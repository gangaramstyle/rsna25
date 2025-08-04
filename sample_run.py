import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    from data_loader import zarr_scan
    from torch.utils.data import DataLoader, IterableDataset
    from rvt_model import RvT
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    import pandas as pd
    import numpy as np
    import os
    import random
    import sys
    import time
    import zarr
    return (
        DataLoader,
        IterableDataset,
        RvT,
        nn,
        np,
        os,
        pd,
        random,
        time,
        torch,
        zarr_scan,
    )


@app.cell
def _(IterableDataset, np, pd, random, torch, zarr_scan):
    def normalize_hu_to_range(hu_array, w_min, w_max, out_range=(-1.0, 1.0)):
        clipped_array = np.clip(hu_array, w_min, w_max)
        scaled_01 = (clipped_array - w_min) / (w_max - w_min)
        out_min, out_max = out_range
        return scaled_01 * (out_max - out_min) + out_min

    class PrismOrderingDataset(IterableDataset):

        def __init__(self, metadata, patch_shape, n_patches):
            super().__init__()
            self.metadata = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
            self.metadata = self.metadata[self.metadata['modality'] == 'CT']
            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_sampled_from_same_study = 10
            self.windows = [
                (0, 100),
                (-20, 180),
                (-800, 2200),
            ]



        def __iter__(self):
            # The __iter__ method is called once per worker process.
            # We need to handle seeding properly for multiprocessing.
            worker_info = torch.utils.data.get_worker_info()

            # If in a worker process, we need to re-seed to ensure each worker
            # generates a different stream of random data.
            if worker_info is not None:
                # A good seed combines a base seed with the worker's ID
                # We use the modulo operator to make sure the seed is in the valid 32-bit range
                seed = (torch.initial_seed() + worker_info.id) % (2**32)

                # This print statement might still not appear depending on your OS/IDE,
                # but the seeding logic below is now correct.
                print(f"Worker {worker_info.id} is using seed: {seed}", flush=True)

                np.random.seed(seed)
                random.seed(seed)

            while True:

                zarr_name = self.metadata.sample(n=1)["zarr_path"].values[0]

                scan = zarr_scan(path_to_scan=zarr_name)

                for i in range(self.n_sampled_from_same_study):
                    w_min, w_max = random.choice(self.windows)

                    subset_1_start, subset_1_shape = scan.get_random_subset_from_scan()
                    idxs_1 = scan.get_random_patch_indices_from_scan_subset(subset_1_start, subset_1_shape, self.n_patches)
                    patches_1 = normalize_hu_to_range(scan.get_patches_from_indices(idxs_1), w_min, w_max)
                    patches_1_pt_space = scan.convert_indices_to_patient_space(idxs_1)
                    subset_1_center = np.array(subset_1_start) + 0.5*np.array(subset_1_shape)
                    subset_1_center = subset_1_center.astype(int)[np.newaxis, :]
                    subset_1_center_pt_space = scan.convert_indices_to_patient_space(subset_1_center)

                    subset_2_start, subset_2_shape = scan.get_random_subset_from_scan()
                    idxs_2 = scan.get_random_patch_indices_from_scan_subset(subset_2_start, subset_2_shape, self.n_patches)
                    patches_2 = normalize_hu_to_range(scan.get_patches_from_indices(idxs_2), w_min, w_max)
                    patches_2_pt_space = scan.convert_indices_to_patient_space(idxs_2)
                    subset_2_center = np.array(subset_2_start) + 0.5*np.array(subset_2_shape)
                    subset_2_center = subset_2_center.astype(int)[np.newaxis, :]
                    subset_2_center_pt_space = scan.convert_indices_to_patient_space(subset_2_center)

                    patches_1_torch = torch.from_numpy(patches_1).to(torch.float32)
                    patches_2_torch = torch.from_numpy(patches_2).to(torch.float32)
                    idx_pt_space_1_torch = torch.from_numpy(patches_1_pt_space).to(torch.float32)
                    idx_pt_space_2_torch = torch.from_numpy(patches_2_pt_space).to(torch.float32)
                    label = torch.tensor(subset_2_center_pt_space - subset_1_center_pt_space, dtype=torch.float32)

                    yield patches_1_torch, patches_2_torch, idx_pt_space_1_torch, idx_pt_space_2_torch, label
    return (PrismOrderingDataset,)


@app.cell
def _(RvT, nn, torch):
    class SaimeseEncoder(nn.Module):
        def __init__(self, *, patch_size, num_classes, dim, depth, heads, mlp_dim, use_rotary = True):
            super().__init__()

            self.branch = RvT(
                patch_size=patch_size,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                use_rotary=use_rotary
            )

            self.trunk = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, num_classes)
            )

        def forward(self, patches_1, coords_1, patches_2, coords_2):
            x1 = self.branch(patches_1, coords_1)
            x2 = self.branch(patches_2, coords_2)
            fused_cls = torch.cat((x1[:, 0], x2[:, 0]), dim = 1)
            return self.trunk(fused_cls)
    return (SaimeseEncoder,)


@app.cell
def _(DataLoader, PrismOrderingDataset, SaimeseEncoder, nn, os, time, torch):
    def train_model():

        OUTPUT_DIR = "trained_models"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # --- Configuration ---
        METADATA_PATH = '/cbica/home/gangarav/data_25_processed/metadata.parquet'

        # Data parameters (MUST match between dataset and model)
        PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
        N_PATCHES = 64            # Number of patches to sample from each scan

        #
        NUM_WORKERS = 6

        # Model hyperparameters
        NUM_CLASSES = 3           # e.g., bleed vs. no_bleed
        MODEL_DIM = 768           # Main dimension of the transformer
        TRANSFORMER_DEPTH = 12     # Number of transformer blocks
        TRANSFORMER_HEADS = 8     # Number of attention heads
        MLP_DIM = 512             # Hidden dimension in the FeedForward network

        # Training parameters
        BATCH_SIZE = 64
        LEARNING_RATE = 1e-4
        TRAINING_STEPS = 300000 # Since the dataset is infinite, we train for a fixed number of steps

        # Check if a GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- Setup Logging ---
        log_filename = f"training_log_{int(time.time())}.csv"
        print(f"Logging training loss to {log_filename}")
        with open(log_filename, 'w') as log_file:
            log_file.write("step,timestamp,loss\n")

        # --- Setup Dataset and DataLoader ---
        print("Setting up data loader...")
        dataset = PrismOrderingDataset(
            metadata=METADATA_PATH,
            patch_shape=PATCH_SHAPE,
            n_patches=N_PATCHES
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, # Use 2 worker processes to load data in parallel
            persistent_workers=(NUM_WORKERS > 0)
        )

        # --- Setup Model ---

        model = SaimeseEncoder(
            patch_size=PATCH_SHAPE,
            num_classes=NUM_CLASSES,
            dim=MODEL_DIM,
            depth=TRANSFORMER_DEPTH,
            heads=TRANSFORMER_HEADS,
            mlp_dim=MLP_DIM,
            use_rotary=True 
        ).to(device)


        # --- Setup Optimizer and Loss Function ---
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # --- Training Loop ---
        print("\nStarting training...")
        model.train()

        data_iterator = iter(dataloader)

        for step in range(TRAINING_STEPS):
            try:
                # 1. Get a batch of data from the infinite dataloader
                # patches_1, patches_2, coords_1, coords_2, labels = 
                patches_1, patches_2, idx_1, idx_2, y = next(data_iterator)

                # Move data to the selected device (GPU or CPU)
                patches_1 = patches_1.to(device)
                patches_2 = patches_2.to(device)
                idx_1 = idx_1.to(device)
                idx_2 = idx_2.to(device)
                y = y.to(device).squeeze(1)

                # 3. Standard training steps
                optimizer.zero_grad()

                # Forward pass
                outputs = model(patches_1, idx_1, patches_2, idx_2)

                # Calculate loss
                loss = criterion(outputs, y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                if (step + 1) % 10 == 0:
                    loss_item = loss.item()
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Step [{step+1}/{TRAINING_STEPS}], Loss: {loss_item:.4f}")
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"{step+1},{timestamp},{loss_item:.4f}\n")

            except StopIteration:
                # This should not happen with an infinite dataset, but is good practice
                print("DataLoader exhausted. Restarting.")
                data_iterator = iter(dataloader)

        print("Training finished.")

        final_model_save_path = os.path.join(OUTPUT_DIR, f"final_model_step_{TRAINING_STEPS}.pth")
        torch.save(model.state_dict(), final_model_save_path)
        print(f"Final model saved to {final_model_save_path}")
    return (train_model,)


@app.cell
def _(train_model):
    train_model()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
