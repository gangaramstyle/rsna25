import marimo

__generated_with = "0.14.16"
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

    def get_random_clamping_range(median, stdev):
        lower_bound = median - 3 * stdev
        upper_bound = median + 3 * stdev
        if lower_bound >= upper_bound:
            return lower_bound, upper_bound
        point1 = np.random.uniform(lower_bound, upper_bound)
        point2 = np.random.uniform(lower_bound, upper_bound)
        return min(point1, point2), max(point1, point2)

    class PrismOrderingDataset(IterableDataset):

        def __init__(self, metadata, patch_shape, n_patches):
            super().__init__()
            self.metadata = pd.read_parquet('/cbica/home/gangarav/data_25_processed/zarr_stats.parquet').head(3250)
            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_sampled_from_same_study = 4

        def __iter__(self):
            # The __iter__ method is called once per worker process.
            # We need to handle seeding properly for multiprocessing.
            worker_info = torch.utils.data.get_worker_info()

            # If in a worker process, we need to re-seed to ensure each worker
            # generates a different stream of random data.
            if worker_info is not None:

                seed = (torch.initial_seed() + worker_info.id) % (2**32)

                # This print statement might still not appear depending on your OS/IDE,
                # but the seeding logic below is now correct.
                print(f"Worker {worker_info.id} is using seed: {seed}", flush=True)

                np.random.seed(seed)
                random.seed(seed)

            while True:

                sample = self.metadata.sample(n=1)

                row_id = sample.index.values[0]
                zarr_name = sample["zarr_path"].values[0]
                median = sample["median"].values[0]
                stdev = sample["stdev"].values[0]

                scan = zarr_scan(path_to_scan=zarr_name)

                for i in range(self.n_sampled_from_same_study):

                    r_wc, r_ww = scan.get_random_wc_ww_for_scan_median_stdev(median, stdev)

                    subset_1_start, subset_1_shape = scan.get_random_subset_from_scan()
                    idxs_1 = scan.get_random_patch_indices_from_scan_subset(subset_1_start, subset_1_shape, self.n_patches)
                    patches_1 = normalize_hu_to_range(scan.get_patches_from_indices(idxs_1), r_wc - 0.5*r_ww, r_wc + 0.5*r_ww)
                    patches_1_pt_space = scan.convert_indices_to_patient_space(idxs_1)
                    subset_1_center = np.array(subset_1_start) + 0.5*np.array(subset_1_shape)
                    subset_1_center = subset_1_center.astype(int)[np.newaxis, :]
                    subset_1_center_pt_space = scan.convert_indices_to_patient_space(subset_1_center)

                    subset_2_start, subset_2_shape = scan.get_random_subset_from_scan()
                    idxs_2 = scan.get_random_patch_indices_from_scan_subset(subset_2_start, subset_2_shape, self.n_patches)
                    patches_2 = normalize_hu_to_range(scan.get_patches_from_indices(idxs_2), r_wc - 0.5*r_ww, r_wc + 0.5*r_ww)
                    patches_2_pt_space = scan.convert_indices_to_patient_space(idxs_2)
                    subset_2_center = np.array(subset_2_start) + 0.5*np.array(subset_2_shape)
                    subset_2_center = subset_2_center.astype(int)[np.newaxis, :]
                    subset_2_center_pt_space = scan.convert_indices_to_patient_space(subset_2_center)

                    patches_1_torch = torch.from_numpy(patches_1).to(torch.float32)
                    patches_2_torch = torch.from_numpy(patches_2).to(torch.float32)
                    idx_pt_space_1_torch = torch.from_numpy(patches_1_pt_space).to(torch.float32)
                    idx_pt_space_2_torch = torch.from_numpy(patches_2_pt_space).to(torch.float32)
                    label = torch.tensor(subset_2_center_pt_space - subset_1_center_pt_space, dtype=torch.float32)

                    yield patches_1_torch, patches_2_torch, idx_pt_space_1_torch, idx_pt_space_2_torch, row_id, label
    return (PrismOrderingDataset,)


@app.cell
def _(RvT, nn):
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

            return x1, x2
    return (SaimeseEncoder,)


@app.cell
def _(nn, torch):
    def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
        """
        Computes the Supervised Contrastive Loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): A tensor of shape [N, D] where N is the batch size
                                       and D is the embedding dimension. Embeddings should be
                                       L2 normalized.
            labels (torch.Tensor): A tensor of shape [N] with the sample ID for each embedding.
            temperature (float): The temperature scaling factor.

        Returns:
            torch.Tensor: The calculated loss.
        """
        device = embeddings.device
        n = embeddings.shape[0]

        # 1. Calculate all-pairs similarity
        # The result is a matrix of shape [N, N]
        similarity_matrix = embeddings @ embeddings.t()

        # 2. Create the positive-pair mask
        # The mask will be True where labels are the same, False otherwise.
        # labels.unsqueeze(0) creates a row vector [1, N]
        # labels.unsqueeze(1) creates a column vector [N, 1]
        # Broadcasting them results in a [N, N] matrix of pairwise label comparisons.
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 3. Discard self-similarity from positives
        # We create a mask to remove the diagonal (where an embedding is compared to itself)
        # an embedding cannot be its own positive pair.
        identity_mask = torch.eye(n, device=device).bool()
        positives_mask = labels_matrix & ~identity_mask

        # 4. Create mask for negative pairs
        # Negatives are all pairs that are not self and not positive.
        negatives_mask = ~labels_matrix

        # The original NT-Xent loss can be simplified for this multi-positive case.
        # For each anchor, the loss is -log( sum(exp(sim_pos)) / sum(exp(sim_all_others)) )

        # To prevent log(0) issues for anchors with no positive pairs, we can mask them out.
        # However, the formulation below handles this gracefully.

        # We need a mask to exclude the diagonal from the denominator's log-sum-exp
        logits_mask = ~identity_mask

        # Apply temperature scaling
        similarity_matrix /= temperature

        # For each row (anchor), we compute the log-softmax over all other samples.
        # The similarity_matrix[logits_mask] flattens the matrix, removing the diagonal.
        # .reshape(n, n - 1) makes it a [N, N-1] matrix where each row corresponds
        # to the similarities of one anchor to all N-1 other samples.
        log_probs = nn.functional.log_softmax(similarity_matrix[logits_mask].reshape(n, n - 1), dim=1)

        # The positives_mask now needs to align with the log_probs matrix.
        # We remove the diagonal from positives_mask as well.
        positives_mask_for_loss = positives_mask[logits_mask].reshape(n, n - 1)

        # For each anchor, we want to sum the log-probabilities of its positive pairs.
        # We use the positive mask to select these probabilities.
        # We normalize by the number of positive pairs for each anchor to get the mean.
        # Adding a small epsilon (1e-7) to the denominator prevents division by zero
        # in case an anchor has no positive pairs (it's the only one of its class).
        num_positives_per_row = positives_mask_for_loss.sum(dim=1)
        loss = - (positives_mask_for_loss * log_probs).sum(dim=1) / (num_positives_per_row + 1e-7)

        # We average the loss over all anchors that had at least one positive pair.
        # This prevents anchors with no positives from contributing a 0 to the mean.
        loss = loss[num_positives_per_row > 0].mean()

        return loss
    return (supervised_contrastive_loss,)


@app.cell
def _(
    DataLoader,
    PrismOrderingDataset,
    SaimeseEncoder,
    nn,
    os,
    supervised_contrastive_loss,
    time,
    torch,
):
    def train_model():
        OUTPUT_DIR = "trained_models"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # --- Configuration ---
        METADATA_PATH = '/cbica/home/gangarav/data_25_processed/zarr_stats.parquet'

        # Data parameters (MUST match between dataset and model)
        PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
        N_PATCHES = 100            # Number of patches to sample from each scan

        #
        NUM_WORKERS = 16

        # Model hyperparameters
        NUM_CLASSES = 3           # e.g., bleed vs. no_bleed
        MODEL_DIM = 288           # Main dimension of the transformer
        TRANSFORMER_DEPTH = 8     # Number of transformer blocks
        TRANSFORMER_HEADS = 8     # Number of attention heads
        MLP_DIM = 512             # Hidden dimension in the FeedForward network

        # Training parameters
        BATCH_SIZE = 256
        LEARNING_RATE = 1e-4
        TRAINING_STEPS = 100000 # Since the dataset is infinite, we train for a fixed number of steps

        # Check if a GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- Setup Logging ---
        log_filename = f"log_{int(time.time())}.csv"
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
            persistent_workers=(NUM_WORKERS > 0),
            pin_memory=True,
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
        position_criterion = nn.MSELoss()

        # --- Training Loop ---
        print("\nStarting training...")
        model.train()

        data_iterator = iter(dataloader)

        for step in range(TRAINING_STEPS):
            try:
                # 1. Get a batch of data from the infinite dataloader
                # patches_1, patches_2, coords_1, coords_2, labels = 
                patches_1, patches_2, idx_1, idx_2, scan, y = next(data_iterator)

                # Move data to the selected device (GPU or CPU)
                patches_1 = patches_1.to(device, non_blocking=True)
                patches_2 = patches_2.to(device, non_blocking=True)
                idx_1 = idx_1.to(device, non_blocking=True)
                idx_2 = idx_2.to(device, non_blocking=True)
                scan = scan.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).squeeze(1)

                # 3. Standard training steps
                optimizer.zero_grad()

                # Forward pass
                x1, x2 = model(patches_1, idx_1, patches_2, idx_2)

                fused_scan_cls = torch.cat((x1[:, 0], x2[:, 0]), dim = 0)
                fused_scan_cls = nn.functional.normalize(fused_scan_cls, p=2, dim=1)
                fused_scan_y = torch.cat((scan, scan), dim = 0)
                scan_loss = supervised_contrastive_loss(fused_scan_cls, fused_scan_y)

                fused_pos_cls = torch.cat((x1[:, 1], x2[:, 1]), dim=1)
                pos_prediction = model.trunk(fused_pos_cls)
                pos_loss = position_criterion(pos_prediction, y)

                total_loss = scan_loss + 5.0 * pos_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                if (step + 1) % 10 == 0:
                    # Get the scalar value of each loss for logging
                    total_loss_item = total_loss.item()

                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                    # A more informative print statement
                    print(f"Step [{step+1}/{TRAINING_STEPS}], "
                          f"Total Loss: {total_loss_item:.4f}, "
                          f"POS Loss: {pos_loss.item():.4f}, "
                          f"SCAN Loss: {scan_loss.item():.4f}")

                    # Log all three loss values for better analysis later
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"{step+1},{timestamp},{total_loss_item:.4f},{pos_loss.item()},{scan_loss.item()}\n")

                # Interim Checkpoint Saving
                if (step + 1) % 2500 == 0:
                    # Define a unique filename for the checkpoint
                    checkpoint_path = os.path.join(OUTPUT_DIR, f"{log_filename}_s_{step+1}.pth")

                    # Save the model's state dictionary
                    torch.save(model.state_dict(), checkpoint_path)

                    print(f"\n--- Saved interim checkpoint at step {step+1} to {checkpoint_path} ---\n")

            except StopIteration:
                # This should not happen with an infinite dataset, but is good practice
                print("DataLoader exhausted. Restarting.")
                data_iterator = iter(dataloader)

        print("Training finished.")

        final_model_save_path = os.path.join(OUTPUT_DIR, f"{log_filename}_s_{TRAINING_STEPS}.pth")
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
