import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from data_loader import zarr_scan
    from torch.utils.data import DataLoader, IterableDataset
    from rvt_model import RvT, PosEmbedding3D
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    from x_transformers import CrossAttender
    import pandas as pd
    import numpy as np
    import os
    import random
    import sys
    import time
    import zarr
    return (
        CrossAttender,
        DataLoader,
        IterableDataset,
        PosEmbedding3D,
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
    class PrismOrderingDataset(IterableDataset):

        def __init__(self, metadata, patch_shape, n_patches, n_sampled_from_same_study=8):
            super().__init__()
            stats_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/zarr_stats.parquet')
            og_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
            merged_df = pd.merge(
                og_pd,
                stats_pd,
                on='zarr_path',
                how='left'
            )
            self.metadata = merged_df.head(500)

            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_sampled_from_same_study = n_sampled_from_same_study

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0 # Get worker ID

            if worker_info is not None:
                seed = (torch.initial_seed() + worker_info.id) % (2**32)
                np.random.seed(seed)
                random.seed(seed)

            while True:
                print(f"[Worker {worker_id}] Sampling a study...")
                sample = self.metadata.sample(n=1)
                zarr_name = sample["zarr_path"].values[0]
                print(f"[Worker {worker_id}] Selected scan: {zarr_name}")

                row_id = sample.index.values[0]
                median = sample["median"].values[0]
                stdev = sample["stdev"].values[0]

                # 2. Instantiate the scan loader with all necessary info
                try:
                    scan = zarr_scan(
                        path_to_scan=zarr_name,
                        median=median,
                        stdev=stdev,
                        patch_shape=self.patch_shape
                    )
                except (ValueError, FileNotFoundError) as e:
                    print(f"[Worker {worker_id}] CRITICAL: Skipping scan {zarr_name} due to error: {e}")
                    continue


                print(f"[Worker {worker_id}] Generating pairs for {zarr_name}...")
                for _ in range(self.n_sampled_from_same_study):
                    patches_1, patches_2, coords_1, coords_2, label, sample_1, sample_2 = scan.generate_training_pair(
                        n_patches=self.n_patches,
                        to_torch=True
                    )

                    print(f"[Worker {worker_id}] Yielding a training sample.")
                    yield patches_1, patches_2, coords_1, coords_2, row_id, label
    return (PrismOrderingDataset,)


@app.cell
def _(CrossAttender, RvT, nn, np, torch):
    def interleave_tensors(tensors):
        tensor1, tensor2 = tensors
        if tensor1.shape != tensor2.shape:
            raise ValueError("Both input tensors must have the same shape.")

        stacked_tensor = torch.stack([tensor1, tensor2], dim=-1)
        B, S, D, _ = stacked_tensor.shape
        interleaved_tensor = stacked_tensor.reshape(B, S, D * 2)

        return interleaved_tensor

    class MIMHead(nn.Module):
        def __init__(self, dim, patch_size, depth, layer_dropout):
            super().__init__()
            self.mim = CrossAttender(dim = dim, depth = depth, layer_dropout=layer_dropout)
            self.to_pixels = nn.Linear(dim, np.prod(patch_size))
            self.patch_size = patch_size

        def forward(self, pos, context):
            unmasked = self.mim(pos, context=context)
            unmasked = self.to_pixels(unmasked)
            unmasked = unmasked.view(unmasked.size(0), unmasked.size(1), *self.patch_size)
            return unmasked

    class SaimeseEncoder(nn.Module):
        def __init__(self, *, patch_size, register_count, num_classes, dim, depth, heads, mlp_dim, use_rotary = True):
            super().__init__()

            self.branch = RvT(
                patch_size=patch_size,
                register_count=register_count,
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
            self.mim = MIMHead(dim, patch_size, depth=4, layer_dropout=0.2)

        def forward(self, patches_1, coords_1, patches_2, coords_2):
            x1 = self.branch(patches_1, coords_1)
            x2 = self.branch(patches_2, coords_2)

            return x1, x2
    return SaimeseEncoder, interleave_tensors


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
    return


@app.cell
def _(
    DataLoader,
    PosEmbedding3D,
    PrismOrderingDataset,
    SaimeseEncoder,
    interleave_tensors,
    nn,
    os,
    time,
    torch,
):
    def train_model():
        OUTPUT_DIR = "FULL_trained_models"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # --- Configuration ---
        METADATA_PATH = '/cbica/home/gangarav/data_25_processed/zarr_stats.parquet'

        # Data parameters (MUST match between dataset and model)
        PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
        N_PATCHES = 64            # Number of patches to sample from each scan

        #
        NUM_WORKERS = 16

        # Model hyperparameters
        NUM_CLASSES = 3           # e.g., bleed vs. no_bleed
        MODEL_DIM = 288           # Main dimension of the transformer
        TRANSFORMER_DEPTH = 12     # Number of transformer blocks
        TRANSFORMER_HEADS = 12     # Number of attention heads
        MLP_DIM = 512             # Hidden dimension in the FeedForward network
        REGISTER_COUNT = 8

        # Training parameters
        BATCH_SIZE = 128
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
            register_count=REGISTER_COUNT,
            num_classes=NUM_CLASSES,
            dim=MODEL_DIM,
            depth=TRANSFORMER_DEPTH,
            heads=TRANSFORMER_HEADS,
            mlp_dim=MLP_DIM,
            use_rotary=True 
        ).to(device)

        coordinate_encoder = PosEmbedding3D(MODEL_DIM//2, max_freq = 3)

        # --- Setup Optimizer and Loss Function ---
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        position_criterion = nn.BCEWithLogitsLoss() 
        mim_loss = nn.SmoothL1Loss()

        # --- Training Loop ---
        print("\nStarting training...")
        model.train()

        data_iterator = iter(dataloader)

        for step in range(TRAINING_STEPS):
            try:
                patches_1, patches_2, idx_1, idx_2, scan, y = next(data_iterator)
                print(f"======={step}======")

                # Split patches and indices for input vs label
                split_size = 50

                input_patches_1 = patches_1[:,:split_size]
                label_patches_1 = patches_1[:,split_size:]
                input_idx_1 = idx_1[:,:split_size]
                label_idx_1 = idx_1[:,split_size:]

                input_patches_2 = patches_2[:,:split_size]
                label_patches_2 = patches_2[:,split_size:]
                input_idx_2 = idx_2[:,:split_size]
                label_idx_2 = idx_2[:,split_size:]

                # Move data to the selected device (GPU or CPU)
                input_patches_1 = input_patches_1.to(device, non_blocking=True)
                input_patches_2 = input_patches_2.to(device, non_blocking=True)
                input_idx_1 = input_idx_1.to(device, non_blocking=True)
                input_idx_2 = input_idx_2.to(device, non_blocking=True)
                label_patches_1 = label_patches_1.to(device, non_blocking=True)
                label_patches_2 = label_patches_2.to(device, non_blocking=True)
                label_idx_1 = label_idx_1.to(device, non_blocking=True)
                label_idx_2 = label_idx_2.to(device, non_blocking=True)
                scan = scan.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).squeeze(1)

                y_class = (y > 0).to(torch.float32)

                # 3. Standard training steps
                optimizer.zero_grad()

                # Forward pass
                x1, x2 = model(input_patches_1, input_idx_1, input_patches_2, input_idx_2)

                # fused_scan_cls = torch.cat((x1[:, 0], x2[:, 0]), dim = 0)
                # fused_scan_cls = nn.functional.normalize(fused_scan_cls, p=2, dim=1)
                # fused_scan_y = torch.cat((scan, scan), dim = 0)

                # scan_loss = supervised_contrastive_loss(fused_scan_cls, fused_scan_y)

                fused_pos_cls = torch.cat((x1[:, 1], x2[:, 1]), dim=1)
                pos_prediction = model.trunk(fused_pos_cls)
                pos_loss = position_criterion(pos_prediction, y_class)

                mim_prediction_1 = model.mim(
                    interleave_tensors(coordinate_encoder(label_idx_1)),
                    torch.cat((x1[:, :2], x1[:,2+REGISTER_COUNT:]), dim=1)
                )
                mim_prediction_2 = model.mim(
                    interleave_tensors(coordinate_encoder(label_idx_2)),
                    torch.cat((x2[:, :2], x2[:,2+REGISTER_COUNT:]), dim=1)
                )

                mim_1_loss = mim_loss(mim_prediction_1, label_patches_1)
                mim_2_loss = mim_loss(mim_prediction_2, label_patches_2)

                total_loss = mim_1_loss + mim_2_loss + pos_loss

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
                          f"MIM1 Loss: {mim_1_loss.item():.4f}, "
                          f"MIM2 Loss: {mim_2_loss.item():.4f}") #, "
                          # f"SCAN Loss: {scan_loss.item():.4f}")

                    # Log all three loss values for better analysis later
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"{step+1},{timestamp},{total_loss_item:.4f},{pos_loss.item():.4f},{mim_1_loss.item():.4f},{mim_2_loss.item():.4f}\n")

                # Interim Checkpoint Saving
                if (step + 1) % 10000 == 0:
                    # Define a unique filename for the checkpoint
                    checkpoint_path = os.path.join(OUTPUT_DIR, f"MIM_{log_filename}_s_{step+1}.pth")

                    # Save the model's state dictionary
                    torch.save(model.state_dict(), checkpoint_path)

                    print(f"\n--- Saved interim checkpoint at step {step+1} to {checkpoint_path} ---\n")

            except StopIteration:
                # This should not happen with an infinite dataset, but is good practice
                print("DataLoader exhausted. Restarting.")
                data_iterator = iter(dataloader)

        print("Training finished.")

        final_model_save_path = os.path.join(OUTPUT_DIR, f"MIM_{log_filename}_s_{TRAINING_STEPS}.pth")
        torch.save(model.state_dict(), final_model_save_path)
        print(f"Final model saved to {final_model_save_path}")
    return (train_model,)


@app.cell
def _(train_model):
    train_model()
    return


if __name__ == "__main__":
    app.run()
