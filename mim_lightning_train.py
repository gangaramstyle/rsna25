import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import wandb
    import sys
    import os
    import shutil
    import torch
    from torch import optim, nn, utils, Tensor
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import lightning as L
    from pytorch_lightning.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from rvt_model import RvT, PosEmbedding3D
    from torch.utils.data import DataLoader, IterableDataset
    from typing import Optional, Tuple, Dict, Any
    from data_loader import zarr_scan
    from x_transformers import CrossAttender
    import pandas as pd
    import numpy as np
    import random
    import time
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score


@app.cell
def _():
    # define the LightningModule
    class RadiographyEncoder(L.LightningModule):
        def __init__(
            self,
            *,
            # model hyperparameters
            encoder_dim,
            encoder_depth,
            encoder_heads,
            mlp_dim,
            n_registers,
            # position embedding
            pos_max_freq,
            use_rotary,
            use_absolute,
            # training runtime hyperparameters
            batch_size,
            learning_rate,
            # dataset hyperparameters
            patch_size,
            patch_jitter,
            # objectives
            pos_objective_mode,
            window_objective,
            scan_contrastive_objective,
            mim_objective,

        ):
            # view objectives refer to:
            #  x, y, z axes
            #  (window width, window center)
            #  (zoom scale, rotation x, y, z)
            self.NUM_POS_OBJECTIVES = 3
            self.NUM_WINDOW_OBJECTIVES = 2
            self.n_registers = n_registers

            super().__init__()

            self.save_hyperparameters()
            self.encoder = RvT(
                patch_size=patch_size,
                register_count=n_registers,
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
                mlp_dim=mlp_dim,
                use_rotary=use_rotary,
                pos_max_freq=pos_max_freq,
                use_absolute=use_absolute,
            )

            # CAUTION: relative view head requires two concatenated
            # encoder outputs because it calculates the relative
            # difference* between the objectives
            self.relative_pos_head = nn.Sequential(
                nn.LayerNorm(encoder_dim * 2), nn.Linear(encoder_dim * 2, self.NUM_POS_OBJECTIVES)
            )

            self.relative_window_head = nn.Sequential(
                nn.LayerNorm(encoder_dim * 2), nn.Linear(encoder_dim * 2, self.NUM_WINDOW_OBJECTIVES)
            )

            if self.hparams.pos_objective_mode == "classification":
                self.pos_view_criterion = nn.BCEWithLogitsLoss()
            elif self.hparams.pos_objective_mode == "regression":
                self.pos_view_criterion = nn.MSELoss()

            self.window_view_criterion = nn.BCEWithLogitsLoss()

            self.mim = MIMHead(encoder_dim, pos_max_freq, patch_size, depth=4, layer_dropout=0.2)
            self.mim_loss = nn.SmoothL1Loss()

            self.validation_step_outputs = []

        def raw_encoder_emb_to_scan_view_registers_patches(self, emb):
            # 0: global scan embedding
            # 1: local view embedding
            # 2 - 2+registers: register tokens
            # 2+registers - end: patch embeddings
            return (
                emb[:, 0:1],
                emb[:, 1:2],
                emb[:, 2 : 2 + self.n_registers],
                emb[:, 2 + self.n_registers :],
            )

        def training_step(self, batch, batch_idx):
            patches_1, patches_2, patch_coords_1, patch_coords_2, aux_patches_1, aux_patches_2, aux_coords_1, aux_coords_2, label, row_id = batch

            # Split patches and indices for input vs label
            split_size = 50

            input_patches_1 = patches_1[:,:split_size]
            mim_patches_1 = patches_1[:,split_size:]
            input_coords_1 = patch_coords_1[:,:split_size]
            mim_coords_1 = patch_coords_1[:,split_size:]

            input_patches_2 = patches_2[:,:split_size]
            mim_patches_2 = patches_2[:,split_size:]
            input_coords_2 = patch_coords_2[:,:split_size]
            mim_coords_2 = patch_coords_2[:,split_size:]

            all_patches_1 = torch.cat([input_patches_1, aux_patches_1], dim=1)
            all_coords_1 = torch.cat([input_coords_1, aux_coords_1], dim=1)
            emb1 = self.encoder(all_patches_1, all_coords_1)

            all_patches_2 = torch.cat([input_patches_2, aux_patches_2], dim=1)
            all_coords_2 = torch.cat([input_coords_2, aux_coords_2], dim=1)
            emb2 = self.encoder(all_patches_2, all_coords_2)

            scan_cls_1, view_cls_1, registers_1, patch_emb_1 = (
                self.raw_encoder_emb_to_scan_view_registers_patches(emb1)
            )
            scan_cls_2, view_cls_2, registers_2, patch_emb_2 = (
                self.raw_encoder_emb_to_scan_view_registers_patches(emb2)
            )

            fused_view_cls = torch.cat((view_cls_1.squeeze(), view_cls_2.squeeze()), dim=1)

            fused_scan_cls = torch.cat((scan_cls_1.squeeze(), scan_cls_2.squeeze()), dim = 0)
            fused_scan_cls = nn.functional.normalize(fused_scan_cls, p=2, dim=1)
            fused_scan_ids = torch.cat((row_id, row_id), dim = 0)

            # --- DYNAMIC LOSS CALCULATION ---
            total_loss = 0.0

            pos_label = label[:, :self.NUM_POS_OBJECTIVES]
            if self.hparams.pos_objective_mode == "classification":
                pos_target = (pos_label > 0).to(torch.float32)
            elif self.hparams.pos_objective_mode == "regression":
                # Target is the actual distance value
                pos_target = (pos_label / 100).to(torch.float32)

            pos_prediction = self.relative_pos_head(fused_view_cls)
            pos_loss = self.pos_view_criterion(pos_prediction, pos_target)

            total_loss += pos_loss
            self.log("pos_loss", pos_loss)

            if self.hparams.window_objective:
                window_label = label[:, -self.NUM_WINDOW_OBJECTIVES:]
                window_target = (window_label > 0).to(torch.float32)

                window_prediction = self.relative_window_head(fused_view_cls)
                window_loss = self.window_view_criterion(window_prediction, window_target)

                total_loss += window_loss
                self.log("window_loss", window_loss)

            if self.hparams.scan_contrastive_objective:
                scan_loss = supervised_contrastive_loss(fused_scan_cls, fused_scan_ids)/4.0
                total_loss += scan_loss
                self.log("scan_loss", scan_loss)

            if self.hparams.mim_objective:
                mim_prediction_1 = self.mim(
                    mim_coords_1,
                    torch.cat((scan_cls_1, view_cls_1, patch_emb_1), dim=1)
                )
                mim_prediction_2 = self.mim(
                    mim_coords_2,
                    torch.cat((scan_cls_2, view_cls_2, patch_emb_2), dim=1)
                )

                mim_1_loss = self.mim_loss(mim_prediction_1, mim_patches_1)
                mim_2_loss = self.mim_loss(mim_prediction_2, mim_patches_2)

                self.log("mim_loss", mim_1_loss + mim_2_loss)
                total_loss += mim_1_loss + mim_2_loss

            self.log("loss", total_loss)

            return total_loss

        def validation_step(self, batch, batch_idx):
            # The validation dataloader yields (patches, centers, location)
            patches, centers, locations = batch

            # Get the view embedding
            emb = self.encoder(patches, centers)
            view_embedding = emb[:, 1]

            # Store the outputs for later use in `on_validation_epoch_end`
            # .detach().cpu() is important to avoid GPU memory leaks
            output = {"embeddings": view_embedding.detach().cpu(), "locations": locations}
            self.validation_step_outputs.append(output)
            return output

        # --- NEW: ON_VALIDATION_EPOCH_END ---
        def on_validation_epoch_end(self):
            if not self.validation_step_outputs:
                print("No validation outputs to process.")
                return

            # --- 1. Aggregate all embeddings and locations from batches ---
            all_embeddings = torch.cat([x["embeddings"] for x in self.validation_step_outputs]).numpy()

            # Locations might be a list of tuples, so we flatten it
            all_locations = []
            for x in self.validation_step_outputs:
                all_locations.extend(x["locations"])

            # --- 2. Perform Clustering and ARI Calculation (your logic) ---
            target_locations = ["Left Middle Cerebral Artery", "Right Middle Cerebral Artery"]
            filtered_indices = [i for i, loc in enumerate(all_locations) if loc in target_locations]

            if not filtered_indices:
                print("No validation data found for target locations. Skipping ARI calculation.")
                self.validation_step_outputs.clear()  # IMPORTANT: Clear stored outputs
                return

            filtered_embeddings = all_embeddings[filtered_indices]
            filtered_locations = [all_locations[i] for i in filtered_indices]

            # --- 3. Create and Log Visualization to wandb ---
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(filtered_embeddings)

            n_clusters = len(target_locations)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_3d)

            ari_score = adjusted_rand_score(filtered_locations, cluster_labels)

            # Log the ARI score to wandb
            self.log("val_ari_score", ari_score, prog_bar=True)
            print(f"Validation ARI Score: {ari_score:.4f}")

            fig = plt.figure(figsize=(20, 9))
            fig.suptitle(
                f"Clustering (Step {self.global_step}) - Ground Truth vs. K-Means", fontsize=16
            )

            # Plot 1: Ground Truth
            ax1 = fig.add_subplot(121, projection="3d")
            unique_locs = list(set(filtered_locations))
            color_map = {loc: plt.cm.viridis(i / len(unique_locs)) for i, loc in enumerate(unique_locs)}
            gt_colors = [color_map[loc] for loc in filtered_locations]
            ax1.scatter(
                embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=gt_colors, alpha=0.7
            )
            ax1.set_title("Ground Truth Labels")

            # Plot 2: K-Means Predictions
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.scatter(
                embeddings_3d[:, 0],
                embeddings_3d[:, 1],
                embeddings_3d[:, 2],
                c=cluster_labels,
                cmap="viridis",
                alpha=0.7,
            )
            ax2.set_title(f"K-Means Clusters (ARI: {ari_score:.2f})")

            # Log the figure to Weights & Biases
            self.logger.experiment.log({"validation_clusters": wandb.Image(fig)})
            plt.close(fig)  # Close the figure to free memory

            # --- 4. IMPORTANT: Clear the stored outputs ---
            self.validation_step_outputs.clear()

        def configure_optimizers(self):
            # Use the learning_rate from hparams so it can be configured by sweeps
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            return optimizer
    return (RadiographyEncoder,)


@app.class_definition
class MIMHead(nn.Module):
    def __init__(self, dim, pos_max_freq, patch_size, depth, layer_dropout):
        super().__init__()
        self.pos_emb = PosEmbedding3D(dim, max_freq = pos_max_freq)
        self.mim = CrossAttender(dim = dim, depth = depth, layer_dropout=layer_dropout)
        self.to_pixels = nn.Linear(dim, np.prod(patch_size))
        self.patch_size = patch_size

    def forward(self, coords, context):
        sin, cos = self.pos_emb(coords)
        sin_unrepeated = sin[:, :, 0::2]
        cos_unrepeated = cos[:, :, 0::2]
        pos = PosEmbedding3D.interleave_two_tensors(sin_unrepeated, cos_unrepeated)

        unmasked = self.mim(pos, context=context)
        unmasked = self.to_pixels(unmasked)
        unmasked = unmasked.view(unmasked.size(0), unmasked.size(1), *self.patch_size)
        return unmasked


@app.function
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


@app.class_definition
class PrismOrderingDataset(IterableDataset):

    def __init__(self, metadata, include_nifti, patch_shape, position_space, n_patches, n_aux_patches, n_sampled_from_same_study, scratch_dir):
        super().__init__()
        self.metadata = pd.read_parquet('/cbica/home/gangarav/rsna_any/rsna_2025/combined_metadata.parquet')
        if not include_nifti:
            self.metadata = self.metadata.query("original_format != 'nifti'")

        self.patch_shape = patch_shape
        self.n_patches = n_patches
        self.n_sampled_from_same_study = n_sampled_from_same_study

        self.position_space=position_space
        self.n_aux_patches=n_aux_patches

        self.scratch_dir = scratch_dir
        if self.scratch_dir:
            print(f"Using scratch directory: {self.scratch_dir}")
            os.makedirs(self.scratch_dir, exist_ok=True)

    def _stage_data_to_scratch(self, source_path, worker_id):
        """
        Checks for data on scratch and copies it if not present.
        Handles race conditions with a lock file.
        Returns the path to the data on scratch.
        """
        scan_basename = os.path.basename(source_path)
        dest_path = os.path.join(self.scratch_dir, scan_basename)
        lock_path = dest_path + ".lock"

        # If the final destination already exists, we're done.
        if os.path.exists(dest_path):
            print(f"[Worker {worker_id}] Found {scan_basename} in scratch.")
            return dest_path

        # --- Lock file mechanism to prevent race conditions ---
        try:
            # Atomic operation: try to create the lock directory.
            # If it fails, another worker is already copying.
            os.makedirs(lock_path)

            print(f"[Worker {worker_id}] Lock acquired. Copying {scan_basename} to scratch...")
            # If a previous copy failed, the dest_path might exist but be incomplete.
            # It's safer to remove it before copying.
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)

            shutil.copytree(source_path, dest_path)
            print(f"[Worker {worker_id}] Copy finished for {scan_basename}.")

        except FileExistsError:
            print(f"[Worker {worker_id}] Waiting for another worker to copy {scan_basename}...")
            # Wait until the lock file is gone.
            while os.path.exists(lock_path):
                time.sleep(2) # Wait for 2 seconds before checking again
            print(f"[Worker {worker_id}] Lock released for {scan_basename}. Proceeding.")

        finally:
            # The worker that acquired the lock must remove it.
            if os.path.exists(lock_path):
                os.rmdir(lock_path)

        # Double-check that the data is now there after waiting
        if not os.path.exists(dest_path):
            raise FileNotFoundError(f"Worker {worker_id} waited for copy, but {dest_path} was not found. The copy may have failed.")

        return dest_path

    def generate_training_pair(self, scan, n_patches: int, position_space = "patient", n_aux_patches: int = 0, debug: bool = False) -> tuple:
        """
        A high-level helper for training loops that generates a pair of samples.
        """

        wc1, ww1 = scan.get_random_wc_ww_for_scan()
        wc2, ww2 = scan.get_random_wc_ww_for_scan()
        sample_1_data = scan.train_sample(n_patches=n_patches - n_aux_patches, wc=wc1, ww=ww1)
        sample_2_data = scan.train_sample(n_patches=n_patches - n_aux_patches, wc=wc2, ww=ww2)

        if position_space == "patient":
            patch_key = 'patch_centers_pt'
            center_key = 'subset_center_pt'
        elif position_space == "pixel":
            patch_key = 'patch_centers_idx'
            center_key = 'subset_center_idx'
        else:
            # Use a more specific exception type
            raise ValueError(f"Invalid position_space setting was given: {position_space}")

        patches_1 = sample_1_data['normalized_patches']
        patches_2 = sample_2_data['normalized_patches']

        patch_coords_1 = sample_1_data[patch_key] - sample_1_data[center_key]
        patch_coords_2 = sample_2_data[patch_key] - sample_2_data[center_key]

        pos_label = (sample_2_data[center_key] - sample_1_data[center_key]).squeeze(0)

        # window based relative view information
        window_label = np.array([wc2 - wc1, ww2 - ww1])

        label = np.concatenate((pos_label, window_label))

        if n_aux_patches > 0:
            sample_1_aux = scan.train_sample(n_patches=n_aux_patches, patch_jitter=5, wc=wc1, ww=ww1)
            sample_2_aux = scan.train_sample(n_patches=n_aux_patches, patch_jitter=5, wc=wc2, ww=ww2)

            aux_patches_1 = sample_1_aux['normalized_patches']
            aux_patches_2 = sample_2_aux['normalized_patches']

            aux_coords_1 = sample_1_aux[patch_key] - sample_1_data[center_key]
            aux_coords_2 = sample_2_aux[patch_key] - sample_2_data[center_key]
        else:
            # make dummy empty values
            sample_1_aux = None
            sample_2_aux = None
            aux_patches_1 = np.empty((0, *patches_1.shape[1:]), dtype=patches_1.dtype)
            aux_patches_2 = np.empty((0, *patches_2.shape[1:]), dtype=patches_2.dtype)
            aux_coords_1 = np.empty((0, *patch_coords_1.shape[1:]), dtype=patch_coords_1.dtype)
            aux_coords_2 = np.empty((0, *patch_coords_2.shape[1:]), dtype=patch_coords_2.dtype)


        tensors = [
            patches_1, patches_2, patch_coords_1, patch_coords_2,
            aux_patches_1, aux_patches_2, aux_coords_1, aux_coords_2,
            label
        ]
        if debug:
            return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors), sample_1_data, sample_2_data, sample_1_aux, sample_2_aux

        return tuple(torch.from_numpy(arr).to(torch.float32) for arr in tensors)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0 # Get worker ID

        if worker_info is None:
            # Case: num_workers = 0. The main process gets all the data.
            worker_id = 0
            worker_metadata = self.metadata
        else:
            # Case: num_workers > 0.
            worker_id = worker_info.id
            # Split the metadata DataFrame among workers.
            # The slice notation [start:stop:step] is used here.
            # worker_id::num_workers gives each worker a unique, non-overlapping subset.
            worker_metadata = self.metadata.iloc[worker_id::worker_info.num_workers]

            # Seed each worker differently to ensure random shuffling is unique
            seed = (torch.initial_seed() + worker_id) % (2**32)
            np.random.seed(seed)
            random.seed(seed)

        print(f"[Worker {worker_id}] assigned {len(worker_metadata)} scans.")

        while True:

            shuffled_worker_metadata = worker_metadata.sample(frac=1)

            for _, sample in shuffled_worker_metadata.iterrows():
                print(f"[Worker {worker_id}] Sampling a study...")
                source_zarr_path = sample["zarr_path"]
                print(f"[Worker {worker_id}] Selected scan: {source_zarr_path}")

                # --- CORE LOGIC CHANGE ---
                path_to_load = source_zarr_path
                if self.scratch_dir:
                    try:
                        path_to_load = self._stage_data_to_scratch(source_zarr_path, worker_id)
                    except (Exception) as e:
                        print(f"[Worker {worker_id}] CRITICAL: Failed to stage {source_zarr_path} to scratch. Skipping. Error: {e}")
                        continue
                # -------------------------

                row_id = sample.name
                median = sample["median"]
                stdev = sample["stdev"]

                # 2. Instantiate the scan loader with all necessary info
                try:
                    scan = zarr_scan(
                        path_to_scan=path_to_load,
                        median=median,
                        stdev=stdev,
                        patch_shape=self.patch_shape
                    )
                except (ValueError, FileNotFoundError) as e:
                    print(f"[Worker {worker_id}] CRITICAL: Skipping scan {source_zarr_path} due to error: {e}")
                    continue


                print(f"[Worker {worker_id}] Generating pairs for {source_zarr_path}...")
                for _ in range(self.n_sampled_from_same_study):
                    training_pair = self.generate_training_pair(
                        scan,
                        n_patches=self.n_patches,
                        position_space=self.position_space,
                        n_aux_patches=self.n_aux_patches
                    )

                    yield (*training_pair, row_id)


@app.class_definition
class ValidationDataset(IterableDataset):
    def __init__(self, prism_shape=(6, 64, 64), patch_shape=None, n_patches=None):
        super().__init__()
        # NOTE: Make sure this path is correct on your system

        self.metadata = pd.read_parquet('aneurysm_cleaned_6_64_64.parquet')
        # Excluding a known problematic series
        self.metadata = self.metadata[self.metadata['series_uid'] != '1.2.826.0.1.3680043.8.498.40511751565674479940947446050421785002']
        self.prism_shape = prism_shape
        self.patch_shape = patch_shape
        self.n_patches = n_patches
        print(f"Initialized validation dataset with {len(self.metadata)} samples.")

    def __iter__(self):
        # No need for worker splitting if num_workers=0, which is typical for smaller validation sets
        for _, row in self.metadata.iterrows():
            try:
                zarr_path = row["zarr_path"]
                median = row["median"]
                stdev = row["stdev"]
                z, y, x = row['aneurysm_z'], row['aneurysm_y'], row['aneurysm_x']
                location = row['location']

                scan = zarr_scan(
                    path_to_scan=zarr_path,
                    median=median,
                    stdev=stdev,
                    patch_shape=self.patch_shape
                )

                sample = scan.train_sample(
                    n_patches=self.n_patches,
                    subset_start=(z - self.prism_shape[0] / 2, y - self.prism_shape[1] / 2, x - self.prism_shape[2] / 2),
                    subset_shape=self.prism_shape,
                )

                patches = torch.from_numpy(sample["normalized_patches"]).to(torch.float32)
                patch_coords = torch.from_numpy(sample['patch_centers_pt'] - sample['subset_center_pt']).to(torch.float32)

                # Yield data in the format expected by validation_step
                yield patches, patch_coords, location
            except Exception as e:
                print(f"Skipping validation sample due to error: {e} in {row.get('zarr_path', 'N/A')}")
                continue


@app.cell
def _(RadiographyEncoder):
    def train_run(default_config=None):
        """
        Main training function that can be called by a sweep agent or for a single run.
        Handles both starting new runs and resuming existing ones.
        """
        # --- 1. Initialize Weights & Biases ---
        # `wandb.init()` will automatically pick up hyperparameters from a sweep agent.
        # If resuming, it will use the WANDB_RUN_ID environment variable.
        # The `resume="allow"` option is key to enabling this behavior.
        run = wandb.init(resume="allow")

        print(run.config)
        print("======\n")
        checkpoint_dir = f'/cbica/home/gangarav/checkpoints/{run.id}'

        # Pull the final config from wandb. This includes sweep params and defaults.
        cfg = run.config

        # --- 2. Handle Resuming ---
        ckpt_path = None
        if run.resumed:
            print(f"Resuming run {run.id}...")
            # Check for the 'last.ckpt' created by lightning's ModelCheckpoint.
            potential_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
            if os.path.exists(potential_ckpt):
                ckpt_path = potential_ckpt
                print(f"Found checkpoint to resume from: {ckpt_path}")
            else:
                print(f"WARNING: Run {run.id} is being resumed, but no 'last.ckpt' found in {checkpoint_dir}. Starting training from scratch but logging to the same run.")

        # --- 3. Setup Model ---
        model = RadiographyEncoder(
            encoder_dim=cfg.encoder_dim,
            encoder_depth=cfg.encoder_depth,
            encoder_heads=cfg.encoder_heads,
            mlp_dim=cfg.mlp_dim,
            n_registers=cfg.n_registers,
            pos_max_freq=cfg.max_freq,
            use_rotary=cfg.use_rotary,
            use_absolute=cfg.use_absolute,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            patch_size=(1, 16, 16),
            patch_jitter=1.0,
            pos_objective_mode=cfg.pos_objective_mode,
            window_objective=cfg.window_objective,
            scan_contrastive_objective=cfg.scan_contrastive_objective,
            mim_objective=cfg.mim_objective,
        )

        # --- 4. Setup Data ---
        PATCH_SHAPE = (1, 16, 16)
        N_PATCHES = 64
        NUM_WORKERS = 10
        METADATA_PATH = '/cbica/home/gangarav/rsna_any/rsna_2025/combined_metadata.parquet'
        scratch_dir = os.environ.get('TMP')
        if scratch_dir is None:
            # If the variable isn't set, raise an error or use a default.
            # For a cluster job, it's better to fail loudly.
            raise ValueError("Environment variable TMP is not set. This is required for the scratch directory.")
            # scratch_dir = "/scratch/gangarav/"

        dataset = PrismOrderingDataset(
            include_nifti=cfg.include_nifti,
            metadata=METADATA_PATH,
            patch_shape=PATCH_SHAPE,
            n_patches=N_PATCHES,
            position_space=cfg.position_space,
            n_aux_patches=cfg.n_aux_patches,
            scratch_dir=scratch_dir,
            n_sampled_from_same_study=cfg.num_repeated_study_samples
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=(NUM_WORKERS > 0),
            pin_memory=True,
        )

        val_dataset = ValidationDataset(
            patch_shape=PATCH_SHAPE,
            n_patches=N_PATCHES
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 2, # Can often use a larger batch size for validation
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

        # --- 5. Setup Callbacks and Logger ---
        # The logger will automatically use the run initialized by wandb.init()
        wandb_logger = WandbLogger(project="rsna25-prism-ordering", log_model=False)

        # Checkpoints are saved in a directory named after the unique wandb run ID
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{step}',
            save_top_k=3,
            monitor='loss',
            mode='min',
            every_n_train_steps=5000,
            save_last=True,
        )

        # --- 6. Setup Trainer ---
        trainer = L.Trainer(
            max_epochs=-1, # For iterable datasets, steps are better than epochs
            max_steps=5000000, # Example: set a max number of steps
            callbacks=[checkpoint_callback],
            accumulate_grad_batches=10,
            logger=wandb_logger,
            log_every_n_steps=25,
            val_check_interval=5000,
            num_sanity_val_steps=0,
        )

        # --- 7. Start Training ---
        # The `ckpt_path` argument tells the trainer to resume from a checkpoint.
        # If ckpt_path is None, it starts a new training run.
        trainer.fit(
            model=model,
            train_dataloaders=dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )

        wandb.finish()
    return (train_run,)


@app.cell
def _():
    # wandb_logger = WandbLogger(project="rsna25-prism-ordering")

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=f'checkpoints/{wandb_logger.version}', # Save in a run-specific folder
    #     filename='best_model_all', # Always name the file 'best_model.ckpt'
    #     save_top_k=1,          # Only save the single best checkpoint
    #     monitor='view_loss',   # Metric to monitor
    #     mode='min',            # 'min' because lower loss is better
    #     every_n_train_steps=5000,
    #     save_on_train_epoch_end=False
    # )

    # trainer = L.Trainer(limit_train_batches=100000, max_epochs=1000, callbacks=[checkpoint_callback], logger=wandb_logger, log_every_n_steps=25)
    # trainer.fit(model=encoder, train_dataloaders=dataloader) #, val_dataloaders=) #
    return


@app.cell
def _(train_run):
    default_config = {
        'encoder_dim': 288,
        'encoder_depth': 12,
        'encoder_heads': 12,
        'mlp_dim': 512,
        'n_registers': 8,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'pos_objective_mode': 'regression',
        'position_space': 'patient',
        'n_aux_patches': 4,
        'window_objective': True,
        'num_repeated_study_samples': 1,
        'scan_contrastive_objective': True,
        'mim_objective': False,
    }
    train_run() #default_config=default_config)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
