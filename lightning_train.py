import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
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
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rvt_model import RvT, PosEmbedding3D
    from torch.utils.data import DataLoader, IterableDataset
    from typing import Optional, Tuple, Dict, Any
    from data_loader import zarr_scan
    import pandas as pd
    import numpy as np
    import random

    # define the LightningModule
    class RadiographyEncoder(L.LightningModule):
        def __init__(
            self, *,
            # model hyperparameters
            encoder_dim,
            encoder_depth,
            encoder_heads,
            mlp_dim,
            n_registers,
            use_rotary,
            # training runtime hyperparameters
            batch_size,
            learning_rate,
            training_steps,
            # dataset hyperparameters
            patch_size,
            patch_jitter,
            # objectives
        ):
            #view objectives refer to:
            #  x, y, z axes
            #  (window width, window center)
            #  (zoom scale, rotation x, y, z)
            NUM_VIEW_OBJECTIVES = 3
            self.n_registers=n_registers

            super().__init__()
            self.save_hyperparameters()
            self.encoder = RvT(
                patch_size=patch_size,
                register_count=n_registers,
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
                mlp_dim=mlp_dim,
                use_rotary=use_rotary
            )

            # CAUTION: relative view head requires two concatenated
            # encoder outputs because it calculates the relative
            # difference* between the objectives
            self.relative_view_head = nn.Sequential(
                nn.LayerNorm(encoder_dim * 2),
                nn.Linear(encoder_dim * 2, NUM_VIEW_OBJECTIVES)
            )

            self.view_criterion = nn.BCEWithLogitsLoss() 

        def raw_encoder_emb_to_scan_view_registers_patches(self, emb):
            # 0: global scan embedding
            # 1: local view embedding
            # 2 - 2+registers: register tokens
            # 2+registers - end: patch embeddings
            return emb[:, 0], emb[:, 1], emb[:, 2:2+self.n_registers], emb[:, 2+self.n_registers:]


        def training_step(self, batch, batch_idx):
            patches_1, patches_2, coords_1, coords_2, row_id, label = batch

            # TODO: allow swapping view target between
            # - regression versus classification task
            # - patient space versus pixel space
            view_target = (label > 0).to(torch.float32)

            # TODO: allow swapping between
            # - absolute versus relative position embedding
            # - patient space versus pixel space
            emb1 = self.encoder(patches_1, coords_1)
            emb2 = self.encoder(patches_2, coords_2)

            # scan_cls_1, view_cls_1, registers_1, patch_emb_1 = self.raw_encoder_emb_to_scan_view_registers_patches(emb1)
            # scan_cls_2, view_cls_2, registers_2, patch_emb_2 = self.raw_encoder_emb_to_scan_view_registers_patches(emb2)

            # TODO: allow specifying which objectives will be run

            fused_view_cls = torch.cat((emb1[:, 1], emb2[:, 1]), dim=1)
            view_prediction = self.relative_view_head(fused_view_cls)
            loss = self.view_criterion(view_prediction, view_target)

            print(loss)

            self.log("view_loss", loss)
            return loss

        # def validation_step(self, batch, batch_idx):
        #     # this is the validation loop
        #     x, _ = batch
        #     x = x.view(x.size(0), -1)
        #     z = self.encoder(x)
        #     x_hat = self.decoder(z)
        #     val_loss = F.mse_loss(x_hat, x)
        #     self.log("val_loss", val_loss)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-4)
            return optimizer


    encoder = RadiographyEncoder(
        encoder_dim=288,
        encoder_depth=12,
        encoder_heads=12,
        mlp_dim=512,
        n_registers=8,
        use_rotary=True,
        # training runtime hyperparameters
        batch_size=128,
        learning_rate=1e-4,
        training_steps=5,
        # dataset hyperparameters
        patch_size=(1, 16, 16),
        patch_jitter=1.0,
    )
    return (
        DataLoader,
        IterableDataset,
        L,
        ModelCheckpoint,
        WandbLogger,
        encoder,
        np,
        os,
        pd,
        random,
        shutil,
        sys,
        torch,
        zarr_scan,
    )


@app.cell
def _():
    return


@app.cell
def _(
    DataLoader,
    IterableDataset,
    np,
    os,
    pd,
    random,
    shutil,
    sys,
    time,
    torch,
    zarr_scan,
):
    class PrismOrderingDataset(IterableDataset):

        def __init__(self, metadata, patch_shape, n_patches, n_sampled_from_same_study=128, scratch_dir="/scratch/gangarav/"):
            super().__init__()
            stats_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/zarr_stats.parquet')
            og_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
            merged_df = pd.merge(
                og_pd,
                stats_pd,
                on='zarr_path',
                how='left'
            )
            self.metadata = merged_df.head(4)

            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_sampled_from_same_study = n_sampled_from_same_study

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
                        patches_1, patches_2, coords_1, coords_2, label = scan.generate_training_pair(
                            n_patches=self.n_patches,
                            to_torch=True
                        )

                        yield patches_1, patches_2, coords_1, coords_2, row_id, label

    # --- Configuration ---
    METADATA_PATH = '/cbica/home/gangarav/data_25_processed/zarr_stats.parquet'

    # Data parameters (MUST match between dataset and model)
    PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
    N_PATCHES = 64            # Number of patches to sample from each scan

    # setup data
    dataset = PrismOrderingDataset(
        metadata=METADATA_PATH,
        patch_shape=PATCH_SHAPE,
        n_patches=N_PATCHES,
        scratch_dir=sys.argv[1]
    )

    NUM_WORKERS=8
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=NUM_WORKERS, # Use 2 worker processes to load data in parallel
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=True,
    )
    # include a validation set
    return (dataloader,)


@app.cell
def _(L, ModelCheckpoint, WandbLogger, dataloader, encoder):
    wandb_logger = WandbLogger(project="rsna25-prism-ordering")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{wandb_logger.version}', # Save in a run-specific folder
        filename='best_model', # Always name the file 'best_model.ckpt'
        save_top_k=1,          # Only save the single best checkpoint
        monitor='view_loss',   # Metric to monitor
        mode='min'             # 'min' because lower loss is better
    )

    trainer = L.Trainer(limit_train_batches=100000, max_epochs=1000, logger=wandb_logger, log_every_n_steps=25)
    trainer.fit(model=encoder, train_dataloaders=dataloader) #, val_dataloaders=) #
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
