import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import torch
    from torch import optim, nn, utils, Tensor
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from rvt_model import RvT, PosEmbedding3D
    from torch.utils.data import DataLoader, IterableDataset
    from typing import Optional, Tuple, Dict, Any
    from data_loader import zarr_scan
    import pandas as pd
    import numpy as np
    import random

    return (
        DataLoader,
        IterableDataset,
        L,
        RvT,
        mo,
        nn,
        optim,
        pd,
        torch,
        zarr_scan,
    )


@app.cell
def _(L, RvT, nn, optim, torch):

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
            print(view_prediction)
            print(view_target)
            loss = self.view_criterion(view_prediction, view_target)

            self.log("view_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-4)
            return optimizer

    return (RadiographyEncoder,)


@app.cell
def _(
    DataLoader,
    IterableDataset,
    pd,
    source_zarr_path,
    torch,
    worker_id,
    zarr_scan,
):
    class PrismOrderingDataset(IterableDataset):

        def __init__(self, patch_shape, n_patches):
            super().__init__()
            stats_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/zarr_stats.parquet')
            og_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
            merged_df = pd.merge(
                og_pd,
                stats_pd,
                on='zarr_path',
                how='left'
            )
            self.metadata = merged_df

            self.patch_shape = patch_shape
            self.n_patches = n_patches

        def __iter__(self):

            while True:

                for _, sample in self.metadata.iterrows():

                    path_to_load = sample["zarr_path"]
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


                    wc, ww = scan.get_random_wc_ww_for_scan()
                    sample_1_data = scan.train_sample(n_patches=self.n_patches, wc=wc, ww=ww)
                    sample_2_data = scan.train_sample(n_patches=self.n_patches, wc=wc, ww=ww)

                    patches_1 = sample_1_data['normalized_patches']
                    patches_2 = sample_2_data['normalized_patches']
                    patch_coords_1 = sample_1_data['patch_centers_pt'] - sample_1_data['subset_center_pt']
                    patch_coords_2 = sample_2_data['patch_centers_pt'] - sample_2_data['subset_center_pt']

                    label = sample_2_data['subset_center_pt'] - sample_1_data['subset_center_pt']
                    label = label.squeeze(0)

                    patches_1 = torch.from_numpy(patches_1).to(torch.float32)
                    patches_2 = torch.from_numpy(patches_2).to(torch.float32)
                    patch_coords_1 = torch.from_numpy(patch_coords_1).to(torch.float32)
                    patch_coords_2 = torch.from_numpy(patch_coords_2).to(torch.float32)
                    label = torch.from_numpy(label).to(torch.float32)

                    yield patches_1, patches_2, patch_coords_1, patch_coords_2, row_id, label, sample_1_data, sample_2_data, {"path_to_scan": path_to_load, "median": median, "stdev": stdev, "patch_shape": self.patch_shape}


    # Data parameters (MUST match between dataset and model)
    PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
    N_PATCHES = 64            # Number of patches to sample from each scan

    # setup data
    dataset = PrismOrderingDataset(
        patch_shape=PATCH_SHAPE,
        n_patches=N_PATCHES,
    )

    NUM_WORKERS=0
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=NUM_WORKERS, # Use 2 worker processes to load data in parallel
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=True,
    )

    iterator = iter(dataloader)
    # include a validation set
    return (iterator,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run")
    run_btn
    return (run_btn,)


@app.cell
def _(iterator, run_btn):
    if run_btn.value:
        batch = next(iterator)
        patches_1, patches_2, patch_coords_1, patch_coords_2, row_id, label, sample_1_data, sample_2_data, scan_metadata = batch
    return (
        label,
        patch_coords_1,
        patch_coords_2,
        patches_1,
        patches_2,
        row_id,
        sample_1_data,
        sample_2_data,
        scan_metadata,
    )


@app.cell
def _(
    RadiographyEncoder,
    label,
    patch_coords_1,
    patch_coords_2,
    patches_1,
    patches_2,
    row_id,
    torch,
):
    model = RadiographyEncoder.load_from_checkpoint("/gpfs/fs001/cbica/home/gangarav/rsna25/checkpoints/None/best_model_all.ckpt")
    model.eval()
    device = next(model.parameters()).device

    embeddings = []
    locations = []

    b = patches_1, patches_2, patch_coords_1, patch_coords_2, row_id, label
    b = tuple(t.to(device) for t in b)
    with torch.no_grad():
        print(model.training_step(b, 0))
    print(label)
    return


@app.cell
def _(mo, sample_1_data, sample_2_data, scan_metadata, torch, zarr_scan):
    _scan = zarr_scan(
            path_to_scan=scan_metadata["path_to_scan"][0],
            median=scan_metadata["median"][0],
            stdev=scan_metadata["stdev"][0],
            patch_shape=scan_metadata["patch_shape"][0]
        )
    _px = _scan.get_scan_array_copy()

    _px = _scan.create_rgb_scan_with_boxes(
        _px,
        [torch.stack(sample_1_data["subset_start"]).reshape(3).cpu().numpy().astype(int)],
        torch.stack(sample_1_data["subset_shape"]).reshape(3).cpu().numpy().astype(int),
        color=(255, 0, 0)
    )

    _px = _scan.create_rgb_scan_with_boxes(
        _px,
        sample_1_data["patch_indices"].squeeze(),
        (1, 16, 16),
        color=(0, 255, 0)
    )

    _px = _scan.create_rgb_scan_with_boxes(
        _px,
        [torch.stack(sample_2_data["subset_start"]).reshape(3).cpu().numpy().astype(int)],
        torch.stack(sample_2_data["subset_shape"]).reshape(3).cpu().numpy().astype(int),
        color=(255, 0, 0)
    )

    pox = _scan.create_rgb_scan_with_boxes(
        _px,
        sample_2_data["patch_indices"].squeeze(),
        (1, 16, 16),
        color=(0, 255, 0)
    )

    sloder = mo.ui.slider(start=0, stop=_px.shape[0]-1, value=sample_1_data["subset_center_idx"][0,0,0].item())
    sloder
    return pox, sloder


@app.cell
def _(label, mo, pox, sample_1_data, sample_2_data, sloder):
    mo.hstack([
        mo.image(src=pox[sloder.value], width=256),
        mo.image(src=pox[sample_1_data["subset_center_idx"][0,0,0].item()], width=256),
        mo.image(src=pox[sample_2_data["subset_center_idx"][0,0,0].item()], width=256),
        [label[0,2].item(),
        label[0,0].item(),
        label[0,1].item()]
    ], justify="start", gap=0)
    return


@app.cell
def _(centers, mo, patches, pox, sloder):
    mo.hstack([
        mo.vstack([
            mo.image(src=pox[sloder.value], width=512)
        ]),
        mo.vstack([
            mo.hstack([mo.image(src=patches[0,_i+_j*8,0], height=32, width=32) for _i in range(8)], justify="start", gap=0) for _j in range(8)
        ] + [patches.max(),patches.min(),centers.max(),centers.min()], gap=0)
    ], widths=[512, 512])
    return


@app.cell
def _():
    return


@app.cell
def _():
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # Filter data for Right and Left Middle Cerebral Artery
    # filtered_indices = [i for i, loc in enumerate(locations) 
    #                    if loc in ["Right Middle Cerebral Artery", "Left Middle Cerebral Artery"]]
    # filtered_embeddings = embeddings[filtered_indices]
    # filtered_locations = [locations[i] for i in filtered_indices]

    # # Perform 3D PCA
    # pca = PCA(n_components=3)
    # embeddings_3d = pca.fit_transform(filtered_embeddings)

    # # Create color mapping for locations
    # unique_locations = list(set(filtered_locations))
    # color_map = {loc: plt.cm.Set3(i / len(unique_locations)) 
    #              for i, loc in enumerate(unique_locations)}
    # colors = [color_map[loc] for loc in filtered_locations]

    # # Create 3D scatter plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(embeddings_3d[:, 0], 
    #                     embeddings_3d[:, 1], 
    #                     embeddings_3d[:, 2],
    #                     c=colors,
    #                     alpha=0.7,
    #                     s=50)

    # # Add legend for locations
    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
    #                                markerfacecolor=color_map[loc], 
    #                                markersize=10, label=loc) 
    #                   for loc in unique_locations]
    # ax.legend(handles=legend_elements, title='Locations', bbox_to_anchor=(1.05, 1), loc='upper left')

    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # ax.set_title('3D PCA of Embeddings Colored by Location')

    # plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
