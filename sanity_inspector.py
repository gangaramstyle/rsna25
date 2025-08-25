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
    from lightning.pytorch.callbacks import ModelCheckpoint
    from rvt_model import RvT, PosEmbedding3D
    from torch.utils.data import DataLoader, IterableDataset
    from typing import Optional, Tuple, Dict, Any
    from data_loader import zarr_scan
    import pandas as pd
    import numpy as np
    import random
    import time
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from lightning_train import PrismOrderingDataset
    return PrismOrderingDataset, mo, os, torch, zarr_scan


@app.cell
def _(PrismOrderingDataset):
    PATCH_SHAPE = (1, 16, 16)
    N_PATCHES = 64
    METADATA_PATH = '/cbica/home/gangarav/data_25_processed/metadata.parquet'
    scratch_dir = "/scratch/gangarav/"

    dataset = PrismOrderingDataset(
        metadata=METADATA_PATH,
        patch_shape=PATCH_SHAPE,
        n_patches=N_PATCHES,
        position_space='patient',
        n_aux_patches=10,
        scratch_dir=scratch_dir,
        n_sampled_from_same_study=4
    )
    return (dataset,)


@app.cell
def _(dataset, mo):
    table = mo.ui.table(dataset.metadata, selection='single')
    table
    return (table,)


@app.cell
def _(mo, table, zarr_scan):
    row = table.value.iloc[0]
    scan = zarr_scan(
        path_to_scan=row["zarr_path"],
        median=row["median"],
        stdev=row["stdev"]
    )
    run_btn = mo.ui.run_button(label="Run")
    run_btn
    return run_btn, scan


@app.cell
def _(dataset, run_btn, scan):
    if run_btn.value:
        batch, sample_1, sample_2, aux_1, aux_2 = dataset.generate_training_pair(
            scan,
            n_patches=8,
            n_aux_patches=4,
            position_space='pixel',
            debug=True
        )
        batch
    return aux_1, aux_2, batch, sample_1, sample_2


@app.cell
def _(sample_1):
    sample_1
    return


@app.cell
def _(aux_1, aux_2, batch, mo, sample_1, sample_2):
    mo.vstack([
        mo.md(f"patch 1 (shape, min, max) - {sample_1['subset_center_idx'][0]}"),
        mo.hstack([batch[0].shape, round(batch[0].min().item(), 2), round(batch[0].max().item(), 2)], justify="start"),
        batch[2],
        mo.md(f"patch 2 (shape, min, max) - {sample_2['subset_center_idx'][0]}"),
        mo.hstack([batch[1].shape, round(batch[1].min().item(), 2), round(batch[1].max().item(), 2)], justify="start"),
        batch[3],
        mo.md(f"aux 1 (shape, min, max) - {aux_1['subset_center_idx'][0]}"),
        mo.hstack([batch[4].shape, round(batch[4].min().item(), 2), round(batch[4].max().item(), 2)], justify="start"),
        batch[6],
        mo.md(f"aux 2 (shape, min, max) - {aux_2['subset_center_idx'][0]}"),
        mo.hstack([batch[5].shape, round(batch[5].min().item(), 2), round(batch[5].max().item(), 2)], justify="start"),
        batch[7],
    ])
    return


@app.cell
def _(batch, mo, px, slider):
    mo.hstack([
        mo.image(src=px[slider.value]),
        mo.vstack([
            mo.md("Patches Sample 1"),
            mo.hstack([mo.image(src=_patch[0], width=32, height=32) for _patch in batch[0]]),
            mo.md("Patches Sample 2"),
            mo.hstack([mo.image(src=_patch[0], width=32, height=32) for _patch in batch[1]]),
            mo.md("Aux Sample 1"),
            mo.hstack([mo.image(src=_patch[0], width=32, height=32) for _patch in batch[4]]),
            mo.md("Aux Sample 2"),
            mo.hstack([mo.image(src=_patch[0], width=32, height=32) for _patch in batch[5]]),
        ])
    ])
    return


@app.cell
def _(aux_1, aux_2, mo, sample_1, sample_2, scan):
    px = scan.get_scan_array_copy()

    px = scan.create_rgb_scan_with_boxes(
        px,
        [sample_1["subset_start"]],
        sample_1["subset_shape"],
        color=(255, 0, 0)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        sample_1["patch_indices"],
        [1, 16, 16],
        color=(0, 255, 0)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        [sample_2["subset_start"]],
        sample_2["subset_shape"],
        color=(0, 0, 255)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        sample_2["patch_indices"],
        [1, 16, 16],
        color=(0, 255, 0)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        [aux_1["subset_start"]],
        aux_1["subset_shape"],
        color=(255, 100, 100)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        aux_1["patch_indices"],
        [1, 16, 16],
        color=(0, 255, 0)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        [aux_2["subset_start"]],
        aux_2["subset_shape"],
        color=(100, 100, 255)
    )

    px = scan.create_rgb_scan_with_boxes(
        px,
        aux_2["patch_indices"],
        [1, 16, 16],
        color=(0, 255, 0)
    )

    slider = mo.ui.slider(steps=range(px.shape[0]), value=sample_1["subset_start"][0])
    slider
    return px, slider


@app.cell
def _():
    # if run_btn.value:
    #     batch = next(iterator)
    #     patches_1, patches_2, patch_coords_1, patch_coords_2, aux_patches_1, aux_patches_2, aux_coords_1, aux_coords_2, label, row_id = batch
    return


@app.cell
def _(scan_metadata):
    scan_metadata
    return


@app.cell
def _():
    # scan_metadata = dataset.metadata.iloc[row_id]
    # scan = zarr_scan(
    #     path_to_scan=scan_metadata["zarr_path"],
    #     median=scan_metadata["median"],
    #     stdev=scan_metadata["stdev"],
    # )

    # px = scan.get_scan_array_copy()

    # px = scan.create_rgb_scan_with_boxes(
    #     px,
    #     [patches_1],
    #     torch.stack(sample_1_data["subset_shape"]).reshape(3).cpu().numpy().astype(int),
    #     color=(255, 0, 0)
    # )

    # _px = _scan.create_rgb_scan_with_boxes(
    #     _px,
    #     sample_1_data["patch_indices"].squeeze(),
    #     (1, 16, 16),
    #     color=(0, 255, 0)
    # )

    # _px = _scan.create_rgb_scan_with_boxes(
    #     _px,
    #     [torch.stack(sample_2_data["subset_start"]).reshape(3).cpu().numpy().astype(int)],
    #     torch.stack(sample_2_data["subset_shape"]).reshape(3).cpu().numpy().astype(int),
    #     color=(255, 0, 0)
    # )

    # pox = _scan.create_rgb_scan_with_boxes(
    #     _px,
    #     sample_2_data["patch_indices"].squeeze(),
    #     (1, 16, 16),
    #     color=(0, 255, 0)
    # )

    # sloder = mo.ui.slider(start=0, stop=_px.shape[0]-1, value=sample_1_data["subset_center_idx"][0,0,0].item())
    # sloder
    # px = scan.get_scan_array_copy()
    return


@app.cell
def _(RadiographyEncoder, os, torch):
    def get_model(RUN_ID):
        CHECKPOINT_PATH = f"/cbica/home/gangarav/checkpoints/{RUN_ID}/last.ckpt"

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

        # --- Step 4: Load the Model from the Checkpoint ---
        print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")

        # This is the magic command. It will automatically read the hyperparameters
        # saved in the .ckpt file and instantiate the model with them.
        model = RadiographyEncoder.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

        model.eval()

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    model = get_model('3d640eau')
    return (model,)


@app.cell
def _(
    label,
    model,
    patch_coords_1,
    patch_coords_2,
    patches_1,
    patches_2,
    row_id,
    torch,
):
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
def _(label):
    label
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
