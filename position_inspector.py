import marimo

__generated_with = "0.14.17"
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
    import marimo as mo
    import pydicom
    return mo, pd, zarr_scan


@app.cell
def _(mo, pd):
    label_table = mo.ui.table(pd.read_parquet('aneurysm_labels_cleaned_6_64_64.parquet'), selection='single')
    label_table
    return (label_table,)


@app.cell
def _(label_table, zarr_scan):
    row = label_table.value.iloc[0]
    z = row['aneurysm_z']
    y = row['aneurysm_y']
    x = row['aneurysm_x']
    s = zarr_scan(path_to_scan=row['zarr_path'], median=row['median'], stdev=row['stdev'])
    return s, x, y, z


@app.cell
def _(mo, s, x, y, z):
    box_half_size = 32
    box_size = 2 * box_half_size
    blue_color = (0, 0, 255)
    red_color = (255, 0, 0)

    sampled_data = s.train_sample(
        n_patches=64,
        subset_start=(z-3, y-box_half_size, x-box_half_size),
        subset_shape=(6, box_size, box_size),
    )

    px = s.get_scan_array_copy()

    px = s.create_rgb_scan_with_boxes(
        px,
        [sampled_data["subset_start"]],
        sampled_data["subset_shape"],
        color=blue_color
    )

    px = s.create_rgb_scan_with_boxes(
        px,
        sampled_data["patch_indices"],
        (1, 16, 16),
        color=(0, 256, 0)
    )

    slider = mo.ui.slider(start=0, stop=px.shape[0]-1, value=z)
    slider
    return px, sampled_data, slider


@app.cell
def _(mo, px, slider):
    mo.vstack([
        mo.image(src=px[slider.value], width=512)
    ])
    return


@app.cell
def _(mo, sampled_data):
    mo.vstack([
        mo.hstack([mo.image(src=sampled_data["normalized_patches"][_i+_j*7,0], width=32) for _i in range(8)], justify="start", gap=0) for _j in range(8)
    ], gap=0)
    return


if __name__ == "__main__":
    app.run()
