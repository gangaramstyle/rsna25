import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    import numpy as np
    import os
    import random
    import marimo as mo
    import zarr
    import pandas as pd
    import fastparquet
    import matplotlib.pyplot as plt


@app.cell
def _():
    metadata = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
    table = mo.ui.table(metadata, selection='single')
    table
    return (table,)


@app.cell
def _(scan):
    q = scan.get_scan_array_copy().flatten()

    # Remove top 5 most frequent values
    unique, counts = np.unique(q, return_counts=True)
    top_5_indices = np.argsort(counts)[-5:][::-1]
    top_5_values = unique[top_5_indices]

    # Create mask to exclude top 5 values
    mask = ~np.isin(q, top_5_values)
    q_filtered = q[mask]

    # Create histogram of remaining pixel values
    plt.figure(figsize=(10, 6))
    plt.hist(q_filtered, bins=100, alpha=0.7, color='steelblue', edgecolor='black', range=(-32768, 32767))
    median_val = np.median(q_filtered)
    plt.axvline(median_val, color='crimson', linestyle='--', linewidth=2, label=f'Median (excl. top 5): {median_val:.1f}')

    plt.title('Histogram of Pixel Values in Scan (Excluding Top 5 Most Frequent)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim(-32768, 32767)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca()

    return


@app.cell
def _():
    # Create text input fields for min and max values
    min_textbox = mo.ui.text(
        label="Min value for clipping",
        value="1000",
        kind="text"
    )

    max_textbox = mo.ui.text(
        label="Max value for clipping", 
        value="1100",
        kind="text"
    )

    # Display the textboxes
    mo.vstack([
        mo.md("### Set clipping values for scan visualization"),
        min_textbox,
        max_textbox
    ])

    return max_textbox, min_textbox


@app.cell
def _(max_textbox, min_textbox, table):
    metadata_row = table.value
    zarr_name = metadata_row["zarr_path"].values[0]
    scan = zarr_scan(path_to_scan=zarr_name)
    subset_1_start, subset_1_shape = scan.get_random_subset_from_scan()
    idxs_1 = scan.get_random_patch_indices_from_scan_subset(subset_1_start, subset_1_shape, 50)
    patches_1 = scan.get_patches_from_indices(idxs_1)
    patches_1_pt_space = scan.convert_indices_to_patient_space(idxs_1)
    subset_1_center = np.array(subset_1_start) + 0.5*np.array(subset_1_shape)
    subset_1_center = subset_1_center.astype(int)[np.newaxis, :]
    subset_1_center_pt_space = scan.convert_indices_to_patient_space(subset_1_center)

    subset_2_start, subset_2_shape = scan.get_random_subset_from_scan()
    idxs_2 = scan.get_random_patch_indices_from_scan_subset(subset_2_start, subset_2_shape, 50)
    patches_2 = scan.get_patches_from_indices(idxs_2)
    patches_2_pt_space = scan.convert_indices_to_patient_space(idxs_2)
    subset_2_center = np.array(subset_2_start) + 0.5*np.array(subset_2_shape)
    subset_2_center = subset_2_center.astype(int)[np.newaxis, :]
    subset_2_center_pt_space = scan.convert_indices_to_patient_space(subset_2_center)


    scan_pixels = np.clip(scan.get_scan_array_copy(), int(min_textbox.value), int(max_textbox.value))
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [subset_1_start],
        subset_1_shape,
        (255, 0, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        idxs_1,
        [1, 16, 16],
        (0, 255, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [subset_2_start],
        subset_2_shape,
        (0, 0, 255)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        idxs_2,
        [1, 16, 16],
        (0, 255, 0)
    )

    slider = mo.ui.slider(start=0, stop=scan_pixels.shape[0]-1,value=int(scan_pixels.shape[0]/2))
    return scan, scan_pixels, slider


@app.cell
def _(slider):
    slider
    return


@app.cell
def _(scan_pixels, slider):
    mo.image(scan_pixels[slider.value], width=512)
    return


@app.cell
def _():
    return


@app.class_definition
class zarr_scan():

    def __init__(self, path_to_scan, patch_shape=(1, 16, 16)):
        super().__init__()
        self.zarr_store = zarr.open(path_to_scan, mode='r')
        self.patch_shape = patch_shape

    def get_scan_array_copy(self):
        return self.zarr_store['pixel_data'][:]

    def create_rgb_scan_with_boxes(
        self,
        array,
        box_indices: np.ndarray,
        box_size: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1
    ):
        gray_volume = array.astype(np.float32)
        if len(gray_volume.shape) == 3:
            d, h, w = gray_volume.shape

            vol_min = gray_volume.min()
            vol_max = gray_volume.max()

            value_range = vol_max - vol_min if vol_max > vol_min else 1
            norm_volume = (255 * (gray_volume - vol_min) / value_range).astype(np.uint8)

            rgb_volume = np.stack([norm_volume] * 3, axis=-1)
        else:
            # gray_volume is actually rgb volume
            rgb_volume = gray_volume
            d, h, w, c = rgb_volume.shape

        patch_d, patch_h, patch_w = box_size

        for pz, py, px in box_indices:
            # Define the box boundaries, clamping to the array dimensions
            z_min, z_max = max(0, pz), min(d, pz + patch_d)
            y_min, y_max = max(0, py), min(h, py + patch_h)
            x_min, x_max = max(0, px), min(w, px + patch_w)

            # Draw the 12 edges of the cube directly into the Zarr array
            t = thickness # shorthand

            # 4 edges along the Z-axis (vertical pillars)
            rgb_volume[z_min:z_max, y_min:y_min+t, x_min:x_max] = color
            rgb_volume[z_min:z_max, y_min:y_max, x_min:x_min+t] = color
            rgb_volume[z_min:z_max, y_max-t:y_max, x_min:x_max] = color
            rgb_volume[z_min:z_max, y_min:y_max, x_max-t:x_max] = color

        return rgb_volume


    def get_random_subset_from_scan(self, min_subset_shape=(4,32,32), max_subset_shape=None, percent_outer_to_ignore=(0.0, 0.2, 0.2)):
        array_shape = self.zarr_store['pixel_data'].shape
        depth, cols, rows = array_shape

        if not all(array_shape[i] >= self.patch_shape[i] for i in range(3)):
            raise ValueError("array_shape must be larger than or equal to patch_shape in all dimensions.")

        if max_subset_shape is None:
            max_subset_shape = tuple(dim // 4 for dim in array_shape)

        # Ensure min/max subset shapes are valid
        print(min_subset_shape)
        print(self.patch_shape)
        print(max_subset_shape)
        if not all(min_subset_shape[i] >= self.patch_shape[i] for i in range(3)):
            raise ValueError("min_subset_shape must be >= patch_shape in all dimensions.")
        if not all(max_subset_shape[i] >= min_subset_shape[i] for i in range(3)):
            raise ValueError("max_subset_shape must be >= min_subset_shape in all dimensions.")

        d_buffer, c_buffer, r_buffer = percent_outer_to_ignore
        d_buffer = depth * d_buffer
        c_buffer = cols * c_buffer
        r_buffer = rows * r_buffer

        # a) Determine the random size of the subset
        # The size must be between min_subset_shape and min(max_subset_shape, array_shape)
        subset_d = np.random.randint(min_subset_shape[0], min(max_subset_shape[0], depth - 2*d_buffer) + 1)
        subset_c = np.random.randint(min_subset_shape[1], min(max_subset_shape[1], cols - 2*c_buffer) + 1)
        subset_r = np.random.randint(min_subset_shape[2], min(max_subset_shape[2], rows - 2*r_buffer) + 1)
        subset_shape = (subset_d, subset_c, subset_r)

        # b) Determine the random starting position of the subset
        # The starting point must allow the subset to fit entirely within the main array.
        max_start_d = depth - subset_d - d_buffer
        max_start_c = cols - subset_c - c_buffer
        max_start_r = rows - subset_r - r_buffer

        subset_start_d = np.random.randint(d_buffer, max_start_d + 1)
        subset_start_c = np.random.randint(c_buffer, max_start_c + 1)
        subset_start_r = np.random.randint(r_buffer, max_start_r + 1)
        subset_start = (subset_start_d, subset_start_c, subset_start_r)

        return subset_start, subset_shape

    def get_random_patch_indices_from_scan_subset(self, subset_start, subset_shape, n_patches):
        patch_d, patch_c, patch_r = self.patch_shape
        subset_start_d, subset_start_c, subset_start_r = subset_start
        subset_d, subset_c, subset_r = subset_shape

        # The starting index of a patch must allow it to fit entirely within the subset.
        max_patch_start_d = subset_d - patch_d
        max_patch_start_c = subset_c - patch_c
        max_patch_start_r = subset_r - patch_r

        # Generate N random starting indices *relative to the subset's corner*
        # This is much more efficient than a for-loop.
        patch_starts_relative_d = np.random.randint(0, max_patch_start_d + 1, size=n_patches)
        patch_starts_relative_c = np.random.randint(0, max_patch_start_c + 1, size=n_patches)
        patch_starts_relative_r = np.random.randint(0, max_patch_start_r + 1, size=n_patches)

        # --- 3. Convert Patch Indices to Absolute Coordinates ---

        # Add the subset's starting offset to get the final indices in the main array's space.
        absolute_starts_d = subset_start_d + patch_starts_relative_d
        absolute_starts_c = subset_start_c + patch_starts_relative_c
        absolute_starts_r = subset_start_r + patch_starts_relative_r

        # Stack the coordinates into an (N, 3) array for easy use.
        # The format is (z, y, x) which corresponds to (depth, cols, rows).
        patch_indices = np.column_stack((
            absolute_starts_d,
            absolute_starts_c,
            absolute_starts_r
        ))
        return patch_indices

    def get_patches_from_indices(self, patch_indices):
        n_patches = patch_indices.shape[0]
        patch_d, patch_c, patch_r = self.patch_shape

        # 1. Create index offsets for a single patch
        # d_range.shape -> (1, 1, 1) or (1,) for patch_d=1
        # c_range.shape -> (1, 16, 1)
        # r_range.shape -> (1, 1, 16)
        d_range = np.arange(patch_d).reshape(1, -1, 1, 1)
        c_range = np.arange(patch_c).reshape(1, 1, -1, 1)
        r_range = np.arange(patch_r).reshape(1, 1, 1, -1)

        # 2. Get the starting indices for each patch
        # start_indices.shape -> (N, 1, 1, 1)
        start_indices = patch_indices.reshape(n_patches, -1, 1, 1, 1)
        start_d, start_c, start_r = start_indices[:, 0], start_indices[:, 1], start_indices[:, 2]

        # 3. Broadcast starting indices with offsets to get absolute indices for all patches
        # This creates a grid of indices for each dimension.
        # final_d.shape -> (N, 1, 1, 1)
        # final_c.shape -> (N, 1, 16, 1)
        # final_r.shape -> (N, 1, 1, 16)
        final_d = start_d + d_range
        final_c = start_c + c_range
        final_r = start_r + r_range

        # 4. Use advanced indexing to gather all patches at once
        # NumPy broadcasts the index arrays to a common shape (N, 1, 16, 16)
        # and extracts the corresponding elements.
        return self.zarr_store['pixel_data'][final_d, final_c, final_r]

    def convert_indices_to_patient_space(self, patch_indices):
        affine_matrix = self.zarr_store['slice_affines']

        affine_matrix_indices = patch_indices[:, 0]
        selected_affine_matrices = affine_matrix[affine_matrix_indices] # Shape: (4, 4, 3)
        coords_rc = patch_indices[:, 1:]
        ones_column = np.ones((coords_rc.shape[0], 1))
        homogeneous_coords = np.hstack([coords_rc, ones_column]) # Shape: (4, 3)

        # Reshape homogeneous_coords to (4, 3, 1) to enable matmul broadcasting
        reshaped_coords = homogeneous_coords[:, :, np.newaxis]

        # Perform the batched matrix multiplication
        matmul_results = selected_affine_matrices @ reshaped_coords
        # The result is of shape (4, 4, 1), so we squeeze out the last dimension
        matmul_results = np.squeeze(matmul_results, axis=2)
        matmul_results = matmul_results/100.0 #/100 to normalize to 10 cm increments

        return matmul_results[:, :3]


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
