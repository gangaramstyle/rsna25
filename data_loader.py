import marimo

__generated_with = "0.14.16"
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
    from typing import Optional
    import pydicom


@app.cell
def _():
    stats_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/zarr_stats.parquet')
    og_pd = pd.read_parquet('/cbica/home/gangarav/data_25_processed/metadata.parquet')
    merged_df = pd.merge(
        og_pd,
        stats_pd,
        on='zarr_path',
        how='left'
    )
    metadata = merged_df
    table = mo.ui.table(metadata, selection='single')
    table
    return (table,)


@app.cell
def _(table):
    metadata_row = table.value
    return (metadata_row,)


@app.cell(hide_code=True)
def _():
    # _q = 0
    # def get_patient_space_coords(row_col, image_position, image_orientation, pixel_spacing):
    #     """
    #     Calculates the 3D patient space coordinates for a given pixel (row, col).

    #     This function implements the logic described in the DICOM Standard, Part 3,
    #     Section C.7.6.2.1.1 (Image Plane Module).

    #     Args:
    #         row_col (tuple or list): The (row, col) pixel indices.
    #         image_position (list or np.ndarray): The 'Image Position (Patient)' (0020,0032) value.
    #         image_orientation (list or np.ndarray): The 'Image Orientation (Patient)' (0020,0037) value.
    #         pixel_spacing (list or np.ndarray): The 'Pixel Spacing' (0028,0030) value [RowSpacing, ColSpacing].

    #     Returns:
    #         np.ndarray: A 3-element NumPy array representing the (x, y, z) coordinates
    #                     in the patient coordinate system.
    #     """
    #     row, col = row_col

    #     # Ensure inputs are NumPy arrays for vectorized operations
    #     image_pos = np.array(image_position, dtype=float)
    #     image_orient = np.array(image_orientation, dtype=float)
    #     pixel_sp = np.array(pixel_spacing, dtype=float)

    #     # Unpack the orientation vectors
    #     # Row vector (direction of change as column index increases)
    #     row_vector = image_orient[:3]
    #     # Column vector (direction of change as row index increases)
    #     col_vector = image_orient[3:]

    #     # Unpack pixel spacing
    #     row_spacing = pixel_sp[0]
    #     col_spacing = pixel_sp[1]

    #     # Calculate the final position
    #     # Formula: StartPoint + (Movement along rows) + (Movement along columns)
    #     # Note: DICOM standard uses (i, j) for (col, row), which can be confusing.
    #     # Here we use explicit variable names to be clear.
    #     patient_coords = image_pos + (col * col_spacing * row_vector) + (row * row_spacing * col_vector)

    #     return patient_coords

    # # XYZ
    # patch_idx = [0, 0, 100]
    # dicom_dir = metadata_row["raw_path"].values[0]
    # dicom_files = sorted(
    #     [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')],
    #     key=lambda fpath: int(pydicom.dcmread(fpath, stop_before_pixels=True).InstanceNumber)
    # )
    # dicom_path = os.path.join(dicom_dir, dicom_files[patch_idx[0]])
    # ds = pydicom.dcmread(dicom_path)
    # ds

    # img_pos_patient = ds.ImagePositionPatient
    # img_orient_patient = ds.ImageOrientationPatient
    # pixel_spc = ds.PixelSpacing
    # print("image pos patient:", img_pos_patient)
    # print("image orientation patient:", img_orient_patient)
    # print("pixel_spacing:", pixel_spc)

    # pixel_coordinate = patch_idx[1:]

    # # --- Calculation ---
    # patient_coordinate_3d = get_patient_space_coords(
    #     row_col=pixel_coordinate,
    #     image_position=img_pos_patient,
    #     image_orientation=img_orient_patient,
    #     pixel_spacing=pixel_spc
    # )
    # mo.vstack([
    #     patient_coordinate_3d,
    #     mo.image(src=ds.pixel_array)
    # ])

    return


@app.cell
def _(metadata_row):
    zarr_name = metadata_row["zarr_path"].values[0]
    patch_shape = [1, 32, 32]
    scan = zarr_scan(path_to_scan=zarr_name, median=metadata_row["median"].values[0], stdev=metadata_row["stdev"].values[0], patch_shape=patch_shape)

    r_wc, r_ww = scan.get_random_wc_ww_for_scan()

    subset_1_start, subset_1_shape = scan.get_random_subset_from_scan()
    idxs_1 = scan.get_random_patch_indices_from_scan_subset(subset_1_start, subset_1_shape, 5)
    patches_1 = scan.normalize_pixels_to_range(scan.get_patches_from_indices(idxs_1), r_wc - 0.5*r_ww, r_wc + 0.5*r_ww)
    patches_1_center = (idxs_1 + 0.5*np.array(patch_shape)).astype(int)
    patches_1_pt_space = scan.convert_indices_to_patient_space(patches_1_center)
    subset_1_center = np.array(subset_1_start) + 0.5*np.array(subset_1_shape)
    subset_1_center = subset_1_center.astype(int)[np.newaxis, :]
    subset_1_center_pt_space = scan.convert_indices_to_patient_space(subset_1_center)

    subset_2_start, subset_2_shape = scan.get_random_subset_from_scan()
    idxs_2 = scan.get_random_patch_indices_from_scan_subset(subset_2_start, subset_2_shape, 5)
    patches_2 = scan.normalize_pixels_to_range(scan.get_patches_from_indices(idxs_2), r_wc - 0.5*r_ww, r_wc + 0.5*r_ww)
    patches_2_center = (idxs_2 + 0.5*np.array(patch_shape)).astype(int)
    patches_2_pt_space = scan.convert_indices_to_patient_space(patches_2_center)
    subset_2_center = np.array(subset_2_start) + 0.5*np.array(subset_2_shape)
    subset_2_center = subset_2_center.astype(int)[np.newaxis, :]
    subset_2_center_pt_space = scan.convert_indices_to_patient_space(subset_2_center)


    scan_pixels = scan.normalize_pixels_to_range(scan.get_scan_array_copy(), r_wc - 0.5*r_ww, r_wc + 0.5*r_ww)
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [subset_1_start],
        subset_1_shape,
        None,
        (255, 0, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        idxs_1,
        patch_shape,
        None,
        (0, 255, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [subset_2_start],
        subset_2_shape,
        None,
        (0, 0, 255)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        idxs_2,
        patch_shape,
        None,
        (0, 255, 0)
    )

    slider = mo.ui.slider(start=0, stop=scan_pixels.shape[0]-1,value=int(scan_pixels.shape[0]/2))
    return scan_pixels, slider


@app.cell
def _(slider):
    slider
    return


@app.cell
def _(scan_pixels, slider):
    mo.image(scan_pixels[slider.value], width=512)
    return


@app.class_definition
class zarr_scan():

    def __init__(self, path_to_scan, median, stdev, patch_shape=(1, 16, 16)):
        self.zarr_store = zarr.open(path_to_scan, mode='r')
        self.patch_shape = patch_shape
        self.med = median
        self.std = stdev

    # not to be used in training loops, this is just helpful for visualizing steps
    def get_scan_array_copy(self):
        return self.zarr_store['pixel_data'][:]

    def create_rgb_scan_with_boxes(
        self,
        array,
        box_indices: np.ndarray,
        box_size: np.ndarray,
        pixel_range: Optional[tuple[float, float]] = None,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1
    ):

        volume = array.astype(np.float32)
        if len(volume.shape) == 3:
            # Grayscale volume
            d, r, c = volume.shape

            if pixel_range is None:
                vol_min = volume.min()
                vol_max = volume.max()
            else:
                vol_min, vol_max = pixel_range

            clipped_volume = np.clip(volume, vol_min, vol_max)
            value_range = vol_max - vol_min if vol_max > vol_min else 1
            normalized = (clipped_volume - vol_min) / value_range
            norm_volume_uint8 = (normalized * 255).astype(np.uint8)
            rgb_volume = np.stack([norm_volume_uint8] * 3, axis=-1)
        else:
            # Already an RGB volume
            d, r, c, _ = volume.shape
            rgb_volume = volume

        box_d, box_r, box_c = box_size

        for bd, br, bc in box_indices:
            d_min, d_max = max(0, bd), min(d, bd + box_d)
            r_min, r_max = max(0, br), min(r, br + box_r)
            c_min, c_max = max(0, bc), min(c, bc + box_c)

            t = thickness # shorthand

            rgb_volume[d_min:d_max, r_min:r_min+t, c_min:c_max] = color
            rgb_volume[d_min:d_max, r_min:r_max, c_min:c_min+t] = color
            rgb_volume[d_min:d_max, r_max-t:r_max, c_min:c_max] = color
            rgb_volume[d_min:d_max, r_min:r_max, c_max-t:c_max] = color

        return rgb_volume

    def get_random_wc_ww_for_scan(self):
        return random.uniform(self.med-self.std, self.med+self.std), random.uniform(self.std, 6*self.std)

    def normalize_pixels_to_range(self, pixel_array, w_min, w_max, out_range=(-1.0, 1.0)):
        clipped_array = np.clip(pixel_array, w_min, w_max)
        scaled_01 = (clipped_array - w_min) / (w_max - w_min)
        out_min, out_max = out_range
        return scaled_01 * (out_max - out_min) + out_min

    def get_random_subset_from_scan(self, min_subset_shape=(4,32,32), max_subset_shape=None, percent_outer_to_ignore=(0.0, 0.2, 0.2)):
        array_shape = self.zarr_store['pixel_data'].shape
        depth, rows, cols = array_shape

        if not all(array_shape[i] >= self.patch_shape[i] for i in range(3)):
            raise ValueError("array_shape must be larger than or equal to patch_shape in all dimensions.")

        if max_subset_shape is None:
            max_subset_shape = tuple(dim // 4 for dim in array_shape)

        # Ensure min/max subset shapes are valid
        if not all(min_subset_shape[i] >= self.patch_shape[i] for i in range(3)):
            raise ValueError("min_subset_shape must be >= patch_shape in all dimensions.")
        if not all(max_subset_shape[i] >= min_subset_shape[i] for i in range(3)):
            raise ValueError("max_subset_shape must be >= min_subset_shape in all dimensions.")

        d_buffer, r_buffer, c_buffer = percent_outer_to_ignore
        d_buffer = depth * d_buffer
        r_buffer = rows * r_buffer
        c_buffer = cols * c_buffer

        # a) Determine the random size of the subset
        # The size must be between min_subset_shape and min(max_subset_shape, array_shape)
        subset_d = np.random.randint(min_subset_shape[0], min(max_subset_shape[0], depth - 2*d_buffer) + 1)
        subset_r = np.random.randint(min_subset_shape[1], min(max_subset_shape[1], rows - 2*r_buffer) + 1)
        subset_c = np.random.randint(min_subset_shape[2], min(max_subset_shape[2], cols - 2*c_buffer) + 1)
        subset_shape = (subset_d, subset_r, subset_c)

        # b) Determine the random starting position of the subset
        # The starting point must allow the subset to fit entirely within the main array.
        max_start_d = depth - subset_d - d_buffer
        max_start_r = rows - subset_r - r_buffer
        max_start_c = cols - subset_c - c_buffer

        subset_start_d = np.random.randint(d_buffer, max_start_d + 1)
        subset_start_r = np.random.randint(r_buffer, max_start_r + 1)
        subset_start_c = np.random.randint(c_buffer, max_start_c + 1)
        subset_start = (subset_start_d, subset_start_r, subset_start_c)

        return subset_start, subset_shape

    def get_random_patch_indices_from_scan_subset(self, subset_start, subset_shape, n_patches):
        patch_d, patch_r, patch_c = self.patch_shape
        subset_start_d, subset_start_r, subset_start_c = subset_start
        subset_d, subset_r, subset_c = subset_shape

        # The starting index of a patch must allow it to fit entirely within the subset.
        max_patch_start_d = subset_d - patch_d
        max_patch_start_r = subset_r - patch_r
        max_patch_start_c = subset_c - patch_c

        # Generate N random starting indices *relative to the subset's corner*
        # This is much more efficient than a for-loop.
        patch_starts_relative_d = np.random.randint(0, max_patch_start_d + 1, size=n_patches)
        patch_starts_relative_r = np.random.randint(0, max_patch_start_r + 1, size=n_patches)
        patch_starts_relative_c = np.random.randint(0, max_patch_start_c + 1, size=n_patches)

        # --- 3. Convert Patch Indices to Absolute Coordinates ---

        # Add the subset's starting offset to get the final indices in the main array's space.
        absolute_starts_d = subset_start_d + patch_starts_relative_d
        absolute_starts_r = subset_start_r + patch_starts_relative_r
        absolute_starts_c = subset_start_c + patch_starts_relative_c

        # Stack the coordinates into an (N, 3) array for easy use.
        # The format is (z, y, x) which corresponds to (depth, cols, rows).
        patch_indices = np.column_stack((
            absolute_starts_d,
            absolute_starts_r,
            absolute_starts_c
        ))
        return patch_indices

    def get_patches_from_indices(self, patch_indices):
        n_patches = patch_indices.shape[0]
        patch_d, patch_r, patch_c = self.patch_shape

        d_range = np.arange(patch_d).reshape(1, -1, 1, 1)
        r_range = np.arange(patch_r).reshape(1, 1, -1, 1)
        c_range = np.arange(patch_c).reshape(1, 1, 1, -1)

        start_indices = patch_indices.reshape(n_patches, -1, 1, 1, 1)
        start_d, start_r, start_c = start_indices[:, 0], start_indices[:, 1], start_indices[:, 2]

        final_d = start_d + d_range
        final_r = start_r + r_range
        final_c = start_c + c_range

        return self.zarr_store['pixel_data'][final_d, final_r, final_c]

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
        matmul_results = matmul_results

        return matmul_results[:, :3]


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
