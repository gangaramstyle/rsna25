import marimo

__generated_with = "0.14.16"
app = marimo.App(width="columns")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
    from math import pi
    from einops import rearrange, repeat
    import numpy as np
    import os
    import random
    import marimo as mo
    import zarr
    import pandas as pd
    import fastparquet
    import matplotlib.pyplot as plt
    from typing import Optional, Tuple, Dict, Any
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


@app.cell
def _(metadata_row):
    zarr_name = metadata_row["zarr_path"].values[0]
    patch_shape = [1, 16, 16]
    # n_patches
    scan = zarr_scan(path_to_scan=zarr_name, median=metadata_row["median"].values[0], stdev=metadata_row["stdev"].values[0], patch_shape=patch_shape)
    return patch_shape, scan


@app.cell
def _(scan):
    sample = scan.train_sample(128)
    return (sample,)


@app.cell
def _():
    # dim = 6*100
    # model = PosEmbedding3D(dim, max_freq=3)

    # # Get min and max for each axis
    # min_vals = (sample["patch_centers_idx"] - sample["subset_center_idx"]).min(axis=0)
    # max_vals = (sample["patch_centers_idx"] - sample["subset_center_idx"]).max(axis=0)
    # print(min_vals, max_vals)
    # # Create smooth ranges for each axis
    # n_points = 100
    # x_range = torch.linspace(min_vals[0], max_vals[0], n_points)
    # y_range = torch.linspace(min_vals[1], max_vals[1], n_points)
    # z_range = torch.linspace(min_vals[2], max_vals[2], n_points)

    # # Create 100x3 tensor
    # new_array = torch.stack([x_range, y_range, z_range], dim=1)
    # new_array = new_array.unsqueeze(0)  # Shape: (1, 100, 3)

    # sin, cos = model(new_array)

    # mo.image(src=sin[0], width=512)
    return


@app.cell
def _(sample):
    sample
    return


@app.cell
def _(scan_pixels, slider):
    mo.image(scan_pixels[slider.value], width=512)
    return


@app.cell
def _(patch_shape, sample, scan):
    scan_pixels = scan.normalize_pixels_to_range(scan.get_scan_array_copy(), sample["wc"] - 0.5*sample["ww"], sample["wc"] + 0.5*sample["ww"])

    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [sample["subset_start"]],
        sample["subset_shape"],
        None,
        (255, 0, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        sample["patch_indices"],
        patch_shape,
        None,
        (0, 255, 0)
    )

    slider = mo.ui.slider(start=0, stop=scan_pixels.shape[0]-1,value=int(sample["subset_center_idx"][0,0]))
    slider
    return scan_pixels, slider


@app.class_definition
class zarr_scan():

    def __init__(self, path_to_scan, median, stdev, patch_shape=(1, 16, 16)):
        self.zarr_store = zarr.open(path_to_scan, mode='r')
        self.patch_shape = np.array(patch_shape)
        self.med = median
        self.std = stdev

    def generate_training_pair(self, n_patches: int, to_torch: bool = True) -> tuple:
        """
        A high-level helper for training loops that generates a pair of samples.
        """

        wc, ww = self.get_random_wc_ww_for_scan()
        sample_1_data = self.train_sample(n_patches=n_patches, wc=wc, ww=ww)
        sample_2_data = self.train_sample(n_patches=n_patches, wc=wc, ww=ww)

        patches_1 = sample_1_data['normalized_patches']
        patches_2 = sample_2_data['normalized_patches']
        patch_coords_1 = sample_1_data['patch_centers_idx'] - sample_1_data['subset_center_idx']
        patch_coords_2 = sample_2_data['patch_centers_idx'] - sample_2_data['subset_center_idx']

        label = sample_2_data['subset_center_idx'] - sample_1_data['subset_center_idx']

        if to_torch:
            patches_1 = torch.from_numpy(patches_1).to(torch.float32)
            patches_2 = torch.from_numpy(patches_2).to(torch.float32)
            patch_coords_1 = torch.from_numpy(patch_coords_1).to(torch.float32)
            patch_coords_2 = torch.from_numpy(patch_coords_2).to(torch.float32)
            label = torch.from_numpy(label).to(torch.float32)

        return patches_1, patches_2, patch_coords_1, patch_coords_2, label, sample_1_data, sample_2_data

    def train_sample(
        self,
        n_patches: int,
        *, # Force subsequent arguments to be keyword-only for clarity
        subset_start: Optional[Tuple[int, int, int]] = None,
        subset_shape: Optional[Tuple[int, int, int]] = None,
        patch_indices: Optional[np.ndarray] = None,
        wc: Optional[float] = None,
        ww: Optional[float] = None
    ) -> Dict[str, Any]:
        results = {}

        if wc is None or ww is None:
            wc, ww = self.get_random_wc_ww_for_scan()
        results['wc'], results['ww'] = wc, ww
        results['w_min'], results['w_max'] = wc - 0.5 * ww, wc + 0.5 * ww

        if patch_indices is not None:
            results['patch_indices'] = patch_indices
            n_patches = patch_indices.shape[0]
            # If indices are provided, we don't know the parent subset
        else:
            if subset_start is None or subset_shape is None:
                subset_start, subset_shape = self.get_random_subset_from_scan(n_patches)
            results['subset_start'], results['subset_shape'] = subset_start, subset_shape

            patch_indices = self.get_stratified_random_patch_indices(
                subset_start, subset_shape, n_patches
            )
            results['patch_indices'] = patch_indices

        if 'subset_start' in results and 'subset_shape' in results:
            ss_start = np.array(results['subset_start'])
            ss_shape = np.array(results['subset_shape'])
            subset_center_idx = (ss_start + 0.5 * ss_shape).astype(int)[np.newaxis, :]
            results['subset_center_idx'] = subset_center_idx
            results['subset_center_pt'] = self.convert_indices_to_patient_space(subset_center_idx)

        raw_patches = self.get_patches_from_indices(results['patch_indices'])
        results['raw_patches'] = raw_patches

        normalized_patches = self.normalize_pixels_to_range(
            raw_patches, results['w_min'], results['w_max']
        )
        results['normalized_patches'] = normalized_patches

        patch_centers_idx = (results['patch_indices'] + 0.5 * self.patch_shape).astype(int)
        results['patch_centers_idx'] = patch_centers_idx

        patch_centers_pt = self.convert_indices_to_patient_space(patch_centers_idx)
        results['patch_centers_pt'] = patch_centers_pt

        return results

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
        # Ensure w_max is greater than w_min to avoid division by zero
        if w_max <= w_min:
            w_max = w_min + 1e-6

        clipped_array = np.clip(pixel_array, w_min, w_max)
        scaled_01 = (clipped_array - w_min) / (w_max - w_min)
        out_min, out_max = out_range
        return scaled_01 * (out_max - out_min) + out_min


    def get_random_subset_from_scan(self, n_patches: int, min_subset_shape=(4,32,32), max_subset_shape=(10,128,128), percent_outer_to_ignore=(0.05, 0.2, 0.2), volume_scale_factor = 1.0):
        """
        Selects a random subset from the scan, with its shape determined by the
        number of patches requested. The volume of the subset will be approximately
        n_patches * patch_volume, while respecting min/max shape constraints.
        """
        array_shape = np.array(self.zarr_store['pixel_data'].shape)

        if not all(array_shape >= self.patch_shape):
            raise ValueError("array_shape must be larger than or equal to patch_shape in all dimensions.")

        # 1. Define bounds for subset shape based on scan size and constraints
        d_buffer = int(array_shape[0] * percent_outer_to_ignore[0])
        r_buffer = int(array_shape[1] * percent_outer_to_ignore[1])
        c_buffer = int(array_shape[2] * percent_outer_to_ignore[2])

        # Min shape must be at least as large as a patch
        min_bounds = np.maximum(min_subset_shape, self.patch_shape).astype(int)

        # Max shape is constrained by user input and the available scan size after buffering
        max_bounds = np.minimum(max_subset_shape, array_shape - 2 * np.array([d_buffer, r_buffer, c_buffer])).astype(int)

        if not np.all(max_bounds >= min_bounds):
             raise ValueError(f"Could not find a valid subset shape. Check min/max_subset_shape and scan dimensions. Min: {min_bounds}, Max: {max_bounds}")

        # 2. Calculate the target volume for the subset
        patch_volume = np.prod(self.patch_shape)
        target_volume = n_patches * patch_volume * volume_scale_factor

        # 3. Generate a random, flexible shape that approximates the target volume
        # We do this sequentially, constraining each subsequent dimension's random range.
        min_d, min_r, min_c = min_bounds
        max_d, max_r, max_c = max_bounds

        # Determine valid range for depth (d)
        # The range for 'd' is constrained by what is possible for 'r' and 'c'.
        d_lower_for_rand = max(min_d, target_volume / (max_r * max_c))
        d_upper_for_rand = min(max_d, target_volume / (min_r * min_c))
        # Ensure the range is valid before sampling
        if d_lower_for_rand > d_upper_for_rand:
            d_lower_for_rand = d_upper_for_rand
        subset_d = np.random.randint(int(d_lower_for_rand), int(d_upper_for_rand) + 1)

        # Determine valid range for rows (r), given the chosen depth
        target_area_rc = target_volume / subset_d
        r_lower_for_rand = max(min_r, target_area_rc / max_c)
        r_upper_for_rand = min(max_r, target_area_rc / min_c)

        if r_lower_for_rand > r_upper_for_rand:
            r_lower_for_rand = r_upper_for_rand
        subset_r = np.random.randint(int(r_lower_for_rand), int(r_upper_for_rand) + 1)

        rounded_subset_r = (subset_r // self.patch_shape[1]) * self.patch_shape[1]
        # Calculate the final dimension (c) to meet the target volume, then clip to its bounds
        subset_c_float = target_area_rc / rounded_subset_r
        subset_c = int(round(subset_c_float))
        subset_c = np.clip(subset_c, min_c, max_c)

        subset_shape = (subset_d, subset_r, subset_c)

        # print("d:", d_lower_for_rand, d_upper_for_rand, "->", subset_d)
        # print("r:", r_lower_for_rand, r_upper_for_rand, "->", subset_r)
        # print("c:", target_area_rc, subset_r, target_area_rc / subset_r, subset_r//self.patch_shape[1], "->", subset_c)
        # print("patch vol: ", n_patches * patch_volume)
        # print("target vol: ", target_volume)
        # print("actual vol: ", np.product(subset_shape))

        # 4. Determine the random starting position for the generated subset shape
        max_start_d = array_shape[0] - subset_d - d_buffer
        max_start_r = array_shape[1] - subset_r - r_buffer
        max_start_c = array_shape[2] - subset_c - c_buffer

        # Ensure start buffer is not larger than max start pos
        start_d = np.random.randint(d_buffer, max_start_d + 1)
        start_r = np.random.randint(r_buffer, max_start_r + 1)
        start_c = np.random.randint(c_buffer, max_start_c + 1)

        subset_start = (start_d, start_r, start_c)

        return subset_start, subset_shape

    def get_stratified_random_patch_indices(self, 
                                            subset_start, 
                                            subset_shape, 
                                            n_patches, 
                                            grid_density_factor=1.0, 
                                            randomness_factor=0.33):
        """
        Generates patch start indices using a stratified sampling approach.

        The method divides the subset into a 3D grid, samples grid cells
        (with replacement), and then jitters the patch location within or 
        around the selected grid cell.

        Args:
            subset_start (tuple or list): The (d, r, c) start coordinate of the subset 
                                          within the larger scan.
            subset_shape (tuple or list): The (d, r, c) shape of the subset from 
                                          which to sample.
            n_patches (int): The total number of patches to sample.
            grid_density_factor (float): Adjusts the density of the sampling grid.
                - 1.0: The volume of each grid cell is roughly `subset_volume / n_patches`.
                - > 1.0: More, smaller grid cells. Increases stratification.
                - < 1.0: Fewer, larger grid cells. Approaches pure random sampling.
            randomness_factor (float): Controls the jitter of the patch start relative
                                     to its grid cell center.
                - 0.0: The patch is perfectly centered within its random offset.
                - 1.0: The patch can be placed anywhere within its grid cell.
                - > 1.0: The patch can be placed outside its own grid cell,
                         creating overlap with neighbors.

        Returns:
            np.ndarray: An array of shape (n_patches, 3) containing the absolute
                        (d, r, c) start coordinates for each patch.
        """
        patch_d, patch_r, patch_c = self.patch_shape
        subset_start_d, subset_start_r, subset_start_c = subset_start
        subset_d, subset_r, subset_c = subset_shape

        patch_vol = patch_d * patch_r * patch_c
        subset_vol = subset_d * subset_r * subset_c

        # This prevents division by zero if patch_vol or n_patches is 0.
        if patch_vol == 0 or n_patches == 0:
            return np.empty((0, 3), dtype=int)

        grid_cell_shape = np.array([patch_d, patch_r, patch_c])

        # Number of grid cells along each dimension
        n_grid_cells = np.maximum(1, (np.array(subset_shape) / grid_cell_shape)).astype(int)

        # Recalculate the actual grid cell shape to perfectly tile the subset
        grid_cell_shape_actual = np.array(subset_shape) / n_grid_cells

        # --- 2. Generate all possible Grid Centers ---
        # Create coordinates for the center of each grid cell.
        grid_centers_d = np.arange(subset_d)
        grid_centers_r = np.linspace(grid_cell_shape_actual[1] / 2, subset_shape[1] - grid_cell_shape_actual[1] / 2, n_grid_cells[1])
        grid_centers_c = np.linspace(grid_cell_shape_actual[2] / 2, subset_shape[2] - grid_cell_shape_actual[2] / 2, n_grid_cells[2])

        # Create a meshgrid and flatten it to get a list of all grid center coordinates
        zz, yy, xx = np.meshgrid(grid_centers_d, grid_centers_r, grid_centers_c, indexing='ij')
        all_grid_centers = np.vstack([zz.ravel(), yy.ravel(), xx.ravel()]).T

        # --- 3. Sample Grid Centers with Replacement ---
        # This is the "cleaner" way to handle the probabilistic selection. Instead
        # of looping with a certain probability, we simply draw n_patches samples
        # from the list of all centers. `replace=True` allows a center to be
        # chosen multiple times, achieving your desired outcome.
        n_total_centers = len(all_grid_centers)
        if n_total_centers == 0:
            return np.empty((0, 3), dtype=int)

        rng = np.random.default_rng()
        random_indices = np.concatenate([rng.permutation(np.arange(n_total_centers)), rng.permutation(np.arange(n_total_centers))])[:n_patches]
        sampled_centers = all_grid_centers[random_indices]

        # --- 4. Sample a Patch Near Each Selected Grid Center ---
        # For each center, calculate a random offset.
        half_grid_cell = grid_cell_shape_actual / 2.0

        # The random offset is scaled by the randomness_factor
        max_offset = half_grid_cell * randomness_factor

        # Generate random offsets for each patch from a uniform distribution
        random_offsets = np.random.uniform(-max_offset, max_offset, size=(n_patches, 3))
        random_offsets[:,0] = 0 # don't need to randomize the D axis

        # The ideal start is the (center + offset) - half_patch_size
        patch_half_shape = np.array(self.patch_shape) / 2.0
        patch_half_shape[0] = 0
        relative_starts = (sampled_centers + random_offsets) - patch_half_shape

        # --- 5. Clamp Patch Starts to Valid Subset Boundaries ---
        # Ensure the patch does not go outside the subset dimensions.
        max_patch_start = np.array(subset_shape) - np.array(self.patch_shape)
        # Ensure max start is not negative (if subset is smaller than patch)
        max_patch_start = np.maximum(0, max_patch_start)

        # np.clip is perfect for this clamping operation
        clamped_relative_starts = np.clip(relative_starts, 0, max_patch_start)

        # --- 6. Convert to Absolute Coordinates and Return ---
        absolute_starts = (np.array(subset_start) + clamped_relative_starts).astype(int)

        return absolute_starts


    def get_true_random_patch_indices_from_scan_subset(self, subset_start, subset_shape, n_patches):
        patch_d, patch_r, patch_c = self.patch_shape
        subset_start_d, subset_start_r, subset_start_c = subset_start
        subset_d, subset_r, subset_c = subset_shape

        max_patch_start_d = subset_d - patch_d
        max_patch_start_r = subset_r - patch_r
        max_patch_start_c = subset_c - patch_c

        patch_starts_relative_d = np.random.randint(0, max_patch_start_d + 1, size=n_patches)
        patch_starts_relative_r = np.random.randint(0, max_patch_start_r + 1, size=n_patches)
        patch_starts_relative_c = np.random.randint(0, max_patch_start_c + 1, size=n_patches)

        absolute_starts_d = subset_start_d + patch_starts_relative_d
        absolute_starts_r = subset_start_r + patch_starts_relative_r
        absolute_starts_c = subset_start_c + patch_starts_relative_c

        patch_indices = np.column_stack((
            absolute_starts_d,
            absolute_starts_r,
            absolute_starts_c
        ))
        return patch_indices

    def get_patches_from_indices(self, patch_indices):
        n_patches = patch_indices.shape[0]
        patch_d, patch_r, patch_c = self.patch_shape

        # Use advanced indexing with broadcasting for efficiency
        # Create indexers for each dimension
        d_idx = patch_indices[:, 0, np.newaxis] + np.arange(patch_d)
        r_idx = patch_indices[:, 1, np.newaxis] + np.arange(patch_r)
        c_idx = patch_indices[:, 2, np.newaxis] + np.arange(patch_c)

        # The zarr array needs to be indexed carefully to avoid loading huge chunks
        # This approach fetches each patch individually.
        # For Zarr, it's often better to iterate if patches are far apart.
        # However, for a small number of patches, this is fine.
        patches = np.array([
            self.zarr_store['pixel_data'][d:d+patch_d, r:r+patch_r, c:c+patch_c]
            for d, r, c in patch_indices
        ])

        return patches

    def convert_indices_to_patient_space(self, patch_indices):
        affine_matrices = self.zarr_store['slice_affines'][:] # Load into memory

        # We need the affine matrix for the *starting slice* of each patch
        slice_indices = patch_indices[:, 0]
        selected_affine_matrices = affine_matrices[slice_indices]

        # Homogeneous coordinates for (r, c) -> (r, c, 1)
        # Note: DICOM standard is often (x, y) which maps to (c, r)
        # We assume indices are (d, r, c) and affines map (r, c)
        coords_rc = patch_indices[:, 1:]
        ones_column = np.ones((coords_rc.shape[0], 1))
        # Create homogeneous coordinates in (c, r, 1) order for standard affine multiplication
        homogeneous_coords = np.hstack([coords_rc[:, ::-1], ones_column])

        # Batched matrix multiplication: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
        patient_coords = np.matmul(
            selected_affine_matrices,
            homogeneous_coords[:, :, np.newaxis]
        ).squeeze(axis=2)

        return patient_coords


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
