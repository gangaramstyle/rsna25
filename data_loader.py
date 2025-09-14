import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")

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
    from rvt_model import PosEmbedding3D
    import nibabel as nib


@app.cell
def _():
    metadata_df = pd.read_parquet('/cbica/home/gangarav/rsna_any/rsna_2025/nifti_combined_metadata.parquet')

    #pd.read_parquet('/cbica/home/gangarav/rsna25/aneurysm_labels_cleaned_6_64_64.parquet')
    metadata_df = metadata_df[metadata_df['series_uid'] != '1.2.826.0.1.3680043.8.498.40511751565674479940947446050421785002']
    metadata_df = metadata_df[metadata_df['patient_id'].notnull()]
    metadata_df = metadata_df[~metadata_df['series_uid'].str.contains('dwi', case=False, na=False)]
    table = mo.ui.table(metadata_df, selection='single')

    table
    return metadata_df, table


@app.cell
def _():
    patch_size_input = mo.ui.text(value="16", label="Patch Size:")
    n_patches_input = mo.ui.text(value="64", label="Num Patches:")
    n_aux_patches_input = mo.ui.text(value="16", label="Num Aux Patches:")
    run_button = mo.ui.run_button()
    mo.vstack([patch_size_input, n_patches_input, n_aux_patches_input, run_button])
    return n_aux_patches_input, n_patches_input, patch_size_input, run_button


@app.cell
def _(
    metadata_df,
    n_aux_patches_input,
    n_patches_input,
    patch_size_input,
    run_button,
    table,
):
    if run_button.value:
        patch_size = int(patch_size_input.value)
        n_patches = int(n_patches_input.value)
        n_aux_patches = int(n_aux_patches_input.value)

        if table.value.empty:
            metadata_row = metadata_df.sample(1)
        else:
            metadata_row = table.value

        zarr_name = metadata_row["zarr_path"].values[0]

        patch_shape = (1, patch_size, patch_size)
        scan = zarr_scan(path_to_scan=zarr_name, median=metadata_row["median"].values[0], stdev=metadata_row["stdev"].values[0], patch_shape=patch_shape)
        sample = scan.train_sample(n_patches)
    return patch_shape, sample, scan


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
    sample["normalized_patches"].max(), sample["normalized_patches"].min()
    return


@app.cell
def _(sample, scan_pixels, slider):
    patches_for_display = sample["normalized_patches"].squeeze()
    patches_for_display[:,0,:] = -1
    patches_for_display[:,:,0] = 1

    mo.hstack([
        mo.image(np.flipud(scan_pixels[slider.value]), height=512),
        mo.image(rearrange(
            patches_for_display,
            '(h_grid w_grid) h w -> (h_grid h) (w_grid w)',
            h_grid=2
        ), height=512)
    ])
    return


@app.cell
def _(patch_shape, sample, scan):
    scan_pixels = scan.normalize_pixels_to_range(scan.get_scan_array_copy(), sample["wc"] - 0.5*sample["ww"], sample["wc"] + 0.5*sample["ww"])

    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [sample["subset_start"]],
        sample["subset_shape"],
        None,
        (100, 0, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        sample["patch_indices"],
        patch_shape,
        None,
        (0, 100, 0)
    )

    POINT_SIZE=2
    p1_r = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[1]-1-POINT_SIZE)
    p1_c = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[2]-1-POINT_SIZE)

    slider = mo.ui.slider(start=0, stop=scan_pixels.shape[0]-1,value=int(sample["subset_center_idx"][0,0]))
    mo.vstack([
        slider,
        p1_r,
        p1_c,
    ])
    return POINT_SIZE, p1_c, p1_r, scan_pixels, slider


@app.cell
def _(POINT_SIZE, p1_c, p1_r, sample, scan, scan_pixels, slider):
    # Create slice tuples for the point marker
    point_slices = [slice(None)] * 3
    point_slices[0] = slider.value
    point_slices[1] = slice(p1_r.value - POINT_SIZE, p1_r.value + POINT_SIZE)
    point_slices[2] = slice(p1_c.value - POINT_SIZE, p1_c.value + POINT_SIZE)

    drawn = scan_pixels[:].copy()
    drawn[tuple(point_slices)] = [256, 0, 0]

    point = np.array([slider.value, p1_r.value, p1_c.value])
    mo.hstack([
        mo.vstack([
            mo.image(np.flipud(drawn[slider.value]), width=512),
            scan.convert_indices_to_patient_space(point)
        ]),
        sample
    ])
    return


@app.cell
def _(sample):
    pos_emb = PosEmbedding3D(600, max_freq = 30)
    coords = torch.from_numpy(sample["patch_centers_pt"] - sample["subset_center_pt"]).unsqueeze(0).to(torch.float32)

    sin, cos = pos_emb(torch.asinh(coords))
    mo.vstack([
        mo.image(src=sin[0]),
        mo.image(src=cos[0])
    ])
    return (coords,)


@app.cell
def _(coords):
    coords.max(dim=1)[0], coords.min(dim=1)[0]
    return


@app.class_definition
class zarr_scan():

    def convert_to_scrollable_coordinates(self, coordinates):
        if self.scrollable_axis == 0:  # sagittal
            axes = [0, 2, 1]
        elif self.scrollable_axis == 1:  # coronal
            axes = [1, 2, 0]
        else:  # axial
            axes = [2, 1, 0]

        # Handle numpy arrays, torch tensors, and regular lists/tuples
        if hasattr(coordinates, 'transpose'):
            # For numpy arrays and torch tensors
            return coordinates[..., axes]
            # return coordinates.transpose(*axes)
        else:
            # For regular Python sequences
            return type(coordinates)(coordinates[i] for i in axes)

    def convert_from_scrollable_coordinates(self, coordinates):
        if self.scrollable_axis == 0:  # sagittal
            axes = [0, 2, 1]
        elif self.scrollable_axis == 1:  # coronal
            axes = [2, 0, 1]
        else:  # axial
            axes = [2, 1, 0]

        # Handle numpy arrays, torch tensors, and regular lists/tuples
        if hasattr(coordinates, 'transpose'):
            # For numpy arrays and torch tensors
            return coordinates[..., axes]
            # return coordinates.transpose(*axes)
        else:
            # For regular Python sequences
            return type(coordinates)(coordinates[i] for i in axes)    


    def __init__(self, path_to_scan, median, stdev, patch_shape=(1, 16, 16)):
        self.zarr_store = zarr.open(path_to_scan, mode='r')
        self.patch_shape = np.array(patch_shape)
        self.med = median
        self.std = stdev

        attrs = self.zarr_store.attrs
        arr_shape = [attrs["shape_RL"], attrs["shape_PA"], attrs["shape_IS"]]
        self.scrollable_axis = np.argmax(np.abs(arr_shape - np.mean(arr_shape)))

        if self.scrollable_axis == 0: #sagittal
            axes = [0, 2, 1]
        elif self.scrollable_axis == 1: # coronal
            axes = [1, 2, 0]
        else: # axial
            axes = [2, 1, 0]

        self.scrollable_arr_shape = tuple(arr_shape[i] for i in axes)

    # def generate_training_pair(self, n_patches: int, to_torch: bool = True) -> tuple:
    #     """
    #     A high-level helper for training loops that generates a pair of samples.
    #     """

    #     wc1, ww1 = self.get_random_wc_ww_for_scan()
    #     wc2, ww2 = self.get_random_wc_ww_for_scan()
    #     sample_1_data = self.train_sample(n_patches=n_patches, wc=wc1, ww=ww1)
    #     sample_2_data = self.train_sample(n_patches=n_patches, wc=wc2, ww=ww2)

    #     patches_1 = sample_1_data['normalized_patches']
    #     patches_2 = sample_2_data['normalized_patches']
    #     patch_coords_1 = sample_1_data['patch_centers_pt'] - sample_1_data['subset_center_pt']
    #     patch_coords_2 = sample_2_data['patch_centers_pt'] - sample_2_data['subset_center_pt']

    #     print(wc1, wc2)
    #     print(ww1, ww2)

    #     # position based relative view information
    #     pos_label = sample_2_data['subset_center_pt'] - sample_1_data['subset_center_pt']
    #     pos_label = pos_label.squeeze(0)

    #     # window based relative view information
    #     window_label = np.array([wc2 - wc1, ww2 - ww1])

    #     label = np.concatenate((pos_label, window_label))

    #     if to_torch:
    #         patches_1 = torch.from_numpy(patches_1).to(torch.float32)
    #         patches_2 = torch.from_numpy(patches_2).to(torch.float32)
    #         patch_coords_1 = torch.from_numpy(patch_coords_1).to(torch.float32)
    #         patch_coords_2 = torch.from_numpy(patch_coords_2).to(torch.float32)
    #         label = torch.from_numpy(label).to(torch.float32)

    #     return patches_1, patches_2, patch_coords_1, patch_coords_2, label

    def train_sample(
        self,
        n_patches: int,
        patch_jitter: float = 0.33,
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
                subset_start, subset_shape, n_patches, patch_jitter
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

        if self.scrollable_axis == 0:  # sagittal
            axes = [0, 2, 1]
        elif self.scrollable_axis == 1:  # coronal
            axes = [1, 2, 0]
        else:  # axial
            axes = [2, 1, 0]


        # 1. Get the axes for the transpose operation (must be positive)
        transpose_axes = [abs(ax) for ax in axes]

        # 2. Get the new axes that need to be flipped (the indices of negative numbers)
        flip_axes = [i for i, ax in enumerate(axes) if ax < 0]

        # 3. Get the data from storage
        scan_data = self.zarr_store['pixel_data'][:]

        # 4. Perform the transposition first
        transposed_data = scan_data.transpose(transpose_axes)

        # 5. If any axes were marked for flipping, flip them now
        if flip_axes:
            return np.flip(transposed_data, axis=tuple(flip_axes))
        else:
            return transposed_data

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
        return random.uniform(self.med-self.std, self.med+self.std), random.uniform(2*self.std, 6*self.std)

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
        array_shape = self.scrollable_arr_shape

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

        ### NEW

        unclamped_absolute_starts = np.array(subset_start) + relative_starts
        max_scan_start = np.array(self.scrollable_arr_shape) - np.array(self.patch_shape)
        max_scan_start = np.maximum(0, max_scan_start) # Ensure it's not negative
        clamped_absolute_starts = np.clip(unclamped_absolute_starts, 0, max_scan_start)

        return clamped_absolute_starts.astype(int)

        ### END NEW

        # # --- 5. Clamp Patch Starts to Valid Subset Boundaries ---
        # # Ensure the patch does not go outside the subset dimensions.
        # max_patch_start = np.array(subset_shape) - np.array(self.patch_shape)
        # # Ensure max start is not negative (if subset is smaller than patch)
        # max_patch_start = np.maximum(0, max_patch_start)

        # # np.clip is perfect for this clamping operation
        # clamped_relative_starts = np.clip(relative_starts, 0, max_patch_start)

        # # --- 6. Convert to Absolute Coordinates and Return ---
        # absolute_starts = (np.array(subset_start) + clamped_relative_starts).astype(int)

        # return absolute_starts


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
        patch_d, patch_r, patch_c = self.convert_from_scrollable_coordinates(self.patch_shape)

        patch_indices = self.convert_from_scrollable_coordinates(patch_indices)

        # --- Step 1: Calculate the bounding box for all patches ---
        # Find the top-left-front corner of the entire region
        min_coords = np.min(patch_indices, axis=0)

        # Find the bottom-right-back corner of the entire region
        # We need the start of the last patch + its size
        max_coords = np.max(patch_indices, axis=0)
        end_coords = max_coords + np.array([patch_d, patch_r, patch_c])

        min_d, min_r, min_c = min_coords
        end_d, end_r, end_c = end_coords

        # --- Step 2: Make ONE large read from the Zarr store ---
        # This reads the entire bounding box into a single NumPy array in memory.
        # This is the ONLY interaction with the Zarr store.
        super_patch = self.zarr_store['pixel_data'][min_d:end_d, min_r:end_r, min_c:end_c]

        # --- Step 3: Extract individual patches from the in-memory super_patch ---
        # Pre-allocate the final array for efficiency
        patches = np.empty((n_patches, 1, 16, 16), dtype=super_patch.dtype)

        for i, (d, r, c) in enumerate(patch_indices):
            # Calculate the patch's position *relative* to the super_patch's origin
            d_rel, r_rel, c_rel = d - min_d, r - min_r, c - min_c

            # Slice the patch from the in-memory NumPy array (this is extremely fast)
            patches[i] = super_patch[
                d_rel : d_rel + patch_d,
                r_rel : r_rel + patch_r,
                c_rel : c_rel + patch_c
            ].flatten().reshape(1, 16, 16)

        return patches

    def convert_indices_to_patient_space(self, voxel_indices):
        affine_matrices = self.zarr_store['slice_affines'][:] # Load into memory

        if len(affine_matrices.shape) == 2:
            # the source data is likely a nifti and just use the singular 4x4 matrix for the transform
            voxel_indices = self.convert_from_scrollable_coordinates(voxel_indices)
            patient_coords = nib.affines.apply_affine(affine_matrices, voxel_indices)
        #     print(patient_coords.shape)
        # else:
        #     # 1. Get the slice index 'd' for each point.
        #     slice_indices = voxel_indices[:, 0].astype(int)

        #     # 2. Select the specific 4x4 affine matrix for each point's slice.
        #     # This results in a stack of shape (N, 4, 4).
        #     selected_affines = affine_matrices[slice_indices]

        #     # 3. Get the in-plane row and column indices (r, c).
        #     coords_rc = voxel_indices[:, 1:]  # Shape: (N, 2)

        #     # 4. Construct the 4-element homogeneous voxel coordinates [r, c, 0, 1].
        #     # The '0' indicates the position is on the 2D plane itself.
        #     # The '1' is the homogeneous coordinate.
        #     num_points = voxel_indices.shape[0]
        #     zeros_col = np.zeros((num_points, 1))
        #     ones_col = np.ones((num_points, 1))

        #     # We construct the vector [r, c, 0, 1] because our affine matrix was built
        #     # with column 0 for 'r' steps and column 1 for 'c' steps.
        #     homogeneous_coords = np.hstack([coords_rc, zeros_col, ones_col]) # Shape: (N, 4)

        #     # 5. Perform batched matrix multiplication.
        #     # We need to reshape homogeneous_coords for matmul: (N, 4) -> (N, 4, 1)
        #     # The operation is: (N, 4, 4) @ (N, 4, 1) -> (N, 4, 1)
        #     patient_coords_homogeneous = np.matmul(
        #         selected_affines,
        #         homogeneous_coords[:, :, np.newaxis]
        #     )

        #     # 6. Squeeze the result and extract the (x, y, z) components.
        #     # The result is (N, 4, 1), squeeze to (N, 4), then take the first 3 columns.
        #     patient_coords = patient_coords_homogeneous.squeeze(axis=2)[:, :3]
        return patient_coords/10


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
