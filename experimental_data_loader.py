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
    from skimage.transform import resize


@app.cell(hide_code=True)
def _():
    metadata_df = pd.read_parquet('/cbica/home/gangarav/rsna_any/rsna_2025/nifti_combined_metadata.parquet')
    #pd.read_parquet('/cbica/home/gangarav/rsna25/aneurysm_labels_cleaned_6_64_64.parquet')
    metadata_df = metadata_df[metadata_df['series_uid'] != '1.2.826.0.1.3680043.8.498.40511751565674479940947446050421785002']
    table = mo.ui.table(metadata_df, selection='single')

    table
    return metadata_df, table


@app.cell(hide_code=True)
def _():
    patch_size_input = mo.ui.text(value="16", label="Patch Size:")
    n_patches_input = mo.ui.text(value="64", label="Num Patches:")
    n_aux_patches_input = mo.ui.text(value="16", label="Num Aux Patches:")
    run_button = mo.ui.run_button()
    mo.vstack([patch_size_input, n_patches_input, n_aux_patches_input, run_button])
    return n_aux_patches_input, n_patches_input, patch_size_input, run_button


@app.cell(hide_code=True)
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
    return sample, scan


@app.cell
def _(sample):
    sample["normalized_patches"].max(), sample["normalized_patches"].min()
    return


@app.cell
def _(sample, scan, scan_pixels, slider_IS, slider_PA, slider_RL):
    patches_for_display = sample["normalized_patches"].squeeze()
    patches_for_display = np.rot90(patches_for_display, k=1, axes=(1, 2))
    patches_for_display[:,0,:] = -1
    patches_for_display[:,:,0] = 1

    if scan.scrollable_axis == 0:  # sagittal
        d = slider_RL.value
        r = slider_IS.value
        c = slider_PA.value
    elif scan.scrollable_axis == 1:  # coronal
        d = slider_PA.value
        r = slider_IS.value
        c = slider_RL.value
    else:  # axial
        d = slider_IS.value
        r = slider_PA.value
        c = slider_RL.value

    mo.hstack([
        mo.image(np.flipud(scan_pixels[d]), height=512),
        mo.image(rearrange(
            patches_for_display,
            '(h_grid w_grid) h w -> (h_grid h) (w_grid w)',
            h_grid=2
        ), height=512)
    ])
    return c, d, r


@app.cell
def _(sample, scan):
    scan_pixels = scan.normalize_pixels_to_range(scan.get_scan_array_copy(), sample["wc"] - 0.5*sample["ww"], sample["wc"] + 0.5*sample["ww"])

    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        [sample["subset_start"]],
        sample["subset_shape"],
        None,
        (200, 0, 0)
    )
    scan_pixels = scan.create_rgb_scan_with_boxes(
        scan_pixels,
        sample["patch_indices"],
        sample["patch_shape"],
        None,
        (0, 200, 0)
    )

    if scan.scrollable_axis == 0:  # sagittal
        axes = [0, 2, 1, 3]
    elif scan.scrollable_axis == 1:  # coronal
        axes = [1, 2, 0, 3]
    else:  # axial
        axes = [2, 1, 0, 3]



    POINT_SIZE=2

    slider_RL = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[0]-1-POINT_SIZE, value=int(sample["subset_center_idx"][0,0]))
    slider_PA = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[1]-1-POINT_SIZE, value=int(sample["subset_center_idx"][0,1]))
    slider_IS = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[2]-1-POINT_SIZE, value=int(sample["subset_center_idx"][0,2]))
    sliders = [slider_RL, slider_PA, slider_IS]

    scan_pixels = scan_pixels.transpose(axes)

    # p1_r = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[1]-1-POINT_SIZE)
    # p1_c = mo.ui.slider(start=POINT_SIZE, stop=scan_pixels.shape[2]-1-POINT_SIZE)

    # slider = mo.ui.slider(start=0, stop=scan_pixels.shape[0]-1)# , value=int(sample["subset_center_idx"][0,0]))
    # mo.vstack([
    #     slider,
    #     p1_r,
    #     p1_c,
    # ])
    mo.vstack([
        slider_RL,
        slider_PA,
        slider_IS
    ])
    return POINT_SIZE, scan_pixels, slider_IS, slider_PA, slider_RL


@app.cell
def _(
    POINT_SIZE,
    c,
    d,
    r,
    sample,
    scan,
    scan_pixels,
    slider_IS,
    slider_PA,
    slider_RL,
):
    # Create slice tuples for the point marker
    point_slices = [slice(None)] * 3
    point_slices[0] = d
    point_slices[1] = slice(r - POINT_SIZE, r + POINT_SIZE)
    point_slices[2] = slice(c - POINT_SIZE, c + POINT_SIZE)

    drawn = scan_pixels[:].copy()
    drawn[tuple(point_slices)] = [256, 0, 0]

    point = np.array([slider_RL.value, slider_PA.value, slider_IS.value])
    mo.hstack([
        mo.vstack([
            mo.image(np.flipud(drawn[d]), width=512),
            scan.convert_indices_to_patient_space(point)
        ]),
        sample
    ])
    return


@app.cell
def _(sample):
    sample["normalized_patches"].shape
    return


@app.cell
def _(sample):
    pos_emb = PosEmbedding3D(600, max_freq = 30)
    coords = torch.from_numpy(sample["patch_centers_pt"] - sample["subset_center_pt"]).unsqueeze(0).to(torch.float32)
    print(coords)
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

    def __init__(self, path_to_scan, median, stdev, patch_shape=(1, 16, 16)):
        self.zarr_store = zarr.open(path_to_scan, mode='r')     
        self.med = median
        self.std = stdev

        attrs = self.zarr_store.attrs        
        pixel_shape = [attrs["px_shape_RL"], attrs["px_shape_PA"], attrs["px_shape_IS"]]
        self.arr_shape = [attrs["shape_RL"], attrs["shape_PA"], attrs["shape_IS"]]
        self.scrollable_axis = np.argmax(np.abs(self.arr_shape - np.mean(self.arr_shape)))

        if self.scrollable_axis == 0:  # sagittal
            axes = [0, 2, 1]
        elif self.scrollable_axis == 1:  # coronal
            axes = [2, 0, 1]
        else:  # axial
            axes = [2, 1, 0]

        self.orientation_normalized_patch_shape = np.array(type(patch_shape)(patch_shape[i] for i in axes))
        self.orientation_normalized_patch_shape = np.ceil(self.orientation_normalized_patch_shape/pixel_shape)



    def train_sample(
        self,
        n_patches: int,
        patch_jitter: float = 0.33,
        *, # Force subsequent arguments to be keyword-only for clarity
        subset_center: Optional[Tuple[int, int, int]] = None,
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
            if subset_center is None:
                subset_start, subset_shape = self.get_random_subset_from_scan(n_patches)
            else:
                subset_shape = 4 * self.orientation_normalized_patch_shape
                subset_start = subset_center - (2*self.orientation_normalized_patch_shape)
                subset_shape = subset_shape.astype(int)
                subset_start = subset_start.astype(int)
            print(subset_center, subset_start, subset_shape)
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
        results['patch_shape'] = raw_patches.shape[1:]

        normalized_patches = self.normalize_pixels_to_range(
            raw_patches, results['w_min'], results['w_max']
        )
        normalized_patches = self.reshape_patches(normalized_patches)
        results['normalized_patches'] = normalized_patches


        patch_centers_idx = (results['patch_indices'] + 0.5 * self.orientation_normalized_patch_shape).astype(int)
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
            RL, PA, IS = volume.shape

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
            RL, PA, IS, _ = volume.shape
            rgb_volume = volume

        box_RL, box_PA, box_IS = box_size

        for bRL, bPA, bIS in box_indices:
            RL_min, RL_max = max(0, bRL), min(RL, bRL + box_RL - 1)
            PA_min, PA_max = max(0, bPA), min(PA, bPA + box_PA - 1)
            IS_min, IS_max = max(0, bIS), min(IS, bIS + box_IS - 1)

            t = thickness # shorthand

            rgb_volume[RL_min:RL_max, PA_min:PA_min+t, IS_min:IS_min+t] = color
            rgb_volume[RL_min:RL_max, PA_min:PA_min+t, IS_max:IS_max+t] = color
            rgb_volume[RL_min:RL_max, PA_max:PA_max+t, IS_min:IS_min+t] = color
            rgb_volume[RL_min:RL_max, PA_max:PA_max+t, IS_max:IS_max+t] = color

            rgb_volume[RL_min:RL_min+t, PA_min:PA_max, IS_min:IS_min+t] = color
            rgb_volume[RL_min:RL_min+t, PA_min:PA_max, IS_max:IS_max+t] = color
            rgb_volume[RL_max:RL_max+t, PA_min:PA_max, IS_min:IS_min+t] = color
            rgb_volume[RL_max:RL_max+t, PA_min:PA_max, IS_max:IS_max+t] = color

            rgb_volume[RL_min:RL_min+t, PA_min:PA_min+t, IS_min:IS_max] = color
            rgb_volume[RL_min:RL_min+t, PA_max:PA_max+t, IS_min:IS_max] = color
            rgb_volume[RL_max:RL_max+t, PA_min:PA_min+t, IS_min:IS_max] = color
            rgb_volume[RL_max:RL_max+t, PA_max:PA_max+t, IS_min:IS_max] = color


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


    def get_random_subset_from_scan(self, n_patches: int, volume_scale_factor = 1.0):
        """
        Selects a random subset from the scan, with its shape determined by the
        number of patches requested. The volume of the subset will be approximately
        n_patches * patch_volume, while respecting min/max shape constraints.
        """
        min_subset_shape = 4 * self.orientation_normalized_patch_shape
        max_subset_shape = 8 * self.orientation_normalized_patch_shape
        percent_outer_to_ignore = (0.1, 0.1, 0.1)
        array_shape = self.arr_shape

        if not all(array_shape >= self.orientation_normalized_patch_shape):
            raise ValueError("array_shape must be larger than or equal to patch_shape in all dimensions.")

        # 1. Define bounds for subset shape based on scan size and constraints
        RL_buffer = int(array_shape[0] * percent_outer_to_ignore[0])
        PA_buffer = int(array_shape[1] * percent_outer_to_ignore[1])
        IS_buffer = int(array_shape[2] * percent_outer_to_ignore[2])

        # Min shape must be at least as large as a patch
        min_bounds = np.maximum(min_subset_shape, self.orientation_normalized_patch_shape).astype(int)

        # Max shape is constrained by user input and the available scan size after buffering
        max_bounds = np.minimum(max_subset_shape, array_shape - 2 * np.array([RL_buffer, PA_buffer, IS_buffer])).astype(int)

        if not np.all(max_bounds >= min_bounds):
             raise ValueError(f"Could not find a valid subset shape. Check min/max_subset_shape and scan dimensions. Min: {min_bounds}, Max: {max_bounds}")

        # 2. Calculate the target volume for the subset
        patch_volume = np.prod(self.orientation_normalized_patch_shape)
        target_volume = n_patches * patch_volume * volume_scale_factor

        # 3. Generate a random, flexible shape that approximates the target volume
        # We do this sequentially, constraining each subsequent dimension's random range.
        min_RL, min_PA, min_IS = min_bounds
        max_RL, max_PA, max_IS = max_bounds

        subset_RL = random.randint(min_RL, max_RL)
        subset_PA = random.randint(min_PA, max_PA)
        subset_IS = random.randint(min_IS, max_IS)
        subset_shape = [subset_RL, subset_PA, subset_IS]

        # 4. Determine the random starting position for the generated subset shape
        max_start_RL = array_shape[0] - subset_RL - RL_buffer
        max_start_PA = array_shape[1] - subset_PA - PA_buffer
        max_start_IS = array_shape[2] - subset_IS - IS_buffer

        # Ensure start buffer is not larger than max start pos
        start_RL = np.random.randint(RL_buffer, max_start_RL + 1)
        start_PA = np.random.randint(PA_buffer, max_start_PA + 1)
        start_IS = np.random.randint(IS_buffer, max_start_IS + 1)

        subset_start = (start_RL, start_PA, start_IS)

        return subset_start, subset_shape

    def get_stratified_random_patch_indices(self, 
                                            subset_start, 
                                            subset_shape, 
                                            n_patches, 
                                            randomness_factor=0.33):

        patch_RL, patch_PA, patch_IS = self.orientation_normalized_patch_shape
        subset_start_RL, subset_start_PA, subset_start_IS = subset_start
        subset_RL, subset_PA, subset_PA = subset_shape

        patch_vol = patch_RL * patch_PA * patch_IS
        subset_vol = subset_RL * subset_PA * subset_PA

        # This prevents division by zero if patch_vol or n_patches is 0.
        if patch_vol == 0 or n_patches == 0:
            return np.empty((0, 3), dtype=int)

        grid_cell_shape = np.array([patch_RL, patch_PA, patch_IS])

        # Number of grid cells along each dimension
        n_grid_cells = np.maximum(1, (np.array(subset_shape) / grid_cell_shape)).astype(int)

        # Recalculate the actual grid cell shape to perfectly tile the subset
        grid_cell_shape_actual = np.array(subset_shape) / n_grid_cells

        # --- 2. Generate all possible Grid Centers ---
        # Create coordinates for the center of each grid cell.
        grid_centers_RL = np.linspace(grid_cell_shape_actual[0] / 2, subset_shape[0] - grid_cell_shape_actual[0] / 2, n_grid_cells[0])
        grid_centers_PA = np.linspace(grid_cell_shape_actual[1] / 2, subset_shape[1] - grid_cell_shape_actual[1] / 2, n_grid_cells[1])
        grid_centers_IS = np.linspace(grid_cell_shape_actual[2] / 2, subset_shape[2] - grid_cell_shape_actual[2] / 2, n_grid_cells[2])

        # Create a meshgrid and flatten it to get a list of all grid center coordinates
        RL, PA, IS = np.meshgrid(grid_centers_RL, grid_centers_PA, grid_centers_IS, indexing='ij')
        all_grid_centers = np.vstack([RL.ravel(), PA.ravel(), IS.ravel()]).T

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
        patch_half_shape = (np.array(self.orientation_normalized_patch_shape) - 1) / 2.0
        relative_starts = (sampled_centers + random_offsets) - patch_half_shape


        unclamped_absolute_starts = np.array(subset_start) + relative_starts
        max_scan_start = np.array(self.arr_shape) - np.array(self.orientation_normalized_patch_shape)
        max_scan_start = np.maximum(0, max_scan_start) # Ensure it's not negative
        clamped_absolute_starts = np.clip(unclamped_absolute_starts, 0, max_scan_start)

        return clamped_absolute_starts.astype(int)


    def get_patches_from_indices(self, patch_indices):
        n_patches = patch_indices.shape[0]
        patch_shape = self.orientation_normalized_patch_shape.astype(int)
        patch_shape[np.argmin(patch_shape)] = 1

        patch_RL, patch_PA, patch_IS = patch_shape

        # --- Step 1: Calculate the bounding box for all patches ---
        # Find the top-left-front corner of the entire region
        min_coords = np.min(patch_indices, axis=0)

        # Find the bottom-right-back corner of the entire region
        # We need the start of the last patch + its size
        max_coords = np.max(patch_indices, axis=0)
        end_coords = max_coords + np.array([patch_RL, patch_PA, patch_IS])

        min_RL, min_PA, min_IS = min_coords.astype(int)
        end_RL, end_PA, end_IS = end_coords.astype(int)

        # --- Step 2: Make ONE large read from the Zarr store ---
        # This reads the entire bounding box into a single NumPy array in memory.
        # This is the ONLY interaction with the Zarr store.
        super_patch = self.zarr_store['pixel_data'][min_RL:end_RL, min_PA:end_PA, min_IS:end_IS]

        # --- Step 3: Extract individual patches from the in-memory super_patch ---
        # Pre-allocate the final array for efficiency
        patches = np.empty((n_patches, patch_RL, patch_PA, patch_IS), dtype=super_patch.dtype)

        for i, (RL, PA, IS) in enumerate(patch_indices):
            # Calculate the patch's position *relative* to the super_patch's origin
            RL_rel, PA_rel, IS_rel = RL - min_RL, PA - min_PA, IS - min_IS

            # Slice the patch from the in-memory NumPy array (this is extremely fast)
            patches[i] = super_patch[
                RL_rel : RL_rel + patch_RL,
                PA_rel : PA_rel + patch_PA,
                IS_rel : IS_rel + patch_IS
            ]

        return patches

    def reshape_patches(self, patches):
        # squeeze to convert to (n_patches, H, W)
        axis_to_squeeze = np.argmin(patches.shape[1:]) + 1 # +1 to account for n_patches dim
        n_patches = patches.shape[0]
        squeezed_patches = np.squeeze(patches, axis=axis_to_squeeze)

        # convert to (n_patches, 16, 16)
        target_shape = (n_patches, 16, 16)
        resized_patches = resize(squeezed_patches, 
                                 target_shape, 
                                 anti_aliasing=True, 
                                 preserve_range=True)

        final_patches = np.expand_dims(resized_patches, axis=1)

        # Cast back to original dtype if needed, as resize may change it
        final_patches = final_patches.astype(patches.dtype)
        return final_patches

    def convert_indices_to_patient_space(self, voxel_indices):
        affine_matrices = self.zarr_store['slice_affines'][:] # Load into memory

        if len(affine_matrices.shape) == 2:
            # the source data is likely a nifti and just use the singular 4x4 matrix for the transform

            # voxel_indices = self.convert_from_scrollable_coordinates(voxel_indices)
            patient_coords = nib.affines.apply_affine(affine_matrices, voxel_indices)

        return patient_coords/10


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
