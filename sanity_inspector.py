import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import wandb
    import os
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from data_loader import zarr_scan
    from lightning_train import RadiographyEncoder
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, IterableDataset
    import altair as alt
    return (
        DataLoader,
        IterableDataset,
        RadiographyEncoder,
        alt,
        mo,
        np,
        os,
        pd,
        torch,
        zarr_scan,
    )


@app.cell
def _(RadiographyEncoder, mo, os, torch):
    mo.md("### 1. Load Model from `wandb` Run")

    # Input for the user to provide the wandb Run ID
    run_id_input = mo.ui.text(
        value="30bqtiji", # A default value to start with
        label="Enter `wandb` Run ID:"
    )

    def get_model(run_id):
        """
        Loads the RadiographyEncoder model from a specified checkpoint path.
        """
        if not run_id:
            return None, "Please enter a valid wandb Run ID."

        # Construct the path to the checkpoint file
        checkpoint_path = f"/cbica/home/gangarav/checkpoints/{run_id}/last.ckpt"

        if not os.path.exists(checkpoint_path):
            return None, mo.md(f"**Error**: Checkpoint not found at `{checkpoint_path}`.")

        try:
            # Load the model using the class method from the training script.
            # Lightning automatically handles hyperparameters.
            model = RadiographyEncoder.load_from_checkpoint(checkpoint_path=checkpoint_path)
            model.eval()  # Set the model to evaluation mode

            # Move model to GPU if available, otherwise CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            status_message = mo.md(f"✅ Model from run `{run_id}` loaded successfully on `{device}`.")
            return model, status_message
        except Exception as e:
            return None, mo.md(f"**Error**: Failed to load model. Reason: {e}")

    # Display the input field
    run_id_input
    return get_model, run_id_input


@app.cell
def _(get_model, run_id_input):
    # This cell executes the model loading function when the Run ID changes
    # and stores the result.
    model, status = get_model(run_id_input.value)
    # Display the status message (e.g., success or error)
    status
    return (model,)


@app.cell
def _(mo, pd):
    metadata_df = pd.read_parquet('/cbica/home/gangarav/rsna25/aneurysm_labels_cleaned_6_64_64.parquet')
    metadata_df = metadata_df[metadata_df['SeriesInstanceUID'] != '1.2.826.0.1.3680043.8.498.40511751565674479940947446050421785002']
    scan_table = mo.ui.table(metadata_df, page_size=10)

    run_button = mo.ui.run_button(label="Generate and Compare Embeddings")
    mo.vstack([
        scan_table,
        run_button
    ])
    return run_button, scan_table


@app.cell
def _(np):
    PRISM_SHAPE = np.array([1, 128, 128])
    PATCH_SHAPE = np.array([1, 16, 16])
    N_PATCHES = 64

    # if scan_table.value is not None and not scan_table.value.empty:
    #     selected_row = scan_table.value.iloc[0]
    #     scan = zarr_scan(
    #         path_to_scan=selected_row["zarr_path"],
    #         median=selected_row["median"],
    #         stdev=selected_row["stdev"],
    #         patch_shape=PATCH_SHAPE
    #     )
    # else:
    #     scan = None
    #     selected_row = None

    # mo.md(f"**Scan Loaded:** `{os.path.basename(selected_row['zarr_path'])}`")
    return N_PATCHES, PATCH_SHAPE, PRISM_SHAPE


@app.cell
def _():
    # if scan:
    #     max_z, max_y, max_x = scan.zarr_store['pixel_data'].shape - PRISM_SHAPE

    #     # Use the first aneurysm location from metadata_df as default for prism 1
    #     default_z1 = selected_row['aneurysm_z'] if len(metadata_df) > 0 else max_z // 3
    #     default_y1 = selected_row['aneurysm_y'] if len(metadata_df) > 0 else max_y // 3
    #     default_x1 = selected_row['aneurysm_x'] if len(metadata_df) > 0 else max_x // 3

    #     # Ensure defaults are within bounds
    #     default_z1 = max(0, min(default_z1, max_z))
    #     default_y1 = max(0, min(default_y1, max_y))
    #     default_x1 = max(0, min(default_x1, max_x))

    #     # Sliders now control the CENTER of the prism
    #     z1 = mo.ui.slider(0, max_z, label="Z₁ Center", value=default_z1)
    #     y1 = mo.ui.slider(0, max_y, label="Y₁ Center", value=default_y1)
    #     x1 = mo.ui.slider(0, max_x, label="X₁ Center", value=default_x1)

    #     z2 = mo.ui.slider(0, max_z, label="Z₂ Center", value=2 * max_z // 3)
    #     y2 = mo.ui.slider(0, max_y, label="Y₂ Center", value=2 * max_y // 3)
    #     x2 = mo.ui.slider(0, max_x, label="X₂ Center", value=2 * max_x // 3)


    #     controls = mo.hstack([
    #         mo.vstack([mo.md("**Prism 1**"), z1, y1, x1]),
    #         mo.vstack([mo.md("**Prism 2**"), z2, y2, x2]),
    #     ], justify='space-around')

    # mo.vstack([controls, run_button])
    return


@app.cell
def _(
    DataLoader,
    IterableDataset,
    N_PATCHES,
    PATCH_SHAPE,
    scan_table,
    torch,
    zarr_scan,
):
    class InteractiveDataset(IterableDataset):
        """Interactive dataset for validation with both aneurysm-centered and random patches."""

        def __init__(self, prism_shape=(6, 64, 64), patch_shape=None, n_patches=None, n_windows=5, n_repeated=2):
            super().__init__()
            # Use the table widget's current selection as metadata source
            self.metadata = scan_table.value
            self.prism_shape = prism_shape
            self.patch_shape = patch_shape
            self.n_patches = n_patches
            self.n_windows = n_windows
            self.n_repeated = n_repeated
            print(f"Initialized interactive dataset with {len(self.metadata)} samples.")

        def __iter__(self):
            """Yield both aneurysm-centered and random patches for each scan."""
            for _, row in self.metadata.iterrows():
                try:
                    # Extract scan information
                    zarr_path = row["zarr_path"]
                    median = row["median"]
                    stdev = row["stdev"]
                    z, y, x = row['aneurysm_z'], row['aneurysm_y'], row['aneurysm_x']
                    location = row['location']

                    # Load scan
                    scan = zarr_scan(
                        path_to_scan=zarr_path,
                        median=median,
                        stdev=stdev,
                        patch_shape=self.patch_shape
                    )

                    for w_id in range(self.n_windows):

                        # Yield aneurysm-centered patches
                        sample = scan.train_sample(
                            n_patches=self.n_patches,
                            subset_start=(z - self.prism_shape[0] / 2, y - self.prism_shape[1] / 2, x - self.prism_shape[2] / 2),
                            subset_shape=self.prism_shape,
                        )

                        patches = torch.from_numpy(sample["normalized_patches"]).float()
                        patch_coords = torch.from_numpy(sample['patch_centers_pt'] - sample['subset_center_pt']).float()

                        yield patches, patch_coords, location, zarr_path, sample['wc'], sample['ww'], sample['subset_start'][0], sample['subset_start'][1], sample['subset_start'][2], sample['subset_center_pt'][0][0], sample['subset_center_pt'][0][1], sample['subset_center_pt'][0][2]

                        for r_id in range(self.n_repeated):
                            # Yield random patches for comparison
                            sample = scan.train_sample(
                                n_patches=self.n_patches,
                                subset_shape=self.prism_shape,
                            )

                            patches = torch.from_numpy(sample["normalized_patches"]).float()
                            patch_coords = torch.from_numpy(sample['patch_centers_pt'] - sample['subset_center_pt']).float()

                            yield patches, patch_coords, "random", zarr_path, sample['wc'], sample['ww'], sample['subset_start'][0], sample['subset_start'][1], sample['subset_start'][2], sample['subset_center_pt'][0][0], sample['subset_center_pt'][0][1], sample['subset_center_pt'][0][2]

                except Exception as e:
                    print(f"Skipping sample due to error: {e} in {row.get('zarr_path', 'N/A')}")
                    continue

    interactive_dataset = InteractiveDataset(
        patch_shape=PATCH_SHAPE,
        n_patches=N_PATCHES,
        n_windows=10,
        n_repeated=10
    )

    interactive_loader = DataLoader(
        interactive_dataset,
        batch_size=64
    )
    return (interactive_loader,)


@app.cell
def _(interactive_loader, model, pd, run_button, torch):
    # Run model on interactive loader outputs
    device = next(model.parameters()).device
    results = []
    if run_button.value:
        for patches, patch_coords, location, zarr_path, wc, ww, prism_start_z, prism_start_y, prism_start_x, subset_center_z, subset_center_y, subset_center_x in interactive_loader:

            patches = patches.to(device)
            patch_coords = patch_coords.to(device)

            with torch.no_grad():
                embeddings = model.encoder(patches, patch_coords)[:, 1]

            # Convert to numpy for easier handling
            embeddings_np = embeddings.cpu().numpy()

            # Create DataFrame with proper lengths
            batch_results = pd.DataFrame({
                'embedding': list(embeddings_np),
                'location': location,
                'wc': [round(float(w), 1) for w in wc],
                'ww': [round(float(w), 1) for w in ww],
                'zarr': zarr_path,
                'prism_start_z': [int(_z) for _z in prism_start_z],
                'prism_start_y': [int(_y) for _y in prism_start_y],
                'prism_start_x': [int(_x) for _x in prism_start_x],
                'subset_center_z': [float(_z) for _z in subset_center_z],
                'subset_center_y': [float(_y) for _y in subset_center_y],
                'subset_center_x': [float(_x) for _x in subset_center_x],
            })
            results.append(batch_results)

        results_df = pd.concat(results, ignore_index=True)
        results_df
    return device, results_df


@app.cell
def _(np, results_df):
    # Prepare data from results_df
    all_embeddings = np.vstack(results_df['embedding'].values)
    all_locations = results_df['location'].tolist()

    # Filter for actual locations (not 'random')
    filter_mask = results_df['location'] != 'random'
    filtered_embeddings = all_embeddings[filter_mask]
    filtered_locations = [loc for loc, m in zip(all_locations, filter_mask) if m]

    # Perform UMAP for visualization
    import umap
    umap_reducer = umap.UMAP(n_components=2, random_state=42).fit(filtered_embeddings)
    return filter_mask, filtered_embeddings, umap_reducer


@app.cell
def _(alt, filter_mask, filtered_embeddings, mo, results_df, umap_reducer):
    # Create DataFrame for plotting based on results_df with new columns
    plot_df = results_df[filter_mask].copy()
    embeddings_2d = umap_reducer.transform(filtered_embeddings)
    plot_df['UMAP1'] = embeddings_2d[:, 0]
    plot_df['UMAP2'] = embeddings_2d[:, 1]
    plot_df = plot_df.drop(columns=['embedding'])

    domain_values = plot_df['location'].unique().tolist()
    range_values = [
        'rgba(31, 119, 180, 0.5)', 
        'rgba(255, 127, 14, 0.5)', 
        'rgba(44, 160, 44, 0.5)', 
        'rgba(214, 39, 40, 0.3)', 
        'rgba(148, 103, 189, 0.3)', 
        'rgba(140, 86, 75, 0.3)', 
        'rgba(227, 119, 194, 0.3)', 
        'rgba(127, 127, 127, 0.3)', 
        'rgba(188, 189, 34, 0.3)', 
        'rgba(23, 190, 207, 0.3)'
    ][:len(domain_values)]

    chart = mo.ui.altair_chart(alt.Chart(plot_df).mark_circle(size=100).encode(
        x='UMAP1',
        y='UMAP2',
        color=alt.Color('location:N', 
                       scale=alt.Scale(domain=domain_values, range=range_values),
                       legend=alt.Legend(title="True Location")),
        tooltip=['location', 'wc', 'ww', 'prism_start_z', 'prism_start_y', 'prism_start_x']
    ))

    chart
    return (chart,)


@app.cell
def _(chart, mo, results_df):
    selected_zarr = chart.value.iloc[0]['zarr']
    filtered_results_df = results_df[results_df['zarr'] == selected_zarr].copy()
    prism_table = mo.ui.table(filtered_results_df, selection="single")
    prism_table
    return filtered_results_df, prism_table


@app.cell
def _(PATCH_SHAPE, PRISM_SHAPE, chart, mo, np, prism_table, zarr_scan):
    # Create visualization using your class method
    selected_plot_df = chart.value.iloc[0]
    comparison_prism = prism_table.value.iloc[0]

    # Load the scan for the selected row
    scan = zarr_scan(
        path_to_scan=selected_plot_df["zarr"],
        median=selected_plot_df["wc"],
        stdev=selected_plot_df["ww"],
        patch_shape=PATCH_SHAPE
    )

    # Get normalized scan pixels
    scan_pixels = scan.get_scan_array_copy()
    scan_pixels_norm = scan.normalize_pixels_to_range(
        scan_pixels, 
        selected_plot_df["wc"], 
        selected_plot_df["ww"], 
        out_range=(0, 1)
    )

    # Calculate prism start coordinates (center - half prism shape)
    prism_start = np.array([
        selected_plot_df["prism_start_z"],
        selected_plot_df["prism_start_y"],
        selected_plot_df["prism_start_x"]
    ])

    # Create visualization with the selected prism
    viz_scan = scan.create_rgb_scan_with_boxes(
        scan_pixels_norm, 
        [prism_start.astype(int)], 
        PRISM_SHAPE, 
        color=(200, 0, 0)
    )

    # Calculate prism start coordinates (center - half prism shape)
    prism_start = np.array([
        comparison_prism["prism_start_z"],
        comparison_prism["prism_start_y"],
        comparison_prism["prism_start_x"]
    ])

    # Create visualization with the selected prism
    viz_scan = scan.create_rgb_scan_with_boxes(
        viz_scan, 
        [prism_start.astype(int)], 
        PRISM_SHAPE, 
        color=(0, 200, 0)
    )


    viz_slider = mo.ui.slider(0, viz_scan.shape[0] - 1, value=selected_plot_df["prism_start_z"])
    viz_slider
    return (
        comparison_prism,
        prism_start,
        selected_plot_df,
        viz_scan,
        viz_slider,
    )


@app.cell
def _(
    comparison_prism,
    mo,
    prism_start,
    selected_plot_df,
    viz_scan,
    viz_slider,
):
    mo.hstack([
        mo.image(src=viz_scan[viz_slider.value]),
        mo.image(src=viz_scan[prism_start[0]]),
        mo.vstack([
            comparison_prism["subset_center_z"] - selected_plot_df["subset_center_z"],
            comparison_prism["subset_center_y"] - selected_plot_df["subset_center_y"],
            comparison_prism["subset_center_x"] - selected_plot_df["subset_center_x"],
        ])
    ])
    return


@app.cell
def _(device, filtered_results_df, model, prism_table, torch):
    _row1 = torch.tensor(filtered_results_df['embedding'].iloc[0]).to(device)
    _row2 = torch.tensor(prism_table.value['embedding'].iloc[0]).to(device)

    fused_view_cls = torch.cat((_row1, _row2), dim=0)

    model.relative_pos_head(fused_view_cls.unsqueeze(0))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
