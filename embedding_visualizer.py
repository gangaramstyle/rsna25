import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import zarr
    from data_loader import zarr_scan
    from rvt_model import RvT
    return RvT, mo, nn, np, pd, torch, zarr_scan


@app.cell
def _(pd):
    # Load metadata
    metadata_path = '/cbica/home/gangarav/data_25_processed/zarr_stats.parquet'
    metadata = pd.read_parquet(metadata_path).head(3250)
    return (metadata,)


@app.cell
def _(metadata, mo):
    # Create mo table for row selection
    row_selector = mo.ui.table(
        data=metadata.head(1000),
    )
    row_selector
    return (row_selector,)


@app.cell
def _(SaimeseEncoder, np, row_selector, torch, zarr_scan):
    def normalize_hu_to_range(hu_array, w_min, w_max, out_range=(-1.0, 1.0)):
        clipped_array = np.clip(hu_array, w_min, w_max)
        scaled_01 = (clipped_array - w_min) / (w_max - w_min)
        out_min, out_max = out_range
        return scaled_01 * (out_max - out_min) + out_min

    # Load model and compute embeddings for all selected scans
    embeddings = []
    scans = []

    if row_selector.value is not None:
        model = SaimeseEncoder(
            patch_size=(1, 16, 16),
            num_classes=3,
            dim=288,
            depth=8,
            heads=8,
            mlp_dim=512,
            use_rotary=True
        )
        model.load_state_dict(torch.load("/gpfs/fs001/cbica/home/gangarav/rsna25/trained_models/contrastive_step_10000.pth", map_location='cpu'))
        model.eval()

        for _, selected_row in row_selector.value.iterrows():
            zarr_path = selected_row['zarr_path']
            print(zarr_path)

            # Load scan
            scan = zarr_scan(path_to_scan=zarr_path)

            # Get patches and compute embedding
            w_min, w_max = selected_row['median'] - 3*selected_row['stdev'], selected_row['median'] + 3*selected_row['stdev']

            subset_start = (np.array(scan.zarr_store["pixel_data"].shape) // 4).astype(int)
            subset_shape = (np.array(scan.zarr_store["pixel_data"].shape) // 2).astype(int)

            #scan.get_random_subset_from_scan()
            idxs = scan.get_random_patch_indices_from_scan_subset(subset_start, subset_shape, 64)
            patches = scan.get_patches_from_indices(idxs)

            patches = normalize_hu_to_range(patches, w_min, w_max)

            coords = scan.convert_indices_to_patient_space(idxs)
            center = np.array(subset_start) + 0.5*np.array(subset_shape)
            center = center.astype(int)[np.newaxis, :]
            center_pt = scan.convert_indices_to_patient_space(center)
            coords = coords - center_pt

            patches_torch = torch.from_numpy(patches).to(torch.float32).unsqueeze(0)
            coords_torch = torch.from_numpy(coords).to(torch.float32).unsqueeze(0)

            with torch.no_grad():
                embedding = model.branch(patches_torch, coords_torch)[:, 0]
                embeddings.append(embedding)
                scans.append(zarr_path)

    embeddings = torch.stack(embeddings).squeeze() if embeddings else None
    embeddings
    return embeddings, scans


@app.cell
def _():
    return


@app.cell
def _(embeddings, mo, pd, scans):
    from sklearn.manifold import TSNE
    import umap.umap_ as umap
    import altair as alt

    # --- Using UMAP ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)

    df_umap = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    df_umap['scan'] = scans

    chart_umap = alt.Chart(df_umap).mark_circle(size=50).encode(
        x=alt.X('UMAP1', title='UMAP Component 1'),
        y=alt.Y('UMAP2', title='UMAP Component 2'),
        tooltip=['UMAP1', 'UMAP2']
    ).properties(
        title='UMAP projection of SimCLR embeddings',
        width=600,
        height=500
    ).interactive()

    # # --- Using t-SNE ---
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, metric='cosine')
    # tsne_results = tsne.fit_transform(embeddings)

    # df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    # df_tsne['scan'] = scans

    # chart_tsne = alt.Chart(df_tsne).mark_circle(size=50).encode(
    #     x=alt.X('t-SNE1', title='t-SNE Component 1'),
    #     y=alt.Y('t-SNE2', title='t-SNE Component 2'),
    #     tooltip=['t-SNE1', 't-SNE2', 'scan']
    # ).properties(
    #     title='t-SNE projection of SimCLR embeddings',
    #     width=600,
    #     height=500
    # ).interactive()

    # chart_tsne

    chart = mo.ui.altair_chart(chart_umap)
    chart
    return (chart,)


@app.cell
def _(chart, mo, zarr_scan):
    if chart.value is not None:
        _zarr_path = chart.value['scan'].iloc[0]
        _scan = zarr_scan(path_to_scan=_zarr_path)
        _z = _scan.get_scan_array_copy()
    mo.image(src=_z[int(_z.shape[0]//2)])
    return


@app.cell
def _(RvT, nn):
    class SaimeseEncoder(nn.Module):
        def __init__(self, *, patch_size, num_classes, dim, depth, heads, mlp_dim, use_rotary = True):
            super().__init__()

            self.branch = RvT(
                patch_size=patch_size,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                use_rotary=use_rotary
            )

            self.trunk = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, num_classes)
            )

        def forward(self, patches_1, coords_1, patches_2, coords_2):
            x1 = self.branch(patches_1, coords_1)
            x2 = self.branch(patches_2, coords_2)

            return x1, x2
    return (SaimeseEncoder,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
