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
        np,
        optim,
        os,
        pd,
        torch,
        zarr_scan,
    )


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


@app.cell
def _(L, RvT, nn, optim, torch):
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
            loss = self.view_criterion(view_prediction, view_target)

            self.log("view_loss", loss)
            return loss

        # def validation_step(self, batch, batch_idx):
        #     # this is the validation loop
        #     x, _ = batch
        #     x = x.view(x.size(0), -1)
        #     z = self.encoder(x)
        #     x_hat = self.decoder(z)
        #     val_loss = F.mse_loss(x_hat, x)
        #     self.log("val_loss", val_loss)

        def configure_optimizers(self):
            # Use the learning_rate from hparams so it can be configured by sweeps
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            return optimizer
    return (RadiographyEncoder,)


@app.cell
def _(DataLoader, IterableDataset, pd, zarr_scan):
    class ValidationDataset(IterableDataset):

        def __init__(self, prism_shape=(6, 64, 64), patch_shape=None, n_patches=None):
            super().__init__()
            self.metadata = pd.read_parquet('aneurysm_labels_cleaned_6_64_64.parquet')
            self.metadata = self.metadata[self.metadata['SeriesInstanceUID'] != '1.2.826.0.1.3680043.8.498.40511751565674479940947446050421785002']
            self.prism_shape = prism_shape
            self.patch_shape = patch_shape
            self.n_patches = n_patches

        def __iter__(self):
            for _, row in self.metadata.iterrows():
                zarr_path = row["zarr_path"]
                print(zarr_path)
                row_id = row.name
                median = row["median"]
                stdev = row["stdev"]
                z = row['aneurysm_z']
                y = row['aneurysm_y']
                x = row['aneurysm_x']
                location = row['location']


                scan = zarr_scan(
                    path_to_scan=zarr_path,
                    median=median,
                    stdev=stdev,
                    patch_shape=self.patch_shape
                )

                sample = scan.train_sample(
                    n_patches=self.n_patches,
                    subset_start=(z-self.prism_shape[0]/2, y-self.prism_shape[1]/2, x-self.prism_shape[2]/2),
                    subset_shape=self.prism_shape,
                )

                yield sample["normalized_patches"], sample['patch_centers_pt'] - sample['subset_center_pt'], location #, sample, {"path_to_scan": zarr_path, "median": median, "stdev": stdev, "patch_shape": self.patch_shape}


    PATCH_SHAPE = (1, 16, 16) # (Depth, Height, Width) of each patch
    N_PATCHES = 64            # Number of patches to sample from each scan

    dataset = ValidationDataset(
        patch_shape=PATCH_SHAPE,
        n_patches=N_PATCHES,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
    )
    iterator = iter(dataloader)
    return dataloader, iterator


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run")
    run_btn
    return (run_btn,)


@app.cell
def _(iterator, run_btn):
    if run_btn.value:
        batch = next(iterator)
        patches, centers, location, imdt, scan_metadata = batch
    return centers, imdt, patches, scan_metadata


@app.cell
def _():
    return


@app.cell
def _(imdt, mo, scan_metadata, torch, zarr_scan):
    _scan = zarr_scan(
            path_to_scan=scan_metadata["path_to_scan"][0],
            median=scan_metadata["median"][0],
            stdev=scan_metadata["stdev"][0],
            patch_shape=scan_metadata["patch_shape"][0]
        )
    _px = _scan.get_scan_array_copy()

    _px = _scan.create_rgb_scan_with_boxes(
        _px,
        [torch.stack(imdt["subset_start"]).reshape(3).cpu().numpy().astype(int)],
        torch.stack(imdt["subset_shape"]).reshape(3).cpu().numpy().astype(int),
        color=(256, 0, 0)
    )

    pox = _scan.create_rgb_scan_with_boxes(
        _px,
        imdt["patch_indices"].squeeze(),
        (1, 16, 16),
        color=(0, 256, 0)
    )

    sloder = mo.ui.slider(start=0, stop=_px.shape[0]-1, value=imdt["subset_center_idx"][0,0,0].item())
    sloder
    return pox, sloder


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
def _(dataloader, model, np, torch):
    embeddings = []
    locations = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for _patches, _centers, _location in dataloader:
            _patches = _patches.float().to(device)
            _centers = _centers.float().to(device)

            _emb = model.encoder(_patches, _centers)


            embeddings.append(_emb[:, 1].cpu().numpy())
            locations.extend(_location)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, locations


@app.cell
def _():
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    return KMeans, PCA, adjusted_rand_score, plt


@app.cell
def _():
    # # Filter data for Right and Left Middle Cerebral Artery
    # filtered_indices = [i for i, loc in enumerate(locations) 
    #                    if loc in ["Basilar Tip", "Left Infraclinoid Internal Carotid Artery"]]
    # filtered_embeddings = embs[filtered_indices]
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
def _(KMeans, PCA, adjusted_rand_score, embeddings, locations, plt):
    # --- 1. Data Preparation (same as before) ---
    # Filter data for the locations of interest
    target_locations = [
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery"
    ]
    filtered_indices = [i for i, loc in enumerate(locations) if loc in target_locations]

    # Ensure you have data to process
    if not filtered_indices:
        print("No data found for the specified locations. Exiting.")
    else:
        filtered_embeddings = embeddings[filtered_indices]
        filtered_locations = [locations[i] for i in filtered_indices]
    
        # --- 2. Perform K-Means Clustering ---
        # We set n_clusters to the number of unique locations we expect to find.
        n_clusters = len(target_locations)
    
        print(f"Performing K-Means clustering with k={n_clusters}...")
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        # Fit K-Means on the original high-dimensional embeddings
        cluster_labels = kmeans.fit_predict(filtered_embeddings)
    
        # --- 3. Evaluate the Clustering ---
        # The Adjusted Rand Index (ARI) measures the similarity between two clusterings.
        # A score of 1.0 means the cluster labels are a perfect match to the ground truth labels.
        # A score around 0.0 means the clustering is random.
        ari_score = adjusted_rand_score(filtered_locations, cluster_labels)
        print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
        print("(ARI of 1.0 is a perfect match, 0.0 is random)")
    
        # --- 4. Prepare for Visualization ---
        # Use PCA ONLY to reduce dimensions for the 3D plot.
        # The clustering was done on the original data.
        print("Using PCA for 3D visualization...")
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(filtered_embeddings)
    
        # --- 5. Create Side-by-Side Plots for Comparison ---
        fig = plt.figure(figsize=(20, 9))
        fig.suptitle('Comparison of Ground Truth vs. K-Means Clustering', fontsize=16)
    
        # -- Plot 1: Ground Truth Labels --
        ax1 = fig.add_subplot(121, projection='3d')
        unique_locations = list(set(filtered_locations))
        color_map = {loc: plt.cm.viridis(i / len(unique_locations)) for i, loc in enumerate(unique_locations)}
        ground_truth_colors = [color_map[loc] for loc in filtered_locations]
    
        ax1.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                    c=ground_truth_colors, alpha=0.7, s=50)
    
        # Add legend for ground truth
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=color_map[loc], markersize=10, label=loc) 
                           for loc in unique_locations]
        ax1.legend(handles=legend_elements, title='True Locations', bbox_to_anchor=(1.15, 1))
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.set_title('Ground Truth Labels (in PCA space)')
    
        # -- Plot 2: K-Means Cluster Labels --
        ax2 = fig.add_subplot(122, projection='3d')
        # Color points based on the cluster label found by K-Means
        ax2.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                    c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title(f'K-Means Predicted Clusters (ARI: {ari_score:.2f})')
    
        plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make space for suptitle and legend
        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
