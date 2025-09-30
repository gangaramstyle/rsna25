import os
import argparse
import pandas as pd
import numpy as np
import torch
import zarr
from PIL import Image
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import torchvision.transforms.functional as TF

# Assuming dino_zarr_data_loader.py is in the same directory
from dino_zarr_data_loader import zarr_scan


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16


class ScanSliceDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing individual scan slices."""
    def __init__(self, rgb_scan):
        self.rgb_scan = rgb_scan
        self.height, self.width, _ = self.rgb_scan[0].shape
        self.r_patches = self.height // PATCH_SIZE
        self.c_patches = self.width // PATCH_SIZE
        self.target_size = (self.r_patches * PATCH_SIZE, self.c_patches * PATCH_SIZE)

    def __len__(self):
        return self.rgb_scan.shape[0]

    def __getitem__(self, idx):
        img_slice = self.rgb_scan[idx]
        img_tensor = torch.from_numpy(img_slice).permute(2, 0, 1)
        image_resized = TF.resize(img_tensor, self.target_size)
        image_resized_norm = TF.normalize(image_resized.float() / 255.0, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_resized_norm


def get_model(model_name="facebook/dinov2-base"):
    """Initializes and returns the DINOv2 model."""
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    model.eval()
    return model


def embed_batch(image_batch, model):
    """
    Generates embeddings for a batch of image slices using the DINOv2 model.
    """
    device = next(model.parameters()).device
    inputs = image_batch.to(device).half()

    with torch.no_grad():
        print(inputs.shape)
        outputs = model(inputs)
        last_hidden_states = outputs.last_hidden_state

        cls_token = last_hidden_states[:, 0, :]
        patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]

        num_patches_h = (inputs.shape[2] // PATCH_SIZE)
        num_patches_w = (inputs.shape[3] // PATCH_SIZE)
        print(num_patches_h, num_patches_w)
        patch_features = patch_features_flat.unflatten(1, (num_patches_h, num_patches_w))

    return patch_features.cpu().numpy()


def process_scan(metadata_row, model, output_dir, batch_size, num_workers):
    """
    Processes a single scan, generates embeddings in batches, and saves them to a Zarr file.
    """
    zarr_name = metadata_row["zarr_path"]
    series_uid = metadata_row["series_uid"]
    output_path = os.path.join(output_dir, f"{series_uid}.zarr")

    if os.path.exists(output_path):
        print(f"Skipping {series_uid}, already processed.")
        return

    try:
        scan = zarr_scan(path_to_scan=zarr_name, median=metadata_row["median"], stdev=metadata_row["stdev"])
        scan_pixels = scan.get_scan_array_copy()
    
        if scan.scrollable_axis == 0:  # sagittal
            axes = [0, 2, 1]
        elif scan.scrollable_axis == 1:  # coronal
            axes = [1, 2, 0]
        else:  # axial
            axes = [2, 1, 0]
    
        scan_pixels = np.flip(scan_pixels.transpose(axes), 1)
    
        scale = 768 / max(scan_pixels.shape[1], scan_pixels.shape[2])
        new_shape = (scan_pixels.shape[0], int(scan_pixels.shape[1] * scale), int(scan_pixels.shape[2] * scale))
    
        px = np.array([np.array(Image.fromarray(slice_).resize((new_shape[2], new_shape[1]), Image.LANCZOS)) for slice_ in scan_pixels])
        rgb = scan.create_rgb_scan_with_boxes(px, [], (1, 1, 1))
        print(rgb.shape)
    
    
        # Create Dataset and DataLoader
        dataset = ScanSliceDataset(rgb)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
        all_embeddings = []
        for batch in tqdm(dataloader, desc=f"Embedding {series_uid}", leave=False):
            print(batch.shape)
            embeddings_batch = embed_batch(batch, model)
            print(embeddings_batch.shape)
            all_embeddings.append(embeddings_batch)
    
        embeddings = np.concatenate(all_embeddings, axis=0)
    
        # Save to Zarr
        z = zarr.open(output_path, mode='w', shape=embeddings.shape,
                      chunks=(1, None, None, None), dtype='float32')
        z[:] = embeddings
    
        print(f"Successfully processed and saved {series_uid}")

    except Exception as e:
        print(f"Error processing {series_uid}: {e}")


def find_optimal_batch_size(model, input_shape=(768, 768, 3)):
    """
    A simple utility to help determine a reasonable batch size for your GPU.
    """
    print("Finding optimal batch size...")
    batch_size = 1
    while True:
        try:
            dummy_input = torch.randn(batch_size, 3, input_shape[0], input_shape[1]).cuda().half()
            with torch.no_grad():
                model(dummy_input)
            print(f"Batch size {batch_size} fits in memory.")
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch size {batch_size}. Optimal batch size is likely {batch_size // 2}.")
                return batch_size // 2
            else:
                raise e

model = get_model("facebook/dinov3-vith16plus-pretrain-lvd1689m")


# To use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)

metadata_df = pd.read_parquet('/cbica/home/gangarav/rsna_any/rsna_2025/nifti_combined_metadata.parquet')

for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    process_scan(row, model, 'tmp', 8, 2)
