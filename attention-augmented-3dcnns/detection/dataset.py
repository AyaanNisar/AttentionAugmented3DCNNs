import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class DetectionPatchDataset(Dataset):
    def __init__(self, meta_csv, patches_dir, transform=None):
        self.meta = pd.read_csv(meta_csv)
        self.patches_dir = patches_dir
        self.transform = transform
        
        # Filter out malformed patches during initialization
        valid_indices = []
        print(f"Checking {len(self.meta)} patches for valid shapes...")
        
        for idx, row in self.meta.iterrows():
            patch_path = os.path.join(self.patches_dir, row['filename'])
            if not os.path.exists(patch_path):
                print(f"Warning: Patch file {row['filename']} not found, skipping.")
                continue
                
            try:
                patch = np.load(patch_path)
                if patch.shape == (32, 32, 32):  # Expected shape
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Patch {row['filename']} has shape {patch.shape}, skipping.")
            except Exception as e:
                print(f"Warning: Error loading patch {row['filename']}: {e}")
                continue
        
        # Keep only valid patches
        self.meta = self.meta.iloc[valid_indices].reset_index(drop=True)
        print(f"Kept {len(self.meta)} valid patches out of {len(pd.read_csv(meta_csv))} total")
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        patch = np.load(os.path.join(self.patches_dir, row['filename']))
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)  # [1, D, H, W]
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        if self.transform:
            patch = self.transform(patch)
        return patch, label 