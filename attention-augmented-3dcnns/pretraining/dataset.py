import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pretraining.transforms import get_simclr_augmentations

class SimCLRDataset(Dataset):
    def __init__(self, patches_dir, transform=None):
        self.patches_dir = patches_dir
        self.files = [f for f in os.listdir(patches_dir) if f.endswith('.npy')]
        self.transform = transform or get_simclr_augmentations()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(os.path.join(self.patches_dir, self.files[idx]))
        patch = patch.astype(np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)  # [1, 32, 32, 32]
        if patch.shape != (1, 32, 32, 32):
            print(f"Warning: Patch {self.files[idx]} has shape {patch.shape}, skipping.")
            return self.__getitem__((idx + 1) % len(self))
        view1 = self.transform(patch)
        view2 = self.transform(patch)
        return view1, view2 