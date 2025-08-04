import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, path_to_data, mask_ratio=0.2, seed=42):
        super().__init__()
        self.x = np.load(path_to_data).astype(np.float32)
        self.D = self.x.shape[1]
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

        if len(self.x) < 2:
            raise ValueError("Dataset must contain at least 2 samples for batch normalization")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_i = self.x[idx]
        x_i = x_i.astype(np.float32)

        num_mask = max(1, int(self.D * self.mask_ratio))  # At least 1 feature masked
        S = self.rng.choice(self.D, num_mask, replace=False)
        S_mask = np.zeros(self.D, dtype=np.float32)
        S_mask[S] = 1.0

        S_prime_mask = 1.0 - S_mask
        x_input = x_i * S_prime_mask

        return {
            'x_full': torch.tensor(x_i, dtype=torch.float32),
            'x_input': torch.tensor(x_input, dtype=torch.float32),
            'mask': torch.tensor(S_mask, dtype=torch.float32),
        }