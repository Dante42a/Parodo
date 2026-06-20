"""
Multi-source periodontal OPG dataset.

Each sample has:
  image  : (3, H, W) float32 tensor, normalized
  mask   : (H, W) int64 tensor, values in {0,1,2,3}
             0 = background
             1 = tooth
             2 = alveolar bone level
             3 = cortical plate

Images come from DENTEX 2023 and/or Tufts Dental DB.
Masks are stored as single-channel PNGs where pixel value = class index.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A


class PeriodontalDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        transform: Optional[A.Compose] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        ids = Path(split_file).read_text().strip().splitlines()
        meta = pd.read_csv(self.root / "metadata.csv", index_col="image_id")
        self.samples = [sid for sid in ids if sid in meta.index]
        self.meta = meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid = self.samples[idx]
        img_path = self.root / "images" / f"{sid}.png"
        mask_path = self.root / "masks" / f"{sid}.png"

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            m = augmented["mask"]
            mask = m.long() if isinstance(m, torch.Tensor) else torch.from_numpy(m).long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask, sid
