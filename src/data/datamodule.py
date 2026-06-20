import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data.dataset import PeriodontalDataset
from src.data.transforms import build_train_transforms, build_val_transforms


class PeriodontalDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        cfg = self.cfg
        split_dir = cfg.data.split_dir
        self.train_ds = PeriodontalDataset(
            root=cfg.data.root,
            split_file=f"{split_dir}/train.txt",
            transform=build_train_transforms(cfg),
        )
        self.val_ds = PeriodontalDataset(
            root=cfg.data.root,
            split_file=f"{split_dir}/val.txt",
            transform=build_val_transforms(cfg),
        )
        self.test_ds = PeriodontalDataset(
            root=cfg.data.root,
            split_file=f"{split_dir}/test.txt",
            transform=build_val_transforms(cfg),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )
