import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(cfg: DictConfig) -> A.Compose:
    h, w = cfg.data.image_size
    aug = cfg.augmentation
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        A.RandomBrightnessContrast(p=aug.brightness_contrast_p),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=aug.clahe_p),
        A.ElasticTransform(alpha=50, sigma=5, p=aug.elastic_p),
        A.GaussNoise(p=aug.gauss_noise_p),
        A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32),
            fill=0, p=aug.coarse_dropout_p
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_transforms(cfg: DictConfig) -> A.Compose:
    h, w = cfg.data.image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
