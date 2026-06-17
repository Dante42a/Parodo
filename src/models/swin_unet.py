"""
Swin-UNet: Swin Transformer encoder (timm) + custom transposed-conv decoder.

Input is padded to a multiple of `pad_to_multiple` before encoding,
then cropped back to the original size in the output mask.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from omegaconf import DictConfig


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # align spatial dims in case of rounding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, encoder_name: str, pretrained: bool, num_classes: int, pad_to_multiple: int = 32):
        super().__init__()
        self.pad_multiple = pad_to_multiple

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        encoder_channels = self.encoder.feature_info.channels()  # e.g. [128, 256, 512, 1024]

        # Build decoder top-down
        ch = encoder_channels
        self.dec3 = DecoderBlock(ch[3], ch[2], 256)
        self.dec2 = DecoderBlock(256, ch[1], 128)
        self.dec1 = DecoderBlock(128, ch[0], 64)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 32),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = math.ceil(h / self.pad_multiple) * self.pad_multiple - h
        pw = math.ceil(w / self.pad_multiple) * self.pad_multiple - w
        x = F.pad(x, (0, pw, 0, ph))
        return x, (h, w)

    def forward(self, x):
        orig_size = x.shape[2:]
        x, (h, w) = self._pad(x)

        feats = self.encoder(x)  # list of 4 feature maps

        d = self.dec3(feats[3], feats[2])
        d = self.dec2(d, feats[1])
        d = self.dec1(d, feats[0])
        d = self.dec0(d)
        out = self.head(d)

        # crop padding
        out = out[:, :, :h, :w]
        return out


def build_swin_unet(cfg: DictConfig) -> nn.Module:
    m = cfg.model
    pretrained = m.get("encoder_weights", "imagenet22k") is not None
    return SwinUNet(
        encoder_name=m.encoder_name,
        pretrained=pretrained,
        num_classes=m.classes,
        pad_to_multiple=m.get("pad_to_multiple", 32),
    )
