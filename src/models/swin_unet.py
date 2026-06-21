"""
Swin-UNet: Swin Transformer encoder (timm) + custom transposed-conv decoder.

Input is padded to a multiple of `pad_to_multiple` before encoding,
then cropped back to the original size in the output mask.

pad_to_multiple must be divisible by (patch_size * window_size):
  swin_base_patch4_window7 → 4*7 = 28
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
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, encoder_name: str, pretrained: bool, num_classes: int,
                 pad_to_multiple: int = 28, img_size: tuple = (532, 1036)):
        super().__init__()
        self.pad_multiple = pad_to_multiple

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )
        encoder_channels = self.encoder.feature_info.channels()  # e.g. [128, 256, 512, 1024]

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
        x, (h, w) = self._pad(x)
        padded_h, padded_w = x.shape[2], x.shape[3]

        feats = self.encoder(x)
        # Swin outputs NHWC → convert to NCHW for Conv2d decoder
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        d = self.dec3(feats[3], feats[2])
        d = self.dec2(d, feats[1])
        d = self.dec1(d, feats[0])
        d = self.dec0(d)

        # Swin downsamples ×32 but decoder only upsamples ×16 → upsample the rest
        d = F.interpolate(d, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
        out = self.head(d)

        out = out[:, :, :h, :w]
        return out


def build_swin_unet(cfg: DictConfig) -> nn.Module:
    m = cfg.model
    pretrained = m.get("encoder_weights", "imagenet22k") is not None
    pad_multiple = m.get("pad_to_multiple", 28)

    # Compute padded input size so encoder knows exact spatial dims at init time
    H, W = cfg.data.image_size
    pH = math.ceil(H / pad_multiple) * pad_multiple
    pW = math.ceil(W / pad_multiple) * pad_multiple

    return SwinUNet(
        encoder_name=m.encoder_name,
        pretrained=pretrained,
        num_classes=m.classes,
        pad_to_multiple=pad_multiple,
        img_size=(pH, pW),
    )
