"""
End-to-end pipeline: OPG image → ImageDiagnosis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.pipeline.stage1_segment import run_segmentation, SegmentationOutput
from src.pipeline.stage2_measure import measure_bone_loss
from src.pipeline.stage2_classify import classify_image, ImageDiagnosis
from src.utils.visualization import overlay_masks


@dataclass
class PipelineResult:
    segmentation: SegmentationOutput
    diagnosis: ImageDiagnosis
    overlay: np.ndarray    # (H, W, 3) uint8 — original + colored masks


class PeriodontalPipeline:
    """Load model (PyTorch or ONNX) and run the full two-stage pipeline."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_or_path,
        cfg: Optional[DictConfig] = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg = cfg or OmegaConf.load("configs/base.yaml")

        if isinstance(model_or_path, (str, Path)):
            path = str(model_or_path)
            if path.endswith(".onnx"):
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.model = None
            else:
                # PyTorch checkpoint
                from src.models.model_factory import create_model
                self.model = create_model(self.cfg)
                state = torch.load(path, map_location="cpu")
                self.model.load_state_dict(state.get("state_dict", state))
                self.model.to(self.device).eval()
                self.ort_session = None
        else:
            self.model = model_or_path
            self.ort_session = None

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 BGR/RGB/gray image to (1, 3, H, W) tensor."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        h, w = self.cfg.data.image_size
        image = cv2.resize(image, (w, h))
        image = image.astype(np.float32) / 255.0
        image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def run(self, image: np.ndarray) -> PipelineResult:
        tensor = self._preprocess(image)

        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            logits = self.ort_session.run(None, {input_name: tensor.numpy()})[0]
            import torch.nn.functional as F
            logits_t = torch.from_numpy(logits)
            probs = F.softmax(logits_t, dim=1)[0].numpy()
            mask = probs.argmax(axis=0)
            from src.pipeline.stage1_segment import SegmentationOutput, CLASS_TOOTH, CLASS_BONE, CLASS_CORTICAL
            seg = SegmentationOutput(
                mask_full=mask,
                mask_tooth=(mask == CLASS_TOOTH),
                mask_bone=(mask == CLASS_BONE),
                mask_cortical=(mask == CLASS_CORTICAL),
                probs=probs,
            )
        else:
            seg = run_segmentation(self.model, tensor, device=self.device)

        measurements = measure_bone_loss(
            seg.mask_tooth, seg.mask_bone, seg.mask_cortical,
            cej_fraction=self.cfg.stage2.cej_fraction,
        )

        diagnosis = classify_image(measurements, self.cfg)

        h, w = self.cfg.data.image_size
        display_img = cv2.resize(image, (w, h)) if image.shape[:2] != (h, w) else image
        overlay = overlay_masks(display_img, seg.mask_full, measurements)

        return PipelineResult(segmentation=seg, diagnosis=diagnosis, overlay=overlay)
