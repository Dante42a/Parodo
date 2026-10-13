"""
End-to-end pipeline: OPG image → ImageDiagnosis.

Two-model architecture:
  Model 1 — Swin-UNet (ONNX or .ckpt): semantic masks for bone and cortical plate.
  Model 2 — YOLOv8-seg (.pt):          per-tooth instance masks.

If no YOLO model is provided, falls back to connected-components on tooth semantic mask.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.pipeline.stage1_segment import (
    run_segmentation,
    run_tooth_instances,
    fallback_tooth_instances,
    SegmentationOutput,
)
from src.pipeline.stage2_measure import measure_bone_loss
from src.pipeline.stage2_classify import classify_image, ImageDiagnosis
from src.utils.visualization import overlay_masks

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    segmentation: SegmentationOutput
    diagnosis: ImageDiagnosis
    overlay: np.ndarray    # (H, W, 3) uint8


class PeriodontalPipeline:
    """Load models and run the full two-stage pipeline."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        seg_model_or_path,
        cfg: Optional[DictConfig] = None,
        yolo_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg = cfg or OmegaConf.load("configs/base.yaml")

        # ── Semantic segmentation model (Swin-UNet) ─────────────────────────
        if isinstance(seg_model_or_path, (str, Path)):
            path = str(seg_model_or_path)
            if path.endswith(".onnx"):
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(
                    path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.seg_model = None
            else:
                from src.models.model_factory import create_model
                # Все веса придут из чекпойнта — предобученные ImageNet-веса энкодера не скачиваем
                # (конфиг общий с обучением, где они нужны; здесь они лишние ~170 МБ и требуют интернет)
                model_cfg = OmegaConf.merge(self.cfg, {"model": {"encoder_weights": None}})
                self.seg_model = create_model(model_cfg)
                state = torch.load(path, map_location="cpu", weights_only=False)
                raw = state.get("state_dict", state)
                if all(k.startswith("model.") for k in raw if not k.startswith("criterion")):
                    raw = {k[len("model."):]: v for k, v in raw.items()
                           if not k.startswith("criterion")}
                self.seg_model.load_state_dict(raw)
                self.seg_model.to(self.device).eval()
                self.ort_session = None
        else:
            self.seg_model = seg_model_or_path
            self.ort_session = None

        # ── YOLO tooth instance model ────────────────────────────────────────
        self.yolo_model = None
        if yolo_path:
            self.load_yolo(yolo_path)
        else:
            logger.warning("No YOLO model — falling back to connected-components for tooth instances.")

    def load_yolo(self, yolo_path: str) -> None:
        # Путь задан явно — его отсутствие считаем ошибкой, а не поводом молча работать без YOLO
        if not Path(yolo_path).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_path}")
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_path)
        logger.info("YOLO tooth model loaded: %s", yolo_path)

    def _preprocess(self, image: np.ndarray) -> tuple:
        """Return (tensor (1,3,H,W), rgb_hwc uint8) at cfg image_size."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        h, w = self.cfg.data.image_size
        resized = cv2.resize(image, (w, h))

        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor, resized

    def run(self, image: np.ndarray) -> PipelineResult:
        tensor, display_img = self._preprocess(image)
        H, W = self.cfg.data.image_size

        # ── Stage 1a: semantic segmentation ─────────────────────────────────
        if self.ort_session is not None:
            import torch.nn.functional as F
            input_name = self.ort_session.get_inputs()[0].name
            logits_np = self.ort_session.run(None, {input_name: tensor.numpy()})[0]
            logits_t = torch.from_numpy(logits_np)
            probs = F.softmax(logits_t, dim=1)[0].numpy()
            mask = probs.argmax(axis=0)
            from src.pipeline.stage1_segment import (
                CLASS_TOOTH, CLASS_BONE, CLASS_CORTICAL, SegmentationOutput
            )
            seg = SegmentationOutput(
                mask_full=mask,
                mask_tooth=(mask == CLASS_TOOTH),
                mask_bone=(mask == CLASS_BONE),
                mask_cortical=(mask == CLASS_CORTICAL),
                probs=probs,
            )
        else:
            seg = run_segmentation(self.seg_model, tensor, device=self.device)

        # ── Stage 1b: tooth instance detection ──────────────────────────────
        if self.yolo_model is not None:
            instances = run_tooth_instances(
                self.yolo_model,
                display_img,
                target_hw=(H, W),
                mask_tooth=seg.mask_tooth,
            )
        else:
            instances = fallback_tooth_instances(seg.mask_tooth)

        # ── Stage 2: measure + classify ──────────────────────────────────────
        measurements = measure_bone_loss(
            instances,
            seg.mask_bone,
            seg.mask_cortical,
            cej_fraction=self.cfg.stage2.cej_fraction,
        )

        diagnosis = classify_image(measurements, self.cfg)

        overlay = overlay_masks(display_img, seg.mask_full, measurements, instances=instances)

        return PipelineResult(segmentation=seg, diagnosis=diagnosis, overlay=overlay)
