"""
Gradio app for periodontal disease detection from OPG X-rays.

Launch:
    python app/app.py

Or with a specific model path:
    python app/app.py --model app/models/best_model.onnx
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.pipeline.full_pipeline import PeriodontalPipeline
from src.utils.visualization import draw_diagnosis_card

DISCLAIMER = (
    "⚠️ ВАЖНО: Данная система предназначена исключительно для исследовательских целей. "
    "Она НЕ является сертифицированным медицинским диагностическим инструментом. "
    "Все результаты должны интерпретироваться квалифицированным врачом-стоматологом. "
    "Не используйте эти результаты как основу для клинических решений без врачебной консультации."
)

CONDITION_COLORS = {
    "Normal": "🟢",
    "Periodontitis": "🔴",
    "Periodontosis": "🟡",
}

SEVERITY_COLORS = {
    "Mild": "🟡",
    "Medium": "🟠",
    "Severe": "🔴",
    None: "—",
}


def load_pipeline(model_path: str, yolo_path: str = None) -> PeriodontalPipeline:
    cfg = OmegaConf.load("configs/base.yaml")
    return PeriodontalPipeline(model_path, cfg=cfg, yolo_path=yolo_path)


def analyze_opg(image: np.ndarray, pipeline: PeriodontalPipeline):
    """Main analysis function called by Gradio."""
    if image is None:
        return None, "", "", DISCLAIMER

    # Gradio gives RGB; pipeline handles any format
    result = pipeline.run(image)
    diag = result.diagnosis

    # Overlay image (RGB)
    overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB) \
        if result.overlay.shape[2] == 3 else result.overlay

    # Per-tooth table as Markdown
    rows = ["| Зуб | Потеря кости | Кортикальная пластинка | Диагноз | Тяжесть |",
            "|-----|-------------|----------------------|---------|---------|"]
    for t in result.diagnosis.per_tooth:
        cort = f"{t.cortical_coverage:.0%}"
        cond_icon = CONDITION_COLORS.get(t.condition, "")
        sev = t.severity or "—"
        rows.append(f"| {t.tooth_id} | {t.bone_loss_ratio:.0%} | {cort} | {cond_icon} {t.condition} | {sev} |")
    table_md = "\n".join(rows)

    # Overall diagnosis card
    cond_icon = CONDITION_COLORS.get(diag.overall_condition, "")
    sev_icon = SEVERITY_COLORS.get(diag.overall_severity)
    summary = (
        f"### {cond_icon} Основной диагноз: **{diag.overall_condition}**\n\n"
        f"**Распространённость:** {diag.spread or 'N/A'}  \n"
        f"**Тяжесть:** {sev_icon} {diag.overall_severity or 'N/A'}  \n"
        f"**Поражено зубов:** {diag.num_affected}\n\n"
        f"---\n\n{diag.disclaimer}"
    )

    return overlay_rgb, table_md, summary


def build_demo(model_path: str, yolo_path: str = None) -> gr.Blocks:
    pipeline = load_pipeline(model_path, yolo_path=yolo_path)

    with gr.Blocks(title="Анализ тканей пародонта по ОПТГ") as demo:
        gr.Markdown("# 🦷 Анализ тканей пародонта по ОПТГ\n"
                    "*Исследовательский инструмент. Не является медицинским диагностическим прибором.*")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Загрузите ОПТГ-снимок", type="numpy")
                analyze_btn = gr.Button("Анализировать", variant="primary")

                gr.Examples(
                    examples=["app/assets/sample_opg.png"] if Path("app/assets/sample_opg.png").exists() else [],
                    inputs=[input_image],
                )

            with gr.Column(scale=2):
                output_overlay = gr.Image(label="Сегментация (зубы / кость / кортикальная пластинка)")

        with gr.Row():
            output_table = gr.Markdown(label="Анализ по зубам")

        with gr.Row():
            output_summary = gr.Markdown(label="Итоговый диагноз")

        analyze_btn.click(
            fn=lambda img: analyze_opg(img, pipeline),
            inputs=[input_image],
            outputs=[output_overlay, output_table, output_summary],
        )

        gr.Markdown(
            "---\n"
            "**Цветовая схема масок:**  "
            "🔴 Зубы  |  🔵 Альвеолярная кость  |  🟡 Кортикальная пластинка\n\n"
            "**Степени тяжести:**  "
            "Лёгкая < 1/3 корня  |  Средняя < 2/3  |  Тяжёлая > 2/3"
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="app/models/best_model.onnx")
    parser.add_argument("--yolo", default=None, help="Path to YOLOv8-seg .pt checkpoint for tooth instances")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found at {args.model}.")
        print("Export with: python scripts/export_onnx.py --config configs/<best>.yaml --checkpoint <ckpt>")
        return

    demo = build_demo(args.model, yolo_path=args.yolo)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
