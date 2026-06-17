"""
Aggregate metrics from reports/results/metrics_summary.csv into
a formatted comparison table and save plots.

Usage:
    python scripts/compare_models.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS_FILE = "reports/results/metrics_summary.csv"
FIGURES_DIR = Path("reports/figures")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METRICS_FILE)
    print("\n=== Model Comparison Table ===")
    print(df.to_string(index=False))

    # Bar chart: mIoU per model
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="mIoU", ax=ax, palette="Blues_d")
    ax.set_title("mIoU on Test Set — Architecture Comparison")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mIoU")
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "miou_comparison.png", dpi=150)
    plt.close(fig)

    # Dice per class heatmap
    dice_cols = [c for c in df.columns if c.startswith("Dice_")]
    if dice_cols:
        dice_df = df.set_index("model")[dice_cols].rename(
            columns=lambda c: c.replace("Dice_", "")
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(dice_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
        ax.set_title("Dice Score per Class — Architecture Comparison")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dice_per_class_heatmap.png", dpi=150)
        plt.close(fig)

    # FPS vs mIoU scatter
    if "FPS" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row in df.iterrows():
            ax.scatter(row["FPS"], row["mIoU"], s=100)
            ax.annotate(row["model"], (row["FPS"] + 0.3, row["mIoU"]), fontsize=9)
        ax.set_xlabel("FPS (higher = faster)")
        ax.set_ylabel("mIoU (higher = better)")
        ax.set_title("Speed vs Accuracy Trade-off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "speed_vs_accuracy.png", dpi=150)
        plt.close(fig)

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
