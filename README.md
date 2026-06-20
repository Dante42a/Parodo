# Parodo — Анализ тканей пародонта по ОПТГ

Исследовательская система для автоматического выявления заболеваний пародонта на панорамных рентгеновских снимках зубов (ортопантомограммах). Учебный проект практики.

> ⚠️ Не является медицинским диагностическим инструментом. Результаты требуют интерпретации врача-стоматолога.

---

## Что делает система

Принимает ОПТГ-снимок → сегментирует зубы, альвеолярную кость и кортикальную пластинку → измеряет степень убыли кости относительно длины корня каждого зуба → ставит диагноз по правилам.

**Три диагностические категории:**

| Диагноз | Признак на снимке |
|---------|-------------------|
| Норма | Кость не убыла, кортикальная пластинка сохранена |
| Пародонтит | Кортикальная пластинка отсутствует (локальный или генерализованный) |
| Пародонтоз | Равномерная убыль кости при сохранённой кортикальной пластинке |

**Степень тяжести** определяется по `bone_loss_ratio = потеря_кости / длина_корня`:
- Лёгкая: < 1/3 корня
- Средняя: < 2/3 корня
- Тяжёлая: > 2/3 корня

---

## Архитектура

```
ОПТГ-снимок
    │
    ▼
[Этап 1: Сегментация]   ← одна из 6 моделей
    │
    ├─ маска зубов
    ├─ маска альвеолярной кости
    └─ маска кортикальной пластинки
    │
    ▼
[Этап 2: Геометрия + классификация]
    │
    ├─ bone_loss_ratio на каждый зуб
    ├─ cortical_coverage
    └─ итоговый диагноз + степень тяжести
```

### 6 сравниваемых архитектур

| # | Модель | Библиотека | Особенность |
|---|--------|-----------|-------------|
| 1 | U-Net | segmentation-models-pytorch | Базовая линия, ResNet-50 |
| 2 | U-Net++ | segmentation-models-pytorch | Плотные skip-соединения, EfficientNet-B4 |
| 3 | Swin-UNet | timm | Чистый трансформер, Swin-B |
| 4 | SegFormer | HuggingFace | All-MLP декодер, MiT-B3 |
| 5 | YOLOv8-seg | ultralytics | Instance segmentation |
| 6 | DeepLabV3+ | segmentation-models-pytorch | ASPP, ResNet-101 |

---

## Структура проекта

```
Parodo/
├── configs/            # YAML-конфиги для каждой модели (base.yaml + 6 моделей)
├── data/
│   ├── raw/            # Исходные датасеты (DENTEX 2023, Tufts)
│   ├── processed/      # Предобработанные PNG + маски + metadata.csv
│   └── splits/         # train.txt / val.txt / test.txt
├── src/
│   ├── data/           # Dataset, DataModule, аугментации (Albumentations)
│   ├── models/         # 6 архитектур + model_factory.py
│   ├── losses/         # Dice + BCE / Focal Loss
│   ├── metrics/        # mIoU, Dice, AUROC, F1
│   ├── pipeline/       # stage1 (сегментация), stage2 (геометрия + диагноз)
│   ├── training/       # PyTorch Lightning LightningModule
│   └── utils/          # Визуализация масок
├── scripts/
│   ├── download_dentex.py   # Скачать датасет DENTEX 2023
│   ├── preprocess.py        # CLAHE + resize + генерация масок
│   ├── create_splits.py     # Разбивка 70/15/15 по пациентам
│   ├── train.py             # Обучение: python scripts/train.py --config configs/unet.yaml
│   ├── evaluate.py          # Метрики на тестовой выборке
│   ├── compare_models.py    # Сводная таблица + графики
│   ├── export_onnx.py       # Экспорт лучшей модели в ONNX
│   └── generate_report.py   # Генератор отчёта .docx для универа
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_preview.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_segmentation_results.ipynb
│   ├── 05_geometric_analysis.ipynb   # Разбор расчёта bone_loss_ratio
│   ├── 06_classification_eval.ipynb  # AUROC, ROC-кривые
│   └── 07_model_comparison.ipynb     # Итоговая таблица архитектур
├── app/
│   └── app.py          # Gradio UI
└── requirements.txt
```

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Скачать датасет DENTEX 2023 (Zenodo DOI: 10.5281/zenodo.7812257)
python scripts/download_dentex.py

# 3. Предобработка: CLAHE + resize до 1024×512 + генерация масок зубов
python scripts/preprocess.py

# 4. Разметить маски кости и кортикальной пластинки в LabelMe (~200 снимков)
#    Экспортировать COCO JSON и смержить в data/processed/masks/

# 5. Создать сплиты (70/15/15, группировка по пациенту)
python scripts/create_splits.py

# 6. Проверить что всё работает (2 батча)
python scripts/train.py --config configs/unet.yaml --fast_dev_run

# 7. Обучить все модели (по очереди)
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/unetpp.yaml
python scripts/train.py --config configs/segformer.yaml
python scripts/train.py --config configs/swin_unet.yaml
python scripts/train.py --config configs/deeplabv3plus.yaml
# YOLOv8-seg обучается отдельно через Ultralytics CLI

# 8. Смотреть метрики
mlflow ui --port 5000

# 9. Сравнить архитектуры
python scripts/evaluate.py --config configs/unet.yaml --checkpoint checkpoints/best.ckpt
python scripts/compare_models.py   # → reports/figures/

# 10. Запустить приложение
python scripts/export_onnx.py --config configs/unet.yaml --checkpoint checkpoints/best.ckpt
python app/app.py   # → http://localhost:7860
```

---

## Метрики сравнения

| Модель | mIoU | Dice_tooth | Dice_bone | Dice_cortical | AUROC (macro) | F1 (macro) | FPS |
|--------|------|-----------|-----------|--------------|--------------|-----------|-----|
| U-Net | — | — | — | — | — | — | — |
| U-Net++ | — | — | — | — | — | — | — |
| Swin-UNet | — | — | — | — | — | — | — |
| SegFormer | — | — | — | — | — | — | — |
| YOLOv8-seg | — | — | — | — | — | — | — |
| DeepLabV3+ | — | — | — | — | — | — | — |

*Таблица заполняется после обучения.*

---

## Датасеты

- **[DENTEX 2023](https://zenodo.org/record/7812257)** (MICCAI Challenge) — ~3000 ОПТГ с COCO-разметкой зубов
- **[Tufts Dental Database](https://tdd.ece.tufts.edu/)** — ~1000 снимков с поэлементными метками

Маски альвеолярной кости и кортикальной пластинки отсутствуют в публичных датасетах — требуется ручная разметка ~200 снимков (LabelMe/CVAT) с использованием SAM для предварительной сегментации.

---

## Технологический стек

`PyTorch 2.7` · `PyTorch Lightning` · `segmentation-models-pytorch` · `HuggingFace Transformers` · `timm` · `Ultralytics YOLOv8` · `Albumentations` · `MLflow` · `Gradio` · `python-docx`

---

## Ограничения

- ОПТГ — двумерная проекция 3D-анатомии; потеря кости может занижаться/завышаться
- Положение ЦЭГ определяется анатомическим приближением (40/60 % высоты зуба), не прямым детектированием
- Объём размеченных данных ограничен (~200 снимков с полными масками)
- Система **не является** медицинским диагностическим прибором
