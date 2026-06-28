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

### 5 обученных архитектур (+ бонусная)

| # | Модель | Backbone | mIoU (test) | FPS | Статус |
|---|--------|----------|-------------|-----|--------|
| 1 | U-Net | ResNet-50 | 0.588 | 51.6 | ✅ обучена |
| 2 | U-Net++ | EfficientNet-B4 | 0.609 | 23.8 | ✅ обучена |
| 3 | SegFormer | MiT-B3 | 0.589 | 23.2 | ✅ обучена |
| 4 | DeepLabV3+ | ResNet-101 | 0.623 | 41.2 | ✅ обучена |
| 5 | **Swin-UNet** | **Swin-B (IN-22k)** | **0.638** | **21.1** | ✅ **лучшая** |
| — | YOLOv8-seg | YOLOv8m | — | — | instance seg., отдельная парадигма |

> mIoU считается по классам background + tooth (маски bone/cortical в процессе разметки)

---

## Структура проекта

```
Parodo/
├── configs/            # YAML-конфиги для каждой модели (base.yaml + 5 моделей)
├── data/
│   ├── raw/            # Исходные датасеты (DENTEX 2023)
│   ├── processed/      # Предобработанные PNG + маски + metadata.csv
│   ├── annotations/    # LabelMe JSON — ручная разметка bone/cortical (~100 снимков)
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

# 2. Скачать датасет DENTEX 2023 (Zenodo record 7812323, ~10.9 GB)
python scripts/download_dentex.py

# 3. Предобработка: CLAHE + resize до 1024×512 + генерация масок зубов
python scripts/preprocess.py
# → 1571 снимок в data/processed/

# 4. Создать сплиты (70/15/15)
python scripts/create_splits.py
# → train=1101, val=235, test=235

# 5. Проверить пайплайн (2 батча)
python scripts/train.py --config configs/unet.yaml --fast_dev_run

# 6. Обучить модели
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/unetpp.yaml
python scripts/train.py --config configs/segformer.yaml
python scripts/train.py --config configs/deeplabv3plus.yaml
python scripts/train.py --config configs/swin_unet.yaml

# Возобновить обучение с чекпоинта:
python scripts/train.py --config configs/swin_unet.yaml --resume checkpoints/<name>.ckpt

# 7. Оценить на тесте
python scripts/evaluate.py --config configs/unet.yaml --checkpoint checkpoints/<name>.ckpt

# 8. Смотреть метрики в MLflow
mlflow ui --port 5000

# 9. Запустить приложение
python scripts/export_onnx.py --config configs/swin_unet.yaml --checkpoint checkpoints/<name>.ckpt
python app/app.py   # → http://localhost:7860
```

---

## Метрики сравнения (тестовая выборка, 235 снимков)

| Модель | mIoU | IoU_bg | Dice_tooth | Dice_bone | Dice_cortical | FPS |
|--------|------|--------|-----------|-----------|--------------|-----|
| U-Net | 0.588 | 0.853 | 0.369 | — | — | 51.6 |
| U-Net++ | 0.609 | 0.865 | 0.403 | — | — | 23.8 |
| SegFormer | 0.589 | 0.863 | 0.357 | — | — | 23.2 |
| DeepLabV3+ | 0.623 | 0.865 | 0.426 | — | — | 41.2 |
| **Swin-UNet** | **0.638** | 0.864 | **0.462** | — | — | 21.1 |

> «—» — разметка bone/cortical в процессе (~100 снимков аннотировано, обучение предстоит)

---

## Датасеты

- **[DENTEX 2023](https://zenodo.org/records/7812323)** (MICCAI Challenge) — 1571 уникальных ОПТГ с COCO-разметкой зубов по квадрантам

Маски альвеолярной кости и кортикальной пластинки в публичных датасетах отсутствуют. Выполняется ручная разметка в LabelMe: ~100 снимков аннотировано (класс `bone`), аннотации хранятся в `data/annotations/`.

---

## Технологический стек

`PyTorch 2.7` · `PyTorch Lightning` · `segmentation-models-pytorch` · `HuggingFace Transformers` · `timm` · `Ultralytics YOLOv8` · `Albumentations` · `MLflow` · `Gradio` · `python-docx`

---

## Ограничения

- ОПТГ — двумерная проекция 3D-анатомии; потеря кости может занижаться/завышаться
- Положение ЦЭГ определяется анатомическим приближением (40/60 % высоты зуба), не прямым детектированием
- Объём размеченных данных ограничен (~200 снимков с полными масками)
- Система **не является** медицинским диагностическим прибором
