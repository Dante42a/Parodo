# Parodo — Анализ тканей пародонта по ОПТГ

Исследовательская система для автоматического выявления заболеваний пародонта на панорамных рентгеновских снимках зубов (ортопантомограммах). Учебный проект практики.

> ⚠️ Не является медицинским диагностическим инструментом. Результаты требуют интерпретации врача-стоматолога.

---

## Что делает система

Принимает ОПТГ-снимок → детектирует зубы по FDI-нотации → сегментирует альвеолярную кость и кортикальную пластинку → измеряет степень убыли кости относительно длины корня каждого зуба → ставит диагноз по правилам.

**Три диагностические категории:**

| Диагноз | Признак на снимке |
|---------|-------------------|
| Норма | Убыль кости < 10% длины корня |
| Пародонтит | Кортикальная пластинка отсутствует (локальный или генерализованный) |
| Пародонтоз | Равномерная убыль кости при сохранённой кортикальной пластинке |

**Степень тяжести** определяется по `bone_loss_ratio = потеря_кости / длина_корня`:
- Лёгкая: < 1/3 корня
- Средняя: < 2/3 корня
- Тяжёлая: > 2/3 корня

---

## Архитектура — двухмодельный пайплайн

```
ОПТГ-снимок
    │
    ├─► [YOLOv8m-seg]  ← AKUDENTAL (333 снимка, 35 FDI-классов)
    │       └─ per-tooth instance masks + FDI-метки
    │
    └─► [Swin-UNet]    ← ручная разметка bone/cortical (~101 снимок)
            ├─ маска альвеолярной кости
            └─ маска кортикальной пластинки
                │
                ▼
    [Этап 2: Геометрия + классификация]
            ├─ bone_loss_ratio на каждый зуб
            ├─ cortical_coverage
            └─ диагноз + степень тяжести
```

---

## Модели

### Инстанс-сегментация зубов (AKUDENTAL, 333 снимка, 35 FDI-классов)

| Модель | Box mAP50 | Mask mAP50 | Статус |
|--------|-----------|------------|--------|
| **YOLOv8m-seg** | **0.970** | **0.960** | ✅ используется в пайплайне |

### Семантическая сегментация кости и кортикальной пластинки

| # | Модель | Backbone | mIoU (val) | Статус |
|---|--------|----------|------------|--------|
| 1 | **SegFormer** | **MiT-B3** | **0.610** | ✅ **лучшая на полном датасете** |
| 2 | U-Net++ | EfficientNet-B4 | 0.592 | ✅ обучена (bone) |
| 3 | DeepLabV3+ | ResNet-101 | 0.591 | ✅ обучена (bone) |
| 4 | Swin-UNet | Swin-B (IN-22k) | 0.557 | ✅ обучена (bone) |
| 5 | U-Net | ResNet-50 | 0.528 | ✅ обучена (bone) |

> Все 5 моделей обучены на едином сплите bone-датасета (214 train / 37 val, 251 размеченный снимок) с одинаковыми аугментациями — честное сравнение. На полном датасете лидирует SegFormer.

---

## Структура проекта

```
Parodo/
├── configs/            # YAML-конфиги для каждой модели (base.yaml + модели)
├── data/
│   ├── raw/
│   │   ├── akudental/  # AKUDENTAL — 333 ОПТГ, 35 FDI-классов, CC-BY 4.0
│   │   └── dentex/     # DENTEX 2023 — 1571 ОПТГ
│   ├── processed/      # Предобработанные PNG + маски + metadata.csv
│   ├── annotations/    # LabelMe JSON — ручная разметка bone/cortical (~101 снимок)
│   └── splits/         # train.txt / val.txt / test.txt
├── src/
│   ├── data/           # Dataset, DataModule, аугментации (Albumentations)
│   ├── models/         # 5 архитектур + model_factory.py
│   ├── losses/         # Dice + BCE / Focal Loss
│   ├── metrics/        # mIoU, Dice, AUROC, F1
│   ├── pipeline/       # stage1 (сегментация), stage2 (геометрия + диагноз)
│   ├── training/       # PyTorch Lightning LightningModule
│   └── utils/          # Визуализация масок
├── scripts/
│   ├── prepare_akudental.py     # Подготовка AKUDENTAL датасета (85/15 сплит)
│   ├── train_yolo_akudental.py  # Обучение YOLOv8m-seg на AKUDENTAL
│   ├── preprocess.py            # CLAHE + resize + генерация масок
│   ├── create_splits.py         # Разбивка 70/15/15 по пациентам
│   ├── train.py                 # Обучение: python scripts/train.py --config configs/unet.yaml
│   ├── evaluate.py              # Метрики на тестовой выборке
│   └── compare_models.py        # Сводная таблица + графики
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_preview.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_segmentation_results.ipynb
│   ├── 05_geometric_analysis.ipynb
│   ├── 06_classification_eval.ipynb
│   └── 07_model_comparison.ipynb
├── app/
│   └── app.py          # Gradio UI
├── checkpoints/
│   ├── swin_unet-epoch=10-val_mIoU=0.7739.ckpt
│   └── yolov8/teeth_akudental/weights/best.pt
└── requirements.txt
```

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Подготовить AKUDENTAL датасет (требует клонирования репозитория AKUDENTAL)
python scripts/prepare_akudental.py

# 3. Обучить YOLOv8m-seg на зубах (35 FDI-классов)
python scripts/train_yolo_akudental.py
# → checkpoints/yolov8/teeth_akudental/weights/best.pt

# 4. Предобработка для семантических моделей
python scripts/preprocess.py
python scripts/create_splits.py

# 5. Проверить пайплайн (2 батча)
python scripts/train.py --config configs/unet.yaml --fast_dev_run

# 6. Обучить модели (сравнение архитектур)
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/unetpp.yaml
python scripts/train.py --config configs/segformer.yaml
python scripts/train.py --config configs/deeplabv3plus.yaml
python scripts/train.py --config configs/swin_unet_bone.yaml

# Возобновить обучение с чекпоинта:
python scripts/train.py --config configs/swin_unet.yaml --resume checkpoints/<name>.ckpt

# 7. Оценить на тесте
python scripts/evaluate.py --config configs/swin_unet.yaml --checkpoint checkpoints/<name>.ckpt

# 8. Сводная таблица сравнения
python scripts/compare_models.py

# 9. Смотреть метрики в MLflow
mlflow ui --port 5000

# 10. Запустить приложение
python app/app.py \
  --model checkpoints/swin_unet-epoch=10-val_mIoU=0.7739.ckpt \
  --yolo  checkpoints/yolov8/teeth_akudental/weights/best.pt
# → http://localhost:7860
```

---

## Метрики сравнения

### Инстанс-сегментация зубов (AKUDENTAL, val 49 снимков)

| Модель | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|--------|-----------|--------------|------------|---------------|
| **YOLOv8m-seg** | **0.970** | **0.722** | **0.960** | **0.665** |

### Семантическая сегментация кости/кортикальной (val, bone-датасет)

| Модель | mIoU | Dice_tooth | Dice_bone | Dice_cortical | FPS |
|--------|------|------------|-----------|---------------|-----|
| U-Net | 0.593 | 0.369 | — | — | 51.6 |
| U-Net++ | — | — | — | — | — |
| SegFormer | — | — | — | — | — |
| DeepLabV3+ | 0.656 | 0.426 | — | — | 41.2 |
| **Swin-UNet** | **0.774** | **0.462** | — | — | 21.1 |

> «—» — модели будут дообучены после пополнения bone/cortical разметки

---

## Датасеты

| Датасет | Снимков | Разметка | Использование |
|---------|---------|----------|---------------|
| **[AKUDENTAL](https://github.com/melihoz/akudental)** | 333 | 35 FDI-классов, 16 точек/зуб, CC-BY 4.0 | Детекция зубов (YOLOv8) |
| **[DENTEX 2023](https://zenodo.org/records/7812323)** | 1571 | COCO-полигоны зубов | Предобучение семантических моделей |
| **Victoria's annotations** | ~101 | LabelMe, классы `bone` / `cortical` | Сегментация кости (Swin-UNet и др.) |

> Маски альвеолярной кости и кортикальной пластинки в публичных датасетах отсутствуют. Выполняется ручная разметка в LabelMe.

---

## Технологический стек

`PyTorch 2.7` · `PyTorch Lightning` · `segmentation-models-pytorch` · `HuggingFace Transformers` · `timm` · `Ultralytics YOLOv8` · `Albumentations` · `MLflow` · `Gradio`

---

## Ограничения

- ОПТГ — двумерная проекция 3D-анатомии; потеря кости может занижаться или завышаться
- Положение ЦЭГ определяется анатомическим приближением (40/60 % высоты зуба), не прямым детектированием
- Объём размеченных данных по кости ограничен (~101 снимок); классификатор пока не использует кортикальную пластинку
- Система **не является** медицинским диагностическим прибором
