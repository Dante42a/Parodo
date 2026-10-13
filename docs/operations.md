# Запуск, мониторинг и откат

## Локальная проверка без весов

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ci.txt ruff
pytest
uvicorn src.api.main:app --port 8000
curl http://localhost:8000/healthz
```

`healthz` проверяет, что процесс жив. `readyz` отвечает 200, когда модель загружена, и 503 — пока она
грузится (`loading`), не настроена (`not_configured`) или не загрузилась (`load_failed`, причина — в `error`).
Там же — вычисленная версия модели и sha256 весов. Анализ без готовой модели завершается HTTP 503.

Образ API содержит библиотеки инференса (torch CPU, segmentation-models-pytorch, ultralytics — версии
закреплены в `requirements-app.txt`), но не библиотеки обучения.

## Демонстрация с весами

1. Разместить проверенные веса в `checkpoints/` вне Git.
2. Скопировать `.env.example` в локальный `.env` и указать пути (версия модели вычисляется сама).
3. Запустить `docker compose up --build`.
4. UI: `http://localhost`; OpenAPI и Prometheus по умолчанию доступны только локально на сервере: `127.0.0.1:8000` и `127.0.0.1:9090`.

## Мониторинг

API публикует число запросов по статусам и гистограмму длительности. Не логируются изображения, имена пациентов или содержимое результата. Для учебной демонстрации контролируются ошибки 422/503 и задержка анализа.

## Релиз и откат

Версия релиза включает commit SHA, контрольные суммы весов, конфигурацию и версию правил. Откат выполняется возвратом к предыдущему тегу контейнера и соответствующему комплекту весов. Обновление модели без повторной приёмки запрещено.

### Как устроен деплой

```text
PR → CI (lint, test, docker) → merge в main → CI на main публикует образы в GHCR (тег = sha)
   → Deploy (.github/workflows/deploy.yml) → сервер: deploy/remote_deploy.sh
```

На сервере (`/opt/parodo`):

```text
checkpoints/          веса, общие для всех релизов: semantic.ckpt, yolo.pt (кладутся вручную)
releases/<sha>/       код релиза + .env с PARODO_TAG=<sha>; хранятся последние 5
current -> releases/  рабочий релиз
```

`remote_deploy.sh` раскладывает релиз, скачивает образы, поднимает compose-проект `parodo` и ждёт
`/readyz` = 200 (до 3 минут). Успех — `current` переключается на новый релиз. Нет — в лог выводится
причина из `/readyz` и логи API, поднимается предыдущий релиз, деплой завершается ошибкой (красный в Actions).

### Ручной деплой и откат

GitHub → Actions → **Deploy** → Run workflow → `sha` нужного коммита `main` (пусто — последний).
Откат = деплой предыдущего sha: образы этого коммита уже есть в GHCR, пересборка не нужна.

### Обновление весов

1. `python scripts/export_weights.py …` — веса без состояния оптимизатора, печатает sha256.
2. Скопировать на сервер и положить в `/opt/parodo/checkpoints/` (нужен `sudo`).
3. Перезапустить API: Actions → Deploy → Run workflow (повторный деплой текущего sha).
4. Сверить `model_info.semantic_sha256` в `/readyz` с напечатанным в п.1.
