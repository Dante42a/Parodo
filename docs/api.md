# API анализа ОПТГ — справочник для фронтенда

Для кого: разработчик интерфейса (Streamlit, `frontend/streamlit_app.py`).
Источник истины — код: `src/api/main.py` (эндпоинты), `src/api/schemas.py` (поля ответов).
Интерактивная схема всех запросов: `http://<api>:8000/docs` (OpenAPI, генерируется FastAPI).

Адрес API внутри Docker Compose: `http://api:8000` (переменная `PARODO_API_URL`); снаружи сервера API
не опубликован — к нему ходит только фронтенд.

## Коротко

| Запрос | Зачем фронтенду |
|---|---|
| `GET /readyz` | при открытии страницы — готова ли модель, её версия, предупреждения |
| `POST /v1/analyze` | отправить снимок, получить результат |
| `GET /healthz` | фронтенду не нужен — это проверка «процесс жив» для Docker |
| `GET /metrics` | фронтенду не нужен — метрики для Prometheus |

## `GET /readyz` — готова ли модель

Модель грузится при старте API (на сервере ~10 с после запуска контейнера). Пока она не готова,
анализ невозможен — `/readyz` объясняет почему.

| HTTP | `status` | Что показать пользователю |
|---|---|---|
| 200 | `ready` | можно загружать снимок; показать `model_version` |
| 503 | `loading` | «Модель загружается, подождите…» — повторить запрос через 2–5 с |
| 503 | `not_configured` | «Сервис не настроен» — ошибка администрирования, текст из `error` |
| 503 | `load_failed` | «Модель не загрузилась» — текст из `error` (там конкретная причина) |
| 503 | `not_started` | как `loading` (кратковременно сразу после старта) |

Пример ответа (модель готова):

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "deeplabv3plus-resnet101@a718ab51+yolo@67098f88+cfg@710b2e8f",
  "error": null,
  "warnings": [],
  "model_info": {
    "arch": "deeplabv3plus/resnet101",
    "semantic_sha256": "a718ab51da6bf4df…",
    "yolo_sha256": "67098f88fcce9935…",
    "config": "configs/deeplabv3plus.yaml",
    "config_sha256": "710b2e8f6523c9fa…",
    "semantic_path": "/models/semantic.ckpt",
    "yolo_path": "/models/yolo.pt"
  }
}
```

- `warnings` — модель работает, но с оговоркой. Сейчас возможна одна: YOLO не подключён, зубы выделяются
  упрощённым алгоритмом. Стоит показать как `st.warning`.
- `model_info` — для администрирования (сверка файлов весов), пользователю показывать не обязательно.

## `POST /v1/analyze` — анализ снимка

Запрос: `multipart/form-data`, поле **`file`**. Тип файла (`Content-Type` части) обязателен:
`image/png`, `image/jpeg` или `image/webp`. Размер — до 20 МБ, изображение — не меньше 64×64.

```python
response = requests.post(
    f"{API_URL}/v1/analyze",
    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
    timeout=180,
)
```

**Время ответа** на сервере (1 CPU) — около 12 с на снимок. Анализы выполняются строго по одному:
если кто-то уже анализирует, запрос ждёт в очереди до 120 с. Поэтому `timeout=180` в клиенте — правильно,
и нужен индикатор ожидания (`st.spinner`).

### Ответ 200

```json
{
  "request_id": "f1837d60-73ee-4c78-86b5-1a62dc623291",
  "status": "ok",
  "overall_condition": "Periodontitis",
  "spread": "Generalized",
  "overall_severity": "Medium",
  "num_affected": 24,
  "reasons": [],
  "rule_version": "research-rules-v1",
  "model_version": "deeplabv3plus-resnet101@a718ab51+yolo@67098f88+cfg@710b2e8f",
  "processing_ms": 12186.57,
  "teeth": [
    {"tooth_id": 1, "condition": "Normal", "severity": null,
     "bone_loss_ratio": 0.0, "cortical_coverage": 0.0, "rule_id": "KB-02"},
    {"tooth_id": 2, "condition": "Periodontitis", "severity": "Mild",
     "bone_loss_ratio": 0.31, "cortical_coverage": 0.0, "rule_id": "KB-03-05"}
  ],
  "overlay_png_base64": "iVBORw0KGgoAAAANSUhEUgAABAAAAAIA…",
  "disclaimer": "Исследовательский прототип. Не является медицинским изделием…"
}
```

(реальный ответ на снимке `train_0.png`; массив `teeth` сокращён — там 30 зубов)

### Поля результата

| Поле | Значения | Как показывать |
|---|---|---|
| `status` | `ok` — результат получен; `needs_review` — автоматического вывода нет, нужен врач | при `needs_review` — выделить, показать `reasons` |
| `overall_condition` | `Normal`, `Periodontitis`, `Periodontosis`, `InsufficientData` | итоговая группа по снимку (худшая по зубам) |
| `overall_severity` | `Mild`, `Medium`, `Severe`, `null` (при `Normal`) | тяжесть |
| `spread` | `Local` (меньше 4 поражённых зубов), `Generalized` (4 и больше), `null` | распространённость |
| `num_affected` | число | сколько зубов не в норме |
| `reasons` | список строк | пояснения и предупреждения — **показывать всегда**, если не пуст (`st.warning`) |
| `model_version` | строка | версия модели — показывать мелко; меняется при смене весов или правил |
| `rule_version` | строка | версия правил классификации |
| `processing_ms` | число | время анализа (без ожидания в очереди) |
| `overlay_png_base64` | PNG в base64, 1024×512 | снимок с наложенной разметкой — главная картинка результата |
| `disclaimer` | строка | **обязательно показывать** под результатом |
| `request_id` | UUID | для обращения в поддержку/поиска в логах |

Поля зуба (`teeth[]`):

| Поле | Смысл |
|---|---|
| `tooth_id` | порядковый номер найденного зуба на снимке. **Не номер по FDI** — соответствие номерам зубной формулы API пока не отдаёт |
| `condition` | `Normal` / `Periodontitis` (`Periodontosis` сейчас не назначается — кортикальная пластинка размечена мало) |
| `severity` | `Mild` (потеря кости ≤ 1/3 корня), `Medium` (≤ 2/3), `Severe` (> 2/3), `null` для `Normal` |
| `bone_loss_ratio` | доля длины корня без кости, 0…1 (`0.31` = 31 %). Норма — меньше 0.10 |
| `cortical_coverage` | доля кортикальной пластинки у гребня, 0…1. В классификации пока не участвует |
| `rule_id` | какое правило базы знаний сработало (`KB-02` — норма, `KB-03-05` — убыль кости) |

Картинка разметки:

```python
overlay = Image.open(io.BytesIO(base64.b64decode(result["overlay_png_base64"])))
st.image(overlay, use_container_width=True)
```

### Ошибки

Тело ошибки — JSON `{"detail": "<текст на русском>"}`. Текст можно показывать пользователю как есть.

| HTTP | Когда | Что показать | Повторять? |
|---|---|---|---|
| 422 | файл не тот: формат, пустой, битый, больше 20 МБ, меньше 64×64 | `detail` — пользователь должен выбрать другой файл | нет |
| 503 | модель не готова (`detail` — причина, как в `/readyz`) или сервис занят другим анализом дольше 120 с | `detail` + «повторите позже» | да, через некоторое время |
| 500 | сбой анализа конкретного снимка; `detail` = «анализ не выполнен, код ошибки XXXX» | `detail` — код нужен разработчикам для поиска в логах | можно один раз |

Особый случай 422: если в запросе вообще нет поля `file`, FastAPI отвечает своей ошибкой валидации —
`detail` там **список**, а не строка. Из Streamlit с `files={"file": …}` так не бывает.

Пример обработки в Streamlit:

```python
try:
    response = requests.post(f"{API_URL}/v1/analyze", files=files, timeout=180)
except requests.RequestException:
    st.error("API недоступен — проверьте, что сервис запущен")
else:
    if response.ok:
        result = response.json()
        ...  # показать результат
    else:
        detail = response.json().get("detail", response.text)
        if response.status_code == 422:
            st.error(f"Файл не подходит: {detail}")
        elif response.status_code == 503:
            st.warning(f"Сервис временно недоступен: {detail}. Повторите позже.")
        else:
            st.error(f"Ошибка анализа: {detail}")
```

Проверка готовности при открытии страницы:

```python
ready = requests.get(f"{API_URL}/readyz", timeout=5)
info = ready.json()
if ready.status_code == 200:
    st.caption(f"Модель: {info['model_version']}")
    for warning in info["warnings"]:
        st.warning(warning)
elif info["status"] in ("loading", "not_started"):
    st.info("Модель загружается, обновите страницу через несколько секунд")
else:
    st.error(f"Модель недоступна: {info['error']}")
```

## Что стоит поправить в текущем `frontend/streamlit_app.py`

- При ошибке сейчас выводится сырой ответ (`exc.response.text` — JSON целиком). Лучше показывать `detail`
  и различать 422 / 503 / 500 — см. пример выше.
- Готовность модели не проверяется: при `loading` пользователь получит 503 только после нажатия
  кнопки. Проверка `/readyz` при открытии страницы снимет этот вопрос.
- Таблица `teeth` выводится как есть — `bone_loss_ratio` и `cortical_coverage` удобнее показывать в процентах,
  а `tooth_id` подписать как «№ на снимке».
