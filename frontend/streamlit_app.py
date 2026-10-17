import base64
import binascii
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("PARODO_API_URL", "http://localhost:8000")
READY_TIMEOUT_S = 5
CONDITION_LABELS = {
    "Normal": "Без признаков изменений",
    "Periodontitis": "Пародонтит",
    "Periodontosis": "Пародонтоз",
    "InsufficientData": "Недостаточно данных",
}
SEVERITY_LABELS = {"Mild": "Лёгкая", "Medium": "Средняя", "Severe": "Тяжёлая"}
SPREAD_LABELS = {"Local": "Локальная", "Generalized": "Распространённая"}

st.set_page_config(page_title="Parodo", page_icon="🦷", layout="wide")
st.title("🦷 Parodo - исследовательский анализ ОПТГ")
st.error(
    "Прототип не является медицинским изделием. Результат нельзя использовать "
    "для постановки диагноза; требуется интерпретация врача-стоматолога."
)


def check_readiness() -> tuple[bool, dict | None]:
    try:
        response = requests.get(f"{API_URL}/readyz", timeout=READY_TIMEOUT_S)
    except requests.RequestException:
        return False, None
    try:
        info = response.json()
    except ValueError:
        return False, None
    return response.status_code == 200 and info.get("status") == "ready", info


def show_readiness() -> bool:
    ready, info = check_readiness()
    if info is None:
        st.warning("API пока недоступен. Проверьте, что сервис запущен, и обновите страницу.")
        return False
    status = info.get("status")
    if ready:
        st.success("API и модель готовы. Можно загрузить снимок.")
        if info.get("model_version"):
            st.caption(f"Версия модели: {info['model_version']}")
        for warning in info.get("warnings", []):
            if "YOLO" in warning or "зубы" in warning.lower():
                st.warning("Зубы выделяются упрощённым алгоритмом; точность результата может быть ниже.")
            else:
                st.warning("Модель работает с ограничением. Результат требует особенно внимательной проверки врача.")
        return True
    if status in {"loading", "not_started"}:
        st.info("Модель ещё загружается. Подождите несколько секунд и обновите страницу.")
    elif status == "not_configured":
        st.error("Сервис анализа не настроен. Обратитесь к администратору API.")
    elif status == "load_failed":
        st.error("Модель не загрузилась. Обратитесь к администратору API и повторите позже.")
    else:
        st.warning("Сервис анализа временно недоступен. Обновите страницу через несколько секунд.")
    return False


api_ready = show_readiness()
uploaded = st.file_uploader("ОПТГ-снимок", type=["png", "jpg", "jpeg", "webp"])


def response_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return "Сервис вернул неожиданный ответ. Повторите попытку позже."
    if isinstance(detail, str) and detail:
        return detail
    return "Сервис не смог обработать запрос. Повторите попытку позже."


def friendly_reason(reason: str) -> str:
    if "PARODO_YOLO_PATH" in reason or "YOLO" in reason:
        return "Зубы выделены упрощённым алгоритмом; точность результата может быть ниже."
    return reason


if uploaded is not None:
    try:
        original = Image.open(io.BytesIO(uploaded.getvalue()))
    except (Image.UnidentifiedImageError, OSError):
        st.error("Файл не удалось открыть как изображение. Выберите корректный PNG, JPEG или WebP.")
        original = None
    if original is not None:
        st.image(original, caption="Исходное изображение", use_container_width=True)
    if original is not None and st.button("Выполнить анализ", type="primary", disabled=not api_ready):
        with st.spinner("Модель обрабатывает снимок, это может занять около 12 секунд…"):
            try:
                response = requests.post(
                    f"{API_URL}/v1/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=180,
                )
            except requests.RequestException:
                st.error("Не удалось связаться с API. Проверьте сервис и повторите попытку.")
            else:
                if response.status_code == 200:
                    try:
                        result = response.json()
                        overlay = Image.open(
                            io.BytesIO(base64.b64decode(result["overlay_png_base64"], validate=True))
                        )
                    except (ValueError, KeyError, binascii.Error, Image.UnidentifiedImageError, OSError, TypeError):
                        st.error("API вернул неполный результат. Повторите анализ позже.")
                    else:
                        st.subheader("Изображения")
                        left, right = st.columns(2)
                        with left:
                            st.image(original, caption="Исходный снимок", use_container_width=True)
                        with right:
                            st.image(overlay, caption="Снимок с наложенной маской", use_container_width=True)

                        st.subheader("Итоговая оценка")
                        if result.get("status") == "needs_review":
                            st.warning("Автоматический вывод требует проверки врача.")
                        else:
                            st.success("Автоматическая оценка получена.")
                        summary = st.columns(4)
                        summary[0].metric(
                            "Состояние", CONDITION_LABELS.get(result.get("overall_condition"), "—")
                        )
                        summary[1].metric(
                            "Тяжесть", SEVERITY_LABELS.get(result.get("overall_severity"), "—")
                        )
                        summary[2].metric("Поражено зубов", result.get("num_affected", 0))
                        summary[3].metric(
                            "Распространённость", SPREAD_LABELS.get(result.get("spread"), "—")
                        )
                        st.caption(
                            f"Модель: {result.get('model_version', '—')} · "
                            f"Правила: {result.get('rule_version', '—')} · "
                            f"Время анализа: {result.get('processing_ms', 0):.0f} мс"
                        )
                        for reason in result.get("reasons", []):
                            st.warning(friendly_reason(reason))

                        teeth = result.get("teeth", [])
                        if teeth:
                            teeth_df = pd.DataFrame(teeth).rename(
                                columns={
                                    "tooth_id": "№ на снимке",
                                    "condition": "Состояние",
                                    "severity": "Тяжесть",
                                    "bone_loss_ratio": "Потеря кости",
                                    "cortical_coverage": "Кортикальная пластинка",
                                    "rule_id": "Правило",
                                }
                            )
                            for column in ("Потеря кости", "Кортикальная пластинка"):
                                teeth_df[column] = teeth_df[column].map(lambda value: f"{value:.0%}")
                            teeth_df["Состояние"] = teeth_df["Состояние"].map(
                                lambda value: CONDITION_LABELS.get(value, value)
                            )
                            teeth_df["Тяжесть"] = teeth_df["Тяжесть"].map(
                                lambda value: SEVERITY_LABELS.get(value, value)
                            )
                            teeth_df["Тяжесть"] = teeth_df["Тяжесть"].fillna("—")
                            st.subheader("Показатели по зубам")
                            st.dataframe(teeth_df, use_container_width=True, hide_index=True)
                        st.info(result.get("disclaimer", "Исследовательский результат требует проверки врача."))
                elif response.status_code == 422:
                    st.error(f"Файл не подходит: {response_detail(response)} Выберите другой снимок.")
                elif response.status_code == 503:
                    st.warning("Сервис временно занят или модель не готова. Подождите и повторите анализ.")
                else:
                    st.error("Анализ не выполнен. Повторите попытку позже или обратитесь к администратору API.")
