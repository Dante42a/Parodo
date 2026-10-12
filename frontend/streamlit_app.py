import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("PARODO_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Parodo", page_icon="🦷", layout="wide")
st.title("🦷 Parodo — исследовательский анализ ОПТГ")
st.error(
    "Прототип не является медицинским изделием. Результат нельзя использовать "
    "для постановки диагноза; требуется интерпретация врача-стоматолога."
)

uploaded = st.file_uploader("ОПТГ-снимок", type=["png", "jpg", "jpeg", "webp"])
if uploaded is not None:
    st.image(uploaded, caption="Исходное изображение", use_container_width=True)
    if st.button("Выполнить анализ", type="primary"):
        with st.spinner("Модель обрабатывает снимок…"):
            try:
                response = requests.post(
                    f"{API_URL}/v1/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                detail = getattr(exc.response, "text", str(exc))
                st.error(f"API недоступен: {detail}")
            else:
                left, right = st.columns(2)
                with left:
                    st.subheader("Визуализация")
                    overlay = Image.open(io.BytesIO(base64.b64decode(result["overlay_png_base64"])))
                    st.image(overlay, use_container_width=True)
                with right:
                    st.subheader("Исследовательский результат")
                    st.metric("Статус", result["status"])
                    st.write("Группа:", result["overall_condition"])
                    st.write("Тяжесть:", result.get("overall_severity") or "—")
                    st.write("Версия модели:", result["model_version"])
                    st.write("Версия правил:", result["rule_version"])
                    st.write("Время:", f'{result["processing_ms"]:.0f} мс')
                if result["teeth"]:
                    st.dataframe(pd.DataFrame(result["teeth"]), use_container_width=True)
                for reason in result.get("reasons", []):
                    st.warning(reason)
                st.caption(result["disclaimer"])
