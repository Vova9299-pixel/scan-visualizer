from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.processor import draw_debug_overlay, process_scan_folder


APP_TITLE = "Intraocular 3D volume calculator"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f6f7fb;
            }

            .block-container {
                max-width: 920px;
                margin: 0 auto;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3 {
                text-align: center;
            }

            .app-subtitle {
                text-align: center;
                color: #4a5568;
                margin: 0 auto 1.5rem auto;
                max-width: 680px;
            }

            .upload-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 1.25rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 1rem;
            }

            .stButton > button {
                width: 100%;
                border-radius: 12px;
                min-height: 3rem;
                font-weight: 600;
            }

            [data-testid="stFileUploader"] {
                width: 100%;
            }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 0.75rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1rem;
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                    padding-bottom: 1.25rem;
                }

                h1 {
                    font-size: 1.7rem;
                    line-height: 1.25;
                }

                h2 {
                    font-size: 1.2rem;
                }

                .app-subtitle {
                    font-size: 0.95rem;
                    margin-bottom: 1rem;
                }

                [data-testid="stHorizontalBlock"] {
                    gap: 0.75rem;
                    flex-direction: column;
                }

                [data-testid="column"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .upload-card {
                    padding: 1rem;
                    border-radius: 14px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_uploaded_images(files: List[io.BytesIO], target_dir: Path) -> int:
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    saved = 0
    for file in files:
        ext = Path(file.name).suffix.lower()
        if ext not in allowed:
            continue
        out_path = target_dir / file.name
        out_path.write_bytes(file.getbuffer())
        saved += 1
    return saved


def _extract_uploaded_zip(uploaded_zip: io.BytesIO, target_dir: Path) -> int:
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    count = 0
    with zipfile.ZipFile(uploaded_zip) as archive:
        for member in archive.namelist():
            member_path = Path(member)
            if member_path.suffix.lower() not in allowed or member_path.name.startswith("."):
                continue
            data = archive.read(member)
            output_name = member_path.name
            (target_dir / output_name).write_bytes(data)
            count += 1
    return count


def _build_3d_figure(process_result) -> go.Figure:
    fig = go.Figure()

    for obj in process_result.objects:
        for points_raw, angle in zip(obj.viz_contours, obj.viz_angles):
            angle_rad = np.deg2rad(angle)
            x_3d = points_raw[:, 0] * np.cos(angle_rad)
            y_3d = points_raw[:, 1]
            z_3d = points_raw[:, 0] * np.sin(angle_rad)

            fig.add_trace(
                go.Scatter3d(
                    x=x_3d,
                    y=y_3d,
                    z=z_3d,
                    mode="lines",
                    line={"width": 2},
                    name=f"Object {obj.object_id}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return fig


def _show_debug_frame(process_result, frame_index: int) -> None:
    debug_data = process_result.debug_data
    if not debug_data["angles"]:
        st.info("Нет данных для 2D-отладки.")
        return

    frame_index = max(0, min(frame_index, len(debug_data["angles"]) - 1))
    angle = debug_data["angles"][frame_index]
    scan_number = debug_data["scan_numbers"][frame_index]
    contours = debug_data["contours"][frame_index]
    colors = debug_data["colors"][frame_index]

    if scan_number != -1 and scan_number in process_result.scan_to_image_map:
        image_idx = process_result.scan_to_image_map[scan_number]
        base_image = process_result.images[image_idx]
        scan_label = f"Скан #{scan_number}"
    else:
        shape = process_result.images[0].shape if process_result.images else (512, 512, 3)
        base_image = np.zeros(shape, dtype=np.uint8)
        scan_label = "Интерполированный кадр"

    rendered = draw_debug_overlay(base_image, contours, colors)
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

    st.caption(f"{scan_label}, угол: {angle:.2f}°, контуров: {len(contours)}")
    st.image(rendered_rgb, use_container_width=True)


st.set_page_config(page_title=APP_TITLE, layout="centered")
_inject_styles()

st.title(APP_TITLE)
st.markdown(
    """
    <p class="app-subtitle">
        Загрузите набор снимков отдельными файлами или ZIP-архивом,
        затем запустите обработку и просмотрите 2D/3D результаты прямо в браузере.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Изображения (PNG/JPG/BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )
    uploaded_zip = st.file_uploader("Или ZIP-архив с изображениями", type=["zip"])
    run_clicked = st.button("Запустить обработку", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if run_clicked:
    with tempfile.TemporaryDirectory(prefix="scan_web_") as tmp_dir:
        work_dir = Path(tmp_dir)
        files_count = 0

        if uploaded_zip is not None:
            files_count += _extract_uploaded_zip(uploaded_zip, work_dir)
        if uploaded_files:
            files_count += _save_uploaded_images(uploaded_files, work_dir)

        if files_count == 0:
            st.error("Не найдено загруженных изображений для обработки.")
            st.stop()

        with st.spinner("Обработка изображений..."):
            try:
                result = process_scan_folder(work_dir)
            except Exception as exc:
                st.exception(exc)
                st.stop()

        st.session_state["process_result"] = result
        st.success(f"Обработка завершена. Загружено файлов: {files_count}")

result = st.session_state.get("process_result")

if result is not None:
    st.subheader("Итоговые метрики")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Суммарный объем (мм3)", f"{result.total_volume_mm3:.3f}")
    metric_b.metric("Суммарный объем (мл)", f"{result.total_volume_ml:.4f}")
    metric_c.metric("Средняя плотность (%)", f"{result.average_density_percent:.2f}")

    st.subheader("Объекты")
    if result.objects:
        for obj in result.objects:
            st.write(
                f"Object {obj.object_id}: {obj.volume_mm3:.3f} мм3 "
                f"({obj.volume_ml:.4f} мл), срезов: {len(obj.final_contours)}"
            )
    else:
        st.warning("Объекты не найдены.")

    st.subheader("3D визуализация")
    st.plotly_chart(_build_3d_figure(result), use_container_width=True)

    st.subheader("2D отладочный просмотр")
    frame_idx = st.slider(
        "Кадр",
        min_value=0,
        max_value=max(0, len(result.debug_data["angles"]) - 1),
        value=0,
        step=1,
    )
    _show_debug_frame(result, frame_idx)
