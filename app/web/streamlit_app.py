from __future__ import annotations

import base64
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
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "mntk-logo.png"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(159, 170, 255, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(117, 201, 255, 0.18), transparent 24%),
                    linear-gradient(180deg, #f8f8fc 0%, #f3f5f9 100%);
                color: #162033;
            }

            .block-container {
                max-width: 1040px;
                margin: 0 auto;
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }

            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
            }

            h1, h2, h3 {
                color: #142033;
                letter-spacing: -0.03em;
            }

            .hero-card,
            .object-card {
                background: #ffffff;
                border: 1px solid rgba(20, 32, 51, 0.08);
                box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
            }

            .hero-card {
                position: relative;
                overflow: hidden;
                border-radius: 28px;
                padding: 1.35rem 1.5rem 1.55rem 1.5rem;
                margin-bottom: 1.25rem;
                border: 1px solid rgba(31, 59, 143, 0.12);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 255, 0.96));
                box-shadow: 0 24px 54px rgba(31, 59, 143, 0.10);
            }

            .hero-card::before {
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 6px;
                background: linear-gradient(90deg, #213f9a 0%, #395fd8 55%, #6f8eff 100%);
                pointer-events: none;
            }

            .hero-logo-wrap {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0.4rem 0 0.9rem 0;
            }

            .hero-logo {
                display: block;
                width: min(760px, 100%);
                margin: 0 auto;
            }

            .hero-divider {
                width: min(780px, 100%);
                height: 1px;
                margin: 0.15rem auto 1rem auto;
                background: linear-gradient(90deg, rgba(31, 59, 143, 0), rgba(31, 59, 143, 0.22), rgba(31, 59, 143, 0));
            }

            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                background: rgba(33, 63, 154, 0.08);
                color: #2947a5;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.95rem;
            }

            .hero-title {
                font-family: "Manrope", sans-serif;
                font-size: 2.45rem;
                line-height: 1.08;
                font-weight: 800;
                color: #1a2f73;
                text-align: center;
                max-width: 820px;
                margin: 0 auto 0.75rem auto;
            }

            .hero-subtitle {
                text-align: center;
                color: #4e5d74;
                margin: 0 auto;
                max-width: 760px;
                font-size: 1rem;
                line-height: 1.65;
            }

            .hero-note {
                text-align: center;
                color: #6d7890;
                font-size: 0.92rem;
                margin-top: 0.85rem;
            }

            .section-title {
                font-size: 1.45rem;
                font-weight: 800;
                color: #132033;
                margin: 0 0 0.3rem 0;
            }

            .section-subtitle {
                color: #64748b;
                margin-bottom: 1rem;
            }

            .stButton > button {
                width: 100%;
                border-radius: 14px;
                min-height: 3.15rem;
                font-weight: 800;
                background: linear-gradient(135deg, #1f3b8f, #4064d8);
                border: 0;
                box-shadow: 0 16px 32px rgba(64, 100, 216, 0.28);
            }

            [data-testid="stFileUploader"] {
                width: 100%;
            }

            [data-testid="stFileUploaderDropzone"] {
                border-radius: 18px;
                border: 1px dashed rgba(31, 59, 143, 0.22);
                background: linear-gradient(180deg, rgba(244, 247, 255, 0.9), rgba(255, 255, 255, 0.95));
            }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid rgba(20, 32, 51, 0.08);
                border-radius: 22px;
                padding: 0.9rem;
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            }

            [data-testid="stMetricLabel"] {
                color: #69788f;
                font-weight: 700;
            }

            [data-testid="stMetricValue"] {
                color: #122038;
            }

            .object-card {
                border-radius: 22px;
                padding: 1rem 1.05rem;
                margin-bottom: 0.85rem;
            }

            .object-title {
                font-size: 1.02rem;
                font-weight: 800;
                color: #142033;
                margin-bottom: 0.35rem;
            }

            .object-meta {
                color: #5e6e86;
                line-height: 1.65;
                font-size: 0.94rem;
            }

            .stSlider {
                padding-top: 0.35rem;
            }

            [data-testid="stPlotlyChart"],
            [data-testid="stImage"] {
                border-radius: 22px;
                overflow: hidden;
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1rem;
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                    padding-bottom: 1.25rem;
                }

                .hero-card {
                    padding: 1rem 1rem 1.05rem 1rem;
                    border-radius: 22px;
                }

                .hero-logo-wrap {
                    padding: 0.25rem 0 0.75rem 0;
                }

                .hero-divider {
                    margin-bottom: 0.85rem;
                }

                .hero-title {
                    font-size: 1.82rem;
                    line-height: 1.14;
                }

                .hero-subtitle {
                    font-size: 0.94rem;
                }

                .hero-note {
                    font-size: 0.86rem;
                }

                .section-title {
                    font-size: 1.2rem;
                }

                [data-testid="stHorizontalBlock"] {
                    gap: 0.75rem;
                    flex-direction: column;
                }

                [data-testid="column"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .object-card {
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
            if member_path.name.startswith("."):
                continue
            is_image = member_path.suffix.lower() in allowed
            is_settings = member_path.name.lower() == "settings.json"
            if not is_image and not is_settings:
                continue
            data = archive.read(member)
            output_name = member_path.name
            (target_dir / output_name).write_bytes(data)
            if is_image:
                count += 1
    return count


def _build_3d_figure(process_result) -> go.Figure:
    fig = go.Figure()
    scan_count = max(1, len(process_result.scan_numbers))
    original_color_cache: dict[int, str] = {}

    def color_for_scan(scan_number: int, is_original: bool) -> str:
        if not is_original or scan_number == -1:
            return "rgba(170, 178, 196, 0.35)"

        if scan_number not in original_color_cache:
            idx = process_result.scan_to_image_map.get(scan_number, 0)
            hue = int(179 * idx / max(1, scan_count))
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            original_color_cache[scan_number] = f"rgb({int(bgr[2])}, {int(bgr[1])}, {int(bgr[0])})"
        return original_color_cache[scan_number]

    for obj in process_result.objects:
        for idx, (points_raw, angle) in enumerate(zip(obj.viz_contours, obj.viz_angles)):
            angle_rad = np.deg2rad(angle)
            x_3d = points_raw[:, 0] * np.cos(angle_rad)
            y_3d = points_raw[:, 1]
            z_3d = points_raw[:, 0] * np.sin(angle_rad)
            scan_number = obj.viz_scan_numbers[idx] if idx < len(obj.viz_scan_numbers) else -1
            is_original = obj.is_original[idx] if idx < len(obj.is_original) else False
            line_color = color_for_scan(scan_number, is_original)
            line_width = 3 if is_original else 1.6
            opacity = 0.95 if is_original else 0.28

            fig.add_trace(
                go.Scatter3d(
                    x=x_3d,
                    y=y_3d,
                    z=z_3d,
                    mode="lines",
                    line={"width": line_width, "color": line_color},
                    opacity=opacity,
                    name=f"Object {obj.object_id}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="orbit",
        scene={
            "xaxis_title": "X (горизонталь)",
            "yaxis_title": "Y (высота)",
            "zaxis_title": "Z (глубина)",
            "aspectmode": "data",
            "bgcolor": "#0f172a",
            "xaxis": {"showbackground": False, "gridcolor": "rgba(255,255,255,0.12)", "color": "#dbe4ff"},
            "yaxis": {"showbackground": False, "gridcolor": "rgba(255,255,255,0.12)", "color": "#dbe4ff"},
            "zaxis": {"showbackground": False, "gridcolor": "rgba(255,255,255,0.12)", "color": "#dbe4ff"},
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 1.6, "y": 1.2, "z": 1.45},
            },
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


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""

    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_hero() -> None:
    logo_markup = ""
    logo_uri = _logo_data_uri()
    if logo_uri:
        logo_markup = (
            f'<img class="hero-logo" src="{logo_uri}" '
            'alt="ФГАУ НМИЦ МНТК Микрохирургия глаза" />'
        )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-logo-wrap">{logo_markup}</div>
            <div class="hero-divider"></div>
            <div style="text-align:center;">
                <span class="eyebrow">Клиническая визуализация</span>
            </div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">
                Загрузите серию снимков, запустите расчёт и получите
                2D/3D визуализацию, объём и плотность в одном веб-интерфейсе.
            </div>
            <div class="hero-note">Инструмент для анализа внутриглазных снимков и оценки объёма на базе серии срезов.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_object_cards(process_result) -> None:
    if not process_result.objects:
        st.warning("Объекты не найдены.")
        return

    left_col, right_col = st.columns(2)
    for idx, obj in enumerate(process_result.objects):
        target_col = left_col if idx % 2 == 0 else right_col
        with target_col:
            st.markdown(
                f"""
                <div class="object-card">
                    <div class="object-title">Объект {obj.object_id}</div>
                    <div class="object-meta">
                        Объём: <strong>{obj.volume_mm3:.3f} мм3</strong><br/>
                        Объём: <strong>{obj.volume_ml:.4f} мл</strong><br/>
                        Срезов в модели: <strong>{len(obj.final_contours)}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.set_page_config(page_title=APP_TITLE, layout="centered")
_inject_styles()
_render_hero()

with st.container():
    _render_section(
        "Исходные данные",
        "Загрузите снимки в поддерживаемом формате и запустите расчёт одним нажатием.",
    )
    uploaded_files = st.file_uploader(
        "Изображения (PNG/JPG/BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )
    uploaded_zip = st.file_uploader("Или ZIP-архив с изображениями", type=["zip"])
    settings_file = st.file_uploader(
        "Файл настроек settings.json (необязательно)",
        type=["json"],
        help="Загрузите тот же settings.json, который использовался в исходном приложении, чтобы метрики совпадали.",
    )
    run_clicked = st.button("Запустить обработку", type="primary", use_container_width=True)

if run_clicked:
    with tempfile.TemporaryDirectory(prefix="scan_web_") as tmp_dir:
        work_dir = Path(tmp_dir)
        files_count = 0

        if uploaded_zip is not None:
            files_count += _extract_uploaded_zip(uploaded_zip, work_dir)
        if uploaded_files:
            files_count += _save_uploaded_images(uploaded_files, work_dir)
        if settings_file is not None:
            (work_dir / "settings.json").write_bytes(settings_file.getbuffer())

        if files_count == 0:
            st.error("Не найдено загруженных изображений для обработки.")
            st.stop()

        with st.spinner("Обработка изображений..."):
            try:
                settings_path = work_dir / "settings.json"
                result = process_scan_folder(
                    work_dir,
                    settings_path=settings_path if settings_path.exists() else None,
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()

        st.session_state["process_result"] = result
        st.success(f"Обработка завершена. Загружено файлов: {files_count}")

result = st.session_state.get("process_result")

if result is not None:
    _render_section(
        "Итоговые метрики",
        "Ключевые показатели по текущему набору снимков.",
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Суммарный объем (мм3)", f"{result.total_volume_mm3:.3f}")
    metric_b.metric("Суммарный объем (мл)", f"{result.total_volume_ml:.4f}")
    metric_c.metric("Средняя плотность (%)", f"{result.average_density_percent:.2f}")

    _render_section(
        "Найденные объекты",
        "Список выделенных объектов с их рассчитанным объёмом.",
    )
    _render_object_cards(result)

    _render_section(
        "3D-визуализация",
        "Интерактивная модель для визуальной оценки формы и структуры.",
    )
    st.caption("Оси фиксированы: X — горизонталь, Y — высота, Z — глубина.")
    st.plotly_chart(
        _build_3d_figure(result),
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "pan3d",
                "tableRotation",
                "resetCameraLastSave3d",
                "hoverClosest3d",
            ],
        },
    )

    _render_section(
        "2D-отладочный просмотр",
        "Покадровый просмотр контуров и промежуточной визуализации.",
    )
    frame_count = len(result.debug_data["angles"])
    if frame_count > 1:
        frame_idx = st.slider(
            "Кадр",
            min_value=0,
            max_value=frame_count - 1,
            value=0,
            step=1,
        )
    else:
        frame_idx = 0
    _show_debug_frame(result, frame_idx)
