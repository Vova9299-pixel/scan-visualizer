from __future__ import annotations

import base64
import io
import json
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

from app.core.processor import draw_debug_overlay, get_default_settings_values, process_scan_folder


APP_TITLE = "Intraocular 3D volume calculator"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "mntk-logo.png"
ABOUT_DESKTOP_TITLE = "3D Scan Processor (Multi-Object)"
ABOUT_DESKTOP_VERSION = "2.2"


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


def _default_settings_text() -> str:
    return json.dumps(get_default_settings_values(), ensure_ascii=False, indent=2)


def _render_secondary_tools() -> None:
    default_settings_text = _default_settings_text()
    st.session_state.setdefault("settings_editor_text", "")
    st.session_state.setdefault("settings_editor_upload_sig", "")

    with st.popover("⋯", use_container_width=True):
        settings_tab, about_tab = st.tabs(["Настройки", "О приложении"])

        with settings_tab:
            settings_import = st.file_uploader(
                "Импортировать settings.json",
                type=["json"],
                help="Файл будет загружен в редактор ниже. Если редактор пустой, используются встроенные настройки или settings.json из ZIP.",
                key="settings_json_popover",
            )
            if settings_import is not None:
                uploaded_bytes = settings_import.getvalue()
                upload_sig = f"{settings_import.name}:{len(uploaded_bytes)}"
                if st.session_state["settings_editor_upload_sig"] != upload_sig:
                    try:
                        parsed = json.loads(uploaded_bytes.decode("utf-8"))
                        st.session_state["settings_editor_text"] = json.dumps(parsed, ensure_ascii=False, indent=2)
                        st.session_state["settings_editor_upload_sig"] = upload_sig
                    except Exception:
                        st.warning("Не удалось прочитать загруженный `settings.json`. Проверьте кодировку и формат JSON.")

            settings_action_col_a, settings_action_col_b = st.columns(2)
            with settings_action_col_a:
                if st.button("Подставить шаблон", use_container_width=True, key="settings_fill_template"):
                    st.session_state["settings_editor_text"] = default_settings_text
            with settings_action_col_b:
                if st.button("Очистить", use_container_width=True, key="settings_clear_editor"):
                    st.session_state["settings_editor_text"] = ""
                    st.session_state["settings_editor_upload_sig"] = ""

            st.text_area(
                "Редактор settings.json",
                key="settings_editor_text",
                height=320,
                help="Если редактор не пустой, его содержимое будет сохранено как `settings.json` перед запуском обработки.",
            )

            current_settings_text = st.session_state.get("settings_editor_text", "").strip()
            if current_settings_text:
                try:
                    json.loads(current_settings_text)
                    st.caption("JSON валиден и будет применён при запуске обработки.")
                except json.JSONDecodeError as exc:
                    st.warning(f"JSON пока невалиден: {exc.msg} (строка {exc.lineno})")

            download_col_a, download_col_b = st.columns(2)
            with download_col_a:
                st.download_button(
                    "Скачать шаблон",
                    data=default_settings_text,
                    file_name="settings.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with download_col_b:
                st.download_button(
                    "Скачать текущее",
                    data=current_settings_text or default_settings_text,
                    file_name="settings-current.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with about_tab:
            st.markdown(
                f"""
                **{ABOUT_DESKTOP_TITLE}**

                Версия desktop: `{ABOUT_DESKTOP_VERSION}`

                Веб-версия сохраняет исходную вычислительную логику из `new_code.py`:
                распознавание срезов, построение 3D-модели, интерполяцию, расчёт объёма и плотности.

                Здесь же доступны browser-friendly аналоги desktop-функций:
                `3D`-ориентир, `2D`-отладка, diagnostics report и работа с `settings.json`.
                """
            )


def _build_3d_figure(process_result) -> go.Figure:
    fig = go.Figure()
    scan_count = max(1, len(process_result.scan_numbers))
    original_color_cache: dict[int, str] = {}
    all_points: list[np.ndarray] = []
    axis_ranges = None

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
            all_points.append(np.column_stack([x_3d, y_3d, z_3d]))
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

    if all_points:
        coords = np.concatenate(all_points, axis=0)
        span = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            float(np.ptp(coords[:, 2])),
            1.0,
        )
        half_span = span * 0.62
        center = np.array(
            [
                float(np.mean(coords[:, 0])),
                float(np.mean(coords[:, 1])),
                float(np.mean(coords[:, 2])),
            ]
        )
        axis_ranges = {
            "x": [float(center[0] - half_span), float(center[0] + half_span)],
            "y": [float(center[1] - half_span), float(center[1] + half_span)],
            "z": [float(center[2] - half_span), float(center[2] + half_span)],
        }
        axis_len = span * 0.18
        triad_origin = np.array(
            [
                float(coords[:, 0].min() - span * 0.28),
                float(coords[:, 1].min() - span * 0.28),
                float(coords[:, 2].min() - span * 0.28),
            ]
        )

        triad_specs = [
            ("X", np.array([axis_len, 0.0, 0.0]), "#f97316"),
            ("Y", np.array([0.0, axis_len, 0.0]), "#22c55e"),
            ("Z", np.array([0.0, 0.0, axis_len]), "#3b82f6"),
        ]
        for label, delta, color in triad_specs:
            end = triad_origin + delta
            fig.add_trace(
                go.Scatter3d(
                    x=[triad_origin[0], end[0]],
                    y=[triad_origin[1], end[1]],
                    z=[triad_origin[2], end[2]],
                    mode="lines+text",
                    text=["", label],
                    textposition="top center",
                    line={"width": 7, "color": color},
                    textfont={"size": 12, "color": color},
                    hoverinfo="skip",
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
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 1},
            "bgcolor": "#0f172a",
            "xaxis": {
                "showbackground": False,
                "gridcolor": "rgba(255,255,255,0.12)",
                "color": "#dbe4ff",
                "range": axis_ranges["x"] if axis_ranges else None,
            },
            "yaxis": {
                "showbackground": False,
                "gridcolor": "rgba(255,255,255,0.12)",
                "color": "#dbe4ff",
                "range": axis_ranges["y"] if axis_ranges else None,
            },
            "zaxis": {
                "showbackground": False,
                "gridcolor": "rgba(255,255,255,0.12)",
                "color": "#dbe4ff",
                "range": axis_ranges["z"] if axis_ranges else None,
            },
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 2.05, "y": 1.55, "z": 0.75},
            },
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return fig


def _render_debug_details(process_result, frame_index: int, frame_position: int, total_frames: int) -> None:
    debug_data = process_result.debug_data
    angle = debug_data["angles"][frame_index]
    scan_number = debug_data["scan_numbers"][frame_index]
    contours = debug_data["contours"][frame_index]
    is_original = scan_number != -1

    if is_original:
        title = f"Кадр {frame_position + 1}/{total_frames} (Оригинал)"
        subtitle = f"Скан: {scan_number} | Угол: {angle:.2f}°"
    else:
        title = f"Кадр {frame_position + 1}/{total_frames} (Интерполированный)"
        subtitle = f"Угол: {angle:.2f}°"

    st.caption(f"{title} | {subtitle}")
    details = [f"Найдено контуров на кадре: {len(contours)}"]
    if contours:
        for idx, contour in enumerate(contours, start=1):
            if contour is None or len(contour) == 0:
                continue
            area_px = cv2.contourArea(contour.astype(np.float32))
            area_mm2 = area_px / max(process_result.scale_x * process_result.scale_y, 1e-9)
            details.append(f"Контур {idx}: {area_mm2:.3f} мм2")
    st.markdown("\n".join(f"- {item}" for item in details))


def _build_debug_figure(process_result, frame_index: int) -> go.Figure:
    debug_data = process_result.debug_data
    if not debug_data["angles"]:
        return go.Figure()

    frame_index = max(0, min(frame_index, len(debug_data["angles"]) - 1))
    scan_number = debug_data["scan_numbers"][frame_index]
    contours = debug_data["contours"][frame_index]
    colors = debug_data["colors"][frame_index]

    if scan_number != -1 and scan_number in process_result.scan_to_image_map:
        image_idx = process_result.scan_to_image_map[scan_number]
        base_image = process_result.images[image_idx]
    else:
        shape = process_result.images[0].shape if process_result.images else (512, 512, 3)
        base_image = np.zeros(shape, dtype=np.uint8)

    rendered = draw_debug_overlay(base_image, contours, colors)
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

    fig = go.Figure(go.Image(z=rendered_rgb))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=min(820, max(420, int(rendered_rgb.shape[0] * 1.05))),
        uirevision=f"debug-{frame_index}",
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, scaleanchor="x", autorange="reversed")
    return fig


def _show_debug_frame(process_result, frame_index: int) -> None:
    st.plotly_chart(
        _build_debug_figure(process_result, frame_index),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )


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
    section_col, actions_col = st.columns([0.9, 0.1])
    with section_col:
        _render_section(
            "Исходные данные",
            "Загрузите снимки в поддерживаемом формате и запустите расчёт одним нажатием.",
        )
    with actions_col:
        _render_secondary_tools()
    uploaded_files = st.file_uploader(
        "Изображения (PNG/JPG/BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )
    uploaded_zip = st.file_uploader("Или ZIP-архив с изображениями", type=["zip"])
    run_clicked = st.button("Запустить обработку", type="primary", use_container_width=True)

if run_clicked:
    with tempfile.TemporaryDirectory(prefix="scan_web_") as tmp_dir:
        work_dir = Path(tmp_dir)
        files_count = 0

        if uploaded_zip is not None:
            files_count += _extract_uploaded_zip(uploaded_zip, work_dir)
        if uploaded_files:
            files_count += _save_uploaded_images(uploaded_files, work_dir)

        settings_editor_text = st.session_state.get("settings_editor_text", "").strip()
        if settings_editor_text:
            try:
                settings_payload = json.loads(settings_editor_text)
            except json.JSONDecodeError as exc:
                st.error(f"`settings.json` невалиден: {exc.msg} (строка {exc.lineno}).")
                st.stop()
            (work_dir / "settings.json").write_text(
                json.dumps(settings_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Суммарный объем (мм3)", f"{result.total_volume_mm3:.3f}")
    metric_b.metric("Суммарный объем (мл)", f"{result.total_volume_ml:.4f}")
    metric_c.metric("Средняя плотность (%)", f"{result.average_density_percent:.2f}")
    metric_d.metric("Найдено объектов", str(len(result.objects)))
    if result.processing_summary:
        st.caption(result.processing_summary)

    if result.processing_report:
        with st.expander("Предупреждения и диагностический отчёт"):
            st.code(result.processing_report, language="text")

    _render_section(
        "Найденные объекты",
        "Список выделенных объектов с их рассчитанным объёмом.",
    )
    _render_object_cards(result)

    _render_section(
        "3D-визуализация",
        "Интерактивная модель для визуальной оценки формы и структуры.",
    )
    st.caption("Оси фиксированы: X — горизонталь, Y — высота, Z — глубина. В сцене добавлен цветной XYZ-ориентир.")
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
        "Покадровый просмотр контуров и промежуточной визуализации. Поддерживаются zoom и pan.",
    )
    frame_count = len(result.debug_data["angles"])
    show_interpolated = st.checkbox("Показывать интерполированные", value=True)
    available_frame_indices = (
        list(range(frame_count))
        if show_interpolated
        else [idx for idx, scan_number in enumerate(result.debug_data["scan_numbers"]) if scan_number != -1]
    )

    if not available_frame_indices:
        st.info("Нет исходных кадров для отображения без интерполированных срезов.")
    else:
        frame_state_key = "debug_frame_position"
        max_frame_position = len(available_frame_indices) - 1
        current_frame_position = min(
            st.session_state.get(frame_state_key, 0),
            max_frame_position,
        )

        if len(available_frame_indices) > 1:
            nav_prev_col, nav_next_col = st.columns(2)
            with nav_prev_col:
                if st.button("← Предыдущее", use_container_width=True, key="debug_prev"):
                    current_frame_position = (current_frame_position - 1) % len(available_frame_indices)
            with nav_next_col:
                if st.button("Следующее →", use_container_width=True, key="debug_next"):
                    current_frame_position = (current_frame_position + 1) % len(available_frame_indices)

            current_frame_position = st.slider(
                "Кадр",
                min_value=0,
                max_value=max_frame_position,
                value=current_frame_position,
                step=1,
            )

        st.session_state[frame_state_key] = current_frame_position
        frame_idx = available_frame_indices[current_frame_position]
        _render_debug_details(result, frame_idx, current_frame_position, len(available_frame_indices))
        _show_debug_frame(result, frame_idx)
