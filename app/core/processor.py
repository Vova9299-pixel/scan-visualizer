from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ProcessingObjectResult:
    object_id: int
    volume_mm3: float
    volume_ml: float
    final_contours: List[np.ndarray]
    final_angles: List[float]
    final_scan_numbers: List[int]
    is_original: List[bool]
    viz_contours: List[np.ndarray]
    viz_angles: List[float]
    viz_scan_numbers: List[int]


@dataclass
class ProcessingResult:
    total_volume_mm3: float
    total_volume_ml: float
    average_density_percent: float
    objects: List[ProcessingObjectResult]
    debug_data: Dict[str, List]
    images: List[np.ndarray]
    scan_numbers: List[int]
    scan_to_image_map: Dict[int, int]
    scale_x: float
    scale_y: float
    processing_summary: str
    processing_report: str


def _make_dummy_qt_class(name: str):
    return type(name, (), {})


def _install_gui_stubs() -> None:
    """
    new_code.py импортирует PyQt6/pyvistaqt на уровне модуля.
    Для запуска в web-контексте создаем безопасные заглушки GUI.
    """
    if "PyQt6" not in sys.modules:
        pyqt6 = types.ModuleType("PyQt6")
        qtwidgets = types.ModuleType("PyQt6.QtWidgets")
        qtgui = types.ModuleType("PyQt6.QtGui")
        qtcore = types.ModuleType("PyQt6.QtCore")

        qtwidgets.QDialog = _make_dummy_qt_class("QDialog")
        qtwidgets.QMainWindow = _make_dummy_qt_class("QMainWindow")
        qtwidgets.QWidget = _make_dummy_qt_class("QWidget")
        qtwidgets.QMessageBox = _make_dummy_qt_class("QMessageBox")
        qtwidgets.QApplication = _make_dummy_qt_class("QApplication")
        qtwidgets.QGraphicsScene = _make_dummy_qt_class("QGraphicsScene")
        qtwidgets.QGraphicsView = _make_dummy_qt_class("QGraphicsView")
        qtwidgets.QTextEdit = _make_dummy_qt_class("QTextEdit")

        qtgui.QAction = _make_dummy_qt_class("QAction")
        qtgui.QImage = _make_dummy_qt_class("QImage")
        qtgui.QPixmap = _make_dummy_qt_class("QPixmap")

        qtcore.QRectF = _make_dummy_qt_class("QRectF")
        qtcore.Qt = _make_dummy_qt_class("Qt")

        pyqt6.QtWidgets = qtwidgets
        pyqt6.QtGui = qtgui
        pyqt6.QtCore = qtcore

        sys.modules["PyQt6"] = pyqt6
        sys.modules["PyQt6.QtWidgets"] = qtwidgets
        sys.modules["PyQt6.QtGui"] = qtgui
        sys.modules["PyQt6.QtCore"] = qtcore

    if "pyvistaqt" not in sys.modules:
        pyvistaqt = types.ModuleType("pyvistaqt")
        pyvistaqt.QtInteractor = _make_dummy_qt_class("QtInteractor")
        sys.modules["pyvistaqt"] = pyvistaqt


def _load_legacy_module():
    _install_gui_stubs()
    project_root = Path(__file__).resolve().parents[2]
    source_file = project_root / "new_code.py"

    module_name = "legacy_scan_processor"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {source_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_default_settings_values() -> Dict[str, Any]:
    legacy = _load_legacy_module()
    values: Dict[str, Any] = {}
    for key, value in legacy.Settings.__dict__.items():
        if key.startswith("__") or callable(value):
            continue
        if isinstance(value, tuple):
            values[key] = list(value)
        elif isinstance(value, (int, float, bool, str, list)):
            values[key] = value
    return values


def _build_debug_data(
    results: List[Dict],
    scan_to_image_map: Dict[int, int],
    scan_count: int,
) -> Dict[str, List]:
    frames_data: Dict[float, Dict[str, List]] = {}
    gray_color_bgr = (180, 180, 180)

    for res in results:
        for i, contour in enumerate(res["final_contours"]):
            angle = res["final_angles"][i]
            is_orig = res["is_original"][i]
            scan_num = res["final_scan_numbers"][i]

            if angle not in frames_data:
                frames_data[angle] = {"contours": [], "colors": [], "scan_number": -1}

            color_bgr = gray_color_bgr
            if is_orig:
                frames_data[angle]["scan_number"] = scan_num
                if scan_num in scan_to_image_map and scan_count > 0:
                    idx = scan_to_image_map[scan_num]
                    hue = np.array(cv2.cvtColor(
                        np.uint8([[[int(180 * idx / max(1, scan_count)), 255, 255]]]),
                        cv2.COLOR_HSV2BGR,
                    ))[0][0]
                    color_bgr = (int(hue[0]), int(hue[1]), int(hue[2]))

            frames_data[angle]["contours"].append(contour)
            frames_data[angle]["colors"].append(color_bgr)

    sorted_frames = sorted(frames_data.items(), key=lambda item: item[0])
    return {
        "angles": [item[0] for item in sorted_frames],
        "contours": [item[1]["contours"] for item in sorted_frames],
        "colors": [item[1]["colors"] for item in sorted_frames],
        "scan_numbers": [item[1]["scan_number"] for item in sorted_frames],
    }


def process_scan_folder(folder_path: str | Path, settings_path: Optional[str | Path] = None) -> ProcessingResult:
    legacy = _load_legacy_module()
    if settings_path is not None:
        settings_file = Path(settings_path)
        if settings_file.exists():
            legacy.Settings.load(str(settings_file))
    legacy.reset_error_collector()

    folder = Path(folder_path)
    reader = legacy.DataReader(folder)
    image_processor = legacy.ImageProcessor()

    images, arrow_angles, scan_numbers, image_shape = reader.read_images()
    if image_shape is None:
        raise ValueError("Не удалось определить разрешение изображений")

    image_height, image_width = image_shape[:2]
    scan_to_image_map = {num: i for i, num in enumerate(scan_numbers) if num is not None}

    if all(n is not None for n in scan_numbers):
        n_items = len(images)
        if n_items == 0:
            raise ValueError("Изображения не найдены")
        angles = [i * (180.0 / n_items) for i in range(n_items)]
    else:
        angles = arrow_angles

    builder = legacy.ModelBuilder(
        image_width,
        image_height,
        n_resample_points=legacy.Settings.RESAMPLE_N_POINTS_DEFAULT,
    )
    legacy.get_error_collector().set_file_counts(len(images), len(images))

    all_slices_contours = []
    all_densities = []
    for i, img in enumerate(images):
        processed = image_processor.process_image(i, img)
        if isinstance(processed, tuple):
            contours, mean_density = processed
        else:
            contours, mean_density = [], 0.0
        all_densities.append(float(mean_density))
        all_slices_contours.append(contours)

    final_mean_density = float(np.mean(all_densities)) if all_densities else 0.0

    results = builder.process_and_build_all_models(
        all_slices_contours,
        scan_numbers,
        angles,
        center=(image_width / 2, image_height / 2),
    )

    for res in results:
        contours_list = res["final_contours"]
        angles_list = res["final_angles"]

        half_contours = []
        half_angles = []
        for contour_px, ang in zip(contours_list, angles_list):
            if contour_px is None or len(contour_px) < 3:
                continue
            n_full = contour_px.shape[0]
            n_half = n_full // 2 + 1
            right_half = contour_px[:n_half]
            left_half = contour_px[n_half - 1 :]
            half_contours.append(right_half)
            half_angles.append(ang)
            half_contours.append(left_half)
            half_angles.append((ang + 180.0) % 360.0)

        vol_radial = builder.volume_radial_integration(half_contours, half_angles)
        res["volume_mm3"] = float(vol_radial)

    total_volume_mm3 = float(sum(r["volume_mm3"] for r in results))
    total_volume_ml = total_volume_mm3 / legacy.Settings.VOLUME_DIVIDER
    norm_density = final_mean_density / 255.0 * 100.0

    debug_data = _build_debug_data(results, scan_to_image_map, len(scan_numbers))

    objects = [
        ProcessingObjectResult(
            object_id=res["id"],
            volume_mm3=float(res["volume_mm3"]),
            volume_ml=float(res["volume_mm3"]) / legacy.Settings.VOLUME_DIVIDER,
            final_contours=res["final_contours"],
            final_angles=res["final_angles"],
            final_scan_numbers=res["final_scan_numbers"],
            is_original=res["is_original"],
            viz_contours=res["viz_contours"],
            viz_angles=res["viz_angles"],
            viz_scan_numbers=res["final_scan_numbers"],
        )
        for res in sorted(results, key=lambda x: x["id"])
    ]

    return ProcessingResult(
        total_volume_mm3=total_volume_mm3,
        total_volume_ml=float(total_volume_ml),
        average_density_percent=float(norm_density),
        objects=objects,
        debug_data=debug_data,
        images=images,
        scan_numbers=list(scan_numbers),
        scan_to_image_map=scan_to_image_map,
        scale_x=float(builder.scale_x),
        scale_y=float(builder.scale_y),
        processing_summary=str(legacy.get_error_collector().get_summary()),
        processing_report=str(legacy.get_error_collector().get_grouped_report()),
    )


def draw_debug_overlay(image: np.ndarray, contours: List[np.ndarray], colors: List[Tuple[int, int, int]]) -> np.ndarray:
    frame = image.copy()
    for contour, color in zip(contours, colors):
        if contour is not None and len(contour) > 0:
            cv2.drawContours(frame, [contour.astype(np.int32)], -1, color, 2)
    return frame
