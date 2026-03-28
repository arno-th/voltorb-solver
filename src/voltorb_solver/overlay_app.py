from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import gettempdir

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from voltorb_solver.image_import.screen_parser import Region, ScreenBoardParser, ScreenParseResult


@dataclass(slots=True)
class OverlayState:
    last_input_path: str | None = None
    last_output_path: str | None = None
    parse_result: ScreenParseResult | None = None


def _map_image_to_overlay(overlay_w: int, overlay_h: int, image_w: int, image_h: int) -> QRect:
    if overlay_w <= 0 or overlay_h <= 0 or image_w <= 0 or image_h <= 0:
        return QRect(0, 0, 0, 0)

    image_ar = image_w / image_h
    overlay_ar = overlay_w / overlay_h
    if image_ar > overlay_ar:
        target_w = overlay_w
        target_h = int(round(target_w / image_ar))
        x = 0
        y = (overlay_h - target_h) // 2
    else:
        target_h = overlay_h
        target_w = int(round(target_h * image_ar))
        x = (overlay_w - target_w) // 2
        y = 0
    return QRect(x, y, max(1, target_w), max(1, target_h))


def _map_region_rect(region: Region, mapping_rect: QRect, image_w: int, image_h: int) -> QRect:
    sx = mapping_rect.width() / max(image_w, 1)
    sy = mapping_rect.height() / max(image_h, 1)
    x = mapping_rect.x() + int(round(region.x * sx))
    y = mapping_rect.y() + int(round(region.y * sy))
    w = int(round(region.w * sx))
    h = int(round(region.h * sy))
    return QRect(x, y, max(1, w), max(1, h))


class OverlayBorderWindow(QWidget):
    def __init__(self, color: QColor, thickness: int = 3) -> None:
        super().__init__()
        self._thickness = thickness
        self._color = color
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: none;"
        )

    def show_for_rect(self, rect: QRect, side: str) -> None:
        if side == "top":
            self.setGeometry(rect.x(), rect.y(), rect.width(), self._thickness)
        elif side == "bottom":
            self.setGeometry(rect.x(), rect.bottom() - self._thickness + 1, rect.width(), self._thickness)
        elif side == "left":
            self.setGeometry(rect.x(), rect.y(), self._thickness, rect.height())
        else:
            self.setGeometry(rect.right() - self._thickness + 1, rect.y(), self._thickness, rect.height())
        self.show()


class OverlayLabelWindow(QWidget):
    def __init__(self, text: str, color: QColor) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        label = QLabel(text, self)
        label.setStyleSheet(
            "color: #ffffff;"
            "padding: 2px 8px;"
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "font-weight: 700;"
            "font-size: 12px;"
        )
        label.adjustSize()
        self.resize(label.size())


class OverlayCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._regions: list[Region] = []
        self._image_size: tuple[int, int] | None = None
        self._colors = [
            QColor(15, 188, 249),
            QColor(255, 158, 0),
            QColor(32, 201, 151),
            QColor(255, 99, 132),
            QColor(153, 102, 255),
            QColor(255, 205, 86),
            QColor(0, 200, 83),
        ]

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())

    def set_overlay_data(self, regions: list[Region], image_w: int, image_h: int) -> None:
        self._regions = list(regions)
        self._image_size = (image_w, image_h)
        self.update()

    def clear_overlay(self) -> None:
        self._regions = []
        self._image_size = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._regions or self._image_size is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        mapping_rect = _map_image_to_overlay(self.width(), self.height(), *self._image_size)

        for idx, region in enumerate(self._regions):
            color = self._colors[idx % len(self._colors)]
            pen = QPen(color, 3)
            painter.setPen(pen)

            rect = _map_region_rect(region, mapping_rect, *self._image_size)
            painter.drawRect(rect)

            label_rect = QRect(rect.left() + 6, max(20, rect.top() - 22), 220, 20)
            painter.fillRect(label_rect, QColor(0, 0, 0, 140))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(label_rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, region.name)

        painter.end()

    # Mapping logic is shared with X11-safe border overlays.


class X11SafeOverlay:
    def __init__(self, colors: list[QColor], border_thickness: int = 3) -> None:
        self._colors = colors
        self._border_thickness = border_thickness
        self._segments: list[OverlayBorderWindow] = []
        self._labels: list[OverlayLabelWindow] = []
        self._visible = False
        self._regions: list[Region] = []
        self._image_size: tuple[int, int] | None = None

    def set_overlay_data(self, regions: list[Region], image_w: int, image_h: int) -> None:
        self._regions = list(regions)
        self._image_size = (image_w, image_h)
        if self._visible:
            self._render()

    def clear_overlay(self) -> None:
        self._regions = []
        self._image_size = None
        self._hide_all_windows()

    def show(self) -> None:
        self._visible = True
        self._render()

    def hide(self) -> None:
        self._visible = False
        self._hide_all_windows()

    def _hide_all_windows(self) -> None:
        for window in self._segments:
            window.hide()
            window.deleteLater()
        for label in self._labels:
            label.hide()
            label.deleteLater()
        self._segments.clear()
        self._labels.clear()

    def _render(self) -> None:
        self._hide_all_windows()
        if self._image_size is None or not self._regions:
            return

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        mapping = _map_image_to_overlay(geo.width(), geo.height(), *self._image_size)
        # Translate from screen-local to absolute coordinates.
        mapping.translate(geo.x(), geo.y())

        for idx, region in enumerate(self._regions):
            color = self._colors[idx % len(self._colors)]
            rect = _map_region_rect(region, mapping, *self._image_size)
            for side in ("top", "bottom", "left", "right"):
                segment = OverlayBorderWindow(color, thickness=self._border_thickness)
                segment.show_for_rect(rect, side)
                self._segments.append(segment)

            label = OverlayLabelWindow(region.name, color)
            label_x = rect.left() + 6
            label_y = max(geo.y() + 8, rect.top() - label.height() - 4)
            label.move(label_x, label_y)
            label.show()
            self._labels.append(label)


class OverlayControlWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voltorb Overlay Tool")
        self.setMinimumSize(560, 230)

        self.parser = ScreenBoardParser()
        self.state = OverlayState()
        self.overlay = OverlayCanvas()
        self.x11_overlay = X11SafeOverlay(self.overlay._colors)
        self._platform_name = (QGuiApplication.platformName() or "").lower()
        self._x11_safe_mode = self._is_x11_platform()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Live Overlay + Screenshot Labeling")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        self.status = QLabel("No screenshot parsed yet.")
        self.status.setStyleSheet("color: #334155;")
        layout.addWidget(self.status)

        button_row = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Screen")
        self.capture_btn.clicked.connect(self.capture_screen)
        self.load_btn = QPushButton("Load Screenshot...")
        self.load_btn.clicked.connect(self.load_screenshot)
        self.save_btn = QPushButton("Save Labeled Screenshot")
        self.save_btn.clicked.connect(self.save_labeled)
        self.clear_btn = QPushButton("Clear Overlay")
        self.clear_btn.clicked.connect(self.clear_overlay)
        button_row.addWidget(self.capture_btn)
        button_row.addWidget(self.load_btn)
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.clear_btn)
        layout.addLayout(button_row)

        self.overlay_btn = QPushButton("Show Overlay")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.toggled.connect(self.toggle_overlay)
        layout.addWidget(self.overlay_btn)

        self.mode_btn = QPushButton(
            "Overlay Mode: X11 Safe" if self._x11_safe_mode else "Overlay Mode: Transparent Canvas"
        )
        self.mode_btn.clicked.connect(self.toggle_overlay_mode)
        layout.addWidget(self.mode_btn)

        help_text = QLabel(
            "Use Capture/Load to parse a screenshot, then enable Show Overlay to draw detected regions"
            " over your desktop view. On i3/X11, use X11 Safe mode if transparent overlays appear black."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #475569;")
        layout.addWidget(help_text)

        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f8fafc;
                color: #0f172a;
            }
            QPushButton {
                background: #1d4ed8;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton:checked {
                background: #14532d;
            }
            QPushButton:hover {
                background: #1e40af;
            }
            """
        )

    def capture_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self._show_error("No active display found for screen capture.")
            return

        output_path = str(Path(gettempdir()) / "voltorb_overlay_capture.png")
        pixmap = screen.grabWindow(0)
        if pixmap.isNull() or not pixmap.save(output_path):
            self._show_error("Failed to capture the current screen.")
            return
        self._parse_and_apply(output_path)

    def load_screenshot(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Screenshot",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not selected:
            return
        self._parse_and_apply(selected)

    def save_labeled(self) -> None:
        if self.state.last_input_path is None:
            self._show_error("Capture or load a screenshot first.")
            return

        target, _ = QFileDialog.getSaveFileName(
            self,
            "Save Labeled Screenshot",
            str(Path(self.state.last_input_path).with_name("labeled_screenshot.png")),
            "PNG (*.png)",
        )
        if not target:
            return

        try:
            result = self.parser.annotate(self.state.last_input_path, target)
        except Exception as exc:
            self._show_error(f"Failed to save labeled screenshot: {exc}")
            return

        self.state.last_output_path = target
        self.state.parse_result = result
        self.status.setText(f"Saved labeled screenshot: {target}")

    def clear_overlay(self) -> None:
        self.overlay.clear_overlay()
        self.x11_overlay.clear_overlay()
        self.state.parse_result = None
        self.state.last_input_path = None
        self.status.setText("Overlay cleared.")

    def toggle_overlay(self, checked: bool) -> None:
        self.overlay_btn.setText("Hide Overlay" if checked else "Show Overlay")
        if checked:
            self._show_active_overlay()
        else:
            self._hide_all_overlays()

    def toggle_overlay_mode(self) -> None:
        self._x11_safe_mode = not self._x11_safe_mode
        self.mode_btn.setText(
            "Overlay Mode: X11 Safe" if self._x11_safe_mode else "Overlay Mode: Transparent Canvas"
        )

        if self.overlay_btn.isChecked():
            self._hide_all_overlays()
            self._show_active_overlay()

    def _show_active_overlay(self) -> None:
        if self._x11_safe_mode:
            self.overlay.hide()
            self.x11_overlay.show()
            return
        self.x11_overlay.hide()
        self.overlay.showFullScreen()

    def _hide_all_overlays(self) -> None:
        self.overlay.hide()
        self.x11_overlay.hide()

    def _is_x11_platform(self) -> bool:
        if "xcb" in self._platform_name:
            return True
        session_type = (os.environ.get("XDG_SESSION_TYPE") or "").lower()
        return session_type == "x11"

    def _parse_and_apply(self, image_path: str) -> None:
        try:
            result = self.parser.parse(image_path)
        except Exception as exc:
            self._show_error(f"Failed to parse screenshot: {exc}")
            return

        self.state.last_input_path = image_path
        self.state.parse_result = result
        self.overlay.set_overlay_data(result.regions, result.image_width, result.image_height)
        self.x11_overlay.set_overlay_data(result.regions, result.image_width, result.image_height)

        warning_text = ""
        if result.warnings:
            warning_text = f" Warnings: {' | '.join(result.warnings)}"
        self.status.setText(
            f"Parsed {len(result.regions)} regions from {Path(image_path).name}.{warning_text}"
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Voltorb Overlay", message)


def run_overlay_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = OverlayControlWindow()
    window.show()
    return app.exec()