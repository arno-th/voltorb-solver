from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import gettempdir

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
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
    selected_screen_index: int = 0
    target_window_id: int | None = None
    target_window_name: str | None = None


OVERLAY_COLORS = [
    QColor(15, 188, 249),
    QColor(255, 158, 0),
    QColor(32, 201, 151),
    QColor(255, 99, 132),
    QColor(153, 102, 255),
    QColor(255, 205, 86),
    QColor(0, 200, 83),
]


def _bind_widget_to_screen(widget: QWidget, screen) -> None:
    if screen is None:
        return

    # Some window managers ignore off-screen geometry unless the toplevel
    # window is explicitly assigned to a target QScreen.
    if hasattr(widget, "setScreen"):
        try:
            widget.setScreen(screen)  # type: ignore[attr-defined]
        except Exception:
            pass

    handle = widget.windowHandle()
    if handle is not None:
        handle.setScreen(screen)


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
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: none;"
        )

    def show_for_rect(self, rect: QRect, side: str, screen=None) -> None:
        _bind_widget_to_screen(self, screen)
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
            | Qt.WindowType.X11BypassWindowManagerHint
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
        self._target_screen = QGuiApplication.primaryScreen()
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

        self._update_geometry_for_target_screen()

    def set_target_screen(self, screen) -> None:
        self._target_screen = screen
        self._update_geometry_for_target_screen()

    def _update_geometry_for_target_screen(self) -> None:
        if self._target_screen is not None:
            _bind_widget_to_screen(self, self._target_screen)
            self.setGeometry(self._target_screen.availableGeometry())

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
        self._target_screen = QGuiApplication.primaryScreen()

    def set_target_screen(self, screen) -> None:
        self._target_screen = screen
        if self._visible:
            self._render()

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

        screen = self._target_screen
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
                segment.show_for_rect(rect, side, screen=screen)
                self._segments.append(segment)

            label = OverlayLabelWindow(region.name, color)
            _bind_widget_to_screen(label, screen)
            label_x = rect.left() + 6
            label_y = max(geo.y() + 8, rect.top() - label.height() - 4)
            label.move(label_x, label_y)
            label.show()
            self._labels.append(label)


class OverlayControlWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voltorb Solver X11 Overlay")
        self.setMinimumSize(780, 360)

        self.parser = ScreenBoardParser()
        self.state = OverlayState()
        self.x11_overlay = X11SafeOverlay(OVERLAY_COLORS)
        self._screens = QGuiApplication.screens()

        root = QWidget()
        root.setObjectName("RootPanel")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_card = QFrame()
        header_card.setObjectName("Card")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(6)

        title = QLabel("Voltorb Solver X11 Overlay")
        title.setObjectName("TitleLabel")
        title.setWordWrap(False)
        header_layout.addWidget(title)

        subtitle = QLabel(
            "Capture a specific monitor, parse board regions, and render the overlay on that same monitor."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)

        self.status = QLabel()
        self.status.setObjectName("StatusPill")
        self.status.setWordWrap(True)
        header_layout.addWidget(self.status)
        layout.addWidget(header_card)

        monitor_card = QFrame()
        monitor_card.setObjectName("Card")
        monitor_layout = QVBoxLayout(monitor_card)
        monitor_layout.setContentsMargins(14, 12, 14, 12)
        monitor_layout.setSpacing(8)

        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(8)
        monitor_label = QLabel("Capture/Overlay Monitor")
        monitor_label.setObjectName("FieldLabel")
        monitor_row.addWidget(monitor_label)
        self.monitor_combo = QComboBox()
        self.monitor_combo.setObjectName("MonitorCombo")
        self.monitor_combo.currentIndexChanged.connect(self._on_monitor_changed)
        monitor_row.addWidget(self.monitor_combo)
        self.refresh_monitors_btn = QPushButton("Refresh Monitors")
        self.refresh_monitors_btn.setObjectName("SecondaryButton")
        self.refresh_monitors_btn.clicked.connect(self._refresh_monitor_list)
        monitor_row.addWidget(self.refresh_monitors_btn)
        monitor_layout.addLayout(monitor_row)

        self.monitor_hint = QLabel("No monitor selected.")
        self.monitor_hint.setObjectName("HintLabel")
        self.monitor_hint.setWordWrap(True)
        monitor_layout.addWidget(self.monitor_hint)

        self.window_hint = QLabel("Capture target: selected monitor.")
        self.window_hint.setObjectName("HintLabel")
        self.window_hint.setWordWrap(True)
        monitor_layout.addWidget(self.window_hint)

        layout.addWidget(monitor_card)

        actions_card = QFrame()
        actions_card.setObjectName("Card")
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(14, 12, 14, 12)
        actions_layout.setSpacing(8)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.capture_btn = QPushButton("Capture Screen")
        self.capture_btn.setObjectName("PrimaryButton")
        self.capture_btn.clicked.connect(self.capture_screen)
        self.target_window_btn = QPushButton("Pick Target Window")
        self.target_window_btn.setObjectName("SecondaryButton")
        self.target_window_btn.clicked.connect(self._handle_target_window_button)
        self.load_btn = QPushButton("Load Screenshot...")
        self.load_btn.setObjectName("SecondaryButton")
        self.load_btn.clicked.connect(self.load_screenshot)
        self.save_btn = QPushButton("Save Labeled Screenshot")
        self.save_btn.setObjectName("SecondaryButton")
        self.save_btn.clicked.connect(self.save_labeled)
        self.clear_btn = QPushButton("Clear Overlay")
        self.clear_btn.setObjectName("DangerButton")
        self.clear_btn.clicked.connect(self.clear_overlay)
        button_row.addWidget(self.capture_btn)
        button_row.addWidget(self.target_window_btn)
        button_row.addWidget(self.load_btn)
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.clear_btn)
        actions_layout.addLayout(button_row)

        self.overlay_btn = QPushButton("Enable Overlay")
        self.overlay_btn.setObjectName("AccentToggle")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.toggled.connect(self.toggle_overlay)
        actions_layout.addWidget(self.overlay_btn)
        layout.addWidget(actions_card)

        help_text = QLabel(
            "Tip: Pick a target emulator window for reliable repeated captures. If no target window is set,"
            " capture uses the selected monitor."
        )
        help_text.setObjectName("HintLabel")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        layout.addStretch(1)

        self._apply_styles()
        self._update_target_window_button()
        self._set_status("No screenshot parsed yet.", level="info")
        self._refresh_monitor_list()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            #RootPanel {
                background: #eef3f7;
                color: #12202f;
            }

            #Card {
                background: #ffffff;
                border: 1px solid #d7e1ea;
                border-radius: 12px;
            }

            #TitleLabel {
                font-size: 22px;
                font-weight: 700;
                color: #0d233a;
            }

            #SubtitleLabel {
                font-size: 13px;
                color: #465c72;
            }

            #FieldLabel {
                font-size: 13px;
                font-weight: 600;
                color: #1f3b54;
            }

            #HintLabel {
                font-size: 12px;
                color: #5b7288;
            }

            #StatusPill {
                border-radius: 8px;
                border: 1px solid #d1deea;
                background: #f5f9fc;
                color: #2f4d66;
                padding: 8px 10px;
                font-size: 12px;
                font-weight: 600;
            }

            QComboBox#MonitorCombo {
                min-height: 34px;
                border: 1px solid #c5d4e1;
                border-radius: 8px;
                padding: 5px 8px;
                background: #ffffff;
                color: #17324a;
            }

            QComboBox#MonitorCombo:hover {
                border-color: #97b0c6;
            }

            QComboBox#MonitorCombo:focus {
                border-color: #4f87b8;
            }

            QPushButton {
                min-height: 34px;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }

            QPushButton#PrimaryButton {
                background: #1767a9;
            }

            QPushButton#PrimaryButton:hover {
                background: #0f578f;
            }

            QPushButton#SecondaryButton {
                background: #5b7288;
            }

            QPushButton#SecondaryButton:hover {
                background: #495f74;
            }

            QPushButton#DangerButton {
                background: #b44949;
            }

            QPushButton#DangerButton:hover {
                background: #9d3e3e;
            }

            QPushButton#AccentToggle {
                background: #1f7d5f;
            }

            QPushButton#AccentToggle:hover {
                background: #18684f;
            }

            QPushButton#AccentToggle:checked {
                background: #14533f;
            }
            """
        )

    def _set_status(self, message: str, level: str = "info") -> None:
        style_map = {
            "info": ("#f5f9fc", "#d1deea", "#2f4d66"),
            "success": ("#edf8f1", "#bfe1cd", "#24553a"),
            "warning": ("#fff8eb", "#efd7a6", "#6f5315"),
            "error": ("#fdeeee", "#e6b6b6", "#7c2d2d"),
        }
        bg, border, fg = style_map.get(level, style_map["info"])
        self.status.setStyleSheet(
            f"border-radius: 8px; border: 1px solid {border}; background: {bg}; color: {fg};"
            "padding: 8px 10px; font-size: 12px; font-weight: 600;"
        )
        self.status.setText(message)

    def _update_monitor_hint(self) -> None:
        screen = self._get_selected_screen()
        if screen is None:
            self.monitor_hint.setText("No monitor selected.")
            return

        geo = screen.geometry()
        self.monitor_hint.setText(
            f"Selected monitor geometry: {geo.width()}x{geo.height()} at ({geo.x()}, {geo.y()})."
        )

    def _update_window_hint(self) -> None:
        if self.state.target_window_id is None:
            self.window_hint.setText("Capture target: selected monitor.")
            return

        name = self.state.target_window_name or "Unnamed window"
        self.window_hint.setText(
            f"Capture target window: #{self.state.target_window_id} ({name})."
        )

    def _update_target_window_button(self) -> None:
        if self.state.target_window_id is None:
            self.target_window_btn.setText("Pick Target Window")
            self.target_window_btn.setToolTip("Select an emulator window for direct capture.")
            return

        self.target_window_btn.setText("Clear Target Window")
        self.target_window_btn.setToolTip("Clear the selected target window and capture from monitor.")

    def _handle_target_window_button(self) -> None:
        if self.state.target_window_id is None:
            self.pick_target_window()
            return
        self.clear_target_window()

    def capture_screen(self) -> None:
        output_path = str(
            Path(gettempdir()) / f"voltorb_overlay_capture_monitor_{self.state.selected_screen_index + 1}.png"
        )

        was_overlay_visible = self.overlay_btn.isChecked()
        if was_overlay_visible:
            self.x11_overlay.hide()

        if self.state.target_window_id is not None:
            pixmap = self._capture_window(self.state.target_window_id)
        else:
            screen = self._get_selected_screen()
            if screen is None:
                if was_overlay_visible:
                    self.x11_overlay.show()
                self._show_error("No monitor selected for screen capture.")
                return
            pixmap = screen.grabWindow(0)

        if pixmap is None or pixmap.isNull() or not pixmap.save(output_path):
            if was_overlay_visible:
                self.x11_overlay.show()
            target_desc = (
                f"window #{self.state.target_window_id}"
                if self.state.target_window_id is not None
                else "selected monitor"
            )
            self._show_error(f"Failed to capture {target_desc}.")
            return

        self._parse_and_apply(output_path)

        if was_overlay_visible:
            self.x11_overlay.show()

    def pick_target_window(self) -> None:
        if shutil.which("xwininfo") is None:
            self._show_error("`xwininfo` not found. Install x11-utils to use target window capture.")
            return

        self._set_status(
            "Click the emulator window to lock capture target...",
            level="info",
        )

        try:
            result = subprocess.run(
                ["xwininfo", "-int"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            self._show_error(f"Failed to run xwininfo: {exc}")
            return

        if result.returncode != 0:
            self._set_status("Window selection cancelled.", level="warning")
            return

        output = result.stdout
        match = re.search(r"Window id:\s+(\d+)\s+\"(.*)\"", output)
        if not match:
            self._show_error("Could not parse selected window from xwininfo output.")
            return

        self.state.target_window_id = int(match.group(1))
        self.state.target_window_name = match.group(2).strip() or "Unnamed window"

        self._align_monitor_to_target_window()
        self._update_window_hint()
        self._update_target_window_button()
        self._set_status(
            f"Selected target window #{self.state.target_window_id}: {self.state.target_window_name}",
            level="success",
        )

    def clear_target_window(self) -> None:
        self.state.target_window_id = None
        self.state.target_window_name = None
        self._update_window_hint()
        self._update_target_window_button()
        self._set_status("Cleared target window. Capture will use selected monitor.", level="info")

    def _capture_window(self, window_id: int):
        # Try each screen; one of them should return the target window content.
        best_pixmap = None
        best_area = 0
        for screen in QGuiApplication.screens():
            pixmap = screen.grabWindow(window_id)
            if pixmap.isNull():
                continue
            area = pixmap.width() * pixmap.height()
            if area > best_area:
                best_area = area
                best_pixmap = pixmap

        if best_pixmap is None:
            selected_screen = self._get_selected_screen()
            return selected_screen.grabWindow(0) if selected_screen else None
        return best_pixmap

    def _align_monitor_to_target_window(self) -> None:
        if self.state.target_window_id is None or shutil.which("xwininfo") is None:
            return

        try:
            result = subprocess.run(
                ["xwininfo", "-id", str(self.state.target_window_id), "-int"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return

        if result.returncode != 0:
            return

        x_match = re.search(r"Absolute upper-left X:\s+(-?\d+)", result.stdout)
        y_match = re.search(r"Absolute upper-left Y:\s+(-?\d+)", result.stdout)
        w_match = re.search(r"Width:\s+(\d+)", result.stdout)
        h_match = re.search(r"Height:\s+(\d+)", result.stdout)
        if not (x_match and y_match and w_match and h_match):
            return

        x = int(x_match.group(1))
        y = int(y_match.group(1))
        w = int(w_match.group(1))
        h = int(h_match.group(1))
        cx = x + w // 2
        cy = y + h // 2

        for idx, screen in enumerate(self._screens):
            geo = screen.geometry()
            if geo.contains(cx, cy):
                if idx != self.state.selected_screen_index:
                    self.monitor_combo.setCurrentIndex(idx)
                return

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
        self._set_status(f"Saved labeled screenshot: {target}", level="success")

    def clear_overlay(self) -> None:
        self.x11_overlay.clear_overlay()
        self.state.parse_result = None
        self.state.last_input_path = None
        self._set_status("Overlay cleared.", level="info")

    def toggle_overlay(self, checked: bool) -> None:
        self.overlay_btn.setText("Disable Overlay" if checked else "Enable Overlay")
        if checked:
            self._show_active_overlay()
        else:
            self._hide_all_overlays()

    def _show_active_overlay(self) -> None:
        screen = self._get_selected_screen()
        if screen is None:
            self._show_error("No monitor selected for overlay.")
            self.overlay_btn.blockSignals(True)
            self.overlay_btn.setChecked(False)
            self.overlay_btn.setText("Enable Overlay")
            self.overlay_btn.blockSignals(False)
            return

        self.x11_overlay.set_target_screen(screen)

        # X11-safe mode is the only supported overlay path in this build.
        self.x11_overlay.show()

    def _hide_all_overlays(self) -> None:
        self.x11_overlay.hide()

    def _monitor_label(self, index: int, screen) -> str:
        geo = screen.geometry()
        name = screen.name() or "Unknown"
        return f"Monitor {index + 1}: {name} ({geo.width()}x{geo.height()} @ {geo.x()},{geo.y()})"

    def _refresh_monitor_list(self) -> None:
        self._screens = QGuiApplication.screens()

        if not self._screens:
            self.monitor_combo.clear()
            self.state.selected_screen_index = 0
            return

        previous = self.state.selected_screen_index
        selected = min(previous, len(self._screens) - 1)

        self.monitor_combo.blockSignals(True)
        self.monitor_combo.clear()
        for idx, screen in enumerate(self._screens):
            self.monitor_combo.addItem(self._monitor_label(idx, screen), idx)
        self.monitor_combo.setCurrentIndex(selected)
        self.monitor_combo.blockSignals(False)

        self.state.selected_screen_index = selected
        self.x11_overlay.set_target_screen(self._screens[selected])
        self._update_monitor_hint()
        self._update_window_hint()

        if self.overlay_btn.isChecked():
            self._show_active_overlay()

    def _on_monitor_changed(self, combo_index: int) -> None:
        if combo_index < 0:
            return
        self.state.selected_screen_index = combo_index

        screen = self._get_selected_screen()
        if screen is not None:
            self.x11_overlay.set_target_screen(screen)
        self._update_monitor_hint()

        if self.overlay_btn.isChecked():
            self._show_active_overlay()

    def _get_selected_screen(self):
        if not self._screens:
            return None
        index = min(max(self.state.selected_screen_index, 0), len(self._screens) - 1)
        return self._screens[index]

    def _parse_and_apply(self, image_path: str) -> None:
        try:
            result = self.parser.parse(image_path)
        except Exception as exc:
            self._show_error(f"Failed to parse screenshot: {exc}")
            return

        self.state.last_input_path = image_path
        self.state.parse_result = result
        self.x11_overlay.set_overlay_data(result.regions, result.image_width, result.image_height)

        warning_text = ""
        if result.warnings:
            warning_text = f" Warnings: {' | '.join(result.warnings)}"
        monitor_text = f" Monitor: {self.state.selected_screen_index + 1}."
        level = "warning" if result.warnings else "success"
        self._set_status(
            f"Parsed {len(result.regions)} regions from {Path(image_path).name}.{monitor_text}{warning_text}",
            level=level,
        )

    def _show_error(self, message: str) -> None:
        self._set_status(message, level="error")
        QMessageBox.critical(self, "Voltorb Solver X11 Overlay", message)


def run_overlay_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = OverlayControlWindow()
    window.show()
    return app.exec()