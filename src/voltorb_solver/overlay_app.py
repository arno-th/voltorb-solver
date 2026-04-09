from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from tempfile import gettempdir

from PySide6.QtCore import Qt, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QSpinBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from voltorb_solver.game_state import GameState
from voltorb_solver.image_import.parser import ImageParser, TileDebugArtifacts
from voltorb_solver.image_import.screen_parser import Region, ScreenBoardParser
from voltorb_solver.solver import solve_game_state
from voltorb_solver.stats import StatsManager
from voltorb_solver.ui.stats_panel import StatsPanel

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None

try:
    from pynput import keyboard as _pynput_kb  # type: ignore
except Exception:
    _pynput_kb = None


def _expected_tile_value(probs: dict[int, float]) -> float:
    return probs[1] * 1.0 + probs[2] * 2.0 + probs[3] * 3.0


def _recommendation_bucket(bomb_probability: float, is_useful: bool) -> int:
    if bomb_probability <= 0.0:
        return 0
    if is_useful:
        return 1
    return 2


class _GlobalHotkeyListener:
    """Manages a pynput GlobalHotKeys listener that fires a callback from any focused window."""

    # Map Qt portable-text modifier/key names to pynput bracket names.
    _QT_TO_PYNPUT: dict[str, str] = {
        "ctrl": "<ctrl>", "shift": "<shift>", "alt": "<alt>", "meta": "<cmd>",
        "space": "<space>", "return": "<enter>", "enter": "<enter>",
        "tab": "<tab>", "escape": "<esc>", "backspace": "<backspace>",
        "delete": "<delete>", "insert": "<insert>",
        "home": "<home>", "end": "<end>",
        "pageup": "<page_up>", "pagedown": "<page_down>",
        "up": "<up>", "down": "<down>", "left": "<left>", "right": "<right>",
    }

    def __init__(self, callback) -> None:
        self._callback = callback
        self._listener = None
        self._combo: str = ""

    def set_key_sequence(self, seq: QKeySequence) -> None:
        combo = self._qt_to_pynput(seq.toString(QKeySequence.SequenceFormat.PortableText))
        if combo == self._combo:
            return
        self._combo = combo
        self._restart()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _restart(self) -> None:
        self.stop()
        if not self._combo or _pynput_kb is None:
            return
        try:
            self._listener = _pynput_kb.GlobalHotKeys({self._combo: self._callback})
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            pass

    @classmethod
    def _qt_to_pynput(cls, qt_str: str) -> str:
        """Convert a Qt PortableText key sequence like 'Ctrl+P' to '<ctrl>+<P>'."""
        if not qt_str:
            return ""
        parts = []
        for part in qt_str.split("+"):
            lower = part.lower()
            if lower in cls._QT_TO_PYNPUT:
                parts.append(cls._QT_TO_PYNPUT[lower])
            elif lower.startswith("f") and lower[1:].isdigit() and 1 <= int(lower[1:]) <= 24:
                parts.append(f"<{lower}>")
            elif len(part) == 1:
                parts.append(lower)
            else:
                return ""  # Unknown token — cannot map
        return "+".join(parts)


@dataclass(slots=True)
class OverlayState:
    last_input_path: str | None = None
    selected_screen_index: int = 0
    target_window_id: int | None = None
    target_window_name: str | None = None
    target_window_class: str | None = None


OVERLAY_COLORS = [
    QColor(15, 188, 249),
    QColor(255, 158, 0),
    QColor(32, 201, 151),
    QColor(255, 99, 132),
    QColor(153, 102, 255),
    QColor(255, 205, 86),
    QColor(0, 200, 83),
]

# Regions of the captured frame used for post-click text-box checks.
# (left_frac, top_frac, right_frac, bottom_frac) relative to the game area.
# Used as a fallback when no board has been parsed yet.
_TEXTBOX_REGION = (0.07, 0.974, 0.10, 0.993)
_TEXTBOX_TEMPLATE_PATH = Path("assets/templates/textbox_indicator.png")
_TEXTBOX_MATCH_THRESHOLD = 0.90

# Textbox indicator position as (x, y, w, h) fractions of the board dimensions.
# Origin (0, 0) = top-left of the board; (1, 1) = bottom-right of the board.
# Values outside [0, 1] are valid (e.g. y > 1 = below the board).
_TEXTBOX_BOARD_X = 0.08
_TEXTBOX_BOARD_Y = 1.024
_TEXTBOX_BOARD_W = 0.05
_TEXTBOX_BOARD_H = 0.048

_TEXTBOX_GAME_CLEAR_REGION = (0.07, 0.9, 0.2, 0.935)
_TEXTBOX_GAME_CLEAR_TEMPLATE_PATH = Path("assets/templates/textbox_game_clear.png")
_TEXTBOX_GAME_CLEAR_MATCH_THRESHOLD = 0.90

# Game-clear textbox board-relative defaults (x, y, w, h fractions of board).
_TEXTBOX_GAME_CLEAR_BOARD_X = 0.014
_TEXTBOX_GAME_CLEAR_BOARD_Y = 0.8
_TEXTBOX_GAME_CLEAR_BOARD_W = 0.386
_TEXTBOX_GAME_CLEAR_BOARD_H = 0.13

_TEXTBOX_PLAY_LEVEL_REGION = (0.07, 0.9, 0.2, 0.935)
_TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH = Path("assets/templates/textbox_play_level.png")
_TEXTBOX_PLAY_LEVEL_MATCH_THRESHOLD = 0.90

# Play-level textbox board-relative defaults (x, y, w, h fractions of board).
_TEXTBOX_PLAY_LEVEL_BOARD_X = 0.014
_TEXTBOX_PLAY_LEVEL_BOARD_Y = 0.8
_TEXTBOX_PLAY_LEVEL_BOARD_W = 0.386
_TEXTBOX_PLAY_LEVEL_BOARD_H = 0.13

_TEXTBOX_GAME_FAILED_REGION = (0.07, 0.9, 0.2, 0.935)
_TEXTBOX_GAME_FAILED_TEMPLATE_PATH = Path("assets/templates/textbox_game_failed.png")
_TEXTBOX_GAME_FAILED_MATCH_THRESHOLD = 0.90

# Game-failed textbox board-relative defaults (x, y, w, h fractions of board).
_TEXTBOX_GAME_FAILED_BOARD_X = 0.014
_TEXTBOX_GAME_FAILED_BOARD_Y = 0.8
_TEXTBOX_GAME_FAILED_BOARD_W = 0.386
_TEXTBOX_GAME_FAILED_BOARD_H = 0.13


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


def _prob_to_rgb(p: float) -> tuple[int, int, int]:
    """Interpolate green→yellow→red for voltorb probability 0→1."""
    p = max(0.0, min(1.0, p))
    if p <= 0.5:
        t = p * 2
        r = int(22 + 212 * t)
        g = int(163 + 16 * t)
        b = int(74 - 66 * t)
    else:
        t = (p - 0.5) * 2
        r = int(234 - 14 * t)
        g = int(179 - 141 * t)
        b = int(8 + 30 * t)
    return r, g, b


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


class ProbabilityLabelWindow(QWidget):
    """Frameless always-on-top window overlaying a single tile with its voltorb probability."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def update_content(self, text: str, r: int, g: int, b: int) -> None:
        self._label.setText(text)
        self._label.setStyleSheet(
            f"color: white; font-weight: 700; font-size: 15px;"
            f" background-color: rgba({r},{g},{b},210);"
            f" border-radius: 6px;"
        )

    def place_at(self, rect: QRect, screen=None) -> None:
        _bind_widget_to_screen(self, screen)
        # Show a small badge in the bottom-right corner rather than covering the whole tile.
        badge_w = max(42, rect.width() // 2)
        badge_h = max(22, rect.height() // 3)
        bx = rect.right() - badge_w + 1
        by = rect.bottom() - badge_h + 1
        self.setGeometry(bx, by, badge_w, badge_h)
        self._label.setGeometry(0, 0, badge_w, badge_h)


class SimpleProbabilityOverlay:
    """Renders voltorb probability percentages on top of each unrevealed tile."""

    def __init__(self) -> None:
        self._windows: list[ProbabilityLabelWindow] = []
        self._visible = False
        self._target_screen = QGuiApplication.primaryScreen()
        self._tile_data: list[tuple[tuple[int, int, int, int], float, bool]] = []
        self._image_size: tuple[int, int] | None = None
        self._mapping_rect: QRect | None = None

    def set_target_screen(self, screen) -> None:
        self._target_screen = screen
        if self._visible:
            self._render()

    def set_data(
        self,
        tile_data: list[tuple[tuple[int, int, int, int], float, bool]],
        image_w: int,
        image_h: int,
        mapping_rect: QRect | None,
    ) -> None:
        """Update tile data: list of (region_xywh, bomb_probability, is_recommended)."""
        self._tile_data = list(tile_data)
        self._image_size = (image_w, image_h)
        self._mapping_rect = QRect(mapping_rect) if mapping_rect is not None else None
        if self._visible:
            self._render()

    def clear(self) -> None:
        self._tile_data = []
        self._image_size = None
        self._mapping_rect = None
        self._hide_all()

    def show(self) -> None:
        self._visible = True
        self._render()

    def hide(self) -> None:
        self._visible = False
        self._hide_all()

    def _hide_all(self) -> None:
        for w in self._windows:
            w.hide()
            w.deleteLater()
        self._windows.clear()

    def _render(self) -> None:
        self._hide_all()
        if not self._tile_data or self._image_size is None:
            return
        screen = self._target_screen
        if screen is None:
            return
        image_w, image_h = self._image_size
        if self._mapping_rect is not None:
            mapping = QRect(self._mapping_rect)
        else:
            geo = screen.geometry()
            mapping = _map_image_to_overlay(geo.width(), geo.height(), image_w, image_h)
            mapping.translate(geo.x(), geo.y())

        for xywh, prob, is_recommended in self._tile_data:
            region = Region("_", *xywh)
            rect = _map_region_rect(region, mapping, image_w, image_h)
            rc, gc, bc = _prob_to_rgb(prob)
            star = "★" if is_recommended else ""
            win = ProbabilityLabelWindow()
            win.update_content(f"{star}{prob:.0%}", rc, gc, bc)
            win.place_at(rect, screen)
            win.show()
            self._windows.append(win)


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
        self._mapping_rect: QRect | None = None

    def set_target_screen(self, screen) -> None:
        self._target_screen = screen
        if self._visible:
            self._render()

    def set_overlay_data(self, regions: list[Region], image_w: int, image_h: int) -> None:
        self._regions = list(regions)
        self._image_size = (image_w, image_h)
        if self._visible:
            self._render()

    def set_mapping_rect(self, rect: QRect | None) -> None:
        self._mapping_rect = QRect(rect) if rect is not None else None
        if self._visible:
            self._render()

    def clear_overlay(self) -> None:
        self._regions = []
        self._image_size = None
        self._mapping_rect = None
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
        # Use full screen geometry so overlay coordinates match screen capture space.
        # availableGeometry() excludes bars/panels (e.g. i3 bar) and introduces offsets.
        if self._mapping_rect is not None:
            mapping = QRect(self._mapping_rect)
        else:
            geo = screen.geometry()
            mapping = _map_image_to_overlay(geo.width(), geo.height(), *self._image_size)
            # Translate from screen-local to absolute coordinates.
            mapping.translate(geo.x(), geo.y())
        geo = screen.geometry()

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
    # Emitted from background thread once the xdotool click subprocess finishes.
    _play_click_done = Signal()
    # Emitted from the pynput listener thread to marshal hotkey activation to main thread.
    _hotkey_activated = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voltorb Solver X11 Overlay")
        self.setMinimumSize(780, 360)

        self.parser = ScreenBoardParser()
        self.clue_parser = ImageParser()
        self.state = OverlayState()
        self.x11_overlay = X11SafeOverlay(OVERLAY_COLORS)
        self.simple_overlay = SimpleProbabilityOverlay()
        self._screens = QGuiApplication.screens()
        self._cached_regions: list[Region] = []
        self._last_capture_signature: tuple[str, int | None, int, int, int, int] | None = None
        self._clue_parsed_values: dict[str, tuple[int, int]] = {}
        self._last_image_size: tuple[int, int] | None = None
        self._last_parse_regions: list[Region] = []
        self._clue_dataset_root = Path("assets/parser_debug/clue_dataset")
        self._tile_dataset_root = Path("assets/parser_debug/tile_dataset")
        self._tile_parsed_values: dict[str, int | None] = {}
        self._game_state = GameState()
        self._play_level_running = False
        self._play_iteration = 0
        self._play_dialog_steps = 0
        self._play_click_delay_ms = 500
        self._play_dialog_delay_ms = 500
        self._play_poll_delay_ms = 500
        self._play_forever = False
        self._hotkey_sequence = QKeySequence("P")
        self._hotkey_shortcut: QShortcut | None = None
        self._global_hotkey = _GlobalHotkeyListener(self._hotkey_activated.emit)
        self._hotkey_activated.connect(self._on_hotkey_triggered)
        self._play_click_done.connect(self._play_after_click)
        self._anchor_board_rect: tuple[int, int, int, int] | None = None  # (left, top, right, bottom) image px
        self._anchor_image_size: tuple[int, int] | None = None

        self._textbox_offsets_path = Path("assets/templates/textbox_offsets.json")
        self._textbox_x = _TEXTBOX_BOARD_X
        self._textbox_y = _TEXTBOX_BOARD_Y
        self._textbox_w = _TEXTBOX_BOARD_W
        self._textbox_h = _TEXTBOX_BOARD_H
        self._game_clear_x = _TEXTBOX_GAME_CLEAR_BOARD_X
        self._game_clear_y = _TEXTBOX_GAME_CLEAR_BOARD_Y
        self._game_clear_w = _TEXTBOX_GAME_CLEAR_BOARD_W
        self._game_clear_h = _TEXTBOX_GAME_CLEAR_BOARD_H
        self._play_level_x = _TEXTBOX_PLAY_LEVEL_BOARD_X
        self._play_level_y = _TEXTBOX_PLAY_LEVEL_BOARD_Y
        self._play_level_w = _TEXTBOX_PLAY_LEVEL_BOARD_W
        self._play_level_h = _TEXTBOX_PLAY_LEVEL_BOARD_H
        self._game_failed_x = _TEXTBOX_GAME_FAILED_BOARD_X
        self._game_failed_y = _TEXTBOX_GAME_FAILED_BOARD_Y
        self._game_failed_w = _TEXTBOX_GAME_FAILED_BOARD_W
        self._game_failed_h = _TEXTBOX_GAME_FAILED_BOARD_H
        self._load_textbox_offsets()

        # last-line dedup: (normalized_key, block_number, repeat_count)
        self._log_last_key: str | None = None
        self._log_last_block: int = -1
        self._log_last_count: int = 0

        root = QWidget()
        root.setObjectName("RootPanel")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setObjectName("MainSplitter")
        root_layout.addWidget(splitter)

        bottom_pane = QWidget()
        layout = QVBoxLayout(bottom_pane)
        layout.setContentsMargins(0, 8, 0, 0)
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

        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(60)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        header_layout.addWidget(self.log_view)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(bottom_pane)

        splitter.addWidget(header_card)
        splitter.addWidget(scroll_area)
        splitter.setSizes([200, 400])

        # ── Runtime section ──────────────────────────────────────────────────
        runtime_card = QFrame()
        runtime_card.setObjectName("Card")
        runtime_layout = QVBoxLayout(runtime_card)
        runtime_layout.setContentsMargins(14, 12, 14, 12)
        runtime_layout.setSpacing(8)

        runtime_header = QLabel("Game Controls")
        runtime_header.setObjectName("FieldLabel")
        runtime_layout.addWidget(runtime_header)

        runtime_btn_row = QHBoxLayout()
        runtime_btn_row.setSpacing(8)
        self.start_play_btn = QPushButton("Start + Play")
        self.start_play_btn.setObjectName("PrimaryButton")
        self.start_play_btn.clicked.connect(self._start_and_play)
        self.refresh_tiles_btn = QPushButton("Refresh Tiles")
        self.refresh_tiles_btn.setObjectName("SecondaryButton")
        self.refresh_tiles_btn.clicked.connect(self._refresh_tiles)
        self.clear_game_btn = QPushButton("Clear Game")
        self.clear_game_btn.setObjectName("DangerButton")
        self.clear_game_btn.clicked.connect(self.clear_overlay)
        runtime_btn_row.addWidget(self.start_play_btn)
        runtime_btn_row.addWidget(self.refresh_tiles_btn)
        runtime_btn_row.addWidget(self.clear_game_btn)
        self.forever_loop_chk = QCheckBox("Forever loop")
        self.forever_loop_chk.setToolTip("Automatically start a new game when Play Level prompt is found")
        self.forever_loop_chk.setChecked(self._play_forever)
        self.forever_loop_chk.toggled.connect(lambda v: setattr(self, '_play_forever', v))
        runtime_btn_row.addSpacing(12)
        runtime_btn_row.addWidget(self.forever_loop_chk)
        runtime_btn_row.addStretch(1)
        runtime_layout.addLayout(runtime_btn_row)

        layout.addWidget(runtime_card)

        # ── Configuration section (collapsible) ──────────────────────────────
        config_card = QFrame()
        config_card.setObjectName("Card")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(14, 12, 14, 12)
        config_layout.setSpacing(6)

        config_header_row = QHBoxLayout()
        config_header_row.setSpacing(8)
        config_header_label = QLabel("Configuration")
        config_header_label.setObjectName("FieldLabel")
        config_header_row.addWidget(config_header_label)
        config_header_row.addStretch(1)
        self.config_toggle_btn = QPushButton("\u25b6 Show")
        self.config_toggle_btn.setObjectName("SecondaryButton")
        self.config_toggle_btn.clicked.connect(self._toggle_config_section)
        config_header_row.addWidget(self.config_toggle_btn)
        config_layout.addLayout(config_header_row)

        self.config_content = QWidget()
        config_content_layout = QVBoxLayout(self.config_content)
        config_content_layout.setContentsMargins(0, 4, 0, 0)
        config_content_layout.setSpacing(8)

        # Monitor selection
        target_row = QHBoxLayout()
        target_row.setSpacing(8)
        monitor_label = QLabel("Monitor")
        monitor_label.setObjectName("FieldLabel")
        target_row.addWidget(monitor_label)
        self.monitor_combo = QComboBox()
        self.monitor_combo.setObjectName("MonitorCombo")
        self.monitor_combo.currentIndexChanged.connect(self._on_monitor_changed)
        target_row.addWidget(self.monitor_combo)
        self.refresh_monitors_btn = QPushButton("Refresh")
        self.refresh_monitors_btn.setObjectName("SecondaryButton")
        self.refresh_monitors_btn.setToolTip("Refresh the monitor list")
        self.refresh_monitors_btn.clicked.connect(self._refresh_monitor_list)
        target_row.addWidget(self.refresh_monitors_btn)
        self.monitor_hint = QLabel()
        self.monitor_hint.setObjectName("HintLabel")
        target_row.addWidget(self.monitor_hint)
        target_row.addSpacing(16)
        self.target_window_btn = QPushButton("Pick Window")
        self.target_window_btn.setObjectName("SecondaryButton")
        self.target_window_btn.clicked.connect(self._handle_target_window_button)
        target_row.addWidget(self.target_window_btn)
        self.window_name_label = QLabel("No window selected.")
        self.window_name_label.setObjectName("HintLabel")
        self.window_name_label.setWordWrap(False)
        target_row.addWidget(self.window_name_label, 1)
        config_content_layout.addLayout(target_row)

        # Timing delays and hotkey
        self.click_delay_spin = QSpinBox()
        self.click_delay_spin.setRange(100, 5000)
        self.click_delay_spin.setSingleStep(100)
        self.click_delay_spin.setValue(self._play_click_delay_ms)
        self.click_delay_spin.setSuffix(" ms")
        self.click_delay_spin.setToolTip("Delay between clicking a tile and checking for a text box")
        self.click_delay_spin.valueChanged.connect(lambda v: setattr(self, '_play_click_delay_ms', v))

        self.dialog_delay_spin = QSpinBox()
        self.dialog_delay_spin.setRange(100, 5000)
        self.dialog_delay_spin.setSingleStep(100)
        self.dialog_delay_spin.setValue(self._play_dialog_delay_ms)
        self.dialog_delay_spin.setSuffix(" ms")
        self.dialog_delay_spin.setToolTip(
            "Delay between dialog-advance clicks and after detecting Game Clear"
        )
        self.dialog_delay_spin.valueChanged.connect(
            lambda v: setattr(self, '_play_dialog_delay_ms', v)
        )

        self.poll_delay_spin = QSpinBox()
        self.poll_delay_spin.setRange(100, 5000)
        self.poll_delay_spin.setSingleStep(100)
        self.poll_delay_spin.setValue(self._play_poll_delay_ms)
        self.poll_delay_spin.setSuffix(" ms")
        self.poll_delay_spin.setToolTip(
            "Interval between clicks while waiting for the Play Level prompt after Game Clear"
        )
        self.poll_delay_spin.valueChanged.connect(
            lambda v: setattr(self, '_play_poll_delay_ms', v)
        )

        self.hotkey_edit = QKeySequenceEdit(self._hotkey_sequence)
        self.hotkey_edit.setMaximumSequenceLength(1)
        self.hotkey_edit.setFixedWidth(120)
        self.hotkey_edit.setToolTip("Global hotkey to start or stop automated play")
        self.hotkey_edit.keySequenceChanged.connect(self._on_hotkey_changed)

        timing_grid = QGridLayout()
        timing_grid.setHorizontalSpacing(8)
        timing_grid.setVerticalSpacing(6)
        timing_grid.setColumnStretch(4, 1)
        timing_grid.addWidget(QLabel("Click delay:"), 0, 0)
        timing_grid.addWidget(self.click_delay_spin, 0, 1)
        timing_grid.addWidget(QLabel("Dialog delay:"), 0, 2)
        timing_grid.addWidget(self.dialog_delay_spin, 0, 3)
        timing_grid.addWidget(QLabel("Poll interval:"), 1, 0)
        timing_grid.addWidget(self.poll_delay_spin, 1, 1)
        timing_grid.addWidget(QLabel("Hotkey:"), 1, 2)
        timing_grid.addWidget(self.hotkey_edit, 1, 3)
        config_content_layout.addLayout(timing_grid)

        config_layout.addWidget(self.config_content)
        self.config_content.setVisible(False)
        layout.addWidget(config_card)

        # ── Debug section (collapsible) ───────────────────────────────────────
        debug_card = QFrame()
        debug_card.setObjectName("Card")
        debug_card_layout = QVBoxLayout(debug_card)
        debug_card_layout.setContentsMargins(14, 12, 14, 12)
        debug_card_layout.setSpacing(6)

        debug_header_row = QHBoxLayout()
        debug_header_row.setSpacing(8)
        debug_section_label = QLabel("Debug Tools")
        debug_section_label.setObjectName("FieldLabel")
        debug_header_row.addWidget(debug_section_label)
        debug_header_row.addStretch(1)
        self.debug_toggle_btn = QPushButton("▶ Show")
        self.debug_toggle_btn.setObjectName("SecondaryButton")
        self.debug_toggle_btn.clicked.connect(self._toggle_debug_section)
        debug_header_row.addWidget(self.debug_toggle_btn)
        debug_card_layout.addLayout(debug_header_row)

        self.debug_content = QWidget()
        debug_content_layout = QVBoxLayout(self.debug_content)
        debug_content_layout.setContentsMargins(0, 4, 0, 0)
        debug_content_layout.setSpacing(8)

        options_row = QHBoxLayout()
        options_row.setSpacing(8)
        self.debug_checkbox = QCheckBox("Debug")
        self.debug_checkbox.setObjectName("DebugCheck")
        options_row.addWidget(self.debug_checkbox)
        options_row.addStretch(1)
        debug_content_layout.addLayout(options_row)

        self.relabel_btn = QPushButton("Label/relabel game")
        self.relabel_btn.setObjectName("SecondaryButton")
        self.relabel_btn.clicked.connect(self.relabel_regions)
        self.parse_anchors_btn = QPushButton("Parse Anchors")
        self.parse_anchors_btn.setObjectName("SecondaryButton")
        self.parse_anchors_btn.clicked.connect(self._parse_anchors_debug)
        self.show_anchor_region_btn = QPushButton("Show Anchor Region")
        self.show_anchor_region_btn.setObjectName("SecondaryButton")
        self.show_anchor_region_btn.clicked.connect(self._show_anchor_region_overlay)
        self.capture_anchor_region_btn = QPushButton("Capture Anchor Region")
        self.capture_anchor_region_btn.setObjectName("SecondaryButton")
        self.capture_anchor_region_btn.setToolTip(
            "Capture a screenshot and save the cropped anchor board region to the debug folder"
        )
        self.capture_anchor_region_btn.clicked.connect(self._capture_anchor_region_image)
        self.parse_all_clues_btn = QPushButton("Parse clues")
        self.parse_all_clues_btn.setObjectName("PrimaryButton")
        self.parse_all_clues_btn.clicked.connect(self.parse_all_clues)
        self.parse_tiles_btn = QPushButton("Parse tiles")
        self.parse_tiles_btn.setObjectName("SecondaryButton")
        self.parse_tiles_btn.clicked.connect(self.parse_tiles)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setObjectName("DangerButton")
        self.clear_btn.clicked.connect(self.clear_overlay)

        debug_btn_row = QHBoxLayout()
        debug_btn_row.setSpacing(8)
        debug_btn_row.addWidget(self.relabel_btn)
        debug_btn_row.addWidget(self.parse_anchors_btn)
        debug_btn_row.addWidget(self.show_anchor_region_btn)
        debug_btn_row.addWidget(self.capture_anchor_region_btn)
        debug_btn_row.addWidget(self.parse_all_clues_btn)
        debug_btn_row.addWidget(self.parse_tiles_btn)
        debug_btn_row.addWidget(self.clear_btn)
        debug_content_layout.addLayout(debug_btn_row)

        self.overlay_btn = QPushButton("Enable Debug Overlay")
        self.overlay_btn.setObjectName("AccentToggle")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.toggled.connect(self.toggle_overlay)
        self.prob_overlay_btn = QPushButton("Enable Prob Overlay")
        self.prob_overlay_btn.setObjectName("ProbToggle")
        self.prob_overlay_btn.setCheckable(True)
        self.prob_overlay_btn.toggled.connect(self.toggle_prob_overlay)
        overlay_btns_row = QHBoxLayout()
        overlay_btns_row.setSpacing(8)
        overlay_btns_row.addWidget(self.overlay_btn)
        overlay_btns_row.addWidget(self.prob_overlay_btn)
        debug_content_layout.addLayout(overlay_btns_row)

        # ── Capture region crop ──────────────────────────────────────────────
        crop_label = QLabel("Capture Region Crop")
        crop_label.setObjectName("FieldLabel")
        debug_content_layout.addWidget(crop_label)

        crop_row = QHBoxLayout()
        crop_row.setSpacing(8)
        self.region_picker_combo = QComboBox()
        self.region_picker_combo.setObjectName("MonitorCombo")
        self.region_picker_combo.currentIndexChanged.connect(self._on_region_picker_changed)
        self.subsection_combo = QComboBox()
        self.subsection_combo.setObjectName("MonitorCombo")
        self.capture_region_btn = QPushButton("Capture")
        self.capture_region_btn.setObjectName("PrimaryButton")
        self.capture_region_btn.clicked.connect(self._capture_region_crop)
        crop_row.addWidget(self.region_picker_combo, 2)
        crop_row.addWidget(self.subsection_combo, 1)
        crop_row.addWidget(self.capture_region_btn)
        debug_content_layout.addLayout(crop_row)

        # ── Text-box detection ───────────────────────────────────────────────
        textbox_label = QLabel("Text-box Detection 1")
        textbox_label.setObjectName("FieldLabel")
        debug_content_layout.addWidget(textbox_label)

        textbox_row = QHBoxLayout()
        textbox_row.setSpacing(8)
        self.check_textbox_btn = QPushButton("Check Text Box")
        self.check_textbox_btn.setObjectName("PrimaryButton")
        self.check_textbox_btn.clicked.connect(self._check_textbox_template)
        self.save_textbox_tpl_btn = QPushButton("Save as Template")
        self.save_textbox_tpl_btn.setObjectName("SecondaryButton")
        self.save_textbox_tpl_btn.clicked.connect(self._capture_textbox_template)
        self.show_textbox_region_btn = QPushButton("Show Region")
        self.show_textbox_region_btn.setObjectName("SecondaryButton")
        self.show_textbox_region_btn.clicked.connect(self._show_textbox_region_overlay)
        textbox_row.addWidget(self.check_textbox_btn)
        textbox_row.addWidget(self.save_textbox_tpl_btn)
        textbox_row.addWidget(self.show_textbox_region_btn)
        textbox_row.addStretch(1)
        debug_content_layout.addLayout(textbox_row)

        # Board-relative offset controls for Textbox 1
        offsets_row = QHBoxLayout()
        offsets_row.setSpacing(6)
        for attr, label_text in (
            ("textbox_x_spin", "X:"),
            ("textbox_y_spin", "Y:"),
            ("textbox_w_spin", "W:"),
            ("textbox_h_spin", "H:"),
        ):
            lbl = QLabel(label_text)
            lbl.setObjectName("FieldLabel")
            offsets_row.addWidget(lbl)
            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 5.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setFixedWidth(72)
            spin.setToolTip("Board-relative fraction: origin (0,0) = board top-left, (1,1) = board bottom-right")
            setattr(self, attr, spin)
            offsets_row.addWidget(spin)
        offsets_row.addStretch(1)
        self.textbox_x_spin.setValue(self._textbox_x)
        self.textbox_y_spin.setValue(self._textbox_y)
        self.textbox_w_spin.setValue(self._textbox_w)
        self.textbox_h_spin.setValue(self._textbox_h)
        for spin in (self.textbox_x_spin, self.textbox_y_spin,
                     self.textbox_w_spin, self.textbox_h_spin):
            spin.valueChanged.connect(self._on_textbox_offsets_changed)
        debug_content_layout.addLayout(offsets_row)

        textbox_game_clear_label = QLabel("Game Clear Text-box")
        textbox_game_clear_label.setObjectName("FieldLabel")
        debug_content_layout.addWidget(textbox_game_clear_label)

        textbox_game_clear_row = QHBoxLayout()
        textbox_game_clear_row.setSpacing(8)
        self.check_textbox_game_clear_btn = QPushButton("Check Text Box")
        self.check_textbox_game_clear_btn.setObjectName("PrimaryButton")
        self.check_textbox_game_clear_btn.clicked.connect(self._check_textbox_game_clear_template)
        self.save_textbox_game_clear_tpl_btn = QPushButton("Save as Template")
        self.save_textbox_game_clear_tpl_btn.setObjectName("SecondaryButton")
        self.save_textbox_game_clear_tpl_btn.clicked.connect(self._capture_textbox_game_clear_template)
        self.show_textbox_game_clear_region_btn = QPushButton("Show Region")
        self.show_textbox_game_clear_region_btn.setObjectName("SecondaryButton")
        self.show_textbox_game_clear_region_btn.clicked.connect(self._show_textbox_game_clear_region_overlay)
        textbox_game_clear_row.addWidget(self.check_textbox_game_clear_btn)
        textbox_game_clear_row.addWidget(self.save_textbox_game_clear_tpl_btn)
        textbox_game_clear_row.addWidget(self.show_textbox_game_clear_region_btn)
        textbox_game_clear_row.addStretch(1)
        debug_content_layout.addLayout(textbox_game_clear_row)

        # Board-relative offset controls for Game Clear
        gc_offsets_row = QHBoxLayout()
        gc_offsets_row.setSpacing(6)
        for attr, label_text in (
            ("game_clear_x_spin", "X:"),
            ("game_clear_y_spin", "Y:"),
            ("game_clear_w_spin", "W:"),
            ("game_clear_h_spin", "H:"),
        ):
            lbl = QLabel(label_text)
            lbl.setObjectName("FieldLabel")
            gc_offsets_row.addWidget(lbl)
            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 5.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setFixedWidth(72)
            spin.setToolTip("Board-relative fraction: origin (0,0) = board top-left, (1,1) = board bottom-right")
            setattr(self, attr, spin)
            gc_offsets_row.addWidget(spin)
        gc_offsets_row.addStretch(1)
        self.game_clear_x_spin.setValue(self._game_clear_x)
        self.game_clear_y_spin.setValue(self._game_clear_y)
        self.game_clear_w_spin.setValue(self._game_clear_w)
        self.game_clear_h_spin.setValue(self._game_clear_h)
        for spin in (self.game_clear_x_spin, self.game_clear_y_spin,
                     self.game_clear_w_spin, self.game_clear_h_spin):
            spin.valueChanged.connect(self._on_game_clear_offsets_changed)
        debug_content_layout.addLayout(gc_offsets_row)

        textbox_play_level_label = QLabel("Play Level Text-box")
        textbox_play_level_label.setObjectName("FieldLabel")
        debug_content_layout.addWidget(textbox_play_level_label)

        textbox_play_level_row = QHBoxLayout()
        textbox_play_level_row.setSpacing(8)
        self.check_textbox_play_level_btn = QPushButton("Check Text Box")
        self.check_textbox_play_level_btn.setObjectName("PrimaryButton")
        self.check_textbox_play_level_btn.clicked.connect(self._check_textbox_play_level_template)
        self.save_textbox_play_level_tpl_btn = QPushButton("Save as Template")
        self.save_textbox_play_level_tpl_btn.setObjectName("SecondaryButton")
        self.save_textbox_play_level_tpl_btn.clicked.connect(self._capture_textbox_play_level_template)
        self.show_textbox_play_level_region_btn = QPushButton("Show Region")
        self.show_textbox_play_level_region_btn.setObjectName("SecondaryButton")
        self.show_textbox_play_level_region_btn.clicked.connect(self._show_textbox_play_level_region_overlay)
        textbox_play_level_row.addWidget(self.check_textbox_play_level_btn)
        textbox_play_level_row.addWidget(self.save_textbox_play_level_tpl_btn)
        textbox_play_level_row.addWidget(self.show_textbox_play_level_region_btn)
        textbox_play_level_row.addStretch(1)
        debug_content_layout.addLayout(textbox_play_level_row)

        pl_offsets_row = QHBoxLayout()
        pl_offsets_row.setSpacing(6)
        for attr, label_text in (
            ("play_level_x_spin", "X:"),
            ("play_level_y_spin", "Y:"),
            ("play_level_w_spin", "W:"),
            ("play_level_h_spin", "H:"),
        ):
            lbl = QLabel(label_text)
            lbl.setObjectName("FieldLabel")
            pl_offsets_row.addWidget(lbl)
            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 5.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setFixedWidth(72)
            spin.setToolTip("Board-relative fraction: origin (0,0) = board top-left, (1,1) = board bottom-right")
            setattr(self, attr, spin)
            pl_offsets_row.addWidget(spin)
        pl_offsets_row.addStretch(1)
        self.play_level_x_spin.setValue(self._play_level_x)
        self.play_level_y_spin.setValue(self._play_level_y)
        self.play_level_w_spin.setValue(self._play_level_w)
        self.play_level_h_spin.setValue(self._play_level_h)
        for spin in (self.play_level_x_spin, self.play_level_y_spin,
                     self.play_level_w_spin, self.play_level_h_spin):
            spin.valueChanged.connect(self._on_play_level_offsets_changed)
        debug_content_layout.addLayout(pl_offsets_row)

        textbox_game_failed_label = QLabel("Game Failed Text-box")
        textbox_game_failed_label.setObjectName("FieldLabel")
        debug_content_layout.addWidget(textbox_game_failed_label)

        textbox_game_failed_row = QHBoxLayout()
        textbox_game_failed_row.setSpacing(8)
        self.check_textbox_game_failed_btn = QPushButton("Check Text Box")
        self.check_textbox_game_failed_btn.setObjectName("PrimaryButton")
        self.check_textbox_game_failed_btn.clicked.connect(self._check_textbox_game_failed_template)
        self.save_textbox_game_failed_tpl_btn = QPushButton("Save as Template")
        self.save_textbox_game_failed_tpl_btn.setObjectName("SecondaryButton")
        self.save_textbox_game_failed_tpl_btn.clicked.connect(self._capture_textbox_game_failed_template)
        self.show_textbox_game_failed_region_btn = QPushButton("Show Region")
        self.show_textbox_game_failed_region_btn.setObjectName("SecondaryButton")
        self.show_textbox_game_failed_region_btn.clicked.connect(self._show_textbox_game_failed_region_overlay)
        textbox_game_failed_row.addWidget(self.check_textbox_game_failed_btn)
        textbox_game_failed_row.addWidget(self.save_textbox_game_failed_tpl_btn)
        textbox_game_failed_row.addWidget(self.show_textbox_game_failed_region_btn)
        textbox_game_failed_row.addStretch(1)
        debug_content_layout.addLayout(textbox_game_failed_row)

        gf_offsets_row = QHBoxLayout()
        gf_offsets_row.setSpacing(6)
        for attr, label_text in (
            ("game_failed_x_spin", "X:"),
            ("game_failed_y_spin", "Y:"),
            ("game_failed_w_spin", "W:"),
            ("game_failed_h_spin", "H:"),
        ):
            lbl = QLabel(label_text)
            lbl.setObjectName("FieldLabel")
            gf_offsets_row.addWidget(lbl)
            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 5.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setFixedWidth(72)
            spin.setToolTip("Board-relative fraction: origin (0,0) = board top-left, (1,1) = board bottom-right")
            setattr(self, attr, spin)
            gf_offsets_row.addWidget(spin)
        gf_offsets_row.addStretch(1)
        self.game_failed_x_spin.setValue(self._game_failed_x)
        self.game_failed_y_spin.setValue(self._game_failed_y)
        self.game_failed_w_spin.setValue(self._game_failed_w)
        self.game_failed_h_spin.setValue(self._game_failed_h)
        for spin in (self.game_failed_x_spin, self.game_failed_y_spin,
                     self.game_failed_w_spin, self.game_failed_h_spin):
            spin.valueChanged.connect(self._on_game_failed_offsets_changed)
        debug_content_layout.addLayout(gf_offsets_row)

        self.debug_content.setVisible(False)
        debug_card_layout.addWidget(self.debug_content)
        layout.addWidget(debug_card)

        # ── Statistics card ──────────────────────────────────────────────────────────────
        stats_card = QFrame()
        stats_card.setObjectName("Card")
        stats_card_layout = QVBoxLayout(stats_card)
        stats_card_layout.setContentsMargins(14, 12, 14, 12)
        stats_card_layout.setSpacing(8)
        stats_card_header = QLabel("Statistics")
        stats_card_header.setObjectName("FieldLabel")
        stats_card_layout.addWidget(stats_card_header)
        self.stats_panel = StatsPanel()
        stats_card_layout.addWidget(self.stats_panel)
        layout.addWidget(stats_card)

        layout.addStretch(1)

        self.stats = StatsManager()

        self._apply_styles()
        self._update_target_window_button()
        self._update_window_name_label()
        self._set_status("No screenshot parsed yet.", level="info")
        self._refresh_monitor_list()
        self._install_hotkey()
        self._restore_window_from_prefs()

    def _install_hotkey(self) -> None:
        """Install the play/pause hotkey globally (pynput) with QShortcut as fallback."""
        if self._hotkey_shortcut is not None:
            self._hotkey_shortcut.setEnabled(False)
            self._hotkey_shortcut.deleteLater()
            self._hotkey_shortcut = None

        # Global listener (works even when the emulator window has focus).
        if _pynput_kb is not None:
            self._global_hotkey.set_key_sequence(self._hotkey_sequence)
        else:
            # pynput unavailable — fall back to QShortcut (app-focus only).
            if not self._hotkey_sequence.isEmpty():
                self._hotkey_shortcut = QShortcut(self._hotkey_sequence, self)
                self._hotkey_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
                self._hotkey_shortcut.activated.connect(self._on_hotkey_triggered)

    def _on_hotkey_triggered(self) -> None:
        # Ignore triggers while recording a new shortcut.
        if hasattr(self, "hotkey_edit") and self.hotkey_edit.hasFocus():
            return
        self._start_and_play()

    def _on_hotkey_changed(self, seq: QKeySequence) -> None:
        self._hotkey_sequence = seq
        self._install_hotkey()

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

            QTextEdit#LogView {
                border-radius: 8px;
                border: 1px solid #d1deea;
                background: #f8fbfd;
                color: #1a2e40;
                padding: 6px 8px;
                font-size: 11px;
                font-family: monospace;
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

            QPushButton#ProbToggle {
                background: #7c3aed;
            }

            QPushButton#ProbToggle:hover {
                background: #6d28d9;
            }

            QPushButton#ProbToggle:checked {
                background: #5b21b6;
            }

            QSplitter#MainSplitter::handle {
                height: 8px;
                background: #d7e1ea;
                border-radius: 4px;
                margin: 2px 0;
            }

            QSplitter#MainSplitter::handle:hover {
                background: #97b0c6;
            }
            """
        )

    def _toggle_config_section(self) -> None:
        visible = self.config_content.isVisible()
        self.config_content.setVisible(not visible)
        self.config_toggle_btn.setText("\u25bc Hide" if not visible else "\u25b6 Show")

    def _toggle_debug_section(self) -> None:
        visible = self.debug_content.isVisible()
        self.debug_content.setVisible(not visible)
        self.debug_toggle_btn.setText("\u25bc Hide" if not visible else "\u25b6 Show")

    def _update_region_picker(self) -> None:
        """Repopulate the region picker combo from the current parsed regions."""
        self.region_picker_combo.blockSignals(True)
        prev_text = self.region_picker_combo.currentText()
        self.region_picker_combo.clear()
        for region in self._last_parse_regions:
            self.region_picker_combo.addItem(region.name)
        # Restore previous selection if still present.
        idx = self.region_picker_combo.findText(prev_text)
        self.region_picker_combo.setCurrentIndex(max(0, idx))
        self.region_picker_combo.blockSignals(False)
        self._on_region_picker_changed(self.region_picker_combo.currentIndex())

    def _on_region_picker_changed(self, _index: int) -> None:
        name = self.region_picker_combo.currentText()
        self.subsection_combo.blockSignals(True)
        self.subsection_combo.clear()
        if self._is_clue_region(name):
            self.subsection_combo.addItems(["full", "total", "voltorbs"])
        else:
            self.subsection_combo.addItem("full")
        self.subsection_combo.blockSignals(False)

    def _capture_region_crop(self) -> None:
        if not self._last_parse_regions:
            self._show_error("No regions parsed yet. Run 'Start Game' or 'Label/relabel game' first.")
            return
        if _cv2 is None:
            self._show_error("OpenCV (cv2) is required for region crop capture.")
            return

        region_name = self.region_picker_combo.currentText()
        subsection = self.subsection_combo.currentText()
        region = next((r for r in self._last_parse_regions if r.name == region_name), None)
        if region is None:
            self._show_error(f"Region '{region_name}' not found in parsed regions.")
            return

        # Take a fresh screenshot.
        if self.state.target_window_id is not None:
            pixmap = self._capture_window(self.state.target_window_id)
        else:
            screen = self._get_selected_screen()
            if screen is None:
                self._show_error("No monitor selected for capture.")
                return
            pixmap = screen.grabWindow(0)

        if pixmap is None or pixmap.isNull():
            self._show_error("Failed to capture screenshot for region crop.")
            return

        # Save to a temp file and reload via cv2.
        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_region_capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        if not pixmap.save(tmp_path):
            self._show_error("Failed to save temporary screenshot.")
            return

        image = _cv2.imread(tmp_path)
        if image is None:
            self._show_error("Failed to read temporary screenshot.")
            return

        # Determine crop box.
        if subsection == "total":
            x, y, w, h = self._subregion_from_bounds(region, self.clue_parser._TOTAL_OCR_BOUNDS)
        elif subsection == "voltorbs":
            x, y, w, h = self._subregion_from_bounds(region, self.clue_parser._VOLTORB_OCR_BOUNDS)
        else:
            x, y, w, h = region.x, region.y, region.w, region.h

        img_h, img_w = image.shape[:2]
        x0 = max(0, min(x, img_w - 1))
        y0 = max(0, min(y, img_h - 1))
        x1 = max(x0 + 1, min(x + w, img_w))
        y1 = max(y0 + 1, min(y + h, img_h))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            self._show_error("Computed crop is empty — region may be off-screen.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self._is_clue_region(region_name):
            out_dir = self._clue_dataset_root / "unknown"
            axis_tag = "row" if region_name.startswith("r") else "col"
            sub_tag = "t" if subsection == "total" else "v" if subsection == "voltorbs" else subsection
            filename = f"clue_{axis_tag}_{sub_tag}_{stamp}.png"
        elif self._is_tile_region(region_name):
            out_dir = self._tile_dataset_root / "unknown"
            safe_name = re.sub(r"[^\w\-]", "_", region_name)
            filename = f"{stamp}_{safe_name}_{subsection}.png"
        else:
            out_dir = Path(gettempdir())
            safe_name = re.sub(r"[^\w\-]", "_", region_name)
            filename = f"{stamp}_{safe_name}_{subsection}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        _cv2.imwrite(str(out_path), crop)

        self._set_status(
            f"Saved {region_name} ({subsection}) crop → {out_path.name}",
            level="success",
        )

    def _show_textbox_region_overlay_for(
        self,
        region: tuple[float, float, float, float],
        color: QColor,
        label: str,
        is_fallback: bool = False,
    ) -> None:
        screen = self._get_selected_screen()
        if screen is None:
            self._show_error("No monitor selected.")
            return

        # Use the same geometry logic as _show_anchor_region_overlay: query the
        # window directly when a target window is selected so that the coordinate
        # origin matches the anchor region display.
        if self.state.target_window_id is not None:
            geometry = self._query_window_geometry(self.state.target_window_id)
            if geometry is None:
                self._show_error("Could not query window geometry.")
                return
            gx, gy, gw, gh = geometry
        else:
            geo = screen.geometry()
            img_size = self._last_image_size or self._anchor_image_size
            if img_size is not None:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *img_size)
                mr.translate(geo.x(), geo.y())
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = geo.x(), geo.y(), geo.width(), geo.height()

        l_f, t_f, r_f, b_f = region
        rx = gx + int(gw * l_f)
        ry = gy + int(gh * t_f)
        rw = max(1, int(gw * (r_f - l_f)))
        rh = max(1, int(gh * (b_f - t_f)))
        region_rect = QRect(rx, ry, rw, rh)

        # Image-local coords (comparable to crop=(cx0,cy0)-(cx1,cy1) in capture status).
        img_x0 = int(gw * l_f)
        img_y0 = int(gh * t_f)
        img_x1 = img_x0 + rw
        img_y1 = img_y0 + rh

        segments: list[OverlayBorderWindow] = []
        for side in ("top", "bottom", "left", "right"):
            seg = OverlayBorderWindow(color, thickness=3)
            seg.show_for_rect(region_rect, side, screen=screen)
            segments.append(seg)

        fallback_note = " \u26a0 Fallback region \u2014 run \u2018Label/relabel\u2019 to enable board-relative offsets." if is_fallback else ""
        self._set_status(
            f"{label}: screen=({rx},{ry}) image-local=({img_x0},{img_y0})\u2013({img_x1},{img_y1}) {rw}\u00d7{rh}px{fallback_note}",
            level="warning" if is_fallback else "info",
        )

        def _hide() -> None:
            for s in segments:
                s.hide()
                s.deleteLater()

        QTimer.singleShot(4000, _hide)

    def _show_textbox_region_overlay(self) -> None:
        region, is_fallback = self._get_textbox_region()
        self._show_textbox_region_overlay_for(region, QColor(255, 80, 200), "Textbox 1", is_fallback)

    def _show_textbox_game_clear_region_overlay(self) -> None:
        region, is_fallback = self._get_game_clear_region()
        self._show_textbox_region_overlay_for(region, QColor(80, 200, 255), "Game Clear", is_fallback)

    def _show_textbox_play_level_region_overlay(self) -> None:
        region, is_fallback = self._get_play_level_region()
        self._show_textbox_region_overlay_for(region, QColor(200, 255, 80), "Play Level", is_fallback)

    def _show_textbox_game_failed_region_overlay(self) -> None:
        region, is_fallback = self._get_game_failed_region()
        self._show_textbox_region_overlay_for(region, QColor(255, 120, 50), "Game Failed", is_fallback)

    # ── Text-box detection ───────────────────────────────────────────────────

    def _grab_textbox_crop(self, region: tuple[float, float, float, float]) -> tuple["np.ndarray", str, tuple[int, int, int, int], tuple[int, int]] | None:  # type: ignore[name-defined]
        """Capture the game window and return the textbox crop (numpy array) + temp path."""
        if _cv2 is None:
            self._show_error("OpenCV (cv2) is required for text-box detection.")
            return None

        # Use the same capture path as relabeling: _capture_window works on
        # composited desktops, grabWindow(0) (root) does not.
        if self.state.target_window_id is not None:
            pixmap = self._capture_window(self.state.target_window_id)
            # Window-local coords: the pixmap origin is (0,0) = top-left of the window.
            use_window_local = True
        else:
            screen = self._get_selected_screen()
            if screen is None:
                self._show_error("No monitor selected.")
                return None
            pixmap = screen.grabWindow(0)
            use_window_local = False

        if pixmap is None or pixmap.isNull():
            self._show_error("Failed to capture screenshot for text-box detection.")
            return None

        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_textbox_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        if not pixmap.save(tmp_path):
            self._show_error("Failed to save temporary screenshot.")
            return None

        image = _cv2.imread(tmp_path)
        if image is None:
            self._show_error("Failed to read temporary screenshot.")
            return None

        img_h, img_w = image.shape[:2]

        if use_window_local:
            # The pixmap IS the window content; game area fills the whole image.
            gx, gy, gw, gh = 0, 0, img_w, img_h
        else:
            screen = self._get_selected_screen()
            screen_geo = screen.geometry() if screen else None
            sx = screen_geo.x() if screen_geo else 0
            sy = screen_geo.y() if screen_geo else 0
            mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
            if mapping_rect is not None:
                gx = mapping_rect.x() - sx
                gy = mapping_rect.y() - sy
                gw = mapping_rect.width()
                gh = mapping_rect.height()
            elif self._last_image_size is not None and screen_geo is not None:
                mr = _map_image_to_overlay(screen_geo.width(), screen_geo.height(), *self._last_image_size)
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = 0, 0, img_w, img_h

        l_f, t_f, r_f, b_f = region
        x0 = max(0, gx + int(gw * l_f))
        y0 = max(0, gy + int(gh * t_f))
        x1 = min(img_w, gx + int(gw * r_f))
        y1 = min(img_h, gy + int(gh * b_f))

        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            self._show_error(
                f"Textbox crop region is empty (game area: {gx},{gy} {gw}x{gh},"
                f" crop: {x0},{y0}-{x1},{y1}, image: {img_w}x{img_h})."
            )
            return None

        return crop, tmp_path, (x0, y0, x1, y1), (img_w, img_h)

    def _check_textbox_template_for(
        self,
        region: tuple[float, float, float, float],
        template_path: Path,
        threshold: float,
        label: str,
    ) -> None:
        if _cv2 is None:
            self._show_error("OpenCV (cv2) is required for text-box detection.")
            return

        result = self._grab_textbox_crop(region)
        if result is None:
            return
        crop, _, (cx0, cy0, cx1, cy1), (iw, ih) = result

        if not template_path.exists():
            self._set_status(
                f"No template saved for {label} yet. Use 'Save as Template' while visible.",
                level="warning",
            )
            return

        template = _cv2.imread(str(template_path))
        if template is None:
            self._set_status(f"Failed to load template for {label}.", level="error")
            return

        th, tw = template.shape[:2]
        ch, cw = crop.shape[:2]
        if tw > cw or th > ch:
            scale = min(cw / tw, ch / th)
            template = _cv2.resize(
                template,
                (max(1, int(tw * scale)), max(1, int(th * scale))),
                interpolation=_cv2.INTER_AREA,
            )

        gray_crop = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
        gray_tpl = _cv2.cvtColor(template, _cv2.COLOR_BGR2GRAY)
        result_map = _cv2.matchTemplate(gray_crop, gray_tpl, _cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = _cv2.minMaxLoc(result_map)

        if score >= threshold:
            self._set_status(
                f"{label} detected! Match score: {score:.1%} (threshold {threshold:.0%}).",
                level="success",
            )
        else:
            self._set_status(
                f"{label} NOT detected. Match score: {score:.1%} (threshold {threshold:.0%}).",
                level="warning",
            )

    def _capture_textbox_template_for(
        self,
        region: tuple[float, float, float, float],
        template_path: Path,
        label: str,
    ) -> None:
        result = self._grab_textbox_crop(region)
        if result is None:
            return
        crop, full_screenshot_path, (cx0, cy0, cx1, cy1), (iw, ih) = result

        # Save the full screenshot for debugging alongside the template.
        debug_screenshot_path = template_path.with_suffix(".debug_capture.png")
        import shutil as _shutil
        try:
            _shutil.copy2(full_screenshot_path, str(debug_screenshot_path))
        except Exception:
            pass

        is_black = int(crop.max()) == 0 if crop.size > 0 else True

        template_path.parent.mkdir(parents=True, exist_ok=True)
        if not _cv2.imwrite(str(template_path), crop):
            self._show_error(f"Failed to save template to {template_path}.")
            return

        h, w = crop.shape[:2]
        black_warning = " \u26a0 Crop is all-black — wrong region or textbox not visible?" if is_black else ""
        self._set_status(
            f"Saved {label} ({w}\u00d7{h}px) crop=({cx0},{cy0})-({cx1},{cy1}) image={iw}\u00d7{ih}"
            f" \u2192 {template_path}{black_warning}",
            level="warning" if is_black else "success",
        )

    def _load_textbox_offsets(self) -> None:
        """Load board-relative textbox offsets from JSON; silently ignore missing file."""
        import json
        if not self._textbox_offsets_path.exists():
            return
        try:
            data = json.loads(self._textbox_offsets_path.read_text())
            self._textbox_x = float(data.get("x", self._textbox_x))
            self._textbox_y = float(data.get("y", self._textbox_y))
            self._textbox_w = float(data.get("w", self._textbox_w))
            self._textbox_h = float(data.get("h", self._textbox_h))
            self._game_clear_x = float(data.get("gc_x", self._game_clear_x))
            self._game_clear_y = float(data.get("gc_y", self._game_clear_y))
            self._game_clear_w = float(data.get("gc_w", self._game_clear_w))
            self._game_clear_h = float(data.get("gc_h", self._game_clear_h))
            self._play_level_x = float(data.get("pl_x", self._play_level_x))
            self._play_level_y = float(data.get("pl_y", self._play_level_y))
            self._play_level_w = float(data.get("pl_w", self._play_level_w))
            self._play_level_h = float(data.get("pl_h", self._play_level_h))
            self._game_failed_x = float(data.get("gf_x", self._game_failed_x))
            self._game_failed_y = float(data.get("gf_y", self._game_failed_y))
            self._game_failed_w = float(data.get("gf_w", self._game_failed_w))
            self._game_failed_h = float(data.get("gf_h", self._game_failed_h))
        except Exception:
            pass

    def _save_textbox_offsets(self) -> None:
        """Persist current board-relative textbox offsets to JSON."""
        import json
        self._textbox_offsets_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "x": self._textbox_x,
            "y": self._textbox_y,
            "w": self._textbox_w,
            "h": self._textbox_h,
            "gc_x": self._game_clear_x,
            "gc_y": self._game_clear_y,
            "gc_w": self._game_clear_w,
            "gc_h": self._game_clear_h,
            "pl_x": self._play_level_x,
            "pl_y": self._play_level_y,
            "pl_w": self._play_level_w,
            "pl_h": self._play_level_h,
            "gf_x": self._game_failed_x,
            "gf_y": self._game_failed_y,
            "gf_w": self._game_failed_w,
            "gf_h": self._game_failed_h,
        }
        self._textbox_offsets_path.write_text(json.dumps(data, indent=2))

    def _on_textbox_offsets_changed(self) -> None:
        self._textbox_x = self.textbox_x_spin.value()
        self._textbox_y = self.textbox_y_spin.value()
        self._textbox_w = self.textbox_w_spin.value()
        self._textbox_h = self.textbox_h_spin.value()
        self._save_textbox_offsets()

    def _on_game_clear_offsets_changed(self) -> None:
        self._game_clear_x = self.game_clear_x_spin.value()
        self._game_clear_y = self.game_clear_y_spin.value()
        self._game_clear_w = self.game_clear_w_spin.value()
        self._game_clear_h = self.game_clear_h_spin.value()
        self._save_textbox_offsets()

    def _on_play_level_offsets_changed(self) -> None:
        self._play_level_x = self.play_level_x_spin.value()
        self._play_level_y = self.play_level_y_spin.value()
        self._play_level_w = self.play_level_w_spin.value()
        self._play_level_h = self.play_level_h_spin.value()
        self._save_textbox_offsets()

    def _on_game_failed_offsets_changed(self) -> None:
        self._game_failed_x = self.game_failed_x_spin.value()
        self._game_failed_y = self.game_failed_y_spin.value()
        self._game_failed_w = self.game_failed_w_spin.value()
        self._game_failed_h = self.game_failed_h_spin.value()
        self._save_textbox_offsets()

    def _get_textbox_region(self) -> tuple[tuple[float, float, float, float], bool]:
        """Return (region, is_fallback). Prefers anchor rect; falls back to tile-based or hardcoded."""
        anchor = self._board_region_from_anchor(
            self._textbox_x, self._textbox_y, self._textbox_w, self._textbox_h,
        )
        if anchor is not None:
            return anchor

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if tile_regions and self._last_image_size is not None:
            img_w, img_h = self._last_image_size
            if img_w > 0 and img_h > 0:
                board_left = min(r.x for r in tile_regions)
                board_top = min(r.y for r in tile_regions)
                board_right = max(r.x + r.w for r in tile_regions)
                board_bottom = max(r.y + r.h for r in tile_regions)
                board_w = board_right - board_left
                board_h = board_bottom - board_top

                x0 = board_left + int(self._textbox_x * board_w)
                y0 = board_top + int(self._textbox_y * board_h)
                x1 = x0 + int(self._textbox_w * board_w)
                y1 = y0 + int(self._textbox_h * board_h)

                x0 = max(0, min(x0, img_w - 1))
                x1 = max(x0 + 1, min(x1, img_w))
                y0 = max(0, min(y0, img_h - 1))
                y1 = max(y0 + 1, min(y1, img_h))

                return (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h), False

        return _TEXTBOX_REGION, True

    def _get_game_clear_region(self) -> tuple[tuple[float, float, float, float], bool]:
        """Return (region, is_fallback). Prefers anchor rect; falls back to tile-based or hardcoded."""
        anchor = self._board_region_from_anchor(
            self._game_clear_x, self._game_clear_y, self._game_clear_w, self._game_clear_h,
        )
        if anchor is not None:
            return anchor

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if tile_regions and self._last_image_size is not None:
            img_w, img_h = self._last_image_size
            if img_w > 0 and img_h > 0:
                board_left = min(r.x for r in tile_regions)
                board_top = min(r.y for r in tile_regions)
                board_right = max(r.x + r.w for r in tile_regions)
                board_bottom = max(r.y + r.h for r in tile_regions)
                board_w = board_right - board_left
                board_h = board_bottom - board_top

                x0 = board_left + int(self._game_clear_x * board_w)
                y0 = board_top + int(self._game_clear_y * board_h)
                x1 = x0 + int(self._game_clear_w * board_w)
                y1 = y0 + int(self._game_clear_h * board_h)

                x0 = max(0, min(x0, img_w - 1))
                x1 = max(x0 + 1, min(x1, img_w))
                y0 = max(0, min(y0, img_h - 1))
                y1 = max(y0 + 1, min(y1, img_h))

                return (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h), False

        return _TEXTBOX_GAME_CLEAR_REGION, True

    def _get_play_level_region(self) -> tuple[tuple[float, float, float, float], bool]:
        """Return (region, is_fallback). Prefers anchor rect; falls back to tile-based or hardcoded."""
        anchor = self._board_region_from_anchor(
            self._play_level_x, self._play_level_y, self._play_level_w, self._play_level_h,
        )
        if anchor is not None:
            return anchor

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if tile_regions and self._last_image_size is not None:
            img_w, img_h = self._last_image_size
            if img_w > 0 and img_h > 0:
                board_left = min(r.x for r in tile_regions)
                board_top = min(r.y for r in tile_regions)
                board_right = max(r.x + r.w for r in tile_regions)
                board_bottom = max(r.y + r.h for r in tile_regions)
                board_w = board_right - board_left
                board_h = board_bottom - board_top

                x0 = board_left + int(self._play_level_x * board_w)
                y0 = board_top + int(self._play_level_y * board_h)
                x1 = x0 + int(self._play_level_w * board_w)
                y1 = y0 + int(self._play_level_h * board_h)

                x0 = max(0, min(x0, img_w - 1))
                x1 = max(x0 + 1, min(x1, img_w))
                y0 = max(0, min(y0, img_h - 1))
                y1 = max(y0 + 1, min(y1, img_h))

                return (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h), False

        return _TEXTBOX_PLAY_LEVEL_REGION, True

    def _get_game_failed_region(self) -> tuple[tuple[float, float, float, float], bool]:
        """Return (region, is_fallback). Prefers anchor rect; falls back to tile-based or hardcoded."""
        anchor = self._board_region_from_anchor(
            self._game_failed_x, self._game_failed_y, self._game_failed_w, self._game_failed_h,
        )
        if anchor is not None:
            return anchor

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if tile_regions and self._last_image_size is not None:
            img_w, img_h = self._last_image_size
            if img_w > 0 and img_h > 0:
                board_left = min(r.x for r in tile_regions)
                board_top = min(r.y for r in tile_regions)
                board_right = max(r.x + r.w for r in tile_regions)
                board_bottom = max(r.y + r.h for r in tile_regions)
                board_w = board_right - board_left
                board_h = board_bottom - board_top

                x0 = board_left + int(self._game_failed_x * board_w)
                y0 = board_top + int(self._game_failed_y * board_h)
                x1 = x0 + int(self._game_failed_w * board_w)
                y1 = y0 + int(self._game_failed_h * board_h)

                x0 = max(0, min(x0, img_w - 1))
                x1 = max(x0 + 1, min(x1, img_w))
                y0 = max(0, min(y0, img_h - 1))
                y1 = max(y0 + 1, min(y1, img_h))

                return (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h), False

        return _TEXTBOX_GAME_FAILED_REGION, True

    def _check_textbox_template(self) -> None:
        region, _ = self._get_textbox_region()
        self._check_textbox_template_for(region, _TEXTBOX_TEMPLATE_PATH, _TEXTBOX_MATCH_THRESHOLD, "Textbox 1")

    def _capture_textbox_template(self) -> None:
        region, _ = self._get_textbox_region()
        self._capture_textbox_template_for(region, _TEXTBOX_TEMPLATE_PATH, "Textbox 1")

    def _check_textbox_game_clear_template(self) -> None:
        region, _ = self._get_game_clear_region()
        self._check_textbox_template_for(region, _TEXTBOX_GAME_CLEAR_TEMPLATE_PATH, _TEXTBOX_GAME_CLEAR_MATCH_THRESHOLD, "Game Clear")

    def _capture_textbox_game_clear_template(self) -> None:
        region, _ = self._get_game_clear_region()
        self._capture_textbox_template_for(region, _TEXTBOX_GAME_CLEAR_TEMPLATE_PATH, "Game Clear")

    def _check_textbox_play_level_template(self) -> None:
        region, _ = self._get_play_level_region()
        self._check_textbox_template_for(region, _TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH, _TEXTBOX_PLAY_LEVEL_MATCH_THRESHOLD, "Play Level")

    def _capture_textbox_play_level_template(self) -> None:
        region, _ = self._get_play_level_region()
        self._capture_textbox_template_for(region, _TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH, "Play Level")

    def _check_textbox_game_failed_template(self) -> None:
        region, _ = self._get_game_failed_region()
        self._check_textbox_template_for(region, _TEXTBOX_GAME_FAILED_TEMPLATE_PATH, _TEXTBOX_GAME_FAILED_MATCH_THRESHOLD, "Game Failed")

    def _capture_textbox_game_failed_template(self) -> None:
        region, _ = self._get_game_failed_region()
        self._capture_textbox_template_for(region, _TEXTBOX_GAME_FAILED_TEMPLATE_PATH, "Game Failed")

    def _start_and_play(self) -> None:
        """Parse the board then immediately begin automated play. Click again to stop."""
        if self._play_level_running:
            self._play_stop("Play Level stopped by user.")
            return

        if self.state.target_window_id is None:
            self._show_error("No target window selected. Use 'Pick Target Window' first.")
            return
        if shutil.which("xdotool") is None:
            self._show_error("`xdotool` not found. Install xdotool to use Start + Play.")
            return

        if not _TEXTBOX_TEMPLATE_PATH.exists():
            self._set_status("Warning: textbox template not found — dialog detection disabled.", "warning")
        if not _TEXTBOX_GAME_CLEAR_TEMPLATE_PATH.exists():
            self._set_status("Warning: game-clear template not found — completion detection disabled.", "warning")

        # Resume from current state if a board is already parsed (i.e. was paused mid-level).
        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if self.state.last_input_path is not None and tile_regions:
            self._play_level_running = True
            self._play_dialog_steps = 0
            self.start_play_btn.setText("Stop Play")
            self._set_status("Resuming play from current board state…", "info")
            QTimer.singleShot(0, self._play_step)
            return

        # Capture one screenshot up-front to: detect board corners via anchors and
        # check whether the Play Level dialog is already showing before the board appears.
        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_overlay_capture_monitor_{self.state.selected_screen_index + 1}.png"
        )
        capture_signature = self._build_capture_signature()
        if not self._capture_game_screenshot(tmp_path):
            return

        self._detect_and_cache_anchor_board_rect(tmp_path)
        # Pre-populate signature so play-level click can resolve screen coords before relabeling.
        self._last_capture_signature = capture_signature

        self._play_level_running = True
        self._play_iteration = 0
        self._play_dialog_steps = 0
        self.start_play_btn.setText("Stop Play")

        # Check if the Play Level dialog is showing before the board is visible.
        # Search directly in the already-captured tmp_path rather than re-capturing, so
        # there is no dependency on anchor coordinates or region fractions being correct.
        _play_level_visible = False
        if _TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH.exists() and _cv2 is not None:
            _img = _cv2.imread(tmp_path)
            _tpl = _cv2.imread(str(_TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH))
            if _img is not None and _tpl is not None:
                _ih, _iw = _img.shape[:2]
                _th, _tw = _tpl.shape[:2]
                if _tw <= _iw and _th <= _ih:
                    _gray_img = _cv2.cvtColor(_img, _cv2.COLOR_BGR2GRAY)
                    _gray_tpl = _cv2.cvtColor(_tpl, _cv2.COLOR_BGR2GRAY)
                    _res = _cv2.matchTemplate(_gray_img, _gray_tpl, _cv2.TM_CCOEFF_NORMED)
                    _, _score, _, _ = _cv2.minMaxLoc(_res)
                    _play_level_visible = _score >= _TEXTBOX_PLAY_LEVEL_MATCH_THRESHOLD
        if _play_level_visible:
            self._set_status("Play Level dialog detected — clicking to start new game…", "info")
            QTimer.singleShot(0, self._play_start_from_play_level)
            return

        # Normal flow: board is visible — parse it and start playing.
        if not self.prob_overlay_btn.isChecked():
            self.prob_overlay_btn.setChecked(True)
        if not self._parse_and_apply(
            tmp_path, capture_signature=capture_signature, relabel_reason="label/relabel game"
        ):
            self._play_stop("Failed to parse screenshot.", "error")
            return
        self.parse_all_clues()
        self.parse_tiles()
        if not [r for r in self._last_parse_regions if self._is_tile_region(r.name)]:
            self._play_stop("No tile regions found after parsing.", "error")
            return

        self._set_status("Play Level started…", "info")
        QTimer.singleShot(0, self._play_step)

    def _click_best_tile(self) -> None:
        if self._last_capture_signature is None or self._last_image_size is None:
            self._show_error("No game parsed yet. Use 'Start Game' first.")
            return

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if not tile_regions:
            self._show_error("No tile regions found. Run 'Start Game' first.")
            return

        snapshot = solve_game_state(self._game_state)
        if snapshot.total_configurations == 0:
            self._show_error("No valid configurations — cannot determine best tile.")
            return

        # Priority: guaranteed-safe tiles first (including safe 1s), then risky useful tiles.
        best_region: Region | None = None
        best_prob = float("inf")
        best_rank: tuple[int, float, float, int, int] | None = None
        for region in tile_regions:
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            if not m:
                continue
            r_idx, c_idx = int(m.group(1)), int(m.group(2))
            if self._game_state.board[r_idx][c_idx].revealed:
                continue
            pos = (r_idx, c_idx)
            prob = snapshot.bomb_probabilities.get(pos, 1.0)
            probs = snapshot.value_probabilities.get(pos, {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})
            expected_value = _expected_tile_value(probs)
            rank = (
                _recommendation_bucket(prob, pos in snapshot.useful_positions),
                prob,
                -expected_value,
                r_idx,
                c_idx,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_prob = prob
                best_region = region

        if best_region is None:
            self._show_error("No recommended tile found (all unrevealed tiles may already be resolved).")
            return

        # Map tile region to absolute screen coordinates.
        image_w, image_h = self._last_image_size
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is None:
            screen = self._get_selected_screen()
            if screen is None:
                self._show_error("No monitor selected.")
                return
            geo = screen.geometry()
            mapping_rect = _map_image_to_overlay(geo.width(), geo.height(), image_w, image_h)
            mapping_rect.translate(geo.x(), geo.y())

        rect = _map_region_rect(best_region, mapping_rect, image_w, image_h)
        cx = rect.center().x()
        cy = rect.center().y()

        if shutil.which("xdotool") is None:
            self._show_error("`xdotool` not found. Install xdotool to use Click Tile.")
            return

        tile_name = best_region.name
        window_id = self.state.target_window_id

        def _do_click() -> None:
            # Small delay to let Qt finish processing the button-release event.
            time.sleep(0.3)
            try:
                # Focus the emulator window: try wmctrl first (hex window id),
                # fall back to xdotool windowfocus.
                if window_id is not None:
                    if shutil.which("wmctrl"):
                        subprocess.run(
                            ["wmctrl", "-ia", hex(window_id)],
                            check=False, capture_output=True,
                        )
                    else:
                        subprocess.run(
                            ["xdotool", "windowfocus", "--sync", str(window_id)],
                            check=False, capture_output=True,
                        )
                    time.sleep(0.15)

                # Move cursor to tile centre.
                subprocess.run(
                    ["xdotool", "mousemove", "--sync", str(cx), str(cy)],
                    check=False, capture_output=True,
                )
                time.sleep(0.1)

                # Press and release separately — some apps ignore the compound click.
                subprocess.run(
                    ["xdotool", "mousedown", "--clearmodifiers", "1"],
                    check=False, capture_output=True,
                )
                time.sleep(0.08)
                subprocess.run(
                    ["xdotool", "mouseup", "--clearmodifiers", "1"],
                    check=False, capture_output=True,
                )
            except Exception as exc:
                self._set_status(f"Failed to click tile: {exc}", level="error")
                return

        # Show the equivalent manual command so you can test it in a terminal.
        manual_cmd = f"xdotool mousemove --sync {cx} {cy} mousedown --clearmodifiers 1 mouseup --clearmodifiers 1"
        self._set_status(
            f"Clicking {tile_name} at ({cx}, {cy}) — bomb prob {best_prob:.1%}.  "
            f"Manual: {manual_cmd}",
            level="success",
        )
        threading.Thread(target=_do_click, daemon=True).start()

    # ── Play Level (main-thread state machine) ─────────────────────────────
    #
    # All Qt / screen-capture operations stay on the main thread.
    # The only background thread is the xdotool subprocess; when it finishes
    # it emits _play_click_done which is connected to _play_after_click on the
    # main thread.  This avoids the QTimer-from-background-thread pitfall where
    # singleShot callbacks are posted to a non-existent event loop.



    def _play_stop(self, reason: str = "Play Level complete.", level: str = "info") -> None:
        """Stop the state machine and reset the button."""
        self._play_level_running = False
        self.start_play_btn.setText("Start + Play")
        # Restore overlays in case they were hidden during a dialog sequence.
        if self.overlay_btn.isChecked():
            self.x11_overlay.show()
        if self.prob_overlay_btn.isChecked():
            self.simple_overlay.show()
        self._set_status(reason, level)

    def _play_step(self) -> None:
        """Main-thread: find best tile and dispatch xdotool click to background thread."""
        if not self._play_level_running:
            return

        self._play_iteration += 1
        step = self._play_iteration
        self._set_status(f"── Step {step}: finding best tile…")

        click_info = self._play_find_best_tile()
        if click_info is None:
            snapshot = solve_game_state(self._game_state)
            if snapshot.total_configurations == 0:
                self._play_stop("No valid board configurations — solver has no solutions. Check clues.", "error")
            else:
                n_useful = len(snapshot.useful_positions)
                n_unrevealed = sum(
                    1 for r in range(5) for c in range(5)
                    if not self._game_state.board[r][c].revealed
                )
                self._play_stop(
                    f"No clickable tile — {n_unrevealed} unrevealed, {n_useful} useful, "
                    f"{snapshot.total_configurations} configs. "
                    "Level may be complete or all remaining tiles are bombs."
                )
            return

        tile_name, cx, cy = click_info
        m = re.match(r"^\((\d),(\d)\)$", tile_name)
        if m:
            snapshot = solve_game_state(self._game_state)
            pos = (int(m.group(1)), int(m.group(2)))
            bomb_prob = snapshot.bomb_probabilities.get(pos, 1.0)
            self._set_status(
                f"Step {step}: clicking {tile_name} at ({cx},{cy}) "
                f"[bomb prob {bomb_prob:.1%}, {snapshot.total_configurations} configs]"
            )
        else:
            self._set_status(f"Step {step}: clicking {tile_name} at ({cx},{cy})")

        window_id = self.state.target_window_id

        def _click_bg() -> None:
            self._play_do_xdotool_click(cx, cy, window_id)
            self._play_click_done.emit()  # cross-thread signal → main thread

        threading.Thread(target=_click_bg, daemon=True).start()

    def _play_after_click(self) -> None:
        """Main-thread slot: called via signal once the xdotool click finishes."""
        if not self._play_level_running:
            return
        step = self._play_iteration
        delay_ms = self._play_click_delay_ms
        self._set_status(f"Step {step}: click sent — waiting {delay_ms} ms for game response…")
        self._play_dialog_steps = 0
        QTimer.singleShot(delay_ms, self._play_check_dialog_step)

    def _play_check_dialog_step(self) -> None:
        """Main-thread: check for dialog textbox; advance or move to board refresh."""
        if not self._play_level_running:
            return

        MAX_DIALOG_STEPS = 30
        self._play_dialog_steps += 1
        step = self._play_iteration
        ds = self._play_dialog_steps

        self._set_status(f"  Step {step} dialog {ds}: checking for textbox…")
        # Capture the screen once and run all template checks in a single pass.
        has_textbox, is_clear, is_failed = self._play_check_templates_now([
            (self._get_textbox_region()[0], _TEXTBOX_TEMPLATE_PATH, _TEXTBOX_MATCH_THRESHOLD),
            (self._get_game_clear_region()[0], _TEXTBOX_GAME_CLEAR_TEMPLATE_PATH, _TEXTBOX_GAME_CLEAR_MATCH_THRESHOLD),
            (self._get_game_failed_region()[0], _TEXTBOX_GAME_FAILED_TEMPLATE_PATH, _TEXTBOX_GAME_FAILED_MATCH_THRESHOLD),
        ])

        if not has_textbox:
            self._set_status(f"  Step {step} dialog {ds}: no textbox — re-evaluating board…")
            # Re-enable overlays now that the textbox is gone.
            if self.overlay_btn.isChecked():
                self.x11_overlay.show()
            if self.prob_overlay_btn.isChecked():
                self.simple_overlay.show()
            self._refresh_tiles()
            if self._play_level_running:
                QTimer.singleShot(1000, self._play_step)
            return

        self._set_status(f"  Step {step} dialog {ds}: textbox present — checking for Game Clear / Game Failed…")
        # Hide overlays while a textbox is on screen so the click lands cleanly.
        if ds == 1:
            self.x11_overlay.hide()
            self.simple_overlay.hide()
        if is_clear:
            self._set_status("  Game Clear detected — waiting for Play Level prompt…", "success")
            self.stats.record_win()
            self.stats_panel.refresh(self.stats.lifetime, self.stats.session)
            self._play_dialog_steps = 0
            QTimer.singleShot(self._play_dialog_delay_ms, self._play_wait_for_play_level)
            return
        if is_failed:
            self._set_status("  Game Failed (voltorb!) detected — advancing to Play Level prompt…", "warning")
            self.stats.record_bomb()
            self.stats_panel.refresh(self.stats.lifetime, self.stats.session)
            self._play_dialog_steps = 0
            QTimer.singleShot(self._play_dialog_delay_ms, self._play_wait_for_play_level_after_fail)
            return

        if ds >= MAX_DIALOG_STEPS:
            self._play_stop(f"Dialog loop exceeded {MAX_DIALOG_STEPS} steps — stopping.", "error")
            return

        self._set_status(f"  Step {step} dialog {ds}: not game clear — advancing dialog…")
        self._play_click_textbox_center()
        QTimer.singleShot(self._play_dialog_delay_ms, self._play_check_dialog_step)

    def _play_wait_for_play_level_after_fail(self) -> None:
        """After game failed: keep clicking game-failed region centre until Play Level prompt appears."""
        if not self._play_level_running:
            return

        MAX_WAIT_STEPS = 30
        self._play_dialog_steps += 1
        ds = self._play_dialog_steps

        has_play_level = self._play_check_template_now(
            self._get_play_level_region()[0],
            _TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH,
            _TEXTBOX_PLAY_LEVEL_MATCH_THRESHOLD,
        )

        if has_play_level:
            if self._play_forever:
                self._set_status("  Play Level prompt detected after game failed — starting new game…", "warning")
                self._play_iteration = 0
                self._play_dialog_steps = 0
                QTimer.singleShot(0, self._play_start_from_play_level)
            else:
                self._set_status("  Play Level prompt detected after game failed — stopping.", "warning")
                self._play_level_running = False
                self.start_play_btn.setText("Start + Play")
                self.clear_overlay()
            return

        if ds >= MAX_WAIT_STEPS:
            self._play_stop(f"Timed out waiting for Play Level prompt after fail ({MAX_WAIT_STEPS} steps).", "error")
            return

        self._set_status(f"  Waiting for Play Level after fail (step {ds}) — clicking game-failed region…")
        gf_region, _ = self._get_game_failed_region()
        l_f, t_f, r_f, b_f = gf_region
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is not None:
            gx, gy, gw, gh = (
                mapping_rect.x(), mapping_rect.y(),
                mapping_rect.width(), mapping_rect.height(),
            )
        else:
            screen = self._get_selected_screen()
            if screen is None:
                self._play_stop("No monitor selected.", "error")
                return
            geo = screen.geometry()
            if self._last_image_size:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *self._last_image_size)
                mr.translate(geo.x(), geo.y())
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = geo.x(), geo.y(), geo.width(), geo.height()

        cx = gx + int(gw * (l_f + r_f) / 2)
        cy = gy + int(gh * (t_f + b_f) / 2)
        self._play_do_xdotool_click(cx, cy, self.state.target_window_id)
        QTimer.singleShot(self._play_poll_delay_ms, self._play_wait_for_play_level_after_fail)

    def _play_wait_for_play_level(self) -> None:
        """After game clear: keep clicking game-clear region centre until Play Level prompt appears."""
        if not self._play_level_running:
            return

        MAX_WAIT_STEPS = 30
        self._play_dialog_steps += 1
        ds = self._play_dialog_steps

        has_play_level = self._play_check_template_now(
            self._get_play_level_region()[0],
            _TEXTBOX_PLAY_LEVEL_TEMPLATE_PATH,
            _TEXTBOX_PLAY_LEVEL_MATCH_THRESHOLD,
        )

        if has_play_level:
            if self._play_forever:
                self._set_status("  Play Level prompt detected — starting new game…", "success")
                self._play_iteration = 0
                self._play_dialog_steps = 0
                QTimer.singleShot(0, self._play_start_from_play_level)
            else:
                self._set_status("  Play Level prompt detected — game clear complete.", "success")
                self._play_level_running = False
                self.start_play_btn.setText("Start + Play")
                self.clear_overlay()
            return

        if ds >= MAX_WAIT_STEPS:
            self._play_stop(f"Timed out waiting for Play Level prompt after {MAX_WAIT_STEPS} steps.", "error")
            return

        self._set_status(f"  Waiting for Play Level prompt (step {ds}) — clicking game clear region…")
        gc_region, _ = self._get_game_clear_region()
        l_f, t_f, r_f, b_f = gc_region
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is not None:
            gx, gy, gw, gh = (
                mapping_rect.x(), mapping_rect.y(),
                mapping_rect.width(), mapping_rect.height(),
            )
        else:
            screen = self._get_selected_screen()
            if screen is None:
                self._play_stop("No monitor selected.", "error")
                return
            geo = screen.geometry()
            if self._last_image_size:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *self._last_image_size)
                mr.translate(geo.x(), geo.y())
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = geo.x(), geo.y(), geo.width(), geo.height()

        cx = gx + int(gw * (l_f + r_f) / 2)
        cy = gy + int(gh * (t_f + b_f) / 2)
        self._play_do_xdotool_click(cx, cy, self.state.target_window_id)
        QTimer.singleShot(self._play_poll_delay_ms, self._play_wait_for_play_level)

    # ── Start from Play Level dialog ─────────────────────────────────────────
    #
    # Called when Start + Play detects the Play Level prompt before the board
    # is visible.  Clicks the prompt to launch the next game, then waits for
    # the board to load before entering the tile-click loop.

    def _play_click_play_level_region(self) -> None:
        """Click the centre of the Play Level region (works before relabeling)."""
        region, _ = self._get_play_level_region()
        l_f, t_f, r_f, b_f = region
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is not None:
            gx, gy, gw, gh = (
                mapping_rect.x(), mapping_rect.y(),
                mapping_rect.width(), mapping_rect.height(),
            )
        else:
            screen = self._get_selected_screen()
            if screen is None:
                return
            geo = screen.geometry()
            img_size = self._last_image_size or self._anchor_image_size
            if img_size is not None:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *img_size)
                mr.translate(geo.x(), geo.y())
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = geo.x(), geo.y(), geo.width(), geo.height()
        cx = gx + int(gw * (l_f + r_f) / 2)
        cy = gy + int(gh * (t_f + b_f) / 2)
        self._play_do_xdotool_click(cx, cy, self.state.target_window_id)

    def _play_start_from_play_level(self) -> None:
        """Click 'Play Level' to start the next game, then wait and relabel."""
        if not self._play_level_running:
            return
        self._set_status("Clicking Play Level to start new game…", "info")
        self._play_click_play_level_region()
        QTimer.singleShot(1500, self._play_relabel_and_start_loop)

    def _play_relabel_and_start_loop(self) -> None:
        """After clicking Play Level: relabel the board and begin the play loop."""
        if not self._play_level_running:
            return
        self.relabel_regions()
        self.parse_all_clues()
        self.parse_tiles()
        if not self.prob_overlay_btn.isChecked():
            self.prob_overlay_btn.setChecked(True)
        if not [r for r in self._last_parse_regions if self._is_tile_region(r.name)]:
            self._play_stop(
                "No tile regions found after relabeling — board may not be ready yet. Try again.",
                "error",
            )
            return
        self._set_status("Board loaded — starting play loop…", "info")
        QTimer.singleShot(0, self._play_step)

    def _play_check_template_now(
        self,
        region: tuple[float, float, float, float],
        template_path: Path,
        threshold: float,
    ) -> bool:
        """Template match called directly on the main thread — no threading needed."""
        return self._play_check_templates_now([(region, template_path, threshold)])[0]

    def _play_check_templates_now(
        self,
        checks: list[tuple[tuple[float, float, float, float], Path, float]],
    ) -> list[bool]:
        """Capture the screen once and match multiple templates in a single pass."""
        n = len(checks)
        if _cv2 is None:
            return [False] * n

        if self.state.target_window_id is not None:
            pixmap = self._capture_window(self.state.target_window_id)
            use_window_local = True
        else:
            screen = self._get_selected_screen()
            if screen is None:
                return [False] * n
            pixmap = screen.grabWindow(0)
            use_window_local = False

        if pixmap is None or pixmap.isNull():
            return [False] * n

        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_textbox_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        if not pixmap.save(tmp_path):
            return [False] * n

        image = _cv2.imread(tmp_path)
        if image is None:
            return [False] * n

        img_h, img_w = image.shape[:2]

        if use_window_local:
            gx, gy, gw, gh = 0, 0, img_w, img_h
        else:
            screen = self._get_selected_screen()
            screen_geo = screen.geometry() if screen else None
            sx = screen_geo.x() if screen_geo else 0
            sy = screen_geo.y() if screen_geo else 0
            mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
            if mapping_rect is not None:
                gx = mapping_rect.x() - sx
                gy = mapping_rect.y() - sy
                gw = mapping_rect.width()
                gh = mapping_rect.height()
            elif self._last_image_size is not None and screen_geo is not None:
                mr = _map_image_to_overlay(screen_geo.width(), screen_geo.height(), *self._last_image_size)
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = 0, 0, img_w, img_h

        results: list[bool] = []
        for region, template_path, threshold in checks:
            if not template_path.exists():
                results.append(False)
                continue

            l_f, t_f, r_f, b_f = region
            x0 = max(0, gx + int(gw * l_f))
            y0 = max(0, gy + int(gh * t_f))
            x1 = min(img_w, gx + int(gw * r_f))
            y1 = min(img_h, gy + int(gh * b_f))
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                results.append(False)
                continue

            template = _cv2.imread(str(template_path))
            if template is None:
                results.append(False)
                continue

            th, tw = template.shape[:2]
            ch, cw = crop.shape[:2]
            if tw > cw or th > ch:
                scale = min(cw / tw, ch / th)
                template = _cv2.resize(
                    template,
                    (max(1, int(tw * scale)), max(1, int(th * scale))),
                    interpolation=_cv2.INTER_AREA,
                )

            gray_crop = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
            gray_tpl = _cv2.cvtColor(template, _cv2.COLOR_BGR2GRAY)
            result_map = _cv2.matchTemplate(gray_crop, gray_tpl, _cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = _cv2.minMaxLoc(result_map)
            results.append(score >= threshold)

        return results

    def _play_find_best_tile(self) -> tuple[str, int, int] | None:
        """Return (tile_name, screen_cx, screen_cy) using safe-first tile priority."""
        if self._last_capture_signature is None or self._last_image_size is None:
            return None

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        snapshot = solve_game_state(self._game_state)
        if snapshot.total_configurations == 0:
            return None

        best_region: Region | None = None
        best_rank: tuple[int, float, float, int, int] | None = None
        for region in tile_regions:
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            if not m:
                continue
            r_idx, c_idx = int(m.group(1)), int(m.group(2))
            if self._game_state.board[r_idx][c_idx].revealed:
                continue
            pos = (r_idx, c_idx)
            prob = snapshot.bomb_probabilities.get(pos, 1.0)
            probs = snapshot.value_probabilities.get(pos, {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})
            expected_value = _expected_tile_value(probs)
            rank = (
                _recommendation_bucket(prob, pos in snapshot.useful_positions),
                prob,
                -expected_value,
                r_idx,
                c_idx,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_region = region

        if best_region is None:
            return None

        image_w, image_h = self._last_image_size
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is None:
            screen = self._get_selected_screen()
            if screen is None:
                return None
            geo = screen.geometry()
            mapping_rect = _map_image_to_overlay(geo.width(), geo.height(), image_w, image_h)
            mapping_rect.translate(geo.x(), geo.y())

        rect = _map_region_rect(best_region, mapping_rect, image_w, image_h)
        return best_region.name, rect.center().x(), rect.center().y()

    def _play_do_xdotool_click(self, cx: int, cy: int, window_id: int | None) -> None:
        """Perform a mouse click at (cx, cy) via xdotool (safe to call from any thread)."""
        try:
            if window_id is not None:
                if shutil.which("wmctrl"):
                    subprocess.run(["wmctrl", "-ia", hex(window_id)], check=False, capture_output=True)
                else:
                    subprocess.run(
                        ["xdotool", "windowfocus", "--sync", str(window_id)],
                        check=False, capture_output=True,
                    )
                time.sleep(0.15)
            subprocess.run(["xdotool", "mousemove", "--sync", str(cx), str(cy)], check=False, capture_output=True)
            time.sleep(0.1)
            subprocess.run(["xdotool", "mousedown", "--clearmodifiers", "1"], check=False, capture_output=True)
            time.sleep(0.08)
            subprocess.run(["xdotool", "mouseup", "--clearmodifiers", "1"], check=False, capture_output=True)
        except Exception:
            pass

    def _play_click_textbox_center(self) -> None:
        """Click the centre of the textbox region to advance a dialog box."""
        region, _ = self._get_textbox_region()
        l_f, t_f, r_f, b_f = region

        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is not None:
            gx, gy, gw, gh = (
                mapping_rect.x(), mapping_rect.y(),
                mapping_rect.width(), mapping_rect.height(),
            )
        else:
            screen = self._get_selected_screen()
            if screen is None:
                return
            geo = screen.geometry()
            if self._last_image_size:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *self._last_image_size)
                mr.translate(geo.x(), geo.y())
                gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()
            else:
                gx, gy, gw, gh = geo.x(), geo.y(), geo.width(), geo.height()

        cx = gx + int(gw * (l_f + r_f) / 2)
        cy = gy + int(gh * (t_f + b_f) / 2)
        self._play_do_xdotool_click(cx, cy, self.state.target_window_id)

    def _play_click_game_center(self) -> None:
        """Click the centre of the game area to advance a dialog box."""
        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        if mapping_rect is not None:
            cx = mapping_rect.center().x()
            cy = mapping_rect.center().y()
        else:
            screen = self._get_selected_screen()
            if screen is None:
                return
            geo = screen.geometry()
            if self._last_image_size:
                mr = _map_image_to_overlay(geo.width(), geo.height(), *self._last_image_size)
                mr.translate(geo.x(), geo.y())
                cx = mr.center().x()
                cy = mr.center().y()
            else:
                cx = geo.center().x()
                cy = geo.center().y()
        self._play_do_xdotool_click(cx, cy, self.state.target_window_id)

    def _refresh_tiles(self) -> None:
        if self.state.last_input_path is None:
            self._show_error("No labeled game found. Use 'Start Game' first.")
            return
        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if not tile_regions:
            self._show_error("No tile regions found. Use 'Start Game' to label the board first.")
            return

        was_prob_overlay_on = self.prob_overlay_btn.isChecked()
        if was_prob_overlay_on:
            self.simple_overlay.hide()

        # Capture a fresh screenshot into the existing path so parse_tiles() picks it up.
        if self.state.target_window_id is not None:
            pixmap = self._capture_window(self.state.target_window_id)
        else:
            screen = self._get_selected_screen()
            if screen is None:
                if was_prob_overlay_on:
                    self.simple_overlay.show()
                self._show_error("No monitor selected for screen capture.")
                return
            pixmap = screen.grabWindow(0)

        if pixmap is None or pixmap.isNull() or not pixmap.save(self.state.last_input_path):
            if was_prob_overlay_on:
                self.simple_overlay.show()
            self._show_error("Failed to capture a fresh screenshot for tile refresh.")
            return

        self.parse_tiles()

        if was_prob_overlay_on:
            # parse_tiles already called _recompute_probability_overlay which populated the
            # overlay data; just re-show the windows.
            self.simple_overlay.show()

    def _normalize_log_key(self, message: str) -> str:
        """Replace digit runs with '#' so repeated-but-numbered messages share a dedup key."""
        return re.sub(r"\d+", "#", message)

    def _set_status(self, message: str, level: str = "info") -> None:
        color_map = {
            "info":    "#2f4d66",
            "success": "#1a6b3a",
            "warning": "#7a5a00",
            "error":   "#8b1a1a",
        }
        fg = color_map.get(level, color_map["info"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        safe_msg = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        key = self._normalize_log_key(message)
        if key == self._log_last_key and self._log_last_block >= 0:
            self._log_last_count += 1
            count_html = f' <span style="color:#888;">\u00d7{self._log_last_count}</span>'
            html = f'<span style="color:{fg};">[{timestamp}] {safe_msg}{count_html}</span>'
            block = self.log_view.document().findBlockByNumber(self._log_last_block)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.insertHtml(html)
                sb = self.log_view.verticalScrollBar()
                sb.setValue(sb.maximum())
                return
            # Block no longer valid — fall through to re-append

        html = f'<span style="color:{fg};">[{timestamp}] {safe_msg}</span>'
        self.log_view.append(html)
        self._log_last_key = key
        self._log_last_block = self.log_view.document().blockCount() - 1
        self._log_last_count = 1
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_monitor_hint(self) -> None:
        screen = self._get_selected_screen()
        if screen is None:
            self.monitor_hint.setText("")
            return

        geo = screen.geometry()
        self.monitor_hint.setText(
            f"{geo.width()}×{geo.height()} at ({geo.x()}, {geo.y()})"
        )

    def _update_window_name_label(self) -> None:
        if self.state.target_window_id is None:
            self.window_name_label.setText("No window selected.")
        else:
            name = self.state.target_window_name or "Unnamed window"
            self.window_name_label.setText(name)
            self.window_name_label.setToolTip(f"Window ID: {self.state.target_window_id}")

    def _update_window_hint(self) -> None:
        self._update_window_name_label()

    def _update_target_window_button(self) -> None:
        if self.state.target_window_id is None:
            self.target_window_btn.setText("Pick Window")
            self.target_window_btn.setToolTip("Select an emulator window for direct capture.")
            return

        self.target_window_btn.setText("Clear Window")
        self.target_window_btn.setToolTip("Clear the selected target window and capture from monitor.")

    def _handle_target_window_button(self) -> None:
        if self.state.target_window_id is None:
            self.pick_target_window()
            return
        self.clear_target_window()

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
        self.state.target_window_class = self._query_window_class(self.state.target_window_id)

        self._align_monitor_to_target_window()
        self._save_prefs()
        self._update_window_hint()
        self._update_target_window_button()
        self._set_status(
            f"Selected target window #{self.state.target_window_id}: {self.state.target_window_name}",
            level="success",
        )

    def clear_target_window(self) -> None:
        self.state.target_window_id = None
        self.state.target_window_name = None
        self.state.target_window_class = None
        self._save_prefs()
        self._update_window_hint()
        self._update_target_window_button()
        self._set_status("Cleared target window. Capture will use selected monitor.", level="info")

    # ── Persistent preferences ────────────────────────────────────────────────────

    _PREFS_PATH = Path.home() / ".local" / "share" / "voltorb-solver" / "prefs.json"

    def _save_prefs(self) -> None:
        """Persist user preferences (currently: last target window name)."""
        try:
            self._PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            if self._PREFS_PATH.exists():
                try:
                    data = json.loads(self._PREFS_PATH.read_text())
                except Exception:
                    data = {}
            data["target_window_name"] = self.state.target_window_name
            data["target_window_class"] = self.state.target_window_class
            screen = self._get_selected_screen()
            data["monitor_name"] = screen.name() if screen is not None else None
            self._PREFS_PATH.write_text(json.dumps(data, indent=2))
            if self.state.target_window_name is not None:
                self._set_status(
                    f"Window preference saved: \"{self.state.target_window_name}\"",
                    level="info",
                )
        except Exception:
            pass

    def _restore_window_from_prefs(self) -> None:
        """On startup, restore saved monitor and window preferences."""
        try:
            data = json.loads(self._PREFS_PATH.read_text())
        except Exception:
            return

        # Restore monitor by name (e.g. "HDMI-1").
        # Block signals so setCurrentIndex doesn't trigger _on_monitor_changed → _save_prefs,
        # which would overwrite target_window_name with null before we restore it below.
        saved_monitor: str | None = data.get("monitor_name")
        if saved_monitor:
            for idx, screen in enumerate(self._screens):
                if screen.name() == saved_monitor:
                    if idx != self.state.selected_screen_index:
                        self.monitor_combo.blockSignals(True)
                        self.monitor_combo.setCurrentIndex(idx)
                        self.monitor_combo.blockSignals(False)
                        self.state.selected_screen_index = idx
                        self.x11_overlay.set_target_screen(screen)
                        self.simple_overlay.set_target_screen(screen)
                        self._update_monitor_hint()
                    break

        # Restore target window. Prefer WM_CLASS (stable) over title (may change).
        saved_name: str | None = data.get("target_window_name")
        saved_class: str | None = data.get("target_window_class")
        if not (saved_name or saved_class) or shutil.which("xdotool") is None:
            return
        ids: list[str] = []
        if saved_class:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--class", saved_class],
                    capture_output=True, text=True, check=False,
                )
                ids = [l.strip() for l in result.stdout.splitlines() if l.strip().isdigit()]
            except Exception:
                pass
        if not ids and saved_name:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--name", re.escape(saved_name)],
                    capture_output=True, text=True, check=False,
                )
                ids = [l.strip() for l in result.stdout.splitlines() if l.strip().isdigit()]
            except Exception:
                pass
        if not ids:
            return
        self.state.target_window_id = int(ids[0])
        self.state.target_window_name = saved_name
        self.state.target_window_class = saved_class
        self._align_monitor_to_target_window()
        self._update_window_hint()
        self._update_target_window_button()

    def _capture_window(self, window_id: int):
        # Try the selected screen first — it's almost always the right one and
        # avoids slow grabWindow calls on every other screen.
        selected = self._get_selected_screen()
        if selected is not None:
            pixmap = selected.grabWindow(window_id)
            if not pixmap.isNull():
                return pixmap

        # Fall back: try remaining screens and pick the largest capture.
        best_pixmap = None
        best_area = 0
        for screen in QGuiApplication.screens():
            if screen is selected:
                continue
            pixmap = screen.grabWindow(window_id)
            if pixmap.isNull():
                continue
            area = pixmap.width() * pixmap.height()
            if area > best_area:
                best_area = area
                best_pixmap = pixmap

        if best_pixmap is None:
            return selected.grabWindow(0) if selected is not None else None
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

    def relabel_regions(self) -> None:
        output_path = str(
            Path(gettempdir()) / f"voltorb_overlay_capture_monitor_{self.state.selected_screen_index + 1}.png"
        )
        capture_signature = self._build_capture_signature()
        if not self._capture_game_screenshot(output_path):
            return
        self._parse_and_apply(
            output_path, capture_signature=capture_signature, relabel_reason="label/relabel game"
        )

    def _capture_game_screenshot(self, output_path: str) -> bool:
        """Capture the game window (or selected screen) to output_path.

        Hides the overlay during capture and restores it afterwards.
        Returns ``True`` on success, ``False`` on failure (error is already shown).
        """
        was_overlay_visible = self.overlay_btn.isChecked()
        if was_overlay_visible:
            self.x11_overlay.hide()
        try:
            if self.state.target_window_id is not None:
                pixmap = self._capture_window(self.state.target_window_id)
            else:
                screen = self._get_selected_screen()
                if screen is None:
                    self._show_error("No monitor selected for screen capture.")
                    return False
                pixmap = screen.grabWindow(0)

            if pixmap is None or pixmap.isNull() or not pixmap.save(output_path):
                target_desc = (
                    f"window #{self.state.target_window_id}"
                    if self.state.target_window_id is not None
                    else "selected monitor"
                )
                self._show_error(f"Failed to capture {target_desc}.")
                return False
            return True
        finally:
            if was_overlay_visible:
                self.x11_overlay.show()

    def _detect_and_cache_anchor_board_rect(self, image_path: str) -> bool:
        """Try to locate the board via TR + BL anchor templates and cache the result.

        Returns ``True`` if both corners were found with sufficient confidence.
        """
        if _cv2 is None:
            return False
        rect = self.parser.find_board_corner_rect(image_path)
        if rect is None:
            return False
        img = _cv2.imread(image_path, _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        self._anchor_board_rect = rect
        self._anchor_image_size = (img.shape[1], img.shape[0])
        return True

    def _capture_anchor_region_image(self) -> None:
        """Capture a screenshot, crop to the cached anchor board rect, and save it for template building."""
        if _cv2 is None:
            self._show_error("OpenCV (cv2) is required for anchor region capture.")
            return
        if self._anchor_board_rect is None or self._anchor_image_size is None:
            self._set_status("No anchor region cached — run 'Parse Anchors' first.", "warning")
            return

        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_anchor_region_full_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        if not self._capture_game_screenshot(tmp_path):
            return

        image = _cv2.imread(tmp_path)
        if image is None:
            self._show_error("Failed to read captured screenshot.")
            return

        left, top, right, bottom = self._anchor_board_rect
        img_h, img_w = image.shape[:2]
        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(img_w, right)
        y1 = min(img_h, bottom)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            self._show_error("Anchor region crop is empty — board rect may be off-screen.")
            return

        out_dir = self._clue_dataset_root.parent / "anchor_region"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{stamp}_anchor_region.png"
        _cv2.imwrite(str(out_path), crop)
        self._set_status(f"Anchor region saved → {out_path}", "success")

    def _parse_anchors_debug(self) -> None:
        """Capture a screenshot, detect TR+BL anchors, and report the board rect."""
        tmp_path = str(
            Path(gettempdir())
            / f"voltorb_anchor_capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        if not self._capture_game_screenshot(tmp_path):
            return
        found = self._detect_and_cache_anchor_board_rect(tmp_path)
        if not found:
            self._set_status("Anchor detection failed — no board rect found. Check TR+BL anchor templates.", "warning")
            return
        left, top, right, bottom = self._anchor_board_rect  # type: ignore[misc]
        img_w, img_h = self._anchor_image_size  # type: ignore[misc]
        self._set_status(
            f"Anchors detected: board px ({left},{top})–({right},{bottom})  "
            f"{right - left}×{bottom - top}px  image {img_w}×{img_h}",
            "success",
        )

    def _show_anchor_region_overlay(self) -> None:
        """Draw a border overlay showing the board rect derived from TR+BL anchor templates."""
        if self._anchor_board_rect is None or self._anchor_image_size is None:
            self._set_status("No anchor rect cached — run 'Parse Anchors' first.", "warning")
            return

        left, top, right, bottom = self._anchor_board_rect
        img_w, img_h = self._anchor_image_size

        screen = self._get_selected_screen()
        if screen is None:
            self._show_error("No monitor selected.")
            return

        # Compute the mapping from image pixels to screen coordinates directly from
        # the current window/screen geometry — do NOT rely on _last_capture_signature,
        # which may be stale or from a different capture (e.g. a prior screen capture
        # while a window is now selected).
        if self.state.target_window_id is not None:
            geometry = self._query_window_geometry(self.state.target_window_id)
            if geometry is None:
                self._show_error("Could not query window geometry.")
                return
            gx, gy, gw, gh = geometry
        else:
            geo = screen.geometry()
            mr = _map_image_to_overlay(geo.width(), geo.height(), img_w, img_h)
            mr.translate(geo.x(), geo.y())
            gx, gy, gw, gh = mr.x(), mr.y(), mr.width(), mr.height()

        # Convert image-pixel board corners → screen coords.
        scale_x = gw / img_w
        scale_y = gh / img_h
        rx = gx + int(left * scale_x)
        ry = gy + int(top * scale_y)
        rw = max(1, int((right - left) * scale_x))
        rh = max(1, int((bottom - top) * scale_y))
        board_rect = QRect(rx, ry, rw, rh)

        color = QColor(255, 200, 0)  # gold
        segments: list[OverlayBorderWindow] = []
        for side in ("top", "bottom", "left", "right"):
            seg = OverlayBorderWindow(color, thickness=3)
            seg.show_for_rect(board_rect, side, screen=screen)
            segments.append(seg)

        self._set_status(
            f"Anchor board region: screen=({rx},{ry}) {rw}×{rh}px  "
            f"image px ({left},{top})–({right},{bottom})",
            "info",
        )

        def _hide() -> None:
            for s in segments:
                s.hide()
                s.deleteLater()

        QTimer.singleShot(4000, _hide)

    def _board_region_from_anchor(
        self,
        bx: float,
        by: float,
        bw: float,
        bh: float,
    ) -> tuple[tuple[float, float, float, float], bool] | None:
        """Compute a textbox region from the cached anchor board rect.

        ``bx``, ``by``, ``bw``, ``bh`` are board-relative fractions where
        (0, 0) = board top-left and (1, 1) = board bottom-right.
        Returns ``(frac_region, False)`` or ``None`` when the anchor rect is unavailable.
        """
        if self._anchor_board_rect is None or self._anchor_image_size is None:
            return None
        left, top, right, bottom = self._anchor_board_rect
        img_w, img_h = self._anchor_image_size
        board_w = right - left
        board_h = bottom - top
        if board_w <= 0 or board_h <= 0:
            return None
        x0 = left + int(bx * board_w)
        y0 = top + int(by * board_h)
        x1 = x0 + int(bw * board_w)
        y1 = y0 + int(bh * board_h)
        x0 = max(0, min(x0, img_w - 1))
        x1 = max(x0 + 1, min(x1, img_w))
        y0 = max(0, min(y0, img_h - 1))
        y1 = max(y0 + 1, min(y1, img_h))
        return (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h), False

    def clear_overlay(self) -> None:
        self.x11_overlay.clear_overlay()
        self.simple_overlay.clear()
        self.state.last_input_path = None
        self._cached_regions = []
        self._last_capture_signature = None
        self._clue_parsed_values = {}
        self._tile_parsed_values = {}
        self._game_state.reset()
        self._last_image_size = None
        self._last_parse_regions = []
        self._update_region_picker()
        self._set_status("Overlay cleared.", level="info")

    def toggle_overlay(self, checked: bool) -> None:
        self.overlay_btn.setText("Disable Debug Overlay" if checked else "Enable Debug Overlay")
        if checked:
            self._show_active_overlay()
        else:
            self.x11_overlay.hide()

    def toggle_prob_overlay(self, checked: bool) -> None:
        self.prob_overlay_btn.setText("Disable Prob Overlay" if checked else "Enable Prob Overlay")
        if checked:
            screen = self._get_selected_screen()
            if screen is None:
                self._show_error("No monitor selected for overlay.")
                self.prob_overlay_btn.blockSignals(True)
                self.prob_overlay_btn.setChecked(False)
                self.prob_overlay_btn.setText("Enable Prob Overlay")
                self.prob_overlay_btn.blockSignals(False)
                return
            self.simple_overlay.set_target_screen(screen)
            self.simple_overlay.show()
        else:
            self.simple_overlay.hide()

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
        self.simple_overlay.set_target_screen(self._screens[selected])
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
            self.simple_overlay.set_target_screen(screen)
        self._update_monitor_hint()
        self._save_prefs()

        if self.overlay_btn.isChecked():
            self._show_active_overlay()

    def _get_selected_screen(self):
        if not self._screens:
            return None
        index = min(max(self.state.selected_screen_index, 0), len(self._screens) - 1)
        return self._screens[index]

    def _query_window_class(self, window_id: int) -> str | None:
        """Return the WM_CLASS instance name for a window, or None if unavailable."""
        if shutil.which("xprop") is None:
            return None
        try:
            result = subprocess.run(
                ["xprop", "-id", str(window_id), "WM_CLASS"],
                capture_output=True, text=True, check=False,
            )
        except Exception:
            return None
        # Output: WM_CLASS(STRING) = "melonDS", "melonDS"
        m = re.search(r'WM_CLASS\(STRING\)\s*=\s*"([^"]+)"', result.stdout)
        return m.group(1) if m else None

    def _query_window_geometry(self, window_id: int) -> tuple[int, int, int, int] | None:
        if shutil.which("xwininfo") is None:
            return None

        try:
            result = subprocess.run(
                ["xwininfo", "-id", str(window_id), "-int"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        x_match = re.search(r"Absolute upper-left X:\s+(-?\d+)", result.stdout)
        y_match = re.search(r"Absolute upper-left Y:\s+(-?\d+)", result.stdout)
        w_match = re.search(r"Width:\s+(\d+)", result.stdout)
        h_match = re.search(r"Height:\s+(\d+)", result.stdout)
        if not (x_match and y_match and w_match and h_match):
            return None
        return (
            int(x_match.group(1)),
            int(y_match.group(1)),
            int(w_match.group(1)),
            int(h_match.group(1)),
        )

    def _build_capture_signature(self) -> tuple[str, int | None, int, int, int, int] | None:
        if self.state.target_window_id is not None:
            geometry = self._query_window_geometry(self.state.target_window_id)
            if geometry is None:
                return None
            x, y, w, h = geometry
            return ("window", self.state.target_window_id, x, y, w, h)

        screen = self._get_selected_screen()
        if screen is None:
            return None
        geo = screen.geometry()
        return ("screen", self.state.selected_screen_index, geo.x(), geo.y(), geo.width(), geo.height())

    def _should_relabel_reason(
        self,
        capture_signature: tuple[str, int | None, int, int, int, int] | None,
    ) -> str | None:
        if not self._cached_regions:
            return "initial capture"
        if capture_signature is None:
            return "capture geometry could not be verified"
        if self._last_capture_signature != capture_signature:
            return "capture window geometry changed"
        return None

    def _parse_and_apply(
        self,
        image_path: str,
        *,
        capture_signature: tuple[str, int | None, int, int, int, int] | None,
        relabel_reason: str,
    ) -> bool:
        try:
            result = self.parser.parse(image_path)
        except Exception as exc:
            self._show_error(f"Failed to parse screenshot: {exc}")
            return False

        self.state.last_input_path = image_path
        self._last_image_size = (result.image_width, result.image_height)
        self._last_parse_regions = list(result.regions)
        self._clue_parsed_values = {}
        overlay_regions = self._expand_with_clue_subregions(result.regions)
        self._cached_regions = list(overlay_regions)
        self._last_capture_signature = capture_signature
        self.x11_overlay.set_mapping_rect(self._mapping_rect_for_signature(capture_signature))
        self.x11_overlay.set_overlay_data(overlay_regions, result.image_width, result.image_height)
        self._update_region_picker()

        warning_text = ""
        if result.warnings:
            warning_text = f" Warnings: {' | '.join(result.warnings)}"
        method_text = f" Methods: {result.method_summary()}."
        monitor_text = f" Monitor: {self.state.selected_screen_index + 1}."
        level = "warning" if result.warnings else "success"
        self._set_status(
            f"Parsed {len(result.regions)} base regions (+ clue subregions) from {Path(image_path).name} ({relabel_reason}).{method_text}{monitor_text}{warning_text}",
            level=level,
        )
        return True

    def _expand_with_clue_subregions(self, regions: list[Region]) -> list[Region]:
        total_bounds = getattr(self.clue_parser, "_TOTAL_OCR_BOUNDS", (0.05, 0.02, 0.95, 0.42))
        voltorb_bounds = getattr(self.clue_parser, "_VOLTORB_OCR_BOUNDS", (0.52, 0.50, 0.98, 0.98))

        expanded = []
        for region in regions:
            if not self._is_clue_region(region.name):
                expanded.append(region)
                continue

            total_rect = self._subregion_from_bounds(region, total_bounds)
            voltorb_rect = self._subregion_from_bounds(region, voltorb_bounds)

            expanded.append(Region(f"{region.name}.total", *total_rect))
            expanded.append(Region(f"{region.name}.voltorbs", *voltorb_rect))
        return expanded

    def _is_clue_region(self, name: str) -> bool:
        if len(name) < 2:
            return False
        if name[0] not in {"r", "c"}:
            return False
        return name[1:].isdigit()

    def _is_tile_region(self, name: str) -> bool:
        return bool(re.match(r"^\(\d,\d\)$", name))

    def _subregion_from_bounds(
        self,
        region: Region,
        bounds: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        x0_f, y0_f, x1_f, y1_f = bounds
        x0 = region.x + max(0, min(int(round(region.w * x0_f)), region.w - 1))
        y0 = region.y + max(0, min(int(round(region.h * y0_f)), region.h - 1))
        x1 = region.x + max(x0 - region.x + 1, min(int(round(region.w * x1_f)), region.w))
        y1 = region.y + max(y0 - region.y + 1, min(int(round(region.h * y1_f)), region.h))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _apply_all_label_updates(self) -> None:
        """Push all parsed labels (clue and tile) to the overlay in a single pass."""
        if self._last_image_size is None:
            return
        labeled = []
        for region in self._cached_regions:
            # Clue subregion labels: "r0.total" → "r0.total=12", "r0.voltorbs" → "r0.v=2"
            if "." in region.name:
                clue_name, field = region.name.rsplit(".", 1)
                values = self._clue_parsed_values.get(clue_name)
                if values is not None:
                    voltorbs, total = values
                    if field == "total":
                        new_name = f"{clue_name}.total={total}"
                    elif field == "voltorbs":
                        new_name = f"{clue_name}.v={voltorbs}"
                    else:
                        new_name = region.name
                    labeled.append(Region(new_name, region.x, region.y, region.w, region.h))
                    continue
            # Tile region labels: "(0,1)" → "(0,1)=2", "(0,1)=V", "(0,1)=-"
            elif self._is_tile_region(region.name) and region.name in self._tile_parsed_values:
                raw_value = self._tile_parsed_values[region.name]
                if raw_value is None:
                    display = "-"
                elif raw_value == 0:
                    display = "V"
                else:
                    display = str(raw_value)
                labeled.append(Region(f"{region.name}={display}", region.x, region.y, region.w, region.h))
                continue
            labeled.append(region)
        self.x11_overlay.set_overlay_data(labeled, *self._last_image_size)

    def _apply_clue_label_updates(self) -> None:
        self._apply_all_label_updates()

    def _apply_tile_label_updates(self) -> None:
        self._apply_all_label_updates()

    def parse_all_clues(self) -> None:
        if self.state.last_input_path is None:
            self._show_error("Label the game first.")
            return

        clue_regions = [r for r in self._last_parse_regions if self._is_clue_region(r.name)]
        if not clue_regions:
            self._show_error("No clue regions found. Label the game first.")
            return

        debug = self.debug_checkbox.isChecked()
        debug_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if debug else None
        parsed_count = 0
        failed: list[str] = []
        for region in clue_regions:
            axis = "col" if region.name.startswith("c") else "row"
            if debug:
                artifacts = self.clue_parser.debug_parse_clue_from_screenshot(
                    self.state.last_input_path,
                    (region.x, region.y, region.w, region.h),
                    output_root=self._clue_dataset_root / "debug_parse",
                    region_name=region.name,
                    run_id=debug_run_id,
                    axis=axis,
                )
                pair = (
                    (artifacts.voltorbs_value, artifacts.total_value)
                    if artifacts is not None
                    and artifacts.voltorbs_value is not None
                    and artifacts.total_value is not None
                    else None
                )
            else:
                pair = self.clue_parser.parse_clue_from_screenshot(
                    self.state.last_input_path,
                    (region.x, region.y, region.w, region.h),
                    fast=not region.name.startswith("c"),
                    axis=axis,
                )
            if pair is not None:
                self._clue_parsed_values[region.name] = pair
                parsed_count += 1
            else:
                failed.append(region.name)
        self._apply_clue_label_updates()
        self._push_clues_to_game_state()
        self._recompute_probability_overlay()
        if failed:
            self._set_status(
                f"Parsed {parsed_count}/{len(clue_regions)} clues. Failed: {', '.join(failed)}.",
                level="warning",
            )
        else:
            self._set_status(
                f"Parsed all {parsed_count} clues. Overlay labels updated.",
                level="success",
            )

    def parse_tiles(self) -> None:
        if self.state.last_input_path is None:
            self._show_error("Label the game first.")
            return

        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if not tile_regions:
            self._show_error("No tile regions found. Label the game first.")
            return

        debug = self.debug_checkbox.isChecked()
        debug_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if debug else None
        prev_unmatched = len(self.clue_parser._unmatched_saved_hashes)

        # Reset tile state so removed/changed tiles are not carried over from a previous parse.
        self._tile_parsed_values = {}
        for r_idx in range(5):
            for c_idx in range(5):
                self._game_state.set_tile_hidden(r_idx, c_idx)

        for region in tile_regions:
            if debug:
                artifacts = self.clue_parser.debug_parse_tile_from_screenshot(
                    self.state.last_input_path,
                    (region.x, region.y, region.w, region.h),
                    output_root=self._tile_dataset_root / "debug_parse",
                    region_name=region.name,
                    run_id=debug_run_id,
                )
                value = artifacts.result if artifacts is not None else None
            else:
                value = self.clue_parser.parse_tile_from_screenshot(
                    self.state.last_input_path,
                    (region.x, region.y, region.w, region.h),
                )
            self._tile_parsed_values[region.name] = value
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            if m:
                r_idx, c_idx = int(m.group(1)), int(m.group(2))
                if value is None:
                    self._game_state.set_tile_hidden(r_idx, c_idx)
                else:
                    self._game_state.set_tile_revealed(r_idx, c_idx, value)

        self._apply_tile_label_updates()
        self._recompute_probability_overlay()

        n_revealed = sum(1 for v in self._tile_parsed_values.values() if v is not None)
        n_total = len(tile_regions)
        n_unmatched = len(self.clue_parser._unmatched_saved_hashes) - prev_unmatched
        n_closed = n_total - n_revealed - n_unmatched

        if n_unmatched > 0 and n_revealed == 0:
            self._set_status(
                f"No tiles classified. {n_unmatched} open tile(s) saved to "
                "assets/parser_debug/tile_dataset/unknown/ for labeling.",
                level="warning",
            )
        elif n_unmatched > 0:
            self._set_status(
                f"Classified {n_revealed} revealed + {n_closed} closed. "
                f"{n_unmatched} open tile(s) saved to unknown/ for labeling.",
                level="warning",
            )
        elif n_revealed == 0:
            self._set_status(
                f"All {n_total} tiles appear face-down (closed). Overlay labels updated.",
                level="success",
            )
        elif n_revealed < n_total:
            self._set_status(
                f"Classified {n_revealed} revealed + {n_closed} closed ({n_total} total). "
                "Overlay labels updated.",
                level="success",
            )
        else:
            self._set_status(
                f"Classified all {n_revealed} tiles. Overlay labels updated.",
                level="success",
            )

    def _push_clues_to_game_state(self) -> None:
        for name, (voltorbs, total) in self._clue_parsed_values.items():
            idx = int(name[1:])
            try:
                if name[0] == "r":
                    self._game_state.set_row_clue(idx, voltorbs, total)
                else:
                    self._game_state.set_col_clue(idx, voltorbs, total)
            except (ValueError, IndexError):
                pass

    def _recompute_probability_overlay(self) -> None:
        if self._last_image_size is None:
            return
        tile_regions = [r for r in self._last_parse_regions if self._is_tile_region(r.name)]
        if not tile_regions:
            return

        snapshot = solve_game_state(self._game_state)
        if snapshot.total_configurations == 0:
            return

        unrevealed: list[Region] = []
        for region in tile_regions:
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            if not m:
                continue
            r_idx, c_idx = int(m.group(1)), int(m.group(2))
            if not self._game_state.board[r_idx][c_idx].revealed:
                unrevealed.append(region)

        if not unrevealed:
            self.simple_overlay.clear()
            return

        recommended_name: str | None = None
        best_rank: tuple[int, float, float, int, int] | None = None
        for region in unrevealed:
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            assert m
            pos = (int(m.group(1)), int(m.group(2)))
            prob = snapshot.bomb_probabilities.get(pos, 1.0)
            probs = snapshot.value_probabilities.get(pos, {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})
            expected_value = _expected_tile_value(probs)
            rank = (
                _recommendation_bucket(prob, pos in snapshot.useful_positions),
                prob,
                -expected_value,
                int(m.group(1)),
                int(m.group(2)),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                recommended_name = region.name

        data: list[tuple[tuple[int, int, int, int], float, bool]] = []
        for region in unrevealed:
            m = re.match(r"^\((\d),(\d)\)$", region.name)
            assert m
            prob = snapshot.bomb_probabilities.get((int(m.group(1)), int(m.group(2))), 1.0)
            data.append(((region.x, region.y, region.w, region.h), prob, region.name == recommended_name))

        mapping_rect = self._mapping_rect_for_signature(self._last_capture_signature)
        self.simple_overlay.set_data(data, *self._last_image_size, mapping_rect)

    def _show_error(self, message: str) -> None:
        self._set_status(message, level="error")
        QMessageBox.critical(self, "Voltorb Solver X11 Overlay", message)

    def _mapping_rect_for_signature(
        self,
        capture_signature: tuple[str, int | None, int, int, int, int] | None,
    ) -> QRect | None:
        if capture_signature is None:
            return None
        _kind, _id, x, y, w, h = capture_signature
        if w <= 0 or h <= 0:
            return None
        return QRect(x, y, w, h)


def run_overlay_app() -> int:
    app = QApplication.instance() or QApplication([])
    window = OverlayControlWindow()
    window.show()
    return app.exec()