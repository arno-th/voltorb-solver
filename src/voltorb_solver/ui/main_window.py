from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from voltorb_solver.game_state import BOARD_SIZE
from voltorb_solver.image_import.crop_dialog import CropDialog
from voltorb_solver.image_import.parser import ImageParser
from voltorb_solver.recalc_service import RecalculationService
from voltorb_solver.stats import StatsManager
from voltorb_solver.ui.board_widget import BoardWidget
from voltorb_solver.ui.solver_panel import SolverPanel
from voltorb_solver.ui.stats_panel import StatsPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voltorb Solver")
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)

        self.service = RecalculationService()
        self.parser = ImageParser()
        self.stats = StatsManager()

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.setCentralWidget(root)

        header = QLabel("Voltorb Solver")
        header.setObjectName("title")
        subtitle = QLabel("Track clues, enter reveals quickly, and follow the safest plays.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(header)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        left_panel = self._card("Board")
        left = left_panel.layout()
        middle_panel = self._card("Clues")
        middle = middle_panel.layout()
        right_panel = self._card("Guidance")
        right = right_panel.layout()
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        self.board_widget = BoardWidget()
        self.board_widget.tile_clicked.connect(self._on_tile_selected)
        left.addWidget(self.board_widget, 1)

        self.selected_tile_label = QLabel("Selected Tile: R1C1")
        self.selected_tile_label.setObjectName("selectedTile")
        left.addWidget(self.selected_tile_label)

        tile_editor = QHBoxLayout()
        tile_editor.setSpacing(6)
        self.row_input = QSpinBox()
        self.row_input.setRange(1, BOARD_SIZE)
        self.col_input = QSpinBox()
        self.col_input.setRange(1, BOARD_SIZE)
        self.row_input.valueChanged.connect(self._on_selected_coords_changed)
        self.col_input.valueChanged.connect(self._on_selected_coords_changed)
        tile_editor.addWidget(QLabel("Row"))
        tile_editor.addWidget(self.row_input)
        tile_editor.addWidget(QLabel("Col"))
        tile_editor.addWidget(self.col_input)
        left.addLayout(tile_editor)

        value_row = QHBoxLayout()
        value_row.setSpacing(6)
        self._value_buttons: dict[int, QPushButton] = {}
        for value in (0, 1, 2, 3):
            btn = QPushButton(str(value))
            btn.clicked.connect(lambda _checked=False, v=value: self._set_selected_tile(v))
            value_row.addWidget(btn)
            self._value_buttons[value] = btn
        self.hidden_btn = QPushButton("Set Hidden")
        self.hidden_btn.clicked.connect(lambda _checked=False: self._set_selected_tile(None))
        value_row.addWidget(self.hidden_btn)
        left.addLayout(value_row)

        self.impossible_label = QLabel("Impossible: -")
        self.impossible_label.setObjectName("impossible")
        left.addWidget(self.impossible_label)

        actions = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.undo_btn = QPushButton("Undo")
        self.won_btn = QPushButton("Won! ✓")
        self.won_btn.setObjectName("wonBtn")
        self.import_btn = QPushButton("Import Image")
        self.parse_single_clue_btn = QPushButton("Parse Single Clue")
        self.reset_btn.clicked.connect(self._reset)
        self.undo_btn.clicked.connect(self._undo)
        self.won_btn.clicked.connect(self._handle_round_won)
        self.import_btn.clicked.connect(self._import_image)
        self.parse_single_clue_btn.clicked.connect(self._parse_single_clue)
        actions.addWidget(self.reset_btn)
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.won_btn)
        actions.addWidget(self.import_btn)
        actions.addWidget(self.parse_single_clue_btn)
        left.addLayout(actions)

        middle.addWidget(QLabel("Row Clues (Voltorbs, Sum)"))
        self.row_spinners = self._build_clue_editor(middle, is_row=True)

        middle.addWidget(QLabel("Column Clues (Voltorbs, Sum)"))
        self.col_spinners = self._build_clue_editor(middle, is_row=False)
        middle.addStretch(1)

        self.solver_panel = SolverPanel()
        right.addWidget(self.solver_panel, 1)

        stats_divider = QFrame()
        stats_divider.setFrameShape(QFrame.Shape.HLine)
        stats_divider.setStyleSheet("color: #d4e2ee;")
        right.addWidget(stats_divider)

        stats_header = QLabel("Statistics")
        stats_header.setObjectName("sectionTitle")
        right.addWidget(stats_header)

        self.stats_panel = StatsPanel()
        self.stats_panel.clear_requested.connect(self._clear_lifetime_stats)
        right.addWidget(self.stats_panel)

        self._loading_clues = False
        self._loading_selection = False
        self._selected_tile = (0, 0)
        self._apply_styles()
        self._sync_selected_inputs()
        self._render_all()
        self._refresh_stats()

    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        card_layout.addWidget(title_label)
        return card

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget#root {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f4fbf8, stop: 1 #e7f0fb);
                color: #1f2a37;
            }
            QLabel#title {
                font-size: 30px;
                font-weight: 700;
                color: #155e75;
            }
            QLabel#subtitle {
                color: #355070;
                font-size: 14px;
                margin-bottom: 4px;
            }
            QFrame#card {
                background: #ffffff;
                border: 1px solid #d4e2ee;
                border-radius: 12px;
            }
            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: 700;
                color: #274c77;
            }
            QLabel#selectedTile {
                font-weight: 600;
                color: #0f766e;
            }
            QLabel#impossible {
                color: #7f1d1d;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QPushButton:disabled {
                background-color: #9db4d3;
                color: #f4f8ff;
            }
            QPushButton#wonBtn {
                background-color: #16a34a;
            }
            QPushButton#wonBtn:hover {
                background-color: #15803d;
            }
            QPushButton#wonBtn:pressed {
                background-color: #166534;
            }
            QSpinBox {
                background: #f8fbff;
                border: 1px solid #c7d7eb;
                border-radius: 7px;
                padding: 4px 6px;
                min-height: 28px;
            }
            QListWidget {
                border: 1px solid #d4e2ee;
                border-radius: 8px;
                background: #fcfeff;
            }
            """
        )

    def _build_clue_editor(self, parent_layout: QVBoxLayout, is_row: bool) -> list[tuple[QSpinBox, QSpinBox]]:
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        spinners: list[tuple[QSpinBox, QSpinBox]] = []
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("Voltorbs"), 0, 1)
        grid.addWidget(QLabel("Sum"), 0, 2)
        for idx in range(BOARD_SIZE):
            label = QLabel(f"{'R' if is_row else 'C'}{idx + 1}")
            bombs = QSpinBox()
            bombs.setRange(0, 5)
            total = QSpinBox()
            total.setRange(0, 15)

            bombs.valueChanged.connect(
                lambda _v, i=idx, row_mode=is_row: self._on_clue_changed(i, row_mode)
            )
            total.valueChanged.connect(
                lambda _v, i=idx, row_mode=is_row: self._on_clue_changed(i, row_mode)
            )

            grid.addWidget(label, idx + 1, 0)
            grid.addWidget(bombs, idx + 1, 1)
            grid.addWidget(total, idx + 1, 2)
            spinners.append((bombs, total))

        parent_layout.addLayout(grid)
        return spinners

    def _render_all(self) -> None:
        result = self.service.current
        recommended = (
            (result.safest_moves[0].row, result.safest_moves[0].col)
            if result.safest_moves
            else None
        )
        self.board_widget.render(self.service.state, result.snapshot, self._selected_tile, recommended)
        self.solver_panel.render(result.snapshot, result.safest_moves, result.best_ev_moves)
        self._sync_clue_widgets()
        self._refresh_selection_ui()

    def _refresh_stats(self) -> None:
        self.stats_panel.refresh(self.stats.lifetime, self.stats.session)

    def _clear_lifetime_stats(self) -> None:
        self.stats.reset_lifetime()
        self._refresh_stats()

    def _sync_clue_widgets(self) -> None:
        self._loading_clues = True
        try:
            for idx in range(BOARD_SIZE):
                rb, rs = self.row_spinners[idx]
                rb.setValue(self.service.state.row_clues[idx].voltorbs)
                rs.setValue(self.service.state.row_clues[idx].total)

                cb, cs = self.col_spinners[idx]
                cb.setValue(self.service.state.col_clues[idx].voltorbs)
                cs.setValue(self.service.state.col_clues[idx].total)
        finally:
            self._loading_clues = False

    def _on_clue_changed(self, idx: int, is_row: bool) -> None:
        if self._loading_clues:
            return
        self.service.push_state()
        if is_row:
            bombs, total = self.row_spinners[idx]
            self.service.state.set_row_clue(idx, bombs.value(), total.value())
        else:
            bombs, total = self.col_spinners[idx]
            self.service.state.set_col_clue(idx, bombs.value(), total.value())
        self.service.recalculate()
        self._render_all()

    def _on_tile_selected(self, row: int, col: int) -> None:
        self._selected_tile = (row, col)
        self._sync_selected_inputs()
        self._render_all()

    def _on_selected_coords_changed(self, _value: int) -> None:
        if self._loading_selection:
            return
        self._selected_tile = (self.row_input.value() - 1, self.col_input.value() - 1)
        self._render_all()

    def _sync_selected_inputs(self) -> None:
        self._loading_selection = True
        try:
            row, col = self._selected_tile
            self.row_input.setValue(row + 1)
            self.col_input.setValue(col + 1)
        finally:
            self._loading_selection = False

    def _set_selected_tile(self, value: int | None) -> None:
        row, col = self._selected_tile
        impossible = self.service.current.snapshot.impossible_values.get((row, col), set())
        if value is not None and value in impossible:
            QMessageBox.warning(
                self,
                "Impossible Move",
                f"{value} cannot be placed at R{row + 1}C{col + 1} with the current clues.",
            )
            return

        self.service.push_state()
        if value is None:
            self.service.state.set_tile_hidden(row, col)
        else:
            self.service.state.set_tile_revealed(row, col, value)
        self.service.recalculate()
        self._render_all()
        if value == 0:
            self._handle_bomb_revealed()

    def _refresh_selection_ui(self) -> None:
        row, col = self._selected_tile
        self.selected_tile_label.setText(f"Selected Tile: R{row + 1}C{col + 1}")
        impossible = self.service.current.snapshot.impossible_values.get((row, col), set())
        impossible_text = ", ".join(str(v) for v in sorted(impossible)) or "-"
        self.impossible_label.setText(f"Impossible: {impossible_text}")
        for value, button in self._value_buttons.items():
            button.setEnabled(value not in impossible)

    def _handle_bomb_revealed(self) -> None:
        reply = QMessageBox.question(
            self,
            "Voltorb Hit!",
            "You revealed a Voltorb! Record this as a loss and start a new round?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.stats.record_bomb()
            self._refresh_stats()
            self._reset()

    def _handle_round_won(self) -> None:
        self.stats.record_win()
        self._refresh_stats()
        self._reset()

    def _reset(self) -> None:
        self.service.push_state()
        self.service.state.reset()
        self._selected_tile = (0, 0)
        self._sync_selected_inputs()
        self.service.recalculate()
        self._render_all()

    def _undo(self) -> None:
        if not self.service.undo():
            QMessageBox.information(self, "Undo", "No history available.")
            return
        self._render_all()

    def _import_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Screenshot",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not image_path:
            return

        crop_dialog = CropDialog(image_path, self)
        if crop_dialog.exec() != crop_dialog.DialogCode.Accepted:
            return

        rect = crop_dialog.get_rect()
        parse_result = self.parser.parse_image(image_path, rect)

        self.service.push_state()
        for idx, clue in enumerate(parse_result.row_clues):
            self.service.state.set_row_clue(idx, clue.voltorbs, clue.total)
        for idx, clue in enumerate(parse_result.col_clues):
            self.service.state.set_col_clue(idx, clue.voltorbs, clue.total)
        for (row, col), value in parse_result.revealed_tiles.items():
            self.service.state.set_tile_revealed(row, col, value)

        self.service.recalculate()
        self._render_all()

        if parse_result.warnings:
            QMessageBox.warning(self, "Import Warnings", "\n".join(parse_result.warnings))
        else:
            QMessageBox.information(self, "Import", "Image import complete.")

    def _parse_single_clue(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Clue Screenshot",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not image_path:
            return

        pair = self.parser.parse_clue_box(image_path)
        if pair is None:
            QMessageBox.warning(
                self,
                "Parse Single Clue",
                "Could not confidently parse this clue box. Try a tighter crop and good contrast.",
            )
            return

        voltorbs, total = pair
        QMessageBox.information(
            self,
            "Parse Single Clue",
            f"Parsed clue: voltorbs={voltorbs}, total={total}",
        )
