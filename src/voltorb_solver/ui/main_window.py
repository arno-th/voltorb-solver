from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from voltorb_solver.game_state import BOARD_SIZE
from voltorb_solver.image_import.crop_dialog import CropDialog
from voltorb_solver.image_import.parser import ImageParser
from voltorb_solver.recalc_service import RecalculationService
from voltorb_solver.ui.board_widget import BoardWidget
from voltorb_solver.ui.solver_panel import SolverPanel
from voltorb_solver.ui.value_picker import ValuePickerDialog


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voltorb Solver")
        self.resize(1200, 760)

        self.service = RecalculationService()
        self.parser = ImageParser()

        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        left = QVBoxLayout()
        middle = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(middle, 2)
        layout.addLayout(right, 2)

        self.board_widget = BoardWidget()
        self.board_widget.tile_clicked.connect(self._on_tile_clicked)
        left.addWidget(QLabel("Board"))
        left.addWidget(self.board_widget)

        actions = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.undo_btn = QPushButton("Undo")
        self.import_btn = QPushButton("Import Image")
        self.reset_btn.clicked.connect(self._reset)
        self.undo_btn.clicked.connect(self._undo)
        self.import_btn.clicked.connect(self._import_image)
        actions.addWidget(self.reset_btn)
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.import_btn)
        left.addLayout(actions)

        middle.addWidget(QLabel("Row Clues (Voltorbs, Sum)"))
        self.row_spinners = self._build_clue_editor(middle, is_row=True)

        middle.addWidget(QLabel("Column Clues (Voltorbs, Sum)"))
        self.col_spinners = self._build_clue_editor(middle, is_row=False)
        middle.addStretch(1)

        self.solver_panel = SolverPanel()
        right.addWidget(self.solver_panel)

        self._loading_clues = False
        self._render_all()

    def _build_clue_editor(self, parent_layout: QVBoxLayout, is_row: bool) -> list[tuple[QSpinBox, QSpinBox]]:
        grid = QGridLayout()
        spinners: list[tuple[QSpinBox, QSpinBox]] = []
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

            grid.addWidget(label, idx, 0)
            grid.addWidget(bombs, idx, 1)
            grid.addWidget(total, idx, 2)
            spinners.append((bombs, total))

        parent_layout.addLayout(grid)
        return spinners

    def _render_all(self) -> None:
        result = self.service.current
        self.board_widget.render(self.service.state, result.snapshot)
        self.solver_panel.render(result.snapshot, result.safest_moves, result.best_ev_moves)
        self._sync_clue_widgets()

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

    def _on_tile_clicked(self, row: int, col: int) -> None:
        impossible = self.service.current.snapshot.impossible_values.get((row, col), set())
        dialog = ValuePickerDialog(impossible, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        self.service.push_state()
        if dialog.selected_value is None:
            self.service.state.set_tile_hidden(row, col)
        else:
            self.service.state.set_tile_revealed(row, col, dialog.selected_value)
        self.service.recalculate()
        self._render_all()

    def _reset(self) -> None:
        self.service.push_state()
        self.service.state.reset()
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
