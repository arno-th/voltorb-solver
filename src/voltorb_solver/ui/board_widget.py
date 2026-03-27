from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QPushButton, QWidget

from voltorb_solver.game_state import BOARD_SIZE, GameState
from voltorb_solver.solver import SolverSnapshot


class BoardWidget(QWidget):
    tile_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buttons: list[list[QPushButton]] = []

        grid = QGridLayout(self)
        grid.setSpacing(4)

        for r in range(BOARD_SIZE):
            row_buttons: list[QPushButton] = []
            for c in range(BOARD_SIZE):
                btn = QPushButton("?")
                btn.setMinimumSize(70, 70)
                btn.clicked.connect(lambda _checked=False, row=r, col=c: self.tile_clicked.emit(row, col))
                grid.addWidget(btn, r, c)
                row_buttons.append(btn)
            self._buttons.append(row_buttons)

    def render(self, state: GameState, snapshot: SolverSnapshot) -> None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = self._buttons[r][c]
                tile = state.board[r][c]
                if tile.revealed and tile.value is not None:
                    btn.setText(str(tile.value))
                    if tile.value == 0:
                        btn.setStyleSheet("background-color: #c44; color: white; font-weight: bold;")
                    else:
                        btn.setStyleSheet("background-color: #3a7; color: white; font-weight: bold;")
                else:
                    bomb_p = snapshot.bomb_probabilities.get((r, c), 0.0)
                    btn.setText(f"?\nB:{bomb_p:.0%}")
                    btn.setStyleSheet("background-color: #263238; color: #f5f5f5;")
