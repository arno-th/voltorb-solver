from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QWidget

from voltorb_solver.game_state import BOARD_SIZE, GameState
from voltorb_solver.solver import SolverSnapshot


class BoardWidget(QWidget):
    tile_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buttons: list[list[QPushButton]] = []
        self._selected: tuple[int, int] | None = None

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(8)

        for r in range(BOARD_SIZE):
            row_buttons: list[QPushButton] = []
            for c in range(BOARD_SIZE):
                btn = QPushButton("?")
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                btn.clicked.connect(lambda _checked=False, row=r, col=c: self.tile_clicked.emit(row, col))
                self._grid.addWidget(btn, r, c)
                row_buttons.append(btn)
            self._buttons.append(row_buttons)

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 320)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_button_sizes()

    def _update_button_sizes(self) -> None:
        margins = self._grid.contentsMargins()
        spacing_total = self._grid.spacing() * (BOARD_SIZE - 1)
        usable_w = max(100, self.width() - margins.left() - margins.right() - spacing_total)
        usable_h = max(100, self.height() - margins.top() - margins.bottom() - spacing_total)
        side = max(46, min(140, min(usable_w, usable_h) // BOARD_SIZE))

        for row in self._buttons:
            for btn in row:
                btn.setFixedSize(side, side)

@staticmethod
    def _bomb_prob_color(p: float) -> str:
        """Interpolate green→yellow→red based on voltorb probability."""
        # Stop 0: #86efac (green-300), stop 0.5: #fde68a (amber-200), stop 1: #fca5a5 (red-300)
        if p <= 0.5:
            t = p * 2
            r = int(134 + 119 * t)
            g = int(239 - 9 * t)
            b = int(172 - 34 * t)
        else:
            t = (p - 0.5) * 2
            r = int(253 - t)
            g = int(230 - 65 * t)
            b = int(138 + 27 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def render(
        self,
        state: GameState,
        snapshot: SolverSnapshot,
        selected: tuple[int, int] | None = None,
        recommended: tuple[int, int] | None = None,
    ) -> None:
        self._selected = selected
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = self._buttons[r][c]
                tile = state.board[r][c]
                is_selected = self._selected == (r, c)
                is_recommended = recommended == (r, c) and not is_selected
                if is_selected:
                    border = "3px solid #0ea5a0"
                elif is_recommended:
                    border = "3px solid #22c55e"
                else:
                    border = "1px solid #b6ccdf"
                if tile.revealed and tile.value is not None:
                    btn.setText(str(tile.value))
                    if tile.value == 0:
                        btn.setStyleSheet(
                            f"background-color: #ef4444; color: white; font-weight: 700; font-size: 20px;"
                            f" border: {border}; border-radius: 12px;"
                        )
                    else:
                        btn.setStyleSheet(
                            f"background-color: #10b981; color: white; font-weight: 700; font-size: 20px;"
                            f" border: {border}; border-radius: 12px;"
                        )
                else:
                    bomb_p = snapshot.bomb_probabilities.get((r, c), 0.0)
                    bg = self._bomb_prob_color(bomb_p)
                    label = "★" if is_recommended else "?"
                    btn.setText(f"{label}\nB:{bomb_p:.0%}")
                    btn.setStyleSheet(
                        f"background-color: {bg}; color: #1f3a5f; font-weight: 600;"
                        f" border: {border}; border-radius: 12px;"
                    )
