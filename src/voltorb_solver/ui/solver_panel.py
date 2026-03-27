from __future__ import annotations

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from voltorb_solver.advisor import MoveSuggestion
from voltorb_solver.solver import SolverSnapshot


class SolverPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.status = QLabel("Configurations: 0")
        self.errors = QLabel("")
        self.errors.setStyleSheet("color: #d84315;")

        self.safest_list = QListWidget()
        self.best_ev_list = QListWidget()

        layout.addWidget(self.status)
        layout.addWidget(QLabel("Safest Moves"))
        layout.addWidget(self.safest_list)
        layout.addWidget(QLabel("Best Expected Value"))
        layout.addWidget(self.best_ev_list)
        layout.addWidget(self.errors)

    def render(
        self,
        snapshot: SolverSnapshot,
        safest_moves: list[MoveSuggestion],
        best_ev_moves: list[MoveSuggestion],
    ) -> None:
        self.status.setText(f"Configurations: {snapshot.total_configurations}")
        self.errors.setText("; ".join(snapshot.errors))

        self.safest_list.clear()
        self.best_ev_list.clear()

        for move in safest_moves:
            self.safest_list.addItem(
                f"R{move.row + 1}C{move.col + 1} | B:{move.bomb_probability:.1%} | EV:{move.expected_value:.2f}"
            )

        for move in best_ev_moves:
            self.best_ev_list.addItem(
                f"R{move.row + 1}C{move.col + 1} | EV:{move.expected_value:.2f} | B:{move.bomb_probability:.1%}"
            )
