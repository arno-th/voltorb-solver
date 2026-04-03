from __future__ import annotations

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

from voltorb_solver.advisor import MoveSuggestion
from voltorb_solver.solver import SolverSnapshot


class SolverPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.status = QLabel("Configurations: 0")
        self.status.setObjectName("solverStatus")
        self.errors = QLabel("")
        self.errors.setObjectName("solverErrors")

        self.safest_list = QListWidget()
        self.best_ev_list = QListWidget()

        layout.addWidget(self.status)
        safest_label = QLabel("Safest Moves")
        safest_label.setObjectName("solverSection")
        layout.addWidget(safest_label)
        layout.addWidget(self.safest_list)
        ev_label = QLabel("Best Expected Value")
        ev_label.setObjectName("solverSection")
        layout.addWidget(ev_label)
        layout.addWidget(self.best_ev_list)
        layout.addWidget(self.errors)

        self.setStyleSheet(
            """
            QLabel#solverStatus {
                color: #0f172a;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#solverSection {
                color: #1d4f91;
                font-weight: 700;
            }
            QLabel#solverErrors {
                color: #b91c1c;
                font-weight: 600;
            }
            """
        )

    def render(
        self,
        snapshot: SolverSnapshot,
        safest_moves: list[MoveSuggestion],
        best_ev_moves: list[MoveSuggestion],
    ) -> None:
        n_useful = len(snapshot.useful_positions)
        self.status.setText(
            f"Configurations: {snapshot.total_configurations} | Useful tiles: {n_useful}"
        )
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
