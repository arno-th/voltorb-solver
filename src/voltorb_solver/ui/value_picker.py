from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ValuePickerDialog(QDialog):
    def __init__(self, impossible_values: set[int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Tile Value")
        self.selected_value: int | None = None

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Choose the revealed value for this tile:"))

        grid = QGridLayout()
        root.addLayout(grid)

        for idx, value in enumerate((0, 1, 2, 3)):
            btn = QPushButton(str(value))
            btn.setEnabled(value not in impossible_values)
            btn.clicked.connect(lambda _checked=False, v=value: self._pick(v))
            grid.addWidget(btn, idx // 2, idx % 2)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        hide_button = QPushButton("Set Hidden")
        hide_button.clicked.connect(self._pick_hidden)
        button_box.addButton(hide_button, QDialogButtonBox.ButtonRole.ActionRole)
        button_box.rejected.connect(self.reject)

        root.addWidget(button_box)

    def _pick(self, value: int) -> None:
        self.selected_value = value
        self.accept()

    def _pick_hidden(self) -> None:
        self.selected_value = None
        self.accept()
