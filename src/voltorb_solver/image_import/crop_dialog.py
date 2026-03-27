from __future__ import annotations

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class CropDialog(QDialog):
    def __init__(self, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self._pixmap = QPixmap(image_path)

        layout = QVBoxLayout(self)
        preview = QLabel()
        preview.setPixmap(self._pixmap.scaledToWidth(460))
        preview.setToolTip("Use rectangle values below to crop the relevant board area.")
        layout.addWidget(preview)

        form = QFormLayout()
        self.x = QSpinBox()
        self.y = QSpinBox()
        self.w = QSpinBox()
        self.h = QSpinBox()

        width = max(1, self._pixmap.width())
        height = max(1, self._pixmap.height())

        self.x.setRange(0, width - 1)
        self.y.setRange(0, height - 1)
        self.w.setRange(1, width)
        self.h.setRange(1, height)
        self.w.setValue(width)
        self.h.setValue(height)

        form.addRow("X", self.x)
        form.addRow("Y", self.y)
        form.addRow("Width", self.w)
        form.addRow("Height", self.h)
        layout.addLayout(form)

        self.x.valueChanged.connect(self._clamp_size)
        self.y.valueChanged.connect(self._clamp_size)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _clamp_size(self) -> None:
        max_w = max(1, self._pixmap.width() - self.x.value())
        max_h = max(1, self._pixmap.height() - self.y.value())
        self.w.setMaximum(max_w)
        self.h.setMaximum(max_h)
        if self.w.value() > max_w:
            self.w.setValue(max_w)
        if self.h.value() > max_h:
            self.h.setValue(max_h)

    def get_rect(self) -> tuple[int, int, int, int]:
        return self.x.value(), self.y.value(), self.w.value(), self.h.value()
