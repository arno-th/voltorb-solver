from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voltorb_solver.stats import RoundCounts


class _WinRateBar(QWidget):
    """Horizontal stacked bar: green = wins, red = unlucky bombs, purple = miscalc bombs."""

    _WIN_COLOR = QColor("#22c55e")
    _UNLUCKY_COLOR = QColor("#ef4444")
    _MISCALC_COLOR = QColor("#7c3aed")
    _EMPTY_COLOR = QColor("#e2e8f0")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wins = 0
        self._unlucky = 0
        self._miscalc = 0
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, wins: int, unlucky: int, miscalc: int) -> None:
        self._wins = wins
        self._unlucky = unlucky
        self._miscalc = miscalc
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        w, h = r.width(), r.height()
        radius = h // 2
        total = self._wins + self._unlucky + self._miscalc

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._EMPTY_COLOR)
        p.drawRoundedRect(r, radius, radius)

        if total == 0:
            return

        win_w = max(0, min(w, round(w * self._wins / total)))
        unlucky_w = max(0, min(w - win_w, round(w * self._unlucky / total)))
        miscalc_w = w - win_w - unlucky_w

        if win_w > 0:
            p.setBrush(self._WIN_COLOR)
            p.setClipRect(0, 0, win_w, h)
            p.drawRoundedRect(r, radius, radius)
            p.setClipping(False)

        if unlucky_w > 0:
            p.setBrush(self._UNLUCKY_COLOR)
            p.setClipRect(win_w, 0, unlucky_w, h)
            p.drawRoundedRect(r, radius, radius)
            p.setClipping(False)

        if miscalc_w > 0:
            p.setBrush(self._MISCALC_COLOR)
            p.setClipRect(win_w + unlucky_w, 0, miscalc_w, h)
            p.drawRoundedRect(r, radius, radius)
            p.setClipping(False)


def _key_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet("color: #475569; font-size: 12px;")
    return lbl


def _val_label(color: str = "#1f2a37") -> QLabel:
    lbl = QLabel("—")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 12px;")
    return lbl


class _StatsColumn(QWidget):
    """Compact stats column for one scope (lifetime or session)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel(title)
        header.setObjectName("statsColHeader")
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)

        grid.addWidget(_key_label("Rounds"), 0, 0)
        self._rounds = _val_label()
        grid.addWidget(self._rounds, 0, 1)

        grid.addWidget(_key_label("Wins"), 1, 0)
        self._wins = _val_label("#16a34a")
        grid.addWidget(self._wins, 1, 1)

        grid.addWidget(_key_label("Bombs"), 2, 0)
        self._bombs = _val_label("#dc2626")
        grid.addWidget(self._bombs, 2, 1)

        grid.addWidget(_key_label("  Unlucky"), 3, 0)
        self._unlucky = _val_label("#f97316")
        grid.addWidget(self._unlucky, 3, 1)

        grid.addWidget(_key_label("  Miscalc"), 4, 0)
        self._miscalc = _val_label("#7c3aed")
        grid.addWidget(self._miscalc, 4, 1)

        grid.addWidget(_key_label("Win Rate"), 5, 0)
        self._rate = _val_label()
        grid.addWidget(self._rate, 5, 1)

        layout.addLayout(grid)

    def refresh(self, counts: RoundCounts) -> None:
        self._rounds.setText(str(counts.rounds_played))
        self._wins.setText(str(counts.wins))
        self._bombs.setText(str(counts.bombs_hit))
        self._unlucky.setText(str(counts.unlucky_bombs))
        self._miscalc.setText(str(counts.miscalc_bombs))
        self._rate.setText(f"{counts.win_rate:.1%}" if counts.rounds_played else "—")


class StatsPanel(QWidget):
    """Statistics panel showing lifetime and session win/loss data with bar charts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # ── Two stat columns side by side ──────────────────────────────────
        cols_row = QHBoxLayout()
        cols_row.setSpacing(0)

        self._lifetime_col = _StatsColumn("Lifetime")
        cols_row.addWidget(self._lifetime_col, 1)

        divider = QWidget()
        divider.setFixedWidth(1)
        divider.setStyleSheet("background: #d4e2ee;")
        divider.setContentsMargins(0, 0, 0, 0)
        cols_row.addSpacing(12)
        cols_row.addWidget(divider)
        cols_row.addSpacing(12)

        self._session_col = _StatsColumn("This Session")
        cols_row.addWidget(self._session_col, 1)

        outer.addLayout(cols_row)

        # ── Win-rate bar chart ─────────────────────────────────────────────
        bars = QGridLayout()
        bars.setHorizontalSpacing(8)
        bars.setVerticalSpacing(4)
        bars.setColumnStretch(1, 1)

        bars.addWidget(_key_label("Lifetime"), 0, 0)
        self._lifetime_bar = _WinRateBar()
        bars.addWidget(self._lifetime_bar, 0, 1)
        self._lifetime_pct = QLabel("—")
        self._lifetime_pct.setFixedWidth(40)
        self._lifetime_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lifetime_pct.setStyleSheet("font-size: 11px; color: #475569;")
        bars.addWidget(self._lifetime_pct, 0, 2)

        bars.addWidget(_key_label("Session"), 1, 0)
        self._session_bar = _WinRateBar()
        bars.addWidget(self._session_bar, 1, 1)
        self._session_pct = QLabel("—")
        self._session_pct.setFixedWidth(40)
        self._session_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._session_pct.setStyleSheet("font-size: 11px; color: #475569;")
        bars.addWidget(self._session_pct, 1, 2)

        outer.addLayout(bars)

        self.setStyleSheet(
            """
            QLabel#statsColHeader {
                font-weight: 700;
                color: #1d4f91;
                font-size: 13px;
            }
            """
        )

    def refresh(self, lifetime: RoundCounts, session: RoundCounts) -> None:
        self._lifetime_col.refresh(lifetime)
        self._session_col.refresh(session)

        self._lifetime_bar.set_data(lifetime.wins, lifetime.unlucky_bombs, lifetime.miscalc_bombs)
        self._session_bar.set_data(session.wins, session.unlucky_bombs, session.miscalc_bombs)

        self._lifetime_pct.setText(
            f"{lifetime.win_rate:.1%}" if lifetime.rounds_played else "—"
        )
        self._session_pct.setText(
            f"{session.win_rate:.1%}" if session.rounds_played else "—"
        )
