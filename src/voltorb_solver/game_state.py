from __future__ import annotations

from dataclasses import dataclass, field

BOARD_SIZE = 5


@dataclass(slots=True)
class Clue:
    voltorbs: int = 0
    total: int = 0

    def is_valid(self) -> bool:
        return 0 <= self.voltorbs <= BOARD_SIZE and 0 <= self.total <= 15


@dataclass(slots=True)
class Tile:
    revealed: bool = False
    value: int | None = None

    def set_hidden(self) -> None:
        self.revealed = False
        self.value = None

    def set_revealed(self, value: int) -> None:
        if value not in (0, 1, 2, 3):
            raise ValueError("Tile value must be 0, 1, 2, or 3.")
        self.revealed = True
        self.value = value


@dataclass(slots=True)
class GameState:
    row_clues: list[Clue] = field(default_factory=lambda: [Clue() for _ in range(BOARD_SIZE)])
    col_clues: list[Clue] = field(default_factory=lambda: [Clue() for _ in range(BOARD_SIZE)])
    board: list[list[Tile]] = field(
        default_factory=lambda: [[Tile() for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    )

    def copy(self) -> GameState:
        copied = GameState()
        copied.row_clues = [Clue(c.voltorbs, c.total) for c in self.row_clues]
        copied.col_clues = [Clue(c.voltorbs, c.total) for c in self.col_clues]
        copied.board = [
            [Tile(revealed=tile.revealed, value=tile.value) for tile in row] for row in self.board
        ]
        return copied

    def reset(self) -> None:
        for r in range(BOARD_SIZE):
            self.row_clues[r] = Clue()
            self.col_clues[r] = Clue()
            for c in range(BOARD_SIZE):
                self.board[r][c].set_hidden()

    def set_row_clue(self, row: int, voltorbs: int, total: int) -> None:
        self._validate_index(row)
        clue = Clue(voltorbs=voltorbs, total=total)
        if not clue.is_valid():
            raise ValueError("Invalid row clue.")
        self.row_clues[row] = clue

    def set_col_clue(self, col: int, voltorbs: int, total: int) -> None:
        self._validate_index(col)
        clue = Clue(voltorbs=voltorbs, total=total)
        if not clue.is_valid():
            raise ValueError("Invalid column clue.")
        self.col_clues[col] = clue

    def set_tile_revealed(self, row: int, col: int, value: int) -> None:
        self._validate_pos(row, col)
        self.board[row][col].set_revealed(value)

    def set_tile_hidden(self, row: int, col: int) -> None:
        self._validate_pos(row, col)
        self.board[row][col].set_hidden()

    def revealed_tiles(self) -> dict[tuple[int, int], int]:
        result: dict[tuple[int, int], int] = {}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tile = self.board[r][c]
                if tile.revealed and tile.value is not None:
                    result[(r, c)] = tile.value
        return result

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= BOARD_SIZE:
            raise IndexError("Index out of bounds.")

    def _validate_pos(self, row: int, col: int) -> None:
        self._validate_index(row)
        self._validate_index(col)
