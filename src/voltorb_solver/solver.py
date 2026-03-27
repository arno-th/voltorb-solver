from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from voltorb_solver.game_state import BOARD_SIZE, GameState


@dataclass(slots=True)
class SolverSnapshot:
    total_configurations: int
    value_probabilities: dict[tuple[int, int], dict[int, float]]
    bomb_probabilities: dict[tuple[int, int], float]
    impossible_values: dict[tuple[int, int], set[int]]
    errors: list[str]


def _row_patterns(target_voltorbs: int, target_sum: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for row in product((0, 1, 2, 3), repeat=BOARD_SIZE):
        if row.count(0) == target_voltorbs and sum(row) == target_sum:
            patterns.append(row)
    return patterns


def solve_game_state(state: GameState) -> SolverSnapshot:
    for clue in state.row_clues + state.col_clues:
        if not clue.is_valid():
            return SolverSnapshot(0, {}, {}, {}, ["Invalid clue values."])

    revealed = state.revealed_tiles()
    row_options: list[list[tuple[int, ...]]] = []

    for r in range(BOARD_SIZE):
        base = _row_patterns(state.row_clues[r].voltorbs, state.row_clues[r].total)
        constrained: list[tuple[int, ...]] = []
        for row in base:
            ok = True
            for c in range(BOARD_SIZE):
                expected = revealed.get((r, c))
                if expected is not None and row[c] != expected:
                    ok = False
                    break
            if ok:
                constrained.append(row)
        if not constrained:
            return SolverSnapshot(0, {}, {}, {}, [f"No valid rows for row {r + 1}."])
        row_options.append(constrained)

    counts = {
        (r, c): {0: 0, 1: 0, 2: 0, 3: 0}
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
    }

    col_sum = [0] * BOARD_SIZE
    col_voltorbs = [0] * BOARD_SIZE
    total = 0

    board_values: list[tuple[int, ...] | None] = [None] * BOARD_SIZE

    def backtrack(row_idx: int) -> None:
        nonlocal total
        if row_idx == BOARD_SIZE:
            for c in range(BOARD_SIZE):
                if col_sum[c] != state.col_clues[c].total:
                    return
                if col_voltorbs[c] != state.col_clues[c].voltorbs:
                    return

            total += 1
            for r in range(BOARD_SIZE):
                row = board_values[r]
                assert row is not None
                for c in range(BOARD_SIZE):
                    counts[(r, c)][row[c]] += 1
            return

        for row in row_options[row_idx]:
            valid = True
            for c, value in enumerate(row):
                next_sum = col_sum[c] + value
                next_voltorbs = col_voltorbs[c] + (1 if value == 0 else 0)

                if next_sum > state.col_clues[c].total:
                    valid = False
                    break
                if next_voltorbs > state.col_clues[c].voltorbs:
                    valid = False
                    break

                remaining = BOARD_SIZE - row_idx - 1
                # Each remaining tile in column contributes at least 0 and at most 3.
                min_possible_sum = next_sum
                max_possible_sum = next_sum + remaining * 3
                if state.col_clues[c].total < min_possible_sum or state.col_clues[c].total > max_possible_sum:
                    valid = False
                    break

                min_possible_voltorbs = next_voltorbs
                max_possible_voltorbs = next_voltorbs + remaining
                target_voltorbs = state.col_clues[c].voltorbs
                if target_voltorbs < min_possible_voltorbs or target_voltorbs > max_possible_voltorbs:
                    valid = False
                    break

            if not valid:
                continue

            for c, value in enumerate(row):
                col_sum[c] += value
                if value == 0:
                    col_voltorbs[c] += 1

            board_values[row_idx] = row
            backtrack(row_idx + 1)
            board_values[row_idx] = None

            for c, value in enumerate(row):
                col_sum[c] -= value
                if value == 0:
                    col_voltorbs[c] -= 1

    backtrack(0)

    if total == 0:
        return SolverSnapshot(0, {}, {}, {}, ["No full-board configurations satisfy the clues."])

    value_probabilities: dict[tuple[int, int], dict[int, float]] = {}
    bomb_probabilities: dict[tuple[int, int], float] = {}
    impossible_values: dict[tuple[int, int], set[int]] = {}

    for pos, value_counts in counts.items():
        probs = {value: value_counts[value] / total for value in (0, 1, 2, 3)}
        value_probabilities[pos] = probs
        bomb_probabilities[pos] = probs[0]
        impossible_values[pos] = {value for value, p in probs.items() if p == 0.0}

    return SolverSnapshot(
        total_configurations=total,
        value_probabilities=value_probabilities,
        bomb_probabilities=bomb_probabilities,
        impossible_values=impossible_values,
        errors=[],
    )
