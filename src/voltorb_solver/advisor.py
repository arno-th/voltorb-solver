from __future__ import annotations

from dataclasses import dataclass

from voltorb_solver.game_state import GameState
from voltorb_solver.solver import SolverSnapshot


@dataclass(slots=True)
class MoveSuggestion:
    row: int
    col: int
    bomb_probability: float
    expected_value: float


def _expected_value_for_pos(snapshot: SolverSnapshot, pos: tuple[int, int], bomb_penalty: float) -> float:
    probs = snapshot.value_probabilities[pos]
    return (
        probs[0] * bomb_penalty
        + probs[1] * 1.0
        + probs[2] * 2.0
        + probs[3] * 3.0
    )


def suggest_moves(
    state: GameState,
    snapshot: SolverSnapshot,
    bomb_penalty: float = -1.0,
    top_n: int = 5,
) -> tuple[list[MoveSuggestion], list[MoveSuggestion]]:
    if snapshot.total_configurations == 0:
        return ([], [])

    candidates: list[MoveSuggestion] = []
    for r in range(5):
        for c in range(5):
            tile = state.board[r][c]
            if tile.revealed:
                continue
            pos = (r, c)
            candidates.append(
                MoveSuggestion(
                    row=r,
                    col=c,
                    bomb_probability=snapshot.bomb_probabilities[pos],
                    expected_value=_expected_value_for_pos(snapshot, pos, bomb_penalty),
                )
            )

    safest = sorted(
        candidates,
        key=lambda move: (move.bomb_probability, -move.expected_value, move.row, move.col),
    )[:top_n]
    best_ev = sorted(
        candidates,
        key=lambda move: (-move.expected_value, move.bomb_probability, move.row, move.col),
    )[:top_n]

    return safest, best_ev
