from __future__ import annotations

from dataclasses import dataclass

from voltorb_solver.game_state import GameState
from voltorb_solver.solver import SolverSnapshot


SAFE_BOMB_PROBABILITY_EPSILON = 1e-12


@dataclass(slots=True)
class MoveSuggestion:
    row: int
    col: int
    bomb_probability: float
    expected_value: float
    is_useful: bool  # P(2) > 0 or P(3) > 0 — worth pressing


def _expected_value_for_pos(snapshot: SolverSnapshot, pos: tuple[int, int], bomb_penalty: float) -> float:
    probs = snapshot.value_probabilities[pos]
    return (
        probs[0] * bomb_penalty
        + probs[1] * 1.0
        + probs[2] * 2.0
        + probs[3] * 3.0
    )


def _is_guaranteed_safe(bomb_probability: float) -> bool:
    return bomb_probability <= SAFE_BOMB_PROBABILITY_EPSILON


def _safest_sort_key(move: MoveSuggestion) -> tuple[int, float, float, int, int]:
    # Priority: guaranteed-safe tiles first (including safe 1s), then risky useful,
    # then risky non-useful. Within each bucket prefer lower bomb risk then EV.
    if _is_guaranteed_safe(move.bomb_probability):
        bucket = 0
    elif move.is_useful:
        bucket = 1
    else:
        bucket = 2
    return (bucket, move.bomb_probability, -move.expected_value, move.row, move.col)


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
                    is_useful=pos in snapshot.useful_positions,
                )
            )

    useful = [c for c in candidates if c.is_useful]

    # For safety-first play, show guaranteed-safe tiles first even when they are
    # definitely 1s, so the solver can clear free clicks before taking risk.
    safest = sorted(candidates, key=_safest_sort_key)[:top_n]
    best_ev = sorted(
        useful if useful else candidates,
        key=lambda move: (-move.expected_value, move.bomb_probability, move.row, move.col),
    )[:top_n]

    return safest, best_ev
