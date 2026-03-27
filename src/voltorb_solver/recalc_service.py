from __future__ import annotations

from dataclasses import dataclass

from voltorb_solver.advisor import MoveSuggestion, suggest_moves
from voltorb_solver.game_state import GameState
from voltorb_solver.solver import SolverSnapshot, solve_game_state


@dataclass(slots=True)
class RecalcResult:
    snapshot: SolverSnapshot
    safest_moves: list[MoveSuggestion]
    best_ev_moves: list[MoveSuggestion]


class RecalculationService:
    def __init__(self, initial_state: GameState | None = None) -> None:
        self.state = initial_state or GameState()
        self._history: list[GameState] = []
        self.current = RecalcResult(
            snapshot=solve_game_state(self.state),
            safest_moves=[],
            best_ev_moves=[],
        )
        self.recalculate()

    def push_state(self) -> None:
        self._history.append(self.state.copy())

    def undo(self) -> bool:
        if not self._history:
            return False
        self.state = self._history.pop()
        self.recalculate()
        return True

    def recalculate(self) -> RecalcResult:
        snapshot = solve_game_state(self.state)
        safest, best_ev = suggest_moves(self.state, snapshot)
        self.current = RecalcResult(snapshot=snapshot, safest_moves=safest, best_ev_moves=best_ev)
        return self.current
