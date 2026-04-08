from voltorb_solver.advisor import suggest_moves
from voltorb_solver.game_state import GameState
from voltorb_solver.solver import SolverSnapshot, solve_game_state


def test_advisor_returns_ranked_moves() -> None:
    state = GameState()
    for i in range(5):
        state.set_row_clue(i, 1, 4)
        state.set_col_clue(i, 1, 4)

    state.set_tile_revealed(0, 0, 0)
    snapshot = solve_game_state(state)
    safest, best_ev = suggest_moves(state, snapshot)

    assert safest
    assert best_ev
    assert safest[0].bomb_probability <= safest[-1].bomb_probability
    assert best_ev[0].expected_value >= best_ev[-1].expected_value


def test_safest_prioritizes_guaranteed_safe_non_useful_tiles() -> None:
    state = GameState()

    value_probabilities: dict[tuple[int, int], dict[int, float]] = {}
    bomb_probabilities: dict[tuple[int, int], float] = {}

    for r in range(5):
        for c in range(5):
            pos = (r, c)
            value_probabilities[pos] = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
            bomb_probabilities[pos] = 1.0

    # Guaranteed-safe but non-useful (always a 1).
    value_probabilities[(0, 0)] = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0}
    bomb_probabilities[(0, 0)] = 0.0

    # Risky but useful (can be a 2).
    value_probabilities[(0, 1)] = {0: 0.10, 1: 0.0, 2: 0.90, 3: 0.0}
    bomb_probabilities[(0, 1)] = 0.10

    snapshot = SolverSnapshot(
        total_configurations=10,
        value_probabilities=value_probabilities,
        bomb_probabilities=bomb_probabilities,
        impossible_values={},
        useful_positions={(0, 1)},
        errors=[],
    )

    safest, best_ev = suggest_moves(state, snapshot)

    assert safest[0].row == 0 and safest[0].col == 0
    assert best_ev[0].row == 0 and best_ev[0].col == 1
