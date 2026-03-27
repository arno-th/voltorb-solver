from voltorb_solver.advisor import suggest_moves
from voltorb_solver.game_state import GameState
from voltorb_solver.solver import solve_game_state


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
