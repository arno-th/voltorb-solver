from voltorb_solver.game_state import GameState
from voltorb_solver.solver import solve_game_state


def test_solver_single_deterministic_board() -> None:
    state = GameState()

    # This setup forces every row and column to be exactly [0, 1, 1, 1, 1].
    for i in range(5):
        state.set_row_clue(i, voltorbs=1, total=4)
        state.set_col_clue(i, voltorbs=1, total=4)

    # Reveal diagonal as bombs to pin the unique board.
    for i in range(5):
        state.set_tile_revealed(i, i, 0)

    snapshot = solve_game_state(state)

    assert snapshot.errors == []
    assert snapshot.total_configurations == 1

    for i in range(5):
        assert snapshot.bomb_probabilities[(i, i)] == 1.0


def test_solver_detects_contradiction() -> None:
    state = GameState()
    # Impossible row clue: sum 15 with all bombs.
    state.set_row_clue(0, voltorbs=5, total=15)
    snapshot = solve_game_state(state)
    assert snapshot.total_configurations == 0
    assert snapshot.errors
