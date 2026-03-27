from voltorb_solver.recalc_service import RecalculationService


def test_recalculate_after_state_update() -> None:
    service = RecalculationService()
    for i in range(5):
        service.state.set_row_clue(i, 1, 4)
        service.state.set_col_clue(i, 1, 4)
    service.recalculate()
    initial_configs = service.current.snapshot.total_configurations

    service.push_state()
    service.state.set_tile_revealed(0, 0, 0)
    service.recalculate()

    assert service.current.snapshot.total_configurations != initial_configs


def test_undo_restores_previous_state() -> None:
    service = RecalculationService()
    service.push_state()
    service.state.set_row_clue(0, 2, 8)
    service.recalculate()

    assert service.undo() is True
    assert service.state.row_clues[0].voltorbs == 0
    assert service.state.row_clues[0].total == 0
