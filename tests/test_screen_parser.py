from pathlib import Path

from voltorb_solver.image_import.screen_parser import ScreenBoardParser


def _sample_image() -> Path:
    return Path(__file__).resolve().parent.parent / "GameBoard.png"


def test_screen_parser_detects_expected_regions() -> None:
    sample = _sample_image()
    assert sample.exists()

    parser = ScreenBoardParser()
    result = parser.parse(str(sample))
    names = {region.name for region in result.regions}

    assert "game_panel" in names
    assert "board_grid" in names
    assert "row_clues" in names
    assert "col_clues" in names

    by_name = result.by_name()
    panel = by_name["game_panel"]
    board = by_name["board_grid"]

    assert panel.w > 100
    assert panel.h > 100
    assert board.w > 80
    assert board.h > 80

    # Board should be inside game panel bounds.
    assert panel.x <= board.x <= panel.x + panel.w
    assert panel.y <= board.y <= panel.y + panel.h
    assert board.x + board.w <= panel.x + panel.w
    assert board.y + board.h <= panel.y + panel.h


def test_screen_parser_annotation_writes_output(tmp_path: Path) -> None:
    sample = _sample_image()
    assert sample.exists()

    output = tmp_path / "labeled.png"
    parser = ScreenBoardParser()
    result = parser.annotate(str(sample), str(output))

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(result.regions) >= 4
