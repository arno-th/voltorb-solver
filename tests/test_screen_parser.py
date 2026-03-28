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

    expected_board = {f"({r},{c})" for r in range(5) for c in range(5)}
    expected_rows = {f"r{r}" for r in range(5)}
    expected_cols = {f"c{c}" for c in range(5)}
    expected_names = expected_board | expected_rows | expected_cols

    assert len(names) == 35
    assert names == expected_names

    by_name = result.by_name()

    top_left = by_name["(0,0)"]
    top_right = by_name["(0,4)"]
    bottom_left = by_name["(4,0)"]
    r0 = by_name["r0"]
    r4 = by_name["r4"]
    c0 = by_name["c0"]
    c4 = by_name["c4"]

    assert top_left.w > 10
    assert top_left.h > 10
    assert r0.h > 10
    assert c0.w > 10

    # Row clues are to the right of the board rows.
    assert r0.x >= top_right.x + top_right.w
    assert r0.y <= top_left.y
    assert r4.y + r4.h >= bottom_left.y + bottom_left.h

    # Column clues are below the board columns.
    assert c0.y >= bottom_left.y + bottom_left.h
    assert c0.x <= top_left.x
    assert c4.x + c4.w >= top_right.x + top_right.w


def test_screen_parser_annotation_writes_output(tmp_path: Path) -> None:
    sample = _sample_image()
    assert sample.exists()

    output = tmp_path / "labeled.png"
    parser = ScreenBoardParser()
    result = parser.annotate(str(sample), str(output))

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(result.regions) == 35
