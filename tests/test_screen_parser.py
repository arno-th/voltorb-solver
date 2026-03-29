from pathlib import Path

import numpy as np

from voltorb_solver.image_import.screen_parser import Region, ScreenBoardParser


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
    assert r0.y <= top_left.y + max(6, top_left.h // 2)

    # Column clues are below the board columns.
    assert c0.y >= bottom_left.y + bottom_left.h
    assert c0.x <= top_left.x
    assert c4.x + c4.w >= top_right.x + max(6, top_right.w // 2)

    # Clue bands remain aligned with the tile grid index for both axes.
    for idx in range(5):
        row_tile = by_name[f"({idx},0)"]
        row_clue = by_name[f"r{idx}"]
        row_tile_cy = row_tile.y + row_tile.h / 2.0
        row_clue_cy = row_clue.y + row_clue.h / 2.0
        assert abs(row_clue_cy - row_tile_cy) <= max(20.0, row_tile.h * 0.55)

        col_tile = by_name[f"(0,{idx})"]
        col_clue = by_name[f"c{idx}"]
        col_tile_cx = col_tile.x + col_tile.w / 2.0
        col_clue_cx = col_clue.x + col_clue.w / 2.0
        assert abs(col_clue_cx - col_tile_cx) <= max(20.0, col_tile.w * 0.55)


def test_screen_parser_annotation_writes_output(tmp_path: Path) -> None:
    sample = _sample_image()
    assert sample.exists()

    output = tmp_path / "labeled.png"
    parser = ScreenBoardParser()
    result = parser.annotate(str(sample), str(output))

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(result.regions) == 35


def test_screen_parser_rejects_off_grid_template_refinement(monkeypatch) -> None:
    parser = ScreenBoardParser()

    board = (500, 500, 500, 500)
    parser._grid_layout = {
        "x_centers": [550.0, 650.0, 750.0, 850.0, 950.0],
        "y_centers": [550.0, 650.0, 750.0, 850.0, 950.0],
        "tile_side": 90,
    }
    regions = parser._build_regions(1600, 1600, (0, 0, 1600, 1600), board)
    image = np.zeros((1600, 1600, 3), dtype=np.uint8)

    monkeypatch.setattr(parser, "_load_clue_templates", lambda: [np.ones((20, 20), dtype=np.uint8)])

    def _fake_match(
        _gray_image: np.ndarray,
        expected_regions: list[Region],
        _templates: list[np.ndarray],
        *,
        axis: str,
    ) -> list[Region]:
        if axis == "row":
            return [
                Region(name=f"r{idx}", x=region.x - 200, y=region.y, w=region.w, h=region.h)
                for idx, region in enumerate(expected_regions)
            ]
        return [
            Region(name=f"c{idx}", x=region.x, y=region.y - 200, w=region.w, h=region.h)
            for idx, region in enumerate(expected_regions)
        ]

    monkeypatch.setattr(parser, "_match_clue_regions", _fake_match)

    warnings: list[str] = []
    refined, row_method, col_method = parser._refine_clue_regions_with_templates(
        image,
        regions,
        board,
        warnings,
    )

    assert row_method == "heuristic-clue"
    assert col_method == "heuristic-clue"
    assert any("Rejected template row clue matches" in warning for warning in warnings)
    assert any("Rejected template column clue matches" in warning for warning in warnings)

    original_by_name = {region.name: region for region in regions}
    refined_by_name = {region.name: region for region in refined}
    for idx in range(5):
        row_name = f"r{idx}"
        col_name = f"c{idx}"
        assert refined_by_name[row_name] == original_by_name[row_name]
        assert refined_by_name[col_name] == original_by_name[col_name]
