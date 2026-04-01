from pathlib import Path
import types

from PIL import Image
import numpy as np
import pytest

import voltorb_solver.image_import.parser as parser_module
from voltorb_solver.image_import.parser import ImageParser


def test_image_parser_graceful_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 200), color="white").save(image_path)

    parser = ImageParser()
    result = parser.parse_image(str(image_path), (0, 0, 200, 200))

    assert len(result.row_clues) == 5
    assert len(result.col_clues) == 5
    # In environments with OCR libs installed, warnings can still be present due to weak content.
    assert isinstance(result.warnings, list)


def test_parse_image_warns_when_cv2_unavailable(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    parser = ImageParser()

    monkeypatch.setattr(parser_module, "cv2", None)

    result = parser.parse_image(str(image_path), (0, 0, 100, 100))

    assert any("opencv-python" in warning for warning in result.warnings)


def test_parse_known_voltorb_screenshot_when_ocr_available() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "GameBoard.png"
    if not sample_path.exists():
        pytest.skip("GameBoard.png sample image not available in repository")

    parser = ImageParser()
    width, height = Image.open(sample_path).size
    result = parser.parse_image(str(sample_path), (0, 0, width, height))

    if any("Tesseract OCR engine is unavailable" in warning for warning in result.warnings):
        pytest.skip("Tesseract runtime not available for OCR regression test")

    parsed_rows = [(clue.voltorbs, clue.total) for clue in result.row_clues]
    parsed_cols = [(clue.voltorbs, clue.total) for clue in result.col_clues]

    assert len(parsed_rows) == 5
    assert len(parsed_cols) == 5
    assert all(0 <= bombs <= 5 and 0 <= total <= 15 for bombs, total in parsed_rows)
    assert all(0 <= bombs <= 5 and 0 <= total <= 15 for bombs, total in parsed_cols)


def test_parse_clue_box_returns_pair_with_stubbed_parser(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "clue.png"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    parser = ImageParser()

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.COLOR_RGB2BGR = 1
    fake_cv2.cvtColor = lambda img, _code: img

    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser, "_ocr_number_field", lambda *_args, **_kwargs: 2 if _kwargs["field_name"] == "voltorbs" else 8)

    result = parser.parse_clue_box(str(image_path))

    assert result == (2, 8)


def test_parse_clue_box_returns_none_when_unreadable(monkeypatch) -> None:
    parser = ImageParser()

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.COLOR_BGR2GRAY = 2
    fake_cv2.cvtColor = lambda img, _code: img
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser, "_ocr_number_field", lambda *_args, **_kwargs: None)

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = parser.parse_clue_box(arr)

    assert result is None


def test_parse_clue_from_screenshot_crops_and_parses(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (120, 80), color="white").save(image_path)

    parser = ImageParser()
    fake_cv2 = types.SimpleNamespace()
    fake_cv2.imread = lambda _path: np.zeros((80, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)

    captured_shapes: list[tuple[int, int]] = []

    def fake_parse_clue_box(crop, *, fast=True):
        captured_shapes.append((crop.shape[1], crop.shape[0]))
        return (1, 7)

    monkeypatch.setattr(parser, "parse_clue_box", fake_parse_clue_box)

    result = parser.parse_clue_from_screenshot(str(image_path), (10, 20, 30, 25))

    assert result == (1, 7)
    assert captured_shapes == [(30, 25)]


def test_parse_clue_from_screenshot_rejects_invalid_box(tmp_path: Path) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (50, 50), color="white").save(image_path)

    parser = ImageParser()
    result = parser.parse_clue_from_screenshot(str(image_path), (0, 0, 0, 10))

    assert result is None


def test_extract_clue_crop_returns_expected_shape(monkeypatch) -> None:
    parser = ImageParser()

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.imread = lambda _path: np.zeros((60, 90, 3), dtype=np.uint8)
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)

    crop = parser.extract_clue_crop("dummy.png", (10, 20, 15, 12))

    assert crop is not None
    assert crop.shape[:2] == (12, 15)


def test_save_clue_crop_writes_image(monkeypatch, tmp_path: Path) -> None:
    parser = ImageParser()
    written: list[tuple[str, tuple[int, ...]]] = []

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.imread = lambda _path: np.zeros((40, 50, 3), dtype=np.uint8)

    def fake_imwrite(path: str, img: np.ndarray) -> bool:
        written.append((path, img.shape))
        return True

    fake_cv2.imwrite = fake_imwrite
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)

    out_path = tmp_path / "out" / "crop.png"
    ok = parser.save_clue_crop("dummy.png", (5, 6, 10, 8), out_path)

    assert ok is True
    assert len(written) == 1
    assert written[0][0] == str(out_path)
    assert written[0][1][:2] == (8, 10)


def test_split_clue_fields_returns_stable_regions() -> None:
    parser = ImageParser()
    crop = np.zeros((100, 120, 3), dtype=np.uint8)

    split = parser.split_clue_fields(crop)

    assert split is not None
    voltorbs_roi, total_roi = split
    assert voltorbs_roi.shape[:2] == (48, 45)
    assert total_roi.shape[:2] == (40, 83)


def test_debug_parse_clue_from_screenshot_writes_artifacts_and_log(monkeypatch, tmp_path: Path) -> None:
    parser = ImageParser()

    written: list[str] = []

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.COLOR_BGR2GRAY = 1
    fake_cv2.THRESH_BINARY_INV = 4
    fake_cv2.THRESH_OTSU = 8
    fake_cv2.imread = lambda _path: np.zeros((80, 100, 3), dtype=np.uint8)
    fake_cv2.cvtColor = lambda img, _code: img[:, :, 0]
    fake_cv2.threshold = lambda img, _a, _b, _c: (0, img)
    fake_cv2.imwrite = lambda path, _img: written.append(path) or True

    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser, "_match_number_template", lambda *_args, **_kwargs: (2, 0.95))

    artifacts = parser.debug_parse_clue_from_screenshot(
        "dummy.png",
        (10, 20, 30, 25),
        output_root=tmp_path / "debug_parse",
        region_name="r2",
        run_id="batch_001",
    )

    assert artifacts is not None
    assert len(written) >= 2
    assert artifacts.voltorbs_value == 2
    assert artifacts.total_value == 2
    assert artifacts.log_path.exists()
    assert artifacts.log_path.parent.name == "r2"
    assert artifacts.log_path.parent.parent.name == "batch_001"
    parent_dirs = {
        artifacts.raw_voltorbs_path.parent,
        artifacts.raw_total_path.parent,
        artifacts.normalized_voltorbs_path.parent,
        artifacts.normalized_total_path.parent,
        artifacts.log_path.parent,
    }
    assert len(parent_dirs) == 1

    log_text = artifacts.log_path.read_text(encoding="utf-8")
    assert "region=r2" in log_text
    assert "voltorbs_text='match:2 score=0.950'" in log_text
    assert "total_text='match:2 score=0.950'" in log_text


def test_save_unmatched_template_sample_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    parser = ImageParser()

    monkeypatch.setattr(parser, "_project_root", lambda: tmp_path)
    roi = np.zeros((12, 12, 3), dtype=np.uint8)
    roi[3:9, 5:7] = 255

    parser._save_unmatched_template_sample(roi, kind="top", source_tag="unit")

    out_dir = tmp_path / "assets/templates/raw/clue_unknown"
    assert out_dir.exists()
    manifest = out_dir / "manifest.csv"
    assert manifest.exists()
    rows = manifest.read_text(encoding="utf-8")
    assert "kind,path,source" in rows
    assert ",top," in rows


# ---------------------------------------------------------------------------
# Tile state classification
# ---------------------------------------------------------------------------


def _make_tile_cv2(match_scores: list[float]):
    """Return a fake cv2 namespace that yields successive match scores from the list."""
    call_idx = [0]

    def fake_matchTemplate(img, tmpl, method):
        score = match_scores[min(call_idx[0], len(match_scores) - 1)]
        call_idx[0] += 1
        return np.array([[score]], dtype=np.float32)

    fake = types.SimpleNamespace()
    fake.COLOR_BGR2GRAY = 6
    fake.INTER_AREA = 3
    fake.TM_CCOEFF_NORMED = 5
    fake.cvtColor = lambda img, code: img[:, :, 0] if img.ndim == 3 else img
    fake.resize = lambda img, size, **kw: np.full((size[1], size[0]), 64, dtype=np.uint8)
    fake.matchTemplate = fake_matchTemplate
    fake.minMaxLoc = lambda result: (0.0, float(result[0, 0]), (0, 0), (0, 0))
    return fake


def _preloaded_parser(closed_templates, state_bank):
    """Return an ImageParser with tile template banks pre-populated (skip disk loading)."""
    parser = ImageParser()
    parser._closed_tile_templates_loaded = True
    parser._tile_state_templates_loaded = True
    parser._closed_tile_templates = list(closed_templates)
    parser._tile_state_bank = dict(state_bank)
    return parser


def test_parse_tile_state_returns_none_with_empty_closed_bank(monkeypatch) -> None:
    parser = _preloaded_parser([], {})
    fake_cv2 = types.SimpleNamespace()
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)

    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    result = parser.parse_tile_state(crop)

    assert result is None


def test_parse_tile_state_detects_closed_tile(monkeypatch) -> None:
    dummy_tpl = np.full((64, 64), 64, dtype=np.uint8)
    parser = _preloaded_parser([dummy_tpl], {})
    # Single match call at high score → closed tile → None
    monkeypatch.setattr(parser_module, "cv2", _make_tile_cv2([1.0]))

    crop = np.full((80, 80, 3), 64, dtype=np.uint8)
    result = parser.parse_tile_state(crop)

    assert result is None  # unopen tile


def test_parse_tile_state_matches_value(monkeypatch) -> None:
    dummy_tpl = np.full((64, 64), 64, dtype=np.uint8)
    parser = _preloaded_parser([dummy_tpl], {2: [dummy_tpl]})
    # First call (closed check) → low score; second call (value match) → high score
    monkeypatch.setattr(parser_module, "cv2", _make_tile_cv2([0.2, 0.92]))

    crop = np.full((80, 80, 3), 64, dtype=np.uint8)
    result = parser.parse_tile_state(crop)

    assert result == 2


def test_parse_tile_state_saves_unknown_when_open_and_no_match(monkeypatch, tmp_path: Path) -> None:
    dummy_tpl = np.full((64, 64), 64, dtype=np.uint8)
    parser = _preloaded_parser([dummy_tpl], {1: [dummy_tpl]})
    monkeypatch.setattr(parser, "_project_root", lambda: tmp_path)

    saved: list[str] = []
    fake = _make_tile_cv2([0.2, 0.2])  # all match attempts fail below threshold
    fake.imwrite = lambda path, img: saved.append(path) or True
    monkeypatch.setattr(parser_module, "cv2", fake)

    crop = np.full((64, 64, 3), 64, dtype=np.uint8)
    result = parser.parse_tile_state(crop, region_name="(2,3)", image_path="test.png")

    assert result is None
    assert len(saved) > 0
    out_dir = tmp_path / "assets/parser_debug/tile_dataset/unknown"
    assert out_dir.exists()


def test_parse_tile_from_screenshot_crops_and_delegates(monkeypatch) -> None:
    parser = ImageParser()

    captured: list[tuple[int, ...]] = []

    def fake_parse_tile_state(crop, **kwargs):
        captured.append(crop.shape)
        return 3

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.imread = lambda _path: np.zeros((100, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser, "parse_tile_state", fake_parse_tile_state)

    result = parser.parse_tile_from_screenshot("dummy.png", (10, 20, 30, 25))

    assert result == 3
    assert len(captured) == 1
    assert captured[0][:2] == (25, 30)  # numpy shape (h, w)
