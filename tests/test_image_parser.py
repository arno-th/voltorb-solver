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


def test_tesseract_unavailable_warning_is_actionable(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    parser = ImageParser()

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.COLOR_RGB2BGR = 1
    fake_cv2.COLOR_BGR2GRAY = 2
    fake_cv2.THRESH_BINARY = 4
    fake_cv2.THRESH_OTSU = 8
    fake_cv2.cvtColor = lambda img, _code: img
    fake_cv2.GaussianBlur = lambda img, _ksize, _sigma: img
    fake_cv2.threshold = lambda img, _t, _m, _f: (0, img)

    fake_pytesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"),
        get_tesseract_version=lambda: (_ for _ in ()).throw(RuntimeError("missing")),
    )

    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser_module, "pytesseract", fake_pytesseract)

    result = parser.parse_image(str(image_path), (0, 0, 100, 100))

    assert any("set `TESSERACT_CMD`" in warning for warning in result.warnings)


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

    fake_pytesseract = types.SimpleNamespace()

    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser_module, "pytesseract", fake_pytesseract)
    monkeypatch.setattr(parser, "_configure_tesseract_runtime", lambda: True)
    monkeypatch.setattr(parser, "_parse_clue_rects", lambda _img, _rects: [(2, 10)])

    result = parser.parse_clue_box(str(image_path))

    assert result == (2, 10)


def test_parse_clue_box_returns_none_when_unreadable(monkeypatch) -> None:
    parser = ImageParser()

    fake_cv2 = types.SimpleNamespace()
    fake_pytesseract = types.SimpleNamespace()

    monkeypatch.setattr(parser_module, "cv2", fake_cv2)
    monkeypatch.setattr(parser_module, "pytesseract", fake_pytesseract)
    monkeypatch.setattr(parser, "_configure_tesseract_runtime", lambda: True)
    monkeypatch.setattr(parser, "_parse_clue_rects", lambda _img, _rects: [None])

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

    def fake_parse_clue_box(crop):
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
