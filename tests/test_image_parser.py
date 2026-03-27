from pathlib import Path
import types

from PIL import Image

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
