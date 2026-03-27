from pathlib import Path

from PIL import Image

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
