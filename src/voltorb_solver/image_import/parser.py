from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image
import numpy as np

from voltorb_solver.game_state import BOARD_SIZE, Clue

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency behavior
    cv2 = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency behavior
    pytesseract = None


@dataclass(slots=True)
class ParseResult:
    row_clues: list[Clue] = field(default_factory=lambda: [Clue() for _ in range(BOARD_SIZE)])
    col_clues: list[Clue] = field(default_factory=lambda: [Clue() for _ in range(BOARD_SIZE)])
    revealed_tiles: dict[tuple[int, int], int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ImageParser:
    def parse_image(self, image_path: str, crop_rect: tuple[int, int, int, int]) -> ParseResult:
        result = ParseResult()
        img = Image.open(image_path)
        x, y, w, h = crop_rect
        cropped = img.crop((x, y, x + w, y + h))

        if cv2 is None or pytesseract is None:
            result.warnings.append(
                "OCR dependencies unavailable (opencv-python/pytesseract). Imported image unchanged."
            )
            return result

        cv_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        try:
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        except Exception:
            result.warnings.append(
                "Tesseract OCR engine is unavailable at runtime. Install `tesseract` or enter clues manually."
            )
            return result
        tokens = self._extract_numeric_tokens(data)

        if len(tokens) < 20:
            result.warnings.append(
                "Could not confidently read all clues from OCR. Please correct clues manually."
            )
            return result

        pairs = self._make_pairs(tokens)
        row_pairs = pairs[:BOARD_SIZE]
        col_pairs = pairs[BOARD_SIZE : BOARD_SIZE * 2]

        if len(col_pairs) < BOARD_SIZE:
            result.warnings.append("OCR returned too few clue pairs. Applied partial import only.")

        for idx, pair in enumerate(row_pairs):
            if idx >= BOARD_SIZE:
                break
            result.row_clues[idx] = Clue(voltorbs=pair[0], total=pair[1])

        for idx, pair in enumerate(col_pairs):
            if idx >= BOARD_SIZE:
                break
            result.col_clues[idx] = Clue(voltorbs=pair[0], total=pair[1])

        result.warnings.append(
            "OCR clue ordering is heuristic. Verify imported clues before trusting recommendations."
        )
        return result

    def _extract_numeric_tokens(self, data: dict) -> list[int]:
        tokens: list[tuple[int, int, int]] = []
        texts = data.get("text", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        confs = data.get("conf", [])

        count = min(len(texts), len(lefts), len(tops), len(confs))
        for i in range(count):
            text = str(texts[i]).strip()
            if not text:
                continue
            try:
                confidence = float(confs[i])
            except (ValueError, TypeError):
                confidence = -1
            if confidence < 30:
                continue
            for match in re.findall(r"\d+", text):
                value = int(match)
                if 0 <= value <= 15:
                    tokens.append((tops[i], lefts[i], value))

        tokens.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in tokens]

    def _make_pairs(self, values: list[int]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        usable = values[: BOARD_SIZE * 4]
        for i in range(0, len(usable), 2):
            if i + 1 >= len(usable):
                break
            a = usable[i]
            b = usable[i + 1]
            # Ensure pair is (voltorbs, sum).
            if a <= 5 and b <= 15:
                pairs.append((a, b))
            elif b <= 5 and a <= 15:
                pairs.append((b, a))
            else:
                # If OCR was noisy, clamp to valid ranges to avoid crashes and prompt manual correction.
                pairs.append((min(a, 5), min(b, 15)))
        return pairs
