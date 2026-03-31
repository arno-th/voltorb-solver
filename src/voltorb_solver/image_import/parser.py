from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import statistics
from datetime import datetime
from collections import Counter
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


@dataclass(slots=True)
class ClueDebugArtifacts:
    raw_voltorbs_path: Path
    raw_total_path: Path
    preprocessed_voltorbs_path: Path
    preprocessed_total_path: Path
    log_path: Path
    voltorbs_text: str
    total_text: str
    voltorbs_value: int | None
    total_value: int | None


class ImageParser:
    # Normalized clue-field bounds (x0, y0, x1, y1) inside a single clue box.
    _TOTAL_TOKEN_BOUNDS = (0.45, 0.02, 0.95, 0.42)
    _VOLTORB_TOKEN_BOUNDS = (0.40, 0.50, 0.98, 0.98)
    _TOTAL_OCR_BOUNDS = (0.10, 0.02, 0.95, 0.42)
    _VOLTORB_OCR_BOUNDS = (0.58, 0.50, 0.98, 0.98)

    def _clue_ocr_backend_order(self) -> list[str]:
        return ["tesseract"]

    def extract_clue_crop(
        self,
        image_path: str,
        clue_box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x, y, w, h = clue_box
        if w <= 0 or h <= 0:
            return None

        if cv2 is None:
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        img_h, img_w = image.shape[:2]
        x0 = max(0, min(x, img_w - 1))
        y0 = max(0, min(y, img_h - 1))
        x1 = max(x0 + 1, min(x + w, img_w))
        y1 = max(y0 + 1, min(y + h, img_h))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return crop.copy()

    def split_clue_fields(self, clue_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (voltorbs_roi, total_roi) for one clue crop using fixed normalized bounds."""
        if clue_crop.size == 0:
            return None

        total_roi = self._crop_by_bounds(clue_crop, self._TOTAL_OCR_BOUNDS)
        voltorbs_roi = self._crop_by_bounds(clue_crop, self._VOLTORB_OCR_BOUNDS)
        if total_roi.size == 0 or voltorbs_roi.size == 0:
            return None
        return voltorbs_roi, total_roi

    def save_clue_crop(
        self,
        image_path: str,
        clue_box: tuple[int, int, int, int],
        output_path: str | Path,
    ) -> bool:
        if cv2 is None:
            return False

        crop = self.extract_clue_crop(image_path, clue_box)
        if crop is None:
            return False

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), crop))

    def parse_clue_from_screenshot(
        self,
        image_path: str,
        clue_box: tuple[int, int, int, int],
        *,
        fast: bool = True,
    ) -> tuple[int, int] | None:
        crop = self.extract_clue_crop(image_path, clue_box)
        if crop is None:
            return None

        return self.parse_clue_box(crop, fast=fast)

    def debug_parse_clue_from_screenshot(
        self,
        image_path: str,
        clue_box: tuple[int, int, int, int],
        *,
        output_root: str | Path,
        region_name: str = "clue",
    ) -> ClueDebugArtifacts | None:
        if cv2 is None or pytesseract is None:
            return None
        if not self._configure_tesseract_runtime():
            return None

        crop = self.extract_clue_crop(image_path, clue_box)
        if crop is None:
            return None

        split = self.split_clue_fields(crop)
        if split is None:
            return None

        voltorbs_roi, total_roi = split
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_root_path = Path(output_root)
        raw_dir = output_root_path / "raw"
        pre_dir = output_root_path / "preprocessed"
        log_dir = output_root_path / "logs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pre_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Keep filenames stable and sortable for quick manual inspection.
        file_prefix = f"{region_name}_{ts}"
        raw_voltorbs_path = raw_dir / f"{file_prefix}_voltorbs.png"
        raw_total_path = raw_dir / f"{file_prefix}_total.png"
        preprocessed_voltorbs_path = pre_dir / f"{file_prefix}_voltorbs_bw.png"
        preprocessed_total_path = pre_dir / f"{file_prefix}_total_bw.png"
        log_path = log_dir / f"{file_prefix}.log"

        cv2.imwrite(str(raw_voltorbs_path), voltorbs_roi)
        cv2.imwrite(str(raw_total_path), total_roi)

        voltorbs_pre = self._preprocess_clue_debug_roi(voltorbs_roi)
        total_pre = self._preprocess_clue_debug_roi(total_roi)
        cv2.imwrite(str(preprocessed_voltorbs_path), voltorbs_pre)
        cv2.imwrite(str(preprocessed_total_path), total_pre)

        ocr_config = "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
        voltorbs_text = pytesseract.image_to_string(voltorbs_pre, config=ocr_config).strip()
        total_text = pytesseract.image_to_string(total_pre, config=ocr_config).strip()

        voltorbs_value = self._extract_first_int(voltorbs_text)
        total_value = self._extract_first_int(total_text)

        x, y, w, h = clue_box
        log_lines = [
            f"source={image_path}",
            f"region={region_name}",
            f"clue_box=({x},{y},{w},{h})",
            f"raw_voltorbs={raw_voltorbs_path}",
            f"raw_total={raw_total_path}",
            f"preprocessed_voltorbs={preprocessed_voltorbs_path}",
            f"preprocessed_total={preprocessed_total_path}",
            f"tesseract_config={ocr_config}",
            f"voltorbs_text={voltorbs_text!r}",
            f"total_text={total_text!r}",
            f"voltorbs_value={voltorbs_value}",
            f"total_value={total_value}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return ClueDebugArtifacts(
            raw_voltorbs_path=raw_voltorbs_path,
            raw_total_path=raw_total_path,
            preprocessed_voltorbs_path=preprocessed_voltorbs_path,
            preprocessed_total_path=preprocessed_total_path,
            log_path=log_path,
            voltorbs_text=voltorbs_text,
            total_text=total_text,
            voltorbs_value=voltorbs_value,
            total_value=total_value,
        )

    def parse_clue_box(self, image: str | np.ndarray, *, fast: bool = True) -> tuple[int, int] | None:
        debug_lines: list[str] = [
            f"parse_clue_box fast={fast}",
        ]
        cv_img = self._to_cv_image(image)
        if cv_img is None or cv2 is None:
            debug_lines.append("image_unavailable_or_dependencies_missing")
            self.last_clue_debug = debug_lines
            return None

        img_h, img_w = cv_img.shape[:2]
        debug_lines[0] = f"parse_clue_box fast={fast} shape={img_w}x{img_h}"
        debug_lines.append(f"backend_order={self._clue_ocr_backend_order()}")

        split = self.split_clue_fields(cv_img)
        if split is None:
            debug_lines.append("split_clue_fields_failed")
            self.last_clue_debug = debug_lines
            return None

        voltorbs_roi, total_roi = split
        voltorbs = self._ocr_number_field(
            voltorbs_roi,
            min_value=0,
            max_value=5,
            fast=fast,
            field_name="voltorbs",
            debug_lines=debug_lines,
        )
        total = self._ocr_number_field(
            total_roi,
            min_value=0,
            max_value=15,
            fast=fast,
            field_name="total",
            debug_lines=debug_lines,
        )

        pair: tuple[int, int] | None = None
        if voltorbs is not None and total is not None:
            candidate = (voltorbs, total)
            if self._is_plausible_clue(*candidate):
                pair = candidate
                debug_lines.append(f"candidate_pair={candidate} accepted")
            else:
                debug_lines.append(f"candidate_pair={candidate} rejected_plausibility")
        else:
            debug_lines.append(f"candidate_missing voltorbs={voltorbs} total={total}")

        debug_lines.append(f"final_pair={pair}")
        self.last_clue_debug = debug_lines
        return pair

    def _to_cv_image(self, image: str | np.ndarray) -> np.ndarray | None:
        if cv2 is None:
            return None

        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        cv_img = image.copy()
        if cv_img.ndim == 2:
            cv_img = np.dstack([cv_img, cv_img, cv_img])
        elif cv_img.ndim != 3 or cv_img.shape[2] < 3:
            return None
        return cv_img

    def _preprocess_clue_debug_roi(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def _extract_first_int(self, raw_text: str) -> int | None:
        match = re.search(r"\d+", raw_text)
        if match is None:
            return None
        return int(match.group(0))

    def _ocr_number_field(
        self,
        roi: np.ndarray,
        *,
        min_value: int,
        max_value: int,
        fast: bool,
        field_name: str,
        debug_lines: list[str] | None = None,
    ) -> int | None:
        if cv2 is None or roi.size == 0:
            if debug_lines is not None:
                debug_lines.append(f"field={field_name} unavailable")
            return None

        backends = self._clue_ocr_backend_order()
        if debug_lines is not None:
            debug_lines.append(f"field={field_name} backends={backends}")

        best_value: int | None = None
        best_score = -1.0
        for _backend in backends:
            value, score = self._ocr_number_field_tesseract(
                roi,
                min_value=min_value,
                max_value=max_value,
                fast=fast,
                field_name=field_name,
                debug_lines=debug_lines,
            )

            if value is not None and score > best_score:
                best_value = value
                best_score = score

        if debug_lines is not None:
            if best_value is None:
                debug_lines.append(f"field={field_name} result=None")
            else:
                debug_lines.append(f"field={field_name} winner={best_value} score={best_score:.2f}")
        return best_value

    def _ocr_number_field_tesseract(
        self,
        roi: np.ndarray,
        *,
        min_value: int,
        max_value: int,
        fast: bool,
        field_name: str,
        debug_lines: list[str] | None = None,
    ) -> tuple[int | None, float]:
        if pytesseract is None:
            if debug_lines is not None:
                debug_lines.append(f"field={field_name} backend=tesseract unavailable")
            return None, -1.0

        if not self._configure_tesseract_runtime():
            if debug_lines is not None:
                debug_lines.append(f"field={field_name} backend=tesseract runtime_unavailable")
            return None, -1.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        nearest = cv2.resize(gray, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_NEAREST)
        variants = [
            ("nearest", nearest),
            ("otsu", cv2.threshold(nearest, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ]
        if not fast:
            variants.append(
                ("otsu_inv", cv2.threshold(nearest, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
            )

        psm_modes = (10, 8) if fast else (10, 8, 7)
        config_base = "--oem 3 -c tessedit_char_whitelist=0123456789"
        scores: dict[int, float] = {}

        for variant_name, variant in variants:
            for psm in psm_modes:
                try:
                    data = pytesseract.image_to_data(
                        variant,
                        output_type=pytesseract.Output.DICT,
                        config=f"{config_base} --psm {psm}",
                    )
                except Exception:
                    continue

                texts = data.get("text", [])
                confs = data.get("conf", [])
                count = min(len(texts), len(confs))
                for i in range(count):
                    text = str(texts[i]).strip()
                    if not text:
                        continue

                    try:
                        confidence = float(confs[i])
                    except Exception:
                        confidence = -1.0

                    digits = re.findall(r"\d+", text)
                    if not digits:
                        continue

                    token = digits[0]
                    value = int(token)
                    weight = max(1.0, confidence if confidence > 0 else 1.0)
                    if min_value <= value <= max_value:
                        scores[value] = scores.get(value, 0.0) + weight

                    if len(token) == 2 and token.startswith("1"):
                        short_value = int(token[1])
                        if min_value <= short_value <= max_value:
                            scores[short_value] = scores.get(short_value, 0.0) + weight * 0.6

                    if token == "10" and min_value == 0:
                        scores[0] = scores.get(0, 0.0) + weight * 0.6

                if debug_lines is not None:
                    debug_lines.append(
                        "field="
                        f"{field_name} backend=tesseract variant={variant_name} psm={psm} "
                        f"partial_scores={dict(sorted(scores.items()))}"
                    )

        if not scores:
            return None, -1.0

        value, score = max(scores.items(), key=lambda item: item[1])
        return value, float(score)

    def _crop_by_bounds(
        self,
        image: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        x0_f, y0_f, x1_f, y1_f = bounds
        x0 = max(0, min(int(round(w * x0_f)), w - 1))
        y0 = max(0, min(int(round(h * y0_f)), h - 1))
        x1 = max(x0 + 1, min(int(round(w * x1_f)), w))
        y1 = max(y0 + 1, min(int(round(h * y1_f)), h))
        return image[y0:y1, x0:x1]

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

        if not self._configure_tesseract_runtime():
            result.warnings.append(
                "Tesseract OCR engine is unavailable at runtime. Install `tesseract`, add it to PATH, or set `TESSERACT_CMD`."
            )
            return result

        cv_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)

        row_pairs, col_pairs = self._parse_structured_board_clues(cv_img)
        parsed_rows = 0
        parsed_cols = 0
        for idx, pair in enumerate(row_pairs):
            if pair is None:
                continue
            result.row_clues[idx] = Clue(voltorbs=pair[0], total=pair[1])
            parsed_rows += 1
        for idx, pair in enumerate(col_pairs):
            if pair is None:
                continue
            result.col_clues[idx] = Clue(voltorbs=pair[0], total=pair[1])
            parsed_cols += 1

        if parsed_rows == BOARD_SIZE and parsed_cols == BOARD_SIZE:
            return result

        if parsed_rows or parsed_cols:
            result.warnings.append(
                f"Structured clue detection parsed {parsed_rows}/{BOARD_SIZE} rows and {parsed_cols}/{BOARD_SIZE} columns."
            )

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        try:
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            result.warnings.append(
                "Tesseract OCR engine is unavailable at runtime. Install `tesseract`, add it to PATH, or set `TESSERACT_CMD`."
            )
            return result
        except pytesseract.TesseractError as exc:
            result.warnings.append(
                f"OCR failed with Tesseract error: {exc}. Enter clues manually."
            )
            return result
        except Exception as exc:
            result.warnings.append(
                f"OCR failed while reading image: {exc}. Enter clues manually."
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

    def _parse_structured_board_clues(
        self, cv_img: np.ndarray
    ) -> tuple[list[tuple[int, int] | None], list[tuple[int, int] | None]]:
        if cv2 is None:
            return [None for _ in range(BOARD_SIZE)], [None for _ in range(BOARD_SIZE)]

        board_box = self._detect_board_box(cv_img)
        if board_box is None:
            board_box = self._fallback_board_box(cv_img.shape[1], cv_img.shape[0])

        left, top, step, tile_size = board_box
        row_rects, col_rects = self._build_clue_rects(cv_img.shape[1], cv_img.shape[0], left, top, step, tile_size)

        row_pairs = self._parse_clue_rects(cv_img, row_rects)
        col_pairs = self._parse_clue_rects(cv_img, col_rects)
        return row_pairs, col_pairs

    def _fallback_board_box(self, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        # Heuristic fallback aligned with screen parser defaults.
        board_w = int(round(image_w * 0.57))
        board_h = int(round(image_h * 0.42))
        board_x = int(round(image_w * 0.02))
        board_y = int(round(image_h * 0.50))

        side = max(20, int(round(min(board_w, board_h) / 6.0)))
        step = max(24, int(round((min(board_w, board_h) - side) / max(1, BOARD_SIZE - 1))))
        return board_x, board_y, step, side

    def _detect_board_box(self, cv_img: np.ndarray) -> tuple[int, int, int, int] | None:
        if cv2 is None:
            return None

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        min_side = max(24, int(min(h, w) * 0.05))
        max_side = max(48, int(min(h, w) * 0.2))

        centers: list[tuple[float, float, int]] = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            if len(approx) != 4:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
                continue
            ar = bw / max(1, bh)
            if not (0.85 <= ar <= 1.15):
                continue

            cx = x + bw / 2.0
            cy = y + bh / 2.0
            side = int(round((bw + bh) / 2.0))
            centers.append((cx, cy, side))

        if len(centers) < BOARD_SIZE * BOARD_SIZE:
            return None

        deduped: list[tuple[float, float, int]] = []
        for cx, cy, side in sorted(centers, key=lambda v: (v[1], v[0], v[2])):
            if any(abs(cx - ox) <= 4 and abs(cy - oy) <= 4 for ox, oy, _ in deduped):
                continue
            deduped.append((cx, cy, side))

        if len(deduped) < BOARD_SIZE * BOARD_SIZE:
            return None

        x_clusters = self._cluster_axis([v[0] for v in deduped], tolerance=12)
        y_clusters = self._cluster_axis([v[1] for v in deduped], tolerance=12)
        if len(x_clusters) < BOARD_SIZE or len(y_clusters) < BOARD_SIZE:
            return None

        x_clusters = sorted(x_clusters, key=lambda c: c[1], reverse=True)[:BOARD_SIZE]
        y_clusters = sorted(y_clusters, key=lambda c: c[1], reverse=True)[:BOARD_SIZE]
        x_centers = sorted(c[0] for c in x_clusters)
        y_centers = sorted(c[0] for c in y_clusters)

        if len(x_centers) < BOARD_SIZE or len(y_centers) < BOARD_SIZE:
            return None

        dx = [x_centers[i + 1] - x_centers[i] for i in range(BOARD_SIZE - 1)]
        dy = [y_centers[i + 1] - y_centers[i] for i in range(BOARD_SIZE - 1)]
        step = int(round(statistics.median(dx + dy)))
        if step <= 0:
            return None

        tile_size = int(round(statistics.median(v[2] for v in deduped)))
        left = int(round(x_centers[0] - tile_size / 2))
        top = int(round(y_centers[0] - tile_size / 2))
        return left, top, step, tile_size

    def _cluster_axis(self, values: list[float], tolerance: int) -> list[tuple[float, int]]:
        if not values:
            return []

        sorted_vals = sorted(values)
        clusters: list[list[float]] = [[sorted_vals[0]]]
        for value in sorted_vals[1:]:
            if abs(value - clusters[-1][-1]) <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        return [(sum(cluster) / len(cluster), len(cluster)) for cluster in clusters]

    def _build_clue_rects(
        self,
        image_w: int,
        image_h: int,
        board_left: int,
        board_top: int,
        step: int,
        tile_size: int,
    ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        panel_size = int(round(max(18, min(step * 0.78, tile_size * 1.05))))
        offset = int(round(step * 0.30))
        align_shift = int(round(step * 0.07))
        col_left_shift = int(round(step * 0.03))

        board_right = board_left + (BOARD_SIZE - 1) * step + tile_size
        board_bottom = board_top + (BOARD_SIZE - 1) * step + tile_size

        row_x = board_right + offset
        col_y = board_bottom + offset

        row_rects: list[tuple[int, int, int, int]] = []
        col_rects: list[tuple[int, int, int, int]] = []

        for idx in range(BOARD_SIZE):
            y = board_top - align_shift + idx * step
            row_rects.append(self._clip_rect(row_x, y, panel_size, panel_size, image_w, image_h))

            x = board_left - col_left_shift + idx * step
            col_rects.append(self._clip_rect(x, col_y, panel_size, panel_size, image_w, image_h))

        return row_rects, col_rects

    def _clip_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_w: int,
        image_h: int,
    ) -> tuple[int, int, int, int]:
        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        max_w = max(1, image_w - x)
        max_h = max(1, image_h - y)
        return x, y, min(w, max_w), min(h, max_h)

    def _parse_clue_rects(
        self,
        cv_img: np.ndarray,
        rects: list[tuple[int, int, int, int]],
        sample_offsets: list[tuple[int, int]] | None = None,
        fast_ocr: bool = False,
        debug_lines: list[str] | None = None,
    ) -> list[tuple[int, int] | None]:
        pairs: list[tuple[int, int] | None] = []
        img_h, img_w = cv_img.shape[:2]
        # Jitter ROI sampling improves robustness against tiny geometry shifts.
        if sample_offsets is None:
            sample_offsets = [
                (0, 0),
                (-2, 0),
                (2, 0),
                (0, -2),
                (0, 2),
                (-1, -1),
                (1, 1),
            ]

        top_token_map = {
            "15": 5,
            "ii": 5,
            "is": 5,
            "os": 5,
            "if": 3,
            "ie": 3,
            "ue": 3,
            "li": 7,
            "l3": 2,
        }
        bottom_token_map = {
            "al": 1,
            "ge": 2,
            "ag": 0,
            "az": 3,
            "10": 0,
            "iz": 2,
            "ae": 1,
            "ii": 1,
            "2": 2,
        }

        for x, y, w, h in rects:
            candidates: list[tuple[int, int]] = []
            if debug_lines is not None:
                debug_lines.append(f"rect=({x},{y},{w},{h})")

            for sample_idx, (dx, dy) in enumerate(sample_offsets):
                sx, sy, sw, sh = self._clip_rect(x + dx, y + dy, w, h, img_w, img_h)
                roi = cv_img[sy : sy + sh, sx : sx + sw]
                if roi.size == 0:
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] offset=({dx},{dy}) roi=empty")
                    continue

                top_token_roi = self._crop_by_bounds(roi, self._TOTAL_TOKEN_BOUNDS)
                bottom_token_roi = self._crop_by_bounds(roi, self._VOLTORB_TOKEN_BOUNDS)
                top_token = self._read_pixel_token(top_token_roi)
                bottom_token = self._read_pixel_token(bottom_token_roi)
                if debug_lines is not None:
                    debug_lines.append(
                        f"sample[{sample_idx}] offset=({dx},{dy}) tokens: top='{top_token}' bottom='{bottom_token}'"
                    )

                token_total = top_token_map.get(top_token)
                token_voltorbs = bottom_token_map.get(bottom_token)
                if token_total is not None and token_voltorbs is not None:
                    token_pair = (token_voltorbs, token_total)
                    if self._is_plausible_clue(*token_pair):
                        candidates.append(token_pair)
                        if debug_lines is not None:
                            debug_lines.append(
                                f"sample[{sample_idx}] token_pair={token_pair} accepted"
                            )
                    elif debug_lines is not None:
                        debug_lines.append(
                            f"sample[{sample_idx}] token_pair={token_pair} rejected_plausibility"
                        )
                    continue

                top_roi = self._crop_by_bounds(roi, self._TOTAL_OCR_BOUNDS)
                bottom_roi = self._crop_by_bounds(roi, self._VOLTORB_OCR_BOUNDS)

                total = self._ocr_number(top_roi, min_value=0, max_value=15, fast=fast_ocr)
                voltorbs = self._ocr_number(bottom_roi, min_value=0, max_value=5, fast=fast_ocr)
                if total is None:
                    total = self._ocr_number_pixel_token(top_roi, min_value=0, max_value=15, kind="top")
                if voltorbs is None:
                    voltorbs = self._ocr_number_pixel_token(bottom_roi, min_value=0, max_value=5, kind="bottom")
                if debug_lines is not None:
                    debug_lines.append(
                        f"sample[{sample_idx}] ocr: voltorbs={voltorbs} total={total}"
                    )

                if total is None or voltorbs is None:
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] skipped_missing_value")
                    continue

                ocr_pair = (voltorbs, total)
                if self._is_plausible_clue(*ocr_pair):
                    candidates.append(ocr_pair)
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] ocr_pair={ocr_pair} accepted")
                elif debug_lines is not None:
                    debug_lines.append(f"sample[{sample_idx}] ocr_pair={ocr_pair} rejected_plausibility")

            if not candidates:
                pairs.append(None)
                if debug_lines is not None:
                    debug_lines.append("candidates=[] -> result=None")
                continue

            votes = Counter(candidates)
            (winner, count) = max(votes.items(), key=lambda item: item[1])
            if debug_lines is not None:
                debug_lines.append(f"candidates={candidates}")
                debug_lines.append(f"votes={dict(votes)} winner={winner} count={count}")
            # Require either majority agreement or at least 2 votes for the winner.
            if count >= 2 or len(votes) == 1:
                pairs.append(winner)
                if debug_lines is not None:
                    debug_lines.append(f"accepted_winner={winner}")
            else:
                pairs.append(None)
                if debug_lines is not None:
                    debug_lines.append("winner_rejected_low_agreement")

        return pairs

    def _is_plausible_clue(self, voltorbs: int, total: int) -> bool:
        if not (0 <= voltorbs <= BOARD_SIZE and 0 <= total <= 15):
            return False

        safe_tiles = BOARD_SIZE - voltorbs
        min_total = safe_tiles  # all non-voltorbs are 1
        max_total = safe_tiles * 3  # all non-voltorbs are 3
        return min_total <= total <= max_total

    def _read_pixel_token(self, roi: np.ndarray) -> str:
        if pytesseract is None or cv2 is None or roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
        enlarged = cv2.resize(thresholded, None, fx=16.0, fy=16.0, interpolation=cv2.INTER_NEAREST)
        try:
            text = pytesseract.image_to_string(enlarged, config="--oem 3 --psm 10")
        except Exception:
            return ""
        return re.sub(r"[^A-Za-z0-9]", "", text).lower()

    def _ocr_number(self, roi: np.ndarray, min_value: int, max_value: int, fast: bool = False) -> int | None:
        if pytesseract is None or cv2 is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        nearest = cv2.resize(gray, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_NEAREST)
        if fast:
            variants = [
                nearest,
                cv2.threshold(nearest, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            ]
        else:
            variants = [
                nearest,
                cv2.threshold(nearest, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.threshold(nearest, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            ]

        char_map = {
            "O": 0,
            "o": 0,
            "Q": 0,
            "D": 0,
            "I": 1,
            "l": 1,
            "|": 1,
            "Z": 2,
            "z": 2,
            "S": 5,
            "s": 5,
            "T": 7,
            "V": 7,
            "v": 7,
        }

        config_base = "--oem 3 -c tessedit_char_whitelist=0123456789"
        psm_modes = (10,) if fast else (10, 7)
        candidate_scores: dict[int, int] = {}
        for variant in variants:
            for psm in psm_modes:
                try:
                    text = pytesseract.image_to_string(variant, config=f"{config_base} --psm {psm}")
                except Exception:
                    continue

                digits = re.findall(r"\d+", text)
                if digits:
                    token = digits[0]
                    value = int(token)
                    if min_value <= value <= max_value:
                        candidate_scores[value] = candidate_scores.get(value, 0) + 3

                    # Pixel-font OCR often adds a leading "1" where the actual value is single-digit.
                    if len(token) == 2 and token.startswith("1"):
                        short_value = int(token[1])
                        if min_value <= short_value <= max_value:
                            candidate_scores[short_value] = candidate_scores.get(short_value, 0) + 2

                    if token == "10" and min_value == 0:
                        candidate_scores[0] = candidate_scores.get(0, 0) + 2

                stripped = text.strip()
                if len(stripped) == 1 and stripped in char_map:
                    value = char_map[stripped]
                    if min_value <= value <= max_value:
                        candidate_scores[value] = candidate_scores.get(value, 0) + 1

        if candidate_scores:
            return max(candidate_scores.items(), key=lambda item: item[1])[0]

        return None

    def _ocr_number_pixel_token(self, roi: np.ndarray, min_value: int, max_value: int, kind: str) -> int | None:
        if pytesseract is None or cv2 is None or roi.size == 0:
            return None

        h, w = roi.shape[:2]
        if kind == "top":
            roi = roi[0:h, int(w * 0.45) : int(w * 0.95)]
        else:
            roi = roi[0:h, int(w * 0.20) : int(w * 0.98)]

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
        enlarged = cv2.resize(thresholded, None, fx=16.0, fy=16.0, interpolation=cv2.INTER_NEAREST)

        try:
            text = pytesseract.image_to_string(enlarged, config="--oem 3 --psm 10")
        except Exception:
            return None

        token = re.sub(r"[^A-Za-z0-9]", "", text).lower()
        if not token:
            return None

        # Fallback map tuned for pixel-art clue digits where OCR returns letter-like artifacts.
        if kind == "top":
            token_map = {
                "15": 5,
                "ii": 5,
                "is": 5,
                "os": 5,
                "if": 3,
                "ie": 3,
                "ue": 3,
                "li": 7,
                "l3": 2,
            }
        else:
            token_map = {
                "al": 1,
                "ge": 2,
                "ag": 0,
                "az": 3,
                "10": 0,
                "iz": 2,
                "ae": 1,
                "if": 1,
                "ii": 1,
                "ile": 0,
                "le": 0,
                "id": 0,
                "ig": 0,
            }

        if token in token_map:
            value = token_map[token]
            if min_value <= value <= max_value:
                return value

        return None

    def _configure_tesseract_runtime(self) -> bool:
        if pytesseract is None:
            return False

        cached = getattr(self, "_tesseract_runtime_ready", None)
        if cached is not None:
            return bool(cached)

        configured_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        candidates: list[str] = []

        env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
        if env_cmd:
            candidates.append(env_cmd)

        if configured_cmd:
            candidates.append(configured_cmd)

        which_cmd = shutil.which("tesseract")
        if which_cmd:
            candidates.append(which_cmd)

        # Common install paths help when GUI launches without shell PATH initialization.
        candidates.extend(
            [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
            ]
        )

        seen: set[str] = set()
        for cmd in candidates:
            cmd = cmd.strip()
            if not cmd or cmd in seen:
                continue
            seen.add(cmd)

            if os.path.isabs(cmd) and not (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):
                continue

            pytesseract.pytesseract.tesseract_cmd = cmd
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                continue
            self._tesseract_runtime_ready = True
            return True

        self._tesseract_runtime_ready = False
        return False

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
