from __future__ import annotations

import csv
import hashlib
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
    normalized_voltorbs_path: Path
    normalized_total_path: Path
    log_path: Path
    voltorbs_text: str
    total_text: str
    voltorbs_value: int | None
    total_value: int | None
    voltorbs_score: float
    total_score: float


class ImageParser:
    # Normalized clue-field bounds (x0, y0, x1, y1) inside a single clue box.
    # Used by overlay_app.py to generate debug sub-region overlays.
    _TOTAL_OCR_BOUNDS = (0.30, 0.02, 0.99, 0.42)
    _VOLTORB_OCR_BOUNDS = (0.615, 0.50, 0.99, 0.98)

    _TEMPLATE_MIN_SCORE = 0.85

    def __init__(self) -> None:
        self._templates_loaded = False
        self._template_bank: dict[str, dict[int, list[np.ndarray]]] = {"top": {}, "bottom": {}}
        self._unmatched_saved_hashes: set[str] = set()
        self.last_clue_debug: list[str] = []

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_project_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self._project_root() / p

    def _parse_manual_value(self, value: str) -> tuple[int, int] | None:
        token = value.strip()
        if not token:
            return None
        parts = [part.strip() for part in token.split(",")]
        if len(parts) != 2:
            return None
        try:
            voltorbs = int(parts[0])
            total = int(parts[1])
        except ValueError:
            return None
        return voltorbs, total

    def _binarize(self, roi: np.ndarray) -> np.ndarray | None:
        """Otsu threshold on the per-pixel channel maximum; returns full-size binary image (no crop).

        Digit strokes are dark gray (all BGR channels ≈ 65) while clue-box backgrounds
        are always colored — orange (R≈227), green (G≈170), etc.  Taking max(B, G, R)
        for each pixel gives ≈65 at digit locations regardless of the background hue,
        so global Otsu reliably separates digits from the background without being
        confused by which color the background uses.  Falls back to the raw array when
        the input is already single-channel.
        """
        if cv2 is None or roi.size == 0:
            return None
        channel = np.max(roi, axis=2) if roi.ndim == 3 else roi
        return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def _prepare_template_image(self, roi: np.ndarray) -> np.ndarray | None:
        bw = self._binarize(roi)
        if bw is None:
            return None
        # Normalize polarity so foreground digits are white on black for stable matching.
        if int(np.count_nonzero(bw)) > (bw.size // 2):
            bw = 255 - bw

        ys, xs = np.where(bw > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return bw[y0:y1, x0:x1]

    def _load_clue_templates(self) -> None:
        if self._templates_loaded or cv2 is None:
            return

        project_root = self._project_root()

        manifest_path = project_root / "assets/parser_debug/clue_dataset/manifest.csv"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if (row.get("split", "").strip().lower() != "labeled"):
                            continue
                        label = self._parse_manual_value(row.get("manual_value", ""))
                        rel_crop = row.get("crop_path", "").strip()
                        if label is None or not rel_crop:
                            continue
                        crop_path = self._resolve_project_path(rel_crop)
                        crop = cv2.imread(str(crop_path))
                        if crop is None:
                            continue
                        split = self.split_clue_fields(crop)
                        if split is None:
                            continue
                        voltorbs_roi, total_roi = split
                        voltorbs_norm = self._prepare_template_image(voltorbs_roi)
                        total_norm = self._prepare_template_image(total_roi)
                        if voltorbs_norm is not None:
                            self._template_bank["bottom"].setdefault(label[0], []).append(voltorbs_norm)
                        if total_norm is not None:
                            self._template_bank["top"].setdefault(label[1], []).append(total_norm)
            except Exception:
                pass

        # Also load manually curated templates from assets/templates and assets/templates/raw.
        for root_rel in ("assets/templates", "assets/templates/raw"):
            root = project_root / root_rel
            if not root.exists():
                continue
            for image_path in root.rglob("*.png"):
                stem = image_path.stem.lower()
                top_match = re.search(r"clue_t_(\d+)$", stem)
                bottom_match = re.search(r"clue_v_(\d+)$", stem)
                if top_match is None and bottom_match is None:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                normalized = self._prepare_template_image(image)
                if normalized is None:
                    continue
                if top_match is not None:
                    value = int(top_match.group(1))
                    self._template_bank["top"].setdefault(value, []).append(normalized)
                if bottom_match is not None:
                    value = int(bottom_match.group(1))
                    self._template_bank["bottom"].setdefault(value, []).append(normalized)

        self._templates_loaded = True

    def _match_number_template(
        self,
        roi: np.ndarray,
        *,
        kind: str,
        min_value: int,
        max_value: int,
        save_if_unmatched: bool,
        source_tag: str,
    ) -> tuple[int | None, float]:
        if cv2 is None or roi.size == 0:
            return None, -1.0

        self._load_clue_templates()
        bw = self._binarize(roi)
        if bw is None:
            if save_if_unmatched:
                self._save_unmatched_template_sample(roi, kind=kind, source_tag=source_tag)
            return None, -1.0

        best_value: int | None = None
        best_score = -1.0
        for value, templates in self._template_bank.get(kind, {}).items():
            if not (min_value <= value <= max_value):
                continue
            for template in templates:
                if template.shape[0] > bw.shape[0] or template.shape[1] > bw.shape[1]:
                    continue
                result = cv2.matchTemplate(bw, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                score = float(max_val)
                if score > best_score:
                    best_score = score
                    best_value = value

        if best_value is not None and best_score >= self._TEMPLATE_MIN_SCORE:
            return best_value, best_score

        if save_if_unmatched:
            self._save_unmatched_template_sample(roi, kind=kind, source_tag=source_tag)
        return None, best_score

    def _save_unmatched_template_sample(self, roi: np.ndarray, *, kind: str, source_tag: str) -> None:
        if cv2 is None or roi.size == 0:
            return

        normalized = self._prepare_template_image(roi)
        hash_source = normalized if normalized is not None else roi
        digest = hashlib.sha1(hash_source.tobytes()).hexdigest()[:12]
        key = f"{kind}:{digest}"
        if key in self._unmatched_saved_hashes:
            return
        self._unmatched_saved_hashes.add(key)

        out_dir = self._project_root() / "assets/templates/raw/clue_unknown"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{stamp}_{kind}_{digest}.png"
        out_path = out_dir / file_name
        cv2.imwrite(str(out_path), roi)

        manifest_path = out_dir / "manifest.csv"
        if not manifest_path.exists():
            manifest_path.write_text("timestamp,kind,path,source\n", encoding="utf-8")
        with manifest_path.open("a", encoding="utf-8") as fh:
            rel = out_path.relative_to(self._project_root())
            fh.write(f"{stamp},{kind},{rel.as_posix()},{source_tag}\n")

    def _clue_ocr_backend_order(self) -> list[str]:
        return ["template"]

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
        run_id: str | None = None,
    ) -> ClueDebugArtifacts | None:
        if cv2 is None:
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

        # Batch mode layout: <output_root>/<run_id>/<region_name>/...
        # Single mode layout: <output_root>/<region_name>_<timestamp>/...
        if run_id:
            run_dir = output_root_path / run_id / region_name
        else:
            run_dir = output_root_path / f"{region_name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_voltorbs_path = run_dir / "raw_voltorbs.png"
        raw_total_path = run_dir / "raw_total.png"
        normalized_voltorbs_path = run_dir / "normalized_voltorbs.png"
        normalized_total_path = run_dir / "normalized_total.png"
        log_path = run_dir / "debug.log"

        cv2.imwrite(str(raw_voltorbs_path), voltorbs_roi)
        cv2.imwrite(str(raw_total_path), total_roi)

        voltorbs_value, voltorbs_score = self._match_number_template(
            voltorbs_roi,
            kind="bottom",
            min_value=0,
            max_value=5,
            save_if_unmatched=True,
            source_tag="debug:voltorbs",
        )
        total_value, total_score = self._match_number_template(
            total_roi,
            kind="top",
            min_value=0,
            max_value=15,
            save_if_unmatched=True,
            source_tag="debug:total",
        )

        voltorbs_norm = self._binarize(voltorbs_roi)
        if voltorbs_norm is not None:
            cv2.imwrite(str(normalized_voltorbs_path), voltorbs_norm)
        total_norm = self._binarize(total_roi)
        if total_norm is not None:
            cv2.imwrite(str(normalized_total_path), total_norm)

        voltorbs_text = (
            f"match:{voltorbs_value} score={voltorbs_score:.3f}"
            if voltorbs_value is not None
            else f"no_match score={voltorbs_score:.3f}"
        )
        total_text = (
            f"match:{total_value} score={total_score:.3f}"
            if total_value is not None
            else f"no_match score={total_score:.3f}"
        )

        x, y, w, h = clue_box
        log_lines = [
            f"source={image_path}",
            f"run_id={run_id}",
            f"region={region_name}",
            f"clue_box=({x},{y},{w},{h})",
            f"voltorbs_text={voltorbs_text!r}",
            f"total_text={total_text!r}",
            f"voltorbs_value={voltorbs_value}",
            f"total_value={total_value}",
            f"voltorbs_score={voltorbs_score:.3f}",
            f"total_score={total_score:.3f}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return ClueDebugArtifacts(
            raw_voltorbs_path=raw_voltorbs_path,
            raw_total_path=raw_total_path,
            normalized_voltorbs_path=normalized_voltorbs_path,
            normalized_total_path=normalized_total_path,
            log_path=log_path,
            voltorbs_text=voltorbs_text,
            total_text=total_text,
            voltorbs_value=voltorbs_value,
            total_value=total_value,
            voltorbs_score=voltorbs_score,
            total_score=total_score,
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

        kind = "bottom" if field_name == "voltorbs" else "top"
        value, score = self._match_number_template(
            roi,
            kind=kind,
            min_value=min_value,
            max_value=max_value,
            save_if_unmatched=True,
            source_tag=f"field:{field_name}",
        )

        if debug_lines is not None:
            if value is None:
                debug_lines.append(f"field={field_name} template_result=None score={score:.3f}")
            else:
                debug_lines.append(f"field={field_name} template_winner={value} score={score:.3f}")
        return value

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

        if cv2 is None:
            result.warnings.append("Image parsing dependency unavailable (opencv-python). Imported image unchanged.")
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

        result.warnings.append(
            "Template matching could not parse all clues. Unmatched clue fields were saved under assets/templates/raw/clue_unknown for manual labeling."
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

        for x, y, w, h in rects:
            candidates: list[tuple[tuple[int, int], float]] = []
            if debug_lines is not None:
                debug_lines.append(f"rect=({x},{y},{w},{h})")

            for sample_idx, (dx, dy) in enumerate(sample_offsets):
                sx, sy, sw, sh = self._clip_rect(x + dx, y + dy, w, h, img_w, img_h)
                roi = cv_img[sy : sy + sh, sx : sx + sw]
                if roi.size == 0:
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] offset=({dx},{dy}) roi=empty")
                    continue

                top_roi = self._crop_by_bounds(roi, self._TOTAL_OCR_BOUNDS)
                bottom_roi = self._crop_by_bounds(roi, self._VOLTORB_OCR_BOUNDS)

                total, total_score = self._match_number_template(
                    top_roi,
                    kind="top",
                    min_value=0,
                    max_value=15,
                    save_if_unmatched=True,
                    source_tag=f"rect:{x},{y},{w},{h}:sample:{sample_idx}:top",
                )
                voltorbs, voltorbs_score = self._match_number_template(
                    bottom_roi,
                    kind="bottom",
                    min_value=0,
                    max_value=5,
                    save_if_unmatched=True,
                    source_tag=f"rect:{x},{y},{w},{h}:sample:{sample_idx}:bottom",
                )
                if debug_lines is not None:
                    debug_lines.append(
                        f"sample[{sample_idx}] template: voltorbs={voltorbs} s={voltorbs_score:.3f} total={total} s={total_score:.3f}"
                    )

                if total is None or voltorbs is None:
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] skipped_missing_value")
                    continue

                template_pair = (voltorbs, total)
                if self._is_plausible_clue(*template_pair):
                    combined_score = (voltorbs_score + total_score) / 2.0
                    candidates.append((template_pair, combined_score))
                    if debug_lines is not None:
                        debug_lines.append(f"sample[{sample_idx}] template_pair={template_pair} accepted")
                elif debug_lines is not None:
                    debug_lines.append(f"sample[{sample_idx}] template_pair={template_pair} rejected_plausibility")

            if not candidates:
                pairs.append(None)
                if debug_lines is not None:
                    debug_lines.append("candidates=[] -> result=None")
                continue

            votes = Counter(pair for pair, _score in candidates)
            avg_scores = {
                pair: float(np.mean([score for p, score in candidates if p == pair])) for pair in votes
            }
            winner, count = max(votes.items(), key=lambda item: (item[1], avg_scores[item[0]]))
            if debug_lines is not None:
                debug_lines.append(f"candidates={candidates}")
                debug_lines.append(f"votes={dict(votes)} avg_scores={avg_scores} winner={winner} count={count}")
            # Require at least 2 independent jitter samples to agree.
            if count >= 2:
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
        # Preserve API for compatibility; now based on template matching instead of OCR text.
        value, score = self._match_number_template(
            roi,
            kind="top",
            min_value=0,
            max_value=15,
            save_if_unmatched=False,
            source_tag="token",
        )
        if value is None:
            return ""
        return f"{value}@{score:.2f}"

    def _ocr_number(self, roi: np.ndarray, min_value: int, max_value: int, fast: bool = False) -> int | None:
        _fast = fast
        value, _score = self._match_number_template(
            roi,
            kind="top",
            min_value=min_value,
            max_value=max_value,
            save_if_unmatched=True,
            source_tag="ocr_number",
        )
        return value

    def _ocr_number_pixel_token(self, roi: np.ndarray, min_value: int, max_value: int, kind: str) -> int | None:
        kind_key = "top" if kind == "top" else "bottom"
        value, _score = self._match_number_template(
            roi,
            kind=kind_key,
            min_value=min_value,
            max_value=max_value,
            save_if_unmatched=True,
            source_tag=f"pixel_token:{kind_key}",
        )
        return value

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
