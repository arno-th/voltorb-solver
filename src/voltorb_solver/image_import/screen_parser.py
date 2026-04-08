from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import itertools
import os
from pathlib import Path
import argparse

import cv2
import numpy as np


@dataclass(slots=True)
class Region:
    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class ScreenParseResult:
    image_width: int
    image_height: int
    regions: list[Region]
    warnings: list[str]
    region_methods: dict[str, str]

    def by_name(self) -> dict[str, Region]:
        return {region.name: region for region in self.regions}

    def method_summary(self) -> str:
        board_methods = {method for name, method in self.region_methods.items() if name.startswith("(")}
        row_methods = {
            method
            for name, method in self.region_methods.items()
            if name.startswith("r") and name[1:].isdigit()
        }
        col_methods = {
            method
            for name, method in self.region_methods.items()
            if name.startswith("c") and name[1:].isdigit()
        }

        board = ",".join(sorted(board_methods)) if board_methods else "unknown"
        rows = ",".join(sorted(row_methods)) if row_methods else "unknown"
        cols = ",".join(sorted(col_methods)) if col_methods else "unknown"
        return f"tiles((0,0)-(4,4))={board}; rows(r0-r4)={rows}; cols(c0-c4)={cols}"


class ScreenBoardParser:
    """Detects coarse Voltorb Flip regions from a full screenshot."""
    TILE_TEMPLATE_MIN_SCORE = 0.80
    CLUE_TEMPLATE_MIN_SCORE = 0.52

    def __init__(self, debug_dir: str | Path | None = None) -> None:
        self._clue_templates: list[np.ndarray] = []
        self._tile_templates: list[np.ndarray] = []
        self._tile_template_names: list[str] = []
        self._anchor_templates: list[np.ndarray] = []
        self._anchor_template_names: list[str] = []
        self._bl_anchor_templates: list[np.ndarray] = []
        self._bl_anchor_template_names: list[str] = []
        self._grid_layout: dict[str, object] | None = None
        env_debug = os.environ.get("VOLTORB_PARSER_DEBUG_DIR", "").strip()
        selected_debug_dir = debug_dir if debug_dir is not None else (env_debug or None)
        self._debug_dir = Path(selected_debug_dir).expanduser().resolve() if selected_debug_dir else None
        self._debug_run_dir: Path | None = None
        env_tile_threshold = os.environ.get("VOLTORB_TILE_TEMPLATE_MIN_SCORE", "").strip()
        if env_tile_threshold:
            try:
                threshold = float(env_tile_threshold)
                self._tile_template_min_score = max(0.0, min(1.0, threshold))
            except ValueError:
                self._tile_template_min_score = self.TILE_TEMPLATE_MIN_SCORE
        else:
            self._tile_template_min_score = self.TILE_TEMPLATE_MIN_SCORE

    def parse(self, image_path: str) -> ScreenParseResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        self._grid_layout = None
        self._begin_debug_run(image_path, image)

        image_h, image_w = image.shape[:2]
        warnings: list[str] = []

        panel = self._detect_game_panel(image)
        if panel is None:
            warnings.append("Could not detect game panel from screenshot; using full image.")
            panel = (0, 0, image_w, image_h)
        self._debug_log(f"panel={panel}")

        px, py, pw, ph = panel
        panel_roi = image[py : py + ph, px : px + pw]
        self._debug_write_image("panel_roi.png", panel_roi)

        board_method = "tile-template-grid"
        board = self._detect_board_grid_from_templates(image, panel)
        if board is None:
            raise ValueError("Could not detect board grid from tile templates.")
        self._debug_log(f"board_method={board_method} board={board}")

        regions = self._build_regions(image_w, image_h, panel, board)
        regions = self._refine_clue_regions_with_templates(image, regions, board)

        region_methods: dict[str, str] = {}
        for region in regions:
            if region.name.startswith("("):
                region_methods[region.name] = board_method
                continue
            if region.name.startswith("r") and region.name[1:].isdigit():
                region_methods[region.name] = "template-clue"
                continue
            if region.name.startswith("c") and region.name[1:].isdigit():
                region_methods[region.name] = "template-clue"
                continue
            region_methods[region.name] = "unknown"

        result = ScreenParseResult(
            image_width=image_w,
            image_height=image_h,
            regions=regions,
            warnings=warnings,
            region_methods=region_methods,
        )
        self._debug_log(f"method_summary={result.method_summary()}")
        if warnings:
            self._debug_log("warnings=" + " | ".join(warnings))
        self._end_debug_run()
        return result

    def annotate(self, image_path: str, output_path: str) -> ScreenParseResult:
        result = self.parse(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        for idx, region in enumerate(result.regions):
            color = self._label_color(idx)
            x2 = region.x + region.w
            y2 = region.y + region.h
            cv2.rectangle(image, (region.x, region.y), (x2, y2), color, 2)
            cv2.putText(
                image,
                region.name,
                (region.x + 4, max(region.y - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise ValueError(f"Failed to write annotated image: {output_path}")
        return result

    def _detect_game_panel(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Voltorb board/game panel is strongly green in DS screenshots.
        mask = cv2.inRange(hsv, (35, 35, 35), (95, 255, 255))
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_h, image_w = image.shape[:2]
        min_area = image_w * image_h * 0.03
        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area:
                continue
            aspect = w / max(h, 1)
            if 0.45 <= aspect <= 1.45:
                candidates.append((x, y, w, h))

        if not candidates:
            return None
        return max(candidates, key=lambda c: c[2] * c[3])

    def _detect_board_grid_from_templates(
        self,
        image: np.ndarray,
        panel: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        tile_templates = self._load_tile_templates()
        if not tile_templates:
            return None

        px, py, pw, ph = panel
        roi = image[py : py + ph, px : px + pw]
        if roi.size == 0:
            self._debug_log("tile_template: empty panel ROI")
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self._debug_write_image("tile_match_gray_panel.png", gray)
        centers, sides, stats = self._match_tile_centers(gray, tile_templates)
        self._debug_log(
            "tile_template: "
            f"templates={stats['template_count']} "
            f"raw_hits={stats['raw_hit_count']} "
            f"nms_kept={stats['nms_kept_count']} "
            f"best_score={stats['best_score']:.4f} "
            f"strict_threshold={stats.get('strict_threshold', self._tile_template_min_score):.2f} "
            f"relaxed_threshold={stats.get('relaxed_threshold', self._tile_template_min_score):.2f} "
            f"relaxed_hits={stats.get('relaxed_hit_count', 0)} "
            f"relaxed_used={stats.get('used_relaxed_fallback', False)}"
        )
        per_template = stats.get("per_template", [])
        if per_template:
            template_lines = []
            for info in per_template:
                template_lines.append(
                    f"{info['name']}: max={info['max_score']:.4f}, hits={info['hit_count']}"
                )
            self._debug_log("tile_template per-template: " + " | ".join(template_lines))

        if stats["draw_overlay"] is not None:
            self._debug_write_image("tile_match_overlay.png", stats["draw_overlay"])

        if len(centers) < 12:
            self._debug_log("tile_template: failed due to insufficient centers (<12)")
            return None

        # Optional anchor can reduce false-positive clusters when available.
        anchor = self._find_anchor(gray)
        if anchor is not None and stats["draw_overlay"] is not None:
            ax_i = int(round(anchor["reference_x"]))
            ay_i = int(round(anchor["reference_y"]))
            overlay_with_anchor = stats["draw_overlay"].copy()
            cv2.drawMarker(
                overlay_with_anchor,
                (ax_i, ay_i),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                overlay_with_anchor,
                f"anchor ({ax_i},{ay_i})",
                (ax_i + 8, max(16, ay_i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            self._debug_write_image("tile_match_overlay_with_anchor.png", overlay_with_anchor)
        if anchor is not None:
            ax = anchor["reference_x"]
            ay = anchor["reference_y"]
            if anchor["corner"] == "top-right":
                filtered_centers = [
                    center for center in centers if center[0] <= ax + 12 and center[1] >= ay - 12
                ]
            else:
                filtered_centers = [
                    center for center in centers if center[0] >= ax - 12 and center[1] >= ay - 12
                ]
            min_centers_after_filter = max(25, int(round(len(centers) * 0.60)))
            if len(filtered_centers) >= min_centers_after_filter:
                centers = filtered_centers
                self._debug_log(
                    "tile_template: "
                    f"anchor_corner={anchor['corner']} "
                    f"anchor_ref=({anchor['reference_x']:.1f},{anchor['reference_y']:.1f}) "
                    f"centers_after_anchor_filter={len(centers)}"
                )
            else:
                self._debug_log(
                    "tile_template: "
                    f"anchor_corner={anchor['corner']} "
                    f"anchor_ref=({anchor['reference_x']:.1f},{anchor['reference_y']:.1f}) "
                    f"ignored_anchor_filter_kept={len(filtered_centers)} "
                    f"required={min_centers_after_filter}"
                )

        tolerance = max(8, int(round(np.median(sides) * 0.35))) if sides else 10
        x_clusters = self._cluster_axis([x for x, _y in centers], tolerance=tolerance)
        y_clusters = self._cluster_axis([y for _x, y in centers], tolerance=tolerance)
        self._debug_log(
            f"tile_template: tolerance={tolerance}, x_clusters={len(x_clusters)}, y_clusters={len(y_clusters)}"
        )
        if len(x_clusters) < 5 or len(y_clusters) < 5:
            self._debug_log("tile_template: failed due to insufficient clusters")
            return None

        x_clusters = self._select_grid_clusters(x_clusters, expected=5)
        y_clusters = self._select_grid_clusters(y_clusters, expected=5)
        x_centers = sorted(cluster[0] for cluster in x_clusters)
        y_centers = sorted(cluster[0] for cluster in y_clusters)

        if len(x_centers) < 5 or len(y_centers) < 5:
            return None

        if sides:
            side = int(round(np.median(sides)))
        else:
            side = int(round(min(pw, ph) * 0.16))

        self._grid_layout = {
            "x_centers": [px + center for center in x_centers],
            "y_centers": [py + center for center in y_centers],
            "tile_side": int(side),
        }
        board_x = int(round(px + x_centers[0] - side / 2))
        board_y = int(round(py + y_centers[0] - side / 2))
        board_w = int(round((x_centers[-1] - x_centers[0]) + side))
        board_h = int(round((y_centers[-1] - y_centers[0]) + side))
        self._debug_log(
            f"tile_template: success board_local=({board_x - px},{board_y - py},{board_w},{board_h}) side={side}"
        )
        return board_x, board_y, board_w, board_h

    def _match_tile_centers(
        self,
        gray_image: np.ndarray,
        tile_templates: list[np.ndarray],
    ) -> tuple[list[tuple[float, float]], list[int], dict[str, object]]:
        points: list[tuple[float, float, int, float]] = []
        relaxed_points: list[tuple[float, float, int, float]] = []
        template_maxima: list[float] = []
        per_template: list[dict[str, object]] = []
        strict_threshold = self._tile_template_min_score
        relaxed_threshold = max(0.68, strict_threshold - 0.08)

        for idx, template in enumerate(tile_templates):
            template_name = (
                self._tile_template_names[idx] if idx < len(self._tile_template_names) else f"template_{idx}"
            )
            th, tw = template.shape[:2]
            if tw > gray_image.shape[1] or th > gray_image.shape[0]:
                template_maxima.append(-1.0)
                per_template.append(
                    {
                        "name": template_name,
                        "max_score": -1.0,
                        "hit_count": 0,
                    }
                )
                continue

            response = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(response)
            template_maxima.append(float(max_val))
            y_idxs, x_idxs = np.where(response >= strict_threshold)
            hit_count = len(y_idxs)
            per_template.append(
                {
                    "name": template_name,
                    "max_score": float(max_val),
                    "hit_count": int(hit_count),
                }
            )
            for y, x in zip(y_idxs.tolist(), x_idxs.tolist()):
                cx = x + tw / 2.0
                cy = y + th / 2.0
                score = float(response[y, x])
                points.append((cx, cy, int(round((tw + th) / 2.0)), score))

            if relaxed_threshold < strict_threshold:
                ry_idxs, rx_idxs = np.where(response >= relaxed_threshold)
                for y, x in zip(ry_idxs.tolist(), rx_idxs.tolist()):
                    score = float(response[y, x])
                    if score >= strict_threshold:
                        continue
                    cx = x + tw / 2.0
                    cy = y + th / 2.0
                    relaxed_points.append((cx, cy, int(round((tw + th) / 2.0)), score))

        if not points:
            return [], [], {
                "template_count": len(tile_templates),
                "template_maxima": template_maxima,
                "per_template": per_template,
                "raw_hit_count": 0,
                "nms_kept_count": 0,
                "best_score": max(template_maxima) if template_maxima else -1.0,
                "draw_overlay": None,
            }

        # Non-maximum suppression by center distance keeps strongest nearby response.
        points.sort(key=lambda item: item[3], reverse=True)
        min_sep = max(8.0, np.median([point[2] for point in points]) * 0.42)

        def _nms(source_points: list[tuple[float, float, int, float]]) -> list[tuple[float, float, int]]:
            selected: list[tuple[float, float, int]] = []
            for cx, cy, side, _score in source_points:
                if any(abs(cx - kx) <= min_sep and abs(cy - ky) <= min_sep for kx, ky, _ks in selected):
                    continue
                selected.append((cx, cy, side))
            return selected

        def _grid_score(candidate: list[tuple[float, float, int]]) -> tuple[int, int, int, int]:
            if not candidate:
                return (0, -10, -10, 0)
            cand_centers = [(cx, cy) for cx, cy, _ in candidate]
            cand_sides = [side for _cx, _cy, side in candidate]
            tolerance = max(8, int(round(np.median(cand_sides) * 0.35))) if cand_sides else 10
            x_clusters = self._cluster_axis([x for x, _y in cand_centers], tolerance=tolerance)
            y_clusters = self._cluster_axis([y for _x, y in cand_centers], tolerance=tolerance)
            x_count = len(x_clusters)
            y_count = len(y_clusters)
            score_primary = min(x_count, 5) + min(y_count, 5)
            score_shape = -abs(x_count - 5) - abs(y_count - 5)
            # Prefer denser candidates as a final tie-breaker.
            return (score_primary, score_shape, min(len(candidate), 40), -max(0, len(candidate) - 40))

        strict_kept = _nms(points)
        used_relaxed_fallback = False
        if relaxed_points:
            merged_points = points + relaxed_points
            merged_points.sort(key=lambda item: item[3], reverse=True)
            relaxed_kept = _nms(merged_points)

            strict_score = _grid_score(strict_kept)
            relaxed_score = _grid_score(relaxed_kept)
            if relaxed_score > strict_score:
                kept = relaxed_kept
                points = merged_points
                used_relaxed_fallback = True
            else:
                kept = strict_kept
        else:
            kept = strict_kept

        centers = [(cx, cy) for cx, cy, _side in kept]
        sides = [side for _cx, _cy, side in kept]

        overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        for cx, cy, _side, _score in points[:5000]:
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 1, (0, 0, 255), -1)
        for cx, cy, side in kept:
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), max(2, int(round(side * 0.12))), (0, 255, 0), 1)

        best_score = max(score for _cx, _cy, _side, score in points)
        return centers, sides, {
            "template_count": len(tile_templates),
            "template_maxima": template_maxima,
            "per_template": per_template,
            "raw_hit_count": len(points),
            "nms_kept_count": len(kept),
            "best_score": float(best_score),
            "strict_threshold": float(strict_threshold),
            "relaxed_threshold": float(relaxed_threshold),
            "relaxed_hit_count": len(relaxed_points),
            "used_relaxed_fallback": used_relaxed_fallback,
            "draw_overlay": overlay,
        }

    def _find_anchor(self, gray_image: np.ndarray) -> dict[str, float | str] | None:
        anchors = self._load_anchor_templates()
        if not anchors:
            return None

        best_score = -1.0
        best_data: dict[str, float | str] | None = None

        for idx, anchor in enumerate(anchors):
            template_name = (
                self._anchor_template_names[idx]
                if idx < len(self._anchor_template_names)
                else f"anchor_{idx}.png"
            )
            name_lower = template_name.lower()
            is_top_right = (
                "top_right" in name_lower
                or "top-right" in name_lower
                or "_tr_" in name_lower
                or name_lower.startswith("tr_")
                or name_lower.endswith("_tr.png")
                or "anchor_tr" in name_lower
            )
            ah, aw = anchor.shape[:2]
            if aw > gray_image.shape[1] or ah > gray_image.shape[0]:
                continue

            response = cv2.matchTemplate(gray_image, anchor, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)
            if max_val > best_score:
                best_score = float(max_val)
                top_left_x = float(max_loc[0])
                top_left_y = float(max_loc[1])
                is_bottom_left = (
                    "bottom_left" in name_lower
                    or "bottom-left" in name_lower
                    or "_bl_" in name_lower
                    or name_lower.startswith("bl_")
                    or name_lower.endswith("_bl.png")
                    or "anchor_bl" in name_lower
                )
                if is_top_right:
                    reference_x = top_left_x + float(aw)
                    reference_y = top_left_y
                    corner = "top-right"
                elif is_bottom_left:
                    reference_x = top_left_x
                    reference_y = top_left_y + float(ah)
                    corner = "bottom-left"
                else:
                    reference_x = top_left_x
                    reference_y = top_left_y
                    corner = "top-left"
                best_data = {
                    "top_left_x": top_left_x,
                    "top_left_y": top_left_y,
                    "reference_x": reference_x,
                    "reference_y": reference_y,
                    "corner": corner,
                    "template_name": template_name,
                }

        if best_data is None or best_score < 0.52:
            return None
        best_data["score"] = best_score
        return best_data

    def _build_regions(
        self,
        image_w: int,
        image_h: int,
        panel: tuple[int, int, int, int],
        board: tuple[int, int, int, int],
    ) -> list[Region]:
        _px, _py, _pw, _ph = panel
        bx, by, bw, bh = board

        grid_layout = self._grid_layout
        board_cols: list[tuple[int, int]]
        board_rows: list[tuple[int, int]]
        row_rows: list[tuple[int, int]]
        col_cols: list[tuple[int, int]]
        row_x: int
        row_w: int
        col_y: int
        col_h: int

        if grid_layout is not None:
            x_centers = [float(v) for v in grid_layout.get("x_centers", [])]
            y_centers = [float(v) for v in grid_layout.get("y_centers", [])]
            tile_side = int(grid_layout.get("tile_side", 0))
            if len(x_centers) == 5 and len(y_centers) == 5 and tile_side > 0:
                step_x = self._estimate_grid_step(x_centers, tile_side)
                step_y = self._estimate_grid_step(y_centers, tile_side)
                clue_side = int(round(max(18.0, min(min(step_x, step_y) * 0.78, tile_side * 1.05))))
                col_clue_w = int(round(clue_side + max(4.0, step_x * 0.12)))
                col_clue_h = int(round(clue_side + max(4.0, step_y * 0.14)))
                row_gap = int(round(step_x * 0.30))
                col_gap = int(round(step_y * 0.30))
                row_align_shift = int(round(step_y * 0.07))
                col_right_shift = int(round(step_x * 0.03))
                col_up_shift = int(round(step_y * 0.08))

                board_cols = [(int(round(cx - tile_side / 2.0)), tile_side) for cx in x_centers]
                board_rows = [(int(round(cy - tile_side / 2.0)), tile_side) for cy in y_centers]

                row_x = int(round(bx + bw + row_gap))
                row_w = clue_side
                col_y = int(round(by + bh + col_gap - col_up_shift))
                col_h = col_clue_h

                row_rows = [
                    (int(round(cy - clue_side / 2.0 - row_align_shift)), clue_side)
                    for cy in y_centers
                ]
                col_cols = [
                    (int(round(cx - col_clue_w / 2.0 + col_right_shift)), col_clue_w)
                    for cx in x_centers
                ]
            else:
                row_gap = int(round(bw * 0.03))
                row_w = int(round(bw * 0.26))
                row_x = bx + bw + row_gap

                col_gap = int(round(bh * 0.03))
                col_h = int(round(bh * 0.21))
                col_y = by + bh + col_gap

                board_cols = self._split_axis(bx, bw, 5)
                board_rows = self._split_axis(by, bh, 5)
                row_rows = self._split_axis(by, bh, 5)
                col_cols = self._split_axis(bx, bw, 5)
        else:
            row_gap = int(round(bw * 0.03))
            row_w = int(round(bw * 0.26))
            row_x = bx + bw + row_gap

            col_gap = int(round(bh * 0.03))
            col_h = int(round(bh * 0.21))
            col_y = by + bh + col_gap

            board_cols = self._split_axis(bx, bw, 5)
            board_rows = self._split_axis(by, bh, 5)
            row_rows = self._split_axis(by, bh, 5)
            col_cols = self._split_axis(bx, bw, 5)

        regions: list[Region] = []

        for r_idx, (tile_y, tile_h) in enumerate(board_rows):
            for c_idx, (tile_x, tile_w) in enumerate(board_cols):
                regions.append(
                    self._make_region(
                        f"({r_idx},{c_idx})",
                        tile_x,
                        tile_y,
                        tile_w,
                        tile_h,
                        image_w,
                        image_h,
                    )
                )

        for r_idx, (clue_y, clue_h) in enumerate(row_rows):
            regions.append(
                self._make_region(
                    f"r{r_idx}",
                    row_x,
                    clue_y,
                    row_w,
                    clue_h,
                    image_w,
                    image_h,
                )
            )

        for c_idx, (clue_x, clue_w) in enumerate(col_cols):
            regions.append(
                self._make_region(
                    f"c{c_idx}",
                    clue_x,
                    col_y,
                    clue_w,
                    col_h,
                    image_w,
                    image_h,
                )
            )

        return regions

    def _refine_clue_regions_with_templates(
        self,
        image: np.ndarray,
        regions: list[Region],
        board: tuple[int, int, int, int],
    ) -> list[Region]:
        templates = self._load_clue_templates()
        if not templates:
            raise ValueError("No clue templates found; cannot locate clue boxes.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        region_by_name = {region.name: region for region in regions}

        expected_rows: list[Region] = [
            region_by_name[f"r{idx}"] for idx in range(5) if f"r{idx}" in region_by_name
        ]
        expected_cols: list[Region] = [
            region_by_name[f"c{idx}"] for idx in range(5) if f"c{idx}" in region_by_name
        ]
        if len(expected_rows) != 5 or len(expected_cols) != 5:
            raise ValueError("Could not find all 5 expected row and column clue regions.")

        row_matches = self._match_clue_regions(gray, expected_rows, templates, axis="row")
        if row_matches is None:
            raise ValueError("Template row clue matching was low confidence.")
        col_matches = self._match_clue_regions(gray, expected_cols, templates, axis="col")
        if col_matches is None:
            raise ValueError("Template column clue matching was low confidence.")

        accepted_rows = self._accept_template_clue_matches(expected_rows, row_matches, board, axis="row")
        accepted_cols = self._accept_template_clue_matches(expected_cols, col_matches, board, axis="col")

        sorted_rows = sorted(accepted_rows, key=lambda region: region.y)
        sorted_cols = sorted(accepted_cols, key=lambda region: region.x)

        merged: list[Region] = []
        for region in regions:
            if region.name.startswith("r") and region.name[1:].isdigit():
                merged.append(sorted_rows[int(region.name[1:])])
                continue
            if region.name.startswith("c") and region.name[1:].isdigit():
                merged.append(sorted_cols[int(region.name[1:])])
                continue
            merged.append(region)

        return merged

    def _accept_template_clue_matches(
        self,
        expected_regions: list[Region],
        candidate_regions: list[Region],
        board: tuple[int, int, int, int],
        *,
        axis: str,
    ) -> list[Region]:
        if len(expected_regions) != len(candidate_regions):
            raise ValueError(
                f"Rejected template {axis} clue matches due to count mismatch."
            )

        sort_key = (lambda region: region.y) if axis == "row" else (lambda region: region.x)
        expected_sorted = sorted(expected_regions, key=sort_key)
        candidates_sorted = sorted(candidate_regions, key=sort_key)

        bx, by, bw, bh = board
        board_right = bx + bw
        board_bottom = by + bh

        accepted: list[Region] = []
        for idx, (expected, candidate) in enumerate(zip(expected_sorted, candidates_sorted)):
            exp_cx = expected.x + expected.w / 2.0
            exp_cy = expected.y + expected.h / 2.0
            cand_cx = candidate.x + candidate.w / 2.0
            cand_cy = candidate.y + candidate.h / 2.0

            if axis == "row":
                primary_delta = abs(cand_cy - exp_cy)
                secondary_delta = abs(cand_cx - exp_cx)
                primary_tol = max(4.0, expected.h * 0.40)
                secondary_tol = max(8.0, expected.w * 0.85)
                if candidate.x < board_right - 2:
                    raise ValueError(
                        "Rejected template row clue matches that crossed into board area."
                    )
            else:
                primary_delta = abs(cand_cx - exp_cx)
                secondary_delta = abs(cand_cy - exp_cy)
                primary_tol = max(4.0, expected.w * 0.40)
                secondary_tol = max(8.0, expected.h * 0.85)
                if candidate.y < board_bottom - 2:
                    raise ValueError(
                        "Rejected template column clue matches that crossed into board area."
                    )

            width_delta = abs(candidate.w - expected.w)
            height_delta = abs(candidate.h - expected.h)
            width_tol = max(6.0, expected.w * 0.60)
            height_tol = max(6.0, expected.h * 0.60)

            if (
                primary_delta > primary_tol
                or secondary_delta > secondary_tol
                or width_delta > width_tol
                or height_delta > height_tol
            ):
                raise ValueError(
                    f"Rejected template {axis} clue matches due to large offset/scale drift."
                )

            accepted.append(
                Region(
                    name=f"{axis[0]}{idx}",
                    x=candidate.x,
                    y=candidate.y,
                    w=candidate.w,
                    h=candidate.h,
                )
            )

        return accepted

    def _match_clue_regions(
        self,
        gray_image: np.ndarray,
        expected_regions: list[Region],
        templates: list[np.ndarray],
        *,
        axis: str,
    ) -> list[Region] | None:
        image_h, image_w = gray_image.shape[:2]
        matched: list[Region] = []

        for idx, expected in enumerate(expected_regions):
            if axis == "row":
                pad_x = int(round(expected.w * 0.9))
                pad_y = int(round(expected.h * 0.4))
            else:
                pad_x = int(round(expected.w * 0.4))
                pad_y = int(round(expected.h * 1.1))

            sx = max(0, expected.x - pad_x)
            sy = max(0, expected.y - pad_y)
            ex = min(image_w, expected.x + expected.w + pad_x)
            ey = min(image_h, expected.y + expected.h + pad_y)
            search = gray_image[sy:ey, sx:ex]
            if search.size == 0:
                return None

            expected_area = expected.w * expected.h
            best_score = -1.0
            best_rect: tuple[int, int, int, int] | None = None

            for template in templates:
                th, tw = template.shape[:2]
                if tw > search.shape[1] or th > search.shape[0]:
                    continue

                area_ratio = (tw * th) / max(expected_area, 1)
                if not (0.35 <= area_ratio <= 1.85):
                    continue

                response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)
                if max_val > best_score:
                    best_score = float(max_val)
                    best_rect = (sx + max_loc[0], sy + max_loc[1], tw, th)

            # Minimum confidence tuned to avoid replacing decent heuristic boxes with noise.
            if best_rect is None or best_score < self.CLUE_TEMPLATE_MIN_SCORE:
                return None

            x, y, w, h = best_rect
            matched.append(
                Region(
                    name=f"{axis[0]}{idx}",
                    x=max(0, min(x, image_w - 1)),
                    y=max(0, min(y, image_h - 1)),
                    w=max(1, min(w, image_w - x)),
                    h=max(1, min(h, image_h - y)),
                )
            )

        return matched

    def _estimate_grid_step(self, centers: list[float], fallback: int) -> float:
        if len(centers) < 2:
            return float(max(1, fallback))

        ordered = sorted(centers)
        diffs = [ordered[idx + 1] - ordered[idx] for idx in range(len(ordered) - 1)]
        positive = [diff for diff in diffs if diff > 1.0]
        if not positive:
            return float(max(1, fallback))
        return float(np.median(positive))

    def _load_clue_templates(self) -> list[np.ndarray]:
        if self._clue_templates:
            return self._clue_templates

        repo_root = Path(__file__).resolve().parents[3]
        templates_dir = repo_root / "assets" / "templates"
        if not templates_dir.exists():
            return []

        candidates = sorted(
            path
            for path in templates_dir.glob("*.png")
            if "clue" in path.stem.lower() and not path.stem.lower().endswith("tmp")
        )
        if not candidates:
            return []

        loaded: list[np.ndarray] = []
        max_area = 0
        raw_templates: list[np.ndarray] = []
        for path in candidates:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            h, w = template.shape[:2]
            if h < 8 or w < 8:
                continue
            raw_templates.append(template)
            max_area = max(max_area, h * w)

        # Keep larger clue-box-like templates, skip tiny glyph-only crops.
        min_area = int(round(max_area * 0.55)) if max_area > 0 else 0
        for template in raw_templates:
            h, w = template.shape[:2]
            if h * w < min_area:
                continue
            loaded.append(template)

        self._clue_templates = loaded
        return self._clue_templates

    def _load_tile_templates(self) -> list[np.ndarray]:
        if self._tile_templates:
            return self._tile_templates

        templates, names = self._load_templates_by_keyword("tile")
        self._tile_templates = templates
        self._tile_template_names = names
        return self._tile_templates

    @staticmethod
    def _is_tr_anchor_name(name: str) -> bool:
        nl = name.lower()
        return (
            "top_right" in nl or "top-right" in nl or "_tr_" in nl
            or nl.startswith("tr_") or nl.endswith("_tr.png") or "anchor_tr" in nl
        )

    @staticmethod
    def _is_bl_anchor_name(name: str) -> bool:
        nl = name.lower()
        return (
            "bottom_left" in nl or "bottom-left" in nl or "_bl_" in nl
            or nl.startswith("bl_") or nl.endswith("_bl.png") or "anchor_bl" in nl
        )

    def _load_anchor_templates(self) -> list[np.ndarray]:
        if self._anchor_templates:
            return self._anchor_templates

        templates, names = self._load_templates_by_keyword("anchor")

        # Populate BL cache as a side effect while we have all templates in hand.
        if not self._bl_anchor_templates:
            bl_pairs = [(t, n) for t, n in zip(templates, names) if self._is_bl_anchor_name(n)]
            self._bl_anchor_templates = [t for t, _ in bl_pairs]
            self._bl_anchor_template_names = [n for _, n in bl_pairs]

        preferred_templates: list[np.ndarray] = []
        preferred_names: list[str] = []

        for template, name in zip(templates, names):
            if self._is_tr_anchor_name(name):
                preferred_templates.append(template)
                preferred_names.append(name)

        if preferred_templates:
            self._anchor_templates = preferred_templates
            self._anchor_template_names = preferred_names
            return self._anchor_templates

        self._anchor_templates = templates
        self._anchor_template_names = names
        return self._anchor_templates

    def _load_bl_anchor_templates(self) -> tuple[list[np.ndarray], list[str]]:
        """Return bottom-left corner anchor templates (lazy-loaded)."""
        if not self._bl_anchor_templates:
            self._load_anchor_templates()  # populates BL cache as a side effect
        return self._bl_anchor_templates, self._bl_anchor_template_names

    def _match_corner_anchor(
        self,
        gray: np.ndarray,
        templates: list[np.ndarray],
        names: list[str],
        corner: str,
    ) -> dict[str, float] | None:
        """Find the best-scoring template match for one corner type.

        Returns a dict with keys ``reference_x``, ``reference_y``, ``score``,
        or ``None`` when no template scores above the minimum threshold.
        """
        _MIN_SCORE = 0.52
        best_score = -1.0
        best: dict[str, float] | None = None
        for template, _name in zip(templates, names):
            ah, aw = template.shape[:2]
            if aw > gray.shape[1] or ah > gray.shape[0]:
                continue
            response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if max_val > best_score:
                best_score = float(max_val)
                tlx, tly = float(max_loc[0]), float(max_loc[1])
                if corner == "top-right":
                    ref_x, ref_y = tlx + float(aw), tly
                else:  # bottom-left
                    ref_x, ref_y = tlx, tly + float(ah)
                best = {"reference_x": ref_x, "reference_y": ref_y, "score": best_score}
        return best if best is not None and best_score >= _MIN_SCORE else None

    def find_board_corner_rect(self, image_path: str) -> tuple[int, int, int, int] | None:
        """Match top-right and bottom-left anchor templates and return the board bounding box.

        Returns ``(left, top, right, bottom)`` in image pixels if both corners are
        found with sufficient confidence, otherwise ``None``.
        """
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        tr_templates = self._load_anchor_templates()
        tr_names = self._anchor_template_names
        bl_templates, bl_names = self._load_bl_anchor_templates()
        if not tr_templates or not bl_templates:
            return None
        tr = self._match_corner_anchor(gray, tr_templates, tr_names, "top-right")
        bl = self._match_corner_anchor(gray, bl_templates, bl_names, "bottom-left")
        if tr is None or bl is None:
            return None
        left = int(round(bl["reference_x"]))
        top = int(round(tr["reference_y"]))
        right = int(round(tr["reference_x"]))
        bottom = int(round(bl["reference_y"]))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _load_templates_by_keyword(self, keyword: str) -> tuple[list[np.ndarray], list[str]]:
        repo_root = Path(__file__).resolve().parents[3]
        templates_dir = repo_root / "assets" / "templates"
        if not templates_dir.exists():
            return [], []

        candidates = sorted(
            path
            for path in templates_dir.glob("*.png")
            if keyword in path.stem.lower()
            and not path.stem.lower().endswith("tmp")
            and not path.stem.lower().startswith("test_")
            and "_test" not in path.stem.lower()
        )

        loaded: list[np.ndarray] = []
        names: list[str] = []
        for path in candidates:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            h, w = template.shape[:2]
            if h < 8 or w < 8:
                continue
            loaded.append(template)
            names.append(path.name)
        return loaded, names

    def _begin_debug_run(self, image_path: str, image: np.ndarray) -> None:
        if self._debug_dir is None:
            self._debug_run_dir = None
            return

        run_name = f"{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir = self._debug_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._debug_run_dir = run_dir

        self._debug_log(f"input={image_path}")
        self._debug_log(f"image_shape={image.shape[1]}x{image.shape[0]}")
        self._debug_write_image("input.png", image)

    def _end_debug_run(self) -> None:
        self._debug_run_dir = None

    def _debug_log(self, message: str) -> None:
        if self._debug_run_dir is None:
            return
        log_path = self._debug_run_dir / "debug.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    def _debug_write_image(self, name: str, image: np.ndarray) -> None:
        if self._debug_run_dir is None:
            return
        cv2.imwrite(str(self._debug_run_dir / name), image)

    def _make_region(
        self,
        name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        image_w: int,
        image_h: int,
    ) -> Region:
        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))
        return Region(name=name, x=x, y=y, w=w, h=h)

    def _cluster_axis(self, values: list[float], tolerance: int) -> list[tuple[float, int]]:
        if not values:
            return []

        values = sorted(values)
        clusters: list[list[float]] = [[values[0]]]
        for value in values[1:]:
            if abs(value - clusters[-1][-1]) <= tolerance:
                clusters[-1].append(value)
                continue
            clusters.append([value])
        return [(sum(cluster) / len(cluster), len(cluster)) for cluster in clusters]

    def _select_grid_clusters(
        self,
        clusters: list[tuple[float, int]],
        *,
        expected: int,
    ) -> list[tuple[float, int]]:
        if len(clusters) <= expected:
            return sorted(clusters, key=lambda item: item[0])

        best_subset: list[tuple[float, int]] | None = None
        best_key: tuple[float, float, float] | None = None

        for subset in itertools.combinations(clusters, expected):
            ordered = sorted(subset, key=lambda item: item[0])
            centers = [item[0] for item in ordered]
            counts = [item[1] for item in ordered]
            steps = [centers[idx + 1] - centers[idx] for idx in range(expected - 1)]
            if any(step <= 0 for step in steps):
                continue

            median_step = float(np.median(steps))
            if median_step <= 0:
                continue
            # Prefer regular spacing first, then stronger support and larger span.
            spacing_mad = float(np.median([abs(step - median_step) for step in steps]))
            spacing_norm = spacing_mad / median_step
            support = float(sum(counts))
            span = centers[-1] - centers[0]
            key = (spacing_norm, -support, -span)

            if best_key is None or key < best_key:
                best_key = key
                best_subset = ordered

        if best_subset is not None:
            return best_subset
        return sorted(clusters, key=lambda item: item[1], reverse=True)[:expected]

    def _split_axis(self, start: int, total: int, count: int) -> list[tuple[int, int]]:
        if count <= 0:
            return []

        end = start + max(1, total)
        edges = [start + int(round(i * (end - start) / count)) for i in range(count + 1)]

        spans: list[tuple[int, int]] = []
        for idx in range(count):
            axis_start = edges[idx]
            axis_end = edges[idx + 1]
            spans.append((axis_start, max(1, axis_end - axis_start)))
        return spans

    def _label_color(self, idx: int) -> tuple[int, int, int]:
        palette = [
            (15, 188, 249),
            (255, 158, 0),
            (32, 201, 151),
            (255, 99, 132),
            (153, 102, 255),
            (255, 205, 86),
            (0, 200, 83),
        ]
        return palette[idx % len(palette)]


def main() -> int:
    arg_parser = argparse.ArgumentParser(description="Label Voltorb Flip screenshot regions")
    arg_parser.add_argument("input_image", help="Path to screenshot image")
    arg_parser.add_argument("output_image", help="Path to save labeled image")
    args = arg_parser.parse_args()

    parser = ScreenBoardParser()
    result = parser.annotate(args.input_image, args.output_image)
    print(f"Labeled image written to: {args.output_image}")
    for region in result.regions:
        method = result.region_methods.get(region.name, "unknown")
        print(
            f"- {region.name}: x={region.x}, y={region.y}, w={region.w}, h={region.h}, method={method}"
        )
    print(f"Method summary: {result.method_summary()}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())