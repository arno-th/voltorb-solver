from __future__ import annotations

from dataclasses import dataclass
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

    def by_name(self) -> dict[str, Region]:
        return {region.name: region for region in self.regions}


class ScreenBoardParser:
    """Detects coarse Voltorb Flip regions from a full screenshot."""

    def parse(self, image_path: str) -> ScreenParseResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_h, image_w = image.shape[:2]
        warnings: list[str] = []

        panel = self._detect_game_panel(image)
        if panel is None:
            warnings.append("Could not detect game panel from screenshot; using full image.")
            panel = (0, 0, image_w, image_h)

        board = self._detect_board_grid(image, panel)
        if board is None:
            warnings.append("Could not detect board grid by tile contours; using panel-relative fallback.")
            board = self._fallback_board(panel)

        regions = self._build_regions(image_w, image_h, panel, board)
        return ScreenParseResult(
            image_width=image_w,
            image_height=image_h,
            regions=regions,
            warnings=warnings,
        )

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

    def _detect_board_grid(
        self, image: np.ndarray, panel: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        px, py, pw, ph = panel
        roi = image[py : py + ph, px : px + pw]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_side = max(14, int(min(pw, ph) * 0.035))
        max_side = max(30, int(min(pw, ph) * 0.17))
        boxes: list[tuple[float, float, int]] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < min_side or h < min_side or w > max_side or h > max_side:
                continue
            aspect = w / max(h, 1)
            if not 0.75 <= aspect <= 1.25:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            side = int(round((w + h) / 2.0))
            boxes.append((cx, cy, side))

        if len(boxes) < 18:
            return None

        x_clusters = self._cluster_axis([item[0] for item in boxes], tolerance=10)
        y_clusters = self._cluster_axis([item[1] for item in boxes], tolerance=10)
        if len(x_clusters) < 5 or len(y_clusters) < 5:
            return None

        x_clusters = sorted(x_clusters, key=lambda c: c[1], reverse=True)[:5]
        y_clusters = sorted(y_clusters, key=lambda c: c[1], reverse=True)[:5]
        x_centers = sorted(cluster[0] for cluster in x_clusters)
        y_centers = sorted(cluster[0] for cluster in y_clusters)

        side = int(round(np.median([item[2] for item in boxes])))
        board_x = int(round(px + x_centers[0] - side / 2))
        board_y = int(round(py + y_centers[0] - side / 2))
        board_w = int(round((x_centers[-1] - x_centers[0]) + side))
        board_h = int(round((y_centers[-1] - y_centers[0]) + side))
        return board_x, board_y, board_w, board_h

    def _fallback_board(self, panel: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        px, py, pw, ph = panel
        board_w = int(round(pw * 0.57))
        board_h = int(round(ph * 0.42))
        board_x = int(round(px + pw * 0.02))
        board_y = int(round(py + ph * 0.50))
        return board_x, board_y, board_w, board_h

    def _build_regions(
        self,
        image_w: int,
        image_h: int,
        panel: tuple[int, int, int, int],
        board: tuple[int, int, int, int],
    ) -> list[Region]:
        px, py, pw, ph = panel
        bx, by, bw, bh = board

        row_gap = int(round(bw * 0.03))
        row_w = int(round(bw * 0.26))
        row_h = bh
        row_x = bx + bw + row_gap
        row_y = by

        col_gap = int(round(bh * 0.03))
        col_h = int(round(bh * 0.21))
        col_x = bx
        col_y = by + bh + col_gap
        col_w = bw

        header_h = int(round(ph * 0.17))
        score_h = int(round(ph * 0.24))
        controls_h = int(round(ph * 0.12))

        regions = [
            self._make_region("game_panel", px, py, pw, ph, image_w, image_h),
            self._make_region("header", px, py, pw, header_h, image_w, image_h),
            self._make_region(
                "score_area",
                px,
                py + header_h,
                pw,
                score_h,
                image_w,
                image_h,
            ),
            self._make_region("board_grid", bx, by, bw, bh, image_w, image_h),
            self._make_region("row_clues", row_x, row_y, row_w, row_h, image_w, image_h),
            self._make_region("col_clues", col_x, col_y, col_w, col_h, image_w, image_h),
            self._make_region(
                "controls",
                px,
                py + ph - controls_h,
                pw,
                controls_h,
                image_w,
                image_h,
            ),
        ]
        return regions

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
        print(f"- {region.name}: x={region.x}, y={region.y}, w={region.w}, h={region.h}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())