#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def _prepare(image: np.ndarray, *, blur: bool, binary: bool) -> np.ndarray:
    prepared = image
    if blur:
        prepared = cv2.GaussianBlur(prepared, (3, 3), 0)
    if binary:
        prepared = cv2.threshold(prepared, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return prepared


def _same_size_score(template: np.ndarray, candidate: np.ndarray) -> float:
    if template.shape != candidate.shape:
        candidate = cv2.resize(
            candidate,
            (template.shape[1], template.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    result = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
    # For equal-size images, matchTemplate returns a 1x1 response map.
    return float(result[0, 0])


def _sliding_score(template: np.ndarray, candidate: np.ndarray) -> tuple[float, tuple[int, int], tuple[int, int]]:
    if candidate.shape[0] < template.shape[0] or candidate.shape[1] < template.shape[1]:
        scale = min(
            candidate.shape[1] / template.shape[1],
            candidate.shape[0] / template.shape[0],
        )
        scale = max(scale, 1e-3)
        new_w = max(1, int(round(template.shape[1] * scale)))
        new_h = max(1, int(round(template.shape[0] * scale)))
        template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    response = cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response)
    top_left = (int(max_loc[0]), int(max_loc[1]))
    size = (int(template.shape[1]), int(template.shape[0]))
    return float(max_val), top_left, size


def _label(score: float) -> str:
    if score >= 0.92:
        return "excellent"
    if score >= 0.82:
        return "strong"
    if score >= 0.68:
        return "moderate"
    if score >= 0.50:
        return "weak"
    return "poor"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score how well a candidate image matches a tile template."
    )
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="Path to template image (for example assets/templates/game_tile_0_tpl.png)",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to candidate image to compare against template",
    )
    parser.add_argument(
        "--mode",
        choices=("same", "sliding"),
        default="same",
        help="same: compare full crop vs template (resizes candidate to template size). "
        "sliding: find best match location inside candidate.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Apply Otsu binarization to both images before matching.",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Apply slight blur before matching.",
    )

    args = parser.parse_args()

    template = _prepare(_load_gray(args.template), blur=args.blur, binary=args.binary)
    candidate = _prepare(_load_gray(args.candidate), blur=args.blur, binary=args.binary)

    print(f"Template: {args.template} ({template.shape[1]}x{template.shape[0]})")
    print(f"Candidate: {args.candidate} ({candidate.shape[1]}x{candidate.shape[0]})")
    print(f"Mode: {args.mode} | binary={args.binary} | blur={args.blur}")

    if args.mode == "same":
        score = _same_size_score(template, candidate)
        print(f"Match score: {score:.4f} ({score * 100:.1f}%)")
        print(f"Quality: {_label(score)}")
        return 0

    score, top_left, match_size = _sliding_score(template, candidate)
    print(f"Best match score: {score:.4f} ({score * 100:.1f}%)")
    print(f"Quality: {_label(score)}")
    print(f"Best match top-left: {top_left}")
    print(f"Matched template size: {match_size[0]}x{match_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
