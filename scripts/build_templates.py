#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def compute_default_size(paths: list[Path]) -> tuple[int, int]:
    widths: list[int] = []
    heights: list[int] = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)
    return int(median(widths)), int(median(heights))


def process_image(src: Path, dst: Path, size: tuple[int, int] | None) -> tuple[int, int]:
    with Image.open(src) as image:
        gray = ImageOps.grayscale(image)
        if size is None:
            output = gray
        else:
            # Nearest-neighbor keeps pixel-art edges sharp.
            output = gray.resize(size, Image.Resampling.NEAREST)

    dst.parent.mkdir(parents=True, exist_ok=True)
    output.save(dst.with_suffix(".png"), format="PNG")
    return output.size


def build_templates(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int] | None,
    suffix: str,
) -> int:
    images = find_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return 1

    print(f"Found {len(images)} raw template image(s)")
    if size is None:
        print("Target size: keep original dimensions")
    else:
        print(f"Target size: {size[0]}x{size[1]}")
    print(f"Output directory: {output_dir}")

    for src in images:
        rel = src.relative_to(input_dir)
        out_name = f"{rel.stem}{suffix}"
        dst = output_dir / rel.parent / out_name
        out_w, out_h = process_image(src, dst, size)
        print(f"wrote: {dst.with_suffix('.png')} ({out_w}x{out_h})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw template screenshots to grayscale templates."
    )
    parser.add_argument(
        "--input",
        default="assets/templates/raw",
        type=Path,
        help="Input directory with raw template screenshots.",
    )
    parser.add_argument(
        "--output",
        default="assets/templates",
        type=Path,
        help="Output directory for processed templates.",
    )
    parser.add_argument("--width", type=int, default=0, help="Optional target width.")
    parser.add_argument("--height", type=int, default=0, help="Optional target height.")
    parser.add_argument(
        "--resize-to-median",
        action="store_true",
        help="Resize all templates to median raw size. By default, original dimensions are preserved.",
    )
    parser.add_argument(
        "--suffix",
        default="_tpl",
        help="Suffix added to each processed filename before .png.",
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    images = find_images(input_dir)
    if args.width > 0 and args.height > 0:
        target_size = (args.width, args.height)
    elif args.resize_to_median:
        if not images:
            print(f"No images found in: {input_dir}")
            return 1
        target_size = compute_default_size(images)
    else:
        target_size = None

    return build_templates(input_dir, output_dir, target_size, args.suffix)


if __name__ == "__main__":
    raise SystemExit(main())
