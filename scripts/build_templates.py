#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def process_image(src: Path, dst: Path) -> tuple[int, int]:
    with Image.open(src) as image:
        output = ImageOps.grayscale(image)

    dst.parent.mkdir(parents=True, exist_ok=True)
    output.save(dst.with_suffix(".png"), format="PNG")
    return output.size


def build_templates(input_dir: Path, output_dir: Path, suffix: str) -> int:
    images = find_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return 1

    print(f"Found {len(images)} raw template image(s)")
    print(f"Output directory: {output_dir}")

    for src in images:
        rel = src.relative_to(input_dir)
        out_name = f"{rel.stem}{suffix}"
        dst = output_dir / rel.parent / out_name
        out_w, out_h = process_image(src, dst)
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
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix added to each processed filename before .png.",
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    return build_templates(input_dir, output_dir, args.suffix)


if __name__ == "__main__":
    raise SystemExit(main())
