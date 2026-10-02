"""Crop and resize finished images for social platforms.

    python -m pipeline.export --src output/ava/week-example/picked --format feed story

Output is JPEG with all metadata dropped. ComfyUI PNGs embed the full workflow
(prompts, seeds, model names) in the file; never upload the raw PNGs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

FORMATS = {
    "feed": (1080, 1350),    # Instagram / Facebook / Threads 4:5 portrait
    "story": (1080, 1920),   # Stories, Reels/TikTok covers, 9:16
    "square": (1080, 1080),
    "x": (1200, 1500),       # X shows 4:5 uncropped in timeline
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def crop_to_aspect(img: Image.Image, width: int, height: int, anchor: float = 0.35) -> Image.Image:
    """Crop to width:height. ``anchor`` is the vertical position kept when trimming height
    (0 = keep top, 0.5 = center); faces usually sit in the upper third."""
    w, h = img.size
    target = width / height
    if w / h > target:  # too wide: trim sides evenly
        new_w = round(h * target)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:  # too tall: trim top/bottom around the anchor
        new_h = round(w / target)
        top = round((h - new_h) * anchor)
        box = (0, top, w, top + new_h)
    return img.crop(box).resize((width, height), Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of picked images")
    ap.add_argument("--format", nargs="+", choices=list(FORMATS) + ["all"], default=["feed"])
    ap.add_argument("--anchor", type=float, default=0.35, help="vertical crop anchor 0..1 (default 0.35)")
    ap.add_argument("--out", default="exports")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args(argv)

    formats = list(FORMATS) if "all" in args.format else args.format
    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        ap.error(f"no images in {src}")

    for fmt in formats:
        w, h = FORMATS[fmt]
        dest = Path(args.out) / fmt
        dest.mkdir(parents=True, exist_ok=True)
        for path in files:
            with Image.open(path) as im:
                out = crop_to_aspect(im.convert("RGB"), w, h, args.anchor)
            # A fresh RGB image saved without exif/pnginfo carries no metadata.
            out.save(dest / f"{path.stem}.jpg", "JPEG", quality=args.quality, optimize=True)
        print(f"{len(files)} image(s) -> {dest} ({w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
