"""Turn hand-picked images into a kohya_ss / OneTrainer LoRA dataset.

Copy the 25-40 best, most consistent images into one folder, then:
    python -m pipeline.dataset --persona personas/ava.yaml --src output/ava/picked --out datasets/ava

Produces datasets/ava/img/<repeats>_<trigger>/ with a resized PNG and a .txt caption
per image. Captions are "<trigger>, <shot prompt>": the scene is described so the
LoRA learns that only the face/body belongs to the trigger word.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

from .config import load_persona

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def scenes_from_log(log_path: Path) -> dict[str, str]:
    """Map image file name -> the shot's scene text, from generate's log.csv."""
    if not log_path.exists():
        return {}
    with open(log_path, newline="", encoding="utf-8") as f:
        return {Path(row["file"]).name: row.get("scene", "") for row in csv.DictReader(f)}


def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--src", required=True, help="folder of hand-picked images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="generate log.csv for captions (default: output/<slug>/log.csv)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args(argv)

    persona = load_persona(args.persona)
    if not persona.trigger:
        ap.error("set 'trigger' in the persona file first (e.g. 'avlx woman')")
    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        ap.error(f"no images in {src}")
    if len(files) < 15:
        print(f"warning: only {len(files)} images; 25-40 gives a more reliable LoRA", file=sys.stderr)

    scenes = scenes_from_log(Path(args.log or f"output/{persona.slug}/log.csv"))
    dest = Path(args.out) / "img" / f"{args.repeats}_{persona.trigger}"
    dest.mkdir(parents=True, exist_ok=True)

    uncaptioned = 0
    for i, path in enumerate(files, 1):
        with Image.open(path) as im:
            img = resize_max(im.convert("RGB"), args.max_side)
        stem = f"{persona.slug}_{i:03d}"
        img.save(dest / f"{stem}.png")
        scene = scenes.get(path.name, "")
        if not scene:
            uncaptioned += 1
        caption = f"{persona.trigger}, {scene}" if scene else persona.trigger
        (dest / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")

    print(f"{len(files)} images -> {dest}")
    if uncaptioned:
        print(f"{uncaptioned} image(s) not found in the log got a trigger-only caption; "
              f"edit their .txt files to describe outfit/pose/background.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
