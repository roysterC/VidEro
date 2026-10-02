"""Render a shot list for a persona through a running ComfyUI.

Examples:
    python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml
    python -m pipeline.generate --persona personas/ava.yaml --shots shots/week-example.yaml --preset quality
    python -m pipeline.generate --persona personas/ava.yaml --shots shots/02-dataset.yaml \
        --template workflows/faceref.json --no-lora
    python -m pipeline.generate ... --dry-run      # print prompts, render nothing
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

from .comfy import ComfyClient, ComfyError
from .config import SIZES, load_persona, load_render_config, load_shots
from .prompts import GuardrailError, build_prompts
from .workflows import fill_template, load_template, sdxl_graph

LOG_FIELDS = [
    "timestamp", "file", "shot_set", "shot_id", "tier", "seed", "preset", "checkpoint",
    "template", "lora", "lora_strength", "width", "height", "scene", "prompt", "negative", "prompt_id",
]


def plan_jobs(persona, shots, *, use_lora: bool, only: list[str] | None, count: int | None):
    """Expand shots into individual renders. Raises GuardrailError before anything is queued."""
    jobs = []
    for shot in shots:
        if only and shot.id not in only:
            continue
        positive, negative = build_prompts(persona, shot, use_lora)
        n = count or shot.count
        for i in range(n):
            seed = shot.seed + i if shot.seed is not None else random.randint(0, 2**48)
            jobs.append({"shot": shot, "seed": seed, "positive": positive, "negative": negative})
    if only and not jobs:
        raise SystemExit(f"no shots matched --only {only}")
    return jobs


def append_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--config", default="config/render.yaml")
    ap.add_argument("--preset", help="render preset from the config (default: config's default_preset)")
    ap.add_argument("--template", help="API-format ComfyUI workflow with %%PROMPT%%/%%NEGATIVE%% placeholders")
    ap.add_argument("--only", nargs="+", metavar="SHOT_ID", help="render only these shot ids")
    ap.add_argument("--count", type=int, help="override images per shot")
    ap.add_argument("--no-lora", action="store_true", help="ignore the persona LoRA even if one is set")
    ap.add_argument("--out", default="output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_render_config(args.config)
    preset_name = args.preset or cfg.default_preset
    if preset_name not in cfg.presets:
        ap.error(f"unknown preset {preset_name!r}; choose from {list(cfg.presets)}")
    preset = cfg.presets[preset_name]
    persona = load_persona(args.persona)
    shots = load_shots(args.shots)
    use_lora = bool(persona.lora_file) and not args.no_lora and not args.template
    template = load_template(args.template) if args.template else None

    try:
        jobs = plan_jobs(persona, shots, use_lora=use_lora, only=args.only, count=args.count)
    except GuardrailError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    shot_set = Path(args.shots).stem
    print(f"{len(jobs)} image(s) for {persona.name} from {shot_set} "
          f"[preset={preset_name}, lora={'on' if use_lora else 'off'}, template={args.template or 'built-in SDXL'}]")

    if args.dry_run:
        seen = set()
        for j in jobs:
            if j["shot"].id in seen:
                continue
            seen.add(j["shot"].id)
            print(f"\n[{j['shot'].id}] tier={j['shot'].tier} size={j['shot'].size}\n  + {j['positive']}\n  - {j['negative']}")
        return 0

    client = ComfyClient(cfg.comfy_url)
    try:
        if not template:
            ckpts = client.list_options("CheckpointLoaderSimple", "ckpt_name")
            if preset.checkpoint not in ckpts:
                print(f"checkpoint {preset.checkpoint!r} not found in ComfyUI/models/checkpoints. "
                      f"Installed: {ckpts}", file=sys.stderr)
                return 1
            if use_lora:
                loras = client.list_options("LoraLoader", "lora_name")
                if persona.lora_file not in loras:
                    print(f"LoRA {persona.lora_file!r} not found in ComfyUI/models/loras. Installed: {loras}",
                          file=sys.stderr)
                    return 1
    except ComfyError as e:
        print(e, file=sys.stderr)
        return 1

    out_dir = Path(args.out) / persona.slug / shot_set
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out) / persona.slug / "log.csv"

    for n, j in enumerate(jobs, 1):
        shot = j["shot"]
        width, height = SIZES[shot.size]
        prefix = f"videro/{persona.slug}/{shot_set}/{shot.id}"
        if template:
            graph = fill_template(template, positive=j["positive"], negative=j["negative"],
                                  seed=j["seed"], width=width, height=height, prefix=prefix)
        else:
            graph = sdxl_graph(preset=preset, positive=j["positive"], negative=j["negative"],
                               width=width, height=height, seed=j["seed"], prefix=prefix,
                               lora=(persona.lora_file, persona.lora_strength) if use_lora else None)
        try:
            prompt_id = client.queue(graph)
            entry = client.wait(prompt_id)
            images = client.output_images(entry)
            if not images:
                raise ComfyError("render finished but produced no output image")
        except ComfyError as e:
            print(f"[{n}/{len(jobs)}] {shot.id} seed={j['seed']}: {e}", file=sys.stderr)
            return 1

        for k, img in enumerate(images):
            suffix = f"_{k}" if len(images) > 1 else ""
            dest = out_dir / f"{shot.id}_{j['seed']}{suffix}.png"
            dest.write_bytes(client.download(img))
            append_log(log_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file": dest.as_posix(), "shot_set": shot_set, "shot_id": shot.id, "tier": shot.tier,
                "seed": j["seed"], "preset": preset_name,
                "checkpoint": "" if template else preset.checkpoint,
                "template": args.template or "",
                "lora": persona.lora_file if use_lora else "",
                "lora_strength": persona.lora_strength if use_lora else "",
                "width": width, "height": height, "scene": shot.prompt,
                "prompt": j["positive"], "negative": j["negative"], "prompt_id": prompt_id,
            })
            print(f"[{n}/{len(jobs)}] {dest}")

    print(f"done -> {out_dir}  (log: {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
