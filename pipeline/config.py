"""Loading of persona, shot list and render config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SIZES = {
    "portrait": (832, 1216),   # 2:3, crops cleanly to Instagram 4:5
    "tall": (768, 1344),       # ~9:16, stories / reels covers
    "square": (1024, 1024),
    "landscape": (1216, 832),
}
TIERS = ("public", "private")


@dataclass
class Persona:
    slug: str
    name: str
    age: int
    identity: str
    style: str = ""
    negative: str = ""
    trigger: str = ""
    lora_file: str | None = None
    lora_strength: float = 0.85


@dataclass
class Shot:
    id: str
    prompt: str
    tier: str = "public"
    size: str = "portrait"
    count: int = 4
    seed: int | None = None


@dataclass
class Preset:
    checkpoint: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    hires_scale: float | None = None
    hires_denoise: float = 0.35
    hires_steps: int | None = None


@dataclass
class RenderConfig:
    comfy_url: str
    default_preset: str
    presets: dict[str, Preset] = field(default_factory=dict)


def _read(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def load_persona(path: str | Path) -> Persona:
    d = _read(path)
    lora = d.get("lora") or {}
    return Persona(
        slug=d["slug"],
        name=d["name"],
        age=int(d["age"]),
        identity=_clean(d["identity"]),
        style=_clean(d.get("style")),
        negative=_clean(d.get("negative")),
        trigger=_clean(d.get("trigger")),
        lora_file=lora.get("file") or None,
        lora_strength=float(lora.get("strength", 0.85)),
    )


def load_shots(path: str | Path) -> list[Shot]:
    d = _read(path)
    shots = []
    for s in d.get("shots") or []:
        shot = Shot(
            id=s["id"],
            prompt=_clean(s["prompt"]),
            tier=s.get("tier", "public"),
            size=s.get("size", "portrait"),
            count=int(s.get("count", 4)),
            seed=s.get("seed"),
        )
        if shot.tier not in TIERS:
            raise ValueError(f"{path}: shot {shot.id!r} has unknown tier {shot.tier!r} (use {TIERS})")
        if shot.size not in SIZES:
            raise ValueError(f"{path}: shot {shot.id!r} has unknown size {shot.size!r} (use {list(SIZES)})")
        shots.append(shot)
    ids = [s.id for s in shots]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: shot ids must be unique")
    return shots


def load_render_config(path: str | Path) -> RenderConfig:
    d = _read(path)
    presets = {}
    for name, p in (d.get("presets") or {}).items():
        hires = p.get("hires") or {}
        presets[name] = Preset(
            checkpoint=p["checkpoint"],
            steps=int(p["steps"]),
            cfg=float(p["cfg"]),
            sampler=p["sampler"],
            scheduler=p["scheduler"],
            hires_scale=float(hires["scale"]) if hires.get("scale") else None,
            hires_denoise=float(hires.get("denoise", 0.35)),
            hires_steps=int(hires["steps"]) if hires.get("steps") else None,
        )
    default = d.get("default_preset") or next(iter(presets), "")
    if default not in presets:
        raise ValueError(f"{path}: default_preset {default!r} is not defined under presets")
    return RenderConfig(comfy_url=d.get("comfy_url", "http://127.0.0.1:8188"), default_preset=default, presets=presets)
