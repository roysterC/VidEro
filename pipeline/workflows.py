"""ComfyUI workflow graphs (API format).

Two ways to get a graph:
- ``sdxl_graph``: a built-in SDXL text-to-image graph (+ optional LoRA and hires pass)
  using only core ComfyUI nodes.
- ``fill_template``: any workflow you exported from ComfyUI with "Export (API)".
  Put %PROMPT% and %NEGATIVE% in its text boxes; seed, size and filename prefix
  are overridden automatically.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .config import Preset

PROMPT_TOKEN = "%PROMPT%"
NEGATIVE_TOKEN = "%NEGATIVE%"


def sdxl_graph(
    *,
    preset: Preset,
    positive: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    lora: tuple[str, float] | None = None,
) -> dict:
    g: dict = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": preset.checkpoint}},
    }
    model, clip = ["ckpt", 0], ["ckpt", 1]
    if lora:
        name, strength = lora
        g["lora"] = {
            "class_type": "LoraLoader",
            "inputs": {"model": model, "clip": clip, "lora_name": name,
                       "strength_model": strength, "strength_clip": strength},
        }
        model, clip = ["lora", 0], ["lora", 1]

    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": clip}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}}
    g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["sample"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": preset.steps, "cfg": preset.cfg,
            "sampler_name": preset.sampler, "scheduler": preset.scheduler, "denoise": 1.0,
        },
    }
    last = ["sample", 0]

    if preset.hires_scale:
        g["upscale"] = {
            "class_type": "LatentUpscaleBy",
            "inputs": {"samples": last, "upscale_method": "nearest-exact", "scale_by": preset.hires_scale},
        }
        g["hires"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["upscale", 0],
                "seed": seed, "steps": preset.hires_steps or preset.steps, "cfg": preset.cfg,
                "sampler_name": preset.sampler, "scheduler": preset.scheduler, "denoise": preset.hires_denoise,
            },
        }
        last = ["hires", 0]

    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": last, "vae": ["ckpt", 2]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g


def load_template(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        graph = json.load(f)
    if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n for n in graph.values()):
        raise ValueError(f"{path}: not an API-format workflow (use 'Export (API)' in ComfyUI)")
    text = json.dumps(graph)
    if PROMPT_TOKEN not in text:
        raise ValueError(f"{path}: put {PROMPT_TOKEN} in the positive prompt box before exporting")
    return graph


def fill_template(
    template: dict, *, positive: str, negative: str, seed: int, width: int, height: int, prefix: str
) -> dict:
    g = copy.deepcopy(template)
    for node in g.values():
        inputs = node.get("inputs", {})
        cls = node.get("class_type", "")
        for key, val in list(inputs.items()):
            if isinstance(val, str):
                inputs[key] = val.replace(PROMPT_TOKEN, positive).replace(NEGATIVE_TOKEN, negative)
            elif key in ("seed", "noise_seed") and isinstance(val, int):
                inputs[key] = seed
        if cls == "EmptyLatentImage":
            inputs["width"], inputs["height"] = width, height
        if cls == "SaveImage":
            inputs["filename_prefix"] = prefix
    return g
