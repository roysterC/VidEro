"""Prompt assembly and content guardrails.

The guardrails are deliberately strict: a render is refused before anything is
queued if a prompt contains youth cues or asks to resemble a real person.
"""

from __future__ import annotations

import re

from .config import Persona, Shot

MIN_PERSONA_AGE = 21

# Terms that push a model toward someone who looks under age. "girl" is included
# because SDXL skews noticeably younger on it; write "woman" instead.
YOUTH_TERMS = [
    "child", "children", "childlike", "child-like", "kid", "kids", "minor", "minors",
    "teen", "teens", "teenage", "teenager", "preteen", "pre-teen", "underage", "under age",
    "loli", "lolita", "shota", "jailbait", "barely legal",
    "girl", "girls", "boy", "boys", "young girl", "little girl",
    "schoolgirl", "schoolboy", "school uniform", "high school", "highschool", "middle school",
    "baby face", "babyface", "petite", "flat chest", "flat chested",
]

# Terms that ask the model to copy a real person's likeness.
LIKENESS_TERMS = [
    "lookalike", "look-alike", "look alike", "looks like", "resembling", "resembles",
    "doppelganger", "celebrity", "famous", "deepfake",
]

# Explicit ages like "19 year old", "17yo", "18-year-old".
_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:-|\s)?\s*(?:yo|y/o|y\.o\.|years?(?:\s*|-)old|year-old)\b", re.I)

SAFETY_NEGATIVE = (
    "child, teen, underage, young-looking, childlike, baby face, school uniform, "
    "real person, celebrity likeness"
)
PUBLIC_NEGATIVE = "nude, naked, nsfw, topless, nipples, explicit, genitals, see-through"
QUALITY_NEGATIVE = (
    "deformed, disfigured, bad anatomy, extra fingers, missing fingers, fused fingers, "
    "extra limbs, blurry, lowres, jpeg artifacts, watermark, text, logo, plastic skin, "
    "cartoon, 3d render, cgi"
)


class GuardrailError(ValueError):
    pass


def _term_pattern(term: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term)]
    return re.compile(r"\b" + r"[\s_-]*".join(parts) + r"\b", re.I)


_YOUTH = [(t, _term_pattern(t)) for t in YOUTH_TERMS]
_LIKENESS = [(t, _term_pattern(t)) for t in LIKENESS_TERMS]


def check_prompt(text: str, where: str = "prompt") -> None:
    """Raise GuardrailError if the positive prompt text breaks a content rule."""
    for term, pat in _YOUTH:
        if pat.search(text):
            hint = " (write 'woman' instead)" if term in ("girl", "girls") else ""
            raise GuardrailError(f"{where}: youth cue {term!r} is not allowed{hint}")
    for term, pat in _LIKENESS:
        if pat.search(text):
            raise GuardrailError(f"{where}: real-person likeness cue {term!r} is not allowed")
    for m in _AGE_RE.finditer(text):
        if int(m.group(1)) < MIN_PERSONA_AGE:
            raise GuardrailError(f"{where}: stated age {m.group(0)!r} is under {MIN_PERSONA_AGE}")


def check_persona(persona: Persona) -> None:
    if persona.age < MIN_PERSONA_AGE:
        raise GuardrailError(f"persona {persona.slug!r}: age must be {MIN_PERSONA_AGE}+ (got {persona.age})")
    check_prompt(persona.identity, f"persona {persona.slug!r} identity")
    check_prompt(persona.style, f"persona {persona.slug!r} style")


def build_prompts(persona: Persona, shot: Shot, use_lora: bool) -> tuple[str, str]:
    """Return (positive, negative) for one shot. Runs the guardrails first."""
    check_persona(persona)
    check_prompt(shot.prompt, f"shot {shot.id!r}")

    parts = []
    if use_lora and persona.trigger:
        parts.append(persona.trigger)
    parts += [persona.identity, shot.prompt, persona.style]
    positive = ", ".join(p.strip().strip(",") for p in parts if p)

    neg = [SAFETY_NEGATIVE]
    if shot.tier == "public":
        neg.append(PUBLIC_NEGATIVE)
    neg.append(QUALITY_NEGATIVE)
    if persona.negative:
        neg.append(persona.negative)
    negative = ", ".join(neg)
    return positive, negative
