# VidEro — AI Video Content Plan

Goal: run a social media account for an AI-generated persona, using the Eromify MCP
to produce video, and use that account as a funnel to a paid destination (website,
chat, or exclusive content — to be decided later).

This document covers the **content creation pipeline**. Monetization destination is
out of scope for now.

---

## 1. Ground rules (decide these first, they constrain everything else)

| Rule | Why |
|---|---|
| Persona is **fully synthetic** and clearly **adult** (21+ looking, adult styling, adult context). No real person's face, name or likeness, no "looks like <celebrity>" prompts, no reference photos of people who haven't signed a release. | Non-consensual likeness/deepfake laws (US TAKE IT DOWN Act, UK OSA, EU AI Act) and every platform's ToS. Single biggest legal risk. |
| Never prompt for youthful cues (school uniforms, "petite/young/teen", childlike bodies or settings), even as a joke. | Instant permanent bans, and potentially criminal. |
| **Two content tiers**: *Public* (SFW / suggestive) for social, *Explicit* only behind an age-verified paid platform. | Instagram, TikTok, YouTube ban nudity/sexual content; X and Reddit allow it only with NSFW labelling. |
| Label content as AI-generated everywhere (platform AI toggle + bio). | Required by TikTok/Meta/YouTube and by AI-friendly paid platforms (e.g. Fanvue); undisclosed AI personas get accounts removed. |
| Keep a generation log (prompt, model, seed/job id, date, which persona). | Proves content is synthetic and original if a takedown or dispute comes in. |

---

## 2. Eromify MCP setup

Eromify exposes an MCP server at `https://api.eromify.com/mcp` that can generate images
and videos, upload references, and check credits from inside Claude.

Connect it:

- **claude.ai / Claude app:** Settings → Connectors → *Add custom connector* →
  `https://api.eromify.com/mcp` → approve the OAuth consent.
- **Claude Code CLI (local):**
  ```bash
  claude mcp add --transport http eromify https://api.eromify.com/mcp
  ```
  then run `/mcp` to authenticate.

> The current cloud session does **not** have it connected yet — none of the
> Eromify tools are available here. Once connected, list the actual tool names and
> parameters and update section 4 with them.

Cost reference (third-party review, Aug 2026 — verify in the dashboard):

- Image generation/edit: ~100 credits
- Video generation: ~1,500 credits
- Packs: ~$6 → ~$48 one-time. A video is ~15× an image, so **iterate on stills,
  animate only approved frames.**
- Reported issues: lost credits / slow support. Start with the smallest pack, test
  the full pipeline, then scale.

---

## 3. Pipeline overview

```
 Persona bible ──► Identity lock ──► Still frames ──► Image-to-video ──► Post-production ──► Publish
 (once)            (once)            (cheap, many)    (expensive, few)    (captions, cuts)     (per platform)
```

### Step 1 — Persona bible (once)
A single source of truth so every prompt is consistent. Fill `personas/<name>.md`:

- Name, stated age (21+), backstory, location vibe, personality, voice/tone for captions
- Fixed physical description: face shape, eye colour, hair (colour/length/style),
  skin tone, body type, distinguishing marks (mole, tattoo) — these anchor consistency
- Wardrobe palette and 3–5 recurring locations (bedroom, gym, beach, café, car)
- Content pillars, e.g. *Get ready with me*, *Gym*, *Travel*, *After dark teaser*

### Step 2 — Identity lock (once)
- Generate 20–40 candidate headshots from the bible; pick the best one.
- Generate a **reference set** from it: front, ¾, profile, full body, 2–3 outfits,
  2–3 lighting setups.
- Register/upload it as the Eromify influencer (identity-lock).
- **Consistency test** before spending on video: generate 10 stills in very
  different scenes. If the face drifts in more than ~2 of 10, refine the references.

### Step 3 — Still frames (the cheap iteration loop)
Per video idea, generate 4–8 stills of the *first frame*, pick 1–2. Prompt structure:

```
[persona identity ref] + [shot type] + [action/pose] + [outfit] + [location]
+ [lighting] + [camera/lens] + [mood] + [tier constraint]
```

Example (Public tier):
> Influencer "<name>", medium shot, sitting on bed edge putting on earrings, black
> satin slip dress, warm bedroom with fairy lights, golden-hour window light, 35mm,
> shallow depth of field, playful smirk at camera, tasteful, fully clothed

### Step 4 — Image-to-video (the expensive step)
- Always **image-to-video** from an approved still — far better identity consistency
  than text-to-video.
- Keep motion prompts simple, one action per clip:
  > slow push-in, she glances up at camera and smiles, hair moves slightly, subtle hand gesture
- Model choice (test each once with the same still, then standardize):
  - **Kling** — strong human motion and faces; good default
  - **Veo** — most cinematic/realistic, often the most restricted on suggestive content
  - **Hailuo / MiniMax, Seedance, WAN** — cheaper/faster options for B-roll and loops
- Clips are ~5–10 s. Build longer videos by chaining clips (last frame of clip N →
  first frame of clip N+1).

### Step 5 — Post-production
- Trim, cut 2–4 clips into a 7–20 s short, 9:16 vertical, 1080×1920.
- Add text hook in first second, captions, trending/licensed audio (added inside
  each app to avoid copyright strikes).
- Optional voice: TTS voice that stays consistent with the persona.
- Tools: CapCut or DaVinci Resolve manually; later automate with `ffmpeg` scripts in
  this repo.
- QC checklist before posting: hands/fingers, face drift, teeth, flicker, text
  artifacts, anything that reads as underage → reject.

### Step 6 — Publish and funnel
| Platform | What goes there | Funnel mechanism |
|---|---|---|
| Instagram Reels / TikTok | Public tier only, lifestyle + flirty | Link-in-bio page (age gate) |
| X (Twitter) | Suggestive + mild NSFW with sensitive-media flag | Pinned post + bio link |
| Reddit | NSFW subs that allow AI (check each sub's rules) | Profile link |
| Paid destination (TBD) | Explicit tier, age-verified | Subscriptions / PPV / chat |

---

## 4. Production cadence and budget (starting point)

| Item | Per week | Credits (approx.) |
|---|---|---|
| Stills explored | ~60 | 6,000 |
| Videos generated (incl. ~30% rejects) | ~10 | 15,000 |
| Published shorts | 5–7 | — |

Weekly loop with Claude + Eromify MCP:
1. Monday: Claude drafts 7 post ideas from persona bible + content pillars.
2. Generate stills for all 7, approve.
3. Animate approved stills, QC, re-roll failures.
4. Edit, caption, schedule.
5. Log prompts/job ids in `logs/` and note which hooks performed best.

---

## 5. Repo structure (proposed)

```
docs/content-plan.md        ← this file
personas/<name>.md          ← persona bible
prompts/stills/*.md         ← reusable still-frame prompt templates per pillar
prompts/motion/*.md         ← reusable motion prompts
logs/YYYY-MM-DD.csv         ← prompt, model, job id, persona, tier, status
scripts/                    ← ffmpeg assembly, caption burn-in, export presets
```

---

## 6. Next steps

1. Connect the Eromify MCP (section 2) and record its real tool list.
2. Write the first persona bible.
3. Buy the smallest credit pack; run identity lock + 10-still consistency test.
4. Run a model bake-off: same still → Kling / Veo / Hailuo / Seedance, compare.
5. Produce 5 public-tier shorts, post, and measure before scaling.
6. Decide the paid destination (site vs chat vs subscription platform) — separate plan.
