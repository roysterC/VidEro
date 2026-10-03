# Static image pipeline: AI influencer on a home PC

Target machine: Windows, RTX 2070 Super (8 GB VRAM), Ryzen 5 5600X, 16 GB RAM.

```
§2 Casting        text-only headshots → pick ONE hero face
§3 Dataset        hero face as reference → ~45 varied images → keep 25–40
§4 LoRA training  dataset → persona LoRA (the "identity lock")
§5 Consistency    10 fixed-seed test scenes → pass if 8/10 look like the same person
§6 Production     weekly shot list → pick best → export for each platform → post

Why §3 + §4 exist: a text prompt alone invents a slightly different face on every
render. The dataset (§3) shows the model one face in many situations; the LoRA (§4)
makes it remember that face under a trigger word, so every post is the same person.
```

Everything is driven by three kinds of files:

| File | What it is |
|---|---|
| `personas/<slug>.yaml` | The persona bible: fixed identity description, style, LoRA trigger and file |
| `shots/*.yaml` | Shot lists: one entry per scene, with tier, aspect ratio and number of candidates |
| `config/render.yaml` | ComfyUI address and render presets (checkpoint, steps, sampler) |

---

## 0. One-time setup (about 1 hour)

### 0.1 Windows prep for 16 GB of RAM
- Pagefile: *System → Advanced system settings → Performance → Advanced → Virtual memory*,
  leave it **System managed** on an SSD. Model loading spikes RAM. Without a pagefile,
  ComfyUI crashes instead of slowing down.
- Close the browser tabs you don't need while rendering.
- Update the NVIDIA driver (Studio or Game Ready, either works).
- An upgrade to 32 GB of RAM (~$50–70 DDR4) is the best-value improvement.

### 0.2 ComfyUI
1. Download **ComfyUI Windows portable (NVIDIA)** from the ComfyUI GitHub releases page and
   extract it with 7-Zip to a short path on an SSD, e.g. `C:\AI\ComfyUI_windows_portable`.
2. Run `run_nvidia_gpu.bat`. The UI opens at <http://127.0.0.1:8188>.
3. If you get *out of memory* errors, edit the `.bat` and add `--lowvram` to the end of the
   `python main.py` line.
4. Install **ComfyUI-Manager** if your build doesn't already show a *Manager* button.
   You need it to install custom nodes in step 2.

### 0.3 Models
Put these in `ComfyUI\models\checkpoints\`:

| File | Source | Use |
|---|---|---|
| `RealVisXL_V5.0_Lightning_fp16.safetensors` | Hugging Face `SG161222/RealVisXL_V5.0_Lightning` (or Civitai) | Default: photoreal, 6 steps, fast on 8 GB |
| `RealVisXL_V5.0_fp16.safetensors` *(optional)* | Hugging Face `SG161222/RealVisXL_V5.0` | `quality` preset, 30 steps |

You can use any SDXL photoreal checkpoint (Juggernaut XL, etc.). Put its exact filename in
`config/render.yaml`. **Don't use Flux on this card.** It runs, but at 1.5–3 minutes per image.

### 0.4 This repo's scripts
1. Install Python 3.11+ from python.org and tick *Add to PATH*.
2. In this repo folder:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m unittest discover -s tests -t .
   ```
3. Smoke test with ComfyUI running:
   ```powershell
   python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml --count 1
   ```
   Images land in `output\ava\01-casting\` and every render is logged to
   `output\ava\log.csv` (prompt, seed, model, ComfyUI job id).

Expect roughly **10–20 s per image** with the `fast` preset and **45–90 s** with `final`
(hires pass) on a 2070 Super. These are estimates; your first run tells you the real numbers.

---

## 1. Define the persona

Copy `personas/ava.yaml` to `personas/<yourname>.yaml` and edit it.

- `identity` is the face and body. Be specific and **never change it** afterwards. Use concrete
  anchors such as eye colour, hair colour, length and cut, freckles, a beauty mark, and build.
- `age` must be **21+**. The scripts refuse to render otherwise.
- `trigger` is a rare made-up token plus "woman" (e.g. `mrlx woman`). It's used after the LoRA exists.
- Fully fictional: no real person's face, name or photos as a base.

Check the assembled prompts without rendering anything:
```powershell
python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml --dry-run
```

**Built-in guardrails.** A shot list is refused before anything renders if it contains:
- youth cues: teen, girl, school uniform, petite, ages under 21, …
- likeness cues: "looks like", lookalike, celebrity, …

Safety terms are always added to the negative prompt. `tier: public` shots also get
nudity terms in the negative prompt so they stay postable on Instagram and TikTok.

---

## 2. Casting: find the face

```powershell
python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml
```
- 40 headshots, about 10 minutes.
- Pick **one** hero face. Clear, front-facing, neutral light, nothing odd.
- If none look right, tweak `identity` and re-run. This is the cheapest moment to change her.
- Copy the hero image to `ComfyUI\input\hero_<slug>.png`.

## 3. Dataset: same face, many situations

A text description alone won't give the same face twice, so this step uses the hero face as an
image reference through IPAdapter.

1. **Install the custom node.** In ComfyUI Manager, install *ComfyUI_IPAdapter_plus*, then
   restart. Download the models listed in that node's README for the
   **PLUS FACE (portraits)** preset:
   - an SDXL `ip-adapter-plus-face` model → `models\ipadapter\`
   - the ViT-H CLIP vision model → `models\clip_vision\`

   This preset needs no `insightface`, which is painful to install on Windows. If likeness is
   too weak, upgrade later to the FaceID presets.
2. **Build the workflow once in the UI.** Start from the default SDXL workflow with your checkpoint and add:
   - `Load Image` (hero face) → `IPAdapter Unified Loader` (preset *PLUS FACE (portraits)*)
     → `IPAdapter Advanced` (weight ~0.75) → into the KSampler's `model`.
   - In the positive prompt box type exactly `%PROMPT%`. In the negative prompt box type `%NEGATIVE%`.
   - Click Run once by hand to check it works. Then **Workflow → Export (API)** and save it as
     `workflows\faceref.json` in this repo.
3. **Render the dataset.**
   ```powershell
   python -m pipeline.generate --persona personas/ava.yaml --shots shots/02-dataset.yaml --template workflows/faceref.json
   ```
   The script fills in prompt, negative, seed, size and filename. About 45 images.
4. **Curate hard.** Copy the 25–40 best into `output\ava\picked\`. Keep only images where:
   - it's clearly the same face;
   - hands, eyes and teeth are clean;
   - there's real variety: close-up, ¾, profile, full body, different light, outfits and backgrounds.

   Duplicates and bad hands get baked into the LoRA.

## 4. Train the persona LoRA

```powershell
python -m pipeline.dataset --persona personas/ava.yaml --src output/ava/picked --out datasets/ava
```
This makes `datasets\ava\img\10_avlx woman\` with resized images and a caption per image. Each
caption is the trigger plus the scene, taken from the render log. Describing the scene teaches the
LoRA that only the person belongs to the trigger word.

**Option A: train on your PC** with kohya_ss (bmaltais GUI) or OneTrainer. Settings that fit 8 GB on SDXL:

| Setting | Value |
|---|---|
| Base model | the same SDXL checkpoint, or base SDXL 1.0 |
| Network dim / alpha | 16 / 8 |
| Resolution | 1024, bucketing on |
| Batch size | 1 |
| Optimizer | Adafactor (relative_step off), LR 1e-4 |
| Train text encoder | off (UNet only) |
| Mixed precision | **fp16** (the 2070 Super has no bf16) |
| Gradient checkpointing | on |
| Cache latents (to disk) | on |
| Epochs | ~10 (≈ images × repeats × epochs ≈ 3,000–4,000 steps), save every 2 epochs |

Expect a few hours. Close everything else. If it runs out of memory or crawls, use Option B.

**Option B: rent a GPU for one hour.** RunPod or Vast.ai, an RTX 4090 with a kohya_ss template:
upload `datasets\ava`, train, download the `.safetensors`. Usually around $1.

Then:
- Copy the LoRA to `ComfyUI\models\loras\ava_v1.safetensors`.
- In `personas/ava.yaml`, set `lora.file: ava_v1.safetensors`.
- From now on the trigger word is added to every prompt automatically.

## 5. Consistency test

```powershell
python -m pipeline.generate --persona personas/ava.yaml --shots shots/03-consistency.yaml
```
The seeds are fixed, so you can compare LoRA versions (`ava_v1`, `ava_v2`) or strengths fairly.

| Result | Fix |
|---|---|
| Same face in at least 8 of 10 | Pass |
| Face drifts | Raise `lora.strength` toward 1.0, or retrain with more close-ups |
| Every image looks like a dataset photo (same outfit/background) | Lower the strength to ~0.7, or retrain with more variety |

## 6. Weekly production

1. Copy `shots/week-example.yaml` to `shots/week-2026-10-05.yaml` and write 7 scenes from the persona's pillars.
2. Draft:
   ```powershell
   python -m pipeline.generate --persona personas/ava.yaml --shots shots/week-2026-10-05.yaml
   ```
3. Re-render the winners sharper with `--preset final --only mon-coffee tue-gym ...`.
4. **QC each pick.** Reject on any of these:
   - hands and fingers;
   - eyes, teeth, earrings;
   - text or logos;
   - warped background lines;
   - anything that reads as young.

   Fix small defects with ComfyUI inpainting.
5. Put the picks in a folder and export:
   ```powershell
   python -m pipeline.export --src output/ava/week-2026-10-05/picked --format feed story
   ```
   The output is JPEG with **all metadata stripped**. ComfyUI's PNGs contain your full prompts and
   workflow, so never upload the raw PNGs.

---

## 7. Posting to social media

| Platform | Format | Notes |
|---|---|---|
| Instagram (feed) | `feed` 1080×1350 | Turn on the **AI info** label when posting. Public tier only. |
| Instagram / Facebook Stories | `story` 1080×1920 | Link stickers allowed, so link to your landing page |
| Threads | `feed` | Same images as Instagram |
| X | `x` 1200×1500 | Suggestive is fine. Flag media as sensitive for anything spicier. Put "AI-generated" in the bio |
| TikTok photo mode | `story` | Toggle **AI-generated content**. Public tier only |

**Account setup:**
- A new email and phone number per persona.
- An Instagram **Creator** account, which unlocks scheduling and insights.
- The bio says the persona is an AI creator.

**Warm-up:**
- Week 1: post every 1–2 days, and engage manually (likes, comments in your niche) for 15–20 minutes a day.
- From week 2: daily posts and 3–5 stories.
- No links in the first week or two. New accounts that link out immediately get throttled.

**Scheduling:**
- Meta Business Suite (free) schedules Instagram and Facebook posts and stories from a PC.
- X has built-in scheduling.

Automated posting through the APIs can come later. The Instagram Graph API needs a Business
account, a Facebook app and a publicly hosted image URL.

**Captions:** write them in the persona's voice (`personality` in the persona file).
- One line of hook, one line of story, then a question.
- 3–5 niche hashtags, not 30.

**Track** saves, shares, profile visits and follows per post. Next week's shot list should copy
the scenes that won.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cannot reach ComfyUI` | Start `run_nvidia_gpu.bat` first. Check `comfy_url` in `config/render.yaml` |
| `checkpoint ... not found` | The filename in `config/render.yaml` must match `models\checkpoints` exactly |
| CUDA out of memory | Add `--lowvram` to the `.bat`. Use the `fast` preset. Close other GPU apps |
| ComfyUI closes while loading a model | The system ran out of RAM: enable the pagefile, close apps, upgrade to 32 GB |
| Plastic, airbrushed skin | Keep "realistic skin texture with pores" in `style`. Lower CFG. Try the `quality` preset |
| Everyone looks like the same stock model | Make `identity` more distinctive (freckles, beauty mark, specific hair cut) |
| `refused: youth cue 'girl'` | Working as intended. Write "woman" |
