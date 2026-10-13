# Hyper-realistic images and video of a locked character (RunPod)

This is a step-by-step playbook for producing photo-real stills and short videos of **one
character who looks the same in every post**, using rented GPUs on RunPod and this repo's
scripts on your own PC.

```
Part 1  RunPod setup         network volume + ComfyUI pod + models          (once, ~45 min)
Part 2  Lock the character   casting → dataset → LoRA → consistency test    (once, ~1 day)
Part 3  Realism workflow     LoRA + hires + FaceDetailer + upscale          (once, ~30 min)
Part 4  Prompting            phone-photo prompts that don't look like AI     (every shot)
Part 5  Pick and QC stills   identity check, defects, inpainting             (every shot)
Part 6  Video                still → Wan 2.2 or Kling → interpolate → export (per clip)
Part 7  Session routine      start, render, stop: keep the bill small       (every session)
```

**Why this order:** the face is locked by the **LoRA** (Part 2), and realism comes from the
**workflow** (Part 3) and the **prompt** (Part 4). Video (Part 6) never invents the face. It
always animates a still that already passed QC. Skip a step and the later ones can't fix it.

> **Ground rules (non-negotiable).** The character is fully fictional and clearly an adult (21+).
> No real person's face, name or photos as input, and no "looks like <celebrity>". Never use
> youthful cues. Label posts as AI-generated. The scripts enforce part of this. See
> [content-plan.md §1](content-plan.md#1-ground-rules-decide-these-first-they-constrain-everything-else).

**Rough cost** (RTX 4090 at about $0.34–0.69/hr, plus about $7–10/month for 100 GB of storage; check live prices):

| Job | GPU time | Approx. cost |
|---|---|---|
| Setup + model downloads | 0.5–1 h | $0.50 |
| Casting + dataset (Part 2) | 1–2 h | $1 |
| LoRA training | 0.5–1 h | $0.50 |
| A week of stills (7 posts × 4 drafts + finals) | 1–1.5 h | $0.50–1 |
| 5-second video clip, Wan 2.2 at 720p | 5–15 min each on a 4090 | $0.05–0.15 each |

---

## Part 1 — RunPod setup (once)

### 1.1 Create a network volume
The network volume is your persistent drive. Models, LoRA files and renders live there.
Without one, everything disappears when you terminate a pod.

1. RunPod → **Storage → New Network Volume**.
2. **Size: 150 GB.** SDXL is about 7 GB per checkpoint and Wan 2.2 14B is about 30–40 GB in
   total, plus renders.
3. **Datacenter:** pick one that lists **RTX 4090** *and* a 48 GB card (L40S or RTX A6000) as
   available. The volume can only attach to pods in the same datacenter.

### 1.2 Deploy a pod
1. **Pods → Deploy**, and choose the datacenter of your volume.
2. **GPU:**
   - **RTX 4090 (24 GB)** for stills, LoRA training and 480p–720p video tests.
   - **L40S / A6000 (48 GB)** or **H100** only when you render a lot of 720p video.
3. **Template:** the official **RunPod PyTorch** template (the newest 2.x version listed). It
   gives you Python, CUDA, PyTorch and JupyterLab. ComfyUI itself goes on your volume in step 1.3.
   *Why not a ComfyUI template?* Most install ComfyUI on the **container disk**, which is wiped every
   time the pod stops. Custom nodes, their Python packages and anything in `models/` would vanish.
4. **Attach the network volume.** It mounts at `/workspace`, the only folder that survives a stop.
5. **Edit Template → Expose HTTP Ports:** `8888,8188` (Jupyter and ComfyUI).
6. Deploy. When it's running, **Connect** shows the links:
   - Jupyter: `https://<POD_ID>-8888.proxy.runpod.net`
   - ComfyUI (after step 1.3): `https://<POD_ID>-8188.proxy.runpod.net`

Always use **the same template and version** for this volume. The Python environment in step 1.3 is
built against that image's Python and PyTorch.

### 1.3 Install ComfyUI on the volume (once)
Open Jupyter → **Terminal** and run this once. Everything goes under `/workspace`, so it all survives
stops and new pods.

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI

# Python environment on the volume. --system-site-packages reuses the image's PyTorch + CUDA
# (no 5 GB download), and every package you add later is stored on the volume.
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
python -c "import torch; print(torch.__version__, 'CUDA OK' if torch.cuda.is_available() else 'NO CUDA')"

# ComfyUI-Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager custom_nodes/comfyui-manager
pip install -r custom_nodes/comfyui-manager/requirements.txt

# Start script, used at the beginning of every session
cat > /workspace/start-comfy.sh <<'SH'
#!/bin/bash
cd /workspace/ComfyUI
source venv/bin/activate
nohup python main.py --listen 0.0.0.0 --port 8188 > /workspace/comfy.log 2>&1 &
echo "ComfyUI starting. Follow the log with: tail -f /workspace/comfy.log"
SH
chmod +x /workspace/start-comfy.sh
/workspace/start-comfy.sh
```

When the log shows `To see the GUI go to: http://0.0.0.0:8188`, open the ComfyUI link from **Connect**.

**Every session from now on:** start the pod, open a Jupyter terminal, run `/workspace/start-comfy.sh`.
For terminal commands in this guide, first run:

```bash
export COMFY=/workspace/ComfyUI
source $COMFY/venv/bin/activate      # so pip, python and hf use ComfyUI's environment
```

> **Already installed into a ComfyUI template's container disk?** Rescue the downloads before you stop
> that pod. Run the `git clone` line above, then move your models across, e.g.
> `mv /path/to/old/ComfyUI/models/checkpoints/* /workspace/ComfyUI/models/checkpoints/`
> (repeat for `loras`, `ipadapter`, `clip_vision`, `ultralytics`, `upscale_models`). Custom
> nodes are quicker to reinstall than to move. Then stop that pod and continue with 1.2 on the
> PyTorch template.
>
> If you ever switch to a different template or version, rebuild the environment:
> `rm -rf /workspace/ComfyUI/venv`, then re-run the `venv` and `pip install` lines above plus
> step 3 of 1.4's Option 2. Models and node folders are kept.

### 1.4 Install the custom nodes
| Node pack (GitHub repo) | Search the Manager for | Used for |
|---|---|---|
| `cubiq/ComfyUI_IPAdapter_plus` | `IPAdapter` | Face reference while building the dataset (Part 2) |
| `ltdrdata/ComfyUI-Impact-Pack` | `Impact` | FaceDetailer (Part 3) |
| `ltdrdata/ComfyUI-Impact-Subpack` | `Impact` | The face detector loader (Part 3) |
| `ssitu/ComfyUI_UltimateSDUpscale` | `UltimateSDUpscale` (one word) | Tiled final upscale (Part 3, optional) |
| `Fannovel16/ComfyUI-Frame-Interpolation` | `Frame Interpolation` | RIFE, to smooth video from 16 to 32 fps (Part 6) |
| `Kosinkadink/ComfyUI-VideoHelperSuite` | `Video Helper` | Load and save video, extract frames (Part 6) |

**Option 1: the Manager.** In ComfyUI → **Manager → Custom Nodes Manager**, set the filter to **All**,
search for each term above, click **Install**, then **Restart**. The Manager searches display names,
not repo names, so search short words. If a pack shows as already installed, the template included it.

**Option 2: the terminal (use this if the Manager can't find a pack).** In the Jupyter terminal:

```bash
# 1. ComfyUI's own Python (on the volume, from step 1.3)
export COMFY=/workspace/ComfyUI
export PY=$COMFY/venv/bin/python

# 2. Clone the packs into custom_nodes
cd $COMFY/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack
git clone --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# 3. Install their Python dependencies into ComfyUI's own Python
$PY -m pip install -r ComfyUI-Impact-Pack/requirements.txt
$PY -m pip install -r ComfyUI-Impact-Subpack/requirements.txt
$PY -m pip install -r ComfyUI-VideoHelperSuite/requirements.txt
(cd ComfyUI-Frame-Interpolation && $PY install.py)
```

Skip any `git clone` that says the folder already exists. Then restart ComfyUI (**Manager →
Restart**, or stop and start the pod) and check that a few nodes appear: double-click the canvas and
search for *FaceDetailer*, *IPAdapter Advanced*, *RIFE VFI* and *Video Combine*.

Use `$PY -m pip` (or activate the venv first), not the system `pip`. Plain `pip` installs into the
container disk's Python: ComfyUI never sees it, and it's wiped at the next stop. That causes
"import failed" errors in the startup log. The Manager already installs into the right place,
because it uses the Python that runs ComfyUI.

### 1.5 Download the models
Use the Jupyter terminal, with ComfyUI's environment active (`source /workspace/ComfyUI/venv/bin/activate`).
`hf` is the Hugging Face CLI installed in step 1.3; older versions call it `huggingface-cli download`.
All files land in `/workspace/ComfyUI/models`, on the volume.

**Hugging Face token (recommended).** Without one, `hf` warns *"You are sending unauthenticated
requests to the HF Hub"*. Downloads still work, but they're slower and rate-limited. Fix it once:
1. On huggingface.co go to **Settings → Access Tokens → Create new token**, type **Read**.
2. RunPod → **Secrets → Create Secret**, name it `hf_token`, and paste the token.
3. When deploying the pod: **Edit Template → Environment Variables**, set `HF_TOKEN` =
   `{{ RUNPOD_SECRET_hf_token }}`.
4. Check with `hf auth whoami`, which should print your username.

For a one-off session you can run `export HF_TOKEN=hf_...` in the terminal instead. Avoid
`hf auth login` here: it stores the token under `/root` on the container disk, which is wiped at every stop.

```bash
cd $COMFY/models

# Photoreal SDXL checkpoints (full model for finals, Lightning for fast drafts)
hf download SG161222/RealVisXL_V5.0 RealVisXL_V5.0_fp16.safetensors --local-dir checkpoints
hf download SG161222/RealVisXL_V5.0_Lightning RealVisXL_V5.0_Lightning_fp16.safetensors --local-dir checkpoints

# IPAdapter face model + the CLIP vision encoder (the file must have this exact name)
mkdir -p ipadapter clip_vision
hf download h94/IP-Adapter sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors --local-dir /tmp/ipa
mv /tmp/ipa/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors ipadapter/
hf download h94/IP-Adapter models/image_encoder/model.safetensors --local-dir /tmp/ipa
mv /tmp/ipa/models/image_encoder/model.safetensors clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors

# Face detector for FaceDetailer
mkdir -p ultralytics/bbox
hf download Bingsu/adetailer face_yolov8m.pt --local-dir ultralytics/bbox
```

Two more come from **Manager → Model Manager**. Search for and install:
- an **upscale model**: `4x-UltraSharp` or `4x_NMKD-Siax_200k` (goes to `models/upscale_models`);
- **SAM ViT-B**, if FaceDetailer asks for a SAM model.

The **Wan 2.2 video models** are big. Download them only when you reach Part 6; the ComfyUI
template there lists the exact files.

Restart ComfyUI and check that the checkpoints appear in a *Load Checkpoint* node's dropdown.

### 1.6 Point this repo at the pod (on your PC)
Pull the latest version of the repo first (`git pull`). Older versions of `pipeline/comfy.py` are
blocked by RunPod's proxy with `HTTP 403: error code: 1010`.

**Don't put the pod URL in `config/render.yaml`.** This repo is public, and anyone with the URL can
use your pod: run jobs on your GPU, read your files, and install custom nodes, which means running
their own code. Use a local override instead. It's git-ignored, so it's never committed.

Create `config/render.local.yaml` in your repo folder on your PC, containing one line:

```yaml
comfy_url: https://<POD_ID>-8188.proxy.runpod.net
```

The scripts read `config/render.yaml` first, then apply any keys from `render.local.yaml` on top.
For a one-off, the `COMFY_URL` environment variable overrides both
(PowerShell: `$env:COMFY_URL = "https://<POD_ID>-8188.proxy.runpod.net"`).

The pod ID changes whenever you deploy a **new** pod. Stopping and starting the same pod keeps it.
If a pod URL ever ends up somewhere public, terminate that pod and deploy a new one. The network
volume keeps everything.

Test:
```powershell
python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml --count 1
```
If it prints `cannot reach ComfyUI`, open the URL in a browser first. The pod may still be starting.

**Optional: a high-quality preset for 24 GB.** The repo's presets are tuned for an 8 GB card.
On a 4090 you can afford the full model with a hires pass. Add this under `presets:` in
`config/render.yaml`:

```yaml
  hq:
    checkpoint: RealVisXL_V5.0_fp16.safetensors
    steps: 30
    cfg: 4.5
    sampler: dpmpp_2m
    scheduler: karras
    hires: {scale: 1.5, denoise: 0.4, steps: 20}
```

---

## Part 2 — Lock the character (once)

If you already have a trained LoRA that passes the consistency test, skip to Part 3. The full
reasoning for each step is in [image-pipeline.md](image-pipeline.md) §1–§5. This is the
RunPod version.

### 2.1 Persona file
Copy `personas/ava.yaml` to `personas/<slug>.yaml`. Then:
- `identity`: concrete, distinctive and **never edited again**. Include eye colour, exact hair colour,
  length and cut, one or two marks (freckles, a beauty mark) and build.
- `trigger`: a rare token plus class word, e.g. `mrlx woman`.
- `style`: keep the phone-photo realism wording from `ava.yaml`.

Check the assembled prompts: `python -m pipeline.generate --persona personas/<slug>.yaml --shots shots/01-casting.yaml --dry-run`

### 2.2 Casting
```powershell
python -m pipeline.generate --persona personas/<slug>.yaml --shots shots/01-casting.yaml --preset hq
```
- Pick **one** hero face: front-facing, even light, clean eyes and teeth, something you'd recognise again.
- Unhappy? Edit `identity` and re-run. Don't proceed with a face you only half like.
- Upload the hero to the pod's `$COMFY/input/hero_<slug>.png`, either by dragging it into Jupyter's
  file browser or into any *Load Image* node.

### 2.3 Dataset
1. In `workflows/faceref-quality.json`, set `"image": "hero_<slug>.png"` and confirm
   `"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"`.
2. Render:
   ```powershell
   python -m pipeline.generate --persona personas/<slug>.yaml --shots shots/02-dataset.yaml --template workflows/faceref-quality.json
   ```
3. **Curate hard:** copy the 25–40 best images into `output/<slug>/picked/`.
   - **Keep:** clearly the same face; clean hands, eyes and teeth; real variety (about 40% close-ups,
     some ¾ and profile shots, about 20% full body, several lighting setups, outfits and places).
   - **Reject:** near-duplicates, waxy skin, anything distorted. Every flaw you keep gets trained in.
4. Build the training set:
   ```powershell
   python -m pipeline.dataset --persona personas/<slug>.yaml --src output/<slug>/picked --out datasets/<slug>
   ```

### 2.4 Train the LoRA on RunPod
1. **Stop** the ComfyUI pod. Deploy a **kohya_ss** template pod (RTX 4090) on the **same network volume**.
2. Upload `datasets/<slug>` to `/workspace/datasets/<slug>` (zip it, upload through Jupyter, then `unzip`).
3. In the kohya GUI (*LoRA* tab), use these settings:

   | Setting | Value on a 4090 |
   |---|---|
   | Pretrained model | `/workspace/.../checkpoints/RealVisXL_V5.0_fp16.safetensors` (SDXL checkbox on) |
   | Image folder | `/workspace/datasets/<slug>/img` (holds `10_<trigger>`) |
   | Network dim / alpha | 16 / 8 |
   | Resolution · bucketing | 1024,1024 · on |
   | Batch size | 2 |
   | Optimizer · LR | Adafactor (`scale_parameter=False relative_step=False warmup_init=False`) · 1e-4 |
   | Train text encoder | off (UNet only) |
   | Mixed / save precision | **bf16** / bf16. Don't use fp16 here: the SDXL VAE can produce NaN errors in fp16 |
   | Gradient checkpointing · cache latents | on · on |
   | Epochs · save every | 10 · 2 epochs |
   | Total steps | about 3,000–4,000 (images × 10 repeats × epochs ÷ batch) |

4. Train (about 30–60 minutes). Copy every saved epoch (`<slug>_v1-000004.safetensors`, `-000006`, …,
   final) into `$COMFY/models/loras/`.
5. **Stop the kohya pod.** Start the ComfyUI pod again.

### 2.5 Consistency test (choose the epoch and strength)
In `personas/<slug>.yaml` set `lora.file` to one epoch and `lora.strength: 0.85`, then:
```powershell
python -m pipeline.generate --persona personas/<slug>.yaml --shots shots/03-consistency.yaml --preset hq
```
The seeds are fixed, so switch `lora.file` and re-run to compare epochs fairly.

| Result | Action |
|---|---|
| Same person in **8+ of 10** | **Locked.** Write down the epoch and strength. Never change them casually. |
| Face drifts | Use a later epoch, or raise strength to 0.9–1.0 |
| Same outfit or background creeping into every image | Use an earlier epoch, or lower strength to 0.7–0.8 |
| Right face, plastic skin | Go to Part 3. The workflow fixes this; retraining won't |

---

## Part 3 — The hyper-realism still workflow (build once)

The built-in graph (`--preset hq`) gives you LoRA + hires. For hyper-realism, add two
more stages: **FaceDetailer**, which re-renders the face at high resolution, and a **model
upscale**. You'll build it once in the ComfyUI canvas, export it, and drive it from the repo
like the faceref workflow.

```
[A] Checkpoint ─► LoRA (0.85) ─► KSampler 30 steps @ 832×1216
[B]          └─► Latent upscale 1.5× ─► KSampler denoise 0.40          (texture pass)
[C]          └─► VAE Decode ─► FaceDetailer (same model + prompt)       (face at ~1024 px)
[D]          └─► Upscale model 4× ─► scale 0.5 ─► Save                  (≈ 2496×3648)
```

### 3.1 Build it node by node
Double-click the canvas to search for each node.

**[A] Base**
1. **Load Checkpoint**: `RealVisXL_V5.0_fp16.safetensors`.
2. **LoraLoader**: `model` and `clip` from the checkpoint. `lora_name` = your locked epoch.
   `strength_model` = `strength_clip` = your locked strength (e.g. 0.85).
3. **CLIP Text Encode** (positive), `clip` from the LoRA loader. Text:
   `mrlx woman, %PROMPT%`. **Type your trigger yourself.** With `--template`, the script doesn't add the
   trigger or the LoRA. It only replaces `%PROMPT%` (identity + scene + style).
4. **CLIP Text Encode** (negative), `clip` from the LoRA loader. Text: `%NEGATIVE%`.
5. **Empty Latent Image**: 832×1216 (the script overwrites this per shot size).
6. **KSampler**: model from the LoRA loader. Steps **30**, cfg **4.5**, sampler **dpmpp_2m**, scheduler
   **karras**, denoise **1.0**.

**[B] Hires texture pass**
7. **Upscale Latent By**: `nearest-exact`, **1.5**.
8. **KSampler** (second): the same model and conditioning, latent from step 7. Steps **20**, cfg **4.5**,
   dpmpp_2m / karras, denoise **0.40**. Values from 0.35 to 0.45 work: lower keeps the composition, higher adds
   more skin detail but can change the face.
9. **VAE Decode**, with the VAE from the checkpoint.

**[C] FaceDetailer.** This is the biggest realism gain for anything wider than a close-up.
10. **UltralyticsDetectorProvider**: `bbox/face_yolov8m.pt`.
11. **FaceDetailer**:
    - `image` from VAE Decode;
    - `model` and `clip` **from the LoRA loader** (otherwise the re-rendered face isn't your character);
    - `vae` from the checkpoint; `positive` and `negative` from steps 3 and 4;
    - `bbox_detector` from step 10.

    | FaceDetailer setting | Value |
    |---|---|
    | guide_size / max_size | 768 / 1024 |
    | steps · cfg · sampler · scheduler | 20 · 4.5 · dpmpp_2m · karras |
    | denoise | **0.35–0.45** (higher = more detail, more risk of identity drift) |
    | feather | 8 |
    | noise_mask · force_inpaint | on · on |
    | bbox_threshold · bbox_dilation · bbox_crop_factor | 0.5 · 10 · 3.0 |
    | drop_size | 10 |
    | sam_model_opt | leave empty (bounding box is enough for faces) |

**[D] Final upscale**
12. **Load Upscale Model**: `4x-UltraSharp` (or `4x_NMKD-Siax_200k`, which is softer and more natural on skin).
13. **Upscale Image (using Model)**: image from FaceDetailer.
14. **Upscale Image By**: `lanczos`, **0.5**. 4× then 0.5× gives a clean 2× result.
15. **Save Image**.

> Optional replacement for [D]: **Ultimate SD Upscale** with the LoRA model, upscale_by 2,
> tile 1024, denoise **0.2**, seam fix off. It's slower, but it adds real micro-texture instead of
> sharpening. Never go above 0.25 denoise or the face starts to drift.

### 3.2 Save it as a repo template
1. Test it once in the UI: replace `%PROMPT%` with a real prompt, run, and check the result.
2. Put `%PROMPT%` back, so the text reads `mrlx woman, %PROMPT%`.
3. **Workflow → Export (API)** → save as `workflows/locked-realism.json` in this repo.
4. Render any shot list with it:
   ```powershell
   python -m pipeline.generate --persona personas/<slug>.yaml --shots shots/week-2026-10-12.yaml --template workflows/locked-realism.json --count 4
   ```
   The script fills in prompt, negative, seed, size and filename, and logs everything to
   `output/<slug>/log.csv`. The log's `lora` column will be empty with templates, because the LoRA is
   baked into the workflow. Note the LoRA version in the shot list's comments instead.

### 3.3 Optional: extra lock with IPAdapter
If one or two faces still drift, add **IPAdapter Advanced** between the LoRA loader and both
KSamplers. Feed it the hero image with weight **0.3–0.4**, start_at 0, end_at **0.6**. Keep it low:
the LoRA does the locking, and IPAdapter only nudges. High weights bring back the waxy look.

---

## Part 4 — Prompting for hyper-realism

### 4.1 The formula
With `--template`, the full prompt the model sees is:

```
<trigger> , <identity from the persona> , <scene from the shot list> , <style from the persona>
```

So the **shot list only describes the scene**. Write it in this order:

```
[shot type] + [action / pose] + [outfit] + [location] + [light] + [camera feel] + [imperfection]
```

| Slot | Good choices |
|---|---|
| Shot type | close-up, head-and-shoulders, waist-up, full body, mirror selfie, over-the-shoulder |
| Action | concrete and mid-movement: "tucking hair behind her ear", "laughing at her phone", "reaching for a cup" |
| Outfit | specific fabric and colour: "oatmeal ribbed knit cardigan", "faded black denim jacket" |
| Location | a real, slightly messy place: "small kitchen with dishes in the sink", "parked car, seatbelt on" |
| Light | "overcast daylight from a window on the left", "warm lamp light at night", "harsh midday sun, hard shadows", "phone flash in a dark bar" |
| Camera feel | "shot on iPhone, front camera", "slightly off-centre framing", "candid, taken by a friend" |
| Imperfection | "flyaway hairs", "slight motion blur on her hand", "creased shirt", "smudged mascara" |

### 4.2 Words to use and words to avoid
- **Use** (already in the persona `style`; add more in the scene when needed): raw unedited phone photo,
  candid, visible pores, natural skin texture, fine facial hair, uneven lighting, slight sensor noise,
  real-world surroundings.
- **Avoid**: masterpiece, best quality, 8k, ultra detailed, hyperrealistic, instagram aesthetic,
  bokeh, studio lighting, flawless skin, perfect, beautiful, gorgeous. These all pull toward the
  glossy "AI model" face.
- **Negative** (persona `negative`, plus the safety terms the script adds): smooth skin, beauty filter,
  airbrushed, glossy skin, porcelain skin, plastic, oversaturated, hdr, cgi, 3d render, illustration,
  perfect symmetry, heavy makeup.

### 4.3 Example scene lines (shot-list `prompt:` values)
```yaml
shots:
  - id: kitchen-morning
    tier: public
    prompt: "waist-up, leaning on a small kitchen counter holding a mug with both hands, oversized grey hoodie, messy bun with flyaway hairs, dishes in the background, overcast window light from the left, candid, taken by a friend"
  - id: car-selfie
    tier: public
    prompt: "selfie from the driver's seat of a parked car, seatbelt on, white t-shirt, sunglasses pushed up into her hair, harsh afternoon sun through the windshield, front camera, slightly off-centre"
  - id: bar-flash
    tier: public
    prompt: "head-and-shoulders, laughing in a dim bar, black slip top, phone flash photo, slight red-eye, dark background, drink glass in foreground, slight motion blur"
  - id: street-overcast
    tier: public
    size: tall
    prompt: "full body walking across a crosswalk, long camel coat, tote bag, wet pavement, overcast grey sky, shot from across the street, natural colours"
```

### 4.4 Tuning cheatsheet
| You see | Change |
|---|---|
| Face is right but the skin looks plastic | FaceDetailer denoise +0.05; use NMKD-Siax instead of UltraSharp; add a realism LoRA at 0.3 (see 4.5) |
| Over-contrast or over-saturated | cfg 4.5 → 3.5 in all samplers |
| Face drifts in wide shots | Raise FaceDetailer guide_size to 1024; lower its denoise to 0.35 |
| Same expression in every image | Put a concrete action and emotion in the scene ("mid-laugh", "squinting into the sun") |
| Background looks like a render | Name ordinary objects and some mess; add "real-world surroundings, natural colours" |
| Hands are wrong | Give the hands a job ("holding a mug with both hands"); inpaint (Part 5.3) |

### 4.5 Optional: realism LoRA
Search Civitai for an **SDXL** "amateur photo" or "skin detail" LoRA. Add a second **LoraLoader**
*after* your character LoRA at **0.3–0.5** strength. Re-run the consistency test after adding it: some
realism LoRAs pull the face toward their own training data.

---

## Part 5 — Pick and QC stills

### 5.1 Identity check
- Open the hero image and the candidate side by side. Compare **eye colour, eye shape, nose, lip shape,
  the marks** (freckles, beauty mark) and **hairline**.
- If you have to talk yourself into it, it's a reject.

### 5.2 Realism and defect check (reject on any)
- [ ] Hands: count the fingers; check nails and knuckles.
- [ ] Eyes: same size and direction, catchlights on the same side as the light.
- [ ] Teeth, ears, earrings: symmetrical where they should be.
- [ ] Text, signs and logos: none garbled.
- [ ] Straight lines: door frames, tiles and railings stay straight.
- [ ] Skin: pores visible at 100% zoom, no wax.
- [ ] Nothing reads as young; correct tier for the destination.
- [ ] **Phone test:** view it at Instagram size on your phone. AI tells show up faster there.

### 5.3 Fix small defects with inpainting
1. In ComfyUI, drop the image on a **Load Image** node. Right-click → **Open in MaskEditor** and paint the bad hand or eye.
2. Use **VAE Encode (for Inpainting)** with grow_mask_by 8, the LoRA model, the same prompt, and a
   KSampler with denoise **0.5–0.7**.
3. Render 4 options and keep the best. For faces, prefer re-running FaceDetailer over manual inpainting.

### 5.4 Export stills
```powershell
python -m pipeline.export --src output/<slug>/<week>/picked --format feed story --grain 4
```
This crops to 1080×1350 and 1080×1920, adds subtle phone-sensor grain and **strips all metadata**.
Never post the raw PNGs: they contain your full prompts and workflow.

---

## Part 6 — Video from a locked still

The rule: **video never generates the face from text.** Every clip starts from a still that
already passed Part 5. Image-to-video keeps the identity; text-to-video doesn't.

### 6.1 Prepare the start frame
1. Pick a passed still with the **face large in the frame** (close-up to waist-up). Full-body shots
   lose the face during motion.
2. Re-render or crop it to **9:16**. Wan works best at **720×1280** (or 480×832 for fast tests).
   Use the export script (`--format story`), then resize to 720×1280.
3. Upload it to `$COMFY/input/`.

### 6.2 Option A — Wan 2.2 image-to-video on RunPod (open model, no per-clip fees)
1. In ComfyUI: **Workflow → Browse Templates → Video → "Wan 2.2 14B Image to Video"**.
2. ComfyUI shows the missing models with download links. They come from Hugging Face
   `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (the I2V **high-noise** and **low-noise** 14B diffusion
   models, the **umt5_xxl** text encoder and **wan_2.1_vae**). Use the **fp8_scaled** versions on a 4090.
   Download each into the folder the dialog names (`diffusion_models`, `text_encoders`, `vae`).
   That's about 30–40 GB, so it stays on your network volume.
3. Set the template:

   | Setting | Value |
   |---|---|
   | Load Image | your start frame |
   | Width × height | 720×1280 (finals) · 480×832 (tests) |
   | Length | **81** frames, about 5 s at Wan's native 16 fps |
   | Steps / CFG | the template defaults (the high-noise model runs the first half of the steps, low-noise the rest) |
   | Seed | fixed while you tune the prompt, then vary it to get alternatives |

   *Speed option:* a Wan 2.2 "lightx2v" or "lightning" 4-step LoRA, loaded on both the high- and low-noise
   models, cuts render time several-fold. Use the 4–8 steps and **CFG 1** that its model card gives.
   Expect slightly less motion detail.
4. **Motion prompt:** describe **only the motion and the camera**, not the face or outfit (the image
   carries those):
   ```
   she looks up from her phone and smiles at the camera, tucks a strand of hair behind her ear,
   subtle handheld phone camera movement, natural daylight, realistic motion
   ```
   **Negative:** `morphing, warping, distorted face, extra fingers, flicker, cartoon, cgi, blurry face, static image`
5. Render 2–4 seeds per clip and keep the best. On a 4090, a 720p clip takes several minutes. Test at
   480p first, then render finals at 720p. For lots of 720p clips, rent a 48 GB card for that session.

### 6.3 Option B — hosted image-to-video (Kling, Veo, Hailuo, Seedance)
Upload the same start frame, use the same motion prompt, 5 s, 9:16. Kling is a strong default for
human motion; Veo is the most cinematic but strictest on content. Run a **bake-off** once: the same
still and prompt through each model, pick a winner, and standardise on it. Download the highest quality
available.

### 6.4 Keeping the identity in motion
- **One simple action per clip.** Turning, hair flips and hands crossing the face are where faces melt.
- Keep head turns **under about 45°**. Profile-to-front turns invent a new face.
- Prefer **slow camera moves** (push-in, slight handheld) over subject running or dancing.
- **5 seconds maximum per generation.** Longer clips drift.
- Reject any clip where the face changes, even briefly. Viewers notice.

### 6.5 Longer videos: chain clips
1. Take the last frame of a good clip:
   ```bash
   ffmpeg -sseof -0.1 -i clip1.mp4 -frames:v 1 -update 1 clip1_last.png
   ```
2. **Check the face on that frame.** If it drifted, run it through FaceDetailer (Part 3) with the LoRA first.
   This re-locks the identity before the next clip.
3. Use it as the start frame of clip 2. Repeat 2–4 times, then join:
   ```bash
   printf "file 'clip1.mp4'\nfile 'clip2.mp4'\nfile 'clip3.mp4'\n" > list.txt
   ffmpeg -f concat -safe 0 -i list.txt -c copy joined.mp4
   ```
   Alternatively, use Wan's **first–last-frame** (FLF2V) template with two locked stills as the endpoints. This
   is the most reliable way to keep the face for transitions.

### 6.6 Finish: smooth, upscale, grain, export
1. **Interpolate** 16 → 32 fps in ComfyUI: Load Video (VideoHelperSuite) → **RIFE VFI** (multiplier 2)
   → Video Combine at 32 fps. Motion looks far less "AI" at 30+ fps.
2. **Upscale** 720×1280 → 1080×1920 and **add grain**, then encode for Instagram/TikTok with metadata stripped:
   ```bash
   ffmpeg -i joined_32fps.mp4 \
     -vf "scale=1080:1920:flags=lanczos,noise=alls=5:allf=t,fps=30" \
     -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
     -map_metadata -1 -movflags +faststart -an final_reel.mp4
   ```
   `-an` drops audio. Add music **inside the app** so it's licensed.
3. **Video QC** (reject on any): face change at any frame, flicker, melting hands, warping background,
   garbled text, anything that reads as young.
4. Post with the platform's **AI-generated** label switched on.

---

## Part 7 — Session routine (keep the bill small)

**Start**
1. RunPod → your pod → **Start**. Open a Jupyter terminal, run `/workspace/start-comfy.sh`, and wait until ComfyUI opens in the browser.
2. If you deployed a new pod, update `comfy_url` in `config/render.local.yaml`.

**Work**
3. Stills: render drafts with `--count 4` → pick → re-render or finish the winners → QC → export.
4. Video: start frames → test at 480p → finals at 720p → interpolate → ffmpeg export.
5. Download what you keep. Renders also stay in `$COMFY/output` on the volume.

**Stop**
6. **Stop** the pod as soon as you finish. GPU billing stops; only storage continues.
7. **Terminate** only pods whose disk you don't need. The network volume survives.
8. Once a month, clean `$COMFY/output` on the volume, since storage is billed per GB.

**Never** leave a pod running overnight "for tomorrow". A 4090 left on costs about $8–16 a day.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cannot reach ComfyUI` from the scripts | Pod still starting, or the pod ID changed. Open the proxy URL in a browser; update `comfy_url` in `config/render.local.yaml` |
| `HTTP 403: error code: 1010` | Old `pipeline/comfy.py`. Run `git pull`; the current version sends a User-Agent RunPod accepts |
| Nodes, packages or models gone after a stop | They were outside `/workspace`. Follow 1.2–1.3 so ComfyUI and its venv live on the volume |
| ComfyUI link shows "502" | ComfyUI isn't running. Run `/workspace/start-comfy.sh` and check `tail -f /workspace/comfy.log` |
| `workflow rejected` with a missing node class | A custom node isn't installed on this pod (Part 1.4). Install it and restart ComfyUI |
| Checkpoint or LoRA not in the dropdown | Wrong folder, or ComfyUI not restarted. Check `ls $COMFY/models/loras` |
| IPAdapter "ClipVision model not found" | The CLIP vision file must be named exactly `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` |
| FaceDetailer changes the face | Its `model` input isn't from the LoRA loader, or its denoise is above 0.45 |
| Every image has the same outfit or background | LoRA over-trained. Use an earlier epoch or strength 0.7–0.8 |
| Skin is waxy even with FaceDetailer | Drop IPAdapter (or keep its weight ≤ 0.4); use the full checkpoint, not Lightning; lower cfg to 3.5–4 |
| kohya: NaN loss or black samples | Use bf16, not fp16 (SDXL VAE) |
| Wan out of memory on a 4090 | Use the fp8_scaled models, render at 480×832, or shorten to 49 frames. For 720p volume, rent a 48 GB card |
| Video face drifts mid-clip | Start from a closer shot, use one action, shorten the clip, or use first–last-frame with two locked stills |
| Pod won't start ("no GPUs available") | The volume's datacenter is out of that GPU. Wait, pick another GPU type there, or copy the volume to another region |

## Files this guide adds or uses

| Path | What |
|---|---|
| `personas/<slug>.yaml` | Identity, trigger, locked LoRA file and strength |
| `shots/*.yaml` | Scenes only (Part 4.1 order) |
| `config/render.yaml` | Shared render presets; optional `hq` preset |
| `config/render.local.yaml` | **You create this** (git-ignored): `comfy_url` = your RunPod proxy URL |
| `workflows/faceref-quality.json` | Dataset building (Part 2.3) |
| `workflows/locked-realism.json` | **You create this** in Part 3.2: LoRA + hires + FaceDetailer + upscale |
| `output/<slug>/log.csv` | Every render's prompt, seed and job id. Keep it |
