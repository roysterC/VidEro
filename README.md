# VidEro

Tooling for a fully synthetic AI influencer: render images locally with ComfyUI, lock the
persona's identity with a LoRA, and export posts for social media.

- [docs/image-pipeline.md](docs/image-pipeline.md): **start here.** Setup on a home PC and the step-by-step workflow
- [docs/content-plan.md](docs/content-plan.md): overall content and funnel plan (images now, video later)
- [docs/guide/AI-Influencer-Guide.pdf](docs/guide/AI-Influencer-Guide.pdf): shareable step-by-step guide (PDF). Edit `ai-influencer-guide.html`, then rebuild with `python docs/guide/build.py`

```powershell
pip install -r requirements.txt
python -m pipeline.generate --persona personas/ava.yaml --shots shots/01-casting.yaml --dry-run
```

| Command | Does |
|---|---|
| `python -m pipeline.generate` | Renders a shot list for a persona through ComfyUI and logs every image |
| `python -m pipeline.dataset` | Turns hand-picked renders into a captioned LoRA training set |
| `python -m pipeline.export` | Crops and resizes for Instagram, Stories and X, and strips metadata |

Tests: `python -m unittest discover -s tests -t .`

Rules built into the pipeline: the persona is fictional and 21+, prompts with youth or
real-person-likeness cues are refused, and public-tier shots are kept non-nude.
