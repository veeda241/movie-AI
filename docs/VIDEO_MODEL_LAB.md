# Own Video Model Lab

Gradio toolkit for **our own** Causal 3D-VAE + Spatiotemporal DiT.
No Hugging Face / CogVideoX engines in the UI.

## Phases

| Phase | Focus |
| --- | --- |
| **A — Data** | Optical-flow filter, dense recaption, aspect buckets (letterbox) |
| **B — Architecture** | Stronger causal VAE (~8×S / 4×T), spacetime-patch DiT + RoPE-lite, curriculum stages |
| **C — Later** | Video DPO, CFG distill, open SOTA backbone fine-tune |

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-video-lab.txt
python gradio_video_lab.py
```

Open http://127.0.0.1:7860

### Workflow

1. **Checklist** — live YOU vs LAB status  
2. **Data** — smoke or curate `data/video_lab/raw/` (enable optical-flow gates) → **Recaption**  
3. **Labels** — edit caption / camera / lighting / motion / aesthetic  
4. **Train** — pick Stage 1/2/3 + bucket → Train VAE → Train DiT  
5. **Generate** — prompt → MP4  

Manual feature list: [OWN_MODEL_CHECKLIST.md](OWN_MODEL_CHECKLIST.md)

## Layout

```text
video_lab/
  app.py
  models/          # causal_vae.py, dit.py, text_encoder.py
  data/            # curate, optical_flow, recaption, buckets, dataset, smoke
  train/           # train_vae, train_dit, curriculum
  infer/           # research_generate
data/video_lab/
  raw/ smoke/ manifest.jsonl
outputs/video_lab/
  vae/ dit/ samples/
```

## Notes

- After Phase B architecture changes, **retrain VAE and DiT** (old checkpoints are incompatible).
- Train **one aspect bucket at a time** (VRAM-safe).
- Smoke data is for plumbing only — add real clips for quality.
