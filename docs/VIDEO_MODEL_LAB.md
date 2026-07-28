# Own Video Model Lab

Gradio toolkit for **our own** Causal 3D-VAE + Spatiotemporal DiT.

The default path is local research training. An optional **Experimental (Wan LoRA)** tab can fine-tune **Wan2.1-T2V-1.3B** with LoRA; that is **not** the own-model path and needs `requirements-cogvideo.txt`.

## Phases

| Phase | Focus |
| --- | --- |
| **A — Data** | Optical-flow filter, dense recaption, aspect buckets (letterbox) |
| **B — Architecture** | Causal VAE (~8×S / 4×T), spacetime-patch DiT + RoPE-lite, curriculum |
| **C — Optional** | Wan LoRA fine-tune; later DPO / distill / larger backbones |

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video-lab.txt
.\.venv\Scripts\python.exe gradio_video_lab.py
```

Open http://127.0.0.1:7860

### Workflow

1. **Checklist** — YOU vs LAB status  
2. **Data** — Pexels / HF Wan / drop MP4s → Curate → Recaption  
3. **Labels** — caption / camera / lighting / motion / aesthetic  
4. **Train** — Stage / niche → Train VAE → Train DiT  
5. **Generate** — niche prompts at train resolution/frames  

Manual feature list: [OWN_MODEL_CHECKLIST.md](OWN_MODEL_CHECKLIST.md)  
Niche GPU profiles: [NICHE_TRAINING.md](NICHE_TRAINING.md)

## Layout

```text
video_lab/
  app.py
  models/          # causal_vae.py, dit.py, text_encoder.py
  data/            # curate, optical_flow, recaption, buckets, pexels, hf_wan
  train/           # train_vae, train_dit, curriculum, niche_profile
  infer/           # research_generate (+ optional finetune_generate)
data/video_lab/
  raw/ smoke/ manifest.jsonl
outputs/video_lab/
  vae/ dit/ samples/ lora/
```

## Notes

- After architecture changes, **retrain VAE and DiT**.  
- Train one aspect bucket at a time on small GPUs.  
- Smoke data is plumbing only.  
- Movie Flow (port 3000) does **not** use these checkpoints automatically.
