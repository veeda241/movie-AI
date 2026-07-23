# Own model — manual features YOU add

Open Gradio **Checklist** for live ✅/⬜. This file mirrors the same list.

## Phase A — data (do these first)

| Who | Feature | Where |
| --- | --- | --- |
| **YOU** | Copy real MP4/WebM clips **or** download via Pexels | `data/video_lab/raw/` / Data → Pexels |
| **LAB** | Scene cut + optical-flow filter | Data tab (min/max flow) |
| **LAB** | Aspect `bucket` on each row | Curate |
| **YOU** | Base `caption` + camera / lighting / motion | **Labels** tab |
| **YOU** | Aesthetic / tags / negative | **Labels** tab |
| **LAB** | `dense_caption` | Data → Recaption (or Labels save) |

## Phase B — train our upgraded model

| Who | Feature | Where |
| --- | --- | --- |
| **LAB** | Causal VAE ~8× spatial / 4× temporal | Train → VAE |
| **LAB** | Spacetime-patch DiT + RoPE-lite | Train → DiT |
| **LAB** | Curriculum Stage 1 / 2 / 3 | Train stage radio |
| **LAB** | Generate sample | Generate tab |

## Phase C — optional / later

- Experimental CogVideoX LoRA (Gradio tab; separate from own VAE/DiT)  
- Video DPO / RLAIF  
- CFG / step distillation  
- Multi-million clip ingest  

## Example manifest row

```json
{
  "path": "C:/hackathon/Gemini_CLI/movie-AI/data/video_lab/raw/my_clip.mp4",
  "caption": "neon city street at night, wet asphalt, slow dolly forward",
  "dense_caption": "neon city street at night, wet asphalt, slow dolly forward. Camera: dolly forward. Lighting: neon night. Motion: camera push-in, light rain. Style tags: city, rain, cinematic. Avoid: watermark, shaky blur, logos",
  "camera": "dolly forward",
  "lighting": "neon night",
  "motion": "camera push-in, light rain",
  "aesthetic": 8,
  "tags": ["city", "rain", "cinematic"],
  "negative": "watermark, shaky blur, logos",
  "bucket": "square_128",
  "flow_mean": 0.8,
  "flow_var": 0.3,
  "fps": 24,
  "frames": 48
}
```

Template: `data/video_lab/manifest_template_example.jsonl`
