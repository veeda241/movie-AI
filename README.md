# Movie-AI

Two clear apps in one repo. They share code but **do not share the same video model**.

| App | What it is | Run | Port |
|-----|------------|-----|------|
| **Movie Flow** | Product studio (image / video / multi-agent movie) | API + Next.js | `8000` + `3000` |
| **Own Video Model Lab** | Research bench to train **your** VAE+DiT | Gradio | `7860` |

```text
Movie Flow:     web (:3000) → api (:8000) → movie_pipeline (Wan remote or local fallback)
Own Model Lab:  gradio_video_lab.py (:7860) → video_lab/ (local VAE + DiT train/generate)
```

---

## Fresh machine setup

```powershell
git clone -b feature/video-lab-own-model https://github.com/veeda241/movie-AI.git
cd movie-AI

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` (`HF_TOKEN`, `PEXELS_API_KEY` as needed).

### Own Video Model Lab (extra)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video-lab.txt
# GPU recommended:
# .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Optional CogVideo LoRA (experimental only)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cogvideo.txt
```

---

## Run Movie Flow (product)

```powershell
# Terminal 1 — API
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Web
cd web
copy .env.local.example .env.local
npm install
npm run dev
```

Open http://localhost:3000

Ops monitor (optional): `.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py`

---

## Run Own Video Model Lab

```powershell
.\.venv\Scripts\python.exe gradio_video_lab.py
```

Open http://127.0.0.1:7860

### Lab workflow

1. **Data** — download Pexels / HF Wan actions, or drop MP4s into `data/video_lab/raw/` → Curate → Recaption  
2. **Labels** — fix captions  
3. **Train** — VAE then DiT (`niche_laptop` on 6GB GPUs)  
4. **Generate** — prompts in your training niche; match train resolution/frames  

CLI train:

```powershell
.\.venv\Scripts\python.exe scripts\download_hf_wan_actions.py
.\.venv\Scripts\python.exe scripts\train_niche.py --profile niche_laptop --vae-steps 3000 --dit-steps 18000
```

Checkpoints: `outputs/video_lab/vae/`, `outputs/video_lab/dit/`  
Samples: `outputs/video_lab/samples/`

Docs: [docs/VIDEO_MODEL_LAB.md](docs/VIDEO_MODEL_LAB.md) · [docs/NICHE_TRAINING.md](docs/NICHE_TRAINING.md) · [docs/OWN_MODEL_CHECKLIST.md](docs/OWN_MODEL_CHECKLIST.md)

---

## Honest expectations

- Lab quality ≈ your data + GPU time. ~hundreds of niche clips ≠ Wan/Veo.  
- Movie Flow video uses **remote Wan** (needs `HF_TOKEN`) or a **local cinematic fallback** — not the lab DiT.  
- Gradio tab **Experimental (CogVideo LoRA)** is optional and separate from own VAE/DiT.

---

## Repo map

```text
api/              Movie Flow FastAPI
web/              Movie Flow Next.js UI
movie_pipeline/   Agents + Motif video client
video_lab/        Own VAE/DiT + Gradio app
gradio_video_lab.py
streamlit_app.py
scripts/          download_*, train_niche, train_lora
docs/
data/video_lab/   raw clips + manifests (local; not in git)
outputs/video_lab/ checkpoints + samples (local; not in git)
```
