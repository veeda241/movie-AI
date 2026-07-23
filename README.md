# Movie-AI

Two apps in one repository. They share the repo but **do not use the same video model**.

| App | Purpose | How to run | URL |
|-----|---------|------------|-----|
| **Movie Flow** | Product studio — image, video, multi-agent movie | FastAPI + Next.js | http://localhost:3000 |
| **Own Video Model Lab** | Train/test **your** Causal VAE + DiT | Gradio | http://127.0.0.1:7860 |

```text
Movie Flow:      web (:3000)  →  api (:8000)  →  movie_pipeline
                 (Wan remote if HF_TOKEN set, else local cinematic fallback)

Own Model Lab:   gradio_video_lab.py (:7860)  →  video_lab/
                 Data → Labels → Train VAE/DiT → Generate
```

**Branch with this stack:** `feature/video-lab-own-model`  
`main` is older and does **not** include the Video Model Lab.

---

## Prerequisites

- Windows / PowerShell (examples below)
- Python **3.10+**
- Node.js **18+** (Movie Flow web only)
- Optional: NVIDIA GPU + CUDA PyTorch (strongly recommended for the lab)

---

## 1. Clone the correct branch

```powershell
cd "C:\Users\AI\Downloads\Spiderboy projects"   # or any folder you prefer
git clone -b feature/video-lab-own-model https://github.com/veeda241/movie-AI.git
cd movie-AI
```

If you already cloned `main`:

```powershell
cd movie-AI
git fetch origin
git checkout feature/video-lab-own-model
git pull origin feature/video-lab-own-model
```

Confirm lab files exist:

```powershell
dir requirements-video-lab.txt
dir video_lab
dir gradio_video_lab.py
```

---

## 2. Python environment

```powershell
py -3 -m venv .venv
# If py fails:  python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You do **not** need `Activate.ps1`. Always call:

```powershell
.\.venv\Scripts\python.exe ...
```

### Env file

```powershell
copy .env.example .env
notepad .env
```

Useful keys:

| Key | Used by |
|-----|---------|
| `HF_TOKEN` | Movie Flow remote LLM / Wan video |
| `PEXELS_API_KEY` | Lab stock-video downloads ([Pexels API](https://www.pexels.com/api/)) |
| `JWT_SECRET` | Movie Flow auth (change in production) |

---

## 3. Own Video Model Lab setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video-lab.txt
```

GPU (recommended):

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Optional — CogVideo LoRA (experimental)

Separate from own VAE/DiT. Needs more disk/VRAM:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cogvideo.txt
```

---

## 4. Run apps

### A) Own Video Model Lab (Gradio)

```powershell
.\.venv\Scripts\python.exe gradio_video_lab.py
```

Open **http://127.0.0.1:7860**

Tabs: Checklist → Data → Labels → Train → Generate  
(Optional tab: **Experimental (CogVideo LoRA)**)

### B) Movie Flow (product)

**Terminal 1 — API**

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Web**

```powershell
cd web
copy .env.local.example .env.local
npm install
npm run dev
```

Open **http://localhost:3000**

### C) Streamlit ops monitor (optional)

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

---

## 5. Lab data + train (CLI)

Raw videos are **not** in git. Download on each machine:

```powershell
# HF Wan action packs (~10 clips each)
.\.venv\Scripts\python.exe scripts\download_hf_wan_actions.py

# Optional Pexels niche stock (needs PEXELS_API_KEY)
.\.venv\Scripts\python.exe scripts\download_pexels.py --query "ocean waves" --count 50
```

Curate → train:

```powershell
.\.venv\Scripts\python.exe -c "from video_lab import MANIFEST_PATH, RAW_DIR; from video_lab.data.curate import build_manifest_from_raw; from video_lab.data.recaption import recaption_manifest; build_manifest_from_raw(RAW_DIR, manifest_path=MANIFEST_PATH, run_scene_cut=False, use_optical_flow=True); recaption_manifest(); print(MANIFEST_PATH)"

.\.venv\Scripts\python.exe scripts\check_project.py
.\.venv\Scripts\python.exe scripts\train_niche.py --profile niche_laptop --vae-steps 3000 --dit-steps 18000
```

| Profile | GPU | Size |
|---------|-----|------|
| `niche_laptop` | ~6GB | 256² × 8f |
| `niche_24gb` | ~24GB | 256² × 24f |

Outputs:

- Checkpoints → `outputs/video_lab/vae/`, `outputs/video_lab/dit/`
- Samples → `outputs/video_lab/samples/`

More detail: [docs/NICHE_TRAINING.md](docs/NICHE_TRAINING.md) · [docs/VIDEO_MODEL_LAB.md](docs/VIDEO_MODEL_LAB.md) · [docs/OWN_MODEL_CHECKLIST.md](docs/OWN_MODEL_CHECKLIST.md)

---

## Requirements files

| File | Install when |
|------|----------------|
| `requirements.txt` | Always (Movie Flow API / core) |
| `requirements-video-lab.txt` | Own VAE/DiT Gradio lab |
| `requirements-model.txt` | Extra generative / SDXL helpers |
| `requirements-cogvideo.txt` | Experimental CogVideo LoRA only |

---

## Honest expectations

- Lab quality ≈ **your data + train time**. Hundreds of niche clips ≠ Wan / Veo.
- Generate with **in-domain** prompts and the **same resolution/frames** you trained.
- Movie Flow video ≠ lab DiT. Lab does not automatically power the product studio.
- Videos provided by [Pexels](https://www.pexels.com) when using that downloader — credit creators.

---

## Repo map

```text
api/                 Movie Flow FastAPI (auth, credits, jobs)
web/                 Movie Flow Next.js UI
movie_pipeline/      Multi-agent planning + Motif video client
video_lab/           Own VAE/DiT models, train, infer, Gradio UI
gradio_video_lab.py  Lab launcher (:7860)
streamlit_app.py     Ops monitor
scripts/             download_pexels, download_hf_wan_actions, train_niche, train_lora
docs/                Lab + niche guides
data/video_lab/      Local clips + manifests (gitignored)
outputs/video_lab/   Local checkpoints + samples (gitignored)
.env.example         Template for secrets
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `requirements-video-lab.txt` not found | You are on `main` — checkout `feature/video-lab-own-model` |
| `.\.venv\Scripts\python.exe` not found | Create venv: `py -3 -m venv .venv` |
| `Activate.ps1` fails | Skip it; call `.\.venv\Scripts\python.exe` directly |
| Import / missing files | `.\.venv\Scripts\python.exe scripts\check_project.py` |
| Flat / meaningless generate | Train on real clips; use niche prompts; match 256 / ~8 frames |
| No GPU in lab | Install CUDA torch (see §3) |
| Movie Flow video looks fake | Set `HF_TOKEN` for remote Wan, or accept local fallback |
