# Movie-AI / Movie Flow

Two products in one repo:

1. **Movie Flow** — a Google Flow–style creative studio (image, video, multi-agent movie) with a Next.js UI and FastAPI backend.
2. **Own Video Model Lab** — a Gradio R&D bench to train and test **our own** VAE + DiT video model (no Hugging Face / CogVideoX in that UI).

```text
┌─────────────────────────────────────────────────────────────┐
│  Movie Flow (product)                                       │
│  web/ (Next.js :3000)  →  api/ (FastAPI :8000)              │
│                           → movie_pipeline/ (agents + gen)  │
│  streamlit_app.py (ops monitor)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Own Video Model Lab (research)                             │
│  gradio_video_lab.py → video_lab/ (:7860)                   │
│  Data → Labels → Train VAE/DiT → Generate                   │
└─────────────────────────────────────────────────────────────┘
```

They share the repo but run on **different ports**. The lab does **not** automatically power Movie Flow video yet.

---

## What each part does

| Piece | Path | Purpose |
|--------|------|---------|
| Product UI | `web/` | Landing, login, Create studio, projects, billing, team |
| SaaS API | `api/` | Auth (JWT), credits, projects, jobs, Stripe, orgs |
| Engine | `movie_pipeline/` | Multi-agent movie planning + image/video generation |
| Ops | `streamlit_app.py` | Simple monitor against the API |
| Own model lab | `video_lab/` + `gradio_video_lab.py` | Train/test local Causal VAE + Spatiotemporal DiT |
| Docs | `docs/` | Lab guide + manual feature checklist |
| Data (lab) | `data/video_lab/` | Raw clips, smoke clips, `manifest.jsonl` |
| Outputs (lab) | `outputs/video_lab/` | VAE/DiT checkpoints + sample MP4s |
| Storage (product) | `storage/` | Local DB / assets for Movie Flow |

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (for the web app)
- Optional: NVIDIA GPU + CUDA PyTorch (much faster for the video lab)

---

## Setup (once)

```powershell
cd C:\hackathon\Gemini_CLI\movie-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` as needed. **`HF_TOKEN` is optional** for Movie Flow remote LLM/video; without it, the pipeline uses local planning + cinematic fallbacks.

For the video lab also install:

```powershell
python -m pip install -r requirements-video-lab.txt
```

GPU (recommended for training/generate):

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 1) Movie Flow — product studio

### Start API

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start web

```powershell
cd web
copy .env.local.example .env.local
npm install
npm run dev
```

Open **http://localhost:3000** → register → use **Create** for Image / Video / Movie.

### Modes

- **Image** — still generation (credits)
- **Video** — short clip generation
- **Movie** — multi-agent pipeline (script → shots → clips → assemble)

### Ops monitor (optional)

```powershell
$env:MOVIE_FLOW_API_URL = "http://127.0.0.1:8000"
python -m streamlit run streamlit_app.py
```

### Offline CLI pipeline

```powershell
python movie_pipeline/main.py
```

### Product features (summary)

- Auth + **credits** (signup grant; image / video / movie costs differ)
- Projects, assets, ZIP export
- Billing (Stripe when keys set; demo upgrade otherwise)
- Teams / orgs / invites
- Pricing: Starter / Pro / Enterprise

---

## 2) Own Video Model Lab — research

This is where you build **our** model (not Veo, not CogVideoX in the UI).

```powershell
.\.venv\Scripts\Activate.ps1
python gradio_video_lab.py
```

Open **http://127.0.0.1:7860**

### Tabs (workflow)

| Tab | What you do |
|-----|-------------|
| **Checklist** | Live ✅/⬜ — YOU vs LAB (Phase A data + Phase B architecture) |
| **Data** | Smoke / curate with **optical-flow** gates → **Recaption** for `dense_caption` |
| **Labels** | Edit caption / camera / lighting / motion / aesthetic / tags |
| **Train** | Curriculum Stage 1/2/3 + aspect bucket → Train VAE → Train DiT |
| **Generate** | Prompt → MP4 from local checkpoints |

### Manual data you should add

1. Put real MP4/WebM into `data/video_lab/raw/` **or** download from [Pexels](https://www.pexels.com) (Videos provided by Pexels):
   - Add `PEXELS_API_KEY=` to `.env` ([free key](https://www.pexels.com/api/))
   - `python scripts/download_pexels.py --query "ocean waves" --count 200`
   - Or Gradio **Data → Download from Pexels**
2. **Data → Curate raw → manifest**
3. **Labels** / **Recaption**, then Train

Example row:

```json
{
  "path": "C:/hackathon/Gemini_CLI/movie-AI/data/video_lab/raw/my_clip.mp4",
  "caption": "neon city street at night, wet asphalt, slow dolly forward",
  "camera": "dolly forward",
  "lighting": "neon night",
  "motion": "camera push-in, light rain",
  "aesthetic": 8,
  "tags": ["city", "rain", "cinematic"],
  "negative": "watermark, shaky blur, logos",
  "fps": 24,
  "frames": 48
}
```

Full checklist: [docs/OWN_MODEL_CHECKLIST.md](docs/OWN_MODEL_CHECKLIST.md)  
Lab details: [docs/VIDEO_MODEL_LAB.md](docs/VIDEO_MODEL_LAB.md)  
Niche 256–512p / 24GB: [docs/NICHE_TRAINING.md](docs/NICHE_TRAINING.md) (`scripts/train_niche.py`)

### Honest expectations

- Smoke/toy training → abstract color motion (not photoreal Veo quality)
- Competing with commercial models needs **lots of real clips + longer GPU training + larger nets**
- RTX 3050-class GPUs are fine for research; not for Veo-scale training

---

## Repo map (quick)

```text
movie-AI/
├── api/                 # FastAPI SaaS
├── web/                 # Next.js product UI
├── movie_pipeline/      # Agents + Motif/image clients
├── video_lab/           # Own VAE/DiT models, train, infer, Gradio app
├── data/video_lab/      # Lab datasets + manifest.jsonl
├── outputs/video_lab/   # Checkpoints + samples
├── docs/                # OWN_MODEL_CHECKLIST, VIDEO_MODEL_LAB
├── storage/             # Product DB / files
├── gradio_video_lab.py  # Launch lab UI
├── streamlit_app.py     # Ops monitor
├── requirements.txt
├── requirements-video-lab.txt
└── .env.example
```

---

## Environment variables

See [`.env.example`](.env.example).

| Area | Vars |
|------|------|
| Optional remote models | `HF_TOKEN`, `HF_TEXT_MODEL`, `HF_VIDEO_*` |
| API | `JWT_SECRET`, `DATABASE_URL`, `STORAGE_ROOT`, `CORS_ORIGINS` |
| Billing | `STRIPE_*` |
| Web | `web/.env.local` → API base URL |

---

## Optional: SDXL keyframes (movie pipeline)

```powershell
python -m pip install -r requirements-model.txt
$env:GENERATE_KEYFRAMES = "true"
```

Uses SDXL for keyframes when GPU/deps allow; otherwise local stills.

---

## Typical day

**Use the product**

1. Start API (`:8000`) + web (`:3000`)
2. Register → Create Image/Video/Movie

**Improve our video model**

1. Start Gradio (`:7860`)
2. Checklist → add clips → Labels → Train VAE → Train DiT → Generate
3. Restart Gradio after code changes

---

## License

See [LICENSE](LICENSE).
