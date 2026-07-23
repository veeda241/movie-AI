# Niche training (1k–10k clips, 256–512p)

Your **RTX 3050 6GB** can only approximate this tier. A real niche run wants **~24 GB** VRAM and **1k–10k labeled clips**.

## Profiles

| Profile | Size | Frames | GPU | Command |
|---------|------|--------|-----|---------|
| `niche_laptop` | 256² | 8 | 6GB try | `python scripts/train_niche.py --profile niche_laptop` |
| `niche_24gb` | 256² | 24 | 20–24GB | `python scripts/train_niche.py --profile niche_24gb` |
| `niche_24gb_512` | 512² | 16 | 22–24GB+ | `python scripts/train_niche.py --profile niche_24gb_512` |

Also available in Gradio **Train** as curriculum stages `Niche laptop` / `Niche 24GB`.

## On this laptop (now)

```powershell
.\.venv\Scripts\Activate.ps1
# Expand smoke a bit, then try 256² / 8f with AMP
.\.venv\Scripts\python.exe scripts\train_niche.py --profile niche_laptop --refresh-smoke --vae-steps 200 --dit-steps 300
```

If you OOM: stay on Stage 2 (128² / 16f) until you rent a 24GB GPU.

Generate **at the same** frames/resolution you trained (e.g. 8f @ 256), not 48f @ 256.

## Cloud (24GB) — recommended path

1. Rent **1× RTX 4090 / A5000 / L40S 24GB** (RunPod / Vast.ai / Lambda).
2. Clone repo, create venv, install `requirements-video-lab.txt` + CUDA torch.
3. Copy **1k–10k** MP4s into `data/video_lab/raw/`.
4. On the machine:

```bash
python gradio_video_lab.py   # or headless:
# Data: curate + recaption (or use Gradio once)
python scripts/train_niche.py --profile niche_24gb
```

5. Download `outputs/video_lab/vae/` and `dit/` back to the laptop for Generate.

**Rough cloud cost:** one long weekend on a 24GB spot instance is often tens of USD, not thousands — enough to beat weeks on 6GB for Stage-3 niche quality.

## Data you still must add (YOU)

Without ~1k real clips + dense captions, niche resolution only upscales **smoke patterns**.  
Pipeline: drop clips → Data curate (optical flow) → Labels/Recaption → train niche profile.

### Hugging Face Wan action packs (linoyts)

Small Wan-2.1-generated action datasets (~10 clips each) with prompts — good for **motion niche** training, not full foundation pretrain.

```powershell
# Optional but recommended for rate limits:
# set HF_TOKEN in .env

.\.venv\Scripts\python.exe scripts\download_hf_wan_actions.py --limit 3   # smoke
.\.venv\Scripts\python.exe scripts\download_hf_wan_actions.py             # all ~28 packs
```

Then **Curate → Recaption → Train**. Generate with matching action prompts (blink, clap, pour liquid, …).

### Pexels download (stock footage)

[Videos provided by Pexels](https://www.pexels.com). Set `PEXELS_API_KEY=` in `.env`.

```powershell
.\.venv\Scripts\python.exe scripts\download_pexels.py --query "ocean waves" --count 200
```

Clips land in `data/video_lab/raw/` with credits in `pexels_index.jsonl`.


## Honest bar

| You have | Result |
|----------|--------|
| 8 smoke clips @ 256 | Colored abstract motion, not a product model |
| 1k curated clips @ 256 on 24GB | Plausible **niche** look for that domain |
| 10k+ @ 512 + weeks | Stronger niche; still not Veo |
