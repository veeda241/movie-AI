from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from video_lab import MANIFEST_PATH, RAW_DIR, ensure_dirs
from video_lab.data.buckets import list_bucket_labels, parse_bucket_choice
from video_lab.data.curate import build_manifest_from_raw, ingest_folder
from video_lab.data.hf_wan_datasets import build_hf_wan_manifest, download_all_wan_action_datasets
from video_lab.data.manifest_edit import (
    CAMERA_CHOICES,
    LIGHTING_CHOICES,
    MOTION_CHOICES,
    autofill_empty_labels,
    get_row,
    row_labels,
    suggest_from_caption,
    update_row,
)
from video_lab.data.recaption import recaption_manifest
from video_lab.data.smoke import ensure_smoke_manifest
from video_lab.infer.finetune_generate import generate_finetune_video
from video_lab.infer.research_generate import generate_research_video
from video_lab.roadmap import checklist_markdown, scan_checklist, write_manifest_template
from video_lab.train.curriculum import apply_stage_to_train_kwargs, list_stage_labels, resolve_stage
from video_lab.train.train_dit import train_dit
from video_lab.train.train_vae import train_vae
from video_lab.utils.device import get_device


def _append(log: str, line: str) -> str:
    return (log + "\n" + line).strip() if log else line


def _duration_label(frames: float, fps: float) -> str:
    frames = max(1, int(frames))
    fps = max(1, int(fps))
    return f"Duration ≈ **{frames / fps:.2f} s** ({frames} frames @ {fps} fps)"


def _frames_from_duration(duration_sec: float, fps: float) -> int:
    fps = max(4, int(fps))
    frames = int(round(float(duration_sec) * fps))
    frames = max(8, frames - (frames % 4))
    return min(frames, int(10 * fps))


def ui_prepare_smoke(log: str):
    ensure_dirs()
    path = ensure_smoke_manifest(MANIFEST_PATH)
    return _append(log, f"Smoke dataset ready: {path}"), str(path)


def ui_curate(
    raw_path: str,
    scene_cut: bool,
    use_flow: bool,
    min_flow: float,
    max_flow: float,
    log: str,
):
    try:
        ensure_dirs()
        src = Path(raw_path) if raw_path.strip() else RAW_DIR
        if src.exists() and src.is_dir():
            copied = ingest_folder(src, RAW_DIR)
            log = _append(log, f"Ingested {len(copied)} clips into {RAW_DIR}")
        manifest = build_manifest_from_raw(
            RAW_DIR,
            manifest_path=MANIFEST_PATH,
            run_scene_cut=scene_cut,
            min_flow=float(min_flow),
            max_flow=float(max_flow),
            use_optical_flow=bool(use_flow),
        )
        n = sum(1 for _ in open(manifest, encoding="utf-8") if _.strip())
        if n == 0:
            manifest = ensure_smoke_manifest(MANIFEST_PATH)
            log = _append(log, "No raw clips passed filters — using smoke dataset.")
            n = sum(1 for _ in open(manifest, encoding="utf-8") if _.strip())
        return _append(log, f"Manifest rows: {n} -> {manifest}"), str(manifest)
    except Exception:
        return _append(log, traceback.format_exc()), ""


def ui_recaption(log: str):
    try:
        n, path = recaption_manifest()
        return _append(log, f"Recaptioned {n} row(s) → {path}"), str(path)
    except Exception:
        return _append(log, traceback.format_exc()), ""


def ui_pexels_download(query: str, count: float, min_dur: float, max_dur: float, log: str):
    lines: list[str] = []

    def log_fn(msg: str):
        lines.append(msg)

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except Exception:
        pass
    try:
        from video_lab.data.pexels_download import download_pexels_videos

        q = (query or "").strip() or "nature forest"
        summary = download_pexels_videos(
            q,
            target_count=int(count),
            min_duration=int(min_dur),
            max_duration=int(max_dur),
            log_fn=log_fn,
        )
        for line in lines:
            log = _append(log, line)
        return _append(
            log,
            f"Pexels done: {summary.get('downloaded')} new clips. "
            f"[Videos provided by Pexels](https://www.pexels.com)",
        )
    except Exception:
        for line in lines:
            log = _append(log, line)
        return _append(log, traceback.format_exc())


def ui_hf_wan_download(log: str):
    lines: list[str] = []

    def log_fn(msg: str):
        lines.append(msg)

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except Exception:
        pass
    try:
        summary = download_all_wan_action_datasets(log_fn=log_fn)
        for line in lines:
            log = _append(log, line)
        return _append(
            log,
            f"HF Wan actions done: downloaded={summary.get('downloaded')} "
            f"skipped={summary.get('skipped')} failed={summary.get('failed')}",
        )
    except Exception:
        for line in lines:
            log = _append(log, line)
        return _append(log, traceback.format_exc())


def ui_train_vae(steps, stage_choice, bucket_choice, dit_size_ignored, min_aes_override, log):
    lines: list[str] = []

    def log_fn(msg: str):
        lines.append(msg)

    try:
        kw = apply_stage_to_train_kwargs(stage_choice)
        if bucket_choice:
            b = parse_bucket_choice(bucket_choice)
            kw["bucket"] = b.name
            kw["height"] = b.height
            kw["width"] = b.width
        if min_aes_override is not None and float(min_aes_override) > 0:
            kw["min_aesthetic"] = float(min_aes_override)
        ckpt = train_vae(
            steps=int(steps),
            frames=int(kw["frames"]),
            height=int(kw["height"]),
            width=int(kw["width"]),
            min_aesthetic=float(kw["min_aesthetic"]),
            bucket=kw.get("bucket"),
            train_stage=kw.get("train_stage"),
            log_fn=log_fn,
        )
        for line in lines:
            log = _append(log, line)
        return _append(log, f"VAE done: {ckpt}"), str(ckpt)
    except Exception:
        return _append(log, traceback.format_exc()), ""


def ui_train_dit(steps, stage_choice, bucket_choice, dit_size, min_aes_override, log):
    lines: list[str] = []

    def log_fn(msg: str):
        lines.append(msg)

    try:
        kw = apply_stage_to_train_kwargs(stage_choice)
        if bucket_choice:
            b = parse_bucket_choice(bucket_choice)
            kw["bucket"] = b.name
            kw["height"] = b.height
            kw["width"] = b.width
        if min_aes_override is not None and float(min_aes_override) > 0:
            kw["min_aesthetic"] = float(min_aes_override)
        ckpt = train_dit(
            steps=int(steps),
            frames=int(kw["frames"]),
            height=int(kw["height"]),
            width=int(kw["width"]),
            min_aesthetic=float(kw["min_aesthetic"]),
            dit_size=dit_size,
            bucket=kw.get("bucket"),
            train_stage=kw.get("train_stage"),
            log_fn=log_fn,
        )
        for line in lines:
            log = _append(log, line)
        return _append(log, f"DiT done: {ckpt}"), str(ckpt)
    except Exception:
        return _append(log, traceback.format_exc()), ""


def ui_generate_video(prompt, steps, seed, duration_sec, fps, size, log):
    lines: list[str] = []

    def log_fn(msg: str):
        lines.append(msg)

    try:
        ensure_dirs()
        prompt = (prompt or "").strip() or "blue particles drifting over a dark gradient"
        duration_sec = float(duration_sec)
        fps = int(fps)
        frames = _frames_from_duration(duration_sec, fps)
        path = generate_research_video(
            prompt,
            steps=int(steps),
            seed=int(seed),
            frames=frames,
            fps=fps,
            duration_sec=duration_sec,
            height=int(size),
            width=int(size),
            log_fn=log_fn,
        )
        path = str(Path(path).resolve())
        for line in lines:
            log = _append(log, line)
        if not Path(path).exists():
            return _append(log, f"ERROR: file missing after generate: {path}"), None
        return _append(log, f"OK — ~{duration_sec:.1f}s video ready: {path}"), path
    except Exception:
        return _append(log, traceback.format_exc()), None


def ui_refresh_checklist():
    items = scan_checklist()
    done = sum(1 for i in items if i.done)
    return checklist_markdown(items), f"{done}/{len(items)} checks complete"


def ui_write_template(log: str):
    path = write_manifest_template()
    return _append(log, f"Wrote editable template: {path}"), str(path)


def ui_label_choices():
    return gr.update(choices=row_labels(), value=row_labels()[0])


def ui_load_label_row(choice: str):
    try:
        idx = int(str(choice).split(":", 1)[0])
    except (ValueError, IndexError):
        idx = 0
    row = get_row(idx)
    tags = row.get("tags") or []
    tags_s = ", ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)
    return (
        idx,
        str(row.get("path", "")),
        str(row.get("caption", "")),
        str(row.get("dense_caption", "")),
        str(row.get("camera", "")),
        str(row.get("lighting", "")),
        str(row.get("motion", "")),
        float(row.get("aesthetic", 5) or 5),
        tags_s,
        str(row.get("negative", "")),
    )


def ui_suggest_labels(caption: str):
    sug = suggest_from_caption(caption)
    return (
        sug.get("camera", ""),
        sug.get("lighting", ""),
        sug.get("motion", ""),
        float(sug.get("aesthetic", 6)),
        ", ".join(sug.get("tags") or []),
        sug.get("negative", ""),
    )


def ui_save_label_row(index, caption, camera, lighting, motion, aesthetic, tags, negative, log: str):
    try:
        update_row(
            int(index),
            caption=caption,
            camera=camera,
            lighting=lighting,
            motion=motion,
            aesthetic=float(aesthetic),
            tags=tags,
            negative=negative,
        )
        choices = row_labels()
        return (
            _append(log, f"Saved row {int(index)} (dense_caption refreshed)"),
            gr.update(choices=choices, value=choices[int(index) % len(choices)]),
        )
    except Exception:
        return _append(log, traceback.format_exc()), gr.update()


def ui_autofill_labels(log: str):
    try:
        n, path = autofill_empty_labels()
        choices = row_labels()
        return (
            _append(log, f"Autofilled/densified → {path} (touched ~{n})"),
            gr.update(choices=choices, value=choices[0] if choices else None),
        )
    except Exception:
        return _append(log, traceback.format_exc()), gr.update()


def ui_stage_info(stage_choice: str):
    s = resolve_stage(stage_choice)
    b = parse_bucket_choice(s.bucket)
    return f"{s.description} → {s.frames}f @ {b.width}x{b.height}, min_aesthetic={s.min_aesthetic}"


def build_app() -> gr.Blocks:
    ensure_dirs()
    write_manifest_template()
    device = get_device()
    bucket_choices = list_bucket_labels()
    stage_choices = list_stage_labels()
    with gr.Blocks(title="Own Video Model Lab") as demo:
        gr.Markdown(
            f"""
# Own Video Model Lab
Train and test **our local** Causal VAE + DiT (research bench).

**Main path:** Data → Labels → Train (VAE then DiT) → Generate  
**Not Movie Flow:** this UI does not power the product studio on port 3000.  
**Not Wan/Veo:** quality matches what you train (niche clips), not commercial models.

Device: `{device}`
"""
        )

        with gr.Tabs():
            with gr.Tab("Checklist"):
                with gr.Row():
                    refresh = gr.Button("Refresh checklist", variant="primary")
                    mk_template = gr.Button("Write manifest template file")
                status = gr.Textbox(label="Progress", interactive=False)
                checklist_md = gr.Markdown(checklist_markdown())
                template_log = gr.Textbox(label="Template log", lines=3)
                template_path = gr.Textbox(label="Template path")
                refresh.click(ui_refresh_checklist, outputs=[checklist_md, status])
                mk_template.click(ui_write_template, inputs=[template_log], outputs=[template_log, template_path])
                demo.load(ui_refresh_checklist, outputs=[checklist_md, status])

            with gr.Tab("Generate"):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="ocean waves crashing on a sandy beach, sunny day",
                )
                with gr.Row():
                    duration = gr.Slider(0.5, 10, value=0.7, step=0.1, label="Duration (seconds)")
                    fps = gr.Slider(8, 24, value=12, step=1, label="FPS")
                    size = gr.Slider(64, 256, value=256, step=32, label="Resolution")
                duration_md = gr.Markdown(_duration_label(_frames_from_duration(0.7, 12), 12))

                def _upd_dur(d, p):
                    f = _frames_from_duration(d, p)
                    return _duration_label(f, p)

                duration.change(_upd_dur, inputs=[duration, fps], outputs=duration_md)
                fps.change(_upd_dur, inputs=[duration, fps], outputs=duration_md)
                with gr.Row():
                    steps = gr.Slider(4, 50, value=24, step=1, label="Diffusion steps (per chunk)")
                    seed = gr.Number(value=0, label="Seed", precision=0)
                gr.Markdown(
                    "Generate from **your last Train checkpoints** (`outputs/video_lab/`). "
                    "Match **resolution/frames** to training (e.g. 256², ~0.7s @ 12fps for niche_laptop). "
                    "Use prompts in your training niche (ocean, blinking, pouring liquid, …). "
                    "Longer durations only stitch short chunks — they do not unlock Wan-level quality."
                )
                gen_btn = gr.Button("Generate Video", variant="primary", size="lg")
                video_out = gr.Video(label="Generated video", height=360)
                log = gr.Textbox(label="Log", lines=10)
                gen_btn.click(
                    ui_generate_video,
                    inputs=[prompt, steps, seed, duration, fps, size, log],
                    outputs=[log, video_out],
                )

            with gr.Tab("Data"):
                gr.Markdown(
                    f"Drop clips into `{RAW_DIR}`, or download from "
                    "[Pexels](https://www.pexels.com) (Videos provided by Pexels). "
                    "Optical-flow drops slideshows / chaotic shake. Then **Recaption**."
                )
                with gr.Accordion("Download HF Wan action packs", open=False):
                    gr.Markdown(
                        "Downloads `linoyts/wan_*` action clips into raw/. "
                        "Optional `HF_TOKEN` in `.env` helps rate limits."
                    )
                    hf_wan_btn = gr.Button("Download all HF Wan action datasets", variant="secondary")
                with gr.Accordion("Download from Pexels", open=True):
                    gr.Markdown(
                        "Set `PEXELS_API_KEY` in `.env` "
                        "([get a free key](https://www.pexels.com/api/)). "
                        "Credit creators are stored automatically."
                    )
                    pexels_q = gr.Textbox(label="Niche query", value="ocean waves")
                    with gr.Row():
                        pexels_n = gr.Slider(10, 500, value=50, step=10, label="Clip count")
                        pexels_min = gr.Slider(2, 10, value=3, step=1, label="Min seconds")
                        pexels_max = gr.Slider(5, 30, value=15, step=1, label="Max seconds")
                    pexels_btn = gr.Button("Download Pexels videos", variant="secondary")
                raw = gr.Textbox(label="Raw folder (optional)", value=str(RAW_DIR))
                scene = gr.Checkbox(label="Run scene cut", value=False)
                use_flow = gr.Checkbox(label="Optical-flow filter (Farneback)", value=True)
                with gr.Row():
                    min_flow = gr.Slider(0.0, 2.0, value=0.15, step=0.05, label="Min flow mean")
                    max_flow = gr.Slider(1.0, 30.0, value=12.0, step=0.5, label="Max flow mean")
                data_log = gr.Textbox(label="Data log", lines=8)
                with gr.Row():
                    b_smoke = gr.Button("Create smoke dataset")
                    b_curate = gr.Button("Curate raw → manifest", variant="primary")
                    b_recap = gr.Button("Recaption manifest")
                manifest_out = gr.Textbox(label="Manifest path")
                hf_wan_btn.click(ui_hf_wan_download, inputs=[data_log], outputs=[data_log])
                pexels_btn.click(
                    ui_pexels_download,
                    inputs=[pexels_q, pexels_n, pexels_min, pexels_max, data_log],
                    outputs=[data_log],
                )
                b_smoke.click(ui_prepare_smoke, inputs=[data_log], outputs=[data_log, manifest_out])
                b_curate.click(
                    ui_curate,
                    inputs=[raw, scene, use_flow, min_flow, max_flow, data_log],
                    outputs=[data_log, manifest_out],
                )
                b_recap.click(ui_recaption, inputs=[data_log], outputs=[data_log, manifest_out])

            with gr.Tab("Labels"):
                label_log = gr.Textbox(label="Label log", lines=4)
                with gr.Row():
                    row_dd = gr.Dropdown(choices=row_labels(), label="Clip row", value=row_labels()[0])
                    reload_rows = gr.Button("Reload rows")
                    autofill_btn = gr.Button("Autofill + densify")
                row_idx = gr.Number(value=0, label="Row index", precision=0, visible=False)
                path_box = gr.Textbox(label="Path", interactive=False)
                cap_box = gr.Textbox(label="caption", lines=2)
                dense_box = gr.Textbox(label="dense_caption (auto)", lines=3, interactive=False)
                with gr.Row():
                    cam_box = gr.Dropdown(choices=CAMERA_CHOICES, label="camera", allow_custom_value=True)
                    light_box = gr.Dropdown(choices=LIGHTING_CHOICES, label="lighting", allow_custom_value=True)
                    motion_box = gr.Dropdown(choices=MOTION_CHOICES, label="motion", allow_custom_value=True)
                with gr.Row():
                    aes_box = gr.Slider(0, 10, value=6, step=1, label="aesthetic")
                    tags_box = gr.Textbox(label="tags (comma-separated)")
                neg_box = gr.Textbox(label="negative")
                with gr.Row():
                    suggest_btn = gr.Button("Suggest from caption")
                    save_btn = gr.Button("Save row", variant="primary")

                reload_rows.click(ui_label_choices, outputs=[row_dd]).then(
                    ui_load_label_row,
                    inputs=[row_dd],
                    outputs=[row_idx, path_box, cap_box, dense_box, cam_box, light_box, motion_box, aes_box, tags_box, neg_box],
                )
                row_dd.change(
                    ui_load_label_row,
                    inputs=[row_dd],
                    outputs=[row_idx, path_box, cap_box, dense_box, cam_box, light_box, motion_box, aes_box, tags_box, neg_box],
                )
                suggest_btn.click(
                    ui_suggest_labels,
                    inputs=[cap_box],
                    outputs=[cam_box, light_box, motion_box, aes_box, tags_box, neg_box],
                )
                save_btn.click(
                    ui_save_label_row,
                    inputs=[row_idx, cap_box, cam_box, light_box, motion_box, aes_box, tags_box, neg_box, label_log],
                    outputs=[label_log, row_dd],
                ).then(
                    ui_load_label_row,
                    inputs=[row_dd],
                    outputs=[row_idx, path_box, cap_box, dense_box, cam_box, light_box, motion_box, aes_box, tags_box, neg_box],
                )
                autofill_btn.click(ui_autofill_labels, inputs=[label_log], outputs=[label_log, row_dd])
                demo.load(
                    ui_load_label_row,
                    inputs=[row_dd],
                    outputs=[row_idx, path_box, cap_box, dense_box, cam_box, light_box, motion_box, aes_box, tags_box, neg_box],
                )

            with gr.Tab("Train"):
                gr.Markdown(
                    "Progressive stages. **VAE first**, then **DiT**. "
                    "Use **Niche laptop** (256²/8f) on 6GB, or **Niche 24GB** on a rented card. "
                    "See `docs/NICHE_TRAINING.md` + `scripts/train_niche.py`."
                )
                train_log = gr.Textbox(label="Train log", lines=8)
                stage = gr.Radio(choices=stage_choices, value=stage_choices[1], label="Curriculum stage")
                stage_info = gr.Markdown(ui_stage_info(stage_choices[1]))
                stage.change(ui_stage_info, inputs=[stage], outputs=stage_info)
                with gr.Row():
                    bucket = gr.Dropdown(choices=bucket_choices, value=bucket_choices[0], label="Aspect bucket")
                    dit_size = gr.Radio(choices=["small", "medium"], value="small", label="DiT size")
                    min_aes = gr.Slider(0, 10, value=0, step=1, label="Override min aesthetic (0=use stage)")
                with gr.Row():
                    steps_vae = gr.Slider(50, 2000, value=400, step=50, label="VAE steps")
                    steps_dit = gr.Slider(50, 2000, value=600, step=50, label="DiT steps")
                with gr.Row():
                    b_vae = gr.Button("Train VAE", variant="primary")
                    b_dit = gr.Button("Train DiT", variant="primary")
                ckpt_box = gr.Textbox(label="Last checkpoint")
                b_vae.click(
                    ui_train_vae,
                    inputs=[steps_vae, stage, bucket, dit_size, min_aes, train_log],
                    outputs=[train_log, ckpt_box],
                )
                b_dit.click(
                    ui_train_dit,
                    inputs=[steps_dit, stage, bucket, dit_size, min_aes, train_log],
                    outputs=[train_log, ckpt_box],
                )

            with gr.Tab("Experimental (CogVideo LoRA)"):
                gr.Markdown(
                    "### Optional — not the own-model path\n\n"
                    "This tab fine-tunes **Hugging Face CogVideoX** (2B default / 5B if VRAM allows) "
                    "with LoRA on your clips. Separate from Train VAE/DiT above. "
                    "Needs disk + CUDA (`pip install -r requirements-cogvideo.txt`).\n\n"
                    "**Workflow:** Build manifest → Train LoRA → Generate with LoRA.\n"
                    "Generate uses the base model recorded in `lora_meta.pt` (must match training).\n"
                )

                with gr.Row():
                    ft_manifest_btn = gr.Button("Build manifest from HF Wan clips", variant="secondary")
                ft_manifest_out = gr.Textbox(label="Manifest path", interactive=False)
                ft_count = gr.Textbox(label="Clip count", interactive=False)

                gr.Markdown("---\n### Training settings")
                with gr.Row():
                    ft_steps = gr.Slider(50, 2000, value=200, step=50, label="Steps")
                    ft_rank = gr.Slider(4, 64, value=16, step=4, label="LoRA rank")
                    ft_lr = gr.Textbox(value="1e-4", label="Learning rate")

                ft_train_btn = gr.Button("Train LoRA", variant="primary", size="lg")
                ft_log = gr.Textbox(label="Training log", lines=12)
                ft_status = gr.Textbox(label="Adapter path", interactive=False)

                gr.Markdown("---\n### Generate with LoRA")
                ft_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="a person blinking, realistic video",
                )
                with gr.Row():
                    ft_steps_gen = gr.Slider(4, 50, value=20, step=1, label="Diffusion steps")
                    ft_seed = gr.Number(value=42, label="Seed", precision=0)
                    ft_duration = gr.Slider(0.5, 10, value=2.0, step=0.1, label="Duration (s)")
                ft_gen_btn = gr.Button("Generate with LoRA", variant="secondary")
                ft_video_out = gr.Video(label="Generated video", height=360)

                # Wire up manifest builder
                def _ui_build_hf_wan_manifest_fn(log: str):
                    try:
                        path = build_hf_wan_manifest(raw_dir=RAW_DIR)
                        count = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
                        return _append(log, f"Built manifest: {path}"), str(path), f"{count} clips"
                    except Exception:
                        return _append(log, traceback.format_exc()), "", ""

                ft_manifest_btn.click(
                    _ui_build_hf_wan_manifest_fn,
                    inputs=[ft_log],
                    outputs=[ft_log, ft_manifest_out, ft_count],
                )

                # Wire up training
                def _ui_train_lora_fn(steps, rank, lr_str, manifest_path, log):
                    lines: list[str] = []
                    def log_fn(msg: str):
                        lines.append(msg)

                    try:
                        from video_lab.train.train_lora_t2v import train_lora_t2v

                        lr = float(lr_str)
                        path = manifest_path.strip() or None
                        if path and not Path(path).exists():
                            path = str(build_hf_wan_manifest(raw_dir=RAW_DIR))

                        result = train_lora_t2v(
                            manifest_path=Path(path) if path else None,
                            steps=int(steps),
                            rank=int(rank),
                            lr=lr,
                            log_fn=log_fn,
                        )
                        for line in lines:
                            log = _append(log, line)
                        return log, str(result)
                    except Exception:
                        for line in lines:
                            log = _append(log, line)
                        return _append(log, traceback.format_exc()), ""

                ft_train_btn.click(
                    _ui_train_lora_fn,
                    inputs=[ft_steps, ft_rank, ft_lr, ft_manifest_out, ft_log],
                    outputs=[ft_log, ft_status],
                )

                # Wire up generation with LoRA
                def _ui_generate_lora_fn(prompt, steps, seed, duration, log):
                    lines: list[str] = []
                    def log_fn(msg: str):
                        lines.append(msg)

                    try:
                        path = generate_finetune_video(
                            prompt=prompt,
                            steps=int(steps),
                            seed=int(seed),
                            frames=max(8, int(float(duration) * 12)),
                            fps=12,
                            height=256,
                            width=256,
                            log_fn=log_fn,
                        )
                        for line in lines:
                            log = _append(log, line)
                        if not Path(path).exists():
                            return _append(log, "ERROR: video file not generated"), None
                        return log, path
                    except Exception:
                        for line in lines:
                            log = _append(log, line)
                        return _append(log, traceback.format_exc()), None

                ft_gen_btn.click(
                    _ui_generate_lora_fn,
                    inputs=[ft_prompt, ft_steps_gen, ft_seed, ft_duration, ft_log],
                    outputs=[ft_log, ft_video_out],
                )

    return demo


def main():
    demo = build_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
