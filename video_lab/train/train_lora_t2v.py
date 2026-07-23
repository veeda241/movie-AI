from __future__ import annotations

from pathlib import Path

from video_lab.config import LabConfig


def train_lora_t2v(
    *,
    manifest_path: Path | None = None,
    base_model: str | None = None,
    steps: int | None = None,
    rank: int | None = None,
    log_fn=None,
) -> Path:
    """
    LoRA fine-tune entrypoint for CogVideoX.

    Full Diffusers CogVideoX LoRA training is heavy; this script:
    - validates the environment
    - writes a ready-to-run accelerate command + config snapshot
    - optionally runs a tiny PEFT adapter init smoke save for UI wiring
    """
    cfg = LabConfig()
    manifest_path = Path(manifest_path or cfg.manifest_path)
    base_model = base_model or cfg.base_t2v_model
    steps = steps or cfg.lora_steps
    rank = rank or cfg.lora_rank
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)

    cmd_path = cfg.lora_dir / "train_lora_command.txt"
    command = (
        f"accelerate launch -m video_lab.train.train_lora_t2v "
        f"--manifest {manifest_path} --model {base_model} --steps {steps} --rank {rank}"
    )
    cmd_path.write_text(command, encoding="utf-8")
    if log_fn:
        log_fn(f"Wrote LoRA command helper: {cmd_path}")

    try:
        import torch
        from peft import LoraConfig

        # Lightweight smoke: save LoRA config + placeholder adapter meta
        meta = {
            "base_model": base_model,
            "rank": rank,
            "steps": steps,
            "manifest": str(manifest_path),
            "note": "Run on GPU with Diffusers CogVideoX training for full fine-tune. "
            "This smoke artifact wires the Gradio Fine-tune tab.",
        }
        out = cfg.lora_dir / "lora_meta.pt"
        torch.save({"meta": meta, "lora_config": LoraConfig(r=rank, lora_alpha=rank).to_dict()}, out)
        if log_fn:
            log_fn(f"Saved LoRA meta smoke checkpoint: {out}")
            log_fn(
                "For full CogVideoX LoRA training, use a CUDA GPU and Diffusers example "
                "train_cogvideox_lora.py pointed at your manifest captions."
            )
        return out
    except Exception as exc:
        fallback = cfg.lora_dir / "lora_meta.txt"
        fallback.write_text(f"LoRA setup pending dependencies: {exc}\n{command}\n", encoding="utf-8")
        if log_fn:
            log_fn(str(exc))
        return fallback
