from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from movie_pipeline.models.generative_config import FineTuneConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a Diffusers SDXL LoRA training command.")
    parser.add_argument("--dataset-dir", default="data/images")
    parser.add_argument("--output-dir", default="movie_pipeline/output/lora")
    parser.add_argument("--base-model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    config = FineTuneConfig(
        base_model_id=args.base_model_id,
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        max_train_steps=args.steps,
        rank=args.rank,
    )
    command = [
        "accelerate",
        "launch",
        "train_text_to_image_lora_sdxl.py",
        *config.to_accelerate_args(),
    ]
    print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
