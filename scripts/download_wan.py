"""Pre-download the Wan2.1-T2V-1.3B model with visible progress.

Run BEFORE submitting a Wan video job so the API doesn't silently block for
hours inside ``from_pretrained``. The model is ~12 GB total (sharded text
encoder + transformer + VAE + tokenizer).

    python scripts/download_wan.py

Optional flags::

    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers   HF repo id
    --cache  C:\\path\\to\\hf\\hub              override HF cache dir
    --clear-no-exist                          wipe stale .no_exist markers first
    --verify-only                             just print which files are missing

Why this exists: huggingface_hub caches 404 results in ``.no_exist`` for 24h.
If a partial download failed yesterday, the worker today will silently skip
the missing files and block on the rest. Running with ``--clear-no-exist``
removes those markers so the downloader actually retries.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def _try_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def _cache_dir_for_model(model_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    safe = "models--" + model_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / safe


def _clear_no_exist(model_id: str) -> int:
    cache = _cache_dir_for_model(model_id)
    no_exist = cache / ".no_exist"
    if not no_exist.exists():
        print(f"  no .no_exist dir at {no_exist}")
        return 0
    n = sum(1 for _ in no_exist.rglob("*") if _.is_file())
    shutil.rmtree(no_exist, ignore_errors=True)
    print(f"  removed {n} stale .no_exist entries from {no_exist}")
    return n


def _missing_files(model_id: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    remote = set(api.list_repo_files(model_id))
    cache = _cache_dir_for_model(model_id) / "snapshots"
    if not cache.exists():
        return sorted(remote)
    present: set[str] = set()
    for snap in cache.iterdir():
        for f in snap.rglob("*"):
            if f.is_file() and f.exists():
                # The snapshot holds symlinks; if the symlink target exists we
                # treat it as present. The file may live in .no_exist with
                # the same path; that path is still missing.
                rel = str(f.relative_to(snap)).replace("\\", "/")
                present.add(rel)
    return sorted(remote - present)


def _download(model_id: str) -> int:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    print(f"Downloading {model_id} (this can take 5-30 min on a 50 Mbps link)…")
    last_t = time.time()
    last_bytes = 0

    class _TqdmWrap(tqdm):
        def update(self, n: float = 1.0) -> bool:  # type: ignore[override]
            nonlocal last_t, last_bytes
            now = time.time()
            res = super().update(n)
            if now - last_t >= 2.0:
                rate = (self.n - last_bytes) / max(now - last_t, 0.001)
                print(
                    f"  [{self.n}/{self.total}] {_human(self.n)}/{_human(self.total or 0)} "
                    f"@ {_human(rate)}/s   {self.desc or ''}",
                    flush=True,
                )
                last_t = now
                last_bytes = self.n
            return res

    snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "text_encoder/*.json",
            "text_encoder/*.safetensors",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
            "scheduler/*",
            "model_index.json",
        ],
        tqdm_class=_TqdmWrap,
        max_workers=2,
    )
    print("Done.")
    return 0


def main() -> int:
    _try_utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--cache", default=None, help="override HF cache dir")
    p.add_argument("--clear-no-exist", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    if args.cache:
        import os

        os.environ["HF_HUB_CACHE"] = args.cache

    print(f"Model: {args.model}")
    print(f"Cache: {_cache_dir_for_model(args.model)}")

    if args.clear_no_exist:
        _clear_no_exist(args.model)

    missing = _missing_files(args.model)
    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for f in missing[:20]:
            print(f"  - {f}")
        if len(missing) > 20:
            print(f"  … and {len(missing) - 20} more")
    else:
        print("\nAll files present in cache. Nothing to download.")

    if args.verify_only:
        return 0 if not missing else 2

    if missing:
        return _download(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
