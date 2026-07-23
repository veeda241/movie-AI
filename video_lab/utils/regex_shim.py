"""
Ensure the pure-Python ``_regex`` shim is installed in the ``regex`` package
directory so that the blocked C extension (``_regex.pyd``) is bypassed.

The compiled ``_regex.cp312-win_amd64.pyd`` is often blocked by Windows
Application Control policy when the project lives under ``Downloads``.  This
helper writes a pure-Python replacement into the ``regex`` package so that
``transformers`` and ``diffusers`` / ``CogVideoXPipeline`` can import without
crashing.

Call ``ensure_regex_shim()`` at application startup *before* any code imports
``regex``, ``transformers`` or ``diffusers``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _find_regex_package() -> Path | None:
    """Return the path to the installed ``regex`` package directory, or None."""
    # Use importlib.util.find_spec — it does NOT load the module or trigger
    # the blocked .pyd, so it's safe to call before the shim is installed.
    try:
        import importlib.util

        spec = importlib.util.find_spec("regex")
        if spec and spec.origin:
            pkg = Path(spec.origin).resolve().parent
            if pkg.is_dir():
                return pkg
    except Exception:
        pass
    return None


def ensure_regex_shim() -> bool:
    """Write the pure-Python ``_regex`` shim into the ``regex`` package directory
    and disable the compiled ``.pyd`` if present.

    Returns True if the shim is in place and the C extension has been disabled,
    False otherwise.
    """
    pkg_dir = _find_regex_package()
    if pkg_dir is None:
        print("WARNING [regex_shim]: could not find regex package directory.", file=sys.stderr)
        return False

    # --- disable the blocked C extension (.pyd) ---
    for pyd in pkg_dir.glob("_regex*.pyd"):
        disabled = pyd.with_name(pyd.name + ".disabled")
        if not disabled.exists():
            pyd.rename(disabled)

    # --- write/refresh the shim file from source ---
    shim_path = pkg_dir / "_regex.py"
    shim_src_path = Path(__file__).resolve().parent / "_regex_shim_source.py"

    if not shim_src_path.exists():
        print(
            f"WARNING [regex_shim]: source file {shim_src_path} not found — "
            f"shim not installed.",
            file=sys.stderr,
        )
        return shim_path.exists()

    shutil.copy2(shim_src_path, shim_path)
    return True
