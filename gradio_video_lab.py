"""Launch the Video Model Lab Gradio UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

# Ensure pure-Python _regex shim when Windows blocks the native regex DLL.
try:
    from video_lab.utils.regex_shim import ensure_regex_shim

    ensure_regex_shim()
except Exception:
    pass

from video_lab.app import main

if __name__ == "__main__":
    main()
