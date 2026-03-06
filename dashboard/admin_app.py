from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Draftkings" / "admin_app.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Expected admin app at {TARGET}")

# Keep the dashboard launcher synced with the primary admin app implementation.
runpy.run_path(str(TARGET), run_name="__main__")
