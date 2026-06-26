#!/usr/bin/env python3
"""Generate web/src/lib/config.defaults.json from the central Python config.

The SPA's cost model / tier architectures / security + advantage weights / demo data sizes are a
single source of truth in src/qsh/config/defaults.py. The full deployment serves them at /api/config;
the public demo (zero /api) bakes this generated JSON. Run this after editing defaults.py:

    python scripts/export_config.py

So the frontend never hand-duplicates the backend's numbers (kills the FE<->BE cost-model drift).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsh.config import frontend_config  # noqa: E402

OUT = ROOT / "web" / "src" / "lib" / "config.defaults.json"


def main() -> int:
    data = frontend_config()
    data["_generated_by"] = "scripts/export_config.py (do not edit by hand; edit src/qsh/config/defaults.py)"
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
