#!/usr/bin/env python3
"""cdxgen's cryptographic assets for web/ beside the CBOM's, as Markdown for the CI job summary (Round 86, 2026-11-23).

Report only, the owner's choice of 2026-11-23: cdxgen 13.8.0 reads a copy of web/ with --include-crypto in CI, and this
lists what it found next to `web/public/cbom/qsh.cdx.json`, matched by name with case, spaces and punctuation ignored.
It never fails: a missing or unreadable document is reported as such. The CBOM stays this repository's own list; cdxgen
follows node:crypto, WebCrypto and JWT calls, so it is not expected to see the noble libraries, node-seal or the Python
side (docs/development.md, CBOM).

    python3 scripts/compare_cdxgen_crypto.py <cdxgen-web.cdx.json> <web/public/cbom/qsh.cdx.json>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def crypto_assets(doc: dict) -> list[dict]:
    return [c for c in doc.get("components", []) if c.get("type") == "cryptographic-asset"]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1].strip())
        return 0
    print("### cdxgen 13.8.0 on web/ (report only)\n")
    found_path, cbom_path = Path(argv[1]), Path(argv[2])
    try:
        found = json.loads(found_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"cdxgen wrote no readable document ({type(e).__name__}); see the job's log.")
        return 0
    cbom = json.loads(cbom_path.read_text(encoding="utf-8"))
    theirs, ours = crypto_assets(found), crypto_assets(cbom)
    ours_by_key = {key(c["name"]): c["name"] for c in ours}
    print(f"cdxgen listed {len(found.get('components', []))} components, {len(theirs)} of them cryptographic assets; "
          f"the CBOM lists {len(ours)}.\n")
    if theirs:
        print("| cdxgen's asset | In the CBOM as |")
        print("|---|---|")
        for c in theirs:
            print(f"| {c.get('name', '?')} | {ours_by_key.get(key(c.get('name', '')), '-')} |")
        print()
    missed = sorted(set(ours_by_key) - {key(c.get('name', '')) for c in theirs})
    print(f"CBOM assets cdxgen did not list ({len(missed)}): {', '.join(ours_by_key[k] for k in missed) or 'none'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
