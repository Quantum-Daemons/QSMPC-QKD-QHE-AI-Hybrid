#!/usr/bin/env python
"""Record the SHA-256 of every file the browser fetches from the Hugging Face Hub, at the revision each model is pinned to.

A pin names a commit, and says nothing about what a server returns for it. The browser checks each file it fetches against
this record before it runs or reads it (`web/src/lib/pinnedFetch.ts`), so a file altered on the Hub or in transit is refused.
The record is built from the bytes themselves: for a Git LFS file the Hub also states the digest, and the two have to agree.

    python scripts/pin_hub_hashes.py           # writes models/hub_file_hashes.json
    python scripts/pin_hub_hashes.py --check   # compares with the Hub, writes nothing, exits 1 on a difference

Run it after `scripts/publish_hf.py` pins a new revision; `tests/test_r86_hub_file_hashes.py` fails until it has been.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import registry  # noqa: E402

OUT = REPO / "models" / "hub_file_hashes.json"
#: The one file the browser fetches that the registry does not list as an artifact.
SAMPLE = "eval_sample.json"
_HEADERS = {"User-Agent": "qsh-pin-hub-hashes"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS), timeout=120) as r:
        return r.read()


def _tree(repo: str, rev: str) -> dict[str, dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    return {e["path"]: e for e in json.loads(_get(url)) if e.get("type") == "file"}


def _fetched_by_the_browser(spec: dict, tree: dict[str, dict]) -> list[str]:
    paths = [a["path"] for a in spec["artifacts"] if a["serves"] in ("browser", "both") and a["fmt"] in ("onnx", "json")]
    missing = [p for p in paths if p not in tree]
    if missing:
        raise SystemExit(f"{spec['key']} @ {spec['hf_revision']}: the registry lists {missing}, which that revision does not hold")
    return sorted({*paths, *([SAMPLE] if SAMPLE in tree else [])})


def digest(repo: str, rev: str, path: str, entry: dict) -> str:
    quoted = "/".join(urllib.parse.quote(p) for p in path.split("/"))
    got = hashlib.sha256(_get(f"https://huggingface.co/{repo}/resolve/{rev}/{quoted}")).hexdigest()
    stated = (entry.get("lfs") or {}).get("oid")
    if stated and stated != got:
        raise SystemExit(f"{repo}@{rev}:{path}: the bytes hash to {got}, and the Hub's LFS record says {stated}")
    return got


def build() -> dict[str, dict]:
    record: dict[str, dict] = {}
    for specs in registry.frontend_models().values():
        for spec in specs:
            repo, rev = spec["hf_repo_id"], spec["hf_revision"]
            if not repo or not rev:
                continue
            tree = _tree(repo, rev)
            files = {p: digest(repo, rev, p, tree[p]) for p in _fetched_by_the_browser(spec, tree)}
            record[spec["key"]] = {"revision": rev, "files": files}
            print(f"{spec['key']}: {len(files)} files at {rev[:12]}")
    return dict(sorted(record.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the Hub and write nothing")
    args = ap.parse_args()
    record = build()
    if args.check:
        on_disk = json.loads(OUT.read_text(encoding="utf-8"))
        if on_disk != record:
            changed = sorted(k for k in {*on_disk, *record} if on_disk.get(k) != record.get(k))
            print(f"differs from {OUT.name}: {changed}", file=sys.stderr)
            return 1
        print(f"{len(record)} models match {OUT.name}")
        return 0
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(record)} models, {sum(len(v['files']) for v in record.values())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
