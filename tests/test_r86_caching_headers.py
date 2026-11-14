"""The app shell is revalidated on every load, and only a hashed asset that exists is kept for a year (Round 86).

WHY THIS EXISTS (2026-11-15). Neither Caddyfile sent Cache-Control. After that day's deploy, Chrome kept the app shell
it had fetched an hour earlier, by its heuristic of a tenth of the time since Last-Modified, and loaded the previous
bundle from its cache. The new image no longer served that bundle: `index-PNyoH7-C.js` answered 404. A visitor with the
old shell and no copy of the old bundle gets a blank page until the heuristic runs out.

WHAT IS ASSERTED. Both Caddyfiles say the same:
- a file under /assets/ that exists on disk is "public, max-age=31536000, immutable", because Vite names every
  file there by its content hash;
- everything else is "no-cache": the shell, the app's routes, the data and licence files, and a 404 for a hash the
  image does not have. A cached 404 would keep a rollback broken in that browser.

The matcher's root is the file server's. The behaviour was read from a real Caddy (caddy:2-alpine) serving a
client-only build of this tree, with `curl -I`; the deployed site is read the same way after the merge.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CADDYFILES = [REPO / "docker" / "env" / "Caddyfile.client-only", REPO / "docker" / "env" / "Caddyfile"]

HASHED = re.compile(
    r"\t@hashed \{\n\t\tpath /assets/\*\n\t\tfile \{\n\t\t\troot (\S+)\n\t\t\}\n\t\}\n"
    r"\theader @hashed Cache-Control \"public, max-age=31536000, immutable\"\n")
UNHASHED = re.compile(
    r"\t@unhashed \{\n\t\tnot \{\n\t\t\tpath /assets/\*\n\t\t\tfile \{\n\t\t\t\troot (\S+)\n\t\t\t\}\n\t\t\}\n\t\}\n"
    r"\theader @unhashed Cache-Control \"no-cache\"\n")


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_an_existing_hashed_asset_is_kept_and_everything_else_revalidated(path: Path) -> None:
    text = path.read_text()
    hashed, unhashed = HASHED.search(text), UNHASHED.search(text)
    assert hashed, f"{path.name}: no year-long Cache-Control for the existing files under /assets/"
    assert unhashed, f"{path.name}: no no-cache for the shell, the routes, the data and a missing asset"
    served = re.search(r"^\t\troot \* (\S+)$", text, re.M)
    assert served and hashed.group(1) == unhashed.group(1) == served.group(1), (
        f"{path.name}: the matchers look for files under a root the file server does not serve")
    # Top-level, so they apply to the routes the default handle rewrites to index.html, and to its 404s.
    assert text.index("header @unhashed") < text.index("\thandle {"), f"{path.name}: the caching rule is inside a handle"


def test_vite_keeps_the_hash_in_every_asset_name() -> None:
    """The year-long rule is safe only while each name under /assets/ is its content's hash, Vite's default."""
    config = (REPO / "web" / "vite.config.ts").read_text()
    for option in ("assetFileNames", "chunkFileNames", "entryFileNames"):
        for m in re.finditer(rf"{option}\s*:\s*[\"'`]([^\"'`]+)[\"'`]", config):
            assert "[hash]" in m.group(1), f"vite.config.ts names {option} without [hash]: {m.group(1)}"
