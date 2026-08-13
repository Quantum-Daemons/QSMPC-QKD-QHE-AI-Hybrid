"""node-seal ships Microsoft SEAL 4.1.2, and upstream has since shipped a major security update.

MEASURED, not inferred:

  - `web/package.json` declares `node-seal: ^7.0.0`; 7.0.0 is also the newest version on npm, and
    npm's `time.modified` for the package is 2025-11-08. Nothing has been published since.
  - The installed package's own README says "Currently aligned with Microsoft SEAL 4.1.2".
  - The GitHub API for microsoft/SEAL lists v4.4.3 (2026-08-04) as latest, and the release body of
    v4.4.0 (2026-07-25) opens with the words "Major security update."

So the browser's CKKS runs on a release line that upstream has superseded for security reasons, and
there is no route to a newer one: node-seal is a single-maintainer package whose SEAL submodule
points at a personal fork, so upstream fixes do not flow in even when they land.

EXPOSURE, stated honestly rather than either ignored or inflated. The 4.3.x/4.4.x fixes concentrate
on deserialising untrusted input (bounded key deserialisation, oversized headers, decompression
bombs) and on C API misuse (null-pointer dereferences, C++ exceptions crossing the ABI boundary).
The demo encrypts and decrypts the visitor's own data inside one tab: no third party supplies a
ciphertext, key or serialised context, so the untrusted-deserialisation class largely does not
apply here. What is NOT ruled out is the C API hardening, since node-seal v7 exposes that binding
surface more directly than v6 did.

This file is a DRIFT GATE, not a fix. There is nothing to fix until node-seal publishes. It fails
when the declared version changes, so that a bump has to be accompanied by re-reading this note and
re-checking which SEAL the new release carries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "web" / "package.json"

#: The state this note was written against. Changing either value must be a deliberate act.
_DECLARED_RANGE = "^7.0.0"
_BUNDLED_SEAL = "4.1.2"
#: Newest upstream SEAL at the time of writing, and the first release that is a security update.
_UPSTREAM_LATEST = "4.4.3"
_UPSTREAM_SECURITY = "4.4.0"


def test_the_declared_node_seal_range_has_not_moved_silently() -> None:
    """If this fails, node-seal was bumped - re-verify which SEAL the new version carries."""
    pkg = json.loads(_PKG.read_text(encoding="utf-8"))
    declared = (pkg.get("dependencies") or {}).get("node-seal")
    assert declared == _DECLARED_RANGE, (
        f"web/package.json declares node-seal {declared!r}, this gate was written against "
        f"{_DECLARED_RANGE!r}. A bump is good news, but it invalidates the exposure note in this "
        f"file's docstring: check the new package's README for the SEAL version it aligns with, "
        f"confirm whether it is at or past {_UPSTREAM_SECURITY} (the 'Major security update' "
        f"release), and rewrite the note before updating these constants.")


def test_the_caret_range_cannot_silently_cross_a_major() -> None:
    """`^7.0.0` admits 7.x only. Recorded because the semantics are what bound the exposure.

    A `^` range would happily install 7.1.0 with a different SEAL underneath and no code change
    anywhere in this repository. That is acceptable only because the check above pins the declared
    range and the lockfile pins the resolved version; if either goes away, this stops being true.
    """
    assert _DECLARED_RANGE.startswith("^7."), _DECLARED_RANGE
    lock = _PKG.parent / "package-lock.json"
    assert lock.exists(), "no lockfile: the resolved node-seal version would float between installs"
    resolved = json.loads(lock.read_text(encoding="utf-8"))
    entry = (resolved.get("packages") or {}).get("node_modules/node-seal")
    assert entry, "node-seal absent from the lockfile"
    assert re.fullmatch(r"7\.\d+\.\d+", entry["version"]), entry["version"]


def test_the_recorded_gap_is_still_a_gap() -> None:
    """Anti-tautology: the constants must actually describe a version behind upstream.

    If someone updates `_BUNDLED_SEAL` to match `_UPSTREAM_LATEST` without the underlying package
    changing, the two tests above would still pass while the note became false.
    """
    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    assert parts(_BUNDLED_SEAL) < parts(_UPSTREAM_SECURITY) <= parts(_UPSTREAM_LATEST), (
        f"bundled {_BUNDLED_SEAL} is no longer behind the security release {_UPSTREAM_SECURITY}; "
        f"if node-seal caught up, delete this gate and say so in the checklist rather than leaving "
        f"a passing test that asserts a resolved problem")
