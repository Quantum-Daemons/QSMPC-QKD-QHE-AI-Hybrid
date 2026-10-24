"""The browser's network key schedules must reproduce the official vectors they claim, and be seen to fail.

WHY THIS EXISTS

`web/src/lib/netkeys/` computes ML-KEM-768 (FIPS 203), a Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s handshake,
the WireGuard handshake on top of it, and the IKEv2 key schedule with RFC 9370 and RFC 8784 layers. Each
module says which of its outputs are held to an official vector. `web/scripts/check_net_key_schedules.mjs`
is what makes that sentence true: it bundles the real modules and runs every stored vector through them.
This file runs that script from the Python suite, which is the job CI escalates with
`QSH_REQUIRE_BROWSER_GATES=1`, so the gate cannot pass by being skipped.

WHAT IS ASSERTED

1. The node check exits 0 and prints its OK line.
2. Every vector file records where its bytes came from: a URL and the SHA-256 of the upstream file, plus
   the commit for git-hosted sources and the ACVP tcIds. A vector without provenance is an assertion
   about this repository's own copy, which is the thing it exists to check.
3. The SEEDED NEGATIVE: for every corruption family the script defines, flipping one byte of one
   expected value in memory makes the check exit 1 and name the corrupted case. A comparison that has
   never been seen to fail cannot be told apart from one that compares a value with itself.
4. The test and the script agree on the set of families, so a family added to one is not silently
   absent from the other.

WHAT IS DELIBERATELY NOT ASSERTED

That the upstream files still hash to the recorded SHA-256 today: CI is offline by design. The hashes are
what make a later re-fetch comparable. Nor anything about the WireGuard framing or the RFC 9370 / RFC 8784
layers beyond the node check's properties - those have no official vectors, and saying more here would
be the overclaim references/net-key-schedules.md exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SCRIPT = WEB / "scripts" / "check_net_key_schedules.mjs"
VECTORS = WEB / "src" / "lib" / "netkeys" / "vectors"

#: family -> text the failure output must contain. Mirrors CORRUPTIONS in the node script; the last test
#: below fails if the two sets drift apart.
CORRUPTIONS = {
    "mlkem": "ML-KEM decapsulation tcId 88",
    "noise": "Noise message 4",
    "ikev2": "HMAC-SHA-384/g^ir1024/N64/COUNT0 DKM",
    "rfc5903": "RFC5903-8.1 g^ir",
    "wginit": "Hi = Hash(Ci || Identifier)",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _require_browser_or_skip() -> None:
    """Skip without the toolchain - unless the environment says skipping is a failure.

    Same contract as the other cross-language gates (`test_qsdc_agrees_across_languages.py`):
    `web/node_modules` is gitignored, so a bare skip would let a fresh checkout report green with the
    vectors unchecked. CI sets `QSH_REQUIRE_BROWSER_GATES=1` and installs Node, so there it fails.
    """
    missing = []
    if not ESBUILD.exists():
        missing.append("web/node_modules/.bin/esbuild")
    if shutil.which("node") is None:
        missing.append("node")
    if not missing:
        return
    msg = (
        f"{' and '.join(missing)} absent, so the network key-schedule vectors cannot be run. Run `npm ci` "
        "in web/. This is a SKIP, not a pass: the ACVP, Noise and CAVP vectors are unchecked, not satisfied."
    )
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
    pytest.skip(msg)


def _run(corrupt: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "QSH_NETKEYS_CORRUPT"}
    if corrupt is not None:
        env["QSH_NETKEYS_CORRUPT"] = corrupt
    return subprocess.run(["node", str(SCRIPT)], cwd=WEB, env=env, capture_output=True, text=True, timeout=600)


def _load(name: str) -> dict:
    return json.loads((VECTORS / name).read_text(encoding="ascii"))


def _walk(node, key: str) -> list:
    """Every value stored under `key`, anywhere in a JSON tree."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                out.append(v)
            out.extend(_walk(v, key))
    elif isinstance(node, list):
        for v in node:
            out.extend(_walk(v, key))
    return out


# ----------------------------------------------------------------------------------------------------
# Provenance - no toolchain needed, so these run everywhere
# ----------------------------------------------------------------------------------------------------

def test_every_vector_file_records_a_source_url_and_sha256() -> None:
    files = sorted(VECTORS.glob("*.json"))
    assert len(files) >= 4, f"only {len(files)} vector files under {VECTORS.relative_to(REPO)}; has the tree moved?"
    for path in files:
        doc = json.loads(path.read_text(encoding="ascii"))
        urls = [u for u in _walk(doc, "url") if isinstance(u, str)]
        shas = _walk(doc, "sha256")
        assert urls and all(u.startswith("https://") for u in urls), f"{path.name}: no https source URL recorded"
        assert shas and all(isinstance(s, str) and _SHA256.match(s) for s in shas), (
            f"{path.name}: every recorded source needs the SHA-256 of the upstream file"
        )
        retrieved = _walk(doc, "retrieved")
        assert retrieved and all(re.match(r"^\d{4}-\d{2}-\d{2}$", r) for r in retrieved), (
            f"{path.name}: no retrieval date; a hash without a date cannot be compared with a later fetch"
        )


def test_the_acvp_subset_names_its_commit_files_and_tcids() -> None:
    doc = _load("mlkem768_acvp.json")
    src = doc["source"]
    assert src["repository"] == "https://github.com/usnistgov/ACVP-Server"
    assert _COMMIT.match(src["commit"]), "the ACVP commit must be a full 40-hex SHA"
    prefix = f"https://raw.githubusercontent.com/usnistgov/ACVP-Server/{src['commit']}/gen-val/json-files/"
    paths = {f["path"] for f in src["files"]}
    for fn in ("ML-KEM-keyGen-FIPS203", "ML-KEM-encapDecap-FIPS203"):
        for name in ("prompt.json", "expectedResults.json"):
            assert f"gen-val/json-files/{fn}/{name}" in paths, f"{fn}/{name} is not recorded as a source"
    assert all(f["url"].startswith(prefix) and _SHA256.match(f["sha256"]) for f in src["files"])
    assert doc["parameterSet"] == "ML-KEM-768"
    for fn in ("keyGen", "encapsulation", "decapsulation"):
        block = doc[fn]
        assert 2 <= len(block["cases"]) <= 4, f"{fn}: {len(block['cases'])} cases; the subset is 2-4 per function"
        assert block["tcIds"] == [c["tcId"] for c in block["cases"]], f"{fn}: tcIds do not list the stored cases"


def test_the_noise_and_cavp_files_name_their_upstream() -> None:
    noise = _load("noise_ikpsk2.json")
    assert _COMMIT.match(noise["source"]["commit"]) and noise["source"]["commit"] in noise["source"]["url"]
    assert all(v["protocol_name"] == "Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s" for v in noise["vectors"])
    cavp = _load("ikev2_cavp.json")
    assert cavp["source"]["url"].startswith("https://csrc.nist.gov/")
    assert _SHA256.match(cavp["source"]["member"]["sha256"]), "the .rsp inside the zip needs its own hash"
    assert cavp["ecp256Rfc5903"]["source"]["url"].startswith("https://www.rfc-editor.org/")
    wg = _load("wireguard_initial_state.json")
    assert wg["authoritative"] is False, "boringtun's constants must stay labelled as non-authoritative"
    assert _COMMIT.match(wg["source"]["commit"]) and wg["source"]["commit"] in wg["source"]["url"]


def test_the_files_this_gate_covers_are_ascii() -> None:
    """The CI language gate rejects CJK; these files promise ASCII, which is stricter, so it is checked."""
    named = [SCRIPT, REPO / "references" / "net-key-schedules.md", Path(__file__)]
    missing = [str(p.relative_to(REPO)) for p in named if not p.is_file()]
    assert not missing, f"{missing} missing; a file that is not there is not 'clean'"
    paths = [p for p in sorted((WEB / "src" / "lib" / "netkeys").rglob("*")) if p.is_file()] + named
    assert len(paths) >= 11, f"only {len(paths)} files in scope; the glob is not finding the modules"
    offenders = [str(p.relative_to(REPO)) for p in paths if not p.read_bytes().isascii()]
    assert not offenders, f"non-ASCII bytes in {offenders}"


# ----------------------------------------------------------------------------------------------------
# The vectors themselves, through the real modules
# ----------------------------------------------------------------------------------------------------

def test_the_node_check_passes_on_the_stored_vectors() -> None:
    _require_browser_or_skip()
    r = _run()
    assert r.returncode == 0, f"check:netkeys failed:\n{r.stdout}\n{r.stderr}"
    assert "check:netkeys OK" in r.stdout, r.stdout


@pytest.mark.parametrize("family", sorted(CORRUPTIONS))
def test_one_flipped_expected_byte_fails_the_check(family: str) -> None:
    """The seeded negative. Exit 1 (a failed check), not 2 (unknown family) or a crash."""
    _require_browser_or_skip()
    r = _run(corrupt=family)
    out = r.stdout + r.stderr
    assert "NEGATIVE CONTROL" in r.stderr, f"the corruption was not applied:\n{out}"
    assert r.returncode == 1, f"flipping one expected byte ({family}) gave exit {r.returncode}:\n{out}"
    assert "check:netkeys FAILED" in r.stderr and CORRUPTIONS[family] in r.stderr, (
        f"the failure does not name the corrupted case {CORRUPTIONS[family]!r}:\n{out}"
    )
    assert "check:netkeys OK" not in r.stdout


def test_this_file_and_the_script_corrupt_the_same_families() -> None:
    """An unknown family makes the script list its own; that list must equal this file's."""
    _require_browser_or_skip()
    r = _run(corrupt="no-such-family")
    assert r.returncode == 2, f"an unknown family should exit 2, got {r.returncode}:\n{r.stderr}"
    m = re.search(r"expected one of ([a-z0-9, ]+)", r.stderr)
    assert m, r.stderr
    assert {f.strip() for f in m.group(1).split(",")} == set(CORRUPTIONS), (
        "the node script's corruption families and this file's have drifted apart"
    )
