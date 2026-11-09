"""Every file the browser fetches from the Hugging Face Hub is checked against a recorded SHA-256 (Round 86).

A pin names a commit and says nothing about what a server returns for it. `scripts/pin_hub_hashes.py` records the digest of
each file the browser fetches, and `web/src/lib/pinnedFetch.ts` refuses a file that does not match or has no entry.

WHAT IS ASSERTED. The record covers every pinned model and is for the revision each is pinned to, so a republish that moved a
pin fails here until the script is run. The check refuses altered bytes and a missing entry, in the code the browser runs.
Every Hub fetch goes through it, and a file is not read until it has passed.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
RECORD = json.loads((REPO / "models" / "hub_file_hashes.json").read_text(encoding="utf-8"))
PINNED = [s for specs in registry.frontend_models().values() for s in specs if s["hf_repo_id"] and s["hf_revision"]]


def test_the_scan_found_the_pinned_models() -> None:
    assert len(PINNED) >= 11


@pytest.mark.parametrize("spec", PINNED, ids=lambda s: s["key"])
def test_the_record_is_for_the_revision_this_model_is_pinned_to(spec: dict) -> None:
    assert RECORD.get(spec["key"], {}).get("revision") == spec["hf_revision"], "the pin moved: run scripts/pin_hub_hashes.py"


@pytest.mark.parametrize("spec", PINNED, ids=lambda s: s["key"])
def test_every_file_the_browser_fetches_has_a_digest(spec: dict) -> None:
    hashes = spec["file_sha256"]
    assert hashes and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values())
    wanted = {a["path"] for a in spec["artifacts"] if a["fmt"] == "onnx" and a["serves"] in ("browser", "both")}
    wanted |= {a["path"] for a in spec["artifacts"] if a["path"].endswith("weights.json")}
    if not spec["eval_sample_reason"]:
        wanted.add("eval_sample.json")
    assert wanted <= set(hashes), sorted(wanted - set(hashes))


def test_no_recorded_model_has_lost_its_pin() -> None:
    assert set(RECORD) == {s["key"] for s in PINNED}


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|(?<!:)//[^\n]*", "", text, flags=re.S)


@pytest.mark.parametrize(("name", "calls"), [("onnxInference.ts", 2), ("forecastStudent.ts", 1), ("clientCompute.ts", 3)])
def test_each_hub_fetch_is_checked_before_it_is_read(name: str, calls: int) -> None:
    source = _strip_comments((WEB / "src" / "lib" / name).read_text(encoding="utf-8"))
    assert source.count("checkPinned(") == calls
    # After a check, the body is read from the bytes that were checked and never from the response a second time.
    assert not re.search(r"(huggingface\.co|target\.url|modelUrl|\bfetch\(url)[^\n]*\n(?:[^\n]*\n){0,6}?[^\n]*await res\.json\(\)", source), name


def test_nothing_else_in_the_browser_fetches_a_hub_file_unchecked() -> None:
    sites = []
    for path in (WEB / "src").rglob("*.ts*"):
        source = _strip_comments(path.read_text(encoding="utf-8"))
        if "huggingface.co/" in source and "/resolve/" in source:
            sites.append(path.name)
    # clientAI hands the LLM's URL to the WebLLM engine, which has its own integrity option and fetches the shards itself.
    assert set(sites) == {"clientCompute.ts", "onnxInference.ts", "clientAI.ts"}, sites


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    d = tmp_path_factory.mktemp("pinned")
    bundle, runner = d / "b.mjs", d / "r.mjs"
    entry = f"export * from {json.dumps(str(WEB / 'src/lib/pinnedFetch'))};\n"
    subprocess.run([str(ESBUILD), "--bundle", "--loader=ts", "--format=esm", "--platform=node", '--define:import.meta.env={"BASE_URL":"/"}',
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True, input=entry.encode())
    runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + r"""
const body = new TextEncoder().encode('{"a":1}');
const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", body))].map((b) => b.toString(16).padStart(2, "0")).join("");
const altered = new TextEncoder().encode('{"a":2}');
const outcome = async (f) => { try { return { ok: await f() }; } catch (e) { return { refused: e.name, message: e.message }; } };
const out = {
  digest,
  matches: await outcome(() => m.checkPinned(body, digest, "f")),
  altered: await outcome(() => m.checkPinned(altered, digest, "f")),
  missing: await outcome(() => m.checkPinned(body, undefined, "f")),
};
console.log(JSON.stringify(out));
""")
    return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])


def test_a_matching_file_passes_and_returns_its_digest(browser) -> None:
    assert browser["matches"]["ok"] == browser["digest"]


def test_altered_bytes_are_refused(browser) -> None:
    case = browser["altered"]
    assert case["refused"] == "HubFileRefused" and "not used" in case["message"], case


def test_a_file_with_no_recorded_digest_is_refused(browser) -> None:
    case = browser["missing"]
    assert case["refused"] == "HubFileRefused" and "no SHA-256 is recorded" in case["message"], case
