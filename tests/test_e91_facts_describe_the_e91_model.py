"""An E91 run's security facts must describe the E91 model, not BB84's decoy-state bound.

WHY THIS EXISTS

With E91 (CHSH) selected, the run log read "E91 (entanglement / CHSH)" and the security-facts table
captioned that run's QBER as "this run's two-intensity decoy-state bound (Ma, Qi, Zhao & Lo, Phys.
Rev. A 72, 012326); post-processing aborts above 0.11". E91 in this simulator has no decoy states:
its key rate is the device-dependent 1 - h(e) - f h(e) on the Werner-model error rate, and with
f = 1.16 it reaches zero at 0.0981, not at 0.11. Observed on the live demo on 2026-10-23.

THE SHAPE, WHICH IS THE PART WORTH KEEPING

The two rows above that one were already protocol-aware, behind a constant whose comment reads "A
shared constant is what stops the next row from being the next omission". The constant was declared
INSIDE the QKD branch; the QBER row sits outside it and could not read it. A fix that names its own
failure mode and still scopes itself too narrowly to prevent it.

The same rows said "GLLP secure fraction on the observed QBER". Since Round 45 the E91 bound takes
the MODEL error rate e0, not the sampled QBER, so "observed" described code that no longer exists.

WHAT IS ASSERTED

Behaviourally, through the real browser module: for an E91 run every QKD fact names the E91 model
and none names decoy states, GLLP or the observed QBER; for a BB84 run the decoy-state wording is
unchanged. And both E91 emit sites expose the model error rate, so S = 2 sqrt(2)(1 - 2 e0) can be
checked exactly from the page.

THE SAME TABLE CALLED EVERY QKD ROW "MEASURED"

Under the heading "what this run measured", with the warning "The bits are real; there are not
enough of them." No QKD device exists in this project, and the browser's bb84() returns a COUNT,
never key bytes (checklist SC2). The QKD rows are now kind "simulated", for both protocols, and the
banner every page renders says the quantum channel is simulated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
CLIENT = WEB / "src" / "lib" / "clientCompute.ts"


def _require_browser_or_skip() -> None:
    if ESBUILD.exists():
        return
    msg = ("web/node_modules/.bin/esbuild is absent, so the browser module cannot be executed. Run "
           "`npm ci` in web/. This is a SKIP rather than a pass: the labels are unchecked.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
    pytest.skip(msg)


#: Packages in `clientCompute.ts`'s import graph that the facts table never calls. They are
#: STUBBED, not bundled: node-seal is imported as a Vite `?url` wasm asset and the ML runtimes expect a
#: browser, so a plain bundle fails. A CommonJS Proxy stub satisfies any named import esbuild sees, and
#: nothing in `securityFacts` reaches it - if something ever did, the call would return undefined and
#: the assertions below would fail rather than pass.
_STUB = r"^(node-seal|onnxruntime-web|@mlc-ai/web-llm|@huggingface/transformers|emp-wasm-engine|mpc-framework|summon-ts|gif\.js|html-to-image|papaparse)(/.*)?$|\?url$|\.wasm$"

_BUILD = """
const esbuild = require(%(esbuild)s);
esbuild.build({
  entryPoints: [%(entry)s], bundle: true, format: "esm", platform: "node", outfile: %(out)s,
  logLevel: "error",
  plugins: [{ name: "stub-heavy", setup(b) {
    b.onResolve({ filter: new RegExp(%(stub)s) }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ loader: "js",
      contents: "module.exports = new Proxy(function () {}, { get: (t, k) => k === '__esModule' ? false : t, apply: () => undefined });" }));
  } }],
}).catch((e) => { console.error(e.message); process.exit(1); });
"""


def _facts(protocol: str, key_bits: int = 145000) -> list[dict]:
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, build, runner = d / "entry.ts", d / "bundle.mjs", d / "build.cjs", d / "run.mjs"
        entry.write_text(f"export {{ securityFacts }} from {json.dumps(str(WEB / 'src/lib/clientCompute'))};\n")
        build.write_text(_BUILD % {
            "esbuild": json.dumps(str(WEB / "node_modules" / "esbuild")), "entry": json.dumps(str(entry)),
            "out": json.dumps(str(bundle)), "stub": json.dumps(_STUB)})
        subprocess.run(["node", str(build)], cwd=WEB, check=True, capture_output=True, text=True)
        cfg = {"qkd": "bb84-decoy", "qkd_protocol": protocol, "usecase": "telecom_ids",
               "qkd_distance_km": 50, "qkd_misalignment": 0.015, "qkd_eve": 0}
        metrics = {"qber": 0.03, "skr_per_pulse": 0.007286, "secure_key_bits": key_bits}
        runner.write_text(
            f"globalThis.window = globalThis; globalThis.self = globalThis;\n"
            f"const m = await import({json.dumps(bundle.as_uri())});\n"
            f"console.log(JSON.stringify(m.securityFacts({json.dumps(cfg)}, {json.dumps(metrics)})));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout.strip().splitlines()[-1])


def _qkd_text(facts: list[dict]) -> str:
    rows = [f for f in facts if f.get("label") in ("QBER", "Secret-key rate", "QKD key material")]
    assert len(rows) == 3, f"expected the three QKD rows, got {[f.get('label') for f in facts]}"
    return " | ".join(f"{f['label']}: {f.get('source', '')} {f.get('warn') or ''}" for f in rows)


def test_an_e91_run_is_described_by_the_e91_model() -> None:
    text = _qkd_text(_facts("e91"))
    for bad in ("decoy-state bound", "two-intensity", "GLLP", "observed QBER", "aborts above 0.11"):
        assert bad not in text, f"an E91 run's facts say {bad!r}:\n{text}"
    assert "Werner-model error rate" in text and "0.0981" in text, text
    assert "S > 2.424" in text, "the device-independent comparison must stay beside the E91 QBER"


def test_a_bb84_run_keeps_its_decoy_wording() -> None:
    """Anti-tautology: the branch must split, not delete the BB84 description."""
    text = _qkd_text(_facts("bb84"))
    assert "two-intensity decoy-state bound" in text, text
    assert "Werner" not in text, text


def test_the_protocol_test_is_at_function_scope() -> None:
    """The mechanism that failed: a constant declared inside a branch the last row could not see."""
    src = CLIENT.read_text()
    body = src[src.index("export function securityFacts("):]
    body = body[: body.index("\nexport function ")]
    decl = body.index("const isE91")
    first_facts = body.index("const facts: SecurityFact[]")
    assert decl < first_facts, "isE91 must be declared before the facts list, at function scope"
    assert body.count("const isE91") == 1, "one declaration, read by every row"


def test_both_e91_sites_expose_the_model_error_rate() -> None:
    src = CLIENT.read_text()
    chsh = len(re.findall(r'metric\(stage,\s*"chsh_s"', src))
    model = len(re.findall(r'metric\(stage,\s*"e91_model_qber"', src))
    assert chsh == model == 2, (
        f"{chsh} chsh_s emit site(s) and {model} e91_model_qber site(s). The fibre and satellite QKD "
        "stages each run E91; a model error rate on one and not the other is the one-twin defect.")


@pytest.mark.parametrize("protocol", ["e91", "bb84"])
def test_every_qkd_row_is_labelled_simulated(protocol: str) -> None:
    facts = _facts(protocol)
    qkd = [f for f in facts if f.get("label") in ("QBER", "Secret-key rate", "QKD key material",
                                                     "Decoy-state necessity")]
    assert len(qkd) == 4, [f.get("label") for f in facts]
    wrong = [f"{f['label']}: {f.get('kind')}" for f in qkd if f.get("kind") != "simulated"]
    assert not wrong, f"QKD rows not labelled simulated on a {protocol} run: {wrong}"
    # Anti-tautology: the PQC rows are still what they were.
    kinds = {f["label"]: f.get("kind") for f in facts}
    assert kinds.get("KEM") == "standard" and kinds.get("KEM ciphertext") == "measured", kinds
    # The 1..255 band is the only one whose warning carried the sentence, so ask for it.
    band = _facts(protocol, key_bits=94)
    warn = next(f.get("warn") or "" for f in band if f.get("label") == "QKD key material")
    assert "below the 256" in warn, f"the 1..255-bit warning did not fire: {warn!r}"
    assert "bits are real" not in warn and "simulated channel model" in warn, warn


def test_the_panel_renders_the_simulated_kind_and_the_banner_says_so() -> None:
    panel = (WEB / "src" / "components" / "BenchmarkPanel.tsx").read_text()
    assert re.search(r'f\.kind\s*===\s*"simulated"\s*\?\s*"simulated', panel), (
        "BenchmarkPanel renders every non-standard row as 'measured'; a simulated kind is invisible")
    banner = (WEB / "src" / "lib" / "demoMode.ts").read_text()
    m = re.search(r"export const DEMO_BANNER\s*=([^;]*);", banner, flags=re.S)
    assert m and "quantum channel (QKD, QSDC) is" in m.group(1) and "simulated" in m.group(1), (
        "the banner every page renders no longer says the quantum channel is simulated")
