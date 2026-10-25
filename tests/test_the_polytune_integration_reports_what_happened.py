"""The Polytune integration checks its own answer, and reports a failure as the failure it was.

WHY THIS EXISTS

Round 68 reviewed how the demo uses Polytune. Polytune itself held up at the commit pinned here:
- `mpc()`, its only public entry, always runs the preprocessing among the parties by OT extension.
  The trusted-dealer mode is crate-private, so the demo cannot reach it.
- Its randomness is `rand`'s thread-local CSPRNG, seeded from the OS, which on wasm32 means
  `crypto.getRandomValues`. Shared randomness comes from a commit-then-reveal coin toss. The fixed
  seeds in the source are in tests.
- The statistical security parameter is 40.
Two security issues are open upstream (#63, the KOS15 consistency check; #100, the AES-CTR PRG), and
docs/threat-model.md records what they mean here.

The integration did not hold up in two places.
- A failure of either half of the hybrid returned `null`. The caller then reported "node-seal did
  not load", and published the cost model's split under the keys the measured path fills
  (`fhe_linear_ms`, `mpc_nonlinear_ms`, `mpc_comm_kb`). This is the same defect the comment beside it
  describes, in three more keys. An MPC failure can be the protocol aborting on a failed MAC or
  commitment check. That is what malicious security looks like from the honest side, and it read as
  a missing library.
- Nothing checked the circuit's answer. In this tab both parties' shares are present, so the same
  argmax can be computed in the clear and compared.

WHAT IS ASSERTED

- Behaviourally, through clientMPC.ts bundled with a stub module: the in-the-clear index is computed
  independently of the module, with the circuit's tie rule, so a wrong answer from the module is
  visible; and a module that throws makes `secureArgmax` reject rather than return something.
- Statically, in clientCompute.ts: the hybrid reports which half failed and why; the fallback names
  that cause in `encrypted_tier` and in its log line; and the measured branch reports
  `mpc_matches_clear` and says so loudly on a disagreement. The keys themselves are held by
  test_a_config_constant_never_wears_a_measurements_name.py.
- The crate refuses inputs it cannot index (more than 255 values) before compiling anything.
- The Polytune commit is the one reviewed. Moving it fails this test until the review is redone and
  REVIEWED_POLYTUNE_REV is updated with it.
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
LIB = WEB / "src" / "lib"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: The Polytune commit Round 68 reviewed: preprocessor, randomness, RHO and the open upstream issues.
REVIEWED_POLYTUNE_REV = "47dc37840a241ba0e27c1b2234059746a1ba05b7"

_BUILD = """
const esbuild = require(%(esbuild)s);
esbuild.build({
  entryPoints: [%(entry)s], bundle: true, format: "esm", platform: "node", outfile: %(out)s,
  logLevel: "error",
  plugins: [{ name: "stub-mpc", setup(b) {
    b.onResolve({ filter: /\\/wasm\\/qsh_mpc\\/qsh_mpc\\.js$/ }, (a) => ({ path: a.path, namespace: "mpcstub" }));
    b.onLoad({ filter: /.*/, namespace: "mpcstub" }, () => ({ loader: "js", contents: %(module)s }));
  } }],
}).catch((e) => { console.error(e.message); process.exit(1); });
"""

#: A module whose answer the test controls: a fixed index, or a throw, chosen per call.
_MODULE = """
export default async function init() {}
export async function secure_argmax(a, b) {
  const plan = globalThis.__plan.shift();
  if (plan === "throw") throw new Error("MAC check failed (stub)");
  return { arg: plan, mpcMs: 0, bytes: 1, messages: 1 };
}
"""


def _run(script: str) -> dict:
    if not ESBUILD.exists():
        msg = ("web/node_modules/.bin/esbuild is absent, so clientMPC.ts cannot be executed. Run `npm ci` "
               "in web/. This is a SKIP rather than a pass.")
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, build, runner = d / "entry.ts", d / "bundle.mjs", d / "build.cjs", d / "run.mjs"
        entry.write_text(f"export {{ secureArgmax }} from {json.dumps(str(LIB / 'clientMPC'))};\n")
        build.write_text(_BUILD % {"esbuild": json.dumps(str(WEB / "node_modules" / "esbuild")),
                                   "entry": json.dumps(str(entry)), "out": json.dumps(str(bundle)),
                                   "module": json.dumps(_MODULE)})
        subprocess.run(["node", str(build)], cwd=WEB, check=True, capture_output=True, text=True)
        runner.write_text(f"const m = await import({json.dumps(bundle.as_uri())});\n" + script)
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_clear_index_is_independent_of_the_module_and_breaks_ties_like_the_circuit() -> None:
    res = _run("""
const out = {};
globalThis.__plan = [1, 0, 2];
out.wrong = await m.secureArgmax([1.5, -0.5]);           // the module claims 1; the clear argmax is 0
out.tie = await m.secureArgmax([0.25, 0.25, -1]);       // equal values: the first maximum wins
out.right = await m.secureArgmax([-1, 0, 2]);
globalThis.__plan = ["throw"];
try { await m.secureArgmax([0, 1]); out.thrown = false; } catch (e) { out.thrown = String(e.message); }
console.log(JSON.stringify(out));
""")
    assert res["wrong"]["out"] == [1] and res["wrong"]["clearOut"] == [0], (
        "the in-the-clear index must come from the logits, not from the module, or a wrong answer "
        "from the circuit agrees with itself")
    assert res["tie"]["clearOut"] == [0], "ties must go to the first maximum, as the circuit's strict > does"
    assert res["right"]["out"] == res["right"]["clearOut"] == [2]
    assert res["thrown"] and "MAC check failed" in res["thrown"], (
        "a protocol failure inside the module must reach the caller as a rejection, with its message")


def test_the_hybrid_says_which_half_failed_and_why() -> None:
    src = (LIB / "clientCompute.ts").read_text()
    assert re.search(r"export interface HybridFailure \{\s*failed: \"fhe\" \| \"mpc\";\s*reason: string;", src)
    assert "): Promise<RealHybridRun | HybridFailure> {" in src, "realClientHybrid went back to returning null"
    body = src.split("export async function realClientHybrid", 1)[1].split("\n}\n", 1)[0]
    assert re.search(r'half = "mpc";\s*const mpc = await secureArgmax\(', body), (
        "the secure argmax must run with `half` set to \"mpc\", or its failure is reported as the FHE's")
    assert 'half = "mpc"; await warmMPC();' in body
    assert "return { failed: half, reason:" in body
    assert "return null" not in body, "a failure reduced to null loses which half failed"


def test_the_fallback_names_the_cause_and_the_run_checks_the_circuit() -> None:
    src = (LIB / "clientCompute.ts").read_text()
    assert 'if (!("failed" in real)) {' in src
    assert "the garbled-circuit MPC did not run to completion (${real.reason})" in src
    assert "the in-browser CKKS computation failed (${real.reason})" in src
    assert 'metric(stage, "encrypted_tier", `failed: ${cause}`);' in src, "the tier status must carry the cause"
    assert re.search(r'metric\(stage, "mpc_matches_clear", mpcAgrees \? 1 : 0\)', src)
    assert "garbled-circuit argmax DISAGREES with the same argmax in the clear" in src
    mpc = (LIB / "clientMPC.ts").read_text()
    assert "clearOut: number[];" in mpc and "clearOut: [clear]" in mpc


def test_the_crate_refuses_inputs_it_cannot_index() -> None:
    src = (REPO / "wasm" / "qsh-mpc" / "src" / "lib.rs").read_text()
    guard = src.index("if a.len() > 255 {")
    assert guard < src.index("let code = argmax_source(a.len());"), "the bound must be checked before compiling"
    assert "fn refuses_inputs_it_cannot_index()" in src and "run(256, 256)" in src


def test_the_polytune_commit_is_the_one_reviewed() -> None:
    cargo = (REPO / "wasm" / "qsh-mpc" / "Cargo.toml").read_text()
    m = re.search(r'polytune\s*=\s*\{[^}]*rev\s*=\s*"([0-9a-f]{40})"', cargo)
    assert m, "Polytune must stay pinned to a full commit hash"
    assert m.group(1) == REVIEWED_POLYTUNE_REV, (
        f"Polytune moved to {m.group(1)[:7]}. Redo the Round 68 review against it before shipping: "
        "the public mpc() still uses the untrusted (OT-extension) preprocessor, the RNG is still a "
        "CSPRNG, RHO is unchanged, and upstream issues #63 and #100 are re-read. Then update "
        "REVIEWED_POLYTUNE_REV and docs/threat-model.md.")
