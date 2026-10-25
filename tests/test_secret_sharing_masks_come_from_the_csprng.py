"""Secret-sharing masks come from the browser's CSPRNG, and the argmax shares are uniform mod 2^32.

WHY THIS EXISTS

Until Round 66 both of the demo's secret-sharing masks were drawn from the run's seeded generator,
mulberry32 with 32 bits of state, keyed on a seed the page shows:

- the secure argmax in clientMPC.ts: a_k = floor(u() * 2^16) and b_k = q_k - a_k + 2^16. Anyone with
  the seed could recompute every mask. A mask bounded to 2^16 also hides a value only statistically:
  b_k for two logits up to 1024 apart differ in distribution by up to 1024 / 2^16 = 1/64;
- the secure sum in clientCompute.ts: mask = floor(u() * p), recomputable in the same way, and biased
  because no odd prime divides 2^32.

It is the twin of a Round 40 finding about the QOTP mask in quantumSim.ts, which was disclosed in place
rather than changed, because that key decides the circuit's output and a CSPRNG key would make the
page's result irreproducible. The two masks here decide nothing: the argmax and the sum are the same
whatever the masks are. So they were changed, not disclosed. The argmax's line never contained the
word "mask", which is how it stayed out of the Round 40 sweep.

WHAT IS ASSERTED

Behaviourally, through the real browser modules bundled by esbuild:
- secureArgmax hands the module Uint32Array shares with a_k + b_k = q_k mod 2^32, takes no generator,
  and draws fresh masks across the whole 32-bit range on every call;
- randomBelow(p) stays in [0, p), is close to uniform for a small p, and refuses p outside [1, 2^32];
  randomWords fills arrays longer than one getRandomValues call.
Statically:
- the secure sum draws its masks with randomBelow and still advances the run's generator once per
  secret, so a seed selects the same downstream inputs as before;
- the Garble circuit adds in 64 bits and casts back to 32, with no offset, and its test wraps;
- across web/src, no statement draws a share or a mask from a seeded generator, except the two that
  are disclosed in place, named below with their reasons. Removing a disclosure revokes its waiver.
"""

from __future__ import annotations

import json
import math
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

#: Statements that draw share material from a seeded generator ON PURPOSE, each disclosed in place.
#: The key is the file and the statement; the value is the disclosure that must stay beside it.
WAIVED = {
    ("web/src/lib/clientCompute.ts", "const s0 = (u() * 2 - 1) * 4"):
        "representative, not cryptographic",  # mpcNonlinear: the cost model, which runs no protocol
}
#: The QOTP demo key is drawn bit by bit (`if (u() < 0.5) X(q)`), so the statement scan cannot see it;
#: its waiver is the Round 40 disclosure, held here explicitly.
QOTP_DISCLOSURE = "anyone can recompute this mask"

_STUB_MPC = r"/wasm/qsh_mpc/qsh_mpc\.js$"
_MPC_MODULE = """
export default async function init() {}
export async function secure_argmax(a, b) {
  globalThis.__mpcCalls.push({ a: Array.from(a), b: Array.from(b),
    aType: a.constructor.name, bType: b.constructor.name });
  const q = Array.from(a, (x, k) => (x + b[k]) >>> 0);
  let arg = 0;
  for (let k = 1; k < q.length; k++) if (q[k] > q[arg]) arg = k;
  return { arg, mpcMs: 0, bytes: 1, messages: 1 };
}
"""

_BUILD = """
const esbuild = require(%(esbuild)s);
esbuild.build({
  entryPoints: [%(entry)s], bundle: true, format: "esm", platform: "node", outfile: %(out)s,
  logLevel: "error",
  plugins: [{ name: "stub-mpc", setup(b) {
    b.onResolve({ filter: new RegExp(%(stub)s) }, (a) => ({ path: a.path, namespace: "mpcstub" }));
    b.onLoad({ filter: /.*/, namespace: "mpcstub" }, () => ({ loader: "js", contents: %(module)s }));
  } }],
}).catch((e) => { console.error(e.message); process.exit(1); });
"""


def _require_browser_or_skip() -> None:
    if ESBUILD.exists():
        return
    msg = ("web/node_modules/.bin/esbuild is absent, so the browser modules cannot be executed. Run "
           "`npm ci` in web/. This is a SKIP rather than a pass: the masks are unchecked.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
    pytest.skip(msg)


def _run(export: str, module: str, script: str) -> dict:
    """Bundle `export` from web/src/lib/<module>.ts, run `script` against it, return its JSON."""
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, build, runner = d / "entry.ts", d / "bundle.mjs", d / "build.cjs", d / "run.mjs"
        entry.write_text(f"export {{ {export} }} from {json.dumps(str(LIB / module))};\n")
        build.write_text(_BUILD % {
            "esbuild": json.dumps(str(WEB / "node_modules" / "esbuild")), "entry": json.dumps(str(entry)),
            "out": json.dumps(str(bundle)), "stub": json.dumps(_STUB_MPC), "module": json.dumps(_MPC_MODULE)})
        subprocess.run(["node", str(build)], cwd=WEB, check=True, capture_output=True, text=True)
        runner.write_text(f"globalThis.__mpcCalls = [];\nconst m = await import({json.dumps(bundle.as_uri())});\n"
                          + script)
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout.strip().splitlines()[-1])


def _q(logit: float) -> int:
    """clientMPC.ts's quantisation: clamp(round((logit + 8) * 64), 0, 1024), with JS rounding."""
    return min(1024, max(0, math.floor((logit + 8) * 64 + 0.5)))


def test_the_argmax_shares_are_uniform_mod_2_32_and_fresh() -> None:
    logits = [0.3, -0.2, 1.7, 1.69]
    res = _run("secureArgmax", "clientMPC", f"""
const logits = {json.dumps(logits)};
const first = await m.secureArgmax(logits);
await m.secureArgmax(logits, () => 0.5);   // a generator passed anyway must not reach the masks
for (let i = 0; i < 16; i++) await m.secureArgmax(logits);
console.log(JSON.stringify({{ first, calls: globalThis.__mpcCalls }}));
""")
    q = [_q(x) for x in logits]
    calls = res["calls"]
    assert len(calls) == 18
    assert res["first"]["out"] == [q.index(max(q))], "the argmax index came back wrong"
    for c in calls:
        assert (c["aType"], c["bType"]) == ("Uint32Array", "Uint32Array"), (
            "the shares must reach the module as Uint32Array, the type its sharing mod 2^32 is defined on")
        assert [(a + b) % 2**32 for a, b in zip(c["a"], c["b"])] == q, (
            "a_k + b_k is not q_k mod 2^32: the shares do not reconstruct the quantised logits")
    masks = [x for c in calls for x in c["a"]]
    assert calls[0]["a"] != calls[1]["a"], "two calls drew the same masks: they are not fresh"
    assert len(set(masks)) > 60, f"only {len(set(masks))} distinct masks in 72 draws"
    assert max(masks) >= 2**31 and min(masks) < 2**31, "the masks do not span the 32-bit range"
    assert sum(x >= 2**16 for x in masks) > 60, "the masks look bounded to 2^16, as before Round 66"


def test_random_below_is_in_range_uniform_and_refuses_bad_moduli() -> None:
    res = _run("randomBelow, randomWords", "csprng", """
const out = { ranges: {} };
for (const p of [1, 2, 3, 101, 2147483647, 4294967296]) {
  const xs = Array.from({ length: 3000 }, () => m.randomBelow(p));
  out.ranges[p] = { min: Math.min(...xs), max: Math.max(...xs), ints: xs.every(Number.isInteger) };
}
const c = [0, 0, 0];
for (let i = 0; i < 30000; i++) c[m.randomBelow(3)]++;
out.counts3 = c;
out.refused = [0, -1, 1.5, 4294967297, NaN, Infinity].map((p) => {
  try { m.randomBelow(p); return false; } catch (e) { return e instanceof RangeError; }
});
const w = m.randomWords(20000);
out.words = { len: w.length, type: w.constructor.name, tailFilled: w.subarray(16384).some((x) => x !== 0),
  max: Math.max(...w) };
console.log(JSON.stringify(out));
""")
    for p, r in res["ranges"].items():
        assert r["ints"] and 0 <= r["min"] and r["max"] < int(p), f"randomBelow({p}) left [0, {p})"
    assert res["ranges"]["1"]["max"] == 0
    assert res["ranges"]["4294967296"]["max"] >= 2**31, "randomBelow(2^32) does not reach the top half"
    # 30000 draws over 3 values: each count is 10000 with standard deviation about 82.
    assert all(abs(n - 10000) < 600 for n in res["counts3"]), f"randomBelow(3) is not uniform: {res['counts3']}"
    assert all(res["refused"]), f"randomBelow accepted a modulus outside [1, 2^32]: {res['refused']}"
    w = res["words"]
    assert (w["len"], w["type"]) == (20000, "Uint32Array")
    assert w["tailFilled"], "randomWords left the part past the first getRandomValues call unfilled"
    assert w["max"] >= 2**31


def test_random_below_rejects_rather_than_reducing_with_bias() -> None:
    src = (LIB / "csprng.ts").read_text()
    assert "crypto.getRandomValues" in src
    assert re.search(r"const limit = 2 \*\* 32 - \(2 \*\* 32 % p\);", src), "randomBelow lost its rejection bound"
    assert "if (w[0] < limit) return w[0] % p;" in src, "randomBelow no longer redraws above the bound"


def _code(line: str) -> str:
    return re.sub(r"(?<![:\"'`])//.*$", "", line)


def test_the_secure_sum_draws_masks_from_the_csprng_and_keeps_the_seed_stream() -> None:
    src = (LIB / "clientCompute.ts").read_text()
    assert 'import { randomBelow } from "./csprng";' in src
    assert "Math.floor(u() * p)" not in src, "the secure sum draws its masks from the seeded generator again"
    assert re.search(r"secrets\.map\(\(s\) => \{\s*u\(\);\s*const mask = randomBelow\(p\);", src), (
        "the secure sum must draw each mask with randomBelow(p) AND still advance the run's generator "
        "once per secret, or every seeded figure after it - the row private_inference encrypts - moves")


def test_client_mpc_takes_no_generator() -> None:
    src = "\n".join(_code(x) for x in (LIB / "clientMPC.ts").read_text().splitlines())
    assert 'import { randomWords } from "./csprng";' in src
    assert re.search(r"export async function secureArgmax\(logits: number\[\]\): Promise<MPCResult>", src), (
        "secureArgmax takes a generator again; its masks must not come from the caller's seeded stream")
    assert not re.search(r"\brnd\(|Math\.random\(|\brng\(", src)
    assert "BIG" not in src, "the bounded 2^16 mask is back"


def test_the_circuit_shares_mod_2_32_and_its_test_wraps() -> None:
    src = (REPO / "wasm" / "qsh-mpc" / "src" / "lib.rs").read_text()
    body = src.split("pub fn argmax_source", 1)[1].split("\n}\n", 1)[0]
    assert "a: [u32; {n}], b: [u32; {n}]" in body, "the circuit's inputs are not u32 shares"
    assert body.count("as u64) + (b[") == 2 and body.count(") as u32;") == 2, (
        "the circuit must add in 64 bits and cast back to 32 (Garble panics on a 32-bit overflow)")
    assert "65536" not in body and "BIG" not in body, "the circuit subtracts an offset again"
    tests = src.split("mod tests", 1)[1]
    assert "u32::MAX" in tests and "wrapping_sub" in tests, "the crate's test no longer exercises the wrap"


def test_no_other_share_or_mask_comes_from_a_seeded_generator() -> None:
    word = re.compile(r"\b(mask\w*|shares?|s0|s1|pad\w*|otp)\b", re.I)
    seeded = re.compile(r"(?<![\w.])(u|rnd|rand|rng\(\w*\))\(\)|Math\.random\(\)")
    found = {}
    for f in sorted((WEB / "src").rglob("*.ts*")):
        rel = str(f.relative_to(REPO))
        for line in f.read_text().splitlines():
            code = _code(line)
            if word.search(code) and seeded.search(code):
                stmt = next((k[1] for k in WAIVED if k[0] == rel and k[1] in code), code.strip())
                found[(rel, stmt)] = line.strip()
    unexpected = sorted(set(found) - set(WAIVED))
    assert not unexpected, (
        "share or mask material drawn from a seeded generator - use csprng.ts, or disclose it in place "
        f"and waive it here with the reason: {unexpected}")
    for (rel, stmt), disclosure in WAIVED.items():
        assert (rel, stmt) in found, f"waived statement no longer present, drop its waiver: {rel}: {stmt}"
        assert disclosure in (REPO / rel).read_text(), f"{rel}: the disclosure that justifies its waiver is gone"
    qsim = (LIB / "quantumSim.ts").read_text()
    assert QOTP_DISCLOSURE in qsim and "if (u() < 0.5) X(q); if (u() < 0.5) Z(q);" in qsim, (
        "the QOTP demo key changed or lost its Round 40 disclosure; re-decide it rather than let it pass")
