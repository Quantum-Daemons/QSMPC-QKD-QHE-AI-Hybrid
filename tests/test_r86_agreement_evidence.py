"""The evidence under the private-inference scatter computes what its caption says (Round 86).

`web/src/lib/agreementStats.ts` gives the Bland-Altman bias and limits of agreement (Bland and Altman, Lancet 1986) and
each pair's precision in bits as Lattigo's `ckks.GetPrecisionStats` defines it, -log2 of the absolute error. Held here
against numpy on pairs built so that each figure is known: the bias and the sample standard deviation (n - 1), the limits
at 1.96 standard deviations, the count within them, the bins of whole bits, the median and the lowest precision, and the
exact pairs left out of the bins. The page draws them under every encrypted-agreement scatter.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-evidence-"));
const out = join(dir, "bundle.mjs");
await build({ entryPoints: [resolve(WEB, "src/lib/agreementStats.ts")], bundle: true, format: "esm", platform: "node",
              outfile: out, logLevel: "error" });
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const cases = JSON.parse(readFileSync(process.argv[3], "utf8"));
// JSON has no NaN, so a missing number arrives as null and is made NaN here.
console.log(JSON.stringify(cases.map((c) => m.agreementStats(c.map((p) => ({ enc: p.enc ?? NaN, plain: p.plain ?? NaN }))))));
"""


@pytest.fixture(scope="module")
def stats(tmp_path_factory):
    if not ESBUILD.exists() or shutil.which("node") is None:
        pytest.skip("node or esbuild absent, so the module cannot run here. This is a SKIP, not a pass.")
    d = tmp_path_factory.mktemp("r86evidence")
    rng = np.random.default_rng(7)
    plain = rng.normal(0.0, 2.0, 500)
    enc = plain + rng.normal(3e-9, 2e-9, 500)
    exact = [{"enc": 1.0, "plain": 1.0}, {"enc": -2.5, "plain": -2.5}] + [{"enc": 0.5 + 2.0 ** -20, "plain": 0.5}]
    cases = [[{"enc": float(e), "plain": float(p)} for e, p in zip(enc, plain, strict=True)], exact,
             [{"enc": 1.0, "plain": 2.0}], [], [{"enc": None, "plain": 1.0}, {"enc": 3.0, "plain": 1.0}]]
    (d / "cases.json").write_text(json.dumps(cases, allow_nan=False))
    (d / "h.mjs").write_text(HARNESS)
    r = subprocess.run(["node", str(d / "h.mjs"), str(WEB), str(d / "cases.json")], capture_output=True, text=True,
                       timeout=120, cwd=WEB)
    assert r.returncode == 0, r.stderr
    return cases, json.loads(r.stdout)


def test_bias_limits_and_count_within_match_numpy(stats) -> None:
    cases, out = stats
    pairs, s = cases[0], out[0]
    d = np.array([p["enc"] - p["plain"] for p in pairs])
    assert s["n"] == len(d)
    assert math.isclose(s["bias"], d.mean(), rel_tol=1e-9)
    assert math.isclose(s["sd"], d.std(ddof=1), rel_tol=1e-9)
    lo, hi = d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1)
    assert math.isclose(s["limits"][0], lo, rel_tol=1e-9) and math.isclose(s["limits"][1], hi, rel_tol=1e-9)
    assert s["within"] == int(((d >= lo) & (d <= hi)).sum())
    assert 0.9 < s["within"] / s["n"] < 0.99, "1.96 SD should hold about 95% of normal differences"
    mean = np.array([(p["enc"] + p["plain"]) / 2 for p in pairs])
    assert np.allclose([q["x"] for q in s["points"]], mean) and np.allclose([q["y"] for q in s["points"]], d)


def test_precision_bits_are_lattigo_s_minus_log2_of_the_absolute_error(stats) -> None:
    cases, out = stats
    pairs, s = cases[0], out[0]
    bits = -np.log2(np.abs(np.array([p["enc"] - p["plain"] for p in pairs])))
    want = {int(k): int(c) for k, c in zip(*np.unique(np.floor(bits), return_counts=True), strict=True)}
    assert {b["bits"]: b["count"] for b in s["bits"]} == want
    assert [b["bits"] for b in s["bits"]] == sorted(want), "the bins are in ascending order"
    assert math.isclose(s["medianBits"], float(np.median(bits)), rel_tol=1e-12)
    assert math.isclose(s["minBits"], float(bits.min()), rel_tol=1e-12)


def test_exact_pairs_are_counted_apart_and_never_binned(stats) -> None:
    _, out = stats
    s = out[1]
    assert s["exact"] == 2 and s["n"] == 3
    assert s["bits"] == [{"bits": 20, "count": 1}] and s["medianBits"] == 20 and s["minBits"] == 20


def test_degenerate_inputs(stats) -> None:
    _, out = stats
    assert out[2]["n"] == 1 and out[2]["sd"] == 0, "one pair has no spread"
    assert out[3] is None, "no pairs, no statistics"
    assert out[4]["n"] == 1 and out[4]["bias"] == 2, "a pair that is not a number is left out"


def test_the_scatter_draws_the_evidence_under_it() -> None:
    src = (WEB / "src" / "components" / "VerificationChart.tsx").read_text()
    chart = src[src.index("export function EncryptedAgreementChart"):]
    chart = chart[:chart.index("\n}\n")]
    assert "<AgreementEvidence plot={plot} />" in chart
    assert "Bland and Altman" in src and "GetPrecisionStats" in src
