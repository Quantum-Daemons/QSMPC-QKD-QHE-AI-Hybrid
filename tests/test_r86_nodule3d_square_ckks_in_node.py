"""The 3D nodule student's encrypted pass under the page's real CKKS, in node (Round 86, batch 4).

`web/src/lib/clientFHESquare3d.ts` runs a 28 x 28 x 28 crop end to end under CKKS with no layer decrypted in between,
under the two parameter sets `references/medical-nodule3d-square-student-preregistration.md` names. This bundles it
with esbuild, as tests/test_r86_satellite_student_ckks_in_node.py bundles the forecaster, runs it against the REAL
node-seal, and checks it against a NumPy reference of the network written here, independently of the TypeScript:

- the TypeScript's float64 logits are the reference's (the taps, the folded bias, the square, the pooling folded into
  the dense weights, and the dense layer's layout, all as PyTorch's Conv3d, AvgPool3d and Flatten lay them out);
- each decrypted logit is within the pre-registered $`10^{-3}`$ of the plaintext one, and the decision is the same;
- the input is 27 taps in 6 blocks under A2 and 3 under B, and the output two ciphertexts;
- the contribution map sums to the logit difference minus the dense biases' difference.

The weights are drawn here from a stated seed, at about the size a trained network's are, with a kernel that is not
symmetric, so a tap or an axis taken in the wrong order changes the logits. The trained student's own crops are
checked by the test that comes with it. About ten seconds of CKKS, and about 0.6 GB.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
SIDE, CH = 28, 4
SETS = {"A2": {"N": 8192, "coeff_mod_bits": [48, 34, 34, 34, 52], "scale_bits": 34},
        "B": {"N": 16384, "coeff_mod_bits": [60, 40, 40, 40, 60], "scale_bits": 40}}

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], DIR = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare3d"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  // The one Vite-only specifier: the SEAL wasm's `?url` import becomes its path on disk.
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(DIR + "/weights.json", "utf8"));
const v = JSON.parse(readFileSync(DIR + "/crop.json", "utf8"));
const runs = {};
for (const set of Object.keys(w.ckks)) runs[set] = await m.runSquare3d(v, w, set);
console.log(JSON.stringify({ plain: m.plainSquare3dLogits(v, w), sum: m.contributionSum(v, w), runs,
  voxels: m.VOXELS, side: m.SIDE3 }));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                                     ("web/node_modules/node-seal", (WEB / "node_modules" / "node-seal").exists()))
               if not ok]
    if not missing:
        return
    msg = f"{', '.join(missing)} absent, so the 3D student's CKKS path cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def _weights_and_crop(seed: int = 20261124) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    # A nodule-like blob in a noisy field, as a crop reads after (u / 255 - 0.5) / 0.5: values in [-1, 1].
    z, y, x = np.meshgrid(*(np.arange(SIDE) - 13.2,) * 3, indexing="ij")
    blob = np.exp(-(z ** 2 / 30 + y ** 2 / 18 + x ** 2 / 24))
    crop = np.clip(-0.8 + 1.6 * blob + 0.05 * rng.standard_normal((SIDE,) * 3), -1, 1)
    conv = rng.normal(0, 0.25, (CH, 1, 3, 3, 3))
    conv[:, 0, 0, 0, 0] += 0.6          # not symmetric under any flip of an axis
    dense = rng.normal(0, 0.02, (2, CH * 14 ** 3))
    w = {"kind": "cnn3d-1conv-square-1dense",
         "conv": {"in_ch": 1, "out_ch": CH, "kernel": 3, "padding": 1, "activation": "square", "batch_norm": "folded",
                  "weight": conv.tolist(), "bias": rng.normal(0, 0.1, CH).tolist()},
         "pool": {"type": "avg", "stride": 2},
         "dense": [{"in": CH * 14 ** 3, "out": 2, "weight": dense.tolist(), "bias": [0.3, -0.2], "activation": "none"}],
         "classes": ["benign", "malignant"], "ckks": SETS}
    return w, crop


def _reference_logits(w: dict, crop: np.ndarray) -> np.ndarray:
    """Conv3d (cross-correlation, zero padding 1), bias, square, AvgPool3d(2), Flatten over (C, D, H, W), Linear."""
    k = np.asarray(w["conv"]["weight"])[:, 0]
    padded = np.pad(crop, 1)
    out = np.zeros((CH, SIDE, SIDE, SIDE))
    for c in range(CH):
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    out[c] += k[c, dz, dy, dx] * padded[dz:dz + SIDE, dy:dy + SIDE, dx:dx + SIDE]
        out[c] += w["conv"]["bias"][c]
    pooled = (out ** 2).reshape(CH, 14, 2, 14, 2, 14, 2).mean(axis=(2, 4, 6))
    d = w["dense"][0]
    return np.asarray(d["weight"]) @ pooled.reshape(-1) + np.asarray(d["bias"])


def test_the_encrypted_3d_pass_gives_the_networks_logits_under_both_sets() -> None:
    _require_node_or_skip()
    w, crop = _weights_and_crop()
    ref = _reference_logits(w, crop)
    # A logit difference of a few units: large enough that the decision is not a coin toss for CKKS noise.
    assert 0.5 < abs(ref[1] - ref[0]) < 50, ref
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "weights.json").write_text(json.dumps(w))
        (d / "crop.json").write_text(json.dumps(crop.reshape(-1).tolist()))
        (d / "harness.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(WEB), str(d / "bundle.mjs"), str(d)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=600)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert (r["side"], r["voxels"]) == (SIDE, SIDE ** 3)
    np.testing.assert_allclose(r["plain"], ref, rtol=1e-12, atol=1e-9)
    total, expected = r["sum"]["total"], r["sum"]["expected"]
    assert abs(total - expected) < 1e-9 * max(1.0, abs(expected)), (total, expected)
    assert set(r["runs"]) == set(SETS)
    for name, run in r["runs"].items():
        blocks = -(-SIDE ** 3 // (SETS[name]["N"] // 2))
        assert (name, blocks) in {("A2", 6), ("B", 3)}
        assert run["ciphertexts"] == {"input": 27 * blocks, "output": 2}, (name, run["ciphertexts"])
        np.testing.assert_allclose(run["plain"], ref, rtol=1e-12, atol=1e-9)
        assert run["maxAbsError"] <= 1e-3, (name, run["maxAbsError"])
        assert run["decisionAgrees"] and run["predicted"] == int(ref[1] > ref[0]), (name, run)
        assert run["bytes"]["input"] > 0 and run["bytes"]["output"] > 0
