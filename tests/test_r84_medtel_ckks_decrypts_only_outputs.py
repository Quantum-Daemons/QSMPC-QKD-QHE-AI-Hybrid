"""The browser's CKKS client decrypts each layer's OUTPUTS, bias included, and nothing else.

WHY THIS EXISTS (Round 84, scratchpad audit83 page-medical-telecom.md row 6)

Every CKKS path in `web/src/lib/clientFHE.ts` (the single-row linear layer, the held-out batch and the
convolution) decrypted the whole rotate-and-add result and added the bias in plaintext afterwards. So
the bias was never part of the encrypted computation, although the published student's biases are not
zero, and the decrypted vector held the reduction's running partial sums in every other slot: slot k
minus slot k + 1 is the k-th product w_k x_k, and a key holder who knows x reads the weights off one
query. The medical student also spent 136 rotations on two dense layers it could evaluate in 12.

WHAT IS ASSERTED, BY RUNNING THE BROWSER'S OWN MODULE AGAINST THE REAL node-seal

`clientFHE.ts` is bundled with esbuild and run in node, with node-seal wrapped so that every vector the
client DECODES after a decryption is recorded: the key holder's whole real-valued view.
  - Each path decrypts W.x + b at its output slots: the bias is already in what was decrypted.
  - Every other slot is flooded: uniform-looking values of the documented half-width, uncorrelated with
    the partial sums, including after dividing out the mask's own (public) encoding error.
  - The published student still gives the plaintext reference's logits and class, and its dense layers
    take 12 rotations (8 + 4), its whole network 28.
  - Why the flood is there at all (evidence for the comment on `CKKS_FLOOD_BITS`, not a regression
    test): with node-seal itself, a 0/1 mask WITHOUT the flood leaves the partial sums recoverable by
    dividing out the mask's encoding error; and with TenSEAL's complex decoder (the same SEAL encoder),
    that error is real, so the imaginary parts of the masked slots carry nothing.

On the baseline the first group fails at its first assertion: slot 0 holds W.x without the bias and the
other slots hold partial sums of order 1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: `CKKS_FLOOD_BITS` in clientFHE.ts: the half-width of the flood, as a power of two.
FLOOD_BITS = 16

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const REPO = resolve(WEB, "..");
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const realSeal = resolve(WEB, "node_modules/node-seal/dist/index_throws.js");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "spy", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
    // node-seal, wrapped: every plaintext a Decryptor writes is marked, and every decode of a marked
    // plaintext is recorded - all 4,096 real parts, which is what the key holder can read.
    b.onResolve({ filter: /^node-seal\/throws$/ }, () => ({ path: "spyseal", namespace: "spyseal" }));
    b.onLoad({ filter: /.*/, namespace: "spyseal" }, () => ({ resolveDir: WEB, loader: "js", contents: `
      import real from ${JSON.stringify(realSeal)};
      export default async function (opts) {
        const seal = await real(opts);
        const marked = new WeakSet();
        globalThis.__views = [];
        globalThis.__seal = seal;
        const Ctx = seal.SEALContext;
        seal.SEALContext = function (...a) { const c = new Ctx(...a); globalThis.__ctx = c; return c; };
        const dec = seal.Decryptor.prototype.decrypt;
        seal.Decryptor.prototype.decrypt = function (ct, pt) { const r = dec.call(this, ct, pt); marked.add(pt); return r; };
        const decode = seal.CKKSEncoder.prototype.decodeFloat64;
        seal.CKKSEncoder.prototype.decodeFloat64 = function (pt) {
          const v = decode.call(this, pt);
          if (marked.has(pt)) globalThis.__views.push(Array.from(v));
          return v;
        };
        return seal;
      }` }));
  } }],
});
const m = await import(`file://${OUT}`);
await m.warmFHE();
const seal = globalThis.__seal, ctx = globalThis.__ctx, slots = 4096, scale = 2 ** 40;
const enc = new seal.CKKSEncoder(ctx);
// The mask's own encoding error, which the key holder can compute exactly by encoding the same mask.
const maskEps = (stride, count) => {
  const v = new Float64Array(slots); for (let i = 0; i < count; i++) v[i * stride] = 1;
  const pt = new seal.Plaintext(); enc.encode(v, scale, pt);
  return Array.from(enc.decodeFloat64(pt), (d, i) => d - v[i]);
};
const take = () => { const v = globalThis.__views; globalThis.__views = []; return v; };
// Seeded inputs (mulberry32), so the run is reproducible.
let s = 84; const u = () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const out = { flood: m.CKKS_FLOOD_BITS, cases: {} };

// 1. The single-row linear layer, 30 inputs (block 32), one class, over 40 queries: one query leaves
//    only 62 slots holding a partial sum, too few for a correlation to mean anything.
{
  const n = 30, w = Array.from({ length: n }, () => (u() * 2 - 1) * 0.4);
  const xs = [], logits = [];
  for (let q = 0; q < 40; q++) {
    const x = Array.from({ length: n }, () => u() * 4 - 2);
    xs.push(x);
    logits.push((await m.encryptedLinear([w], [0.123], x)).logits[0]);
  }
  out.cases.linear = { inputs: xs, W: [w], b: [0.123], stride: 32, count: 1, block: 32, views: take(),
                       logits, eps: maskEps(32, 1) };
}
// 2. The held-out batch path: 8 features (block 8), 512 rows, which fill one ciphertext.
{
  const n = 8, rows = Array.from({ length: 512 }, () => Array.from({ length: n }, () => u() * 4 - 2));
  const w = Array.from({ length: n }, () => (u() * 2 - 1) * 0.5);
  const r = await m.encryptedLinearBatch([w], [-0.2], rows);
  out.cases.batch = { rows, W: [w], b: [-0.2], stride: 8, count: rows.length, block: 8, views: take(),
                      logits: r.logits.map((l) => l[0]), eps: maskEps(8, rows.length) };
}
// 3. The convolution: one 3x3 kernel, padding 1, over 14x14 (196 patches in blocks of 16).
{
  const img = Array.from({ length: 14 }, () => Array.from({ length: 14 }, () => u() * 2 - 1));
  const k = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => (u() * 2 - 1) * 0.5));
  const r = await m.encryptedConv2d(img, [k], [0.152], 1);
  out.cases.conv = { img, kernel: k, b: [0.152], stride: 16, count: 196, block: 16, views: take(),
                     logits: r.maps[0], eps: maskEps(16, 196), maxAbsError: r.maxAbsError };
}
// 4. The published student, on the bundled WDBC row tiled into its 14x14 input, as the page does.
{
  const w = JSON.parse(readFileSync(resolve(REPO, "models/medical_breastmnist_student/weights.json"), "utf8"));
  const ds = JSON.parse(readFileSync(resolve(WEB, "public/data/medical/breast_cancer_sample.json"), "utf8"));
  const res = [];
  for (const row of ds.heldout.samples.slice(0, 5)) {
    const f = row.slice(0, 8);
    const img = Array.from({ length: 14 }, (_, y) => Array.from({ length: 14 }, (_, x) => f[(y * 14 + x) % 8]));
    const r = await m.encryptedStudentInference(w, img);
    res.push({ logits: r.logits, err: r.logitMaxAbsError, prediction: r.prediction, rotations: r.rotations,
               denseRotations: r.denseRotations, totalRotations: r.totalRotations, views: take().length });
  }
  out.student = res;
}
// 5. Evidence for the flood (node-seal itself, outside clientFHE): the mask WITHOUT it.
{
  const pk = (() => { const kg = new seal.KeyGenerator(ctx); return { kg, pk: kg.createPublicKey(), sk: kg.secretKey() }; })();
  const encr = new seal.Encryptor(ctx, pk.pk), decr = new seal.Decryptor(ctx, pk.sk), ev = new seal.Evaluator(ctx);
  const gal = pk.kg.createGaloisKeysWithSteps(Int32Array.from([1, 2, 4, 8]));
  const x = new Float64Array(slots), wt = new Float64Array(slots);
  const kern = Array.from({ length: 9 }, () => (u() * 2 - 1) * 0.4);
  for (let p = 0; p < 196; p++) for (let t = 0; t < 9; t++) { x[p * 16 + t] = u() * 4 - 2; wt[p * 16 + t] = kern[t]; }
  const ptx = new seal.Plaintext(); enc.encode(x, scale, ptx); const ct = new seal.Ciphertext(); encr.encrypt(ptx, ct);
  const ptw = new seal.Plaintext(); enc.encode(wt, scale, ptw);
  let acc = new seal.Ciphertext(); ev.multiplyPlain(ct, ptw, acc);
  for (let st = 1; st < 16; st <<= 1) { const r = new seal.Ciphertext(); ev.rotateVector(acc, st, gal, r); const nx = new seal.Ciphertext(); ev.add(acc, r, nx); acc = nx; }
  const mv = new Float64Array(slots); for (let p = 0; p < 196; p++) mv[p * 16] = 1;
  const ptm = new seal.Plaintext(); enc.encode(mv, scale, ptm); ev.multiplyPlainInplace(acc, ptm);
  const dpt = new seal.Plaintext(); decr.decrypt(acc, dpt);
  out.maskOnly = { x: Array.from(x), w: Array.from(wt), view: Array.from(enc.decodeFloat64(dpt)), eps: maskEps(16, 196) };
  take();
}
console.log(JSON.stringify(out));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the browser's CKKS paths cannot run. Run `npm ci` in "
           "web/. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("medtel_ckks")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, "the browser's CKKS paths did not run:\n" + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _partial_sums(products: np.ndarray, block: int) -> np.ndarray:
    """What slot k holds after the rotate-and-add: the sum of `block` products from k, cyclically."""
    n = products.size
    ext = np.concatenate([products, products[:block]])
    c = np.concatenate([[0.0], np.cumsum(ext)])
    return c[np.arange(n) + block] - c[np.arange(n)]


def _products(case: dict) -> list[np.ndarray]:
    """The element-wise products each recorded view's rotate-and-add summed, in slot order."""
    slots = len(case["views"][0])
    if "kernel" in case:
        img = np.pad(np.asarray(case["img"]), 1)
        k = np.asarray(case["kernel"]).ravel()
        p = np.zeros(slots)
        for oy in range(14):
            for ox in range(14):
                patch = img[oy:oy + 3, ox:ox + 3].ravel()
                p[(oy * 14 + ox) * 16:(oy * 14 + ox) * 16 + 9] = patch * k
        return [p]
    w = np.asarray(case["W"][0])
    out = []
    for rows in ([case["rows"]] if "rows" in case else [[x] for x in case["inputs"]]):
        p = np.zeros(slots)
        for r, row in enumerate(rows):
            p[r * case["block"]:r * case["block"] + len(w)] = np.asarray(row) * w
        out.append(p)
    return out


def _corr_bound(n: int) -> float:
    """Six standard deviations of a sample correlation under independence: about 1e-9 to exceed."""
    return 6.0 / np.sqrt(n)


@pytest.mark.parametrize("name", ["linear", "batch", "conv"])
def test_the_bias_is_inside_what_the_client_decrypts(run: dict, name: str) -> None:
    case = run["cases"][name]
    idx = np.arange(case["count"]) * case["stride"]
    outs, expected = [], []
    for view, products in zip(case["views"], _products(case)):
        outs.append(np.asarray(view)[idx])
        expected.append(_partial_sums(products, case["block"])[idx] + case["b"][0])
    outs, expected = np.concatenate(outs), np.concatenate(expected)
    # The decrypted output slots ARE W.x + b: on the baseline they were W.x, the bias added afterwards.
    assert np.max(np.abs(outs - expected)) < 1e-6, (
        f"{name}: the decrypted outputs are not W.x + b (max deviation "
        f"{np.max(np.abs(outs - expected)):.3e}); the bias was added after decryption")
    assert np.max(np.abs(np.asarray(case["logits"]) - expected)) < 1e-6


@pytest.mark.parametrize("name", ["linear", "batch", "conv"])
def test_every_other_slot_is_flooded_not_a_partial_sum(run: dict, name: str) -> None:
    case = run["cases"][name]
    # The documented width when the module exports none, so a baseline fails on the view itself.
    bits = run.get("flood", FLOOD_BITS)
    eps = np.asarray(case["eps"])
    slots = len(case["views"][0])
    outputs = set((np.arange(case["count"]) * case["stride"]).tolist())
    other = np.array([k for k in range(slots) if k not in outputs])
    vals, live_v, live_s, div_v, div_s, residual = [], [], [], [], [], 0.0
    for view, products in zip(case["views"], _products(case)):
        view = np.asarray(view)
        sums = _partial_sums(products, case["block"])
        vals.append(view[other])
        live = other[np.abs(sums[other]) > 1e-9]
        live_v.append(view[live]); live_s.append(sums[live])
        ok = live[np.abs(eps[live]) > 0]
        div_v.append(view[ok] / eps[ok]); div_s.append(sums[ok])
        residual = max(residual, float(np.max(np.abs(eps[other] * sums[other]))))
    vals = np.concatenate(vals)
    # Uniform on [-2^b, 2^b): median magnitude 2^(b-1), nothing beyond 2^b.
    assert np.median(np.abs(vals)) > 2.0 ** (bits - 3), (
        f"{name}: the non-output slots are not flooded (median magnitude {np.median(np.abs(vals)):.3e}); "
        "they hold the rotate-and-add's partial sums")
    assert np.max(np.abs(vals)) <= 2.0 ** bits * (1 + 1e-9)
    # Neither the slots nor the slots divided by the mask's public encoding error track the sums. The
    # unmasked baseline gives a correlation near 1; independent values stay within six sigmas of 0.
    lv, ls = np.concatenate(live_v), np.concatenate(live_s)
    assert lv.size >= 2000, f"{name}: only {lv.size} slots hold a partial sum, too few to test"
    assert abs(np.corrcoef(lv, ls)[0, 1]) < _corr_bound(lv.size), f"{name}: the masked slots track the partial sums"
    dv, ds = np.concatenate(div_v), np.concatenate(div_s)
    assert abs(np.corrcoef(dv, ds)[0, 1]) < _corr_bound(dv.size), (
        f"{name}: dividing out the mask's encoding error recovers the partial sums")
    # The residual the flood covers, as `CKKS_FLOOD_BITS` documents it: below 1e-9 on these layouts.
    assert residual < 1e-9, f"{name}: the mask's residual reaches {residual:.3e}"
    assert run.get("flood") == FLOOD_BITS, "clientFHE.ts no longer exports the flood width it documents"


def test_the_published_student_keeps_its_answer_and_packs_its_dense_layers(run: dict) -> None:
    for r in run["student"]:
        assert r["err"] < 1e-6, f"the encrypted student's logits deviate by {r['err']:.3e}"
        assert r["prediction"] == int(np.argmax(r["logits"]))
        # 16 for the convolution (4 per channel), 8 + 4 for the two packed dense layers.
        assert (r["rotations"], r["denseRotations"], r["totalRotations"]) == (16, 12, 28), (
            f"rotations conv/dense/total {r['rotations']}/{r['denseRotations']}/{r['totalRotations']}; "
            "row at a time the dense layers took 136")
        # One decryption per conv channel and one per dense layer, each a masked output view.
        assert r["views"] == 4 + 2


def test_without_the_flood_the_mask_leaves_the_partial_sums_recoverable(run: dict) -> None:
    """Evidence for the flood (node-seal itself): a 0/1 mask alone is not enough."""
    mo = run["maskOnly"]
    view, eps = np.asarray(mo["view"]), np.asarray(mo["eps"])
    sums = _partial_sums(np.asarray(mo["x"]) * np.asarray(mo["w"]), 16)
    outs = set(range(0, 196 * 16, 16))
    idx = np.array([k for k in range(view.size - 16) if k not in outs and abs(sums[k]) > 1e-6 and eps[k] != 0])
    assert np.median(np.abs(view[idx])) < 1e-9, "the 0/1 mask did not zero the slots to the eye"
    c = np.corrcoef(view[idx] / eps[idx], sums[idx])[0, 1]
    assert c > 0.999, f"dividing out the mask's encoding error recovers the sums only to correlation {c:.4f}"


def test_the_masks_encoding_error_is_real_so_the_imaginary_parts_carry_nothing() -> None:
    """Evidence for flooding the real parts only. node-seal decodes no imaginary part; TenSEAL does."""
    pytest.importorskip("tenseal")
    import tenseal.sealapi as sa

    n = 8192
    parms = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(n)
    parms.set_coeff_modulus(sa.CoeffModulus.Create(n, [60, 40, 40, 60]))
    ctx = sa.SEALContext(parms, True, sa.SEC_LEVEL_TYPE.TC128)
    kg = sa.KeyGenerator(ctx)
    pk = sa.PublicKey()
    kg.create_public_key(pk)
    enc, encr = sa.CKKSEncoder(ctx), sa.Encryptor(ctx, pk)
    dec, ev = sa.Decryptor(ctx, kg.secret_key()), sa.Evaluator(ctx)
    slots, scale = n // 2, 2.0 ** 40
    rng = np.random.default_rng(84)
    for stride, count in ((16, 196), (256, 16), (8, 512), (32, 1)):
        m = np.zeros(slots)
        m[np.arange(count) * stride] = 1.0
        pm = sa.Plaintext()
        enc.encode(m.tolist(), scale, pm)
        eps = np.asarray(enc.decode_complex(pm)) - m
        assert np.max(np.abs(eps.imag)) < 1e-15 < np.median(np.abs(eps.real)), (stride, count)
        # A ciphertext of arbitrary real content, times the mask: the masked slots' imaginary parts.
        v = rng.normal(size=slots)
        pv = sa.Plaintext()
        enc.encode(v.tolist(), scale, pv)
        ct = sa.Ciphertext()
        encr.encrypt(pv, ct)
        ev.multiply_plain_inplace(ct, pm)
        d = sa.Plaintext()
        dec.decrypt(ct, d)
        view = np.asarray(enc.decode_complex(d))
        other = np.setdiff1d(np.arange(slots), np.arange(count) * stride)
        assert np.max(np.abs(view.imag[other])) < 1e-14, (stride, count, np.max(np.abs(view.imag[other])))
