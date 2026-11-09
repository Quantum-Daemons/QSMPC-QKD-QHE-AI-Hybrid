"""The confirmed findings of the Round 86 private-inference review, fixed at their source and held.

WHY THIS EXISTS (Round 86, by the owner's choice to fix every confirmed finding)

A review of both private-inference tiers, run from a scratchpad against 85e12a7, confirmed:

- PI-1: the full deployment's reported class came from an UNMASKED decryption. The linear model's
  logits were read from `backend.decrypt(dot)` whole, bias added in the clear, so slots 1.. held the
  rotate-and-sum's partial sums: slot k minus slot k + 1 is x_k * w_k, and a key holder who knows x
  reads the weight row off one query. The threat model said every slot the key holder decodes is
  masked, which was true of the hybrid's layers only. Round 84 fixed the same leak in `_fhe_linear`
  and the convolution; `encrypted_linear_inference` was the twin it did not reach.
- PI-3: a NaN feature encoded to a silently wrong logit that no precision check flagged (each compared
  with `>= 1`, which NaN fails), and +-Infinity trapped node-seal's WASM.
- PI-4: the garbled-circuit argmax quantised logits in steps of 1/64 over +-8, so a closer pair tied and
  went to the lower index, and its self-test shared the quantisation, so it could not see that.
- PI-5: the browser added its flood as a plaintext, so the decrypted ciphertext carried no fresh
  encryption randomness; the server encrypts its masks for that reason.
- PI-6, PI-10, PI-11, PI-12: texts and metrics that read better than what ran.
- PI-7, PI-8, PI-9: documents that described the flood's sizing, the ciphertext bytes and the reaction
  attack's setting more favourably than the code and the paper support.
- PI-15: FHE config fields nothing read; PI-17: a failed CKKS init cached for the life of the tab.

WHAT IS ASSERTED

- PI-1, functionally on TenSEAL: decoding every slot of everything the linear inference decrypts, with
  the key holder's own key, no linear combination of the other slots predicts any product x_k * w_k, and
  the attack of record (adjacent-slot differences) fails; the same detector recovers the products
  exactly from the unmasked reference path, so it is live. The masked path also runs on OpenFHE.
- Every CKKS decryption on the server is the masked one or one of three named exceptions, by AST.
- In node, on the repository's own clientFHE.ts and node-seal: a non-finite value is refused by name
  before it reaches the encoder, and a failed init is retried rather than cached.
- The rest at their source: the argmax's step and its float self-test, the encrypted flood, the labels,
  the four-place rates and their baseline, the docs' wording, the removed config fields.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "qsh"
WEB = REPO / "web"
LIB = WEB / "src" / "lib"
DOCS = REPO / "docs"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: Queries per attack, as in tests/test_r84_c1_slot_masking.py: under the null the sample correlation
#: has a standard deviation near 1/sqrt(T) = 0.06, so 0.3 sits five of those away from it, while the
#: unmasked decryption this replaces gives 1.000.
TRIALS = 240
CORR_BOUND = 0.3
R2_BOUND = 0.2


def _strip_ts(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def _read(path: Path) -> str:
    return path.read_text()


# --------------------------------------------------------------------------------------------------
# PI-1: the linear inference's decryptions show the logits and nothing else
# --------------------------------------------------------------------------------------------------


class _ClientDecode:
    """Wraps a TenSEAL backend; decodes, with the key holder's own key, every slot it is asked to decrypt."""

    def __init__(self, inner):
        import tenseal.sealapi as sa

        self.inner, self.name, self.views, self._sa = inner, inner.name, [], sa
        sc = inner.context.seal_context().data
        self._dec = sa.Decryptor(sc, inner.context.secret_key().data)
        self._enc = sa.CKKSEncoder(sc)

    def decrypt(self, ct):
        raw = ct.ciphertext()[0] if hasattr(ct, "ciphertext") else ct
        pt = self._sa.Plaintext()
        self._dec.decrypt(raw, pt)
        self.views.append(np.asarray(self._enc.decode_complex(pt)))
        return self.inner.decrypt(ct)

    def __getattr__(self, attr):
        return getattr(self.inner, attr)


def _held_out_r2(features: np.ndarray, target: np.ndarray) -> float:
    half = len(target) // 2
    a = np.column_stack([features, np.ones(len(target))])
    coef, *_ = np.linalg.lstsq(a[:half], target[:half], rcond=None)
    resid = target[half:] - a[half:] @ coef
    return 1.0 - float(resid @ resid) / float(((target[half:] - target[half:].mean()) ** 2).sum())


def _leaks(views: np.ndarray, xs: np.ndarray, w: np.ndarray) -> dict:
    """The two attacks on one output row's decodes: the best held-out R^2 and adjacent-slot correlation."""
    n = w.size
    idx = np.r_[1:n, views.shape[1] - n + 1:views.shape[1]]
    feats = np.concatenate([views[:, idx].real, views[:, idx].imag], axis=1)
    r2, corr = [], []
    for k in range(n):
        target = xs[:, k] * w[k]
        r2.append(_held_out_r2(feats, target))
        if k < n - 1:
            corr.append(abs(float(np.corrcoef((views[:, k] - views[:, k + 1]).real, target)[0, 1])))
    return {"r2": max(r2), "corr": max(corr)}


@pytest.fixture(scope="module")
def tenseal_backend():
    pytest.importorskip("tenseal")
    from qsh.fhe.backends import get_backend

    return get_backend("tenseal")


def test_the_linear_inference_decrypts_its_logits_and_nothing_else(tenseal_backend) -> None:
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference

    rng = np.random.default_rng(86)
    n_in, n_out = 8, 2
    W, b = rng.normal(size=(n_out, n_in)) * 0.4, rng.normal(size=n_out) * 0.1
    spy = _ClientDecode(tenseal_backend)
    xs = rng.normal(size=(TRIALS, n_in))
    for x in xs:
        out = encrypted_linear_inference(LinearModel(W, b), x, spy)
        # The logits survive the masks: slot 0 decrypts to W x + b, to CKKS precision.
        assert np.max(np.abs(np.asarray(out["logits"]) - (W @ x + b))) < 1e-4, out
    assert len(spy.views) == TRIALS * n_out, "each logit is one decryption"
    views = np.asarray(spy.views).reshape(TRIALS, n_out, -1)
    for j in range(n_out):
        got = _leaks(views[:, j], xs, W[j])
        assert got["r2"] < R2_BOUND, f"logit {j}: the other slots predict a product x_k * w_k ({got})"
        assert got["corr"] < CORR_BOUND, f"logit {j}: slot k - slot k+1 tracks x_k * w_k ({got})"
        # And every other slot reads at the masks' width, real and imaginary parts.
        other = views[:, j, 1:]
        assert np.median(np.abs(other.real)) > 2**20 and np.median(np.abs(other.imag)) > 2**20


def test_the_detector_finds_the_leak_in_the_unmasked_reference(tenseal_backend) -> None:
    """Without this, a detector that never fires would pass the test above."""
    from qsh.mpc.secret_inference import _fhe_linear

    rng = np.random.default_rng(87)
    w = rng.normal(size=8) * 0.4
    spy = _ClientDecode(tenseal_backend)
    xs = rng.normal(size=(40, 8))
    for x in xs:
        _fhe_linear(spy, x, w[None, :], np.zeros(1))      # mask=None: the documented plaintext fallback
    got = _leaks(np.asarray(spy.views), xs, w)
    assert got["corr"] > 0.99 and got["r2"] > 0.99, f"the unmasked decode no longer shows the products: {got}"


def test_the_masked_linear_inference_runs_on_openfhe() -> None:
    pytest.importorskip("openfhe")
    from qsh.fhe.backends import get_backend
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference

    backend = get_backend("openfhe")
    seen: list[np.ndarray] = []
    inner = backend.decrypt
    backend.decrypt = lambda ct: (seen.append(np.asarray(inner(ct))), seen[-1])[1]  # type: ignore[method-assign]
    rng = np.random.default_rng(88)
    W, b = rng.normal(size=(2, 8)) * 0.4, rng.normal(size=2) * 0.1
    x = rng.normal(size=8)
    out = encrypted_linear_inference(LinearModel(W, b), x, backend)
    assert np.max(np.abs(np.asarray(out["logits"]) - (W @ x + b))) < 1e-4
    assert len(seen) == 2 and all(np.median(np.abs(np.real(v[1:]))) > 2**20 for v in seen), (
        "OpenFHE's other slots are not masked")


#: The server's CKKS decryptions that are NOT the masked one, each with the reason it may stay.
UNMASKED_ALLOWED = {
    ("fhe/backends.py", "decrypt"): "the backend's own method",
    ("fhe/backends.py", "decrypt_slots"): "the backend's own method",
    ("fhe/masking.py", "masked_decrypt"): "the masked decryption itself",
    ("mpc/secret_inference.py", "_fhe_linear"): "mask=None: the plaintext fallback, client_learns says so",
    ("fhe/conv.py", "fhe_conv2d"): "mask=None: the same fallback for the convolution",
    ("qhe/qcnn_pauli.py", "under_ckks"): "the key holder reads its own readout, and a dot product's result has no partial sums to expose",
    ("usecases/genomics/__init__.py", "encrypted_scores"):
        "the panel check; its other slots hold products of the PUBLISHED PGS weights (docstring, threat model)",
}


def test_every_server_ckks_decryption_is_masked_or_named() -> None:
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("pqc/"):
            continue                                            # AES-GCM, not CKKS
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "decrypt"):
                    found.add((rel, fn.name))
    stray = found - set(UNMASKED_ALLOWED)
    assert not stray, (f"a CKKS decryption outside masked_decrypt: {sorted(stray)}. Route it through "
                       "qsh.fhe.masking.masked_decrypt, or name it in UNMASKED_ALLOWED with its reason")
    assert ("fhe/masking.py", "masked_decrypt") in found, "the scan no longer sees the masked decryption"
    linear = (SRC / "fhe" / "ckks_inference.py").read_text()
    assert "masked_decrypt(backend, enc_dot" in linear and "backend.decrypt(" not in linear


def test_the_stage_log_says_how_the_logits_were_decrypted() -> None:
    stages = (SRC / "orchestration" / "stages.py").read_text()
    assert "each logit decrypted with its bias added under" in stages
    assert "every other slot masked, so the key holder reads the logits and" in stages


# --------------------------------------------------------------------------------------------------
# The browser: PI-3 and PI-17 in node, on the repository's own clientFHE.ts and node-seal
# --------------------------------------------------------------------------------------------------

_BUILD = r"""
import { createRequire } from "node:module";
import { resolve } from "node:path";
const [WEB, OUT, ENTRY] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: ENTRY, resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    // node-seal's loader, failing on its FIRST call only: a transient wasm-load failure.
    b.onResolve({ filter: /^node-seal\/throws$/ }, (a) =>
      a.namespace === "flaky" ? undefined : ({ path: "flaky-seal", namespace: "flaky" }));
    b.onLoad({ filter: /.*/, namespace: "flaky" }, () => ({ resolveDir: WEB, loader: "js", contents: `
      import real from "node-seal/throws";
      let calls = 0;
      export default async function SEAL(opts) {
        if (calls++ === 0) throw new Error("transient wasm load failure (test)");
        return real(opts);
      }` }));
  } }],
});
"""


def _require_node_or_skip() -> None:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                               ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so clientFHE.ts cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def browser_fhe(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r86fhe")
    (d / "build.mjs").write_text(_BUILD)
    entry = f"export {{ encryptedLinear }} from {json.dumps(str(LIB / 'clientFHE'))};"
    subprocess.run(["node", str(d / "build.mjs"), str(WEB), str(d / "bundle.mjs"), entry], cwd=WEB,
                   check=True, capture_output=True, text=True, timeout=300)
    (d / "run.mjs").write_text(f"""
const m = await import({json.dumps((d / "bundle.mjs").as_uri())});
const W = [[0.3, -0.2, 0.5, 0.1], [-0.4, 0.25, 0.05, 0.2]], b = [0.1, -0.05];
const x = [0.7, -1.2, 0.4, 2.0];
const attempt = async (W, b, x) => {{
  try {{ const r = await m.encryptedLinear(W, b, x); return {{ ok: true, logits: r.logits, err: r.maxAbsError }}; }}
  catch (e) {{ return {{ ok: false, message: String(e && e.message || e) }}; }}
}};
const out = {{}};
out.first = await attempt(W, b, x);                    // the flaky loader's one failure
out.retried = await attempt(W, b, x);                  // the cached rejection, before the fix
out.nanFeature = await attempt(W, b, [0.7, NaN, 0.4, 2.0]);
out.infFeature = await attempt(W, b, [0.7, -1.2, Infinity, 2.0]);
out.nanWeight = await attempt([[0.3, NaN, 0.5, 0.1], W[1]], b, x);
out.after = await attempt(W, b, x);                    // the instance is still usable
console.log(JSON.stringify(out));
""")
    proc = subprocess.run(["node", str(d / "run.mjs")], cwd=WEB, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_failed_ckks_init_is_retried_not_cached(browser_fhe) -> None:
    assert not browser_fhe["first"]["ok"] and "transient" in browser_fhe["first"]["message"], browser_fhe["first"]
    assert browser_fhe["retried"]["ok"], (
        f"the second run still fails with the first one's error: the rejected init was cached "
        f"({browser_fhe['retried']})")
    W = np.array([[0.3, -0.2, 0.5, 0.1], [-0.4, 0.25, 0.05, 0.2]])
    plain = W @ np.array([0.7, -1.2, 0.4, 2.0]) + np.array([0.1, -0.05])
    assert np.allclose(browser_fhe["retried"]["logits"], plain, atol=1e-6)


@pytest.mark.parametrize("case", ["nanFeature", "infFeature", "nanWeight"])
def test_a_non_finite_value_is_refused_by_name(browser_fhe, case) -> None:
    r = browser_fhe[case]
    assert not r["ok"], f"{case}: encrypted and decrypted to {r.get('logits')} with error {r.get('err')}"
    assert "not a finite number" in r["message"], f"{case}: refused, but not by the check: {r['message']}"
    assert "memory access out of bounds" not in r["message"]
    assert browser_fhe["after"]["ok"], "a refused input left the CKKS instance unusable"


def test_every_encode_goes_through_the_finite_check() -> None:
    src = _strip_ts(_read(LIB / "clientFHE.ts"))
    body = re.search(r"function encodeFinite\([^)]*\)[^{]*\{(.*?)\n\}", src, flags=re.S)
    assert body, "encodeFinite moved"
    calls = re.findall(r"\bencoder\.encode\(", src)
    assert len(calls) == 1 and "encoder.encode(" in body.group(1), (
        f"{len(calls)} raw encoder.encode calls: every encode must go through encodeFinite")
    assert len(re.findall(r"\bencodeFinite\(", src)) >= 11, "an encode site stopped checking its values"
    compute = _strip_ts(_read(LIB / "clientCompute.ts"))
    assert not re.search(r"maxAbs(?:Error)?\s*>=\s*1\b", compute), "a precision check that NaN passes is back"
    assert "!(maxAbs < 1)" in compute and "!(real.maxAbsError < 1)" in compute


def test_a_rejected_init_resets_the_cache_in_the_source() -> None:
    src = _strip_ts(_read(LIB / "clientFHE.ts"))
    assert re.search(r"_ctx \?\?= init\(\)\.catch\(\(e\) => \{ _ctx = null; throw e; \}\)", src)


# --------------------------------------------------------------------------------------------------
# PI-4, PI-5: the argmax's step, its float self-test, and the encrypted flood
# --------------------------------------------------------------------------------------------------


def test_the_argmax_resolves_logits_a_sixty_fourth_apart() -> None:
    src = _strip_ts(_read(LIB / "clientMPC.ts"))
    assert re.search(r"const SCALE = 2 \*\* 20;", src) and re.search(r"const OFFSET = 512;", src)
    assert re.search(r"const QMAX = 2 \* OFFSET \* SCALE;", src)
    scale, offset = 2**20, 512
    assert 2 * offset * scale <= 2**31, "the quantised logits must stay below 2^31 to compare as u32"
    # The shipped logits reach |24.9| and the closest decided pair is 0.0071 apart (the review):
    # both are resolved now, and neither was at 1/64 over +-8.
    assert 1 / scale < 0.0071 and offset > 24.9
    compute = _strip_ts(_read(LIB / "clientCompute.ts"))
    assert "mpcFloatIndex: argmax(fhe.logits)" in compute
    assert 'metric(stage, "mpc_matches_float_argmax", floatAgrees ? 1 : 0)' in compute


def test_the_browser_flood_is_encrypted_before_it_is_added() -> None:
    src = _strip_ts(_read(LIB / "clientFHE.ts"))
    fn = re.search(r"function decryptOutputs\(.*?\n\}", src, flags=re.S)
    assert fn, "decryptOutputs moved"
    body = fn.group(0)
    assert "encryptor.encrypt(ptAdd, ctAdd)" in body and "evaluator.add(masked, ctAdd, flooded)" in body
    assert "decryptor.decrypt(flooded, out)" in body
    assert "addPlain" not in body, "the flood is added as a plaintext again"


# --------------------------------------------------------------------------------------------------
# PI-6, PI-10, PI-11, PI-12: what a page and a log say ran
# --------------------------------------------------------------------------------------------------


def test_the_genomics_log_no_longer_says_the_partial_sums_are_readable() -> None:
    compute = _read(LIB / "clientCompute.ts")
    assert "from which the key holder could read each weighted product" not in compute
    assert "every other slot is zeroed by the output mask and" in compute
    assert "flooded (+/-2^${CKKS_FLOOD_BITS}) before decryption" in compute


def test_the_medical_cnn_says_its_input_is_not_an_image() -> None:
    compute = _read(LIB / "clientCompute.ts")
    assert 'metric(stage, "fhe_student_input", "this page\'s tabular row tiled to fill the image; not an image")' in compute
    assert "the image, not an image, so the CNN's class is not a diagnosis." in compute


def test_the_two_party_demo_names_what_its_inference_phase_runs() -> None:
    compute = _read(LIB / "clientCompute.ts")
    alice_bob = (SRC / "benchmark" / "alice_bob_e2e.py").read_text()
    # The browser's two-party mirror, `clientAliceBob`, that carried the plaintext placeholder phase had no caller and was
    # removed in Round 86; the Python demo below is the one that runs.
    assert '"Private inference (ONNX)"' not in compute and "function clientAliceBob" not in compute
    assert '"Private inference (CKKS)"' not in alice_bob
    assert '"CKKS timing on a random model (single party)"' in alice_bob


def test_rates_print_to_four_places_beside_their_baseline() -> None:
    compute = _strip_ts(_read(LIB / "clientCompute.ts"))
    assert "const r4 = (x: number) => Math.round(x * 10000) / 10000;" in compute
    rates = 0
    for name in ("encrypted_agreement", "encrypted_agreement_over_decidable"):
        for m in re.finditer(rf'metric\(stage, "{name}",\s*(.*?)\);', compute, flags=re.S):
            assert "r2(" not in m.group(1), f"{name} is rounded to two places again: {m.group(0)}"
            if "/" in m.group(1):                        # a rate; the one-row path emits 0 or 1 exactly
                rates += 1
                assert "r4(" in m.group(1), m.group(0)
    assert rates == 4, f"{rates} agreement rates found, expected the batch's and the held-out's two each"
    assert 'metric(stage, "encrypted_accuracy_majority_baseline", r4(ho.majorityBaseline))' in compute
    fhe = _strip_ts(_read(LIB / "clientFHE.ts"))
    assert "majorityBaseline: number | null;" in fhe and "Math.max(...counts.values()) / labels.length" in fhe
    # And on the page: the multiclass chart prints its accuracy, so it prints the baseline beside it.
    assert "majority: majorityBaseline }" in fhe
    chart = _read(WEB / "src" / "components" / "VerificationChart.tsx")
    assert "always calling the most common class would score" in chart


# --------------------------------------------------------------------------------------------------
# PI-7, PI-8, PI-9, the SEAL residual risk, PI-15
# --------------------------------------------------------------------------------------------------


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_flood_is_described_as_fixed_with_its_margin_reported() -> None:
    tm = _flat(_read(DOCS / "threat-model.md"))
    masking = _flat((SRC / "fhe" / "masking.py").read_text())
    for text, where in ((tm, "threat-model.md"), (masking, "masking.py")):
        assert "sizes its flood from the error the run measures" not in text, where
        assert "so the flood is sized against the error the run measures" not in text, where
    assert "The server tier's flood is a fixed $`2^{30}`$ (`benchmark.mpc_mask_bits`)" in tm
    assert "nothing fails a run whose margin is small" in tm and "nothing fails a run whose margin is small" in masking


def test_each_tiers_ciphertext_bytes_name_their_serialisation() -> None:
    mb = _flat(_read(DOCS / "models-and-benchmarks.md"))
    assert "The two tiers also count different bytes for the same ciphertext" in mb
    assert "TenSEAL's `serialize()`, which compresses" in mb
    assert "node-seal's `saveToArray(ComprModeType.none)`, uncompressed: 393,329 bytes" in mb
    fhe = _read(LIB / "clientFHE.ts")
    assert fhe.count("saveToArray(seal.ComprModeType.none)") >= 4, "the browser's bytes changed serialisation"


def test_the_reaction_attack_is_scoped_to_full_galois_sets_and_flooding_is_an_inference() -> None:
    tm = _flat(_read(DOCS / "threat-model.md"))
    assert "and noise flooding does not close it, because" not in tm, "the flooding sentence is the paper's again"
    assert "The abstract does not discuss noise flooding; this project's reading, not the paper's" in tm
    assert "The setting is not OpenFHE's bootstrap keys alone" in tm
    assert "calls `generate_galois_keys()` with no step list" in tm
    backends = (SRC / "fhe" / "backends.py").read_text()
    assert "ctx.generate_galois_keys()" in backends, (
        "the TenSEAL backend's Galois keys changed: reword the threat model's reaction-attack scope")


def test_the_masking_claim_names_the_linear_stage_and_the_one_exception() -> None:
    tm = _flat(_read(DOCS / "threat-model.md"))
    assert "the linear model whose class the stage reports, `src/qsh/fhe/ckks_inference.py`" in tm
    assert "One decryption is deliberately not masked: genomics' panel check (`encrypted_scores`)" in tm


def test_the_seal_residual_risk_carries_versions_and_its_check() -> None:
    tm = _flat(_read(DOCS / "threat-model.md"))
    for needle in ("node-seal 7.0.0 (SEAL 4.1.2)", "TenSEAL 0.3.18", "bundles SEAL 4.3.3", "TenSEAL 0.3.16 (SEAL 4.1.1)",
                   "PR #168", "7.1.0", "neither tier deserializes a SEAL object it did not create",
                   "`scripts/audit_deps.sh seal`"):
        assert needle in tm, f"the SEAL residual risk lost {needle!r}"
    script = _read(REPO / "scripts" / "audit_deps.sh")
    assert "audit_seal()" in script and "seal) audit_seal ;;" in script
    assert "all) audit_npm && audit_seal && audit_polytune && audit_python && audit_rust ;;" in script
    workflow = _read(REPO / ".github" / "workflows" / "audit.yml")
    assert "bash scripts/audit_deps.sh seal" in workflow


def _seal_check(tmp_path: Path, locked: str | None, newest: str) -> subprocess.CompletedProcess[str]:
    packages = {"": {"name": "x"}}
    if locked is not None:
        packages["node_modules/node-seal"] = {"version": locked}
    (tmp_path / "package-lock.json").write_text(json.dumps({"lockfileVersion": 3, "packages": packages}))
    env = {**os.environ, "AUDIT_WEB_DIR": str(tmp_path), "NPM_VIEW": f"echo {newest}"}
    return subprocess.run(["sh", str(REPO / "scripts" / "audit_deps.sh"), "seal"], capture_output=True,
                          text=True, env=env, timeout=60)


@pytest.mark.skipif(shutil.which("node") is None, reason="the check reads the lockfile with node")
def test_the_seal_check_fails_on_a_newer_release_and_passes_on_the_locked_one(tmp_path: Path) -> None:
    same = _seal_check(tmp_path, "7.0.0", "7.0.0")
    assert same.returncode == 0 and "is npm's newest" in same.stdout, same.stdout + same.stderr
    newer = _seal_check(tmp_path, "7.0.0", "7.1.0")
    assert newer.returncode == 1 and "node-seal 7.1.0 is published and 7.0.0 is locked" in newer.stderr
    missing = _seal_check(tmp_path, None, "7.1.0")
    assert missing.returncode == 1 and "could not read" in missing.stderr


def test_the_dead_fhe_config_fields_are_gone() -> None:
    from dataclasses import fields

    from qsh.config.defaults import FHEConfig

    names = {f.name for f in fields(FHEConfig)}
    assert not names & {"backend", "max_level", "use_gpu", "use_bootstrap"}, names
    import yaml

    fhe = yaml.safe_load((REPO / "config.yaml").read_text())["fhe"]
    assert not set(fhe) & {"backend", "max_level", "use_gpu", "use_bootstrap"}, fhe
    assert set(fhe) <= names, f"config.yaml sets FHE fields the dataclass does not have: {set(fhe) - names}"
    # Every field that remains has a reader outside the config package.
    readers = "\n".join(p.read_text() for p in SRC.rglob("*.py") if "config" not in p.parts)
    for name in names:
        assert re.search(rf"\.{name}\b", readers), f"FHEConfig.{name} is read by nothing"


def test_the_models_doc_states_what_their_records_say_about_provenance() -> None:
    """PI-16: the doc's three claims about the records hold, so a retrain that fixes one fails here."""
    metrics = json.loads((REPO / "models" / "metrics.json").read_text())
    shas = {k: v["git_sha"] for k, v in metrics.items() if isinstance(v, dict) and v.get("git_sha")}
    assert shas and all(s.endswith("-dirty") for s in shas.values()), (
        f"a record carries a clean git_sha now ({shas}): update docs/models-and-benchmarks.md's provenance list")
    students = [k for k in metrics if "student" in k]
    with_sha = sorted(k for k in students if metrics[k].get("git_sha"))
    # Round 86: the satellite student gained its own trainer, which records one; the other students are
    # recorded by their teachers' trainers, which do not.
    assert students and with_sha == ["satellite_comms.student"], (
        f"the students that record a git_sha changed ({with_sha}): update the provenance list")
    digests = sorted(k for k, v in metrics.items() if isinstance(v, dict) and v.get("data_digest"))
    assert digests == ["finance_aml.gbm"], f"data digests changed ({digests}): update the provenance list"
    doc = _flat(_read(DOCS / "models-and-benchmarks.md"))
    for needle in ("every recorded `git_sha` ends in `-dirty`",
                   "of the students, only `satellite_comms.student` records a `git_sha`",
                   "only `finance_aml.gbm` records a digest of its training data",
                   "name commits of the history before it was rewritten"):
        assert needle in doc, f"the provenance list lost {needle!r}"
