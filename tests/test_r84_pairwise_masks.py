"""Round 84 (secagg): every pairwise mask comes from a seed the two clients of a pair agreed by X25519.

WHY THIS EXISTS

The federated secure aggregation's masks cancelled, and hid nothing from the aggregator. In the
browser (`web/src/lib/clientQFL.ts`) every pair's mask was an HKDF-SHA256 output of the run's one PQC
session key, which the aggregator side of the handshake holds, so it could strip every mask. In
Python (`src/qsh/ai/federated.py`) `SecureAggregator` was only ever built with no keys, so every pair
key was `sha256(f"pair-{i}-{j}")`, a public constant. By the owner's decision both tiers now follow
the pairwise key agreement of Bonawitz et al. (ACM CCS 2017, doi:10.1145/3133956.3133982; the masking
of Protocol 3 in arXiv:1611.04482): each client draws an X25519 key pair per run, each pair i < j
derives its seed from the X25519 shared secret by HKDF-SHA256, each round's mask comes from the seed,
and the aggregator receives only the public keys and the masked updates.

THE BYTES, written out here rather than imported, so the test does not agree with the code by
construction:

    salt        b"qsh/secagg/salt/v1"
    info        b"qsh/secagg/pair/v1" + u32be(i) + u32be(j)
    seed        HKDF-SHA256(X25519(sk_i, pk_j), salt, info, 32)
    round mask  SHAKE-256(seed + b"qsh/secagg/mask/v1" + u32be(round)), little-endian uint32 words
    Python      each word w is added as the real w / 2**31 - 1

WHAT IS ASSERTED, AND WHAT EACH FAILS ON AT THE BASE COMMIT (f5f1fce)

- Python: the masks cancel in the sum (to the 1e-9 the existing tests assert), per run and per round.
  Base: `SecureAggregator` has no `public_keys` and no `private_keys`.
- Python: the pair seed is the HKDF of the X25519 secret, both members of a pair reach it, and the
  masks are its SHAKE-256 expansion. Base: no `pair_seed`, and the masks came from the public constant.
- Python: the default keys are fresh per run, so the public constant no longer unmasks an update, and
  fixed private keys reproduce a run. Base: the constant unmasks every update; `federated_train` has no
  `secagg_private_keys`.
- Python and the browser derive byte-identical public keys, pair seeds and per-round masks for the
  same fixed private keys, and the browser's masked shares are exactly the quantised updates plus
  those masks. Base: `clientQFL.ts` exports no key agreement.
- The browser's masks do not depend on the session key (two session keys, the same masks) and a
  session-key holder cannot unmask a share with the derivation the base used. Base: the masks are
  keyed by the session key, so a holder unmasks every share.
- The aggregator's inputs, in both tiers, contain no private key, pair seed or shared secret, and the
  aggregation step reads the masked updates alone. Base: no aggregator input is observable.
- The browser pipeline's federated stage, run in node end to end for medical and finance, reports the
  same figures the base reported, names the key agreement, and publishes fresh public keys per run;
  its source reads no session key. Base: the stage passes `_sessionKey` and names the session key.

The garbled-circuit argmax is a gitignored wasm build, so a plaintext argmax stands in for it in the
pipeline run, as in tests/test_r84_finance_browser_run.py; nothing below asserts anything about it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

SALT = b"qsh/secagg/salt/v1"
PAIR_LABEL = b"qsh/secagg/pair/v1"
MASK_LABEL = b"qsh/secagg/mask/v1"

#: Four fixed, obviously public X25519 private keys: this file asserts the construction, not key secrecy.
K = 4
FIXED = [hashlib.sha256(b"qsh/secagg/test-key/%d" % c).digest() for c in range(K)]
#: Two stand-ins for a run's PQC session key.
SESSION_A = hashlib.sha256(b"qsh/secagg/test-session-key/a").digest()
SESSION_B = hashlib.sha256(b"qsh/secagg/test-session-key/b").digest()
ROUNDS = (0, 1, 2, 7)
P = 5
#: Four clients' updates and uneven shard sizes, as web/scripts/check_qfl.mjs uses.
UPDATES = [[0.8125, -1.5, 0.03125, 2.0, -0.4], [-0.25, 0.5, 1.125, -0.75, 0.3],
           [1.5, 0.25, -0.5, 0.0625, -1.1], [-0.0625, 2.25, 0.75, -1.375, 0.9]]
SIZES = [37, 21, 29, 13]
WEIGHTS = [n / sum(SIZES) for n in SIZES]
SCALE = 1 << 20

#: The browser pipeline's federated figures at the base commit f5f1fce (seed 0), measured with the
#: harness below on the base tree. The masks cancel exactly in the ring, so a change to the masks
#: must not move any of them.
BASE_FIGURES = {
    "medical_fl": {"roundAcc": [0.6, 0.69, 0.75, 0.78, 0.78, 0.8], "finalAcc": 0.8, "baselineAcc": 0.53,
                   "evalRows": 80, "clients": 4, "rounds": 6, "pairs": 6},
    "finance_aml": {"roundAcc": [0.76, 0.75, 0.78, 0.79, 0.79, 0.79], "finalAcc": 0.79, "baselineAcc": 0.45,
                    "evalRows": 80, "clients": 4, "rounds": 6, "pairs": 6},
}


# ---------------------------------------------------------------------------------------------
# The construction, computed here independently of the code under test
# ---------------------------------------------------------------------------------------------

def _public(sk: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(sk).public_key().public_bytes_raw()


def _shared(i: int, j: int) -> bytes:
    return X25519PrivateKey.from_private_bytes(FIXED[i]).exchange(
        X25519PublicKey.from_public_bytes(_public(FIXED[j])))


def _seed(i: int, j: int) -> bytes:
    lo, hi = min(i, j), max(i, j)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=SALT,
                info=PAIR_LABEL + struct.pack(">II", lo, hi)).derive(_shared(lo, hi))


def _words(seed: bytes, rnd: int, n: int) -> list[int]:
    raw = hashlib.shake_256(seed + MASK_LABEL + struct.pack(">I", rnd)).digest(4 * n)
    return list(struct.unpack(f"<{n}I", raw))


def _js_round(x: float) -> int:
    """JavaScript's Math.round: half rounds up, unlike Python's round."""
    return math.floor(x + 0.5)


def _plain_share(u: int) -> list[int]:
    return [_js_round(v * WEIGHTS[u] * SCALE) % 2 ** 32 for v in UPDATES[u]]


def _old_session_key_mask(key: bytes, u: int, v: int, rnd: int, n: int) -> list[int]:
    """The base commit's browser derivation: HKDF-SHA256 of the session key, the pair and round in the info."""
    okm = HKDF(algorithm=hashes.SHA256(), length=4 * n, salt=None,
               info=f"qsh/qfl-pairwise-mask/v1|{u}|{v}|r{rnd}".encode()).derive(key)
    return list(struct.unpack(f"<{n}I", okm))


def _contains_any(blob: bytes, secrets: list[bytes]) -> list[str]:
    return [s.hex()[:16] for s in secrets if s in blob]


def _secrets() -> list[bytes]:
    """Every secret of the fixed run: the private keys, the shared secrets and the pair seeds."""
    out = list(FIXED)
    for i in range(K):
        for j in range(i + 1, K):
            out += [_shared(i, j), _seed(i, j)]
    return out


# ---------------------------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------------------------

def _federated():
    from qsh.ai import federated
    return federated


@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.parametrize("rnd", [0, 3])
def test_python_masks_cancel_in_the_sum(n: int, rnd: int) -> None:
    fed = _federated()
    agg = fed.SecureAggregator(n)
    ups = [np.random.default_rng(10 * n + c).normal(size=(17,)) for c in range(n)]
    masked = [agg.mask_update(c, u, round_index=rnd) for c, u in enumerate(ups)]
    assert len(agg.public_keys) == n and all(len(pk) == 32 for pk in agg.public_keys)
    np.testing.assert_allclose(agg.aggregate(masked), np.sum(ups, axis=0), atol=1e-9)
    assert not any(np.allclose(m, u) for m, u in zip(masked, ups)), "an update went out unmasked"


def test_python_pair_seed_is_the_hkdf_of_the_x25519_secret() -> None:
    fed = _federated()
    own = [X25519PrivateKey.from_private_bytes(k) for k in FIXED]
    pub = [_public(k) for k in FIXED]
    assert fed.SECAGG_SALT == SALT and fed.SECAGG_PAIR_LABEL == PAIR_LABEL and fed.SECAGG_MASK_LABEL == MASK_LABEL
    assert fed.pair_info(1, 3) == PAIR_LABEL + b"\x00\x00\x00\x01\x00\x00\x00\x03" and len(fed.pair_info(1, 3)) == 26
    for i in range(K):
        for j in range(i + 1, K):
            # Each member of the pair, from its own private key and the other's public key.
            assert fed.pair_seed(own[i], pub[j], i, j) == _seed(i, j)
            assert fed.pair_seed(own[j], pub[i], i, j) == _seed(i, j)
    agg = fed.SecureAggregator(K, private_keys=FIXED)
    assert agg.public_keys == pub
    for rnd in ROUNDS:
        for i in range(K):
            for j in range(i + 1, K):
                words = np.array(_words(_seed(i, j), rnd, P), dtype=np.float64)
                assert fed.pair_mask_words(_seed(i, j), rnd, P).tolist() == _words(_seed(i, j), rnd, P)
                np.testing.assert_array_equal(fed._mask(_seed(i, j), (P,), rnd), words / 2 ** 31 - 1)
    # Client c's masked update is its update plus the masks of its pairs, signed by the order.
    u = np.array(UPDATES[1])
    expected = u.copy()
    for j in (0, 2, 3):
        m = np.array(_words(_seed(1, j), 2, P), dtype=np.float64) / 2 ** 31 - 1
        expected += m if 1 < j else -m
    np.testing.assert_array_equal(agg.mask_update(1, u, round_index=2), expected)


def test_python_default_keys_are_fresh_per_run_and_not_the_public_constant() -> None:
    fed = _federated()
    a, b = fed.SecureAggregator(K), fed.SecureAggregator(K)
    assert set(a.public_keys).isdisjoint(b.public_keys), "two runs drew the same key pair"
    u = np.array(UPDATES[0])
    assert not np.allclose(a.mask_update(0, u, 1), b.mask_update(0, u, 1)), "two runs masked alike"
    # The base commit's default: every pair key sha256(f"pair-{i}-{j}"), each round's mask SHAKE-256 of
    # the key and the round, as reals in [-1, 1). Anyone holding the constant could unmask an update.
    def constant_mask(i: int, j: int, rnd: int) -> np.ndarray:
        raw = hashlib.shake_256(hashlib.sha256(f"pair-{i}-{j}".encode()).digest()
                                + b"|qsh/qfl-pairwise-mask/v1|r" + str(rnd).encode()).digest(8 * P)
        return (np.frombuffer(raw, dtype=np.uint64).astype(np.float64) / 2 ** 64 - 0.5) * 2.0
    masked = a.mask_update(0, u, 1)
    stripped = masked - sum(constant_mask(0, j, 1) for j in range(1, K))
    assert not np.allclose(stripped, u), "the public constant still unmasks client 0's update"


def test_python_fixed_private_keys_reproduce_a_run() -> None:
    fed = _federated()
    from qsh.data.synthetic import synthetic_transactions

    first = fed.SecureAggregator(K, private_keys=FIXED)
    second = fed.SecureAggregator(K, private_keys=FIXED)
    u = np.array(UPDATES[2])
    np.testing.assert_array_equal(first.mask_update(2, u, 5), second.mask_update(2, u, 5))
    X, y = synthetic_transactions(n=400, seed=0)
    parts = fed.partition(X[:320], y[:320], n_clients=K, iid=True, seed=0)
    runs = [fed.federated_train(parts, kind="mlp", rounds=2, eval_data=(X[320:], y[320:]), seed=0,
                                secagg_private_keys=FIXED, return_model=True) for _ in range(2)]
    w = [np.concatenate([p.detach().cpu().numpy().ravel() for p in m.parameters()]) for _, m in runs]
    np.testing.assert_array_equal(w[0], w[1])
    assert runs[0][0].round_acc == runs[1][0].round_acc


def test_python_aggregator_inputs_hold_no_private_key() -> None:
    fed = _federated()
    agg = fed.SecureAggregator(K, private_keys=FIXED)
    masked = [agg.mask_update(c, np.array(UPDATES[c]), 3) for c in range(K)]
    # What reaches the aggregator: the relayed public keys and the masked updates.
    view = b"".join(agg.public_keys) + b"".join(m.tobytes() for m in masked)
    assert not _contains_any(view, _secrets())
    assert agg.private_keys is None, "the aggregator object kept the clients' private keys"
    assert "private_keys" not in repr(agg)
    # The aggregation step is a function of the masked updates alone: no instance, no key.
    assert list(inspect.signature(fed.SecureAggregator.aggregate).parameters) == ["masked_updates"]
    np.testing.assert_allclose(fed.SecureAggregator.aggregate(masked), np.sum(UPDATES, axis=0), atol=1e-9)


# ---------------------------------------------------------------------------------------------
# The browser, bundled with esbuild and run in node
# ---------------------------------------------------------------------------------------------

_MODULE_HARNESS = r"""
import * as m from %(bundle)s;
const hex = (u) => Buffer.from(u).toString("hex");
const fromHex = (h) => Uint8Array.from(Buffer.from(h, "hex"));
const input = %(input)s;
const keys = input.keys.map(fromHex);
const kps = m.secaggKeyPairs(keys.length, keys);
const publicKeys = kps.map((kp) => kp.publicKey);
const seeds = await Promise.all(kps.map((kp, u) => m.secaggClientSeeds(u, kp.privateKey, publicKeys)));
const masks = {};
for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++)
  masks[`${i},${j}`] = Object.fromEntries(input.rounds.map((r) => [r, m.secaggMaskWords(seeds[i][j], r, input.P)]));
const rounds = Object.fromEntries(input.rounds.map((r) => {
  const { aggregate, maskedShares } = m.secureAggregate(input.updates, input.weights, seeds, r);
  return [r, { aggregate, maskedShares, direct: m.aggregateMaskedShares(maskedShares) }];
}));
// clientQFL under two session keys, the same fixed private keys: what the aggregator receives.
const views = [];
for (const sessionKey of input.sessionKeys.map(fromHex)) {
  const seen = [];
  const q = await m.clientQFL(0, { clients: keys.length, rounds: 3, privateKeys: keys, sessionKey,
    onAggregatorInput: (r, v) => seen.push({ r, publicKeys: v.publicKeys.map(hex), maskedShares: v.maskedShares }) });
  views.push({ seen, roundAcc: q.roundAcc, publicKeys: q.publicKeys, keys: Object.keys(q) });
}
const fresh = [];
for (let t = 0; t < 2; t++) {
  const q = await m.clientQFL(0, { clients: keys.length, rounds: 2 });
  fresh.push({ publicKeys: q.publicKeys, roundAcc: q.roundAcc, keyAgreementMs: q.keyAgreementMs });
}
console.log(JSON.stringify({
  constants: { salt: hex(m.SECAGG_SALT), pair: hex(m.SECAGG_PAIR_LABEL), mask: hex(m.SECAGG_MASK_LABEL),
               info13: hex(m.secaggPairInfo(1, 3)) },
  publicKeys: publicKeys.map(hex),
  seeds: seeds.map((row) => row.map((s) => (s ? hex(s) : null))),
  masks, rounds, views, fresh,
}));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r84secagg")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/clientQFL'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    payload = {"keys": [k.hex() for k in FIXED], "rounds": list(ROUNDS), "P": P, "updates": UPDATES,
               "weights": WEIGHTS, "sessionKeys": [SESSION_A.hex(), SESSION_B.hex()]}
    runner.write_text(_MODULE_HARNESS % {"bundle": json.dumps(bundle.as_uri()), "input": json.dumps(payload)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_both_languages_use_the_same_documented_bytes(browser) -> None:
    c = browser["constants"]
    assert bytes.fromhex(c["salt"]) == SALT and bytes.fromhex(c["pair"]) == PAIR_LABEL
    assert bytes.fromhex(c["mask"]) == MASK_LABEL
    assert bytes.fromhex(c["info13"]) == PAIR_LABEL + struct.pack(">II", 1, 3)


def test_both_languages_derive_the_same_public_keys_and_pair_seeds(browser) -> None:
    fed = _federated()
    assert [bytes.fromhex(h) for h in browser["publicKeys"]] == [_public(k) for k in FIXED]
    py = fed.SecureAggregator(K, private_keys=FIXED)
    for u in range(K):
        for v in range(K):
            got = browser["seeds"][u][v]
            if u == v:
                assert got is None
                continue
            # Client u's own derivation, in the browser, equals the HKDF of the X25519 secret, which
            # is also what Python's client u holds.
            assert bytes.fromhex(got) == _seed(u, v) == py._seeds[u][v], (u, v)


def test_both_languages_derive_the_same_round_masks(browser) -> None:
    fed = _federated()
    for i in range(K):
        for j in range(i + 1, K):
            per_round = browser["masks"][f"{i},{j}"]
            for rnd in ROUNDS:
                expected = _words(_seed(i, j), rnd, P)
                assert per_round[str(rnd)] == expected == fed.pair_mask_words(_seed(i, j), rnd, P).tolist()
            assert len({tuple(per_round[str(r)]) for r in ROUNDS}) == len(ROUNDS), "a mask repeated across rounds"


def test_the_browser_shares_are_the_updates_plus_the_x25519_masks_and_cancel(browser) -> None:
    for rnd in ROUNDS:
        out = browser["rounds"][str(rnd)]
        plain = [_plain_share(u) for u in range(K)]
        for u in range(K):
            expected = list(plain[u])
            for v in range(K):
                if v == u:
                    continue
                m = _words(_seed(u, v), rnd, P)
                expected = [(e + w) % 2 ** 32 if v > u else (e - w) % 2 ** 32 for e, w in zip(expected, m)]
            assert out["maskedShares"][u] == expected, (rnd, u)
            assert out["maskedShares"][u] != plain[u], f"client {u}'s share went out unmasked"
        # The ring sum of the masked shares is the ring sum of the plain shares, exactly.
        total = [sum(col) % 2 ** 32 for col in zip(*plain)]
        ref = [(t - 2 ** 32 if t >= 2 ** 31 else t) / SCALE for t in total]
        assert out["aggregate"] == ref == out["direct"]


def test_the_browser_masks_do_not_depend_on_the_session_key(browser) -> None:
    """Two session keys, the same fixed private keys: what the aggregator receives is identical."""
    a, b = browser["views"]
    assert [v["r"] for v in a["seen"]] == [0, 1, 2], "clientQFL showed no aggregator input"
    assert a["seen"] == b["seen"]
    assert a["publicKeys"] == b["publicKeys"] == [_public(k).hex() for k in FIXED]
    assert a["roundAcc"] == b["roundAcc"]


def test_a_session_key_holder_cannot_unmask_a_browser_share(browser) -> None:
    """The base's derivation, keyed by the session key, recovers nothing; the pair seeds recover all."""
    out = browser["rounds"]["1"]
    for u in range(K):
        share, plain = out["maskedShares"][u], _plain_share(u)
        for key in (SESSION_A, SESSION_B):
            guess = list(share)
            for v in range(K):
                if v == u:
                    continue
                m = _old_session_key_mask(key, min(u, v), max(u, v), 1, P)
                guess = [(g - w) % 2 ** 32 if v > u else (g + w) % 2 ** 32 for g, w in zip(guess, m)]
            assert guess != plain, f"the session key unmasked client {u}'s share"
            # And the guess is not even near the update: the quantised values are below 2^22 in size.
            assert any(min(g, 2 ** 32 - g) > 2 ** 24 for g in guess), (u, guess)


def test_the_browser_aggregator_input_holds_no_private_key(browser) -> None:
    secrets = _secrets() + [SESSION_A, SESSION_B]
    for view in browser["views"]:
        assert view["seen"]
        blob = b"".join(bytes.fromhex(pk) for pk in view["seen"][0]["publicKeys"])
        for step in view["seen"]:
            blob += b"".join(struct.pack(f"<{len(s)}I", *s) for s in step["maskedShares"])
        assert not _contains_any(blob, secrets)
        assert not {"privateKeys", "seeds", "pairSeeds", "sessionKey"} & set(view["keys"]), view["keys"]


def test_the_browser_draws_fresh_keys_per_run_and_the_figures_do_not_move(browser) -> None:
    a, b = browser["fresh"]
    assert len(a["publicKeys"]) == K and set(a["publicKeys"]).isdisjoint(b["publicKeys"])
    assert a["roundAcc"] == b["roundAcc"], "the masks moved a reported figure"
    assert a["keyAgreementMs"] >= 0 and b["keyAgreementMs"] >= 0


# ---------------------------------------------------------------------------------------------
# The browser pipeline, end to end in node
# ---------------------------------------------------------------------------------------------

_PIPELINE_HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-secagg-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /(^|\/)clientMPC$/ }, () => ({ path: "fakempc", namespace: "fakempc" }));
    b.onLoad({ filter: /.*/, namespace: "fakempc" }, () => ({ loader: "js", contents: `
      export async function warmMPC() {}
      export async function secureArgmax(l) {
        let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i;
        return { out: [b], compares: Math.max(0, l.length - 1), commBytes: 0, mpcMs: 0, messages: 0, clearOut: [b] };
      }` }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const result = {};
for (const [name, usecase] of [["medical", "medical_fl"], ["medical2", "medical_fl"], ["finance", "finance_aml"]]) {
  const logs = [], metrics = {};
  const config = { seed: 0, qfl: true };
  const end = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    m.runClientPipeline(usecase, config, {
      onEvent: (e) => {
        if (e.type === "log" && e.stage === "federated_training") logs.push(String(e.message ?? ""));
        if (e.type === "metric") (metrics[e.stage] ??= {})[e.data.name] = e.data.value;
      },
      onEnd: (s) => { clearTimeout(timer); done(s); },
    });
  });
  const sk = config._sessionKey;
  result[name] = { state: end?.state ?? null, logs, metrics: metrics.federated_training ?? null,
                   qfl: end?.artifacts?.qfl ?? null, sessionKey: sk ? Buffer.from(sk).toString("hex") : null };
}
console.log(JSON.stringify(result));
process.exit(0);
"""


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84secaggpipe") / "harness.mjs"
    script.write_text(_PIPELINE_HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name, usecase", [("medical", "medical_fl"), ("medical2", "medical_fl"),
                                           ("finance", "finance_aml")])
def test_the_pipeline_federates_on_agreed_seeds_with_the_base_figures(pipeline, name, usecase) -> None:
    run = pipeline[name]
    assert run["state"] == "done"
    q, met = run["qfl"], run["metrics"]
    for k, v in BASE_FIGURES[usecase].items():
        assert q[k] == v, f"{usecase}: {k} is {q[k]}, the base reported {v}; the masks must not move a figure"
    assert met["federated_accuracy"] == BASE_FIGURES[usecase]["finalAcc"]
    assert met["federated_baseline_accuracy"] == BASE_FIGURES[usecase]["baselineAcc"]
    assert met["qfl_aggregation"] == "size-weighted FedAvg + 6 pairwise masks from X25519-agreed pair seeds, fixed-point mod 2^32"
    assert "session key" not in met["qfl_aggregation"]
    assert any("X25519 key pair for this run" in line and "Bonawitz et al., CCS 2017" in line
               and "the PQC session key seeds no mask" in line for line in run["logs"]), run["logs"]
    # What the run publishes about the agreement: four 32-byte public keys and nothing secret.
    assert len(q["publicKeys"]) == 4 and all(re.fullmatch(r"[0-9a-f]{64}", pk) for pk in q["publicKeys"])
    assert run["sessionKey"] and run["sessionKey"] not in q["publicKeys"]
    assert not {"privateKeys", "seeds", "pairSeeds", "sessionKey"} & set(q)


def test_every_pipeline_run_draws_fresh_key_pairs(pipeline) -> None:
    keys = [set(pipeline[n]["qfl"]["publicKeys"]) for n in ("medical", "medical2", "finance")]
    assert keys[0].isdisjoint(keys[1]) and keys[0].isdisjoint(keys[2]) and keys[1].isdisjoint(keys[2])
    assert pipeline["medical"]["sessionKey"] != pipeline["medical2"]["sessionKey"]


def test_the_federated_stage_reads_no_session_key() -> None:
    """The stage's source, comments stripped: no `_sessionKey`, and clientQFL takes none."""
    src = (WEB / "src" / "lib" / "clientCompute.ts").read_text()
    start = src.index('case "federated_training": {')
    body = src[start:src.index("\n      case ", start + 10)]
    code = "\n".join(re.sub(r"//.*$", "", ln) for ln in body.splitlines())
    assert "_sessionKey" not in code and "sessionKey" not in code
    qfl = (WEB / "src" / "lib" / "clientQFL.ts").read_text()
    sig = qfl[qfl.index("export async function clientQFL("):qfl.index("): Promise<QFLResult>")]
    assert "sessionKey" not in sig
    # The derivation itself names X25519 and HKDF on WebCrypto, and no session key reaches it.
    seed_fn = qfl[qfl.index("export async function secaggPairSeed("):qfl.index("export async function secaggClientSeeds(")]
    assert "x25519.getSharedSecret(" in seed_fn and '"HKDF"' in seed_fn and "sessionKey" not in seed_fn


# ---------------------------------------------------------------------------------------------
# What the visitor reads: the caption, the blurb and the architecture box
# ---------------------------------------------------------------------------------------------

_TEXT_HARNESS = r"""
import { createRequire } from "node:module";
import { resolve } from "node:path";
const [WEB, OUT] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
await build({
  stdin: { contents: `export { captionOf } from ${JSON.stringify(resolve(WEB, "src/lib/useCaseFlows"))};
export { USECASE_META } from ${JSON.stringify(resolve(WEB, "src/lib/useCases"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stubs", setup(b) {
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
console.log(JSON.stringify({ medical: m.captionOf("medical_fl"), finance: m.captionOf("finance_aml"),
                             blurb: m.USECASE_META.medical_fl.blurb }));
"""


@pytest.fixture(scope="module")
def texts(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r84secaggtext")
    (d / "h.mjs").write_text(_TEXT_HARNESS)
    proc = subprocess.run(["node", str(d / "h.mjs"), str(WEB), str(d / "b.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


#: What each visitor text said at the base commit, and must not say again.
_STALE = ("derived from the session key", "derive from the session", "derives from the one session key",
          "hide nothing from it", "hide no update from it", "could unmask each update")


@pytest.mark.parametrize("key", ["medical", "finance", "blurb"])
def test_the_visitor_texts_state_the_key_agreement_and_its_limits(texts, key) -> None:
    t = texts[key]
    assert "agrees its mask seed by X25519" in t, t
    assert "cannot remove a mask alone" in t, t
    assert not [s for s in _STALE if s in t], t
    assert "authentication of the public keys" in t and "no dropout recovery" in t, t


def test_the_architecture_box_names_the_pair_seeds() -> None:
    src = (WEB / "src" / "components" / "ArchitectureDiagram.tsx").read_text()
    box = re.search(r'\{ id: "fl",.*?lines: \[(.*?)\]', src, re.S)
    assert box, "the federated box moved"
    assert json.loads(f"[{box.group(1)}]") == ["finance, medical:", "FedAvg; pairwise masks", "seeded by X25519 per pair"]
    desc = re.sub(r"\s+", " ", src[src.index('<desc id="arch-desc">'):src.index("</desc>")])
    assert "masks with seeds each pair of clients agrees by X25519, not with the session key" in desc
