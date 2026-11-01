"""The full deployment's secure aggregation masks in the browser's ring, and its shares are the browser's.

WHY THIS EXISTS (Round 85)

Both tiers agreed each pair's mask seed by X25519 and expanded it with SHAKE-256 to the same words
(Round 84), but only the browser used them as ring elements: it quantises each weighted update to
``round(x * 2^20)`` and adds and subtracts the words modulo 2^32, so the masks cancel exactly and a
share is the quantised update plus a mask the aggregator cannot derive. The Python twin
(``qsh.ai.federated``) turned each word into a real in [-1, 1) and added it in float64: the masks
cancelled only to rounding, and a real mask of that size over an update of order one hides it only
partially. The owner chose the exact ring for Python too.

WHAT IS ASSERTED

- For the same fixed private keys, updates, shard weights and rounds, every client's masked share in
  Python is the browser's ``maskedShare`` to the bit, and the two aggregates are equal floats.
- The aggregate is exactly the sum of the quantised updates, not the real sum to a tolerance.
- Quantisation rounds as JavaScript's ``Math.round`` does, including the halves and the one value
  ``floor(x + 0.5)`` gets wrong, and refuses a coordinate the ring cannot hold, as the browser does.

On the base commit (6bb8657) all three fail: Python's shares are float64 reals, and it has no quantisation.
"""

from __future__ import annotations

import importlib.util
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

_spec = importlib.util.spec_from_file_location("r84_masks", REPO / "tests" / "test_r84_pairwise_masks.py")
_r84 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r84)

#: Scaled values at the edges of rounding: halves both ways, and 0.49999999999999994, which
#: ``floor(x + 0.5)`` takes to 1 because the sum rounds up to 1.0 in double precision.
EDGES = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994, 1e-300, -1e-300]

_HARNESS = r"""
import * as m from %(bundle)s;
const fromHex = (h) => Uint8Array.from(Buffer.from(h, "hex"));
const input = %(input)s;
const keys = input.keys.map(fromHex);
const kps = m.secaggKeyPairs(keys.length, keys);
const publicKeys = kps.map((kp) => kp.publicKey);
const seeds = await Promise.all(kps.map((kp, u) => m.secaggClientSeeds(u, kp.privateKey, publicKeys)));
const rounds = Object.fromEntries(input.rounds.map((r) => {
  const { aggregate, maskedShares } = m.secureAggregate(input.updates, input.weights, seeds, r);
  return [r, { aggregate, maskedShares }];
}));
// The edges, through the share a lone client with no peers would send: its quantised update.
const edges = input.edges.map((x) => m.maskedShare(0, [x / 2 ** 20], 1, [null], 0)[0]);
let refused = null;
try { m.maskedShare(0, [3000], 1, [null, null, null, null], 0); } catch (e) { refused = String(e.message); }
console.log(JSON.stringify({ rounds, edges, refused }));
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
    d = tmp_path_factory.mktemp("r85ring")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/clientQFL'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    payload = {"keys": [k.hex() for k in _r84.FIXED], "rounds": list(_r84.ROUNDS), "updates": _r84.UPDATES,
               "weights": _r84.WEIGHTS, "edges": EDGES}
    runner.write_text(_HARNESS % {"bundle": json.dumps(bundle.as_uri()), "input": json.dumps(payload)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_python_shares_are_the_browsers_to_the_bit(browser) -> None:
    from qsh.ai import federated as fed

    agg = fed.SecureAggregator(_r84.K, private_keys=_r84.FIXED)
    for rnd in _r84.ROUNDS:
        b = browser["rounds"][str(rnd)]
        shares = [agg.mask_update(c, np.array(_r84.UPDATES[c]) * _r84.WEIGHTS[c], round_index=rnd)
                  for c in range(_r84.K)]
        assert all(s.dtype == np.uint32 for s in shares)
        assert [s.tolist() for s in shares] == b["maskedShares"], f"round {rnd}"
        assert fed.SecureAggregator.aggregate(shares).tolist() == b["aggregate"], f"round {rnd}"


def test_the_aggregate_is_the_exact_sum_of_the_quantised_updates() -> None:
    from qsh.ai import federated as fed

    rng = np.random.default_rng(85)
    for n in (2, 3, 5, 8):
        agg = fed.SecureAggregator(n)
        ups = [rng.normal(scale=3.0, size=41) / n for _ in range(n)]
        shares = [agg.mask_update(c, u, round_index=4) for c, u in enumerate(ups)]
        exact = [sum(_r84._js_round(u[k] * fed.SECAGG_SCALE) for u in ups) / fed.SECAGG_SCALE
                 for k in range(41)]
        assert fed.SecureAggregator.aggregate(shares).tolist() == exact, n


def test_quantisation_rounds_as_javascript_does_and_refuses_what_the_ring_cannot_hold(browser) -> None:
    from qsh.ai import federated as fed

    got = fed.quantise(np.array(EDGES) / 2 ** 20, 1).tolist()
    assert got == browser["edges"]
    # Half up, as Math.round: 0.5 -> 1, -0.5 -> 0, -2.5 -> -2; and 0.49999999999999994 -> 0.
    assert got[:8] == [1, 0, 2, (-1) % 2**32, 3, (-2) % 2**32, 0, 0]
    assert browser["refused"] and "out of ring range" in browser["refused"]
    with pytest.raises(ValueError, match="out of the ring's range"):
        fed.quantise(np.array([3000.0]), 4)
    with pytest.raises(ValueError, match="out of the ring's range"):
        fed.quantise(np.array([np.nan]), 4)
