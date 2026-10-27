"""The browser's "ckks + mpyc-shamir" cost model carries the MPyC tier's own measured communication.

WHY THIS EXISTS

The cost-model row is labelled "ckks + mpyc-shamir (cost model, not measured)", and until Round 75 it
charged 256 bytes per secure comparison. Nothing could have measured that figure: the MPyC tier ran
as one in-process party, so no byte crossed a wire. Once three parties ran over sockets
(qsh/mpc/parties.py), a comparison measured 15,120 bytes across them, 59x the constant. The Python
side carried its own copy of the same 256 in a second config field, which is gone.

Communication is a property of the protocol and its parameters, not of the machine, so it is pinned
here by measurement. Protocol time depends on the host and is recorded with its provenance in
config/defaults.py instead.

WHAT IS ASSERTED

- Each secure operation the hybrid uses (ReLU, argmax, max-pool) sends, per comparison, the bytes the
  cost model charges, within 2 percent.
- The measurement runs at the shipped party settings, so the figure describes the tier as configured.
- No second per-comparison byte constant has come back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mpyc")

from qsh.config import get_config
from qsh.mpc.parties import run_parties

REPO = Path(__file__).resolve().parents[1]


def _charged() -> int:
    return int(get_config().benchmark.cost_per_op["mpc_comm_bytes_per_cmp"])


def _shipped() -> dict:
    b = get_config().benchmark
    return {"m": b.mpc_parties, "threshold": b.mpc_threshold, "sec_param": b.mpc_sec_param}


# As the hybrid runs them: the client's masked values, the server's share and, where the result goes
# back under FHE, the server's output mask.
_MASK = [1e6, -2e5, 3e4, -4e3, 5e2, -6e1, 7.0, -8e5]
_RELU = [-1.0, 0.5, 2.0, -0.25, 1.5, -3.0, 0.0, 4.0]
_ARGMAX = [0.1, 0.9, -0.4, 0.3, 2.2, -1.0, 0.7, 0.05]


@pytest.mark.parametrize("op,job,comparisons", [
    ("relu", {"values": [v + m for v, m in zip(_RELU, _MASK, strict=True)],
              "server_share": [-m for m in _MASK], "out_mask": _MASK[::-1]}, 8),
    ("argmax", {"values": [v + m for v, m in zip(_ARGMAX, _MASK, strict=True)],
                "server_share": [-m for m in _MASK]}, 7),
    ("maxpool", {"windows": [[0.1 + 5.0, 0.4, -0.2, 0.3], [1.0, -1.0, 0.5 - 7.0, 0.2]],
                 "server_share": [-5.0, 0, 0, 0, 0, 0, 7.0, 0], "out_mask": [9.0, -9.0]}, 6),
], ids=["relu", "argmax", "maxpool"])
def test_each_secure_op_sends_what_the_cost_model_charges(op: str, job: dict, comparisons: int) -> None:
    run = run_parties(op, **job, **_shipped())
    per_comparison = run["bytes_sent"] / comparisons
    assert per_comparison == pytest.approx(_charged(), rel=0.02), (
        f"{op}: {per_comparison:.0f} B per comparison measured, {_charged()} charged")


def test_the_measurement_is_at_the_shipped_settings() -> None:
    run = run_parties("relu", [1.0, -1.0], **_shipped())
    b = get_config().benchmark
    assert (run["parties"], run["threshold"], run["sec_param"]) == (
        b.mpc_parties, b.mpc_threshold, b.mpc_sec_param)


def test_one_per_comparison_byte_constant() -> None:
    src = (REPO / "src" / "qsh" / "config" / "defaults.py").read_text()
    assert "mpc_comm_bytes_per_compare" not in src, "the second copy of the byte constant is back"
    assert len(re.findall(r'"mpc_comm_bytes_per_cmp":', src)) == 1


def test_the_counted_comparisons_are_the_ones_the_bytes_pay_for() -> None:
    """The hybrid's count, the projection's count and the bytes on the wire describe the same work.

    Until Round 75 the hybrid counted the final argmax over k classes as k comparisons (MPyC runs
    k - 1), and the projection added a max-pool layer's per-window cost once per LAYER.
    """
    import numpy as np

    from qsh.benchmark.runner import _cmps
    from qsh.fhe.backends import get_backend
    from qsh.mpc import ConvSpec, DenseSpec, FlattenSpec, PoolSpec, hybrid_private_inference

    rng = np.random.default_rng(5)
    specs = [
        ConvSpec(rng.normal(size=(2, 1, 3, 3)) * 0.3, rng.normal(size=2) * 0.1,
                 in_shape=(1, 8, 8), stride=1, padding=1, activation="relu"),
        PoolSpec(pool="max", size=2, stride=2),
        FlattenSpec(),
        DenseSpec(rng.normal(size=(4, 2 * 4 * 4)) * 0.2, rng.normal(size=4) * 0.1, activation="relu"),
        DenseSpec(rng.normal(size=(3, 4)) * 0.2, rng.normal(size=3) * 0.1, activation="none"),
    ]
    r = hybrid_private_inference(specs, rng.normal(size=(1, 8, 8)) * 0.3, get_backend("tenseal"))
    expected = 2 * 8 * 8 + 2 * 4 * 4 * (2 * 2 - 1) + 4 + (3 - 1)
    assert r["mpc_comparisons"] == expected == _cmps(specs)
    assert r["mpc_comm_kb"] * 1024 / expected == pytest.approx(_charged(), rel=0.02)
