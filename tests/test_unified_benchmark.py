"""Round-7 unification: the live E2E pipeline run IS the benchmark.

A single pipeline run must produce ``ctx.artifacts["benchmark_result"]`` assembled from that run:
the live MPC-FHE row (real measured FHE-linear + MPC-non-linear latency) + a QSMPC-FHE comparison
row + quantum advantage. Telecom/satellite (no private inference) get the quantum row only.
"""

import pytest

pytest.importorskip("mpyc")
pytest.importorskip("tenseal")

from qsh.orchestration.pipeline import build_pipeline


def _run(usecase, config):
    pipe, ctx = build_pipeline(usecase, config)
    pipe.run(ctx)
    return pipe, ctx


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics", "medical_fl"])
def test_inference_usecase_unifies_run_and_benchmark(usecase):
    pipe, ctx = _run(usecase, {"model_size": "medium", "seed": 0, "rounds": 3})
    assert pipe.state.value == "done"
    # the live run produced the MPC-FHE row from the real hybrid (split + comm)
    row = ctx.artifacts["mpc_fhe_row"]
    assert row["fhe_linear_ms"] > 0 and row["mpc_nonlinear_ms"] > 0
    assert row["mpc_comm_kb"] > 0 and row["projected"] is False
    # the model is a per-use-case architecture with a parameter count (not a generic MLP)
    assert row["model_arch"] and row["model_params"] > 0
    # per-stage split metrics are on the event bus (drive the diagram + RunMonitor)
    m = pipe.bus.metrics()
    assert m["fhe_linear_ms"] > 0 and m["mpc_nonlinear_ms"] > 0
    # the benchmark is assembled from the run: MPC-FHE + QSMPC-FHE + advantage
    br = ctx.artifacts["benchmark_result"]
    assert br["live"] is True
    paths = {r["path"] for r in br["results"]}
    assert paths == {"MPC-FHE", "QSMPC-FHE"}
    assert br["advantage"] and br["advantage"]["score"] > 0
    # the benchmark rides into the ML-DSA-signed summary
    assert "benchmark_result" in ctx.artifacts["summary"]["artifacts"]


def test_large_tier_is_projected():
    _, ctx = _run("finance_aml", {"model_size": "large", "seed": 0})
    row = ctx.artifacts["mpc_fhe_row"]
    assert row["projected"] is True
    assert row["fhe_linear_ms"] > 0 and row["mpc_nonlinear_ms"] > 0


def test_medical_cnn_runs_genuine_conv():
    """Medical with ``dataset=medmnist`` runs a GENUINE convolution under FHE+MPC: the MPC-FHE row is
    a real CNN (arch_class 'cnn'), the MPC ran every comparison the convolution's ReLU needs, and the
    decision is faithful.

    Uses real MedMNIST when available; an offline CI box falls back to the logged synthetic-image set
    (the conv mechanism + assertions hold either way). A tiny 10x10 / 2-channel / 1-round config keeps
    the genuine encrypted run fast.

    Until Round 75 this also asserted that the FHE time exceeded the MPC time. That held while the MPC
    was one in-process party. With three real parties it is a property of the machine: 1,460 ms
    against 475 ms on the 20-core development box, and 2,421 ms against 2,868 ms on a two-core CI
    runner, where three party processes share two cores. It is asserted in work instead.
    """
    _, ctx = _run("medical_fl", {"dataset": "medmnist", "img_size": 10, "cnn_channels": 2,
                                 "medmnist_n": 64, "hospitals": 2, "rounds": 1, "seed": 0})
    row = ctx.artifacts["mpc_fhe_row"]
    assert (row.get("extra") or {}).get("arch_class") == "cnn"
    assert "CNN" in row["model_arch"]
    assert row["fhe_linear_ms"] > 0 and row["mpc_nonlinear_ms"] > 0
    extra = row.get("extra") or {}
    assert extra.get("avg_comparisons", 0) >= 2 * 10 * 10    # every conv output's ReLU ran under MPC
    assert extra.get("client_learns") == "the class index"   # through the masked conversions
    assert row["accuracy"] == 1.0                            # decision matches plaintext (faithful)
    assert row["projected"] is False                         # medium tier runs for real (no fallback)
    assert ctx.data["image"].shape == (1, 10, 10)            # a real medical image was inferred
    assert ctx.artifacts["benchmark_result"]["live"] is True


@pytest.mark.parametrize("usecase", ["telecom_ids", "satellite_comms"])
def test_compact_usecase_has_quantum_row_only(usecase):
    _, ctx = _run(usecase, {"seed": 0})
    br = ctx.artifacts["benchmark_result"]
    assert br["live"] is True
    paths = [r["path"] for r in br["results"]]
    assert paths == ["QSMPC-FHE"]            # no inference -> no MPC-FHE row
    assert br["advantage"] is None           # compact panel (security row only)
