"""The GPU CKKS backend computes what the CPU one does, on the operations the hybrid uses.

WHY THIS EXISTS (Round 86, the owner's choice of the whole GPU path). `HeongpuBackend` raised NotImplementedError: HEonGPU has
no Python API, and the benchmark's "GPU" row measured an upstream example, not this project's arithmetic. The binding in
`native/heongpu_py` now runs the seven operations of the backend protocol. Skipped where it is not built (`scripts/build_heongpu.sh`).
"""

from __future__ import annotations

import numpy as np
import pytest

from qsh.fhe.backends import heongpu_available

pytestmark = pytest.mark.skipif(not heongpu_available(), reason="the HEonGPU module is not built here, or there is no GPU")


@pytest.fixture(scope="module")
def gpu():
    from qsh.fhe.backends import get_backend

    return get_backend("heongpu")


@pytest.fixture(scope="module")
def cpu():
    from qsh.fhe.backends import get_backend

    return get_backend("tenseal")


def test_the_ring_is_the_configured_one(gpu, cpu) -> None:
    assert gpu.slot_count == cpu.slot_count == 4096


def test_each_operation_matches_plaintext(gpu) -> None:
    rng = np.random.default_rng(7)
    x, y, w = rng.normal(size=33), rng.normal(size=33), rng.normal(size=33)
    ct = gpu.encrypt(x)
    assert np.allclose(gpu.decrypt(ct), x, atol=1e-6)
    assert np.allclose(gpu.decrypt(gpu.add(ct, gpu.encrypt(y))), x + y, atol=1e-6)
    assert np.allclose(gpu.decrypt(gpu.add_plain(ct, y)), x + y, atol=1e-6)
    assert np.allclose(gpu.decrypt(gpu.add_plain(ct, 0.5)), x + 0.5, atol=1e-6)
    assert np.allclose(gpu.decrypt(gpu.mul_plain(ct, w)), x * w, atol=1e-6)
    assert abs(gpu.decrypt(gpu.dot_plain(ct, w))[0] - float(w @ x)) < 1e-6


def test_a_dot_product_agrees_with_the_cpu_backend(gpu, cpu) -> None:
    rng = np.random.default_rng(11)
    x, w = rng.normal(size=64), rng.normal(size=64)
    g = gpu.decrypt(gpu.dot_plain(gpu.encrypt(x), w))[0]
    c = cpu.decrypt(cpu.dot_plain(cpu.encrypt(x), w))[0]
    assert abs(g - c) < 1e-5 and abs(g - float(w @ x)) < 1e-6


def test_the_mask_covers_every_slot_and_both_parts(gpu) -> None:
    """`add_plain_slots` is what the hybrid's masked conversion relies on: a fresh encryption over every slot."""
    rng = np.random.default_rng(3)
    x, w = rng.normal(size=33), rng.normal(size=33)
    d = gpu.dot_plain(gpu.encrypt(x), w)
    mask = rng.uniform(-2**30, 2**30, size=gpu.slot_count) + 1j * rng.uniform(-2**30, 2**30, size=gpu.slot_count)
    out = gpu.decrypt(gpu.add_plain_slots(d, mask))
    assert len(out) == gpu.slot_count and len(out.imag) == gpu.slot_count
    assert abs(out[0] - (float(w @ x) + mask[0].real)) < 1e-2, "a 2**30 mask leaves about 10 bits of a 2**40 scale"
    assert abs(out.imag[1] - mask[1].imag) < 1e-2


def test_the_selector_offers_it_only_where_it_runs() -> None:
    from qsh.orchestration.crypto_agility import default_registry

    assert "heongpu-gpu" in default_registry().options("fhe")


@pytest.mark.parametrize(("usecase", "config"), [
    ("finance_aml", {"model_size": "medium", "seed": 0, "rounds": 3}),
    ("medical_fl", {"dataset": "medmnist", "img_size": 10, "cnn_channels": 2, "medmnist_n": 64, "hospitals": 2,
                    "rounds": 1, "seed": 0}),
], ids=["dense", "cnn"])
def test_a_run_that_selects_it_times_its_row_on_it(usecase, config) -> None:
    """The live benchmark row runs on the backend the run selected, for the dense and the convolutional model.

    Until Round 86 `_hybrid_bench_row` turned every name but TenSEAL's and OpenFHE's into TenSEAL, so a run that selected
    HEonGPU encrypted its inference on the GPU and timed its benchmark row on the CPU, under one selection.
    """
    pytest.importorskip("mpyc")
    from qsh.orchestration.pipeline import build_pipeline

    pipe, ctx = build_pipeline(usecase, {**config, "fhe": "heongpu-gpu"})
    pipe.run(ctx)
    assert pipe.state.value == "done"
    row = ctx.artifacts["mpc_fhe_row"]
    assert row["backend"] == "heongpu-ckks-gpu + mpyc-shamir" and row["device"] == "gpu", row["backend"]
    assert row["accuracy"] == 1.0 and row["fhe_linear_ms"] > 0 and row["mpc_nonlinear_ms"] > 0
