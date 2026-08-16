"""Hybrid private inference: linear layers under FHE (CKKS), non-linear under MPC (MPyC).

The 2026-standard secure-inference split (EncFormer, npj/IACR 2026; the Gazelle→Delphi→BOLT→
Bumblebee→BLB lineage; the RhombusEnd2End/HEonGPU reference): **linear** ops (matmul/conv) run
homomorphically under CKKS (GPU-accelerable via HEonGPU), while **non-linear** ops (ReLU, argmax,
comparison) run under **secret-sharing MPC** — they are cheap homomorphically only as low-degree
polynomials, so a real hybrid delegates them to MPC. At each FHE↔MPC boundary the CKKS result is
decrypted-and-secret-shared (and re-encrypted for the next linear layer) — EncFormer's
"stage-compatible patterns" minimise this conversion cost.

Here the non-linear half uses **MPyC** (MIT, Shamir secret sharing + secure comparison). The PoC
runs the parties in-process (LAN-local simulation); a real deployment runs one party per host.
Communication is reported as an estimate from the comparison protocol's bit-complexity (the
in-process sim exchanges no real bytes).
"""

from __future__ import annotations

import logging
import time

import numpy as np

# keep MPyC quiet inside the benchmark
logging.getLogger("mpyc").setLevel(logging.ERROR)

# Representative 2-party communication per L-bit secure comparison (OT/edaBits-class): ~256 B.
#: Bytes on the wire per secure comparison. Lives in config so the demo and the backend cannot
#: disagree about communication cost; see qsh.config.defaults.BenchmarkConfig.
def _mpc_comm_bytes_per_compare() -> int:
    from ..config import get_config
    return get_config().benchmark.mpc_comm_bytes_per_compare


def mpyc_available() -> bool:
    try:
        import mpyc  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def _import_mpc():
    """Import MPyC's runtime singleton. MPyC parses ``sys.argv`` on first import (it has its own
    CLI for party config); neutralise argv so it never consumes the host app's flags (uvicorn /
    the benchmark CLI). Cached after the first import."""
    import sys

    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        from mpyc.runtime import mpc
        return mpc
    finally:
        sys.argv = saved


def _run(coro_factory):
    """Run one MPyC session (start → compute → shutdown) and return the opened result."""
    mpc = _import_mpc()

    async def main():
        await mpc.start()
        try:
            return await coro_factory(mpc)
        finally:
            await mpc.shutdown()

    return mpc.run(main())


def secure_relu(vec, *, bits: int = 64) -> list[float]:
    """ReLU on secret-shared values via MPyC secure comparison: ReLU([x]) = [x>=0]·[x]."""
    async def f(mpc):
        sec = mpc.SecFxp(bits)
        xs = [sec(float(v)) for v in vec]
        return await mpc.output([(x >= 0) * x for x in xs])
    return [float(v) for v in _run(f)]


def secure_argmax(vec, *, bits: int = 64) -> int:
    """argmax over secret-shared values via MPyC (secure comparisons)."""
    async def f(mpc):
        sec = mpc.SecFxp(bits)
        xs = [sec(float(v)) for v in vec]
        return await mpc.output(mpc.argmax(xs)[0])
    return int(_run(f))


def secure_maxpool(planes, pool, *, bits: int = 64):
    """Max-pooling on secret-shared values via MPyC secure ``max`` per window.

    ``planes`` is a decrypted ``(C, H, W)`` array (the conv/ReLU output crossing the FHE→MPC
    boundary); ``pool`` is a :class:`qsh.mpc.layers.PoolSpec`. Each pooling window's maximum is
    computed under secret-sharing (genuine secure comparison, like ReLU/argmax). Returns
    ``(pooled (C, Hout, Wout), n_comparisons)`` — ``n_comparisons`` counts ``(win-1)`` per
    window for the MPC-communication estimate.
    """
    x = np.asarray(planes, dtype=float)
    c, h, w = x.shape
    size, stride = int(pool.size), int(pool.stride)
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    windows = [
        x[ci, i * stride: i * stride + size, j * stride: j * stride + size].ravel()
        for ci in range(c) for i in range(ho) for j in range(wo)
    ]

    async def f(mpc):
        sec = mpc.SecFxp(bits)
        maxima = [mpc.max([sec(float(v)) for v in win]) for win in windows]
        return await mpc.output(maxima)

    pooled = np.asarray([float(v) for v in _run(f)], dtype=float).reshape(c, ho, wo)
    n_compare = len(windows) * max(size * size - 1, 0)
    return pooled, n_compare


def _fhe_linear(backend, x, W, b):
    """One linear layer W·x + b evaluated under CKKS on ``backend`` (encrypt → dot → decrypt)."""
    ct = backend.encrypt(np.asarray(x, dtype=float))
    return [backend.decrypt(backend.dot_plain(ct, W[j]))[0] + float(b[j])
            for j in range(W.shape[0])]


def hybrid_private_inference(layers, x, backend, *, use_mpc: bool = True) -> dict:
    """Run a small network as **FHE-linear + MPC-non-linear** and report the per-stage split.

    ``layers`` may be either the legacy ``list[(W (out,in), b (out,))]`` (a dense MLP — ReLU
    between layers, argmax on the final logits) **or** a list of tagged specs
    (:mod:`qsh.mpc.layers`: ``ConvSpec``/``DenseSpec``/``PoolSpec``/``FlattenSpec``) for a CNN.
    :func:`~qsh.mpc.layers.normalize_layers` upgrades the legacy form, so the **dense path is
    behaviour-identical** to before.

    Dispatch:
      * ``ConvSpec``  → genuine encrypted convolution (``fhe.conv.fhe_conv2d``) → ``fhe_linear_ms``
      * ``DenseSpec`` → ``_fhe_linear`` (CKKS matmul)                            → ``fhe_linear_ms``
      * ReLU after conv/dense → ``secure_relu`` (MPyC)                           → ``mpc_nonlinear_ms``
      * ``PoolSpec`` avg → linear (FHE-side, no comm); max → ``secure_maxpool``  → ``mpc_nonlinear_ms``
      * final decision → ``secure_argmax`` (MPyC)                               → ``mpc_nonlinear_ms``

    With ``use_mpc=False`` the non-linear ops run in plaintext (the client-demo fallback).
    """
    from ..fhe.conv import avgpool2d, fhe_conv2d, maxpool2d
    from .layers import (
        ConvSpec,
        DenseSpec,
        FlattenSpec,
        PoolSpec,
        normalize_layers,
        plaintext_forward,
    )

    specs = normalize_layers(layers)
    arch_class = "cnn" if any(isinstance(s, ConvSpec) for s in specs) else "mlp"

    h = np.asarray(x, dtype=float)
    if arch_class == "mlp":
        h = h.ravel()
    fhe_ms = mpc_ms = 0.0
    n_compare = 0
    logits = None

    for s in specs:
        if isinstance(s, DenseSpec):
            hv = h.ravel()[: s.weight.shape[1]]
            t = time.perf_counter()
            z = _fhe_linear(backend, hv, s.weight, s.bias)          # linear under FHE (GPU-capable)
            fhe_ms += (time.perf_counter() - t) * 1000
            if s.activation == "relu":                              # hidden ReLU under MPC
                t = time.perf_counter()
                h = np.asarray(secure_relu(z) if use_mpc else np.maximum(0.0, z), dtype=float)
                mpc_ms += (time.perf_counter() - t) * 1000
                n_compare += len(z)
            else:                                                   # final logits
                h = np.asarray(z, dtype=float)
                logits = z
        elif isinstance(s, ConvSpec):
            t = time.perf_counter()
            planes = fhe_conv2d(backend, h.reshape(s.in_shape), s)  # genuine encrypted conv
            fhe_ms += (time.perf_counter() - t) * 1000
            if s.activation == "relu":                              # ReLU on the conv output (MPC)
                t = time.perf_counter()
                flat = planes.ravel()
                relu = secure_relu(flat) if use_mpc else np.maximum(0.0, flat)
                planes = np.asarray(relu, dtype=float).reshape(planes.shape)
                mpc_ms += (time.perf_counter() - t) * 1000
                n_compare += int(planes.size)
            h = planes
        elif isinstance(s, PoolSpec):
            if s.pool == "max" and use_mpc:                         # max-pool under MPC
                t = time.perf_counter()
                h, nc = secure_maxpool(h, s)
                mpc_ms += (time.perf_counter() - t) * 1000
                n_compare += nc
            else:                                                   # avg-pool is linear → FHE-side
                t = time.perf_counter()
                h = avgpool2d(h, s.size, s.stride) if s.pool == "avg" else maxpool2d(h, s.size, s.stride)
                fhe_ms += (time.perf_counter() - t) * 1000
        elif isinstance(s, FlattenSpec):
            h = h.ravel()

    if logits is None:                                             # network ended without a final dense
        logits = np.asarray(h, dtype=float).ravel().tolist()

    t = time.perf_counter()                                        # decision: argmax under MPC
    pred = secure_argmax(logits) if use_mpc else int(np.argmax(logits))
    mpc_ms += (time.perf_counter() - t) * 1000
    n_compare += len(logits)

    pt = plaintext_forward(specs, x)                              # plaintext reference (decision-consistency)
    pt_pred = int(np.argmax(pt))

    comm_kb = round(n_compare * _mpc_comm_bytes_per_compare() / 1024, 2) if use_mpc else 0.0
    return {
        "prediction": int(pred),
        "decision_matches_plaintext": int(pred) == pt_pred,
        "logits": [round(float(v), 6) for v in logits],
        "fhe_linear_ms": round(fhe_ms, 3),
        "mpc_nonlinear_ms": round(mpc_ms, 3),
        "mpc_comm_kb_estimated": comm_kb,
        "mpc_comparisons": n_compare,
        "mpc_engine": "mpyc-shamir" if use_mpc else "plaintext-fallback",
        "n_layers": len(specs),
        "arch_class": arch_class,
    }
