"""Hybrid private inference: linear layers under FHE (CKKS), non-linear under MPC (MPyC).

The 2026-standard secure-inference split (EncFormer, npj/IACR 2026; the Gazelle→Delphi→BOLT→
Bumblebee→BLB lineage; the RhombusEnd2End/HEonGPU reference): **linear** ops (matmul/conv) run
homomorphically under CKKS, while **non-linear** ops (ReLU, argmax,
comparison) run under **secret-sharing MPC** — they are cheap homomorphically only as low-degree
polynomials, so a real hybrid delegates them to MPC. Here the CKKS half runs on the configured CPU
backend (TenSEAL or OpenFHE): HEonGPU is a build-gated reference adapter with no Python API
(qsh.fhe.backends.HeongpuBackend raises), and no HEonGPU run is measured in this repository.

The non-linear half uses **MPyC** (MIT, Shamir secret sharing + secure comparison), run as THREE
parties in separate processes over loopback TCP (:mod:`qsh.mpc.parties`): threshold t = 1,
statistical security 40, and the bytes on the wire MEASURED. A real deployment runs one party per
host, so these timings are loopback figures.

THE CONVERSIONS ARE MASKED IN SLOT 0 ONLY, AND THE CONVERSION IS NOT MODEL-PRIVATE. The roles: a
CLIENT, who holds the CKKS key and the input; a SERVER, who holds the model; a HELPER, the third
Shamir party.
- FHE to MPC: the server evaluates ``W h + b`` on the client's ciphertext and adds a fresh mask ``r``
  INSIDE the ciphertext, and the decrypt call returns slot 0, ``z + r``. The MPC runs on the sum of
  the client's ``z + r`` and the server's ``-r``. WHAT THAT DOES NOT HIDE, confirmed locally on the
  TenSEAL backend: ``_fhe_linear`` adds ``b + r`` as a SCALAR, which TenSEAL adds to EVERY slot, and
  ``ct.dot`` leaves running partial sums of the products ``x_i * w_i`` in the other slots. The
  client holds the secret key, so it can decode the whole plaintext: differences between slots
  cancel ``r`` and give ``z`` and each product ``x_i * w_i``, and since the client knows ``x`` it
  recovers the server's weight row from ONE query. So the statement "the client decrypts ``z + r``
  and never ``z``" holds for slot 0 only. The fix (independent uniform masks in every slot, or a
  one-hot selector before masking, plus noise flooding, and a test that decodes all slots) is not
  made here; ``tests/test_mpc.py`` checks slot 0 only. The convolution path adds its mask the same
  way and has not been checked slot by slot.
- MPC to FHE: the MPC adds a second server mask ``s`` to its result and opens ``a + s`` to the client
  alone. The client encrypts that, and the server adds ``-s`` inside the ciphertext before the next
  layer, so neither party holds ``a`` in the clear; the next linear layer's ciphertext then carries
  the same slot leak as above.
- The decision: the MPC opens the class index to the client, and nothing else; the CKKS slots
  around it are the gap described above.
Linear steps between non-linear ones (average pooling, flattening) run on each party's share
locally, which an additive sharing allows. This is the shape of Gazelle's and Delphi's conversions;
they mask exactly over a ring, and this one masks over the reals because CKKS is approximate.

What real masks give, IN SLOT 0: ``r`` is uniform on ``[-2^b, 2^b)`` from the OS CSPRNG,
``b = mpc_mask_bits``. Seen through it, two candidate values within ``B`` of zero are ``B / 2^b``
apart in statistical distance, per value, and the slot-0 view of a run at most ``n`` times that for
``n`` masked values. Each run reports ``n``, its largest ``|value|`` and that bound; the bound says
nothing about the other CKKS slots, which leak as described above. Statistical, not perfect, and
smaller than the MPC's own 2^-40 for the shipped b = 30: the mask must fit 64-bit fixed point (its
integer part holds 2^31), and larger masks cost precision (measured: 5e-7 at 2^30, 8e-6 at 2^36,
2e-4 at 2^40 through a CKKS round trip on TenSEAL).

UNTIL ROUND 75 the conversions were decrypt-then-share and open-to-everyone: the client decrypted
each linear output in the clear, party 0 secret-shared it, and every party received each ReLU
result. Whoever held the CKKS key saw every intermediate, which is the precondition of the
weight-recovery attack docs/architecture.md cites (IACR eprint 2026/1527), while that document said
it was absent. The bias was also added by the client after decryption, outside the ciphertext.
Before that the MPC itself was ONE in-process party (m = 1, t = 0), where a share is the value.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

# keep MPyC quiet inside the benchmark
logging.getLogger("mpyc").setLevel(logging.ERROR)


def mpyc_available() -> bool:
    try:
        import mpyc  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def _mpc_settings() -> tuple[int, int, int]:
    """(parties, threshold, statistical security) from config - one source for every MPC call."""
    from ..config import get_config
    b = get_config().benchmark
    return int(b.mpc_parties), int(b.mpc_threshold), int(b.mpc_sec_param)


def _mask_bits(bits: int) -> int:
    """The mask magnitude, checked against the fixed-point range it has to fit."""
    from ..config import get_config
    b = int(get_config().benchmark.mpc_mask_bits)
    # SecFxp(bits) keeps bits // 2 fractional bits, so its integer part holds 2^(bits // 2 - 1);
    # a mask of 2^b plus the value it hides must stay inside that.
    if not 1 <= b <= bits // 2 - 2:
        raise ValueError(f"mpc_mask_bits = {b} does not fit {bits}-bit fixed point (at most {bits // 2 - 2})")
    return b


def _masks(shape, bits: int) -> np.ndarray:
    """Masks uniform on [-2^bits, 2^bits), from the OS CSPRNG: the client must not predict them."""
    n = int(np.prod(shape))
    u = (np.frombuffer(os.urandom(8 * n), dtype=np.uint64) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    return ((2.0 * u - 1.0) * 2.0 ** bits).reshape(shape)


def _secure(op: str, *, stats: list | None = None, bits: int = 64, **job):
    """Run one secure operation with the configured parties; append what it measured to `stats`."""
    from .parties import run_parties
    m, t, k = _mpc_settings()
    run = run_parties(op, m=m, threshold=t, sec_param=k, bits=bits, **job)
    if stats is not None:
        stats.append(run)
    return run["result"]


def _listed(v) -> list[float] | None:
    return None if v is None else [float(a) for a in np.asarray(v, dtype=float).ravel()]


def secure_relu(vec, *, server_share=None, out_mask=None, bits: int = 64,
                stats: list | None = None) -> list[float]:
    """ReLU via MPyC secure comparison, ReLU([x]) = [x >= 0]·[x], on ``x = vec + server_share``.

    Opened to the client alone, with ``out_mask`` added first when the server gives one.
    """
    return [float(v) for v in _secure("relu", values=_listed(vec), server_share=_listed(server_share),
                                      out_mask=_listed(out_mask), bits=bits, stats=stats)]


def secure_argmax(vec, *, server_share=None, bits: int = 64, stats: list | None = None) -> int:
    """argmax via MPyC secure comparisons on ``x = vec + server_share``: the index, to the client."""
    return int(_secure("argmax", values=_listed(vec), server_share=_listed(server_share),
                       bits=bits, stats=stats))


def _windows(x: np.ndarray, size: int, stride: int) -> list[list[float]]:
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    return [x[ci, i * stride: i * stride + size, j * stride: j * stride + size].ravel().tolist()
            for ci in range(c) for i in range(ho) for j in range(wo)]


def secure_maxpool(planes, pool, *, server_share=None, out_mask=None, bits: int = 64,
                   stats: list | None = None):
    """Max-pooling via MPyC secure ``max`` per window, on ``planes + server_share``.

    ``planes`` and ``server_share`` are ``(C, H, W)``; ``pool`` is a :class:`qsh.mpc.layers.PoolSpec`.
    ``out_mask``, shaped like the result, is added before the client receives it. Returns
    ``(pooled (C, Hout, Wout), n_comparisons)``, counting ``size * size - 1`` per window.
    """
    x = np.asarray(planes, dtype=float)
    c, h, w = x.shape
    size, stride = int(pool.size), int(pool.stride)
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    windows = _windows(x, size, stride)
    server = None if server_share is None else [
        v for win in _windows(np.asarray(server_share, dtype=float).reshape(c, h, w), size, stride) for v in win]
    maxima = _secure("maxpool", windows=windows, server_share=server, out_mask=_listed(out_mask),
                     bits=bits, stats=stats)
    pooled = np.asarray([float(v) for v in maxima], dtype=float).reshape(c, ho, wo)
    return pooled, len(windows) * max(size * size - 1, 0)


def _fhe_linear(backend, x, W, b, *, server_share=None, mask=None):
    """One linear layer ``W·x + b`` under CKKS on ``backend``.

    Unmasked (``mask=None``): encrypt, dot, decrypt, then add the bias - the plaintext fallback.
    Masked: the key holder encrypts its share ``x``, the server adds ``server_share`` and then, per
    output, the bias and its mask inside the ciphertext, so slot 0 decrypts to
    ``W @ (x + share) + b + mask``. The bias and mask go in as one SCALAR, which the backend adds
    to every slot, so the other slots still let the key holder recover the unmasked value and the
    weights (module docstring). Slot 0 is the only slot this masks.
    """
    ct = backend.encrypt(np.asarray(x, dtype=float))
    if mask is None:
        if server_share is not None:
            raise ValueError("a server share without a mask would decrypt the unmasked value")
        return [backend.decrypt(backend.dot_plain(ct, W[j]))[0] + float(b[j]) for j in range(W.shape[0])]
    if server_share is not None:
        ct = backend.add_plain(ct, np.asarray(server_share, dtype=float))
    return [backend.decrypt(backend.add_plain(backend.dot_plain(ct, W[j]), float(b[j]) + float(mask[j])))[0]
            for j in range(W.shape[0])]


def hybrid_private_inference(layers, x, backend, *, use_mpc: bool = True) -> dict:
    """Run a small network as **FHE-linear + MPC-non-linear** and report the per-stage split.

    ``layers`` may be either the legacy ``list[(W (out,in), b (out,))]`` (a dense MLP — ReLU
    between layers, argmax on the final logits) **or** a list of tagged specs
    (:mod:`qsh.mpc.layers`: ``ConvSpec``/``DenseSpec``/``PoolSpec``/``FlattenSpec``) for a CNN.

    Dispatch, with the masked conversions of the module docstring between the two halves:
      * ``ConvSpec``  → genuine encrypted convolution (``fhe.conv.fhe_conv2d``) → ``fhe_linear_ms``
      * ``DenseSpec`` → ``_fhe_linear`` (CKKS matmul)                            → ``fhe_linear_ms``
      * ReLU after conv/dense → ``secure_relu`` (MPyC)                           → ``mpc_nonlinear_ms``
      * ``PoolSpec`` avg → on each share locally; max → ``secure_maxpool``     → ``mpc_nonlinear_ms``
      * final decision → ``secure_argmax`` (MPyC), the index to the client      → ``mpc_nonlinear_ms``

    With ``use_mpc=False`` the non-linear ops run in plaintext after an unmasked decryption: the
    fallback, with no privacy from the key holder.

    The harness plays all three roles in one process, so it can also reconstruct the logits and the
    largest masked value, reported to check fidelity and the masks' bound. The MPC opens only the
    class index to the client, but the client can see more than that: each masked linear output
    is masked in CKKS slot 0 only, and decoding the other slots with its key gives it the unmasked
    outputs and the weights (module docstring).
    """
    from .layers import ConvSpec, normalize_layers, plaintext_forward

    specs = normalize_layers(layers)
    arch_class = "cnn" if any(isinstance(s, ConvSpec) for s in specs) else "mlp"
    run = _masked_forward(specs, x, backend, arch_class) if use_mpc else _plain_forward(specs, x, backend, arch_class)
    pt = plaintext_forward(specs, x)                              # plaintext reference (decision-consistency)
    logits = np.asarray(run["logits"], dtype=float)
    m, t, k = _mpc_settings()
    runs = run["runs"]
    mpc_ms, overhead_ms = run["mpc_wall_ms"], 0.0
    if use_mpc:
        # The parties' own clock, from "all connected" to the opened result. The caller's clock also
        # counts starting the party processes, connecting them and shutting them down, once per
        # session: a cost of this harness that parties standing between requests would not pay, so it
        # is reported beside the protocol time rather than inside it.
        protocol_ms = sum(float(r["mpc_ms"]) for r in runs)
        overhead_ms, mpc_ms = max(0.0, mpc_ms - protocol_ms), protocol_ms
    out = {
        "prediction": int(run["prediction"]),
        "decision_matches_plaintext": int(run["prediction"]) == int(np.argmax(pt)),
        "logits": [round(float(v), 6) for v in logits],          # the simulator's reconstruction
        "logit_max_abs_error": float(np.max(np.abs(logits - pt))) if logits.size == pt.size else None,
        "fhe_linear_ms": round(run["fhe_ms"], 3),
        "mpc_nonlinear_ms": round(mpc_ms, 3),      # protocol time inside the parties (Round 75)
        "mpc_session_overhead_ms": round(overhead_ms, 3),   # process start, connect, shutdown
        "mpc_comm_kb": round(sum(r["bytes_sent"] for r in runs) / 1024, 2) if use_mpc else 0.0,
        "mpc_sessions": len(runs),
        "mpc_parties": m if use_mpc else 0,
        "mpc_threshold": t if use_mpc else 0,
        "mpc_sec_param": k if use_mpc else 0,
        "mpc_comparisons": run["n_compare"],
        "mpc_engine": f"mpyc-shamir, {m} parties (t={t}), loopback TCP" if use_mpc else "plaintext-fallback",
        # What the MPC OPENS to the client. Not everything the client can learn: the CKKS slots
        # beyond slot 0 are unmasked (module docstring). The string is kept as-is until that is
        # fixed, because tests and the benchmark row read it.
        "client_learns": "the class index" if use_mpc else "every intermediate value",
        "n_layers": len(specs),
        "arch_class": arch_class,
    }
    if use_mpc:
        b, n, top = run["mask_bits"], run["masked_values"], run["max_abs_masked"]
        out.update({
            "mask_bits": b, "masked_values": n, "max_abs_masked_value": round(top, 6),
            # per value B / 2^b; the slot-0 view of the run at most n times that. Not a bound on
            # the other CKKS slots, which are unmasked (module docstring).
            "mask_statistical_distance": float(n * top / 2.0 ** b),
        })
    return out


def _plain_forward(specs, x, backend, arch_class: str) -> dict:
    """The fallback: unmasked decryption and plaintext non-linear layers. No MPC, no privacy."""
    from ..fhe.conv import avgpool2d, fhe_conv2d, maxpool2d
    from .layers import ConvSpec, DenseSpec, FlattenSpec, PoolSpec

    h = np.asarray(x, dtype=float)
    if arch_class == "mlp":
        h = h.ravel()
    fhe_ms = mpc_ms = 0.0
    n_compare, logits = 0, None
    for s in specs:
        if isinstance(s, DenseSpec):
            t = time.perf_counter()
            z = np.asarray(_fhe_linear(backend, h.ravel()[: s.weight.shape[1]], s.weight, s.bias))
            fhe_ms += (time.perf_counter() - t) * 1000
            if s.activation == "relu":
                t = time.perf_counter()
                h = np.maximum(0.0, z)
                mpc_ms += (time.perf_counter() - t) * 1000
                n_compare += len(z)
            else:
                h = logits = z
        elif isinstance(s, ConvSpec):
            t = time.perf_counter()
            h = fhe_conv2d(backend, h.reshape(s.in_shape), s)
            fhe_ms += (time.perf_counter() - t) * 1000
            if s.activation == "relu":
                h = np.maximum(0.0, h)
                n_compare += int(h.size)
        elif isinstance(s, PoolSpec):
            t = time.perf_counter()
            h = avgpool2d(h, s.size, s.stride) if s.pool == "avg" else maxpool2d(h, s.size, s.stride)
            fhe_ms += (time.perf_counter() - t) * 1000
        elif isinstance(s, FlattenSpec):
            h = h.ravel()
    logits = np.asarray(h if logits is None else logits, dtype=float).ravel()
    n_compare += max(len(logits) - 1, 0)
    return {"prediction": int(np.argmax(logits)), "logits": logits, "fhe_ms": fhe_ms, "mpc_wall_ms": mpc_ms,
            "n_compare": n_compare, "runs": []}


def _masked_forward(specs, x, backend, arch_class: str, *, bits: int = 64) -> dict:
    """The protocol of the module docstring. ``hc`` is the client's share, ``hs`` the server's."""
    from ..fhe.conv import avgpool2d, fhe_conv2d, out_hw
    from .layers import ConvSpec, DenseSpec, FlattenSpec, PoolSpec

    mb = _mask_bits(bits)
    hc = np.asarray(x, dtype=float)                 # the client's input: its own, so unmasked
    if arch_class == "mlp":
        hc = hc.ravel()
    hs = None                                       # the server holds no share of it yet
    fhe_ms = mpc_ms = 0.0
    n_compare = masked = 0
    top = 0.0                                       # the simulator's largest |masked value|
    runs: list = []

    def track(value) -> None:
        nonlocal masked, top
        v = np.asarray(value, dtype=float)
        masked += int(v.size)
        top = max(top, float(np.max(np.abs(v))) if v.size else 0.0)

    def relu(y, r, shape):
        """MPC ReLU on (y, -r); the client gets a + s, the server keeps -s."""
        nonlocal mpc_ms, n_compare
        s_out = _masks(shape, mb)
        t = time.perf_counter()
        a_plus_s = np.asarray(secure_relu(y, server_share=-r, out_mask=s_out, bits=bits, stats=runs))
        mpc_ms += (time.perf_counter() - t) * 1000
        n_compare += int(np.prod(shape))
        track(a_plus_s.reshape(shape) - s_out)      # the ReLU output, masked on its way back
        return a_plus_s.reshape(shape), -s_out

    for s in specs:
        if isinstance(s, DenseSpec):
            n_in, n_out = s.weight.shape[1], s.weight.shape[0]
            c_in = hc.ravel()[:n_in]
            s_in = None if hs is None else hs.ravel()[:n_in]
            r = _masks((n_out,), mb)
            t = time.perf_counter()
            y = np.asarray(_fhe_linear(backend, c_in, s.weight, s.bias, server_share=s_in, mask=r))
            fhe_ms += (time.perf_counter() - t) * 1000
            track(y - r)                            # the linear output, masked on its way out
            if s.activation == "relu":
                hc, hs = relu(y, r, (n_out,))
            else:                                   # logits, held as shares (y, -r)
                hc, hs = y, -r
        elif isinstance(s, ConvSpec):
            ho, wo = out_hw(s)
            shape = (int(s.weight.shape[0]), ho, wo)
            r = _masks(shape, mb)
            t = time.perf_counter()
            y = fhe_conv2d(backend, hc.reshape(s.in_shape), s, mask=r,
                           server_share=None if hs is None else hs.reshape(s.in_shape))
            fhe_ms += (time.perf_counter() - t) * 1000
            track(y - r)
            if s.activation == "relu":
                hc, hs = relu(y.ravel(), r.ravel(), shape)
            else:
                hc, hs = y, -r
        elif isinstance(s, PoolSpec):
            if s.pool == "max":
                c, h, w = hc.shape
                ho, wo = (h - s.size) // s.stride + 1, (w - s.size) // s.stride + 1
                s_out = _masks((c, ho, wo), mb)
                t = time.perf_counter()
                pooled, nc = secure_maxpool(hc, s, server_share=hs, out_mask=s_out, bits=bits, stats=runs)
                mpc_ms += (time.perf_counter() - t) * 1000
                n_compare += nc
                track(pooled - s_out)
                hc, hs = pooled, -s_out
            else:                                   # linear: each party pools its own share
                t = time.perf_counter()
                hc = avgpool2d(hc, s.size, s.stride)
                hs = None if hs is None else avgpool2d(hs, s.size, s.stride)
                fhe_ms += (time.perf_counter() - t) * 1000
        elif isinstance(s, FlattenSpec):
            hc = hc.ravel()
            hs = None if hs is None else hs.ravel()

    c_logits = hc.ravel()
    s_logits = None if hs is None else hs.ravel()
    t = time.perf_counter()                         # the decision: the index, to the client only
    pred = secure_argmax(c_logits, server_share=s_logits, bits=bits, stats=runs)
    mpc_ms += (time.perf_counter() - t) * 1000
    n_compare += max(len(c_logits) - 1, 0)          # argmax of n is n - 1 comparisons
    logits = c_logits if s_logits is None else c_logits + s_logits    # the simulator's view only
    return {"prediction": pred, "logits": logits, "fhe_ms": fhe_ms, "mpc_wall_ms": mpc_ms,
            "n_compare": n_compare, "runs": runs, "mask_bits": mb, "masked_values": masked,
            "max_abs_masked": top}
