"""Genuine FHE convolution via im2col + CKKS dot products.

The CKKS backends (:mod:`qsh.fhe.backends`) expose ``dot_plain(ct, vec)`` =
``<encrypted vector, plaintext vector>`` returned in slot 0 — which is *exactly* a
convolution patch dot-product. So a real **encrypted** convolution needs no new crypto and
no ciphertext rotations: im2col the (encrypted) image into receptive-field patches and, for
each output ``(channel oc, spatial position p)``, compute
``dot_plain(patch_ct[p], kernel[oc]) + bias[oc]``.

This is the same primitive the dense layer (``_fhe_linear``) already uses — the **image
patches are the encrypted operand**, the **kernels are plaintext server weights** (just like
the dense ``W[j]`` rows). Decrypting each conv output before the next non-linear layer is the
documented FHE↔MPC boundary (decrypt-and-secret-share for ReLU/maxpool, re-encrypt for the
next linear layer — EncFormer's "stage-compatible patterns").

Cost is honest and bounded: a tiny 14×14 / 1-block / 4-channel CNN is ``14·14`` patches ×
``4`` out-channels = 196 encrypts + 784 ``dot_plain`` calls → seconds on TenSEAL CPU
(cf. FFConv ~0.37 s/MNIST image, HTCNN 163 images in 10.4 s on one CPU thread). If a larger
tier is slow the lever is *optimization* (batch/SIMD slot packing) — never a silent fallback.
"""

from __future__ import annotations

import numpy as np

from ..mpc.layers import ConvSpec


def _pad(x_chw: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return x_chw
    return np.pad(x_chw, ((0, 0), (padding, padding), (padding, padding)))


def out_hw(conv: ConvSpec) -> tuple[int, int]:
    """Output spatial size ``(Hout, Wout)`` for ``conv`` over its ``in_shape``."""
    _, h, w = conv.in_shape
    kh, kw = int(conv.weight.shape[2]), int(conv.weight.shape[3])
    ho = (h + 2 * conv.padding - kh) // conv.stride + 1
    wo = (w + 2 * conv.padding - kw) // conv.stride + 1
    return ho, wo


def im2col_patches(x_chw, conv: ConvSpec) -> tuple[np.ndarray, tuple[int, int]]:
    """im2col: each receptive field flattened **channel-major** to ``in_ch·kh·kw``.

    Returns ``(patches (n_pos, in_ch·kh·kw), (Hout, Wout))`` where ``n_pos = Hout·Wout`` in
    row-major (``oh``-major) order. The flatten order matches :func:`flatten_kernels` so the
    dot product equals the convolution.
    """
    x = np.asarray(x_chw, dtype=float).reshape(conv.in_shape)
    xp = _pad(x, conv.padding)
    _, kh, kw = int(conv.weight.shape[1]), int(conv.weight.shape[2]), int(conv.weight.shape[3])
    ho, wo = out_hw(conv)
    s = conv.stride
    plen = int(conv.weight.shape[1]) * kh * kw
    patches = np.empty((ho * wo, plen), dtype=float)
    p = 0
    for oh in range(ho):
        for ow in range(wo):
            recf = xp[:, oh * s: oh * s + kh, ow * s: ow * s + kw]   # (in_ch, kh, kw)
            patches[p] = recf.ravel()                                # channel-major (C, kh, kw)
            p += 1
    return patches, (ho, wo)


def flatten_kernels(conv: ConvSpec) -> np.ndarray:
    """``(out_ch, in_ch·kh·kw)`` — each row a kernel flattened to match the im2col patch order."""
    oc = int(conv.weight.shape[0])
    return conv.weight.reshape(oc, -1)


def conv2d_plaintext(x_chw, conv: ConvSpec) -> np.ndarray:
    """Numpy reference convolution → ``(out_ch, Hout, Wout)``.

    Uses the *same* im2col + kernel flatten as :func:`fhe_conv2d`, so the encrypted result
    matches this to within CKKS noise.
    """
    patches, (ho, wo) = im2col_patches(x_chw, conv)
    K = flatten_kernels(conv)                              # (oc, plen)
    out = patches @ K.T + conv.bias                        # (n_pos, oc)
    return out.T.reshape(int(conv.weight.shape[0]), ho, wo)


def fhe_conv2d(backend, x_chw, conv: ConvSpec, *, server_share=None, mask=None) -> np.ndarray:
    """Genuine encrypted convolution → ``(out_ch, Hout, Wout)`` (decrypted result).

    For each im2col patch: encrypt it, then for each output channel compute
    ``decrypt(dot_plain(patch_ct, kernel[oc]))[0] + bias[oc]``. The convolution arithmetic
    happens on the ciphertext; only the per-output scalar is decrypted (exactly as the dense
    ``_fhe_linear`` decrypts each logit).

    With ``mask`` (shape ``(out_ch, Hout, Wout)``) this is the hybrid's masked conversion: the key
    holder encrypts ITS share ``x_chw``, the server adds ``server_share`` (same shape) inside each
    patch's ciphertext, and the bias plus the server's mask go in before decryption, so what is
    decrypted is ``conv(x) + mask`` and never ``conv(x)``.
    """
    patches, (ho, wo) = im2col_patches(x_chw, conv)
    K = flatten_kernels(conv)                              # (oc, plen)
    oc = int(conv.weight.shape[0])
    out = np.empty((patches.shape[0], oc), dtype=float)
    if mask is None:
        if server_share is not None:
            raise ValueError("a server share without a mask would decrypt the unmasked value")
        for p in range(patches.shape[0]):
            ct = backend.encrypt(patches[p])              # encrypt the receptive field once
            for j in range(oc):
                out[p, j] = backend.decrypt(backend.dot_plain(ct, K[j]))[0] + float(conv.bias[j])
        return out.T.reshape(oc, ho, wo)
    m = np.asarray(mask, dtype=float).reshape(oc, ho * wo).T     # (patch, channel)
    server = None if server_share is None else im2col_patches(server_share, conv)[0]
    for p in range(patches.shape[0]):
        ct = backend.encrypt(patches[p])                  # the key holder encrypts its share
        if server is not None:
            ct = backend.add_plain(ct, server[p])         # the server adds its own
        for j in range(oc):
            masked = backend.add_plain(backend.dot_plain(ct, K[j]), float(conv.bias[j]) + float(m[p, j]))
            out[p, j] = backend.decrypt(masked)[0]
    return out.T.reshape(oc, ho, wo)


def conv_cost(conv: ConvSpec) -> dict:
    """Op counts for one conv layer (for the benchmark cost model / projection)."""
    oc, ic, kh, kw = (int(v) for v in conv.weight.shape)
    ho, wo = out_hw(conv)
    return {
        "encrypts": ho * wo,                              # one ciphertext per patch
        "dot_plain": ho * wo * oc,                        # one dot per (patch, out-channel)
        "macs": ho * wo * oc * ic * kh * kw,
        "out_shape": (oc, ho, wo),
    }


def avgpool2d(x_chw, size: int = 2, stride: int = 2) -> np.ndarray:
    """Average pooling (linear — runs FHE-side, no MPC). Returns ``(C, Hout, Wout)``."""
    x = np.asarray(x_chw, dtype=float)
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    out = np.empty((c, ho, wo), dtype=float)
    for i in range(ho):
        for j in range(wo):
            out[:, i, j] = x[:, i * stride: i * stride + size, j * stride: j * stride + size].mean(axis=(1, 2))
    return out


def maxpool2d(x_chw, size: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling — plaintext reference (the secure version is ``mpc.secure_maxpool``)."""
    x = np.asarray(x_chw, dtype=float)
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    out = np.empty((c, ho, wo), dtype=float)
    for i in range(ho):
        for j in range(wo):
            out[:, i, j] = x[:, i * stride: i * stride + size, j * stride: j * stride + size].max(axis=(1, 2))
    return out


def pool_windows(x_chw, size: int = 2, stride: int = 2):
    """Yield ``(c, i, j, window_values_flat)`` for each pooling window — used by secure_maxpool."""
    x = np.asarray(x_chw, dtype=float)
    c, h, w = x.shape
    ho, wo = (h - size) // stride + 1, (w - size) // stride + 1
    for ci in range(c):
        for i in range(ho):
            for j in range(wo):
                win = x[ci, i * stride: i * stride + size, j * stride: j * stride + size].ravel()
                yield ci, i, j, win, (ho, wo)
