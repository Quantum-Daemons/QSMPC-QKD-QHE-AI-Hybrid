"""Tagged layer specs for the hybrid private-inference engine.

The hybrid engine (:mod:`qsh.mpc.secret_inference`) evaluates a small neural network as
**linear layers under FHE (CKKS)** + **non-linear layers under MPC (MPyC secret-sharing)**.
To support genuine *convolutional* models — not only dense MLPs — it dispatches on these
*tagged specs* instead of bare ``(W, b)`` tuples:

  * :class:`DenseSpec`   -- fully-connected ``W·x + b``        (linear  → FHE), optional ReLU (→ MPC)
  * :class:`ConvSpec`    -- 2D convolution                      (linear  → FHE), optional ReLU (→ MPC)
  * :class:`PoolSpec`    -- avg-pool (linear, FHE-side) or max-pool (secure compare → MPC)
  * :class:`FlattenSpec` -- ``(C,H,W) → (C·H·W,)``              (structural, free)

:func:`normalize_layers` upgrades the *legacy* ``list[(W, b)]`` representation to a
``DenseSpec`` list (so every existing caller keeps working unchanged — the dense fast-path is
byte-identical), and :func:`plaintext_forward` is the numpy reference used to check that the
encrypted/secret-shared run preserves the plaintext decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DenseSpec:
    """Fully-connected layer ``W·x + b`` (linear → FHE), optional ReLU after (→ MPC)."""

    weight: np.ndarray            # (out, in)
    bias: np.ndarray              # (out,)
    activation: str = "relu"      # "relu" (under MPC) | "none"

    def __post_init__(self) -> None:
        self.weight = np.atleast_2d(np.asarray(self.weight, dtype=float))
        self.bias = np.asarray(self.bias, dtype=float).ravel()


@dataclass
class ConvSpec:
    """2D convolution (linear → FHE), optional ReLU after (→ MPC).

    ``weight`` is ``(out_ch, in_ch, kh, kw)``; ``in_shape`` is this layer's input ``(C, H, W)``.
    """

    weight: np.ndarray            # (out_ch, in_ch, kh, kw)
    bias: np.ndarray              # (out_ch,)
    in_shape: tuple               # (C, H, W) of this layer's input
    stride: int = 1
    padding: int = 0
    activation: str = "relu"

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=float)
        self.bias = np.asarray(self.bias, dtype=float).ravel()
        self.in_shape = tuple(int(v) for v in self.in_shape)


@dataclass
class PoolSpec:
    """Spatial pooling. ``avg`` → linear (FHE-side, no comm); ``max`` → secure compare (MPC)."""

    pool: str = "avg"             # "avg" | "max"
    size: int = 2
    stride: int = 2


@dataclass
class FlattenSpec:
    """``(C,H,W) → (C·H·W,)``. Structural, free."""


# the spec types, as a tuple for isinstance() dispatch
LAYER_SPECS = (DenseSpec, ConvSpec, PoolSpec, FlattenSpec)


def normalize_layers(layers):
    """Upgrade a legacy ``list[(W, b)]`` to a ``DenseSpec`` list (ReLU between, none on last).

    Idempotent: a list already containing specs is returned unchanged. This is the
    backward-compat seam — ``runner.usecase_arch_layers`` / ``_tier_mlp`` keep returning
    ``(W, b)`` tuples and every existing test/caller works without modification.
    """
    items = list(layers)
    if not items:
        return items
    if isinstance(items[0], LAYER_SPECS):
        return items
    specs: list = []
    n = len(items)
    for i, (W, b) in enumerate(items):
        specs.append(DenseSpec(W, b, activation="relu" if i < n - 1 else "none"))
    return specs


def is_conv_stack(layers) -> bool:
    """True if the (normalized) stack contains any convolution — i.e. a CNN, not an MLP."""
    return any(isinstance(s, ConvSpec) for s in normalize_layers(layers))


def plaintext_forward(layers, x):
    """Numpy reference forward pass over specs → final logits (1-D vector).

    Used for decision-consistency: the encrypted FHE-linear + secret-shared MPC-non-linear run
    must yield the same argmax as this plaintext reference.
    """
    # lazy import avoids an import cycle (conv.py imports ConvSpec from this module)
    from ..fhe.conv import avgpool2d, conv2d_plaintext, maxpool2d

    h = np.asarray(x, dtype=float)
    for s in normalize_layers(layers):
        if isinstance(s, DenseSpec):
            h = s.weight @ h.ravel()[: s.weight.shape[1]] + s.bias
            if s.activation == "relu":
                h = np.maximum(0.0, h)
        elif isinstance(s, ConvSpec):
            h = conv2d_plaintext(h.reshape(s.in_shape), s)
            if s.activation == "relu":
                h = np.maximum(0.0, h)
        elif isinstance(s, PoolSpec):
            h = avgpool2d(h, s.size, s.stride) if s.pool == "avg" else maxpool2d(h, s.size, s.stride)
        elif isinstance(s, FlattenSpec):
            h = h.ravel()
    return h.ravel()
