"""Secure multiparty computation (MPC) layer.

Houses the *non-linear* half of the hybrid private-inference split: linear layers run under
FHE (CKKS, GPU-capable), non-linear layers (ReLU, argmax, comparison) run under secret-sharing
MPC — the standard 2026 architecture (EncFormer; Gazelle→Delphi→BOLT→Bumblebee→BLB;
RhombusEnd2End/HEonGPU). The MPC engine is **MPyC** (MIT, pure-Python, Shamir secret sharing
with secure comparison), installed from PyPI (the wheel is the same code as upstream lschoe/mpyc;
the submodule was dropped in Round 17 as redundant). MP-SPDZ + hpmpc/PIGEON remain as
heavyweight production references.
"""

from .layers import (
    ConvSpec,
    DenseSpec,
    FlattenSpec,
    PoolSpec,
    is_conv_stack,
    normalize_layers,
    plaintext_forward,
)
from .secret_inference import (
    hybrid_private_inference,
    mpyc_available,
    secure_argmax,
    secure_maxpool,
    secure_relu,
)

__all__ = [
    # spec types + helpers (qsh.mpc.layers)
    "ConvSpec",
    "DenseSpec",
    "FlattenSpec",
    "PoolSpec",
    "normalize_layers",
    "is_conv_stack",
    "plaintext_forward",
    # hybrid engine (qsh.mpc.secret_inference)
    "hybrid_private_inference",
    "mpyc_available",
    "secure_argmax",
    "secure_maxpool",
    "secure_relu",
]
