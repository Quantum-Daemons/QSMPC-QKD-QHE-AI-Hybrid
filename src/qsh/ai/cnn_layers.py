"""Bridge a (trained) PyTorch ``SmallCNN`` to the hybrid engine's layer specs.

The genuine-FHE convolution path (:mod:`qsh.fhe.conv` + :mod:`qsh.mpc.secret_inference`) evaluates
a network expressed as :mod:`qsh.mpc.layers` specs. :func:`smallcnn_to_specs` walks a trained
``SmallCNN`` and emits that spec list **using the trained weights**, so the encrypted run is a real
inference of the real model (not a random-weight stand-in). :func:`small_medmnist_cnn` builds the
tiny FHE-feasible configuration, :func:`downsample` shrinks images to the FHE-feasible resolution,
and :func:`random_cnn_specs` builds a representative-weight stack for the standalone cost benchmark.
"""

from __future__ import annotations

import numpy as np


def downsample(x, size: int):
    """Area-resize images to ``(size, size)``. Accepts ``(C,H,W)`` or a batch ``(N,C,H,W)``."""
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    single = t.ndim == 3
    if single:
        t = t.unsqueeze(0)
    t = F.interpolate(t, size=(int(size), int(size)), mode="area")
    out = t.cpu().numpy()
    return out[0] if single else out


def small_medmnist_cnn(in_ch: int = 1, n_classes: int = 2, img_size: int = 14,
                       channels: tuple[int, ...] = (4,), hidden: int = 16, pool: str = "avg"):
    """The tiny FHE-feasible CNN: 1 conv block (in->4 ch, 3x3, pad 1) + avg-pool/2 + small head.

    With ``channels=(4,)`` it is one block (FHE-feasible, ~seconds encrypted); ``channels=(4, 8)``
    is the two-block variant used as the projected 'large' tier.
    """
    from .models import SmallCNN

    return SmallCNN(n_classes=n_classes, in_ch=in_ch, channels=channels, pool=pool,
                    img_size=img_size, hidden=hidden, one_block=(len(channels) == 1))


def smallcnn_to_specs(model, in_shape):
    """Convert a trained ``SmallCNN`` into a hybrid spec list (ConvSpec/PoolSpec/FlattenSpec/DenseSpec).

    ``in_shape`` is the input ``(C, H, W)``. Conv/Linear weights are taken from the trained model;
    a following activation module tags the preceding layer's ``activation="relu"`` (run under MPC).
    The final dense layer is forced to ``activation="none"`` (those are the logits).
    """
    import torch.nn as nn

    from ..mpc.layers import ConvSpec, DenseSpec, FlattenSpec, PoolSpec

    specs: list = []
    c, h, w = (int(v) for v in in_shape)
    last = None                                              # conv/dense awaiting an activation tag
    for m in list(model.features) + list(model.classifier):
        if isinstance(m, nn.Conv2d):
            W = m.weight.detach().cpu().numpy()
            b = m.bias.detach().cpu().numpy() if m.bias is not None else np.zeros(W.shape[0])
            pad = m.padding[0] if isinstance(m.padding, tuple) else int(m.padding)
            stride = m.stride[0] if isinstance(m.stride, tuple) else int(m.stride)
            spec = ConvSpec(W, b, in_shape=(c, h, w), stride=stride, padding=pad, activation="none")
            specs.append(spec)
            last = spec
            kh, kw = int(W.shape[2]), int(W.shape[3])
            h = (h + 2 * pad - kh) // stride + 1
            w = (w + 2 * pad - kw) // stride + 1
            c = int(W.shape[0])
        elif isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            b = m.bias.detach().cpu().numpy()
            spec = DenseSpec(W, b, activation="none")
            specs.append(spec)
            last = spec
        elif isinstance(m, nn.AvgPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else int(m.kernel_size[0])
            specs.append(PoolSpec(pool="avg", size=k, stride=k))
            h, w, last = h // k, w // k, None
        elif isinstance(m, nn.MaxPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else int(m.kernel_size[0])
            specs.append(PoolSpec(pool="max", size=k, stride=k))
            h, w, last = h // k, w // k, None
        elif isinstance(m, nn.Flatten):
            specs.append(FlattenSpec())
            last = None
        else:                                               # an activation (ReLU / Square)
            if last is not None:
                last.activation = "relu"
    if specs and hasattr(specs[-1], "activation"):
        specs[-1].activation = "none"                       # final logits layer: no activation
    return specs


def random_cnn_specs(in_shape=(1, 14, 14), channels=(4,), hidden: int = 16, pool: str = "avg",
                     seed: int = 0):
    """Representative-weight CNN spec stack for the standalone cost benchmark.

    The encrypted hybrid preserves the plaintext decision for *any* weights, so random weights are
    faithful for the FHE-linear/MPC-non-linear **cost** metric (the live medical run uses the real
    trained weights via :func:`smallcnn_to_specs`).
    """
    import torch

    torch.manual_seed(seed)
    model = small_medmnist_cnn(in_ch=int(in_shape[0]), img_size=int(in_shape[1]),
                               channels=channels, hidden=hidden, pool=pool)
    return smallcnn_to_specs(model, in_shape)
