"""The server's masks on a ciphertext the key holder will decrypt: every slot, both parts (Round 84).

One implementation for the dense layer (``qsh.mpc.secret_inference._fhe_linear``) and the
convolution (``qsh.fhe.conv.fhe_conv2d``), because the defect this replaces lived in both: each
added its bias and mask as one scalar, which CKKS broadcasts to every slot, so the other slots kept
the rotation-sum's partial sums with the mask cancelling between them.

What the server adds to the ciphertext holding ``z`` in slot 0, before the key holder decrypts:

- slot 0's real part: ``slot0``, the bias plus the mask ``r`` it shares into the MPC as ``-r``;
- every other real part and every imaginary part: an independent mask, uniform on
  ``[-2^b, 2^b)`` from the OS CSPRNG, which the server keeps and never uses again.

The second line is also the noise flood. Besides the partial sums, the slots hold the CKKS error,
which depends on the computation, and the imaginary parts hold nothing else; a uniform mask of
half-width ``2^b`` over each of them hides an error ``e`` up to a statistical distance ``|e| / 2^b``,
so the flood is sized against the error the run measures (``ClientView.err_max``). The vector goes
in ENCRYPTED under the public key, which also re-randomises the ciphertext (see the backends).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def uniform_masks(shape, bits: int) -> np.ndarray:
    """Masks uniform on [-2^bits, 2^bits), from the OS CSPRNG: the client must not predict them."""
    n = int(np.prod(shape))
    u = (np.frombuffer(os.urandom(8 * n), dtype=np.uint64) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    return ((2.0 * u - 1.0) * 2.0 ** bits).reshape(shape)


def slot_masks(n_slots: int, bits: int, *, imaginary: bool = True) -> np.ndarray:
    """An independent mask for every slot's real part and, when ``imaginary``, its imaginary part."""
    u = uniform_masks((2, n_slots), bits)
    return u[0] + 1j * (u[1] if imaginary else 0.0)


@dataclass
class ClientView:
    """What the key holder's decodes hold besides the masks, summed for the bound on its view.

    The harness's bookkeeping: it plays the server as well, so it knows each mask; the client does
    not. ``values`` counts the masked real coordinates decoded (two per slot when both parts are
    masked), ``l1`` sums the magnitude each one hides, and ``err_max`` is the largest imaginary
    content, which is pure CKKS error.
    """

    values: int = 0
    l1: float = 0.0
    err_max: float = 0.0
    imaginary_flooded: bool = True

    def record(self, shown, mask: np.ndarray) -> None:
        re = np.asarray(shown, dtype=float) - mask.real
        self.values += int(re.size)
        self.l1 += float(np.abs(re).sum())
        im = getattr(shown, "imag", None)
        if im is None or not np.any(mask.imag):
            # Not masked on this backend, so not part of a bound this ledger can state.
            self.imaginary_flooded = False
            return
        c = np.asarray(im, dtype=float) - mask.imag
        self.values += int(c.size)
        self.l1 += float(np.abs(c).sum())
        self.err_max = max(self.err_max, float(np.max(np.abs(c))))


def masked_decrypt(backend, ct, slot0: float, bits: int, view: ClientView | None = None) -> float:
    """Mask every slot of ``ct`` and decrypt it as the key holder does; return its share, slot 0.

    ``slot0`` goes into slot 0's real part; everything else gets the independent masks above.
    """
    m = slot_masks(int(backend.slot_count), bits, imaginary=bool(getattr(backend, "complex_slots", False)))
    m[0] = complex(float(slot0), m[0].imag)
    shown = backend.decrypt(backend.add_plain_slots(ct, m))
    if view is not None:
        view.record(shown, m)
    return float(shown[0])
