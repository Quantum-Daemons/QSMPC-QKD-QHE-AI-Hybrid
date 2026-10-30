"""Federated learning with pairwise-mask aggregation (the masking structure of secure aggregation).

Implements FedAvg across N virtual sites (hospitals / banks) where raw data never
leaves a client, plus pairwise masking: each pair of clients adds and subtracts the same
pseudo-random mask, derived from a per-pair key and the round. The masks cancel EXACTLY in the
sum, so the aggregate is right. What they HIDE depends entirely on who holds the pair keys, and in
this repository the answer is: nothing from a key holder.

  * Python (this module): no caller supplies keys, so ``SecureAggregator`` derives every pair key
    from a PUBLIC constant, ``sha256("pair-i-j")``. Anyone can recompute every mask and unmask any
    client's update. The masks are also bounded reals in [-1, 1) added in float64, not uniform
    values over a finite ring, so even secret keys would hide an update only partially, not
    perfectly.
  * Browser (``web/src/lib/clientQFL.ts``): every pair's mask is an HKDF output of the ONE PQC
    session key, which the aggregator also holds, so the aggregator can unmask every update.

Bonawitz et al. (CCS 2017) derive each pair's seed by a key agreement between the two clients of
that pair, which is what makes the masks hide individual updates from the server; that is not
implemented here. So this module does NOT defend the gradient-leakage risk noted in the QFL survey
and Li-Deng: it demonstrates that masks cancel.

This is the local "full implementation" FL path, kept framework-light so it stays robust across
Flower releases.

Scope, stated plainly: even a correct secure aggregation hides only an *individual client's* update
from the aggregator. It does not bound what the aggregate or the trained model leaks, and FedAvg on
its own is not a privacy guarantee - gradient inversion against a shared model is an established
attack. A claim of formal privacy would need a stated DP-SGD budget, which this path does not
provide.

An earlier version of this docstring advertised an ``as_flower_client`` adapter exposing a Flower
``NumPyClient``. No such function was ever implemented, so the reference is removed rather than left
as a phantom API for a reader to go looking for.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model, device


# --------------------------------------------------------------------------- #
#  Data partitioning
# --------------------------------------------------------------------------- #
def partition(X: np.ndarray, y: np.ndarray, n_clients: int, *, iid: bool = True,
              seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a dataset across ``n_clients`` (IID or label-skewed non-IID)."""
    rng = np.random.default_rng(seed)
    if iid:
        idx = rng.permutation(len(X))
        return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]
    order = np.argsort(y)                                   # sort by label -> skew
    shards = np.array_split(order, n_clients * 2)
    rng.shuffle(shards)
    parts = []
    for c in range(n_clients):
        sel = np.concatenate([shards[2 * c], shards[2 * c + 1]])
        parts.append((X[sel], y[sel]))
    return parts


# --------------------------------------------------------------------------- #
#  Pairwise masks (from caller-supplied pair keys; by default a public constant)
# --------------------------------------------------------------------------- #
def _mask(seed_key: bytes, shape: tuple[int, ...], round_index: int) -> np.ndarray:
    """Deterministic pseudo-random mask from a shared key and ROUND (SHAKE-256 seeded).

    The round is domain-separated into the seed, mirroring `qflPairMask` in
    `web/src/lib/clientQFL.ts`, whose HKDF info string is `qsh/qfl-pairwise-mask/v1|u|v|r<round>`.

    Until Round 39 this took only the pair key, so `federated_train`'s loop reused ONE mask for every
    round of a run while the browser twin derived a fresh one and documented that requirement. The
    masks still cancelled and the aggregate was never wrong; what reuse leaks is the per-client
    update DELTA, because differencing two rounds of one client's masked share gives
    `(u1 + m) - (u2 + m) = u1 - u2` with no key and no cryptanalysis. That is exactly the quantity
    pairwise masking exists to hide. `tests/test_federated_mask_is_per_round.py` asserts the delta is
    unrecoverable rather than asserting that a `round_index` parameter exists - a signature-only fix
    passes the latter and changes nothing, which is how the combiner's length tag reached one of two
    languages and stayed there for two rounds.
    """
    n = int(np.prod(shape))
    seed = seed_key + b"|qsh/qfl-pairwise-mask/v1|r" + str(int(round_index)).encode()
    raw = hashlib.shake_256(seed).digest(8 * n)
    vals = np.frombuffer(raw, dtype=np.uint64).astype(np.float64)
    return ((vals / 2 ** 64) - 0.5).reshape(shape) * 2.0


@dataclass
class SecureAggregator:
    """Sum pairwise-masked client updates; the masks cancel exactly in the sum.

    Pairwise keys ``keys[(i,j)]`` (i<j) seed antisymmetric masks: client i adds the
    mask, client j subtracts it. Summing all masked updates cancels every mask. An individual
    update is hidden only from someone who cannot derive the pair keys, and with the default keys
    (a public constant, below) that is nobody.
    """

    n_clients: int
    keys: dict[tuple[int, int], bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.keys:
            # DEMO ONLY, and not a privacy guarantee: these pairwise keys are derived from a
            # public constant, so anyone can recompute every mask and unmask an individual
            # update. The masks are real and they cancel exactly; what is absent is the secrecy
            # of the key material. No caller in this repository passes `keys` (federated_train
            # below does not), so this branch is the one that runs. Hiding needs each pair's key
            # agreed between the two clients of that pair (Bonawitz et al.); a key every party and
            # the aggregator share, like the browser's session key, would not hide anything from
            # the aggregator either. See docs/threat-model.md.
            for i in range(self.n_clients):
                for j in range(i + 1, self.n_clients):
                    self.keys[(i, j)] = hashlib.sha256(f"pair-{i}-{j}".encode()).digest()

    def mask_update(self, client: int, update: np.ndarray, round_index: int = 0) -> np.ndarray:
        """Mask one client's update for `round_index`.

        `round_index` is keyword-defaulted so the existing three call sites keep working, but the
        FedAvg loop MUST pass the real round - a default that silently means "round 0 forever" is
        the reuse this signature was widened to remove.
        """
        out = update.copy()
        for j in range(self.n_clients):
            if j == client:
                continue
            i0, j0 = min(client, j), max(client, j)
            # Keyed on the ORDERED pair, so both members of a pair derive the identical mask and the
            # antisymmetric +/- below still cancels exactly. Folding the round in per-client instead
            # of per-pair would break cancellation into a plausible-looking wrong aggregate.
            m = _mask(self.keys[(i0, j0)], update.shape, round_index)
            out += m if client < j else -m
        return out

    def aggregate(self, masked_updates: list[np.ndarray]) -> np.ndarray:
        """Sum masked updates -> the masks cancel, leaving the true sum."""
        return np.sum(masked_updates, axis=0)


# --------------------------------------------------------------------------- #
#  FedAvg
# --------------------------------------------------------------------------- #
def _flatten(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def _unflatten(model: nn.Module, vec: np.ndarray) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.tensor(vec[off:off + n].reshape(p.shape), dtype=p.dtype, device=p.device)
        off += n


def _local_train(model: nn.Module, X, y, epochs, lr, dev, batch_size=32) -> np.ndarray:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size, shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return _flatten(model)


@dataclass
class FederatedResult:
    global_acc: float
    round_acc: list[float]
    n_clients: int
    secure_aggregation: bool


def federated_train(
    parts: list[tuple[np.ndarray, np.ndarray]],
    kind: str = "mlp",
    *,
    rounds: int = 8,
    local_epochs: int = 1,
    lr: float = 0.05,
    batch_size: int = 32,
    secure_aggregation: bool = True,
    eval_data: tuple[np.ndarray, np.ndarray] | None = None,
    model_kwargs: dict | None = None,
    seed: int = 0,
    return_model: bool = False,
) -> FederatedResult:
    """Run FedAvg across ``parts`` with optional pairwise masking (public demo keys; see the module
    docstring for what that does and does not hide).

    With ``return_model=True`` returns ``(FederatedResult, global_model)`` so callers (e.g. the
    medical CNN use case) can run a genuine private inference with the federated-trained weights.
    """
    torch.manual_seed(seed)
    # The QFL hybrid runs its VQC on PennyLane's default.qubit (CPU); keep the whole net on CPU so the
    # classical params and the quantum simulator share a device (mixing cuda params + CPU statevector errors).
    dev = torch.device("cpu") if kind == "qfl" else device()
    n_clients = len(parts)
    model_kwargs = dict(model_kwargs or {})
    if kind in ("mlp", "qfl"):                              # qfl = hybrid quantum-classical (HybridQNN)
        model_kwargs.setdefault("n_features", parts[0][0].shape[1])
    classes = int(max(p[1].max() for p in parts)) + 1
    model_kwargs.setdefault("n_classes", classes)

    global_model = build_model(kind, **model_kwargs).to(dev)
    global_vec = _flatten(global_model)
    agg = SecureAggregator(n_clients) if secure_aggregation else None
    sizes = np.array([len(p[0]) for p in parts], dtype=float)
    weights = sizes / sizes.sum()
    round_acc: list[float] = []

    for rnd in range(rounds):
        updates = []
        for c, (Xc, yc) in enumerate(parts):
            local = build_model(kind, **model_kwargs).to(dev)
            _unflatten(local, global_vec)
            vec = _local_train(local, Xc, yc, local_epochs, lr, dev, batch_size)
            weighted = vec * weights[c]                  # FedAvg weighting
            # `rnd` is what makes the mask fresh. This loop is where the reuse actually lived: the
            # index was discarded as `_` for the whole life of the function.
            updates.append(agg.mask_update(c, weighted, rnd) if agg else weighted)
        global_vec = agg.aggregate(updates) if agg else np.sum(updates, axis=0)
        if eval_data is not None:
            _unflatten(global_model, global_vec)
            round_acc.append(_eval(global_model, *eval_data, dev))

    _unflatten(global_model, global_vec)
    final = round_acc[-1] if round_acc else (
        _eval(global_model, *eval_data, dev) if eval_data else float("nan"))
    result = FederatedResult(final, round_acc, n_clients, secure_aggregation)
    if return_model:
        return result, global_model.eval()
    return result


@torch.no_grad()
def _eval(model: nn.Module, X, y, dev) -> float:
    model.eval()
    pred = model(torch.tensor(X, dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()
    return float((pred == y).mean())
