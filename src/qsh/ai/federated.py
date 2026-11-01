"""Federated learning with pairwise-mask secure aggregation (Bonawitz et al., CCS 2017).

Implements FedAvg across N virtual sites (hospitals / banks) where raw data never
leaves a client, plus pairwise masking: each pair of clients adds and subtracts the same
pseudo-random mask, derived from a seed that pair agreed by X25519 and from the round. The masks
cancel in the sum, so the aggregate is right, and an aggregator that holds only the clients' public
keys and their masked updates cannot remove a mask on its own.

The construction is byte-identical to the browser's (``web/src/lib/clientQFL.ts``;
``tests/test_r84_pairwise_masks.py`` asserts it for the same private keys):

  * each client draws a fresh X25519 key pair per run (or a caller passes fixed private keys, for a
    test or a reproduction);
  * the pair i < j derives ``seed_ij = HKDF-SHA256(X25519(sk_i, pk_j), salt=SECAGG_SALT,
    info=SECAGG_PAIR_LABEL + u32be(i) + u32be(j), length=32)``, the same 32 bytes from either side;
  * the round-r mask is ``SHAKE-256(seed_ij + SECAGG_MASK_LABEL + u32be(r))`` read as little-endian
    uint32 words; client i adds it for every j > i and subtracts it for every j < i;
  * all of it modulo 2^32, on the update in fixed point: a weighted coordinate x is carried as
    ``round(x * 2**20) mod 2**32`` (JavaScript's rounding), and the aggregate is the sum of the shares
    modulo 2^32, read back through the signed half and divided by 2^20 (Round 85). The masks cancel
    exactly, the aggregate is the exact sum of the quantised updates, and a share on its own is the
    quantised update plus at least one mask the aggregator cannot derive, which SHAKE-256 makes
    indistinguishable from uniform over the ring. Until Round 85 the masks were added as reals in
    [-1, 1) in float64, which cancelled only to rounding and hid an update only partially.

What it does NOT give, stated plainly:

  * no dropout recovery: Bonawitz et al.'s second, per-user mask and the Shamir shares that let the
    server finish a round without a dropped user are not implemented, so a client that drops breaks
    the round;
  * no malicious security, and no authentication of the public keys: an aggregator that substitutes
    its own keys can agree a seed with each client and unmask it, and that adversary is out of scope;
  * ``federated_train`` runs every client and the aggregator in one process, so the parties are
    notional; the aggregation step (``SecureAggregator.aggregate``) reads the masked updates only.

So this module does NOT defend the gradient-leakage risk noted in the QFL survey and Li-Deng: even a
complete secure aggregation hides only an *individual client's* update from the aggregator. It does
not bound what the aggregate or the trained model leaks, and FedAvg on its own is not a privacy
guarantee - gradient inversion against a shared model is an established attack. A claim of formal
privacy would need a stated DP-SGD budget, which this path does not provide.

This is the local "full implementation" FL path, kept framework-light so it stays robust across
Flower releases.

An earlier version of this docstring advertised an ``as_flower_client`` adapter exposing a Flower
``NumPyClient``. No such function was ever implemented, so the reference is removed rather than left
as a phantom API for a reader to go looking for.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
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
#  Pairwise key agreement and masks (Bonawitz et al., CCS 2017)
# --------------------------------------------------------------------------- #
#: The fixed HKDF salt of the pair-seed derivation (18 ASCII bytes), as ``SECAGG_SALT`` in
#: ``web/src/lib/clientQFL.ts``.
SECAGG_SALT = b"qsh/secagg/salt/v1"
#: The HKDF info is this 18-byte label followed by the pair's indices i < j, each a 4-byte big-endian
#: unsigned integer: 26 bytes in all.
SECAGG_PAIR_LABEL = b"qsh/secagg/pair/v1"
#: The per-round mask's domain label, between the seed and the round in SHAKE-256's input.
SECAGG_MASK_LABEL = b"qsh/secagg/mask/v1"


#: The fixed-point ring of the masked shares, the browser's (``QFL_SCALE``, ``QFL_RING`` in
#: clientQFL.ts): 2^-20 resolution per coordinate, carried modulo 2^32.
SECAGG_SCALE = 1 << 20
SECAGG_RING = 1 << 32


def max_share(n_clients: int) -> float:
    """The largest quantised contribution, in magnitude, whose sum over ``n_clients`` cannot overflow
    the signed half of the ring (``qflMaxShare`` in clientQFL.ts)."""
    return 2 ** 31 / max(1, n_clients)


def quantise(update: np.ndarray, n_clients: int) -> np.ndarray:
    """An update in the ring: ``round(x * 2**20)`` as uint32 two's complement, ``maskedShare``'s first step.

    Rounded as JavaScript's ``Math.round`` rounds, half up, so the two tiers agree to the bit; Python's
    and NumPy's ``round`` take halves to even. ``floor(x) + (x - floor(x) >= 0.5)`` is exact where
    ``floor(x + 0.5)`` is not (0.49999999999999994 + 0.5 rounds to 1.0). A coordinate past
    :func:`max_share` raises, as the browser does: the ring would wrap it into the wrong half and the
    aggregate would be a wrong number that still looks like one.
    """
    x = np.asarray(update, dtype=np.float64) * SECAGG_SCALE
    q = np.floor(x)
    q = q + ((x - q) >= 0.5)
    bad = ~np.isfinite(q) | (np.abs(q) > max_share(n_clients))
    if bad.any():
        k = int(np.flatnonzero(bad.ravel())[0])
        raise ValueError(f"secure aggregation: coordinate {k} is out of the ring's range "
                         f"(q={q.ravel()[k]}, limit {max_share(n_clients):.0f} for {n_clients} clients)")
    return q.astype(np.int64).astype(np.uint32)


def dequantise(acc: np.ndarray) -> np.ndarray:
    """A sum of shares modulo 2^32, read through the signed half and scaled back to reals."""
    signed = np.asarray(acc, dtype=np.uint32).astype(np.int64)
    signed[signed >= SECAGG_RING // 2] -= SECAGG_RING
    return signed.astype(np.float64) / SECAGG_SCALE


def pair_info(i: int, j: int) -> bytes:
    """The HKDF info of the pair (i, j): ``SECAGG_PAIR_LABEL + u32be(i) + u32be(j)``, i < j."""
    if not (0 <= i < j <= 0xFFFFFFFF):
        raise ValueError(f"a pair is named by its two client indices in order, got ({i}, {j})")
    return SECAGG_PAIR_LABEL + struct.pack(">II", i, j)


def pair_seed(own_private_key: X25519PrivateKey, peer_public_key: bytes, i: int, j: int) -> bytes:
    """The 32-byte seed of the pair (i, j) as ONE of its two clients computes it.

    ``HKDF-SHA256(X25519(own, peer), salt=SECAGG_SALT, info=pair_info(i, j), length=32)`` (RFC 7748,
    RFC 5869). The other client of the pair reaches the same bytes from its own private key; the
    aggregator, which relays both public keys, cannot. ``exchange`` refuses a low-order peer key
    whose shared secret would be all zeros (RFC 7748 section 6.1).
    """
    info = pair_info(i, j)
    shared = own_private_key.exchange(X25519PublicKey.from_public_bytes(bytes(peer_public_key)))
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=SECAGG_SALT, info=info).derive(shared)


def pair_mask_words(seed: bytes, round_index: int, n: int) -> np.ndarray:
    """The round's mask words for one pair: ``n`` little-endian uint32 words of
    ``SHAKE-256(seed + SECAGG_MASK_LABEL + u32be(round_index))``, as ``secaggMaskWords`` derives them.

    SHAKE-256 because a parameter vector can be longer than the 255 x 32 = 8,160 bytes HKDF-SHA256 can
    expand to (RFC 5869 section 2.3); one derivation serves both tiers byte for byte.
    """
    if not 0 <= int(round_index) <= 0xFFFFFFFF:
        raise ValueError(f"round {round_index} is not a uint32")
    raw = hashlib.shake_256(seed + SECAGG_MASK_LABEL + struct.pack(">I", int(round_index))).digest(4 * n)
    return np.frombuffer(raw, dtype="<u4")


@dataclass
class SecureAggregator:
    """Pairwise-masked aggregation whose mask seeds each pair of clients agrees by X25519.

    Construction draws every client's X25519 key pair for this run (fresh from the OS CSPRNG, or the
    32-byte ``private_keys`` a caller passes, for a test or a reproduction: a fixed key is a public
    one), publishes the public keys in ``public_keys``, and gives each client its seed with every peer,
    computed from that client's own private key. The private keys are not kept.

    ``mask_update(c, ...)`` is client c's side and uses client c's seeds alone; ``aggregate`` is the
    aggregator's side and reads the masked updates alone. Every client and the aggregator still run in
    this one process, so the parties are notional.
    """

    n_clients: int
    private_keys: Sequence[bytes] | None = field(default=None, repr=False)
    #: The clients' X25519 public keys for this run (32 bytes each): what the aggregator relays.
    public_keys: list[bytes] = field(init=False, default_factory=list)
    #: Measured wall time of the key agreement (every key pair and every client's seeds), in ms.
    key_agreement_ms: float = field(init=False, default=0.0)
    _seeds: list[dict[int, bytes]] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        t0 = time.perf_counter()
        n = self.n_clients
        if self.private_keys is None:
            own = [X25519PrivateKey.generate() for _ in range(n)]
        else:
            if len(self.private_keys) != n:
                raise ValueError(f"{len(self.private_keys)} private keys for {n} clients")
            own = [X25519PrivateKey.from_private_bytes(bytes(k)) for k in self.private_keys]
        self.public_keys = [k.public_key().public_bytes_raw() for k in own]
        # Client c's seeds, each from c's OWN private key and the peer's relayed public key. Both
        # members of a pair compute the pair's seed independently, as they would on two hosts.
        self._seeds = [{j: pair_seed(own[c], self.public_keys[j], min(c, j), max(c, j))
                        for j in range(n) if j != c} for c in range(n)]
        # Each client needs its private key only to derive its seeds; drop the references here.
        self.private_keys = None
        del own
        self.key_agreement_ms = (time.perf_counter() - t0) * 1000.0

    def mask_update(self, client: int, update: np.ndarray, round_index: int = 0) -> np.ndarray:
        """Mask one client's update for `round_index`, with that client's own seeds.

        `round_index` is keyword-defaulted so the existing three call sites keep working, but the
        FedAvg loop MUST pass the real round - a default that silently means "round 0 forever" is
        the reuse this signature was widened to remove.

        The share is uint32, in the ring (see the module docstring): the update, already weighted by
        the caller, quantised, plus this client's pair masks, all modulo 2^32.
        """
        share = quantise(update, self.n_clients).ravel()
        for j, seed in self._seeds[client].items():
            # One seed per unordered pair, so both members derive the identical mask and the
            # antisymmetric +/- below cancels. Folding the round in per-client instead of per-pair
            # would break cancellation into a plausible-looking wrong aggregate. uint32 arithmetic
            # wraps, so `+` and `-` here are the ring's.
            m = pair_mask_words(seed, round_index, share.size)
            share = share + m if client < j else share - m
        return share.reshape(np.shape(update))

    @staticmethod
    def aggregate(masked_updates: list[np.ndarray]) -> np.ndarray:
        """The aggregator's whole computation: sum the masked shares modulo 2^32; the masks cancel
        exactly, leaving the sum of the quantised updates, read back as reals. It is given the masked
        shares and nothing else."""
        acc = np.zeros(np.shape(masked_updates[0]), dtype=np.uint64)
        for share in masked_updates:
            acc += np.asarray(share, dtype=np.uint32)
        return dequantise(acc % SECAGG_RING)


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
    secagg_private_keys: Sequence[bytes] | None = None,
) -> FederatedResult:
    """Run FedAvg across ``parts`` with optional pairwise masking, each pair's mask seed agreed by
    X25519 (see the module docstring for what that does and does not hide).

    ``secagg_private_keys`` fixes the clients' X25519 private keys (32 bytes each) for a test or a
    reproduction; by default every run draws fresh ones.

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
    agg = (SecureAggregator(n_clients, private_keys=secagg_private_keys)
           if secure_aggregation else None)
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
