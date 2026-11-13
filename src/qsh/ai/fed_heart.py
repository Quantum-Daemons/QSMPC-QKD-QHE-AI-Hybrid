"""Four real hospitals, each alone, federated and pooled: FLamby's Fed-Heart-Disease benchmark, twinned with the page.

WHY THIS EXISTS (Round 86, the owner's choice of 2026-11-13)

The medical page federated shards of one corpus, so its federation had no hospital in it. This module runs FLamby's
Fed-Heart-Disease baseline on the four UCI hospitals the page ships (`scripts/data/build_heart_disease_sites.py`):

  * the model: logistic regression, 13 standardised features to one logit, FLamby's `Baseline`, initialised as
    PyTorch initialises `nn.Linear(13, 1)`, uniform in +/-1/sqrt(13);
  * each hospital ALONE and all four POOLED: Adam (lr 0.001, betas 0.9 and 0.999, eps 1e-8), batches of 4, 50 epochs
    (FLamby's `NUM_EPOCHS_POOLED`, `BATCH_SIZE`, `LR`, `Optimizer`);
  * FEDERATED: FedAvg, 15 rounds of 100 local SGD steps at lr 0.001 (FLamby's `get_nb_max_rounds(100)` and its
    benchmark's FedAvg setting), each hospital's model weighted by its training rows, through the page's
    secure aggregation: each weighted update quantised as `maskedShare` quantises it and summed modulo 2^32, which is
    what the masked shares sum to, since the masks cancel exactly;
  * the metric: FLamby's, accuracy at 0.5 on each hospital's test rows, averaged over the four.

A TWIN of `web/src/lib/clientFedHeart.ts`: the same random stream (mulberry32, `rng` in quantumSim.ts), the same order
of operations in float64, so the page's figures are this module's to the last bits
(`tests/test_r86_fed_heart_twin.py`). The streams: initial weights from rng(seed * 1000 + 7), hospital k's batches
from rng(seed * 1000 + 100 + k) alone and rng(seed * 1000 + 200 + k) federated, the pooled batches from
rng(seed * 1000 + 300).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .federated import dequantise, quantise

LR = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
BATCH = 4
EPOCHS = 50
ROUNDS = 15
LOCAL_STEPS = 100
SAMPLE = Path(__file__).resolve().parents[3] / "web" / "public" / "data" / "medical" / "heart_disease_sites.json"


def mulberry32(seed: int) -> Callable[[], float]:
    """`rng` in web/src/lib/quantumSim.ts, bit for bit: 32-bit integer arithmetic, a float in [0, 1)."""
    a = (seed & 0xFFFFFFFF) or 1

    def imul(x: int, y: int) -> int:
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def shuffled(n: int, u: Callable[[], float]) -> list[int]:
    """Fisher-Yates from the end, `Math.floor(u() * (i + 1))`, as clientQFL shuffles."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(u() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _grad(theta: list[float], X: list[list[float]], y: list[int], batch: list[int]) -> list[float]:
    """Mean binary cross-entropy gradient over the batch; theta is 13 weights then the bias."""
    g = [0.0] * len(theta)
    d = len(theta) - 1
    for i in batch:
        z = theta[d]
        for k in range(d):
            z += theta[k] * X[i][k]
        r = _sigmoid(z) - y[i]
        for k in range(d):
            g[k] += r * X[i][k]
        g[d] += r
    m = len(batch)
    return [v / m for v in g]


@dataclass
class _Adam:
    m: list[float]
    v: list[float]
    #: The bias corrections' powers as running products, not `BETA1 ** t`: a power goes through each language's own
    #: `pow`, which may differ in the last bit, and a product of doubles does not.
    b1t: float = 1.0
    b2t: float = 1.0

    def step(self, theta: list[float], g: list[float]) -> None:
        self.b1t *= BETA1
        self.b2t *= BETA2
        bc1, bc2 = 1 - self.b1t, 1 - self.b2t
        for k in range(len(theta)):
            self.m[k] = BETA1 * self.m[k] + (1 - BETA1) * g[k]
            self.v[k] = BETA2 * self.v[k] + (1 - BETA2) * g[k] * g[k]
            denom = math.sqrt(self.v[k]) / math.sqrt(bc2) + EPS
            theta[k] -= (LR / bc1) * self.m[k] / denom


def init_theta(seed: int, d: int = 13) -> list[float]:
    u = mulberry32(seed * 1000 + 7)
    bound = 1 / math.sqrt(d)
    return [(u() * 2 - 1) * bound for _ in range(d + 1)]


def train_adam(theta: list[float], X: list[list[float]], y: list[int], u: Callable[[], float]) -> list[float]:
    th, opt = list(theta), _Adam([0.0] * len(theta), [0.0] * len(theta))
    for _ in range(EPOCHS):
        order = shuffled(len(y), u)
        for s in range(0, len(order), BATCH):
            opt.step(th, _grad(th, X, y, order[s:s + BATCH]))
    return th


def train_sgd_steps(theta: list[float], X: list[list[float]], y: list[int], u: Callable[[], float]) -> list[float]:
    """LOCAL_STEPS batches, a fresh shuffle each time the client's rows run out, as a DataLoader is iterated."""
    th = list(theta)
    order, pos = shuffled(len(y), u), 0
    for _ in range(LOCAL_STEPS):
        if pos >= len(order):
            order, pos = shuffled(len(y), u), 0
        batch = order[pos:pos + BATCH]
        pos += BATCH
        g = _grad(th, X, y, batch)
        for k in range(len(th)):
            th[k] -= LR * g[k]
    return th


def accuracy(theta: list[float], X: list[list[float]], y: list[int]) -> float:
    d = len(theta) - 1
    hits = 0
    for i in range(len(y)):
        z = theta[d] + sum(theta[k] * X[i][k] for k in range(d))
        hits += int((1 if _sigmoid(z) > 0.5 else 0) == y[i])
    return hits / len(y)


@dataclass
class FedHeartResult:
    #: Test accuracy on each hospital's own rows, for each arm, and FLamby's average over the four.
    alone: list[list[float]] = field(default_factory=list)     # alone[k][j]: hospital k's model on hospital j's test
    federated: list[float] = field(default_factory=list)
    pooled: list[float] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)          # the shared initial model, before any training

    @staticmethod
    def mean(v: list[float]) -> float:
        return sum(v) / len(v)


def run(sample: dict | None = None, seed: int = 0) -> FedHeartResult:
    s = sample or json.loads(SAMPLE.read_text())
    cs = s["centres"]
    tests = [(c["test"]["X"], c["test"]["y"]) for c in cs]
    theta0 = init_theta(seed)
    res = FedHeartResult()
    res.initial = [accuracy(theta0, X, y) for X, y in tests]
    for k, c in enumerate(cs):
        th = train_adam(theta0, c["train"]["X"], c["train"]["y"], mulberry32(seed * 1000 + 100 + k))
        res.alone.append([accuracy(th, X, y) for X, y in tests])
    Xp = [r for c in cs for r in c["train"]["X"]]
    yp = [v for c in cs for v in c["train"]["y"]]
    pooled = train_adam(theta0, Xp, yp, mulberry32(seed * 1000 + 300))
    res.pooled = [accuracy(pooled, X, y) for X, y in tests]
    sizes = [len(c["train"]["y"]) for c in cs]
    total = sum(sizes)
    streams = [mulberry32(seed * 1000 + 200 + k) for k in range(len(cs))]
    glob = list(theta0)
    for _ in range(ROUNDS):
        acc = np.zeros(len(glob), dtype=np.uint64)
        for k, c in enumerate(cs):
            local = train_sgd_steps(glob, c["train"]["X"], c["train"]["y"], streams[k])
            acc += quantise(np.asarray(local, dtype=np.float64) * (sizes[k] / total), len(cs))
        glob = dequantise((acc % (1 << 32)).astype(np.uint32)).tolist()
    res.federated = [accuracy(glob, X, y) for X, y in tests]
    return res
