#!/usr/bin/env python3
"""Murmur detection on CirCor DigiScope under differential privacy (Round 86, the owner's choices of 2026-11-14).

Present against absent, per patient record (records labelled unknown are left out), scored on a held-out fifth of the
PEOPLE. 70 people in the public training set attended both screening campaigns and so have two patient IDs (the table's
Additional ID; `scripts/data/prepare_circor.py` joins them), so the person, not the patient ID, is the unit of the split
and of privacy: everything a person contributes is clipped together.

Models (`--models`):
  head-panns    a logistic head on PANNs CNN14 embeddings (AudioSet, CC BY 4.0; `panns_cnn14.py`), reduced to 128 by a PCA
                fitted on public windows only: the recipe Abadi et al. (CCS 2016) used for CIFAR-10, features learned on
                public data and only the last layer trained privately;
  head-public   the same on the 128 features of the heart-sound card's CNN trained on the PhysioNet/CinC 2016 recordings;
  head-logmel   the same on each window's log-mel band means and deviations (80 features), nothing learned;
  cnn           that CNN with GroupNorm, trained end to end on CirCor (the comparison Tramer and Boneh, ICLR 2021, draw).
  Every head's features are standardised with public statistics, so nothing private sets them.
Arms:
  central       DP-SGD (Abadi et al.): people Poisson-sampled (64 expected a step), each person's gradient clipped to C,
                Gaussian noise sigma C on the sum; Renyi DP of the sampled Gaussian (Mironov, Talwar and Zhang, 2019)
                over integer orders, converted by Canonne, Kamath and Steinke's Proposition 12.
  central-full  every person every step, Gaussian noise, the exact Gaussian-DP curve (Balle and Wang, ICML 2018): the
                central twin of the distributed arm, which samples no one, so the two differ only by the distribution.
  federated     four sites, every person every round; each person's clipped gradient scaled by 2^14 / C and randomly
                rounded; each site adds its share of discrete Gaussian noise before the secure sum (Kairouz, Liu and
                Steinke, ICML 2021), accounted by `qsh.ai.dp` as the page's rows are; the bound holds while all four add
                their shares.
Each with its non-private twin and a clip-only run, at epsilon 8, 4, 2 and 1, delta 1e-5.
Tuning (`--tune-on`): every setting's learning rate comes from a grid, scored either on a validation fifth of CirCor's
training people (`circor`: the stated epsilon then leaves the choice out; Papernot and Steinke, ICLR 2022) or on a
PhysioNet 2016 stand-in of the same size (`public`, with `--proxy-sources` and `--proxy-cnn-disjoint`: the stated epsilon
covers everything that touched CirCor). The decision thresholds come from the same place (`decision_thresholds`): DP's
clipping drives the logit-0 threshold to call almost no one positive, which AUROC does not show (Wen et al. 2026).
`medical_murmur_dp_search.py` runs the accounted random search and `medical_murmur_dp_objpert.py` objective perturbation.

The data: Oliveira et al., The CirCor DigiScope Phonocardiogram Dataset, version 1.0.3, PhysioNet, under ODC-By 1.0, and
the PhysioNet/CinC Challenge 2016 recordings, ODC-By 1.0 (DATA_LICENSES.md). Nothing derived from a recording, and no
weight, is committed.

Usage: python -I scripts/train/medical_murmur_dp.py --prep <prepared CirCor dir> --public-prep <prepared PhysioNet 2016 dir>
       --out <dir> [--models head-panns,head-public] [--panns Cnn14_mAP=0.431.pth] [--tune-on public --proxy-sources abf
       --proxy-cnn-disjoint] [--grid MODEL[:ARM]=LR,...]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("medical_pcg_fl", HERE / "medical_pcg_fl.py")
pcg = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("medical_pcg_fl", pcg)
_spec.loader.exec_module(pcg)

DEV = pcg.DEV
DELTA = 1e-5
EPSILONS = (8.0, 4.0, 2.0, 1.0)
KINDS = (("non-private", None), ("clip only", None)) + tuple(("DP", e) for e in EPSILONS)
CLIP = 1.0
WINDOWS_PER_PATIENT = 4
POS_WEIGHT = 4.0          # a fixed constant near the public prevalence (about 1 in 5), not computed from the training labels
SITES = 4
SCALE_BITS = 14           # gamma = 2^14 / C: the integer scale of a clipped person's gradient
ORDERS = tuple(range(2, 257))
EXPECTED_BATCH = 64
EPOCHS = 30               # central: steps = EPOCHS / q; the full-batch arms take as many rounds as the central arm steps
PUBLIC_EPOCHS = 12
ARMS = {"cnn": ("central", "federated"), "head-public": ("central", "central-full", "federated"),
        "head-logmel": ("central", "central-full", "federated"), "head-panns": ("central", "central-full", "federated")}
PANNS_DIMS = 128          # principal components of the CNN14 embedding, fitted on the public windows
GRIDS = {("cnn", "central"): (0.02, 0.05, 0.1, 0.2, 0.5), ("cnn", "federated"): (0.1, 0.25, 0.5, 1.0)}
HEAD_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
SEEDS = {"cnn": {"central": (1, 2), "federated": (1,), "test": (1, 2, 3)},
         "head": {"central": (1, 2, 3), "central-full": (1, 2, 3), "federated": (1, 2, 3), "test": (1, 2, 3, 4, 5)}}


# ------------------------------------------------------------------------------------------------------------ accounting

def rdp_sampled_gaussian(q: float, sigma: float, order: int) -> float:
    """Renyi DP of one step of the Poisson-sampled Gaussian mechanism at an integer order (Mironov, Talwar and Zhang,
    2019, section 3.3): log(sum_k C(order, k) (1 - q)^(order - k) q^k exp((k^2 - k) / (2 sigma^2))) / (order - 1)."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return order / (2 * sigma * sigma)
    terms = [math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1) + (order - k) * math.log1p(-q)
             + k * math.log(q) + (k * k - k) / (2 * sigma * sigma) for k in range(order + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (order - 1)


def rdp_to_dp(rdp_total: dict[int, float], delta: float) -> float:
    """The conversion of Canonne, Kamath and Steinke (2020), Proposition 12, minimised over the orders."""
    return min(r + math.log1p(-1 / a) - (math.log(delta) + math.log(a)) / (a - 1) for a, r in rdp_total.items())


def dpsgd_epsilon(q: float, sigma: float, steps: int, delta: float = DELTA) -> float:
    return rdp_to_dp({a: steps * rdp_sampled_gaussian(q, sigma, a) for a in ORDERS}, delta)


def _bisect(eps_of, target: float) -> float:
    lo, hi = 1e-3, 1.0
    while eps_of(hi) > target:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if eps_of(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_for(target: float, q: float, steps: int, delta: float = DELTA) -> float:
    return _bisect(lambda s: dpsgd_epsilon(q, s, steps, delta), target)


def _qsh_dp():
    repo = HERE.parents[1]
    spec = importlib.util.spec_from_file_location("qsh_ai_dp", repo / "src" / "qsh" / "ai" / "dp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------- data

def load(prep: Path) -> dict:
    with open(prep / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    d = {"patient": np.array([int(r["patient"]) for r in rows]), "person": np.array([int(r["person"]) for r in rows]),
         "murmur": np.array([r["murmur"] for r in rows]), "offset": np.array([int(r["offset"]) for r in rows]),
         "length": np.array([int(r["length"]) for r in rows])}
    keep = d["murmur"] != "Unknown"
    pats = np.unique(d["patient"][keep])
    recs = {p: np.flatnonzero(d["patient"] == p) for p in pats}
    y = np.array([float(d["murmur"][recs[p][0]] == "Present") for p in pats], dtype=np.float32)
    people, person_of = np.unique([d["person"][recs[p][0]] for p in pats], return_inverse=True)
    members = [np.flatnonzero(person_of == k) for k in range(len(people))]
    return {**d, "pats": pats, "recs": recs, "py": y, "people": people, "person_of": person_of, "members": members,
            "person_y": np.array([y[m].max() for m in members], dtype=np.float32)}


def holdout(d: dict, people: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A fifth of these people held out, stratified by whether any of their records has a murmur."""
    rng = np.random.default_rng(seed)
    out = []
    for lab in (0.0, 1.0):
        idx = people[d["person_y"][people] == lab]
        out.extend(rng.choice(idx, len(idx) // 5, replace=False))
    out = np.sort(np.array(out))
    return np.setdiff1d(people, out), out


def split(d: dict, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Training and test people; fixed across training seeds."""
    return holdout(d, np.arange(len(d["members"])), seed)


def records_of(d: dict, people: np.ndarray) -> np.ndarray:
    return np.sort(np.concatenate([d["members"][k] for k in people]))


def sites_of(d: dict, people: np.ndarray, k: int = SITES, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    out = [[] for _ in range(k)]
    for lab in (0.0, 1.0):
        for j, p in enumerate(rng.permutation(people[d["person_y"][people] == lab])):
            out[j % k].append(p)
    return [np.sort(np.array(s)) for s in out]


def owners(d: dict, people: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """The records of these people, and for each record its person's position in `people`."""
    pats = np.concatenate([d["members"][k] for k in people])
    own = np.concatenate([np.full(len(d["members"][k]), j) for j, k in enumerate(people)])
    return pats, torch.from_numpy(own).to(DEV)


# ------------------------------------------------------------------------------------------------------------ the models

class GnCnn(nn.Module):
    """The PhysioNet 2016 card's CNN with GroupNorm in place of BatchNorm, so a person's gradient is its own."""

    def __init__(self, width: int = 16):
        super().__init__()
        layers, c = [], 1
        for k in range(4):
            out = width * 2 ** k
            layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.GroupNorm(4, out), nn.ReLU(), nn.MaxPool2d(2)]
            c = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c, 1)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(m).mean(dim=(2, 3))).squeeze(-1)


def patient_windows(d, sig, feats, patients: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """WINDOWS_PER_PATIENT random windows from each record's recordings: (records, windows, 1, mels, frames)."""
    recs, starts = [], []
    for p in patients:
        r = d["recs"][d["pats"][p]]
        for i in rng.choice(r, WINDOWS_PER_PATIENT):
            n = int(d["length"][i])
            recs.append(i)
            starts.append(int(rng.integers(0, max(n - pcg.WINDOW, 0) + 1)))
    with torch.no_grad():
        m = feats(pcg.batch(sig, d, np.array(recs), np.array(starts)))
    return m.view(len(patients), WINDOWS_PER_PATIENT, *m.shape[1:])


def per_patient_grads(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    params = {k: v.detach() for k, v in model.named_parameters()}
    pw = torch.tensor(POS_WEIGHT, device=DEV)

    def loss_one(p, xb, yb):
        logit = functional_call(model, p, (xb,))
        return F.binary_cross_entropy_with_logits(logit, yb.expand_as(logit), pos_weight=pw)

    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([g[k].reshape(x.shape[0], -1) for k in params], dim=1)


def person_grads_cnn(model, d, sig, feats, people: np.ndarray, rng, chunk: int = 48) -> torch.Tensor:
    """(people, parameters): each person's gradient, the sum over its records of the mean over each record's windows."""
    out = []
    for c in range(0, len(people), chunk):
        pats, own = owners(d, people[c:c + chunk])
        g = per_patient_grads(model, patient_windows(d, sig, feats, pats, rng), torch.from_numpy(d["py"][pats]).to(DEV))
        out.append(torch.zeros(len(people[c:c + chunk]), g.shape[1], device=DEV).index_add_(0, own, g))
    return torch.cat(out)


class Head:
    """Fixed features of every window of every labelled record, and a logistic head on them."""

    def __init__(self, d: dict, x: torch.Tensor, win_patient: np.ndarray):
        self.d, self.x = d, x
        n = len(d["pats"])
        self.counts = torch.from_numpy(np.bincount(win_patient, minlength=n)).to(DEV)
        order = np.argsort(win_patient, kind="stable")
        self.order = torch.from_numpy(order).to(DEV)
        self.starts = torch.from_numpy(np.concatenate([[0], np.cumsum(np.bincount(win_patient, minlength=n))[:-1]])).to(DEV)
        self.win_patient = torch.from_numpy(win_patient).to(DEV)
        self.y = torch.from_numpy(d["py"]).to(DEV)
        self.dim = x.shape[1] + 1

    def person_grads(self, theta: torch.Tensor, people: np.ndarray, g: torch.Generator) -> torch.Tensor:
        pats, own = owners(self.d, people)
        pt = torch.from_numpy(pats).to(DEV)
        j = (torch.rand((len(pats), WINDOWS_PER_PATIENT), device=DEV, generator=g) * self.counts[pt].unsqueeze(1)).long()
        xw = self.x[self.order[self.starts[pt].unsqueeze(1) + j]]
        s = torch.sigmoid(xw @ theta[:-1] + theta[-1])
        dz = torch.where(self.y[pt].unsqueeze(1) > 0.5, POS_WEIGHT * (s - 1), s) / WINDOWS_PER_PATIENT
        gp = torch.cat([(dz.unsqueeze(-1) * xw).sum(1), dz.sum(1, keepdim=True)], 1)
        return torch.zeros(len(people), self.dim, device=DEV).index_add_(0, own, gp)

    @torch.no_grad()
    def scores(self, theta: torch.Tensor, patients: np.ndarray) -> np.ndarray:
        z = self.x @ theta[:-1] + theta[-1]
        tot = torch.zeros(len(self.counts), device=DEV).index_add_(0, self.win_patient, z) / self.counts
        return tot[torch.from_numpy(patients).to(DEV)].cpu().numpy()


# --------------------------------------------------------------------------------------------------------- the DP steps

def clip_rows(g: torch.Tensor, c: float) -> torch.Tensor:
    return g * torch.clamp(c / (g.norm(dim=1, keepdim=True) + 1e-12), max=1.0)


def noisy_sum(gp: torch.Tensor, clipped: bool, sigma: float, g: torch.Generator) -> torch.Tensor:
    s = (clip_rows(gp, CLIP) if clipped else gp).sum(0)
    return s + torch.randn(s.shape, device=DEV, generator=g) * sigma * CLIP if sigma > 0 else s


def discrete_gaussian(shape, variance: float, g: torch.Generator) -> torch.Tensor:
    """N_Z(0, variance) by Canonne, Kamath and Steinke's rejection from a discrete Laplace (NeurIPS 2020, Algorithm 3),
    vectorised with floating-point acceptance: a simulation's sampler, not the page's exact one."""
    n = int(np.prod(shape))
    t = math.floor(math.sqrt(variance)) + 1
    p_geo = torch.tensor(1 - math.exp(-1 / t), device=DEV)
    out = torch.empty(0, dtype=torch.int64, device=DEV)
    while out.numel() < n:
        k = int((n - out.numel()) * 1.6) + 1024
        u1 = 1 - torch.rand(k, device=DEV, generator=g, dtype=torch.float64)      # in (0, 1]: two geometric draws by
        u2 = 1 - torch.rand(k, device=DEV, generator=g, dtype=torch.float64)      # inversion, on {0, 1, ...}
        log_keep = torch.log1p(-p_geo.double())
        y = (torch.floor(torch.log(u1) / log_keep) - torch.floor(torch.log(u2) / log_keep)).to(torch.int64)
        acc = torch.rand(k, device=DEV, generator=g, dtype=torch.float64) < torch.exp(-((y.abs().double() - variance / t) ** 2) / (2 * variance))
        out = torch.cat([out, y[acc]])
    return out[:n].view(shape)


def secure_share(gp: torch.Tensor, clipped: bool, variance: float, g: torch.Generator) -> torch.Tensor:
    """One site's contribution to the secure sum: its people's (clipped) gradients scaled by gamma, each randomly rounded
    to integers, summed, plus the site's discrete Gaussian share. The pairwise masks cancel modulo 2^32 and are not drawn;
    the caller checks the sum stays inside that range."""
    scaled = (clip_rows(gp, CLIP) if clipped else gp).double() * (2.0 ** SCALE_BITS / CLIP)
    low = torch.floor(scaled)
    s = (low + (torch.rand(scaled.shape, device=DEV, generator=g, dtype=torch.float64) < scaled - low)).to(torch.int64).sum(0)
    return s + discrete_gaussian(s.shape, variance, g) if variance > 0 else s


def apply_flat(model, flat: torch.Tensor, opt) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n
    opt.step()


# -------------------------------------------------------------------------------------------------------------- training

def train(model_name: str, arm: str, ctx: dict, people: np.ndarray, sites: list[np.ndarray], steps: int, noise: float,
          clipped: bool, seed: int, lr: float):
    """One run; returns what scores the test records, and the largest secure-sum magnitude seen (0 when not federated)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(people)
    q = EXPECTED_BATCH / n
    gamma = 2.0 ** SCALE_BITS / CLIP
    worst = 0
    if model_name == "cnn":
        model = GnCnn().to(DEV)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        def grads(ps):
            return person_grads_cnn(model, ctx["d"], ctx["sig"], ctx["feats"], ps, rng)

        def step(flat):
            apply_flat(model, flat, opt)
    else:
        head = ctx[model_name]
        theta = torch.zeros(head.dim, device=DEV)
        buf = torch.zeros_like(theta)

        def grads(ps):
            return head.person_grads(theta, ps, g)

        def step(flat):
            nonlocal theta, buf
            buf = 0.9 * buf + flat
            theta = theta - lr * buf
    for _ in range(steps):
        if arm == "central":
            chosen = people[rng.random(n) < q]
            if len(chosen):
                step(noisy_sum(grads(chosen), clipped, noise, g) / (q * n))
        elif arm == "central-full":
            step(noisy_sum(grads(people), clipped, noise, g) / n)
        else:
            total = sum(secure_share(grads(s), clipped, noise, g) for s in sites)
            worst = max(worst, int(total.abs().max()))
            step(total.double().div(gamma * n).float())
    if worst >= 2 ** 31:
        raise RuntimeError("the secure sum would wrap modulo 2^32")
    if model_name == "cnn":
        return (lambda pats: patient_scores(model, ctx["d"], ctx["sig"], ctx["feats"], pats)), worst
    return (lambda pats: head.scores(theta, pats)), worst


@torch.no_grad()
def patient_scores(model, d, sig, feats, patients: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    for p in patients:
        rr, ss = [], []
        for i in d["recs"][d["pats"][p]]:
            st = pcg.window_starts(int(d["length"][i]), pcg.WINDOW, pcg.EVAL_HOP)
            rr.append(np.full(len(st), i))
            ss.append(st)
        out.append(float(model(feats(pcg.batch(sig, d, np.concatenate(rr), np.concatenate(ss)))).mean()))
    model.train()
    return np.array(out)


def auc_people(y: np.ndarray, p: np.ndarray, owner: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC over the records, with a 95% interval from resampling PEOPLE (a person's records go together)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(owner == o) for o in np.unique(owner)]
    vals = []
    for _ in range(n_boot):
        k = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[k].min() != y[k].max():
            vals.append(pcg.roc_auc_score(y[k], p[k]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": float(pcg.roc_auc_score(y, p)), "lo": float(lo), "hi": float(hi)}


def gaussian_exact_epsilon(mu: float, delta: float = DELTA) -> float:
    """The exact (epsilon, delta) of a mu-GDP mechanism, which T Gaussian releases of sensitivity C and noise sigma C are
    with mu = sqrt(T) / sigma: delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2) (Balle and Wang, ICML 2018;
    Dong, Roth and Su, JRSS-B 2022). For continuous Gaussian noise only, so for the central full-batch arm."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def gap(e):
        return norm.cdf(-e / mu + mu / 2) - math.exp(e + norm.logcdf(-e / mu - mu / 2)) - delta
    hi = 1.0
    while gap(hi) > 0:
        hi *= 2
    return brentq(gap, 0.0, hi)


def full_epsilon(sigma: float, steps: int) -> float:
    return gaussian_exact_epsilon(math.sqrt(steps) / sigma)


def decision_thresholds(val_scores: list[np.ndarray], y: np.ndarray) -> dict:
    """Thresholds chosen where the learning rate was (the stand-in's validation records when tuning on public data):
    'prevalence' calls positive the same share as the validation set's positives, 'balanced' maximises balanced accuracy;
    each averaged over the selection seeds."""
    share = float(y.mean())
    prev, bal = [], []
    for sc in val_scores:
        prev.append(float(np.quantile(sc, 1 - share)))
        cand = np.unique(sc)
        accs = [0.5 * ((sc[y == 1] > t).mean() + (sc[y == 0] <= t).mean()) for t in cand]
        bal.append(float(cand[int(np.argmax(accs))]))
    return {"prevalence": float(np.mean(prev)), "balanced": float(np.mean(bal)), "zero": 0.0}


def balanced_at(scores: np.ndarray, y: np.ndarray, t: float) -> dict:
    pred = scores > t
    return {"threshold": t, "balanced_accuracy": float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())),
            "share_predicted_positive": float(pred.mean())}


def noise_for(model_name: str, arm: str, eps: float | None, n_people: int, steps: int, dim: int, dp) -> tuple[float, dict]:
    if eps is None:
        return 0.0, {}
    if arm == "central":
        s = sigma_for(eps, EXPECTED_BATCH / n_people, steps)
        return s, {"sigma": s, "epsilon_accounted": dpsgd_epsilon(EXPECTED_BATCH / n_people, s, steps)}
    if arm == "central-full":
        s = _bisect(lambda x: full_epsilon(x, steps), eps)
        return s, {"sigma": s, "epsilon_accounted": full_epsilon(s, steps)}
    d2 = 2.0 ** SCALE_BITS + math.sqrt(dim)
    var = dp.variance_for(eps, steps, d2, SITES, dim, DELTA)
    return var, {"variance_per_site": var, "noise_sd_over_gamma_c": math.sqrt(SITES * var) / 2.0 ** SCALE_BITS,
                 "epsilon_accounted": dp.epsilon_of(var, steps, d2, SITES, dim, DELTA)}


# --------------------------------------------------------------------------------------------------- the public features

def eval_windows(d: dict, recs) -> tuple[np.ndarray, np.ndarray]:
    rr, ss = [], []
    for i in recs:
        s = pcg.window_starts(int(d["length"][i]), pcg.WINDOW, pcg.EVAL_HOP)
        rr.append(np.full(len(s), i))
        ss.append(s)
    return np.concatenate(rr), np.concatenate(ss)


@torch.no_grad()
def window_features(cnn, feats, sig, d, rr, ss) -> tuple[torch.Tensor, torch.Tensor]:
    """Each window's 128 features from the frozen public CNN, and its log-mel band means and deviations (80)."""
    cnn.eval()
    e, h = [], []
    for s in range(0, len(rr), 1024):
        m = feats(pcg.batch(sig, d, rr[s:s + 1024], ss[s:s + 1024]))
        e.append(cnn.body(m).mean(dim=(2, 3)).float())
        h.append(torch.cat([m.mean(dim=3).flatten(1), m.std(dim=3).flatten(1)], 1).float())
    return torch.cat(e), torch.cat(h)


def public_heads(d: dict, sig, feats, public_prep: Path, out: Path, panns_path: Path | None = None) -> tuple[dict, dict]:
    pub = pcg.load(public_prep)
    psig = torch.from_numpy(np.load(public_prep / "signals.npy")).to(DEV)
    torch.manual_seed(1)
    cnn = pcg.SmallCnn().to(DEV)
    ck = out / "public_cnn.pt"
    if ck.exists():
        cnn.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
    else:
        pcg.train(cnn, feats, psig, pub, np.arange(len(pub["y"])), PUBLIC_EPOCHS,
                  lambda e: 1e-3 * 0.5 * (1 + math.cos(math.pi * e / PUBLIC_EPOCHS)), 64, 1)
        torch.save(cnn.state_dict(), ck)
    pe, ph = window_features(cnn, feats, psig, pub, *eval_windows(pub, np.arange(len(pub["y"]))))
    del psig
    allrecs = np.concatenate([d["recs"][p] for p in d["pats"]])
    rr, ss = eval_windows(d, allrecs)
    rec_patient = {int(r): k for k, p in enumerate(d["pats"]) for r in d["recs"][p]}
    win_patient = np.array([rec_patient[int(r)] for r in rr])
    ce, ch = window_features(cnn, feats, sig, d, rr, ss)
    heads = {"head-public": Head(d, (ce - pe.mean(0)) / (pe.std(0) + 1e-6), win_patient),
             "head-logmel": Head(d, (ch - ph.mean(0)) / (ph.std(0) + 1e-6), win_patient)}
    info = {"recordings": len(pub["y"]), "epochs": PUBLIC_EPOCHS, "lr": 1e-3, "batch": 64, "seed": 1,
            "public_windows": len(pe), "circor_windows": len(rr)}
    if panns_path:
        cache = out / "panns_embeddings.pt"
        if cache.exists():
            emb = torch.load(cache, map_location=DEV, weights_only=True)
        else:
            cnn14 = load_cnn14(panns_path)
            psig = torch.from_numpy(np.load(public_prep / "signals.npy")).to(DEV)
            emb = {"public": panns_embeddings(cnn14, psig, pub, *eval_windows(pub, np.arange(len(pub["y"])))),
                   "circor": panns_embeddings(cnn14, sig, d, rr, ss)}
            del psig, cnn14
            torch.save(emb, cache)
        proj = panns_projection(emb["public"])
        heads["head-panns"] = Head(d, proj(emb["circor"]), win_patient)
        info["panns"] = {"weights": panns_path.name, "embedding": 2048, "pca_dims": PANNS_DIMS, "pca_fitted_on": "public windows"}
        PANNS_CACHE["public"] = emb["public"]
        PANNS_CACHE["proj"] = proj
    return heads, info


PANNS_CACHE: dict = {}


def proxy_dataset(pub: dict, recs: np.ndarray) -> dict:
    """PhysioNet 2016 recordings shaped like `load`'s output, each recording its own record and its own person (the files
    name no patients; the data are public, so the unit only shapes the noise, it protects no one)."""
    y = pub["y"][recs].astype(np.float32)
    return {"offset": pub["offset"], "length": pub["length"], "pats": recs, "recs": {int(r): np.array([r]) for r in recs},
            "py": y, "people": recs, "person_of": np.arange(len(recs)), "members": [np.array([k]) for k in range(len(recs))],
            "person_y": y}


def public_proxy(public_prep: Path, feats, n_people: int, share: float, out: Path, sources: str = "abcdef",
                 disjoint: bool = False) -> tuple[dict, dict]:
    """The tuning stand-in made of public data only: n_people PhysioNet 2016 recordings with CirCor's share of positives,
    the heads' CNN trained on the other recordings so that the stand-in's features are not its own training data; with
    disjoint, only on the databases the stand-in is not drawn from, so that, as with CirCor, the features meet a new
    population."""
    pub = pcg.load(public_prep)
    psig = torch.from_numpy(np.load(public_prep / "signals.npy")).to(DEV)
    rng = np.random.default_rng(11)
    allowed = np.isin(pub["source"], list(sources))
    pos, neg = np.flatnonzero((pub["y"] == 1) & allowed), np.flatnonzero((pub["y"] == 0) & allowed)
    n_pos = round(n_people * share)
    chosen = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_people - n_pos, replace=False)]))
    rest = np.flatnonzero(~allowed) if disjoint else np.setdiff1d(np.arange(len(pub["y"])), chosen)
    torch.manual_seed(1)
    cnn = pcg.SmallCnn().to(DEV)
    ck = out / f"public_cnn_proxy_{sources}{'_disjoint' if disjoint else ''}.pt"
    if ck.exists():
        cnn.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
    else:
        pcg.train(cnn, feats, psig, pub, rest, PUBLIC_EPOCHS, lambda e: 1e-3 * 0.5 * (1 + math.cos(math.pi * e / PUBLIC_EPOCHS)), 64, 1)
        torch.save(cnn.state_dict(), ck)
    re_, rh = window_features(cnn, feats, psig, pub, *eval_windows(pub, rest))
    rr, ss = eval_windows(pub, chosen)
    pos_of = {int(r): k for k, r in enumerate(chosen)}
    win_patient = np.array([pos_of[int(r)] for r in rr])
    ce, ch = window_features(cnn, feats, psig, pub, rr, ss)
    d = proxy_dataset(pub, chosen)
    ctx = {"d": d, "sig": psig, "feats": feats,
           "head-public": Head(d, (ce - re_.mean(0)) / (re_.std(0) + 1e-6), win_patient),
           "head-logmel": Head(d, (ch - rh.mean(0)) / (rh.std(0) + 1e-6), win_patient)}
    if "public" in PANNS_CACHE:
        # the public windows' CNN14 embeddings in eval_windows order over all recordings; pick the stand-in's windows
        allr, _ = eval_windows(pub, np.arange(len(pub["y"])))
        keep = torch.from_numpy(np.isin(allr, chosen)).to(DEV)
        ctx["head-panns"] = Head(d, PANNS_CACHE["proj"](PANNS_CACHE["public"][keep]), win_patient)
    info = {"recordings": int(n_people), "positives": int(n_pos), "cnn_recordings": int(len(rest)), "seed": 11, "drawn_from": sources, "cnn_disjoint": disjoint,
            "sources": {s: int((pub["source"][chosen] == s).sum()) for s in sorted(set(pub["source"]))}}
    return ctx, info


def panns_embeddings(cnn14, sig, d, rr, ss, batch: int = 48) -> torch.Tensor:
    """CNN14's 2,048-d embedding of each 4-second window, upsampled from 2 kHz to its 32 kHz."""
    _spec = importlib.util.spec_from_file_location("panns_cnn14", HERE / "panns_cnn14.py")
    panns = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(panns)
    out = []
    for s in range(0, len(rr), batch):
        w = panns.upsample(pcg.batch(sig, d, rr[s:s + batch], ss[s:s + batch]), panns.RATE // pcg.RATE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(cnn14(w).float())
    return torch.cat(out)


def panns_projection(public_emb: torch.Tensor, dims: int = PANNS_DIMS):
    """A PCA fitted on the public windows only, then standardised with the public windows' statistics."""
    mu = public_emb.mean(0)
    _, _, vt = torch.linalg.svd(public_emb - mu, full_matrices=False)
    basis = vt[:dims].T
    z = (public_emb - mu) @ basis
    zm, zs = z.mean(0), z.std(0) + 1e-6
    return lambda e: (((e - mu) @ basis) - zm) / zs


def load_cnn14(path: Path):
    _spec = importlib.util.spec_from_file_location("panns_cnn14", HERE / "panns_cnn14.py")
    panns = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(panns)
    return panns.Cnn14Embedding.from_checkpoint(path).to(DEV)


# ------------------------------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--public-prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="cnn,head-public,head-logmel")
    ap.add_argument("--grid", action="append", default=[], metavar="MODEL[:ARM]=LR,LR,...",
                    help="replace a model's (or one arm's) learning-rate grid, e.g. head-public=0.003,0.01,0.03")
    ap.add_argument("--panns", type=Path, default=None, help="Cnn14_mAP=0.431.pth: adds the head-panns model")
    ap.add_argument("--tune-on", choices=("circor", "public"), default="circor",
                    help="circor: validation people from CirCor's training people; public: a PhysioNet 2016 stand-in of the "
                         "same size, so tuning spends no CirCor privacy")
    ap.add_argument("--proxy-sources", default="abcdef", help="the Challenge 2016 databases the public stand-in is drawn from")
    ap.add_argument("--proxy-cnn-disjoint", action="store_true",
                    help="train the stand-in's feature CNN only on the databases the stand-in is not drawn from")
    ap.add_argument("--smoke", action="store_true", help="a few steps, one seed, two rates: that the pipeline runs")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dp = _qsh_dp()
    d = load(a.prep)
    sig = torch.from_numpy(np.load(a.prep / "signals.npy")).to(DEV)
    feats = pcg.Features().to(DEV)
    train_people, test_people = split(d)
    fit_people, val_people = holdout(d, train_people, 7)
    models = a.models.split(",")
    override = {}
    for g in a.grid:
        key, vals = g.split("=")
        model_name, _, arm = key.partition(":")
        for ar in ([arm] if arm else ARMS[model_name]):
            override[(model_name, ar)] = tuple(float(v) for v in vals.split(","))
    ctx = {"d": d, "sig": sig, "feats": feats}
    report = {"environment": pcg.environment(), "delta": DELTA, "clip": CLIP, "windows_per_record": WINDOWS_PER_PATIENT,
              "pos_weight": POS_WEIGHT, "expected_batch": EXPECTED_BATCH, "epochs": EPOCHS, "sites": SITES, "scale_bits": SCALE_BITS,
              "data": {"records_labelled": len(d["pats"]), "people": len(d["members"]),
                       "people_with_two_records": int(sum(len(m) == 2 for m in d["members"])),
                       "train_people": len(train_people), "test_people": len(test_people),
                       "test_records": len(records_of(d, test_people)),
                       "test_records_present": int(d["py"][records_of(d, test_people)].sum()),
                       "fit_people": len(fit_people), "val_people": len(val_people)},
              "params": {"cnn": int(sum(p.numel() for p in GnCnn().parameters()))}, "selection": [], "runs": []}
    if any(m.startswith("head") for m in models):
        heads, report["public_cnn"] = public_heads(d, sig, feats, a.public_prep, a.out, a.panns)
        ctx.update(heads)
        report["params"].update({k: int(h.dim) for k, h in heads.items()})
    if a.tune_on == "public":
        share = float(d["person_y"][train_people].mean())
        sel_ctx, report["tuned_on"] = public_proxy(a.public_prep, feats, len(train_people), share, a.out, a.proxy_sources,
                                                   a.proxy_cnn_disjoint)
        sel_fit, sel_val = holdout(sel_ctx["d"], np.arange(len(sel_ctx["d"]["pats"])), 7)
        report["tuned_on"].update({"kind": "public", "fit": len(sel_fit), "val": len(sel_val), "positive_share": share})
    else:
        sel_ctx, sel_fit, sel_val = ctx, fit_people, val_people
        report["tuned_on"] = {"kind": "circor", "fit": len(fit_people), "val": len(val_people)}
    sel_d = sel_ctx["d"]
    sel_val_recs = records_of(sel_d, sel_val)
    test_recs = records_of(d, test_people)
    test_owner = d["person_of"][test_recs]
    for model_name in models:
        family = "cnn" if model_name == "cnn" else "head"
        dim = report["params"][model_name]
        for arm in ARMS[model_name]:
            grid = override.get((model_name, arm), GRIDS.get((model_name, arm), HEAD_GRID))
            sel_seeds = SEEDS[family][arm]
            test_seeds = SEEDS[family]["test"]
            if a.smoke:
                grid, sel_seeds, test_seeds = grid[:2], (1,), (1,)
            for kind, eps in KINDS:
                clipped = kind != "non-private"
                fit_steps = 3 if a.smoke else int(EPOCHS * len(sel_fit) / EXPECTED_BATCH)
                fit_noise, _ = noise_for(model_name, arm, eps, len(sel_fit), fit_steps, dim, dp)
                fit_sites = sites_of(sel_d, sel_fit)
                tried, val_scores = [], {}
                y_sel = sel_d["py"][sel_val_recs]
                for lr in grid:
                    v = []
                    for seed in sel_seeds:
                        score, _ = train(model_name, arm, sel_ctx, sel_fit, fit_sites, fit_steps, fit_noise, clipped, seed, lr)
                        sv = score(sel_val_recs)
                        val_scores.setdefault(lr, []).append(sv)
                        v.append(float(pcg.roc_auc_score(y_sel, sv)))
                    tried.append({"lr": lr, "val_auroc": v})
                best = max(tried, key=lambda t: np.mean(t["val_auroc"]))
                thresholds = decision_thresholds(val_scores[best["lr"]], y_sel)
                report["selection"].append({"model": model_name, "arm": arm, "kind": kind, "epsilon": eps, "tried": tried,
                                            "chosen_lr": best["lr"], "thresholds": thresholds})
                steps = 3 if a.smoke else int(EPOCHS * len(train_people) / EXPECTED_BATCH)
                noise, extra = noise_for(model_name, arm, eps, len(train_people), steps, dim, dp)
                sites = sites_of(d, train_people)
                for seed in test_seeds:
                    t = time.time()
                    score, worst = train(model_name, arm, ctx, train_people, sites, steps, noise, clipped, seed, best["lr"])
                    s_test = score(test_recs)
                    pred = s_test > 0
                    y_test = d["py"][test_recs]
                    run = {"model": model_name, "arm": arm, "kind": kind, "epsilon": eps, "seed": seed, "lr": best["lr"],
                           "steps": steps, "train_s": time.time() - t,
                           "auroc": auc_people(y_test, s_test, test_owner),
                           # at the logit threshold 0, with the class weight the loss used: a collapse onto one class shows
                           # as a balanced accuracy near 0.5 and a share predicted positive near 0 or 1 (Wen et al. 2026)
                           "balanced_accuracy": float(0.5 * (pred[y_test == 1].mean() + (~pred[y_test == 0]).mean())),
                           "share_predicted_positive": float(pred.mean()),
                           "at_thresholds": {name: balanced_at(s_test, y_test, t) for name, t in thresholds.items()}, **extra}
                    if arm == "federated":
                        run["max_abs_sum"] = worst
                    report["runs"].append(run)
                    print(json.dumps({k: run[k] for k in ("model", "arm", "kind", "epsilon", "seed", "lr")} | {"auroc": round(run["auroc"]["auroc"], 3)}), flush=True)
                with open(a.out / "report.json", "w") as f:
                    json.dump(report, f, indent=1)
    report["total_s"] = time.time() - t0
    with open(a.out / "report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
