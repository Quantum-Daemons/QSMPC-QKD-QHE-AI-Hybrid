"""AI-driven QKD parameter optimization (OptiQKD, WS1).

A compact, honest, **measured** optimizer that tunes decoy-state + efficient-BB84 parameters
(``mu_signal``, ``basis_prob``, ``pp_threshold``) to maximise the secret-key rate (SKR) over the
analytic BB84 channel model in :mod:`~qsh.qkd.bb84`. It reports the SKR improvement measured against
the **default-parameter baseline on the same simulated channel** — a simulation, not deployed QKD, and
not a hardware claim. This is the concrete stand-in for the report's WS1 "OptiQKD" theme
(arXiv:2603.04192) built from commercial-OK OSS (scikit-learn) only — no RL / torch.

Methods (config-selectable via ``qkd.optimize_method``):
  * ``"coordinate"`` — gradient-free coordinate ascent on the true simulator.
  * ``"surrogate"``  — sample the space, fit a GradientBoostingRegressor, argmax on a dense grid, then
                       polish by coordinate ascent on the true model (default).
  * ``"gp"``         — GaussianProcessRegressor (Bayesian) surrogate + polish.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .bb84 import QKDParams, run_bb84
from .channel import ChannelParams

# --- optimizer internals (algorithm hyperparameters, not user-facing tunables) ---
_GBM_ESTIMATORS = 100
_GBM_MAX_DEPTH = 3
_SURROGATE_GRID = 2000        # dense random grid the fitted surrogate is argmaxed over
_POLISH_ROUNDS = 6            # coordinate-ascent polish rounds on the TRUE simulator
_POLISH_STEP_DIV = 8.0        # initial per-dimension step = (hi-lo)/this
_POLISH_SHRINK = 0.6          # step shrink factor each round


@dataclass
class OptimizeResult:
    """Outcome of an OptiQKD run."""

    best_params: QKDParams
    best_skr: float
    baseline_skr: float
    improvement_pct: float
    best_qber: float
    baseline_qber: float
    n_evals: int
    method: str
    history: list
    convergence_evals: int = 0          # evaluations until the running best first reached 95% of the final SKR
    convergence_curve: list | None = None   # running best-SKR after each evaluation (the adaptation profile)


def _lhs(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Latin-hypercube samples in the unit cube [0,1]^d."""
    out = np.zeros((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def _space_from_config(space: dict | None, protocol: str = "bb84") -> dict:
    from ..config import get_config

    q = get_config().qkd
    if space:
        return space
    if protocol == "e91":     # E91 tunes only the basis bias + post-processing threshold (no decoy intensities)
        return {"basis_prob": (q.opt_basis_prob_min, q.opt_basis_prob_max),
                "pp_threshold": (q.opt_pp_threshold_min, q.opt_pp_threshold_max)}
    return {
        "mu_signal": (q.opt_mu_signal_min, q.opt_mu_signal_max),
        "mu_decoy": (q.opt_mu_decoy_min, q.opt_mu_decoy_max),
        "basis_prob": (q.opt_basis_prob_min, q.opt_basis_prob_max),
        "pp_threshold": (q.opt_pp_threshold_min, q.opt_pp_threshold_max),
    }


def evaluate_params(qkd_params: QKDParams, channel: ChannelParams, *, n_pulses: int,
                    seed: int, ecc: float, protocol: str = "bb84") -> tuple[float, float]:
    """Measure (SKR, QBER) for a parameter set on the channel (fixed seed => deterministic).

    ``protocol="e91"`` evaluates the entanglement-based E91/CHSH model (:func:`~qsh.qkd.e91.run_e91`) with the
    tuned ``basis_prob``/``pp_threshold`` (decoy intensities do not apply); otherwise the decoy-state BB84 model.
    """
    if protocol == "e91":
        from ..config import get_config
        from .e91 import run_e91
        r = run_e91(n_pulses, channel, visibility=get_config().qkd.e91_visibility,
                    basis_prob=qkd_params.basis_prob, error_correction_eff=ecc, seed=seed,
                    pp_threshold=qkd_params.pp_threshold)
    else:
        r = run_bb84(n_pulses, channel, error_correction_eff=ecc, seed=seed, qkd_params=qkd_params)
    return r.secret_key_rate, r.qber


def optimize_qkd(channel: ChannelParams | None = None, *, method: str | None = None,
                 n_samples: int | None = None, seed: int = 0, space: dict | None = None,
                 n_pulses: int | None = None, protocol: str | None = None,
                 baseline: QKDParams | None = None) -> OptimizeResult:
    """Tune QKD parameters to maximise SKR; return the measured improvement vs the default baseline.

    ``protocol`` (``None`` -> config) selects the physical model: ``"bb84"`` tunes the decoy-state +
    biased-basis parameters; ``"e91"`` tunes the entanglement-based (CHSH) basis bias + threshold.

    ``baseline`` is the link's own source (``None`` -> the fibre config's ``QKDParams()``). Every
    candidate starts from it, so what the search does not tune - whether a vacuum state is sent, how
    often each state is - stays the link's. The satellite passes Micius's (Round 77); it was measured
    against the fibre's intensities until then.
    """
    from ..config import get_config

    q = get_config().qkd
    channel = channel or ChannelParams(
        distance_km=q.distance_km, fiber_loss_db_per_km=q.fiber_loss_db_per_km,
        detector_efficiency=q.detector_efficiency, dark_count_prob=q.dark_count_prob,
        misalignment=q.misalignment, eve_intercept_fraction=q.eve_intercept_fraction)
    base = baseline or QKDParams()
    method = method or q.optimize_method
    protocol = protocol or getattr(q, "protocol", "bb84")
    n_samples = int(n_samples or q.optimize_n_samples)
    n_pulses = int(n_pulses or q.optimize_n_pulses)
    ecc = q.error_correction_eff
    space = _space_from_config(space, protocol)
    keys = list(space.keys())
    lo = np.array([space[k][0] for k in keys], dtype=float)
    hi = np.array([space[k][1] for k in keys], dtype=float)
    ev_seed = int(seed) + 1                     # one fixed Monte-Carlo realization -> deterministic surface
    rng = np.random.default_rng(seed)
    history: list = []

    def make_params(vec: np.ndarray) -> QKDParams:
        # build QKDParams from whatever tunables the search space carries (mu_signal, mu_decoy, basis_prob,
        # pp_threshold) -> adding a dimension to the space needs no change here
        d = {k: float(v) for k, v in zip(keys, vec)}
        return replace(base, **{k: v for k, v in d.items() if k in QKDParams.__dataclass_fields__})

    def score(vec: np.ndarray) -> float:
        skr, _ = evaluate_params(make_params(vec), channel, n_pulses=n_pulses, seed=ev_seed, ecc=ecc,
                                 protocol=protocol)
        return skr

    baseline_skr, baseline_qber = evaluate_params(
        base, channel, n_pulses=n_pulses, seed=ev_seed, ecc=ecc, protocol=protocol)
    n_evals = 1

    # --- global search: sample + surrogate argmax (surrogate/gp) or midpoint start (coordinate) ---
    if method in ("surrogate", "gp"):
        samples = _lhs(rng, n_samples, len(keys)) * (hi - lo) + lo
        y = np.array([score(s) for s in samples])
        n_evals += len(samples)
        history += [(s.tolist(), float(v)) for s, v in zip(samples, y)]
        if method == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            model = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True,
                                             alpha=1e-9, random_state=seed).fit(samples, y)
        else:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(n_estimators=_GBM_ESTIMATORS, max_depth=_GBM_MAX_DEPTH,
                                              random_state=seed).fit(samples, y)
        grid = _lhs(rng, _SURROGATE_GRID, len(keys)) * (hi - lo) + lo
        best_vec = grid[int(np.argmax(model.predict(grid)))].astype(float)
    else:
        best_vec = (lo + hi) / 2.0

    # --- coordinate-ascent polish on the TRUE simulator ---
    best_skr = score(best_vec)
    n_evals += 1
    step = (hi - lo) / _POLISH_STEP_DIV
    for _ in range(_POLISH_ROUNDS):
        improved = False
        for j in range(len(keys)):
            for delta in (step[j], -step[j]):
                cand = best_vec.copy()
                cand[j] = float(np.clip(cand[j] + delta, lo[j], hi[j]))
                s = score(cand)
                n_evals += 1
                history.append((cand.tolist(), float(s)))
                if s > best_skr:
                    best_skr, best_vec, improved = s, cand, True
        step = step * _POLISH_SHRINK
        if not improved:
            break

    best_params = make_params(best_vec)
    best_skr, best_qber = evaluate_params(best_params, channel, n_pulses=n_pulses, seed=ev_seed, ecc=ecc,
                                          protocol=protocol)
    improvement = (100.0 * (best_skr - baseline_skr) / baseline_skr) if baseline_skr > 1e-12 else (
        100.0 if best_skr > 0 else 0.0)
    # convergence profile: running-best SKR after each evaluation + evals to reach 95% of the final SKR
    # (an honest "adaptation speed" proxy — cf. the papers' real-time re-tune-in-N-blocks framing)
    curve, run_best = [], -1.0
    for _, s in history:
        run_best = max(run_best, s)
        curve.append(run_best)
    final = curve[-1] if curve else best_skr
    conv_evals = next((i + 1 for i, v in enumerate(curve) if v >= 0.95 * final), len(curve))
    return OptimizeResult(
        best_params=best_params, best_skr=best_skr, baseline_skr=baseline_skr,
        improvement_pct=improvement, best_qber=best_qber, baseline_qber=baseline_qber,
        n_evals=n_evals, method=method, history=history,
        convergence_evals=int(conv_evals), convergence_curve=[float(c) for c in curve])
