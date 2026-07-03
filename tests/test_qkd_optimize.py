"""AI-QKD parameter optimization (OptiQKD, WS1) tests — offline, deterministic."""

from qsh.config import get_config
from qsh.qkd.bb84 import QKDParams, run_bb84
from qsh.qkd.channel import ChannelParams
from qsh.qkd.e91 import run_e91
from qsh.qkd.optimize import optimize_qkd

_CH = ChannelParams(distance_km=50, eve_intercept_fraction=0.0)


def test_optimizer_improves_skr_over_baseline_coordinate():
    res = optimize_qkd(_CH, method="coordinate", n_samples=24, seed=0, n_pulses=20_000)
    assert res.best_skr >= res.baseline_skr
    assert res.improvement_pct > 20.0                 # report WS1 target


def test_optimizer_improves_skr_over_baseline_surrogate():
    res = optimize_qkd(_CH, method="surrogate", n_samples=24, seed=0, n_pulses=20_000)
    assert res.improvement_pct > 20.0
    assert res.n_evals > 24


def test_optimizer_deterministic_under_seed():
    a = optimize_qkd(_CH, method="coordinate", n_samples=16, seed=0, n_pulses=20_000)
    b = optimize_qkd(_CH, method="coordinate", n_samples=16, seed=0, n_pulses=20_000)
    assert a.best_params.basis_prob == b.best_params.basis_prob
    assert a.best_skr == b.best_skr


def test_optimizer_respects_search_space():
    p = optimize_qkd(_CH, method="coordinate", seed=0, n_pulses=20_000).best_params
    q = get_config().qkd
    assert q.opt_basis_prob_min <= p.basis_prob <= q.opt_basis_prob_max
    assert q.opt_mu_signal_min <= p.mu_signal <= q.opt_mu_signal_max
    assert q.opt_pp_threshold_min <= p.pp_threshold <= q.opt_pp_threshold_max


def test_qkd_params_have_an_effect():
    """qkd_params must genuinely change the outcome (biased-basis + decoy model), while params=None is stable."""
    ch = ChannelParams(distance_km=40)
    legacy_a = run_bb84(30_000, ch, seed=5)
    legacy_b = run_bb84(30_000, ch, seed=5)
    assert legacy_a.qber == legacy_b.qber and legacy_a.sifted_bits == legacy_b.sifted_bits   # deterministic
    biased = run_bb84(30_000, ch, seed=5, qkd_params=QKDParams(basis_prob=0.85))
    assert biased.sifted_bits != legacy_a.sifted_bits                                          # params matter


def test_optimizer_qber_not_worse():
    res = optimize_qkd(_CH, method="surrogate", n_samples=24, seed=0, n_pulses=20_000)
    assert res.best_qber <= res.baseline_qber + 0.03      # QBER stays controlled while SKR rises


# --- Round 16: real two-intensity decoy-state (mu_decoy) + convergence metric + E91 protocol ---

def test_mu_decoy_genuinely_affects_skr():
    """mu_decoy now moves the SKR through the two-intensity Y1/e1 single-photon bounds (Ma 2005)."""
    ch = ChannelParams(distance_km=50)
    a = run_bb84(20_000, ch, seed=1, qkd_params=QKDParams(mu_decoy=0.05)).secret_key_rate
    b = run_bb84(20_000, ch, seed=1, qkd_params=QKDParams(mu_decoy=0.20)).secret_key_rate
    assert a != b


def test_optimizer_tunes_mu_decoy_in_range():
    p = optimize_qkd(_CH, method="coordinate", seed=0, n_pulses=20_000).best_params
    q = get_config().qkd
    assert q.opt_mu_decoy_min <= p.mu_decoy <= q.opt_mu_decoy_max


def test_optimizer_reports_convergence():
    res = optimize_qkd(_CH, method="surrogate", n_samples=24, seed=0, n_pulses=20_000)
    assert res.convergence_curve and 1 <= res.convergence_evals <= len(res.convergence_curve)
    assert res.convergence_curve == sorted(res.convergence_curve)     # running-best SKR is non-decreasing


def test_e91_bell_violation_gates_key():
    ch = ChannelParams(distance_km=50)
    good = run_e91(50_000, ch, visibility=0.97, seed=1)
    assert good.secret_key_rate > 0 and good.sifted_bits > 0
    bad = run_e91(50_000, ch, visibility=0.60, seed=1)               # CHSH S <= 2 -> no Bell violation
    assert bad.secret_key_rate == 0.0


def test_e91_optimizer_improves_skr():
    res = optimize_qkd(ChannelParams(distance_km=50), protocol="e91", method="coordinate",
                       n_samples=16, seed=0, n_pulses=50_000)
    assert res.best_skr >= res.baseline_skr
    assert res.best_params.basis_prob >= 0.5      # biased basis raises the E91 matching-basis sift ratio
