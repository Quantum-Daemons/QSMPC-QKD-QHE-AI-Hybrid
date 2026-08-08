"""Assert the decoy-state reference model against PUBLISHED values, not against itself.

Every assertion here names the paper and the figure or statement it comes from. If one of these
fails, the model is wrong - not the test - because none of these numbers originated in this
repository.

The published curves are plotted for the efficient BB84 of Lo-Chau-Ardehali, where the sifting
factor q tends to 1. This project runs standard BB84 (q = 1/2), so the comparisons below construct
a channel with ``q_sift=1.0`` to meet the paper on its own terms. Without that, every rate here is
exactly half the published one - close enough to look like a fitting discrepancy and wrong enough
to invalidate the comparison.
"""
from __future__ import annotations

import math

from qsh.qkd.decoy_reference import (
    ChannelSpec, gain, h2, key_rate_decoy, key_rate_nondecoy, max_distance_km,
    optimal_decoy, optimal_nondecoy, qber, tagged_fraction, transmittance,
)

PAPER = ChannelSpec(q_sift=1.0)     # efficient BB84, as Ma et al. plot it
GYS = ChannelSpec()                 # this project's standard-BB84 default


def test_gys_defaults_are_the_published_parameter_set():
    """Ma et al. (2005) Table 1, from Gobby-Yuan-Shields APL 84, 3762 (2004)."""
    assert GYS.alpha_db_per_km == 0.21
    assert GYS.detector_eff == 0.045
    assert GYS.dark_count == 1.7e-6
    assert GYS.misalignment == 0.033
    assert GYS.f_ec == 1.22


def test_binary_entropy_endpoints():
    assert h2(0.5) == 1.0
    assert h2(0.0) == 0.0 and h2(1.0) == 0.0
    assert math.isclose(h2(0.11), 0.4999, abs_tol=1e-3)


def test_dark_count_error_is_one_half():
    """A background click is uncorrelated with the prepared state, so it is wrong half the time.

    At zero mean photon number the only detections are dark counts, so the QBER must be exactly
    e_0 = 1/2 rather than the misalignment figure.
    """
    assert math.isclose(qber(0.0, transmittance(0.0, GYS), GYS), 0.5, rel_tol=1e-9)


def test_qber_floor_sits_just_above_the_misalignment():
    """At short range the QBER tends to e_det from ABOVE, never below it.

    The direction is the physics worth asserting. Dark counts add error at e_0 = 1/2, so they can
    only push the QBER up; a model that returned exactly e_det would be ignoring them, and one that
    returned less would be unphysical. Measured excess here is ~0.0004 percentage points.
    """
    e = qber(0.5, transmittance(0.0, GYS), GYS)
    assert e > GYS.misalignment
    assert math.isclose(e, GYS.misalignment, abs_tol=1e-3)


def test_tagged_fraction_exceeds_one_at_zero_distance_for_mu_0_1():
    """The sharpest published claim, and the one most often gotten wrong.

    Lo, Ma and Chen note that experimentalists "liberally pick 0.1 as a convenient number for
    average photon number without any security justification". With the GYS parameter set, a
    NON-decoy link at mu = 0.1 has a tagged fraction above 1 at ZERO distance: no detection can be
    certified as single-photon, so the key rate is zero even back to back.
    """
    eta0 = transmittance(0.0, GYS)
    delta = tagged_fraction(0.1, eta0, GYS)
    assert delta > 1.0, f"expected Delta > 1, got {delta}"
    assert math.isclose(delta, 1.0417, abs_tol=5e-3)
    assert key_rate_nondecoy(0.1, 0.0, GYS) == 0.0


def test_decoy_reaches_past_140_km():
    """Lo-Ma-Chen (PRL 94, 230504) report decoy BB84 "over 140 km" with these parameters."""
    d = max_distance_km(key_rate_decoy, PAPER)
    assert d > 140.0, f"decoy max distance {d} km, expected > 140"


def test_nondecoy_dies_around_40_km():
    """Without decoy states the same hardware reaches only tens of kilometres."""
    d = max_distance_km(key_rate_nondecoy, PAPER)
    assert 30.0 < d < 50.0, f"non-decoy max distance {d} km, expected ~42"
    assert d < max_distance_km(key_rate_decoy, PAPER) / 3.0


def test_decoy_rate_scales_linearly_in_eta():
    """R/eta is flat at about 0.11 - the decoy method's headline result.

    Linear in eta, versus quadratic without decoy states. Scarani et al., Rev. Mod. Phys. 81, 1301
    (2009) section II.1.3 restates it as t versus t^2.
    """
    for L in (0.0, 20.0, 40.0, 60.0):
        eta = transmittance(L, PAPER)
        _, r = optimal_decoy(L, PAPER)
        assert math.isclose(r / eta, 0.11, abs_tol=0.01), f"L={L}: R/eta = {r / eta}"


def test_nondecoy_rate_scales_quadratically_in_eta():
    """R/eta^2 is about 0.15 where the non-decoy link still produces key at all."""
    eta = transmittance(0.0, PAPER)
    _, r = optimal_nondecoy(0.0, PAPER)
    assert math.isclose(r / eta**2, 0.15, abs_tol=0.01), f"R/eta^2 = {r / eta**2}"


def test_decoy_optimal_mu_is_about_one_half_not_one_tenth():
    """Ma et al. put the decoy optimum near mu = 0.5, well above the customary 0.1."""
    mu, _ = optimal_decoy(25.0, PAPER)
    assert 0.3 < mu < 0.8, f"optimal decoy mu = {mu}, expected ~0.5"


def test_decoy_beats_nondecoy_at_every_distance():
    for L in (0.0, 10.0, 25.0, 50.0, 100.0):
        assert optimal_decoy(L, PAPER)[1] >= optimal_nondecoy(L, PAPER)[1]


def test_gain_is_monotonic_and_bounded():
    eta = transmittance(0.0, GYS)
    prev = -1.0
    for mu in (0.0, 0.1, 0.5, 1.0, 5.0):
        g = gain(mu, eta, GYS)
        assert 0.0 <= g <= 1.0 + GYS.dark_count
        assert g >= prev
        prev = g


def test_standard_bb84_is_exactly_half_the_efficient_rate():
    """The factor that would silently invalidate any comparison against the published figures."""
    for L in (0.0, 30.0):
        assert math.isclose(optimal_decoy(L, GYS)[1] * 2.0, optimal_decoy(L, PAPER)[1], rel_tol=1e-9)
