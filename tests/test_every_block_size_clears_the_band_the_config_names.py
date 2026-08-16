"""Every BB84 block size this project ships must clear the finite-key threshold, or say why not.

WHY THIS EXISTS

`src/qsh/config/defaults.py` justifies its block size at length, and until Round 47 the
justification it gave was applied to ONE of the two keys it governs. `QKDConfig.pulses` was raised
on finite-key grounds; `WebConfig.demo_alice_bob_pulses`, four hundred and sixty lines further down
the same file, sat at 400,000 - a block the same reasoning rules out.

That is this round's defect class in a third dimension. Round 44 changed one language. Round 46
changed one language again, and this round found a fourth instance inside its commit. Here the two
sides are not two languages but two config keys with one shared justification, and the gate that
existed could not see it because it checked call sites rather than values.

The threshold itself is COMPUTED, from `src/qsh/qkd/finite_key.py`, at the channel and the
intensities this project actually ships. The band that used to be quoted here - "~5e3 to 1.1e4
sifted bits", attributed to Rusca et al. - was wrong by about an order of magnitude and appears
nowhere in that paper; see `finite_key.py` for the full record.

HOW THE SCOPE IS DERIVED

By introspecting the config dataclasses for every integer field whose name contains "pulses",
rather than by naming the two keys known to matter. A key added tomorrow is in scope on the day it
is added. A block size that legitimately cannot clear the band - the free-space downlink, where
24.48 dB of loss at 800 km is physics rather than a setting - must be DECLARED here with its
reason, so the exception is visible instead of absent.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the shipped rate is finite-key correct. No finite-key correction is applied anywhere in
this project and the UI says so. What is asserted is the weaker, checkable thing the config comment
actually claims: that the block being reported on is one that COULD carry a key.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Block sizes that cannot clear the threshold, each with the reason it is exempt. A name may only
#: appear here with a justification that is PHYSICAL or that shows the block is not a published
#: quantity - "it would be slow" is not one, since the browser's `bb84()` is analytic and the
#: Python cost is tens of milliseconds.
DECLARED_RESIDUALS = {
    "demo_sat_pulses": (
        "free-space 800 km downlink, 24.48 dB loss. The shortfall is the link budget, not the "
        "setting: clearing the threshold there needs a block far beyond anything this Monte Carlo "
        "can allocate, and `config/defaults.py` records the residual beside the value rather than "
        "engineering it away."
    ),
    "optimize_n_pulses": (
        "not a published block. This is the per-EVALUATION sample the OptiQKD parameter search "
        "draws, and what the page publishes from it is an improvement RATIO plus the optimised "
        "per-pulse fraction, neither of which is a key length. Measured rather than asserted: the "
        "reported improvement is identical to four significant figures at 20,000, 200,000 and "
        "2,000,000 pulses per evaluation - the INVARIANCE across block sizes is the exemption's "
        "justification, and it holds seed by seed. The VALUE varies with the seed: 113.429% at "
        "seed 0, and 113.429 / 113.431 / 106.354 / 109.1885 / 113.3703 for seeds 0-4, each of "
        "those identical at all three pulse counts. That is because the decoy bound is analytic in "
        "the channel and the sampled QBER only enters through an abort test that never fires at "
        "this operating point. THE FIGURE HERE READ 77.922% AND WAS STALE ON ARRIVAL: that is the "
        "pre-Round-47 measurement, and this round's own move of mu_signal 0.5 -> 0.3416 - committed "
        "alongside this file - lowered the baseline SKR from 0.002368 to 0.0019765 bit/pulse while "
        "leaving the optimum unchanged, so the ratio rose. Raising the sample would cost 48x the "
        "work and move no digit of anything displayed."
    ),
}


def _threshold(n_pulses_hint: int = 0) -> int:
    """The positive-key threshold on the shipped fibre, COMPUTED rather than quoted.

    ROUND 47 REPLACED A CONSTANT HERE. This file previously carried
    `FINITE_KEY_FLOOR_SIFTED = 5_000` with the comment that it came from Rusca et al.'s band of
    ~5e3 to 1.1e4 sifted bits. That band was wrong by about an order of magnitude AND was never in
    that paper - see `src/qsh/qkd/finite_key.py`. A gate that hardcodes a threshold is a gate that
    goes stale silently when the threshold is corrected, which is what happened here: this file was
    written earlier in the same round that refuted its own floor.

    AND THE INTENSITIES COME FROM CONFIG TOO. The first version of this helper called
    `minimum_pulses_for_positive_key(ch)` with the module's DEFAULT `FiniteKeyParams`, which are
    mu = 0.5 / nu = 0.1 - the intensities that shipped until this round. It therefore computed a
    threshold of 3.84e7 for a demo that runs at 0.3416 / 0.1417, and reported the shipped 2e7 block
    as failing when it clears its actual threshold of **1.406e7 by a factor of 1.42**. A gate that
    reads the channel from config and the intensities from a default is the same half-a-fix this
    round is named for, committed inside the gate that exists to catch it.

    AND THE THRESHOLD IS 1.406e7, NOT THE 1.31e7 THIS DOCSTRING QUOTED FOR ONE ROUND. This helper
    returns 14,055,773 - measurable by calling it. 13,078,257 (1.31e7) is what `optimise_parameters`
    reports for its FULL solution, which also sets `p_signal` = 0.3764 and `p_z` = 0.5042; neither
    ships (there is no `p_signal` config field and `basis_prob` is 0.5), so quoting the optimiser's
    number as this build's threshold credited the config with two variables it does not carry.
    `src/qsh/config/defaults.py` records the same correction.
    """
    from qsh.config import get_config
    from qsh.qkd.channel import ChannelParams
    from qsh.qkd.finite_key import FiniteKeyParams, minimum_pulses_for_positive_key

    c = get_config().qkd
    ch = ChannelParams(distance_km=c.distance_km, fiber_loss_db_per_km=c.fiber_loss_db_per_km,
                       detector_efficiency=c.detector_efficiency,
                       dark_count_prob=c.dark_count_prob, misalignment=c.misalignment)
    p = FiniteKeyParams(mu_signal=c.mu_signal, mu_decoy=c.mu_decoy,
                        f_ec=c.error_correction_eff)
    n = minimum_pulses_for_positive_key(ch, p)
    assert n is not None, "no block size yields a composable key on the shipped fibre at all"
    return n


def _block_size_fields() -> dict[str, int]:
    """Every integer config field whose name says it is a count of pulses."""
    from qsh.config.defaults import QKDConfig, WebConfig

    found: dict[str, int] = {}
    for cls in (QKDConfig, WebConfig):
        for f in fields(cls):
            if "pulses" in f.name and isinstance(getattr(cls(), f.name), int):
                found[f.name] = int(getattr(cls(), f.name))
    return found


def _sifted_at(n_pulses: int) -> int:
    """Sifted bits this block yields on the SHIPPED fibre channel, from the shipped model."""
    from qsh.qkd.bb84 import QKDParams, run_bb84
    from qsh.qkd.channel import ChannelParams
    from qsh.config import get_config

    c = get_config().qkd
    res = run_bb84(n_pulses, ChannelParams(distance_km=c.distance_km), seed=1,
                   qkd_params=QKDParams())
    return int(res.sifted_bits)


def test_the_scan_has_a_subject() -> None:
    """Non-vacuity: introspection that finds nothing would make every check below pass."""
    found = _block_size_fields()
    assert len(found) >= 3, (
        f"found only {len(found)} block-size config fields ({sorted(found)}); the fibre block, the "
        "two-party demo block and the satellite block exist at minimum. Introspection is not "
        "seeing fields it should, so the assertions below have no subject."
    )
    unknown = set(DECLARED_RESIDUALS) - set(found)
    assert not unknown, (
        f"these declared residuals name config fields that no longer exist: {sorted(unknown)}. A "
        "stale exemption is how a real regression gets waved through."
    )


def test_every_fibre_block_clears_the_finite_key_threshold() -> None:
    """The property the config comment states, applied to every key it governs rather than one.

    Compared in PULSES against the computed threshold, because that is the unit the config fields
    are in and the conversion to sifted bits is exactly what the refuted band got wrong.
    """
    threshold = _threshold()
    offenders: list[str] = []
    for name, n in sorted(_block_size_fields().items()):
        if name in DECLARED_RESIDUALS:
            continue
        if n < threshold:
            offenders.append(f"{name} = {n:_} pulses, below the computed threshold of "
                             f"{threshold:_}")
    assert not offenders, (
        "these shipped block sizes publish an asymptotic rate for a block that a composably-secure "
        "finite-key analysis says carries no key:\n  " + "\n  ".join(offenders)
        + "\n\nRaise them, or declare them in DECLARED_RESIDUALS with a reason that is physical or "
        "shows the block is not a published quantity. The cost of raising is zero in the browser - "
        "`quantumSim.ts::bb84` is analytic, not a per-pulse loop."
    )


def test_the_satellite_block_is_still_the_two_that_must_match() -> None:
    """`_SAT_DEFAULTS["sat_pulses"]` carries a comment saying it is "kept in step with
    `WebConfig.demo_sat_pulses`". A comment is not a mechanism; this is.
    """
    from qsh.config.defaults import WebConfig
    from qsh.usecases.satellite_comms import _SAT_DEFAULTS

    stage = int(_SAT_DEFAULTS["sat_pulses"])
    web = int(WebConfig().demo_sat_pulses)
    assert stage == web, (
        f"the satellite stage runs {stage:_} pulses and the browser's config says {web:_}. These "
        "are two copies of one number with a comment promising they stay equal; they have "
        "diverged, which is the same failure the two-party block had one file over."
    )
