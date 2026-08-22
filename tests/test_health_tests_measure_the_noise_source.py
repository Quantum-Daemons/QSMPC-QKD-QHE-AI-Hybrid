"""The SP 800-90B health tests must measure the NOISE SOURCE, not the conditioner.

WHY THIS EXISTS

`simulator.py` called `health_suite(bits, ...)` where `bits` was the post-Toeplitz output, while
`raw` sat two lines above in the same scope. `health.py`'s opening docstring states the contract that
violates - section 4.4's continuous tests "run on raw samples **before any conditioning**" - and the
comment directly above the call inside `health_suite` said "on the raw stream", describing an
argument the function never received. Three places said raw; the data flow said conditioned.

WHAT THE SYMPTOM IS, AND WHAT IT IS NOT

Not "a broken source passes". `passed` stays True for a merely biased source either way, and that is
CORRECT SP 800-90B behaviour: Repetition Count and Adaptive Proportion catch a source that has failed
outright or gone stuck, not one that is skewed. A test asserting `passed is False` for a biased source
would be asserting the standard is something it is not.

The defect is the ENTROPY ESTIMATE. Measured on this fixture: a p(1) = 0.90 source has raw
min-entropy 0.1472 bit/bit, and the same estimator run after the extractor reports 0.9806 - against
0.9892 for a fair 50/50 source. A badly biased source and a healthy one land a hundredth of a bit
apart, because an extractor produces near-uniform output by construction. The number was never wrong
about what it measured; it measured the Toeplitz stage while every label around it said entropy
source, and it is the figure a reader uses to judge the noise.

WHAT IS ASSERTED

That the reported min-entropy DISCRIMINATES between a biased source and a fair one. That is the
property the conditioned measurement destroys, and it holds regardless of the exact estimator, so it
survives a change to the MCV bound or the confidence level.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsh.qrng.health import health_suite
from qsh.qrng.simulator import generate_quantum_bits, toeplitz_extract


def _streams(p_one: float, n: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = (rng.random(n) < p_one).astype(np.int64)
    return raw, toeplitz_extract(raw, out_len=n // 4, seed=1)


@pytest.mark.parametrize("p_one,ceiling", [(0.90, 0.30), (0.75, 0.60)])
def test_a_biased_source_is_reported_as_biased(p_one, ceiling):
    """The headline entropy figure must reflect the SOURCE, not the extractor."""
    raw, cond = _streams(p_one)
    got = health_suite(cond, raw=raw).min_entropy_per_bit
    assert got < ceiling, (
        f"a p(1)={p_one} source reports {got:.4f} bit/bit, which is not the entropy of that source. "
        "If this rose, the estimate is being taken after conditioning again - an extractor makes any "
        "source look uniform, so the figure would describe the Toeplitz stage under a label that "
        "says entropy source.")


def test_the_estimate_separates_a_biased_source_from_a_fair_one():
    """Anti-tautology: the check above is only meaningful if a fair source scores HIGH here.

    Without this, `min_entropy < 0.30` would also pass against an estimator that returned 0 for
    everything, which would be a different defect wearing the same green tick.
    """
    raw_b, cond_b = _streams(0.90)
    raw_f, cond_f = _streams(0.50)
    h_biased = health_suite(cond_b, raw=raw_b).min_entropy_per_bit
    h_fair = health_suite(cond_f, raw=raw_f).min_entropy_per_bit
    assert h_fair > 0.95, f"a fair source reports {h_fair:.4f}; the estimator is not reading high"
    assert h_fair - h_biased > 0.5, (
        f"fair {h_fair:.4f} vs biased {h_biased:.4f} differ by {h_fair - h_biased:.4f}. The whole "
        "point of measuring pre-conditioning is that these separate; conditioned they sit about "
        "0.01 apart.")


def test_the_conditioned_measurement_would_have_failed_this():
    """The state that shipped, reconstructed rather than described.

    `health_suite(cond)` with no `raw` is exactly the old call. It must NOT satisfy the property
    above - otherwise the fix changed nothing and these tests pass for the wrong reason.
    """
    raw, cond = _streams(0.90)
    old = health_suite(cond).min_entropy_per_bit
    new = health_suite(cond, raw=raw).min_entropy_per_bit
    assert old > 0.90, (
        f"the pre-fix call reports {old:.4f}; it used to report ~0.98 for a p(1)=0.90 source and "
        "that is the defect this file pins. If it no longer does, the extractor or the estimator "
        "changed and this reconstruction needs re-deriving.")
    assert new < 0.30 < old, f"pre-fix {old:.4f} and post-fix {new:.4f} must straddle the threshold"


def test_the_shipped_generator_passes_raw_through():
    """End to end: the real call site, not just the function it calls.

    The behavioural checks above all drive `health_suite` directly, so every one of them would stay
    green if `simulator.py` reverted to `health_suite(bits)`. The shipped simulated source is near
    fair, so no output-value assertion can discriminate here either - a fair source reads ~0.99
    whether the estimate is taken before or after conditioning, which is exactly why the defect
    survived: on the data this project actually generates, the two are indistinguishable.

    So this asserts on the CALL, which is the thing that was wrong. Same technique the register gate
    uses on `__code__.co_names`: a property of the code, checked because the output cannot carry it.
    """
    import ast
    import inspect
    import textwrap

    # PARSED, not grepped. The first version of this check was `"raw=raw" in src`, and it did not
    # fire when the call site was reverted - because the COMMENT explaining why `raw=raw` matters
    # sits directly above the call and contains the string. A gate that matches its own explanatory
    # comment passes for the wrong reason, which is the defect class this whole file is about.
    tree = ast.parse(textwrap.dedent(inspect.getsource(generate_quantum_bits)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "health_suite"]
    assert calls, "the generator no longer calls health_suite; re-derive this test"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "raw" in kw, (
            f"`generate_quantum_bits` calls `health_suite` with keywords {sorted(kw)} and no `raw=`, "
            "so the SP 800-90B tests and the min-entropy estimate are running on CONDITIONED output "
            "again. Every behavioural test in this file would stay green, because they call "
            "`health_suite` directly rather than through the generator.")

    res = generate_quantum_bits(n_bits=32_768, post_processing="toeplitz", seed=3)
    assert res.health.min_entropy_per_bit > 0.0
    assert res.health.min_entropy_bits > 0.0
