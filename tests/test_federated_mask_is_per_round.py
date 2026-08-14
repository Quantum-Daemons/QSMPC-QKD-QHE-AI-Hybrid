"""A pairwise mask must be fresh per round, and this asserts it on the bytes.

WHY THIS EXISTS

`SecureAggregator.mask_update` took `(client, update)` and derived its mask from the pair key alone.
The FedAvg loop in `federated_train` calls it once per round with the same pair keys, so **every
round of a run reused the same mask**. The browser twin does not: `qflPairMask` in
`web/src/lib/clientQFL.ts` puts the round index in the HKDF info string and its docstring states the
reason - *"Deriving per round is what stops a mask from being reused across updates of the same
weights."* Two implementations of one scheme, one of them holding a property the other documents as
required.

WHAT REUSE ACTUALLY COSTS, stated precisely so the fix is not oversold

The masks still cancel, so the aggregate was never wrong, and the demo runs all clients in one tab
so no party is separated from another. What reuse destroys is the property that makes masking worth
doing at all: with a constant mask `m`, an observer of two rounds computes
`(u_1 + m) - (u_2 + m) = u_1 - u_2`, the individual client's update DELTA, in the clear. That is the
per-client information the scheme exists to hide, and it is recoverable by subtraction alone -
no key, no cryptanalysis.

WHY THE ASSERTION IS ON THE MASK AND NOT ON THE SIGNATURE

A test that checks `mask_update` accepts a `round=` argument passes the moment the parameter exists,
whether or not it reaches the derivation. `_length_tag` was added to the Python combiner alone one
commit after a message asserting all four sites change together, and the browser kept deriving a
different key for two rounds. So the assertion here is that the masked output CHANGES with the round
and that the cancellation still holds exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.ai.federated import SecureAggregator  # noqa: E402


def _updates(k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(6,)) for _ in range(k)]


def test_the_mask_changes_between_rounds() -> None:
    """The whole finding: same client, same update, two rounds, two different masked outputs."""
    agg = SecureAggregator(4)
    u = _updates(4)[0]
    r0 = agg.mask_update(0, u, round_index=0)
    r1 = agg.mask_update(0, u, round_index=1)
    assert not np.allclose(r0, r1), (
        "the masked share is identical across rounds, so the mask is being reused. An observer of "
        "two rounds recovers this client's update delta by subtraction: (u1+m)-(u2+m) = u1-u2."
    )


def test_the_recoverable_delta_is_actually_gone() -> None:
    """Assert the consequence, not just that some bytes moved.

    Subtracting two rounds of one client's masked shares must NOT yield the plaintext delta. This is
    the property the reuse destroyed, so it is the property the test states - a "bytes differ" check
    would also pass if the round index perturbed something irrelevant.
    """
    agg = SecureAggregator(4)
    u1, u2 = _updates(4, seed=1)[0], _updates(4, seed=2)[0]
    leaked = agg.mask_update(0, u1, round_index=0) - agg.mask_update(0, u2, round_index=1)
    assert not np.allclose(leaked, u1 - u2), (
        "differencing two rounds of one client's masked shares recovered the plaintext update "
        "delta - the exact per-client information pairwise masking exists to hide"
    )


@pytest.mark.parametrize("rounds", [0, 1, 2, 7])
def test_the_masks_still_cancel_exactly(rounds: int) -> None:
    """Freshness must not cost correctness: within one round the masks must still sum to zero.

    Antisymmetry is what makes the aggregate recoverable. A round index folded in on only one side
    of the pair, or folded in per-client rather than per-pair, would break cancellation - and it
    would break it into a plausible-looking wrong number rather than an error.
    """
    ups = _updates(5, seed=3)
    agg = SecureAggregator(5)
    masked = [agg.mask_update(c, u, round_index=rounds) for c, u in enumerate(ups)]
    np.testing.assert_allclose(agg.aggregate(masked), np.sum(ups, axis=0), atol=1e-9)


def test_every_round_index_gives_a_distinct_mask() -> None:
    """Not just round 0 vs 1 - a derivation that folded the index in weakly could collide."""
    agg = SecureAggregator(3)
    u = _updates(3, seed=4)[0]
    seen = {agg.mask_update(0, u, round_index=r).tobytes() for r in range(12)}
    assert len(seen) == 12, f"round indices collided: {12 - len(seen)} of 12 masks repeated"


def test_the_browser_twin_still_documents_the_same_requirement() -> None:
    """If the TS side ever drops the round from its info string, this fix is half-applied again.

    The Python and TypeScript aggregators are the same scheme in two languages, and this repository
    has twice shipped a change to one of a pair of twins. Reading the browser's derivation here is
    cheap and fails loudly if it drifts.
    """
    src = (REPO / "web" / "src" / "lib" / "clientQFL.ts").read_text()
    assert "qsh/qfl-pairwise-mask/v1|${u}|${v}|r${round}" in src, (
        "clientQFL.ts no longer derives its pairwise mask per round; the two implementations of "
        "this scheme have diverged again"
    )
