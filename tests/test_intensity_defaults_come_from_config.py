"""Every dataclass carrying decoy intensities must take them from config, not spell them.

WHY THIS EXISTS

Round 47 moved the shipped intensities to the finite-key optimum - `mu_signal` 0.5 -> 0.3416,
`mu_decoy` 0.1 -> 0.1417 - and fixed `QKDParams` to read them from `QKDConfig` rather than carry
literals, recording in its own docstring that "every production surface constructs a bare
`QKDParams()`" so a literal there silently outranks the config.

`FiniteKeyParams` is the sibling and it did not follow. It kept `mu_signal: float = 0.5` and
`mu_decoy: float = 0.1` for four rounds, so every bare `FiniteKeyParams()` evaluated Rusca et al.'s
bound at intensities the demo does not run.

WHY NOTHING CAUGHT IT, which is the part worth keeping

The refuted-band test constructs `FiniteKeyParams()` with no arguments and asserts the positive-key
threshold lands inside a 5e6 to 2e8 band. Measured at both pairs: **3.84e7 pulses** at the retired
0.5 / 0.1 and **1.40e7** at the shipped 0.3416 / 0.1417. Both sit inside the band, so that test
passes either way - a gate positioned exactly where the drift happens and unable to see it, because
its tolerance is wider than the effect.

So this asserts the SOURCE of the defaults rather than a downstream number. A band cannot catch a
drift smaller than itself; the identity `default == config` has no tolerance to hide in.

WHAT IS DELIBERATELY NOT ASSERTED

Not that any particular intensity is correct - that is `defaults.py`'s decision and it has moved
once already. Only that one place decides it.
"""

from __future__ import annotations

import dataclasses

from qsh.config import get_config
from qsh.qkd.bb84 import QKDParams
from qsh.qkd.finite_key import FiniteKeyParams

#: Field name on the params dataclass -> field name on `QKDConfig`. Both dataclasses use the same
#: names, but the mapping is written out so a future rename fails loudly here rather than silently
#: comparing an attribute against itself.
INTENSITY_FIELDS = {"mu_signal": "mu_signal", "mu_decoy": "mu_decoy"}


def test_a_bare_construction_matches_the_shipped_config():
    """The property: `X()` and the config agree, for every dataclass that carries intensities."""
    cfg = get_config().qkd
    wrong = []
    for cls in (QKDParams, FiniteKeyParams):
        obj = cls()
        for own, cfg_name in INTENSITY_FIELDS.items():
            if not hasattr(obj, own):
                continue
            got, want = getattr(obj, own), getattr(cfg, cfg_name)
            if got != want:
                wrong.append(f"{cls.__name__}().{own} = {got}, config says {want}")
    assert not wrong, (
        "these intensity defaults disagree with the shipped configuration:\n  "
        + "\n  ".join(wrong)
        + "\n\nA bare construction is how every production surface builds these, so a literal here "
        "outranks the config for every Python caller while the browser reads the config and gets "
        "the other value. That is the split Round 47 fixed in `QKDParams` and left in "
        "`FiniteKeyParams`."
    )


def test_the_defaults_are_wired_to_config_rather_than_copied():
    """A literal that happens to equal the config today would pass the check above and drift tomorrow.

    `default_factory` is the wiring; a plain `default` is a copy. Asserted on the dataclass field so
    the check is about the mechanism, not about today's numbers coinciding.
    """
    for cls in (QKDParams, FiniteKeyParams):
        fields = {f.name: f for f in dataclasses.fields(cls)}
        for own in INTENSITY_FIELDS:
            f = fields.get(own)
            if f is None:
                continue
            assert f.default is dataclasses.MISSING, (
                f"{cls.__name__}.{own} carries a literal default ({f.default!r}). Even if it equals "
                "the config today, it is a second place that decides the value and will drift the "
                "next time the config moves - which is exactly how this happened.")
            assert f.default_factory is not dataclasses.MISSING, (
                f"{cls.__name__}.{own} has neither a default nor a default_factory reading config")


def test_the_band_gate_could_not_have_caught_this():
    """Anti-tautology, against the real gate rather than an invented one.

    The refuted-band test asserts the positive-key threshold falls in 5e6 to 2e8. Measured at both
    intensity pairs, both land inside it, so it is green on the defect and on the fix alike. Recorded
    here as arithmetic so the reason this file exists cannot quietly stop being true.
    """
    lo, hi = 5e6, 2e8
    retired, shipped = 3.84e7, 1.40e7        # measured, 50 km channel
    assert lo < retired < hi, "the retired pair left the band; re-derive"
    assert lo < shipped < hi, "the shipped pair left the band; re-derive"
    assert retired != shipped, "the two pairs must differ or there was nothing to catch"
