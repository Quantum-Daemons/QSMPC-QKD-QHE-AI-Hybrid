"""A stage that logs "mixed into session" must have mixed something into the session.

WHY THIS EXISTS

`QKDKeyAgreementStage` logged *"QKD key derived (QBER=...); mixed into session"* on every run whose
channel delivered 256 or more secure bits, and mixed nothing. `standard_stages()` runs
`PQCHandshakeStage` FIRST; it reads `ctx.data.get("qkd_key")`, which is `None` at that point. The
`ctx.put("qkd_key", ...)` that follows had no reader anywhere - grepped across `src/`, `web/src/`
and `api/`. So the session key was the PQC hybrid alone, under a log line and a
`security.py` composite summary both claiming a PQC+QKD hybrid.

`SatelliteQKDStage` had the same defect with a quieter symptom: its log said only "derived", but its
docstring said "derive + fold in" and the key was equally unused.

This is the security-property-the-run-does-not-provide class that this project has found three times
before on pages. Finding it in the orchestration itself is the same defect one layer down.

WHY THE ASSERTION IS ON THE KEY AND NOT ON THE LOG

A test that greps for the log line passes as soon as the wording is fixed, which was one of the two
options considered and is not the one taken. What has to hold is that the bytes changed, and that
they changed to exactly what the combiner produces - so a future edit that logs the mix, or reorders
the stages, or swaps the combiner mode, is measured against the same predicate.

Parameterised over BOTH pipelines. The two are separate stage classes in separate modules, and
fixing one while leaving the other is how this codebase has repeatedly ended up with one defect in
two places.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("oqs", reason="liboqs-python is required for the PQC handshake")

from qsh.config import get_config  # noqa: E402
from qsh.orchestration.pipeline import standard_stages  # noqa: E402
from qsh.orchestration.stages import PQCHandshakeStage, QKDKeyAgreementStage  # noqa: E402
from qsh.pqc.transport import _mix_qkd  # noqa: E402
from qsh.usecases.satellite_comms import SatelliteQKDStage  # noqa: E402


class _Bus:
    """Collects what the stage claims, so the claim can be read against what it did."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.metrics: dict[str, object] = {}

    def log(self, msg, stage=None):        # noqa: D102 - test double
        self.logs.append(str(msg))

    def metric(self, name, value, stage=None):  # noqa: D102 - test double
        self.metrics[name] = value


class _Ctx:
    def __init__(self, **config) -> None:
        self.config = {"seed": 0, **config}
        self.data: dict = {}
        self.metrics: dict = {}

    def put(self, k, v):
        self.data[k] = v


def _registry():
    from qsh.orchestration.crypto_agility import default_registry

    return default_registry()


def _handshake(ctx, bus, reg):
    PQCHandshakeStage().run(ctx, bus, reg)
    return ctx.data["session_key"]


#: (stage, config that produces a usable key on that stage's channel).
#: Satellite needs a short link: the free-space budget at the default 800 km yields no secure key,
#: which is the honest physics and also means the fold would not run.
PIPELINES = [
    pytest.param(QKDKeyAgreementStage(), {"qkd_distance_km": 1.0, "qkd_misalignment": 0.0},
                 id="standard"),
    pytest.param(SatelliteQKDStage(), {"sat_distance_km": 1.0, "sat_misalignment": 0.0,
                                       "sat_atmospheric_db": 0.5, "sat_pointing_db": 0.5},
                 id="satellite"),
]


@pytest.mark.parametrize(("stage", "config"), PIPELINES)
def test_the_session_key_actually_changes(stage, config) -> None:
    """The bytes must differ after the QKD stage, and differ to exactly what the combiner gives."""
    ctx, bus, reg = _Ctx(**config), _Bus(), _registry()
    before = _handshake(ctx, bus, reg)
    stage.run(ctx, bus, reg)

    if not ctx.data.get("qkd_key"):
        pytest.skip(f"{stage.name} produced no secure key on this channel; the fold cannot run")

    after = ctx.data["session_key"]
    assert after != before, (
        f"{stage.name} put a qkd_key and left the session key unchanged. That is the defect: the "
        "stage claims a PQC+QKD hybrid while the session rests on the PQC half alone."
    )
    expected = _mix_qkd(before, ctx.data["qkd_key"],
                        mode=get_config().crypto.hybrid_combiner_mode)
    assert after == expected, (
        "the session key changed but is not what `_mix_qkd` produces from the pre-stage key and the "
        "QKD key - something else rewrote it, or a different combiner ran than the config selects"
    )


@pytest.mark.parametrize(("stage", "config"), PIPELINES)
def test_the_channels_carry_the_mixed_key_too(stage, config) -> None:
    """`ctx.data["session_key"]` is a copy; a fold that misses the channels desynchronises them.

    The PSI reads `ctx.data["session_key"]` and the transport seals with `channel.session_key`. If
    only one is folded, two things that must be the same key are not, and nothing would notice until
    a seal failed to open.
    """
    ctx, bus, reg = _Ctx(**config), _Bus(), _registry()
    _handshake(ctx, bus, reg)
    stage.run(ctx, bus, reg)
    if not ctx.data.get("qkd_key"):
        pytest.skip(f"{stage.name} produced no secure key on this channel")

    key = ctx.data["session_key"]
    for role in ("channel_initiator", "channel_responder"):
        assert ctx.data[role].session_key == key, f"{role} did not receive the folded key"
    # And the two ends still agree, which is what the fold must not break.
    a, b = ctx.data["channel_initiator"], ctx.data["channel_responder"]
    assert b.open(a.seal(b"round-trip")) == b"round-trip"


@pytest.mark.parametrize(("stage", "config"), PIPELINES)
def test_the_claim_follows_the_mix_not_the_derivation(stage, config) -> None:
    """`qkd_mixed` must report the fold, not that a key was derived.

    The original defect was exactly this substitution: the log fired on `secure_key_bits >= 256`,
    which is a precondition for mixing rather than evidence of it.
    """
    ctx, bus, reg = _Ctx(**config), _Bus(), _registry()
    _handshake(ctx, bus, reg)
    stage.run(ctx, bus, reg)
    assert "qkd_mixed" in bus.metrics, f"{stage.name} does not report whether the mix happened"
    mixed = bus.metrics["qkd_mixed"] == 1
    assert mixed == bool(ctx.data.get("qkd_key")), (
        "qkd_mixed disagrees with whether a key was available to fold"
    )
    if mixed:
        assert any("folded into the session" in m for m in bus.logs), (
            "the run reports a mix but no log line says so"
        )


def test_no_pqc_channel_means_no_claim() -> None:
    """With no handshake, the stage must derive and say it could not fold - not claim a hybrid."""
    ctx, bus, reg = _Ctx(qkd_distance_km=1.0, qkd_misalignment=0.0), _Bus(), _registry()
    QKDKeyAgreementStage().run(ctx, bus, reg)          # no PQCHandshakeStage first
    if not ctx.data.get("qkd_key"):
        pytest.skip("no secure key on this channel")
    assert bus.metrics.get("qkd_mixed") == 0
    assert any("no PQC channel" in m for m in bus.logs)


def test_the_handshake_still_precedes_the_qkd_stage() -> None:
    """The fix is a re-derivation, so the documented stage order must be the one it assumes.

    If someone later reorders these two, the re-derivation becomes dead weight (the handshake would
    already have the key) and this test says so rather than leaving both mechanisms half-applied.
    """
    names = [s.name for s in standard_stages()]
    assert names.index("pqc_handshake") < names.index("qkd_key_agreement"), (
        "the stages were reordered; the QKD stage now re-derives a key the handshake already mixed. "
        "Pick one mechanism - and if the order is the one kept, the browser's FLOWS must move too."
    )
