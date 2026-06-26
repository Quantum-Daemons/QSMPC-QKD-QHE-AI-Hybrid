"""Telecom / defense use case: QKD+PQC hybrid + QML intrusion detection (scaffold).

A carrier runs a PQC-first hybrid link with a QKD overlay and an AI intrusion-detection
plane that classifies QKD attack signatures (intercept-resend, PNS, Trojan, detector
blinding, ...). The pipeline establishes the secure session, runs BB84 under a
configurable eavesdropper, then the QML-IDS scores the session telemetry.

Maps to: AI-QKD (WS1) + QML-IDS (WS2) on a PQC-first hybrid. The classical IDS path is
the report's Phase-1 baseline (runs fast); set ``config['ids']='quantum'`` for the
PennyLane variational classifier (the WS2 research target).

Run:  ``python -m qsh.orchestration.pipeline --usecase telecom_ids``
"""

from __future__ import annotations

from ...orchestration.crypto_agility import CryptoRegistry
from ...orchestration.events import EventBus
from ...orchestration.pipeline import Pipeline
from ...orchestration.stages import (
    Context,
    PQCHandshakeStage,
    QKDKeyAgreementStage,
    QuantumComparisonStage,
    ResultStage,
    SignatureStage,
    Stage,
)


class QMLIDSStage(Stage):
    """Train + evaluate the QKD intrusion-detection plane on attack telemetry."""

    name = "qml_ids"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        mode = ctx.config.get("ids", "classical")
        if mode == "quantum":
            from ...qml_ids.detector import evaluate_quantum

            rep = evaluate_quantum(n_per_class=ctx.config.get("n_per_class", 100),
                                   seed=ctx.config.get("seed", 0))
        else:
            from ...qml_ids.detector import evaluate_classical

            rep = evaluate_classical(n_per_class=ctx.config.get("n_per_class", 300),
                                     seed=ctx.config.get("seed", 0))
        ctx.artifacts["ids_detector"] = rep.detector
        ctx.artifacts["ids_binary_f1"] = round(rep.binary_f1, 4)
        bus.metric("ids_accuracy", round(rep.accuracy, 4), self.name)
        bus.metric("ids_macro_f1", round(rep.macro_f1, 4), self.name)
        bus.metric("ids_binary_f1", round(rep.binary_f1, 4), self.name)
        bus.metric("ids_params", rep.n_params, self.name)
        bus.log(f"QML-IDS ({rep.detector}): acc={rep.accuracy:.3f} "
                f"macroF1={rep.macro_f1:.3f} attack-F1={rep.binary_f1:.3f} "
                f"(target>=0.93)", self.name)


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    from ...config import get_config

    config = dict(config or {})
    qkd_pulses = config.get("qkd_pulses", get_config().qkd.pulses)   # central QKDConfig (no inline literal)
    ctx = Context(
        usecase="telecom_ids",
        config={"qkd": config.get("qkd", "bb84-decoy"),
                "qkd_pulses": qkd_pulses,
                "ids": config.get("ids", "classical"), **config},
    )
    # model an active eavesdropper on the QKD link to exercise the IDS
    ctx.config.setdefault("eve", True)
    stages = [
        PQCHandshakeStage(),
        QKDKeyAgreementStage(),
        QMLIDSStage(),
        QuantumComparisonStage(),
        ResultStage(),
        SignatureStage(),
    ]
    return Pipeline(name="telecom-ids", stages=stages), ctx


def run_demo(config: dict | None = None) -> dict:
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "ids_detector": ctx.artifacts.get("ids_detector"),
        "attack_detection_f1": ctx.artifacts.get("ids_binary_f1"),
        "qber": m.get("qber"),
        "metrics": m,
    }
