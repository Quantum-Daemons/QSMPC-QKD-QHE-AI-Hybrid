"""The full deployment's federated training says what its accuracy is set against (Round 86).

WHY THIS EXISTS

The medical use case's federated CNN (``config={"dataset": "medmnist", "model": "cnn"}``) ended at 0.7282 on
seeds 0, 1 and 2: the evaluation rows' majority class, so it answered one class for every image. Its log
printed "global model accuracy = 0.728 (no raw patient data shared)", which reads as a result. Plain SGD at
0.05 with an unweighted loss on a label-skewed split, two of whose four hospitals hold one class each, never
left that answer; each hospital's own inverse-frequency class weights with Adam at 0.01 do, weakly.

WHAT IS ASSERTED

1. ``federated_train`` records the evaluation rows' majority rate, the balanced accuracy and the AUC, and,
   with ``local_baseline=True``, each client trained alone from the same initial weights for the same epochs.
2. The class weights are each client's own, 0 for a class it does not hold.
3. The medical stage's log sets the accuracy against the majority, the balanced accuracy and the AUC, says
   when the model answers one class, and prints every hospital trained alone; the default tabular run too.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qsh.ai.federated import FederatedResult, _class_weights, federated_train  # noqa: E402


def _skewed_parts(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.6).astype(np.int64)      # about 30 percent class 1
    order = np.argsort(y, kind="stable")
    parts = [(X[order[i::4]], y[order[i::4]]) for i in range(3)]
    ones = order[y[order] == 1][:20]
    parts.append((X[ones], y[ones]))                          # a client holding class 1 alone
    return parts, (X[300:], y[300:])


def test_the_result_carries_the_majority_the_balanced_accuracy_the_auc_and_each_client_alone() -> None:
    parts, ev = _skewed_parts()
    res = federated_train(parts, kind="mlp", rounds=3, eval_data=ev, seed=0, optimizer="adam", lr=0.01,
                          class_weighted=True, local_baseline=True)
    y = ev[1]
    assert res.majority_baseline == pytest.approx(max(np.mean(y == 0), np.mean(y == 1)))
    assert 0.0 <= res.balanced_acc <= 1.0 and res.auc is not None and 0.0 <= res.auc <= 1.0
    assert [s["client"] for s in res.local_only] == [0, 1, 2, 3]
    assert [s["rows"] for s in res.local_only] == [len(p[1]) for p in parts]
    assert res.local_only[3]["labels"] == [0, 20]
    for s in res.local_only:
        assert set(s) == {"client", "rows", "labels", "accuracy", "balanced_accuracy", "auc"}


def test_without_a_local_baseline_none_is_trained_and_the_defaults_are_unchanged() -> None:
    parts, ev = _skewed_parts()
    a = federated_train(parts, kind="mlp", rounds=2, eval_data=ev, seed=0)
    b = federated_train(parts, kind="mlp", rounds=2, eval_data=ev, seed=0, optimizer="sgd", class_weighted=False)
    # One evaluation row of 100 at most: on a GPU the same seed need not give bit-identical weights.
    assert a.local_only == [] and a.global_acc == pytest.approx(b.global_acc, abs=0.011)
    assert a.round_acc == pytest.approx(b.round_acc, abs=0.011)


def test_class_weights_are_the_clients_own_and_zero_for_an_absent_class() -> None:
    w = _class_weights(np.array([1, 1, 1, 0]), 2, torch.device("cpu")).tolist()
    assert w == pytest.approx([2.0, 4 / 6])
    assert _class_weights(np.array([1, 1]), 2, torch.device("cpu")).tolist() == [0.0, 0.5]


def test_an_unknown_optimizer_is_refused() -> None:
    parts, ev = _skewed_parts()
    with pytest.raises(ValueError, match="optimizer must be"):
        federated_train(parts, kind="mlp", rounds=1, eval_data=ev, optimizer="rmsprop")


def _stage_logs(monkeypatch, config: dict, fake: FederatedResult | None = None):
    from qsh.orchestration.crypto_agility import CryptoRegistry
    from qsh.orchestration.events import EventBus
    from qsh.usecases import medical_fl

    _, ctx = medical_fl.build(config)
    if fake is not None:
        import qsh.ai.federated as fed

        monkeypatch.setattr(fed, "federated_train", lambda *a, **k: fake)
    bus = EventBus()
    medical_fl.FederatedTrainingStage().run(ctx, bus, CryptoRegistry())
    return ctx, bus.metrics(), [e.message for e in bus.history if e.type.value == "log"]


def test_the_tabular_run_sets_its_accuracy_against_the_majority(monkeypatch) -> None:
    ctx, metrics, logs = _stage_logs(monkeypatch, {})
    line = next(m for m in logs if m.startswith("global model accuracy = "))
    assert "against the evaluation rows' majority class" in line and "balanced accuracy" in line and "AUC" in line
    assert "answers one class" not in line
    assert metrics["federated_majority_baseline"] == ctx.artifacts["federated_majority_baseline"]
    assert metrics["federated_balanced_accuracy"] > metrics["federated_majority_baseline"] - 1


def test_a_one_class_model_is_called_one(monkeypatch) -> None:
    fake = FederatedResult(0.7282, [0.7282], 4, True, majority_baseline=0.7282, balanced_acc=0.5, auc=0.56,
                           local_only=[{"client": 0, "rows": 102, "labels": [0, 102], "accuracy": 0.7282,
                                        "balanced_accuracy": 0.5, "auc": 0.5}])
    _, metrics, logs = _stage_logs(monkeypatch, {}, fake)
    line = next(m for m in logs if m.startswith("global model accuracy = "))
    assert line.endswith("; the model answers one class for every evaluation row"), line
    alone = next(m for m in logs if m.startswith("each hospital trained alone"))
    assert "hospital 0 (102 images, labels [0, 102]) balanced accuracy 0.500, AUC 0.500" in alone
    assert metrics["local_only_auc_mean"] == 0.5
