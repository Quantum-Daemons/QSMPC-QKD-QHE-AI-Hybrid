"""A published QCNN must be REACHABLE from the stages its use case actually runs.

`tests/test_qcnn_usecases.py` checks the data: the declared set matches the files on disk, the
weights load, and the shipped copy agrees with the trained artifact. All three passed while the
telecom QCNN option was inert, because none of them asks the question that matters to a visitor -
does selecting it change anything?

The gap is a half-stated invariant. The comment above `QCNN_USECASES` says a backend option is shown
"when the artefact implementing it exists", and that is one of two conditions. The other is that a
stage the use case RUNS reads the artefact. Telecom satisfied the first and not the second:
`FLOWS.telecom_ids` contains `qml_ids` and not `private_inference`, and the QCNN branch lived in the
`private_inference` case. So `ParameterPanel` offered a QCNN option that no code path could honour,
which is the same shape as the genomics failure the comment itself recounts - option offered, silent
fallback to CKKS, two panels announcing that a quantum circuit had decided the class.

Nothing local caught it. tsc was clean (the branch simply was not there to fail), every Python gate
was green, and the weights were served with HTTP 200 and the right content-type. It took running the
deployed page and noticing that `qcnn_weights.json` was never fetched.

This test asserts the second condition, in the same source-derived style as the rest of the suite:
no hand-kept list, both facts parsed from the modules that define them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_WEB = Path(__file__).resolve().parents[1] / "web" / "src" / "lib"
_COMPUTE = _WEB / "clientCompute.ts"
_PIPELINE = _WEB / "usePipelineRun.ts"

#: The call every QCNN branch must make. Fetching the weights is the irreducible step - a branch
#: that does not load them cannot be running a trained circuit whatever else it prints.
_LOADER = "loadQcnnWeights("


def _flows() -> dict[str, list[str]]:
    """`FLOWS` from clientCompute.ts, as a dict."""
    src = _COMPUTE.read_text(encoding="utf-8")
    m = re.search(r"const FLOWS: Record<string, string\[\]> = \{(.*?)\n\};", src, re.S)
    assert m, "FLOWS table not found - clientCompute.ts changed shape and this parser is now blind"
    out: dict[str, list[str]] = {}
    for key, body in re.findall(r"(\w+):\s*\[([^\]]*)\]", m.group(1)):
        out[key] = [s.strip().strip('"') for s in body.split(",") if s.strip()]
    assert len(out) >= 5, f"parsed only {len(out)} flows; the table has more than that"
    return out


def _qcnn_usecases() -> set[str]:
    """`QCNN_USECASES` from usePipelineRun.ts."""
    src = _PIPELINE.read_text(encoding="utf-8")
    m = re.search(r"QCNN_USECASES = new Set\(\[([^\]]*)\]\)", src)
    assert m, "QCNN_USECASES not found - usePipelineRun.ts changed shape"
    got = {s.strip().strip('"') for s in m.group(1).split(",") if s.strip()}
    assert got, "QCNN_USECASES parsed empty, which would make every assertion below vacuous"
    return got


def _stage_blocks() -> dict[str, str]:
    """Each `case "<stage>": { ... }` body from the runStage switch, keyed by stage name.

    Split on the case labels rather than by matching braces: the bodies contain template literals,
    regexes and brace characters inside strings, and a brace counter gets those wrong in exactly the
    quiet way this test exists to prevent.
    """
    src = _COMPUTE.read_text(encoding="utf-8")
    starts = [(m.group(1), m.start()) for m in re.finditer(r'^      case "([a-z_]+)":', src, re.M)]
    assert len(starts) >= 6, f"found {len(starts)} case labels; the switch has more than that"
    bounds = [p for _, p in starts] + [len(src)]
    return {name: src[bounds[i]:bounds[i + 1]] for i, (name, _) in enumerate(starts)}


def _stages_that_run_a_qcnn() -> set[str]:
    return {name for name, body in _stage_blocks().items() if _LOADER in body}


@pytest.mark.parametrize("usecase", sorted(_qcnn_usecases()))
def test_a_published_qcnn_is_reachable_from_a_stage_the_use_case_runs(usecase: str) -> None:
    """Shipping weights and registering the use case is not enough; a stage must read them."""
    flows, qcnn_stages = _flows(), _stages_that_run_a_qcnn()
    assert usecase in flows, f"{usecase} declares a QCNN but has no flow"
    reachable = sorted(set(flows[usecase]) & qcnn_stages)
    assert reachable, (
        f"{usecase} is in QCNN_USECASES, so ParameterPanel offers a QCNN backend option, but none of "
        f"the stages it runs ({', '.join(flows[usecase])}) calls {_LOADER!r}. Stages that do: "
        f"{', '.join(sorted(qcnn_stages)) or '(none)'}. The option would be inert: selecting it "
        f"changes nothing and the page falls back to CKKS while claiming a circuit decided the class."
    )


def test_the_check_can_fail() -> None:
    """Anti-tautology: a use case with no QCNN stage must come out unreachable.

    Without this, the test above passing tells you only that the parsers returned something.
    `satellite_comms` runs neither `private_inference` nor `qml_ids`, and ships no weights, so it is
    the honest negative control. If it ever gains a QCNN the assertion below should be moved, not
    deleted - the suite needs some use case that demonstrates the check discriminates.
    """
    flows, qcnn_stages = _flows(), _stages_that_run_a_qcnn()
    assert qcnn_stages, "no stage calls the loader at all; every reachability assertion is vacuous"
    negative = "satellite_comms"
    assert negative in flows and negative not in _qcnn_usecases()
    assert not (set(flows[negative]) & qcnn_stages), (
        f"{negative} now reaches a QCNN stage, so it no longer demonstrates that this check can "
        f"return false; pick another use case that ships no weights")


def test_telecom_specifically_runs_qml_ids_and_not_private_inference() -> None:
    """The exact configuration that produced the inert control, pinned so it cannot come back.

    Telecom is the only QCNN use case whose flow omits `private_inference`, which is why it was the
    one that broke. Pinning it means a future edit that moves the telecom QCNN branch back into
    `private_inference` fails here rather than on the deployed site.
    """
    flow = _flows()["telecom_ids"]
    assert "qml_ids" in flow and "private_inference" not in flow, (
        f"telecom's flow changed to {flow}; the reachability requirement above still holds but the "
        f"reasoning in this file's docstring needs rewriting")
    assert _LOADER in _stage_blocks()["qml_ids"]


def test_every_use_case_with_weights_on_disk_is_declared() -> None:
    """The reverse direction, cheap to state here: weights on disk with no registration are dead.

    `test_qcnn_usecases.py` owns the forward direction. This catches the mirror-image mistake -
    training and committing a circuit, then never adding the use case, so the artefact ships in the
    bundle and nothing can ever select it.
    """
    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    on_disk = {p.parent.name for p in root.glob("*/qcnn_weights.json")}
    assert on_disk, f"no qcnn_weights.json under {root}; this test is not looking where they live"
    declared = {u.split("_")[0] if u != "medical_fl" else "medical" for u in _qcnn_usecases()}
    orphans = sorted(on_disk - declared)
    assert not orphans, (
        f"qcnn_weights.json ships for {orphans} but no use case declares it, so the bundle carries "
        f"weights that no control can select")


#: Every place the demo decides whether a visitor's backend choice survives. All three must agree,
#: and each was individually invisible: the branch was missing, then the control was not rendered,
#: then the key was stripped on the way to the run.
_PANEL = Path(__file__).resolve().parents[1] / "web" / "src" / "components" / "ParameterPanel.tsx"


def _stages_reading_the_backend_key() -> set[str]:
    """Stage case blocks that actually read `config.inference`."""
    return {name for name, body in _stage_blocks().items() if "config.inference" in body}


def test_the_declared_backend_stages_are_the_stages_that_read_the_key() -> None:
    """`INFERENCE_BACKEND_STAGES` is a list, so it can go stale; derive the truth and compare.

    A stage that starts reading `config.inference` without being added here renders no control, and
    one that stops reading it leaves a control that changes nothing. Both are the same defect.
    """
    src = _PIPELINE.read_text(encoding="utf-8")
    m = re.search(r"INFERENCE_BACKEND_STAGES = \[([^\]]*)\]", src)
    assert m, "INFERENCE_BACKEND_STAGES not found in usePipelineRun.ts"
    declared = {s.strip().strip('"') for s in m.group(1).split(",") if s.strip()}
    actual = _stages_reading_the_backend_key()
    assert declared == actual, (
        f"declared backend stages {sorted(declared)} but the stages reading config.inference are "
        f"{sorted(actual)}. Missing from the list means the Inference control is hidden on any use "
        f"case whose only such stage is the missing one; extra means it is shown where nothing reads it")


@pytest.mark.parametrize("usecase", sorted(_qcnn_usecases()))
def test_the_backend_control_is_rendered_for_every_qcnn_use_case(usecase: str) -> None:
    """The third condition: a control must exist to make the selection.

    Reachability alone is not enough. Telecom's QCNN branch was live and unreachable anyway, because
    `ParameterPanel` gated the whole Inference block on `runsInference` - which asks whether the stage
    NAMED `private_inference` runs, and telecom's backend stage is `qml_ids`. The page rendered six
    selects and none of them was a backend selector.
    """
    flows = _flows()
    assert set(flows[usecase]) & _stages_reading_the_backend_key(), (
        f"{usecase} ships a QCNN but no stage it runs reads config.inference, so no control can "
        f"select it")


def test_the_panel_and_the_config_gate_on_the_same_predicate() -> None:
    """Shown and sent must agree, and both must be the READS predicate, not the stage-name one.

    `buildConfig` stripped `inference` for telecom under `runsInference`, so even with the control
    on screen the choice would not have reached the run. Three gates in series; this pins the two
    that are one line each and easy to revert by accident.
    """
    panel, pipeline = _PANEL.read_text(encoding="utf-8"), _PIPELINE.read_text(encoding="utf-8")
    assert "readsInferenceBackend(usecase)" in panel, (
        "ParameterPanel no longer derives the Inference control from readsInferenceBackend; if it "
        "went back to runsInference the control disappears from every qml_ids use case")
    assert re.search(r"if \(readsInferenceBackend\(usecase\) && inference\) c\.inference", pipeline), (
        "buildConfig no longer ships `inference` under readsInferenceBackend, so a selection made on "
        "screen would be dropped before the run sees it")
    assert "runsInference" not in panel, (
        "ParameterPanel still references runsInference; the two predicates differ by exactly the case "
        "this test exists for, so having both in one file invites regating on the wrong one")


def _shipped_weights() -> list[tuple[str, dict]]:
    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    out = [(p.parent.name, json.loads(p.read_text(encoding="utf-8")))
           for p in sorted(root.glob("*/qcnn_weights.json"))]
    assert out, f"no qcnn_weights.json under {root}; these tests are not looking where they live"
    return out


@pytest.mark.parametrize("regime_field", ["test", "classical_lr"])
def test_metric_names_follow_the_regime_they_were_measured_in(regime_field: str) -> None:
    """Exactly one of the minority/macro pair is populated, and it is the applicable one.

    `f1_score(average="binary")` on two classes is a minority-class F1; on K > 2 the same call site
    holds a MACRO F1, for which "minority class" names nothing. The 8-class telecom artifact shipped
    a macro F1 of 0.809846 under `test_minority_f1`, carried `macro_f1_floors` for a metric the file
    never named, and reported `all_positive_f1: 0.0` - a floor of zero for a baseline that cannot be
    constructed with more than two classes. The value was right and the label was wrong, which is
    the failure mode no range check catches.
    """
    minority_key = f"{regime_field}_minority_f1" if regime_field == "test" else "classical_lr_minority_f1"
    macro_key = f"{regime_field}_macro_f1" if regime_field == "test" else "classical_lr_macro_f1"
    for name, w in _shipped_weights():
        multiclass = int((w.get("readout") or {}).get("readout_qubits") or 0) > 0
        minority, macro = w.get(minority_key), w.get(macro_key)
        if multiclass:
            assert minority is None, f"{name}: {minority_key}={minority} on a multiclass artifact"
            assert isinstance(macro, float), f"{name}: {macro_key} missing on a multiclass artifact"
            assert w.get("all_positive_f1") is None, (
                f"{name}: all_positive_f1={w.get('all_positive_f1')} on {w['readout']['n_classes']} "
                f"classes, where an all-positive predictor does not exist")
        else:
            assert isinstance(minority, float), f"{name}: {minority_key} missing on a binary artifact"
            assert macro is None, f"{name}: {macro_key}={macro} on a binary artifact"


def test_the_headline_metric_names_a_field_that_is_populated() -> None:
    """`headline_metric` drives what the browser and the server print; it must resolve to a number.

    The two-way version asked only whether the BINARY problem was imbalanced, so a balanced 8-class
    run fell through to "accuracy" while the trainer's own console printed "MACRO F1 leads". The
    artifact shipped floors for macro F1 and then told every consumer to headline something else.
    """
    fields = {"macro_f1": ("test_macro_f1", "classical_lr_macro_f1"),
              "minority_class_f1": ("test_minority_f1", "classical_lr_minority_f1"),
              "accuracy": ("test_accuracy", "classical_lr_accuracy")}
    for name, w in _shipped_weights():
        head = w.get("headline_metric")
        assert head in fields, f"{name}: headline_metric={head!r} is not one of {sorted(fields)}"
        for key in fields[head]:
            assert isinstance(w.get(key), float), (
                f"{name}: headline_metric is {head!r} but {key} is {w.get(key)!r}, so the consumer "
                f"would print a metric it does not have")
        multiclass = int((w.get("readout") or {}).get("readout_qubits") or 0) > 0
        if multiclass:
            assert head == "macro_f1", (
                f"{name}: a multiclass artifact must headline macro F1 - accuracy on K classes hides "
                f"a model that has stopped predicting some of them, which is what the shipped floors "
                f"exist to scale")


def test_every_artifact_records_the_command_that_made_it() -> None:
    """A published artifact must carry its own invocation, because recovering one cost an epoch sweep.

    `layers: 6` was in the telecom file, so the architecture was recoverable, but epochs and seed were
    not - and the defaults do not reproduce it. `--usecase telecom` gave macro F1 0.4998 against a
    published 0.8098, `--layers 6` alone gave 0.6443, and only `--layers 6 --epochs 400` returned
    0.809846. The demo was displaying six digits from a run nobody could repeat.
    """
    for name, w in _shipped_weights():
        t = w.get("training")
        assert isinstance(t, dict), f"{name}: no `training` block, so the artifact cannot be regenerated"
        for key in ("layers", "epochs", "seed", "command"):
            assert t.get(key) is not None, f"{name}: training.{key} missing"
        assert t["command"].startswith("python scripts/train/qcnn_variational.py"), (
            f"{name}: training.command is {t['command']!r}, which is not a runnable invocation")
        for key in ("layers", "epochs", "seed"):
            assert f"--{key} {t[key]}" in t["command"], (
                f"{name}: training.command disagrees with training.{key}={t[key]}, so following the "
                f"command would not reproduce the file it is attached to")
        assert w.get("layers") == t["layers"], (
            f"{name}: layers={w.get('layers')} but training.layers={t['layers']}")


def test_the_server_and_browser_resolve_the_headline_the_same_way() -> None:
    """Two implementations of one mapping, in different languages, with no shared source.

    `clientCompute.ts` and `stages.py` each choose (metric, classical, baseline, label) from
    `headline_metric`. Nothing makes them agree, and they are read by different audiences - the demo
    page and the API - so a divergence shows up as two documents quoting different numbers for the
    same model. The three shipped artifacts happen to exercise all three branches between them
    (finance minority-class F1, medical accuracy, telecom macro F1), which is why a drift here would
    be visible on one page and not the others.
    """
    server = (Path(__file__).resolve().parents[1] / "src" / "qsh" / "orchestration" / "stages.py"
              ).read_text(encoding="utf-8")
    browser = _COMPUTE.read_text(encoding="utf-8")
    for field in ("test_macro_f1", "classical_lr_macro_f1", "test_minority_f1",
                  "classical_lr_minority_f1", "test_accuracy", "classical_lr_accuracy",
                  "majority_baseline", "all_positive_f1", "uniform_random"):
        assert field in server, f"stages.py no longer reads {field}; the two paths have diverged"
        assert field in browser, f"clientCompute.ts no longer reads {field}"
    # The multiclass baseline is the uniform-random floor in BOTH, not the constant-predictor one:
    # it is the higher of the two the trainer ships, so picking the other would flatter the model.
    assert 'get("uniform_random")' in server or "['uniform_random']" in server
    assert "uniform_random" in browser
    assert "macro_f1" in server and "macro_f1" in browser, (
        "one of the two paths has lost the multiclass branch, so it would fall through to a metric "
        "the artifact leaves null and print nothing or zero")


def test_the_multiclass_readout_avoids_the_nondeterministic_kernel() -> None:
    """The multiclass marginal must not use `index_add_`, whose CUDA path is nondeterministic.

    This is what made the 8-class artifact irreproducible: `index_add_` accumulates with atomicAdd,
    so the same seeded command landed on macro F1 0.809846 or 0.807781 depending on the run, and only
    the multiclass path touched it - which is why the two binary artifacts always reproduced. Two
    wrong causes were investigated first (thread count, PYTHONHASHSEED), both resting on the
    unchecked assumption that training ran on the CPU. It runs on CUDA when one is present.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train" / "qcnn_variational.py"
           ).read_text(encoding="utf-8")
    # The CALL form, not the name: this file and the trainer both discuss `index_add_` in prose, and
    # a substring check on the bare name flagged its own explanation of why it was removed.
    assert ".index_add_(" not in src, (
        "index_add_ is back in the trainer. On CUDA it is documented as nondeterministic, and the "
        "readout it was used for is a plain column sum: p.view(batch, -1, k).sum(1)")
    # ROUND 47 REPLACED THE EXPRESSION THIS USED TO PIN. It asserted the literal string
    # `p.view(p.shape[0], -1, k).sum(1)`, which was the deterministic marginal at the time. But that
    # marginal binned by `i & (k - 1)` - wires 0..m-1 - and Round 47 found that set includes wires
    # pooling has DISCARDED, so the readout now gathers by the surviving wires first. Pinning one
    # spelling made a correct change look like a regression while saying nothing about the property
    # the check exists to protect.
    #
    # The property is that the reduction is deterministic on CUDA. `argsort(stable=True)` and a
    # fixed-shape `.sum()` are; `index_add_` and `scatter_add_` are not. The prohibition above is
    # the real assertion, and this one requires that SOME deterministic reduction is still present
    # rather than one particular way of writing it.
    assert any(form in src for form in (".sum(1)", ".sum(2)", ".sum(dim=")), (
        "no reduction of the state vector into class bins is visible in the trainer. If the readout "
        "was rewritten again, re-verify that repeated runs of the same seeded command give "
        "identical metrics before shipping an artifact from it.")
    assert ".scatter_add_(" not in src, (
        "scatter_add_ accumulates with the same nondeterministic CUDA path as index_add_")


def test_the_shipped_weights_name_their_readout_convention() -> None:
    """A multiclass readout that does not record its bit order cannot be reproduced from the file.

    Reading the bits in another order permutes the class map while leaving accuracy, macro F1 and
    every loss curve numerically identical, so the convention has to travel WITH the weights rather
    than living only in the trainer.
    """
    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    multiclass = []
    for p in sorted(root.glob("*/qcnn_weights.json")):
        w = json.loads(p.read_text(encoding="utf-8"))
        readout = w.get("readout") or {}
        if int(readout.get("readout_qubits") or 0) > 0:
            multiclass.append(p)
            assert readout.get("bit_order"), f"{p} measures multiple qubits but names no bit order"
            assert readout.get("class_index"), f"{p} names no class-index rule"
            n = int(readout.get("n_classes") or 0)
            assert len(w.get("class_names") or []) == n, (
                f"{p} declares {n} classes but lists {len(w.get('class_names') or [])} names")
    assert multiclass, "no multiclass QCNN found; this test is asserting nothing"


# --------------------------------------------------------------------------- #
#  Published-model pinning. Different subject, same rule: a thing the page offers
#  must resolve to something that exists.
# --------------------------------------------------------------------------- #
def _frontend_models() -> list[dict]:
    cfg = json.loads((Path(__file__).resolve().parents[1] / "web" / "src" / "lib"
                      / "config.defaults.json").read_text(encoding="utf-8"))
    out: list[dict] = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "hf_repo_id" in o:
                out.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    assert out, "no published-model entries found; config.defaults.json changed shape"
    return out


def test_every_published_model_is_pinned_to_a_revision() -> None:
    """A published model with no pinned revision is a panel the browser refuses to fill.

    `browserModels()` builds `.../resolve/<hf_revision>/<path>` and deliberately will not fall back
    to `main` - pinning is what stops a visitor silently getting a different model from the one whose
    metrics the page is printing. But nothing required the pin to EXIST, and three models shipped
    without one: finance_aml.gbm, finance_aml.student and telecom_ids.qml. The demo behaved honestly
    (it says "the browser will not fetch an unpinned model") and three published models were
    unreachable anyway.

    Two of the three were `hf_revision_invalidated_by_retrain` - correct behaviour, since re-pinning
    a stale artifact would serve a model whose numbers do not match the page. The fix was to publish,
    not to re-pin. The third had simply never been recorded.

    Offline on purpose: whether the sha RESOLVES is a network fact and lives in the checklist (RT8).
    Whether it exists at all is a repository fact and belongs here.
    """
    unpinned = [m["key"] for m in _frontend_models()
                if m.get("hf_repo_id") and not m.get("hf_revision")]
    assert not unpinned, (
        f"published models with no pinned revision: {unpinned}. The browser will decline to fetch "
        f"them, so each is a Published-model panel that renders an explanation instead of a model. "
        f"Publish (do not hand-pin a stale sha) and let publish_hf.py record the revision.")


def test_a_pin_and_an_invalidation_never_name_the_same_revision() -> None:
    """The record must not assert that its own pin was invalidated.

    `record_metrics` moves `hf_revision` into `hf_revision_invalidated_by_retrain` when a retrain
    lands without a revision, and nothing cleared it on the way back. After a re-publish,
    finance_aml.student held the SAME sha in both fields - because re-uploading identical bytes
    returns the existing commit rather than making a new one. The registry prefers `hf_revision`, so
    this was invisible at runtime and wrong only in the record.
    """
    metrics_path = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    clashes = [k for k, e in metrics.items()
               if isinstance(e, dict) and e.get("hf_revision")
               and e.get("hf_revision") == e.get("hf_revision_invalidated_by_retrain")]
    assert not clashes, (
        f"{clashes} name one revision as both the pin and the retrain-invalidated one. "
        f"publish_hf.py clears the marker it resolves; if this fires, that clearing was lost.")
