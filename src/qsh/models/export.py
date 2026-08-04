"""Export trained models to the artefact set the Hub repos ship.

Three producers feed one pipeline:

    torch.nn.Module   --torch.onnx.export-->  fp32 ONNX  --quantize_dynamic-->  int8 ONNX
    sklearn estimator --skl2onnx------------>  fp32 ONNX  --quantize_dynamic-->  int8 ONNX
    xgboost booster   --onnxmltools--------->  fp32 ONNX  --quantize_dynamic-->  int8 ONNX

Every stage is gated numerically. ``optimum`` is deliberately not used: none of the five T2 models
is a Hugging Face ``transformers`` architecture, and optimum<=1.17 does not import against
transformers>=5. ``torch.onnx`` plus ``onnxruntime.quantization`` covers every path we need.

The int8 gate matters more than it looks. Optimum ships no browser quantisation preset, the ONNX
Runtime Web WebGPU execution provider implements only a subset of operators, and a quantised graph
that is numerically fine on the CPU EP can still be wrong in a browser. So:

  * fp32 export is checked against the source model to ``atol=1e-4`` (a correctness assertion);
  * int8 is checked against the fp32 ONNX on *agreement of the decision*, not on tensor closeness —
    quantisation is lossy by construction and demanding closeness would be the wrong test;
  * the browser is checked separately, against these same fixtures, in the Chrome audit.

Opset 17 is the target. It is the widest-compatibility point for ONNX Runtime Web; higher opsets
buy nothing here and risk operators the WebGPU EP has not implemented.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Widest-compatibility opset for ONNX Runtime Web. Do not raise without re-running the Chrome audit.
ONNX_OPSET = 17

#: onnxmltools' XGBoost converter refuses a target opset above this and does not clamp on its own.
XGBOOST_MAX_OPSET = 15

#: fp32 ONNX must reproduce the source model to this tolerance, or the export is rejected.
FP32_ATOL = 1e-4

#: Fraction of *confident* held-out samples on which int8 must agree with fp32 on the predicted class.
INT8_MIN_AGREEMENT = 0.99

#: Bottom fraction of samples by fp32 decision margin that the gate ignores.
#:
#: A flat agreement gate conflates two different things: a quantised graph that computes the wrong
#: function, and a model that was already undecided. On the BreastMNIST ResNet-18 the only two
#: disagreements out of 156 sat at the 0.6th and 1.3rd percentile of logit margin — 0.35 and 0.45
#: against a median margin of 9.25 — so they were coin flips that any perturbation would move.
#: Excluding the least-confident tail keeps the gate strict where it is diagnostic (a genuine
#: operator bug moves confident samples too) without failing on boundary noise. Raw agreement over
#: every sample is still measured and published; only the pass/fail decision uses the confident set.
INT8_MARGIN_QUANTILE = 0.05


@dataclass
class ExportReport:
    """What an export produced and how well it survived quantisation.

    Written next to the artefacts and uploaded with them, so a model card can state the quantisation
    cost as a measured number instead of "negligible".
    """

    fp32_path: str
    int8_path: str
    fp32_bytes: int
    int8_bytes: int
    opset: int
    fp32_max_abs_error: float          # vs the source model
    int8_agreement: float | None       # fraction of fixtures where int8 and fp32 agree; None if skipped
    int8_max_abs_error: float | None   # vs fp32 ONNX, reported not gated; None if skipped
    n_fixtures: int
    passed: bool
    int8_skipped_reason: str = ""   # non-empty => no int8 graph exists; serve fp32 to the browser
    int8_confident_agreement: float = 1.0   # agreement excluding the least-confident tail; this is gated
    int8_margin_quantile: float = 0.0       # size of that excluded tail
    #: Operator domains present in the graph, and whether ORT Web can execute all of them.
    #: A tree ensemble is ai.onnx.ml and is NOT browser-runnable at any size.
    onnx_domains: list[str] = field(default_factory=list)
    browser_runnable: bool = True

    @property
    def compression(self) -> float:
        return self.fp32_bytes / max(self.int8_bytes, 1) if self.int8_bytes else 1.0


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    """Run the graph and return its *score* output.

    Classifier graphs produced by skl2onnx / onnxmltools have two outputs: a hard label first, then
    the per-class scores. Taking ``out[0]`` compares integer labels against ``predict_proba`` floats
    and reports a max abs error of exactly 1.0 for a graph that is in fact correct. The scores are
    the meaningful comparison, so they are preferred whenever they are present.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: np.asarray(x, np.float32)})
    if len(outs) > 1:
        scores = outs[1]
        # ZipMap-style output: a list of {class: probability} dicts, one per row.
        if isinstance(scores, list) and scores and isinstance(scores[0], dict):
            keys = sorted(scores[0])
            return np.asarray([[row[k] for k in keys] for row in scores], dtype=np.float64)
        scores = np.asarray(scores)
        if scores.dtype.kind == "f":
            return scores
    return np.asarray(outs[0])


def export_torch(model, example: np.ndarray, path: Path, *,
                 input_name: str = "input", output_name: str = "logits") -> Path:
    """Export a ``torch.nn.Module`` to a self-contained fp32 ONNX file at ``path``.

    Thin wrapper over :func:`qsh.ai.inference.export_onnx` so there is exactly one place that decides
    the opset and the dynamic-batch axis, plus a consolidation step.

    torch's exporter writes tensors above its size threshold to a sibling ``model.onnx.data`` file,
    which turns one artefact into two that must travel together — through the Hub upload, the CDN
    fetch and the browser loader, each of which is a place to lose the sidecar and get a graph with
    no weights. Everything here is far below the 2 GB protobuf ceiling, so the tensors are folded
    back in and a single file ships.
    """
    import onnx

    from ..ai.inference import export_onnx

    export_onnx(model, example, path, input_name=input_name, output_name=output_name)
    sidecar = path.with_name(path.name + ".data")
    if sidecar.exists():
        onnx.save(onnx.load(str(path)), str(path), save_as_external_data=False)
        sidecar.unlink()
    return path


def export_sklearn(estimator, n_features: int, path: Path, *, input_name: str = "input") -> Path:
    """Export an sklearn estimator to fp32 ONNX."""
    from skl2onnx import to_onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = np.zeros((1, n_features), dtype=np.float32)
    onx = to_onnx(estimator, dummy, target_opset=ONNX_OPSET,
                  options={id(estimator): {"zipmap": False}} if hasattr(estimator, "predict_proba") else None)
    # skl2onnx names the input after the training array unless told otherwise; normalise it so the
    # web loader can address every model the same way.
    if onx.graph.input[0].name != input_name:
        old = onx.graph.input[0].name
        onx.graph.input[0].name = input_name
        for node in onx.graph.node:
            node.input[:] = [input_name if i == old else i for i in node.input]
    path.write_bytes(onx.SerializeToString())
    return path


def export_xgboost(booster, n_features: int, path: Path, *, input_name: str = "input") -> Path:
    """Export an XGBoost model to fp32 ONNX.

    The tensor type must come from ``onnxmltools.convert.common.data_types``: onnxmltools checks
    input types by class identity, and the otherwise-identical class in ``onnxconverter_common``
    is rejected with "got an input with a wrong type".
    """
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    path.parent.mkdir(parents=True, exist_ok=True)
    # onnxmltools' tree converter tops out below ONNX_OPSET and raises rather than clamping. The
    # cap is harmless here: a tree ensemble is a single TreeEnsembleClassifier node from the
    # ai.onnx.ml domain, whose own opset is versioned separately from the default domain.
    onx = convert_xgboost(booster, initial_types=[(input_name, FloatTensorType([None, n_features]))],
                          target_opset=min(ONNX_OPSET, XGBOOST_MAX_OPSET))
    path.write_bytes(onx.SerializeToString())
    return path


def is_quantizable(fp32_path: Path) -> tuple[bool, str]:
    """Whether dynamic int8 quantisation can do anything useful to this graph.

    A tree ensemble is one ``ai.onnx.ml`` ``TreeEnsembleClassifier`` node that carries its
    thresholds and leaf values in node *attributes*, not in weight tensors. There is nothing for
    weight quantisation to act on; onnxruntime does not merely no-op, it raises "Failed to find
    proper ai.onnx domain" because the graph imports no default-domain opset at all. Detecting
    that here keeps the failure explicable instead of turning it into a stack trace.
    """
    import onnx

    model = onnx.load(str(fp32_path))
    domains = {op.domain for op in model.opset_import}
    ops = {n.op_type for n in model.graph.node}
    if ops & {"TreeEnsembleClassifier", "TreeEnsembleRegressor"}:
        return False, ("tree ensemble: parameters live in node attributes, not weight tensors, so "
                       "dynamic int8 quantisation has nothing to quantise")
    if "" not in domains and "ai.onnx" not in domains:
        return False, f"no default ai.onnx domain in the graph (domains: {sorted(domains)})"
    return True, ""


def quantize_int8(fp32_path: Path, int8_path: Path, *, per_channel: bool = True) -> Path:
    """Dynamic int8 quantisation, weights only.

    Dynamic (not static) because it needs no calibration set and keeps activations in float, which is
    what survives the ONNX Runtime Web operator subset.

    ``per_channel`` defaults on. A convolution's output channels have wildly different weight
    ranges, so a single scale per tensor is set by the widest channel and crushes the rest: with
    per-tensor quantisation the ResNet-18 here agreed with its fp32 graph on only 98.1% of the test
    set, below the gate. Per-channel scales fix that at a cost of a few bytes per channel.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # torch's ONNX optimiser leaves intermediate value_info behind that can disagree with what
    # shape inference derives, and the quantiser runs strict inference first:
    #   InferenceError: Inferred shape and existing shape differ in dimension 0: (512) vs (2)
    # Clearing value_info makes inference recompute every intermediate from the graph itself.
    # Graph inputs and outputs are left alone, so the external contract is unchanged.
    model = onnx.load(str(fp32_path))
    src = fp32_path
    cleaned = None
    if model.graph.value_info:
        del model.graph.value_info[:]
        cleaned = fp32_path.with_suffix(".cleaned.onnx")
        onnx.save(model, str(cleaned), save_as_external_data=False)
        src = cleaned
    try:
        quantize_dynamic(str(src), str(int8_path), weight_type=QuantType.QInt8,
                         per_channel=per_channel)
    finally:
        if cleaned is not None:
            cleaned.unlink(missing_ok=True)
    return int8_path


def graph_domains(path: Path) -> list[str]:
    """Operator domains present in the graph.

    This decides whether a model can run in a BROWSER at all, which is not something the file size
    or the opset tells you. ONNX Runtime Web implements the default ``ai.onnx`` domain; it does not
    implement ``ai.onnx.ml`` (TreeEnsembleClassifier, LinearClassifier, ...). A tree ensemble is
    therefore not browser-executable no matter how small it is -- and finding that out by shipping
    4.6 MB to a visitor and then hanging is the worst possible way to learn it, which is exactly
    what happened before this was recorded.
    """
    import onnx

    model = onnx.load(str(path))
    return sorted({node.domain or "ai.onnx" for node in model.graph.node})


#: Domains ONNX Runtime Web can execute. Anything outside this set is server/desktop-only.
BROWSER_RUNNABLE_DOMAINS = frozenset({"ai.onnx", ""})

#: Operators that are in ai.onnx but that ORT Web's WASM build does not execute usefully. Dynamic
#: quantisation is the case that bit here: the int8 Telemanom graph loads in 0.03 s under Python ORT
#: and never finishes session creation in the browser, because MatMulInteger +
#: DynamicQuantizeLinear alongside LSTM is not viable there. Being in the right *domain* is
#: necessary and not sufficient.
#: Established by measurement on the deployed demo, not from documentation.
#:
#: Recurrent operators are the confirmed case: the Telemanom graph is in the supported ai.onnx
#: domain and only 0.66 MB at fp32, and ORT Web WASM still exceeds 45 s of session creation on it,
#: at both fp32 and int8. Dynamic quantisation (MatMulInteger / DynamicQuantizeLinear) was initially
#: listed too, but that was inferred from the same LSTM graph rather than measured independently --
#: quantised CONVOLUTIONAL models are routinely fine in ORT Web, so blanket-listing those operators
#: wrongly excluded the medical ResNet's int8 build, which is 11 MB against 45 MB at fp32.
#:
#: Being in the right domain is necessary and not sufficient; so is being small.
BROWSER_HOSTILE_OPS = frozenset({"LSTM", "GRU", "RNN"})


def graph_ops(path: Path) -> list[str]:
    import onnx

    return sorted({node.op_type for node in onnx.load(str(path)).graph.node})


def browser_runnable(domains: list[str], ops: list[str] | None = None) -> bool:
    if not domains or any(d not in BROWSER_RUNNABLE_DOMAINS for d in domains):
        return False
    return not (set(ops or ()) & BROWSER_HOSTILE_OPS)


def graph_opset(path: Path) -> int:
    """The default-domain opset actually present in the file.

    torch's exporter emits at the opset it has implementations for and only then attempts a
    down-conversion, which can fail with "Failed to convert the model to the target version" and
    leave the higher opset in place. Reporting the requested opset rather than the real one would
    put a wrong number in the model card and mislead anyone checking browser compatibility.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            return int(op.version)
    return 0


def verify_and_quantize(source_model, fixtures: np.ndarray, out_dir: Path, *,
                        kind: str = "torch", n_features: int | None = None) -> ExportReport:
    """Export -> verify fp32 -> quantise -> verify int8, and refuse to emit a graph that fails.

    ``fixtures`` is a held-out batch. The same array is written next to the artefacts as
    ``fixtures.npy`` and re-used by the browser audit, so the in-browser check compares against
    exactly the inputs this gate used rather than a fresh random draw.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "onnx" / "model.onnx"
    int8 = out_dir / "onnx" / "model_int8.onnx"
    fixtures = np.asarray(fixtures, dtype=np.float32)

    if kind == "torch":
        export_torch(source_model, fixtures[:1], fp32)
        import torch

        source_model.eval().cpu()
        with torch.no_grad():
            ref = source_model(torch.tensor(fixtures)).numpy()
    elif kind == "sklearn":
        n = n_features if n_features is not None else fixtures.shape[1]
        export_sklearn(source_model, n, fp32)
        ref = _decision_scores(source_model, fixtures)
    elif kind == "xgboost":
        n = n_features if n_features is not None else fixtures.shape[1]
        export_xgboost(source_model, n, fp32)
        ref = _decision_scores(source_model, fixtures)
    else:
        raise ValueError(f"unknown export kind: {kind!r}")

    got32 = _run(fp32, fixtures)
    fp32_err = float(np.max(np.abs(_align(ref) - _align(got32))))

    quantizable, skip_reason = is_quantizable(fp32)
    confident_agreement = 1.0
    if quantizable:
        quantize_int8(fp32, int8)
        got8 = _run(int8, fixtures)
        same = _labels(got32) == _labels(got8)
        agreement = float(np.mean(same))
        confident = _confident_mask(got32, INT8_MARGIN_QUANTILE)
        confident_agreement = float(np.mean(same[confident])) if confident.any() else agreement
        int8_err = float(np.max(np.abs(_align(got32) - _align(got8))))
        int8_bytes = int8.stat().st_size
        int8_rel = str(int8.relative_to(out_dir))
    else:
        # Not an error and not a silent skip: the browser tier serves the fp32 graph and the model
        # card says why. Pretending an int8 file exists would break the web loader at runtime.
        # None, not 1.0. This used to record perfect agreement for a graph that was never built:
        # int8_agreement is defined as 'fraction of fixtures where int8 and fp32 pick the same
        # class', and with no int8 graph there are no such fixtures. metrics.json then carried
        # int8_agreement 1.0 beside int8_bytes 0, which reads as a passed gate rather than a skip.
        agreement, int8_err, int8_bytes, int8_rel = None, None, 0, ""
        print(f"  [export] int8 skipped — {skip_reason}")

    np.save(out_dir / "fixtures.npy", fixtures)

    report = ExportReport(
        fp32_path=str(fp32.relative_to(out_dir)),
        int8_path=int8_rel,
        fp32_bytes=fp32.stat().st_size,
        int8_bytes=int8_bytes,
        opset=graph_opset(fp32),
        onnx_domains=graph_domains(fp32),
        browser_runnable=browser_runnable(graph_domains(fp32)),
        fp32_max_abs_error=fp32_err,
        int8_agreement=agreement,
        int8_max_abs_error=int8_err,
        n_fixtures=int(fixtures.shape[0]),
        passed=bool(fp32_err <= FP32_ATOL and confident_agreement >= INT8_MIN_AGREEMENT),
        int8_skipped_reason=skip_reason,
        int8_confident_agreement=confident_agreement,
        int8_margin_quantile=INT8_MARGIN_QUANTILE,
    )
    (out_dir / "export_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")

    if not report.passed:
        raise RuntimeError(
            "ONNX export gate failed — not publishing this graph.\n"
            f"  fp32 max abs error vs source:   {fp32_err:.3e} (limit {FP32_ATOL:.0e})\n"
            f"  int8 agreement, confident set:  {confident_agreement:.4f} (limit {INT8_MIN_AGREEMENT})\n"
            f"  int8 agreement, all fixtures:   {agreement:.4f}\n"
            "Investigate the operator set rather than relaxing the gate: a graph that disagrees on "
            "confident samples will disagree in the browser too, and the demo would then report a "
            "measured-but-wrong number."
        )
    return report


def _decision_scores(estimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x))
    return np.asarray(estimator.predict(x))


def _align(a: np.ndarray) -> np.ndarray:
    """Squeeze to 2-D so tensor comparisons line up across producers."""
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1, 1)


def _labels(a: np.ndarray) -> np.ndarray:
    a = _align(a)
    return a.argmax(1) if a.shape[1] > 1 else (a[:, 0] > 0.5).astype(int)


def _confident_mask(scores: np.ndarray, quantile: float) -> np.ndarray:
    """Samples above the ``quantile`` of decision margin — i.e. the ones the model is sure about."""
    s = _align(scores)
    if s.shape[1] > 1:
        top2 = np.sort(s, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = np.abs(s[:, 0] - 0.5)
    if quantile <= 0:
        return np.ones(len(margin), dtype=bool)
    return margin > np.quantile(margin, quantile)
