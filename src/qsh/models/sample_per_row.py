"""Per-row records of a published graph on its own evaluation sample, computed as the browser computes them (Round 86).

WHY THIS EXISTS

The page re-scores every published graph on its evaluation sample and compared ONE aggregate figure with the record,
within 0.05: a graph four points worse than the record passed, and a graph that answered differently on a hundred rows
whose errors cancelled passed too. Every live comparison read a difference of exactly 0 (2026-11-13), so the tolerance
was hiding nothing it needed to hide. The record now carries each row's decision and its margin to the decision
boundary, and the page compares every row: a disagreement is either a different decision, which fails, or a near tie
within the tolerance below, which two runtimes may break differently and which is reported, not failed.

A TWIN, kept beside its original. Which output is read, how a class is decided and what the margin is follow
`evaluatePublishedModel` in `web/src/lib/onnxInference.ts`; the probabilities follow `outputKind` and
`toProbabilities` in `web/src/lib/calibration.ts`; the preprocessing follows `applyPreprocessing`, which accepts the
three steps below and refuses any other. `tests/test_r86_sample_per_row.py` holds the twin to the browser's text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The largest |margin| difference (classification) or |value| difference (forecast) the page accepts between its
#: own run and the record, and the margin below which a different decision is a near tie rather than a failure.
#: Measured on 2026-11-13 by `tests/test_r86_published_rescore.py` (ONNX Runtime Web's WASM build, one thread, against
#: Python's onnxruntime on the CPU): the largest difference on any published graph was 5.7e-5 (ResNet-50's logits),
#: the smallest 4.6e-7. 1e-3 leaves seventeen times that for the page's threaded and WebGPU runs, while a different
#: graph or preprocessing moves margins by orders of magnitude more (a missing normalisation step took the medical
#: accuracy from 0.8782 to 0.4615).
TOLERANCE = 1e-3

#: The preprocessing steps the page implements (`applyPreprocessing`), and nothing else.
_STEPS = {"divide by 255": lambda x: x / 255.0, "subtract 0.5": lambda x: x - 0.5, "divide by 0.5": lambda x: x / 0.5}


def output_kind(name: str) -> str | None:
    """What a graph's score output holds, from its name: `outputKind` in calibration.ts."""
    return name if name in ("probabilities", "logits") else None


def to_probabilities(data: np.ndarray, n: int, width: int, kind: str) -> np.ndarray:
    """`toProbabilities`: probabilities as they are; logits through a sigmoid (one column) or a shifted softmax."""
    d = np.asarray(data, dtype=np.float64).reshape(n, width)
    if kind == "probabilities":
        return d.copy()
    if width == 1:
        return 1.0 / (1.0 + np.exp(-d))
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def preprocess(samples: list, steps: list[str]) -> np.ndarray:
    """`applyPreprocessing` over every value in float64, then the Float32Array the page builds."""
    x = np.asarray(samples, dtype=np.float64)
    for step in steps:
        if step not in _STEPS:
            raise ValueError(f'unknown preprocessing step "{step}": the page refuses it, so this twin does too')
        x = _STEPS[step](x)
    return x.astype(np.float32)


def _short(v: float) -> float:
    """The shortest decimal that reads back as the same float32: margins and values travel as float32."""
    return float(np.format_float_positional(np.float32(v), unique=True, trim="-"))


def per_row_record(sample: dict, graph: Path) -> dict:
    """Each row's decision and margin (classification) or forecast values (forecast), from `graph` on `sample`."""
    import onnxruntime as ort

    task = sample.get("task", "classification")
    if task not in ("classification", "forecast"):
        raise ValueError(f"a {task} sample is verified by its own path, not by a per-row record")
    n = len(sample["samples"])
    dims = list(sample["input"]["shape"][1:])
    x = preprocess(sample["samples"], sample["input"].get("preprocessing", [])).reshape([n, *dims])
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    names = [o.name for o in sess.get_outputs()]
    # The first float32 output with two or more dimensions, else the first: the page's `found`.
    found = next((i for i, o in enumerate(outs) if isinstance(o, np.ndarray) and o.dtype == np.float32
                  and o.ndim >= 2), 0)
    data, name = np.asarray(outs[found]), names[found]
    width = max(1, data.size // n)
    flat = data.reshape(n, width)
    common = {"rows": n, "read_from": name, "tolerance": TOLERANCE, "graph": graph.name}
    if task == "forecast":
        return {**common, "values": [[_short(v) for v in row] for row in flat],
                "note": "Each window's forecast, from this file's samples and graph by the page's rule; the page "
                        "compares every value."}

    thr = sample.get("expected", {}).get("decision_threshold")
    kind = output_kind(name)
    prob = to_probabilities(flat, n, width, kind) if kind else None
    if thr is not None and prob is None:
        raise ValueError(f'the score output "{name}" is neither probabilities nor logits, so a threshold cannot apply')
    predicted, margin = [], []
    for i in range(n):
        row = flat[i].astype(np.float64)
        if thr is not None and width == 2:
            m = prob[i, 1] - thr
            pred = 1 if m >= 0 else 0
            what = "p(class 1) minus the decision threshold; class 1 at or above 0"
        elif width > 1:
            best = 0
            for c in range(1, width):                    # strictly greater: the first maximum wins
                if row[c] > row[best]:
                    best = c
            runner = max(row[c] for c in range(width) if c != best)
            m, pred = row[best] - runner, best
            what = "the top output minus the runner-up, on the graph's own output"
        else:
            v = prob[i, 0] if prob is not None else row[0]
            m = v - (0.5 if thr is None else thr)
            pred = 1 if m > 0 else 0
            what = "the score minus the threshold; class 1 above 0"
        predicted.append(int(pred))
        margin.append(_short(m))
    return {**common, "predicted": predicted, "margin": margin, "margin_is": what,
            "note": "Each row's class by the page's rule and its margin to the decision boundary, from this file's "
                    "samples, preprocessing and graph with onnxruntime; the page compares every row, and a different "
                    "class within the tolerance of the boundary is a near tie, reported rather than failed."}


def implied_metrics(sample: dict, record: dict) -> dict[str, float]:
    """The figures the page computes from the record's decisions (classification) or values (forecast).

    Each one is set against the figure the sample already states before a record is written: a record whose
    decisions imply another accuracy describes other rows, another graph or another rule."""
    exp = sample["expected"]
    if "values" in record:
        targets = sample.get("targets") or []
        # Over each window's truth, as the page does: a graph that emits more columns than the truth has is read
        # only as far as the truth goes.
        err = [abs(v - t) for row, tr in zip(record["values"], targets, strict=True)
               for v, t in zip(row, tr, strict=False)]
        return {"mae": float(np.mean(err))} if "mae" in exp else {}
    y = np.asarray(sample["labels"])
    p = np.asarray(record["predicted"])
    out: dict[str, float] = {}
    if "accuracy" in exp:
        out["accuracy"] = float((p == y).mean())
    if "minority_class_f1" in exp:
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y != 1)).sum())
        fn = int(((p != 1) & (y == 1)).sum())
        out["minority_class_f1"] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    if "macro_f1" in exp:
        classes = sorted(set(p.tolist()) | set(y.tolist()))
        f1s = []
        for c in classes:
            ctp = int(((p == c) & (y == c)).sum())
            cfp = int(((p == c) & (y != c)).sum())
            cfn = int(((p != c) & (y == c)).sum())
            f1s.append(0.0 if ctp == 0 else 2 * ctp / (2 * ctp + cfp + cfn))
        out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out
