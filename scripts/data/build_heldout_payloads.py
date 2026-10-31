#!/usr/bin/env python3
"""Build the held-out rows the browser scores under CKKS, and the OPS-SAT segments that replace SMAP/MSL.

WHY THIS EXISTS (Round 84)

The private-inference figure is a claim about a model on rows. On finance it encrypted a logistic
surrogate over 400 PaySim rows the surrogate had been FITTED on, under a caption saying "held-out",
beside an overlay that drew a different corpus (AMLworld) for a different model. Telecom scored the
published student on rows a JavaScript mirror of the generator drew in the tab, which the student
never saw but which are not the split it was evaluated on either. This script ships, per use case,
the rows the PUBLISHED model was never fitted on, in the form the browser needs to encrypt them:

  finance    the 8,192 rows of the AMLworld temporal test window that `finance/real_2d.json`
             already draws, as the 32 raw graph features `finance_aml.student` takes, with a
             byte-identical copy of that student's published `weights.json` (mean, scale, W, b).
  telecom    the 600 rows `telecom_ids.qml` and `telecom_ids.student` were evaluated on: the
             held-out quarter of `generate(n_per_class=300, seed=42)` under the trainer's own
             stratified split, which `models/telecom_qml_ids/eval_sample.json` already publishes.
  satellite  NOT a private-inference payload. The satellite page scored a NASA SMAP/MSL window for
             its plaintext anomaly verdict and SHAP, and no provider grants a licence for that
             data (JPL is Caltech; DATA_LICENSES.md section 2). It is replaced by OPS-SAT-AD segments,
             CC BY 4.0, from the benchmark's test partition that `models/satellite_opssat/
             eval_sample.json` already ships. The partition's train half is not on this machine, so
             the demo regression is fitted on 370 of the 529 test segments and the verdict is drawn
             from the other 159; it is a demo model, NOT `satellite_comms.opssat` or its student,
             and the payload says so. The full deployment's CKKS telemetry stage scores the same
             file, from a byte-identical copy this function also writes to data/satellite/.

Nothing here is downloaded. Every input is a file this repository already tracks, or the
repository's own generator.

Run:  .venv/bin/python scripts/data/build_heldout_payloads.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models.registry import DS_AMLWORLD, DS_OPSSAT_AD, DS_QKD_ATTACK_SYNTH  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

WEB = REPO / "web" / "public" / "data"

CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
CDLA_SHARING_1 = "https://cdla.dev/sharing-1-0/"

#: The OPS-SAT-AD dataset record's own credit, beside the article the registry credits. CC BY 4.0
#: asks for the creator "in the manner requested", and the Zenodo record names its four creators in
#: this order (read at source 2026-10-30, recorded in the plots-2d audit).
OPSSAT_RECORD = ("Data: Ruszczak, B., Kotowski, K., Nalepa, J. & Evans, D., 'OPSSAT-AD - anomaly "
                 "detection dataset for satellite telemetry', Zenodo (2024), "
                 "doi:10.5281/zenodo.12588359 (version 1).")

#: The satellite demo sample's provenance. The registry's `BUNDLED_DEMO_SAMPLES["satellite_comms"]`
#: still names SMAP/MSL, and `tests/test_demo_provenance.py` holds the shipped file to that text; the
#: integrator note C3-for-integrator.md sets the registry to exactly this string. It is a constant
#: here, not read from the registry, because reading it would write the SMAP/MSL text into an
#: OPS-SAT file until that note is applied.
SATELLITE_PROVENANCE = (
    "real OPS-SAT-AD telemetry segments (ESA OPS-SAT CubeSat; Zenodo 10.5281/zenodo.12588359, version 1, "
    "CC-BY-4.0): 19 precomputed per-segment features of the benchmark's test partition. The demo's "
    "anomaly regression is fitted here on 370 of those 529 segments and the verdict is drawn from "
    "the other 159; it is not the published satellite_comms.opssat model.")


def _g7(v: float) -> float:
    """Seven significant figures: below the published rows' own precision, so no decision moves."""
    return float(f"{float(v):.7g}")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"  wrote {path.relative_to(REPO)} ({path.stat().st_size:,} B)")


def finance() -> None:
    """The overlay's 8,192 AMLworld test-window rows, as the published student's raw input."""
    gbm = json.loads((REPO / "models" / "finance_aml_gbm" / "eval_sample.json").read_text())
    stu = json.loads((REPO / "models" / "finance_aml_student" / "eval_sample.json").read_text())
    weights_src = REPO / "models" / "finance_aml_student" / "weights.json"
    w = json.loads(weights_src.read_text())
    names = gbm["feature_names"]
    X = np.asarray(gbm["samples"], dtype=np.float64)
    y = np.asarray(gbm["labels"], dtype=int)
    assert gbm["labels"] == stu["labels"], "the teacher's and student's samples are different rows"
    mean, scale = np.asarray(w["input"]["mean"]), np.asarray(w["input"]["scale"])
    Wv, b = np.asarray(w["W"][0]), float(w["b"][0])

    rows = [[_g7(v) for v in r] for r in X]
    Z = (np.asarray(rows) - mean) / scale
    # The rows as shipped, through the student's OWN preprocessing, against the z-scores the student's
    # published sample carries. If they disagreed on one decision the browser would be scoring a
    # different input from the one the model card's figure was computed on.
    z_pub = np.asarray(stu["samples"], dtype=np.float64)
    logit, logit_pub = Z @ Wv + b, z_pub @ Wv + b
    flips = int(((logit > 0) != (logit_pub > 0)).sum())
    assert flips == 0, f"{flips} student decisions differ from the published sample's z-scores"
    shift = float(np.abs(logit - logit_pub).max())

    # The bundled weights are the published file, byte for byte, so the browser runs that model and
    # not a re-serialisation of it.
    (WEB / "finance").mkdir(parents=True, exist_ok=True)
    (WEB / "finance" / "student_weights.json").write_bytes(weights_src.read_bytes())
    print(f"  wrote web/public/data/finance/student_weights.json (copy of "
          f"{weights_src.relative_to(REPO)})")

    pred = (logit > 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    _write(WEB / "finance" / "student_heldout.json", {
        "usecase": "finance_aml",
        "model_key": w["student_key"],
        "teacher_key": w["teacher_key"],
        "weights": "data/finance/student_weights.json",
        "corpus": "IBM AMLworld HI-Small (synthetic, by IBM's own statement)",
        "provenance": ("The 8,192 rows finance/real_2d.json draws, in the same order: a draw from "
                       "the AMLworld temporal test window (the last 30 percent of transactions), "
                       "enriched to 2 percent laundering, as the 32 graph features the published "
                       "student takes before its own z-scoring."),
        # The registry's split, not the sample file's older wording ("of the timeline"): the window
        # is the last 30 percent of the TRANSACTIONS in time order, a different span of days.
        "split": get_spec(w["student_key"]).eval_split,
        "fit_note": ("finance_aml.student and its teacher finance_aml.gbm were fitted on the "
                     "earlier training window (fit 56 percent, validation 14); neither was fitted, "
                     "tuned or calibrated on these test-window rows."),
        "licence": "CDLA-Sharing-1.0",
        "licence_url": CDLA_SHARING_1,
        "attribution": DS_AMLWORLD.attribution,
        "modifications": ("Computed graph features (32 per transaction) of 8,192 test-window rows "
                          "drawn with laundering enriched to 2 percent, rounded to 7 significant "
                          "figures. Enhanced Data shared under the same CDLA-Sharing-1.0."),
        "feature_names": names,
        "input_space": "raw",
        "class_names": ["legitimate", "laundering"],
        "plane": {"x": {"key": "src_fan_out_ratio", "label": "source fan-out ratio"},
                  "y": {"key": "src_burst_fanout", "label": "source burst fan-out", "scale": "log"}},
        "samples": rows,
        "labels": [int(v) for v in y],
        "expected": {
            "rows": int(len(y)), "positives": int(y.sum()),
            "plaintext_flagged": int(pred.sum()),
            "plaintext_minority_class_f1": round(f1, 6),
            "smallest_abs_logit": float(np.abs(logit).min()),
            "max_logit_shift_vs_published_sample": shift,
            "note": ("The published student in plaintext on these rows, computed by this script "
                     "with the shipped weights; the browser's plaintext reference must reproduce "
                     "it. Rounding to 7 significant figures moves no decision: the largest logit "
                     "shift against the student's published z-scored sample is "
                     f"{shift:.2e}, below the smallest |logit| of "
                     f"{float(np.abs(logit).min()):.2e}."),
        },
    })


def telecom() -> None:
    """The held-out quarter of the seed-42 draw the published telecom student was evaluated on."""
    from sklearn.model_selection import train_test_split

    from qsh.qml_ids.attacks import ATTACKS, FEATURES, generate

    X, y, class_names = generate(n_per_class=300, seed=42)
    assert list(class_names) == list(ATTACKS)
    # The trainer's split, call for call (`scripts/train/telecom_ids.py`): test_size 0.25,
    # random_state 42, stratified. Re-derived rather than copied so the fit side is known too.
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
    assert not set(tr) & set(te)
    pub = json.loads((REPO / "models" / "telecom_qml_ids" / "eval_sample.json").read_text())
    gap = float(np.abs(np.asarray(pub["samples"]) - X[te]).max())
    assert gap < 1e-5 and pub["labels"] == [int(v) for v in y[te]], (
        f"the regenerated held-out quarter is not the published eval sample (max gap {gap:.2e})")

    w = json.loads((WEB / "telecom" / "student_weights.json").read_text())
    rows = [[_g7(v) for v in r] for r in X[te]]
    Z = (np.asarray(rows) - np.asarray(w["input"]["mean"])) / np.asarray(w["input"]["scale"])
    scores = Z @ np.asarray(w["W"]).T + np.asarray(w["b"])
    srt = np.sort(scores, axis=1)
    acc = float((scores.argmax(1) == y[te]).mean())
    _write(WEB / "telecom" / "student_heldout.json", {
        "usecase": "telecom_ids",
        "model_key": w["student_key"],
        "teacher_key": w["teacher_key"],
        "weights": "data/telecom/student_weights.json",
        "corpus": "synthetic QKD-session telemetry generated by qsh.qml_ids.attacks",
        "provenance": ("The 600 rows models/telecom_qml_ids/eval_sample.json publishes: the held-out "
                       "quarter of generate(n_per_class=300, seed=42) under the trainer's own "
                       "stratified 75/25 split (random_state 42), 75 per class. Synthetic: no "
                       "number here is a claim about a real link."),
        "split": get_spec(w["teacher_key"]).eval_split,
        "fit_note": ("telecom_ids.qml was fitted, and telecom_ids.student distilled, on the other "
                     "1,800 rows of the same draw; neither saw these 600."),
        "licence": DS_QKD_ATTACK_SYNTH.licence,
        "licence_url": None,
        "attribution": ("Class structure per Al-Kuwari et al., arXiv:2509.14282. The rows are "
                        "produced by this repository's own generator."),
        "modifications": "Rounded to 7 significant figures.",
        "feature_names": list(FEATURES),
        "input_space": "raw",
        "class_names": list(ATTACKS),
        "plane": {"x": {"key": "qber_std", "label": "QBER standard deviation"},
                  "y": {"key": "multiphoton_rate", "label": "multiphoton rate"}},
        "samples": rows,
        "labels": [int(v) for v in y[te]],
        "expected": {
            "rows": int(len(te)), "per_class": 75,
            "plaintext_accuracy": round(acc, 6),
            "smallest_top1_margin": float((srt[:, -1] - srt[:, -2]).min()),
            "note": "The published student in plaintext on these rows, computed by this script.",
        },
    })


def satellite() -> None:
    """OPS-SAT-AD segments for the plaintext anomaly verdict, replacing the unlicensed SMAP/MSL file."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    src = REPO / "models" / "satellite_opssat" / "eval_sample.json"
    d = json.loads(src.read_text())
    names = d["feature_names"]
    X = np.asarray(d["samples"], dtype=np.float64)
    y = np.asarray(d["labels"], dtype=int)
    fit, held = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    fit, held = np.sort(fit), np.sort(held)
    scaler = StandardScaler().fit(X[fit])
    Z = scaler.transform(X)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Z[fit], y[fit])
    s = Z[held] @ lr.coef_[0] + lr.intercept_[0]
    acc = float(((s > 0).astype(int) == y[held]).mean())
    ap = float(average_precision_score(y[held], s))
    base = float(y[held].mean())
    _write(WEB / "satellite" / "opssat_segments.json", {
        "provenance": SATELLITE_PROVENANCE,
        "licence": "CC-BY-4.0",
        "licence_url": CC_BY_4,
        "attribution": f"{DS_OPSSAT_AD.attribution} {OPSSAT_RECORD}",
        "modifications": ("The 19 published per-segment features of the 529 test-partition "
                          "segments, standardised with constants fitted on 370 of them; only the "
                          "other 159 ship here, rounded to 5 decimals, with their labels."),
        "names": names,
        "feature_space": "zscore",
        "input": {"mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scaler.scale_]},
        "model": {"W": [[0.0] * len(names), [float(v) for v in lr.coef_[0]]],
                  "b": [0.0, float(lr.intercept_[0])]},
        "real_model": ("logistic regression over the 19 OPS-SAT-AD segment features, fitted here on "
                       f"{len(fit)} of the 529 test-partition segments (stratified, seed 0); the "
                       f"{len(held)} segments below were not used to fit it or its scaler. A demo "
                       "model, NOT satellite_comms.opssat or satellite_comms.opssat_student, which "
                       "are fitted on the benchmark's train partition."),
        "segments": [[round(float(v), 5) for v in Z[i]] for i in held],
        "labels": [int(v) for v in y[held]],
        "row_index": [int(i) for i in held],
        # E[x] over the FITTING segments, the SHAP reference point. Zero by construction in this
        # z-space; shipped so the reader does not have to know that, and so a re-fit cannot leave
        # the browser averaging the held-out rows instead.
        "baseline": [round(float(v), 9) for v in Z[fit].mean(axis=0)],
        "baseline_note": f"mean of the {len(fit)} fitting segments (the zero vector in this z-space)",
        "measured": {"heldout_rows": int(len(held)), "fit_rows": int(len(fit)),
                     "heldout_accuracy": round(acc, 6), "heldout_auc_pr": round(ap, 6),
                     "heldout_base_rate": round(base, 6),
                     "note": ("This demo regression on its own held-out segments, computed by this "
                              "script. Not comparable with the benchmark's published figures, which "
                              "are fitted on the train partition and scored on all 529.")},
    })
    # The full deployment's CKKS twin (`qsh.data.loaders.load_opssat_segments`) reads the same bytes from
    # data/, which the API image ships and web/public/ is not; tests/test_r84_satpy_opssat_ckks.py holds
    # the two copies byte-identical.
    copy = REPO / "data" / "satellite" / "opssat_segments.json"
    copy.write_bytes((WEB / "satellite" / "opssat_segments.json").read_bytes())
    print(f"  wrote {copy.relative_to(REPO)} (copy of web/public/data/satellite/opssat_segments.json)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["finance", "telecom", "satellite"], default=None)
    args = ap.parse_args()
    for name, fn in (("finance", finance), ("telecom", telecom), ("satellite", satellite)):
        if args.only and name != args.only:
            continue
        print(f"[{name}]")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
