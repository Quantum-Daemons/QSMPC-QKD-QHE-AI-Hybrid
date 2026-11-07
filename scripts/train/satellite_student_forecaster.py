#!/usr/bin/env python
"""The satellite CKKS student as a linear next-step forecaster (Round 86, the owner's choice of 2026-11-07).

WHY A FORECASTER
    The logistic student classified the teacher's flags from 8 rows and could not follow the retrained teacher. On its
    evaluation slice it scored 0 against the teacher's 0.961, and it raised 51 alarms where the teacher raised 2,349.
    A threshold on a forecast error is not a linear function of the window, so no linear classifier can reproduce it,
    but the forecast itself is close to linear in the last rows, and Telemanom's thresholding needs nothing besides the
    forecast and the actual values. So the encrypted part is the forecast: six outputs, each a dot product with the
    last K rows (K x 6 features), which the page evaluates under CKKS with `encryptedLinearBatch`. The client decrypts
    the forecast and thresholds |forecast - actual| itself, in plaintext, with the same procedure the teacher's flags
    come from (`qsh.models.telemanom_threshold`, ported to the page as `telemanomThreshold.ts`).

THE FIT
    Ridge regression in closed form to the TEACHER's step-1 forecasts on the teacher's own unlabelled training windows:
    distillation, the student learning to be the teacher's forecaster rather than an independent one. The penalty is
    chosen on the validation months by how closely the student's forecasts follow the teacher's there. A fit to the
    true next values, chosen the same way against the truth, is recorded beside it for comparison.

FIDELITY
    The student pipeline (its forecast, Telemanom's thresholding at the teacher's own floor, ESA-ADB's scorer) against
    the teacher's over the whole test half, which neither model trained on: ESA-ADB's corrected event-wise F0.5 for
    both, over anomalies alone and over all events; their flag agreement; and their forecast errors against
    persistence on the windows that touch no label.

Reads the published teacher in models/satellite_telemanom (model.safetensors, manifest.json and its record); run
after scripts/train/satellite_telemanom.py.

    python scripts/train/satellite_student_forecaster.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from train import satellite_telemanom as T  # noqa: E402

from qsh.models import distil, export, registry, telemanom_threshold, train  # noqa: E402
from qsh.models.esa_scores import esa_event_wise  # noqa: E402

#: Rows of history the student reads. 8 x 6 = 48 features: one 64-slot block per row under CKKS, as the logistic
#: student's 48 used.
K = 8
#: Ridge penalties tried on the validation months.
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7)
TEACHER = "satellite_comms.telemanom"
STUDENT = "satellite_comms.student"
OUT = REPO / registry.ARTIFACT_DIRS[STUDENT]


def features(z: np.ndarray, rows: np.ndarray, k: int = K) -> np.ndarray:
    """The k rows before each target row, row-major: feature i * 6 + c is channel c at row t - k + i, oldest first."""
    idx = np.asarray(rows)[:, None] + np.arange(-k, 0)
    return z[idx].reshape(len(rows), -1)


def forecast(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The student's forecast, (rows, 6) from (rows, 48)."""
    return X @ W.T + b


def ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with an unpenalised intercept, solved on centred data in float64; W is (outputs, features)."""
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc).T
    return W, y_mean - W @ x_mean


def _teacher_at(model, z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The teacher's step-1 forecast for each target row, from the SEQ_LEN rows before it."""
    import torch

    dev = train._device()
    model.eval().to(dev)
    zt = torch.tensor(z, dtype=torch.float32, device=dev)
    offs = torch.arange(-T.SEQ_LEN, 0, device=dev)
    r = torch.tensor(np.asarray(rows), dtype=torch.long, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(r), 8192):
            out.append(model(zt[r[i:i + 8192, None] + offs])[:, 0, :].double().cpu().numpy())
    return np.concatenate(out)


def _padded(pred: np.ndarray) -> np.ndarray:
    """Forecasts for a period's rows from SEQ_LEN on, with its first SEQ_LEN rows taking the first forecast, as the
    teacher's are padded (ESA-ADB's reconstruction.csv)."""
    return np.concatenate([np.repeat(pred[:1], T.SEQ_LEN, axis=0), pred])


def main() -> int:
    started = time.time()
    rec = registry.achieved(TEACHER) or {}
    split = rec.get("split") or {}
    floor = rec["esa_scores"]["plain_zero_order_hold"]["floor_chosen_on_validation"]
    prep = T.prepare(rec["resample"], split["test_from"], split["validation_from"], hold="last")
    z = prep.zscore(prep.raw_plain).astype(np.float64)
    cut, vstart, sigma = prep.cut, prep.vstart, prep.sigma

    from safetensors.torch import load_file

    teacher_dir = REPO / registry.ARTIFACT_DIRS[TEACHER]
    horizon = int(json.loads((teacher_dir / "manifest.json").read_text())["horizon"])
    model = T.TelemanomLSTM.build(z.shape[1], horizon=horizon)
    model.load_state_dict(load_file(str(teacher_dir / "model.safetensors")))

    # ---- fit -------------------------------------------------------------------------------------------------------
    train_rows = T.clean_starts(prep.any_label, 0, vstart, T.SEQ_LEN + T.HORIZON) + T.SEQ_LEN
    print(f"  {len(train_rows):,} unlabelled training windows; teacher forecasts for their target rows ...")
    X = features(z, train_rows)
    Y_teacher = _teacher_at(model, z, train_rows)
    Y_truth = z[train_rows]
    v_rows = np.arange(vstart + T.SEQ_LEN, cut)
    Xv, Yv_teacher, Yv_truth = features(z, v_rows), _teacher_at(model, z, v_rows), z[v_rows]

    def choose(X_: np.ndarray, Y: np.ndarray, Yv: np.ndarray) -> tuple[float, list[dict], np.ndarray, np.ndarray]:
        table, fits = [], {}
        for alpha in RIDGE_GRID:
            W, b = ridge(X_, Y, alpha)
            mse = float(np.mean((forecast(W, b, Xv) - Yv) ** 2))
            table.append({"alpha": alpha, "validation_mse": round(mse, 9)})
            fits[alpha] = (W, b)
        best = min(table, key=lambda r: (r["validation_mse"], r["alpha"]))["alpha"]
        return best, table, *fits[best]

    alpha, alpha_table, W, b = choose(X, Y_teacher, Yv_teacher)
    alpha_t, _, W_t, b_t = choose(X, Y_truth, Yv_truth)
    print(f"  distilled: alpha {alpha} (validation MSE to the teacher's forecasts "
          f"{min(r['validation_mse'] for r in alpha_table):.6f}); direct fit: alpha {alpha_t}")
    del X, Y_teacher, Y_truth

    # ---- the two pipelines over the test half --------------------------------------------------------------------
    times = prep.df.index[cut:]
    st = prep.labels_all["StartTime"].dt.tz_localize(None)
    en = prep.labels_all["EndTime"].dt.tz_localize(None)
    t_labels = prep.labels_all[(st >= times[0]) & (en <= times[-1])]
    full = (times[0], times[-1])
    rows = np.arange(cut + T.SEQ_LEN, len(z))
    Xt = features(z, rows)
    actual = z[cut:]
    vals = prep.raw_plain[cut:]
    cfg = telemanom_threshold.TelemanomConfig(min_error_value=floor)

    def pipeline(pred_rows: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        err = np.abs(_padded(pred_rows) - actual) * sigma
        _, flagged = telemanom_threshold.flag(err, vals, cfg)
        flagged = flagged.astype(int)
        return flagged, *(esa_event_wise(times, flagged, t_labels, select=sel, full_range=full)
                          for sel in (T.ANOMALIES, T.ALL_EVENTS))

    teacher_pred = _teacher_at(model, z, rows)
    t_flags, t_esa, t_all = pipeline(teacher_pred)
    taught = rec["corrected_event_wise_f0_5"]
    if abs(t_esa["f_0.50"] - taught) > 1e-6:
        raise SystemExit(f"the teacher pipeline gives {t_esa['f_0.50']} here; its record says {taught}")
    s_pred, d_pred = forecast(W, b, Xt), forecast(W_t, b_t, Xt)
    s_flags, s_esa, s_all = pipeline(s_pred)
    _, d_esa, d_all = pipeline(d_pred)
    for name, r_, a_ in (("teacher", t_esa, t_all), ("student (distilled)", s_esa, s_all),
                         ("direct fit", d_esa, d_all)):
        print(f"    {name:20s} F0.5 {r_['f_0.50']:.4f} (tp {r_['tp']} fp {r_['fp']} fn {r_['fn']}) | all events "
              f"{a_['f_0.50']:.4f}")

    # Forecast errors on the windows that touch no label, where a forecaster can be judged as one.
    bad = np.r_[0, np.cumsum(prep.any_label.astype(np.int64))]
    clean = (bad[rows + 1] - bad[rows - T.SEQ_LEN]) == 0
    mae = {name: round(float(np.abs(p[clean] - z[rows][clean]).mean()), 6)
           for name, p in (("student", s_pred), ("teacher", teacher_pred), ("direct_fit", d_pred),
                           ("persistence", z[rows - 1]))}
    mae_to_teacher = round(float(np.abs(s_pred[clean] - teacher_pred[clean]).mean()), 6)
    print(f"  forecast MAE on {int(clean.sum()):,} unlabelled windows: {mae}; student to teacher {mae_to_teacher}")

    # ---- artefacts -------------------------------------------------------------------------------------------------
    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()
    lin.coef_, lin.intercept_ = W.copy(), b.copy()
    lin.n_features_in_ = W.shape[1]
    OUT.mkdir(parents=True, exist_ok=True)
    report = export.verify_and_quantize(lin, Xt[:256].astype(np.float32), OUT, kind="sklearn",
                                        n_features=W.shape[1], int8_required=False)
    (OUT / "weights.json").write_text(json.dumps({
        "kind": "linear-forecaster",
        "note": ("The CKKS-linear student, a next-step forecaster: forecast[c] = W[c] . x + b[c], where x is the last "
                 f"{K} z-scored rows, row-major and oldest first (x[i * 6 + c] is channel c at row t - {K} + i). The "
                 "page evaluates the dot products under CKKS and thresholds |forecast - actual| in plaintext with "
                 "Telemanom's procedure, after converting both to raw units with `input.std`."),
        "teacher_key": TEACHER, "student_key": STUDENT, "k_rows": K,
        "W": W.tolist(), "b": b.tolist(),
        "input": {"mean": prep.mu.tolist(), "std": sigma.tolist(), "channels": list(prep.df.columns)},
        "thresholding": {"min_error_value": floor, **{k: getattr(telemanom_threshold.DEFAULT, k)
                                                       for k in telemanom_threshold.TelemanomConfig.__dataclass_fields__
                                                       if k != "min_error_value"}},
        "ridge_alpha": alpha,
    }, indent=1) + "\n")

    fidelity = distil.measure_fidelity(
        teacher_key=TEACHER, student_key=STUDENT, teacher_pred=t_flags, student_pred=s_flags,
        y_true=prep.mask[cut:].astype(int),
        metric_fn=lambda _y, p: esa_event_wise(times, np.asarray(p).astype(int), t_labels, select=T.ANOMALIES,
                                               full_range=full)["f_0.50"],
        metric_name="corrected_event_wise_f0_5", student_params=int(W.size + b.size),
        teacher_params=int(rec["n_parameters"]), n_scoring_units=int(t_esa["n_selected_ids"]),
        scoring_unit="labelled anomaly events (ESA-ADB's ESAScores rules), over the whole test half")

    def brief(s_: dict, a_: dict) -> dict:
        return {"corrected_event_wise_f0_5": round(float(s_["f_0.50"]), 6), "tp": int(s_["tp"]),
                "fp": int(s_["fp"]), "fn": int(s_["fn"]),
                "all_events_corrected_event_wise_f0_5": round(float(a_["f_0.50"]), 6)}

    train.record_metrics(STUDENT, {
        **fidelity.as_metrics(),
        "student_kind": "linear next-step forecaster, thresholded by the client",
        "k_rows": K,
        "ridge_alpha": alpha,
        "ridge_validation_table": alpha_table,
        "floor": floor,
        "esa_scores": {"teacher": brief(t_esa, t_all), "student": brief(s_esa, s_all),
                       "direct_fit": {**brief(d_esa, d_all), "ridge_alpha": alpha_t}},
        "forecast_mae_unlabelled_windows": mae,
        "forecast_mae_student_to_teacher": mae_to_teacher,
        "n_unlabelled_windows": int(clean.sum()),
        "export": {"fp32_bytes": report.fp32_bytes, "opset": report.opset,
                   **({"int8_skipped_reason": report.int8_skipped_reason} if report.int8_skipped_reason else {})},
        "git_sha": train.git_sha(),
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"\n  wrote {OUT.relative_to(REPO)} and the student's record in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
