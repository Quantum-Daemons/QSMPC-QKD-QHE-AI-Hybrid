#!/usr/bin/env python3
"""Federated learning across 5G-NIDD's two base stations, exactly as pre-registered (Round 86, batch 6).

references/telecom-nidd-federated-preregistration.md fixed everything below before this ran, on the owner's choice of
2026-11-26: the CKKS student's split cut by base station; each station alone and both pooled by the student's own
procedure, imported from its trainer rather than typed again; and FedAvg under the page's secure aggregation
(`qsh.ai.federated.SecureAggregator`, the browser's construction byte for byte) for 40 rounds of 20 local L-BFGS-B
iterations, its penalty and temperature chosen from masked sums of the stations' validation losses.

CPU only. Writes docs/measurements/telecom-nidd-federated-2026-11-26.json and web/public/data/telecom_nidd/federated.json.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/telecom_nidd_federated.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.ai.federated import SecureAggregator, dequantise, quantise  # noqa: E402
from qsh.models import datasets as D  # noqa: E402
from qsh.models.registry import DS_5G_NIDD  # noqa: E402

PREREG = REPO / "references" / "telecom-nidd-federated-preregistration.md"
PREREG_SHA256 = "5e3a137bfb4c0382ce626bb733531f94b9a505b7ec93e8500bdad8122affff0d"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-federated-2026-11-26.json"
WEB = REPO / "web" / "public" / "data" / "telecom_nidd"
ROUNDS = 40
LOCAL_ITERATIONS = 20
#: The student's recorded test macro-F1 (docs/measurements/telecom-nidd-student-2026-11-25.json), which the pooled arm,
#: the same procedure on the same rows, must reproduce.
STUDENT_TEST_MACRO_F1 = 0.974439
STANDARDISATION_TOLERANCE = 1e-5
STATIONS = ("base station 1 (the CSV's first file)", "base station 2 (the CSV's second file)")
KEYS = ("station_1", "station_2")


def _student():
    """The student's trainer as a module: its fit, metrics and constants are the ones every arm here uses."""
    spec = importlib.util.spec_from_file_location("telecom_nidd_student", REPO / "scripts" / "train" / "telecom_nidd_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _student()


def fit_from(Z: np.ndarray, T: np.ndarray, l2: float, theta0: np.ndarray, maxiter: int) -> np.ndarray:
    """`S.fit`'s objective by L-BFGS-B from `theta0` for at most `maxiter` iterations: one station's local step.

    The objective is typed out here because `S.fit` starts from zeros with 1,000 iterations; `main` checks, before any
    arm runs, that from zeros with 1,000 iterations this returns `S.fit`'s parameters bit for bit.
    """
    from scipy.optimize import minimize

    n, d = Z.shape
    k = T.shape[1]

    def f(theta):
        W, b = theta[: k * d].reshape(k, d), theta[k * d:]
        P = S.softmax(Z @ W.T + b)
        loss = -np.sum(T * np.log(np.clip(P, 1e-12, None))) / n + l2 * np.sum(W * W)
        G = (P - T) / n
        return loss, np.concatenate([(G.T @ Z + 2 * l2 * W).ravel(), G.sum(axis=0)])

    return minimize(f, np.asarray(theta0, dtype=np.float64), jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter}).x


def objective(Z: np.ndarray, T: np.ndarray, l2: float, W: np.ndarray, b: np.ndarray) -> float:
    P = S.softmax(Z @ W.T + b)
    return float(-np.sum(T * np.log(np.clip(P, 1e-12, None))) / len(Z) + l2 * np.sum(W * W))


class Channel:
    """The aggregator's side of the federation: masked shares in, their sum out, one mask index per message.

    Every message is checked against the plain sum of the quantised contributions: the masks must cancel exactly.
    """

    def __init__(self) -> None:
        self.agg = SecureAggregator(n_clients=2)
        self.messages = 0

    def sum(self, contributions: list[np.ndarray]) -> np.ndarray:
        shares = [self.agg.mask_update(c, np.asarray(v, dtype=np.float64), self.messages) for c, v in enumerate(contributions)]
        out = SecureAggregator.aggregate(shares)
        acc = np.zeros(np.shape(contributions[0]), dtype=np.uint64)
        for v in contributions:
            acc += quantise(np.asarray(v, dtype=np.float64), 2).astype(np.uint64)
        if not np.array_equal(out, dequantise(acc % (1 << 32))):
            raise SystemExit(f"message {self.messages}: the masks did not cancel; the aggregation is wrong, run nothing more")
        self.messages += 1
        return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")

    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import train_test_split

    x, y, names, classes = D.load_5g_nidd()
    k = len(classes)
    cut = D.NIDD_SECOND_FILE_STARTS_AT_CLEANED
    onehot = np.eye(k)
    # The student's split, exactly: the same call on the row indices draws the same permutation as on the rows.
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    xtr, xte, ytr, yte = x[itr], x[ite], y[itr], y[ite]
    fi, vi = train_test_split(np.arange(len(ytr)), test_size=0.10, random_state=0, stratify=ytr)
    numeric = [i for i, n in enumerate(D.NIDD_FEATURES) if n not in D.NIDD_CATEGORICAL]
    st_tr, st_te = (itr >= cut).astype(int), (ite >= cut).astype(int)
    st_f, st_v = st_tr[fi], st_tr[vi]

    def stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """As the student computes its own: single-precision mean and deviation over the rows, then double."""
        return a[:, numeric].mean(axis=0).astype(np.float64), a[:, numeric].std(axis=0).astype(np.float64)

    def scaler(centre: np.ndarray, scale: np.ndarray):
        scale = scale.copy()
        scale[scale == 0] = 1.0
        return lambda a: np.clip((a[:, numeric] - centre) / scale, -S.CLIP, S.CLIP)

    # ---- the self-check on the local step: S.fit's parameters, bit for bit, from zeros with 1,000 iterations
    probe = np.random.default_rng(0).choice(len(fi), size=4000, replace=False)
    c0, s0 = stats(xtr)
    z0 = scaler(c0, s0)(xtr[fi][probe])
    Wp, bp = S.fit(z0, onehot[ytr[fi][probe]], 1e-3)
    th = fit_from(z0, onehot[ytr[fi][probe]], 1e-3, np.zeros(k * z0.shape[1] + k), 1000)
    if not np.array_equal(np.concatenate([Wp.ravel(), bp]), th):
        raise SystemExit("the local step's objective is not S.fit's: run nothing")

    rows = {KEYS[s]: {"fit": int(np.sum(st_f == s)), "validation": int(np.sum(st_v == s)), "test": int(np.sum(st_te == s))}
            for s in (0, 1)}
    print(f"  rows by station: {rows}")

    def arm(Zf: np.ndarray, yf: np.ndarray, Zv: np.ndarray, yv: np.ndarray, tag: str) -> dict:
        """The student's procedure: the grid by validation log loss (rounded as its table rounds it), then the temperature."""
        table, fits = [], {}
        for l2 in S.L2_GRID:
            W, b = S.fit(Zf, onehot[yf], l2)
            table.append({"l2": l2, "validation_log_loss": round(S.log_loss(S.softmax(Zv @ W.T + b), onehot[yv]), 8)})
            fits[l2] = (W, b)
        best = min(table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
        W, b = fits[best]
        lv = Zv @ W.T + b

        def val_loss(t: float, lv: np.ndarray = lv) -> float:
            return S.log_loss(S.softmax(lv / np.exp(t)), onehot[yv])

        temperature = float(np.exp(minimize_scalar(val_loss, bounds=(-4, 4), method="bounded").x))
        print(f"  {tag:10s} l2 {best:g}  T {temperature:.4f}  ({time.time() - started:.0f} s)")
        return {"penalty_table": table, "l2": best, "temperature": temperature, "W": W, "b": b, "fits": fits}

    def scores(W: np.ndarray, b: np.ndarray, T: float, Zt: np.ndarray, yt: np.ndarray) -> dict:
        P = S.softmax((Zt @ W.T + b) / T)
        pred = P.argmax(1)
        f1, per = S.macro_f1(yt, pred, k)
        return {"rows": int(len(yt)), "macro_f1": round(f1, 6),
                "per_class_f1": dict(zip(classes, [round(v, 6) for v in per], strict=True)),
                "accuracy": round(float(np.mean(pred == yt)), 6), "ece": round(S.ece(P, yt), 6)}

    def by_station(W, b, T, z) -> dict:
        return {KEYS[s]: scores(W, b, T, z(xte[st_te == s]), yte[st_te == s]) for s in (0, 1)}

    def flamby(per: dict) -> float:
        return round(float(np.mean([per[key]["macro_f1"] for key in KEYS])), 6)

    # ---- pooled: the student itself
    centre, scale_raw = stats(xtr)
    z_student = scaler(centre, scale_raw)
    scale = scale_raw.copy()
    scale[scale == 0] = 1.0
    Zf, Zv = z_student(xtr[fi]), z_student(xtr[vi])
    pooled = arm(Zf, ytr[fi], Zv, ytr[vi], "pooled")
    pooled_union = scores(pooled["W"], pooled["b"], pooled["temperature"], z_student(xte), yte)
    reproduces = abs(pooled_union["macro_f1"] - STUDENT_TEST_MACRO_F1) <= 1e-6
    if not reproduces:
        raise SystemExit(f"the pooled arm scores {pooled_union['macro_f1']}, not the student's {STUDENT_TEST_MACRO_F1}: "
                         "it is not the student, so nothing here compares with it; nothing ships")
    pooled_station = by_station(pooled["W"], pooled["b"], pooled["temperature"], z_student)

    # ---- each station alone, standardised by its own training rows
    alone = []
    parts = []
    for s in (0, 1):
        cs, ss = stats(xtr[st_tr == s])
        parts.append((int(np.sum(st_tr == s)), cs, ss))
        zs = scaler(cs, ss)
        a = arm(zs(xtr[fi][st_f == s]), ytr[fi][st_f == s], zs(xtr[vi][st_v == s]), ytr[vi][st_v == s], f"alone {s + 1}")
        per = by_station(a["W"], a["b"], a["temperature"], zs)
        alone.append({"station": STATIONS[s], "penalty_table": a["penalty_table"], "l2": a["l2"],
                      "temperature": a["temperature"], "test": per, "flamby_metric": flamby(per)})

    # ---- the federated standardisation: the 73 numbers per station, combined by the parallel formula
    n_all = sum(n for n, _, _ in parts)
    c_f = sum(n * c for n, c, _ in parts) / n_all
    s_f = np.sqrt(sum(n * (s_ * s_ + (c - c_f) ** 2) for n, c, s_ in parts) / n_all)
    centre_in_deviations = float(np.max(np.abs(c_f - centre) / scale))
    s_fixed = s_f.copy()
    s_fixed[s_fixed == 0] = 1.0
    scale_rel = float(np.max(np.abs(s_fixed - scale) / scale))
    nz = centre != 0
    centre_rel = float(np.max(np.abs(c_f[nz] - centre[nz]) / np.abs(centre[nz]))) if nz.any() else 0.0
    std_holds = centre_in_deviations <= STANDARDISATION_TOLERANCE and scale_rel <= STANDARDISATION_TOLERANCE
    print(f"  federated standardisation: centre within {centre_in_deviations:.2e} deviations, scale within {scale_rel:.2e}")
    if not std_holds:
        raise SystemExit("the stations' combined standardisation is not the student's to 1e-5: run nothing more")
    z_fed = scaler(c_f, s_f)

    # ---- federated: FedAvg under pairwise masks, one run per penalty
    Zf_s = [z_fed(xtr[fi][st_f == s]) for s in (0, 1)]
    yf_s = [ytr[fi][st_f == s] for s in (0, 1)]
    Zv_s = [z_fed(xtr[vi][st_v == s]) for s in (0, 1)]
    yv_s = [ytr[vi][st_v == s] for s in (0, 1)]
    Zt_s = [z_fed(xte[st_te == s]) for s in (0, 1)]
    yt_s = [yte[st_te == s] for s in (0, 1)]
    n_fit = np.array([len(v) for v in yf_s], dtype=np.float64)
    n_val = np.array([len(v) for v in yv_s], dtype=np.float64)
    w_fit, w_val = n_fit / n_fit.sum(), n_val / n_val.sum()
    d = Zf_s[0].shape[1]
    channel = Channel()

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[: k * d].reshape(k, d), theta[k * d:]

    def pooled_val_loss(W: np.ndarray, b: np.ndarray, T: float = 1.0) -> float:
        return float(channel.sum([np.array([w_val[s] * S.log_loss(S.softmax((Zv_s[s] @ W.T + b) / T), onehot[yv_s[s]])])
                                  for s in (0, 1)])[0])

    runs = {}
    for l2 in S.L2_GRID:
        theta = np.zeros(k * d + k)
        curve = []
        local = []
        for r in range(ROUNDS):
            local = [fit_from(Zf_s[s], onehot[yf_s[s]], l2, theta, LOCAL_ITERATIONS) for s in (0, 1)]
            theta = channel.sum([w_fit[s] * local[s] for s in (0, 1)])
            W, b = unpack(theta)
            point = {"round": r + 1}
            for s in (0, 1):
                point[f"{KEYS[s]}_validation_macro_f1"] = round(S.macro_f1(yv_s[s], (Zv_s[s] @ W.T + b).argmax(1), k)[0], 6)
                point[f"{KEYS[s]}_test_macro_f1"] = round(S.macro_f1(yt_s[s], (Zt_s[s] @ W.T + b).argmax(1), k)[0], 6)
            point["test_mean_macro_f1"] = round((point["station_1_test_macro_f1"] + point["station_2_test_macro_f1"]) / 2, 6)
            curve.append(point)
        W, b = unpack(theta)
        runs[l2] = {"theta": theta, "local": local, "curve": curve,
                    "validation_log_loss": round(pooled_val_loss(W, b), 8)}
        print(f"  federated l2 {l2:g}: pooled validation log loss {runs[l2]['validation_log_loss']:.6f}, "
              f"last round's test mean macro-F1 {curve[-1]['test_mean_macro_f1']:.4f} ({time.time() - started:.0f} s)")
    fed_table = [{"l2": l2, "validation_log_loss": runs[l2]["validation_log_loss"]} for l2 in S.L2_GRID]
    fed_l2 = min(fed_table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
    run = runs[fed_l2]
    W_fed, b_fed = unpack(run["theta"])
    messages_before_temperature = channel.messages
    fed_T = float(np.exp(minimize_scalar(lambda t: pooled_val_loss(W_fed, b_fed, float(np.exp(t))),
                                         bounds=(-4, 4), method="bounded").x))
    fed_station = {KEYS[s]: scores(W_fed, b_fed, fed_T, Zt_s[s], yt_s[s]) for s in (0, 1)}
    fed_union = scores(W_fed, b_fed, fed_T, np.vstack(Zt_s), np.concatenate(yt_s))
    gap = sum(w_fit[s] * objective(Zf_s[s], onehot[yf_s[s]], fed_l2, W_fed, b_fed) for s in (0, 1)) \
        - objective(Zf, onehot[ytr[fi]], fed_l2, *pooled["fits"][fed_l2])
    print(f"  federated: l2 {fed_l2:g}, T {fed_T:.4f}, {channel.messages} masked messages")

    # ---- the criteria
    fed_flamby, pooled_flamby = flamby(fed_station), flamby(pooled_station)
    c1 = all(fed_flamby > a["flamby_metric"] for a in alone)
    c2 = pooled_flamby - fed_flamby <= 0.01

    # ---- the page's held-out rows: the student's 2,048, each named by its station
    rng = np.random.default_rng(0)
    take: list[int] = []
    for c in range(k):
        idx = np.flatnonzero(yte == c)
        take.extend(rng.choice(idx, size=max(1, round(S.HELD_OUT_ROWS * len(idx) / len(yte))), replace=False).tolist())
    take_a = np.sort(np.asarray(take[: S.HELD_OUT_ROWS]))
    held = json.loads((WEB / "student_heldout.json").read_text())
    shipped = np.asarray(held["rows"], dtype=np.float64)
    if not (np.array_equal(np.round(z_student(xte[take_a]), 6), shipped) and held["labels"] == yte[take_a].astype(int).tolist()):
        raise SystemExit("student_heldout.json is not the student's 2,048 rows in this split: the stations cannot be named")
    heldout_station = (ite[take_a] >= cut).astype(int)
    dz = float(np.max(np.abs(z_fed(xte[take_a]) - z_student(xte[take_a]))))
    pred_shipped = (shipped @ W_fed.T + b_fed).argmax(1)
    labels = yte[take_a]
    shipped_by_station = {KEYS[s]: {"rows": int(np.sum(heldout_station == s)),
                                    "accuracy": round(float(np.mean(pred_shipped[heldout_station == s] == labels[heldout_station == s])), 6)}
                          for s in (0, 1)}

    teacher = json.loads((REPO / "models" / "metrics.json").read_text())["telecom_ids.nidd"]
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "stations": dict(zip(KEYS, STATIONS, strict=True)), "rows": rows,
        "features": [D.NIDD_FEATURES[i] for i in numeric], "clip": S.CLIP, "classes": list(classes),
        "local_step_is_the_students_objective": True,
        "standardisation": {
            "exchanged_in_the_clear_per_station": 1 + 2 * len(numeric), "tolerance": STANDARDISATION_TOLERANCE,
            "max_centre_difference_in_deviations": centre_in_deviations, "max_scale_relative_difference": scale_rel,
            "max_centre_relative_difference": centre_rel, "holds": bool(std_holds),
            "max_abs_difference_on_the_shipped_rows": dz},
        "pooled": {"is_the_student": True, "reproduces_the_students_record": bool(reproduces),
                   "penalty_table": pooled["penalty_table"], "l2": pooled["l2"], "temperature": pooled["temperature"],
                   "test": pooled_station, "union_test": pooled_union, "flamby_metric": pooled_flamby},
        "alone": alone,
        "two_by_two_test_macro_f1": [[a["test"][key]["macro_f1"] for key in KEYS] for a in alone],
        "federated": {
            "algorithm": "FedAvg (McMahan et al., AISTATS 2017) under pairwise masks agreed by X25519 (SecureAggregator)",
            "rounds": ROUNDS, "local_iterations": LOCAL_ITERATIONS, "weights": w_fit.tolist(),
            "validation_weights": w_val.tolist(), "penalty_table": fed_table, "l2": fed_l2, "temperature": fed_T,
            "test": fed_station, "union_test": fed_union, "flamby_metric": fed_flamby,
            "objective_gap_to_the_pooled_fit": gap,
            "per_round": run["curve"],
            "masked_messages": channel.messages, "masked_messages_for_the_temperature": channel.messages - messages_before_temperature,
            "masks_cancelled_exactly_in_every_message": True, "key_agreement_ms": round(channel.agg.key_agreement_ms, 3),
            "what_the_aggregator_saw": ("the two stations' X25519 public keys, each message's two masked shares, their sums: "
                                        "each round's global model and the pooled validation losses"),
        },
        "criteria": {"1_federated_above_each_station_alone": bool(c1), "2_within_0_01_of_pooled": bool(c2),
                     "federated_minus_pooled": round(fed_flamby - pooled_flamby, 6),
                     "federated_minus_best_alone": round(fed_flamby - max(a["flamby_metric"] for a in alone), 6)},
        "heldout": {"file": "web/public/data/telecom_nidd/student_heldout.json", "rows": int(len(take_a)),
                    "by_station": shipped_by_station},
        "teacher_reference": {"key": "telecom_ids.nidd", "from": "models/metrics.json", "not_run_here": True,
                              "test_macro_f1": teacher.get("macro_f1"),
                              "cross_base_station": {kk: {"macro_f1": v.get("macro_f1"), "lowest_class": v.get("lowest_class")}
                                                     for kk, v in teacher.get("cross_base_station", {}).items()}},
        "wall_clock_s": round(time.time() - started, 1), "device": "cpu",
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}; criterion 1 {'holds' if c1 else 'FAILS'}, criterion 2 {'holds' if c2 else 'FAILS'}")

    def model(W: np.ndarray, b: np.ndarray) -> dict:
        return {"W": W.tolist(), "b": b.tolist()}

    (WEB / "federated.json").write_text(json.dumps({
        "_generated_by": "scripts/train/telecom_nidd_federated.py; do not edit by hand",
        "preregistration": record["preregistration"], "record": str(RECORD.relative_to(REPO)),
        "stations": list(STATIONS), "features": record["features"], "classes": list(classes), "clip": S.CLIP,
        # The global model is the dequantised masked sum, so every coordinate is a multiple of 2^-20 and survives JSON exactly;
        # the last local models are full doubles, from which the page replays that sum bit for bit.
        "federated": {**model(W_fed, b_fed), "temperature": fed_T, "l2": fed_l2, "rounds": ROUNDS,
                      "local_iterations": LOCAL_ITERATIONS, "weights": w_fit.tolist(),
                      "last_local": [model(*unpack(run["local"][s])) for s in (0, 1)]},
        "heldout_station": heldout_station.tolist(),
        "comparison": {"pooled": {"test": {key: pooled_station[key]["macro_f1"] for key in KEYS}, "flamby_metric": pooled_flamby},
                       "alone": [{"test": {key: a["test"][key]["macro_f1"] for key in KEYS}, "flamby_metric": a["flamby_metric"]}
                                 for a in alone],
                       "federated": {"test": {key: fed_station[key]["macro_f1"] for key in KEYS}, "flamby_metric": fed_flamby}},
        "per_round": [{"round": p["round"], **{key: p[f"{key}_test_macro_f1"] for key in KEYS}} for p in run["curve"]],
        "criteria": record["criteria"],
        "shipped_rows_accuracy_by_station": shipped_by_station,
        "attribution": {"dataset": "5G-NIDD", "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                        "citation": DS_5G_NIDD.attribution, "download": DS_5G_NIDD.url},
        "licence": "Apache-2.0", "note": "fitted numbers and figures computed from 5G-NIDD; the held-out rows are student_heldout.json's",
    }, indent=1) + "\n")
    print(f"wrote {(WEB / 'federated.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
