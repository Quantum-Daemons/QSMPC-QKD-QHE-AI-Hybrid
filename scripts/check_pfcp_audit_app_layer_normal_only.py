#!/usr/bin/env python3
"""Three follow-ups to the PFCP held-out-day check, as pre-registered (Round 86, batch 13):
references/telecom-pfcp-audit-app-layer-normal-only-preregistration.md.

Arm A audits each run's labels against its addresses and its attack's message counter (counts and shares, no
address); arm P runs the held-out-day check's teacher and student on the application-layer PFCP message counters;
arm O fits IForest, kNN and HBOS on Normal flows only, on both feature sets. Reads the deposit's archives in place
(nothing extracted) and writes docs/measurements/telecom-pfcp-audit-app-layer-normal-only-<date>.json.

    QSH_DATA_ROOT=... python scripts/check_pfcp_audit_app_layer_normal_only.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "references" / "telecom-pfcp-audit-app-layer-normal-only-preregistration.md"
PREREG_SHA256 = "0209f741613d48e5786d1c94b01272a24f0fbee0f756cb987177c65479ae51e9"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _module("check_pfcp_held_out_day", REPO / "scripts" / "check_pfcp_held_out_day.py")   # puts src/ on the path
from qsh.models import datasets as D  # noqa: E402

APP_IDENTIFIERS = {"", "flow ID", "source IP", "destination IP", "source port", "destination port", "protocol"}
ADDRESSES = {"cic": ("Src IP", "Dst IP"), "app": ("source IP", "destination IP")}
ATTACK_COUNTER = {"Establishment": "PFCPSessionEstablishmentRequest_counter",
                  "Deletion": "PFCPSessionDeletionRequest_counter",
                  "Modification": "PFCPSessionModificationRequest_counter"}
IFOREST_SEEDS, KNN_K, HBOS_BINS = (0, 1, 2, 3, 4), 5, 10


def app_runs(root: Path) -> dict[str, dict]:
    """Each run's application-layer UPF flows at 120 s, by its folder's name, its header's names stripped."""
    out = {}
    for a in H.ARCHIVES:
        with zipfile.ZipFile(root / a) as z:
            for n in z.namelist():
                if n.lower().endswith("/pfcp flows/120s-timeout/upf-120s-flow.csv"):
                    rows = list(csv.reader(io.TextIOWrapper(z.open(n), encoding="utf-8", errors="replace")))
                    out[n.split("/")[1]] = {"archive": a, "member": n, "head": [h.strip() for h in rows[0]], "rows": rows[1:]}
    return out


def attack_of(run: str) -> str:
    return next(k for k in ATTACK_COUNTER if k.lower() in run.lower())


def audit(run: dict, kind: str, name: str) -> dict:
    head = run["head"]
    rows = [r for r in run["rows"] if len(r) == len(head)]
    lab = [r[head.index("Label")].strip() for r in rows]
    s, d = (head.index(c) for c in ADDRESSES[kind])
    src, dst = [r[s].strip() for r in rows], [r[d].strip() for r in rows]
    pair = list(zip(src, dst, strict=True))
    out = {"flows": dict(Counter(lab)), "rows_dropped_malformed": len(run["rows"]) - len(rows)}
    for what, vals in (("source_addresses", src), ("destination_addresses", dst), ("address_pairs", pair)):
        per = {lv: {v for v, lb in zip(vals, lab, strict=True) if lb == lv} for lv in sorted(set(lab))}
        out[what] = {"distinct_per_label": {k: len(v) for k, v in per.items()},
                     "under_both_labels": len(set.intersection(*per.values())) if len(per) == 2 else 0}
    by_pair: dict = {}
    for p, lb in zip(pair, lab, strict=True):
        by_pair.setdefault(p, Counter())[lb] += 1
    out["share_whose_label_is_their_address_pairs_majority"] = sum(c.most_common(1)[0][1] for c in by_pair.values()) / len(lab)
    if kind == "app":
        col = ATTACK_COUNTER[attack_of(name)]
        j = head.index(col)
        nz = {lv: float(np.mean([float(r[j] or 0) > 0 for r, lb in zip(rows, lab, strict=True) if lb == lv]))
              for lv in sorted(set(lab))}
        out["attack_counter"] = {"column": col, "share_non_zero_per_label": nz}
    return out


def app_matrix(run: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """As the held-out-day check's `matrix`, on the stripped header."""
    return H.matrix({"head": run["head"], "rows": run["rows"]}, names)


def splits(R: dict, base: list[str], matrix, held: str):
    train = [n for n in R if n != held]
    Xs, ys = zip(*(matrix(R[n], base) for n in train), strict=True)
    names = [c for j, c in enumerate(base) if np.ptp(np.concatenate([x[:, j] for x in Xs])) > 0]
    cols = [base.index(c) for c in names]
    fx, fy, vx, vy = [], [], [], []
    for x, yy in zip(Xs, ys, strict=True):
        cut = int(round(0.8 * len(yy)))
        fx.append(x[:cut, cols])
        fy.append(yy[:cut])
        vx.append(x[cut:, cols])
        vy.append(yy[cut:])
    Xt, yt = matrix(R[held], base)
    return train, names, np.concatenate(fx), np.concatenate(fy), np.concatenate(vx), np.concatenate(vy), Xt[:, cols], yt


def spread(arms: list[dict], key: str) -> dict:
    vals = [a[key] for a in arms if a[key] is not None]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None


def supervised(Xf, yf, Xv, yv, Xt, yt, T, S) -> dict:
    """The held-out-day check's teacher (five seeds, CPU) and linear student, unchanged."""
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd == 0] = 1.0

    def z(a):
        return np.clip((a - mu) / sd, -H.CLIP, H.CLIP)

    teacher = []
    for seed in H.SEEDS:
        m = T._booster(SimpleNamespace(n_estimators=300, max_depth=8, seed=seed))
        m.set_params(device="cpu")
        m.fit(Xf, yf)
        teacher.append({"seed": seed, **H.arm(m.predict_proba(Xv)[:, 1], yv, m.predict_proba(Xt)[:, 1], yt)})
    onehot = np.eye(2)
    fits = {l2: S.fit(z(Xf), onehot[yf], l2) for l2 in S.L2_GRID}
    loss = {l2: S.log_loss(S.softmax(z(Xv) @ W.T + b), onehot[yv]) for l2, (W, b) in fits.items()}
    l2 = min(loss, key=lambda k: (loss[k], k))
    W, b = fits[l2]
    lv, lt = z(Xv) @ W.T + b, z(Xt) @ W.T + b
    from scipy.optimize import minimize_scalar

    temperature = float(np.exp(minimize_scalar(lambda u: S.log_loss(S.softmax(lv / np.exp(u)), onehot[yv]),
                                               bounds=(-4, 4), method="bounded").x))
    lv, lt = lv / temperature, lt / temperature
    return {"teacher": {"per_seed": teacher, **{k: spread(teacher, k) for k in ("auc", "detection_rate", "false_positive_rate")}},
            "student": {"l2": l2, "temperature": temperature, **H.arm(lv[:, 1] - lv[:, 0], yv, lt[:, 1] - lt[:, 0], yt)}}


def hbos(fit: np.ndarray, x: np.ndarray) -> np.ndarray:
    """HBOS as pre-registered: 10 equal-width bins per input over the fitted range, bin share (c + 1) / (n + 10), a value
    outside the range 1 / (n + 10), the score minus the sum of the log shares."""
    n = len(fit)
    score = np.zeros(len(x))
    for j in range(fit.shape[1]):
        lo, hi = float(fit[:, j].min()), float(fit[:, j].max())
        v = x[:, j]
        p = np.full(len(v), 1.0 / (n + HBOS_BINS))
        if hi > lo:
            edges = np.linspace(lo, hi, HBOS_BINS + 1)
            counts = np.histogram(fit[:, j], bins=edges)[0]
            inside = (v >= lo) & (v <= hi)
            idx = np.clip(np.searchsorted(edges, v[inside], side="right") - 1, 0, HBOS_BINS - 1)
            p[inside] = (counts[idx] + 1.0) / (n + HBOS_BINS)
        else:
            p[v == lo] = (n + 1.0) / (n + HBOS_BINS)
        score -= np.log(p)
    return score


def normal_only(Xf, yf, Xv, yv, Xt, yt) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    normal = Xf[yf == 0]
    mu, sd = normal.mean(0), normal.std(0)
    sd[sd == 0] = 1.0

    def z(a):
        return np.clip((a - mu) / sd, -H.CLIP, H.CLIP)

    zn, zv, zt = z(normal), z(Xv), z(Xt)
    forest = []
    for seed in IFOREST_SEEDS:
        m = IsolationForest(n_estimators=100, max_samples="auto", random_state=seed).fit(zn)
        forest.append({"seed": seed, **H.arm(-m.score_samples(zv), yv, -m.score_samples(zt), yt)})
    nn = NearestNeighbors(n_neighbors=KNN_K).fit(zn)
    knn_v, knn_t = nn.kneighbors(zv)[0].mean(1), nn.kneighbors(zt)[0].mean(1)
    return {"fitted_normal_flows": int(len(normal)),
            "iforest": {"per_seed": forest, **{k: spread(forest, k) for k in ("auc", "detection_rate", "false_positive_rate")}},
            "knn": H.arm(knn_v, yv, knn_t, yt),
            "hbos": H.arm(hbos(zn, zv), yv, hbos(zn, zt), yt)}


def main() -> int:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    root = Path(D.data_root()) / "raw" / "5gc-pfcp"
    cic, app = H.runs(root), app_runs(root)
    if len(cic) != 4 or sorted(cic) != sorted(app):
        raise SystemExit(f"{len(cic)} CICFlowMeter and {len(app)} application-layer runs found, not the same four: run nothing")

    # Arm A first: the audit, which changes no arm.
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "deposit": {"doi": "10.21227/e0gj-ev03", "zenodo": "7888347", "licence": "CC-BY-4.0",
                          "archives_sha256": {a: hashlib.sha256((root / a).read_bytes()).hexdigest() for a in H.ARCHIVES}},
              "arm_a_audit": {n: {"attack": attack_of(n), "cic": audit(cic[n], "cic", n), "app": audit(app[n], "app", n)}
                              for n in sorted(cic)}}
    print(json.dumps({n: (v["cic"]["share_whose_label_is_their_address_pairs_majority"],
                          v["app"]["share_whose_label_is_their_address_pairs_majority"],
                          v["app"]["attack_counter"]["share_non_zero_per_label"]) for n, v in record["arm_a_audit"].items()}),
          flush=True)

    T = H._module("telecom_nidd", REPO / "scripts" / "train" / "telecom_nidd.py")
    S = H._module("telecom_nidd_student", REPO / "scripts" / "train" / "telecom_nidd_student.py")
    heads = {"cic": [r["head"] for r in cic.values()], "app": [r["head"] for r in app.values()]}
    shared = {k: set.intersection(*(set(h) for h in v)) for k, v in heads.items()}
    base = {"cic": [c for c in heads["cic"][0] if c in shared["cic"] and c not in H.IDENTIFIERS and c != "Label"],
            "app": [c for c in heads["app"][0] if c in shared["app"] and c not in APP_IDENTIFIERS and c != "Label"]}
    readers = {"cic": (cic, H.matrix), "app": (app, app_matrix)}
    record["inputs_named"] = {k: len(v) for k, v in base.items()}
    record["arm_p_app_layer"], record["arm_o_normal_only"] = {}, {"cic": {}, "app": {}}
    for held in sorted(cic):
        for kind in ("app", "cic"):
            R, matrix = readers[kind]
            train, names, Xf, yf, Xv, yv, Xt, yt = splits(R, base[kind], matrix, held)
            if not (yt == 1).any() or not (yt == 0).any():
                result = {"scorable": False}
                if kind == "app":
                    record["arm_p_app_layer"][held] = result
                record["arm_o_normal_only"][kind][held] = result
                continue
            info = {"scorable": True, "train_days": train, "inputs": len(names), "fit_rows": int(len(yf)),
                    "validation_rows": int(len(yv)), "held_out_rows": int(len(yt)), "held_out_malicious": int(yt.sum())}
            if kind == "app":
                record["arm_p_app_layer"][held] = {**info, "input_names": names, **supervised(Xf, yf, Xv, yv, Xt, yt, T, S)}
                p = record["arm_p_app_layer"][held]
                print(f"P {held[:40]}: student {p['student']['auc']:.3f}, teacher {p['teacher']['auc']['mean']:.3f}", flush=True)
            o = {**info, **normal_only(Xf, yf, Xv, yv, Xt, yt)}
            record["arm_o_normal_only"][kind][held] = o
            print(f"O {kind} {held[:40]}: iforest {o['iforest']['auc']['mean']:.3f}, knn {o['knn']['auc']:.3f}, "
                  f"hbos {o['hbos']['auc']:.3f}", flush=True)
    import sklearn

    record["tools"] = {"scikit-learn": sklearn.__version__, "numpy": np.__version__}
    out = REPO / "docs" / "measurements" / f"telecom-pfcp-audit-app-layer-normal-only-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
