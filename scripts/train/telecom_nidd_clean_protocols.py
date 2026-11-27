"""5G-NIDD's models under leakage-aware protocols, as pre-registered (Round 86, batch 9).

`references/telecom-nidd-clean-protocols-preregistration.md` fixes the protocols (P0 shipped, P1 cross-station, P2 a time
split in each capture file, P3 LEAF's Seq rule), the input sets (the six columns and TTL removed), the single-feature
check and the arms. Every recipe is the shipped one, imported, not re-implemented: the teacher's `_booster`, the
student's `fit`, the federation's `fit_from` and `Channel`. Each arm writes its part of the record as it ends.

  --check-only     the single-feature check (thresholds only, no model)
  --arm NAME       student | federation | teacher | mlp | crosstestbed
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

PREREG = REPO / "references" / "telecom-nidd-clean-protocols-preregistration.md"
PREREG_SHA256 = "7d4feebfdccb8c2626e614445999ef48ac23dff2c8071f20b2fb1140d5750aed"  # committed in 8e91b10
# One record for every arm, named by the day the re-run began (the single-feature check, 2026-11-27); each arm adds
# its part, so an arm run on a later day writes to the same file.
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-clean-protocols-2026-11-27.json"
SIX = ("sHops", "dHops", "sVid", "dVid", "SrcTCPBase", "DstTCPBase")
TTL = ("sTtl", "dTtl")
FLAG_F1 = 0.99
SEEDS = (0, 1, 2, 3, 4)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _module("telecom_nidd_federated", REPO / "scripts" / "train" / "telecom_nidd_federated.py")
S = F.S
P = _module("nidd_protocols", REPO / "scripts" / "train" / "nidd_protocols.py")


def load():
    x, y, names, classes = D.load_5g_nidd(with_identifiers=True, with_file_label=True)
    names = list(names)
    feat = list(D.NIDD_FEATURES)
    cols = [names.index(n) for n in feat]
    seq = x[:, names.index("Seq")].astype(np.float64)
    fid = P.file_ids(x[:, names.index("IdleTime")])
    station = (np.arange(len(y)) >= D.NIDD_SECOND_FILE_STARTS_AT_CLEANED).astype(int)
    return x[:, cols].astype(np.float32), y.astype(np.int64), feat, list(classes), seq, fid, station


def protocols(y, seq, fid, station) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {"P0": P.p0(y), "P1_train_1_test_2": P.p1(y, station, 0), "P1_train_2_test_1": P.p1(y, station, 1),
            "P2": P.p2(y, fid), "P3": P.p3(y, seq)}


def input_sets(feat: list[str]) -> dict[str, list[int]]:
    numeric = [i for i, n in enumerate(feat) if n not in D.NIDD_CATEGORICAL]
    drop = lambda idx, out: [i for i in idx if feat[i] not in out]  # noqa: E731
    return {"T46": list(range(len(feat))), "T40": drop(range(len(feat)), SIX), "T38": drop(range(len(feat)), SIX + TTL),
            "F36": numeric, "F30": drop(numeric, SIX), "F28": drop(numeric, SIX + TTL)}


def class_f1(y_true, pred, c) -> float:
    tp = float(np.sum((pred == c) & (y_true == c)))
    denom = float(np.sum(pred == c) + np.sum(y_true == c))
    return 2 * tp / denom if denom else 0.0


def scores(y_true, pred, k, station_rows=None) -> dict:
    """Macro-F1 over the classes present in the part (an absent class is left out and named), weighted F1, accuracy;
    with station masks, each station's macro-F1 and their mean (FLamby's metric)."""
    present = [int(c) for c in np.unique(y_true)]
    f1 = np.array([class_f1(y_true, pred, c) for c in present])
    support = np.array([np.sum(y_true == c) for c in present], dtype=np.float64)
    out = {"macro_f1": round(float(f1.mean()), 6), "weighted_f1": round(float(np.dot(f1, support) / support.sum()), 6),
           "accuracy": round(float(np.mean(pred == y_true)), 6), "classes_absent": sorted(set(range(k)) - set(present))}
    if station_rows is not None:
        per = {s: scores(y_true[m], pred[m], k)["macro_f1"] for s, m in station_rows.items() if m.any()}
        out["station_macro_f1"] = per
        out["flamby_metric"] = round(float(np.mean(list(per.values()))), 6)
    return out


def best_threshold(x: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    """One threshold and direction maximising F1 of `sign * x >= sign * thr` on these rows; cuts only where the sorted
    value changes, so tied values fall on one side."""
    best = (-1.0, 0.0, 1)
    for sign in (1, -1):
        v = sign * x.astype(np.float64)
        order = np.argsort(v, kind="stable")
        vs, ts = v[order], target[order]
        starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])     # first index of each distinct value
        tp_from = np.cumsum(ts[::-1])[::-1]                          # positives at or above each index
        n_from = len(vs) - starts
        f1 = 2 * tp_from[starts] / (n_from + ts.sum())
        i = int(np.argmax(f1))
        if f1[i] > best[0]:
            best = (float(f1[i]), float(vs[starts[i]] * sign), sign)
    return best


def single_feature_check(X, y, feat, parts) -> dict:
    """Per protocol, input and class (one against the rest): one threshold fitted on the fit part, scored by F1 on the
    validation part; flagged at FLAG_F1. Every value of at least 0.5 is kept."""
    out = {}
    for pname, (fit, val, _) in parts.items():
        rows = {}
        for j, name in enumerate(feat):
            for c in np.unique(y[fit]):
                _, thr, sign = best_threshold(X[fit, j], y[fit] == c)
                pv, tv = sign * X[val, j] >= sign * thr, y[val] == c
                denom = float(pv.sum() + tv.sum())
                f1v = 2 * float(np.sum(pv & tv)) / denom if denom else 0.0
                if f1v >= 0.5:
                    rows.setdefault(name, {})[int(c)] = round(f1v, 4)
        out[pname] = {"flagged": {n: {c: v for c, v in d.items() if v >= FLAG_F1} for n, d in rows.items()
                                  if any(v >= FLAG_F1 for v in d.values())}, "validation_f1_at_least_0_5": rows}
    return out


def student_arm(X, y, k, cols, fit, val, test, station, weights: bool = False) -> dict:
    """The shipped student's procedure on these parts and inputs."""
    from scipy.optimize import minimize_scalar

    centre = X[np.ix_(fit, cols)].mean(axis=0).astype(np.float64)
    scale = X[np.ix_(fit, cols)].std(axis=0).astype(np.float64)
    scale[scale == 0] = 1.0
    z = lambda r: np.clip((X[np.ix_(r, cols)] - centre) / scale, -S.CLIP, S.CLIP)  # noqa: E731
    Zf, Zv, Zt = z(fit), z(val), z(test)
    onehot = np.eye(k)
    table, fits = [], {}
    for l2 in S.L2_GRID:
        W, b = S.fit(Zf, onehot[y[fit]], l2)
        fits[l2] = (W, b)
        table.append({"l2": l2, "validation_log_loss": round(S.log_loss(S.softmax(Zv @ W.T + b), onehot[y[val]]), 8)})
    best = min(table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
    W, b = fits[best]
    T = float(np.exp(minimize_scalar(lambda t: S.log_loss(S.softmax((Zv @ W.T + b) / np.exp(t)), onehot[y[val]]),
                                     bounds=(-4, 4), method="bounded").x))
    pred = (Zt @ W.T + b).argmax(1)
    st = {f"station_{s + 1}": station[test] == s for s in (0, 1)}
    out = {"penalty_table": table, "l2": best, "temperature": T, "test": scores(y[test], pred, k, st)}
    if weights:
        out["_weights"] = (W, b, centre, scale)
    return out


def teacher_arm(X, y, k, cols, fit, test, station, seed, device="cuda") -> dict:
    T = _module("telecom_nidd", REPO / "scripts" / "train" / "telecom_nidd.py")
    model = T._booster(SimpleNamespace(n_estimators=300, max_depth=8, seed=seed))
    model.set_params(device=device)
    model.fit(X[np.ix_(fit, cols)], y[fit])
    model.set_params(device="cpu")
    pred = model.predict_proba(X[np.ix_(test, cols)]).argmax(1)
    st = {f"station_{s + 1}": station[test] == s for s in (0, 1)}
    return {"seed": seed, "device": device, "test": scores(y[test], pred, k, st)}


def federation_arm(X, y, k, cols, fit, val, test, station) -> dict:
    """The shipped two-station masked FedAvg on P2's parts: each station its own fit and validation rows, the
    standardisation from each station's count, means and population deviations (checked against the union's), 40 rounds
    of 20 local L-BFGS-B iterations weighted by fit rows, the penalty and temperature from masked validation sums."""
    from scipy.optimize import minimize_scalar

    st_fit = [fit[station[fit] == s] for s in (0, 1)]
    st_val = [val[station[val] == s] for s in (0, 1)]
    n = np.array([len(r) for r in st_fit], dtype=np.float64)
    mu = [X[np.ix_(r, cols)].astype(np.float64).mean(axis=0) for r in st_fit]
    var = [X[np.ix_(r, cols)].astype(np.float64).var(axis=0) for r in st_fit]
    centre = (n[0] * mu[0] + n[1] * mu[1]) / n.sum()
    scale = np.sqrt((n[0] * (var[0] + (mu[0] - centre) ** 2) + n[1] * (var[1] + (mu[1] - centre) ** 2)) / n.sum())
    scale[scale == 0] = 1.0
    pooled = X[np.ix_(fit, cols)].astype(np.float64)
    ref_scale = pooled.std(axis=0)
    ref_scale[ref_scale == 0] = 1.0
    dev = float(max(np.max(np.abs(centre - pooled.mean(axis=0)) / ref_scale), np.max(np.abs(scale / ref_scale - 1))))
    if dev > F.STANDARDISATION_TOLERANCE:
        raise SystemExit(f"the stations' combined standardisation is {dev:.2e} from the union's")
    z = lambda r: np.clip((X[np.ix_(r, cols)] - centre) / scale, -S.CLIP, S.CLIP)  # noqa: E731
    Zf, Zv, Zt = [z(r) for r in st_fit], [z(r) for r in st_val], z(test)
    Yf, Yv = [y[r] for r in st_fit], [y[r] for r in st_val]
    onehot, d = np.eye(k), len(cols)
    w_fit = n / n.sum()
    n_val = np.array([len(r) for r in st_val], dtype=np.float64)
    w_val = n_val / n_val.sum()
    channel = F.Channel()
    unpack = lambda th: (th[: k * d].reshape(k, d), th[k * d:])  # noqa: E731

    def val_loss(W, b, T=1.0):
        return float(channel.sum([np.array([w_val[s] * S.log_loss(S.softmax((Zv[s] @ W.T + b) / T), onehot[Yv[s]])])
                                  for s in (0, 1)])[0])

    runs = {}
    for l2 in S.L2_GRID:
        theta = np.zeros(k * d + k)
        for _ in range(F.ROUNDS):
            local = [F.fit_from(Zf[s], onehot[Yf[s]], l2, theta, F.LOCAL_ITERATIONS) for s in (0, 1)]
            theta = channel.sum([w_fit[s] * local[s] for s in (0, 1)])
        runs[l2] = (theta, round(val_loss(*unpack(theta)), 8))
    table = [{"l2": l2, "validation_log_loss": runs[l2][1]} for l2 in S.L2_GRID]
    best = min(table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
    W, b = unpack(runs[best][0])
    T = float(np.exp(minimize_scalar(lambda t: val_loss(W, b, float(np.exp(t))), bounds=(-4, 4), method="bounded").x))
    pred = (Zt @ W.T + b).argmax(1)
    st = {f"station_{s + 1}": station[test] == s for s in (0, 1)}
    return {"penalty_table": table, "l2": best, "temperature": T, "masked_messages": channel.messages,
            "standardisation_max_relative_difference": dev, "test": scores(y[test], pred, k, st)}


def mlp_arm(X, y, k, cols, fit, val, test, station, seed, device) -> dict:
    """The square MLP: inputs -> 64 -> x^2 -> k; Adam 1e-3, batch 512, unweighted cross-entropy, 30 epochs, the epoch
    reported chosen by validation macro-F1; standardised and clipped as the student."""
    import torch

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    centre = X[np.ix_(fit, cols)].mean(axis=0).astype(np.float64)
    scale = X[np.ix_(fit, cols)].std(axis=0).astype(np.float64)
    scale[scale == 0] = 1.0
    z = lambda r: torch.tensor(np.clip((X[np.ix_(r, cols)] - centre) / scale, -S.CLIP, S.CLIP), dtype=torch.float32)  # noqa: E731
    dev = torch.device(device)
    Zf, Zv, Zt = z(fit).to(dev), z(val).to(dev), z(test).to(dev)
    yf = torch.tensor(y[fit], dtype=torch.long, device=dev)

    class Square(torch.nn.Module):
        def forward(self, a):
            return a * a

    net = torch.nn.Sequential(torch.nn.Linear(len(cols), 64), Square(), torch.nn.Linear(64, k)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best = (-1.0, 0, None)
    for e in range(30):
        net.train()
        perm = torch.randperm(len(fit), generator=g).to(dev)
        for i in range(0, len(fit), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Zf[idx]), yf[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Zv).argmax(1).cpu().numpy()
        v = scores(y[val], pv, k)["macro_f1"]
        if v > best[0]:
            best = (v, e + 1, {key: t.clone() for key, t in net.state_dict().items()})
    net.load_state_dict(best[2])
    with torch.no_grad():
        pred = net(Zt).argmax(1).cpu().numpy()
    st = {f"station_{s + 1}": station[test] == s for s in (0, 1)}
    return {"seed": seed, "device": device, "epoch": best[1], "validation_macro_f1": round(best[0], 6),
            "test": scores(y[test], pred, k, st)}


MAPPED = ("Dur", "SrcPkts", "DstPkts", "TotPkts", "SrcBytes", "DstBytes", "TotBytes", "Rate", "SrcRate", "DstRate",
          "Load", "SrcLoad", "DstLoad", "Proto")
NETSLAB_CLASS = {"benign": "Benign", "syn": "SYNFlood", "http_flood": "HTTPFlood", "udp": "UDPFlood", "icmp": "ICMPFlood",
                 "slowloris": "SlowrateDoS", "portscan_udp": "UDPScan", "portscan_tcp": "TCP scan"}


def netslab_inputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """NetsLab-5GORAN-IDD's Zeek connections as the mapped Argus inputs (the pre-registration's map), and their
    attack_type labels as given."""
    import pandas as pd

    d = pd.read_csv(path, usecols=["duration", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "proto",
                                   "attack_type"], low_memory=False)
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(np.float64)  # noqa: E731
    dur, sp, dp, sb, db = num("duration"), num("src_pkts"), num("dst_pkts"), num("src_ip_bytes"), num("dst_ip_bytes")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = lambda n: np.where((dur > 0) & (n > 1), (n - 1) / dur, 0.0)  # noqa: E731
        load = lambda b, n: np.where((dur > 0) & (n > 1), 8 * b * (n - 1) / (n * dur), 0.0)  # noqa: E731
        sl, dl = load(sb, sp), load(db, dp)
        cols = {"Dur": dur, "SrcPkts": sp, "DstPkts": dp, "TotPkts": sp + dp, "SrcBytes": sb, "DstBytes": db,
                "TotBytes": sb + db, "Rate": rate(sp + dp), "SrcRate": rate(sp), "DstRate": rate(dp),
                "Load": sl + dl, "SrcLoad": sl, "DstLoad": dl}
    vocab = {v: float(i) for i, v in enumerate(D.NIDD_CATEGORICAL["Proto"])}
    cols["Proto"] = d["proto"].astype(str).str.lower().map(vocab).fillna(D.NIDD_MISSING).to_numpy(np.float64)
    return np.column_stack([cols[n] for n in MAPPED]).astype(np.float32), d["attack_type"].astype(str).to_numpy()


def crosstestbed_arm(X, y, k, classes, feat, fit, val, path: Path, device: str) -> dict:
    """A teacher and a student on the mapped inputs, trained on 5G-NIDD's P2 parts, scored on NetsLab without
    retraining: on the shared classes (the two TCP scans merged at scoring) and as Benign against any attack."""
    Xn, lab = netslab_inputs(path)
    tcols = [feat.index(n) for n in MAPPED]
    scols = [feat.index(n) for n in MAPPED if n not in D.NIDD_CATEGORICAL]
    tcp_scan = {classes.index("SYNScan"), classes.index("TCPConnectScan")}
    group = lambda c: "TCP scan" if c in tcp_scan else classes[c]  # noqa: E731
    shared = np.array([a in NETSLAB_CLASS for a in lab])
    truth = np.array([NETSLAB_CLASS[a] for a in lab[shared]])
    benign = classes.index("Benign")

    def score(pred: np.ndarray) -> dict:
        got = np.array([group(int(c)) for c in pred[shared]])
        names = sorted(set(truth))
        f1 = {}
        for nm in names:
            tp = float(np.sum((got == nm) & (truth == nm)))
            den = float(np.sum(got == nm) + np.sum(truth == nm))
            f1[nm] = round(2 * tp / den if den else 0.0, 6)
        att_t, att_p = lab != "benign", pred != benign
        tp = float(np.sum(att_t & att_p))
        den = float(att_t.sum() + att_p.sum())
        return {"shared_rows": int(shared.sum()), "per_class_f1": f1, "macro_f1": round(float(np.mean(list(f1.values()))), 6),
                "binary_attack_f1": round(2 * tp / den if den else 0.0, 6),
                "binary_accuracy": round(float(np.mean(att_t == att_p)), 6)}

    T = _module("telecom_nidd", REPO / "scripts" / "train" / "telecom_nidd.py")
    out = {}
    for seed in SEEDS:
        model = T._booster(SimpleNamespace(n_estimators=300, max_depth=8, seed=seed))
        model.set_params(device=device)
        model.fit(X[np.ix_(fit, tcols)], y[fit])
        model.set_params(device="cpu")
        out[f"teacher/seed{seed}"] = score(model.predict_proba(Xn).argmax(1))
    st = student_arm(X, y, k, scols, fit, val, val, np.zeros(len(y), int), weights=True)
    W, b, centre, scale = st["_weights"]
    Xs = np.column_stack([Xn[:, MAPPED.index(feat[c])] for c in scols])
    out["student"] = score((np.clip((Xs - centre) / scale, -S.CLIP, S.CLIP) @ W.T + b).argmax(1))
    out["student"]["l2"] = st["l2"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--arm", choices=("student", "federation", "teacher", "mlp", "crosstestbed"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    X, y, feat, classes, seq, fid, station = load()
    k, parts, sets = len(classes), protocols(y, seq, fid, station), input_sets(feat)
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256}, "classes": classes,
        "rows": int(len(y)), "parts": {p: [int(len(a)) for a in v] for p, v in parts.items()}}
    if args.check_only:
        record["single_feature_check"] = single_feature_check(X, y, feat, parts)
    elif args.arm == "student":
        record["student"] = {f"{s}/{p}": student_arm(X, y, k, sets[s], *parts[p], station)
                             for s in ("F36", "F30", "F28") for p in parts}
    elif args.arm == "teacher":
        record["teacher"] = {f"{s}/{p}/seed{seed}": teacher_arm(X, y, k, sets[s], parts[p][0], parts[p][2], station, seed,
                                                                args.device)
                             for s in ("T46", "T40", "T38") for p in parts for seed in SEEDS}
    elif args.arm == "federation":
        record["federation"] = {s: federation_arm(X, y, k, sets[s], *parts["P2"], station) for s in ("F36", "F30", "F28")}
    elif args.arm == "mlp":
        record["mlp"] = {f"F30/{p}/seed{seed}": mlp_arm(X, y, k, sets["F30"], *parts[p], station, seed, args.device)
                         for p in ("P1_train_1_test_2", "P1_train_2_test_1", "P2") for seed in SEEDS}
    elif args.arm == "crosstestbed":
        nl = Path(D.data_root()) / "raw" / "netslab-5goran-idd" / "Network_Dataset.csv"
        record["crosstestbed"] = crosstestbed_arm(X, y, k, classes, feat, parts["P2"][0], parts["P2"][1], nl, args.device)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
