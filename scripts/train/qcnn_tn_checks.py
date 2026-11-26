#!/usr/bin/env python
"""Tensor-network checks of the medical, telecom and satellite QCNNs (Round 86, batch 6; the owner's choice of 2026-11-26).

As `references/qcnn-tensor-network-checks-preregistration.md` sets out, committed before this ran: for each shipped QCNN,
(1) the largest Schmidt rank of its state after every layer, over its test rows and every cut of the wire order, beside
the bound every state of that width meets, and (2) a matrix-product-state classifier on the same inputs and split as a
comparator. Nothing is retrained or replaced; the record feeds the cards' statement that the circuits are classically
simulable and that no quantum advantage is claimed.

    python scripts/train/qcnn_tn_checks.py            # writes the record and the page's summary
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

import qcnn_variational as Q  # noqa: E402

PREREG = REPO / "references" / "qcnn-tensor-network-checks-preregistration.md"
#: The text with its addendum of 2026-11-26 (the first run traced in single precision); 361e9d33... before it.
PREREG_SHA256 = "07647cb60c7d3b8430eec6fd516c75ca4ec27e2885f3ce1abebd6e623c81aebf"
RECORD = REPO / "docs" / "measurements" / "qcnn-tn-checks-2026-11-26.json"
SITE = REPO / "web" / "public" / "data" / "qcnn_tn_checks.json"

#: The pre-registration's table: key -> (artifact directory, the trainer's use case).
ARTIFACTS = {
    "medical": ("qcnn_medical", "medical"),
    "telecom": ("qcnn_telecom", "telecom"),
    "telecom_q7": ("qcnn_telecom_q7", "telecom"),
    "satellite_opssat": ("qcnn_satellite_opssat", "satellite_opssat"),
    "satellite_esa": ("qcnn_satellite_esa", "satellite_esa"),
}
SCHMIDT_ROWS = 1000
CHIS = (2, 4, 8)


def _finance_module():
    """The finance comparison's own functions: one Schmidt-rank routine and one MPS definition for every page."""
    spec = importlib.util.spec_from_file_location("finance_qcnn_comparison",
                                                  REPO / "scripts" / "train" / "finance_qcnn_comparison.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mps_multiclass(q_fit, y_fit, q_rows: list[np.ndarray], chi: int, k: int, seed: int = 0, epochs: int = 50):
    """The finance arm's MPS with the class index on the last core: k logits on the cross-entropy loss (section 2)."""
    import torch

    torch.manual_seed(seed)
    w = q_fit.shape[1]
    dims = [1] + [chi] * (w - 1) + [k]
    cores = torch.nn.ParameterList()
    for i in range(w):
        eye = torch.zeros(dims[i], 2, dims[i + 1])
        for s in range(2):
            eye[:, s, :] = torch.eye(dims[i], dims[i + 1]) / np.sqrt(2)
        cores.append(torch.nn.Parameter(eye + 0.01 * torch.randn(dims[i], 2, dims[i + 1])))
    bias = torch.nn.Parameter(torch.zeros(k))
    scale = torch.nn.Parameter(torch.ones(()))

    def forward(x):
        phi = torch.stack([torch.cos(np.pi * x / 2), torch.sin(np.pi * x / 2)], dim=2)
        v = torch.ones(len(x), 1)
        for i in range(w):
            v = torch.einsum("na,nab->nb", v, torch.einsum("ns,asb->nab", phi[:, i, :], cores[i]))
        return scale * v * (2 ** (w / 2)) + bias

    xt = torch.tensor(q_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.long)
    opt = torch.optim.Adam([*cores, bias, scale], lr=0.01)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 256):
            j = order[i:i + 256]
            opt.zero_grad()
            lossf(forward(xt[j]), yt[j]).backward()
            opt.step()
    with torch.no_grad():
        return [forward(torch.tensor(r, dtype=torch.float32)).numpy().astype(np.float64) for r in q_rows]


def _inputs(usecase: str, w: dict) -> dict:
    """The trainer's rows, split and encoding for one artifact, checked against the moments the artifact ships."""
    X, y, cols, _ = Q.load(usecase)
    order, cut, vcut = Q.split_for(usecase, y, 0)
    X, y = X[order], y[order].astype(int)
    # The moments in the loader's own dtype, as the trainer computed them: telecom's generator returns float32, whose
    # mean differs from a float64 one by about 1.5e-7, which a first run of this script read as "not these rows".
    mu, sd = X[:cut].mean(0), X[:cut].std(0)
    sd = np.where(sd == 0, 1.0, sd)
    X = X.astype(np.float64)
    gap = max(float(np.max(np.abs(mu - np.asarray(w["feature_mean"])))),
              float(np.max(np.abs(sd - np.asarray(w["feature_std"])))))
    if gap > 1e-9 * max(1.0, float(np.max(np.abs(mu))), float(np.max(sd))):
        raise SystemExit(f"{usecase}: the fit rows' moments differ from the artifact's by {gap:.3g}; not these rows")
    if list(cols) != list(w["feature_names"]):
        raise SystemExit(f"{usecase}: the loader's features {cols} are not the artifact's {w['feature_names']}")
    xs = np.tanh((X - np.asarray(w["feature_mean"])) / np.asarray(w["feature_std"]))
    nq = int(w["n_qubits"])
    if xs.shape[1] < nq:
        xs = np.hstack([xs, np.zeros((len(xs), nq - xs.shape[1]))])
    ref = np.sort(X[:cut], axis=0)
    q = np.stack([np.searchsorted(ref[:, j], X[:, j], side="right") / len(ref) for j in range(X.shape[1])], axis=1)
    return {"X": X, "y": y, "xin": float(w.get("encoding_scale", 1.0)) * xs, "q": q,
            "fit": np.arange(cut), "val": np.arange(cut, vcut), "te": np.arange(vcut, len(y)), "features": list(cols)}


def schmidt_check(key: str, w: dict, d: dict, F) -> dict:
    """Section 1: the largest rank per traced point, and the recorded figure recomputed where every test row is used."""
    import torch

    te = d["te"]
    rows = te if len(te) <= SCHMIDT_ROWS else np.random.default_rng(0).permutation(te)[:SCHMIDT_ROWS]
    nq = int(w["n_qubits"])
    readout_q = int(w["readout"].get("readout_qubits", 0) or 0)
    arch = w["circuit"]["name"]
    trace: list = []
    # In double precision (the addendum of 2026-11-26): single-precision round-off reads as rank at 1e-10.
    with torch.no_grad():
        z = Q.circuit(torch.tensor(d["xin"][rows], dtype=torch.float64), torch.tensor(w["theta"], dtype=torch.float64),
                      nq, readout_qubits=readout_q, entangle=bool(w["training"].get("entangle", True)),
                      layers=int(w["layers"]), arch=arch, layer_trace=trace, complex_dtype=torch.complex128)
    by_point, labels = {}, []
    for label, st in trace:
        name = label if label not in by_point else f"{label} ({sum(1 for x in labels if x.startswith(label)) + 1})"
        labels.append(name)
        by_point[name] = int(F.schmidt_ranks(st.numpy(), nq=nq).max())
    out = {"rows": int(len(rows)), "of_test_rows": int(len(te)), "wires": nq, "bound": 2 ** (nq // 2),
           "largest_rank": max(by_point.values()), "by_point": by_point}
    if len(rows) == len(te):
        scale, bias = np.asarray(w["scale"], dtype=np.float64), np.asarray(w["bias"], dtype=np.float64)
        logits = z.numpy().astype(np.float64) * scale + bias
        y = d["y"][rows]
        if readout_q:
            from sklearn.metrics import f1_score
            pred = logits.argmax(1)
            out["recomputed"] = {"test_accuracy": round(float((pred == y).mean()), 6),
                                 "test_macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 6)}
            out["recorded"] = {"test_accuracy": w["test_accuracy"], "test_macro_f1": w["test_macro_f1"]}
        else:
            from sklearn.metrics import f1_score
            pred = (logits > 0).astype(int)
            out["recomputed"] = {"test_accuracy": round(float((pred == y).mean()), 6),
                                 "test_minority_f1": round(float(f1_score(y, pred, zero_division=0)), 6)}
            out["recorded"] = {"test_accuracy": w["test_accuracy"], "test_minority_f1": w["test_minority_f1"]}
    return out


def mps_check(key: str, w: dict, d: dict, F) -> dict:
    """Section 2: the bond dimension chosen on validation, then the test figures."""
    from sklearn.metrics import average_precision_score as ap
    from sklearn.metrics import f1_score

    q, y, fit, val, te = d["q"], d["y"], d["fit"], d["val"], d["te"]
    multiclass = bool(int(w["readout"].get("readout_qubits", 0) or 0))
    tried, best = {}, None
    for chi in CHIS:
        if multiclass:
            sv, st = mps_multiclass(q[fit], y[fit], [q[val], q[te]], chi, k=int(w["readout"]["n_classes"]))
            v = float(f1_score(y[val], sv.argmax(1), average="macro", zero_division=0))
        else:
            sv, st = F.mps_classifier(q[fit], y[fit], [q[val], q[te]], chi)
            v = float(ap(y[val], sv))
        tried[str(chi)] = round(v, 6)
        print(f"  {key}: mps chi {chi}: validation {'macro F1' if multiclass else 'AP'} {v:.4f}", flush=True)
        if best is None or v > best[0]:
            best = (v, chi, sv, st)
    v, chi, sv, st = best
    out = {"chosen_bond_dimension": chi, "validation_by_bond_dimension": tried,
           "selected_on": "validation macro F1" if multiclass else "validation average precision"}
    yt = y[te]
    if multiclass:
        pred = st.argmax(1)
        out.update({"test_accuracy": round(float((pred == yt).mean()), 6),
                    "test_macro_f1": round(float(f1_score(yt, pred, average="macro", zero_division=0)), 6)})
        return out
    pred = (st > 0).astype(int)
    grid = np.unique(sv)
    f1v = [(float(f1_score(y[val], (sv >= t).astype(int), zero_division=0)), float(t)) for t in grid]
    t_best = max(f1v)[1]
    from sklearn.metrics import balanced_accuracy_score
    out.update({"test_accuracy": round(float((pred == yt).mean()), 6),
                "test_minority_f1": round(float(f1_score(yt, pred, zero_division=0)), 6),
                "test_average_precision": round(float(ap(yt, st)), 6),
                "validation_best_threshold": round(t_best, 6),
                "test_minority_f1_at_validation_best_threshold":
                    round(float(f1_score(yt, (st >= t_best).astype(int), zero_division=0)), 6),
                "test_balanced_accuracy_at_validation_best_threshold":
                    round(float(balanced_accuracy_score(yt, (st >= t_best).astype(int))), 6)})
    return out


def main() -> int:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} is not the committed pre-registration ({got})")
    F = _finance_module()
    t0 = time.time()
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256}, "artifacts": {}}
    for key, (folder, usecase) in ARTIFACTS.items():
        w = json.loads((REPO / "models" / folder / "weights.json").read_text())
        d = _inputs(usecase, w)
        print(f"{key}: {len(d['fit'])} fit, {len(d['val'])} validation, {len(d['te'])} test rows; {w['n_qubits']} wires", flush=True)
        sch = schmidt_check(key, w, d, F)
        print(f"  {key}: largest Schmidt rank {sch['largest_rank']} of the bound {sch['bound']} over {sch['rows']} rows", flush=True)
        mps = mps_check(key, w, d, F)
        multiclass = bool(int(w["readout"].get("readout_qubits", 0) or 0))
        record["artifacts"][key] = {
            "weights": f"models/{folder}/weights.json", "circuit": w["circuit"]["name"], "usecase": usecase,
            "features": d["features"], "split": {"fit": len(d["fit"]), "validation": len(d["val"]), "test": len(d["te"])},
            "schmidt": sch, "mps": mps,
            "qcnn": ({"test_accuracy": w["test_accuracy"], "test_macro_f1": w["test_macro_f1"]} if multiclass else
                     {"test_accuracy": w["test_accuracy"], "test_minority_f1": w["test_minority_f1"]}),
            "logistic_regression": ({"test_accuracy": w["classical_lr_accuracy"], "test_macro_f1": w["classical_lr_macro_f1"]}
                                    if multiclass else {"test_accuracy": w["classical_lr_accuracy"],
                                                        "test_minority_f1": w["classical_lr_minority_f1"]}),
            "headline": "test_macro_f1" if multiclass else "test_minority_f1",
        }
    record["wall_clock_s"] = round(time.time() - t0, 1)
    record["statement"] = ("Every circuit here acts on at most eight wires, so its state is at most 256 amplitudes and is "
                           "simulated exactly on a classical computer; QCNNs on classical data of this kind are classically "
                           "simulable (Bermejo et al., PRX Quantum 7, 020304, 2026). No quantum advantage is claimed.")
    # The first run's single-precision ranks stay in the record as invalid (the addendum), whoever reruns it.
    if RECORD.exists():
        prior = json.loads(RECORD.read_text()).get("invalid_single_precision_run")
        if prior:
            record["invalid_single_precision_run"] = prior
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    # The corpora behind the figures, credited from the registry's dataset records (an aggregate's credit object per
    # corpus, as tests/test_r86_every_shipped_data_file_is_licensed.py asks of figures computed from a corpus).
    from qsh.models import registry as R

    def credit(ds) -> dict:
        return {"dataset": ds.name, "download": ds.url, "citation": ds.attribution or ds.notes, "licence": ds.licence}

    site = {"preregistration": record["preregistration"]["file"], "record": str(RECORD.relative_to(REPO)),
            "statement": record["statement"], "artifacts": {},
            "attribution": {"medical": credit(R.DS_WDBC), "telecom": credit(R.DS_QKD_ATTACK_SYNTH),
                            "satellite_opssat": credit(R.DS_OPSSAT_AD), "satellite_esa": credit(R.DS_ESA_ADB)}}
    for key, r in record["artifacts"].items():
        h = r["headline"]
        site["artifacts"][key] = {
            "circuit": r["circuit"], "wires": r["schmidt"]["wires"], "largest_schmidt_rank": r["schmidt"]["largest_rank"],
            "bound": r["schmidt"]["bound"], "schmidt_rows": r["schmidt"]["rows"], "headline": h,
            "qcnn": r["qcnn"][h], "logistic_regression": r["logistic_regression"][h], "mps": r["mps"][h],
            "mps_bond_dimension": r["mps"]["chosen_bond_dimension"],
            "mps_average_precision": r["mps"].get("test_average_precision"),
            "mps_minority_f1_at_validation_best_threshold": r["mps"].get("test_minority_f1_at_validation_best_threshold"),
            "mps_balanced_accuracy_at_validation_best_threshold":
                r["mps"].get("test_balanced_accuracy_at_validation_best_threshold"),
        }
    SITE.write_text(json.dumps(site, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)} and {SITE.relative_to(REPO)} in {record['wall_clock_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
