"""Fit the finance MLP's and GNN's calibration maps on their trained weights without retraining, and record them (Round 86).

The teacher's and the student's records carry a Platt map fitted on validation and scored on test, with its CORP terms;
the two networks did not. Their trainers now compute it in the run that trains them (`network_calibration` in
`scripts/train/finance_aml.py`), and this applies the same function to the weights already written, so a retrain is not
needed to read what their probabilities mean. A GPU retrain need not reproduce the weights bit for bit, and the decisions
and figures these weights already gave would then belong to different weights.

It writes nothing unless each network reproduces its own record first: the logits of its fixture rows, and its
minority-class F1 over the whole test window at argmax. A record that already has a map keeps it: the published `a` and
`b` are passed in and only the figures beside them are measured again (Round 86, after a refit moved them in the 8th digit).

Needs the AMLworld corpus (`qsh.models.datasets`, `$QSH_DATA_ROOT`).

    .venv/bin/python scripts/finance_network_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]

from finance_aml import (  # noqa: E402
    causal_account_features,
    load_sorted,
    model_inputs,
    network_calibration,
    temporal_windows,
)
from finance_aml_gnn import from_weights  # noqa: E402
from finance_aml_neighbours import neighbourhoods  # noqa: E402

from qsh.models import metrics, registry, train  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: The fixture logits are float32 network outputs; this is the tolerance the browser checks use for the same rows.
FIXTURE_TOL = 1e-4


def _check(key: str, logits_te: np.ndarray, y_te: np.ndarray, fixture_logits: np.ndarray, recomputed_fixture: np.ndarray) -> None:
    rec = registry.achieved(key)
    err = float(np.abs(recomputed_fixture - fixture_logits).max())
    f1 = metrics.minority_class_f1(y_te, logits_te.argmax(1))
    if err > FIXTURE_TOL or len(y_te) != rec["n_test"] or abs(f1 - rec["minority_class_f1"]) > 5e-7:
        raise SystemExit(f"{key}: fixture logits within {err:.2e}, {len(y_te)} test rows, minority-class F1 {f1:.6f}; the record says "
                         f"{rec['n_test']} rows and {rec['minority_class_f1']}")
    print(f"  {key}: reproduced its record, fixtures within {err:.2e}, F1 {f1:.6f} over {len(y_te):,} rows")


def _published_map(key: str):
    """The map the record already publishes, so a rewrite keeps the floats the cards print (a refit moves them in the 8th digit)."""
    from qsh.models import calibration as cal

    c = registry.achieved(key)["calibration"]
    if c.get("method") != "platt" or c.get("on_logit") is not True:
        raise SystemExit(f"{key}: unexpected map {c.get('method')}, on_logit {c.get('on_logit')}")
    return cal.Platt(a=float(c["a"]), b=float(c["b"]), n_calibration=int(c["n_calibration"]),
                     positives_calibration=int(c["positives_calibration"]), on_logit=True, logit_clip_eps=float(c["logit_clip_eps"]))


def _fitted_by(key: str, sha: str) -> str:
    first = registry.achieved(key)["calibration"]["fitted_by"].split("; every figure beside the map")[0]
    return (f"{first}; every figure beside the map rewritten by scripts/finance_network_calibration.py at git {sha} from the trained "
            "weights and the published map, without retraining")


def _record(key: str, calibration: dict) -> None:
    payload = dict(registry.achieved(key))
    payload["calibration"] = calibration
    train.record_metrics(key, payload)
    c = calibration
    print(f"  {key}: Platt a {c['a']:.6f}, b {c['b']:.6f}; test Brier {c['brier_raw']:.8f} -> {c['brier_calibrated']:.8f}, "
          f"CORP MCB {c['corp_raw']['mcb']:.8f} -> {c['corp_calibrated']['mcb']:.8f}, DSC {c['corp_raw']['dsc']:.8f}")


def main() -> int:
    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    n_fit, n_val = len(fit), len(val)
    n_all = n_fit + n_val
    (_, x_val, x_te), names, reach = model_inputs(df, fit, val, te, tr, causal_account_features(df))
    y_val, y_te = val["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)
    sha = train.git_sha()

    # The MLP: robust scaling with its clip, then the dense layers its weights file lists.
    w = json.loads((REPO / registry.ARTIFACT_DIRS["finance_aml.mlp_large"] / "weights.json").read_text())
    if w["features"] != names:
        raise SystemExit("the MLP's feature order is not model_inputs()'s")
    centre, scale, clip = np.asarray(w["scaling"]["center"]), np.asarray(w["scaling"]["scale"]), float(w["scaling"]["clip"])

    def mlp(x: np.ndarray) -> np.ndarray:
        h = np.clip((np.asarray(x, dtype=np.float64) - centre) / scale, -clip, clip)
        for layer in w["layers"]:
            h = h @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
            if layer["activation"] == "relu":
                h = np.maximum(h, 0.0)
        return h

    lt = mlp(x_te)
    _check("finance_aml.mlp_large", lt, y_te, np.asarray(w["fixtures"]["logits"]), mlp(np.asarray(w["fixtures"]["rows"])))
    _record("finance_aml.mlp_large", network_calibration(
        mlp(x_val), y_val, lt, y_te, model="MLP", platt=_published_map("finance_aml.mlp_large"),
        fitted_by=_fitted_by("finance_aml.mlp_large", sha)))

    # The GNN: its query, endpoint and slot scalings, then the network its weights file describes.
    g = json.loads((REPO / registry.ARTIFACT_DIRS["finance_aml.gnn_compact"] / "weights.json").read_text())
    if g["query"]["features"] != names:
        raise SystemExit("the GNN's feature order is not model_inputs()'s")
    nb = neighbourhoods(df, DEVICE, horizon_minutes=reach)
    model = from_weights(g).to(DEVICE)
    q_c, q_s, q_clip = (np.asarray(g["query"]["scaling"][k]) for k in ("center", "scale", "clip"))
    e_c, e_s = (torch.tensor(g["endpoints"]["scaling"][k], dtype=torch.float32) for k in ("center", "scale"))
    s_s = torch.tensor(g["slots"]["scaling"]["scale"], dtype=torch.float32)
    e_clip, s_clip = float(g["endpoints"]["scaling"]["clip"]), float(g["slots"]["scaling"]["clip"])

    def gnn(x: np.ndarray, ends: torch.Tensor, slots: torch.Tensor) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 262_144):
                xq = torch.tensor(np.clip((x[i:i + 262_144] - q_c) / q_s, -q_clip, q_clip), dtype=torch.float32, device=DEVICE)
                en = ((ends[i:i + 262_144] - e_c) / e_s).clamp(-e_clip, e_clip).to(DEVICE)
                sl = (slots[i:i + 262_144] / s_s).clamp(-s_clip, s_clip).to(DEVICE)
                out.append(model(xq, en, sl).cpu().numpy())
        return np.vstack(out)

    gt = gnn(x_te, nb["endpoints"][n_all:], nb["slots"][n_all:])
    fx = (np.asarray(g["fixtures"]["x"]), torch.tensor(g["fixtures"]["endpoints"], dtype=torch.float32),
          torch.tensor(g["fixtures"]["slots"], dtype=torch.float32))
    _check("finance_aml.gnn_compact", gt, y_te, np.asarray(g["fixtures"]["logits"]), gnn(*fx))
    _record("finance_aml.gnn_compact", network_calibration(
        gnn(x_val, nb["endpoints"][n_fit:n_all], nb["slots"][n_fit:n_all]), y_val, gt, y_te, model="GNN",
        platt=_published_map("finance_aml.gnn_compact"), fitted_by=_fitted_by("finance_aml.gnn_compact", sha)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
