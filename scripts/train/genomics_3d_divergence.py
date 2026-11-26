#!/usr/bin/env python3
"""The genomics page's encrypted predictor of 3D folding divergence, as pre-registered (Round 86, batch 6; option D1).

`references/genomics-3d-divergence-preregistration.md` fixed the method before any figure: per window, a ridge
regression of Akita's predicted divergence (from Gilbertson et al.'s data) on the window's ten carrier indicators, the
penalty chosen on validation R^2 and refitted on fit and validation together; the test R^2 and Spearman correlation
beside a comparator that knows only the individual's super-population; the twelve models as one block-diagonal linear
map under CKKS; and the page's criteria. Reads `scripts/data/akita_d1_inputs.py`'s output; writes the record and, when
the criteria hold, the page's payload.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/genomics_3d_divergence.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets  # noqa: E402

PREREG = REPO / "references" / "genomics-3d-divergence-preregistration.md"
PREREG_SHA256 = "2e93f4fab32d8f2852c7b5be59284460e3d5488533adc2cd491e0261222884c2"  # with its two addenda (empty cells; NA effects)
RECORD = REPO / "docs" / "measurements" / "genomics-3d-divergence-2026-11-26.json"
SITE = REPO / "web" / "public" / "data" / "genomics" / "folding_divergence.json"
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
R2_FLOOR, MIN_WINDOWS, CKKS_TOL = 0.5, 6, 1e-4
ZENODO = "https://doi.org/10.5281/zenodo.13900918"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Divergence: Akita predictions from Gilbertson EN, Brand CM, McArthur E, Rinker DC, Kuang S, Pollard KS, Capra JA, "
    "'Machine learning reveals the diversity of human 3D chromatin contact patterns', Mol Biol Evol 41(10):msae209 (2024), "
    "doi:10.1093/molbev/msae209; data: Gilbertson E, Zenodo record 13900918, doi:10.5281/zenodo.13900918, CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). Akita: Fudenberg G, Kelley DR, Pollard KS, 'Predicting 3D genome "
    "folding from DNA sequence with Akita', Nature Methods 17:1111-1117 (2020). Genotypes: The 1000 Genomes Project "
    "Consortium, 'A global reference for human genetic variation', Nature 526:68-74 (2015), doi:10.1038/nature15393, "
    "read through the Ensembl REST API.")


def ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge with an unpenalised intercept: centre, solve, restore."""
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return w, float(my - mx @ w)


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> int:
    from scipy.stats import spearmanr

    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} is not the committed pre-registration ({got})")
    src = datasets.artifacts("akita_d1_inputs.npz")
    meta = json.loads(src.with_suffix(".json").read_text())
    if meta["preregistration"]["sha256"] != PREREG_SHA256:
        raise SystemExit("the inputs were built under another text of the pre-registration")
    d = np.load(src)
    C, Y, part = d["C"].astype(np.float64), d["Y"], d["part"]
    sup = np.asarray(meta["super_population"])
    win_of = np.asarray([v["window"] for v in meta["variants"]])
    W = np.zeros((Y.shape[1], C.shape[1]))
    b = np.zeros(Y.shape[1])
    windows = []
    for w in range(Y.shape[1]):
        cols = np.flatnonzero(win_of == w)
        y = Y[:, w]
        ok = ~np.isnan(y)                                   # the addendum: no value, no figure
        fit, val, te = (part == 0) & ok, (part == 1) & ok, (part == 2) & ok
        scores = {}
        for a in ALPHAS:
            coef, icpt = ridge(C[fit][:, cols], y[fit], a)
            scores[a] = r2(y[val], C[val][:, cols] @ coef + icpt)
        alpha = max(ALPHAS, key=lambda a: (scores[a], -a))
        coef, icpt = ridge(C[fit | val][:, cols], y[fit | val], alpha)
        W[w, cols], b[w] = coef, icpt
        pred = C[te][:, cols] @ coef + icpt
        means = {g: float(y[(part == 0) & ok & (sup == g)].mean()) for g in sorted(set(sup))}
        base = np.asarray([means[g] for g in sup[te]])
        test_r2, base_r2 = r2(y[te], pred), r2(y[te], base)
        windows.append({
            "chr": meta["windows"][w]["chr"], "start": meta["windows"][w]["start"], "variants": int(len(cols)),
            "alpha": alpha, "validation_r2": {str(a): round(v, 6) for a, v in scores.items()},
            "test_rows": int(te.sum()), "test_r2": round(test_r2, 6),
            "test_spearman": round(float(spearmanr(y[te], pred).statistic), 6),
            "super_population_r2": round(base_r2, 6),
            "qualifies": bool(test_r2 >= R2_FLOOR and test_r2 > base_r2),
            "no_value": meta["windows"][w].get("no_value", []),
        })
        print(f"window {w} {windows[-1]['chr']}:{windows[-1]['start']}: alpha {alpha}, test R^2 {test_r2:.4f} "
              f"(super-population {base_r2:.4f}), Spearman {windows[-1]['test_spearman']:.4f}", flush=True)
    qualifying = [w for w, x in enumerate(windows) if x["qualifies"]]
    card = len(qualifying) >= MIN_WINDOWS

    # ---- CKKS (TenSEAL) on every test individual ----------------------------------------------------------------
    from qsh.fhe.backends import get_backend
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference

    be = get_backend("tenseal")
    model = LinearModel(W, b)
    test_idx = np.flatnonzero(part == 2)
    t0 = time.time()
    worst = 0.0
    for i in test_idx:
        res = encrypted_linear_inference(model, C[i], backend=be)
        worst = max(worst, res["max_abs_error"])
    ckks = {"backend": "tenseal", "individuals": int(len(test_idx)), "outputs": int(W.shape[0]),
            "max_abs_error": worst, "criterion_met": bool(worst <= CKKS_TOL), "seconds": round(time.time() - t0, 1)}
    print(f"CKKS: largest error {worst:.3g} over {len(test_idx)} individuals x {W.shape[0]} outputs", flush=True)

    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
        "inputs": meta["counts"], "declared_variants": meta["declared"],
        "windows": windows, "qualifying_windows": qualifying, "card_shown": card,
        "criteria": {"r2_floor": R2_FLOOR, "min_windows": MIN_WINDOWS, "ckks_tolerance": CKKS_TOL},
        "ckks_python": ckks,
        "genotype_caveat": ("The teacher was built from the 1000 Genomes 20190312 GRCh38 release; these carrier indicators "
                            "are Ensembl's phase 3 genotypes, which can differ for some individuals and variants. Not "
                            "measured here."),
        "wall_clock_s": round(time.time() - started, 1),
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(qualifying)} of {len(windows)} windows qualify; card {'shown' if card else 'NOT shown'}; wrote {RECORD.relative_to(REPO)}")
    if not card or not ckks["criterion_met"]:
        if SITE.exists():
            SITE.unlink()
        return 0

    # ---- the page's payload: the qualifying windows, their variants, the 30 shipped individuals ----------------------
    ids = meta["ids"]
    shipped = [ids.index(s) for s in meta["shipped"]]
    keep_cols = np.flatnonzero(np.isin(win_of, qualifying))
    Wq = W[np.ix_(qualifying, keep_cols)]
    site = {
        "what": ("Predicted 3D folding divergence (Akita, from Gilbertson et al.'s data), not a disease risk: one minus the "
                 "Spearman correlation between a person's predicted chromatin contact map for a 1 Mb window and the "
                 "inferred human-archaic ancestor's."),
        "preregistration": record["preregistration"]["file"], "record": str(RECORD.relative_to(REPO)),
        "windows": [{k: windows[w][k] for k in ("chr", "start", "test_r2", "test_spearman", "super_population_r2", "test_rows")}
                    for w in qualifying],
        "variants": [{k: meta["variants"][j][k] for k in ("rsid", "chr", "pos", "ref", "alt")} | {"window": qualifying.index(int(win_of[j]))}
                     for j in keep_cols],
        "W": [[round(float(v), 12) for v in row] for row in Wq], "b": [round(float(v), 12) for v in b[qualifying]],
        "sample_ids": meta["shipped"], "super_population": [meta["super_population"][i] for i in shipped],
        "carriers": [[int(C[i, j]) for j in keep_cols] for i in shipped],
        "teacher": [[None if np.isnan(Y[i, w]) else round(float(Y[i, w]), 10) for w in qualifying] for i in shipped],
        "genotype_caveat": record["genotype_caveat"],
        "licence": "CC-BY-4.0 (the divergence values); none (the 1000 Genomes genotypes)", "licence_url": CC_BY_4,
        "attribution": ATTRIBUTION,
        "modifications": (f"The divergence is one minus the table's Spearman correlation, for {len(qualifying)} windows and "
                          f"the {len(shipped)} individuals the genomics payload ships; their carrier indicators at "
                          f"{len(keep_cols)} variants were read from Ensembl; the weights were fitted here."),
    }
    SITE.write_text(json.dumps(site, separators=(",", ":")) + "\n")
    print(f"wrote {SITE.relative_to(REPO)}: {len(qualifying)} windows, {len(keep_cols)} variants, {len(shipped)} individuals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
