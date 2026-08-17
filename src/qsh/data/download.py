"""Fetch the datasets the pipeline needs, and report what is already present.

``make data`` has invoked ``python -m qsh.data.download`` since the first commit while this module
did not exist, so the target has always failed with ``No module named qsh.data.download``.

Scope is deliberately narrow. The repository bundles small curated samples under ``data/`` so tests
and the demo run offline, and those need no download. What this handles is:

  * MedMNIST, which the ``medmnist`` package fetches on first use;
  * reporting the full-scale corpora used by ``scripts/train/*`` and printing the exact command for
    any that is missing, rather than downloading tens of gigabytes as a side effect of ``make data``.

Nothing here silently substitutes a smaller dataset for a missing one: a training metric measured on
a 4,000-row demo sample is not the metric the model claims, so a missing corpus is reported as
missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _bundled() -> list[tuple[str, Path]]:
    d = REPO / "data"
    return [
        ("finance / PaySim sample", d / "finance" / "paysim_sample.tsv"),
        ("genomics / PGS000001 weights", d / "genomics" / "pgs000001_breast_cancer.tsv"),
        ("genomics / 1000G dosages (400)", d / "genomics" / "g1k_dosages.tsv"),
        ("genomics / 1000G dosages (full)", d / "genomics" / "g1k_dosages_full.tsv"),
        ("genomics / 1000G ancestry panel", d / "genomics" / "g1k_panel.tsv"),
        ("satellite / SMAP-MSL sample", d / "satellite" / "smap_msl_telemetry.tsv"),
    ]


def _full_scale():
    """Full corpora used by scripts/train/*, with the command that fetches each."""
    from ..models import datasets as ds

    return [
        ("finance / IBM AMLworld HI-Small", ds.raw("amlworld", "HI-Small_Trans.csv"), ds.AMLWORLD_FETCH),
        ("satellite / ESA-ADB Mission1", ds.raw("esa", "ESA-Mission1"), ds.ESA_FETCH),
        ("telecom / CIC-IDS2017 (Engelen-corrected)", ds.raw("cicids2017-fixed"), ds.CICIDS_FETCH),
        # 44.4 MiB compressed, so unlike its neighbours this one is genuinely cheap to fetch. It is
        # still listed rather than downloaded: `make data` reports and prints commands, and one
        # corpus quietly breaking that rule is how the rule stops being true.
        #
        # The FIELDS-PRESERVED export, not Combined.csv: same rows, 113 columns against 52, and one
        # of the extra columns (`IdleTime`) is worth 22 accuracy points. Round 50 changed this.
        ("telecom / 5G-NIDD fields-preserved",
         ds.raw("5g-nidd", "fields_preserved", "BTS1_BTS2_fields_preserved.csv"), ds.NIDD_FETCH),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qsh.data.download")
    ap.add_argument("--medmnist", action="store_true",
                    help="also download MedMNIST (BreastMNIST + PneumoniaMNIST)")
    args = ap.parse_args(argv)

    print("bundled samples (committed; no download needed)")
    missing_bundled = 0
    for label, path in _bundled():
        ok = path.exists()
        missing_bundled += 0 if ok else 1
        size = f"{path.stat().st_size/1e3:.0f} kB" if ok else "-"
        print(f"  [{'ok' if ok else '--'}] {label:38s} {size:>10s}  {path.relative_to(REPO)}")

    print("\nfull-scale corpora (training only; not committed)")
    missing = []
    for label, path, how in _full_scale():
        ok = path.exists()
        print(f"  [{'ok' if ok else '--'}] {label:38s} {path}")
        if not ok:
            missing.append((label, how))

    if args.medmnist:
        print("\nMedMNIST")
        try:
            from ..models.datasets import breastmnist

            imgs, labels = breastmnist("train")
            print(f"  [ok] BreastMNIST train: {imgs.shape}, {len(labels)} labels")
        except Exception as exc:
            print(f"  [--] BreastMNIST unavailable: {type(exc).__name__}: {exc}")
            return 1

    if missing:
        print("\nTo fetch the missing corpora (set QSH_DATA_ROOT to a large volume first):")
        for label, how in missing:
            print(f"\n  # {label}\n  {how}")
    else:
        print("\nAll full-scale corpora present.")

    # A missing full-scale corpus is informational: the demo and the tests do not need it.
    return 1 if missing_bundled else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
