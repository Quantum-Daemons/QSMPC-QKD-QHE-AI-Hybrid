#!/usr/bin/env python3
"""The repository's DP-SGD accountant against Google's dp_accounting, at the murmur study's own settings.

`scripts/train/medical_murmur_dp.py` accounts DP-SGD itself: Renyi DP of the Poisson-sampled Gaussian over integer
orders 2..256 (Mironov, Talwar and Zhang 2019) and Canonne, Kamath and Steinke's conversion to (epsilon, delta). This
records what dp_accounting (Apache-2.0) gives for the same mechanism: its RDP accountant, which should agree, and its
privacy-loss-distribution (PLD) accountant, which is tighter (Gopi, Lee and Wutschitz, NeurIPS 2021). The settings are
the murmur record's search route: expected batch 64 of 511 fitting records, 239 steps, delta 1e-5, at the two noise
multipliers it trained with (docs/measurements/murmur-dp-2026-11-14.json).

dp_accounting pins an old typing-extensions, so this runs in its own environment, never the project's:

    python -m venv <dir> && <dir>/bin/pip install dp-accounting==0.7.0
    <dir>/bin/python scripts/measure_dp_accountant_crosscheck.py -o docs/measurements/dp-accountant-crosscheck-<date>.json

tests/test_r86_dp_accountant_crosscheck.py computes the repository's figure in the project's environment and holds it to
this record.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import dp_accounting
from dp_accounting import pld, rdp

REPO = Path(__file__).resolve().parents[1]
FIT, BATCH, STEPS, DELTA = 511, 64, 239, 1e-5
SIGMAS = (1.89138, 3.45294)
PLD_INTERVAL = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    q = BATCH / FIT
    rows = []
    for sigma in SIGMAS:
        event = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), STEPS)
        ra = rdp.RdpAccountant()
        ra.compose(event)
        pa = pld.PLDAccountant(value_discretization_interval=PLD_INTERVAL)
        pa.compose(event)
        rows.append({"noise_multiplier": sigma, "dp_accounting_rdp_epsilon": round(ra.get_epsilon(DELTA), 6),
                     "dp_accounting_pld_epsilon": round(pa.get_epsilon(DELTA), 6)})
    record = {
        "what": "DP-SGD's (epsilon, delta) for the murmur study's settings, from dp_accounting's RDP and PLD accountants",
        "mechanism": "Poisson-sampled Gaussian, self-composed",
        "settings": {"fitting_records": FIT, "expected_batch": BATCH, "sampling_rate": q, "steps": STEPS, "delta": DELTA,
                     "source": "docs/measurements/murmur-dp-2026-11-14.json, the search route"},
        "rows": rows,
        "pld_value_discretization_interval": PLD_INTERVAL,
        "environment": {"dp_accounting": version("dp-accounting"), "python": platform.python_version(),
                        "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
