#!/usr/bin/env python3
"""Kulynych et al.'s bound on a reconstruction's success, added to the k-candidate audit's record (NeurIPS 2025).

*Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy*
(arXiv:2507.06969), Theorem 3.1: under f-DP, an attack whose best blind guess succeeds with probability b succeeds with
probability at most 1 - f(b); for k candidates drawn uniformly, b = 1/k. The trade-off curve f is the accountant's,
computed by riskcal 1.5.1 (MIT, the authors' library), which runs in its own environment beside dp_accounting:

- the heart federation: from its zCDP rho (`riskcal.analysis.get_beta_from_zcdp`), rho for adding or removing a
  record, 4 rho for a swap (zCDP's group privacy for a group of two);
- the murmur study: from the privacy-loss distribution of its Poisson-subsampled Gaussian, composed over its steps
  (dp_accounting, `riskcal.analysis.get_beta_from_pld`), for adding or removing a record.

The bound caps the expected accuracy of guessing every slot, which the record states with its interval. Run with the
audit environment's Python:

    tools_external/dpaudit-venv/bin/python scripts/audit_dp_kcandidate_bound.py docs/measurements/dp-kcandidate-audit-<date>.json
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    from dp_accounting.pld import privacy_loss_distribution as pld_module
    from riskcal import analysis

    path = Path(sys.argv[1])
    record = json.loads(path.read_text())

    def zcdp_bound(rho: float, k: int) -> float:
        return round(1.0 - float(analysis.get_beta_from_zcdp(rho, alpha=1.0 / k)), 6)

    def pld_bound(sigma: float, q: float, steps: int, k: int) -> float:
        pld = pld_module.from_gaussian_mechanism(standard_deviation=sigma, sampling_prob=q,
                                                 use_connect_dots=True).self_compose(steps)
        return round(1.0 - float(analysis.get_beta_from_pld(pld, alpha=1.0 / k)), 6)

    for arm in record["fedheart"].values():
        arm["k2_add_remove"]["reconstruction_bound"] = zcdp_bound(arm["rho"], 2)
        arm["k7_swap"]["reconstruction_bound"] = zcdp_bound(4 * arm["rho"], 7)
    seeded = record["seeded_checks"]["fedheart_a_sixteenth_of_the_variance"]
    seeded["k2_add_remove"]["reconstruction_bound"] = zcdp_bound(seeded["rho_accounted"], 2)
    for arm in record["murmur"].values():
        arm["k2_add_remove"]["reconstruction_bound"] = pld_bound(arm["noise_multiplier"], arm["sampling_rate"], arm["steps"], 2)
    seeded = record["seeded_checks"]["murmur_every_person_every_step"]
    seeded["k2_add_remove"]["reconstruction_bound"] = pld_bound(
        seeded["noise_multiplier"], seeded["accountant_assumes_sampling_rate"], seeded["steps"], 2)
    for group in (record["fedheart"], record["murmur"], record["seeded_checks"]):
        for arm in group.values():
            for key in ("k2_add_remove", "k7_swap"):
                if key in arm:
                    a = arm[key]
                    a["every_slot"]["within_the_bound"] = a["every_slot"]["interval_95"][0] <= a["reconstruction_bound"]
    record["reconstruction_bound"] = {
        "method": "Kulynych et al., NeurIPS 2025 (arXiv:2507.06969), Theorem 3.1: success <= 1 - f(1/k), f from "
                  "riskcal (zCDP for the heart federation, 4 rho for a swap; the PLD of the Poisson-subsampled Gaussian "
                  "for the murmur study)",
        "riskcal": version("riskcal"), "dp_accounting": version("dp-accounting"),
        "reading": "within_the_bound: the lower end of the every-slot accuracy's 95 percent interval is at or under the "
                   "bound; an accuracy past it would contradict the accounting, as the seeded checks are built to."}
    path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(json.dumps(record["reconstruction_bound"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
