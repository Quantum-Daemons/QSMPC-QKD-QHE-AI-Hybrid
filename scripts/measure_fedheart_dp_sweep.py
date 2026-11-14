#!/usr/bin/env python3
"""The medical page's record-level DP arm over many draws of its noise (Round 86, the owner's choice of 2026-11-14:
privacy-utility curves with their spread on the DP cards).

The page runs `fedHeartDp` (web/src/lib/clientFedHeart.ts) once per visit, its noise from the tab's CSPRNG, so each
run's row of accuracies is one draw. This runs the same function in node, built as the page builds it, DRAWS times at
the page's own seed, each draw's noise from the module's seeded source (`seededLabel`, which only tests and this record
use: seeded noise is not private), and records, for each epsilon, the accuracy's mean, standard deviation and the
10th, 50th and 90th percentiles over the draws. The no-noise row draws nothing and is the same every time.

Writes docs/measurements/fedheart-dp-sweep-2026-11-14.json and web/public/data/medical/fedheart_dp_sweep.json.

Usage: python -I scripts/measure_fedheart_dp_sweep.py [--draws 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
RECORD = REPO / "docs" / "measurements" / "fedheart-dp-sweep-2026-11-14.json"
CARD = WEB / "public" / "data" / "medical" / "fedheart_dp_sweep.json"
LABEL = "fedheart-dp-sweep-2026-11-14"

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const [WEB, draws, label, outFile] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-dp-sweep-"));
const out = join(dir, "bundle.mjs");
await build({ stdin: { contents: `export { fedHeartDp, loadHeartSites, FED_HEART_DP } from ${JSON.stringify(resolve(WEB, "src/lib/clientFedHeart"))};`,
                       resolveDir: WEB, loader: "ts" },
              bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
              define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' } });
globalThis.fetch = async (url) => {
  const text = readFileSync(join(WEB, "public", String(url).replace(/^\/+/, "")), "utf8");
  return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const sites = await m.loadHeartSites();
const runs = [];
for (let i = 1; i <= Number(draws); i++) {
  const r = await m.fedHeartDp(sites, 0, { seededLabel: `${label}:${i}` });
  runs.push(r.rows.map((x) => ({ epsilon: Number.isFinite(x.epsilon) ? x.epsilon : null, mean: x.mean, sdTotal: x.sdTotal })));
}
writeFileSync(outFile, JSON.stringify({ runs, delta: m.FED_HEART_DP.delta, rounds: m.FED_HEART_DP.rounds, clip: m.FED_HEART_DP.clip }));
process.exit(0);
"""


def main() -> int:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        script, out = Path(d) / "h.mjs", Path(d) / "out.json"
        script.write_text(HARNESS)
        r = subprocess.run(["node", str(script), str(WEB), str(a.draws), LABEL, str(out)], cwd=WEB,
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            return 1
        res = json.loads(out.read_text())
    eps = [x["epsilon"] for x in res["runs"][0]]
    rows = []
    for k, e in enumerate(eps):
        acc = np.array([run[k]["mean"] for run in res["runs"]])
        rows.append({"epsilon": e, "draws": int(len(acc)), "mean": float(acc.mean()), "sd": float(acc.std(ddof=1)),
                     "p10": float(np.percentile(acc, 10)), "p50": float(np.percentile(acc, 50)),
                     "p90": float(np.percentile(acc, 90)), "min": float(acc.min()), "max": float(acc.max()),
                     "noise_sd_total": res["runs"][0][k]["sdTotal"]})
    record = {
        "_about": ("The medical page's record-level DP arm (fedHeartDp) over many draws of its noise, at the page's seed: "
                   "accuracy on FLamby's metric per epsilon. Written by scripts/measure_fedheart_dp_sweep.py; the noise is "
                   "seeded for the record and is not private."),
        "function": "web/src/lib/clientFedHeart.ts fedHeartDp(sites, 0, { seededLabel })",
        "seed": 0, "draws": a.draws, "seeded_label": LABEL, "delta": res["delta"], "rounds": res["rounds"], "clip": res["clip"],
        "metric": "FLamby's metric: accuracy at 0.5 on each hospital's own test rows, averaged over the four",
        "rows": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()} for row in rows],
        # Each draw's accuracy per epsilon, in the rows' order, so a check can rerun a draw and compare it exactly.
        "per_draw": [[run[k]["mean"] for k in range(len(eps))] for run in res["runs"]],
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    CARD.write_text(json.dumps({**{k: record[k] for k in ("seed", "draws", "delta", "rounds", "clip", "metric", "rows")},
                                "_about": "The medical DP card's band; written by scripts/measure_fedheart_dp_sweep.py."},
                               indent=1) + "\n")
    print(json.dumps(record["rows"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
