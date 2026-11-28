#!/usr/bin/env python3
"""Criterion 3 of the telecom square network from run to run, under each parameter set, as its addendum states
(`references/telecom-nidd-square-student-preregistration-addendum-spread.md`, Round 86, batch 12).

Runs the page's own module (`runSquareTabular`, `web/src/lib/clientFHESquareTabular.ts`) in node with the real
node-seal, 30 times under each set the page file offers, on the 2,048 flows the page ships, each run with fresh keys,
and writes `docs/measurements/telecom-nidd-square-ckks-spread-2026-11-28.json`.

    python scripts/measure_telecom_square_ckks_spread.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ADDENDUM = REPO / "references" / "telecom-nidd-square-student-preregistration-addendum-spread.md"
ADDENDUM_SHA256 = "dabdd902f2c67a1c97cc696f9acf395bd680bb0bd02ad8d5e0ca5d5593794271"
PAGE = WEB / "public" / "data" / "telecom_nidd" / "square_network.json"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-square-ckks-spread-2026-11-28.json"
RUNS, LIMIT = 30, 1e-3

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const [WEB, OUT, PAGE, SET, RUNS] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquareTabular"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(PAGE, "utf8"));
const raw = w.held_out.raw, k = w.classes.length;
const z = new Float64Array(raw.length * w.n_in);
raw.forEach((x, r) => z.set(m.scaleRow(w, x), r * w.n_in));
for (let run = 0; run < Number(RUNS); run++) {
  const r = await m.runSquareTabular(w, z, raw.length, SET);
  let right = 0;
  for (let i = 0; i < r.rows; i++) if (m.argmaxAt(r.logits, k * i, k) === w.held_out.labels[i]) right++;
  console.log(JSON.stringify({ set: SET, run, rows: r.rows, maxAbsError: r.maxAbsError,
                               decisionsDiffer: r.decisionsDiffer, accuracy: right / r.rows }));
}
"""


def main() -> int:
    got = hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()
    if got != ADDENDUM_SHA256:
        raise SystemExit(f"{ADDENDUM.name} hashes to {got[:12]}, not the committed {ADDENDUM_SHA256[:12]}: run nothing")
    page = json.loads(PAGE.read_text())
    sets: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "h.mjs").write_text(HARNESS)
        for name in page["ckks"]:
            out = subprocess.run(["node", str(Path(tmp) / "h.mjs"), str(WEB), str(Path(tmp) / "b.mjs"), str(PAGE), name,
                                  str(RUNS)], check=True, capture_output=True, text=True, cwd=WEB, timeout=3600)
            runs = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
            if len(runs) != RUNS:
                raise SystemExit(f"{name}: {len(runs)} runs returned, not {RUNS}")
            e = [r["maxAbsError"] for r in runs]
            sets[name] = {"runs": RUNS, "rows": runs[0]["rows"], "max_abs_error_min": min(e),
                          "max_abs_error_median": statistics.median(e), "max_abs_error_max": max(e),
                          "runs_above_limit": sum(x > LIMIT for x in e),
                          "runs_with_a_decision_changed": sum(r["decisionsDiffer"] > 0 for r in runs),
                          "accuracies": sorted({r["accuracy"] for r in runs}),
                          "per_run_max_abs_error": e}
            print(f"  {name}: largest error {min(e):.2e} to {max(e):.2e} (median {statistics.median(e):.2e}), "
                  f"{sets[name]['runs_above_limit']} of {RUNS} runs above {LIMIT:g}", flush=True)
    record = {"addendum": {"file": str(ADDENDUM.relative_to(REPO)), "sha256": ADDENDUM_SHA256},
              "page_file_sha256": hashlib.sha256(PAGE.read_bytes()).hexdigest(), "limit": LIMIT,
              "module": "runSquareTabular, web/src/lib/clientFHESquareTabular.ts, node with node-seal",
              "date": date.today().isoformat(), "sets": sets,
              "page_default_set": page["default_set"],
              "default_holds_the_rule_in_every_run": (sets[page["default_set"]]["runs_above_limit"] == 0
                                                     and sets[page["default_set"]]["runs_with_a_decision_changed"] == 0)}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0 if record["default_holds_the_rule_in_every_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
