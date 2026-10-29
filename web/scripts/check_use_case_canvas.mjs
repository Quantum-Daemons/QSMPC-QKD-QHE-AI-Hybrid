/**
 * The use-case canvas draws each stage where the code runs it, the way its data goes.
 *
 * WHY THIS EXISTS (Round 80)
 *
 * The canvas replaced a sequence diagram that drew the satellite's key and telemetry going UP, ground
 * to satellite, on a link the code models as a downlink (Liao et al. 2017's Micius), and that drew
 * two cohorts for a secure sum the run makes over four. A drawing states facts about the code, and
 * nothing checked them. `canvasRow` (src/lib/useCaseFlows.ts) is where the canvas reads them from,
 * and this asserts them there:
 *   1. every stage of every use case has a home on the canvas: components, a lane, or a full-width
 *      row; none falls through to the generic default;
 *   2. the satellite's quantum key and its QSDC telemetry come down the link, satellite to ground,
 *      and the fibre key goes from source to detectors;
 *   3. a many-party stage draws the parties the run has: the configured count before a run, and the
 *      run's own count once it reports one;
 *   4. the private inference's client is the side whose input it protects: Bank B's transaction on
 *      finance, the hospital's and the cohort's data on medical and genomics;
 *   5. a PQC-only profile draws no quantum lane: the key stages are dropped, and the satellite's
 *      telemetry, which the browser then sends under the session key with QSDC skipped, is drawn
 *      as the classical exchange it is. Both figures drew a quantum QSDC arrow there until Round 80;
 *   6. the legend lists the lanes the canvas draws and no others, and the key bus exactly where a
 *      stage reads the session key (the PSI, the federated masks). Until Round 81 it listed all five
 *      everywhere, so telecom and satellite explained CKKS, MPC and key lanes neither draws;
 *   7. the caption says what the run does: the MPC-FHE sentence only where a private inference
 *      runs, no WAN between a satellite and its ground station, and no simulated quantum channel
 *      under a profile that has none. One caption served all five until Round 81;
 *   8. every use case the E2E page's selector offers has a canvas, the generic demo included.
 *
 * HOW IT LOADS THE TS: esbuild bundles the module to a temp ESM file, as check_calibration.mjs does.
 *
 * Run: npm run check:use-case-canvas   (CHECK_MODULE=<path> checks another copy of the module)
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MODULE = process.env.CHECK_MODULE ?? join(WEB, "src/lib/useCaseFlows");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const dir = mkdtempSync(join(tmpdir(), "qsh-canvas-"));
const entry = join(dir, "entry.ts");
writeFileSync(entry, `export { USECASE_SPECS, canvasRow, visibleSteps, kindOf, drawnChannels, captionOf } from ${JSON.stringify(MODULE)};\n`
  + `export { appConfig } from ${JSON.stringify(join(WEB, "src/lib/config"))};\n`
  + `export { USECASES } from ${JSON.stringify(join(WEB, "src/lib/useCases"))};\n`);
const out = join(dir, "bundle.mjs");
execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                     `--outfile=${out}`, "--log-level=error"], { cwd: WEB, stdio: "pipe" });
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });

const cfg = m.appConfig();
const rowsOf = (usecase, config = {}, metrics = {}) =>
  m.visibleSteps(usecase, config).map((s) => ({ s, row: m.canvasRow(usecase, s, m.kindOf(s, config), metrics, config) }));
const find = (rows, stageId, pred = () => true) => rows.find((r) => r.s.stageId === stageId && pred(r.row));

// 1. A home for every stage.
for (const usecase of Object.keys(m.USECASE_SPECS)) {
  for (const { s, row } of rowsOf(usecase)) {
    const placed = row.wide || row.many || (row.a && row.a !== "runs locally") || (row.b && row.b !== "runs locally");
    check("placed", placed, `${usecase}.${s.stageId} has no component, lane or full-width row on the canvas`);
    if (row.a && row.b) check("lane", !!row.flow && row.channel !== "none",
      `${usecase}.${s.stageId} has components on both sides and no lane between them`);
  }
}

// 2. Which way the quantum channel goes.
const sat = rowsOf("satellite_comms");
check("downlink", find(sat, "satellite_qkd")?.row.flow === "ba",
  `the satellite's QKD key is drawn ${find(sat, "satellite_qkd")?.row.flow}; the code models a downlink, satellite to ground ("ba")`);
check("downlink", find(sat, "qsdc_telemetry")?.row.flow === "ba",
  `the satellite's QSDC telemetry is drawn ${find(sat, "qsdc_telemetry")?.row.flow}; it comes down the link ("ba")`);
check("downlink", find(sat, "satellite_qkd")?.row.channel === "quantum", "the satellite's key lane is not a quantum channel");
for (const usecase of ["finance_aml", "medical_fl", "genomics", "telecom_ids"]) {
  const q = find(rowsOf(usecase), "qkd_key_agreement");
  check("fibre", q?.row.flow === "ab" && q?.row.channel === "quantum",
    `${usecase}'s fibre key is drawn ${q?.row.flow} on ${q?.row.channel}; source to detectors on the quantum channel`);
}

// 3. The run's parties.
const hospitals = Number(cfg.federated?.hospitals);
const fl = find(rowsOf("medical_fl"), "federated_training");
check("parties", fl?.row.many?.count === hospitals,
  `medical's federated stage draws ${fl?.row.many?.count} hospitals before a run; the config trains ${hospitals}`);
const fl6 = find(rowsOf("medical_fl", {}, { qfl_clients: 6 }), "federated_training");
check("parties", fl6?.row.many?.count === 6, `a run reporting 6 federated clients draws ${fl6?.row.many?.count}`);
const cohorts = Number(cfg.data?.genomics_biobanks);
const sum = find(rowsOf("genomics"), "qsmpc_compute");
check("parties", sum?.row.many?.count === cohorts && sum?.row.channel === "mpc",
  `genomics' secure sum draws ${sum?.row.many?.count} parties on ${sum?.row.channel}; the run pools ${cohorts} cohorts by shares`);
const sum5 = find(rowsOf("genomics", {}, { qsms_parties: 5 }), "qsmpc_compute");
check("parties", sum5?.row.many?.count === 5, `a run reporting 5 parties to the sum draws ${sum5?.row.many?.count}`);
const psi = find(rowsOf("finance_aml"), "qsmpc_compute");
check("psi", !psi?.row.many && psi?.row.flow === "both" && psi?.row.channel === "classical",
  "finance's PSI is not drawn as a two-way exchange between the two banks under the session key");

// 4. The inference's client.
const clientOf = (usecase) => {
  const r = find(rowsOf(usecase), "private_inference", (row) => row.channel === "fhe")?.row;
  return r ? (/encrypts/.test(r.a ?? "") ? "a" : /encrypts/.test(r.b ?? "") ? "b" : "?") : "none";
};
check("client", clientOf("finance_aml") === "b", `finance's CKKS client is side ${clientOf("finance_aml")}; the input is Bank B's transaction`);
check("client", clientOf("medical_fl") === "a", `medical's CKKS client is side ${clientOf("medical_fl")}; the input is the hospital's`);
check("client", clientOf("genomics") === "a", `genomics' CKKS client is side ${clientOf("genomics")}; the input is the cohort's`);

// 5. No quantum lane under PQC-only.
for (const usecase of Object.keys(m.USECASE_SPECS)) {
  const q = rowsOf(usecase, { qkd: "pqc-only" }).filter(({ row }) => row.channel === "quantum");
  check("pqc-only", q.length === 0, `${usecase} still draws ${q.map((r) => r.s.stageId).join(", ")} under a PQC-only profile`);
}

// 6. The legend: the lanes drawn, and the key bus where the session key is read.
const legend = (usecase, config = {}) => m.drawnChannels(usecase, config);
for (const usecase of ["telecom_ids", "satellite_comms"]) {
  const l = legend(usecase);
  check("legend", !l.lanes.includes("fhe") && !l.lanes.includes("mpc") && !l.keyBus,
    `${usecase}'s legend lists ${[...l.lanes, ...(l.keyBus ? ["key bus"] : [])].join(", ")}; it draws no CKKS, MPC or key lane`);
}
for (const usecase of Object.keys(m.USECASE_SPECS)) {
  const l = legend(usecase);
  const readsKey = rowsOf(usecase).some(({ s, row }) => s.stageId === "federated_training"
    || (s.stageId === "qsmpc_compute" && row.channel === "classical"));
  check("legend", l.keyBus === readsKey,
    `${usecase}'s legend ${l.keyBus ? "lists" : "omits"} the key bus, and ${readsKey ? "a" : "no"} stage reads the session key`);
  check("legend", !legend(usecase, { qkd: "pqc-only" }).lanes.includes("quantum"),
    `${usecase}'s legend lists a quantum lane under a PQC-only profile`);
}
check("legend", legend("genomics").keyBus === false && legend("genomics", { qsmpcOp: "psi" }).keyBus === true,
  "genomics' key bus does not follow its QSMPC op: none for the secure sum, one for the PSI");

// 7. The caption.
for (const usecase of Object.keys(m.USECASE_SPECS)) {
  const infers = rowsOf(usecase).some(({ s }) => s.stageId === "private_inference");
  const cap = m.captionOf(usecase);
  check("caption", /MPC-FHE hybrid/.test(cap) === infers,
    `${usecase}'s caption ${infers ? "omits" : "claims"} the MPC-FHE inference, which its run ${infers ? "performs" : "does not perform"}`);
  check("caption", !/quantum channel is simulated/.test(m.captionOf(usecase, { qkd: "pqc-only" })),
    `${usecase}'s caption describes a simulated quantum channel under a PQC-only profile`);
}
check("caption", !/WAN/.test(m.captionOf("satellite_comms")) && !/WAN/.test(m.captionOf("satellite_comms", { qkd: "pqc-only" })),
  "the satellite's caption puts it and its ground station on a WAN");

// 8. A canvas for every entry in the E2E page's selector (pages/Orchestration.tsx: the use cases,
//    then the generic demo).
for (const id of [...m.USECASES.map((u) => u.id), "demo"])
  check("e2e", id in m.USECASE_SPECS, `the E2E selector offers ${id}, and the canvas has no spec for it`);

if (failures.length) {
  console.error("check:use-case-canvas FAILED");
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check:use-case-canvas OK  ${Object.keys(m.USECASE_SPECS).length} use cases: every stage placed, the downlink `
  + "drawn down, the run's parties counted, the inference client on its side, no quantum lane under PQC-only, "
  + "a legend of the lanes drawn, a caption of what runs, a canvas for every E2E selector entry");
