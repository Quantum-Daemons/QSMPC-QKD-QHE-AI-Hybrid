// Split inference on a chest X-ray (Round 86, the owner's fourth choice of 2026-11-14). The published network up to its
// pooled features runs in this tab on one of the model's 16 published films, and only the 512 features are encrypted:
// the network's own last layer is evaluated on them under CKKS (`encryptedLinear`, clientFHE.ts, unchanged), and the
// decrypted logit gives the decision at the model's operating point. Every file is the model's pinned Hub file or this
// repository's record of its last layer (lib/cxrSplit.ts says which).
import { useEffect, useRef, useState } from "react";
import { modelsFor } from "../lib/config";
import { type CxrHead, decide, filmInput, filmRgba, headLogit } from "../lib/cxrSplit";
import { type ModelInputFile, openPinnedGraph, publishedSampleOf } from "../lib/onnxInference";
import StageBadge from "./StageBadge";

const KEY = "medical_fl.cxr_fedavg";
const SIZE = 224;

interface Outcome {
  film: number; backboneMs: number; graphBytes: number; graphLoadMs: number;
  zPlain: number; zEnc: number; fheMs: number; ctBytes: number | null; rotations: number;
  effusion: boolean; p: number; recorded: number | null;
}

function Film({ pixels, size, label }: { pixels: number[]; size: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(filmRgba(pixels), SIZE, SIZE), 0, 0);
  }, [pixels]);
  return <canvas ref={ref} width={SIZE} height={SIZE} role="img" aria-label={label}
    style={{ width: size, height: size, display: "block", borderRadius: 4 }} />;
}

export default function CxrSplitInference() {
  const spec = modelsFor("medical_fl").find((m) => m.key === KEY);
  const [file, setFile] = useState<ModelInputFile | null>(null);
  const [head, setHead] = useState<CxrHead | null>(null);
  const [chosen, setChosen] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  if (!spec?.hf_revision) return null;
  const views = (file as unknown as { views?: string[] } | null)?.views ?? [];
  const labels = (file?.labels ?? []) as number[];
  const name = (i: number) => `film ${i + 1}, ${labels[i] === 1 ? "effusion" : "no effusion"} by NIH's label, ${views[i] ?? "view not given"}`;

  const loadFilms = async () => {
    setBusy("Fetching the 16 films and the last layer..."); setFailed(null);
    try {
      const [got, h] = await Promise.all([
        publishedSampleOf(spec),
        fetch(`${import.meta.env.BASE_URL}data/medical/cxr_head.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))),
      ]);
      if (got.kind !== "ok") throw new Error(got.kind === "absent" ? "the model's repository has no evaluation sample at its pinned revision" : got.reason);
      if ((h as CxrHead).hf_revision !== spec.hf_revision) throw new Error("the recorded last layer is for another revision of the model");
      setFile(got.file); setHead(h as CxrHead);
    } catch (e) { setFailed((e as Error)?.message ?? String(e)); }
    finally { setBusy(null); }
  };

  const run = async () => {
    if (!file || !head) return;
    setFailed(null); setOutcome(null);
    try {
      setBusy("Running the network up to its pooled features in this tab...");
      const graph = await openPinnedGraph(spec, head.features_graph);
      const input = filmInput(file.samples[chosen] as number[]);
      const t0 = performance.now();
      const { Tensor } = await import("onnxruntime-web/wasm");
      const name0 = graph.session.inputNames[0];
      const out = await graph.session.run({ [name0]: new Tensor("float32", input, [1, 1, SIZE, SIZE]) });
      const features = Array.from(out[graph.session.outputNames[0]].data as Float32Array);
      const backboneMs = performance.now() - t0;
      setBusy("Encrypting the 512 features and evaluating the last layer under CKKS...");
      const { encryptedLinear } = await import("../lib/clientFHE");
      const fhe = await encryptedLinear([head.weights], [head.bias], features);
      const zEnc = fhe.logits[0];
      const d = decide(head, zEnc);
      const recorded = (file.expected as { per_row?: { predicted?: number[] } })?.per_row?.predicted?.[chosen] ?? null;
      setOutcome({ film: chosen, backboneMs, graphBytes: graph.bytes, graphLoadMs: graph.loadMs, zPlain: headLogit(head, features),
                   zEnc, fheMs: fhe.fheMs, ctBytes: fhe.ctBytes, rotations: fhe.rotations, effusion: d.effusion, p: d.p, recorded });
    } catch (e) { setFailed((e as Error)?.message ?? String(e)); }
    finally { setBusy(null); }
  };

  const o = outcome;
  return (
    <div className="card" id="cxr-split" style={{ marginTop: 14 }}>
      <h3>Split inference on a chest X-ray <StageBadge kind="notional" /></h3>
      <p className="small" style={{ marginTop: 4 }}>
        The published chest X-ray model ({KEY}, ResNet-18 trained by FedAvg over four simulated hospitals) runs in two parts.
        In this tab, as on the data owner&apos;s own machine, the network up to its 512 pooled features reads one of the
        model&apos;s 16 published NIH test films; only those 512 numbers are encrypted under CKKS, and the network&apos;s own
        last layer is evaluated on the ciphertext, as it would be at the model&apos;s holder. The owner decrypts the logit and
        applies the model&apos;s operating point. Both parties run in this tab, so the separation is notional; the last layer is
        published, so here it is not a secret either. What the card shows is the split, its cost and its precision.
      </p>
      {!file ? (
        <button className="btn" onClick={loadFilms} disabled={busy != null}>
          {busy ?? "Load the 16 films (3.7 MB from the model's Hub repository)"}
        </button>
      ) : (
        <>
          <div className="cxr-films" role="listbox" aria-label="The model's 16 published films">
            {file.samples.map((px, i) => (
              <button key={i} type="button" role="option" aria-selected={i === chosen} className={`cxr-film${i === chosen ? " chosen" : ""}`}
                onClick={() => { setChosen(i); setOutcome(null); }} title={name(i)}>
                <Film pixels={px as number[]} size={64} label={name(i)} />
                <span className="small">{i + 1}{labels[i] === 1 ? " E" : ""}{views[i] ? ` ${views[i]}` : ""}</span>
              </button>
            ))}
          </div>
          <p className="muted small" style={{ margin: "4px 0 8px" }}>E: effusion by NIH&apos;s label, which was mined from the radiology report.
            AP films are typically taken at the bedside, PA standing.</p>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start" }}>
            <Film pixels={file.samples[chosen] as number[]} size={SIZE} label={name(chosen)} />
            <div style={{ flex: "1 1 260px", minWidth: 240 }}>
              <button className="btn" onClick={run} disabled={busy != null}>{busy ?? `Run split inference on film ${chosen + 1}`}</button>
              {o && o.film === chosen && head ? (
                <table className="small" style={{ marginTop: 8 }}>
                  <tbody>
                    <tr><th scope="row">In this tab, in the clear</th><td>{head.features_graph}: {(o.graphBytes / 1e6).toFixed(1)} MB fetched and checked against its pin ({(o.graphLoadMs / 1000).toFixed(1)} s, then cached), the film to 512 features in {o.backboneMs.toFixed(0)} ms</td></tr>
                    <tr><th scope="row">Crossed, encrypted</th><td>the 512 features in one CKKS ciphertext{o.ctBytes != null ? ` of ${o.ctBytes.toLocaleString("en-US")} bytes` : ""}; the last layer evaluated on it in {o.fheMs.toFixed(0)} ms ({o.rotations} rotations)</td></tr>
                    <tr><th scope="row">Decrypted logit</th><td>{o.zEnc.toFixed(6)}, against {o.zPlain.toFixed(6)} in the clear (difference {Math.abs(o.zEnc - o.zPlain).toExponential(2)})</td></tr>
                    <tr><th scope="row">Decision</th><td><b>{o.effusion ? "effusion" : "no effusion"}</b> at the operating point (p = {o.p.toFixed(3)} against {head.decision_threshold.toFixed(4)}); NIH&apos;s label: {labels[o.film] === 1 ? "effusion" : "no effusion"}{o.recorded != null ? <>; the full int8 model&apos;s recorded decision: {o.recorded === 1 ? "effusion" : "no effusion"} ({(o.recorded === 1) === o.effusion ? "the same" : "different"})</> : null}</td></tr>
                  </tbody>
                </table>
              ) : null}
              {failed ? <p className="small" style={{ color: "var(--danger)" }}>It did not run: {failed}</p> : null}
            </div>
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>
            The last layer is the network&apos;s own (fc), read from the published weights by <code>scripts/export_cxr_head.py</code>,
            which checks that on all {head?.check.films ?? 16} films the features graph followed by it gives the classifier
            graph&apos;s logit to {head ? head.check.max_abs_logit_difference_float32.toExponential(1) : "-"} in float32 and the
            same decisions in int8. Were the last layer a secret, this split would not keep it from the data owner: the
            decrypted logits for 513 feature vectors of the owner&apos;s choosing determine its 512 weights and its bias.
            For research and demonstration only: not for clinical use, and not a medical device.
            Films: NIH Clinical Center, ChestX-ray14 (Wang et al., CVPR 2017), <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC"
            target="_blank" rel="noreferrer">nihcc.app.box.com/v/ChestXray-NIHCC</a>.
          </p>
        </>
      )}
      {!file && failed ? <p className="small" style={{ color: "var(--danger)" }}>The films did not load: {failed}</p> : null}
    </div>
  );
}
