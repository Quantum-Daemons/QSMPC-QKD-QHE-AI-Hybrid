// Signed SHAP feature-attribution chart for a use case's private-inference decision.
// Reads the run's `explanation` artifact (exact linear SHAP, ML-DSA-signed). What EU AI Act Art. 13
// would ask of THIS use case comes from `aiActScope.ts`: until Round 69 the panel printed "Art. 13,
// applicable 2 Dec 2027" on every page, which was right for none of them - two use cases are not
// high-risk at all, and the two that are reach it through Annex I, from 2 August 2028.
// `surfacedBackend` is the backend the run actually used - `artifacts.primary_inference.backend`,
// one of "qcnn-trained" or "ckks". It is deliberately NOT the `inference` control's value: selecting
// QCNN on a use case that ships no trained weights falls back to CKKS, and gating this panel on the
// control made it assert that a quantum circuit produced a class that CKKS produced.
// `isOfReportedClass` says whether the explained model is the one that produced the class this page
// reports. It is TRUE on telecom and satellite, whose flows have no `private_inference` stage, and
// false on finance/genomics/medical, which do. The paragraph below used to assert the false case
// unconditionally - so on the two pages where the attribution is of the reported decision, the panel
// told the reader not to read it that way. It survived because the error points toward caution and so
// reads as rigour. `clientCompute` sets the flag explicitly at each explanation site.
import { AI_ACT_CAVEAT, AI_ACT_SCOPE } from "../lib/aiActScope";

export default function ShapPanel({ explanation, provenance, baseline, publishedModel, surfacedBackend,
  isOfReportedClass, hasTreeAttribution, usecase }: {
  explanation: any; provenance?: string | null; baseline?: string | null;
  publishedModel?: string | null; surfacedBackend?: string | null;
  isOfReportedClass?: boolean | null;
  /** Whether THIS page's model panel renders tree-ensemble attribution. True on telecom only. */
  hasTreeAttribution?: boolean | null;
  /** The use case, for what the EU AI Act would ask of its model. */
  usecase?: string;
}) {
  const aiAct = usecase ? AI_ACT_SCOPE[usecase] : undefined;
  const tops = explanation?.top_features as any[] | undefined;
  if (!tops?.length) return null;
  const max = Math.max(...tops.map((t) => Math.abs(t.contribution)), 1e-6);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>Why — top features (SHAP)</h3>
        <span className="tag" style={{ fontSize: 10 }}>🔏 signed explanation</span>
      </div>
      {/* WHICH model this explains. The stage runs two inferences - the bundled model on this page's
          own data, and a pinned Hugging Face model on its own corpus - and this attribution is of
          the first. Without saying so, a reader who has just seen "Published model: ResNet-18" in
          the provenance card above will take these bars to be the ResNet's. */}
      <p className="muted small" style={{ marginTop: -2 }}>
        Explains the <b>{provenance ?? "pipeline"}</b> on this page's own data
        {baseline ? <> · reference point: {baseline}</> : null}
        {publishedModel ? (
          <> · <b>{publishedModel}</b> also ran this page, on its own corpus, and has <b>no</b> SHAP
          here: exact linear SHAP applies to the linear student, not to a ResNet, a tree ensemble or a
          fixed quantum circuit.</>
        ) : null}
        {/* The tree-ensemble attribution exists on ONE page, and getting this gate right took two
            attempts worth recording.
            Originally the sentence sat inside the `publishedModel` clause and promised the block on
            every page. `AttributionRows` renders only where the record carries the two tree measures -
            telecom - so it pointed readers at a block that was not there. But it was worse than that:
            telecom is precisely the page where `publishedModel` is FALSY, so the sentence appeared on
            the four pages WITHOUT the block and never on the one with it.
            The first fix nested it under `hasTreeAttribution` while leaving it inside `publishedModel`,
            which made it appear nowhere at all - correct in the sense that it no longer lied, and
            useless. It belongs outside: whether this page's model panel shows tree attribution has
            nothing to do with whether a published model also ran. */}
        {hasTreeAttribution ? (
          <> · The published <i>tree ensemble</i> on this page <b>does</b> carry attribution — impurity
          and permutation importance, in the model panel above, computed at training time because a
          forest has no closed-form Shapley value.</>
        ) : null}
      </p>
      {/* Which of these two paragraphs applies is a property of WHAT PRODUCED THE SURFACED CLASS, not
          of the page's flow. It was written as a flow property, and the flow is the wrong predicate:
          telecom's flow has no `private_inference` stage, so this said "this page has no encrypted
          inference stage" - while `runEncryptedTelecomStudent` emitted a full set of `encrypted_*`
          metrics and, after Round 37, a card titled "Private inference against plaintext, this run"
          rendered directly above this sentence. Defensible about the stage graph, false about the
          page.

          Worse, with QCNN selected BOTH this paragraph and the warning below rendered at once,
          saying opposite things - because `explanation_is_of_reported_class` was set before the
          branch that decides what surfaced. That flag is now set after it. */}
      {isOfReportedClass ? (
        <p className="muted small" style={{ marginTop: -2 }}>
          These bars explain <b>the same model that produced the class reported above</b> — read them
          as an attribution of that decision.
        </p>
      ) : (
        <p className="muted small" style={{ marginTop: -2 }}>
          These bars explain the <b>bundled linear model</b>. The <b>class this page reports</b> came
          from a different computation — so read the bars as "what drives this model on this input",
          not as an attribution of the reported class.
        </p>
      )}
      {surfacedBackend === "qcnn-trained" && (
        <div className="note" style={{ marginBottom: 8, borderColor: "var(--warn)", color: "var(--warn)" }}>
          ⚠ The class this page surfaces came from the <b>trained QCNN</b> circuit. These bars
          describe the bundled linear model, which is a different decision.
        </div>
      )}
      {tops.map((t, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, margin: "3px 0" }}>
          <span className="muted" style={{ width: 150, fontSize: 12, textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{t.feature}</span>
          <div style={{ flex: 1, background: "#1a2233", borderRadius: 4, height: 16 }}>
            <div style={{ width: `${Math.max(3, (Math.abs(t.contribution) / max) * 100)}%`, height: "100%", borderRadius: 4, background: t.contribution >= 0 ? "#ff6b6b" : "#4f9dff" }} />
          </div>
          <span className="mono" style={{ width: 66, fontSize: 11, textAlign: "right" }}>{t.contribution >= 0 ? "+" : ""}{Number(t.contribution).toFixed(3)}</span>
        </div>
      ))}
      <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
        red = pushes toward the flagged class · blue = away · exact linear SHAP, ML-DSA-signed
      </p>
      {aiAct ? (
        <p className="muted" style={{ fontSize: 11, marginTop: 2 }}>
          EU AI Act Art. 13 (transparency for high-risk systems). {aiAct.text} {AI_ACT_CAVEAT}
        </p>
      ) : null}
    </div>
  );
}
