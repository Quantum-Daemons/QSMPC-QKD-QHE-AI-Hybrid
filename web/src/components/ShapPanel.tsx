// Signed SHAP feature-attribution chart for a use case's private-inference decision.
// Reads the run's `explanation` artifact (exact linear SHAP, ML-DSA-signed). Aimed at EU AI Act
// Art. 13, which applies from 2027-12-02 (Annex III) after the AI Omnibus deferral - not Aug 2026.
export default function ShapPanel({ explanation, provenance, baseline, publishedModel, surfaced }: {
  explanation: any; provenance?: string | null; baseline?: string | null;
  publishedModel?: string | null; surfaced?: string | null;
}) {
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
          <> · <b>{publishedModel}</b> also ran this page, on its own corpus, and has <b>no</b>{" "}
          attribution here: exact linear SHAP applies to the linear student, not to a ResNet, a tree
          ensemble or a fixed quantum circuit.</>
        ) : null}
      </p>
      {surfaced === "qcnn" && (
        <div className="note" style={{ marginBottom: 8, borderColor: "var(--warn)", color: "var(--warn)" }}>
          ⚠ Inference is set to <b>QCNN</b>, so the class this page surfaces comes from the quantum
          circuit. These bars still describe the CKKS-linear model, which is a different decision.
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
        red = pushes toward the flagged class · blue = away · exact linear SHAP (for EU AI Act Art. 13, applicable 2 Dec 2027; ML-DSA-signed)
      </p>
    </div>
  );
}
