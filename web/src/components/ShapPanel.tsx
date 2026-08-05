// Signed SHAP feature-attribution chart for a use case's private-inference decision.
// Reads the run's `explanation` artifact (exact linear SHAP, ML-DSA-signed). Aimed at EU AI Act
// Art. 13, which applies from 2027-12-02 (Annex III) after the AI Omnibus deferral - not Aug 2026.
export default function ShapPanel({ explanation }: { explanation: any }) {
  const tops = explanation?.top_features as any[] | undefined;
  if (!tops?.length) return null;
  const max = Math.max(...tops.map((t) => Math.abs(t.contribution)), 1e-6);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>Why — top features (SHAP)</h3>
        <span className="tag" style={{ fontSize: 10 }}>🔏 signed explanation</span>
      </div>
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
