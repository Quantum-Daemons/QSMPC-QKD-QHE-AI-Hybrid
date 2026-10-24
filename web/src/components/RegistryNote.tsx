import Details from "./Details";

/**
 * A registry prose field, first sentence visible and the rest folded.
 *
 * The registry text is GENERATED - `scripts/check_generated_prose.py` and CI steps 2-3 reject hand
 * edits to it - so a page that carried 1,300 words of it for one model could only be shortened where it
 * is rendered. The first sentence of each field is written to carry its point, and it stays in the same
 * block as the table it qualifies (checklist UX6); notes shorter than a few lines are shown whole.
 * Licence attribution is never passed through here: it is rendered in full (checklist PM6).
 */
export default function RegistryNote({ label, text }: { label: string; text: string }) {
  // The first sentence: at least 40 characters, ending at ". " before a capital, a bracket or a quote,
  // so "e.g." and decimals such as "0.98" do not end it.
  const m = /^([\s\S]{40,}?[.!?])\s+(?=[A-Z(`"])/.exec(text);
  const rest = m ? text.slice(m[0].length) : "";
  const restWords = rest.trim().split(/\s+/).filter(Boolean).length;
  if (!m || restWords < 25) {
    return <p className="muted small"><strong>{label}:</strong> {text}</p>;
  }
  return (
    <div style={{ margin: "6px 0" }}>
      <p className="muted small" style={{ margin: 0 }}><strong>{label}:</strong> {m[1]}</p>
      <Details summary={`${label}, continued (${restWords} words)`}><p>{rest}</p></Details>
    </div>
  );
}
