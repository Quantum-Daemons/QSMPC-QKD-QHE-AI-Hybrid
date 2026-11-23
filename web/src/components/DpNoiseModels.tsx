// Where the noise is added: central, local and distributed differential privacy side by side, this page's model
// marked (Round 86, the owner's choice of 2026-11-22: DP explained, with a central-versus-local diagram). The local
// figure is arithmetic, not a measurement: n independent draws of the same noise add their variances, so their sum's
// standard deviation is the square root of n times one draw's.
export type NoiseModel = "central" | "distributed";

const KINDS = ["central", "local", "distributed"] as const;

export default function DpNoiseModels({ here, n, units }: {
  /** The model this page's release uses. */
  here: NoiseModel;
  /** How many people (or accounts) the release is about. */
  n: number;
  /** Their plural noun, as the card names them. */
  units: string;
}) {
  const count = n.toLocaleString("en-US");
  const body: Record<(typeof KINDS)[number], { title: string; text: string }> = {
    central: {
      title: "Central",
      text: "One trusted party holds the exact data and adds noise once, to what it releases.",
    },
    local: {
      title: "Local",
      text: `Each of the ${count} ${units} adds noise sized for one before anything is sent: no one has to be trusted, `
        + `but the draws add up, so the total's noise is ${Math.sqrt(n).toFixed(1)} times one draw's (the square root of `
        + `${count}). No page here uses it.`,
    },
    distributed: {
      title: "Distributed",
      text: "Each party adds a share of the noise inside a secure sum, and only the noisy total opens: about the central "
        + "noise, while no party sees another's exact value (Kairouz, Liu and Steinke, ICML 2021).",
    },
  };
  return (
    <div style={{ marginTop: 12 }} id="dp-noise-models">
      <h4>Where the noise is added</h4>
      <div className="grid cols-3" style={{ gap: 10, marginTop: 6 }}>
        {KINDS.map((k) => (
          <div key={k} className={`noise-model${k === here ? " here" : ""}`}>
            <h5>{body[k].title}{k === here ? <>{" "}<span className="tag" style={{ marginLeft: 4 }}>this page</span></> : null}</h5>
            <p className="small">{body[k].text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
