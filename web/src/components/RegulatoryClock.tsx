import { CLOCK, type Firmness, type Lane } from "../lib/regulatoryClock";

/**
 * The regulatory clock as a timeline by jurisdiction plus the full table, so a date is never read
 * without who it binds and how firm it is. Colour carries firmness and the table repeats it in words,
 * so nothing depends on colour alone.
 */
const LANES: Lane[] = ["NIST", "US federal", "US NSS", "EU", "Companies"];
const T0 = 2024, T1 = 2036.6, X0 = 120, X1 = 950;
const x = (t: number) => X0 + ((t - T0) / (T1 - T0)) * (X1 - X0);
const LANE_GAP = 54;   // labels alternate above/below; 40 stacked neighbouring lanes' labels
const laneY = (l: Lane) => 34 + LANES.indexOf(l) * LANE_GAP;
const AXIS_Y = laneY("Companies") + 34;

const COLOR: Record<Firmness, string> = {
  "in force": "var(--accent-2)", requirement: "var(--accent-2)", proposed: "var(--accent)",
  draft: "var(--warn)", deferred: "var(--warn)", target: "var(--quantum)", passed: "var(--muted)",
};

export default function RegulatoryClock() {
  const now = new Date();
  const tNow = now.getFullYear() + now.getMonth() / 12 + (now.getDate() - 1) / 365;
  // Alternate labels above/below within a lane, so neighbours a few months apart do not collide.
  const seen: Record<string, number> = {};
  return (
    <div>
      <figure className="chart-figure" style={{ margin: 0 }}>
        <svg viewBox={`0 0 960 ${AXIS_Y + 26}`} role="img" aria-labelledby="clock-title"
          style={{ width: "100%", minWidth: 640, height: "auto", display: "block" }}>
          <title id="clock-title">Dated post-quantum and AI-transparency obligations, 2024 to 2036, by who they bind</title>
          {LANES.map((l) => (
            <g key={l}>
              <text x={8} y={laneY(l) + 4} fontSize={12} fontWeight={600} fill="var(--text)">{l}</text>
              <line x1={X0} y1={laneY(l)} x2={X1} y2={laneY(l)} stroke="var(--border)" />
            </g>
          ))}
          {Array.from({ length: 13 }, (_, i) => T0 + i).map((yr) => (
            <g key={yr}>
              <line x1={x(yr)} y1={AXIS_Y} x2={x(yr)} y2={AXIS_Y + 5} stroke="var(--muted)" />
              <text x={x(yr)} y={AXIS_Y + 18} fontSize={10.5} textAnchor="middle" fill="var(--muted)">{yr}</text>
            </g>
          ))}
          <line x1={X0} y1={AXIS_Y} x2={X1} y2={AXIS_Y} stroke="var(--muted)" />
          {tNow > T0 && tNow < T1 && (
            <g>
              <line x1={x(tNow)} y1={14} x2={x(tNow)} y2={AXIS_Y} stroke="var(--text)" strokeDasharray="3 3" opacity={0.6} />
              <text x={x(tNow) + 4} y={12} fontSize={10} fill="var(--text)">today</text>
            </g>
          )}
          {CLOCK.map((r, i) => {
            const k = r.lane;
            const n = (seen[k] = (seen[k] ?? -1) + 1);
            const above = n % 2 === 0;
            const hollow = r.firmness === "draft" || r.firmness === "deferred" || r.firmness === "target";
            return (
              <g key={i}>
                <circle cx={x(r.t)} cy={laneY(k)} r={5.5} fill={hollow ? "var(--bg)" : COLOR[r.firmness]}
                  stroke={COLOR[r.firmness]} strokeWidth={2} />
                <text x={x(r.t)} y={laneY(k) + (above ? -9 : 17)} fontSize={10} textAnchor="middle"
                  fill={COLOR[r.firmness]}>{r.short}</text>
              </g>
            );
          })}
        </svg>
        <figcaption className="figure-note">
          Filled = in force or a binding requirement for the stated scope; hollow = draft, target or
          deferred; grey = passed; blue = a rule only proposed so far. Each marker is one row of the
          table below.
        </figcaption>
      </figure>
      <table className="facts-table small" style={{ marginTop: 10 }}>
        <thead><tr><th>When</th><th>Instrument</th><th>Binds</th><th>What</th><th>Firmness</th></tr></thead>
        <tbody>
          {CLOCK.map((r, i) => (
            <tr key={i}>
              <td className="mono">{r.when}</td>
              <td>{r.instrument}</td>
              <td>{r.binds}</td>
              <td>{r.what}</td>
              <td style={{ color: COLOR[r.firmness], whiteSpace: "nowrap" }}>{r.firmness}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
