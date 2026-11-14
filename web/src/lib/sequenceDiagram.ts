/**
 * A run's messages as a Mermaid sequence diagram (Round 86, the owner's fourth choice of 2026-11-14).
 *
 * Read from the same `message` and `holds` events as "Who holds what" (`runTimeline.ts`), up to the timeline's position:
 * one participant per party under the canvas's names, the messages numbered in the order the run sent them with the
 * size it measured, each stage a shaded block headed by its name, and what a party keeps as a note beside it. A dotted
 * arrow is a simulated channel (photons, or the protocol's classical messages modelled rather than sent); a solid one
 * carried real bytes. Text only: the page draws the same diagram itself (`SequenceView.tsx`), and this is what a reader
 * copies or downloads to draw it with Mermaid elsewhere.
 * Mermaid's sequence syntax: https://mermaid.js.org/syntax/sequenceDiagram.html (read 2026-11-14).
 */
import type { PipelineEvent } from "./api";
import { sideName, type HoldsData, type MessageData } from "./runTimeline";
import type { DiagramConfig } from "./useCaseFlows";

/** Text Mermaid reads as plain words: a semicolon ends a statement, `#` opens an entity code, angle brackets and
 *  braces are markup, and a colon in an alias would end it. */
export function mermaidText(s: string, max = 110): string {
  const t = s.replace(/[;#]/g, ",").replace(/[<>{}[\]]/g, "").replace(/:/g, " -").replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max - 3)}...` : t;
}

const SIMULATED = new Set(["photons (simulated)", "classical (simulated)"]);
const size = (b: number | null, n?: number) =>
  b == null ? "size not measured" : `${b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${b.toLocaleString("en-US")} B`}${n && n > 1 ? ` over ${n.toLocaleString("en-US")} messages` : ""}`;

/** The diagram's text for the first `upTo` events. */
export function toMermaidSequence(usecase: string, events: PipelineEvent[], config: DiagramConfig = {},
                                  upTo: number = events.length): string {
  const shown = events.slice(0, Math.max(0, upTo)).filter((e) => e.type === "message" || e.type === "holds");
  // Participants in a fixed order: side A, side B, the many parties, then any other in the order met.
  const sides: string[] = [];
  const meet = (s: string) => { if (!sides.includes(s)) sides.push(s); };
  for (const e of shown) {
    const d = e.data as unknown as MessageData & HoldsData;
    if (e.type === "message") { meet(d.from); meet(d.to); } else meet(d.party);
  }
  const rank = (s: string) => (s === "a" ? 0 : s === "b" ? 1 : s === "many" ? 2 : 3);
  sides.sort((x, y) => rank(x) - rank(y));
  const id = new Map(sides.map((s, i) => [s, `P${i}`]));
  const lines = ["sequenceDiagram", "  autonumber"];
  for (const s of sides) lines.push(`  participant ${id.get(s)} as ${mermaidText(sideName(usecase, s, config), 60)}`);
  let stage: string | null = null;
  const span = () => (sides.length > 1 ? `${id.get(sides[0])},${id.get(sides[sides.length - 1])}` : `${id.get(sides[0])}`);
  for (const e of shown) {
    if (e.stage !== stage) {
      if (stage !== null) lines.push("  end");
      stage = e.stage;
      lines.push("  rect rgba(79, 157, 255, 0.08)", `  Note over ${span()}: ${mermaidText(stage, 60)}`);
    }
    const d = e.data as unknown as MessageData & HoldsData;
    if (e.type === "message") {
      const arrow = SIMULATED.has(d.protection) ? "-->>" : "->>";
      lines.push(`  ${id.get(d.from)}${arrow}${id.get(d.to)}: ${mermaidText(`${d.what}, ${size(d.bytes, d.count)}, ${d.protection}`)}`);
    } else if (d.kept) {
      lines.push(`  Note over ${id.get(d.party)}: keeps ${mermaidText(d.what, 70)}`);
    }
  }
  if (stage !== null) lines.push("  end");
  return lines.join("\n") + "\n";
}
