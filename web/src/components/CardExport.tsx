// A card's own downloads (Round 86, the owner's choice of 2026-11-13): the rows the card draws as CSV, and the card's
// data as JSON, named for the page and the card. Nothing renders until the card has rows, so a button never offers a
// file it cannot fill; each button carries its own accessible name, because six cards on one page each saying "CSV"
// would be six buttons a screen reader cannot tell apart.
import { downloadCSV, downloadJSON } from "../lib/export";

export default function CardExport({ name, label, rows, json }: {
  /** The file stem: "<usecase>-<card>". */
  name: string;
  /** What the card is, for the buttons' accessible names: "the four-hospital matrix". */
  label: string;
  rows: Record<string, unknown>[];
  /** The JSON's content when it is more than the rows; the rows otherwise. */
  json?: unknown;
}) {
  if (rows.length === 0) return null;
  return (
    <span style={{ display: "inline-flex", gap: 6, flexWrap: "wrap" }} data-card-export={name}>
      <button type="button" className="btn ghost" aria-label={`Download ${label} as CSV`}
        onClick={() => downloadCSV(rows, `${name}.csv`)}>⬇ CSV</button>
      <button type="button" className="btn ghost" aria-label={`Download ${label} as JSON`}
        onClick={() => downloadJSON(json ?? rows, `${name}.json`)}>⬇ JSON</button>
    </span>
  );
}
