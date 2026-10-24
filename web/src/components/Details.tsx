import type { ReactNode } from "react";

/**
 * A folded block for METHOD, HISTORY and PROVENANCE - the reading that a first-time visitor can skip.
 *
 * Never for a caveat that qualifies a number shown above it: that stays in the same visual block as
 * the number (checklist UX6), and so do the EU AI Act Art. 50 disclosure, a verification FAIL note and
 * the security-fact warnings. The summary must say what is inside - two "Details" summaries on one
 * page read to a screen reader as duplicate headings (checklist SW2).
 */
export default function Details({ summary, children, open = false }: {
  summary: string; children: ReactNode; open?: boolean;
}) {
  return (
    <details className="details" open={open}>
      <summary>{summary}</summary>
      <div className="details-body">{children}</div>
    </details>
  );
}
