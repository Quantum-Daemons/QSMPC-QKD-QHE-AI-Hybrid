// The top-level pages, in sidebar order. One list for the sidebar and for the not-found pages, so a
// page added to one cannot be missing from the other. `short` is what the collapsed sidebar shows;
// the full label stays the link's accessible name and its tooltip.
export const NAV = [
  { to: "/", label: "Overview", short: "HOME", end: true },
  { to: "/value", label: "Introduction & Value", short: "INTRO" },
  { to: "/validations", label: "Validations", short: "VALID" },
  { to: "/orchestration", label: "E2E Orchestration", short: "E2E" },
  { to: "/usecases", label: "Use Cases", short: "CASES" },
];

/** The collapsed sidebar's label for a use case: the first word of its sidebar label, three letters
 *  ("Finance - AML" is FIN, "Telecom / Defense" is TEL), so a use case added later gets one too. */
export function shortNav(nav: string): string {
  return nav.split(/[^A-Za-z0-9]+/).filter(Boolean)[0]?.slice(0, 3).toUpperCase() ?? "";
}
