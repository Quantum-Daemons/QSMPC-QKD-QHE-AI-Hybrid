import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { NAV } from "../lib/nav";
import { USECASES } from "../lib/useCases";

/**
 * One not-found page for both ways of missing: an unknown path (App's catch-all) and an unknown use-case
 * id (the /usecase/:id route matches, so the catch-all never fires). They were two different pages -
 * one an unstyled heading with a single link, the other a styled list of use cases only - so the same
 * mistake met two answers. Both now list every page and every use case. The two headings stay
 * distinct ("Page not found" / "No such use case"), because the checklist's negative-path rows
 * assert on them.
 */
export default function NotFound({ title, lead }: { title: string; lead: ReactNode }) {
  return (
    <section className="page">
      <h1 className="page-title">{title}</h1>
      <p className="page-sub">{lead}</p>
      <div className="grid cols-2">
        <div className="card">
          <h3>Pages</h3>
          <ul>{NAV.map((n) => <li key={n.to} style={{ marginBottom: 4 }}><Link to={n.to}>{n.label}</Link></li>)}</ul>
        </div>
        <div className="card">
          <h3>Use-case simulators</h3>
          <ul>
            {USECASES.map((u) => (
              <li key={u.id} style={{ marginBottom: 4 }}>
                <Link to={`/usecase/${u.id}`}><code>{u.id}</code></Link>
                <span className="muted"> — {u.title}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
