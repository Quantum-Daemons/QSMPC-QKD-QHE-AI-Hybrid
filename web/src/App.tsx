import { Link, Navigate, NavLink, Route, Routes } from "react-router";
import Overview from "./pages/Overview";
import Value from "./pages/Value";
import Validations from "./pages/Validations";
import Orchestration from "./pages/Orchestration";
import UseCases from "./pages/UseCases";
import UseCasePage from "./pages/UseCasePage";
import { USECASES } from "./lib/useCases";
import NotFound from "./components/NotFound";
import { NAV } from "./lib/nav";


export default function App() {
  return (
    <div className="app">
      {/* SC 2.4.1 Bypass Blocks, Level A. Must be the FIRST focusable element in the document, or it
          is not a bypass - it would sit behind the block it exists to skip. The `<main id="main">`
          below is the target; the id is added there for this and is used nowhere else. */}
      <a className="skip-link" href="#main">Skip to main content</a>
      <aside className="sidebar">
        <div className="brand">
          QSH
          <small>Quantum-Safe Hybrid<br />QKD · QSMPC · QHE/FHE · AI</small>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => (isActive ? "active" : "")}>
              {n.label}
            </NavLink>
          ))}
          <div className="nav-group">Use-case simulators</div>
          {USECASES.map((u) => (
            <NavLink key={u.id} to={`/usecase/${u.id}`}
              className={({ isActive }) => (isActive ? "active sub" : "sub")}>
              {u.nav}
            </NavLink>
          ))}
        </nav>
        <div style={{ marginTop: 28, fontSize: 11 }} className="muted">
          PQC-first hybrid · NIST FIPS 203/204<br />
          {/* Was "CNSA 2.0 · HNDL / Q-Day aware", which reads as a compliance badge. */}
          Tracks CNSA 2.0 (defaults are not CNSA 2.0) · HNDL aware
        </div>
      </aside>
      {/* `tabIndex={-1}` so the skip link can actually MOVE focus here. Without it the browser
          scrolls to the anchor and leaves focus on the link, so the next Tab returns to the second
          sidebar item - the block was skipped visually and not for the keyboard, which is the half
          of this fix that is easy to ship broken. */}
      <main className="main" id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/value" element={<Value />} />
          <Route path="/validations" element={<Validations />} />
          {/* Agentic AI · MCP is now integrated INTO each use-case page (the report frames MCP as a
              per-use-case cross-cutting layer); redirect the old standalone route to the use-case index. */}
          <Route path="/agentic-ai" element={<Navigate to="/usecases" replace />} />
          <Route path="/orchestration" element={<Orchestration />} />
          <Route path="/usecases" element={<UseCases />} />
          <Route path="/usecase/:id" element={<UseCasePage />} />
          {/* Without this, an unknown path rendered the sidebar next to an empty <main>, which reads
              as a broken page rather than a wrong address. */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </main>
    </div>
  );
}

function NotFoundPage() {
  return (
    <NotFound title="Page not found"
      lead={<>That address does not match any page in this app. The use-case simulators live under{" "}
        <code>/usecase/&lt;id&gt;</code>.</>} />
  );
}
