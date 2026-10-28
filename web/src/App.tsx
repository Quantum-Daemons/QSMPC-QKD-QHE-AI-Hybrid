import { Link, Navigate, NavLink, Route, Routes } from "react-router";
import Overview from "./pages/Overview";
import Value from "./pages/Value";
import Validations from "./pages/Validations";
import Orchestration from "./pages/Orchestration";
import UseCases from "./pages/UseCases";
import UseCasePage from "./pages/UseCasePage";
import { USECASES } from "./lib/useCases";
import { useEffect, useState } from "react";
import NotFound from "./components/NotFound";
import { NAV, shortNav } from "./lib/nav";
import { useRunStatus } from "./lib/runStatus";


/** Where the visitor's choice to collapse the sidebar is kept. Per browser and per viewer only: a
 *  convenience, so a blocked or cleared storage just means the default applies again. */
const SIDEBAR_KEY = "qsh.sidebar";
/** Below this width the sidebar starts collapsed when the visitor has not chosen. At 1280 px the
 *  expanded sidebar takes 240 px, and the run pages' diagrams are the widest things on them. */
const NARROW_PX = 1100;
/** Below this width there is no rail: the sidebar is a top bar (styles.css, the 900 px query), so
 *  its links keep their full labels and the run status its full text. */
const RAIL_MIN_PX = 901;

function readSidebarPref(): "collapsed" | "expanded" | null {
  try {
    const v = window.localStorage.getItem(SIDEBAR_KEY);
    return v === "collapsed" || v === "expanded" ? v : null;
  } catch { return null; }
}

function useSidebar() {
  const [pref, setPref] = useState(readSidebarPref);
  const [width, setWidth] = useState(() => window.innerWidth);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const collapsed = width >= RAIL_MIN_PX && (pref ? pref === "collapsed" : width < NARROW_PX);
  const toggle = () => {
    const next = collapsed ? "expanded" : "collapsed";
    setPref(next);
    try { window.localStorage.setItem(SIDEBAR_KEY, next); } catch { /* this visit still toggles */ }
  };
  return { collapsed, toggle };
}

/** The live run's state, in the sidebar since Round 80 (see lib/runStatus.ts). The `role="status"`
 *  element is always rendered, empty while nothing runs, so assistive technology already knows the
 *  region when its first announcement arrives. */
function RunStatusBox({ collapsed }: { collapsed: boolean }) {
  const st = useRunStatus();
  return (
    <div className="run-status" role="status" aria-live="polite">
      {st && (
        <div className="run-status-box" title={st.active ? `${st.state} - ${st.active}` : st.state}>
          <span className={`badge-state ${st.state}`}>{collapsed ? st.state.slice(0, 3) : st.state}</span>
          {collapsed
            ? <span className="sr-only">{st.state}{st.active ? `, stage ${st.active}` : ""}</span>
            : st.active ? <code className="run-stage">{st.active}</code> : null}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const { collapsed, toggle } = useSidebar();
  return (
    <div className={`app${collapsed ? " nav-collapsed" : ""}`}>
      {/* SC 2.4.1 Bypass Blocks, Level A. Must be the FIRST focusable element in the document, or it
          is not a bypass - it would sit behind the block it exists to skip. The `<main id="main">`
          below is the target; the id is added there for this and is used nowhere else. */}
      <a className="skip-link" href="#main">Skip to main content</a>
      <aside className="sidebar">
        <div className="sidebar-head">
          <div className="brand">
            QSH{" "}
            <small>Quantum-Safe Hybrid<br />QKD · QSMPC · QHE/FHE · AI</small>
          </div>
          <button type="button" className="sidebar-toggle" onClick={toggle}
            aria-expanded={!collapsed} aria-controls="sidebar-nav"
            aria-label={collapsed ? "Expand the navigation" : "Collapse the navigation"}
            title={collapsed ? "Expand the navigation" : "Collapse the navigation"}>
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <path d={collapsed ? "M5 2 L10 7 L5 12" : "M9 2 L4 7 L9 12"} fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
        {/* Collapsed, each link shows a short label and keeps its full name as its accessible name
            and tooltip, so the rail reads the same to a screen reader as the expanded list. */}
        <nav className="nav" id="sidebar-nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              aria-label={collapsed ? n.label : undefined} title={collapsed ? n.label : undefined}
              className={({ isActive }) => (isActive ? "active" : "")}>
              {collapsed ? <span className="nav-short" aria-hidden="true">{n.short}</span> : n.label}
            </NavLink>
          ))}
          <div className="nav-group">Use-case simulators</div>
          {USECASES.map((u) => (
            <NavLink key={u.id} to={`/usecase/${u.id}`}
              aria-label={collapsed ? u.nav : undefined} title={collapsed ? u.nav : undefined}
              className={({ isActive }) => (isActive ? "active sub" : "sub")}>
              {collapsed ? <span className="nav-short" aria-hidden="true">{shortNav(u.nav)}</span> : u.nav}
            </NavLink>
          ))}
        </nav>
        <RunStatusBox collapsed={collapsed} />
        <div style={{ marginTop: 28, fontSize: 11 }} className="muted sidebar-note">
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
