import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Value from "./pages/Value";
import Validations from "./pages/Validations";
import AgenticAI from "./pages/AgenticAI";
import Orchestration from "./pages/Orchestration";
import UseCases from "./pages/UseCases";
import UseCasePage from "./pages/UseCasePage";
import { USECASES } from "./lib/useCases";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/value", label: "Introduction & Value" },
  { to: "/validations", label: "Validations" },
  { to: "/agentic-ai", label: "Agentic AI · MCP" },
  { to: "/orchestration", label: "E2E Orchestration" },
  { to: "/usecases", label: "Use Cases" },
];

export default function App() {
  return (
    <div className="app">
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
          CNSA 2.0 · HNDL / Q-Day aware
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/value" element={<Value />} />
          <Route path="/validations" element={<Validations />} />
          <Route path="/agentic-ai" element={<AgenticAI />} />
          <Route path="/orchestration" element={<Orchestration />} />
          <Route path="/usecases" element={<UseCases />} />
          <Route path="/usecase/:id" element={<UseCasePage />} />
        </Routes>
      </main>
    </div>
  );
}
