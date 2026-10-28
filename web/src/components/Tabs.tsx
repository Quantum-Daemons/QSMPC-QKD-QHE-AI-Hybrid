import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

/**
 * Tabs, as the WAI-ARIA Authoring Practices describe them: a `tablist` of `tab` buttons, one visible
 * `tabpanel`, arrow keys moving between tabs (Home and End to the ends), focus following selection.
 * Added in Round 77 for the satellite card, whose Downlink and Network views no longer fit one panel.
 * Inactive panels are not rendered, so a heavy view (a 3D scene, a pass integration) costs nothing
 * until its tab is opened.
 */
export default function Tabs({ tabs, label, initial = 0 }: {
  tabs: { id: string; title: string; content: () => ReactNode }[];
  label: string;
  initial?: number;
}) {
  const [active, setActive] = useState(initial);
  const base = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const move = (i: number) => {
    const next = (i + tabs.length) % tabs.length;
    setActive(next);
    refs.current[next]?.focus();
  };
  const onKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    if (e.key === "ArrowRight") { e.preventDefault(); move(i + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); move(i - 1); }
    else if (e.key === "Home") { e.preventDefault(); move(0); }
    else if (e.key === "End") { e.preventDefault(); move(tabs.length - 1); }
  };
  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="tab-list">
        {tabs.map((t, i) => (
          <button key={t.id} type="button" role="tab" id={`${base}-tab-${t.id}`}
            aria-selected={i === active} aria-controls={`${base}-panel-${t.id}`} tabIndex={i === active ? 0 : -1}
            className={`tab${i === active ? " tab-active" : ""}`} ref={(el) => { refs.current[i] = el; }}
            onClick={() => setActive(i)} onKeyDown={(e) => onKey(e, i)}>
            {t.title}
          </button>
        ))}
      </div>
      {tabs.map((t, i) => (
        <div key={t.id} role="tabpanel" id={`${base}-panel-${t.id}`} aria-labelledby={`${base}-tab-${t.id}`}
          hidden={i !== active} className="tab-panel">
          {i === active ? t.content() : null}
        </div>
      ))}
    </div>
  );
}
