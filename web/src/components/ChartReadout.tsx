import { createContext, useContext, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/**
 * A fixed line under a chart that shows the point being read, in place of a floating tooltip.
 *
 * WHY (Round 74, by the user's decision). WCAG 2.2 SC 1.4.13 asks that content shown on hover can
 * itself be hovered. Recharts' tooltip follows the pointer with `pointer-events: none`, so it never
 * can be. A line in the page flow is not hover content at all: it is always there, covers nothing,
 * and keeps the last point read for a screen magnifier, a touch screen or a reader who paused. As a
 * polite live region it also announces each point a keyboard user arrows to; a chart surface is
 * `role="application"`, where a floating box is not read aloud.
 *
 * HOW IT IS WIRED. Recharts finds a chart's tooltip by component type, and its keyboard layer
 * (`accessibilityLayer`) stops working without one, so every chart still carries a `<Tooltip>`, with
 * `content={<ReadoutTooltip />}`. Recharts calls that with the active payload and the Tooltip's own
 * `formatter` and `labelFormatter`; it renders the values through a portal into this component's
 * line instead of beside the pointer, so the floating box stays empty. The highlight recharts draws
 * on the active category is kept, and hidden once the chart has neither hover nor focus
 * (`styles.css`).
 */

interface ReadoutContextValue {
  el: HTMLElement | null;
  hint: string;
  last: { current: ReactNode };
}

const ReadoutContext = createContext<ReadoutContextValue | null>(null);

export const HINT_KEYBOARD = "Hover the chart, or focus it and press the arrow keys, to read a point here.";
export const HINT_POINTER = "Hover a point to read its values here.";

export function ChartReadout({ hint, children }: { hint: string; children: ReactNode }) {
  const [el, setEl] = useState<HTMLParagraphElement | null>(null);
  const last = useRef<ReactNode>(null);
  return (
    <ReadoutContext.Provider value={{ el, hint, last }}>
      <div className="chart-with-readout">
        {children}
        <p ref={setEl} className="chart-readout" role="status" />
      </div>
    </ReadoutContext.Provider>
  );
}

type Entry = { name?: unknown; value?: unknown; unit?: unknown; dataKey?: unknown; payload?: unknown };
type Formatter = (value: unknown, name: unknown, entry: Entry, index: number, payload: Entry[]) => unknown;

/** A number as the tooltip would have shown it, without sixteen digits of float. */
function show(v: unknown): string {
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Number.isInteger(v) ? v.toLocaleString("en-US") : String(Number(v.toPrecision(6)));
  }
  return String(v ?? "");
}

/** The Tooltip `content` that writes into the readout. Recharts supplies every prop. */
export function ReadoutTooltip(props: {
  active?: boolean; payload?: Entry[]; label?: unknown; formatter?: Formatter;
  labelFormatter?: (label: unknown, payload: Entry[]) => ReactNode; labelName?: string;
}) {
  const ctx = useContext(ReadoutContext);
  if (!ctx?.el) return null;
  const { active, payload, label, formatter, labelFormatter, labelName } = props;
  if (active && payload?.length) {
    const items = payload.map((entry, i) => {
      const out = formatter ? formatter(entry.value, entry.name, entry, i, payload) : entry.value;
      const [value, name] = Array.isArray(out) ? out : [out, entry.name];
      return `${show(name)} ${show(value)}${entry.unit ? String(entry.unit) : ""}`;
    });
    const head = label == null || label === "" ? null
      : labelFormatter ? labelFormatter(label, payload) : `${labelName ? `${labelName} ` : ""}${show(label)}`;
    ctx.last.current = (
      <>
        {head != null ? <b>{head}: </b> : null}
        {items.join(" · ")}
      </>
    );
  }
  return createPortal(ctx.last.current ?? <span className="muted">{ctx.hint}</span>, ctx.el);
}
