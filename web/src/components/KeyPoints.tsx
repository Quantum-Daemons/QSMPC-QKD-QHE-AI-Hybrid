import type { ReactNode } from "react";

/** The few facts a section exists to say, as a list - placed before any prose that explains them. */
export default function KeyPoints({ items, title }: { items: ReactNode[]; title?: string }) {
  return (
    <div className="key-points">
      {title && <div className="key-points-title">{title}</div>}
      <ul>{items.map((it, i) => <li key={i}>{it}</li>)}</ul>
    </div>
  );
}
