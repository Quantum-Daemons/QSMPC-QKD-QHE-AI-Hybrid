/**
 * The sidebar's group label meets 4.5:1 and links in running text are underlined (Round 84).
 *
 * WHY THIS EXISTS
 *
 * Lighthouse on the deployed site (scratchpad ux82, checklist rows UX36 and UX37):
 *   - `color-contrast`: "USE-CASE SIMULATORS" in the sidebar was #6d7687 on #0a0e1a, 4.2:1 at 10 px,
 *     on every route. The rule gave it `--muted` at opacity 0.7.
 *   - `link-in-text-block`: inline links were #4f9dff in #97a3b6 text, 1.08:1, with no underline, so
 *     only colour told a link from its sentence (SC 1.4.1).
 *
 * What this asserts, from styles.css's own values (no browser):
 *   1. the group label's colour, composited at its opacity over both backgrounds it can sit on (the
 *      sidebar's --bg-2, and --bg, which Lighthouse measured against), is at least 4.5:1;
 *   2. links are underlined by default, the sidebar navigation alone opts out, and nothing else in
 *      the stylesheet switches the underline off for links in text.
 *
 * Run: node web/scripts/check_link_and_label_contrast.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const css = readFileSync(join(WEB, "src/styles.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
/** The declarations of the first top-level rule whose selector is exactly `selector`. */
const rule = (selector) => {
  const re = new RegExp(`(^|\\n)\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`);
  return css.match(re)?.[2] ?? null;
};
const decl = (body, prop) => body?.match(new RegExp(`(^|;|\\s)${prop}\\s*:\\s*([^;]+)`))?.[2].trim() ?? null;
const root = rule(":root");
const varOf = (v) => decl(root, v.replace(/^var\((--[\w-]+)\)$/, "$1"));
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lin = (c) => { const s = c / 255; return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; };
const lum = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const contrast = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const over = (fg, bg, alpha) => fg.map((c, i) => Math.round(alpha * c + (1 - alpha) * bg[i]));
const resolveColor = (v) => (v?.startsWith("var(") ? varOf(v) : v);

// 1. The group label.
const label = rule(".nav .nav-group");
check("label", label != null, "no `.nav .nav-group` rule - this check is inert");
if (label) {
  const fg = hex(resolveColor(decl(label, "color")));
  const alpha = Number(decl(label, "opacity") ?? 1);
  for (const [name, bgVar] of [["the sidebar (--bg-2)", resolveColor(decl(rule(".sidebar"), "background"))],
                               ["the page (--bg)", varOf("--bg")]]) {
    const bg = hex(bgVar);
    const ratio = contrast(over(fg, bg, alpha), bg);
    check("label", ratio >= 4.5, `the 10 px group label is ${ratio.toFixed(2)}:1 on ${name}, under 4.5:1`);
  }
}

// 2. Links.
check("links", decl(rule("a"), "text-decoration") === "underline" || /underline/.test(decl(rule("a"), "text-decoration") ?? ""),
  "links are not underlined by default, so a link in running text differs from it by colour alone");
check("links", decl(rule(".nav a"), "text-decoration") === "none", "the sidebar navigation should opt out of the underline");
const offs = [...css.matchAll(/([^{}]+)\{[^}]*text-decoration\s*:\s*none[^}]*\}/g)].map((x) => x[1].trim());
const allowed = new Set([".nav a", ".skip-link"]);
for (const sel of offs) check("links", allowed.has(sel), `\`${sel}\` switches the underline off; only the navigation and the skip link may`);

if (failures.length) {
  console.error(`check_link_and_label_contrast: ${failures.length} failure(s)`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log("check_link_and_label_contrast: OK - the group label clears 4.5:1 and links in text are underlined");
