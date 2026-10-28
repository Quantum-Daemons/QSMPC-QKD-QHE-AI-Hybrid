// No JSX line break silently deletes the space between a value and the word after it.
//
// WHY THIS EXISTS
//
// JSX drops whitespace that contains a line break wherever it touches an expression or an element.
// So text wrapped after `{value}` renders glued to it. The calibration caption read
// "Widest deviation 0.644 at predicted 0.644against observed 0.000", which was measured on the
// deployed finance page in Round 74. The source looked right, with a line break where the space
// was meant to be. tsc and the build were green, and the caption is the chart's only text
// alternative, which a screen reader reads aloud as written.
//
// The authors who knew the rule left a space inside the next node instead (`<b> out of scope</b>`,
// `{cond ? " The live run..." : ""}`) or wrote `{" "}`. This check accepts all of those. It reports
// only text that meets an inline value across a line break with no space on either side. Text that
// opens with punctuation is left alone, because ", and" is meant to attach.
//
// Run: node web/scripts/check_jsx_text_spaces.mjs   (npm run check:jsx-spaces; CI gate)

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(WEB, "src");

/** Elements whose content flows inside a line, so a lost space shows. */
const INLINE = new Set(["a", "abbr", "b", "cite", "code", "dfn", "em", "i", "kbd", "mark", "q", "s",
  "small", "span", "strong", "sub", "sup", "time", "u", "var", "Link", "StageBadge", "ProvenanceBadge"]);
const PUNCT_START = /^[.,;:!?)\]'"\u2019]/;

const tagOf = (n) => ts.isJsxElement(n) ? n.openingElement.tagName.getText()
  : ts.isJsxSelfClosingElement(n) ? n.tagName.getText() : null;

/** Renders nothing: an empty expression (a comment) or a literal null/false/"". */
function rendersNothing(e) {
  if (!e) return true;
  if (ts.isParenthesizedExpression(e)) return rendersNothing(e.expression);
  return e.kind === ts.SyntaxKind.NullKeyword || e.kind === ts.SyntaxKind.FalseKeyword
    || (ts.isIdentifier(e) && e.text === "undefined")
    || ((ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) && e.text === "");
}

/** Whether a node's rendered text begins (side 0) or ends (side 1) with a space that survives. */
function edgeSpace(n, side) {
  if (!n) return false;
  if (ts.isJsxText(n)) {
    return side === 0 ? /^[ \t]+\S/.test(n.text) : /\S[ \t]+$/.test(n.text);
  }
  if (ts.isJsxElement(n) || ts.isJsxFragment(n)) {
    const kids = n.children.filter((k) => !(ts.isJsxText(k) && k.containsOnlyTriviaWhiteSpaces)
      && !(ts.isJsxExpression(k) && !k.expression));
    return edgeSpace(side === 0 ? kids[0] : kids[kids.length - 1], side);
  }
  if (ts.isJsxExpression(n)) return exprSpace(n.expression, side);
  return false;
}

function exprSpace(e, side) {
  if (!e) return false;
  if (ts.isParenthesizedExpression(e)) return exprSpace(e.expression, side);
  if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) {
    return side === 0 ? /^\s/.test(e.text) : /\s$/.test(e.text);
  }
  if (ts.isTemplateExpression(e)) {
    return side === 0 ? /^\s/.test(e.head.text)
      : /\s$/.test(e.templateSpans[e.templateSpans.length - 1].literal.text);
  }
  if (ts.isConditionalExpression(e)) {
    return [e.whenTrue, e.whenFalse].every((b) => rendersNothing(b) || exprSpace(b, side));
  }
  if (ts.isBinaryExpression(e) && e.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken) {
    return exprSpace(e.right, side);
  }
  if (ts.isJsxElement(e) || ts.isJsxFragment(e) || ts.isJsxSelfClosingElement(e)) return edgeSpace(e, side);
  return false;
}

/** An inline value: an expression that renders something, or an inline element. */
function isInline(n) {
  if (!n) return false;
  if (ts.isJsxExpression(n)) {
    return !rendersNothing(n.expression);
  }
  const tag = tagOf(n);
  return tag !== null && INLINE.has(tag);
}

function* tsxFiles(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* tsxFiles(p);
    else if (e.name.endsWith(".tsx")) yield p;
  }
}

const found = [];
let texts = 0;
for (const file of tsxFiles(SRC)) {
  const sf = ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true,
    ts.ScriptKind.TSX);
  const where = (node) => `${relative(WEB, file)}:${sf.getLineAndCharacterOfPosition(node.getStart()).line + 1}`;
  const visit = (node) => {
    if (ts.isJsxElement(node) || ts.isJsxFragment(node)) {
      // Comments and line-break-only whitespace render nothing, so the text meets whatever lies past
      // them: `{value}` then a comment then text on the next line is the same defect.
      const kids = node.children.filter((k) => !(ts.isJsxText(k) && k.containsOnlyTriviaWhiteSpaces)
        && !(ts.isJsxExpression(k) && !k.expression));
      kids.forEach((k, i) => {
        if (!ts.isJsxText(k)) return;
        texts += 1;
        const raw = k.text;
        const prev = kids[i - 1];
        const next = kids[i + 1];
        const after = raw.match(/^[ \t]*\n\s*(\S)/);
        if (after && !PUNCT_START.test(after[1]) && isInline(prev) && !edgeSpace(prev, 1)) {
          found.push(`${where(k)}: "${prev.getText().slice(-30)}" runs into "${raw.trim().slice(0, 30)}"`);
        }
        if (/\S[ \t]*\n\s*$/.test(raw) && isInline(next) && !edgeSpace(next, 0)) {
          found.push(`${where(next)}: "${raw.trim().slice(-30)}" runs into "${next.getText().slice(0, 30)}"`);
        }
        // Two text nodes in a row means a comment sat between them, on its own line: both sides are
        // trimmed at the line break and the words on either side are rendered touching.
        const nextLead = next && ts.isJsxText(next) && next.text.match(/^[ \t]*\n\s*(\S)/);
        if (nextLead && !PUNCT_START.test(nextLead[1]) && /\S[ \t]*\n\s*$/.test(raw)) {
          found.push(`${where(next)}: "${raw.trim().slice(-30)}" runs into "${next.text.trim().slice(0, 30)}" across a comment`);
        }
      });
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

if (texts < 200) {
  console.error(`only ${texts} JSX text nodes examined - did web/src move? A scan of nothing passes.`);
  process.exit(1);
}
if (found.length) {
  console.error("JSX text that loses its space at a line break (it renders glued to the value):");
  for (const f of found) console.error(`  ${f}`);
  console.error("Write the space where it survives: `{value}{\" \"}`, or keep the text on the value's line.");
  process.exit(1);
}
console.log(`PASS  ${texts} JSX text nodes: none loses its space at a line break`);
