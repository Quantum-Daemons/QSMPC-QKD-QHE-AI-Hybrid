"""Public text that claimed more than the demo does, corrected 2026-10-24, and kept corrected.

WHAT WAS CORRECTED, AND THE PRIMARY SOURCE FOR EACH

1. Market-size figures. This project's public material carries none, and the README, the Value page's
   "Market context" table and a `market` line on every use-case page carried analyst forecasts. Two of
   those lines also tied a bank AML workload and a telecom PQC migration to CNSA 2.0, which binds
   national security systems only ("NSA does not mandate requirements on any other entity outside of
   NSS", CNSA 2.0 and Quantum Computing FAQ Ver. 2.2, October 2026).

2. The novelty sentence said the chain ran "PQC key establishment, then QKD-derived key material, then
   ...". In the browser chain it describes, `bb84()` returns a COUNT and never key bytes, and the agent
   panel mixes a fixed constant. It is one sentence in three files (README, Value page, technical
   report), which have to say the same thing.

3. Attributions the sources do not support:
   - BSI was credited with the QKD add-on. TR-02102-1 v2026-01 section 2.2 recommends a quantum-safe
     plus CLASSICAL hybrid and does not mention QKD; section 2.1 does not recommend QKD at all.
   - "Cloudflare commits" - its post is titled "Cloudflare targets 2029 for full post-quantum
     security" and calls its milestones "subject to change".
   - "a FAR rule putting contractors on the 2030 date" - EO 14412 section 6(c) directs a PROPOSED rule
     within 180 days.
   - "NIST IR 8547 deprecates RSA-2048 and ECC-256 by 2030" - the initial public draft deprecates
     112-bit strength after 2030; P-256 (128-bit) is only disallowed after 2035.
   - "validated for finance, healthcare, genomics and telecom/defence" - nothing is validated in any
     of those settings, and the same page says the quantum channel is "simulated, not
     hardware-validated".
   - CNSA 2.0 "procurement that non-compliant vendors can't enter" and "be compliant before bids" -
     the defaults are not CNSA 2.0 parameter sets.

4. Backend advice the measurement reversed (Round 60). The agent panel called WASM "the slower"
   backend and its timeout note advised "a WebGPU-capable browser". On the machine measured, WebGPU
   ran distilGPT-2 about 80 times slower per token and was the backend that could not finish inside
   the time limit (register, Round 59).

WHY A SOURCE-LEVEL GATE

Each of these sat in more than one file, and this project's most-repeated defect is a fix reaching one
copy of several. Comments are stripped from TypeScript before matching, because the comments recording
each correction quote the wording they replaced.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"


def _strip_ts(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def _public_text(include_logs: bool) -> dict[str, str]:
    files = {"README.md": (REPO / "README.md").read_text()}
    for rel in ("report/technical-report.tex", "references/REFERENCES.md"):
        files[rel] = (REPO / rel).read_text()
    for path in sorted((REPO / "docs").glob("*.md")):
        # The two verification records quote superseded wording on purpose; they are history.
        if not include_logs and path.name in ("verification-log.md", "deliverable-verification.md"):
            continue
        if path.name.startswith("browser-verification-checklist"):
            continue  # gitignored, local only
        files[str(path.relative_to(REPO))] = path.read_text()
    for path in sorted(WEB.rglob("*.ts*")):
        files[str(path.relative_to(REPO))] = _strip_ts(path.read_text())
    return files


_MARKET = [
    r"\$\s?\d[\d.,]*\s?(B|M|bn|billion|million)\b",
    r"\bCAGR\b",
    r"Grand View|Precedence Research|Roots Analysis|MarketsandMarkets|Mordor Intelligence|market\.us",
]


def test_no_public_text_carries_a_market_size_figure() -> None:
    hits = []
    for rel, text in _public_text(include_logs=True).items():
        for pat in _MARKET:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}: {m.group(0)!r}")
    assert not hits, (
        "market-size figures are back in public text; this project's public material carries none:\n  "
        + "\n  ".join(hits[:20]))


def test_no_use_case_carries_a_market_line_again() -> None:
    src = _strip_ts((WEB / "lib" / "useCases.ts").read_text())
    assert not re.search(r"^\s*market\s*[:?]", src, flags=re.M), "useCases.ts has a `market` field again"
    for page in ("pages/UseCases.tsx", "pages/UseCasePage.tsx"):
        assert ".market" not in _strip_ts((WEB / page).read_text()), f"{page} renders a market line again"


_NOVELTY = {
    "README.md": REPO / "README.md",
    "web/src/pages/Value.tsx": WEB / "pages" / "Value.tsx",
    "report/technical-report.tex": REPO / "report" / "technical-report.tex",
}
_LINK = "a simulated QKD link (it yields a key length, not key bytes)"


def test_the_novelty_sentence_names_no_qkd_key_material_in_any_copy() -> None:
    for rel, path in _NOVELTY.items():
        text = re.sub(r"\s+", " ", path.read_text().replace("> ", " "))
        if rel.endswith(".tsx"):
            text = re.sub(r"\s+", " ", _strip_ts(path.read_text()))
        m = re.search(r"PQC key establishment(.{0,160}?)secure multiparty computation", text)
        assert m, f"{rel}: the novelty sentence is gone or reworded past recognition"
        middle = m.group(1)
        assert _LINK in middle, f"{rel}: the chain's QKD step reads {middle!r}, not {_LINK!r}"
        assert "QKD-derived" not in middle, f"{rel}: the chain claims QKD-derived key material again"


_RETIRED = [
    ("hybrid framing here is BSI's", "BSI's hybrid is PQC + classical; it does not recommend QKD"),
    ("on BSI's authority", "BSI does not recommend QKD, as an add-on or otherwise"),
    ("taken from **BSI TR-02102-1 v2026-01 section 2.2**", "section 2.2 does not mention QKD"),
    ("Cloudflare commits", "Cloudflare's post says it targets 2029"),
    ("FAR rule putting contractors", "EO 14412 section 6(c) directs a PROPOSED FAR rule"),
    ("deprecates RSA-2048 and ECC-256", "IR 8547 ipd deprecates 112-bit strength; P-256 is 128-bit"),
    ("validated for finance", "nothing is validated in those settings"),
    ("QKD non-recommended standalone", "NSA answers No to QKD for NSS in any configuration"),
    ("position bids for CNSA 2.0", "the defaults are not CNSA 2.0 parameter sets"),
    ("be compliant before bids", "the defaults are not CNSA 2.0 parameter sets"),
    ("weaker of the two commitments", "neither post ranks itself against the other"),
    ("The hybrid (PQC+QKD) channel protects it today", "QKD is simulated here"),
    ("slower WASM backend", "WASM is the default backend, and the faster one on the machine measured"),
    ("a WebGPU-capable browser", "on the machine measured, WebGPU was the backend that could not finish"),
    # Round 70: the browser's OptiQKD runs coordinate ascent over the analytic decoy-state model;
    # scikit-learn is the full deployment's surrogate.
    ("OptiQKD</b> (scikit-learn surrogate", "the browser optimises the analytic decoy-state model"),
]


def test_the_corrected_attributions_stay_corrected() -> None:
    found = []
    for rel, text in _public_text(include_logs=False).items():
        flat = re.sub(r"\s+", " ", text)
        for phrase, why in _RETIRED:
            if phrase in flat:
                found.append(f"{rel}: {phrase!r} ({why})")
    py = {"src/qsh/pqc/hybrid_kem.py": "NSA/NIST/IETF-recommended",
          "src/qsh/qkd/sdn_controller.py": "sufficiently mature from a security point of view"}
    for rel, phrase in py.items():
        if phrase in re.sub(r"\s+", " ", (REPO / rel).read_text()):
            found.append(f"{rel}: {phrase!r}")
    assert not found, "a corrected claim is back:\n  " + "\n  ".join(found)
