"""Each fact several documents restated has one owner, and the other documents link to it (Round 84).

WHY THIS EXISTS

The Round 83 documentation audit (its section 4, "Duplicates and consolidation") found twelve facts
each restated in three or more places, and this project's most repeated documentation defect is a
correction that reaches some copies of a fact and not others. Round 84 gave each fact one owner,
which holds the complete statement, and reduced every other location to a clause and a link:

- what the browser and the full deployment each protect: one table in `docs/threat-model.md`;
- which pages run CKKS and 2PC, and the QKD key blocks: `docs/models-and-benchmarks.md`;
- what ct/pt measures: `docs/models-and-benchmarks.md`;
- the trained QCNNs, simulable and beaten or tied by logistic regression: `docs/use-cases.md`;
- the telecom forest and the finance gap: `docs/findings.md`;
- NSA's and BSI's positions on QKD: `docs/pqc-roadmap.md`;
- the retired 0-100 score: `references/security-scoring-fact-check.md`;
- the provenance badge's four states: `docs/threat-model.md`, once;
- the WebGPU measurements: `docs/findings.md`, once;
- the full-tier commands and the reverse-proxy runbook: `docs/deployment.md`, once;
- the submodules: one table in `DATA_LICENSES.md`;
- the HElib and OpenFHE quotes: `docs/threat-model.md`, with provenance only in `REFERENCES.md`.

Three facts were wrong where they were restated and are checked here too: the report gave the
trained QCNNs "7 or 8 qubits" (the finance QCNN runs at 4); `docs/deployment.md` named `!reset`
where the override file uses `!override`; and the findings' caveat list said "Three of those" above
four caveats.

WHAT IS ASSERTED

- Every relative Markdown link with an anchor resolves to a heading of its target, by GitHub's anchor
  rule, and there are links to check.
- Each owner holds the facts the other copies used to carry, each non-owner links to the owner, and
  a phrase that only a restatement carried is gone from the non-owners.

ROUND 86: the technical report left this repository for a private one, and the checks on its
copy went with it, unchanged; what stays here checks the copies that are published.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

THREAT = "threat-model.md#what-each-step-protects-in-the-browser-and-in-the-full-deployment"
PAGES = "models-and-benchmarks.md#which-pages-encrypt"
KEYS = "models-and-benchmarks.md#the-qkd-key-blocks"
CTPT = "models-and-benchmarks.md#what-ctpt-measures"
QCNN = "use-cases.md#what-the-trained-qcnns-are"
TELECOM = "findings.md#telecom-the-classical-forest-wins"
FINANCE = "findings.md#finance-the-gap-to-the-published-baseline-is-architectural"
ROADMAP = "pqc-roadmap.md#standards-alignment-done--in-progress"
SCORE = "security-scoring-fact-check.md#why-the-existing-score-had-to-go"
BADGE = "threat-model.md#what-a-results-provenance-badge-says"
PROXY = "deployment.md#demo-tier-behind-a-reverse-proxy-that-already-owns-80-and-443"
SUBMODULES = "DATA_LICENSES.md#6a-git-submodules"


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


def _flat(rel: str) -> str:
    return re.sub(r"\s+", " ", _read(rel))


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation other than '-' and '_' dropped, spaces to '-'."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    out: set[str] = set()
    fence = False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        m = None if fence else re.match(r"#{1,6} +(.*?)\s*#*\s*$", line)
        if m:
            slug = _slug(m.group(1))
            n = 0
            while (slug if n == 0 else f"{slug}-{n}") in out:
                n += 1
            out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def _tracked_markdown() -> list[str]:
    files = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO, check=True, capture_output=True,
                           text=True).stdout.split()
    return [f for f in files if (REPO / f).is_file()]


_LINK = re.compile(r"\]\(([^)\s]*#[^)\s]+)\)")


def test_every_anchor_link_resolves_to_a_heading() -> None:
    checked, bad = 0, []
    for rel in _tracked_markdown():
        src = REPO / rel
        for m in _LINK.finditer(src.read_text()):
            target = m.group(1)
            if re.match(r"[a-z]+://", target):
                continue
            path, anchor = target.split("#", 1)
            dest = src if not path else (src.parent / path).resolve()
            if dest.suffix != ".md":
                continue
            checked += 1
            if not dest.is_file():
                bad.append(f"{rel}: {target} (no such file)")
            elif anchor not in _anchors(dest):
                bad.append(f"{rel}: {target} (no heading with that anchor)")
    assert checked >= 40, f"only {checked} anchored links found; the owners' links are missing"
    assert not bad, "anchored links that GitHub would not resolve:\n  " + "\n  ".join(bad)


def _links_to(rel: str, target: str) -> bool:
    """Does `rel` link to `target` (a path relative to docs/, or a bare file name for references/)?"""
    path, anchor = target.split("#")
    name = Path(path).name
    return any(Path(m.group(1).split("#")[0]).name == name and m.group(1).endswith("#" + anchor)
               for m in _LINK.finditer(_read(rel)))


def test_the_threat_model_holds_one_table_of_what_each_step_protects() -> None:
    text = _read("docs/threat-model.md")
    start = text.index("## What each step protects, in the browser and in the full deployment")
    section = text[start:text.index("\n## ", start + 10)]
    assert "| Step | In the browser (the public demo) | In the full deployment |" in section
    rows = [line.split("|")[1].strip() for line in section.splitlines() if line.startswith("| ")][1:]
    for step in ("Session key", "Agent (MCP) channel", "PSI (finance)", "Secure aggregation (finance, medical)",
                 "Secure sum (genomics)", "Private inference"):
        assert step in rows, f"the table has no row for {step!r}: {rows}"
    flat = re.sub(r"\s+", " ", section)
    # The PSI row's facts are the OPRF PSI's since Round 84 (psi): its RFC and suite, what Bank B
    # learns, and its two limits; the keyed hash it stated no longer runs, so its two facts went. The
    # secure-aggregation fact is the X25519 key agreement since Round 84 (secagg); it read the public
    # constant `sha256(f"pair-{i}-{j}")`, which no longer derives any mask.
    for fact in ("No QKD bit enters it", "RFC 9497's OPRF, mode 0x00, suite ristretto255-SHA512",
                 '"Unconditional input secrecy"', "**No malicious security**", "**Not post-quantum**",
                 "derives its mask seed from their X25519 shared secret by HKDF-SHA256",
                 "Every CKKS slot the key holder can decode", "`masked_slots`"):
        assert fact in flat, f"the owner table no longer states {fact!r}"
    for retired in ("Pinkas, Schneider and Zohner", "OPRF-based protocol", "HMAC-SHA256 of both banks' lists"):
        assert retired not in flat, f"the owner table still describes the retired keyed-hash PSI: {retired!r}"
    for rel in ("README.md", "docs/architecture.md", "docs/findings.md", "docs/use-cases.md",
                "docs/models-and-benchmarks.md"):
        assert _links_to(rel, THREAT), f"{rel} restates what the browser protects without the owner's link"
    for rel in ("docs/findings.md", "docs/use-cases.md", "docs/models-and-benchmarks.md", "docs/architecture.md"):
        assert "hash every candidate ID and recover the other's list" not in _flat(rel), (
            f"{rel} restates the PSI's exposure; one clause and a link to the threat model")


def test_which_pages_encrypt_has_one_owner() -> None:
    owner = _flat("docs/models-and-benchmarks.md")
    assert "#### Which pages encrypt" in _read("docs/models-and-benchmarks.md")
    assert "the satellite page runs neither" in owner and "under CKKS (TenSEAL)" in owner
    for rel in ("docs/architecture.md", "docs/findings.md", "docs/use-cases.md", "DATA_LICENSES.md"):
        assert _links_to(rel, PAGES), f"{rel} says which pages encrypt without the owner's link"
    assert "the satellite page runs no CKKS step" not in _flat("DATA_LICENSES.md")


def test_the_trained_qcnns_have_one_owner() -> None:
    owner = _flat("docs/use-cases.md")
    assert "## What the trained QCNNs are" in _read("docs/use-cases.md")
    for fact in ("finance at 4 wires", "0.9705 against 0.9537", "Section IV.B"):
        assert fact in owner, fact
    assert _links_to("docs/findings.md", QCNN)


def test_the_telecom_forest_and_the_finance_gap_have_one_owner() -> None:
    findings = _read("docs/findings.md")
    for heading in ("### Telecom: the classical forest wins",
                    "### Finance: the gap to the published baseline is architectural"):
        assert heading in findings, heading
    flat = re.sub(r"\s+", " ", findings)
    for fact in ("`quantum_beats_classical: false`", "of 200 trees", "1.45x AUPRC", "to 58.60",
                 "per-transaction subgraph patterns", "bit-identical whether or not later rows exist"):
        assert fact in flat, fact
    assert "Four of those deserve their caveat" in flat and "Three of those deserve" not in flat
    for rel in ("docs/use-cases.md", "docs/architecture.md"):
        assert _links_to(rel, TELECOM), rel
        assert "equals the classical product kernel" not in _flat(rel), rel
    roadmap = _read("docs/pqc-roadmap.md")
    assert "The finance AML feature architecture" not in roadmap, "the roadmap restates the finance gap again"
    assert _links_to("docs/pqc-roadmap.md", FINANCE)


def test_nsa_and_bsi_on_qkd_have_one_owner() -> None:
    owner = _flat("docs/pqc-roadmap.md")
    for fact in ("Can I use a QKD system to protect my national security system from a quantum computer?",
                 "does not support the usage of QKD or QC to protect communications in National Security Systems",
                 "Key Derivation and Hybridisation", "should remain secure as long as one of the methods used is secure"):
        assert fact in owner, fact
    for rel in ("docs/threat-model.md", "docs/findings.md", "docs/crypto-agility.md"):
        assert "currently not ready for use from a security point of view" not in _flat(rel), (
            f"{rel} quotes BSI again; one clause and a link to the roadmap")
        assert _links_to(rel, ROADMAP), rel


def test_ctpt_has_one_owner() -> None:
    owner = _flat("docs/models-and-benchmarks.md")
    for fact in ("`ciphertext_bytes / (n_in * 8)`", "about 5222x", "about 1135x", "`cost_per_op.demo_ct_pt_ratio`"):
        assert fact in owner, fact
    for rel in ("docs/findings.md", "docs/architecture.md"):
        assert "1135x" not in _read(rel) and "5222x" not in _read(rel), f"{rel} restates the ct/pt figures"
        assert _links_to(rel, CTPT), rel


def test_the_retired_score_has_one_owner() -> None:
    note = _flat("references/security-scoring-fact-check.md")
    for fact in ("`0.0128`", "assumed 256 bits", "awarded a perfect 100/100", "**Where the composite stands now.**",
                 "$`30 + 30 + 20 + 10 + 10 = 100`$"):
        assert fact in note, fact
    for rel in ("docs/use-cases.md", "docs/models-and-benchmarks.md", "docs/findings.md", "docs/crypto-agility.md"):
        flat = _flat(rel)
        assert not re.search(r"(16|[Ss]ixteen) of the 28", flat), f"{rel} restates the retired score's gap"
        assert _links_to(rel, SCORE), rel


def test_the_badge_the_webgpu_figures_and_the_runbook_are_stated_once() -> None:
    # The four states are listed once, in the badge's own section.
    assert _read("docs/threat-model.md").count("`MEASURED (local WebGPU)`") == 1
    assert "measured-js" not in _read("docs/findings.md") and _links_to("docs/findings.md", BADGE)
    assert _read("docs/findings.md").count("12842") == 1, "the WebGPU figures are stated more than once"
    deployment = _read("docs/deployment.md")
    assert deployment.count("docker compose -f docker/docker-compose.yml exec api") == 1
    assert "pqcqkd_mgmt-net" in deployment and "pqcqkd_mgmt-net" not in _read("docs/findings.md")
    assert _links_to("docs/findings.md", PROXY)
    override = _read("docker/docker-compose.demo.behind-proxy.yml")
    tag = re.search(r"ports:\s*(!\w+)", override).group(1)
    other = {"!override": "!reset", "!reset": "!override"}[tag]
    assert f"`{tag}`" in deployment and other not in deployment, (
        f"the override file uses {tag}; docs/deployment.md must name that tag and not {other}")


def test_the_submodules_have_one_table() -> None:
    licences = _read("DATA_LICENSES.md")
    rows = [line for line in licences.splitlines() if line.startswith("| `submodules/HEonGPU`")]
    assert len(rows) == 1, f"the HEonGPU submodule has {len(rows)} rows in DATA_LICENSES.md"
    assert "chart21/pygeon" in licences and "Do not initialise submodules recursively" in licences
    dev = _read("docs/development.md")
    assert "**submodule**" not in dev and _links_to("docs/development.md", SUBMODULES)


def test_the_key_blocks_and_the_ckks_quotes_have_one_owner() -> None:
    owner = _flat("docs/models-and-benchmarks.md")
    for fact in ("0.56 of the 629,000 under Hoeffding and 0.69 under Chernoff", "about 7 s", "33 s",
                 "5,214,327", "450,923"):
        assert fact in owner, fact
    assert _links_to("README.md", KEYS)
    refs = _read("references/REFERENCES.md")
    assert "868,462" not in refs, "REFERENCES.md restates the model's key; it keeps what each source grounds"
    for quote in ("N_d = 128", "sqrt(D)", "since the scheme itself adds some error"):
        assert quote not in refs, f"REFERENCES.md quotes {quote!r}; the quote lives in the threat model"
        assert quote in _read("docs/threat-model.md"), quote
