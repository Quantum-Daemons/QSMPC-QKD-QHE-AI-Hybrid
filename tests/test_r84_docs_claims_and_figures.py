"""Round 84 documentation defects, corrected, and kept corrected (audit of 2026-10-31).

WHAT WAS WRONG, AND THE SOURCE FOR EACH

1. The use-case flowcharts (README Mermaid and their TikZ twins in the report) gated BB84 on "QBER at
   most 0.11", while the code decides on what the run produced: `qkdOk` in
   `web/src/lib/useCaseFlows.ts` reads `secure_key_bits > 0`, for BB84 and E91 alike. They also drew
   "Explainability (full deployment)" as a dashed box on three pages and not at all on two, while the
   browser computes exact linear SHAP on all five (`linearShap` in `web/src/lib/clientCompute.ts`)
   and signs it with the result (`signed_fields` includes "explanation").
2. The report's crypto table said Polytune runs "ReLU, argmax"; in the browser it runs the argmax
   only, and ReLU and pooling run in plaintext between the medical CNN's encrypted layers. The models
   table printed a dash, which its caption calls "not trained", for the published and pinned
   `telecom_ids.qml`, because the generator read `macro_f1` and that record stores the forest's figure
   as `classical_rf_macro_f1`.
3. Stale figures and library facts, each read at its primary source on 2026-10-31: IACR ePrint
   2026/935 revised 2026-10-23 ("only 5 out of the 21 surveyed frameworks are model-preserving");
   Microsoft SEAL 4.7.2 (2026-10-28) while node-seal 7.0.0 is still aligned with SEAL 4.1.2; liboqs
   0.16.0, the Docker pin, enables HQC by default and builds SLH-DSA (its release notes and
   `.CMake/alg_support.cmake`); OpenSSL 3.5 implements X25519MLKEM768 natively, RFC 10024 (August
   2026) standardises it, and oqs-provider calls itself a testing vehicle and removed composite
   signatures in 0.10.0; TenSEAL's `dot` refuses to run without rotation (Galois) keys; the fibre key
   of the default block is 5,214,327 bits at 50 km; OPS-SAT-AD's concept DOI resolves to version 2,
   whose files differ from the version 1 this project uses; TabPFN-3 handles up to 1M rows under a
   non-commercial licence.
4. Licence claims: the 1000 Genomes slice the browser serves has no licence, and PGS000004's source
   paper is CC BY-NC-ND 4.0 (under review); the five removed PDFs were purged from the history.
5. Seven private checklist row IDs were cited in public reference notes, where no reader can resolve
   them; plain-text formulas sat outside the house MathJax form; two code files' docstrings and
   comments carried non-ASCII characters.

Each test below fails on the base commit of Round 84 (de1111c) and passes after, checked by running
this file against an export of that commit's tree.

ROUND 86: the technical report left this repository for a private one, and the checks on its
copy went with it, unchanged; what stays here checks the copies that are published.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

EXPLAIN = "Explainability: exact linear SHAP, signed"
UC_KEYS = ("uc-finance_aml", "uc-medical_fl", "uc-genomics", "uc-telecom_ids", "uc-satellite_comms")


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


def _readme_figures() -> dict[str, str]:
    text = README.read_text()
    return {m.group(1): m.group(2) for m in
            re.finditer(r"<!-- figure: ([\w-]+) -->\s*```mermaid\n(.*?)```", text, flags=re.S)}


def _ts_function(path: Path, name: str) -> str:
    src = path.read_text()
    start = src.index(f"const {name} = ")
    return src[start:src.index("\n};", start)]


def test_the_flowcharts_gate_bb84_on_the_key_bits_the_code_reads() -> None:
    gate = _ts_function(REPO / "web" / "src" / "lib" / "useCaseFlows.ts", "qkdOk")
    assert re.search(r"secure_key_bits", gate) and re.search(r"bits\s*>\s*0", gate), (
        "qkdOk no longer decides on secure_key_bits > 0; re-derive the flowcharts' gate from the code")
    readme = _readme_figures()
    for key in ("uc-finance_aml", "uc-medical_fl", "uc-genomics", "uc-telecom_ids"):
        assert "QBER at most" not in readme[key], f"README {key} still gates BB84 on a QBER threshold"
        assert "BB84 and E91: secure key bits above 0" in readme[key], (
            f"README {key} does not state the gate the code applies")


def test_every_use_case_figure_draws_the_explanation_the_browser_signs() -> None:
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert cc.count("artifacts.explanation = linearShap(") >= 3, (
        "the browser no longer computes linear SHAP at the telecom, satellite and private-inference sites")
    assert re.search(r'signed_fields:\s*\[[^\]]*"explanation"', cc), "the explanation is no longer signed"
    readme = _readme_figures()
    for key in UC_KEYS:
        assert EXPLAIN in readme[key], f"README {key} does not draw the browser's explanation"
        assert "-.->" not in readme[key] and "classDef full" not in readme[key], (
            f"README {key} still draws a full-deployment-only (dashed) stage")


#: (file, pattern) - wording each correction replaced. Patterns allow a line break wherever the
#: text wraps, so a reflowed paragraph cannot hide a claim from this list.
_CORRECTED = [
    ("README.md", r"2PC for the non-linear part"),
    ("docs/architecture.md", r"no rotations"),
    ("docs/architecture.md", r"MPyC or MP-SPDZ"),
    ("docs/architecture.md", r"slot-0-only masking"),
    ("docs/architecture.md", r"WASM/WebGPU PQC via @noble"),
    ("src/qsh/fhe/conv.py", r"no\s+ciphertext rotations"),
    ("src/qsh/fhe/ckks_inference.py", r"square activation"),
    ("docs/threat-model.md", r"tracks\s+Microsoft SEAL"),
    ("docs/threat-model.md", r"pin versions \+ review"),
    ("docs/threat-model.md", r"nothing has been adopted"),
    ("docs/pqc-roadmap.md", r"not in this liboqs build"),
    ("docs/pqc-roadmap.md", r"disables HQC by default since"),
    ("docs/pqc-roadmap.md", r"liboqs disables it by default"),
    ("docs/pqc-roadmap.md", r"~20 CKKS frameworks"),
    ("docs/development.md", r"SLH-DSA is not in this liboqs build"),
    ("docs/crypto-agility.md", r"the production counterpart to this registry"),
    ("docs/crypto-agility.md", r"composite PQC\s+certificates"),
    ("docs/deployment.md", r"Every\s+table in it is generated"),
    ("docs/models-and-benchmarks.md", r"about 5\.1 Mbit"),
    ("docs/models-and-benchmarks.md", r"the same fraction twice"),
    ("docs/models-and-benchmarks.md", r"permits redistribution since Round 84"),
    ("docs/models-and-benchmarks.md", r"slot-0-only masking"),
    ("docs/findings.md", r"permits redistribution since the satellite sample"),
    ("docs/use-cases.md", r"real small CNN on \*\*real MedMNIST images\*\* with its convolution"),
    ("docs/use-cases.md", r"citing WASM threading"),
    ("DATA_LICENSES.md", r"with the same\s+files"),
    ("DATA_LICENSES.md", r"only the bundled copy is gone"),
    ("DATA_LICENSES.md", r"The registry's label for it"),
    ("DATA_LICENSES.md", r"source paper's own\s+terms were not verified"),
    ("references/REFERENCES.md", r"20% of surveyed implementations"),
    ("references/REFERENCES.md", r"2026/935\*\*, revised 2026-06-03"),
    ("references/REFERENCES.md", r"returns no licence at all"),
    ("references/encrypted-inference-performance.md", r"\| finance \| 700 ms \| yes"),
    ("references/model-choice-vs-2026-practice.md", r"stopped being true in Jan 2025"),
]


def test_corrected_claims_stay_corrected() -> None:
    found = [f"{rel}: /{pat}/" for rel, pat in _CORRECTED if re.search(pat, _read(rel))]
    assert not found, "wording a Round 84 correction replaced is back:\n  " + "\n  ".join(found)


def test_the_corrected_facts_are_stated() -> None:
    expect = [
        ("references/REFERENCES.md", r"5 of the 21 surveyed frameworks"),
        ("docs/pqc-roadmap.md", r"only 5 of them model-preserving"),
        ("docs/pqc-roadmap.md", r"enables\s+`KEM_hqc_1/3/5` by default"),
        ("docs/crypto-agility.md", r"RFC 10024"),
        ("docs/models-and-benchmarks.md", r"5,214,327"),
        ("DATA_LICENSES.md", r"12588359"),
        ("DATA_LICENSES.md", r"\*\*version 2\*\*, record 15108715"),
        ("DATA_LICENSES.md", r"\*\*CC BY-NC-ND 4\.0\*\* \(its version of record"),
        ("DATA_LICENSES.md", r"pending a\s+request to GitHub Support"),
        ("docs/architecture.md", r"rotation \(Galois\) keys"),
        ("src/qsh/fhe/conv.py", r"rotation \(Galois\) keys"),
        ("references/model-choice-vs-2026-practice.md", r"TabPFN-3"),
    ]
    missing = [f"{rel}: /{pat}/" for rel, pat in expect if not re.search(pat, _read(rel))]
    assert not missing, "a corrected fact is no longer stated:\n  " + "\n  ".join(missing)


#: Code spans of ID shape that are not checklist rows: a KEM shared secret and a vacuum yield.
_NOT_ROWS = {"K1", "Y0"}


def test_reference_notes_cite_no_private_checklist_row() -> None:
    """The verification checklist is not published (`.gitignore`), so a row ID in a public note is a
    reference nobody can follow. State the check's content instead."""
    bad = []
    for path in sorted((REPO / "references").glob("*.md")):
        text = path.read_text()
        for m in re.finditer(r"`([A-Z]{1,3}[0-9]{1,3}[a-z]?)`", text):
            if m.group(1) not in _NOT_ROWS:
                bad.append(f"{path.name}: `{m.group(1)}`")
        for m in re.finditer(r"[Cc]hecklist (?:row|item)s?\s+`?[A-Z]{1,3}[0-9]|survives S3\b|^S3 applied",
                             text, flags=re.M):
            bad.append(f"{path.name}: {m.group(0)!r}")
    assert not bad, "private checklist rows cited in public reference notes:\n  " + "\n  ".join(bad)


#: Plain-text formulas the audit listed, each now in the house MathJax form.
_PLAIN_FORMULAS = [
    ("docs/architecture.md", r"B / 2\^30"),
    ("docs/architecture.md", r"`W, x\+b`"),
    ("docs/use-cases.md", r"\|<Z0>\|"),
    ("docs/use-cases.md", r"composed independently with the channel error, a \+ b - 2ab"),
    ("docs/findings.md", r"maps n qubits to n qubits"),
    ("docs/models-and-benchmarks.md", r"2 to the 21st"),
    ("docs/models-and-benchmarks.md", r"tau0 multiplied delta\(n_Z\)"),
    ("docs/models-and-benchmarks.md", r"half-width 2\^30"),
    ("docs/pqc-roadmap.md", r"c1 \|\| c2 \|\| ek1 \|\| ek2"),
    ("docs/crypto-agility.md", r"`e0 \* y0`"),
    ("references/REFERENCES.md", r"pDC = 10\^-8"),
    ("references/REFERENCES.md", r"c_dt = 1 / \(1 \+ R P t_DT\)"),
    ("references/REFERENCES.md", r"beta = ln\(1/eps\)"),
    ("references/REFERENCES.md", r"s\^u_\{Z,0\} = 2 \(tau0"),
    ("references/REFERENCES.md", r"eps_sec = 21 eps"),
    ("references/2026-metric-practice-fact-check.md", r"score 2p/\(1\+p\) at"),
]


def test_listed_formulas_use_the_house_math_form() -> None:
    found = [f"{rel}: /{pat}/" for rel, pat in _PLAIN_FORMULAS if re.search(pat, _read(rel))]
    assert not found, "plain-text formulas outside the MathJax form:\n  " + "\n  ".join(found)


def test_owned_code_text_is_ascii() -> None:
    bad = []
    for rel in ("src/qsh/fhe/conv.py", "src/qsh/fhe/ckks_inference.py"):
        for n, line in enumerate(_read(rel).splitlines(), 1):
            if re.search(r"[^\x00-\x7f]", line):
                bad.append(f"{rel}:{n}")
    for n, line in enumerate(_read("scripts/run_gpu_mpc.sh").splitlines(), 1):
        if line.lstrip().startswith("#") and re.search(r"[^\x00-\x7f]", line):
            bad.append(f"scripts/run_gpu_mpc.sh:{n}")
    assert not bad, f"non-ASCII text: {bad}"
