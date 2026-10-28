"""The report's architecture figures and the README's diagrams carry the same labels.

WHY THIS EXISTS (Round 80, the user's choice). The verification environment, the implementation
workflow and one flowchart per use case are drawn twice: in Mermaid in the README, which GitHub
renders, and in TikZ in the technical report, which pdflatex renders. The user chose two copies over
a shared renderer and a new dependency, on the condition that a test compares their labels. Two
copies of a figure drift the way two copies of a sentence do - this project has recorded that
defect in its prose many times - so each figure is paired by a key and compared label by label.

HOW THE PAIRS ARE FOUND. In the README a figure is the Mermaid block after an HTML comment
`<!-- figure: KEY -->`; in the report it is the TikZ picture after a comment `% figure: KEY`. The
labels are the node texts, the subgraph and region titles, and the edge labels. LaTeX markup is
reduced to plain text before comparing (`\\texttt{x}` to `x`, `\\_` to `_`), so the comparison is about
what a reader sees, in both places, word for word.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
REPORT = REPO / "report" / "technical-report.tex"

#: Every figure drawn in both places. A figure added to one file and not the other fails below.
KEYS = {
    "workflow", "verification-environment",
    "uc-finance_aml", "uc-medical_fl", "uc-genomics", "uc-telecom_ids", "uc-satellite_comms",
}


def _plain(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\\(?:texttt|textbf|emph|textit)\{", "", text)
    text = text.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&").replace("\\\\", " ")
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _readme_figures() -> dict[str, set[str]]:
    text = README.read_text()
    out: dict[str, set[str]] = {}
    for m in re.finditer(r"<!-- figure: ([\w-]+) -->\s*```mermaid\n(.*?)```", text, flags=re.S):
        body = m.group(2)
        labels = set(re.findall(r'\w+\s*(?:\[\(|\(\[|\[|\{|\()"([^"]+)"', body))   # nodes and subgraphs
        labels |= set(re.findall(r"\|([^|]+)\|", body))                            # edge labels
        out[m.group(1)] = {_plain(t) for t in labels}
    return out


def _report_figures() -> dict[str, set[str]]:
    text = REPORT.read_text()
    out: dict[str, set[str]] = {}
    for m in re.finditer(r"% figure: ([\w-]+)\n\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", text, flags=re.S):
        body = m.group(2)
        labels = set(re.findall(r"\\node\[[^\]]*?\]\s*\([\w-]+\)(?:\s*at\s*\([^)]*\))?\s*\{(.+?)\};", body, flags=re.S))
        labels |= set(re.findall(r"label=\{\[regionlabel\][a-z ]+:(.+?)\}\]", body))
        labels |= set(re.findall(r"node\[edgelabel[^\]]*\]\s*\{(.+?)\}", body))
        out[m.group(1)] = {_plain(t) for t in labels if _plain(t)}
    return out


def test_every_figure_is_drawn_in_both_places() -> None:
    readme, report = _readme_figures(), _report_figures()
    assert set(readme) == KEYS, f"README figures: {sorted(readme)}; expected {sorted(KEYS)}"
    assert set(report) == KEYS, f"report figures: {sorted(report)}; expected {sorted(KEYS)}"


def test_each_figure_carries_the_same_labels_in_both_places() -> None:
    readme, report = _readme_figures(), _report_figures()
    for key in sorted(KEYS & set(readme) & set(report)):
        only_readme = readme[key] - report[key]
        only_report = report[key] - readme[key]
        assert not only_readme and not only_report, (
            f"figure {key!r} differs. Only in the README: {sorted(only_readme)}. "
            f"Only in the report: {sorted(only_report)}.")
        assert len(readme[key]) >= 4, f"figure {key!r} parsed to {len(readme[key])} labels; the parser missed it"
