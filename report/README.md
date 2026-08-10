# Technical report

`technical-report.pdf` is the deliverable. It is committed rather than built on demand, because a
report that requires a TeX installation to read is not a report.

## Rebuilding

    python report/build_data.py       # tables + macros, from models/metrics.json and the registry
    python report/build_figures.py    # figures, from live artefacts
    cd report && pdflatex technical-report.tex && pdflatex technical-report.tex

The second `pdflatex` pass resolves the table of contents and cross-references.

## Why the tables are generated

Every number in the report comes from `generated/`, which is produced from the repository's own
artefacts at build time. Nothing is transcribed. This project has found the same stale figure in five
documents at once; a PDF is a worse place for that than a README, because it is a fixed artefact
somebody keeps a copy of.

`build_data.py` emits complete `table` environments rather than bare rows. `\input` inside a
`tabular` does not work - TeX ends the file inside an alignment cell, so the following `\bottomrule`
meets `\noalign` outside a row. It reports "Misplaced \noalign" and *still produces a readable PDF
with rows silently dropped*, which is the same shape of failure as a test suite passing on the wrong
thing.

## Toolchain

`pdflatex` with `lmodern`, not `xelatex` with `fontspec`. Latin Modern is not installed as a system
font here, and xelatex fell back to `nullfont`: the build "succeeded", the PDF had the right page
count, and no glyph rendered. Pinned to what is actually present.
