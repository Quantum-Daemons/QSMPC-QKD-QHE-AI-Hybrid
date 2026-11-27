# Python walkthroughs on a client-only demo: JupyterLite, marimo and pyvista-wasm

Read and tried 2026-11-09 (Round 86); the versions and Pyodide's package list re-read at PyPI and pyodide.org, and the served headers on the demo, on 2026-11-22. The question is whether a walkthrough a reader can re-run in the tab belongs on the public demo, and with what.

## What each is

| | marimo `export html-wasm` | JupyterLite | pyvista-wasm |
|---|---|---|---|
| what it is | a reactive notebook exported as static files, Python on Pyodide | JupyterLab in the browser, kernels in a Web Worker on Pyodide or Xeus | a PyVista-like API over VTK.wasm, for 3D |
| version read | 0.25.2 (PyPI, 2026-10-19), the walkthrough's pin; 0.26.0 reached PyPI on 2026-11-24 (Python 3.11 or later; read 2026-11-27), not adopted yet | `jupyterlite-core` 0.8.7 (PyPI, 2026-10-30; classified "4 - Beta") | 0.17.0 (PyPI, 2026-11-14; classified "2 - Pre-Alpha") |
| hosting | static files over HTTP; a directory export needs its `assets` beside it | static files over HTTP | a library, used inside one of the others |
| reader experience | one page, code locked or editable | a full notebook editor, files kept in the browser's own storage | not applicable |
| stated limits | 2 GB of memory, no OS threads or shared memory, no PDB | not all JupyterLab features work | none stated |

Pyodide 314.0.8, the stable release on 2026-11-22, ships numpy, scipy, scikit-learn, pandas, matplotlib and xgboost, as 314.0.7 did. It does not list onnxruntime or tenseal, so a walkthrough can re-derive this project's analyses in Python, and cannot run its CKKS or ONNX paths in Python in the tab. The walkthrough serves Pyodide 314.0.0, the version marimo 0.25.2 pins (`walkthroughs/pyodide-sources.json`); a bump is the `--refresh` below.

## What was tried

A marimo notebook that fetches the demo's own `/data/telecom_nidd/real_2d.json` and recomputes the figures in its note was exported in run mode and served under the demo's cross-origin isolation headers (`COOP: same-origin`, `COEP: require-corp`):

- It ran with no change to those headers, `crossOriginIsolated` true, and printed its result within 8 seconds.
- The export is 28 MB of static files, and Pyodide 314.0.0 loads from `cdn.jsdelivr.net` by default.
- The export contains a stray `CLAUDE.md` from the marimo package, to remove before publishing.

## Reading

**marimo fits.** The demo's selling point is that a reader can check the result, and the checks most worth re-deriving are short analyses over data the site already serves. It adds no backend and costs nothing unless visited.

**Pyodide had to come from the demo's own origin.** The demo says compute happens in the tab and loads its own code, and by default a marimo export loads Pyodide from a third-party CDN and its lock file from marimo's own host. Both are options: marimo reads them from two attributes on its root element.

**JupyterLite does not fit this demo.** It brings an editor and per-browser file storage. Those suit a teaching site where a reader writes code, and a verification page does not need them.

**pyvista-wasm is not needed.** No use case here has volumetric or mesh data, and the 3D views the demo has (Bloch vectors and satellite geometry) are three.js and small.

## What was built

One notebook, `walkthroughs/nidd_copies.py`, linked from the 5G-NIDD card. Since 2026-11-20 the demo serves it from an origin of its own, and the app's `/walkthrough/` answers 301 there ([`docs/deployment.md`](../docs/deployment.md#on-an-origin-of-its-own-qsh_walkthrough_origin)). It reads 200 evenly spaced pairs of the flows the published 5G-NIDD file holds twice, on all 113 columns, and works out the ceiling they set.

`scripts/build_walkthrough.py` runs in a stage of the web image. It exports the notebook, fetches 16 files (the Pyodide core and 13 wheels) against the sha256 in `walkthroughs/pyodide-sources.json`, writes a lock file naming only those local files, sets the two attributes, drops the note the export carries, and lists the third-party works in `THIRD_PARTY.txt`. The export also runs ruff, through uv, to find the notebook's local imports; the image downloads that wheel first against `walkthroughs/ruff.lock` and the export runs uv offline, so nothing is fetched without a recorded hash (2026-11-19, after a deploy host's slow path to PyPI timed the download out).

| | |
|---|---|
| served | 47 MB of static files, fetched only if a reader opens it, under a policy of its own: inline scripts and `'unsafe-eval'`, which the app's policy refuses, with the app's cross-origin isolation (`COOP: same-origin`, `COEP: require-corp`), HSTS and `nosniff` (read on the demo, 2026-11-22) |
| run, from a cold cache | under 10 seconds on a 20-core machine |
| hosts it reaches | none other than this site: the page ran with every third-party host name in the export replaced by an invalid one |

A marimo or Pyodide bump is `python scripts/build_walkthrough.py --refresh`. `--check`, in CI, holds the record together without the network, and CI runs the notebook under CPython.

## Sources

- marimo, "WASM notebooks" and "Export to WebAssembly HTML", https://docs.marimo.io/guides/wasm/ and https://docs.marimo.io/guides/exporting/webassembly_html/
- JupyterLite documentation, https://jupyterlite.readthedocs.io/en/stable/
- pyvista-wasm documentation, https://pyvista-wasm.readthedocs.io/en/latest/
- Pyodide, "Packages built in Pyodide", https://pyodide.org/en/stable/usage/packages-in-pyodide.html
