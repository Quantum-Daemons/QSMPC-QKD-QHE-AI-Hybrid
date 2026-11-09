# Python walkthroughs on a client-only demo: JupyterLite, marimo and pyvista-wasm

Read and tried 2026-11-09 (Round 86). The question is whether a walkthrough a reader can re-run in the tab belongs on the public demo, and with what.

## What each is

| | marimo `export html-wasm` | JupyterLite | pyvista-wasm |
|---|---|---|---|
| what it is | a reactive notebook exported as static files, Python on Pyodide | JupyterLab in the browser, kernels in a Web Worker on Pyodide or Xeus | a PyVista-like API over VTK.wasm, for 3D |
| version read | 0.25.2 (PyPI) | `jupyterlite-core` 0.7 to 0.8 (its docs) | 0.16.0 (its docs); no maturity statement |
| hosting | static files over HTTP; a directory export needs its `assets` beside it | static files over HTTP | a library, used inside one of the others |
| reader experience | one page, code locked or editable | a full notebook editor, files kept in the browser's own storage | not applicable |
| stated limits | 2 GB of memory, no OS threads or shared memory, no PDB | not all JupyterLab features work | none stated |

Pyodide 314.0.7 ships numpy, scipy, scikit-learn, pandas, matplotlib and xgboost. It does not list onnxruntime or tenseal, so a walkthrough can re-derive this project's analyses in Python, and cannot run its CKKS or ONNX paths in Python in the tab.

## What was tried

A marimo notebook that fetches the demo's own `/data/telecom_nidd/real_2d.json` and recomputes the figures in its note was exported in run mode and served under the demo's cross-origin isolation headers (`COOP: same-origin`, `COEP: require-corp`):

- It ran with no change to those headers, `crossOriginIsolated` true, and printed its result within 8 seconds.
- The export is 28 MB of static files, and Pyodide 314.0.0 loads from `cdn.jsdelivr.net` by default.
- The export contains a stray `CLAUDE.md` from the marimo package, to remove before publishing.

## Reading

**marimo fits, if a walkthrough is wanted.** The demo's selling point is that a reader can check the result, and the checks most worth re-deriving are short analyses over data the site already serves, for example the arithmetic behind the 5G-NIDD ceiling. It adds no backend and costs nothing unless visited.

**The one thing to settle first is where Pyodide comes from.** The demo says compute happens in the tab and loads its own code. By default a marimo export loads Pyodide from a third-party CDN, which adds an origin that can change what runs. Serving it from the demo's own origin removes that and costs about the size of the Pyodide files actually used.

**JupyterLite does not fit this demo.** It brings an editor and per-browser file storage. Those suit a teaching site where a reader writes code, and a verification page does not need them.

**pyvista-wasm is not needed.** No use case here has volumetric or mesh data, and the 3D views the demo has (Bloch vectors and satellite geometry) are three.js and small.

## Sources

- marimo, "WASM notebooks" and "Export to WebAssembly HTML", https://docs.marimo.io/guides/wasm/ and https://docs.marimo.io/guides/exporting/webassembly_html/
- JupyterLite documentation, https://jupyterlite.readthedocs.io/en/stable/
- pyvista-wasm documentation, https://pyvista-wasm.readthedocs.io/en/latest/
- Pyodide, "Packages built in Pyodide", https://pyodide.org/en/stable/usage/packages-in-pyodide.html
