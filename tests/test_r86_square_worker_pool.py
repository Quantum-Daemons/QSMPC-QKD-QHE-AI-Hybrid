"""The square student's batch runs across a pool of Web Workers (the owner's choice of 2026-11-25).

node-seal ships one single-threaded wasm build, so `runSquareStudentOffThread` parallelises a batch of images across
workers, each with its own keys: `squarePoolSize` sizes the pool by `clientGcPool.gcPoolSize`'s rule (half the logical
cores, at most four), never more than the items, and one where the device reports under 8 GB; `chunksOf` cuts the images
into contiguous chunks so the runs come back in the images' own order. This bundles the module and checks both rules.
In the preview on 2026-11-25 the 156 test images under A2 took 3.8 s on four workers and 13.6 s on one, the same 156
decisions and accuracy 0.7051 either way.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"

HARNESS = r"""
import { createRequire } from "node:module";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { squarePoolSize, chunksOf } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const sizes = [[156, 20], [156, 2], [156, 8], [156, 6], [3, 20], [1, 20], [156, 20, 4], [156, 20, 8]]
  .map(([n, c, g]) => m.squarePoolSize(n, c, g));
const items = Array.from({ length: 10 }, (_, i) => i);
const chunks = m.chunksOf(items, 4), few = m.chunksOf([7, 8], 4);
console.log(JSON.stringify({ sizes, chunks, few }));
"""


def test_the_pool_is_sized_and_cut_as_stated() -> None:
    if not ((WEB / "node_modules" / ".bin" / "esbuild").exists()):
        msg = "web/node_modules lacks esbuild, so the pool's rules cannot be bundled. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "harness.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(WEB), str(d / "bundle.mjs")],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=300)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["sizes"] == [4, 1, 4, 3, 3, 1, 1, 4], r["sizes"]
    assert r["chunks"] == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert r["few"] == [[7], [8]]


def test_the_card_says_how_many_workers_ran() -> None:
    src = (WEB / "src" / "lib" / "clientFHESquare.ts").read_text()
    assert "Web Workers at once" in src and "for (const worker of workers) worker.terminate();" in src
