import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP make the page cross-origin isolated -> enables SharedArrayBuffer,
// multithreaded WASM and WebGPU (needed for in-browser FHE / quantum sim / ONNX).
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
  configurePreviewServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

// A copied asset keeps its `//# sourceMappingURL=` comment, and the map is not copied with it:
// `gif.js/dist/gif.worker.js?url` arrives verbatim and names `gif.worker.js.map`, a file the build
// does not ship. Any open DevTools asks for it, and until Round 74 the server answered every missing
// path with index.html, so it was handed HTML where a map belongs. Visitors are not affected. The
// comment is dropped from emitted assets, so nothing names a file that is not served. (A console
// 404 in the Round 74 walk was first blamed on this; it was not - see the register.)
const dropDanglingSourceMapComments = {
  name: "drop-dangling-source-map-comments",
  generateBundle(_options: unknown, bundle: Record<string, any>) {
    for (const file of Object.values(bundle)) {
      if (file.type === "asset" && /\.(m?js)$/.test(file.fileName)) {
        const text = typeof file.source === "string" ? file.source : new TextDecoder().decode(file.source);
        if (/\/\/# sourceMappingURL=/.test(text)) {
          file.source = text.replace(/\n?\/\/# sourceMappingURL=\S+\s*$/, "\n");
        }
      }
    }
  },
};

// THIRD-PARTY NOTICES FOR THE NPM PACKAGES IN THE BUNDLE (Round 86). MIT, ISC, BSD and Apache-2.0 each ask that their
// notice travel with copies of the code, and minification keeps only the comments a package marks for it: a built page
// carried React's, Three.js's and ONNX Runtime's and no other. Every module Rollup puts in a chunk, the workers' chunks
// included, is traced to its package under node_modules, and each package's name, version, declared licence and licence
// file are written to `licenses/js-bundle.txt`, which the sidebar links. `web/scripts/check_js_notices.mjs` reads the
// built file. The wasm crates and the WebLLM libraries have their own notice files beside it.
const bundledPackages = new Map<string, string>();   // package directory -> package name
function tracePackages(bundle: Record<string, any>): void {
  for (const file of Object.values(bundle)) {
    if (file.type !== "chunk") continue;
    for (const id of Object.keys(file.modules ?? {})) {
      const clean = id.replace(/^\0/, "").split("?")[0];
      const at = clean.lastIndexOf("/node_modules/");
      if (at < 0) continue;
      const parts = clean.slice(at + 14).split("/");
      const name = parts[0].startsWith("@") ? `${parts[0]}/${parts[1]}` : parts[0];
      bundledPackages.set(clean.slice(0, at + 14) + name, name);
    }
  }
}
// Four packages in the bundle ship no licence file in their npm package. Their notice is taken from the project's own
// repository, read on 2026-11-14 (gif.js, onnxruntime, victory: MIT). The text states where it came from, and that a
// repository's file may be newer than the version bundled.
const MIT_TERMS = [
  "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated",
  "documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the",
  "rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit",
  "persons to whom the Software is furnished to do so, subject to the following conditions:", "",
  "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the",
  "Software.", "",
  "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE",
  "WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR",
  "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR",
  "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.",
].join("\n");
const ONNXRUNTIME = { source: "https://github.com/microsoft/onnxruntime/blob/main/LICENSE", copyright: "Copyright (c) Microsoft Corporation" };
export const NOTICE_FROM_REPOSITORY: Record<string, { source: string; copyright: string }> = {
  "gif.js": { source: "https://github.com/jnordberg/gif.js/blob/master/LICENSE", copyright: "Copyright (c) 2013-2018 Johan Nordberg" },
  "onnxruntime-common": ONNXRUNTIME,
  "onnxruntime-web": ONNXRUNTIME,
  "victory-vendor": { source: "https://github.com/FormidableLabs/victory/blob/main/LICENSE.txt",
                      copyright: "Copyright (c) 2015-2026 Nearform., and the further notices in that file" },
};

function noticesText(): string {
  const entries = Array.from(bundledPackages.entries()).map(([dir, name]) => {
    let version = "", licence = "";
    try {
      const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
      version = String(pkg.version ?? "");
      licence = typeof pkg.license === "string" ? pkg.license : String(pkg.license?.type ?? "");
    } catch { /* a package with no readable manifest is listed by name, and the check below fails on it */ }
    const file = existsSync(dir) ? readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.(md|txt))?$/i.test(f)) : undefined;
    const repo = NOTICE_FROM_REPOSITORY[name];
    const text = file ? readFileSync(join(dir, file), "utf8").trim()
      : repo && /\bMIT\b/.test(licence) ? `MIT License\n\n${repo.copyright}\n\n${MIT_TERMS}` : "";
    const origin = file ? `From its ${file}:`
      : repo ? `The npm package ships no licence file; the notice below is from ${repo.source} (read 2026-11-14), which may be newer than this version:`
      : "The package ships no licence file.";
    return { name, version, licence, text, origin };
  }).sort((x, y) => x.name.localeCompare(y.name) || x.version.localeCompare(y.version));
  const seen = new Set<string>();
  const unique = entries.filter((e) => { const k = `${e.name}@${e.version}`; if (seen.has(k)) return false; seen.add(k); return true; });
  return [
    "Third-party notices for the npm packages bundled into this page's JavaScript.",
    "Generated at build time by web/vite.config.ts from the modules the bundle contains; one entry per package and version.",
    `${unique.length} packages.`, "",
    ...unique.flatMap((e) => [
      "=".repeat(78), `${e.name} ${e.version}`, `Declared licence: ${e.licence || "none declared"}`,
      e.origin, "", e.text, "",
    ]),
  ].join("\n");
}
const thirdPartyNotices = (write: boolean) => ({
  name: "third-party-notices",
  generateBundle(this: any, _options: unknown, bundle: Record<string, any>) {
    tracePackages(bundle);
    // The workers are bundled while the page's modules are transformed, so by the page's generateBundle every worker
    // chunk has been traced.
    if (write) this.emitFile({ type: "asset", fileName: "licenses/js-bundle.txt", source: noticesText() });
  },
});

// Dev server port + API proxy target are env-driven (VITE_DEV_PORT / VITE_API_TARGET) with the
// historical defaults, so they are no longer hard-coded literals.
const DEV_PORT = Number(process.env.VITE_DEV_PORT ?? 5173);
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), crossOriginIsolation, dropDanglingSourceMapComments, thirdPartyNotices(true)],
  server: {
    port: DEV_PORT,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true, ws: true },
    },
  },
  // ES workers: the satellite student's worker reaches clientFHE.ts, which loads node-seal by a dynamic import, and
  // the default IIFE worker format cannot split code (Round 86).
  worker: { format: "es", plugins: () => [thirdPartyNotices(false)] },
  build: {
    target: "es2022",
    outDir: "dist",
    // The in-browser AI engines are large but are ONLY pulled via dynamic import() (clientAI.ts), so
    // they never enter the entry bundle. Pin them to named vendor chunks for legible build output, and
    // raise the size-warning ceiling (WebLLM's wasm glue is big) without hiding real entry-chunk growth.
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("@huggingface/transformers")) return "vendor-transformers";
          if (id.includes("@mlc-ai/web-llm")) return "vendor-webllm";
          return undefined;
        },
      },
    },
  },
});
