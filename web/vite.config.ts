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

// Dev server port + API proxy target are env-driven (VITE_DEV_PORT / VITE_API_TARGET) with the
// historical defaults, so they are no longer hard-coded literals.
const DEV_PORT = Number(process.env.VITE_DEV_PORT ?? 5173);
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), crossOriginIsolation, dropDanglingSourceMapComments],
  server: {
    port: DEV_PORT,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true, ws: true },
    },
  },
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
