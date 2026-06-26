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

// Dev server port + API proxy target are env-driven (VITE_DEV_PORT / VITE_API_TARGET) with the
// historical defaults, so they are no longer hard-coded literals.
const DEV_PORT = Number(process.env.VITE_DEV_PORT ?? 5173);
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), crossOriginIsolation],
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
