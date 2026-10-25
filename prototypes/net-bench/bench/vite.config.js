import { defineConfig } from "vite";

// The same cross-origin isolation headers the demo serves, so both engines run under the conditions
// they run under there.
const isolation = {
  name: "cross-origin-isolation",
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

export default defineConfig({
  plugins: [isolation],
  build: { target: "es2022", outDir: "dist" },
});
