import { defineConfig } from "vite";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    lib: {
      entry: path.resolve(__dirname, "src/index.ts"),
      name: "HnefataflEngine",
      fileName: (format) => `hnefatafl-engine.${format}.js`,
      formats: ["es", "cjs"],
    },
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      external: [
        "node:path",
      ],
      output: {
        exports: "named",
      },
    },
  },
});
