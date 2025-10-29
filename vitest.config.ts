import {defineConfig} from "vitest/config";
import * as path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
    exclude: ["src/**/test-utils*.test.ts"],
    environment: 'node',
    globals: true,
  },
});
