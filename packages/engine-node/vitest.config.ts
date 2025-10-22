import {defineConfig} from "vitest/config";

export default defineConfig({
  root: "../",
  test: {
    include: ["engine/src/**/*.test.ts"],
    exclude: [],
  },
});
