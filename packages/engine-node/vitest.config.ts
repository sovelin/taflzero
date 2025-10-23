import {defineConfig} from "vitest/config";

export default defineConfig({
  test: {
    include: ["engine/src/**/*.test.ts"],
    exclude: [],
  },
});
