import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  outDir: 'dist',
  format: ['esm'],
  target: 'esnext',
  sourcemap: true,
  clean: true,
  dts: true,
  splitting: false,
  skipNodeModulesBundle: true,
  shims: false,
  minify: false,
});
