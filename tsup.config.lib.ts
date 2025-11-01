import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/lib.ts'],
  outDir: 'dist',
  format: ['esm', 'cjs'],
  target: 'esnext',
  sourcemap: true,
  clean: true,
  dts: true,
  splitting: false,
  skipNodeModulesBundle: true,
  shims: false,
  minify: false,
});
