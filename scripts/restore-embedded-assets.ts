/** Restore missing test/build files from versioned assets. Preserve local rebuilds. */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const root = resolve(import.meta.dir, '..');
let restored = 0;
function missing(path: string, content: Uint8Array | string): void {
  if (existsSync(path)) return;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content, { flag: 'wx' });
  restored++;
}
for (const model of ['tiny', 'base', 'small']) {
  const weights = await import(`../src/embedded/weights-${model}.ts`);
  missing(resolve(root, `weights/fe_${model}_48k.bin`), new Uint8Array(weights.getWeightData()));
  for (const variant of ['scalar', 'simd']) {
    const wasm = await import(`../src/embedded/wasm-${model}-${variant}.ts`);
    const map = await import(`../src/embedded/exports-${model}-${variant}.ts`);
    missing(resolve(root, `dist/wasm/fastenhancer-${model}-${variant}.wasm`), new Uint8Array(wasm.getWasmBytes()));
    missing(resolve(root, `dist/wasm/fastenhancer-${model}-${variant}-exports.json`), JSON.stringify(map.exportMap, null, 2));
  }
}
console.log(`Restored ${restored} missing model assets; existing files were preserved.`);
