import { loadModel } from './loader.js';
import { instantiateWasm } from './wasm-instantiate.js';
import { ModelInitError, ValidationError, WasmLoadError } from './errors.js';
import type { WasmInstance } from './index.js';
import type { ModelOptions } from './offline-worker-tasks.js';

export interface BatchInfo { sampleRate: number; hopSize: number; latency: number; encodedSize: number; featureSize: number; frequencyBins: number; featureChannels: number; blocks: number }
interface OfflineWasm extends WasmInstance {
  _initialize?: () => void;
  _fe_offline_info(field: number): number;
  _fe_offline_encode(state: number, input: number, frames: number, output: number): number;
  _fe_offline_gru(state: number, input: number, frames: number, block: number, first: number, end: number, output: number): number;
  _fe_offline_attention(state: number, input: number, frames: number, block: number, output: number): number;
  _fe_offline_decode(state: number, encoded: number, features: number, frames: number, output: number): number;
}

async function embedded(size: string, variant: string) {
  switch (`${size}-${variant}`) {
    case 'tiny-simd': return import('../embedded/offline-tiny-simd.js');
    case 'base-simd': return import('../embedded/offline-base-simd.js');
    case 'small-simd': return import('../embedded/offline-small-simd.js');
    default: throw new ValidationError('Invalid offline model variant');
  }
}
const resources = new Map<string, Promise<{ bytes: ArrayBuffer; map: Record<string, string>; module: WebAssembly.Module; weights: ArrayBuffer; id: number }>>();
function resource(options: ModelOptions) {
  const key = `${options.model ?? 'small'}:${options.simd ?? 'auto'}:${options.baseUrl ?? ''}`;
  let promise = resources.get(key);
  if (!promise) {
    promise = (async () => {
      const model = await loadModel(options.model ?? 'small', { simd: options.simd, baseUrl: options.baseUrl });
      let bytes: ArrayBuffer, map: Record<string, string>;
      if (options.baseUrl) {
        const base = options.baseUrl.endsWith('/') ? options.baseUrl : `${options.baseUrl}/`;
        const response = await fetch(`${base}fastenhancer-${model.size}-${model.variant}-offline.wasm`);
        if (!response.ok) throw new WasmLoadError(`Could not load the offline engine (${response.status}); deploy the offline WASM asset or select engine: 'channels'`);
        bytes = await response.arrayBuffer();
        const names = ['malloc', 'free', 'fe_init', 'fe_destroy', 'fe_process', 'fe_process_inplace', 'fe_get_input_ptr', 'fe_get_output_ptr', 'fe_get_hop_size', 'fe_get_n_fft', 'fe_set_agc', 'fe_set_hpf', 'fe_reset', 'fe_offline_info', 'fe_offline_encode', 'fe_offline_gru', 'fe_offline_attention', 'fe_offline_decode'];
        map = Object.fromEntries(names.map(name => [`_${name}`, name])); map._initialize = '_initialize';
      } else { const asset = await embedded(model.size, model.variant); bytes = asset.getWasmBytes(); map = asset.exportMap; }
      return { bytes, map, module: await WebAssembly.compile(bytes), weights: model.weightData, id: model.modelSizeId };
    })();
    resources.set(key, promise); promise.catch(() => resources.delete(key));
  }
  return promise;
}

export class OfflineEngine {
  private wasm: OfflineWasm;
  private state: number;
  private destroyed = false;
  readonly info: BatchInfo;
  private constructor(wasm: OfflineWasm, state: number) {
    this.wasm = wasm; this.state = state;
    this.info = { sampleRate: 48000, hopSize: wasm._fe_get_hop_size(state), latency: wasm._fe_get_n_fft(state) - wasm._fe_get_hop_size(state),
      encodedSize: wasm._fe_offline_info(0), featureSize: wasm._fe_offline_info(1), frequencyBins: wasm._fe_offline_info(2), featureChannels: wasm._fe_offline_info(3), blocks: wasm._fe_offline_info(4) };
  }
  static async create(options: ModelOptions): Promise<OfflineEngine> {
    const data = await resource(options);
    const wasm = await instantiateWasm(data.bytes, data.map, data.module) as OfflineWasm;
    wasm._initialize?.();
    const pointer = wasm._malloc(data.weights.byteLength);
    if (!pointer) throw new ModelInitError('Could not allocate offline model weights');
    try {
      new Uint8Array(wasm.HEAPF32.buffer).set(new Uint8Array(data.weights), pointer);
      const state = wasm._fe_init(data.id, pointer, data.weights.byteLength);
      if (!state) throw new ModelInitError('Could not initialize the offline engine');
      return new OfflineEngine(wasm, state);
    } finally { wasm._free(pointer); }
  }
  private execute(inputs: Float32Array[], length: number, invoke: (pointers: number[], output: number) => number): Float32Array {
    if (this.destroyed) throw new ModelInitError('Offline engine is destroyed');
    const pointers: number[] = [];
    let output = 0;
    try {
      for (const input of inputs) {
        const pointer = this.wasm._malloc(input.byteLength);
        if (!pointer) throw new ModelInitError('Offline tile exceeds engine memory');
        pointers.push(pointer); this.wasm.HEAPF32.set(input, pointer / 4);
      }
      output = this.wasm._malloc(length * 4);
      if (!output) throw new ModelInitError('Offline tile exceeds engine memory');
      if (invoke(pointers, output) !== 0) throw new ValidationError('Invalid offline engine tile');
      const result = this.wasm.HEAPF32.slice(output / 4, output / 4 + length);
      if (!result.every(Number.isFinite)) throw new ValidationError('Engine produced non-finite samples');
      return result;
    } finally { if (output) this.wasm._free(output); pointers.forEach(pointer => this.wasm._free(pointer)); }
  }
  encode(input: Float32Array, frames: number): Float32Array {
    if (input.length !== (frames + 1) * this.info.hopSize) throw new ValidationError('Invalid encoder context');
    return this.execute([input], frames * this.info.encodedSize, (p, out) => this.wasm._fe_offline_encode(this.state, p[0], frames, out));
  }
  gru(input: Float32Array, frames: number, block: number, first: number, end: number): Float32Array {
    if (input.length !== frames * this.info.featureSize) throw new ValidationError('Invalid recurrent features');
    return this.execute([input], frames * (end - first) * this.info.featureChannels, (p, out) => this.wasm._fe_offline_gru(this.state, p[0], frames, block, first, end, out));
  }
  attention(input: Float32Array, frames: number, block: number): Float32Array {
    if (input.length !== frames * this.info.featureSize) throw new ValidationError('Invalid attention features');
    return this.execute([input], input.length, (p, out) => this.wasm._fe_offline_attention(this.state, p[0], frames, block, out));
  }
  decode(encoded: Float32Array, features: Float32Array, frames: number): Float32Array {
    if (encoded.length !== frames * this.info.encodedSize || features.length !== frames * this.info.featureSize) throw new ValidationError('Invalid decoder features');
    return this.execute([encoded, features], frames * this.info.hopSize * 2, (p, out) => this.wasm._fe_offline_decode(this.state, p[0], p[1], frames, out));
  }
  destroy(): void { if (!this.destroyed) { this.destroyed = true; this.wasm._fe_destroy(this.state); } }
}
