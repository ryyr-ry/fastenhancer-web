import { test, expect } from '@playwright/test';

for (const size of ['base', 'small'] as const) {
  for (const simd of [false, true]) {
    for (const scenario of ['burst', 'recovery'] as const) {
      test(`${size}/${simd ? 'simd' : 'scalar'}: ${scenario} preserves audio through the real stereo engine`, async ({ page }) => {
        await page.goto('/tests/browser/offline-test.html');
        const result = await page.evaluate(async ({ size, simd, scenario }) => {
          const apiPath = '/dist/api/index.js';
          const { loadModel } = await import(apiPath);
          const model = await loadModel(size, { simd });
          const frames = 40, hop = 512, length = frames * hop;
          const costs = Array(frames).fill(scenario === 'burst' ? 8 : 11);
          if (scenario === 'burst') costs.fill(18, 10, 18);
          else costs.fill(30, 0, 12);

          // Inject measured durations, rather than rely on OS/CI scheduling.
          // All inference, buffering, reset and mixing still use shipped code
          // and real WASM. The production processor has no test-only controls.
          const source = await (await fetch('/dist/worklet/processor.js')).text();
          const registration = "registerProcessor('fastenhancer-processor', FastEnhancerProcessor);";
          const controlled = source.replace(registration, `
            class ControlledProcessor extends FastEnhancerProcessor {
              _runWasmFrame(hop) {
                super._runWasmFrame(hop);
                return ${JSON.stringify(costs)}[this._audioFrameCount - 1];
              }
              _processFrame(hop) {
                super._processFrame(hop);
                this.port.postMessage({ type: 'trace', frame: this._audioFrameCount - 1, mode: this._monitor.mode });
              }
            }
            registerProcessor('fastenhancer-processor', ControlledProcessor);
          `);
          if (controlled === source) throw new Error('Could not instrument the processor');
          const context = new OfflineAudioContext(2, length, 48000);
          const url = URL.createObjectURL(new Blob([controlled], { type: 'application/javascript' }));
          try { await context.audioWorklet.addModule(url); }
          finally { URL.revokeObjectURL(url); }
          const node = new AudioWorkletNode(context, 'fastenhancer-processor', {
            outputChannelCount: [2], channelCount: 2,
            channelCountMode: 'explicit', channelInterpretation: 'discrete',
            processorOptions: { channelCount: 2 },
          });
          const events: boolean[] = [], modes: string[] = [];
          let finish!: () => void;
          const traced = new Promise<void>(resolve => { finish = resolve; });
          const ready = new Promise<void>((resolve, reject) => {
            node.port.onmessage = event => {
              const message = event.data;
              if (message.type === 'ready') resolve();
              if (message.type === 'error') reject(new Error(message.message));
              if (message.type === 'auto_bypass') events.push(message.enabled);
              if (message.type === 'trace') {
                modes[message.frame] = message.mode;
                if (message.frame === frames - 1) finish();
              }
            };
          });
          node.port.postMessage({ type: 'init', wasmBytes: model.wasmBytes, weightBytes: model.weightData,
            exportMap: model.exportMap, modelSize: model.modelSizeId, channelCount: 2 });
          await ready;
          const buffer = context.createBuffer(2, length, 48000);
          for (let i = 0; i < length; i++) {
            buffer.getChannelData(0)[i] = 0.2 * Math.sin(2 * Math.PI * 400 * i / 48000);
            buffer.getChannelData(1)[i] = 0.1 * Math.sin(2 * Math.PI * 1700 * i / 48000);
          }
          const input = context.createBufferSource(); input.buffer = buffer;
          input.connect(node).connect(context.destination); input.start();
          const rendered = await context.startRendering();
          await traced;
          let dryError = 0, wetError = 0;
          const referenceStart = scenario === 'recovery' ? 12 : 0;
          const wetStart = scenario === 'recovery' ? 16 : 0;
          for (let c = 0; c < 2; c++) {
            const mono = await model.createDenoiser({ isolated: true });
            try {
              const reference = new Float32Array(length);
              for (let f = referenceStart; f < frames; f++) {
                reference.set(mono.processFrame(buffer.getChannelData(c).slice(f * hop, (f + 1) * hop)), f * hop);
              }
              const output = rendered.getChannelData(c), original = buffer.getChannelData(c);
              for (let i = wetStart * hop + 384; i < length; i++) wetError = Math.max(wetError, Math.abs(output[i] - reference[i - 384]));
              if (scenario === 'recovery') {
                for (let i = 3 * hop + 384; i < 15 * hop + 384; i++) {
                  dryError = Math.max(dryError, Math.abs(output[i] - original[i - 896]));
                }
              }
            } finally { mono.destroy(); }
          }
          node.port.postMessage({ type: 'destroy' }); node.disconnect();
          return { events, modes, dryError, wetError };
        }, { size, simd, scenario });
        expect(result.wetError).toBeLessThan(1e-6);
        expect(result.dryError).toBeLessThan(1e-6);
        if (scenario === 'burst') {
          expect(result.events).toEqual([]);
          expect(result.modes.every(mode => mode === 'processing')).toBe(true);
        } else {
          expect(result.events).toEqual([true, false]);
          expect(result.modes[2]).toBe('bypass');
          expect(result.modes.slice(12, 15)).toEqual(['recovering', 'recovering', 'recovering']);
          expect(result.modes[15]).toBe('processing');
        }
      });
    }
  }
}
