import { test, expect } from '@playwright/test';

for (const size of ['base', 'small'] as const) {
  for (const simd of [false, true]) {
    for (const scenario of ['burst', 'recovery', 'borderline', 'cold', 'medium-burst', 'coarse-clock'] as const) {
      test(`${size}/${simd ? 'simd' : 'scalar'}: ${scenario} preserves audio through the real stereo engine`, async ({ page }) => {
        await page.goto('/tests/browser/offline-test.html');
        const result = await page.evaluate(async ({ size, simd, scenario }) => {
          const apiPath = '/dist/api/index.js';
          const { loadModel } = await import(apiPath);
          const model = await loadModel(size, { simd });
          const recovering = ['recovery', 'borderline', 'cold'].includes(scenario);
          const frames = ['medium-burst', 'coarse-clock'].includes(scenario) ? 160 : 40;
          const hop = 512, length = frames * hop;
          const costs = Array(frames).fill(recovering ? (scenario === 'borderline' ? 11.5 : 11) : 8);
          if (scenario === 'burst') costs.fill(18, 10, 18);
          if (recovering) costs.fill(30, 0, 12);
          if (scenario === 'cold') costs.fill(19, 13, 15);
          if (scenario === 'medium-burst') costs.fill(13.9, 40, 70);

          // Inject measured durations, rather than rely on OS/CI scheduling.
          // All inference, buffering, reset and mixing still use shipped code
          // and real WASM. The production processor has no test-only controls.
          let source = await (await fetch('/dist/worklet/processor.js')).text();
          if (scenario === 'coarse-clock') {
            source = 'let simulatedTime = 1005;\n' + source
              .replace("const _perf = typeof performance !== 'undefined' ? performance : null;",
                'const _perf = { now: () => Math.floor(simulatedTime / 16) * 16 };')
              .replace("const _dateNow = typeof Date !== 'undefined' ? Date.now : null;",
                'const _dateNow = () => simulatedTime;');
          }
          const registration = "registerProcessor('fastenhancer-processor', FastEnhancerProcessor);";
          const controlled = source.replace(registration, `
            class ControlledProcessor extends FastEnhancerProcessor {
              ${scenario === 'coarse-clock' ? `
              _initWasm(message) {
                super._initWasm(message);
                for (const channel of this._channels) {
                  const process = channel.wasm._fe_process_inplace;
                  channel.wasm._fe_process_inplace = state => {
                    process(state);
                    simulatedTime += 8 / this._channels.length;
                  };
                }
              }
              process(inputs, outputs) {
                simulatedTime += 128 / 48000 * 1000;
                return super.process(inputs, outputs);
              }` : ''}
              _runWasmFrame(hop) {
                const actual = super._runWasmFrame(hop);
                return ${scenario === 'coarse-clock' ? 'actual' : `${JSON.stringify(costs)}[this._audioFrameCount - 1]`};
              }
              _processFrame(hop) {
                super._processFrame(hop);
                this.port.postMessage({ type: 'trace', frame: this._audioFrameCount - 1,
                  mode: this._monitor.mode, timingSource: this._timingSource, measuredMs: this._monitor.averageMs });
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
          let lastTiming: { source: string; ms: number } | undefined;
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
                lastTiming = { source: message.timingSource, ms: message.measuredMs };
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
          // Locate the successful consecutive trial. The severe-load probe now
          // checks three frames, so a backoff interval can end one frame later.
          const referenceStart = recovering ? modes.lastIndexOf('bypass') + 1 : 0;
          const recoveryEnd = recovering ? modes.lastIndexOf('recovering') + 1 : 0;
          const wetStart = recovering ? recoveryEnd + 1 : 0;
          for (let c = 0; c < 2; c++) {
            const mono = await model.createDenoiser({ isolated: true });
            try {
              const reference = new Float32Array(length);
              for (let f = referenceStart; f < frames; f++) {
                reference.set(mono.processFrame(buffer.getChannelData(c).slice(f * hop, (f + 1) * hop)), f * hop);
              }
              const output = rendered.getChannelData(c), original = buffer.getChannelData(c);
              for (let i = wetStart * hop + 384; i < length; i++) wetError = Math.max(wetError, Math.abs(output[i] - reference[i - 384]));
              if (recovering) {
                for (let i = 3 * hop + 384; i < recoveryEnd * hop + 384; i++) {
                  dryError = Math.max(dryError, Math.abs(output[i] - original[i - 896]));
                }
              }
            } finally { mono.destroy(); }
          }
          node.port.postMessage({ type: 'destroy' }); node.disconnect();
          return { events, modes, dryError, wetError, lastTiming, referenceStart, recoveryEnd };
        }, { size, simd, scenario });
        expect(result.wetError).toBeLessThan(1e-6);
        expect(result.dryError).toBeLessThan(1e-6);
        if (!['recovery', 'borderline', 'cold'].includes(scenario)) {
          expect(result.events).toEqual([]);
          expect(result.modes.every(mode => mode === 'processing')).toBe(true);
        } else {
          expect(result.events).toEqual([true, false]);
          expect(result.modes[2]).toBe('bypass');
          expect(result.referenceStart).toBeGreaterThanOrEqual(12);
          expect(result.recoveryEnd).toBeLessThan(24);
          expect(result.modes.slice(result.referenceStart, result.recoveryEnd)).toEqual(Array(5).fill('recovering'));
          expect(result.modes.slice(result.recoveryEnd).every(mode => mode === 'processing')).toBe(true);
        }
        if (scenario === 'coarse-clock') {
          expect(result.lastTiming?.source).toBe('date');
          expect(result.lastTiming?.ms).toBeCloseTo(8, 8);
        }
      });
    }
  }
}
