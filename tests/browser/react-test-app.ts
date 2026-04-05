/**
 * Application for React useDenoiser E2E tests
 *
 * Bundled with bun build and output to tests/browser/vendor/react-test-bundle.js.
 * Loaded from react-test.html.
 *
 * useDenoiser v2: wasmBytes/weightBytes/exportMap are unnecessary.
 * Simply specifying baseUrl lets loadModel fetch resources automatically.
 */
import React, { useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { useDenoiser } from '../../dist/react/useDenoiser.js';

function TestApp() {
  const inputCtxRef = useRef<AudioContext | null>(null);

  const { state, error, outputStream, bypass, start, stop, setBypass, destroy } =
    useDenoiser('small', {
      baseUrl: '/flat/',
      workletUrl: '/dist/worklet/processor.js',
    });

  const handleStart = useCallback(async () => {
    let ctx: AudioContext | undefined;
    try {
      ctx = new AudioContext({ sampleRate: 48000 });
      const osc = ctx.createOscillator();
      osc.frequency.value = 440;
      const dest = ctx.createMediaStreamDestination();
      osc.connect(dest);
      osc.start();
      inputCtxRef.current = ctx;
      await start(dest.stream);
    } catch {
      ctx?.close().catch((e: unknown) => console.warn('AudioContext close failed:', e));
      inputCtxRef.current = null;
    }
  }, [start]);

  const handleStop = useCallback(() => {
    stop();
    if (inputCtxRef.current) {
      inputCtxRef.current.close().catch((e: unknown) => console.warn('AudioContext close failed:', e));
      inputCtxRef.current = null;
    }
  }, [stop]);

  const handleDestroy = useCallback(() => {
    destroy();
    if (inputCtxRef.current) {
      inputCtxRef.current.close().catch((e: unknown) => console.warn('AudioContext close failed:', e));
      inputCtxRef.current = null;
    }
  }, [destroy]);

  return React.createElement('div', {
    id: 'hook-state',
    'data-state': state,
    'data-error': error?.message ?? '',
    'data-has-stream': outputStream ? 'true' : 'false',
    'data-bypass': String(bypass),
  },
    React.createElement('button', { id: 'btn-start', onClick: handleStart }, 'Start'),
    React.createElement('button', { id: 'btn-stop', onClick: handleStop }, 'Stop'),
    React.createElement('button', { id: 'btn-bypass-on', onClick: () => setBypass(true) }, 'Bypass On'),
    React.createElement('button', { id: 'btn-bypass-off', onClick: () => setBypass(false) }, 'Bypass Off'),
    React.createElement('button', { id: 'btn-destroy', onClick: handleDestroy }, 'Destroy'),
  );
}

async function main() {
  const statusEl = document.getElementById('status')!;
  statusEl.textContent = 'React rendering...';

  const root = createRoot(document.getElementById('root')!);
  root.render(React.createElement(TestApp));

  statusEl.textContent = 'ready';
}

main().catch(err => {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = '[FAIL] ' + err.message;
});
