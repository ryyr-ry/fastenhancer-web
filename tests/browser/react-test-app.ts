/**
 * React useDenoiser E2Eテスト用アプリケーション
 *
 * bun build でバンドルして tests/browser/vendor/react-test-bundle.js に出力。
 * react-test.html から読み込まれる。
 *
 * useDenoiser v2: wasmBytes/weightBytes/exportMap は不要。
 * baseUrl を指定するだけでloadModelが自動的にリソースを取得する。
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
    const ctx = new AudioContext({ sampleRate: 48000 });
    try {
      const osc = ctx.createOscillator();
      osc.frequency.value = 440;
      const dest = ctx.createMediaStreamDestination();
      osc.connect(dest);
      osc.start();
      inputCtxRef.current = ctx;
      await start(dest.stream);
    } catch {
      ctx.close().catch((e: unknown) => console.warn('AudioContext close failed:', e));
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
  statusEl.textContent = 'React レンダリング中...';

  const root = createRoot(document.getElementById('root')!);
  root.render(React.createElement(TestApp));

  statusEl.textContent = 'ready';
}

main().catch(err => {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = '[FAIL] ' + err.message;
});
