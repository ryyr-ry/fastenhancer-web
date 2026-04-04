class TestProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.port.onmessage = (e) => {
      if (e.data.type === 'ping') {
        this.port.postMessage({ type: 'pong', received: true });
        return;
      }

      if (e.data.type === 'init') {
        try {
          const wasmBytes = e.data.wasmBytes;
          const wasmModule = new WebAssembly.Module(wasmBytes);
          const instance = new WebAssembly.Instance(wasmModule, {});
          const exports = Object.keys(instance.exports);
          const fnExports = exports.filter(k => typeof instance.exports[k] === 'function');

          let addResult = null;
          let mulResult = null;
          for (const name of fnExports) {
            const fn = instance.exports[name];
            const testAdd = fn(100, 200);
            if (testAdd === 300) { addResult = fn(1, 2); }
            const testMul = fn(2.5, 4.0);
            if (Math.abs(testMul - 10.0) < 0.001) { mulResult = fn(2.0, 3.0); }
          }

          this.port.postMessage({
            type: 'result',
            addResult: addResult,
            mulResult: mulResult,
            exports: exports,
            fnExports: fnExports,
          });
        } catch (err) {
          this.port.postMessage({ type: 'error', message: String(err) });
        }
      }
    };

    this.port.postMessage({ type: 'ready' });
  }

  process(inputs, outputs, parameters) {
    return true;
  }
}

registerProcessor('test-processor', TestProcessor);
