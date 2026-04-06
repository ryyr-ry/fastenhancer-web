export class FastEnhancerError extends Error {
  readonly code: string;

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'FastEnhancerError';
    this.code = 'FAST_ENHANCER_ERROR';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class WasmLoadError extends FastEnhancerError {
  override readonly code = 'WASM_LOAD_FAILED';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'WasmLoadError';
  }
}

export class ModelInitError extends FastEnhancerError {
  override readonly code = 'MODEL_INIT_FAILED';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'ModelInitError';
  }
}

export class AudioContextError extends FastEnhancerError {
  override readonly code = 'AUDIO_CONTEXT_ERROR';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'AudioContextError';
  }
}

export class WorkletError extends FastEnhancerError {
  override readonly code = 'WORKLET_ERROR';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'WorkletError';
  }
}

export class ValidationError extends FastEnhancerError {
  override readonly code = 'VALIDATION_ERROR';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'ValidationError';
  }
}

export class DestroyedError extends FastEnhancerError {
  override readonly code = 'DESTROYED_ERROR';

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'DestroyedError';
  }
}
