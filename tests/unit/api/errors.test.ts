import { describe, it, expect } from 'vitest';
import * as errors from '../../../src/api/errors.js';

const EXPECTED_ERROR_CLASSES = [
  'FastEnhancerError',
  'WasmLoadError',
  'ModelInitError',
  'AudioContextError',
  'WorkletError',
  'ValidationError',
  'DestroyedError',
] as const;

const EXPECTED_CODES: Record<string, string> = {
  FastEnhancerError: 'FAST_ENHANCER_ERROR',
  WasmLoadError: 'WASM_LOAD_FAILED',
  ModelInitError: 'MODEL_INIT_FAILED',
  AudioContextError: 'AUDIO_CONTEXT_ERROR',
  WorkletError: 'WORKLET_ERROR',
  ValidationError: 'VALIDATION_ERROR',
  DestroyedError: 'DESTROYED_ERROR',
};

describe('Error module', () => {
  function getErrorClasses() {
    return Object.values(errors).filter(
      (v): v is new (...args: any[]) => Error =>
        typeof v === 'function' && v.prototype instanceof Error
    );
  }

  it('exports exactly the expected error classes', () => {
    const exportedNames = getErrorClasses().map((c) => c.name).sort();
    expect(exportedNames).toEqual([...EXPECTED_ERROR_CLASSES].sort());
  });

  it('each subclass is instanceof FastEnhancerError and Error', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(instance).toBeInstanceOf(Error);
      expect(instance).toBeInstanceOf(errors.FastEnhancerError);
    }
  });

  it('each error class has the documented code constant', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      const expected = EXPECTED_CODES[ErrorClass.name];
      expect(instance.code).toBe(expected);
    }
  });

  it('each error has a unique code', () => {
    const codes = new Set<string>();
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      codes.add(instance.code);
    }
    expect(codes.size).toBe(getErrorClasses().length);
  });

  it('all errors support cause chains', () => {
    for (const ErrorClass of getErrorClasses()) {
      const cause = new Error('root cause');
      const wrapped = new ErrorClass('wrapped', { cause });
      expect(wrapped.cause).toBe(cause);
    }
  });

  it('all errors preserve the constructor message', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('specific message');
      expect(instance.message).toBe('specific message');
    }
  });

  it('all errors set name to match the class name', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(instance.name).toBe(ErrorClass.name);
    }
  });
});
