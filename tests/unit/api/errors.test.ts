import { describe, it, expect } from 'vitest';
import * as errors from '../../../src/api/errors.js';

describe('Error module', () => {
  function getErrorClasses() {
    return Object.values(errors).filter(
      (v): v is new (...args: any[]) => Error =>
        typeof v === 'function' && v.prototype instanceof Error
    );
  }

  it('exports at least 6 error classes', () => {
    const classes = getErrorClasses();
    expect(classes.length).toBeGreaterThanOrEqual(6);
  });

  it('all errors extend Error', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(instance).toBeInstanceOf(Error);
    }
  });

  it('all errors have a non-empty string code property', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(typeof instance.code).toBe('string');
      expect(instance.code.length).toBeGreaterThan(0);
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

  it('all errors can be checked with instanceof in catch blocks', () => {
    for (const ErrorClass of getErrorClasses()) {
      let caught = false;
      try {
        throw new ErrorClass('test');
      } catch (e) {
        if (e instanceof Error) caught = true;
      }
      expect(caught).toBe(true);
    }
  });

  it('reflects the constructor argument in the message property', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('specific message');
      expect(instance.message).toBe('specific message');
    }
  });
});
