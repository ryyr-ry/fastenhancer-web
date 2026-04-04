import { describe, it, expect } from 'vitest';
import * as errors from '../../../src/api/errors.js';

describe('エラーモジュール', () => {
  function getErrorClasses() {
    return Object.values(errors).filter(
      (v): v is new (...args: any[]) => Error =>
        typeof v === 'function' && v.prototype instanceof Error
    );
  }

  it('6種類以上のエラークラスをエクスポート', () => {
    const classes = getErrorClasses();
    expect(classes.length).toBeGreaterThanOrEqual(6);
  });

  it('全エラーがErrorを継承', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(instance).toBeInstanceOf(Error);
    }
  });

  it('全エラーにcodeプロパティ（string, 非空）', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      expect(typeof instance.code).toBe('string');
      expect(instance.code.length).toBeGreaterThan(0);
    }
  });

  it('各エラーのcodeが一意', () => {
    const codes = new Set<string>();
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('test');
      codes.add(instance.code);
    }
    expect(codes.size).toBe(getErrorClasses().length);
  });

  it('全エラーがcauseチェーンをサポート', () => {
    for (const ErrorClass of getErrorClasses()) {
      const cause = new Error('root cause');
      const wrapped = new ErrorClass('wrapped', { cause });
      expect(wrapped.cause).toBe(cause);
    }
  });

  it('全エラーがcatch文でinstanceof判定可能', () => {
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

  it('messageプロパティにコンストラクタ引数が反映', () => {
    for (const ErrorClass of getErrorClasses()) {
      const instance = new ErrorClass('specific message');
      expect(instance.message).toBe('specific message');
    }
  });
});
