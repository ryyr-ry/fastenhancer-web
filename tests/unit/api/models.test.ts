import { describe, it, expect } from 'vitest';
import { getModels, recommendModel } from '../../../src/api/models.js';

describe('getModels', () => {
  it('3つのモデル情報を返す', () => {
    const models = getModels();
    expect(models).toHaveLength(3);
  });

  it('各モデルにid, label, params, wasmSizeKB, hopSize, description がある', () => {
    const models = getModels();
    for (const m of models) {
      expect(m).toHaveProperty('id');
      expect(m).toHaveProperty('label');
      expect(m).toHaveProperty('params');
      expect(m).toHaveProperty('wasmSizeKB');
      expect(m).toHaveProperty('hopSize');
      expect(m).toHaveProperty('description');
    }
  });

  it('tiny, base, small の順で返される', () => {
    const models = getModels();
    expect(models[0].id).toBe('tiny');
    expect(models[1].id).toBe('base');
    expect(models[2].id).toBe('small');
  });

  it('パラメータ数が昇順', () => {
    const models = getModels();
    expect(models[0].params).toBeLessThan(models[1].params);
    expect(models[1].params).toBeLessThan(models[2].params);
  });

  it('hopSizeは全モデル512', () => {
    const models = getModels();
    for (const m of models) {
      expect(m.hopSize).toBe(512);
    }
  });
});

describe('recommendModel', () => {
  it('引数なしでsmallを推奨', () => {
    const rec = recommendModel();
    expect(rec.id).toBe('small');
  });

  it('priority=quality でsmallを推奨', () => {
    const rec = recommendModel({ priority: 'quality' });
    expect(rec.id).toBe('small');
  });

  it('priority=speed でtinyを推奨', () => {
    const rec = recommendModel({ priority: 'speed' });
    expect(rec.id).toBe('tiny');
  });

  it('priority=balanced でbaseを推奨', () => {
    const rec = recommendModel({ priority: 'balanced' });
    expect(rec.id).toBe('base');
  });

  it('推奨結果にreasonが含まれる', () => {
    const rec = recommendModel();
    expect(rec).toHaveProperty('reason');
    expect(typeof rec.reason).toBe('string');
    expect(rec.reason.length).toBeGreaterThan(0);
  });

  it('不正なpriorityでValidationError', () => {
    expect(() => recommendModel({ priority: 'invalid' as any })).toThrow();
  });
});
