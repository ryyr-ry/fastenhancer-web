import { describe, it, expect } from 'vitest';
import { getModels, recommendModel } from '../../../src/api/models.js';
import { ValidationError } from '../../../src/api/errors.js';

describe('getModels', () => {
  it('returns information for 3 models', () => {
    const models = getModels();
    expect(models).toHaveLength(3);
  });

  it('includes id, label, params, wasmSizeKB, hopSize, and description on each model', () => {
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

  it('returns models in tiny, base, small order', () => {
    const models = getModels();
    expect(models[0].id).toBe('tiny');
    expect(models[1].id).toBe('base');
    expect(models[2].id).toBe('small');
  });

  it('orders parameter counts in ascending order', () => {
    const models = getModels();
    expect(models[0].params).toBeLessThan(models[1].params);
    expect(models[1].params).toBeLessThan(models[2].params);
  });

  it('uses hopSize 512 for all models', () => {
    const models = getModels();
    for (const m of models) {
      expect(m.hopSize).toBe(512);
    }
  });
});

describe('recommendModel', () => {
  it('recommends small with no arguments', () => {
    const rec = recommendModel();
    expect(rec.id).toBe('small');
  });

  it('recommends small for priority=quality', () => {
    const rec = recommendModel({ priority: 'quality' });
    expect(rec.id).toBe('small');
  });

  it('recommends tiny for priority=speed', () => {
    const rec = recommendModel({ priority: 'speed' });
    expect(rec.id).toBe('tiny');
  });

  it('recommends base for priority=balanced', () => {
    const rec = recommendModel({ priority: 'balanced' });
    expect(rec.id).toBe('base');
  });

  it('includes a reason in the recommendation result', () => {
    const rec = recommendModel();
    expect(rec).toHaveProperty('reason');
    expect(typeof rec.reason).toBe('string');
    expect(rec.reason.length).toBeGreaterThan(0);
  });

  it('throws ValidationError for an invalid priority', () => {
    expect(() => recommendModel({ priority: 'invalid' as any })).toThrow(ValidationError);
  });
});
