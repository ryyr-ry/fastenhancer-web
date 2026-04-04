import { ValidationError } from './errors.js';
import { MODEL_CONFIGS } from '../engine/model-config.js';

/**
 * Model catalog and recommendations.
 * Provides available model information and use-case-based recommendations.
 */

export interface ModelInfo {
  id: 'tiny' | 'base' | 'small';
  label: string;
  params: number;
  wasmSizeKB: number;
  hopSize: number;
  description: string;
}

export type ModelPriority = 'speed' | 'balanced' | 'quality';

export interface RecommendModelOptions {
  priority?: ModelPriority;
}

export interface ModelRecommendation extends ModelInfo {
  reason: string;
}

const MODEL_CATALOG: readonly ModelInfo[] = [
  {
    id: 'tiny',
    label: 'Tiny',
    params: 28_000,
    wasmSizeKB: 52,
    hopSize: MODEL_CONFIGS.tiny.hopSize,
    description: 'Lightest model. Suitable for low-latency environments and mobile devices.',
  },
  {
    id: 'base',
    label: 'Base',
    params: 101_000,
    wasmSizeKB: 65,
    hopSize: MODEL_CONFIGS.base.hopSize,
    description: 'Balanced model. Combines quality and speed.',
  },
  {
    id: 'small',
    label: 'Small',
    params: 207_000,
    wasmSizeKB: 75,
    hopSize: MODEL_CONFIGS.small.hopSize,
    description: 'Recommended model. Provides the highest-quality noise removal.',
  },
] as const;

const RECOMMENDATIONS: Record<ModelPriority, { index: number; reason: string }> = {
  speed: {
    index: 0,
    reason: 'Prioritizes processing speed. Runs reliably on mobile and lower-spec environments.',
  },
  balanced: {
    index: 1,
    reason: 'Well-balanced between quality and speed, and also suitable for mobile environments.',
  },
  quality: {
    index: 2,
    reason: 'Highest-quality noise removal. Recommended for most environments.',
  },
};

export function getModels(): ModelInfo[] {
  return MODEL_CATALOG.map(m => ({ ...m }));
}

export function recommendModel(
  options?: RecommendModelOptions,
): ModelRecommendation {
  const priority = options?.priority ?? 'quality';
  const rec = RECOMMENDATIONS[priority];
  if (!rec) {
    const valid = Object.keys(RECOMMENDATIONS).join(', ');
    throw new ValidationError(
      `Invalid priority: '${String(priority)}'. Valid values: ${valid}`,
    );
  }
  const model = MODEL_CATALOG[rec.index];
  return { ...model, reason: rec.reason };
}
