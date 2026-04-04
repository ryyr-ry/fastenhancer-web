import { ValidationError } from './errors.js';
import { MODEL_CONFIGS } from '../engine/model-config.js';

/**
 * モデルカタログとレコメンデーション
 * 利用可能なモデルの情報取得とユースケースに基づく推奨を提供する。
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
    description: '最軽量モデル。低レイテンシ環境やモバイル向け。',
  },
  {
    id: 'base',
    label: 'Base',
    params: 101_000,
    wasmSizeKB: 65,
    hopSize: MODEL_CONFIGS.base.hopSize,
    description: 'バランス型モデル。品質と速度を両立。',
  },
  {
    id: 'small',
    label: 'Small',
    params: 207_000,
    wasmSizeKB: 75,
    hopSize: MODEL_CONFIGS.small.hopSize,
    description: '推奨モデル。最高品質のノイズ除去を提供。',
  },
] as const;

const RECOMMENDATIONS: Record<ModelPriority, { index: number; reason: string }> = {
  speed: {
    index: 0,
    reason: '処理速度を最優先。モバイルや低スペック環境で安定動作。',
  },
  balanced: {
    index: 1,
    reason: '品質と速度のバランスが良く、モバイル環境にも適する。',
  },
  quality: {
    index: 2,
    reason: '最高品質のノイズ除去。多くの環境で推奨。',
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
      `不正なpriority: '${String(priority)}'。有効な値: ${valid}`,
    );
  }
  const model = MODEL_CATALOG[rec.index];
  return { ...model, reason: rec.reason };
}
