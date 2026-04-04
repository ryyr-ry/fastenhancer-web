/*
 * activations.h — 活性化関数インターフェース
 */

#ifndef FE_ACTIVATIONS_H
#define FE_ACTIVATIONS_H

/* スカラー関数 */
float fe_sigmoid(float x);
float fe_silu(float x);

/* バッチ関数 (SIMD最適化) */
void fe_sigmoid_batch(const float* input, float* output, int n);
void fe_silu_batch(const float* input, float* output, int n);

#endif /* FE_ACTIVATIONS_H */
