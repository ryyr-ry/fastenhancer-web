/*
 * activations.h — Activation functions interface
 */

#ifndef FE_ACTIVATIONS_H
#define FE_ACTIVATIONS_H

/* Scalar functions */
float fe_sigmoid(float x);
float fe_silu(float x);

/* Batch functions (SIMD optimized) */
void fe_sigmoid_batch(const float* input, float* output, int n);
void fe_silu_batch(const float* input, float* output, int n);

#endif /* FE_ACTIVATIONS_H */
