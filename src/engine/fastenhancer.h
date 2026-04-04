/*
 * fastenhancer.h — FastEnhancer inference engine public API
 *
 * Usage:
 *   int n = fe_weight_count(FE_MODEL_TINY);
 *   float* weights = load_weights(...);  // float array of n elements
 *   FeState* state = fe_create(FE_MODEL_TINY, weights, n);
 *   fe_process_frame(state, input_512, output_512);
 *   fe_destroy(state);
 */

#ifndef FE_FASTENHANCER_H
#define FE_FASTENHANCER_H

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define FE_PUBLIC_API EMSCRIPTEN_KEEPALIVE
#else
#define FE_PUBLIC_API
#endif

#define FE_MODEL_TINY  0
#define FE_MODEL_BASE  1
#define FE_MODEL_SMALL 2

typedef struct FeState FeState;

/* Returns the required weight count for a model size. Returns -1 if invalid. */
FE_PUBLIC_API int fe_weight_count(int model_size);

/* Initialize the engine. Weight data is copied internally.
 * Returns NULL if weight_count does not match fe_weight_count(model_size). */
FE_PUBLIC_API FeState* fe_create(int model_size, const float* weights, int weight_count);

/* Process one frame (hop_size samples).
 * input: [hop_size] input samples
 * output: [hop_size] denoised output samples */
FE_PUBLIC_API void fe_process_frame(FeState* state, const float* input, float* output);

/* Toggle preprocessing on/off. 0=disabled, non-zero=enabled. */
FE_PUBLIC_API void fe_set_hpf(FeState* state, int enabled);
FE_PUBLIC_API void fe_set_agc(FeState* state, int enabled);

/* Reset internal state (GRU hidden state, STFT overlap). Weights are preserved. */
FE_PUBLIC_API void fe_reset(FeState* state);

/* Release resources. Access to state after this call is undefined. */
FE_PUBLIC_API void fe_destroy(FeState* state);

/* Returns the hop size of the current model. */
FE_PUBLIC_API int fe_get_hop_size(const FeState* state);

#endif /* FE_FASTENHANCER_H */
