/*
 * exports.h — WASM/FFI向け公開API
 */

#ifndef FE_EXPORTS_H
#define FE_EXPORTS_H

#include "fastenhancer.h"
#include <stdint.h>

FE_PUBLIC_API FeState* fe_init(int model_size, const uint8_t* weight_data, int weight_len);
FE_PUBLIC_API int fe_process(FeState* state, const float* input, float* output);
FE_PUBLIC_API int fe_process_buffered(FeState* state);
FE_PUBLIC_API int fe_process_inplace(FeState* state);

FE_PUBLIC_API float* fe_get_input_ptr(FeState* state);
FE_PUBLIC_API float* fe_get_output_ptr(FeState* state);
FE_PUBLIC_API int fe_get_n_fft(const FeState* state);

#endif /* FE_EXPORTS_H */
