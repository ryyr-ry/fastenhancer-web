/*
 * exports.c — WASM/FFI wrapper
 */

#include "exports.h"
#include "weight_format.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_u32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float read_f32_le(const uint8_t* p) {
    union {
        uint32_t u;
        float f;
    } bits;

    bits.u = read_u32_le(p);
    return bits.f;
}

static void decode_f32_le_array(const uint8_t* src, float* dst, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = read_f32_le(src + i * (int)sizeof(float));
    }
}

static uint32_t crc32_compute(const uint8_t* data, int len) {
    uint32_t crc = 0xffffffffu;

    for (int i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int b = 0; b < 8; b++) {
            const uint32_t mask = (uint32_t)-(int)(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }

    return ~crc;
}

FE_PUBLIC_API FeState* fe_init(int model_size, const uint8_t* weight_data, int weight_len) {
    const int expected_count = fe_weight_count(model_size);
    uint32_t header_version;
    uint32_t header_model_size;
    uint32_t header_weight_count;
    uint32_t header_crc32;
    const uint8_t* data_ptr;
    const int data_bytes = expected_count > 0 ? expected_count * (int)sizeof(float) : 0;

    if (!weight_data || weight_len < FE_WEIGHT_HEADER_SIZE) return NULL;
    if (expected_count < 0) return NULL;

    header_version = read_u32_le(weight_data + 4);
    header_model_size = read_u32_le(weight_data + 8);
    header_weight_count = read_u32_le(weight_data + 12);
    header_crc32 = read_u32_le(weight_data + 16);

    if (weight_data[0] != FE_WEIGHT_MAGIC_0 ||
        weight_data[1] != FE_WEIGHT_MAGIC_1 ||
        weight_data[2] != FE_WEIGHT_MAGIC_2 ||
        weight_data[3] != FE_WEIGHT_MAGIC_3) {
        return NULL;
    }
    if (header_version != FE_WEIGHT_VERSION) return NULL;
    if ((int)header_model_size != model_size) return NULL;
    if ((int)header_weight_count != expected_count) return NULL;
    if (weight_len != FE_WEIGHT_HEADER_SIZE + data_bytes) return NULL;

    data_ptr = weight_data + FE_WEIGHT_HEADER_SIZE;
    if (crc32_compute(data_ptr, data_bytes) != header_crc32) return NULL;

    {
        float* temp = (float*)malloc((size_t)data_bytes);
        FeState* state;

        if (!temp) return NULL;
        decode_f32_le_array(data_ptr, temp, expected_count);
        state = fe_create(model_size, temp, expected_count);
        free(temp);
        return state;
    }
}

FE_PUBLIC_API int fe_process(FeState* state, const float* input, float* output) {
    if (!state || !input || !output) return -1;
    fe_process_frame(state, input, output);
    return 0;
}

FE_PUBLIC_API int fe_process_buffered(FeState* state) {
    if (!state) return -1;
    float* in_ptr = fe_get_input_ptr(state);
    float* out_ptr = fe_get_output_ptr(state);
    if (!in_ptr || !out_ptr) return -1;
    fe_process_frame(state, in_ptr, out_ptr);
    return 0;
}

FE_PUBLIC_API int fe_process_inplace(FeState* state) {
    return fe_process_buffered(state);
}
