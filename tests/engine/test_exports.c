/*
 * test_exports.c — Phase 3B-E: WASMエクスポートAPI テスト
 *
 * 検証対象:
 *   - fe_init: 重みヘッダ/CRC検証
 *   - fe_process: state内バッファ経由の処理
 *   - fe_get_input_ptr / fe_get_output_ptr
 *   - fe_get_hop_size / fe_get_n_fft
 *   - fe_set_hpf / fe_set_agc
 */

#include "unity.h"
#include "exports.h"
#include "weight_format.h"
#include "tiny_48k.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PI_F 3.14159265f
#define FRAME_LEN 512

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

static void write_u32_le(uint8_t* dst, uint32_t value) {
    dst[0] = (uint8_t)(value & 0xffu);
    dst[1] = (uint8_t)((value >> 8) & 0xffu);
    dst[2] = (uint8_t)((value >> 16) & 0xffu);
    dst[3] = (uint8_t)((value >> 24) & 0xffu);
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

static void fix_bn_scales(float* w) {
    int p = 0;

    p += FE_C1 * 2 * FE_ENC_K0;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;

    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;
    }

    p += FE_F2 * FE_F1;
    p += FE_C2 * FE_C1;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 1.0f;
    p += FE_C2;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 0.0f;
    p += FE_C2;

    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        p += FE_W_GRU + FE_W_GRU_FC;
        if (b == 0) p += FE_W_PE;
        p += FE_W_MHSA;
    }

    p += FE_C1 * FE_C2;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;
    p += FE_F1 * FE_F2;

    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * 2 * FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;

        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;
    }

    p += FE_C1 * 2 * FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;
}

static uint8_t* create_weight_blob(unsigned int seed, int* out_len) {
    const int count = FE_TOTAL_WEIGHTS;
    const int data_bytes = count * (int)sizeof(float);
    uint8_t* blob = (uint8_t*)malloc(FE_WEIGHT_HEADER_SIZE + data_bytes);
    float* weights = (float*)malloc((size_t)data_bytes);
    unsigned int s = seed;

    for (int i = 0; i < count; i++) {
        weights[i] = lcg_float(&s);
    }
    fix_bn_scales(weights);

    blob[0] = FE_WEIGHT_MAGIC_0;
    blob[1] = FE_WEIGHT_MAGIC_1;
    blob[2] = FE_WEIGHT_MAGIC_2;
    blob[3] = FE_WEIGHT_MAGIC_3;
    write_u32_le(blob + 4, FE_WEIGHT_VERSION);
    write_u32_le(blob + 8, FE_MODEL_TINY);
    write_u32_le(blob + 12, (uint32_t)count);

    for (int i = 0; i < count; i++) {
        memcpy(blob + FE_WEIGHT_HEADER_SIZE + i * (int)sizeof(float), &weights[i], sizeof(float));
    }

    write_u32_le(blob + 16, crc32_compute(blob + FE_WEIGHT_HEADER_SIZE, data_bytes));
    free(weights);

    *out_len = FE_WEIGHT_HEADER_SIZE + data_bytes;
    return blob;
}

static uint8_t* create_misaligned_weight_blob(unsigned int seed, int* out_len, uint8_t** base_ptr) {
    uint8_t* aligned_blob = create_weight_blob(seed, out_len);
    uint8_t* base = (uint8_t*)malloc((size_t)(*out_len + 1));
    uint8_t* shifted = base + 1;

    memcpy(shifted, aligned_blob, (size_t)(*out_len));
    free(aligned_blob);

    *base_ptr = base;
    return shifted;
}

void setUp(void) {}
void tearDown(void) {}

void test_fe_init_accepts_valid_weight_blob(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    FeState* state = fe_init(FE_MODEL_TINY, blob, blob_len);

    TEST_ASSERT_NOT_NULL(state);
    TEST_ASSERT_NOT_NULL(fe_get_input_ptr(state));
    TEST_ASSERT_NOT_NULL(fe_get_output_ptr(state));
    TEST_ASSERT_EQUAL_INT(512, fe_get_hop_size(state));
    TEST_ASSERT_EQUAL_INT(1024, fe_get_n_fft(state));

    fe_destroy(state);
    free(blob);
}

void test_fe_init_rejects_bad_magic(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    blob[0] = 'B';

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_bad_crc(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    blob[16] ^= 0xffu;

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_payload_crc_mismatch(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    blob[FE_WEIGHT_HEADER_SIZE + 7] ^= 0xffu;

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_bad_version(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    write_u32_le(blob + 4, FE_WEIGHT_VERSION + 1u);

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_wrong_model_size(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    write_u32_le(blob + 8, FE_MODEL_SMALL);

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_wrong_weight_count(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    write_u32_le(blob + 12, FE_TOTAL_WEIGHTS - 1u);

    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, blob, blob_len));
    free(blob);
}

void test_fe_init_rejects_too_short_blob(void) {
    uint8_t tiny_blob[4] = { FE_WEIGHT_MAGIC_0, FE_WEIGHT_MAGIC_1, FE_WEIGHT_MAGIC_2, FE_WEIGHT_MAGIC_3 };
    TEST_ASSERT_NULL(fe_init(FE_MODEL_TINY, tiny_blob, 4));
}

void test_fe_init_accepts_misaligned_blob(void) {
    int blob_len = 0;
    uint8_t* base = NULL;
    uint8_t* shifted = create_misaligned_weight_blob(42u, &blob_len, &base);
    FeState* state = fe_init(FE_MODEL_TINY, shifted, blob_len);

    TEST_ASSERT_NOT_NULL(state);

    fe_destroy(state);
    free(base);
}

void test_fe_process_uses_state_buffers(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    FeState* state = fe_init(FE_MODEL_TINY, blob, blob_len);
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < FRAME_LEN; i++) {
        input[i] = 0.4f * sinf(2.0f * PI_F * 1000.0f * i / 48000.0f);
    }

    TEST_ASSERT_EQUAL_INT(0, fe_process(state, input, output));

    {
        int nonzero = 0;
        for (int i = 0; i < FRAME_LEN; i++) {
            TEST_ASSERT_FALSE(isnan(output[i]));
            TEST_ASSERT_FALSE(isinf(output[i]));
            if (fabsf(output[i]) > 1e-10f) nonzero++;
        }
        TEST_ASSERT_TRUE(nonzero > 0);
    }

    fe_destroy(state);
    free(blob);
}

void test_fe_process_rejects_null_args(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    FeState* state = fe_init(FE_MODEL_TINY, blob, blob_len);
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    TEST_ASSERT_EQUAL_INT(-1, fe_process(NULL, input, output));
    TEST_ASSERT_EQUAL_INT(-1, fe_process(state, NULL, output));
    TEST_ASSERT_EQUAL_INT(-1, fe_process(state, input, NULL));

    fe_destroy(state);
    free(blob);
}

void test_fe_set_agc_changes_output(void) {
    int blob_len_a = 0;
    int blob_len_b = 0;
    uint8_t* blob_a = create_weight_blob(42u, &blob_len_a);
    uint8_t* blob_b = create_weight_blob(42u, &blob_len_b);
    FeState* state_a = fe_init(FE_MODEL_TINY, blob_a, blob_len_a);
    FeState* state_b = fe_init(FE_MODEL_TINY, blob_b, blob_len_b);
    float* input_a = fe_get_input_ptr(state_a);
    float* output_a = fe_get_output_ptr(state_a);
    float* input_b = fe_get_input_ptr(state_b);
    float* output_b = fe_get_output_ptr(state_b);

    fe_set_agc(state_b, 1);

    for (int frame = 0; frame < 8; frame++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            const float sample = 0.01f * sinf(2.0f * PI_F * 1000.0f * (frame * FRAME_LEN + i) / 48000.0f);
            input_a[i] = sample;
            input_b[i] = sample;
        }
        TEST_ASSERT_EQUAL_INT(0, fe_process(state_a, input_a, output_a));
        TEST_ASSERT_EQUAL_INT(0, fe_process(state_b, input_b, output_b));
    }

    {
        int differ = 0;
        for (int i = 0; i < FRAME_LEN; i++) {
            if (fabsf(output_a[i] - output_b[i]) > 1e-8f) differ++;
        }
        TEST_ASSERT_TRUE(differ > 0);
    }

    fe_destroy(state_a);
    fe_destroy(state_b);
    free(blob_a);
    free(blob_b);
}

void test_fe_process_internal_buffer_alias_safe(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(42u, &blob_len);
    FeState* state = fe_init(FE_MODEL_TINY, blob, blob_len);
    float* input = fe_get_input_ptr(state);

    fe_set_hpf(state, 1);
    fe_set_agc(state, 1);

    /* 同一バッファを input と output の両方に渡して alias テスト */
    for (int frame = 0; frame < 8; frame++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            input[i] = 0.05f * sinf(2.0f * PI_F * 500.0f * (frame * FRAME_LEN + i) / 48000.0f);
        }

        TEST_ASSERT_EQUAL_INT(0, fe_process(state, input, input));
        for (int i = 0; i < FRAME_LEN; i++) {
            TEST_ASSERT_FALSE(isnan(input[i]));
            TEST_ASSERT_FALSE(isinf(input[i]));
        }
    }

    fe_destroy(state);
    free(blob);
}

void test_fe_set_hpf_toggles_correctly(void) {
    int blob_len = 0;
    uint8_t* blob = create_weight_blob(99u, &blob_len);
    FeState* state = fe_init(FE_MODEL_TINY, blob, blob_len);
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);
    float out_no_hpf[FRAME_LEN];
    float out_with_hpf[FRAME_LEN];

    /* HPF無効で処理 */
    fe_set_hpf(state, 0);
    for (int i = 0; i < FRAME_LEN; i++) {
        input[i] = 0.5f;
    }
    fe_process(state, input, output);
    memcpy(out_no_hpf, output, sizeof(out_no_hpf));

    fe_reset(state);

    /* HPF有効で処理 */
    fe_set_hpf(state, 1);
    for (int i = 0; i < FRAME_LEN; i++) {
        input[i] = 0.5f;
    }
    fe_process(state, input, output);
    memcpy(out_with_hpf, output, sizeof(out_with_hpf));

    /* DC成分がHPFで異なる出力になることを確認 */
    int differ = 0;
    for (int i = 0; i < FRAME_LEN; i++) {
        if (fabsf(out_no_hpf[i] - out_with_hpf[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_TRUE(differ > 0);

    fe_destroy(state);
    free(blob);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_fe_init_accepts_valid_weight_blob);
    RUN_TEST(test_fe_init_rejects_bad_magic);
    RUN_TEST(test_fe_init_rejects_bad_crc);
    RUN_TEST(test_fe_init_rejects_payload_crc_mismatch);
    RUN_TEST(test_fe_init_rejects_bad_version);
    RUN_TEST(test_fe_init_rejects_wrong_model_size);
    RUN_TEST(test_fe_init_rejects_wrong_weight_count);
    RUN_TEST(test_fe_init_rejects_too_short_blob);
    RUN_TEST(test_fe_init_accepts_misaligned_blob);
    RUN_TEST(test_fe_process_uses_state_buffers);
    RUN_TEST(test_fe_process_rejects_null_args);
    RUN_TEST(test_fe_set_agc_changes_output);
    RUN_TEST(test_fe_process_internal_buffer_alias_safe);
    RUN_TEST(test_fe_set_hpf_toggles_correctly);

    return UNITY_END();
}
