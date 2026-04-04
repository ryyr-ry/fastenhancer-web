#include <stdio.h>
#include <stdlib.h>
#include "unity/unity.h"
#include "../../src/engine/exports.h"

#ifdef FE_USE_BASE_48K
#include "../../src/engine/configs/base_48k.h"
#define TEST_BIN_PATH    "weights/fe_base_48k.bin"
#define TEST_BIN_SIZE    406636
#define TEST_MODEL_ID    1
#elif defined(FE_USE_SMALL_48K)
#include "../../src/engine/configs/small_48k.h"
#define TEST_BIN_PATH    "weights/fe_small_48k.bin"
#define TEST_BIN_SIZE    832988
#define TEST_MODEL_ID    2
#else
#include "../../src/engine/configs/tiny_48k.h"
#define TEST_BIN_PATH    "weights/fe_tiny_48k.bin"
#define TEST_BIN_SIZE    113436
#define TEST_MODEL_ID    0
#endif

void setUp(void) {}
void tearDown(void) {}

static uint8_t* read_file(const char* path, int* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { *out_len = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); *out_len = 0; return NULL; }
    uint8_t* buf = (uint8_t*)malloc(sz);
    if (!buf) { fclose(f); *out_len = 0; return NULL; }
    size_t nr = fread(buf, 1, sz, f);
    fclose(f);
    if ((long)nr != sz) {
        free(buf);
        *out_len = 0;
        return NULL;
    }
    *out_len = (int)sz;
    return buf;
}

void test_fe_init_accepts_exported_binary(void) {
    int len = 0;
    uint8_t* data = read_file(TEST_BIN_PATH, &len);
    char msg[128];

    snprintf(msg, sizeof(msg), "%s not found", TEST_BIN_PATH);
    TEST_ASSERT_NOT_NULL_MESSAGE(data, msg);

    snprintf(msg, sizeof(msg), "%s size mismatch (got %d, expected %d)",
             TEST_BIN_PATH, len, TEST_BIN_SIZE);
    TEST_ASSERT_EQUAL_INT_MESSAGE(TEST_BIN_SIZE, len, msg);

    FeState* state = fe_init(TEST_MODEL_ID, data, len);
    snprintf(msg, sizeof(msg), "fe_init failed for %s", TEST_BIN_PATH);
    TEST_ASSERT_NOT_NULL_MESSAGE(state, msg);

    float* in_ptr = fe_get_input_ptr(state);
    float* out_ptr = fe_get_output_ptr(state);
    int hop = fe_get_hop_size(state);
    for (int i = 0; i < hop; i++) in_ptr[i] = 0.0f;

    int ret = fe_process(state, in_ptr, out_ptr);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, ret, "fe_process returned non-zero");

    for (int i = 0; i < hop; i++) {
        snprintf(msg, sizeof(msg), "out[%d] is not finite", i);
        TEST_ASSERT_FALSE_MESSAGE(out_ptr[i] != out_ptr[i], msg);
        TEST_ASSERT_TRUE_MESSAGE(out_ptr[i] > -1e30f && out_ptr[i] < 1e30f, msg);
    }

    fe_destroy(state);
    free(data);
}

void test_fe_init_rejects_wrong_model_id(void) {
    int len = 0;
    uint8_t* data = read_file(TEST_BIN_PATH, &len);
    if (!data) { TEST_IGNORE_MESSAGE("weight file not found"); return; }

    int wrong_id = (TEST_MODEL_ID + 1) % 3;
    FeState* state = fe_init(wrong_id, data, len);
    TEST_ASSERT_NULL_MESSAGE(state, "fe_init should reject mismatched model_id");

    free(data);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_fe_init_accepts_exported_binary);
    RUN_TEST(test_fe_init_rejects_wrong_model_id);
    return UNITY_END();
}
