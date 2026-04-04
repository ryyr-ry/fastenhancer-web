/*
 * fastenhancer.h — FastEnhancer 推論エンジン公開API
 *
 * 使い方:
 *   int n = fe_weight_count(FE_MODEL_TINY);
 *   float* weights = load_weights(...);  // n要素のfloat配列
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

/* モデルサイズに対する必要重み数を返す。不正な場合は-1。 */
FE_PUBLIC_API int fe_weight_count(int model_size);

/* エンジンを初期化。重みデータは内部にコピーされる。
 * weight_count が fe_weight_count(model_size) と一致しない場合NULLを返す。 */
FE_PUBLIC_API FeState* fe_create(int model_size, const float* weights, int weight_count);

/* 1フレーム(hop_sizeサンプル)を処理。
 * input: [hop_size] 入力サンプル
 * output: [hop_size] ノイズ除去済み出力サンプル */
FE_PUBLIC_API void fe_process_frame(FeState* state, const float* input, float* output);

/* 前処理の有効/無効を切り替える。0=無効, 非0=有効。 */
FE_PUBLIC_API void fe_set_hpf(FeState* state, int enabled);
FE_PUBLIC_API void fe_set_agc(FeState* state, int enabled);

/* 内部状態をリセット(GRU隠れ状態、STFTオーバーラップ)。重みは保持。 */
FE_PUBLIC_API void fe_reset(FeState* state);

/* リソース解放。state以降のアクセスは未定義。 */
FE_PUBLIC_API void fe_destroy(FeState* state);

/* 現在のモデルのhopサイズを返す。 */
FE_PUBLIC_API int fe_get_hop_size(const FeState* state);

#endif /* FE_FASTENHANCER_H */
