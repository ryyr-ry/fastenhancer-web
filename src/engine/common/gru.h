/*
 * gru.h — GRU (Gated Recurrent Unit) インターフェース
 */

#ifndef FE_GRU_H
#define FE_GRU_H

/* 実行時に許容される hidden_size の上限 (スタック配列サイズ) */
#define FE_GRU_MAX_HIDDEN 128

typedef struct {
    const float* W_z;     /* 更新ゲート入力重み [hidden × input] */
    const float* U_z;     /* 更新ゲート隠れ重み [hidden × hidden] */
    const float* b_z;     /* 更新ゲートバイアス [hidden] */
    const float* W_r;     /* リセットゲート入力重み */
    const float* U_r;     /* リセットゲート隠れ重み */
    const float* b_r;     /* リセットゲートバイアス */
    const float* W_n;     /* 候補隠れ状態入力重み */
    const float* U_n;     /* 候補隠れ状態隠れ重み */
    const float* b_in_n;  /* 候補状態入力バイアス (rゲート外側) */
    const float* b_hn_n;  /* 候補状態隠れバイアス (rゲート内側) */
    int input_size;
    int hidden_size;
} FeGruWeights;

/* GRU 1ステップ: hidden を in-place 更新 */
void fe_gru_step(const FeGruWeights* w, const float* input, float* hidden);

/* 隠れ状態をゼロリセット */
void fe_gru_reset_hidden(float* hidden, int size);

#endif /* FE_GRU_H */
