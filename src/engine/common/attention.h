/*
 * attention.h — Multi-Head Self-Attention インターフェース
 */

#ifndef FE_ATTENTION_H
#define FE_ATTENTION_H

typedef struct {
    const float* W_q;   /* [c2 × c2] */
    const float* b_q;   /* [c2] */
    const float* W_k;
    const float* b_k;
    const float* W_v;
    const float* b_v;
    const float* W_o;   /* 出力射影 [c2 × c2] */
    const float* b_o;
    int n_heads;
    int head_dim;
    int c2;             /* = n_heads * head_dim */
} FeMhsaWeights;

/* Multi-Head Self-Attention
 * input:    [seq_len × c2]
 * output:   [seq_len × c2]
 * attn_buf: [n_heads × seq_len × seq_len] 一時バッファ
 * scratch:  [4 × seq_len × c2] Q/K/V/attn_out用一時バッファ
 */
void fe_mhsa(const FeMhsaWeights* w, const float* input,
             float* output, float* attn_buf, float* scratch, int seq_len);

/* 数値安定softmax (max subtraction) */
void fe_softmax(const float* input, float* output, int n);

#endif /* FE_ATTENTION_H */
