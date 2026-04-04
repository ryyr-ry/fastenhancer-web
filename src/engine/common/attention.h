/*
 * attention.h — Multi-Head Self-Attention interface
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
    const float* W_o;   /* Output projection [c2 x c2] */
    const float* b_o;
    int n_heads;
    int head_dim;
    int c2;             /* = n_heads * head_dim */
} FeMhsaWeights;

/* Multi-Head Self-Attention
 * input:    [seq_len x c2]
 * output:   [seq_len x c2]
 * attn_buf: [n_heads x seq_len x seq_len] temporary buffer
 * scratch:  [4 x seq_len x c2] temporary buffer for Q/K/V/attn_out
 */
void fe_mhsa(const FeMhsaWeights* w, const float* input,
             float* output, float* attn_buf, float* scratch, int seq_len);

/* Numerically stable softmax (max subtraction) */
void fe_softmax(const float* input, float* output, int n);

#endif /* FE_ATTENTION_H */
