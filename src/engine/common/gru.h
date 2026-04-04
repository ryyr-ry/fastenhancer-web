/*
 * gru.h — GRU (Gated Recurrent Unit) interface
 */

#ifndef FE_GRU_H
#define FE_GRU_H

/* Maximum allowed hidden_size at runtime (stack array size) */
#define FE_GRU_MAX_HIDDEN 128

typedef struct {
    const float* W_z;     /* Update gate input weights [hidden x input] */
    const float* U_z;     /* Update gate hidden weights [hidden x hidden] */
    const float* b_z;     /* Update gate bias [hidden] */
    const float* W_r;     /* Reset gate input weights */
    const float* U_r;     /* Reset gate hidden weights */
    const float* b_r;     /* Reset gate bias */
    const float* W_n;     /* Candidate hidden state input weights */
    const float* U_n;     /* Candidate hidden state hidden weights */
    const float* b_in_n;  /* Candidate state input bias (outside r gate) */
    const float* b_hn_n;  /* Candidate state hidden bias (inside r gate) */
    int input_size;
    int hidden_size;
} FeGruWeights;

/* GRU single step: updates hidden in-place */
void fe_gru_step(const FeGruWeights* w, const float* input, float* hidden);

/* Reset hidden state to zero */
void fe_gru_reset_hidden(float* hidden, int size);

#endif /* FE_GRU_H */
