/*
 * pipeline.h — HPF/AGC preprocessing pipeline
 *
 * HPF: 2nd-order Butterworth high-pass filter (80Hz @ 48kHz)
 * AGC: RMS-based automatic gain control
 *
 * Both are disabled by default. Enable with fe_set_hpf()/fe_set_agc().
 */

#ifndef FE_PIPELINE_H
#define FE_PIPELINE_H

/* HPF state: Direct Form II Transposed biquad */
typedef struct {
    float b0;
    float b1;
    float b2;
    float a1;
    float a2;
    float z1;
    float z2;
} FeHpfState;

/* AGC state */
typedef struct {
    float env;
    float gain;
} FeAgcState;

void fe_hpf_init(FeHpfState* s);
void fe_hpf_process(FeHpfState* s, float* buf, int len);
void fe_hpf_reset(FeHpfState* s);

void fe_agc_init(FeAgcState* s);
void fe_agc_process(FeAgcState* s, float* buf, int len);
void fe_agc_reset(FeAgcState* s);

#endif /* FE_PIPELINE_H */
