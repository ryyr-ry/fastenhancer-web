/*
 * pipeline.h — HPF/AGC 前処理パイプライン
 *
 * HPF: 2次Butterworth高域通過フィルタ (80Hz @ 48kHz)
 * AGC: RMSベース自動ゲイン制御
 *
 * 両方ともデフォルト無効。fe_set_hpf()/fe_set_agc()で有効化。
 */

#ifndef FE_PIPELINE_H
#define FE_PIPELINE_H

/* HPF状態: Direct Form II Transposed biquad */
typedef struct {
    float b0;
    float b1;
    float b2;
    float a1;
    float a2;
    float z1;
    float z2;
} FeHpfState;

/* AGC状態 */
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
