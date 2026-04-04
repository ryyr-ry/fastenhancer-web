/*
 * compression.h — パワー圧縮/逆圧縮インターフェース
 */

#ifndef FE_COMPRESSION_H
#define FE_COMPRESSION_H

/* mag^exponent 圧縮 (NaN/Inf/負値は0にクランプ) */
void fe_power_compress(const float* input, float* output, int n, float exponent);

/* mag^(1/exponent) 逆圧縮 */
void fe_power_decompress(const float* input, float* output, int n, float exponent);

/* 複素スペクトルのパワー圧縮
 * scale = mag^(exponent - 1.0)  (exponent=0.3 → scale = mag^(-0.7))
 * out_re[i] = re[i] * scale,  out_im[i] = im[i] * scale
 * mag=0の場合: out=0 (NaN/Inf防止) */
void fe_power_compress_complex(const float* re, const float* im,
                                float* out_re, float* out_im,
                                int n, float exponent);

/* 複素スペクトルのパワー復号
 * scale = compressed_mag^(1.0/exponent - 1.0)  (exponent=0.3 → scale = mag^2.333)
 * compress → decompress で元の値に戻る */
void fe_power_decompress_complex(const float* re, const float* im,
                                  float* out_re, float* out_im,
                                  int n, float exponent);

#endif /* FE_COMPRESSION_H */
