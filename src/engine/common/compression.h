/*
 * compression.h — Power compression/decompression interface
 */

#ifndef FE_COMPRESSION_H
#define FE_COMPRESSION_H

/* mag^exponent compression (NaN/Inf/negative values clamped to 0) */
void fe_power_compress(const float* input, float* output, int n, float exponent);

/* mag^(1/exponent) decompression */
void fe_power_decompress(const float* input, float* output, int n, float exponent);

/* Power compression of complex spectrum
 * scale = mag^(exponent - 1.0)  (exponent=0.3 -> scale = mag^(-0.7))
 * out_re[i] = re[i] * scale,  out_im[i] = im[i] * scale
 * When mag=0: out=0 (prevents NaN/Inf) */
void fe_power_compress_complex(const float* re, const float* im,
                                float* out_re, float* out_im,
                                int n, float exponent);

/* Power decompression of complex spectrum
 * scale = compressed_mag^(1.0/exponent - 1.0)  (exponent=0.3 -> scale = mag^2.333)
 * compress -> decompress recovers the original values */
void fe_power_decompress_complex(const float* re, const float* im,
                                  float* out_re, float* out_im,
                                  int n, float exponent);

#endif /* FE_COMPRESSION_H */
