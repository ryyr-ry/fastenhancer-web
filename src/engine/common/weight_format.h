/*
 * weight_format.h — 重みバイナリフォーマット定義
 *
 * バイナリレイアウト:
 *   [0..3]   マジック "FEW1" (4 bytes)
 *   [4..7]   バージョン uint32_t LE (現在=1)
 *   [8..11]  モデルサイズ uint32_t LE (0=Tiny, 1=Base, 2=Small)
 *   [12..15] 重み数 uint32_t LE (float32要素数)
 *   [16..19] CRC32 uint32_t LE (float32データ部のみ対象)
 *   [20..]   float32 LE × 重み数
 *
 * ヘッダサイズ: 20 bytes
 * 重みデータはfloat32リトルエンディアン。
 * fe_init() は payload を little-endian バイト列として明示的に復号する。
 * CRC32はfloat32データ部(ヘッダ除外)に対して計算。
 */

#ifndef FE_WEIGHT_FORMAT_H
#define FE_WEIGHT_FORMAT_H

#include <stdint.h>

#define FE_WEIGHT_MAGIC_0   'F'
#define FE_WEIGHT_MAGIC_1   'E'
#define FE_WEIGHT_MAGIC_2   'W'
#define FE_WEIGHT_MAGIC_3   '1'
#define FE_WEIGHT_VERSION   1
#define FE_WEIGHT_HEADER_SIZE 20

typedef struct {
    uint8_t  magic[4];
    uint32_t version;
    uint32_t model_size;
    uint32_t weight_count;
    uint32_t crc32;
} FeWeightHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(FeWeightHeader) == FE_WEIGHT_HEADER_SIZE,
               "FeWeightHeader size must match FE_WEIGHT_HEADER_SIZE");
#else
typedef char fe_weight_header_size_check_[
    (sizeof(FeWeightHeader) == FE_WEIGHT_HEADER_SIZE) ? 1 : -1
];
#endif

#endif /* FE_WEIGHT_FORMAT_H */
