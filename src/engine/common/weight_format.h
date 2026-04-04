/*
 * weight_format.h — Weight binary format definition
 *
 * Binary layout:
 *   [0..3]   Magic "FEW1" (4 bytes)
 *   [4..7]   Version uint32_t LE (currently=1)
 *   [8..11]  Model size uint32_t LE (0=Tiny, 1=Base, 2=Small)
 *   [12..15] Weight count uint32_t LE (number of float32 elements)
 *   [16..19] CRC32 uint32_t LE (computed over float32 data section only)
 *   [20..]   float32 LE x weight count
 *
 * Header size: 20 bytes
 * Weight data is float32 little-endian.
 * fe_init() explicitly decodes the payload as little-endian byte sequence.
 * CRC32 is computed over the float32 data section (excluding header).
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
