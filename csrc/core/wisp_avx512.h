/*
 * wisp_avx512.h — AVX-512 optimized kernels for Intel Ice Lake+ (Xeon Gold 6348)
 * 
 * Usage: Include this header and call functions when __AVX512F__ is defined.
 * Runtime CPU feature detection should be performed before calling.
 */

#ifndef WISP_AVX512_H
#define WISP_AVX512_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Basic vector operations */
void wisp_avx512_scale_float(float* data, size_t count, float scalar);
void wisp_avx512_add(const float* a, const float* b, float* c, size_t count);
void wisp_avx512_multiply(const float* a, const float* b, float* c, size_t count);
float wisp_avx512_dot_product(const float* a, const float* b, size_t count);
float wisp_avx512_l2_norm(const float* x, size_t count);
void wisp_avx512_normalize(float* x, size_t count);
int64_t wisp_avx512_sum_int32(const int32_t* data, size_t count);
float wisp_avx512_max_float(const float* data, size_t count);

/* Parallel operations */
void wisp_avx512_add_parallel(const float* a, const float* b, float* c, size_t count);
void wisp_avx512_multiply_parallel(const float* a, const float* b, float* c, size_t count);
float wisp_avx512_dot_product_parallel(const float* a, const float* b, size_t count);

/* GEMV kernels — critical for expert inference */
void wisp_avx512_gemv_f16(const uint16_t* W, const float* x, float* y,
                          int rows, int cols);
void wisp_avx512_gemv_int4_vnni(const uint8_t* packed, const uint16_t* scales,
                                const uint16_t* zeros, float* x, float* y,
                                int rows, int cols, int group_size);

/* Softmax */
void wisp_avx512_softmax(float* data, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* WISP_AVX512_H */
