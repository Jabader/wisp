/*
 * wisp_avx512.c — AVX-512 optimized kernels for Intel Ice Lake+ (Xeon Gold 6348)
 * 
 * This file contains hand-optimized AVX-512 implementations of common operations:
 * - Vector scaling (float32)
 * - Vector sum (int32)
 * - Vector max (float32)
 * - Matrix-vector multiply (fp16 weights, fp32 compute)
 * - INT4 dequant + GEMV (VNNI accelerated)
 *
 * Compile with: -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -O3
 */

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ======================================================================= *
 * FP16 <-> FP32 conversion using AVX-512DQ
 * ======================================================================= */

static inline __m512 wisp_avx512_cvtph_ps(const __m256i h) {
    return _mm512_cvtph_ps(h);
}

static inline __m256i wisp_avx512_cvtps_ph(const __m512 f) {
    return _mm512_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

/* ======================================================================= *
 * Vector Operations
 * ======================================================================= */

void wisp_avx512_scale_float(float* data, size_t count, float scalar) {
    size_t i = 0;
    __m512 v_scalar = _mm512_set1_ps(scalar);
    
    for (; i + 16 <= count; i += 16) {
        __m512 v_data = _mm512_loadu_ps(data + i);
        __m512 v_result = _mm512_mul_ps(v_data, v_scalar);
        _mm512_storeu_ps(data + i, v_result);
    }
    
    for (; i < count; i++) {
        data[i] *= scalar;
    }
}

int64_t wisp_avx512_sum_int32(const int32_t* data, size_t count) {
    size_t i = 0;
    __m512i v_sum = _mm512_setzero_si512();
    
    for (; i + 16 <= count; i += 16) {
        __m512i v_data = _mm512_loadu_si512(data + i);
        v_sum = _mm512_add_epi32(v_sum, v_data);
    }
    
    int64_t result = 0;
    int32_t temp[16];
    _mm256_storeu_si256((__m256i*)temp, _mm512_extracti32x8_epi32(v_sum, 0));
    _mm256_storeu_si256((__m256i*)(temp + 8), _mm512_extracti32x8_epi32(v_sum, 1));
    
    for (int j = 0; j < 16; j++) {
        result += temp[j];
    }
    
    for (; i < count; i++) {
        result += data[i];
    }
    
    return result;
}

float wisp_avx512_max_float(const float* data, size_t count) {
    if (count == 0) return 0.0f;
    
    size_t i = 0;
    __m512 v_max = _mm512_set1_ps(data[0]);
    
    for (; i + 16 <= count; i += 16) {
        __m512 v_data = _mm512_loadu_ps(data + i);
        v_max = _mm512_max_ps(v_max, v_data);
    }
    
    float temp[16];
    _mm512_storeu_ps(temp, v_max);
    
    float result = temp[0];
    for (int j = 1; j < 16; j++) {
        if (temp[j] > result) result = temp[j];
    }
    
    for (; i < count; i++) {
        if (data[i] > result) result = data[i];
    }
    
    return result;
}

/* ======================================================================= *
 * FP16 GEMV: y = W * x, where W is fp16, x/y are fp32
 * ======================================================================= */

void wisp_avx512_gemv_f16(const uint16_t* W, const float* x, float* y,
                          size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const uint16_t* row = W + r * cols;
        __m512 acc = _mm512_setzero_ps();
        size_t c = 0;
        
        for (; c + 32 <= cols; c += 32) {
            __m256i h_lo = _mm256_loadu_si256((const __m256i*)(row + c));
            __m256i h_hi = _mm256_loadu_si256((const __m256i*)(row + c + 16));
            
            __m512 f_lo = _mm512_cvtph_ps(h_lo);
            __m512 f_hi = _mm512_cvtph_ps(h_hi);
            
            __m512 x_lo = _mm512_loadu_ps(x + c);
            __m512 x_hi = _mm512_loadu_ps(x + c + 16);
            
            acc = _mm512_fmadd_ps(f_lo, x_lo, acc);
            acc = _mm512_fmadd_ps(f_hi, x_hi, acc);
        }
        
        float sum = _mm512_reduce_add_ps(acc);
        
        for (; c < cols; c++) {
            union { uint16_t u; float f; } conv;
            conv.u = row[c];
            uint32_t sign = (uint32_t)(conv.u & 0x8000) << 16;
            uint32_t exp = (conv.u >> 10) & 0x1F;
            uint32_t mant = conv.u & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                if (mant == 0) bits = sign;
                else {
                    exp = 127 - 15 + 1;
                    while (!(mant & 0x400)) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    bits = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7F800000 | (mant << 13);
            } else {
                bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            float wf;
            memcpy(&wf, &bits, 4);
            sum += wf * x[c];
        }
        y[r] = sum;
    }
}

/* ======================================================================= *
 * INT4 GEMV with VNNI: dequant + multiply-accumulate in one pass
 * Format: 2 ints per byte (low nibble first), group_size elements per scale/zero
 * Note: Simplified version without full VNNI - uses standard AVX-512 ops
 * ======================================================================= */

void wisp_avx512_gemv_int4_vnni(const uint8_t* packed, const uint16_t* scales,
                                const uint16_t* zeros, const float* x, float* y,
                                size_t rows, size_t cols, size_t group_size) {
    size_t groups_per_row = (cols + group_size - 1) / group_size;
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_packed = packed + r * (cols / 2);
        const uint16_t* row_scales = scales + r * groups_per_row;
        const uint16_t* row_zeros = zeros + r * groups_per_row;
        
        float sum = 0.0f;
        size_t c = 0;
        
        /* Scalar fallback for correctness - full VNNI impl needs more work */
        for (; c < cols; c++) {
            uint8_t byte = row_packed[c >> 1];
            int nib = (c & 1) ? (byte >> 4) : (byte & 0x0F);
            size_t g = c / group_size;
            float scale = (float)row_scales[g];
            float zero = (float)row_zeros[g];
            sum += ((float)nib - 8.0f) * scale * x[c] + zero * x[c];
        }
        y[r] = sum;
    }
}

/* ======================================================================= *
 * Softmax with AVX-512: exp(x - max) / sum(exp(x - max))
 * Note: Uses scalar expf as _mm512_exp_ps requires SVML library
 * ======================================================================= */

void wisp_avx512_softmax(float* data, size_t count) {
    if (count == 0) return;
    
    float max_val = wisp_avx512_max_float(data, count);
    
    size_t i = 0;
    /* Vector subtraction of max value */
    __m512 v_max = _mm512_set1_ps(max_val);
    for (; i + 16 <= count; i += 16) {
        __m512 v_data = _mm512_loadu_ps(data + i);
        __m512 v_shifted = _mm512_sub_ps(v_data, v_max);
        _mm512_storeu_ps(data + i, v_shifted);
    }
    
    /* Scalar exp for remaining elements and aligned portion */
    for (i = 0; i < count; i++) {
        data[i] = expf(data[i]);
    }
    
    /* Sum using vector reduction */
    float sum = 0.0f;
    i = 0;
    __m512 v_sum = _mm512_setzero_ps();
    for (; i + 16 <= count; i += 16) {
        __m512 v_data = _mm512_loadu_ps(data + i);
        v_sum = _mm512_add_ps(v_sum, v_data);
    }
    
    float temp[16];
    _mm512_storeu_ps(temp, v_sum);
    for (int j = 0; j < 16; j++) {
        sum += temp[j];
    }
    for (; i < count; i++) {
        sum += data[i];
    }
    
    if (sum == 0.0f) sum = 1e-9f;
    
    wisp_avx512_scale_float(data, count, 1.0f / sum);
}
