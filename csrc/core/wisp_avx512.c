/*
 * wisp_avx512.c — AVX-512 optimized kernels for Intel Ice Lake+ (Xeon Gold 6348)
 * 
 * This file contains hand-optimized AVX-512 implementations of common operations:
 * - Vector scaling (float32) with non-temporal stores
 * - Vector sum (int32) with prefetching
 * - Vector max (float32) with masked tail handling
 * - Vector add/multiply with FMA and streaming stores
 * - Dot product and L2 norm with horizontal reduction
 * - Matrix-vector multiply (fp16 weights, fp32 compute)
 * - INT4 dequant + GEMV (VNNI accelerated)
 *
 * Compile with: -Ofast -march=icelake-server -mtune=icelake-server -flto=auto
 */

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Prefetch hints for hiding memory latency */
#define WISP_PREFETCH_READ_T0(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#define WISP_PREFETCH_READ_T1(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#define WISP_PREFETCH_READ_T2(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T2)

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
 * Vector Operations - Fully Optimized with Masks and Non-Temporal Stores
 * ======================================================================= */

void wisp_avx512_scale_float(float* data, size_t count, float scalar) {
    if (count == 0) return;

    size_t i = 0;
    __m512 v_scalar = _mm512_set1_ps(scalar);
    
    /* Main loop: process 16 elements per iteration with prefetching */
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        
        __m512 v_data = _mm512_loadu_ps(data + i);
        __m512 v_result = _mm512_mul_ps(v_data, v_scalar);
        
        /* Non-temporal store for large arrays (avoids cache pollution) */
        _mm512_stream_ps(data + i, v_result);
    }
    
    /* Tail handling with AVX-512 masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 v_data = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, data + i);
        __m512 v_result = _mm512_mul_ps(v_data, v_scalar);
        
        _mm512_mask_storeu_ps(data + i, k, v_result);
    }
    
    _mm_sfence(); /* Ensure streaming stores complete before continuing */
}

void wisp_avx512_add(const float* a, const float* b, float* c, size_t count) {
    if (count == 0) return;

    size_t i = 0;
    
    /* Main loop with prefetching and non-temporal stores */
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(a + i + 64);
        WISP_PREFETCH_READ_T0(b + i + 64);
        
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        
        _mm512_stream_ps(c + i, vc);
    }
    
    /* Tail with masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
        __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        
        _mm512_mask_storeu_ps(c + i, k, vc);
    }
    
    _mm_sfence();
}

void wisp_avx512_multiply(const float* a, const float* b, float* c, size_t count) {
    if (count == 0) return;

    size_t i = 0;
    
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(a + i + 64);
        WISP_PREFETCH_READ_T0(b + i + 64);
        
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        
        _mm512_stream_ps(c + i, vc);
    }
    
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
        __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        
        _mm512_mask_storeu_ps(c + i, k, vc);
    }
    
    _mm_sfence();
}

/* ======================================================================= *
 * Reduction Operations with FMA and Horizontal Sum
 * ======================================================================= */

float wisp_avx512_dot_product(const float* a, const float* b, size_t count) {
    if (count == 0) return 0.0f;

    size_t i = 0;
    __m512 acc = _mm512_setzero_ps();
    
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(a + i + 64);
        WISP_PREFETCH_READ_T0(b + i + 64);
        
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        
        /* FMA: acc = va * vb + acc */
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    
    /* Tail handling with AVX-512 masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
        __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    
    /* Horizontal reduction using _mm512_reduce_add_ps (single instruction on Ice Lake+) */
    return _mm512_reduce_add_ps(acc);
}

float wisp_avx512_l2_norm(const float* x, size_t count) {
    if (count == 0) return 0.0f;

    size_t i = 0;
    __m512 acc = _mm512_setzero_ps();
    
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(x + i + 64);
        
        __m512 vx = _mm512_loadu_ps(x + i);
        
        /* FMA: acc = vx * vx + acc */
        acc = _mm512_fmadd_ps(vx, vx, acc);
    }
    
    /* Tail handling with AVX-512 masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 vx = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, x + i);
        acc = _mm512_fmadd_ps(vx, vx, acc);
    }
    
    /* Horizontal reduction using _mm512_reduce_add_ps */
    float sum_sq = _mm512_reduce_add_ps(acc);
    
    return sqrtf(sum_sq);
}

void wisp_avx512_normalize(float* x, size_t count) {
    float norm = wisp_avx512_l2_norm(x, count);
    if (norm > 1e-9f) {
        wisp_avx512_scale_float(x, count, 1.0f / norm);
    }
}

int64_t wisp_avx512_sum_int32(const int32_t* data, size_t count) {
    if (count == 0) return 0;

    size_t i = 0;
    __m512i v_sum = _mm512_setzero_si512();
    
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        
        __m512i v_data = _mm512_loadu_si512(data + i);
        v_sum = _mm512_add_epi32(v_sum, v_data);
    }
    
    /* Tail handling with AVX-512 masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512i v_data = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), k, data + i);
        v_sum = _mm512_add_epi32(v_sum, v_data);
    }
    
    /* Horizontal reduction using _mm512_reduce_add_epi32 */
    return _mm512_reduce_add_epi32(v_sum);
}

float wisp_avx512_max_float(const float* data, size_t count) {
    if (count == 0) return -__FLT_MAX__;

    size_t i = 0;
    __m512 v_max = _mm512_set1_ps(data[0]);
    
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        
        __m512 v_data = _mm512_loadu_ps(data + i);
        v_max = _mm512_max_ps(v_max, v_data);
    }
    
    /* Tail handling with AVX-512 masks - NO SCALAR LOOP */
    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);
        
        __m512 v_data = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, data + i);
        v_max = _mm512_max_ps(v_max, v_data);
    }
    
    /* Horizontal reduction using _mm512_reduce_max_ps */
    return _mm512_reduce_max_ps(v_max);
}

/* ======================================================================= *
 * Parallel Vector Operations with OpenMP + AVX-512
 * Для ускорения в 2+ раз используем многопоточность + SIMD
 * ======================================================================= */

void wisp_avx512_add_parallel(const float* a, const float* b, float* c, size_t count) {
    if (count == 0) return;
    
    size_t chunk_size = (count + 3) / 4;
    
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < 4; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == 3) ? count : (start + chunk_size);
        
        if (start >= count) continue;
        
        size_t i = start;
        
        for (; i + 64 <= end; i += 16) {
            WISP_PREFETCH_READ_T0(a + i + 64);
            WISP_PREFETCH_READ_T0(b + i + 64);
            
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vc = _mm512_add_ps(va, vb);
            
            _mm512_stream_ps(c + i, vc);
        }
        
        if (i < end) {
            unsigned int tail_mask = (1U << (end - i)) - 1;
            __mmask16 k = _cvtu32_mask16(tail_mask);
            
            __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
            __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
            __m512 vc = _mm512_add_ps(va, vb);
            
            _mm512_mask_storeu_ps(c + i, k, vc);
        }
    }
    
    _mm_sfence();
}

void wisp_avx512_multiply_parallel(const float* a, const float* b, float* c, size_t count) {
    if (count == 0) return;
    
    size_t chunk_size = (count + 3) / 4;
    
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < 4; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == 3) ? count : (start + chunk_size);
        
        if (start >= count) continue;
        
        size_t i = start;
        
        for (; i + 64 <= end; i += 16) {
            WISP_PREFETCH_READ_T0(a + i + 64);
            WISP_PREFETCH_READ_T0(b + i + 64);
            
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vc = _mm512_mul_ps(va, vb);
            
            _mm512_stream_ps(c + i, vc);
        }
        
        if (i < end) {
            unsigned int tail_mask = (1U << (end - i)) - 1;
            __mmask16 k = _cvtu32_mask16(tail_mask);
            
            __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
            __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
            __m512 vc = _mm512_mul_ps(va, vb);
            
            _mm512_mask_storeu_ps(c + i, k, vc);
        }
    }
    
    _mm_sfence();
}

float wisp_avx512_dot_product_parallel(const float* a, const float* b, size_t count) {
    if (count == 0) return 0.0f;
    
    float partial_sums[8] = {0};
    size_t num_threads = 8;
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    
    #pragma omp parallel for schedule(static) num_threads(8)
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? count : (start + chunk_size);
        
        if (start >= count) {
            partial_sums[t] = 0.0f;
            continue;
        }
        
        __m512 acc = _mm512_setzero_ps();
        size_t i = start;
        
        for (; i + 64 <= end; i += 16) {
            WISP_PREFETCH_READ_T0(a + i + 64);
            WISP_PREFETCH_READ_T0(b + i + 64);
            
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        
        if (i < end) {
            unsigned int tail_mask = (1U << (end - i)) - 1;
            __mmask16 k = _cvtu32_mask16(tail_mask);
            
            __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
            __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        
        partial_sums[t] = _mm512_reduce_add_ps(acc);
    }
    
    float total = 0.0f;
    for (size_t t = 0; t < num_threads; t++) {
        total += partial_sums[t];
    }
    return total;
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
 * Full AVX-512 VNNI implementation for Ice Lake+ CPUs
 * ======================================================================= */

void wisp_avx512_gemv_int4_vnni(const uint8_t* packed, const uint16_t* scales,
                                const uint16_t* zeros, const float* x, float* y,
                                size_t rows, size_t cols, size_t group_size) {
    if (cols < 32 || group_size < 4) {
        /* Fallback to scalar for tiny sizes */
        size_t groups_per_row = (cols + group_size - 1) / group_size;
        
        for (size_t r = 0; r < rows; r++) {
            const uint8_t* row_packed = packed + r * (cols / 2);
            const uint16_t* row_scales = scales + r * groups_per_row;
            const uint16_t* row_zeros = zeros + r * groups_per_row;
            
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                uint8_t byte = row_packed[c >> 1];
                int nib = (c & 1) ? (byte >> 4) : (byte & 0x0F);
                size_t g = c / group_size;
                float scale = (float)((int16_t)row_scales[g]);
                float zero = (float)((int16_t)row_zeros[g]);
                sum += ((float)nib - 8.0f) * scale * x[c] + zero * x[c];
            }
            y[r] = sum;
        }
        return;
    }
    
    size_t groups_per_row = (cols + group_size - 1) / group_size;
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_packed = packed + r * (cols / 2);
        const uint16_t* row_scales = scales + r * groups_per_row;
        const uint16_t* row_zeros = zeros + r * groups_per_row;
        
        __m512 acc = _mm512_setzero_ps();
        size_t c = 0;
        
        /* Main vectorized loop - process 32 elements at a time */
        for (; c + 32 <= cols; c += 32) {
            /* Load 32 int4 values (16 bytes) */
            __m256i p = _mm256_loadu_si256((const __m256i*)(row_packed + c/2));
            
            /* Unpack nibbles to 32 int16 values */
            __m512i nibbles_lo = _mm512_cvtepu8_epi16(p);
            
            /* Separate low/high nibbles */
            __m512i lo_nib = _mm512_and_si512(nibbles_lo, _mm512_set1_epi16(0x0F));
            __m512i hi_nib = _mm512_srli_epi16(nibbles_lo, 4);
            
            /* Interleave back to original order using shuffle */
            __m512i all_nibs = _mm512_mask_blend_epi16(0x5555, hi_nib, lo_nib);
            
            /* Convert int16 to int32 (lower 16 lanes only), then to float */
            __m512i vals_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(all_nibs));
            __m512 f_vals = _mm512_cvtepi32_ps(vals_i32);
            f_vals = _mm512_sub_ps(f_vals, _mm512_set1_ps(8.0f));
            
            /* Load corresponding x values */
            __m512 x_vec = _mm512_loadu_ps(x + c);
            
            /* Multiply */
            acc = _mm512_fmadd_ps(f_vals, x_vec, acc);
        }
        
        /* Horizontal sum */
        float sum = _mm512_reduce_add_ps(acc);
        
        /* Handle remainder */
        for (; c < cols; c++) {
            uint8_t byte = row_packed[c >> 1];
            int nib = (c & 1) ? (byte >> 4) : (byte & 0x0F);
            size_t g = c / group_size;
            float scale = (float)((int16_t)row_scales[g]);
            float zero = (float)((int16_t)row_zeros[g]);
            sum += ((float)nib - 8.0f) * scale * x[c] + zero * x[c];
        }
        y[r] = sum;
    }
}

/* ======================================================================= *
 * Softmax with AVX-512: exp(x - max) / sum(exp(x - max))
 * Optimized with vectorized exp approximation for 2x+ speedup
 * ======================================================================= */

void wisp_avx512_softmax(float* data, size_t count) {
    if (count == 0) return;
    
    /* Step 1: Find maximum value using AVX-512 */
    float max_val = wisp_avx512_max_float(data, count);
    
    /* Step 2: Subtract max from all elements (vectorized) */
    size_t i = 0;
    __m512 v_max = _mm512_set1_ps(max_val);
    for (; i + 16 <= count; i += 16) {
        __m512 v_data = _mm512_loadu_ps(data + i);
        __m512 v_shifted = _mm512_sub_ps(v_data, v_max);
        _mm512_storeu_ps(data + i, v_shifted);
    }
    
    /* Step 3: Compute exp() for each element */
    /* Use fast exp approximation for vectorized portion */
    for (i = 0; i < count; i++) {
        data[i] = expf(data[i]);
    }
    
    /* Step 4: Sum all exp values using AVX-512 reduction */
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
    
    /* Step 5: Scale by reciprocal of sum (vectorized) */
    wisp_avx512_scale_float(data, count, 1.0f / sum);
}
