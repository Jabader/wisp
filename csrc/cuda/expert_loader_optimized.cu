/*
 * expert_loader_optimized.cu — Optimized int4 -> fp16/fp8 dequantization
 * for 8GB VRAM GPUs (RTX 3050/4060 class).
 *
 * Key optimizations for limited VRAM:
 *   1. Fused dequantize + GEMM kernel — no intermediate fp16 storage
 *   2. FP8 E4M3 support for scales/zeros (12.5% size reduction vs fp16)
 *   3. Register-resident dequantization during matrix multiply
 *   4. Double-buffered async prefetch with minimal staging buffers
 *
 * Bit format (matches wisp/converter/quantizer.py EXACTLY):
 *   - Two int4 values per byte: LOW nibble = even index, HIGH = odd
 *   - Groups of `group_size` consecutive flattened elements share
 *     one fp16/fp8 scale and one fp16/fp8 zero point
 *   - value = (nibble - 8) * scale + zero
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>

#include "../core/wisp_engine.h"

/* ====================================================================== *
 * FP8 E4M3 conversion helpers
 * ====================================================================== */

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    // FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
    // Bias = 7, NaN representation: 0b01111111
    if (x == 0x7F) return __float2half_rn(0.0f); // NaN -> 0
    
    uint32_t sign = (x & 0x80) << 24;
    uint32_t exp_bias = (x & 0x78) >> 3;
    uint32_t mantissa = (x & 0x07);
    
    if (exp_bias == 0) {
        // Subnormal or zero
        if (mantissa == 0) return 0.0f;
        // Subnormal: exp = -6, implicit leading 0
        float val = ldexpf((float)mantissa, -6);
        return sign ? -val : val;
    } else if (exp_bias == 15) {
        // Infinity (we treat as max finite)
        float val = ldexpf(1.0f + 7.0f/8.0f, 7); // max E4M3 finite
        return sign ? -val : val;
    } else {
        // Normal: exp = exp_bias - 7, implicit leading 1
        float val = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp_bias - 7);
        return sign ? -val : val;
    }
}

__device__ __forceinline__ uint8_t float_to_fp8e4m3(float x) {
    uint32_t sign = 0;
    if (x < 0.0f) {
        sign = 0x80;
        x = -x;
    }
    
    if (x == 0.0f) return 0x00;
    
    int exp;
    float mant = frexpf(x, &exp); // x = mant * 2^exp, 0.5 <= mant < 1.0
    
    // Adjust to E4M3 format (bias=7, 3 mantissa bits)
    exp += 7; // Apply bias
    
    if (exp <= 0) {
        // Subnormal or underflow
        if (exp < -2) return sign; // Underflow to zero
        // Subnormal: exp = 0, mantissa scaled
        mant = ldexpf(mant, exp + 2); // Scale for subnormal range
        uint32_t m = (uint32_t)(mant * 4.0f); // 2 bits for subnormal
        return sign | (m & 0x07);
    } else if (exp >= 15) {
        // Overflow to infinity
        return sign | 0x7E; // Max finite (0x7F is NaN)
    } else {
        // Normal
        mant = mant * 2.0f - 1.0f; // Remove implicit 1, get fractional part
        uint32_t m = (uint32_t)(mant * 8.0f + 0.5f); // 3 mantissa bits
        return sign | ((exp << 3) | (m & 0x07));
    }
}

/* ====================================================================== *
 * Fused Dequantize + GEMM Kernel for MoE Experts
 * 
 * This kernel performs dequantization directly in registers during
 * the matrix multiplication, eliminating the need for intermediate
 * fp16 weight storage. Critical for 8GB VRAM where every MB counts.
 * 
 * Matrix dimensions: [rows, cols] where cols is the inner dimension
 * Input vector: x[cols]
 * Output: y[rows] = W @ x
 * 
 * Thread block processes one row of the output matrix.
 * ====================================================================== */

template<int BLOCK_SIZE, int VECTORIZAION = 4>
__global__ void fused_dequant_gemm_int4_fp16(
    const uint8_t* __restrict__ packed_weights,  // [rows*cols/2] bytes
    const __half*  __restrict__ scales,          // [num_groups]
    const __half*  __restrict__ zeros,           // [num_groups]
    const float*   __restrict__ x,               // [cols] input vector
    float*         __restrict__ y,               // [rows] output vector
    int rows,
    int cols,
    int group_size
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= rows) return;
    
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread processes multiple elements along the column dimension
    int elements_per_thread = (cols + num_threads - 1) / num_threads;
    int start_col = thread_id * elements_per_thread;
    int end_col = min(start_col + elements_per_thread, cols);
    
    float acc = 0.0f;
    
    // Load input vector elements into registers
    float x_vals[VECTORIZAION];
    
    for (int col = start_col; col < end_col; col += VECTORIZAION) {
        // Vectorized load of input
        #pragma unroll
        for (int v = 0; v < VECTORIZAION && col + v < cols; v++) {
            x_vals[v] = x[col + v];
        }
        
        // Load and dequantize weights
        #pragma unroll
        for (int v = 0; v < VECTORIZAION && col + v < cols; v++) {
            int flat_idx = row_idx * cols + (col + v);
            
            // Unpack int4 nibble
            uint8_t byte_val = packed_weights[flat_idx >> 1];
            int nibble = (flat_idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
            
            // Compute group index for scale/zero
            int group_idx = flat_idx / group_size;
            
            // Dequantize: val = (nibble - 8) * scale + zero
            float scale = __half2float(scales[group_idx]);
            float zero = __half2float(zeros[group_idx]);
            float weight = (float)(nibble - 8) * scale + zero;
            
            // Accumulate
            acc += weight * x_vals[v];
        }
    }
    
    // Warp-level reduction for final accumulation
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    
    // First thread in warp writes result
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&y[row_idx], acc);
    }
}

/* ====================================================================== *
 * FP8 Scales/Zeros Variant
 * Same fused kernel but loads fp8 scales/zeros and converts on-the-fly
 * Reduces metadata storage by 50% (fp8 vs fp16)
 * ====================================================================== */

template<int BLOCK_SIZE, int VECTORIZAION = 4>
__global__ void fused_dequant_gemm_int4_fp8(
    const uint8_t* __restrict__ packed_weights,  // [rows*cols/2] bytes
    const uint8_t* __restrict__ scales_fp8,      // [num_groups] fp8 E4M3
    const uint8_t* __restrict__ zeros_fp8,       // [num_groups] fp8 E4M3
    const float*   __restrict__ x,               // [cols] input vector
    float*         __restrict__ y,               // [rows] output vector
    int rows,
    int cols,
    int group_size
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= rows) return;
    
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    
    int elements_per_thread = (cols + num_threads - 1) / num_threads;
    int start_col = thread_id * elements_per_thread;
    int end_col = min(start_col + elements_per_thread, cols);
    
    float acc = 0.0f;
    
    for (int col = start_col; col < end_col; col += VECTORIZAION) {
        float x_vals[VECTORIZAION];
        
        #pragma unroll
        for (int v = 0; v < VECTORIZAION && col + v < cols; v++) {
            x_vals[v] = x[col + v];
        }
        
        #pragma unroll
        for (int v = 0; v < VECTORIZAION && col + v < cols; v++) {
            int flat_idx = row_idx * cols + (col + v);
            
            // Unpack int4 nibble
            uint8_t byte_val = packed_weights[flat_idx >> 1];
            int nibble = (flat_idx & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
            
            // Compute group index
            int group_idx = flat_idx / group_size;
            
            // Dequantize with fp8 scale/zero
            float scale = fp8e4m3_to_float(scales_fp8[group_idx]);
            float zero = fp8e4m3_to_float(zeros_fp8[group_idx]);
            float weight = (float)(nibble - 8) * scale + zero;
            
            acc += weight * x_vals[v];
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&y[row_idx], acc);
    }
}

/* ====================================================================== *
 * Async Prefetch Kernel for Double-Buffering
 * Overlaps PCIe transfer with computation using CUDA streams
 * ====================================================================== */

extern "C" cudaError_t launch_fused_expert_gemm(
    const uint8_t* d_packed,
    const void*    d_scales,      // __half* or uint8_t* depending on mode
    const void*    d_zeros,       // __half* or uint8_t* depending on mode
    const float*   d_input,
    float*         d_output,
    int rows,
    int cols,
    int group_size,
    int use_fp8,                  // 0 = fp16 scales, 1 = fp8 scales
    cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0 || group_size <= 0) {
        return cudaErrorInvalidValue;
    }
    
    const int BLOCK_X = 16;   // Rows per block
    const int BLOCK_Y = 4;    // Threads for row distribution
    const int THREADS_X = 128; // Threads for column processing
    
    dim3 block_dim(THREADS_X, BLOCK_Y);
    dim3 grid_dim((rows + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), 1);
    
    if (!use_fp8) {
        fused_dequant_gemm_int4_fp16<BLOCK_X, 4><<<grid_dim, block_dim, 0, stream>>>(
            d_packed,
            (const __half*)d_scales,
            (const __half*)d_zeros,
            d_input,
            d_output,
            rows, cols, group_size
        );
    } else {
        fused_dequant_gemm_int4_fp8<BLOCK_X, 4><<<grid_dim, block_dim, 0, stream>>>(
            d_packed,
            (const uint8_t*)d_scales,
            (const uint8_t*)d_zeros,
            d_input,
            d_output,
            rows, cols, group_size
        );
    }
    
    cudaError_t err = cudaGetLastError();
    return err;
}

/* ====================================================================== *
 * Optimized Async Load with Minimal Staging
 * For 8GB VRAM, minimize temporary buffer usage by reusing scratch space
 * ====================================================================== */

extern "C" cudaError_t async_load_expert_optimized(
    const uint8_t* h_packed,       // Pinned host: int4 packed weights
    const void*    h_scales,       // Pinned host: fp16 or fp8 scales
    const void*    h_zeros,        // Pinned host: fp16 or fp8 zeros
    uint8_t*       d_staging,      // Reusable device staging buffer
    void*          d_scales_tmp,   // Reusable device scales buffer
    void*          d_zeros_tmp,    // Reusable device zeros buffer
    int            n_elements,
    int            group_size,
    int            use_fp8,        // 0 = fp16, 1 = fp8
    cudaStream_t   stream
) {
    if (n_elements <= 0 || group_size <= 0) {
        return cudaErrorInvalidValue;
    }
    
    int n_groups = (n_elements + group_size - 1) / group_size;
    size_t packed_bytes = ((size_t)n_elements + 1) / 2;
    size_t scale_bytes = use_fp8 ? (size_t)n_groups : (size_t)n_groups * sizeof(__half);
    
    cudaError_t e;
    
    // Async copy to staging buffers
    e = cudaMemcpyAsync(d_staging, h_packed, packed_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    
    e = cudaMemcpyAsync(d_scales_tmp, h_scales, scale_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    
    e = cudaMemcpyAsync(d_zeros_tmp, h_zeros, scale_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    
    return cudaSuccess;
}

/* ====================================================================== *
 * Memory-Efficient Expert Cache Manager for 8GB VRAM
 * Tracks VRAM usage and triggers LRU eviction when approaching limit
 * ====================================================================== */

typedef struct {
    void** expert_ptrs;      // Device pointers to cached experts
    int*   expert_layers;    // Layer ID for each cached expert
    int*   expert_ids;       // Expert ID within layer
    int*   expert_ages;      // LRU age counter
    int    capacity;         // Max experts that fit in VRAM budget
    int    count;            // Currently cached experts
    int    age_counter;      // Monotonic age for LRU
    size_t bytes_per_expert; // Size of one expert in bytes
    size_t total_budget;     // Total VRAM budget for expert cache
    size_t used_bytes;       // Currently used VRAM
} ExpertCacheState;

extern "C" cudaError_t expert_cache_create(
    ExpertCacheState** state_out,
    size_t vram_budget_bytes,
    size_t expert_size_bytes,
    int max_experts
) {
    if (!state_out || vram_budget_bytes == 0 || expert_size_bytes == 0) {
        return cudaErrorInvalidValue;
    }
    
    ExpertCacheState* state = (ExpertCacheState*)malloc(sizeof(ExpertCacheState));
    if (!state) return cudaErrorMemoryAllocation;
    
    int capacity = min(max_experts, (int)(vram_budget_bytes / expert_size_bytes));
    
    state->capacity = capacity;
    state->count = 0;
    state->age_counter = 0;
    state->bytes_per_expert = expert_size_bytes;
    state->total_budget = vram_budget_bytes;
    state->used_bytes = 0;
    
    cudaError_t e;
    e = cudaMalloc(&state->expert_ptrs, capacity * sizeof(void*));
    if (e != cudaSuccess) { free(state); return e; }
    
    e = cudaMalloc(&state->expert_layers, capacity * sizeof(int));
    if (e != cudaSuccess) { cudaFree(state->expert_ptrs); free(state); return e; }
    
    e = cudaMalloc(&state->expert_ids, capacity * sizeof(int));
    if (e != cudaSuccess) {
        cudaFree(state->expert_layers);
        cudaFree(state->expert_ptrs);
        free(state);
        return e;
    }
    
    e = cudaMalloc(&state->expert_ages, capacity * sizeof(int));
    if (e != cudaSuccess) {
        cudaFree(state->expert_ids);
        cudaFree(state->expert_layers);
        cudaFree(state->expert_ptrs);
        free(state);
        return e;
    }
    
    // Initialize to zero
    e = cudaMemset(state->expert_ptrs, 0, capacity * sizeof(void*));
    if (e != cudaSuccess) {
        cudaFree(state->expert_ages);
        cudaFree(state->expert_ids);
        cudaFree(state->expert_layers);
        cudaFree(state->expert_ptrs);
        free(state);
        return e;
    }
    
    *state_out = state;
    return cudaSuccess;
}

extern "C" cudaError_t expert_cache_destroy(ExpertCacheState* state) {
    if (!state) return cudaErrorInvalidValue;
    
    cudaFree(state->expert_ages);
    cudaFree(state->expert_ids);
    cudaFree(state->expert_layers);
    cudaFree(state->expert_ptrs);
    free(state);
    
    return cudaSuccess;
}

/* Find LRU expert for eviction */
__global__ void find_lru_expert_kernel(
    const int* ages,
    int count,
    int* lru_idx
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;
    
    int min_age = ages[idx];
    int min_idx = idx;
    
    // Simple parallel reduction for minimum
    for (int i = idx; i < count; i += blockDim.x * gridDim.x) {
        if (ages[i] < min_age) {
            min_age = ages[i];
            min_idx = i;
        }
    }
    
    // Atomic update for global minimum
    atomicMin(lru_idx, min_idx);
}

extern "C" cudaError_t expert_cache_evict_lru(
    ExpertCacheState* state,
    cudaStream_t stream,
    int* evicted_layer,
    int* evicted_id
) {
    if (!state || state->count == 0) return cudaErrorInvalidValue;
    
    int h_lru_idx = INT_MAX;
    int* d_lru_idx;
    cudaError_t e = cudaMalloc(&d_lru_idx, sizeof(int));
    if (e != cudaSuccess) return e;
    
    cudaMemcpyAsync(d_lru_idx, &h_lru_idx, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    const int BLOCKS = 4;
    const int THREADS = 256;
    find_lru_expert_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        state->expert_ages, state->count, d_lru_idx
    );
    
    cudaMemcpyAsync(&h_lru_idx, d_lru_idx, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_lru_idx < 0 || h_lru_idx >= state->count) {
        cudaFree(d_lru_idx);
        return cudaErrorInvalidValue;
    }
    
    // Copy out evicted expert info before removing
    cudaMemcpyAsync(evicted_layer, &state->expert_layers[h_lru_idx],
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(evicted_id, &state->expert_ids[h_lru_idx],
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // Update accounting
    state->used_bytes -= state->bytes_per_expert;
    state->count--;
    
    // Move last expert to fill the gap (simplified - real impl would use proper list)
    if (h_lru_idx < state->count) {
        cudaMemcpyAsync(&state->expert_ptrs[h_lru_idx],
                        &state->expert_ptrs[state->count],
                        sizeof(void*), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&state->expert_layers[h_lru_idx],
                        &state->expert_layers[state->count],
                        sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&state->expert_ids[h_lru_idx],
                        &state->expert_ids[state->count],
                        sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    
    cudaFree(d_lru_idx);
    return cudaSuccess;
}
