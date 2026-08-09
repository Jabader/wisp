/**
 * WISP C++ Runtime Implementation
 * 
 * Implementation file for high-performance runtime components
 */

#include "wisp_runtime_cpp.h"
#include <deque>
#include <functional>
#include <algorithm>
#include <random>

namespace wisp {

// Explicit template instantiations for common types
template class LockFreeCache<int, void*>;
template class LockFreeCache<size_t, float*>;

// AVX-512 optimized softmax implementation
void softmax_avx512(float* logits, size_t vocab_size) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    
    // Vectorized max reduction with AVX-512
    for (size_t i = 0; i < vocab_size; i += 16) {
        __m512 v_logits = _mm512_loadu_ps(logits + i);
        __m512 v_max = _mm512_max_ps(v_logits, _mm512_set1_ps(max_val));
        
        // Horizontal max
        float lane_max[16];
        _mm512_storeu_ps(lane_max, v_max);
        for (int j = 0; j < 16 && i + j < vocab_size; ++j) {
            if (lane_max[j] > max_val) max_val = lane_max[j];
        }
    }
    
    // Compute exp(x - max) and sum using scalar fallback (AVX-512 exp not available in all compilers)
    float sum = 0.0f;
    
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < vocab_size; i += 16) {
        __m512 v_logits = _mm512_loadu_ps(logits + i);
        __m512 v_inv_sum = _mm512_set1_ps(inv_sum);
        v_logits = _mm512_mul_ps(v_logits, v_inv_sum);
        _mm512_storeu_ps(logits + i, v_logits);
    }
}

// AVX-512 optimized top-k selection
void topk_avx512(const float* logits, size_t vocab_size, int k, 
                 int* indices, float* values) {
    // Simple selection algorithm (can be optimized further)
    std::vector<std::pair<float, int>> scored_indices(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        scored_indices[i] = {logits[i], static_cast<int>(i)};
    }
    
    // Partial sort for top-k
    std::partial_sort(scored_indices.begin(), 
                      scored_indices.begin() + k,
                      scored_indices.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });
    
    for (int i = 0; i < k; ++i) {
        indices[i] = scored_indices[i].second;
        values[i] = scored_indices[i].first;
    }
}

// Categorical sampling with alias method (O(1) sampling after O(n) setup)
int categorical_sample_alias(const float* probs, size_t vocab_size, 
                             std::mt19937& rng) {
    // Build alias table (simplified version)
    std::vector<int> small, large;
    std::vector<float> alias(vocab_size);
    std::vector<float> prob(vocab_size);
    
    float scale = static_cast<float>(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        prob[i] = probs[i] * scale;
        if (prob[i] < 1.0f) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }
    
    while (!small.empty() && !large.empty()) {
        int s = small.back();
        small.pop_back();
        int l = large.back();
        large.pop_back();
        
        prob[s] = prob[s];
        alias[s] = l;
        
        prob[l] = prob[l] + prob[s] - 1.0f;
        if (prob[l] < 1.0f) {
            small.push_back(l);
        } else {
            large.push_back(l);
        }
    }
    
    // Sample in O(1)
    std::uniform_int_distribution<size_t> dist(0, vocab_size - 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    
    size_t idx = dist(rng);
    float coin = uniform(rng);
    
    if (coin < prob[idx]) {
        return static_cast<int>(idx);
    } else {
        return alias[idx];
    }
}

} // namespace wisp
