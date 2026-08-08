#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// Предвыборка данных для скрытия латентности памяти
#define WISP_PREFETCH_READ_T0(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#define WISP_PREFETCH_READ_T1(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)

/**
 * Векторное сложение: c = a + b
 * Использует non-temporal store для больших массивов, чтобы не вымывать кэш.
 * Полностью векторизовано с масками для обработки хвостов.
 */
void wisp_avx512_vec_add(const float* a, const float* b, float* c, size_t count) {
    if (count == 0) return;

    size_t i = 0;
    
    // Основной цикл: обработка по 16 элементов (512 бит)
    // Предвыборка данных на 4 итерации вперед (~256 байт)
    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(a + i + 64);
        WISP_PREFETCH_READ_T0(b + i + 64);
        
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        
        // Non-temporal store для больших данных (не загрязняет кэш)
        _mm512_stream_ps(c + i, vc);
    }

    // Обработка остатка с использованием масок (без скалярного цикла)
    if (i < count) {
        __mmask16 mask = (__mmask16)_kxnz_mask(_cvtu32_mask16((1U << (count - i)) - 1), 0xFFFF); 
        // Более простой способ создания маски для хвоста:
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);

        __m512 va = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, a + i);
        __m512 vb = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        
        _mm512_mask_storeu_ps(c + i, k, vc);
    }
    
    _mm_sfence(); // Гарантия завершения потоковых записей перед продолжением
}

/**
 * Векторное масштабирование: out = in * scalar
 * Использует FMA и маски.
 */
void wisp_avx512_scale_float(float* data, size_t count, float scalar) {
    if (count == 0) return;

    size_t i = 0;
    __m512 v_scalar = _mm512_set1_ps(scalar);

    for (; i + 64 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        
        __m512 v_data = _mm512_loadu_ps(data + i);
        // FMA: data * scalar + 0.0f (хотя mul_ps тоже быстро, FMA универсальнее для других ops)
        __m512 v_result = _mm512_mul_ps(v_data, v_scalar);
        
        _mm512_stream_ps(data + i, v_result);
    }

    if (i < count) {
        unsigned int tail_mask = (1U << (count - i)) - 1;
        __mmask16 k = _cvtu32_mask16(tail_mask);

        __m512 v_data = _mm512_mask_loadu_ps(_mm512_setzero_ps(), k, data + i);
        __m512 v_result = _mm512_mul_ps(v_data, v_scalar);
        
        _mm512_mask_storeu_ps(data + i, k, v_result);
    }
    _mm_sfence();
}

/**
 * Скалярное произведение: sum(a * b)
 * Использует FMA для накопления суммы внутри вектора.
 */
float wisp_avx512_dot_product(const float* a, const float* b, size_t count) {
    if (count == 0) return 0.0f;

    size_t i = 0;
    __m512 v_sum = _mm512_setzero_ps();

    for (; i + 16 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(a + i + 64);
        WISP_PREFETCH_READ_T0(b + i + 64);

        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        
        // FMA: v_sum = va * vb + v_sum
        v_sum = _mm512_fmadd_ps(va, vb, v_sum);
    }

    // Горизонтальное сложение (редукция)
    float result[16] __attribute__((aligned(64)));
    _mm512_store_ps(result, v_sum);

    float acc = 0.0f;
    for (int j = 0; j < 16; j++) {
        acc += result[j];
    }

    // Обработка хвоста
    for (; i < count; i++) {
        acc += a[i] * b[i];
    }

    return acc;
}

/**
 * Квадрат евклидовой нормы: sum(x * x)
 * Критично для нормализации и функций потерь.
 */
float wisp_avx512_norm_sq(const float* x, size_t count) {
    if (count == 0) return 0.0f;

    size_t i = 0;
    __m512 v_sum = _mm512_setzero_ps();

    for (; i + 16 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(x + i + 64);

        __m512 vx = _mm512_loadu_ps(x + i);
        // FMA: v_sum = vx * vx + v_sum
        v_sum = _mm512_fmadd_ps(vx, vx, v_sum);
    }

    float result[16] __attribute__((aligned(64)));
    _mm512_store_ps(result, v_sum);

    float acc = 0.0f;
    #pragma nounroll
    for (int j = 0; j < 16; j++) {
        acc += result[j];
    }

    for (; i < count; i++) {
        acc += x[i] * x[i];
    }

    return acc;
}

/**
 * Поиск максимума
 */
float wisp_avx512_max_float(const float* data, size_t count) {
    if (count == 0) return -__FLT_MAX__;

    size_t i = 0;
    __m512 v_max = _mm512_set1_ps(data[0]);

    for (; i + 16 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        __m512 v_data = _mm512_loadu_ps(data + i);
        v_max = _mm512_max_ps(v_max, v_data);
    }

    float result[16] __attribute__((aligned(64)));
    _mm512_store_ps(result, v_max);

    float max_val = result[0];
    for (int j = 1; j < 16; j++) {
        if (result[j] > max_val) max_val = result[j];
    }

    for (; i < count; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    return max_val;
}

/**
 * Сумма элементов (int32)
 */
int64_t wisp_avx512_sum_int32(const int32_t* data, size_t count) {
    if (count == 0) return 0;

    size_t i = 0;
    __m512i v_sum = _mm512_setzero_si512();

    for (; i + 16 <= count; i += 16) {
        WISP_PREFETCH_READ_T0(data + i + 64);
        __m512i v_data = _mm512_loadu_si512(data + i);
        v_sum = _mm512_add_epi32(v_sum, v_data);
    }

    int32_t temp[16] __attribute__((aligned(64)));
    _mm512_store_si512(temp, v_sum);

    int64_t result = 0;
    for (int j = 0; j < 16; j++) {
        result += temp[j];
    }

    for (; i < count; i++) {
        result += data[i];
    }

    return result;
}
