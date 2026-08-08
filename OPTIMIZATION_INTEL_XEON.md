# Оптимизация WISP Engine для Intel Xeon Gold 6348

## Выполненные оптимизации

### 1. Компиляторные флаги (CMakeLists.txt)

Добавлена поддержка `-march=x86-64-v4` для Intel Xeon Gold 6348 (Ice Lake):

```cmake
target_compile_options(_wisp_core PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra -march=x86-64-v4>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -march=x86-64-v4>
)
```

**Что включает x86-64-v4:**
- AVX-512F: Базовые 512-битные операции
- AVX-512VL: Векторизация на 256/128 бит
- AVX-512DQ: Улучшенные операции с двойной точностью
- AVX-512BW: Байт/слово операции для квантования
- AVX-512VNNI: Векторные нейронные сети (int4 dequant)

### 2. AVX-512 векторизованные ядра (прототип)

Создан файл `wisp_avx512.c` с реализацией:
- `avx512_gemv_f16()` - GEMV с fp16 весами
- `avx512_gemv_int4()` - Fused int4 dequant + GEMV
- `avx512_rmsnorm()` - RMSNorm с векторизацией
- `avx512_swiglu()` - SwiGLU активация
- `avx512_attention_scores()` - Attention score computation
- `avx512_router_logits()` - MoE router
- `avx512_residual_add()` - Residual connection
- `avx512_scale_accum()` - Scale and accumulate
- `avx512_embed_lookup()` - Embedding lookup
- `avx512_rope()` - Rotary embeddings

### 3. Асинхронное упреждающее чтение (прототип)

Создан файл `wisp_async_io.c` с реализацией:
- Thread pool для параллельного чтения с SSD
- Read-ahead буферизация
- Выравнивание по границам страниц (4KB) для DMA
- Интеграция с LRU cache системой

## Рекомендации по дальнейшей глубокой оптимизации

### 1. Полная интеграция AVX-512 ядер

Для полной интеграции AVX-512 ядер в основной движок:

```c
// В wisp_engine.c добавить диспетчеризацию:
#include "wisp_avx512.h"

static void cpu_gemv_f16_avx512(const wisp_half* W, const float* x, float* y,
                                int rows, int cols) {
    avx512_gemv_f16(W, x, y, rows, cols);
}

// В run_moe_expert() использовать AVX-512 версию:
#ifdef __AVX512F__
    avx512_gemv_int4(...);
#else
    cpu_gemv_int4(...);
#endif
```

### 2. Оптимизация int4 dequant с AVX-512 VNNI

Использовать специальные инструкции VNNI для fused dequant+MAC:

```c
#include <immintrin.h>

// VPDPBUSD - Vector Dot Product of Bytes and Signed-Dword Accumulation
__m512i result = _mm512_dpbusd_epi32(acc, nibbles, weights);
```

### 3. Многопоточный prefetch экспертов

Увеличить число потоков prefetch для Ice Lake (32 cores / 64 threads):

```c
AsyncIOConfig config = {
    .num_reader_threads = 16,  // Половина физических ядер
    .read_ahead_bytes = 4 * 1024 * 1024,  // 4MB read-ahead
    .max_pending_ops = 128,
    .use_posix_aio = 1,
    .use_hugepages = 1  // Для больших буферов
};
```

### 4. NUMA-aware аллокация памяти

Intel Xeon Gold 6348 имеет NUMA архитектуру:

```c
#include <numa.h>

// Привязка потоков к NUMA узлам
numa_bind(numa_node_of_cpu(cpu_id));

// Аллокация памяти локально к узлу
void* ptr = numa_alloc_onnode(size, node_id);
```

### 5. Hugepages для больших буферов

```bash
# Настроить hugepages в системе
echo 2048 > /proc/sys/vm/nr_hugepages

# Использовать в коде
mmap(NULL, size, PROT_READ|PROT_WRITE, 
     MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
```

### 6. Оптимизация attention с AVX-512

Flash Attention 2 с AVX-512:
- Tiled matrix multiplication
- Register blocking для softmax
- Kernel fusion (QK^T + scale + mask + softmax)

### 7. Параллелизм на уровне инструкций (ILP)

Развернуть циклы для лучшего использования execution units:

```c
// Loop unrolling для 4 итераций за раз
for (int i = 0; i < n; i += 4) {
    acc0 = _mm512_fmadd_ps(w0, x[i+0], acc0);
    acc1 = _mm512_fmadd_ps(w1, x[i+1], acc1);
    acc2 = _mm512_fmadd_ps(w2, x[i+2], acc2);
    acc3 = _mm512_fmadd_ps(w3, x[i+3], acc3);
}
```

### 8. Prefetching данных

Использовать software prefetch hints:

```c
_mm_prefetch(data + prefetch_distance, _MM_HINT_T0);
_mm_prefetch(data + prefetch_distance + 64, _MM_HINT_T1);
```

### 9. Оптимизация MoE routing

Для Top-K routing с большим числом экспертов:
- Векторизовать softmax
- Использовать bitonic sort для top-k
- Fuse router + expert dispatch

### 10. Профилирование и tuning

Использовать Intel VTune Profiler:

```bash
# Собрать с debug symbols
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Запустить профилирование
vtune -collect hotspots ./python -m wisp.benchmark
```

## Ожидаемый прирост производительности

| Оптимизация | Ожидаемый прирост |
|-------------|-------------------|
| AVX-512 GEMV | 4-8x vs scalar |
| AVX-512 int4 dequant | 8-16x vs scalar |
| Async IO prefetch | 2-3x latency hiding |
| Multi-threaded experts | 16-32x на 32 cores |
| NUMA-aware allocation | 1.5-2x memory bandwidth |
| Hugepages | 10-20% TLB miss reduction |

## Тестирование

После интеграции проверить:

```bash
cd /workspace/build
make -j4
python -c "import wisp; print(wisp.__version__)"
python scripts/benchmark.py --model test --tokens 1000
```

## Примечания

- Intel Xeon Gold 6348: 32 cores / 64 threads, 2.6 GHz base, 48MB L3
- Поддержка AVX-512: все расширения Ice Lake
- Память: 8 каналов DDR4-3200, до 204.8 GB/s bandwidth
- PCIe 4.0: 64 lanes для NVMe SSD
