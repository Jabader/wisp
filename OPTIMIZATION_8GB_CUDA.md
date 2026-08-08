# Оптимизация WISP для GPU с 8GB VRAM (Nvidia 3050/4060 и аналоги)

## Анализ текущего состояния

### Формат хранения экспертов (WBIN2/int4)

Текущий формат эксперта в файле `.bin`:
```
Header (120 байт):
  - magic: u32 = 0x57455850
  - version: u32 = 1
  - layer_id: u32
  - expert_id: u32
  - group_size: u32 (=64)
  - n_mats: u32 (=3)
  - 3x WispExpertMat (32 байта каждая):
      rows, cols: u32
      packed_bytes, scales_bytes, zeros_bytes: u64

Data blobs:
  - packed: uint8[n_elements/2]  (2 int4 значения в байте)
  - scales: fp16[n_groups]       (один scale на группу из 64 элементов)
  - zeros: fp16[n_groups]        (один zero point на группу)
```

**Математика размера эксперта:**
```
expert_size = (rows * cols / 2)                    # packed weights
            + (rows * cols / group_size) * 2       # scales (fp16)
            + (rows * cols / group_size) * 2       # zeros (fp16)
            
Для Mixtral 8x7B (hidden=4096, inter=14336):
  expert_params = 4096 * 14336 * 3 = 176,160,768 параметров
  packed_bytes = 176,160,768 / 2 = 88,080,384 байт
  n_groups = 176,160,768 / 64 = 2,752,512 групп
  scales_bytes = 2,752,512 * 2 = 5,505,024 байт
  zeros_bytes = 2,752,512 * 2 = 5,505,024 байт
  total = 88,080,384 + 5,505,024 + 5,505,024 = 99,090,432 байт (~94.5 MB)
```

### Проблемы для 8GB VRAM

1. **Dense веса не помещаются полностью:**
   - Mixtral 8x7B dense: ~3.2GB (помещается)
   - Mixtral 8x22B dense: ~10.7GB (НЕ помещается)
   - GLM-5.2 dense: ~9.9GB (НЕ помещается)
   - Kimi K3 dense: ~12GB (НЕ помещается)

2. **Экспертный кэш ограничен:**
   - При 8GB VRAM и 3.2GB на dense остаётся ~4.8GB для экспертов
   - Это ~50 экспертов Mixtral 8x7B (из 256 total = 20% в VRAM)
   - Остальные эксперты должны грузиться из RAM/SSD

3. **Пропускная способность:**
   - PCIe 3.0 x16: ~16 GB/s (реально ~12-14 GB/s)
   - PCIe 4.0 x16: ~32 GB/s (реально ~24-28 GB/s)
   - DDR4/DDR5 RAM: 40-100 GB/s
   - NVMe SSD: 3-7 GB/s

## Предлагаемые оптимизации

### 1. Оптимизированный формат WBIN2-GPU

**Цель:** Уменьшить размер экспертов на 25-30% без потери качества

```python
# Новый формат с улучшенным packing
class QuantizedTensorGPU:
    packed: torch.Tensor   # uint8, 4 значения int4 в 2 байтах (unchanged)
    scales: torch.Tensor   # fp8 E4M3, один на группу (вместо fp16)
    zeros: torch.Tensor    # fp8 E4M3, один на группу (вместо fp16)
    block_scales: torch.Tensor  # fp16, один на блок из 8 групп (новая оптимизация)
```

**Преимущества:**
- scales/zeros: fp16 (2 байта) → fp8 (1 байт) = экономия 50% на метаданных
- block_scales позволяют использовать меньшую точность для per-group scales
- Общая экономия: ~10-15% от размера эксперта

**Изменения в quantizer.py:**
```python
def quantize_int4_gpu(weight: torch.Tensor, group_size: int = 64) -> QuantizedTensorGPU:
    # ... та же логика квантования ...
    
    # Конвертируем scales/zeros в fp8
    scales_fp8 = scales.to(torch.float8_e4m3fn)
    zeros_fp8 = zeros.to(torch.float8_e4m3fn)
    
    # Добавляем block-level scales для лучшей точности
    n_blocks = (n_groups + 7) // 8
    block_scales = scales.view(n_blocks, -1).mean(dim=1).to(torch.float16)
    
    return QuantizedTensorGPU(
        packed=packed,
        scales=scales_fp8,
        zeros=zeros_fp8,
        block_scales=block_scales,
        rows=rows, cols=cols, group_size=group_size,
    )
```

### 2. CUDA ядро с fused dequantization + GEMM

**Цель:** Декомпрессия прямо в регистрах GPU во время умножения матриц

```cuda
// expert_loader.cu - новое ядро
__global__ void dequant_gemv_fused_kernel(
    const uint8_t* __restrict__ packed_weights,
    const __half*  __restrict__ scales,
    const __half*  __restrict__ zeros,
    const __half*  __restrict__ input,
    __half*        __restrict__ output,
    int rows, int cols, int group_size
) {
    int row = blockIdx.y;
    int col_tile = blockIdx.x;
    
    // Загружаем chunk входных данных в shared memory
    extern __shared__ __half shared_input[];
    
    // Каждая нить обрабатывает несколько элементов
    float acc = 0.0f;
    
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        // Unpack int4 из packed format
        uint8_t byte = packed_weights[row * (cols/2) + (c >> 1)];
        int nibble = (c & 1) ? (byte >> 4) : (byte & 0x0F);
        
        // Dequantize
        int g = c / group_size;
        float scale = __half2float(scales[row * n_groups + g]);
        float zero = __half2float(zeros[row * n_groups + g]);
        float weight = (nibble - 8.0f) * scale + zero;
        
        // Fused multiply-add
        acc += weight * __half2float(input[c]);
    }
    
    // Reduction в пределах блока
    output[row] = __float2half(acc);
}
```

**Преимущества:**
- Нет отдельного этапа декомпрессии
- Данные декомпрессируются только один раз и сразу используются
- Экономия памяти: не нужно хранить decompressed веса в VRAM

### 3. Агрессивный префетчинг с предсказанием экспертов

**Цель:** Скрыть задержку загрузки экспертов из RAM/SSD

```c
// wisp_engine.c - улучшенный prefetch
typedef struct {
    uint32_t layer;
    uint32_t expert_id;
    float probability;      // Предсказанная вероятность выбора
    int priority;           // Приоритет загрузки
} PrefetchCandidate;

// Предсказание на основе истории + паттернов внимания
static void predict_next_experts(WispEngine* eng, 
                                 PrefetchCandidate* candidates,
                                 int max_candidates) {
    // Используем simple Markov chain на основе последних N токенов
    // Эксперты, выбранные в похожих контекстах, имеют更高的 вероятность
    
    // Для MoE моделей: эксперты часто выбираются кластерами
    // Если выбран эксперт E_i в слое L, вероятно будут выбраны
    // эксперты из того же "кластера" в соседних слоях
}
```

### 4. Multi-GPU стратегия для 8GB карт

**Цель:** Использовать несколько 8GB GPU для больших моделей

```python
# multi_gpu_example.py - оптимизировано для 8GB GPU
def setup_dual_8gb(model_config):
    """
    Стратегия для двух 8GB GPU:
    - GPU 0: dense weights + 50% экспертов каждого слоя
    - GPU 1: 50% экспертов каждого слоя + KV cache
    
    Перекрытие вычислений и передачи данных через PCIe/NVLink
    """
    experts_per_layer = model_config.n_experts
    experts_per_gpu = experts_per_layer // 2
    
    # Распределяем экспертов равномерно по GPU
    gpu0_experts = list(range(0, experts_per_layer, 2))  # Чётные
    gpu1_experts = list(range(1, experts_per_layer, 2))  # Нечётные
    
    return {
        'gpu0': {
            'dense': True,
            'experts': gpu0_experts,
            'kv_cache': False,
        },
        'gpu1': {
            'dense': False,
            'experts': gpu1_experts,
            'kv_cache': True,
        }
    }
```

### 5. Оптимизация LRU кэша для ограниченной VRAM

**Цель:** Максимально эффективно использовать доступную VRAM

```c
// lru_cache.c - адаптивный размер кэша
typedef struct {
    LRUNode* head;
    LRUNode* tail;
    size_t current_bytes;
    size_t max_bytes;
    size_t min_bytes;      // Минимальный гарантированный размер
    float dynamic_fraction; // Динамическая доля от свободной VRAM
    
    // Статистика для адаптации
    int hit_count;
    int miss_count;
    uint64_t last_resize_time_ns;
} AdaptiveLRUCache;

// Автоматическая подстройка размера кэша на основе hit rate
static void adaptive_resize(AdaptiveLRUCache* cache, 
                           size_t free_vram_bytes) {
    float hit_rate = cache->hit_count / 
                    (float)(cache->hit_count + cache->miss_count);
    
    if (hit_rate > 0.8) {
        // Высокий hit rate - можно увеличить кэш
        cache->max_bytes = min(free_vram_bytes * 0.7, 
                              cache->max_bytes * 1.2);
    } else if (hit_rate < 0.5) {
        // Низкий hit rate - уменьшаем кэш, возможно эксперты не те
        cache->max_bytes = max(cache->min_bytes,
                              cache->max_bytes * 0.8);
    }
}
```

### 6. Изменения в auto_config.py для 8GB GPU

```python
# wisp/system/auto_config.py

class AutoConfig:
    def calculate(self, profile: SystemProfile,
                  model: ModelAdapter) -> TierConfig:
        # ... существующий код ...
        
        # Специальная логика для 8GB GPU
        if profile.gpu_count > 0:
            vram_total = profile.gpus[profile.primary_gpu_index].vram_total_bytes
            
            # Для 8GB GPU используем более агрессивную оптимизацию
            if 7 * GB <= vram_total <= 9 * GB:
                # Уменьшаем safety buffer для максимизации доступной памяти
                BUFFER = 300 * MB  # Вместо 500 MB
                
                # Более агрессивное использование VRAM (80% вместо 75%)
                MAX_VRAM_FRACTION = 0.80
                
                # Включаем compression для экспертов
                USE_GPU_COMPRESSION = True
```

### 7. Benchmark скрипт для 8GB GPU

```python
# scripts/benchmark_8gb.py
import torch
import time
from wisp.runtime.engine import WispEngine

def benchmark_8gb_optimization(model_path, batch_size=1):
    """
    Бенчмарк оптимизаций для 8GB GPU:
    1. Базовая версия (int4, fp16 scales)
    2. Оптимизированная (int4, fp8 scales, fused kernel)
    3. С префетчингом
    """
    
    engine = WispEngine(
        model_path=model_path,
        use_gpu=True,
        gpu_indices=[0],
        max_vram_fraction=0.80,  # 80% для 8GB карты
        enable_fused_kernels=True,
        enable_fp8_scales=True,
        prefetch_depth=4,
    )
    
    # Тест prefill
    prompt = "Объясни квантовую запутанность простыми словами"
    start = time.time()
    tokens = engine.generate(prompt, max_tokens=100)
    prefill_time = time.time() - start
    
    # Тест decode
    decode_speed = len(tokens) / (time.time() - start)
    
    print(f"Prefill time: {prefill_time:.2f}s")
    print(f"Decode speed: {decode_speed:.2f} tok/s")
    print(f"VRAM usage: {engine.get_vram_usage() / 1e9:.2f} GB")
    
    return {
        'prefill_time': prefill_time,
        'decode_speed': decode_speed,
        'vram_usage': engine.get_vram_usage(),
    }
```

## План внедрения

### Фаза 1: Базовая оптимизация (1-2 недели)
1. [ ] Обновить `auto_config.py` для 8GB GPU
2. [ ] Добавить fp8 support для scales/zeros
3. [ ] Оптимизировать LRU кэш с адаптивным размером

### Фаза 2: CUDA оптимизации (2-3 недели)
1. [ ] Реализовать fused dequant+GEMM ядра
2. [ ] Добавить double buffering для pre-fetching
3. [ ] Оптимизировать memory layout для coalesced access

### Фаза 3: Multi-GPU поддержка (1-2 недели)
1. [ ] Добавить распределение экспертов по GPU
2. [ ] Реализовать overlap вычислений и transfers
3. [ ] Тестирование на dual 8GB конфигурации

### Фаза 4: Тестирование и бенчмарки (1 неделя)
1. [ ] Создать comprehensive benchmark suite
2. [ ] Протестировать на различных моделях
3. [ ] Документировать результаты

## Ожидаемые результаты

Для **Mixtral 8x7B** на **Nvidia 3050 8GB**:
- **До оптимизации:** ~8-12 tok/s (с частыми page faults)
- **После оптимизации:** ~15-20 tok/s (стабильно)

Для **GLM-5.2** на **dual 3050 8GB**:
- **До оптимизации:** Не помещается
- **После оптимизации:** ~3-5 tok/s (с распределением по GPU)

## Риски и ограничения

1. **fp8 точность:** Может потребоваться calibration для некоторых моделей
2. **PCIe bottleneck:** На старых системах с PCIe 3.0 скорость будет ограничена
3. **Driver support:** Требуется CUDA 11.8+ для fp8 operations

## Заключение

Предложенные оптимизации позволят эффективно использовать GPU с 8GB VRAM для запуска MoE моделей формата WBIN2. Ключевые улучшения:
- Уменьшение размера экспертов на 10-15% через fp8 scales
- Ускорение инференса на 30-50% через fused kernels
- Поддержка больших моделей через multi-GPU распределение
