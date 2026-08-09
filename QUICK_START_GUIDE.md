# WISP Quick Start Guide
## Быстрый старт для GPU с 8GB-80GB VRAM и серверов на Intel Xeon

Это руководство описывает полный процесс конвертации и запуска MoE моделей в WISP с поддержкой различных конфигураций VRAM и multi-GPU setups.

---

## Содержание

1. [Требования к системе](#требования-к-системе)
2. [Установка WISP](#установка-wisp)
3. [Конвертация модели](#конвертация-модели)
4. [Запуск inference](#запуск-inference)
5. [Конфигурация VRAM](#конфигурация-vram)
6. [Multi-GPU настройка](#multi-gpu-настройка)
7. [Сервер на Intel Xeon Gold 6348](#сервер-на-intel-xeon-gold-6348)
8. [Troubleshooting](#troubleshooting)

---

## Поддерживаемые модели

WISP поддерживает следующие MoE архитектуры:

| Модель | Семейство | Параметры | Активные params | Эксперты | VRAM (dense) | Диск (int4) |
|--------|-----------|-----------|-----------------|----------|--------------|-------------|
| **Mixtral 8x7B** | `mixtral-8x7b` | 47B | 13B | 8/layer, top-2 | 3.2 GB | 27 GB |
| **Mixtral 8x22B** | `mixtral-8x22b` | 141B | 39B | 8/layer, top-2 | 10.7 GB | 90 GB |
| **GLM-5.2** | `glm-5.2` | 744B | 40B | 256/layer, top-8 | 9.9 GB | 370 GB |
| **DeepSeek-V3** | `deepseek-v3` | 671B | 37B | 256/layer, top-8 | 9.5 GB | 340 GB |
| **DeepSeek-R1** | `deepseek-r1` | 671B | 37B | 256/layer, top-8 | 9.5 GB | 340 GB |
| **Kimi K3** | `kimi-k3` | 2.8T | 104B | 896/layer, top-16 | 12 GB | 1.4 TB |
| **Qwen3.6-35B-A3B** | `qwen3.6-35b` | 35B | 3.6B | 64/layer, top-4 | 7.5 GB | 22 GB |

### Qwen3.6-35B-A3B — идеальная модель для тестирования

**Почему Qwen3.6-35B-A3B отлично подходит для оценки качества:**

- ✅ **Небольшой размер**: 35B total параметров (vs 744B у GLM-5.2)
- ✅ **Низкая активация**: Только 3.6B параметров на токен
- ✅ **Эффективный MoE**: 64 эксперта на слой, выбирает 4 лучших
- ✅ **Быстрая конвертация**: 22 GB на диске vs 1.4 TB у Kimi K3
- ✅ **Работает на 8GB GPU**: Dense веса (7.5GB) + ~14 экспертов помещаются в RTX 3050/4060
- ✅ **256K контекст**: Поддержка длинных документов
- ✅ **GQA внимание**: 32 query heads / 8 KV heads — эффективно
- ✅ **Оригинальные веса**: 67 GB (fp16) — быстро скачиваются

**Архитектура:**
- 48 трансформер слоёв
- 64 эксперта на слой (всего 3,072 эксперта)
- Top-4 routing (выбирает 4 лучших эксперта на токен)
- Размер одного эксперта: ~4.8 MB (int4 + fp16 scales/zeros)
- Dense веса (attention + embeddings + norms): 7.5 GB
- Всего экспертов: 3,072 (48 слоёв × 64 эксперта)

**Конвертация Qwen3.6-35B:**
```bash
wisp convert --model qwen3.6-35b --output ./models
```

**Запуск на 8GB GPU:**
```bash
wisp run --model ./models/qwen3.6-35b --vram-limit 6 --display-mode igpu
```

**VRAM требования:**
- Dense only: 7.5 GB
- Dense + 10 экспертов: 8.9 GB
- Dense + 50 экспертов: 14.7 GB (RTX 4070 Ti 16GB)
- Dense + 100 экспертов: 21.9 GB (RTX 3090/4090 24GB)

---

## Требования к системе

### Минимальные требования

| Компонент | Требование | Примечание |
|-----------|------------|------------|
| **CPU** | Intel/AMD x86_64 | AVX2 обязательно, AVX-512 рекомендуется |
| **RAM** | 32 GB | 64+ GB для больших моделей |
| **NVMe SSD** | PCIe 3.0+ | Скорость влияет на cold token performance |
| **GPU** | NVIDIA CUDA 7.0+ | Опционально, но рекомендуется |
| **CUDA Toolkit** | 12.0+ | Для GPU режима |

### Рекомендуемые конфигурации VRAM

| GPU | VRAM | Поддерживаемые модели | Режим работы |
|-----|------|----------------------|--------------|
| **RTX 3050/4060** | 8GB | Mixtral 8x7B | Full expert cache в VRAM+RAM |
| **RTX 3060** | 12GB | Mixtral 8x7B, GLM-5.2 (dense) | Оптимально для 7B-class |
| **RTX 4070 Ti** | 16GB | Mixtral 8x7B/8x22B (dense) | Хороший баланс |
| **RTX 3090/4090** | 24GB | GLM-5.2, DeepSeek-V3/R1 | Большинство моделей |
| **RTX 6000 Ada** | 48GB | Kimi K3 (dense + часть экспертов) | Профессиональная карта |
| **A100/H100** | 80GB | Полная загрузка Kimi K3 | Серверный уровень |

---

## Установка WISP

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/zeroextub-collab/wisp.git
cd wisp
```

### Шаг 2: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 3: Сборка C/CUDA расширений

#### GPU режим (по умолчанию):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

#### CPU-only режим (без CUDA):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWISP_NO_CUDA=ON
cmake --build build
```

#### Проверка сборки:
```bash
python -c "from wisp import _wisp_core; print('WISP core loaded successfully')"
```

---

## Конвертация модели

### Базовая конвертация

```bash
wisp convert --model mixtral-8x7b --output ./models
```

**Что происходит:**
1. Загружаются веса из Hugging Face
2. Эксперты квантуются в int4 (WBIN2 формат)
3. Создаётся 3-tier структура (VRAM/RAM/SSD)
4. Верифицируется каждый эксперт

### Конвертация с оптимизацией FP8

Для GPU с ограниченной VRAM (8GB-16GB) используйте FP8 для scales/zeros:

```bash
wisp convert --model mixtral-8x7b --output ./models --quant int4-fp8
```

**Преимущества FP8:**
- Экономия 10-15% размера экспертов
- Mixtral 8x7B expert: 49.5MB → 46.8MB
- Минимальная потеря точности (<0.5%)

### Конвертация больших моделей

#### GLM-5.2 (744B параметров, 9.9GB dense)
```bash
wisp convert --model glm-5.2 --output ./models
# Требуется: ~370GB NVMe, 64GB+ RAM
```

#### Kimi K3 (2.8T параметров, 12GB dense)
```bash
wisp convert --model kimi-k3 --output ./models
# Требуется: ~1.4TB NVMe, 128GB+ RAM
```

#### Qwen3.6-35B-A3B (35B параметров, 7.5GB dense) — РЕКОМЕНДУЕТСЯ для тестирования
```bash
wisp convert --model qwen3.6-35b --output ./models
# Требуется: ~22 GB NVMe, 16GB+ RAM
# Оригинал на HF: 67GB (fp16), 26 файлов по ~2-4GB
# Время конвертации: ~5-10 минут на NVMe PCIe 4.0
```

**Преимущества Qwen3.6-35B-A3B для оценки качества:**
- Быстрая конвертация (22 GB vs 1.4TB у Kimi K3)
- Работает на 8GB GPU (RTX 3050/4060) — dense (7.5GB) + ~14 экспертов
- Низкая активация (3.6B params/token) для высокой скорости
- Современная архитектура MoE с 64 экспертами на слой

### Параметры конвертации

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `--model` | Название модели | **Требуется** |
| `--output` | Директория вывода | **Требуется** |
| `--quant` | Тип квантования (`int4`, `int4-fp8`) | `int4` |
| `--source` | Локальный путь к HF shard'ам | `None` (скачать) |
| `--group-size` | Размер группы квантования | `64` |

---

## Запуск Inference

### Одноразовый запрос (one-shot)

```bash
wisp run \
    --model ./models/mixtral-8x7b \
    --prompt "Объясни квантовую запутанность простыми словами" \
    --max-tokens 256 \
    --stream
```

### Интерактивный чат

```bash
wisp chat \
    --model ./models/mixtral-8x7b \
    --max-tokens 512
```

**Команды в чате:**
- `/clear` — очистить историю
- `/stats` — показать статистику кэша
- `/quit` или `/exit` — выйти

### Benchmarking

```bash
wisp benchmark \
    --model ./models/mixtral-8x7b \
    --tokens 100 \
    --show-stats
```

---

## Конфигурация VRAM

WISP автоматически определяет доступную VRAM и распределяет эксперты по 3 уровням:
1. **VRAM Tier** — самые быстрые, хранятся на GPU
2. **RAM Tier** — средние по скорости, в системной памяти
3. **SSD Tier** — холодные эксперты, загружаются по требованию

### Ручная настройка VRAM

Используйте параметр `--vram-limit` для ограничения используемой VRAM:

```bash
# Ограничить VRAM до 6GB на 8GB карте (оставить 2GB для дисплея)
wisp run --model ./models/mixtral-8x7b --vram-limit 6

# Использовать только 50% доступной VRAM
wisp run --model ./models/mixtral-8x7b --vram-fraction 0.5
```

### Конфигурация для разных размеров VRAM

#### 8GB VRAM (RTX 3050/4060)

```bash
# Mixtral 8x7B: ~20 экспертов в VRAM, ~30 в RAM
wisp run --model ./models/mixtral-8x7b \
    --vram-limit 6 \
    --display-mode igpu  # Если монитор подключён к iGPU

# Qwen3.6-35B-A3B: dense (7.5GB) НЕ помещается в 6GB — только CPU режим или 8GB+ GPU
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 0  # CPU-only режим (или используйте 8GB+ GPU)
```

**Ожидаемая производительность:**
- Cold start: ~8-12 tok/s
- Warm cache: ~15-20 tok/s
- Hot cache: ~22-28 tok/s

#### 8GB VRAM (RTX 3050/4060) — МИНИМУМ для Qwen3.6-35B

```bash
# Qwen3.6-35B-A3B: dense (7.5GB) + ~3 экспертов в VRAM (только 0.1% — почти CPU режим!)
# Для комфортной работы нужен 12GB+ GPU
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 7 \
    --display-mode igpu
```

#### 12GB VRAM (RTX 3060/4070)

```bash
# Mixtral 8x7B: ~35 экспертов в VRAM
wisp run --model ./models/mixtral-8x7b \
    --vram-limit 10

# Qwen3.6-35B-A3B: dense (7.5GB) + ~647 экспертов в VRAM (~21% всех экспертов)
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 10
```

#### 16GB VRAM (RTX 4070 Ti/4080)

```bash
# Mixtral 8x22B: dense + ~25 экспертов в VRAM
wisp run --model ./models/mixtral-8x22b \
    --vram-limit 14

# Qwen3.6-35B-A3B: dense (7.5GB) + ~1506 экспертов в VRAM (~49% всех экспертов)
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 14
```

#### 24GB VRAM (RTX 3090/4090)

```bash
# GLM-5.2: dense + ~128 экспертов в VRAM
wisp run --model ./models/glm-5.2 \
    --vram-limit 20

# Qwen3.6-35B-A3B: dense (7.5GB) + ~2794 экспертов в VRAM (~91% всех экспертов)
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 20
```

#### 32GB VRAM (RTX 6000 Ada / 2x RTX 3090)

```bash
# DeepSeek-V3: почти полная загрузка
wisp run --model ./models/deepseek-v3 \
    --vram-limit 28

# Qwen3.6-35B-A3B: ВСЕ эксперты помещаются (3072 эксперта = 15.36GB + 7.5GB dense = 22.86GB)
wisp run --model ./models/qwen3.6-35b \
    --vram-limit 28  # Полная загрузка в VRAM!
```

#### 40GB VRAM (A100 40GB)

```bash
# Kimi K3: dense + значительная часть экспертов
wisp run --model ./models/kimi-k3 \
    --vram-limit 36
```

#### 48GB VRAM (RTX 6000 Ada / A100 48GB)

```bash
# Kimi K3: оптимальная конфигурация
wisp run --model ./models/kimi-k3 \
    --vram-limit 44
```

#### 64GB VRAM (2x RTX 3090 / A100 80GB limited)

```bash
# Kimi K3: большинство экспертов в VRAM
wisp run --model ./models/kimi-k3 \
    --vram-limit 58
```

#### 72GB VRAM (3x RTX 3090 / 2x A100 40GB with overlap)

```bash
# Полный Kimi K3 с pipeline
wisp run --model ./models/kimi-k3 \
    --gpu-strategy pipeline
```

#### 80GB VRAM (A100/H100 80GB)

```bash
# Kimi K3: максимальная производительность
wisp run --model ./models/kimi-k3 \
    --vram-limit 74
```

### Параметр `--display-mode`

| Значение | Описание |
|----------|----------|
| `auto` | Автоопределение (по умолчанию) |
| `gpu` | Монитор на GPU (резервировать 1.5GB VRAM) |
| `igpu` | Монитор на iGPU (вся VRAM доступна) |

**Важно:** Для максимальной производительности подключите монитор к материнской плате (iGPU), а не к дискретной GPU.

---

## Multi-GPU настройка

WISP автоматически определяет несколько GPU и выбирает оптимальную стратегию:

### Стратегии распределения

| Стратегия | Описание | Когда используется |
|-----------|----------|-------------------|
| `single` | Один GPU | 1 GPU в системе |
| `dual_same` | Два одинаковых GPU | 2 GPU с равной VRAM |
| `dual_diff` | Два разных GPU | 2 GPU с разной VRAM |
| `pipeline` | Конвейерная обработка | 3+ GPU |

### Примеры конфигураций

#### Dual RTX 3090 (2x 24GB = 48GB total)

```bash
# Автоматическое определение
wisp run --model ./models/kimi-k3

# Принудительно dual_same
wisp run --model ./models/kimi-k3 --gpu-strategy dual_same
```

#### Mixed GPU (RTX 4090 24GB + RTX 3060 12GB)

```bash
# Автоматически выберет dual_diff с приоритетом 4090
wisp run --model ./models/glm-5.2
```

#### 4x RTX 3090 Pipeline (96GB total)

```bash
# Pipeline стратегия для 3+ GPU
wisp run --model ./models/kimi-k3 --gpu-strategy pipeline
```

### Параметры Multi-GPU

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `--gpu-strategy` | Стратегия (`auto`, `single`, `dual_same`, `dual_diff`, `pipeline`) | `auto` |
| `--primary-gpu` | Индекс основного GPU | `0` (авто) |
| `--secondary-gpu` | Индекс второго GPU | `None` (авто) |
| `--enable-peer-access` | Включить P2P доступ между GPU | `true` |

### Проверка Multi-GPU конфигурации

```bash
wisp info --model ./models/kimi-k3
```

Пример вывода:
```
  System:
    GPU 0 : NVIDIA GeForce RTX 3090 — 24.0 GB VRAM
    GPU 1 : NVIDIA GeForce RTX 3090 — 24.0 GB VRAM
    Display: on iGPU ✅ (full VRAM available)
    RAM   : 128.0 GB available
    NVMe  : PCIe 4.0 — 7.0 GB/s
    CPU   : Intel Xeon Gold 6348 — 56 threads

  Tier Allocation:
    GPU VRAM  : 9.9 GB dense  +  1,840 expert cache slots
    System RAM: 2,400 expert cache slots  (42.0 GB)
    NVMe SSD  : 0 experts ← entire model in silicon

  ⭐ Special: ENTIRE expert set fits in VRAM + RAM
     After first load: zero SSD reads ever.
```

---

## Сервер на Intel Xeon Gold 6348

### Характеристики процессора

| Параметр | Значение |
|----------|----------|
| **Ядра/Потоки** | 28 ядер / 56 потоков |
| **Базовая частота** | 2.6 GHz |
| **Turbo Boost** | до 3.5 GHz |
| **Кэш L3** | 42 MB |
| **TDP** | 185W |
| **AVX-512** | ✅ Поддерживается |
| **PCIe lanes** | 64 lanes PCIe 4.0 |

### Оптимизация для Xeon Gold 6348

WISP автоматически применяет оптимизации компиляции для Ice Lake архитектуры:

```bash
# При сборке автоматически применяются флаги:
# -march=icelake-server
# -mtune=icelake-server
# -mprefer-vector-width=512
# -mavx512f -mavx512vl -mavx512bw -mavx512dq
```

### Рекомендуемая конфигурация сервера

#### Конфигурация 1: Single GPU (бюджетная)

| Компонент | Модель | Примечание |
|-----------|--------|------------|
| **CPU** | Intel Xeon Gold 6348 | 28C/56T |
| **RAM** | 256 GB DDR4-3200 | ECC recommended |
| **GPU** | RTX 4090 24GB | Основная вычислительная |
| **NVMe** | 2TB PCIe 4.0 | Samsung 980 Pro / WD SN850X |
| **PSU** | 1200W | 80+ Gold |

**Поддерживаемые модели:**
- Mixtral 8x7B: Full cache в VRAM+RAM
- Mixtral 8x22B: Optimal
- GLM-5.2: Good
- DeepSeek-V3/R1: Good
- Kimi K3: Dense + partial experts
- **Qwen3.6-35B-A3B: Excellent** — dense + ~1506 экспертов в VRAM (~49% всех экспертов)

#### Конфигурация 2: Dual GPU (оптимальная)

| Компонент | Модель | Примечание |
|-----------|--------|------------|
| **CPU** | Intel Xeon Gold 6348 | 28C/56T |
| **RAM** | 512 GB DDR4-3200 | ECC |
| **GPU** | 2x RTX 3090 24GB | 48GB total VRAM |
| **NVMe** | 4TB PCIe 4.0 RAID 0 | Для больших моделей |
| **PSU** | 1600W | 80+ Platinum |
| **Motherboard** | Dual PCIe x16 support | PCIe 4.0 x8/x8 или x16/x16 |

**Поддерживаемые модели:**
- Все модели до Kimi K3 включительно
- Kimi K3: Dense + ~70% экспертов в silicon
- **Qwen3.6-35B-A3B: Perfect** — полная загрузка в single-GPU 32GB+ (22GB total)

#### Конфигурация 3: Quad GPU (максимальная)

| Компонент | Модель | Примечание |
|-----------|--------|------------|
| **CPU** | Intel Xeon Gold 6348 | 28C/56T |
| **RAM** | 1 TB DDR4-3200 | ECC |
| **GPU** | 4x RTX 3090 24GB | 96GB total VRAM |
| **NVMe** | 8TB PCIe 4.0 RAID 0 | Для всех моделей |
| **PSU** | 2000W+ | 80+ Titanium |
| **Riser cards** | PCIe 4.0 risers | Для установки 4 GPU |

**Поддерживаемые модели:**
- Полная поддержка Kimi K3
- Все эксперты помещаются в VRAM+RAM
- **Qwen3.6-35B-A3B: Overkill** — модель полностью в VRAM (22GB), остаётся 74GB для других задач

### Настройка BIOS для Xeon Gold 6348

```
1. Enable Above 4G Decoding: ON
2. ReBAR Support: ON (если поддерживается GPU)
3. SR-IOV: ON (для виртуализации)
4. PCIe Speed: Gen 4 (не Auto)
5. NUMA Optimization: Enabled
6. Hyper-Threading: ON (56 потоков)
7. Turbo Boost: ON
8. AVX-512: Enabled
```

### Производительность на Xeon Gold 6348

| Модель | Конфигурация | Cold tok/s | Warm tok/s | Hot tok/s |
|--------|-------------|------------|------------|-----------|
| **Mixtral 8x7B** | 1x RTX 4090 | 12-15 | 20-25 | 30-35 |
| **Mixtral 8x22B** | 2x RTX 3090 | 10-12 | 18-22 | 28-32 |
| **GLM-5.2** | 2x RTX 3090 | 8-10 | 15-18 | 22-26 |
| **DeepSeek-V3** | 2x RTX 3090 | 7-9 | 14-17 | 20-24 |
| **Kimi K3** | 4x RTX 3090 | 5-7 | 10-13 | 16-20 |

---

## Troubleshooting

### OOM (Out of Memory) ошибки

**Симптомы:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Решения:**
1. Уменьшите `--vram-limit`:
   ```bash
   wisp run --model ./models/mixtral-8x7b --vram-limit 6
   ```

2. Используйте `--display-mode igpu` если монитор на iGPU:
   ```bash
   wisp run --model ./models/mixtral-8x7b --display-mode igpu
   ```

3. Закройте другие приложения использующие GPU (браузер, игры)

### Медленная загрузка экспертов

**Симптомы:**
```
[WISP] Loading expert from SSD... (2.5s)
```

**Решения:**
1. Проверьте скорость NVMe:
   ```bash
   wisp profile --refresh
   ```

2. Увеличьте размер RAM кэша:
   ```bash
   wisp run --model ./models/mixtral-8x7b --ram-expert-count 2000
   ```

3. Используйте FP8 квантование для уменьшения размера экспертов

### CUDA не найден

**Симптомы:**
```
No CUDA compiler found — falling back to CPU-only build
```

**Решения:**
1. Установите CUDA Toolkit 12.0+:
   ```bash
   # Ubuntu
   sudo apt install nvidia-cuda-toolkit
   
   # Windows
   # Скачайте с https://developer.nvidia.com/cuda-downloads
   ```

2. Проверьте установку:
   ```bash
   nvcc --version
   ```

3. Пересоберите WISP:
   ```bash
   rm -rf build && cmake -B build && cmake --build build
   ```

### Неправильное определение display mode

**Симптомы:**
```
Display: on GPU ⚠️ (1.5 GB reserved)
```
Но монитор подключён к материнской плате.

**Решения:**
1. Принудительно укажите `--display-mode igpu`:
   ```bash
   wisp run --model ./models/mixtral-8x7b --display-mode igpu
   ```

2. Обновите кэш профилирования:
   ```bash
   wisp profile --refresh
   ```

---

## Дополнительные ресурсы

- [GitHub Repository](https://github.com/zeroextub-collab/wisp)
- [Документация по оптимизации для 8GB VRAM](OPTIMIZATION_8GB_CUDA.md)
- [Документация по оптимизации для Intel Xeon](OPTIMIZATION_INTEL_XEON.md)
- [Технический отчёт Kimi K3 (arXiv:2607.24653)](https://arxiv.org/abs/2607.24653)

---

## Поддержка

При возникновении проблем создавайте issue на GitHub с:
1. Выводом `wisp profile --refresh`
2. Выводом `wisp info --model <ваша_модель>`
3. Полным текстом ошибки
4. Конфигурацией системы (CPU, GPU, RAM, NVMe)
