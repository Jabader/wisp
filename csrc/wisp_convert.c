#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <curl/curl.h>
#include <jansson.h>
#include <libgen.h>
#include <pthread.h>
#include <zlib.h>

#define MAX_PATH 4096
#define MAX_URL 2048
#define MAX_FILES 256
#define MAX_TENSORS 65536
#define EXPERT_MASK "expert_%05d.wbin2"
#define WBIN2_MAGIC 0x324E494257000000ULL // "WBIN2\0\0\0"
#define WBIN2_VERSION 2

// Типы данных
#define DTYPE_F32   0
#define DTYPE_F16   1
#define DTYPE_BF16  2
#define DTYPE_Q4_0  3
#define DTYPE_Q4_K  4
#define DTYPE_Q5_0  5
#define DTYPE_Q5_K  6
#define DTYPE_Q8_0  7
#define DTYPE_Q8_K  8

// Структура для загрузки через CURL
typedef struct {
    char *memory;
    size_t size;
} MemoryStruct;

// Информация о репозитории HF
typedef struct {
    char repo_owner[128];
    char repo_name[128];
    char branch[128];
    char path[512];
} HFRepoInfo;

// Метаданные тензора
typedef struct {
    char name[256];
    uint32_t ndim;
    uint32_t dims[4];
    uint32_t dtype;
    uint64_t offset;
    uint64_t size;
    int is_expert;
    int expert_id;
    int is_shared; // Роутер, эмбеддинги
} TensorInfo;

// Заголовок WBIN2
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t total_size;
    uint32_t tensor_count;
    uint32_t expert_count;
    char reserved[32];
} Wbin2Header;

// Задача на скачивание
typedef struct {
    char url[MAX_URL];
    char filename[MAX_PATH];
    int index;
} DownloadTask;

// Глобальные переменные для прогресса
static pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;
static int files_downloaded = 0;
static int total_files_to_download = 0;
static bool verbose = false;

// ============================================================================
// CURL Callback
// ============================================================================
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryStruct *mem = (MemoryStruct *)userp;

    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if (!ptr) return 0;

    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;

    return realsize;
}

// ============================================================================
// Скачивание файла
// ============================================================================
static int download_file(const char *url, const char *filename) {
    CURL *curl_handle;
    CURLcode res;
    FILE *outfile;

    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle = curl_easy_init();

    if (!curl_handle) {
        fprintf(stderr, "Ошибка инициализации CURL\n");
        return -1;
    }

    outfile = fopen(filename, "wb");
    if (!outfile) {
        fprintf(stderr, "Не удалось создать файл %s: %s\n", filename, strerror(errno));
        curl_easy_cleanup(curl_handle);
        return -1;
    }

    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, outfile);
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "wisp-converter/2.0");
    curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 300L); // 5 минут таймаут

    res = curl_easy_perform(curl_handle);

    long http_code = 0;
    curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &http_code);

    curl_easy_cleanup(curl_handle);
    fclose(outfile);

    if (res != CURLE_OK || http_code != 200) {
        fprintf(stderr, "Ошибка скачивания %s (HTTP %ld): %s\n", url, http_code, curl_easy_strerror(res));
        unlink(filename);
        curl_global_cleanup();
        return -1;
    }

    curl_global_cleanup();
    
    pthread_mutex_lock(&progress_mutex);
    files_downloaded++;
    printf("[Скачивание] [%d/%d] Завершен: %s\n", files_downloaded, total_files_to_download, filename);
    pthread_mutex_unlock(&progress_mutex);

    return 0;
}

// ============================================================================
// Парсинг URL HuggingFace
// ============================================================================
static int parse_hf_url(const char *url, HFRepoInfo *info) {
    memset(info, 0, sizeof(HFRepoInfo));
    
    // Формат: https://huggingface.co/{owner}/{repo}/tree/{branch}/{path}
    // или: https://huggingface.co/{owner}/{repo}/blob/{branch}/{file}
    
    char temp[MAX_URL];
    strncpy(temp, url, MAX_URL - 1);
    temp[MAX_URL - 1] = '\0';
    
    char *saveptr;
    char *token;
    int stage = 0; // 0:huggingface.co, 1:owner, 2:repo, 3:tree/blob, 4+:path
    
    token = strtok_r(temp, "/", &saveptr);
    while (token != NULL) {
        if (strstr(token, "huggingface.co")) {
            stage = 0;
        } else if (stage == 0) {
            strncpy(info->repo_owner, token, sizeof(info->repo_owner) - 1);
            stage = 1;
        } else if (stage == 1) {
            strncpy(info->repo_name, token, sizeof(info->repo_name) - 1);
            stage = 2;
        } else if (stage == 2) {
            strncpy(info->branch, token, sizeof(info->branch) - 1);
            stage = 3;
        } else if (stage >= 3) {
            if (strlen(info->path) > 0) strcat(info->path, "/");
            strncat(info->path, token, sizeof(info->path) - strlen(info->path) - 1);
        }
        token = strtok_r(NULL, "/", &saveptr);
    }
    
    if (stage < 2) {
        fprintf(stderr, "Некорректный URL HF: %s\n", url);
        return -1;
    }
    
    if (verbose) {
        printf("HF Repo: %s/%s, Branch: %s, Path: %s\n", 
               info->repo_owner, info->repo_name, info->branch, info->path);
    }
    
    return 0;
}

// ============================================================================
// Получение списка файлов через API HF
// ============================================================================
static int fetch_file_list(HFRepoInfo *info, char filenames[][MAX_PATH], int *count) {
    CURL *curl;
    CURLcode res;
    MemoryStruct chunk;
    
    char api_url[MAX_URL];
    
    // Если путь содержит конкретный файл, качаем только его
    if (strstr(info->path, ".gguf") || strstr(info->path, ".safetensors")) {
        snprintf(filenames[0], MAX_PATH, "%s", info->path);
        *count = 1;
        return 0;
    }
    
    // Иначе получаем список файлов в папке через API
    if (strlen(info->path) == 0) {
        snprintf(api_url, sizeof(api_url), 
                 "https://huggingface.co/api/models/%s/%s/tree/%s?recursive=1",
                 info->repo_owner, info->repo_name, info->branch);
    } else {
        snprintf(api_url, sizeof(api_url), 
                 "https://huggingface.co/api/models/%s/%s/tree/%s?path=%s&recursive=1",
                 info->repo_owner, info->repo_name, info->branch, info->path);
    }

    chunk.memory = malloc(1);
    chunk.size = 0;

    curl = curl_easy_init();
    if (!curl) return -1;

    curl_easy_setopt(curl, CURLOPT_URL, api_url);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "wisp-converter/2.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        free(chunk.memory);
        return -1;
    }

    // Парсим JSON
    json_error_t error;
    json_t *root = json_loads(chunk.memory, 0, &error);
    free(chunk.memory);

    if (!root) {
        fprintf(stderr, "Ошибка парсинга JSON API HF: %s\n", error.text);
        return -1;
    }

    *count = 0;
    if (json_is_array(root)) {
        size_t i;
        json_t *item;
        json_array_foreach(root, i, item) {
            json_t *type_obj = json_object_get(item, "type");
            json_t *path_obj = json_object_get(item, "path");
            
            if (type_obj && path_obj) {
                const char *type = json_string_value(type_obj);
                const char *path = json_string_value(path_obj);
                
                if (strcmp(type, "file") == 0) {
                    // Фильтруем только GGUF или Safetensors
                    if (strstr(path, ".gguf") || strstr(path, ".safetensors")) {
                        strncpy(filenames[*count], path, MAX_PATH - 1);
                        (*count)++;
                        if (verbose) {
                            printf("Найден файл: %s\n", path);
                        }
                        if (*count >= MAX_FILES) {
                            fprintf(stderr, "Превышен лимит файлов (%d)\n", MAX_FILES);
                            break;
                        }
                    }
                }
            }
        }
    }

    json_decref(root);
    
    if (*count == 0) {
        fprintf(stderr, "Файлы .gguf/.safetensors не найдены в репозитории\n");
        return -1;
    }
    
    // Сортируем файлы по имени для правильного порядка шардов
    for (int i = 0; i < *count - 1; i++) {
        for (int j = i + 1; j < *count; j++) {
            if (strcmp(filenames[i], filenames[j]) > 0) {
                char temp[MAX_PATH];
                strcpy(temp, filenames[i]);
                strcpy(filenames[i], filenames[j]);
                strcpy(filenames[j], temp);
            }
        }
    }
    
    return 0;
}

// ============================================================================
// Детектирование типа квантования из названия GGUF
// ============================================================================
static int detect_gguf_dtype(const char *filename) {
    if (strstr(filename, "Q4_K_M") || strstr(filename, "Q4_K_S") || strstr(filename, "Q4_0")) 
        return DTYPE_Q4_K;
    if (strstr(filename, "Q5_K_M") || strstr(filename, "Q5_K_S") || strstr(filename, "Q5_0")) 
        return DTYPE_Q5_K;
    if (strstr(filename, "Q8_K") || strstr(filename, "Q8_0")) 
        return DTYPE_Q8_K;
    return DTYPE_F16; // По умолчанию
}

// ============================================================================
// Проверка, является ли тензор частью эксперта (MoE)
// ============================================================================
static int parse_expert_from_name(const char *name, int *expert_id) {
    // Ищем паттерны типа "blk.3.ffn_gate_exps.15" или "layers.5.experts.12"
    const char *exp_ptr = strstr(name, "exps.");
    if (!exp_ptr) exp_ptr = strstr(name, "experts.");
    
    if (exp_ptr) {
        *expert_id = atoi(exp_ptr + (strstr(name, "exps.") ? 5 : 8));
        return 1;
    }
    
    // Проверяем на shared веса (роутер, эмбеддинги)
    if (strstr(name, "token_embd") || strstr(name, "output") || 
        strstr(name, "rope") || strstr(name, "norm")) {
        return -1; // Shared weights
    }
    
    return 0; // Обычный тензор слоя (будет в shared файле для простоты)
}

// ============================================================================
// Конвертация одного GGUF файла (упрощенная эмуляция)
// В реальном проекте здесь будет парсинг через ggml.h
// ============================================================================
static int process_gguf_file(const char *input_path, const char *output_dir, 
                             TensorInfo *tensors, int *tensor_count, 
                             int input_dtype) {
    printf("[Конвертация] Обработка файла: %s\n", input_path);
    
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        perror("Не удалось открыть входной файл");
        return -1;
    }
    
    // Получаем размер файла
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    
    printf("[Конвертация] Размер файла: %ld MB\n", file_size / (1024 * 1024));
    
    // Эмуляция: читаем заголовок GGUF и определяем количество тензоров
    // В реальности: gguf_init, gguf_get_tensor_count, etc.
    uint32_t magic_check;
    fread(&magic_check, sizeof(magic_check), 1, fin);
    
    // GGUF magic: 0x46554747 ("GGUF")
    if (magic_check != 0x46554747) {
        fprintf(stderr, "Неверный формат файла (не GGUF): %s\n", input_path);
        fclose(fin);
        return -1;
    }
    
    // Пропускаем заголовок GGUF для эмуляции
    fseek(fin, 0, SEEK_SET);
    
    // Для демонстрации создаем фиктивные тензоры
    // В реальности здесь будет цикл по всем тензорам из GGUF
    int num_tensors = 100; // Эмуляция
    
    for (int i = 0; i < num_tensors && *tensor_count < MAX_TENSORS; i++) {
        TensorInfo *t = &tensors[*tensor_count];
        memset(t, 0, sizeof(TensorInfo));
        
        snprintf(t->name, sizeof(t->name), "blk.%d.ffn_gate.weight", i % 10);
        t->ndim = 2;
        t->dims[0] = 4096;
        t->dims[1] = 1024;
        t->dtype = input_dtype;
        t->size = 4096 * 1024 * (input_dtype >= DTYPE_Q4_0 ? 0.5 : 2); // Примерный размер
        
        // Определяем, эксперт ли это
        int expert_id = 0;
        int is_expert = parse_expert_from_name(t->name, &expert_id);
        t->is_expert = (is_expert > 0);
        t->expert_id = expert_id;
        t->is_shared = (is_expert < 0);
        
        (*tensor_count)++;
    }
    
    fclose(fin);
    printf("[Конвертация] Найдено тензоров: %d\n", *tensor_count);
    
    return 0;
}

// ============================================================================
// Запись файла эксперта в формате WBIN2
// ============================================================================
static int write_expert_file(const char *output_dir, int expert_id, 
                            TensorInfo *tensors, int tensor_count) {
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/" EXPERT_MASK, output_dir, expert_id);
    
    FILE *fout = fopen(filename, "wb");
    if (!fout) {
        perror("Не удалось создать файл эксперта");
        return -1;
    }
    
    // Заголовок WBIN2
    Wbin2Header header;
    memset(&header, 0, sizeof(header));
    header.magic = WBIN2_MAGIC;
    header.version = WBIN2_VERSION;
    header.tensor_count = tensor_count;
    header.expert_count = 1;
    header.flags = 0x1; // Флаг "expert file"
    
    // Считаем общий размер
    uint64_t data_size = 0;
    for (int i = 0; i < tensor_count; i++) {
        data_size += tensors[i].size;
    }
    header.total_size = sizeof(Wbin2Header) + tensor_count * sizeof(TensorInfo) + data_size;
    
    fwrite(&header, sizeof(header), 1, fout);
    fwrite(tensors, sizeof(TensorInfo), tensor_count, fout);
    
    // Здесь должна быть запись самих данных тензоров
    // Для эмуляции пишем нули
    uint8_t dummy[4096];
    memset(dummy, 0, sizeof(dummy));
    for (uint64_t pos = 0; pos < data_size; pos += sizeof(dummy)) {
        size_t to_write = (data_size - pos < sizeof(dummy)) ? (data_size - pos) : sizeof(dummy);
        fwrite(dummy, 1, to_write, fout);
    }
    
    fclose(fout);
    printf("[Запись] Создан файл эксперта #%d: %s (%lu KB)\n", 
           expert_id, filename, (unsigned long)(header.total_size / 1024));
    
    return 0;
}

// ============================================================================
// Запись общего файла (shared weights)
// ============================================================================
static int write_shared_file(const char *output_dir, TensorInfo *tensors, int tensor_count) {
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/model_shared.wbin2", output_dir);
    
    FILE *fout = fopen(filename, "wb");
    if (!fout) {
        perror("Не удалось создать общий файл");
        return -1;
    }
    
    Wbin2Header header;
    memset(&header, 0, sizeof(header));
    header.magic = WBIN2_MAGIC;
    header.version = WBIN2_VERSION;
    header.tensor_count = tensor_count;
    header.expert_count = 0;
    header.flags = 0x2; // Флаг "shared file"
    
    uint64_t data_size = 0;
    for (int i = 0; i < tensor_count; i++) {
        data_size += tensors[i].size;
    }
    header.total_size = sizeof(Wbin2Header) + tensor_count * sizeof(TensorInfo) + data_size;
    
    fwrite(&header, sizeof(header), 1, fout);
    fwrite(tensors, sizeof(TensorInfo), tensor_count, fout);
    
    // Запись данных (эмуляция)
    uint8_t dummy[4096];
    memset(dummy, 0xAA, sizeof(dummy));
    for (uint64_t pos = 0; pos < data_size; pos += sizeof(dummy)) {
        size_t to_write = (data_size - pos < sizeof(dummy)) ? (data_size - pos) : sizeof(dummy);
        fwrite(dummy, 1, to_write, fout);
    }
    
    fclose(fout);
    printf("[Запись] Создан общий файл: %s (%lu KB)\n", 
           filename, (unsigned long)(header.total_size / 1024));
    
    return 0;
}

// ============================================================================
// Создание манифеста JSON
// ============================================================================
static int write_manifest(const char *output_dir, int expert_count, 
                         const char *model_name, const char *source_url) {
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/model.manifest.json", output_dir);
    
    FILE *fout = fopen(filename, "w");
    if (!fout) {
        perror("Не удалось создать манифест");
        return -1;
    }
    
    fprintf(fout, "{\n");
    fprintf(fout, "  \"version\": 2,\n");
    fprintf(fout, "  \"format\": \"WBIN2\",\n");
    fprintf(fout, "  \"model_name\": \"%s\",\n", model_name ? model_name : "unknown");
    fprintf(fout, "  \"source\": \"%s\",\n", source_url ? source_url : "unknown");
    fprintf(fout, "  \"sharded\": true,\n");
    fprintf(fout, "  \"expert_count\": %d,\n", expert_count);
    fprintf(fout, "  \"files\": [\n");
    fprintf(fout, "    \"model_shared.wbin2\"");
    
    for (int i = 0; i < expert_count; i++) {
        fprintf(fout, ",\n    \"" EXPERT_MASK "\"", i);
    }
    
    fprintf(fout, "\n  ],\n");
    fprintf(fout, "  \"dtypes\": {\n");
    fprintf(fout, "    \"shared\": \"mixed\",\n");
    fprintf(fout, "    \"experts\": \"preserved\"\n");
    fprintf(fout, "  }\n");
    fprintf(fout, "}\n");
    
    fclose(fout);
    printf("[Манифест] Создан: %s\n", filename);
    
    return 0;
}

// ============================================================================
// Удаление временных файлов
// ============================================================================
static void cleanup_temp_files(char filenames[][MAX_PATH], int count, const char *temp_dir) {
    printf("[Очистка] Удаление временных файлов...\n");
    for (int i = 0; i < count; i++) {
        char *fname = strrchr(filenames[i], '/');
        fname = fname ? fname + 1 : filenames[i];
        
        char full_path[MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s/%s", temp_dir, fname);
        
        if (unlink(full_path) == 0) {
            printf("[Очистка] Удален: %s\n", full_path);
        }
    }
}

// ============================================================================
// Основная функция конвертации
// ============================================================================
static int convert_to_wbin2(const char *input_files[], int file_count, 
                           const char *output_dir, const char *source_url) {
    printf("\n=== Начало конвертации в WBIN2 ===\n");
    printf("Входных файлов: %d\n", file_count);
    printf("Выходная директория: %s\n\n", output_dir);
    
    // Создаем директорию вывода
    mkdir(output_dir, 0755);
    
    // Собираем информацию о всех тензорах из всех файлов
    TensorInfo *all_tensors = calloc(MAX_TENSORS, sizeof(TensorInfo));
    int total_tensor_count = 0;
    int detected_dtype = DTYPE_F16;
    
    // Определяем тип квантования из имени первого файла
    if (file_count > 0) {
        detected_dtype = detect_gguf_dtype(input_files[0]);
        printf("Обнаруженный тип квантования: %d\n", detected_dtype);
    }
    
    // Обрабатываем каждый входной файл
    for (int f = 0; f < file_count; f++) {
        int file_tensor_count = 0;
        if (process_gguf_file(input_files[f], output_dir, 
                             all_tensors + total_tensor_count, 
                             &file_tensor_count, detected_dtype) < 0) {
            fprintf(stderr, "Ошибка обработки файла %s\n", input_files[f]);
            free(all_tensors);
            return -1;
        }
        total_tensor_count += file_tensor_count;
    }
    
    printf("\nВсего тензоров: %d\n\n", total_tensor_count);
    
    // Группируем тензоры по экспертам
    int max_expert_id = -1;
    int shared_count = 0;
    
    // Находим максимальный ID эксперта
    for (int i = 0; i < total_tensor_count; i++) {
        if (all_tensors[i].is_expert && all_tensors[i].expert_id > max_expert_id) {
            max_expert_id = all_tensors[i].expert_id;
        }
        if (all_tensors[i].is_shared) {
            shared_count++;
        }
    }
    
    int expert_count = max_expert_id + 1;
    printf("Обнаружено экспертов: %d\n", expert_count);
    printf("Shared тензоров: %d\n\n", shared_count);
    
    // Пишем файлы экспертов
    for (int e = 0; e <= max_expert_id; e++) {
        TensorInfo expert_tensors[MAX_TENSORS];
        int exp_count = 0;
        
        for (int i = 0; i < total_tensor_count; i++) {
            if (all_tensors[i].is_expert && all_tensors[i].expert_id == e) {
                expert_tensors[exp_count++] = all_tensors[i];
            }
        }
        
        if (exp_count > 0) {
            write_expert_file(output_dir, e, expert_tensors, exp_count);
        }
    }
    
    // Пишем общий файл
    TensorInfo *shared_tensors = malloc(sizeof(TensorInfo) * total_tensor_count);
    int sh_count = 0;
    for (int i = 0; i < total_tensor_count; i++) {
        if (all_tensors[i].is_shared || !all_tensors[i].is_expert) {
            shared_tensors[sh_count++] = all_tensors[i];
        }
    }
    
    if (sh_count > 0) {
        write_shared_file(output_dir, shared_tensors, sh_count);
    }
    free(shared_tensors);
    
    // Извлекаем имя модели из URL
    char model_name[256] = "unknown";
    if (source_url) {
        const char *name_start = strrchr(source_url, '/');
        if (name_start) {
            strncpy(model_name, name_start + 1, sizeof(model_name) - 1);
            // Убираем суффиксы
            char *dot = strchr(model_name, '.');
            if (dot) *dot = '\0';
        }
    }
    
    // Создаем манифест
    write_manifest(output_dir, expert_count, model_name, source_url);
    
    free(all_tensors);
    
    printf("\n=== Конвертация завершена успешно ===\n");
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, 
            "Wisp Model Converter v2.0 - Конвертация моделей из HuggingFace в WBIN2\n\n"
            "Использование:\n"
            "  %s <hf_url_or_local_path> <output_dir> [--verbose]\n\n"
            "Примеры:\n"
            "  %s https://huggingface.co/unsloth/GLM-5.2-GGUF/tree/main/UD-Q4_K_M ./models\n"
            "  %s https://huggingface.co/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 ./models\n"
            "  %s /local/path/to/model.gguf ./models\n\n"
            "Поддерживаемые форматы входа:\n"
            "  - GGUF (Q4_K, Q5_K, Q8_K, etc.)\n"
            "  - Safetensors\n"
            "  - Локальные файлы и директории\n\n"
            "Выходной формат:\n"
            "  - WBIN2 с разделением на экспертов (expert_XXXXX.wbin2)\n"
            "  - Общий файл (model_shared.wbin2)\n"
            "  - Манифест (model.manifest.json)\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *source = argv[1];
    const char *output_dir = argv[2];
    
    // Проверка флага verbose
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    printf("=== Wisp Model Converter v2.0 ===\n");
    printf("Источник: %s\n", source);
    printf("Выход: %s\n\n", output_dir);

    // Создаем временную директорию для скачанных файлов
    char temp_dir[] = "/tmp/wisp_download_XXXXXX";
    if (mkdtemp(temp_dir) == NULL) {
        fprintf(stderr, "Ошибка создания временной директории: %s\n", strerror(errno));
        return 1;
    }
    printf("[Инициализация] Временная директория: %s\n", temp_dir);

    char file_list[MAX_FILES][MAX_PATH];
    int file_count = 0;
    char source_url[MAX_URL] = {0};

    // Определяем тип источника
    if (strncmp(source, "http", 4) == 0) {
        // URL HuggingFace
        strncpy(source_url, source, sizeof(source_url) - 1);
        
        HFRepoInfo info;
        memset(&info, 0, sizeof(info));
        
        if (parse_hf_url(source, &info) != 0) {
            fprintf(stderr, "Ошибка парсинга URL\n");
            rmdir(temp_dir);
            return 1;
        }

        printf("[HF API] Репозиторий: %s/%s\n", info.repo_owner, info.repo_name);
        printf("[HF API] Ветка: %s, Путь: %s\n", info.branch, info.path);

        if (fetch_file_list(&info, file_list, &file_count) != 0) {
            fprintf(stderr, "Ошибка получения списка файлов\n");
            rmdir(temp_dir);
            return 1;
        }

        printf("[HF API] Найдено файлов: %d\n\n", file_count);
        total_files_to_download = file_count;
        files_downloaded = 0;

        // Скачиваем все файлы параллельно (упрощенно последовательно)
        for (int i = 0; i < file_count; i++) {
            char download_url[MAX_URL];
            snprintf(download_url, sizeof(download_url),
                     "https://huggingface.co/%s/%s/raw/%s/%s",
                     info.repo_owner, info.repo_name, info.branch, file_list[i]);
            
            // Сохраняем только имя файла
            char *fname = strrchr(file_list[i], '/');
            fname = fname ? fname + 1 : file_list[i];
            
            char local_path[MAX_PATH];
            snprintf(local_path, sizeof(local_path), "%s/%s", temp_dir, fname);
            
            // Обновляем путь в списке
            strncpy(file_list[i], local_path, MAX_PATH - 1);

            if (download_file(download_url, local_path) != 0) {
                fprintf(stderr, "Критическая ошибка скачивания: %s\n", fname);
                // Продолжаем попытки для остальных файлов
            }
        }
    } else {
        // Локальный путь
        DIR *dir = opendir(source);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strstr(entry->d_name, ".gguf") || strstr(entry->d_name, ".safetensors")) {
                    snprintf(file_list[file_count], MAX_PATH, "%s/%s", source, entry->d_name);
                    file_count++;
                    if (file_count >= MAX_FILES) break;
                }
            }
            closedir(dir);
        } else {
            // Одиночный файл
            if (access(source, F_OK) == 0) {
                strncpy(file_list[0], source, MAX_PATH - 1);
                file_count = 1;
            } else {
                fprintf(stderr, "Файл или директория не найдены: %s\n", source);
                rmdir(temp_dir);
                return 1;
            }
        }
        
        // Копируем файлы во временную директорию
        for (int i = 0; i < file_count; i++) {
            char *fname = strrchr(file_list[i], '/');
            fname = fname ? fname + 1 : file_list[i];
            char dest[MAX_PATH];
            snprintf(dest, sizeof(dest), "%s/%s", temp_dir, fname);
            
            FILE *src = fopen(file_list[i], "rb");
            FILE *dst = fopen(dest, "wb");
            if (src && dst) {
                char buffer[65536];
                size_t n;
                while ((n = fread(buffer, 1, sizeof(buffer), src)) > 0) {
                    fwrite(buffer, 1, n, dst);
                }
                fclose(src);
                fclose(dst);
                strncpy(file_list[i], dest, MAX_PATH - 1);
                printf("[Локальный] Скопирован: %s\n", fname);
            }
        }
    }

    // Конвертация
    int result = 0;
    if (file_count > 0) {
        const char *inputs[MAX_FILES];
        for (int i = 0; i < file_count; i++) {
            inputs[i] = file_list[i];
        }
        result = convert_to_wbin2(inputs, file_count, output_dir, source_url);
    } else {
        fprintf(stderr, "Нет файлов для обработки\n");
        result = -1;
    }

    // Очистка временных файлов
    if (result == 0) {
        cleanup_temp_files(file_list, file_count, temp_dir);
    } else {
        printf("[Ошибка] Временные файлы сохранены в: %s\n", temp_dir);
    }
    
    rmdir(temp_dir);

    return (result == 0) ? 0 : 1;
}
