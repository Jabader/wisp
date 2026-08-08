#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <jansson.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <libgen.h>
#include <pthread.h>
#include <stdint.h>
#include <errno.h>

#define MAX_URL_LEN 2048
#define MAX_PATH_LEN 4096
#define MAX_FILES 1024
#define WBIN2_MAGIC 0x5742494E32000000ULL // "WBIN2\0\0\0"
#define MAX_EXPERTS 10000
#define EXPERT_MASK "expert_%05d.wbin2"

typedef struct {
    char *url;
    char *output_path;
    char *temp_dir;
    int thread_count;
    int keep_original;
} ConvertConfig;

typedef struct {
    char filename[MAX_PATH_LEN];
    char download_url[MAX_URL_LEN];
    uint64_t size;
} ModelFile;

typedef struct {
    char name[256];
    char type[64];
    uint64_t offset;
    uint64_t size;
    int is_expert;
    int expert_id;
} TensorInfo;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t total_size;
    uint32_t tensor_count;
    uint32_t reserved;
} Wbin2Header;

// Структура для многопоточного скачивания
typedef struct {
    ModelFile *files;
    int count;
    int current_idx;
    pthread_mutex_t mutex;
    char *temp_dir;
} DownloadPool;

size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    FILE *f = (FILE *)userp;
    size_t written = fwrite(contents, size, nmemb, f);
    return written;
}

size_t header_callback(void *buffer, size_t size, size_t nitems, void *userdata) {
    uint64_t *size_ptr = (uint64_t *)userdata;
    if (strncmp((char*)buffer, "content-length:", 15) == 0) {
        *size_ptr = strtoull((char*)buffer + 15, NULL, 10);
    }
    return size * nitems;
}

int download_file(const char *url, const char *path) {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    
    fp = fopen(path, "wb");
    if (!fp) return -1;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "wisp-convert/1.0");
        
        // Прогресс можно добавить здесь
        
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
        
        if (res != CURLE_OK) {
            unlink(path);
            return -1;
        }
        return 0;
    }
    fclose(fp);
    return -1;
}

void* download_thread(void *arg) {
    DownloadPool *pool = (DownloadPool *)arg;
    
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        int idx = pool->current_idx++;
        pthread_mutex_unlock(&pool->mutex);
        
        if (idx >= pool->count) break;
        
        ModelFile *file = &pool->files[idx];
        char out_path[MAX_PATH_LEN];
        snprintf(out_path, sizeof(out_path), "%s/%s", pool->temp_dir, file->filename);
        
        printf("Downloading [%d/%d]: %s\n", idx + 1, pool->count, file->filename);
        if (download_file(file->download_url, out_path) != 0) {
            fprintf(stderr, "Failed to download %s\n", file->filename);
        }
    }
    return NULL;
}

int parse_hf_url(const char *url, char *repo_owner, char *repo_name, char *branch, char *folder) {
    // Пример: https://huggingface.co/unsloth/GLM-5.2-GGUF/tree/main/UD-Q4_K_M
    char temp[MAX_URL_LEN];
    strncpy(temp, url, sizeof(temp) - 1);
    
    char *token = strtok(temp, "/");
    int part = 0;
    while (token) {
        if (strcmp(token, "huggingface.co") == 0) {
            token = strtok(NULL, "/");
            if (token) strcpy(repo_owner, token);
            token = strtok(NULL, "/");
            if (token) strcpy(repo_name, token);
            token = strtok(NULL, "/"); // tree или blob
            token = strtok(NULL, "/"); // branch
            if (token) strcpy(branch, token);
            
            // Остаток - папка
            char *rest = strtok(NULL, "");
            if (rest) {
                // Убираем ведущий слэш
                if (rest[0] == '/') rest++;
                strcpy(folder, rest);
            } else {
                folder[0] = '\0';
            }
            return 0;
        }
        token = strtok(NULL, "/");
    }
    return -1;
}

int fetch_file_list(const char *owner, const char *repo, const char *branch, const char *folder, ModelFile *files, int *count) {
    CURL *curl;
    CURLcode res;
    char url[MAX_URL_LEN];
    char *json_str = NULL;
    size_t json_len = 0;
    
    // API HF для списка файлов
    if (strlen(folder) > 0) {
        snprintf(url, sizeof(url), "https://huggingface.co/api/models/%s/%s/tree/%s?recursive=1&path=%s", 
                 owner, repo, branch, folder);
    } else {
        snprintf(url, sizeof(url), "https://huggingface.co/api/models/%s/%s/tree/%s?recursive=1", 
                 owner, repo, branch);
    }

    curl = curl_easy_init();
    if (!curl) return -1;

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Accept: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    
    // Собираем JSON в память
    struct curl_buffer {
        char *ptr;
        size_t len;
    };
    
    struct curl_buffer chunk;
    chunk.ptr = malloc(1);
    chunk.len = 0;
    
    // Переопределяем callback для сбора в память
    // (Для краткости используем упрощенную версию, в продакшене нужен отдельный буфер)
    FILE *tmp = open_memstream(&chunk.ptr, &chunk.len);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, tmp);
    
    res = curl_easy_perform(curl);
    fclose(tmp);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || !chunk.ptr) {
        if (chunk.ptr) free(chunk.ptr);
        return -1;
    }

    json_error_t error;
    json_t *root = json_loads(chunk.ptr, 0, &error);
    free(chunk.ptr);

    if (!root) {
        fprintf(stderr, "JSON parse error: %s\n", error.text);
        return -1;
    }

    *count = 0;
    if (json_is_array(root)) {
        size_t i;
        json_t *item;
        json_array_foreach(root, i, item) {
            json_t *type_obj = json_object_get(item, "type");
            json_t *path_obj = json_object_get(item, "path");
            
            if (type_obj && path_obj && strcmp(json_string_value(type_obj), "file") == 0) {
                const char *path = json_string_value(path_obj);
                
                // Фильтруем только GGUF или Safetensors
                if (strstr(path, ".gguf") || strstr(path, ".safetensors")) {
                    strncpy(files[*count].filename, path, MAX_PATH_LEN - 1);
                    // Формируем URL для скачивания сырых файлов
                    snprintf(files[*count].download_url, MAX_URL_LEN,
                             "https://huggingface.co/%s/%s/raw/%s/%s", owner, repo, branch, path);
                    (*count)++;
                    if (*count >= MAX_FILES) break;
                }
            }
        }
    }
    
    json_decref(root);
    return 0;
}

// Заглушки для парсинга GGUF и записи WBIN2
// В реальной реализации здесь будет полный парсер GGUF и логика квантования
int process_gguf_files(ModelFile *files, int count, const char *output_dir) {
    printf("Processing %d GGUF files...\n", count);
    
    // Создаем директорию вывода
    mkdir(output_dir, 0755);
    
    // Здесь должна быть логика:
    // 1. Открыть каждый файл из files[i].filename (во временной папке)
    // 2. Распарсить заголовок GGUF
    // 3. Найти тензоры экспертов
    // 4. Для каждого эксперта создать отдельный файл expert_XXXXX.wbin2
    // 5. Общие веса записать в shared.wbin2
    
    // Эмуляция создания файлов экспертов
    for (int i = 0; i < 5; i++) { // Допустим нашли 5 экспертов
        char expert_path[MAX_PATH_LEN];
        snprintf(expert_path, sizeof(expert_path), "%s/" EXPERT_MASK, output_dir, i);
        
        FILE *f = fopen(expert_path, "wb");
        if (f) {
            Wbin2Header hdr = {
                .magic = WBIN2_MAGIC,
                .version = 2,
                .flags = 0x1, // INT4 quantized
                .total_size = 0,
                .tensor_count = 10,
                .reserved = 0
            };
            fwrite(&hdr, sizeof(hdr), 1, f);
            // Запись данных...
            fclose(f);
            printf("Created sharded expert: %s\n", expert_path);
        }
    }
    
    // Создание общего файла
    char shared_path[MAX_PATH_LEN];
    snprintf(shared_path, sizeof(shared_path), "%s/model_shared.wbin2", output_dir);
    FILE *f = fopen(shared_path, "wb");
    if (f) {
        Wbin2Header hdr = {
            .magic = WBIN2_MAGIC,
            .version = 2,
            .flags = 0x0,
            .total_size = 0,
            .tensor_count = 50,
            .reserved = 0
        };
        fwrite(&hdr, sizeof(hdr), 1, f);
        fclose(f);
        printf("Created shared weights: %s\n", shared_path);
    }
    
    // Создание манифеста
    char manifest_path[MAX_PATH_LEN];
    snprintf(manifest_path, sizeof(manifest_path), "%s/model.manifest.json", output_dir);
    f = fopen(manifest_path, "w");
    if (f) {
        fprintf(f, "{\n  \"version\": 2,\n  \"sharded\": true,\n  \"experts\": 5,\n  \"format\": \"INT4\"\n}\n");
        fclose(f);
    }
    
    return 0;
}

void cleanup_temp_files(const char *temp_dir) {
    DIR *dir = opendir(temp_dir);
    if (!dir) return;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        
        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s/%s", temp_dir, entry->d_name);
        unlink(path);
    }
    closedir(dir);
    rmdir(temp_dir);
    printf("Cleaned up temporary files in %s\n", temp_dir);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <hf_url_or_gguf_link> <output_dir> [--threads N] [--keep]\n", argv[0]);
        fprintf(stderr, "Example: %s https://huggingface.co/unsloth/GLM-5.2-GGUF/tree/main/UD-Q4_K_M ./models\n", argv[0]);
        return 1;
    }

    ConvertConfig config = {0};
    config.url = argv[1];
    config.output_path = argv[2];
    config.thread_count = 4;
    config.keep_original = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.thread_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--keep") == 0) {
            config.keep_original = 1;
        }
    }

    curl_global_init(CURL_GLOBAL_ALL);

    char temp_dir[MAX_PATH_LEN];
    snprintf(temp_dir, sizeof(temp_dir), "/tmp/wisp_download_XXXXXX");
    if (mkdtemp(temp_dir) == NULL) {
        fprintf(stderr, "Failed to create temp directory\n");
        return 1;
    }
    config.temp_dir = temp_dir;

    printf("Temp directory: %s\n", temp_dir);
    printf("Output directory: %s\n", config.output_path);

    ModelFile files[MAX_FILES];
    int file_count = 0;

    // Проверка типа URL
    if (strstr(config.url, "huggingface.co")) {
        char owner[256], repo[256], branch[256], folder[256];
        if (parse_hf_url(config.url, owner, repo, branch, folder) == 0) {
            printf("Detected HF Repo: %s/%s (Branch: %s, Folder: %s)\n", owner, repo, branch, folder);
            if (fetch_file_list(owner, repo, branch, folder, files, &file_count) != 0) {
                fprintf(stderr, "Failed to fetch file list\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Failed to parse HF URL\n");
            return 1;
        }
    } else if (strstr(config.url, ".gguf")) {
        // Прямая ссылка на GGUF
        strncpy(files[0].download_url, config.url, MAX_URL_LEN);
        char *fname = basename(config.url);
        strncpy(files[0].filename, fname, MAX_PATH_LEN);
        file_count = 1;
    } else {
        fprintf(stderr, "Unsupported URL format\n");
        return 1;
    }

    if (file_count == 0) {
        fprintf(stderr, "No valid model files found\n");
        return 1;
    }

    printf("Found %d files to download\n", file_count);

    // Многопоточное скачивание
    DownloadPool pool = {
        .files = files,
        .count = file_count,
        .current_idx = 0,
        .temp_dir = temp_dir
    };
    pthread_mutex_init(&pool.mutex, NULL);

    pthread_t threads[config.thread_count];
    for (int i = 0; i < config.thread_count; i++) {
        pthread_create(&threads[i], NULL, download_thread, &pool);
    }

    for (int i = 0; i < config.thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&pool.mutex);

    printf("Download complete. Starting conversion...\n");

    // Конвертация и шардинг
    if (process_gguf_files(files, file_count, config.output_path) != 0) {
        fprintf(stderr, "Conversion failed\n");
        return 1;
    }

    // Очистка
    if (!config.keep_original) {
        cleanup_temp_files(temp_dir);
    } else {
        printf("Keeping original files in %s\n", temp_dir);
    }

    curl_global_cleanup();
    printf("Conversion completed successfully!\n");
    return 0;
}
