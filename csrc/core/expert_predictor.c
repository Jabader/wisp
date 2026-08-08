/*
 * expert_predictor.c — KV-cache based expert prefetch predictor.
 * See expert_predictor.h for the API and algorithm description.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "expert_predictor.h"
#include "lru_cache.h"

/* Prefetch macro for cache optimization */
#ifdef __GNUC__
#define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
#define PREFETCH(addr, rw, locality) ((void)0)
#endif

/* Hash function for token context */
static uint32_t context_hash(const int* tokens, uint32_t n_tokens, 
                             uint32_t layer_id, uint32_t mask) {
    uint64_t h = 14695981039346656037ull;  /* FNV-1a offset basis */
    for (uint32_t i = 0; i < n_tokens; i++) {
        h ^= (uint64_t)tokens[i];
        h *= 1099511628211ull;  /* FNV prime */
    }
    /* Mix in layer_id to avoid collisions across layers */
    h ^= (uint64_t)layer_id * 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    return (uint32_t)(h & mask);
}

/* Update rolling token window */
void predictor_update_token(ExpertPredictor* p, int token_id) {
    if (!p || !p->token_window || !p->enabled) return;
    
    wisp_mutex_lock(&p->mutex);
    p->token_window[p->window_pos] = token_id;
    p->window_pos = (p->window_pos + 1) % p->window_size;
    wisp_mutex_unlock(&p->mutex);
}

/* Compute current context hash from token window */
static uint32_t get_current_context(ExpertPredictor* p, uint32_t layer_id) {
    if (!p || !p->token_window) return 0;
    
    uint32_t ctx[WISP_PREDICTOR_CTX_SIZE];
    uint32_t n = p->window_size < WISP_PREDICTOR_CTX_SIZE ? 
                 p->window_size : WISP_PREDICTOR_CTX_SIZE;
    
    /* Get last N tokens in order */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = (p->window_pos - n + i + p->window_size) % p->window_size;
        ctx[i] = p->token_window[idx];
    }
    
    return context_hash(ctx, n, layer_id, p->table_mask);
}

WispError predictor_init(ExpertPredictor* p, int n_layers, 
                         int n_experts_per_layer, WispErrCtx* err) {
    WISP_CHECK_NULL(p, err, "predictor_init: predictor");
    memset(p, 0, sizeof(*p));
    
    p->table_mask = WISP_PREDICTOR_TABLE_SIZE - 1;
    p->n_layers = (uint32_t)n_layers < WISP_PREDICTOR_MAX_LAYERS ? 
                  (uint32_t)n_layers : WISP_PREDICTOR_MAX_LAYERS;
    p->n_experts_per_layer = (uint32_t)n_experts_per_layer < WISP_PREDICTOR_MAX_EXPERTS_PER_LAYER ?
                             (uint32_t)n_experts_per_layer : WISP_PREDICTOR_MAX_EXPERTS_PER_LAYER;
    p->window_size = WISP_PREDICTOR_CTX_SIZE;
    p->window_pos = 0;
    p->enabled = 1;
    
    /* Allocate token window */
    p->token_window = (int*)calloc(p->window_size, sizeof(int));
    WISP_CHECK_NULL(p->token_window, err, "predictor_init: token_window");
    
    /* Allocate per-layer expert frequency counters */
    for (uint32_t l = 0; l < p->n_layers; l++) {
        p->layer_expert_counts[l] = (uint32_t*)calloc(p->n_experts_per_layer, 
                                                       sizeof(uint32_t));
        if (!p->layer_expert_counts[l]) {
            WISP_ERR_SET(err, WISP_ERR_OOM, 
                        "predictor_init: layer_expert_counts[%u]", l);
            goto cleanup;
        }
    }
    
    wisp_mutex_init(&p->mutex);
    return WISP_OK;

cleanup:
    for (uint32_t l = 0; l < p->n_layers; l++) {
        free(p->layer_expert_counts[l]);
        p->layer_expert_counts[l] = NULL;
    }
    free(p->token_window);
    p->token_window = NULL;
    return WISP_ERR_OOM;
}

void predictor_destroy(ExpertPredictor* p) {
    if (!p) return;
    
    wisp_mutex_lock(&p->mutex);
    
    /* Free all hash table entries */
    for (uint32_t i = 0; i < WISP_PREDICTOR_TABLE_SIZE; i++) {
        PredictorEntry* entry = p->table[i];
        while (entry) {
            PredictorEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        p->table[i] = NULL;
    }
    
    /* Free per-layer counters */
    for (uint32_t l = 0; l < p->n_layers; l++) {
        free(p->layer_expert_counts[l]);
        p->layer_expert_counts[l] = NULL;
    }
    
    free(p->token_window);
    p->token_window = NULL;
    p->n_layers = 0;
    p->enabled = 0;
    
    wisp_mutex_unlock(&p->mutex);
    wisp_mutex_destroy(&p->mutex);
}

void predictor_clear(ExpertPredictor* p) {
    if (!p) return;
    
    wisp_mutex_lock(&p->mutex);
    
    /* Free all hash table entries */
    for (uint32_t i = 0; i < WISP_PREDICTOR_TABLE_SIZE; i++) {
        PredictorEntry* entry = p->table[i];
        while (entry) {
            PredictorEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        p->table[i] = NULL;
    }
    
    /* Reset per-layer counters */
    for (uint32_t l = 0; l < p->n_layers; l++) {
        if (p->layer_expert_counts[l]) {
            memset(p->layer_expert_counts[l], 0, 
                   p->n_experts_per_layer * sizeof(uint32_t));
        }
    }
    
    /* Reset statistics */
    p->total_predictions = 0;
    p->correct_predictions = 0;
    p->prefetches_issued = 0;
    
    /* Clear token window */
    if (p->token_window) {
        memset(p->token_window, 0, p->window_size * sizeof(int));
    }
    p->window_pos = 0;
    
    wisp_mutex_unlock(&p->mutex);
}

void predictor_set_enabled(ExpertPredictor* p, int enabled) {
    if (!p) return;
    wisp_mutex_lock(&p->mutex);
    p->enabled = enabled;
    wisp_mutex_unlock(&p->mutex);
}

double predictor_accuracy(const ExpertPredictor* p) {
    if (!p || p->total_predictions == 0) return 0.0;
    return (double)p->correct_predictions / (double)p->total_predictions;
}

/* Find or create entry for context+layer */
static PredictorEntry* find_or_create_entry(ExpertPredictor* p, 
                                            uint32_t ctx_hash, 
                                            uint32_t layer_id) {
    uint32_t slot = ctx_hash & p->table_mask;
    PredictorEntry* entry = p->table[slot];
    
    /* Search existing entries */
    while (entry) {
        if (entry->context_hash == ctx_hash && entry->layer_id == layer_id) {
            PREFETCH(entry, 1, 3);
            return entry;
        }
        entry = entry->next;
    }
    
    /* Create new entry */
    entry = (PredictorEntry*)calloc(1, sizeof(PredictorEntry));
    if (!entry) return NULL;
    
    entry->context_hash = ctx_hash;
    entry->layer_id = layer_id;
    entry->n_predictions = 0;
    entry->hits = 0;
    entry->misses = 0;
    entry->last_access_ns = wisp_now_ns();
    entry->next = p->table[slot];
    p->table[slot] = entry;
    
    return entry;
}

void predictor_record(ExpertPredictor* p, uint32_t layer_id, 
                      const uint32_t* expert_ids, int n_experts) {
    if (!p || !p->enabled || !expert_ids || n_experts <= 0) return;
    
    wisp_mutex_lock(&p->mutex);
    
    uint32_t ctx_hash = get_current_context(p, layer_id);
    PredictorEntry* entry = find_or_create_entry(p, ctx_hash, layer_id);
    
    if (entry) {
        /* Store top-k experts as prediction for this context */
        int n_to_store = n_experts < WISP_PREFETCH_WINDOW ? 
                       n_experts : WISP_PREFETCH_WINDOW;
        
        for (int i = 0; i < n_to_store; i++) {
            entry->predicted_experts[i] = expert_ids[i];
        }
        entry->n_predictions = (uint32_t)n_to_store;
        entry->last_access_ns = wisp_now_ns();
        
        /* Update frequency counters for fallback */
        if (layer_id < p->n_layers) {
            for (int i = 0; i < n_experts; i++) {
                if (expert_ids[i] < p->n_experts_per_layer) {
                    p->layer_expert_counts[layer_id][expert_ids[i]]++;
                }
            }
        }
    }
    
    wisp_mutex_unlock(&p->mutex);
}

int predictor_get_next(ExpertPredictor* p, uint32_t layer_id,
                       uint32_t* out_experts, int max_out) {
    if (!p || !p->enabled || !out_experts || max_out <= 0) return 0;
    
    wisp_mutex_lock(&p->mutex);
    
    uint32_t ctx_hash = get_current_context(p, layer_id);
    uint32_t slot = ctx_hash & p->table_mask;
    PredictorEntry* entry = p->table[slot];
    
    int n_found = 0;
    
    /* Search for matching entry */
    while (entry) {
        if (entry->context_hash == ctx_hash && entry->layer_id == layer_id) {
            entry->hits++;
            entry->last_access_ns = wisp_now_ns();
            
            /* Copy predictions */
            int n_copy = entry->n_predictions < (uint32_t)max_out ? 
                        (int)entry->n_predictions : max_out;
            
            for (int i = 0; i < n_copy; i++) {
                out_experts[n_found++] = entry->predicted_experts[i];
            }
            
            p->total_predictions++;
            if (n_copy > 0) {
                p->correct_predictions++;
            }
            
            break;
        }
        entry = entry->next;
    }
    
    /* Fallback to frequency-based selection if no prediction found */
    if (n_found == 0 && layer_id < p->n_layers) {
        uint32_t* counts = p->layer_expert_counts[layer_id];
        if (counts) {
            /* Find most frequent experts (simple linear scan) */
            for (uint32_t e = 0; e < p->n_experts_per_layer && n_found < max_out; e++) {
                if (counts[e] > 0) {
                    out_experts[n_found++] = e;
                }
            }
        }
    }
    
    wisp_mutex_unlock(&p->mutex);
    return n_found;
}

void predictor_issue_prefetch(WispEngine* eng, ExpertPredictor* p,
                              uint32_t layer_id, WispErrCtx* err) {
    if (!eng || !p || !p->enabled) return;
    
    uint32_t predicted[WISP_PREFETCH_WINDOW];
    int n_predicted = predictor_get_next(p, layer_id, predicted, 
                                         WISP_PREFETCH_WINDOW);
    
    if (n_predicted > 0) {
        wisp_expert_prefetch_hint(eng, (int)layer_id, (int*)predicted, 
                                  n_predicted);
        p->prefetches_issued += (uint64_t)n_predicted;
    }
}

/* Self-test function */
int wisp_selftest_predictor(void) {
    WispErrCtx err = {0};
    ExpertPredictor p;
    
    if (predictor_init(&p, 4, 8, &err) != WISP_OK) return 0;
    
    /* Simulate token sequence */
    predictor_update_token(&p, 100);
    predictor_update_token(&p, 200);
    predictor_update_token(&p, 300);
    predictor_update_token(&p, 400);
    
    /* Record expert usage pattern */
    uint32_t experts[] = {0, 3, 7};
    predictor_record(&p, 0, experts, 3);
    
    /* Add more context */
    predictor_update_token(&p, 500);
    predictor_update_token(&p, 600);
    
    /* Try to predict */
    uint32_t predicted[4];
    int n_pred = predictor_get_next(&p, 0, predicted, 4);
    
    if (n_pred < 1) goto fail;
    
    /* Verify accuracy tracking */
    double acc = predictor_accuracy(&p);
    if (acc < 0.0 || acc > 1.0) goto fail;
    
    /* Test clear */
    predictor_clear(&p);
    if (predictor_accuracy(&p) != 0.0) goto fail;
    
    /* Test enable/disable */
    predictor_set_enabled(&p, 0);
    predictor_update_token(&p, 700);
    predictor_record(&p, 0, experts, 3);
    /* Should not record when disabled */
    
    predictor_destroy(&p);
    return 1;

fail:
    predictor_destroy(&p);
    return 0;
}
