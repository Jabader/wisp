/*
 * expert_predictor.h — KV-cache based expert prefetch predictor.
 *
 * Tracks expert access patterns per layer using a small KV cache that maps
 * (prev_tokens_context) -> (likely_next_experts). Uses recent token history
 * to predict which experts will be needed on the next operation.
 *
 * Algorithm:
 *   - Maintain a rolling window of last N tokens per sequence
 *   - Hash token sequence to lookup predicted experts for current layer
 *   - Update predictions based on actual expert usage (learning mode)
 *   - Issue prefetch hints WISP_PREFETCH_WINDOW operations ahead
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include "wisp_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WISP_PREDICTOR_MAX_LAYERS 128
#define WISP_PREDICTOR_MAX_EXPERTS_PER_LAYER 4096
#define WISP_PREDICTOR_CTX_SIZE 4        /* Token context window size */
#define WISP_PREDICTOR_TABLE_SIZE 1024   /* Power of 2 hash table */
#define WISP_PREFETCH_WINDOW 2           /* Look-ahead distance */

typedef struct PredictorEntry {
    uint32_t context_hash;
    uint32_t layer_id;
    uint32_t predicted_experts[WISP_PREFETCH_WINDOW];
    uint32_t n_predictions;
    uint32_t hits;
    uint32_t misses;
    uint64_t last_access_ns;
    struct PredictorEntry* next;  /* Hash chain */
} PredictorEntry;

typedef struct ExpertPredictor {
    PredictorEntry* table[WISP_PREDICTOR_TABLE_SIZE];
    uint32_t table_mask;
    
    /* Per-layer access counters for frequency-based fallback */
    uint32_t* layer_expert_counts[WISP_PREDICTOR_MAX_LAYERS];
    uint32_t n_layers;
    uint32_t n_experts_per_layer;
    
    /* Rolling token context */
    int* token_window;
    uint32_t window_size;
    uint32_t window_pos;
    
    /* Statistics */
    uint64_t total_predictions;
    uint64_t correct_predictions;
    uint64_t prefetches_issued;
    
    wisp_mutex_t mutex;
    int enabled;
} ExpertPredictor;

/* Initialize predictor with model config */
WispError predictor_init(ExpertPredictor* p, int n_layers, 
                         int n_experts_per_layer, WispErrCtx* err);

void predictor_destroy(ExpertPredictor* p);

/* Record actual expert selection for learning */
void predictor_record(ExpertPredictor* p, uint32_t layer_id, 
                      const uint32_t* expert_ids, int n_experts);

/* Get predicted experts for next operation */
int predictor_get_next(ExpertPredictor* p, uint32_t layer_id,
                       uint32_t* out_experts, int max_out);

/* Update internal state with new token */
void predictor_update_token(ExpertPredictor* p, int token_id);

/* Clear all learned state */
void predictor_clear(ExpertPredictor* p);

/* Enable/disable prediction */
void predictor_set_enabled(ExpertPredictor* p, int enabled);

/* Get prediction accuracy */
double predictor_accuracy(const ExpertPredictor* p);

/* Issue prefetch hints to engine */
void predictor_issue_prefetch(WispEngine* eng, ExpertPredictor* p,
                              uint32_t layer_id, WispErrCtx* err);

#ifdef __cplusplus
}
#endif
