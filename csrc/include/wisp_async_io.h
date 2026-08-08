#ifndef WISP_ASYNC_IO_H
#define WISP_ASYNC_IO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __linux__
#ifdef WISP_HAS_IO_URING
#include <liburing.h>
#endif
#endif

#define WISP_ASYNC_QUEUE_DEPTH 64
#define WISP_ASYNC_MAX_BUFS 8

typedef struct {
    void* buffer;
    size_t size;
    int64_t file_offset;
    int fd;
    void* user_data;
    bool completed;
    int result;
} WispAsyncRequest;

typedef struct {
#ifdef __linux__
    struct io_uring ring;
    bool is_initialized;
#endif
    int queue_depth;
    
    // Double buffering pools
    void* io_buffers[WISP_ASYNC_MAX_BUFS];
    size_t buffer_size;
    bool buffer_in_use[WISP_ASYNC_MAX_BUFS];
    
    // Request tracking
    WispAsyncRequest pending_requests[WISP_ASYNC_QUEUE_DEPTH];
    int head;
    int tail;
    
    // Fallback mode flag
    bool use_fallback;
} WispAsyncContext;

/**
 * Initialize async I/O context.
 * Uses io_uring on Linux if available, falls back to sync I/O otherwise.
 * 
 * @param ctx Context to initialize
 * @param queue_depth Depth of the submission queue
 * @param buffer_size Size of each IO buffer (should match expert size)
 * @return 0 on success, -1 on failure
 */
int wisp_async_init(WispAsyncContext* ctx, int queue_depth, size_t buffer_size);

/**
 * Submit an asynchronous read request.
 * Non-blocking: returns immediately.
 * 
 * @param ctx Context
 * @param fd File descriptor
 * @param offset File offset to read from
 * @param size Number of bytes to read
 * @param user_data Pointer passed to completion handler
 * @return 0 on success (queued), -1 on error (queue full or I/O error)
 */
int wisp_async_submit_read(WispAsyncContext* ctx, int fd, int64_t offset, size_t size, void* user_data);

/**
 * Poll for completed requests.
 * Processes completions and calls internal handlers.
 * 
 * @param ctx Context
 * @param wait_ms Time to wait in milliseconds (0 for non-blocking poll)
 * @return Number of completed requests processed
 */
int wisp_async_poll(WispAsyncContext* ctx, int wait_ms);

/**
 * Get a completed request.
 * 
 * @param ctx Context
 * @return Pointer to completed request or NULL if none available
 */
WispAsyncRequest* wisp_async_get_completed(WispAsyncContext* ctx);

/**
 * Release a buffer back to the pool.
 * 
 * @param ctx Context
 * @param buffer_ptr Pointer to buffer to release
 */
void wisp_async_release_buffer(WispAsyncContext* ctx, void* buffer_ptr);

/**
 * Get a free buffer from the pool.
 * 
 * @param ctx Context
 * @return Pointer to free buffer or NULL if none available
 */
void* wisp_async_get_free_buffer(WispAsyncContext* ctx);

/**
 * Destroy async context and free resources.
 * 
 * @param ctx Context to destroy
 */
void wisp_async_destroy(WispAsyncContext* ctx);

#endif // WISP_ASYNC_IO_H
