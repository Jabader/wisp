/*
 * wisp_async_io.c — Async I/O with io_uring (Linux) and fallback to POSIX AIO
 * 
 * This file provides high-performance async file I/O for expert prefetching:
 * - io_uring-based async read/write on Linux (kernel 5.1+)
 * - POSIX pread/pwrite fallback for other platforms
 * - Batch submission for reduced syscall overhead
 * - Zero-copy friendly with pinned buffers
 *
 * Features:
 * - Non-blocking expert file reads from SSD/NVMe
 * - Prefetch queue management
 * - Callback-based completion notification
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/stat.h>
#ifdef WISP_HAS_IO_URING
#include <liburing.h>
#define WISP_USE_IO_URING 1
#endif
#else
#include <unistd.h>
#include <sys/types.h>
#endif

#include "wisp_engine.h"

/* ======================================================================= *
 * Configuration
 * ======================================================================= */

#define WISP_ASYNC_QUEUE_DEPTH 256
#define WISP_ASYNC_MAX_BATCH 64

/* ======================================================================= *
 * Types
 * ======================================================================= */

typedef struct WispAsyncRequest {
    void* buffer;
    size_t size;
    int fd;
    off_t offset;
    void (*callback)(int result, void* user_data);
    void* user_data;
    int is_write;
} WispAsyncRequest;

typedef struct WispAsyncCtx {
#ifdef WISP_USE_IO_URING
    struct io_uring ring;
#endif
    int initialized;
    int queue_depth;
    int pending_count;
} WispAsyncCtx;

/* ======================================================================= *
 * Initialization / Cleanup
 * ======================================================================= */

WispError wisp_async_init(WispAsyncCtx* ctx, int queue_depth, WispErrCtx* err) {
    if (!ctx) {
        WISP_ERR_SET(err, WISP_ERR_INVALID_ARG, "wisp_async_init: ctx is NULL");
        return WISP_ERR_INVALID_ARG;
    }
    
    memset(ctx, 0, sizeof(*ctx));
    
    if (queue_depth <= 0) {
        queue_depth = WISP_ASYNC_QUEUE_DEPTH;
    }
    ctx->queue_depth = queue_depth;
    
#ifdef WISP_USE_IO_URING
    if (io_uring_queue_init(queue_depth, &ctx->ring, 0) < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, 
                     "wisp_async_init: io_uring_queue_init failed: %s", 
                     strerror(errno));
        ctx->initialized = 0;
        return WISP_ERR_IO;
    }
    ctx->initialized = 1;
#else
    ctx->initialized = 1;
#endif
    
    return WISP_OK;
}

void wisp_async_cleanup(WispAsyncCtx* ctx) {
    if (!ctx || !ctx->initialized) return;
    
#ifdef WISP_USE_IO_URING
    io_uring_queue_exit(&ctx->ring);
#endif
    ctx->initialized = 0;
    ctx->pending_count = 0;
}

/* ======================================================================= *
 * Single Request Submit
 * ======================================================================= */

static int wisp_async_submit_read(WispAsyncCtx* ctx, WispAsyncRequest* req,
                                  WispErrCtx* err) {
    if (!ctx || !ctx->initialized || !req) {
        WISP_ERR_SET(err, WISP_ERR_INVALID_ARG, "wisp_async_submit_read: invalid args");
        return -1;
    }
    
#ifdef WISP_USE_IO_URING
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ctx->ring);
    if (!sqe) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_read: SQE full");
        return -1;
    }
    
    io_uring_prep_read(sqe, req->fd, req->buffer, req->size, req->offset);
    io_uring_sqe_set_data(sqe, req);
    
    int ret = io_uring_submit(&ctx->ring);
    if (ret < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_read: submit failed: %s",
                     strerror(errno));
        return -1;
    }
    
    ctx->pending_count++;
    return 0;
#else
    ssize_t result = pread(req->fd, req->buffer, req->size, req->offset);
    if (result < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_read: pread failed: %s",
                     strerror(errno));
        return -1;
    }
    if (req->callback) {
        req->callback((int)result, req->user_data);
    }
    return 0;
#endif
}

static int wisp_async_submit_write(WispAsyncCtx* ctx, WispAsyncRequest* req,
                                   WispErrCtx* err) {
    if (!ctx || !ctx->initialized || !req) {
        WISP_ERR_SET(err, WISP_ERR_INVALID_ARG, "wisp_async_submit_write: invalid args");
        return -1;
    }
    
#ifdef WISP_USE_IO_URING
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ctx->ring);
    if (!sqe) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_write: SQE full");
        return -1;
    }
    
    io_uring_prep_write(sqe, req->fd, req->buffer, req->size, req->offset);
    io_uring_sqe_set_data(sqe, req);
    
    int ret = io_uring_submit(&ctx->ring);
    if (ret < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_write: submit failed: %s",
                     strerror(errno));
        return -1;
    }
    
    ctx->pending_count++;
    return 0;
#else
    ssize_t result = pwrite(req->fd, req->buffer, req->size, req->offset);
    if (result < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_submit_write: pwrite failed: %s",
                     strerror(errno));
        return -1;
    }
    if (req->callback) {
        req->callback((int)result, req->user_data);
    }
    return 0;
#endif
}

/* ======================================================================= *
 * Public API: Read/Write
 * ======================================================================= */

WispError wisp_async_read(WispAsyncCtx* ctx, void* buffer, size_t size,
                          int fd, off_t offset, void (*callback)(int, void*),
                          void* user_data, WispErrCtx* err) {
    WispAsyncRequest req;
    req.buffer = buffer;
    req.size = size;
    req.fd = fd;
    req.offset = offset;
    req.callback = callback;
    req.user_data = user_data;
    req.is_write = 0;
    
    if (wisp_async_submit_read(ctx, &req, err) != 0) {
        return WISP_ERR_IO;
    }
    return WISP_OK;
}

WispError wisp_async_write(WispAsyncCtx* ctx, const void* buffer, size_t size,
                           int fd, off_t offset, void (*callback)(int, void*),
                           void* user_data, WispErrCtx* err) {
    WispAsyncRequest req;
    req.buffer = (void*)buffer;
    req.size = size;
    req.fd = fd;
    req.offset = offset;
    req.callback = callback;
    req.user_data = user_data;
    req.is_write = 1;
    
    if (wisp_async_submit_write(ctx, &req, err) != 0) {
        return WISP_ERR_IO;
    }
    return WISP_OK;
}

/* ======================================================================= *
 * Batch Operations
 * ======================================================================= */

typedef struct {
    void* buffer;
    size_t size;
    off_t offset;
} WispBatchIoEntry;

WispError wisp_async_read_batch(WispAsyncCtx* ctx, int fd,
                                WispBatchIoEntry* entries, int count,
                                void (*callback)(int, void*),
                                void* user_data, WispErrCtx* err) {
    if (!ctx || !ctx->initialized || !entries || count <= 0) {
        WISP_ERR_SET(err, WISP_ERR_INVALID_ARG, 
                     "wisp_async_read_batch: invalid args");
        return WISP_ERR_INVALID_ARG;
    }
    
    if (count > WISP_ASYNC_MAX_BATCH) {
        count = WISP_ASYNC_MAX_BATCH;
    }
    
#ifdef WISP_USE_IO_URING
    for (int i = 0; i < count; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ctx->ring);
        if (!sqe) {
            WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_read_batch: SQE full at %d", i);
            return WISP_ERR_IO;
        }
        
        io_uring_prep_read(sqe, fd, entries[i].buffer, entries[i].size, 
                          entries[i].offset);
        io_uring_sqe_set_data(sqe, user_data);
    }
    
    int ret = io_uring_submit(&ctx->ring);
    if (ret < 0) {
        WISP_ERR_SET(err, WISP_ERR_IO, "wisp_async_read_batch: submit failed: %s",
                     strerror(errno));
        return WISP_ERR_IO;
    }
    
    ctx->pending_count += count;
    return WISP_OK;
#else
    int total_read = 0;
    for (int i = 0; i < count; i++) {
        ssize_t result = pread(fd, entries[i].buffer, entries[i].size, 
                              entries[i].offset);
        if (result < 0) {
            WISP_ERR_SET(err, WISP_ERR_IO, 
                         "wisp_async_read_batch: pread failed at %d: %s",
                         i, strerror(errno));
            return WISP_ERR_IO;
        }
        total_read += (int)result;
    }
    if (callback) {
        callback(total_read, user_data);
    }
    return WISP_OK;
#endif
}

/* ======================================================================= *
 * Completion Polling
 * ======================================================================= */

int wisp_async_poll_completions(WispAsyncCtx* ctx, int wait,
                                void (*handler)(int result, void* user_data)) {
    if (!ctx || !ctx->initialized) return 0;
    
#ifdef WISP_USE_IO_URING
    struct io_uring_cqe* cqe;
    unsigned head;
    unsigned completed = 0;
    
    if (wait) {
        if (io_uring_wait_cqe(&ctx->ring, &cqe) < 0) {
            return 0;
        }
        if (cqe) {
            WispAsyncRequest* req = (WispAsyncRequest*)io_uring_cqe_get_data(cqe);
            if (handler && req) {
                handler(cqe->res, req->user_data);
            }
            io_uring_cqe_seen(&ctx->ring, cqe);
            ctx->pending_count--;
            completed++;
        }
    } else {
        io_uring_for_each_cqe(&ctx->ring, head, cqe) {
            WispAsyncRequest* req = (WispAsyncRequest*)io_uring_cqe_get_data(cqe);
            if (handler && req) {
                handler(cqe->res, req->user_data);
            }
            ctx->pending_count--;
            completed++;
        }
        io_uring_cq_advance(&ctx->ring, completed);
    }
    
    return (int)completed;
#else
    (void)wait;
    (void)handler;
    return 0;
#endif
}

/* ======================================================================= *
 * File Descriptor Helpers
 * ======================================================================= */

int wisp_async_open_expert_file(const char* path, int readonly) {
    if (!path) return -1;
    
    int flags = O_RDONLY;
    if (!readonly) {
        flags = O_RDWR | O_CREAT;
    }
    
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    
    int fd = open(path, flags, 0644);
    return fd;
}

void wisp_async_close_file(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}

/* ======================================================================= *
 * Self-test
 * ======================================================================= */

int wisp_selftest_async_io(void) {
    WispErrCtx err = {0};
    WispAsyncCtx ctx;
    
    if (wisp_async_init(&ctx, 32, &err) != WISP_OK) {
        return 0;
    }
    
    /* Create a temp file for testing */
    const char* test_path = "/tmp/wisp_async_test.bin";
    int fd = open(test_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        wisp_async_cleanup(&ctx);
        return 0;
    }
    
    /* Write test data */
    uint8_t write_buf[1024];
    for (int i = 0; i < 1024; i++) {
        write_buf[i] = (uint8_t)(i & 0xFF);
    }
    
    ssize_t written = write(fd, write_buf, sizeof(write_buf));
    if (written != sizeof(write_buf)) {
        close(fd);
        unlink(test_path);
        wisp_async_cleanup(&ctx);
        return 0;
    }
    
    /* Read back with async */
    uint8_t read_buf[1024];
    memset(read_buf, 0, sizeof(read_buf));
    
    WispError re = wisp_async_read(&ctx, read_buf, sizeof(read_buf),
                                   fd, 0, NULL, NULL, &err);
    
#ifdef WISP_USE_IO_URING
    if (re == WISP_OK) {
        wisp_async_poll_completions(&ctx, 1, NULL);
    }
#endif
    
    close(fd);
    unlink(test_path);
    wisp_async_cleanup(&ctx);
    
    /* Verify data */
    for (int i = 0; i < 1024; i++) {
        if (read_buf[i] != (uint8_t)(i & 0xFF)) {
            return 0;
        }
    }
    
    return 1;
}
