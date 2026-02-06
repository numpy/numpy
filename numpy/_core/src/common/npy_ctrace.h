/*
 * NumPy Debug-only C-Level Function Call Tracing
 *
 * This header provides a debug-only mechanism to trace C-level function calls
 * and call nesting executed during NumPy operations. It uses compiler-inserted
 * function instrumentation via -finstrument-functions.
 *
 * This feature is ONLY enabled when NUMPY_DEBUG_CTRACE is defined.
 * It is NOT intended for production use.
 *
 * See numpy/prof.md for the full design document.
 */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_CTRACE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CTRACE_H_

#include "npy_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NUMPY_DEBUG_CTRACE

#include <stddef.h>
#include <stdint.h>

/*
 * Maximum depth of the call stack trace.
 * Calls beyond this depth will be silently dropped.
 */
#ifndef NPY_CTRACE_MAX_DEPTH
#define NPY_CTRACE_MAX_DEPTH 4096
#endif

/*
 * Trace node structure for the thread-local call stack.
 * Implemented as a doubly linked list for efficient push/pop.
 */
typedef struct npy_trace_node {
    void *func;                     /* Function address */
    void *caller;                   /* Caller address */
    struct npy_trace_node *prev;    /* Previous node in stack */
    struct npy_trace_node *next;    /* Next node (for free list) */
    uint32_t depth;                 /* Current nesting depth */
} npy_trace_node_t;

/*
 * Trace output callback type.
 * Called on function entry/exit when tracing is enabled.
 *
 * Parameters:
 *   func   - Address of the function being entered/exited
 *   caller - Address of the calling function
 *   depth  - Current nesting depth
 *   is_entry - 1 for entry, 0 for exit
 */
typedef void (*npy_ctrace_callback_t)(void *func, void *caller,
                                       uint32_t depth, int is_entry);

/*
 * Initialize the tracing subsystem.
 * Must be called before any tracing operations.
 * Thread-safe: can be called multiple times.
 */
__attribute__((no_instrument_function))
void npy_ctrace_init(void);

/*
 * Shutdown the tracing subsystem and free resources.
 * Thread-safe.
 */
__attribute__((no_instrument_function))
void npy_ctrace_shutdown(void);

/*
 * Enable tracing for the current thread.
 * Tracing is disabled by default.
 */
__attribute__((no_instrument_function))
void npy_ctrace_enable(void);

/*
 * Disable tracing for the current thread.
 */
__attribute__((no_instrument_function))
void npy_ctrace_disable(void);

/*
 * Check if tracing is enabled for the current thread.
 * Returns: 1 if enabled, 0 if disabled
 */
__attribute__((no_instrument_function))
int npy_ctrace_is_enabled(void);

/*
 * Set the trace output callback.
 * If NULL, a default callback that prints to stderr is used.
 *
 * Parameters:
 *   callback - The callback function, or NULL for default
 */
__attribute__((no_instrument_function))
void npy_ctrace_set_callback(npy_ctrace_callback_t callback);

/*
 * Get the current call stack depth for this thread.
 * Returns: Current depth (0 if at top level)
 */
__attribute__((no_instrument_function))
uint32_t npy_ctrace_get_depth(void);

/*
 * Snapshot the current call stack.
 * Copies function addresses into the provided buffer.
 *
 * Parameters:
 *   buffer   - Array to store function addresses
 *   max_size - Maximum number of entries to store
 *
 * Returns: Number of entries actually stored
 */
__attribute__((no_instrument_function))
size_t npy_ctrace_snapshot(void **buffer, size_t max_size);

/*
 * Resolve a function address to a symbol name.
 * Uses dladdr() if available, otherwise returns NULL.
 *
 * Parameters:
 *   addr - Function address to resolve
 *   buf  - Buffer to store the symbol name
 *   size - Size of the buffer
 *
 * Returns: Pointer to buf on success, NULL on failure
 */
__attribute__((no_instrument_function))
const char *npy_ctrace_resolve_symbol(void *addr, char *buf, size_t size);

/*
 * Dump the current call stack to stderr.
 * Useful for debugging and diagnostics.
 */
__attribute__((no_instrument_function))
void npy_ctrace_dump_stack(void);

#else /* !NUMPY_DEBUG_CTRACE */

/* No-op stubs when tracing is disabled */
#define npy_ctrace_init()           ((void)0)
#define npy_ctrace_shutdown()       ((void)0)
#define npy_ctrace_enable()         ((void)0)
#define npy_ctrace_disable()        ((void)0)
#define npy_ctrace_is_enabled()     (0)
#define npy_ctrace_set_callback(cb) ((void)0)
#define npy_ctrace_get_depth()      (0)
#define npy_ctrace_snapshot(b, s)   (0)
#define npy_ctrace_resolve_symbol(a, b, s) (NULL)
#define npy_ctrace_dump_stack()     ((void)0)

#endif /* NUMPY_DEBUG_CTRACE */

#ifdef __cplusplus
}
#endif

#endif /* NUMPY_CORE_SRC_COMMON_NPY_CTRACE_H_ */
