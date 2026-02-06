/*
 * NumPy Debug-only C-Level Function Call Tracing - Implementation
 *
 * This file implements the debug-only C-level function call tracing mechanism
 * described in numpy/prof.md.
 *
 * Key features:
 * - Thread-local call stacks using _Thread_local
 * - Fixed-size memory pools to avoid heap allocation in hot paths
 * - Compiler instrumentation hooks (__cyg_profile_func_enter/exit)
 * - Optional symbol resolution via dladdr()
 *
 * Build requirements:
 * - Compile with -finstrument-functions
 * - Define NUMPY_DEBUG_CTRACE
 * - Use -O0 -g3 -fno-inline for best results
 */

#include "npy_config.h"

#ifdef NUMPY_DEBUG_CTRACE

#include "npy_ctrace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Platform-specific includes for symbol resolution */
#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif

/*
 * Thread-local storage for the call stack.
 * We use a fixed-size pool to avoid malloc in the instrumentation hooks.
 */
typedef struct {
    npy_trace_node_t pool[NPY_CTRACE_MAX_DEPTH];
    npy_trace_node_t *head;      /* Top of the call stack */
    npy_trace_node_t *free_list; /* Free nodes for reuse */
    uint32_t depth;              /* Current stack depth */
    int enabled;                 /* Tracing enabled flag */
    int initialized;             /* Pool initialized flag */
} npy_ctrace_tls_t;

/* Check for thread-local storage support */
#if defined(HAVE_THREAD_LOCAL)
    #define NPY_TLS thread_local
#elif defined(HAVE__THREAD_LOCAL)
    #define NPY_TLS _Thread_local
#elif defined(HAVE__THREAD)
    #define NPY_TLS __thread
#elif defined(HAVE___DECLSPEC_THREAD_)
    #define NPY_TLS __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
    #define NPY_TLS __thread
#elif defined(_MSC_VER)
    #define NPY_TLS __declspec(thread)
#else
    #error "No thread-local storage support detected"
#endif

/* Thread-local state */
static NPY_TLS npy_ctrace_tls_t tls_state = {0};

/* Global callback (shared across threads) */
static npy_ctrace_callback_t g_callback = NULL;

/* Global initialization flag */
static int g_initialized = 0;

/*
 * Initialize the thread-local pool.
 * Sets up the free list by linking all nodes.
 */
__attribute__((no_instrument_function))
static void
init_tls_pool(void)
{
    if (tls_state.initialized) {
        return;
    }

    /* Link all nodes into the free list */
    for (size_t i = 0; i < NPY_CTRACE_MAX_DEPTH - 1; i++) {
        tls_state.pool[i].next = &tls_state.pool[i + 1];
        tls_state.pool[i].prev = NULL;
    }
    tls_state.pool[NPY_CTRACE_MAX_DEPTH - 1].next = NULL;
    tls_state.pool[NPY_CTRACE_MAX_DEPTH - 1].prev = NULL;

    tls_state.free_list = &tls_state.pool[0];
    tls_state.head = NULL;
    tls_state.depth = 0;
    tls_state.enabled = 0;
    tls_state.initialized = 1;
}

/*
 * Allocate a node from the free list.
 * Returns NULL if the pool is exhausted.
 */
__attribute__((no_instrument_function))
static npy_trace_node_t *
alloc_node(void)
{
    if (!tls_state.free_list) {
        return NULL;  /* Pool exhausted */
    }

    npy_trace_node_t *node = tls_state.free_list;
    tls_state.free_list = node->next;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

/*
 * Return a node to the free list.
 */
__attribute__((no_instrument_function))
static void
free_node(npy_trace_node_t *node)
{
    if (!node) {
        return;
    }
    node->next = tls_state.free_list;
    node->prev = NULL;
    tls_state.free_list = node;
}

/*
 * Default callback: prints entry/exit to stderr.
 */
__attribute__((no_instrument_function))
static void
default_callback(void *func, void *caller, uint32_t depth, int is_entry)
{
    /* Indent based on depth */
    for (uint32_t i = 0; i < depth; i++) {
        fprintf(stderr, "  ");
    }

    if (is_entry) {
        fprintf(stderr, "-> %p (from %p)\n", func, caller);
    }
    else {
        fprintf(stderr, "<- %p\n", func);
    }
}

/* Public API implementation */

__attribute__((no_instrument_function))
void
npy_ctrace_init(void)
{
    if (g_initialized) {
        return;
    }
    g_callback = default_callback;
    g_initialized = 1;
}

__attribute__((no_instrument_function))
void
npy_ctrace_shutdown(void)
{
    g_initialized = 0;
    g_callback = NULL;
}

__attribute__((no_instrument_function))
void
npy_ctrace_enable(void)
{
    init_tls_pool();
    tls_state.enabled = 1;
}

__attribute__((no_instrument_function))
void
npy_ctrace_disable(void)
{
    tls_state.enabled = 0;
}

__attribute__((no_instrument_function))
int
npy_ctrace_is_enabled(void)
{
    return tls_state.enabled;
}

__attribute__((no_instrument_function))
void
npy_ctrace_set_callback(npy_ctrace_callback_t callback)
{
    g_callback = callback ? callback : default_callback;
}

__attribute__((no_instrument_function))
uint32_t
npy_ctrace_get_depth(void)
{
    return tls_state.depth;
}

__attribute__((no_instrument_function))
size_t
npy_ctrace_snapshot(void **buffer, size_t max_size)
{
    if (!buffer || max_size == 0) {
        return 0;
    }

    size_t count = 0;
    npy_trace_node_t *node = tls_state.head;

    while (node && count < max_size) {
        buffer[count++] = node->func;
        node = node->prev;
    }

    return count;
}

__attribute__((no_instrument_function))
const char *
npy_ctrace_resolve_symbol(void *addr, char *buf, size_t size)
{
    if (!addr || !buf || size == 0) {
        return NULL;
    }

#ifdef HAVE_DLFCN_H
    Dl_info info;
    if (dladdr(addr, &info) && info.dli_sname) {
        strncpy(buf, info.dli_sname, size - 1);
        buf[size - 1] = '\0';
        return buf;
    }
#endif

    /* Fallback: just format the address */
    snprintf(buf, size, "%p", addr);
    return buf;
}

__attribute__((no_instrument_function))
void
npy_ctrace_dump_stack(void)
{
    char symbol_buf[256];

    fprintf(stderr, "=== NumPy C Call Stack (depth=%u) ===\n", tls_state.depth);

    npy_trace_node_t *node = tls_state.head;
    uint32_t frame = 0;

    while (node) {
        const char *symbol = npy_ctrace_resolve_symbol(
            node->func, symbol_buf, sizeof(symbol_buf));

        fprintf(stderr, "#%u: %s (%p)\n", frame, symbol ? symbol : "???",
                node->func);

        node = node->prev;
        frame++;
    }

    fprintf(stderr, "=== End Stack ===\n");
}

/*
 * Compiler instrumentation hooks.
 * These are called automatically by the compiler when -finstrument-functions
 * is enabled. They must always be defined and exported so that instrumented
 * code can link against them.
 */

void
__cyg_profile_func_enter(void *func, void *caller)
{
    /* Skip if not initialized or not enabled */
    if (!tls_state.initialized || !tls_state.enabled) {
        return;
    }

    /* Allocate a node for this call */
    npy_trace_node_t *node = alloc_node();
    if (!node) {
        /* Pool exhausted, silently drop */
        return;
    }

    /* Fill in the node */
    node->func = func;
    node->caller = caller;
    node->depth = tls_state.depth;

    /* Push onto the stack */
    node->prev = tls_state.head;
    if (tls_state.head) {
        tls_state.head->next = node;
    }
    tls_state.head = node;
    tls_state.depth++;

    /* Invoke callback */
    if (g_callback) {
        g_callback(func, caller, node->depth, 1);
    }
}

void
__cyg_profile_func_exit(void *func, void *caller)
{
    /* Skip if not initialized or not enabled */
    if (!tls_state.initialized || !tls_state.enabled) {
        return;
    }

    /* Pop from the stack */
    npy_trace_node_t *node = tls_state.head;
    if (!node) {
        /* Stack underflow - shouldn't happen */
        return;
    }

    /* Invoke callback before popping */
    if (g_callback) {
        g_callback(func, caller, node->depth, 0);
    }

    /* Update head */
    tls_state.head = node->prev;
    if (tls_state.head) {
        tls_state.head->next = NULL;
    }

    /* Return node to free list */
    free_node(node);

    if (tls_state.depth > 0) {
        tls_state.depth--;
    }
}

/*
 * Exported hook entry points for use by other modules.
 * These are called by instrumented code in other shared libraries.
 */
__attribute__((visibility("default")))
void
npy_ctrace_hook_enter(void *func, void *caller)
{
    __cyg_profile_func_enter(func, caller);
}

__attribute__((visibility("default")))
void
npy_ctrace_hook_exit(void *func, void *caller)
{
    __cyg_profile_func_exit(func, caller);
}

#endif /* NUMPY_DEBUG_CTRACE */
