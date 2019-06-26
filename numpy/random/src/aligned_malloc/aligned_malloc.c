#include "aligned_malloc.h"

static NPY_INLINE void *PyArray_realloc_aligned(void *p, size_t n);

static NPY_INLINE void *PyArray_malloc_aligned(size_t n);

static NPY_INLINE void *PyArray_calloc_aligned(size_t n, size_t s);

static NPY_INLINE void PyArray_free_aligned(void *p);