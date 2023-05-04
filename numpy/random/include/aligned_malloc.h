#ifndef _RANDOMDGEN__ALIGNED_MALLOC_H_
#define _RANDOMDGEN__ALIGNED_MALLOC_H_

#include <Python.h>
#include "numpy/npy_common.h"

#define NPY_MEMALIGN 16 /* 16 for SSE2, 32 for AVX, 64 for Xeon Phi */

static inline void *PyArray_realloc_aligned(void *p, size_t n)
{
    void *p1, **p2, *base;
    size_t old_offs, offs = NPY_MEMALIGN - 1 + sizeof(void *);
    if (NPY_UNLIKELY(p != NULL))
    {
        base = *(((void **)p) - 1);
        if (NPY_UNLIKELY((p1 = PyMem_Realloc(base, n + offs)) == NULL))
            return NULL;
        if (NPY_LIKELY(p1 == base))
            return p;
        p2 = (void **)(((Py_uintptr_t)(p1) + offs) & ~(NPY_MEMALIGN - 1));
        old_offs = (size_t)((Py_uintptr_t)p - (Py_uintptr_t)base);
        memmove((void *)p2, ((char *)p1) + old_offs, n);
    }
    else
    {
        if (NPY_UNLIKELY((p1 = PyMem_Malloc(n + offs)) == NULL))
            return NULL;
        p2 = (void **)(((Py_uintptr_t)(p1) + offs) & ~(NPY_MEMALIGN - 1));
    }
    *(p2 - 1) = p1;
    return (void *)p2;
}

static inline void *PyArray_malloc_aligned(size_t n)
{
    return PyArray_realloc_aligned(NULL, n);
}

static inline void *PyArray_calloc_aligned(size_t n, size_t s)
{
    void *p;
    if (NPY_UNLIKELY((p = PyArray_realloc_aligned(NULL, n * s)) == NULL))
        return NULL;
    memset(p, 0, n * s);
    return p;
}

static inline void PyArray_free_aligned(void *p)
{
    void *base = *(((void **)p) - 1);
    PyMem_Free(base);
}

#endif
