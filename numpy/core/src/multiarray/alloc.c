#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#if PY_VERSION_HEX >= 0x03060000
#include <pymem.h>
/* public api in 3.7 */
#if PY_VERSION_HEX < 0x03070000
#define PyTraceMalloc_Track _PyTraceMalloc_Track
#define PyTraceMalloc_Untrack _PyTraceMalloc_Untrack
#endif
#else
#define PyTraceMalloc_Track(...)
#define PyTraceMalloc_Untrack(...)
#endif

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>
#include "numpy/arrayobject.h"
#include <numpy/npy_common.h>
#include "npy_config.h"
#include "alloc.h"


#include <assert.h>

#ifdef NPY_OS_LINUX
#include <sys/mman.h>
#ifndef MADV_HUGEPAGE
/*
 * Use code 14 (MADV_HUGEPAGE) if it isn't defined. This gives a chance of
 * enabling huge pages even if built with linux kernel < 2.6.38
 */
#define MADV_HUGEPAGE 14
#endif
#endif

#define NBUCKETS 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE 7 /* number of cache entries per bucket */
/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];

/* as the cache is managed in global variables verify the GIL is held */
#if defined(NPY_PY3K)
#define NPY_CHECK_GIL_HELD() PyGILState_Check()
#else
#define NPY_CHECK_GIL_HELD() 1
#endif

/*
 * very simplistic small memory block cache to avoid more expensive libc
 * allocations
 * base function for data cache with 1 byte buckets and dimension cache with
 * sizeof(npy_intp) byte buckets
 */
static NPY_INLINE void *
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t))
{
    void * p;
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    assert(NPY_CHECK_GIL_HELD());
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
    p = alloc(nelem * esz);
    if (p) {
#ifdef _PyPyGC_AddMemoryPressure
        _PyPyPyGC_AddMemoryPressure(nelem * esz);
#endif
#ifdef NPY_OS_LINUX
        /* allow kernel allocating huge pages for large arrays */
        if (NPY_UNLIKELY(nelem * esz >= ((1u<<22u)))) {
            npy_uintp offset = 4096u - (npy_uintp)p % (4096u);
            npy_uintp length = nelem * esz - offset;
            /**
             * Intentionally not checking for errors that may be returned by
             * older kernel versions; optimistically tries enabling huge pages.
             */
            madvise((void*)((npy_uintp)p + offset), length, MADV_HUGEPAGE);
        }
#endif
    }
    return p;
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static NPY_INLINE void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    assert(NPY_CHECK_GIL_HELD());
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    dealloc(p);
}


/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void *
npy_alloc_cache(npy_uintp sz)
{
    return _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
}

/* zero initialized data, sz is number of bytes to allocate */
NPY_NO_EXPORT void *
npy_alloc_cache_zero(npy_uintp sz)
{
    void * p;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = PyDataMem_NEW_ZEROED(sz, 1);
    NPY_END_THREADS;
    return p;
}

NPY_NO_EXPORT void
npy_free_cache(void * p, npy_uintp sz)
{
    _npy_free_cache(p, sz, NBUCKETS, datacache, &PyDataMem_FREE);
}

/*
 * dimension/stride cache, uses a different allocator and is always a multiple
 * of npy_intp
 */
NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz)
{
    /*
     * make sure any temporary allocation can be used for array metadata which
     * uses one memory block for both dimensions and strides
     */
    if (sz < 2) {
        sz = 2;
    }
    return _npy_alloc_cache(sz, sizeof(npy_intp), NBUCKETS_DIM, dimcache,
                            &PyArray_malloc);
}

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sz)
{
    /* see npy_alloc_cache_dim */
    if (sz < 2) {
        sz = 2;
    }
    _npy_free_cache(p, sz, NBUCKETS_DIM, dimcache,
                    &PyArray_free);
}


/* malloc/free/realloc hook */
NPY_NO_EXPORT PyDataMem_EventHookFunc *_PyDataMem_eventhook;
NPY_NO_EXPORT void *_PyDataMem_eventhook_user_data;

/*NUMPY_API
 * Sets the allocation event hook for numpy array data.
 * Takes a PyDataMem_EventHookFunc *, which has the signature:
 *        void hook(void *old, void *new, size_t size, void *user_data).
 *   Also takes a void *user_data, and void **old_data.
 *
 * Returns a pointer to the previous hook or NULL.  If old_data is
 * non-NULL, the previous user_data pointer will be copied to it.
 *
 * If not NULL, hook will be called at the end of each PyDataMem_NEW/FREE/RENEW:
 *   result = PyDataMem_NEW(size)        -> (*hook)(NULL, result, size, user_data)
 *   PyDataMem_FREE(ptr)                 -> (*hook)(ptr, NULL, 0, user_data)
 *   result = PyDataMem_RENEW(ptr, size) -> (*hook)(ptr, result, size, user_data)
 *
 * When the hook is called, the GIL will be held by the calling
 * thread.  The hook should be written to be reentrant, if it performs
 * operations that might cause new allocation events (such as the
 * creation/destruction numpy objects, or creating/destroying Python
 * objects which might cause a gc)
 */
NPY_NO_EXPORT PyDataMem_EventHookFunc *
PyDataMem_SetEventHook(PyDataMem_EventHookFunc *newhook,
                       void *user_data, void **old_data)
{
    PyDataMem_EventHookFunc *temp;
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API
    temp = _PyDataMem_eventhook;
    _PyDataMem_eventhook = newhook;
    if (old_data != NULL) {
        *old_data = _PyDataMem_eventhook_user_data;
    }
    _PyDataMem_eventhook_user_data = user_data;
    NPY_DISABLE_C_API
    return temp;
}

/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW(size_t size)
{
    void *result;

    result = malloc(size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
{
    void *result;

    result = calloc(size, elsize);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(NULL, result, size * elsize,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    return result;
}

/*NUMPY_API
 * Free memory for array data.
 */
NPY_NO_EXPORT void
PyDataMem_FREE(void *ptr)
{
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    free(ptr);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, NULL, 0,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
}

/*NUMPY_API
 * Reallocate/resize memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_RENEW(void *ptr, size_t size)
{
    void *result;

    result = realloc(ptr, size);
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    }
    PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (_PyDataMem_eventhook != NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        if (_PyDataMem_eventhook != NULL) {
            (*_PyDataMem_eventhook)(ptr, result, size,
                                    _PyDataMem_eventhook_user_data);
        }
        NPY_DISABLE_C_API
    }
    return result;
}
