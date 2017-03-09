#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>
#include "numpy/arrayobject.h"
#include <numpy/npy_common.h>
#include "npy_config.h"

#define NBUCKETS_DATA 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE_DATA 7 /* number of cache entries per data bucket */
#define NCACHE_DIM 15 /* number of cache entries per dim bucket */

/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE_DATA];
} data_cache_bucket;

typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE_DIM];
} dim_cache_bucket;

static data_cache_bucket datacache[NBUCKETS_DATA];
static dim_cache_bucket dimcache[NBUCKETS_DIM];

/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void *
npy_alloc_cache(npy_uintp sz)
{
    if (sz > 0 && sz-1 < NBUCKETS_DATA && datacache[sz-1].available > 0) {
        return datacache[sz-1].ptrs[--(datacache[sz-1].available)];
    }
    return PyDataMem_NEW(sz);
}

/* zero initialized data, sz is number of bytes to allocate */
NPY_NO_EXPORT void *
npy_alloc_cache_zero(npy_uintp sz)
{
    void * p;    
    if (sz > 0 && sz-1 < NBUCKETS_DATA && datacache[sz-1].available > 0) {
        p = datacache[sz-1].ptrs[--(datacache[sz-1].available)];
        memset(p, 0, sz);
        return p;
    }
    
    p = PyDataMem_NEW_ZEROED(sz, 1);
    return p;
}

NPY_NO_EXPORT void
npy_free_cache(void * p, npy_uintp sz)
{
    if (p != NULL && sz > 0 && sz-1 < NBUCKETS_DATA &&
        datacache[sz-1].available < NCACHE_DATA) {
        datacache[sz-1].ptrs[datacache[sz-1].available++] = p;
        return ;
    }
    PyDataMem_FREE(p);
}

/*
 * dimension/stride cache, uses a different allocator and is always a multiple
 * of npy_intp
 */
NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz)
{
    void * p;
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }

    if (sz-2 < NBUCKETS_DIM && dimcache[sz-2].available > 0) {
            return dimcache[sz-2].ptrs[--(dimcache[sz-2].available)];
    }
    /* type of dimension elements is npy_intp */
    p = PyArray_malloc(sz * sizeof(npy_intp));
    return p;
}

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sz)
{
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }
    
    if (p != NULL && sz-2 < NBUCKETS_DIM &&
        dimcache[sz-2].available < NCACHE_DIM) {
        dimcache[sz-2].ptrs[dimcache[sz-2].available++] = p;
        return ;
    }
    PyArray_free(p);
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
    return result;
}

/*NUMPY_API
 * Free memory for array data.
 */
NPY_NO_EXPORT void
PyDataMem_FREE(void *ptr)
{
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
