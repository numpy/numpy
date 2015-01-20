#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>
#include "numpy/arrayobject.h"
#include <numpy/npy_common.h>
#include "npy_config.h"

#include <assert.h>

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
 * creation/descruction numpy objects, or creating/destroying Python
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

/* Allocator structure */
static PyDataMem_Allocator default_allocator = {
    malloc,
    calloc,
    free,
    realloc
};
static const PyDataMem_Allocator *current_allocator = &default_allocator;

static NPY_INLINE void *
allocator_new(const PyDataMem_Allocator *a, size_t size)
{
    void *result;

    result = a->alloc(size);
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

static NPY_INLINE void *
allocator_new_zeroed(const PyDataMem_Allocator *a, size_t size, size_t elsize)
{
    void *result;

    result = a->zeroed_alloc(size, elsize);
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

static NPY_INLINE void
allocator_free(const PyDataMem_Allocator *a, void *ptr)
{
    a->free(ptr);
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

static NPY_INLINE void *
allocator_renew(const PyDataMem_Allocator *a, void *ptr, size_t size)
{
    void *result;

    result = a->realloc(ptr, size);
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

static NPY_INLINE void *
current_allocator_new(size_t size)
{
    return allocator_new(current_allocator, size);
}

static NPY_INLINE void *
current_allocator_new_zeroed(size_t size, size_t elsize)
{
    return allocator_new_zeroed(current_allocator, size, elsize);
}

static NPY_INLINE void
current_allocator_free(void *ptr)
{
    allocator_free(current_allocator, ptr);
}

static NPY_INLINE void *
current_allocator_renew(void *ptr, size_t size)
{
    return allocator_renew(current_allocator, ptr, size);
}


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
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
    return alloc(nelem * esz);
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static NPY_INLINE void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    dealloc(p);
}

/*
 * clear all cache data in the given cache
 */
static void
_npy_clear_cache(npy_uint msz, cache_bucket * cache, void (*dealloc)(void *))
{
    npy_intp i, nelem;
    for (nelem = 0; nelem < msz; nelem++) {
        for (i = 0; i < cache[nelem].available; i++) {
            dealloc(cache[nelem].ptrs[i]);
        }
        cache[nelem].available = 0;
    }
}


/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void *
npy_alloc_cache(const PyDataMem_Allocator **a, npy_uintp sz)
{
    *a = current_allocator;
    return _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &current_allocator_new);
}

/* zero initialized data, sz is number of bytes to allocate */
NPY_NO_EXPORT void *
npy_alloc_cache_zero(const PyDataMem_Allocator **a, npy_uintp sz)
{
    void * p;
    *a = current_allocator;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &current_allocator_new);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = current_allocator_new_zeroed(sz, 1);
    NPY_END_THREADS;
    return p;
}

NPY_NO_EXPORT void
npy_free_cache(const PyDataMem_Allocator *a, void * p, npy_uintp sz)
{
    if (a == current_allocator) {
        _npy_free_cache(p, sz, NBUCKETS, datacache, &current_allocator_free);
    }
    else {
        allocator_free(a, p);
    }
}

/*
 * dimension/stride cache, uses a different allocator and is always a multiple
 * of npy_intp
 */
NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz)
{
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }
    return _npy_alloc_cache(sz, sizeof(npy_intp), NBUCKETS_DIM, dimcache,
                            &PyArray_malloc);
}

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sz)
{
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }
    _npy_free_cache(p, sz, NBUCKETS_DIM, dimcache,
                    &PyArray_free);
}


/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW(size_t size)
{
    return allocator_new(&default_allocator, size);
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
{
    return allocator_new_zeroed(&default_allocator, size, elsize);
}

/*NUMPY_API
 * Free memory for array data.
 */
NPY_NO_EXPORT void
PyDataMem_FREE(void *ptr)
{
    allocator_free(&default_allocator, ptr);
}

/*NUMPY_API
 * Reallocate/resize memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_RENEW(void *ptr, size_t size)
{
    return allocator_renew(&default_allocator, ptr, size);
}


/*NUMPY_API
 * Get the current allocator for array data.
 */
NPY_NO_EXPORT const PyDataMem_Allocator *
PyDataMem_GetAllocator(void)
{
    return current_allocator;
}

/*NUMPY_API
 * Set the current allocator for array data.  If the parameter is NULL,
 * the allocator is reset to the default Numpy allocator.
 */
NPY_NO_EXPORT void
PyDataMem_SetAllocator(const PyDataMem_Allocator *new_allocator)
{
    /* We must deallocate all cached data areas before switching allocators */
    _npy_clear_cache(NBUCKETS, datacache, &current_allocator_free);
    if (new_allocator == NULL) {
        new_allocator = &default_allocator;
    }
    current_allocator = new_allocator;
}
