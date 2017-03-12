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
#define PyTraceMalloc_Track(...) -2
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
#include <stdio.h>

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>

#if defined MADV_FREE && defined HAVE_MADVISE
/* mark memory reclaimable by other processes */
#define MADVISE_FREE(ptr, size)\
    madvise((void*)((npy_uintp)ptr + (4096 - ((npy_uintp)ptr % 4096))), \
            size, MADV_FREE)
#define NPY_ENABLE_LARGECACHE
#include <unistd.h>
#endif
#endif
#ifndef MADVISE_FREE
#define MADVISE_FREE(ptr, size) 1
#endif

/*
 * Large memory allocation cache.
 * Large allocations on glibc systems need to be faulted into the process
 * before use each use. This can account for a significant fraction of the
 * runtime of frequent array allocation and deallocation.  This can be avoidded
 * by caching used memory blocks inside numpy for later reuse.  To not waste
 * the memory in the cache we mark the memory as reclaimable by other processes
 * (or numpy itself) with MADV_FREE. Should we try to use memory claimed by
 * other processes we will receive newly faulted memory instead just as if it
 * were freed normally.
 * MADV_FREE is supported on Linux >= 4.5 and BSD.
 *
 * The cache is intentionally very simple queue storing the last large
 * deallocations in a queue. New allocations pick the first block of memory
 * that matches in size from the cache. It does not pick larger blocks, while
 * the memory can be reclaimed by other processes it is not available for the
 * page cache used to buffer IO.
 * On each deallocation the cache is aged and if no entries have been removed
 * for too long the oldest memory block is properly deallocated to avoid that
 * large unused blocks stay in the cache too long.
 * In most cases where it is relevant are fast allocation/deallocation cycles
 * that appear in operations with temporaries or indexing. A simple small cache
 * is sufficient for this.
 * It will cache at most half of the available physical memory so that there is
 * always enough free to be used in the page cache.
 */

/* threshold in bytes for large allocation to be considered for the cache */
#define NPY_LARGE_ALLOC 4096 * 128
#define LCACHE_MAXENTRIES 50
typedef struct {
    npy_uintp size;
    void * ptr;
} lcache_bucket;
static unsigned int lcache_entries = 0;
#ifdef NPY_ENABLE_LARGECACHE
static unsigned int lcache_nentries = 16;
#else
static unsigned int lcache_nentries = 0;
#endif
static npy_uintp lcache_size = 0;
static npy_uintp lcache_maxsize = -1;
static lcache_bucket lcache[LCACHE_MAXENTRIES];
static unsigned int lcache_age = 0;

static void remove_lcache_entry(npy_uintp idx, void (*dealloc)(void *))
{
    dealloc(lcache[idx].ptr);
    lcache_size -= lcache[idx].size;
    lcache_entries--;
    lcache[idx].ptr = NULL;
    lcache[idx].size = 0;
    lcache_age = 0;
}

/* set the large allocation cache max size, 0 disables and clears it */
NPY_NO_EXPORT unsigned int
npy_set_lcache_size(unsigned int size)
{
#ifdef NPY_ENABLE_LARGECACHE
    unsigned int i, old_size = lcache_size;
    void * ptr;
    size = size > LCACHE_MAXENTRIES ? LCACHE_MAXENTRIES : size;

    /* disable cache if MADV_FREE does not work */
    ptr = malloc(4096 * 10);
    if (MADVISE_FREE(ptr, 4096) != 0) {
        size = 0;
    }
    free(ptr);

    /* not enough address space on 32 bit to cache large data */
    if (sizeof(ptr) < 8) {
        size = 0;
    }

    /* disable caching when kernel has overcommit disabled */
    {
        FILE * f = fopen("/proc/sys/vm/overcommit_memory", "r");
        if (f == NULL) {
            size = 0;
        }
        else {
            if (fgetc(f) == '2') {
                size = 0;
            }
            fclose(f);
        }
    }

    /*
     * cache at most half the available ram so there is enough left for the
     * page cache used to buffer IO
     */
    {
        npy_intp pages = sysconf(_SC_PHYS_PAGES);
        npy_intp page_size = sysconf(_SC_PAGE_SIZE);
        lcache_maxsize = (pages * page_size / 2);
    }

    for (i=size; i < lcache_size; i++) {
        PyDataMem_FREE(lcache[i].ptr);
        lcache_size -= lcache[i].size;
        lcache[i].ptr = NULL;
        lcache[i].size = 0;
    }
    lcache_nentries = size;
    return old_size;
#else
    lcache_size = 0;
    return 0;
#endif
}

/* fast cache toggle for disabling during tracing */
static void enable_lcache(int enable)
{
    static unsigned int old_lcache_nentries;
    if (!enable) {
        unsigned int i;
        for (i=0; i < lcache_size; i++) {
            PyDataMem_FREE(lcache[i].ptr);
            lcache_size -= lcache[i].size;
            lcache[i].ptr = NULL;
            lcache[i].size = 0;
        }
        if (lcache_nentries != 0) {
            old_lcache_nentries = lcache_nentries;
        }
        lcache_nentries = 0;
    }
    else {
        if (old_lcache_nentries != 0) {
            lcache_nentries = old_lcache_nentries;
        }
    }

}


/* place a block into the cache, marking it as reclaimable */
static void add_to_lcache(void * ptr, npy_uintp size, void (*dealloc)(void *))
{
    if (lcache_size + size > lcache_maxsize || lcache_nentries == 0) {
        dealloc(ptr);
        return;
    }
    if (MADVISE_FREE(ptr, size) == 0) {
        if (lcache_entries == lcache_nentries) {
            remove_lcache_entry(lcache_entries - 1, dealloc);
        }
        if (lcache_entries > 0) {
            memmove(lcache + 1, lcache, sizeof(lcache[0]) * lcache_entries);
        }
        lcache[0].ptr = ptr;
        lcache[0].size = size;
        lcache_size += size;
        lcache_entries++;
        return;
    }
    else {
        dealloc(ptr);
    }
}


/* retrieve a memory block from the cache or allocate if empty */
static void * remove_from_lcache(npy_uintp size, void * (*alloc)(size_t))
{
    npy_uintp i;

    /* find a matching entry */
    for (i=0; i < lcache_entries; i++) {
        if (lcache[i].size == size) {
            void * ptr = lcache[i].ptr;
            lcache_size -= lcache[i].size;
            if (i < lcache_entries) {
                memmove(&lcache[i], &lcache[i] + 1,
                        sizeof(lcache[0]) * (lcache_entries - i - 1));
            }
            lcache_entries--;
            /* successful retrieve, reset age as more successes may follow */
            lcache_age = 0;
            return ptr;
        }
    }
    return alloc(size);
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
    else if (nelem > NPY_LARGE_ALLOC) {
        return remove_from_lcache(nelem * esz, alloc);
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
    /* remove stale entries from the large allocation cache */
    lcache_age++;
    if (NPY_UNLIKELY(lcache_age > 2000) && lcache_entries > 0) {
        remove_lcache_entry(lcache_entries - 1, dealloc);
    }
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    else if (p != NULL && nelem > NPY_LARGE_ALLOC) {
        add_to_lcache(p, nelem, dealloc);
        return;
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
    enable_lcache(newhook == NULL);
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
    int tracking;

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
    tracking = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    enable_lcache(tracking == -2 && _PyDataMem_eventhook == NULL);
#ifdef _PyPyGC_AddMemoryPressure
    if (result) {
        _PyPyPyGC_AddMemoryPressure(size);
    }
#endif
    return result;
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
{
    void *result;
    int tracking;

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
    tracking = PyTraceMalloc_Track(NPY_TRACE_DOMAIN,
                                   (npy_uintp)result, elsize * size);
    enable_lcache(tracking == -2 && _PyDataMem_eventhook == NULL);
#ifdef _PyPyGC_AddMemoryPressure
    if (result) {
        _PyPyPyGC_AddMemoryPressure(elsize * size);
    }
#endif
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
    unsigned int tracking;

    result = realloc(ptr, size);
    if (result != ptr) {
        PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    }
    tracking = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    enable_lcache(tracking == -2 && _PyDataMem_eventhook == NULL);
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
