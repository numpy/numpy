#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <pymem.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"
#include "npy_config.h"
#include "alloc.h"
#include "npy_static_data.h"
#include "templ_common.h"
#include "multiarraymodule.h"

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

/* Do not enable the alloc cache if the GIL is disabled, or if ASAN or MSAN
 * instrumentation is enabled. The cache makes ASAN use-after-free or MSAN
 * use-of-uninitialized-memory warnings less useful. */
#define USE_ALLOC_CACHE 1
#ifdef Py_GIL_DISABLED
# define USE_ALLOC_CACHE 0
#elif defined(__has_feature)
# if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer)
#  define USE_ALLOC_CACHE 0
# endif
#endif

# define NBUCKETS 1024 /* number of buckets for data*/
# define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
# define NCACHE 7 /* number of cache entries per bucket */
/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];

/*
 * This function tells whether NumPy attempts to call `madvise` with
 * `MADV_HUGEPAGE`.  `madvise` is only ever used on linux, so the value
 * of `madvise_hugepage` may be ignored.
 *
 * It is exposed to Python as `np._core.multiarray._get_madvise_hugepage`.
 */
NPY_NO_EXPORT PyObject *
_get_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args))
{
#ifdef NPY_OS_LINUX
    if (npy_thread_unsafe_state.madvise_hugepage) {
        Py_RETURN_TRUE;
    }
#endif
    Py_RETURN_FALSE;
}


/*
 * This function enables or disables the use of `MADV_HUGEPAGE` on Linux
 * by modifying the global static `madvise_hugepage`.
 * It returns the previous value of `madvise_hugepage`.
 *
 * It is exposed to Python as `np._core.multiarray._set_madvise_hugepage`.
 */
NPY_NO_EXPORT PyObject *
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj)
{
    int was_enabled = npy_thread_unsafe_state.madvise_hugepage;
    int enabled = PyObject_IsTrue(enabled_obj);
    if (enabled < 0) {
        return NULL;
    }
    npy_thread_unsafe_state.madvise_hugepage = enabled;
    if (was_enabled) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}


NPY_FINLINE void
indicate_hugepages(void *p, size_t size) {
#ifdef NPY_OS_LINUX
    /* allow kernel allocating huge pages for large arrays */
    if (NPY_UNLIKELY(size >= ((1u<<22u))) &&
        npy_thread_unsafe_state.madvise_hugepage) {
        npy_uintp offset = 4096u - (npy_uintp)p % (4096u);
        npy_uintp length = size - offset;
        /**
         * Intentionally not checking for errors that may be returned by
         * older kernel versions; optimistically tries enabling huge pages.
         */
        madvise((void*)((npy_uintp)p + offset), length, MADV_HUGEPAGE);
    }
#endif
}


/* as the cache is managed in global variables verify the GIL is held */

/*
 * very simplistic small memory block cache to avoid more expensive libc
 * allocations
 * base function for data cache with 1 byte buckets and dimension cache with
 * sizeof(npy_intp) byte buckets
 */
static inline void *
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t))
{
    void * p;
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    assert(PyGILState_Check());
#if USE_ALLOC_CACHE
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
#endif
    p = alloc(nelem * esz);
    if (p) {
#ifdef _PyPyGC_AddMemoryPressure
        _PyPyPyGC_AddMemoryPressure(nelem * esz);
#endif
        indicate_hugepages(p, nelem * esz);
    }
    return p;
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static inline void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    assert(PyGILState_Check());
#if USE_ALLOC_CACHE
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
#endif
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
npy_alloc_cache_zero(size_t nmemb, size_t size)
{
    void * p;
    size_t sz = nmemb * size;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &PyDataMem_NEW);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = PyDataMem_NEW_ZEROED(nmemb, size);
    NPY_END_THREADS;
    if (p) {
        indicate_hugepages(p, sz);
    }
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

/* Similar to array_dealloc in arrayobject.c */
static inline void
WARN_NO_RETURN(PyObject* warning, const char * msg) {
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        PyObject * s;

        s = PyUnicode_FromString("PyDataMem_UserFREE");
        if (s) {
            PyErr_WriteUnraisable(s);
            Py_DECREF(s);
        }
        else {
            PyErr_WriteUnraisable(Py_None);
        }
    }
}


/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW(size_t size)
{
    void *result;

    assert(size != 0);
    result = malloc(size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (ret == -1) {
        free(result);
        return NULL;
    }
    return result;
}

/*NUMPY_API
 * Allocates zeroed memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_NEW_ZEROED(size_t nmemb, size_t size)
{
    void *result;

    result = calloc(nmemb, size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);
    if (ret == -1) {
        free(result);
        return NULL;
    }
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
}

/*NUMPY_API
 * Reallocate/resize memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMem_RENEW(void *ptr, size_t size)
{
    void *result;

    assert(size != 0);
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    result = realloc(ptr, size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (ret == -1) {
        free(result);
        return NULL;
    }
    return result;
}

// The default data mem allocator malloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_malloc(void *NPY_UNUSED(ctx), size_t size)
{
    return _npy_alloc_cache(size, 1, NBUCKETS, datacache, &malloc);
}

// The default data mem allocator calloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserNEW_ZEROED
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_calloc(void *NPY_UNUSED(ctx), size_t nelem, size_t elsize)
{
    void * p;
    size_t sz = nelem * elsize;
    NPY_BEGIN_THREADS_DEF;
    if (sz < NBUCKETS) {
        p = _npy_alloc_cache(sz, 1, NBUCKETS, datacache, &malloc);
        if (p) {
            memset(p, 0, sz);
        }
        return p;
    }
    NPY_BEGIN_THREADS;
    p = calloc(nelem, elsize);
    if (p) {
        indicate_hugepages(p, sz);
    }
    NPY_END_THREADS;
    return p;
}

// The default data mem allocator realloc routine does not make use of a ctx.
// It should be called only through PyDataMem_UserRENEW
// since itself does not handle eventhook and tracemalloc logic.
static inline void *
default_realloc(void *NPY_UNUSED(ctx), void *ptr, size_t new_size)
{
    return realloc(ptr, new_size);
}

// The default data mem allocator free routine does not make use of a ctx.
// It should be called only through PyDataMem_UserFREE
// since itself does not handle eventhook and tracemalloc logic.
static inline void
default_free(void *NPY_UNUSED(ctx), void *ptr, size_t size)
{
    _npy_free_cache(ptr, size, NBUCKETS, datacache, &free);
}

/* Memory handler global default */
PyDataMem_Handler default_handler = {
    "default_allocator",
    1,
    {
        NULL,            /* ctx */
        default_malloc,  /* malloc */
        default_calloc,  /* calloc */
        default_realloc, /* realloc */
        default_free     /* free */
    }
};
/* singleton capsule of the default handler */
PyObject *PyDataMem_DefaultHandler;
PyObject *current_handler;

int uo_index=0;   /* user_override index */

/* Wrappers for the default or any user-assigned PyDataMem_Handler */

NPY_NO_EXPORT void *
PyDataMem_UserNEW(size_t size, PyObject *mem_handler)
{
    void *result;
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;
    }
    assert(size != 0);
    result = handler->allocator.malloc(handler->allocator.ctx, size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (ret == -1) {
        handler->allocator.free(handler->allocator.ctx, result, size);
        return NULL;
    }
    return result;
}

NPY_NO_EXPORT void *
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyObject *mem_handler)
{
    void *result;
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;
    }
    result = handler->allocator.calloc(handler->allocator.ctx, nmemb, size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, nmemb * size);
    if (ret == -1) {
        handler->allocator.free(handler->allocator.ctx, result, size);
        return NULL;
    }
    return result;
}


NPY_NO_EXPORT void
PyDataMem_UserFREE(void *ptr, size_t size, PyObject *mem_handler)
{
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        WARN_NO_RETURN(PyExc_RuntimeWarning,
                     "Could not get pointer to 'mem_handler' from PyCapsule");
        return;
    }
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    handler->allocator.free(handler->allocator.ctx, ptr, size);
}

NPY_NO_EXPORT void *
PyDataMem_UserRENEW(void *ptr, size_t size, PyObject *mem_handler)
{
    void *result;
    PyDataMem_Handler *handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        return NULL;
    }

    assert(size != 0);
    PyTraceMalloc_Untrack(NPY_TRACE_DOMAIN, (npy_uintp)ptr);
    result = handler->allocator.realloc(handler->allocator.ctx, ptr, size);
    int ret = PyTraceMalloc_Track(NPY_TRACE_DOMAIN, (npy_uintp)result, size);
    if (ret == -1) {
        handler->allocator.free(handler->allocator.ctx, result, size);
        return NULL;
    }
    return result;
}

/*NUMPY_API
 * Set a new allocation policy. If the input value is NULL, will reset
 * the policy to the default. Return the previous policy, or
 * return NULL if an error has occurred. We wrap the user-provided
 * functions so they will still call the python and numpy
 * memory management callback hooks.
 */
NPY_NO_EXPORT PyObject *
PyDataMem_SetHandler(PyObject *handler)
{
    PyObject *old_handler;
    PyObject *token;
    if (PyContextVar_Get(current_handler, NULL, &old_handler)) {
        return NULL;
    }
    if (handler == NULL) {
        handler = PyDataMem_DefaultHandler;
    }
    if (!PyCapsule_IsValid(handler, MEM_HANDLER_CAPSULE_NAME)) {
        PyErr_SetString(PyExc_ValueError, "Capsule must be named 'mem_handler'");
        return NULL;
    }
    token = PyContextVar_Set(current_handler, handler);
    if (token == NULL) {
        Py_DECREF(old_handler);
        return NULL;
    }
    Py_DECREF(token);
    return old_handler;
}

/*NUMPY_API
 * Return the policy that will be used to allocate data
 * for the next PyArrayObject. On failure, return NULL.
 */
NPY_NO_EXPORT PyObject *
PyDataMem_GetHandler()
{
    PyObject *handler;
    if (PyContextVar_Get(current_handler, NULL, &handler)) {
        return NULL;
    }
    return handler;
}

NPY_NO_EXPORT PyObject *
get_handler_name(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *arr=NULL;
    if (!PyArg_ParseTuple(args, "|O:get_handler_name", &arr)) {
        return NULL;
    }
    if (arr != NULL && !PyArray_Check(arr)) {
         PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
         return NULL;
    }
    PyObject *mem_handler;
    PyDataMem_Handler *handler;
    PyObject *name;
    if (arr != NULL) {
        mem_handler = PyArray_HANDLER((PyArrayObject *) arr);
        if (mem_handler == NULL) {
            Py_RETURN_NONE;
        }
        Py_INCREF(mem_handler);
    }
    else {
        mem_handler = PyDataMem_GetHandler();
        if (mem_handler == NULL) {
            return NULL;
        }
    }
    handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    name = PyUnicode_FromString(handler->name);
    Py_DECREF(mem_handler);
    return name;
}

NPY_NO_EXPORT PyObject *
get_handler_version(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *arr=NULL;
    if (!PyArg_ParseTuple(args, "|O:get_handler_version", &arr)) {
        return NULL;
    }
    if (arr != NULL && !PyArray_Check(arr)) {
         PyErr_SetString(PyExc_ValueError, "if supplied, argument must be an ndarray");
         return NULL;
    }
    PyObject *mem_handler;
    PyDataMem_Handler *handler;
    PyObject *version;
    if (arr != NULL) {
        mem_handler = PyArray_HANDLER((PyArrayObject *) arr);
        if (mem_handler == NULL) {
            Py_RETURN_NONE;
        }
        Py_INCREF(mem_handler);
    }
    else {
        mem_handler = PyDataMem_GetHandler();
        if (mem_handler == NULL) {
            return NULL;
        }
    }
    handler = (PyDataMem_Handler *) PyCapsule_GetPointer(
            mem_handler, MEM_HANDLER_CAPSULE_NAME);
    if (handler == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    version = PyLong_FromLong(handler->version);
    Py_DECREF(mem_handler);
    return version;
}


/*
 * Internal function to malloc, but add an overflow check similar to Calloc
 */
NPY_NO_EXPORT void *
_Npy_MallocWithOverflowCheck(npy_intp size, npy_intp elsize)
{
    npy_intp total_size;
    if (npy_mul_sizes_with_overflow(&total_size, size, elsize)) {
        return NULL;
    }
    return PyMem_MALLOC(total_size);
}
