#ifndef NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#define NPY_TRACE_DOMAIN 389047
#define MEM_HANDLER_CAPSULE_NAME "mem_handler"

NPY_NO_EXPORT PyObject *
_get_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args));

NPY_NO_EXPORT PyObject *
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj);

NPY_NO_EXPORT void *
PyDataMem_UserNEW(npy_uintp sz, PyObject *mem_handler);

NPY_NO_EXPORT void *
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyObject *mem_handler);

NPY_NO_EXPORT void
PyDataMem_UserFREE(void * p, npy_uintp sd, PyObject *mem_handler);

NPY_NO_EXPORT void *
PyDataMem_UserRENEW(void *ptr, size_t size, PyObject *mem_handler);

NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz);

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sd);

static inline void
npy_free_cache_dim_obj(PyArray_Dims dims)
{
    npy_free_cache_dim(dims.ptr, dims.len);
}

static inline void
npy_free_cache_dim_array(PyArrayObject * arr)
{
    npy_free_cache_dim(PyArray_DIMS(arr), PyArray_NDIM(arr));
}

extern PyDataMem_Handler default_handler;
extern PyObject *current_handler; /* PyContextVar/PyCapsule */

NPY_NO_EXPORT PyObject *
get_handler_name(PyObject *NPY_UNUSED(self), PyObject *obj);
NPY_NO_EXPORT PyObject *
get_handler_version(PyObject *NPY_UNUSED(self), PyObject *obj);

/* Helper to add an overflow check (and avoid inlininig this probably) */
NPY_NO_EXPORT void *
_Npy_MallocWithOverflowCheck(npy_intp size, npy_intp elsize);


static inline void
_npy_init_workspace(
    void **buf, void *static_buf, size_t static_buf_size, size_t elsize, size_t size)
{
    if (NPY_LIKELY(size <= static_buf_size)) {
        *buf = static_buf;
    }
    else {
        *buf = _Npy_MallocWithOverflowCheck(size, elsize);
        if (*buf == NULL) {
            PyErr_NoMemory();
        }
    }
}


/*
 * Helper definition macro for a small work/scratchspace.
 * The `NAME` is the C array to to be defined of with the type `TYPE`.
 *
 * The usage pattern for this is:
 *
 *     NPY_ALLOC_WORKSPACE(arr, PyObject *, 14, n_objects);
 *     if (arr == NULL) {
 *         return -1;  // Memory error is set
 *     }
 *     ...
 *     npy_free_workspace(arr);
 *
 * Notes
 * -----
 * The reason is to avoid allocations in most cases, but gracefully
 * succeed for large sizes as well.
 * With some caches, it may be possible to malloc/calloc very quickly in which
 * case we should not hesitate to replace this pattern.
 */
#define NPY_ALLOC_WORKSPACE(NAME, TYPE, fixed_size, size)  \
    TYPE NAME##_static[fixed_size];                        \
    TYPE *NAME;                                            \
    _npy_init_workspace((void **)&NAME, NAME##_static, (fixed_size), sizeof(TYPE), (size))


static inline void
_npy_free_workspace(void *buf, void *static_buf)
{
    if (buf != static_buf) {
        PyMem_FREE(buf);
    }
}

/* Free a small workspace allocation (macro to fetch the _static name) */
#define npy_free_workspace(NAME)  \
    _npy_free_workspace(NAME, NAME##_static)

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_ */
