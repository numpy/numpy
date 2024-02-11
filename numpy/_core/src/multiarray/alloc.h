#ifndef NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#define NPY_TRACE_DOMAIN 389047

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

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_ */
