#ifndef _NPY_ARRAY_ALLOC_H_
#define _NPY_ARRAY_ALLOC_H_
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#define NPY_TRACE_DOMAIN 389047

NPY_NO_EXPORT void *
npy_alloc_cache(npy_uintp sz);

NPY_NO_EXPORT void *
npy_alloc_cache_zero(npy_uintp sz);

NPY_NO_EXPORT void
npy_free_cache(void * p, npy_uintp sd);

NPY_NO_EXPORT void *
npy_alloc_cache_dim(npy_uintp sz);

NPY_NO_EXPORT void
npy_free_cache_dim(void * p, npy_uintp sd);

static NPY_INLINE void
npy_free_cache_dim_obj(PyArray_Dims dims)
{
    npy_free_cache_dim(dims.ptr, dims.len);
}

static NPY_INLINE void
npy_free_cache_dim_array(PyArrayObject * arr)
{
    npy_free_cache_dim(PyArray_DIMS(arr), PyArray_NDIM(arr));
}

#endif
