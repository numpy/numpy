#ifndef _NPY_ARRAY_ALLOC_H_
#define _NPY_ARRAY_ALLOC_H_
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

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

#endif
