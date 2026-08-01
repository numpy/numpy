#ifndef NUMPY_CORE_SRC_MULTIARRAY_MASKED_COMPRESS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MASKED_COMPRESS_H_

#include "npy_cpu_dispatch.h"
#include "numpy/ndarraytypes.h"
#include "masked_compress.dispatch.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT size_t npy_count_nonzero_mask,
                        (const unsigned char *mask, size_t n))

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT size_t npy_masked_compress,
                        (void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize))

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT size_t npy_masked_expand,
                        (void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize))
#ifdef __cplusplus
}
#endif
#endif
