#ifndef NUMPY_CORE_SRC_MULTIARRAY_MASKED_COMPRESS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MASKED_COMPRESS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
size_t npy_count_nonzero_mask(const unsigned char *mask, size_t n);
size_t npy_masked_compress(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize);
size_t npy_masked_expand(void *dst, const void *src, const unsigned char *mask, size_t n, size_t elsize);
#ifdef __cplusplus
}
#endif
#endif
