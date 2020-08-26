#ifndef _NPY_MULTIARRAY_EINSUM_SUMPROD_H
#define _NPY_MULTIARRAY_EINSUM_SUMPROD_H

#include "simd/simd.h"
#include "common.h"

#define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "einsum.dispatch.h"
#endif

typedef void (*sum_of_products_fn)(int, char **, npy_intp const*, npy_intp);

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT sum_of_products_fn einsum_get_sum_of_products_function,
    (int nop, int type_num, npy_intp itemsize, npy_intp const *fixed_strides))

#endif
