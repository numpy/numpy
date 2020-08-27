#ifndef _NPY_MULTIARRAY_EINSUM_SUMPROD_H
#define _NPY_MULTIARRAY_EINSUM_SUMPROD_H

#include "npy_cpu_dispatch.h"
#include <numpy/npy_common.h>

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "einsum.dispatch.h"
#endif

typedef void (*sum_of_products_fn)(int, char **, npy_intp const*, npy_intp);

NPY_CPU_DISPATCH_DECLARE(NPY_VISIBILITY_HIDDEN sum_of_products_fn einsum_get_sum_of_products_function,
    (int nop, int type_num, npy_intp itemsize, npy_intp const *fixed_strides))

#endif
