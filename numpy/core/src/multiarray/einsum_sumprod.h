#ifndef _NPY_MULTIARRAY_EINSUM_SUMPROD_H
#define _NPY_MULTIARRAY_EINSUM_SUMPROD_H
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>
#include <numpy/halffloat.h>
#include <npy_pycompat.h>

#include <ctype.h>

#include "simd/simd.h"
#include "convert.h"
#include "common.h"
#include "ctors.h"

#define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)
#include <numpy/npy_common.h>

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "einsum.dispatch.h"
#endif

typedef void (*sum_of_products_fn)(int, char **, npy_intp const*, npy_intp);

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT sum_of_products_fn einsum_get_sum_of_products_function,
    (int nop, int type_num, npy_intp itemsize, npy_intp const *fixed_strides))

#endif
