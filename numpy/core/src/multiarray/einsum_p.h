#ifndef _NPY_EINSUM_P_H_
#define _NPY_EINSUM_P_H_

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

#define EINSUM_IS_SSE_ALIGNED(x) ((((npy_intp)x)&0xf) == 0)

/********** PRINTF DEBUG TRACING **************/
#define NPY_EINSUM_DBG_TRACING 0

#if NPY_EINSUM_DBG_TRACING
#define NPY_EINSUM_DBG_PRINT(s) printf("%s", s);
#define NPY_EINSUM_DBG_PRINT1(s, p1) printf(s, p1);
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2) printf(s, p1, p2);
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3) printf(s);
#else
#define NPY_EINSUM_DBG_PRINT(s)
#define NPY_EINSUM_DBG_PRINT1(s, p1)
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2)
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3)
#endif

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "einsum.dispatch.h"
#endif

typedef void (*sum_of_products_fn)(int, char **, npy_intp const*, npy_intp);

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT sum_of_products_fn einsum_get_sum_of_products_function,
    (int nop, int type_num, npy_intp itemsize,npy_intp const *fixed_strides))

#endif // _NPY_EINSUM_P_H_
