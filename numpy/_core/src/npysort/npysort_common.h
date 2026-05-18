#ifndef NUMPY_CORE_SRC_NPYSORT_NPYSORT_COMMON_H_
#define NUMPY_CORE_SRC_NPYSORT_NPYSORT_COMMON_H_

#include <numpy/ndarraytypes.h>
#include "dtypemeta.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared helpers used by the per-dtype sort implementations.  Per-dtype
 * less-than comparisons live on the tags in ``numpy_tag.h``; this header
 * only carries the small handful of helpers used by the generic (cmp-
 * function-driven) sort code and by the argsort variants.
 */

/* Argsort works on indices, swap macro for npy_intp. */
#define INTP_SWAP(a, b) {npy_intp tmp = (b); (b) = (a); (a) = tmp;}

static inline void
get_sort_data_from_array(void *varr, npy_intp *elsize, PyArray_CompareFunc **cmp)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    *elsize = PyArray_ITEMSIZE(arr);
    *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
}

/* Element copy / swap for the generic, comparison-function-driven sort. */
static inline void
GENERIC_COPY(char *a, char *b, size_t len)
{
    memcpy(a, b, len);
}

static inline void
GENERIC_SWAP(char *a, char *b, size_t len)
{
    while (len--) {
        const char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_NPYSORT_NPYSORT_COMMON_H_ */
