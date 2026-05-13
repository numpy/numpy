/* -*- c -*- */

/*
 * Comparator-function-driven version of the heapsort implemented in
 * ``npysort_heapsort.hpp``.  Used by dtypes that register a
 * ``PyArray_CompareFunc`` rather than the type-specialised path.
 * See ``npysort_heapsort.hpp`` for the algorithm description.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"

#include <cstdlib>

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_heapsort_impl(void *start, npy_intp num, void *varr, npy_intp elsize,
                  PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    if (elsize == 0) {
        return 0;  /* no need for sorting elements of no size */
    }
    char *tmp = (char *)malloc(elsize);
    char *a = (char *)start - elsize;
    npy_intp i, j, l;

    if (tmp == NULL) {
        return -NPY_ENOMEM;
    }

    for (l = num >> 1; l > 0; --l) {
        GENERIC_COPY(tmp, a + l * elsize, elsize);
        for (i = l, j = l << 1; j <= num;) {
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                GENERIC_COPY(a + i * elsize, a + j * elsize, elsize);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        GENERIC_COPY(a + i * elsize, tmp, elsize);
    }

    for (; num > 1;) {
        GENERIC_COPY(tmp, a + num * elsize, elsize);
        GENERIC_COPY(a + num * elsize, a + elsize, elsize);
        num -= 1;
        for (i = 1, j = 2; j <= num;) {
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                GENERIC_COPY(a + i * elsize, a + j * elsize, elsize);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        GENERIC_COPY(a + i * elsize, tmp, elsize);
    }

    free(tmp);
    return 0;
}

NPY_NO_EXPORT int
npy_aheapsort_impl(void *vv, npy_intp *tosort, npy_intp n, void *varr,
                   npy_intp elsize, PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    char *v = (char *)vv;
    npy_intp *a, i, j, l, tmp;

    /* The array needs to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n &&
                cmp(v + a[j] * elsize, v + a[j + 1] * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(v + tmp * elsize, v + a[j] * elsize, arr) < 0) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n &&
                cmp(v + a[j] * elsize, v + a[j + 1] * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(v + tmp * elsize, v + a[j] * elsize, arr) < 0) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}
