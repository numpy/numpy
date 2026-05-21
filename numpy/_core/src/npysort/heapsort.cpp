/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include "npysort_heapsort.h"

#include <cstdlib>

#define NOT_USED NPY_UNUSED(unused)
#define PYA_QS_STACK 100
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20
#define SMALL_STRING 16


/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_heapsort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
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
npy_aheapsort(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    char *v = (char *)vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
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
