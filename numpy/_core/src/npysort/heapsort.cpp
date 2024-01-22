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

#include <cstdlib>

#include "heapsort.hpp"

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
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
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
        memcpy(tmp, a + l * elsize, elsize);
        for (i = l, j = l << 1; j <= num;) {
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                memcpy(a + i * elsize, a + j * elsize, elsize);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        memcpy(a + i * elsize, tmp, elsize);
    }

    for (; num > 1;) {
        memcpy(tmp, a + num * elsize, elsize);
        memcpy(a + num * elsize, a + elsize, elsize);
        num -= 1;
        for (i = 1, j = 2; j <= num;) {
            if (j < num &&
                cmp(a + j * elsize, a + (j + 1) * elsize, arr) < 0) {
                ++j;
            }
            if (cmp(tmp, a + j * elsize, arr) < 0) {
                memcpy(a + i * elsize, a + j * elsize, elsize);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        memcpy(a + i * elsize, tmp, elsize);
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
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
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

/***************************************
 * C > C++ dispatch
 ***************************************/

NPY_NO_EXPORT int
heapsort_bool(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Bool *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Byte *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UByte *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Short *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UShort *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Int *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UInt *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Long *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::ULong *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::LongLong *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::ULongLong *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Half *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Float *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Double *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::LongDouble *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CFloat *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CDouble *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CLongDouble *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::DateTime *)start, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::TimeDelta *)start, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Bool *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Byte *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_ubyte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UByte *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_short(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Short *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_ushort(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UShort *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Int *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::UInt *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Long *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_ulong(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::ULong *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::LongLong *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::ULongLong *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Half *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_float(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Float *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_double(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::Double *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::LongDouble *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CFloat *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CDouble *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::CLongDouble *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::DateTime *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
aheapsort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    np::sort::Heap((np::TimeDelta *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
heapsort_string(void *start, npy_intp n, void *varr)
{
    return np::sort::Heap((np::String*)start, n, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
heapsort_unicode(void *start, npy_intp n, void *varr)
{
    return np::sort::Heap((np::Unicode*)start, n, (PyArrayObject *)varr);
}

NPY_NO_EXPORT int
aheapsort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return np::sort::Heap((np::String*)vv, tosort, n, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
aheapsort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return np::sort::Heap((np::Unicode*)vv, tosort, n, (PyArrayObject *)varr);
}
