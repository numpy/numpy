#ifndef __NPY_SORT_HEAPSORT_H__
#define __NPY_SORT_HEAPSORT_H__

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <cstdlib>

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
inline NPY_NO_EXPORT
int heapsort_(type *start, npy_intp n)
{
    type tmp, *a;
    npy_intp i, j, l;

    /* The array needs to be offset by one for heapsort indexing */
    a = start - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && Tag::less(a[j], a[j + 1])) {
                j += 1;
            }
            if (Tag::less(tmp, a[j])) {
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
            if (j < n && Tag::less(a[j], a[j + 1])) {
                j++;
            }
            if (Tag::less(tmp, a[j])) {
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

template <typename Tag, typename type>
inline NPY_NO_EXPORT
int aheapsort_(type *vv, npy_intp *tosort, npy_intp n)
{
    type *v = vv;
    npy_intp *a, i, j, l, tmp;
    /* The arrays need to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && Tag::less(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (Tag::less(v[tmp], v[a[j]])) {
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
            if (j < n && Tag::less(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (Tag::less(v[tmp], v[a[j]])) {
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

/*
 *****************************************************************************
 **                            STRING SORTS                                 **
 *****************************************************************************
 */

template <typename Tag, typename type>
inline NPY_NO_EXPORT
int string_heapsort_(type *start, npy_intp n, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    if (len == 0) {
        return 0;  /* no need for sorting if strings are empty */
    }

    type *tmp = (type *)malloc(PyArray_ITEMSIZE(arr));
    type *a = (type *)start - len;
    npy_intp i, j, l;

    if (tmp == NULL) {
        return -NPY_ENOMEM;
    }

    for (l = n >> 1; l > 0; --l) {
        Tag::copy(tmp, a + l * len, len);
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && Tag::less(a + j * len, a + (j + 1) * len, len))
                j += 1;
            if (Tag::less(tmp, a + j * len, len)) {
                Tag::copy(a + i * len, a + j * len, len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        Tag::copy(a + i * len, tmp, len);
    }

    for (; n > 1;) {
        Tag::copy(tmp, a + n * len, len);
        Tag::copy(a + n * len, a + len, len);
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && Tag::less(a + j * len, a + (j + 1) * len, len))
                j++;
            if (Tag::less(tmp, a + j * len, len)) {
                Tag::copy(a + i * len, a + j * len, len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        Tag::copy(a + i * len, tmp, len);
    }

    free(tmp);
    return 0;
}

template <typename Tag, typename type>
inline NPY_NO_EXPORT
int string_aheapsort_(type *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    type *v = vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    npy_intp *a, i, j, l, tmp;

    /* The array needs to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && Tag::less(v + a[j] * len, v + a[j + 1] * len, len))
                j += 1;
            if (Tag::less(v + tmp * len, v + a[j] * len, len)) {
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
            if (j < n && Tag::less(v + a[j] * len, v + a[j + 1] * len, len))
                j++;
            if (Tag::less(v + tmp * len, v + a[j] * len, len)) {
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

#endif
