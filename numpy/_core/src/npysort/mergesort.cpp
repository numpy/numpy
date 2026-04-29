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

#include <cstdlib>

#define NOT_USED NPY_UNUSED(unused)
#define PYA_QS_STACK 100
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20
#define SMALL_STRING 16

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type, bool reverse>
static void
mergesort0_(type *pl, type *pr, type *pw)
{
    type vp, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        mergesort0_<Tag, type, reverse>(pl, pm, pw);
        mergesort0_<Tag, type, reverse>(pm, pr, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::template cmp<reverse>(*pm, *pj)) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::template cmp<reverse>(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}

template <typename Tag, typename type, bool reverse = false>
NPY_NO_EXPORT int
mergesort_(type *start, npy_intp num)
{
    type *pl, *pr, *pw;

    pl = start;
    pr = pl + num;
    pw = (type *)malloc((num / 2) * sizeof(type));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    mergesort0_<Tag, type, reverse>(pl, pr, pw);

    free(pw);
    return 0;
}

template <typename Tag, typename type, bool reverse>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw)
{
    type vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_<Tag, type, reverse>(pl, pm, v, pw);
        amergesort0_<Tag, type, reverse>(pm, pr, v, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::template cmp<reverse>(v[*pm], v[*pj])) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::template cmp<reverse>(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename Tag, typename type, bool reverse = false>
NPY_NO_EXPORT int
amergesort_(type *v, npy_intp *tosort, npy_intp num)
{
    npy_intp *pl, *pr, *pw;

    pl = tosort;
    pr = pl + num;
    pw = (npy_intp *)malloc((num / 2) * sizeof(npy_intp));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    amergesort0_<Tag, type, reverse>(pl, pr, v, pw);
    free(pw);

    return 0;
}

/*
 
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type, bool reverse>
static void
mergesort0_(type *pl, type *pr, type *pw, type *vp, size_t len)
{
    type *pi, *pj, *pk, *pm;

    if ((size_t)(pr - pl) > SMALL_MERGESORT * len) {
        /* merge sort */
        pm = pl + (((pr - pl) / len) >> 1) * len;
        mergesort0_<Tag, type, reverse>(pl, pm, pw, vp, len);
        mergesort0_<Tag, type, reverse>(pm, pr, pw, vp, len);
        Tag::copy(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::template cmp<reverse>(pm, pj, len)) {
                Tag::copy(pk, pm, len);
                pm += len;
                pk += len;
            }
            else {
                Tag::copy(pk, pj, len);
                pj += len;
                pk += len;
            }
        }
        Tag::copy(pk, pj, pi - pj);
    }
    else {
        /* insertion sort */
        for (pi = pl + len; pi < pr; pi += len) {
            Tag::copy(vp, pi, len);
            pj = pi;
            pk = pi - len;
            while (pj > pl && Tag::template cmp<reverse>(vp, pk, len)) {
                Tag::copy(pj, pk, len);
                pj -= len;
                pk -= len;
            }
            Tag::copy(pj, vp, len);
        }
    }
}

template <typename Tag, typename type, bool reverse = false>
static int
string_mergesort_(type *start, npy_intp num, int elsize)
{
    size_t len = elsize / sizeof(type);
    type *pl, *pr, *pw, *vp;
    int err = 0;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pl = start;
    pr = pl + num * len;
    pw = (type *)malloc((num / 2) * elsize);
    if (pw == NULL) {
        err = -NPY_ENOMEM;
        goto fail_0;
    }
    vp = (type *)malloc(elsize);
    if (vp == NULL) {
        err = -NPY_ENOMEM;
        goto fail_1;
    }
    mergesort0_<Tag, type, reverse>(pl, pr, pw, vp, len);

    free(vp);
fail_1:
    free(pw);
fail_0:
    return err;
}

template <typename Tag, typename type, bool reverse>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw, size_t len)
{
    type *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_<Tag, type, reverse>(pl, pm, v, pw, len);
        amergesort0_<Tag, type, reverse>(pm, pr, v, pw, len);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::template cmp<reverse>(v + (*pm) * len, v + (*pj) * len, len)) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v + vi * len;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::template cmp<reverse>(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename Tag, typename type, bool reverse = false>
static int
string_amergesort_(type *v, npy_intp *tosort, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t elsize = PyArray_ITEMSIZE(arr);
    size_t len = elsize / sizeof(type);
    npy_intp *pl, *pr, *pw;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pl = tosort;
    pr = pl + num;
    pw = (npy_intp *)malloc((num / 2) * sizeof(npy_intp));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    amergesort0_<Tag, type, reverse>(pl, pr, v, pw, len);
    free(pw);

    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

static void
npy_mergesort0(char *pl, char *pr, char *pw, char *vp, npy_intp elsize,
               PyArray_CompareFunc *cmp, void *arr)
{
    char *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT * elsize) {
        /* merge sort */
        pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
        npy_mergesort0(pl, pm, pw, vp, elsize, cmp, arr);
        npy_mergesort0(pm, pr, pw, vp, elsize, cmp, arr);
        GENERIC_COPY(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (cmp(pm, pj, arr) < 0) {
                GENERIC_COPY(pk, pm, elsize);
                pm += elsize;
                pk += elsize;
            }
            else {
                GENERIC_COPY(pk, pj, elsize);
                pj += elsize;
                pk += elsize;
            }
        }
        GENERIC_COPY(pk, pj, pi - pj);
    }
    else {
        /* insertion sort */
        for (pi = pl + elsize; pi < pr; pi += elsize) {
            GENERIC_COPY(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                GENERIC_COPY(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            GENERIC_COPY(pj, vp, elsize);
        }
    }
}

NPY_NO_EXPORT int
npy_mergesort(void *start, npy_intp num, void *varr)
{
    void *arr = varr;
    npy_intp elsize;
    PyArray_CompareFunc *cmp;
    get_sort_data_from_array(arr, &elsize, &cmp);

    return npy_mergesort_impl(start, num, varr, elsize, cmp);
}

NPY_NO_EXPORT int
npy_mergesort_impl(void *start, npy_intp num, void *varr,
                   npy_intp elsize, PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    char *pl = (char *)start;
    char *pr = pl + num * elsize;
    char *pw;
    char *vp;
    int err = -NPY_ENOMEM;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pw = (char *)malloc((num >> 1) * elsize);
    vp = (char *)malloc(elsize);

    if (pw != NULL && vp != NULL) {
        npy_mergesort0(pl, pr, pw, vp, elsize, cmp, arr);
        err = 0;
    }

    free(vp);
    free(pw);

    return err;
}

static void
npy_amergesort0(npy_intp *pl, npy_intp *pr, char *v, npy_intp *pw,
                npy_intp elsize, PyArray_CompareFunc *cmp, void *arr)
{
    char *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        npy_amergesort0(pl, pm, v, pw, elsize, cmp, arr);
        npy_amergesort0(pm, pr, v, pw, elsize, cmp, arr);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (cmp(v + (*pm) * elsize, v + (*pj) * elsize, arr) < 0) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

NPY_NO_EXPORT int
npy_amergesort(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    void *arr = varr;
    npy_intp elsize;
    PyArray_CompareFunc *cmp;
    get_sort_data_from_array(arr, &elsize, &cmp);

    return npy_amergesort_impl(v, tosort, num, varr, elsize, cmp);
}

NPY_NO_EXPORT int
npy_amergesort_impl(void *v, npy_intp *tosort, npy_intp num, void *varr,
                    npy_intp elsize, PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    npy_intp *pl, *pr, *pw;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pl = tosort;
    pr = pl + num;
    pw = (npy_intp *)malloc((num >> 1) * sizeof(npy_intp));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    npy_amergesort0(pl, pr, (char *)v, pw, elsize, cmp, arr);
    free(pw);

    return 0;
}
