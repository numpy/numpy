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

template <typename Tag, typename type>
static void
mergesort0_(type *pl, type *pr, type *pw)
{
    type vp, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        mergesort0_<Tag>(pl, pm, pw);
        mergesort0_<Tag>(pm, pr, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::less(*pm, *pj)) {
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
            while (pj > pl && Tag::less(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}

template <typename Tag, typename type>
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
    mergesort0_<Tag>(pl, pr, pw);

    free(pw);
    return 0;
}

template <typename Tag, typename type>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw)
{
    type vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_<Tag>(pl, pm, v, pw);
        amergesort0_<Tag>(pm, pr, v, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::less(v[*pm], v[*pj])) {
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
            while (pj > pl && Tag::less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename Tag, typename type>
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
    amergesort0_<Tag>(pl, pr, v, pw);
    free(pw);

    return 0;
}

/*
 
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static void
mergesort0_(type *pl, type *pr, type *pw, type *vp, size_t len)
{
    type *pi, *pj, *pk, *pm;

    if ((size_t)(pr - pl) > SMALL_MERGESORT * len) {
        /* merge sort */
        pm = pl + (((pr - pl) / len) >> 1) * len;
        mergesort0_<Tag>(pl, pm, pw, vp, len);
        mergesort0_<Tag>(pm, pr, pw, vp, len);
        Tag::copy(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::less(pm, pj, len)) {
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
            while (pj > pl && Tag::less(vp, pk, len)) {
                Tag::copy(pj, pk, len);
                pj -= len;
                pk -= len;
            }
            Tag::copy(pj, vp, len);
        }
    }
}

template <typename Tag, typename type>
static int
string_mergesort_(type *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t elsize = PyArray_ITEMSIZE(arr);
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
    mergesort0_<Tag>(pl, pr, pw, vp, len);

    free(vp);
fail_1:
    free(pw);
fail_0:
    return err;
}

template <typename Tag, typename type>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw, size_t len)
{
    type *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        amergesort0_<Tag>(pl, pm, v, pw, len);
        amergesort0_<Tag>(pm, pr, v, pw, len);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (Tag::less(v + (*pm) * len, v + (*pj) * len, len)) {
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
            while (pj > pl && Tag::less(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename Tag, typename type>
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
    amergesort0_<Tag>(pl, pr, v, pw, len);
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
               PyArray_CompareFunc *cmp, PyArrayObject *arr)
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
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
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
                npy_intp elsize, PyArray_CompareFunc *cmp, PyArrayObject *arr)
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
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
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

/***************************************
 * C > C++ dispatch
 ***************************************/
NPY_NO_EXPORT int
mergesort_bool(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::bool_tag>((npy_bool *)start, num);
}
NPY_NO_EXPORT int
mergesort_byte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::byte_tag>((npy_byte *)start, num);
}
NPY_NO_EXPORT int
mergesort_ubyte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ubyte_tag>((npy_ubyte *)start, num);
}
NPY_NO_EXPORT int
mergesort_short(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::short_tag>((npy_short *)start, num);
}
NPY_NO_EXPORT int
mergesort_ushort(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ushort_tag>((npy_ushort *)start, num);
}
NPY_NO_EXPORT int
mergesort_int(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::int_tag>((npy_int *)start, num);
}
NPY_NO_EXPORT int
mergesort_uint(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::uint_tag>((npy_uint *)start, num);
}
NPY_NO_EXPORT int
mergesort_long(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::long_tag>((npy_long *)start, num);
}
NPY_NO_EXPORT int
mergesort_ulong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ulong_tag>((npy_ulong *)start, num);
}
NPY_NO_EXPORT int
mergesort_longlong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::longlong_tag>((npy_longlong *)start, num);
}
NPY_NO_EXPORT int
mergesort_ulonglong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ulonglong_tag>((npy_ulonglong *)start, num);
}
NPY_NO_EXPORT int
mergesort_half(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::half_tag>((npy_half *)start, num);
}
NPY_NO_EXPORT int
mergesort_float(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::float_tag>((npy_float *)start, num);
}
NPY_NO_EXPORT int
mergesort_double(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::double_tag>((npy_double *)start, num);
}
NPY_NO_EXPORT int
mergesort_longdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::longdouble_tag>((npy_longdouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_cfloat(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::cfloat_tag>((npy_cfloat *)start, num);
}
NPY_NO_EXPORT int
mergesort_cdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::cdouble_tag>((npy_cdouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_clongdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::clongdouble_tag>((npy_clongdouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_datetime(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::datetime_tag>((npy_datetime *)start, num);
}
NPY_NO_EXPORT int
mergesort_timedelta(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::timedelta_tag>((npy_timedelta *)start, num);
}

NPY_NO_EXPORT int
amergesort_bool(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return amergesort_<npy::bool_tag>((npy_bool *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_byte(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return amergesort_<npy::byte_tag>((npy_byte *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ubyte(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return amergesort_<npy::ubyte_tag>((npy_ubyte *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_short(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return amergesort_<npy::short_tag>((npy_short *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ushort(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return amergesort_<npy::ushort_tag>((npy_ushort *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_int(void *start, npy_intp *tosort, npy_intp num,
               void *NPY_UNUSED(varr))
{
    return amergesort_<npy::int_tag>((npy_int *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_uint(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return amergesort_<npy::uint_tag>((npy_uint *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_long(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return amergesort_<npy::long_tag>((npy_long *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ulong(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return amergesort_<npy::ulong_tag>((npy_ulong *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_longlong(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    return amergesort_<npy::longlong_tag>((npy_longlong *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ulonglong(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    return amergesort_<npy::ulonglong_tag>((npy_ulonglong *)start, tosort,
                                           num);
}
NPY_NO_EXPORT int
amergesort_half(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return amergesort_<npy::half_tag>((npy_half *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_float(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return amergesort_<npy::float_tag>((npy_float *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_double(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return amergesort_<npy::double_tag>((npy_double *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_longdouble(void *start, npy_intp *tosort, npy_intp num,
                      void *NPY_UNUSED(varr))
{
    return amergesort_<npy::longdouble_tag>((npy_longdouble *)start, tosort,
                                            num);
}
NPY_NO_EXPORT int
amergesort_cfloat(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return amergesort_<npy::cfloat_tag>((npy_cfloat *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_cdouble(void *start, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    return amergesort_<npy::cdouble_tag>((npy_cdouble *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_clongdouble(void *start, npy_intp *tosort, npy_intp num,
                       void *NPY_UNUSED(varr))
{
    return amergesort_<npy::clongdouble_tag>((npy_clongdouble *)start, tosort,
                                             num);
}
NPY_NO_EXPORT int
amergesort_datetime(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    return amergesort_<npy::datetime_tag>((npy_datetime *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_timedelta(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    return amergesort_<npy::timedelta_tag>((npy_timedelta *)start, tosort,
                                           num);
}

NPY_NO_EXPORT int
mergesort_string(void *start, npy_intp num, void *varr)
{
    return string_mergesort_<npy::string_tag>((npy_char *)start, num, varr);
}
NPY_NO_EXPORT int
mergesort_unicode(void *start, npy_intp num, void *varr)
{
    return string_mergesort_<npy::unicode_tag>((npy_ucs4 *)start, num, varr);
}
NPY_NO_EXPORT int
amergesort_string(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return string_amergesort_<npy::string_tag>((npy_char *)v, tosort, num,
                                               varr);
}
NPY_NO_EXPORT int
amergesort_unicode(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return string_amergesort_<npy::unicode_tag>((npy_ucs4 *)v, tosort, num,
                                                varr);
}
