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

#include "heapsort.hpp"

namespace np::sort {

constexpr size_t kMergeStack = 100;
constexpr ptrdiff_t kMergeSmall = 20;

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename T>
inline void Merge(T *pl, T *pr, T *pw)
{
    T vp, *pi, *pj, *pk, *pm;

    if (pr - pl > kMergeSmall) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        Merge(pl, pm, pw);
        Merge(pm, pr, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (LessThan(*pm, *pj)) {
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
            while (pj > pl && LessThan(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}

template <typename T>
inline int Merge(T *start, SSize num)
{
    T *pl = start;
    T *pr = pl + num;
    T *pw = (T *)malloc((num / 2) * sizeof(T));
    if (pw == nullptr) {
        return -NPY_ENOMEM;
    }
    Merge(pl, pr, pw);
    free(pw);
    return 0;
}

template <typename T>
inline void MergeArg(SSize *pl, SSize *pr, T *v, SSize *pw)
{
    T vp;
    SSize vi, *pi, *pj, *pk, *pm;

    if (pr - pl > kMergeSmall) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        MergeArg(pl, pm, v, pw);
        MergeArg(pm, pr, v, pw);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (LessThan(v[*pm], v[*pj])) {
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
            while (pj > pl && LessThan(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename T>
inline int MergeArg(T *v, SSize *tosort, SSize num)
{
    SSize *pl = tosort;
    SSize *pr = pl + num;
    SSize *pw = (SSize *)malloc((num / 2) * sizeof(SSize));
    if (pw == nullptr) {
        return -NPY_ENOMEM;
    }
    MergeArg(pl, pr, v, pw);
    free(pw);
    return 0;
}

/*

 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename T>
inline void Merge(T *pl, T *pr, T *pw, T *vp, SSize arr_len)
{
    T *pi, *pj, *pk, *pm;
    SSize len = arr_len / sizeof(T);
    if (pr - pl > kMergeSmall * len) {
        /* merge sort */
        pm = pl + (((pr - pl) / len) >> 1) * len;
        Merge(pl, pm, pw, vp, len);
        Merge(pm, pr, pw, vp, len);
        memcpy(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (LessThan(pm, pj, len)) {
                memcpy(pk, pm, len);
                pm += len;
                pk += len;
            }
            else {
                memcpy(pk, pj, len);
                pj += len;
                pk += len;
            }
        }
        memcpy(pk, pj, pi - pj);
    }
    else {
        /* insertion sort */
        for (pi = pl + len; pi < pr; pi += len) {
            memcpy(vp, pi, len);
            pj = pi;
            pk = pi - len;
            while (pj > pl && LessThan(vp, pk, len)) {
                memcpy(pj, pk, len);
                pj -= len;
                pk -= len;
            }
            memcpy(pj, vp, len);
        }
    }
}

template <typename T>
inline int Merge(T *start, SSize num, PyArrayObject *arr)
{
    SSize arr_len = PyArray_ITEMSIZE(arr);
    /* Items that have zero size don't make sense to sort */
    if (arr_len == 0) {
        return 0;
    }

    SSize len = arr_len / sizeof(T);
    T *pl = start;
    T *pr = pl + num * len;
    T *pw = (T *)malloc((num / 2) * arr_len);
    T *vp = (T *)malloc(arr_len);

    if (pw == nullptr) {
        return -NPY_ENOMEM;
    }
    if (vp == nullptr) {
        free(pw);
        return -NPY_ENOMEM;
    }
    Merge(pl, pr, pw, vp, arr_len);
    free(vp);
    free(pw);
    return 0;
}

template <typename T>
inline void MergeArg(SSize *pl, SSize *pr, T *v, SSize *pw, size_t len)
{
    T *vp;
    SSize vi, *pi, *pj, *pk, *pm;

    if (pr - pl > kMergeSmall) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        MergeArg(pl, pm, v, pw, len);
        MergeArg(pm, pr, v, pw, len);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (LessThan(v + (*pm) * len, v + (*pj) * len, len)) {
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
            while (pj > pl && LessThan(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

template <typename T>
static int MergeArg(T *v, SSize *tosort, SSize num, PyArrayObject *arr)
{
    SSize arr_len = PyArray_ITEMSIZE(arr);

    /* Items that have zero size don't make sense to sort */
    if (arr_len == 0) {
        return 0;
    }

    SSize len = arr_len / sizeof(T);
    SSize *pl = tosort;
    SSize *pr = pl + num;
    SSize *pw = (SSize *)malloc((num / 2) * sizeof(SSize));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    MergeArg(pl, pr, v, pw, len);
    free(pw);
    return 0;
}
} // namespace np::sort
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

    if (pr - pl > np::sort::kMergeSmall * elsize) {
        /* merge sort */
        pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
        npy_mergesort0(pl, pm, pw, vp, elsize, cmp, arr);
        npy_mergesort0(pm, pr, pw, vp, elsize, cmp, arr);
        memcpy(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (cmp(pm, pj, arr) < 0) {
                memcpy(pk, pm, elsize);
                pm += elsize;
                pk += elsize;
            }
            else {
                memcpy(pk, pj, elsize);
                pj += elsize;
                pk += elsize;
            }
        }
        memcpy(pk, pj, pi - pj);
    }
    else {
        /* insertion sort */
        for (pi = pl + elsize; pi < pr; pi += elsize) {
            memcpy(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                memcpy(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            memcpy(pj, vp, elsize);
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

    if (pr - pl > np::sort::kMergeSmall) {
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

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    npy_intp *pl = tosort;
    npy_intp *pr = pl + num;
    npy_intp *pw = (npy_intp *)malloc((num >> 1) * sizeof(npy_intp));
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
    return np::sort::Merge((np::Bool *)start, num);
}
NPY_NO_EXPORT int
mergesort_byte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Byte *)start, num);
}
NPY_NO_EXPORT int
mergesort_ubyte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::UByte *)start, num);
}
NPY_NO_EXPORT int
mergesort_short(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Short *)start, num);
}
NPY_NO_EXPORT int
mergesort_ushort(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::UShort *)start, num);
}
NPY_NO_EXPORT int
mergesort_int(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Int *)start, num);
}
NPY_NO_EXPORT int
mergesort_uint(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::UInt *)start, num);
}
NPY_NO_EXPORT int
mergesort_long(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Long *)start, num);
}
NPY_NO_EXPORT int
mergesort_ulong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::ULong *)start, num);
}
NPY_NO_EXPORT int
mergesort_longlong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::LongLong *)start, num);
}
NPY_NO_EXPORT int
mergesort_ulonglong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::ULongLong *)start, num);
}
NPY_NO_EXPORT int
mergesort_half(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Half *)start, num);
}
NPY_NO_EXPORT int
mergesort_float(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Float *)start, num);
}
NPY_NO_EXPORT int
mergesort_double(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::Double *)start, num);
}
NPY_NO_EXPORT int
mergesort_longdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::LongDouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_cfloat(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::CFloat *)start, num);
}
NPY_NO_EXPORT int
mergesort_cdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::CDouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_clongdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::CLongDouble *)start, num);
}
NPY_NO_EXPORT int
mergesort_datetime(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::DateTime *)start, num);
}
NPY_NO_EXPORT int
mergesort_timedelta(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return np::sort::Merge((np::TimeDelta *)start, num);
}

NPY_NO_EXPORT int
amergesort_bool(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Bool *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_byte(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Byte *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ubyte(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::UByte *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_short(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Short *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ushort(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::UShort *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_int(void *start, npy_intp *tosort, npy_intp num,
               void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Int *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_uint(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::UInt *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_long(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Long *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ulong(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::ULong *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_longlong(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::LongLong *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_ulonglong(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::ULongLong *)start, tosort,
                                           num);
}
NPY_NO_EXPORT int
amergesort_half(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Half *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_float(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Float *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_double(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::Double *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_longdouble(void *start, npy_intp *tosort, npy_intp num,
                      void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::LongDouble *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_cfloat(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::CFloat *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_cdouble(void *start, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::CDouble *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_clongdouble(void *start, npy_intp *tosort, npy_intp num,
                       void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::CLongDouble *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_datetime(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::DateTime *)start, tosort, num);
}
NPY_NO_EXPORT int
amergesort_timedelta(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    return np::sort::MergeArg((np::TimeDelta *)start, tosort, num);
}

NPY_NO_EXPORT int
mergesort_string(void *start, npy_intp num, void *varr)
{
    return np::sort::Merge((np::String*)start, num, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
mergesort_unicode(void *start, npy_intp num, void *varr)
{
    return np::sort::Merge((np::Unicode*)start, num, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
amergesort_string(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return np::sort::MergeArg((np::String*)v, tosort, num, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
amergesort_unicode(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return np::sort::MergeArg((np::Unicode*)v, tosort, num, (PyArrayObject *)varr);
}
