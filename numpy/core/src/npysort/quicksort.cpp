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
 * Quick sort is usually the fastest, but the worst case scenario is O(N^2) so
 * the code switches to the O(NlogN) worst case heapsort if not enough progress
 * is made on the large side of the two quicksort partitions. This improves the
 * worst case while still retaining the speed of quicksort for the common case.
 * This is variant known as introsort.
 *
 *
 * def introsort(lower, higher, recursion_limit=log2(higher - lower + 1) * 2):
 *   # sort remainder with heapsort if we are not making enough progress
 *   # we arbitrarily choose 2 * log(n) as the cutoff point
 *   if recursion_limit < 0:
 *       heapsort(lower, higher)
 *       return
 *
 *   if lower < higher:
 *      pivot_pos = partition(lower, higher)
 *      # recurse into smaller first and leave larger on stack
 *      # this limits the required stack space
 *      if (pivot_pos - lower > higher - pivot_pos):
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *      else:
 *          quicksort(lower, pivot_pos, recursion_limit - 1)
 *          quicksort(pivot_pos + 1, higher, recursion_limit - 1)
 *
 *
 * the below code implements this converted to an iteration and as an
 * additional minor optimization skips the recursion depth checking on the
 * smaller partition as it is always less than half of the remaining data and
 * will thus terminate fast enough
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_cpu_features.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "npysort_heapsort.h"
#include "numpy_tag.h"
#include "simd_qsort.hpp"

#include <cstdlib>
#include <utility>

#define NOT_USED NPY_UNUSED(unused)
/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20
#define SMALL_STRING 16

// Temporarily disable AVX512 sorting on WIN32 and CYGWIN until we can figure
// out why it has test failures
#if defined(_MSC_VER) || defined(__CYGWIN__)
template<typename T>
inline bool quicksort_dispatch(T*, npy_intp)
{
    return false;
}
template<typename T>
inline bool aquicksort_dispatch(T*, npy_intp*, npy_intp)
{
    return false;
}
#else
template<typename T>
inline bool quicksort_dispatch(T *start, npy_intp num)
{
    using TF = typename np::meta::FixedWidth<T>::Type;
    void (*dispfunc)(TF*, intptr_t) = nullptr;
    if (sizeof(T) == sizeof(uint16_t)) {
        #ifndef NPY_DISABLE_OPTIMIZATION
            #include "simd_qsort_16bit.dispatch.h"
        #endif
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
    }
    else if (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
        #ifndef NPY_DISABLE_OPTIMIZATION
            #include "simd_qsort.dispatch.h"
        #endif
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
    }
    if (dispfunc) {
        (*dispfunc)(reinterpret_cast<TF*>(start), static_cast<intptr_t>(num));
        return true;
    }
    return false;
}

template<typename T>
inline bool aquicksort_dispatch(T *start, npy_intp* arg, npy_intp num)
{
    using TF = typename np::meta::FixedWidth<T>::Type;
    void (*dispfunc)(TF*, npy_intp*, npy_intp) = nullptr;
    #ifndef NPY_DISABLE_OPTIMIZATION
        #include "simd_qsort.dispatch.h"
    #endif
    /* x86-simd-sort uses 8-byte int to store arg values, npy_intp is 4 bytes
     * in 32-bit*/
    if (sizeof(npy_intp) == sizeof(int64_t)) {
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSort, <TF>);
    }
    if (dispfunc) {
        (*dispfunc)(reinterpret_cast<TF*>(start), arg, num);
        return true;
    }
    return false;
}
#endif // _MSC_VER || CYGWIN

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static int
quicksort_(type *start, npy_intp num)
{
    type vp;
    type *pl = start;
    type *pr = pl + num - 1;
    type *stack[PYA_QS_STACK];
    type **sptr = stack;
    type *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            heapsort_<Tag>(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(*pi, vp));
                do {
                    --pj;
                } while (Tag::less(vp, *pj));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

template <typename Tag, typename type>
static int
aquicksort_(type *vv, npy_intp *tosort, npy_intp num)
{
    type *v = vv;
    type vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_<Tag>(vv, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(v[*pr], v[*pm])) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v[*pi], vp));
                do {
                    --pj;
                } while (Tag::less(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static int
string_quicksort_(type *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    const size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    type *vp;
    type *pl = start;
    type *pr = pl + (num - 1) * len;
    type *stack[PYA_QS_STACK], **sptr = stack, *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    vp = (type *)malloc(PyArray_ITEMSIZE(arr));
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_heapsort_<Tag>(pl, (pr - pl) / len + 1, varr);
            goto stack_pop;
        }
        while ((size_t)(pr - pl) > SMALL_QUICKSORT * len) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / len) >> 1) * len;
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            if (Tag::less(pr, pm, len)) {
                Tag::swap(pr, pm, len);
            }
            if (Tag::less(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            Tag::copy(vp, pm, len);
            pi = pl;
            pj = pr - len;
            Tag::swap(pm, pj, len);
            for (;;) {
                do {
                    pi += len;
                } while (Tag::less(pi, vp, len));
                do {
                    pj -= len;
                } while (Tag::less(vp, pj, len));
                if (pi >= pj) {
                    break;
                }
                Tag::swap(pi, pj, len);
            }
            pk = pr - len;
            Tag::swap(pi, pk, len);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + len;
                *sptr++ = pr;
                pr = pi - len;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - len;
                pl = pi + len;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + len; pi <= pr; pi += len) {
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
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    free(vp);
    return 0;
}

template <typename Tag, typename type>
static int
string_aquicksort_(type *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    type *v = vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    type *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_aheapsort_<Tag>(vv, pl, pr - pl + 1, varr);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            if (Tag::less(v + (*pr) * len, v + (*pm) * len, len)) {
                std::swap(*pr, *pm);
            }
            if (Tag::less(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            vp = v + (*pm) * len;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (Tag::less(v + (*pi) * len, vp, len));
                do {
                    --pj;
                } while (Tag::less(vp, v + (*pj) * len, len));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * len;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_quicksort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    char *vp;
    char *pl = (char *)start;
    char *pr = pl + (num - 1) * elsize;
    char *stack[PYA_QS_STACK];
    char **sptr = stack;
    char *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    vp = (char *)malloc(elsize);
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_heapsort(pl, (pr - pl) / elsize + 1, varr);
            goto stack_pop;
        }
        while (pr - pl > SMALL_QUICKSORT * elsize) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            if (cmp(pr, pm, arr) < 0) {
                GENERIC_SWAP(pr, pm, elsize);
            }
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            GENERIC_COPY(vp, pm, elsize);
            pi = pl;
            pj = pr - elsize;
            GENERIC_SWAP(pm, pj, elsize);
            /*
             * Generic comparisons may be buggy, so don't rely on the sentinels
             * to keep the pointers from going out of bounds.
             */
            for (;;) {
                do {
                    pi += elsize;
                } while (cmp(pi, vp, arr) < 0 && pi < pj);
                do {
                    pj -= elsize;
                } while (cmp(vp, pj, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                GENERIC_SWAP(pi, pj, elsize);
            }
            pk = pr - elsize;
            GENERIC_SWAP(pi, pk, elsize);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + elsize;
                *sptr++ = pr;
                pr = pi - elsize;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - elsize;
                pl = pi + elsize;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + elsize; pi <= pr; pi += elsize) {
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
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    free(vp);
    return 0;
}

NPY_NO_EXPORT int
npy_aquicksort(void *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    char *v = (char *)vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    char *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_aheapsort(vv, pl, pr - pl + 1, varr);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            if (cmp(v + (*pr) * elsize, v + (*pm) * elsize, arr) < 0) {
                INTP_SWAP(*pr, *pm);
            }
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            vp = v + (*pm) * elsize;
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (cmp(v + (*pi) * elsize, vp, arr) < 0 && pi < pj);
                do {
                    --pj;
                } while (cmp(vp, v + (*pj) * elsize, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}

/***************************************
 * C > C++ dispatch
 ***************************************/

NPY_NO_EXPORT int
quicksort_bool(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::bool_tag>((npy_bool *)start, n);
}
NPY_NO_EXPORT int
quicksort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::byte_tag>((npy_byte *)start, n);
}
NPY_NO_EXPORT int
quicksort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::ubyte_tag>((npy_ubyte *)start, n);
}
NPY_NO_EXPORT int
quicksort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_short *)start, n)) {
        return 0;
    }
    return quicksort_<npy::short_tag>((npy_short *)start, n);
}
NPY_NO_EXPORT int
quicksort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_ushort *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ushort_tag>((npy_ushort *)start, n);
}
NPY_NO_EXPORT int
quicksort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_int *)start, n)) {
        return 0;
    }
    return quicksort_<npy::int_tag>((npy_int *)start, n);
}
NPY_NO_EXPORT int
quicksort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_uint *)start, n)) {
        return 0;
    }
    return quicksort_<npy::uint_tag>((npy_uint *)start, n);
}
NPY_NO_EXPORT int
quicksort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_long *)start, n)) {
        return 0;
    }
    return quicksort_<npy::long_tag>((npy_long *)start, n);
}
NPY_NO_EXPORT int
quicksort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_ulong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ulong_tag>((npy_ulong *)start, n);
}
NPY_NO_EXPORT int
quicksort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_longlong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::longlong_tag>((npy_longlong *)start, n);
}
NPY_NO_EXPORT int
quicksort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_ulonglong *)start, n)) {
        return 0;
    }
    return quicksort_<npy::ulonglong_tag>((npy_ulonglong *)start, n);
}
NPY_NO_EXPORT int
quicksort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Half *)start, n)) {
        return 0;
    }
    return quicksort_<npy::half_tag>((npy_half *)start, n);
}
NPY_NO_EXPORT int
quicksort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_float *)start, n)) {
        return 0;
    }
    return quicksort_<npy::float_tag>((npy_float *)start, n);
}
NPY_NO_EXPORT int
quicksort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((npy_double *)start, n)) {
        return 0;
    }
    return quicksort_<npy::double_tag>((npy_double *)start, n);
}
NPY_NO_EXPORT int
quicksort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::longdouble_tag>((npy_longdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cfloat_tag>((npy_cfloat *)start, n);
}
NPY_NO_EXPORT int
quicksort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::cdouble_tag>((npy_cdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::clongdouble_tag>((npy_clongdouble *)start, n);
}
NPY_NO_EXPORT int
quicksort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::datetime_tag>((npy_datetime *)start, n);
}
NPY_NO_EXPORT int
quicksort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    return quicksort_<npy::timedelta_tag>((npy_timedelta *)start, n);
}

NPY_NO_EXPORT int
aquicksort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::bool_tag>((npy_bool *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::byte_tag>((npy_byte *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ubyte(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ubyte_tag>((npy_ubyte *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_short(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::short_tag>((npy_short *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ushort(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::ushort_tag>((npy_ushort *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_int *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::int_tag>((npy_int *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_uint *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::uint_tag>((npy_uint *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_long *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::long_tag>((npy_long *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ulong(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_ulong *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::ulong_tag>((npy_ulong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_longlong *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::longlong_tag>((npy_longlong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_ulonglong *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::ulonglong_tag>((npy_ulonglong *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::half_tag>((npy_half *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_float(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_float *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::float_tag>((npy_float *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_double(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((npy_double *)vv, tosort, n)) {
        return 0;
    }
    return aquicksort_<npy::double_tag>((npy_double *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::longdouble_tag>((npy_longdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::cfloat_tag>((npy_cfloat *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::cdouble_tag>((npy_cdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                       void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::clongdouble_tag>((npy_clongdouble *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::datetime_tag>((npy_datetime *)vv, tosort, n);
}
NPY_NO_EXPORT int
aquicksort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    return aquicksort_<npy::timedelta_tag>((npy_timedelta *)vv, tosort, n);
}

NPY_NO_EXPORT int
quicksort_string(void *start, npy_intp n, void *varr)
{
    return string_quicksort_<npy::string_tag>((npy_char *)start, n, varr);
}
NPY_NO_EXPORT int
quicksort_unicode(void *start, npy_intp n, void *varr)
{
    return string_quicksort_<npy::unicode_tag>((npy_ucs4 *)start, n, varr);
}

NPY_NO_EXPORT int
aquicksort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aquicksort_<npy::string_tag>((npy_char *)vv, tosort, n,
                                               varr);
}
NPY_NO_EXPORT int
aquicksort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return string_aquicksort_<npy::unicode_tag>((npy_ucs4 *)vv, tosort, n,
                                                varr);
}
