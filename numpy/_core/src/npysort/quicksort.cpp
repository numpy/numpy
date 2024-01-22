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
#include "x86_simd_qsort.hpp"
#include "highway_qsort.hpp"

#include <cstdlib>
#include <utility>

#include "quicksort.hpp"


#define NOT_USED NPY_UNUSED(unused)

/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20
#define SMALL_STRING 16

// Disable AVX512 sorting on CYGWIN until we can figure
// out why it has test failures
template<typename T>
inline bool quicksort_dispatch(T *start, npy_intp num)
{
#if !defined(__CYGWIN__)
    using TF = np::MakeFixedIntegral<T>;
    void (*dispfunc)(TF*, intptr_t) = nullptr;
    if (sizeof(T) == sizeof(uint16_t)) {
        #ifndef NPY_DISABLE_OPTIMIZATION
            #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86) // x86 32-bit and 64-bit
                #include "x86_simd_qsort_16bit.dispatch.h"
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
            #else
                #include "highway_qsort_16bit.dispatch.h"
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
            #endif
        #endif
    }
    else if (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
        #ifndef NPY_DISABLE_OPTIMIZATION
            #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86) // x86 32-bit and 64-bit
                #include "x86_simd_qsort.dispatch.h"
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
            #else
                #include "highway_qsort.dispatch.h"
                NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
            #endif
        #endif
    }
    if (dispfunc) {
        (*dispfunc)(reinterpret_cast<TF*>(start), static_cast<intptr_t>(num));
        return true;
    }
#endif // __CYGWIN__
    (void)start; (void)num; // to avoid unused arg warn
    return false;
}

template<typename T>
inline bool aquicksort_dispatch(T *start, npy_intp* arg, npy_intp num)
{
#if !defined(__CYGWIN__)
    using TF = np::MakeFixedIntegral<T>;
    void (*dispfunc)(TF*, npy_intp*, npy_intp) = nullptr;
    #ifndef NPY_DISABLE_OPTIMIZATION
        #include "x86_simd_argsort.dispatch.h"
    #endif
    NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSort, <TF>);
    if (dispfunc) {
        (*dispfunc)(reinterpret_cast<TF*>(start), arg, num);
        return true;
    }
#endif // __CYGWIN__
    (void)start; (void)arg; (void)num; // to avoid unused arg warn
    return false;
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
                np::sort::Swap(pm, pl, elsize);
            }
            if (cmp(pr, pm, arr) < 0) {
                np::sort::Swap(pr, pm, elsize);
            }
            if (cmp(pm, pl, arr) < 0) {
                np::sort::Swap(pm, pl, elsize);
            }
            memcpy(vp, pm, elsize);
            pi = pl;
            pj = pr - elsize;
            np::sort::Swap(pm, pj, elsize);
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
                np::sort::Swap(pi, pj, elsize);
            }
            pk = pr - elsize;
            np::sort::Swap(pi, pk, elsize);
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
                std::swap(*pm, *pl);
            }
            if (cmp(v + (*pr) * elsize, v + (*pm) * elsize, arr) < 0) {
                std::swap(*pr, *pm);
            }
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                std::swap(*pm, *pl);
            }
            vp = v + (*pm) * elsize;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
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
    np::sort::Quick((np::Bool *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_byte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::Byte *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_ubyte(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::UByte *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_short(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Short *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Short *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_ushort(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::UShort *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::UShort *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_int(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Int *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Int *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_uint(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::UInt *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::UInt *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_long(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Long *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Long *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_ulong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::ULong *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::ULong *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_longlong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::LongLong *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::LongLong *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_ulonglong(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::ULongLong *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::ULongLong *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_half(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Half *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Half *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_float(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Float *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Float *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_double(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    if (quicksort_dispatch((np::Double *)start, n)) {
        return 0;
    }
    np::sort::Quick((np::Double *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_longdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::LongDouble *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_cfloat(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CFloat *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_cdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CDouble *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_clongdouble(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CLongDouble *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_datetime(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::DateTime *)start, n);
    return 0;
}
NPY_NO_EXPORT int
quicksort_timedelta(void *start, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::TimeDelta *)start, n);
    return 0;
}

NPY_NO_EXPORT int
aquicksort_bool(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::Bool *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_byte(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::Byte *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_ubyte(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::UByte *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_short(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::Short *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_ushort(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::UShort *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_int(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::Int *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::Int *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_uint(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::UInt *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::UInt *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_long(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::Long *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::Long *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_ulong(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::ULong *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::ULong *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_longlong(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::LongLong *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::LongLong *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_ulonglong(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::ULongLong *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::ULongLong *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_half(void *vv, npy_intp *tosort, npy_intp n, void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::Half *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_float(void *vv, npy_intp *tosort, npy_intp n,
                 void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::Float *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::Float *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_double(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    if (aquicksort_dispatch((np::Double *)vv, tosort, n)) {
        return 0;
    }
    np::sort::Quick((np::Double *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_longdouble(void *vv, npy_intp *tosort, npy_intp n,
                      void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::LongDouble *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_cfloat(void *vv, npy_intp *tosort, npy_intp n,
                  void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CFloat *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_cdouble(void *vv, npy_intp *tosort, npy_intp n,
                   void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CDouble *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_clongdouble(void *vv, npy_intp *tosort, npy_intp n,
                       void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::CLongDouble *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_datetime(void *vv, npy_intp *tosort, npy_intp n,
                    void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::DateTime *)vv, tosort, n);
    return 0;
}
NPY_NO_EXPORT int
aquicksort_timedelta(void *vv, npy_intp *tosort, npy_intp n,
                     void *NPY_UNUSED(varr))
{
    np::sort::Quick((np::TimeDelta *)vv, tosort, n);
    return 0;
}

NPY_NO_EXPORT int
quicksort_string(void *start, npy_intp n, void *varr)
{
    return np::sort::Quick((np::String*)start, n, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
quicksort_unicode(void *start, npy_intp n, void *varr)
{
    return np::sort::Quick((np::Unicode*)start, n, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
aquicksort_string(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return np::sort::Quick((np::String*)vv, tosort, n, (PyArrayObject *)varr);
}
NPY_NO_EXPORT int
aquicksort_unicode(void *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    return np::sort::Quick((np::Unicode*)vv, tosort, n, (PyArrayObject *)varr);
}
