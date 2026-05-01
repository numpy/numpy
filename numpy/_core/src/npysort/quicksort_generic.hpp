#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_cpu_features.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "npysort_heapsort.h"
#include "numpy_tag.h"
#include "x86_simd_qsort.hpp"
#include "highway_qsort.hpp"

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

// Disable AVX512 sorting on CYGWIN until we can figure
// out why it has test failures
template<typename T>
inline bool quicksort_dispatch(T *start, npy_intp num)
{
#if !defined(__CYGWIN__)
    using TF = typename np::meta::FixedWidth<T>::Type;
    void (*dispfunc)(TF*, intptr_t) = nullptr;
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
    #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86) // x86 32-bit and 64-bit
        #include "x86_simd_qsort_16bit.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
    #else
        #include "highway_qsort_16bit.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
    #endif
    }
    else if constexpr (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
    #if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86) // x86 32-bit and 64-bit
        #include "x86_simd_qsort.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSort, <TF>);
    #else
        #include "highway_qsort.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template QSort, <TF>);
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
    using TF = typename np::meta::FixedWidth<T>::Type;
    void (*dispfunc)(TF*, npy_intp*, npy_intp) = nullptr;
    #include "x86_simd_argsort.dispatch.h"
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
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type, bool reverse = false>
static int
quicksort_(type *start, npy_intp num)
{
    if constexpr (!reverse && Tag::has_sort_dispatch) {
        using T = typename std::conditional<std::is_same_v<Tag, npy::half_tag>, np::Half, typename Tag::type>::type;
        if (quicksort_dispatch((T *)start, num)) {
            return 0;
        }
    }

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
            heapsort_<Tag, type, reverse>(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (npy::cmp<Tag, reverse>(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            if (npy::cmp<Tag, reverse>(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            if (npy::cmp<Tag, reverse>(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (npy::cmp<Tag, reverse>(*pi, vp));
                do {
                    --pj;
                } while (npy::cmp<Tag, reverse>(vp, *pj));
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
            while (pj > pl && npy::cmp<Tag, reverse>(vp, *pk)) {
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

// ``PyArray_SortFunc``-shaped trampoline.
template <typename Tag, typename type, bool reverse = false>
static int
quicksort_impl(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return quicksort_<Tag, type, reverse>((type *)start, num);
}

template <typename Tag, typename type, bool reverse = false>
static int
aquicksort_(type *vv, npy_intp *tosort, npy_intp num)
{
    if constexpr (!reverse && Tag::has_argsort_dispatch) {
        if (aquicksort_dispatch((type *)vv, tosort, num)) {
            return 0;
        }
    }

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
            aheapsort_<Tag, type, reverse>(vv, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (npy::cmp<Tag, reverse>(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            if (npy::cmp<Tag, reverse>(v[*pr], v[*pm])) {
                std::swap(*pr, *pm);
            }
            if (npy::cmp<Tag, reverse>(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (npy::cmp<Tag, reverse>(v[*pi], vp));
                do {
                    --pj;
                } while (npy::cmp<Tag, reverse>(vp, v[*pj]));
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
            while (pj > pl && npy::cmp<Tag, reverse>(vp, v[*pk])) {
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

// ``PyArray_ArgSortFunc``-shaped trampoline.
template <typename Tag, typename type, bool reverse = false>
static int
aquicksort_impl(void *vv, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return aquicksort_<Tag, type, reverse>((type *)vv, tosort, num);
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type, bool reverse = false>
static int
string_quicksort_(type *start, npy_intp num, int elsize)
{
    const size_t len = elsize / sizeof(type);
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

    vp = (type *)malloc(elsize);
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            string_heapsort_<Tag, type, reverse>(pl, (pr - pl) / len + 1, elsize);
            goto stack_pop;
        }
        while ((size_t)(pr - pl) > SMALL_QUICKSORT * len) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / len) >> 1) * len;
            if (npy::cmp<Tag, reverse>(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            if (npy::cmp<Tag, reverse>(pr, pm, len)) {
                Tag::swap(pr, pm, len);
            }
            if (npy::cmp<Tag, reverse>(pm, pl, len)) {
                Tag::swap(pm, pl, len);
            }
            Tag::copy(vp, pm, len);
            pi = pl;
            pj = pr - len;
            Tag::swap(pm, pj, len);
            for (;;) {
                do {
                    pi += len;
                } while (npy::cmp<Tag, reverse>(pi, vp, len));
                do {
                    pj -= len;
                } while (npy::cmp<Tag, reverse>(vp, pj, len));
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
            while (pj > pl && npy::cmp<Tag, reverse>(vp, pk, len)) {
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

template <typename Tag, typename type, bool reverse = false>
static int
string_aquicksort_(type *vv, npy_intp *tosort, npy_intp num, int elsize)
{
    type *v = vv;
    size_t len = elsize / sizeof(type);
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
            string_aheapsort_<Tag, type, reverse>(vv, pl, pr - pl + 1, elsize);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (npy::cmp<Tag, reverse>(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            if (npy::cmp<Tag, reverse>(v + (*pr) * len, v + (*pm) * len, len)) {
                std::swap(*pr, *pm);
            }
            if (npy::cmp<Tag, reverse>(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            vp = v + (*pm) * len;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (npy::cmp<Tag, reverse>(v + (*pi) * len, vp, len));
                do {
                    --pj;
                } while (npy::cmp<Tag, reverse>(vp, v + (*pj) * len, len));
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
            while (pj > pl && npy::cmp<Tag, reverse>(vp, v + (*pk) * len, len)) {
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