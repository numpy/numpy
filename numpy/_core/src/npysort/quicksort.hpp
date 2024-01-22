#ifndef NUMPY_SRC_COMMON_NPYSORT_QUICKSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_QUICKSORT_HPP

#include "heapsort.hpp"
#include "common.hpp"

namespace np::sort {

// pushing largest partition has upper bound of log2(n) space
// we store two pointers each time
constexpr size_t kQuickStack = sizeof(intptr_t) * 8 * 2;
constexpr ptrdiff_t kQuickSmall = 15;

// NUMERIC SORTS
template <typename T>
inline void Quick(T *start, SSize num)
{
    T vp;
    T *pl = start;
    T *pr = pl + num - 1;
    T *stack[kQuickStack];
    T **sptr = stack;
    T *pm, *pi, *pj, *pk;
    int depth[kQuickStack];
    int *psdepth = depth;
    int cdepth = BitScanReverse(static_cast<std::make_unsigned_t<SSize>>(num)) * 2;
    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            Heap(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > kQuickSmall) {
            // quicksort partition
            pm = pl + ((pr - pl) >> 1);
            if (LessThan(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            if (LessThan(*pr, *pm)) {
                std::swap(*pr, *pm);
            }
            if (LessThan(*pm, *pl)) {
                std::swap(*pm, *pl);
            }
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (LessThan(*pi, vp));
                do {
                    --pj;
                } while (LessThan(vp, *pj));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            // push largest partition on stack
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
            while (pj > pl && LessThan(vp, *pk)) {
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
}

// Argsort
template <typename T>
inline void Quick(T *vv, SSize *tosort, SSize num)
{
    T *v = vv;
    T vp;
    SSize *pl = tosort;
    SSize *pr = tosort + num - 1;
    SSize *stack[kQuickStack];
    SSize **sptr = stack;
    SSize *pm, *pi, *pj, *pk, vi;
    int depth[kQuickStack];
    int *psdepth = depth;
    int cdepth = BitScanReverse(static_cast<std::make_unsigned_t<SSize>>(num)) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            Heap(vv, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > kQuickSmall) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (LessThan(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            if (LessThan(v[*pr], v[*pm])) {
                std::swap(*pr, *pm);
            }
            if (LessThan(v[*pm], v[*pl])) {
                std::swap(*pm, *pl);
            }
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (LessThan(v[*pi], vp));
                do {
                    --pj;
                } while (LessThan(vp, v[*pj]));
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
            while (pj > pl && LessThan(vp, v[*pk])) {
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
}

template <typename T>
inline int Quick(T *start, SSize num, PyArrayObject *arr)
{
    const SSize arr_len = PyArray_ITEMSIZE(arr);
    const size_t len = arr_len / sizeof(T);
    T *vp;
    T *pl = start;
    T *pr = pl + (num - 1) * len;
    T *stack[kQuickStack], **sptr = stack, *pm, *pi, *pj, *pk;
    int depth[kQuickStack];
    int *psdepth = depth;
    int cdepth = BitScanReverse(static_cast<std::make_unsigned_t<SSize>>(num)) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    vp = (T *)malloc(arr_len);
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            Heap(pl, (pr - pl) / len + 1, arr);
            goto stack_pop;
        }
        while ((size_t)(pr - pl) > kQuickSmall * len) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / len) >> 1) * len;
            if (LessThan(pm, pl, len)) {
                Swap(pm, pl, len);
            }
            if (LessThan(pr, pm, len)) {
                Swap(pr, pm, len);
            }
            if (LessThan(pm, pl, len)) {
                Swap(pm, pl, len);
            }
            memcpy(vp, pm, arr_len);
            pi = pl;
            pj = pr - len;
            Swap(pm, pj, len);
            for (;;) {
                do {
                    pi += len;
                } while (LessThan(pi, vp, len));
                do {
                    pj -= len;
                } while (LessThan(vp, pj, len));
                if (pi >= pj) {
                    break;
                }
                Swap(pi, pj, len);
            }
            pk = pr - len;
            Swap(pi, pk, len);
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
            memcpy(vp, pi, arr_len);
            pj = pi;
            pk = pi - len;
            while (pj > pl && LessThan(vp, pk, len)) {
                memcpy(pj, pk, arr_len);
                pj -= len;
                pk -= len;
            }
            memcpy(pj, vp, arr_len);
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

template <typename T>
inline int Quick(T *vv, SSize *tosort, SSize num, PyArrayObject *arr)
{
    T *v = vv;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(T);
    T *vp;
    SSize *pl = tosort;
    SSize *pr = tosort + num - 1;
    SSize *stack[kQuickStack];
    SSize **sptr = stack;
    SSize *pm, *pi, *pj, *pk, vi;
    int depth[kQuickStack];
    int *psdepth = depth;
    int cdepth = BitScanReverse(static_cast<std::make_unsigned_t<SSize>>(num)) * 2;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            Heap(vv, pl, pr - pl + 1, arr);
            goto stack_pop;
        }
        while ((pr - pl) > kQuickSmall) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (LessThan(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            if (LessThan(v + (*pr) * len, v + (*pm) * len, len)) {
                std::swap(*pr, *pm);
            }
            if (LessThan(v + (*pm) * len, v + (*pl) * len, len)) {
                std::swap(*pm, *pl);
            }
            vp = v + (*pm) * len;
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (LessThan(v + (*pi) * len, vp, len));
                do {
                    --pj;
                } while (LessThan(vp, v + (*pj) * len, len));
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
            while (pj > pl && LessThan(vp, v + (*pk) * len, len)) {
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

} // np::sort
#endif // NUMPY_SRC_COMMON_NPYSORT_QUICK_HPP
