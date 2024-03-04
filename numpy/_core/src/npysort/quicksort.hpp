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
} // np::sort
#endif // NUMPY_SRC_COMMON_NPYSORT_QUICK_HPP
