#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#include "x86-simd-sort/src/x86simdsort-static-incl.h"

// hasnan is unknown up front, so always true; nans_last defaults to true.
#define DISPATCH_SORT_METHODS(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(TYPE *arr, npy_intp num, npy_intp kth, bool reverse) \
{ \
    x86simdsortStatic::qselect(arr, kth, num, true, reverse); \
} \
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(TYPE *arr, npy_intp num, bool reverse) \
{ \
    x86simdsortStatic::qsort(arr, num, true, reverse); \
} \

namespace np { namespace qsort_simd {
    DISPATCH_SORT_METHODS(uint32_t)
    DISPATCH_SORT_METHODS(int32_t)
    DISPATCH_SORT_METHODS(float)
    DISPATCH_SORT_METHODS(uint64_t)
    DISPATCH_SORT_METHODS(int64_t)
    DISPATCH_SORT_METHODS(double)
}} // namespace np::qsort_simd

#endif // __CYGWIN__
