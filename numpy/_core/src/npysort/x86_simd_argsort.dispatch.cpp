#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#include "x86-simd-sort/src/x86simdsort-static-incl.h"

#define DISPATCH_ARG_METHODS(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(TYPE* arr, npy_intp* arg, npy_intp num, npy_intp kth) \
{ \
    x86simdsortStatic::argselect(arr, reinterpret_cast<size_t*>(arg), kth, num, true); \
} \
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(TYPE* arr, npy_intp *arg, npy_intp size) \
{ \
    x86simdsortStatic::argsort(arr, reinterpret_cast<size_t*>(arg), size, true); \
} \

namespace np { namespace qsort_simd {

    DISPATCH_ARG_METHODS(uint32_t)
    DISPATCH_ARG_METHODS(int32_t)
    DISPATCH_ARG_METHODS(float)
    DISPATCH_ARG_METHODS(uint64_t)
    DISPATCH_ARG_METHODS(int64_t)
    DISPATCH_ARG_METHODS(double)

}} // namespace np::simd

#endif // __CYGWIN__
