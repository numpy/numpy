#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#include "x86-simd-sort/src/x86simdsort-static-incl.h"

namespace np { namespace qsort_simd {
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int32_t *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint32_t *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int64_t*arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint64_t*arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(float *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(double *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int32_t *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint32_t *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int64_t *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint64_t *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(float *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(double *arr, npy_intp num)
{
    x86simdsortStatic::qsort(arr, num, true);
}
}} // namespace np::qsort_simd

#endif // __CYGWIN__
