/*@targets
 * $maxopt $keep_baseline
 * asimd
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "highway_qsort.hpp"
#ifndef __CYGWIN__

#if NPY_HAVE_ASIMD
    #define VQSORT_ONLY_STATIC 1
    #include "hwy/contrib/sort/vqsort-inl.h"
#endif

namespace np { namespace qsort_simd {

#if NPY_HAVE_ASIMD
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int32_t *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint32_t *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int64_t *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint64_t *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(float *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(double *arr, intptr_t size)
{
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
}
#endif // NPY_HAVE_ASIMD

}} // namespace np::simd

#endif // __CYGWIN__
