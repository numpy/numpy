#include "highway_qsort.hpp"
#define VQSORT_ONLY_STATIC 1
#include "hwy/contrib/sort/vqsort-inl.h"

namespace np { namespace highway { namespace qsort_simd {

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

} } } // np::highway::qsort_simd
