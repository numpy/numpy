#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort.hpp"

namespace np::highway::qsort_simd {
template <>
void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half *arr, size_t size)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<hwy::float16_t*>(arr), size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}
template <>
bool NPY_CPU_DISPATCH_CURFX(QSelect)<Half>(Half *arr, size_t num, size_t kth)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSelectStatic(reinterpret_cast<hwy::float16_t*>(arr), num, kth, hwy::SortAscending());
    return true;
#else
    return false;
#endif
}
} // np::highway::qsort_simd
