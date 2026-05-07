#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort_generic.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size, bool descending)
{
#if VQSORT_ENABLED
    using THwy = std::conditional_t<std::is_same_v<T, Half>, hwy::float16_t, T>;
    if (descending) {
        hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortDescending());
    } else {
        hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortAscending());
    }
#endif
}

#if VQSORT_ENABLED
template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, npy_intp, bool);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, npy_intp, bool);
#endif

} // np::highway::qsort_simd
