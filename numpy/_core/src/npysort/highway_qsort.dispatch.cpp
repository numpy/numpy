#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}

template void NPY_CPU_DISPATCH_CURFX(QSort)<int32_t>(int32_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint32_t>(uint32_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<int64_t>(int64_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint64_t>(uint64_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<float>(float*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<double>(double*, npy_intp);

} // np::highway::qsort_simd

