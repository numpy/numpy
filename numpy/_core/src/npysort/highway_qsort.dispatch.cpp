#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, size_t size)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}
template <typename T>
bool NPY_CPU_DISPATCH_CURFX(QSelect)(T *arr, size_t num, size_t kth)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSelectStatic(arr, num, kth, hwy::SortAscending());
    return true;
#else
    return false;
#endif
}

template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<int32_t>(int32_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint32_t>(uint32_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<int64_t>(int64_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint64_t>(uint64_t*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<float>(float*, size_t);
template void NPY_CPU_DISPATCH_CURFX(QSort)<double>(double*, size_t);

template bool NPY_CPU_DISPATCH_CURFX(QSelect)<int16_t>(int16_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<uint16_t>(uint16_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<int32_t>(int32_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<uint32_t>(uint32_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<int64_t>(int64_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<uint64_t>(uint64_t*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<float>(float*, size_t, size_t);
template bool NPY_CPU_DISPATCH_CURFX(QSelect)<double>(double*, size_t, size_t);
} // np::highway::qsort_simd

