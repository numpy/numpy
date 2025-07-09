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
    using THwy = std::conditional_t<std::is_same_v<T, Half>, hwy::float16_t, T>;
    hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}
#if !HWY_HAVE_FLOAT16
template <>
void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half *arr, npy_intp size)
{
    sort::Quick(arr, size);
}
#endif // !HWY_HAVE_FLOAT16

template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, npy_intp);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half*, npy_intp);
#endif

} // np::highway::qsort_simd
