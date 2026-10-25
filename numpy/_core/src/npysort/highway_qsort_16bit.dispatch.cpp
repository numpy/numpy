#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort_generic.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size, bool reverse)
{
#if VQSORT_ENABLED
    using THwy = std::conditional_t<std::is_same_v<T, Half>, hwy::float16_t, T>;
    if (reverse) {
        hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortDescending());
    }
    else {
        hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortAscending());
    }
#else
    if (reverse) {
        sort::Quick<true>(arr, size);
    }
    else {
        sort::Quick<false>(arr, size);
    }
#endif
}

#if !HWY_HAVE_FLOAT16
// Highway's float16 vector sort isn't compiled in; provide a scalar
// specialization so ``Half`` still has a working symbol at link time and
// the primary template body above doesn't try to instantiate
// ``VQSortStatic<hwy::float16_t>``.
template <>
void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half *arr, npy_intp size, bool reverse)
{
    if (reverse) {
        sort::Quick<true>(arr, size);
    }
    else {
        sort::Quick<false>(arr, size);
    }
}
#endif // !HWY_HAVE_FLOAT16

template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, npy_intp, bool);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, npy_intp, bool);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half*, npy_intp, bool);
#endif

} // np::highway::qsort_simd
