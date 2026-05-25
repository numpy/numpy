#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort_generic.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSelect)(T *arr, npy_intp num, npy_intp kth)
{
#if VQSORT_ENABLED
    // Guard the npy_intp (signed) → size_t (unsigned) casts below.
    assert(num > 0 && kth >= 0 && kth < num);
    hwy::HWY_NAMESPACE::VQSelectStatic(arr,
                                       static_cast<size_t>(num),
                                       static_cast<size_t>(kth),
                                       hwy::SortAscending());
#else
    // VQSORT_ENABLED can be 0 on a registered target (e.g. aarch64 ASAN, or a
    // HWY_DISABLE_VQSORT build).  Match the sibling sort dispatch and degrade
    // to a scalar sort rather than fail to compile: a fully sorted array
    // satisfies partition's postcondition at any kth.  (void)kth: unused here.
    (void)kth;
    sort::Quick<false>(arr, num);
#endif
}

// 16-bit integers (int16_t/uint16_t) are plain integers covered by base ASIMD,
// so they share this dispatch.  Half (float16) is omitted: selection.cpp
// excludes it (scalar introselect handles float16 partition).
template void NPY_CPU_DISPATCH_CURFX(QSelect)<int16_t>(int16_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<uint16_t>(uint16_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<int32_t>(int32_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<uint32_t>(uint32_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<int64_t>(int64_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<uint64_t>(uint64_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<float>(float*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<double>(double*, npy_intp, npy_intp);

} // np::highway::qsort_simd
