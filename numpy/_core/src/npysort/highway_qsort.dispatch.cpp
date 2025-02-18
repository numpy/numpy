#include "highway_qsort.hpp"

#if VQSORT_ENABLED

#define DISPATCH_VQSORT(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(TYPE *arr, intptr_t size) \
{ \
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending()); \
} \

namespace np { namespace highway { namespace qsort_simd {

    DISPATCH_VQSORT(int32_t)
    DISPATCH_VQSORT(uint32_t)
    DISPATCH_VQSORT(int64_t)
    DISPATCH_VQSORT(uint64_t)
    DISPATCH_VQSORT(double)
    DISPATCH_VQSORT(float)

} } } // np::highway::qsort_simd

#endif // VQSORT_ENABLED
