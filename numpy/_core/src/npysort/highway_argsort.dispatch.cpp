#include "highway_qsort.hpp"
#include "quicksort.hpp"
#include "common.hpp"

#if VQSORT_ENABLED
#define DISPATCH_VQSORT_u32(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(TYPE *arr, npy_intp *arg, npy_intp size) \
{ \
    if(size <= UINT32_MAX){ \
        hwy::K32V32* kv_pairs = new hwy::K32V32[size]; \
        for (npy_intp i = 0; i < size; i++) { \
            kv_pairs[i].key = arr[i]; \
            kv_pairs[i].value = static_cast<uint32_t>(i); \
        } \
        hwy::HWY_NAMESPACE::VQSortStatic(kv_pairs, static_cast<size_t>(size), hwy::SortAscending()); \
        for (npy_intp i = 0; i < size; i++) { \
            arg[i] = static_cast<npy_intp>(kv_pairs[i].value); \
        } \
        delete[] kv_pairs; \
    } \
    else{ \
        hwy::K64V64* kv_pairs = new hwy::K64V64[size]; \
        for (npy_intp i = 0; i < size; i++) { \
            kv_pairs[i].key = arr[i]; \
            kv_pairs[i].value = static_cast<uint64_t>(i); \
        } \
        hwy::HWY_NAMESPACE::VQSortStatic(kv_pairs, static_cast<size_t>(size), hwy::SortAscending()); \
        for (npy_intp i = 0; i < size; i++) { \
            arg[i] = static_cast<npy_intp>(kv_pairs[i].value); \
        } \
        delete[] kv_pairs; \
    }\
}

#define DISPATCH_VQSORT_u64(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(TYPE *arr, npy_intp *arg, npy_intp size) \
{ \
    hwy::K64V64* kv_pairs = new hwy::K64V64[size]; \
    for (npy_intp i = 0; i < size; i++) { \
        kv_pairs[i].key = arr[i]; \
        kv_pairs[i].value = static_cast<uint64_t>(i); \
    } \
    hwy::HWY_NAMESPACE::VQSortStatic(kv_pairs, static_cast<size_t>(size), hwy::SortAscending()); \
    for (npy_intp i = 0; i < size; i++) { \
        arg[i] = static_cast<npy_intp>(kv_pairs[i].value); \
    } \
    delete[] kv_pairs; \
}

#define AQUICKSORT(TYPE) \
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(TYPE *arr, npy_intp *arg, npy_intp size) \
{ \
    if constexpr (std::is_same_v<TYPE, int32_t>) { \
        np::sort::AQuick<npy::long_tag>(arr, arg, size); \
    } \
    else if constexpr (std::is_same_v<TYPE, int64_t>) { \
        np::sort::AQuick<npy::longlong_tag>(arr, arg, size); \
    } \
    else if constexpr (std::is_same_v<TYPE, float>) { \
        np::sort::AQuick<npy::float_tag>(arr, arg, size); \
    } \
    else if constexpr (std::is_same_v<TYPE, double>) { \
        np::sort::AQuick<npy::double_tag>(arr, arg, size); \
    } \
}

namespace np { namespace highway { namespace qsort_simd {
    DISPATCH_VQSORT_u32(uint32_t)
    DISPATCH_VQSORT_u64(uint64_t)
    AQUICKSORT(int32_t)
    AQUICKSORT(int64_t)
    AQUICKSORT(float)
    AQUICKSORT(double)
}}}  // np::highway::qsort_simd
#endif // VQSORT_ENABLED