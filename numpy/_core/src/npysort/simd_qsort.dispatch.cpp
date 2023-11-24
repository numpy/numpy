/*@targets
 * $maxopt $keep_baseline
 * avx512_skx
 * asimd
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "simd_qsort.hpp"
#ifndef __CYGWIN__

#define USE_HIGHWAY defined(__aarch64__)

#if defined(NPY_HAVE_AVX512_SKX)
    #include "x86-simd-sort/src/avx512-32bit-qsort.hpp"
    #include "x86-simd-sort/src/avx512-64bit-qsort.hpp"
#elif USE_HIGHWAY
    #define VQSORT_ONLY_STATIC 1
    #include "hwy/contrib/sort/vqsort-inl.h"
#endif

namespace np { namespace qsort_simd {

#if defined(NPY_HAVE_AVX512_SKX)
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int32_t *arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint32_t *arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int64_t*arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint64_t*arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(float *arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(double *arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int32_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint32_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int64_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint64_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(float *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(double *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
#elif USE_HIGHWAY
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
#endif

}} // namespace np::simd

#endif // __CYGWIN__
