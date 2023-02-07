/*@targets
 * $maxopt $keep_baseline avx512_icl
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "simd_qsort.hpp"

#ifdef NPY_HAVE_AVX512_ICL
    #include "avx512-16bit-qsort.hpp"
#endif

namespace np { namespace qsort_simd {

#ifdef NPY_HAVE_AVX512_ICL
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, intptr_t size)
{
    avx512_qsort_fp16(reinterpret_cast<uint16_t*>(arr), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
#endif // NPY_HAVE_AVX512_ICL

}} // namespace np::qsort_simd
