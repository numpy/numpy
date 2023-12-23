/*@targets
 * $maxopt $keep_baseline avx512_icl avx512_spr
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "simd_qsort.hpp"

#if defined(NPY_HAVE_AVX512_SPR) && !defined(_MSC_VER)
    #include "x86-simd-sort/src/avx512fp16-16bit-qsort.hpp"
    #include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
#elif defined(NPY_HAVE_AVX512_ICL) && !defined(_MSC_VER)
    #include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
#endif

namespace np { namespace qsort_simd {

#if !defined(_MSC_VER)
#if defined(NPY_HAVE_AVX512_ICL) || defined(NPY_HAVE_AVX512_SPR)
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, intptr_t size)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qsort(reinterpret_cast<_Float16*>(arr), size);
#else
    avx512_qsort_fp16(reinterpret_cast<uint16_t*>(arr), size);
#endif
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
#endif // NPY_HAVE_AVX512_ICL || SPR
#endif // _MSC_VER

}} // namespace np::qsort_simd
