/*@targets
 * $maxopt $keep_baseline avx512_icl avx512_spr
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "simd_qsort.hpp"
#ifndef __CYGWIN__

#if defined(NPY_HAVE_AVX512_SPR)
    #include "x86-simd-sort/src/avx512fp16-16bit-qsort.hpp"
/*
 * Wrapper function declarations to avoid multiple definitions of
 * avx512_qsort<uint16_t> and avx512_qsort<int16_t>
 */
void avx512_qsort_uint16(uint16_t*, intptr_t);
void avx512_qsort_int16(int16_t*, intptr_t);
void avx512_qselect_uint16(uint16_t*, npy_intp, npy_intp);
void avx512_qselect_int16(int16_t*, npy_intp, npy_intp);

#elif defined(NPY_HAVE_AVX512_ICL)
    #include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
/* Wrapper function defintions here: */
void avx512_qsort_uint16(uint16_t* arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
void avx512_qsort_int16(int16_t* arr, intptr_t size)
{
    avx512_qsort(arr, size);
}
void avx512_qselect_uint16(uint16_t* arr, npy_intp kth, npy_intp size)
{
    avx512_qselect(arr, kth, size, true);
}
void avx512_qselect_int16(int16_t* arr, npy_intp kth, npy_intp size)
{
    avx512_qselect(arr, kth, size, true);
}
#endif

namespace np { namespace qsort_simd {

/*
 * QSelect dispatch functions:
 */
#if defined(NPY_HAVE_AVX512_ICL) || defined(NPY_HAVE_AVX512_SPR)
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(Half *arr, npy_intp num, npy_intp kth)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qselect(reinterpret_cast<_Float16*>(arr), kth, num, true);
#else
    avx512_qselect_fp16(reinterpret_cast<uint16_t*>(arr), kth, num, true);
#endif
}

template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint16_t *arr, npy_intp num, npy_intp kth)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qselect_uint16(arr, kth, num);
#else
    avx512_qselect(arr, kth, num);
#endif
}

template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int16_t *arr, npy_intp num, npy_intp kth)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qselect_int16(arr, kth, num);
#else
    avx512_qselect(arr, kth, num);
#endif
}

/*
 * QSort dispatch functions:
 */
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
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qsort_uint16(arr, size);
#else
    avx512_qsort(arr, size);
#endif
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, intptr_t size)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qsort_int16(arr, size);
#else
    avx512_qsort(arr, size);
#endif
}
#endif // NPY_HAVE_AVX512_ICL || SPR

}} // namespace np::qsort_simd

#endif // __CYGWIN__
