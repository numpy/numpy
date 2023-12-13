#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#if defined(NPY_HAVE_AVX512_SPR)
    #include "x86-simd-sort/src/avx512fp16-16bit-qsort.hpp"
    #include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
#elif defined(NPY_HAVE_AVX512_ICL)
    #include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
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
    avx512_qselect(arr, kth, num);
}

template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int16_t *arr, npy_intp num, npy_intp kth)
{
    avx512_qselect(arr, kth, num);
}

/*
 * QSort dispatch functions:
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, npy_intp size)
{
#if defined(NPY_HAVE_AVX512_SPR)
    avx512_qsort(reinterpret_cast<_Float16*>(arr), size, true);
#else
    avx512_qsort_fp16(reinterpret_cast<uint16_t*>(arr), size, true);
#endif
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, npy_intp size)
{
    avx512_qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, npy_intp size)
{
    avx512_qsort(arr, size);
}
#endif // NPY_HAVE_AVX512_ICL || SPR

}} // namespace np::qsort_simd

#endif // __CYGWIN__
