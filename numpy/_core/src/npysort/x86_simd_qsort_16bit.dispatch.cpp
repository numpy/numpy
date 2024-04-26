#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#include "x86-simd-sort/src/x86simdsort-static-incl.h"
/*
 * MSVC doesn't set the macro __AVX512VBMI2__ which is required for the 16-bit
 * functions and therefore we need to manually include this file here
 */
#ifdef _MSC_VER
#include "x86-simd-sort/src/avx512-16bit-qsort.hpp"
#endif

namespace np { namespace qsort_simd {

/*
 * QSelect dispatch functions:
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(Half *arr, npy_intp num, npy_intp kth)
{
#if defined(NPY_HAVE_AVX512_SPR)
    x86simdsortStatic::qselect(reinterpret_cast<_Float16*>(arr), kth, num, true);
#else
    avx512_qselect_fp16(reinterpret_cast<uint16_t*>(arr), kth, num, true, false);
#endif
}

template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(uint16_t *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}

template<> void NPY_CPU_DISPATCH_CURFX(QSelect)(int16_t *arr, npy_intp num, npy_intp kth)
{
    x86simdsortStatic::qselect(arr, kth, num);
}

/*
 * QSort dispatch functions:
 */
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(Half *arr, npy_intp size)
{
#if defined(NPY_HAVE_AVX512_SPR)
    x86simdsortStatic::qsort(reinterpret_cast<_Float16*>(arr), size, true);
#else
    avx512_qsort_fp16(reinterpret_cast<uint16_t*>(arr), size, true, false);
#endif
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(uint16_t *arr, npy_intp size)
{
    x86simdsortStatic::qsort(arr, size);
}
template<> void NPY_CPU_DISPATCH_CURFX(QSort)(int16_t *arr, npy_intp size)
{
    x86simdsortStatic::qsort(arr, size);
}

}} // namespace np::qsort_simd

#endif // __CYGWIN__
