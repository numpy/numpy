/*@targets
 * $maxopt $keep_baseline avx512_icl
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "x86-qsort-icl.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#ifdef NPY_HAVE_AVX512_ICL
#include "avx512-16bit-qsort.hpp"

/***************************************
 * C > C++ dispatch
 ***************************************/
NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_half)(void *arr, npy_intp arrsize)
{
    avx512_qsort_fp16((npy_half*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_short)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_short>((npy_short*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_ushort)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_ushort>((npy_ushort*)arr, arrsize);
}

#endif  // NPY_HAVE_AVX512_ICL
