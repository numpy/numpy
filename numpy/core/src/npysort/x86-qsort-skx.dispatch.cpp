/*@targets
 * $maxopt $keep_baseline avx512_skx
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "x86-qsort-skx.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#ifdef NPY_HAVE_AVX512_SKX
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"

/***************************************
 * C > C++ dispatch
 ***************************************/
NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_long)(void *arr, npy_intp arrsize)
{
    avx512_qsort<int64_t>((int64_t*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_ulong)(void *arr, npy_intp arrsize)
{
    avx512_qsort<uint64_t>((uint64_t*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_double)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_double>((npy_double*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_int)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_int>((npy_int*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_uint)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_uint>((npy_uint*)arr, arrsize);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_float)(void *arr, npy_intp arrsize)
{
    avx512_qsort<npy_float>((npy_float*)arr, arrsize);
}

#endif  // NPY_HAVE_AVX512_SKX
