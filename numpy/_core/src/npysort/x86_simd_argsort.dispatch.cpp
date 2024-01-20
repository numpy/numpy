#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#if defined(NPY_HAVE_AVX512_SKX)
    #include "x86-simd-sort/src/avx512-64bit-argsort.hpp"
#endif

namespace np { namespace qsort_simd {

/* arg methods currently only have AVX-512 versions */
#if defined(NPY_HAVE_AVX512_SKX)
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(int32_t *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(uint32_t *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(int64_t*arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(uint64_t*arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(float *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(double *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    avx512_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(int32_t *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(uint32_t *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(int64_t *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(uint64_t *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(float *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size, true);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(double *arr, npy_intp *arg, npy_intp size)
{
    avx512_argsort(arr, reinterpret_cast<size_t*>(arg), size, true);
}
#endif  // NPY_HAVE_AVX512_SKX

}} // namespace np::simd

#endif // __CYGWIN__
