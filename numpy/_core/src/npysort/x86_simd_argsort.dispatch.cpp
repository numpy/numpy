#include "x86_simd_qsort.hpp"
#ifndef __CYGWIN__

#if defined(NPY_HAVE_AVX512_SKX)
#include "x86-simd-sort/src/avx512-64bit-argsort.hpp"
#elif defined(NPY_HAVE_AVX2)
#include "x86-simd-sort/src/avx2-32bit-half.hpp"
#include "x86-simd-sort/src/avx2-32bit-qsort.hpp"
#include "x86-simd-sort/src/avx2-64bit-qsort.hpp"
#include "x86-simd-sort/src/xss-common-argsort.h"
#endif

namespace {
template<typename T>
void x86_argsort(T* arr, size_t* arg, npy_intp num)
{
#if defined(NPY_HAVE_AVX512_SKX)
    avx512_argsort(arr, arg, num, true);
#elif defined(NPY_HAVE_AVX2)
    avx2_argsort(arr, arg, num, true);
#endif
}

template<typename T>
void x86_argselect(T* arr, size_t* arg, npy_intp kth, npy_intp num)
{
#if defined(NPY_HAVE_AVX512_SKX)
    avx512_argselect(arr, arg, kth, num, true);
#elif defined(NPY_HAVE_AVX2)
    avx2_argselect(arr, arg, kth, num, true);
#endif
}
} // anonymous

namespace np { namespace qsort_simd {

template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(int32_t *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(uint32_t *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(int64_t*arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(uint64_t*arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(float *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(double *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    x86_argselect(arr, reinterpret_cast<size_t*>(arg), kth, num);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(int32_t *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(uint32_t *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(int64_t *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(uint64_t *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(float *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}
template<> void NPY_CPU_DISPATCH_CURFX(ArgQSort)(double *arr, npy_intp *arg, npy_intp size)
{
    x86_argsort(arr, reinterpret_cast<size_t*>(arg), size);
}

}} // namespace np::simd

#endif // __CYGWIN__
