/*
 * Provides wrappers around C11 standard library atomics and MSVC intrinsics
 * to provide basic atomic load and store functionality. This is based on
 * code in CPython's pyatomic.h, pyatomic_std.h, and pyatomic_msc.h
 */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_ATOMIC_H_
#define NUMPY_CORE_SRC_COMMON_NPY_ATOMIC_H_

#include "numpy/npy_common.h"

#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
// TODO: support C++ atomics as well if this header is ever needed in C++
    #include <stdatomic.h>
    #include <stdint.h>
    #define STDC_ATOMICS
#elif _MSC_VER
    #include <intrin.h>
    #define MSC_ATOMICS
#else
    #error "no support for missing C11 atomics except with MSVC"
#endif


static inline npy_uint8 npy_atomic_load_uint8(const npy_uint8 *obj) {
#ifdef STDC_ATOMICS
    return (npy_uint8)atomic_load((const _Atomic(uint8_t)*)obj);
#elif defined(MSC_ATOMICS)
#if defined(_M_X64) || defined(_M_IX86)
    return *(volatile npy_uint8 *)obj;
#elif defined(_M_ARM64)
    return (npy_uint8)__ldar8((unsigned __int8 volatile *)obj);
#else
#error "Unsupported MSVC build configuration, neither x86 or ARM"
#endif
#endif
}

#undef MSC_ATOMICS
#undef STDC_ATOMICS

#endif // NUMPY_CORE_SRC_COMMON_NPY_NPY_ATOMIC_H_
