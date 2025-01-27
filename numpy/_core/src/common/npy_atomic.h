/*
 * Provides wrappers around C11 standard library atomics and MSVC intrinsics
 * to provide basic atomic load and store functionality. This is based on
 * code in CPython's pyatomic.h, pyatomic_std.h, and pyatomic_msc.h
 */

#ifndef NUMPY_CORE_SRC_COMMON_NPY_ATOMIC_H_
#define NUMPY_CORE_SRC_COMMON_NPY_ATOMIC_H_

#include "numpy/npy_common.h"

#ifdef __cplusplus
    extern "C++" {
        #include <atomic>
    }
    #define _NPY_USING_STD using namespace std
    #define _Atomic(tp) atomic<tp>
    #define STDC_ATOMICS
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L \
    && !defined(__STDC_NO_ATOMICS__)
    #include <stdatomic.h>
    #include <stdint.h>
    #define _NPY_USING_STD
    #define STDC_ATOMICS
#elif _MSC_VER
    #include <intrin.h>
    #define MSC_ATOMICS
    #if !defined(_M_X64) && !defined(_M_IX86) && !defined(_M_ARM64)
        #error "Unsupported MSVC build configuration, neither x86 or ARM"
    #endif
#elif defined(__GNUC__) && (__GNUC__ > 4)
    #define GCC_ATOMICS
#elif defined(__clang__)
    #if __has_builtin(__atomic_load)
        #define GCC_ATOMICS
    #endif
#else
    #error "no supported atomic implementation for this platform/compiler"
#endif


static inline npy_uint8
npy_atomic_load_uint8(const npy_uint8 *obj) {
#ifdef STDC_ATOMICS
    _NPY_USING_STD;
    return (npy_uint8)atomic_load((const _Atomic(uint8_t)*)obj);
#elif defined(MSC_ATOMICS)
#if defined(_M_X64) || defined(_M_IX86)
    return *(volatile npy_uint8 *)obj;
#else // defined(_M_ARM64)
    return (npy_uint8)__ldar8((unsigned __int8 volatile *)obj);
#endif
#elif defined(GCC_ATOMICS)
    return __atomic_load_n(obj, __ATOMIC_SEQ_CST);
#endif
}

static inline void*
npy_atomic_load_ptr(const void *obj) {
#ifdef STDC_ATOMICS
    _NPY_USING_STD;
    return atomic_load((const _Atomic(void *)*)obj);
#elif defined(MSC_ATOMICS)
#if SIZEOF_VOID_P == 8
#if defined(_M_X64) || defined(_M_IX86)
    return (void *)*(volatile uint64_t *)obj;
#elif defined(_M_ARM64)
    return (void *)__ldar64((unsigned __int64 volatile *)obj);
#endif
#else
#if defined(_M_X64) || defined(_M_IX86)
    return (void *)*(volatile uint32_t *)obj;
#elif defined(_M_ARM64)
    return (void *)__ldar32((unsigned __int32 volatile *)obj);
#endif
#endif
#elif defined(GCC_ATOMICS)
    return (void *)__atomic_load_n((void * const *)obj, __ATOMIC_SEQ_CST);
#endif
}

static inline void
npy_atomic_store_uint8(npy_uint8 *obj, npy_uint8 value) {
#ifdef STDC_ATOMICS
    _NPY_USING_STD;
    atomic_store((_Atomic(uint8_t)*)obj, value);
#elif defined(MSC_ATOMICS)
    _InterlockedExchange8((volatile char *)obj, (char)value);
#elif defined(GCC_ATOMICS)
    __atomic_store_n(obj, value, __ATOMIC_SEQ_CST);
#endif
}

static inline void
npy_atomic_store_ptr(void *obj, void *value)
{
#ifdef STDC_ATOMICS
    _NPY_USING_STD;
    atomic_store((_Atomic(void *)*)obj, value);
#elif defined(MSC_ATOMICS)
    _InterlockedExchangePointer((void * volatile *)obj, (void *)value);
#elif defined(GCC_ATOMICS)
    __atomic_store_n((void **)obj, value, __ATOMIC_SEQ_CST);
#endif
}

#undef MSC_ATOMICS
#undef STDC_ATOMICS
#undef GCC_ATOMICS

#endif // NUMPY_CORE_SRC_COMMON_NPY_NPY_ATOMIC_H_
