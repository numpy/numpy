#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include <cmath>
#include "simd/simd.hpp"
#include <hwy/highway.h>
namespace {
using namespace np::simd;
#if NPY_HWY

/*******************************************************************************
 ** Defining the SIMD kernels
 ******************************************************************************/
template <typename T, typename SCALAR_F, typename HWY_F>
HWY_ATTR SIMD_MSVC_NOINLINE static void execute_simd_unrolled_loop(T* op, const T* ip, npy_intp len, SCALAR_F scalar_func, HWY_F hwy_func) {
    constexpr int UNROLL = 16;
    const hn::ScalableTag<T> d;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            auto v = hn::LoadU(d, ip + vstep * i);
            hn::StoreU(hwy_func(v), d, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        hn::StoreU(hwy_func(hn::LoadU(d, ip)), d, op);
    }

    // Scalar loop to finish off
    for (; len > 0; len--, ip++, op++) {
        *op = scalar_func(*ip);
    }
}

template <typename T, typename SCALAR_F, typename HWY_F>
HWY_ATTR SIMD_MSVC_NOINLINE static void execute_simd_unrolled_loop_mul(T* op, const T* ip, npy_intp len, SCALAR_F scalar_func, HWY_F hwy_func) {
    constexpr int UNROLL = 16;
    const hn::ScalableTag<T> d;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            auto v = hn::LoadU(d, ip + vstep * i);
            hn::StoreU(hwy_func(v, v), d, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        auto v = hn::LoadU(d, ip);
        hn::StoreU(hwy_func(v, v), d, op);
    }

    // Scalar loop to finish off
    for (; len > 0; len--, ip++, op++) {
        *op = scalar_func(*ip);
    }
}

template <typename T, typename SCALAR_F, typename HWY_F>
HWY_ATTR SIMD_MSVC_NOINLINE static void execute_simd_unrolled_loop_div(T* op, const T* ip, npy_intp len, SCALAR_F scalar_func, HWY_F hwy_func) {
    constexpr int UNROLL = 16;
    const hn::ScalableTag<T> d;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        auto one = hn::Set(d, static_cast<T>(1.0));
        for (int i = 0; i < UNROLL; i++) {
            auto v = hn::LoadU(d, ip + vstep * i);
            hn::StoreU(hwy_func(one, v), d, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        auto v = hn::LoadU(d, ip);
        hn::StoreU(hwy_func(hn::Set(d, static_cast<T>(1.0)), v), d, op);
    }

    // Scalar loop to finish off
    for (; len > 0; len--, ip++, op++) {
        *op = scalar_func(*ip);
    }
}

template <typename T> static void simd_unary_loop_absolute(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::abs(v); }, [](auto v) { return hn::Abs(v); });
}

template <typename T> static void simd_unary_loop_square(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop_mul(op, ip, len, [](T v) { return v * v; }, [](auto v1, auto v2) { return hn::Mul(v1, v2); });
}

template <typename T> static void simd_unary_loop_reciprocal(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop_div(op, ip, len, [](T v) { return static_cast<T>(1.0) / v; }, [](auto v1, auto v2) { return hn::Div(v1, v2); });
}

template <typename T> static void simd_unary_loop_sqrt(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::sqrt(v); }, [](auto v) { return hn::Sqrt(v); });
}

template <typename T> static void simd_unary_loop_floor(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::floor(v); }, [](auto v) { return hn::Floor(v); });
}

template <typename T> static void simd_unary_loop_ceil(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::ceil(v); }, [](auto v) { return hn::Ceil(v); });
}

template <typename T> static void simd_unary_loop_trunc(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::trunc(v); }, [](auto v) { return hn::Trunc(v); });
}

template <typename T> static void simd_unary_loop_rint(T* op, const T* ip, npy_intp len) {
    execute_simd_unrolled_loop(op, ip, len, [](T v) { return std::rint(v); }, [](auto v) { return hn::Round(v); });
}

#endif
template <typename T, void (*SIMD_FUNC)(T*, const T*, npy_intp), T (*SCALAR_FUNC)(T)>
static void execute_unary_ufunc(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    T *ip = reinterpret_cast<T*>(args[0]);
    T *op = reinterpret_cast<T*>(args[1]);
    npy_intp is = steps[0];
    npy_intp os = steps[1];
    npy_intp n = dimensions[0];
#if NPY_HWY
    if (is == sizeof(T) && os == sizeof(T)) {
        SIMD_FUNC(op, ip, n);
        return;
    }
#endif
    for (npy_intp i = 0; i < n; i++, ip = reinterpret_cast<T*>(reinterpret_cast<char*>(ip) + is), op = reinterpret_cast<T*>(reinterpret_cast<char*>(op) + os)) {
        *op = SCALAR_FUNC(*ip);
    }
}
template <typename T> static HWY_INLINE T scalar_abs(T v) { return std::abs(v); }
template <typename T> static HWY_INLINE T scalar_sq(T v) { return v * v; }
template <typename T> static HWY_INLINE T scalar_rec(T v) { return static_cast<T>(1.0) / v; }
template <typename T> static HWY_INLINE T scalar_sqrt(T v) { return std::sqrt(v); }
template <typename T> static HWY_INLINE T scalar_flr(T v) { return std::floor(v); }
template <typename T> static HWY_INLINE T scalar_cel(T v) { return std::ceil(v); }
template <typename T> static HWY_INLINE T scalar_trn(T v) { return std::trunc(v); }
template <typename T> static HWY_INLINE T scalar_rnt(T v) { return std::rint(v); }
} // namespace anonymous


/*******************************************************************************
 ** Defining ufunc inner functions
 ******************************************************************************/
#if NPY_HWY
#define EXPAND_SIMD_FUNC(kind) simd_unary_loop_##kind
#else
#define EXPAND_SIMD_FUNC(kind) nullptr
#endif
#define EXPAND_SIMD_FUNC_signbit nullptr
#define NPY_CREATE_UNARY_UFUNC(TYPE, ctype, kind, SCALAR, IS_ABS) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##kind)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    execute_unary_ufunc<ctype, EXPAND_SIMD_FUNC(kind), SCALAR>(args, dimensions, steps); \
    if (sizeof(ctype) == sizeof(float) && IS_ABS) { \
        npy_clear_floatstatus_barrier(reinterpret_cast<char*>(const_cast<npy_intp*>(dimensions))); \
    } \
}
NPY_CREATE_UNARY_UFUNC(FLOAT, float, absolute, scalar_abs, 1)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, square, scalar_sq, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, reciprocal, scalar_rec, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, sqrt, scalar_sqrt, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, floor, scalar_flr, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, ceil, scalar_cel, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, trunc, scalar_trn, 0)
NPY_CREATE_UNARY_UFUNC(FLOAT, float, rint, scalar_rnt, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, absolute, scalar_abs, 1)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, square, scalar_sq, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, reciprocal, scalar_rec, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, sqrt, scalar_sqrt, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, floor, scalar_flr, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, ceil, scalar_cel, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, trunc, scalar_trn, 0)
NPY_CREATE_UNARY_UFUNC(DOUBLE, double, rint, scalar_rnt, 0)
extern "C" {
#ifndef NPY_DISABLE_OPTIMIZATION
#include "loops_unary_fp.dispatch.h"
#endif
}
