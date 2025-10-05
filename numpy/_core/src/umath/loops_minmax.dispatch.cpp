#include "loops_utils.h"
#include "loops.h"

#include <hwy/highway.h>
#include <hwy/cache_control.h>
#include "simd/simd.hpp"
#include "numpy/npy_common.h"
#include "common.hpp"
#include "fast_loop_macros.h"

namespace {
using namespace np::simd;

template <typename T> struct OpMax {
    using Degraded = std::conditional_t<std::is_same_v<T, long double>, OpMax<double>, OpMax<T>>;
#if NPY_HWY
    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& a, const V& b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return hn::IfThenElse(hn::IsEitherNaN(a, b), Set<T>(NAN), hn::Max(a, b));
        } else {
            return hn::Max(a, b);
        }
    }

    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        if constexpr (std::is_floating_point_v<T>) {
            return hn::AllFalse(_Tag<T>(), hn::IsNaN(v)) ? hn::ReduceMax(_Tag<T>(), v) : NAN;
        } else {
            return hn::ReduceMax(_Tag<T>(), v);
        }
    }
#endif

    NPY_INLINE T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return (a >= b || npy_isnan(a)) ? a : b;
        } else {
            return a > b ? a : b;
        }
    }
};

template <typename T> struct OpMin {
    using Degraded = std::conditional_t<std::is_same_v<T, long double>, OpMin<double>, OpMin<T>>;
#if NPY_HWY
    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& a, const V& b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return hn::IfThenElse(hn::IsEitherNaN(a, b), Set<T>(NAN), hn::Min(a, b));
        } else {
            return hn::Min(a, b);
        }
    }

    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        if constexpr (std::is_floating_point_v<T>) {
            return hn::AllFalse(_Tag<T>(), hn::IsNaN(v)) ? hn::ReduceMin(_Tag<T>(), v) : NAN;
        } else {
            return hn::ReduceMin(_Tag<T>(), v);
        }
    }
#endif

    NPY_INLINE T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return (a <= b || npy_isnan(a)) ? a : b;
        } else {
            return a < b ? a : b;
        }
    }
};

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>> struct OpMaxp {
    using Degraded = std::conditional_t<std::is_same_v<T, long double>, OpMaxp<double>, OpMaxp<T>>;
#if NPY_HWY
    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& a, const V& b) const {
        return hn::Max(hn::IfThenElse(hn::IsNaN(a), b, a), hn::IfThenElse(hn::IsNaN(b), a, b));
    }

    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        auto m = hn::IsNaN(v);
        return hn::AllTrue(_Tag<T>(), m) ? NAN : hn::MaskedReduceMax(_Tag<T>(), hn::Not(m), v);
    }
#endif

    NPY_INLINE T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, float>) {
            return fmaxf(a, b);
        } else if constexpr (std::is_same_v<T, double>) {
            return fmax(a, b);
        } else {
            return fmaxl(a, b);
        }
    }
};

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>> struct OpMinp {
    using Degraded = std::conditional_t<std::is_same_v<T, long double>, OpMinp<double>, OpMinp<T>>;
#if NPY_HWY
    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& a, const V& b) const {
        return hn::Min(hn::IfThenElse(hn::IsNaN(a), b, a), hn::IfThenElse(hn::IsNaN(b), a, b));
    }

    template <typename D = T, typename = std::enable_if_t<kSupportLane<D>>, typename V = Vec<D>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        auto m = hn::IsNaN(v);
        return hn::AllTrue(_Tag<T>(), m) ? NAN : hn::MaskedReduceMin(_Tag<T>(), hn::Not(m), v);
    }
#endif

    NPY_INLINE T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, float>) {
            return fminf(a, b);
        } else if constexpr (std::is_same_v<T, double>) {
            return fmin(a, b);
        } else {
            return fminl(a, b);
        }
    }
};

#if NPY_HWY
template <typename T, typename D = std::conditional_t<(sizeof(T) <= 4), int32_t, int64_t>>
HWY_INLINE HWY_ATTR auto LoadWithStride(const T* src, npy_intp ssrc) {
    auto index = hn::Mul(hn::Iota(_Tag<D>(), 0), Set<D>(ssrc));
    return hn::GatherIndex(_Tag<T>(), src, index);
}

template <typename T, typename D = std::conditional_t<(sizeof(T) <= 4), int32_t, int64_t>>
HWY_INLINE HWY_ATTR void StoreWithStride(Vec<T> vec, T* dst, npy_intp sdst) {
    auto index = hn::Mul(hn::Iota(_Tag<D>(), 0), Set<D>(sdst));
    hn::ScatterIndex(vec, _Tag<T>(), dst, index);
}

/********************************************************************************
 ** Defining the SIMD kernels
 ********************************************************************************/

// contiguous input.
template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
simd_reduce_c(const T* ip, T* op1, npy_intp len)
{
    const OP op_func;
    if (len < 1) {
        return;
    }
    const int vstep = Lanes<T>();
    const int wstep = vstep*8;
    auto acc = Set<T>(op1[0]);
    for (; len >= wstep; len -= wstep, ip += wstep) {
        /*
         * error: '_mm_prefetch' needs target feature mmx on clang-cl
         */
#if !(defined(_MSC_VER) && defined(__clang__))
        hwy::Prefetch(ip + wstep);
#endif
        auto v0 = LoadU(ip + vstep * 0);
        auto v1 = LoadU(ip + vstep * 1);
        auto v2 = LoadU(ip + vstep * 2);
        auto v3 = LoadU(ip + vstep * 3);

        auto v4 = LoadU(ip + vstep * 4);
        auto v5 = LoadU(ip + vstep * 5);
        auto v6 = LoadU(ip + vstep * 6);
        auto v7 = LoadU(ip + vstep * 7);

        auto r01 = op_func(v0, v1);
        auto r23 = op_func(v2, v3);
        auto r45 = op_func(v4, v5);
        auto r67 = op_func(v6, v7);
        acc = op_func(acc, op_func(op_func(r01, r23), op_func(r45, r67)));
    }
    for (; len >= vstep; len -= vstep, ip += vstep) {
        acc = op_func(acc, LoadU(ip));
    }
    T r = op_func(acc);
    // Scalar - finish up any remaining iterations
    for (; len > 0; --len, ++ip) {
        const T in2 = *ip;
        r = op_func(r, in2);
    }
    op1[0] = r;
}

// contiguous inputs and output.
template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
simd_binary_ccc(const T*ip1, const T*ip2,
                      T*op1, npy_intp len)
{
    const OP op_func;
#if HWY_MAX_BYTES == 16
    // Note, 6x unroll was chosen for best results on Apple M1
    const int vectorsPerLoop = 6;
#else
    // To avoid memory bandwidth bottleneck
    const int vectorsPerLoop = 2;
#endif
    const int elemPerVector = Lanes<T>();
    int elemPerLoop = vectorsPerLoop * elemPerVector;

    npy_intp i = 0;

    for (; (i+elemPerLoop) <= len; i += elemPerLoop) {
        auto v0 = LoadU(&ip1[i + 0 * elemPerVector]);
        auto v1 = LoadU(&ip1[i + 1 * elemPerVector]);
    #if HWY_MAX_BYTES == 16
        auto v2 = LoadU(&ip1[i + 2 * elemPerVector]);
        auto v3 = LoadU(&ip1[i + 3 * elemPerVector]);
        auto v4 = LoadU(&ip1[i + 4 * elemPerVector]);
        auto v5 = LoadU(&ip1[i + 5 * elemPerVector]);
    #endif
        auto u0 = LoadU(&ip2[i + 0 * elemPerVector]);
        auto u1 = LoadU(&ip2[i + 1 * elemPerVector]);
    #if HWY_MAX_BYTES == 16
        auto u2 = LoadU(&ip2[i + 2 * elemPerVector]);
        auto u3 = LoadU(&ip2[i + 3 * elemPerVector]);
        auto u4 = LoadU(&ip2[i + 4 * elemPerVector]);
        auto u5 = LoadU(&ip2[i + 5 * elemPerVector]);
    #endif
        auto m0 = op_func(v0, u0);
        auto m1 = op_func(v1, u1);
    #if HWY_MAX_BYTES == 16
        auto m2 = op_func(v2, u2);
        auto m3 = op_func(v3, u3);
        auto m4 = op_func(v4, u4);
        auto m5 = op_func(v5, u5);
    #endif
        StoreU(m0, &op1[i + 0 * elemPerVector]);
        StoreU(m1, &op1[i + 1 * elemPerVector]);
    #if HWY_MAX_BYTES == 16
        StoreU(m2, &op1[i + 2 * elemPerVector]);
        StoreU(m3, &op1[i + 3 * elemPerVector]);
        StoreU(m4, &op1[i + 4 * elemPerVector]);
        StoreU(m5, &op1[i + 5 * elemPerVector]);
    #endif
    }
    for (; (i+elemPerVector) <= len; i += elemPerVector) {
        auto v0 = LoadU(ip1 + i);
        auto u0 = LoadU(ip2 + i);
        auto m0 = op_func(v0, u0);
        StoreU(m0, op1 + i);
    }
    // Scalar - finish up any remaining iterations
    for (; i < len; ++i) {
        const T in1 = ip1[i];
        const T in2 = ip2[i];
        op1[i] = op_func(in1, in2);
    }
}

// non-contiguous for float 32/64-bit memory access
template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
simd_binary(const T* ip1, npy_intp sip1,
            const T* ip2, npy_intp sip2,
                  T* op1, npy_intp sop1,
                  npy_intp len)
{
    const OP op_func;
    const int vstep = Lanes<T>();
    for (; len >= vstep; len -= vstep, ip1 += sip1*vstep,
                         ip2 += sip2*vstep, op1 += sop1*vstep
    ) {
        Vec<T> a, b;
        if (sip1 == 1) {
            a = LoadU(ip1);
        } else {
            a = LoadWithStride(ip1, sip1);
        }
        if (sip2 == 1) {
            b = LoadU(ip2);
        } else {
            b = LoadWithStride(ip2, sip2);
        }
        auto r = op_func(a, b);
        if (sop1 == 1) {
            StoreU(r, op1);
        } else {
            StoreWithStride(r, op1, sop1);
        }
    }
    for (; len > 0; --len, ip1 += sip1, ip2 += sip2, op1 += sop1) {
        const T a = *ip1;
        const T b = *ip2;
        *op1 = op_func(a, b);
    }
}
#endif // NPY_HWY

template <typename T, typename OP, typename D = std::conditional_t<std::is_same_v<T, long double>, double, T>>
HWY_INLINE HWY_ATTR void
minmax(char **args, npy_intp const*dimensions, npy_intp const*steps)
{
    const OP op_func;
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2],
             len = dimensions[0];
    npy_intp i = 0;
#if NPY_HWY
    if constexpr (kSupportLane<T>) {
        if (IS_BINARY_REDUCE) {
            // reduce and contiguous
            if (is2 == sizeof(T)) {
                simd_reduce_c<D, typename OP::Degraded>(
                    (D*)ip2, (D*)op1, len
                );
                goto clear_fp;
            }
        }
        else if (!is_mem_overlap(ip1, is1, op1, os1, len) &&
                 !is_mem_overlap(ip2, is2, op1, os1, len)
        ) {
            // no overlap and operands are binary contiguous
            if (IS_BINARY_CONT(T, T)) {
                simd_binary_ccc<D, typename OP::Degraded>(
                    (D*)ip1, (D*)ip2, (D*)op1, len
                );
                goto clear_fp;
            }
        // unroll scalars faster than non-contiguous vector load/store on Arm
        #if !HWY_TARGET_IS_NEON
            if constexpr (std::is_floating_point_v<T>) {
                if (alignof(T) == sizeof(T) && is1 % sizeof(T) == 0 && is2 % sizeof(T) == 0 && os1 % sizeof(T) == 0) {
                    simd_binary<D, typename OP::Degraded>(
                        (D*)ip1, is1/sizeof(D),
                        (D*)ip2, is2/sizeof(D),
                        (D*)op1, os1/sizeof(D), len
                    );
                    goto clear_fp;
                }
            }
        #endif
        }
    }
#endif
#ifndef NPY_DISABLE_OPTIMIZATION
    // scalar unrolls
    if (IS_BINARY_REDUCE) {
        // Note, 8x unroll was chosen for best results on Apple M1
        npy_intp elemPerLoop = 8;
        if((i+elemPerLoop) <= len){
            T m0 = *((T*)(ip2 + (i + 0) * is2));
            T m1 = *((T*)(ip2 + (i + 1) * is2));
            T m2 = *((T*)(ip2 + (i + 2) * is2));
            T m3 = *((T*)(ip2 + (i + 3) * is2));
            T m4 = *((T*)(ip2 + (i + 4) * is2));
            T m5 = *((T*)(ip2 + (i + 5) * is2));
            T m6 = *((T*)(ip2 + (i + 6) * is2));
            T m7 = *((T*)(ip2 + (i + 7) * is2));

            i += elemPerLoop;
            for(; (i+elemPerLoop)<=len; i+=elemPerLoop){
                T v0 = *((T*)(ip2 + (i + 0) * is2));
                T v1 = *((T*)(ip2 + (i + 1) * is2));
                T v2 = *((T*)(ip2 + (i + 2) * is2));
                T v3 = *((T*)(ip2 + (i + 3) * is2));
                T v4 = *((T*)(ip2 + (i + 4) * is2));
                T v5 = *((T*)(ip2 + (i + 5) * is2));
                T v6 = *((T*)(ip2 + (i + 6) * is2));
                T v7 = *((T*)(ip2 + (i + 7) * is2));

                m0 = op_func(m0, v0);
                m1 = op_func(m1, v1);
                m2 = op_func(m2, v2);
                m3 = op_func(m3, v3);
                m4 = op_func(m4, v4);
                m5 = op_func(m5, v5);
                m6 = op_func(m6, v6);
                m7 = op_func(m7, v7);
            }

            m0 = op_func(m0, m1);
            m2 = op_func(m2, m3);
            m4 = op_func(m4, m5);
            m6 = op_func(m6, m7);

            m0 = op_func(m0, m2);
            m4 = op_func(m4, m6);

            m0 = op_func(m0, m4);

             *((T*)op1) = op_func(*((T*)op1), m0);
        }
    } else{
        // Note, 4x unroll was chosen for best results on Apple M1
        npy_intp elemPerLoop = 4;
        for(; (i+elemPerLoop)<=len; i+=elemPerLoop){
            /* Note, we can't just load all, do all ops, then store all here.
             * Sometimes ufuncs are called with `accumulate`, which makes the
             * assumption that previous iterations have finished before next
             * iteration.  For example, the output of iteration 2 depends on the
             * result of iteration 1.
             */

            T v0 = *((T*)(ip1 + (i + 0) * is1));
            T u0 = *((T*)(ip2 + (i + 0) * is2));
            *((T*)(op1 + (i + 0) * os1)) = op_func(v0, u0);
            T v1 = *((T*)(ip1 + (i + 1) * is1));
            T u1 = *((T*)(ip2 + (i + 1) * is2));
            *((T*)(op1 + (i + 1) * os1)) = op_func(v1, u1);
            T v2 = *((T*)(ip1 + (i + 2) * is1));
            T u2 = *((T*)(ip2 + (i + 2) * is2));
            *((T*)(op1 + (i + 2) * os1)) = op_func(v2, u2);
            T v3 = *((T*)(ip1 + (i + 3) * is1));
            T u3 = *((T*)(ip2 + (i + 3) * is2));
            *((T*)(op1 + (i + 3) * os1)) = op_func(v3, u3);
        }
    }
#endif // NPY_DISABLE_OPTIMIZATION
    ip1 += is1 * i;
    ip2 += is2 * i;
    op1 += os1 * i;
    for (; i < len; ++i, ip1 += is1, ip2 += is2, op1 += os1) {
        const T in1 = *(T*)ip1;
        const T in2 = *(T*)ip2;
        *((T*)op1) = op_func(in1, in2);
    }

    goto clear_fp; // suppress warnings
clear_fp:
    if constexpr (std::is_floating_point_v<T>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
}

template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
minmax_indexed(char *const* args, npy_intp const*dimensions, npy_intp const*steps)
{
    const OP op_func;
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp n = dimensions[0];
    npy_intp shape = steps[3];
    npy_intp i;
    T *indexed;
    for(i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        indexed = (T* )(ip1 + is1 * indx);
        *indexed = op_func(*indexed, *(T*)value);
    }
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
#define DEFINE_UNARY_MINMAX_FUNCTION(TYPE, KIND, INTR, T)                                \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{                                                                                        \
    using FixedType = typename np::meta::FixedWidth<T>::Type;                            \
    minmax<FixedType, Op##INTR<FixedType>>(args, dimensions, steps);                     \
}                                                                                        \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND##_indexed)                        \
(PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,                          \
    npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func))     \
{                                                                                        \
    using FixedType = typename np::meta::FixedWidth<T>::Type;                            \
    minmax_indexed<FixedType, Op##INTR<FixedType>>(args, dimensions, steps);             \
    return 0;                                                                            \
}

#define DEFINE_UNARY_MINMAX_FUNCTION_LD(TYPE, KIND, INTR)                                \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{                                                                                        \
    minmax<long double, Op##INTR<long double>>(args, dimensions, steps);                 \
}                                                                                        \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND##_indexed)                        \
(PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,                          \
    npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func))     \
{                                                                                        \
    minmax_indexed<long double, Op##INTR<long double>>(args, dimensions, steps);         \
    return 0;                                                                            \
}

DEFINE_UNARY_MINMAX_FUNCTION(UBYTE, maximum, Max, npy_ubyte)
DEFINE_UNARY_MINMAX_FUNCTION(USHORT, maximum, Max, npy_ushort)
DEFINE_UNARY_MINMAX_FUNCTION(UINT, maximum, Max, npy_uint)
DEFINE_UNARY_MINMAX_FUNCTION(ULONG, maximum, Max, npy_ulong)
DEFINE_UNARY_MINMAX_FUNCTION(ULONGLONG, maximum, Max, npy_ulonglong)
DEFINE_UNARY_MINMAX_FUNCTION(UBYTE, minimum, Min, npy_ubyte)
DEFINE_UNARY_MINMAX_FUNCTION(USHORT, minimum, Min, npy_ushort)
DEFINE_UNARY_MINMAX_FUNCTION(UINT, minimum, Min, npy_uint)
DEFINE_UNARY_MINMAX_FUNCTION(ULONG, minimum, Min, npy_ulong)
DEFINE_UNARY_MINMAX_FUNCTION(ULONGLONG, minimum, Min, npy_ulonglong)
DEFINE_UNARY_MINMAX_FUNCTION(BYTE, maximum, Max, npy_byte)
DEFINE_UNARY_MINMAX_FUNCTION(SHORT, maximum, Max, npy_short)
DEFINE_UNARY_MINMAX_FUNCTION(INT, maximum, Max, npy_int)
DEFINE_UNARY_MINMAX_FUNCTION(LONG, maximum, Max, npy_long)
DEFINE_UNARY_MINMAX_FUNCTION(LONGLONG, maximum, Max, npy_longlong)
DEFINE_UNARY_MINMAX_FUNCTION(BYTE, minimum, Min, npy_byte)
DEFINE_UNARY_MINMAX_FUNCTION(SHORT, minimum, Min, npy_short)
DEFINE_UNARY_MINMAX_FUNCTION(INT, minimum, Min, npy_int)
DEFINE_UNARY_MINMAX_FUNCTION(LONG, minimum, Min, npy_long)
DEFINE_UNARY_MINMAX_FUNCTION(LONGLONG, minimum, Min, npy_longlong)
DEFINE_UNARY_MINMAX_FUNCTION(FLOAT, maximum, Max, npy_float)
DEFINE_UNARY_MINMAX_FUNCTION(DOUBLE, maximum, Max, npy_double)
DEFINE_UNARY_MINMAX_FUNCTION_LD(LONGDOUBLE, maximum, Max)
DEFINE_UNARY_MINMAX_FUNCTION(FLOAT, fmax, Maxp, npy_float)
DEFINE_UNARY_MINMAX_FUNCTION(DOUBLE, fmax, Maxp, npy_double)
DEFINE_UNARY_MINMAX_FUNCTION_LD(LONGDOUBLE, fmax, Maxp)
DEFINE_UNARY_MINMAX_FUNCTION(FLOAT, minimum, Min, npy_float)
DEFINE_UNARY_MINMAX_FUNCTION(DOUBLE, minimum, Min, npy_double)
DEFINE_UNARY_MINMAX_FUNCTION_LD(LONGDOUBLE, minimum, Min)
DEFINE_UNARY_MINMAX_FUNCTION(FLOAT, fmin, Minp, npy_float)
DEFINE_UNARY_MINMAX_FUNCTION(DOUBLE, fmin, Minp, npy_double)
DEFINE_UNARY_MINMAX_FUNCTION_LD(LONGDOUBLE, fmin, Minp)
#undef DEFINE_UNARY_MINMAX_FUNCTION
#undef DEFINE_UNARY_MINMAX_FUNCTION_LD
