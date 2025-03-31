#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include "numpy/npy_common.h"
#include "common.hpp"
#include <hwy/highway.h>

namespace {
namespace hn = hwy::HWY_NAMESPACE;

const hn::ScalableTag<uint8_t>  u8;
const hn::ScalableTag<int8_t>   s8;
const hn::ScalableTag<uint16_t> u16;
const hn::ScalableTag<int16_t>  s16;
const hn::ScalableTag<uint32_t> u32;
const hn::ScalableTag<int32_t>  s32;
const hn::ScalableTag<uint64_t> u64;
const hn::ScalableTag<int64_t>  s64;
const hn::ScalableTag<float>    f32;
const hn::ScalableTag<double>   f64;

using vec_u8  = hn::Vec<decltype(u8)>;
using vec_s8  = hn::Vec<decltype(s8)>;
using vec_u16 = hn::Vec<decltype(u16)>;
using vec_s16 = hn::Vec<decltype(s16)>;
using vec_u32 = hn::Vec<decltype(u32)>;
using vec_s32 = hn::Vec<decltype(s32)>;
using vec_u64 = hn::Vec<decltype(u64)>;
using vec_s64 = hn::Vec<decltype(s64)>;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_f64 = hn::Vec<decltype(f64)>;

template<typename T>
struct TagSelector;

template<> struct TagSelector<uint8_t>  { static const auto& value() { return u8;  } };
template<> struct TagSelector<int8_t>   { static const auto& value() { return s8;  } };
template<> struct TagSelector<uint16_t> { static const auto& value() { return u16; } };
template<> struct TagSelector<int16_t>  { static const auto& value() { return s16; } };
template<> struct TagSelector<uint32_t> { static const auto& value() { return u32; } };
template<> struct TagSelector<int32_t>  { static const auto& value() { return s32; } };
template<> struct TagSelector<uint64_t> { static const auto& value() { return u64; } };
template<> struct TagSelector<int64_t>  { static const auto& value() { return s64; } };
template<> struct TagSelector<float>    { static const auto& value() { return f32; } };
template<> struct TagSelector<double>   { static const auto& value() { return f64; } };

template<typename T>
constexpr const auto& GetTag() {
    return TagSelector<T>::value();
}

template <typename T>
constexpr bool kSupportLane = false;

template <> constexpr bool kSupportLane<uint8_t>  = true;
template <> constexpr bool kSupportLane<int8_t>   = true;
template <> constexpr bool kSupportLane<uint16_t> = true;
template <> constexpr bool kSupportLane<int16_t>  = true;
template <> constexpr bool kSupportLane<uint32_t> = true;
template <> constexpr bool kSupportLane<int32_t>  = true;
template <> constexpr bool kSupportLane<uint64_t> = true;
template <> constexpr bool kSupportLane<int64_t>  = true;
template <> constexpr bool kSupportLane<float>    = true;
template <> constexpr bool kSupportLane<double>   = true;

template <typename T>
struct OpNegative {
#if NPY_SIMD
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            #if defined(NPY_HAVE_NEON)
                if constexpr (std::is_same_v<T, float>) {
                    return vnegq_f32(v);
                } else {
                    return vnegq_f64(v);
                }
            #else
                // (v ^ signmask)
                const auto signmask = hn::Set(GetTag<T>(), static_cast<T>(-0.));
                return hn::Xor(v, signmask);
            #endif
        } else if constexpr (std::is_signed_v<T>) {
            #if defined(NPY_HAVE_NEON)
                if constexpr (sizeof(T) == 1) {
                    return vnegq_s8(v);
                } else if constexpr (sizeof(T) == 2) {
                    return vnegq_s16(v);
                } else if constexpr (sizeof(T) == 4) {
                    return vnegq_s32(v);
                } else if constexpr (sizeof(T) == 8) {
                    #if defined(__aarch64__)
                        return vnegq_s64(v);
                    #endif
                }
            #endif
            const auto m1 = hn::Set(GetTag<T>(), static_cast<T>(-1));
            return hn::Sub(hn::Xor(v, m1), m1);
        } else {
            #if defined(NPY_HAVE_NEON)
                if constexpr (sizeof(T) == 1) {
                    return hn::BitCast(u8, vnegq_s8(hn::BitCast(s8, v)));
                } else if constexpr (sizeof(T) == 2) {
                    return hn::BitCast(u16, vnegq_s16(hn::BitCast(s16, v)));
                } else if constexpr (sizeof(T) == 4) {
                    return hn::BitCast(u32, vnegq_s32(hn::BitCast(s32, v)));
                } else if constexpr (sizeof(T) == 8) {
                    #if defined(__aarch64__)
                        return hn::BitCast(u64, vnegq_s64(hn::BitCast(s64, v)));
                    #endif
                }
            #endif
                const auto m1 = hn::Set(GetTag<T>(), static_cast<T>(-1));
                return hn::Sub(hn::Xor(v, m1), m1);
        }
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) {
        return -a;
    }
};

template <>
struct OpNegative<long double> {
    HWY_INLINE HWY_ATTR long double operator()(long double a) {
        return -a;
    }
};

template <typename T>
HWY_INLINE HWY_ATTR auto LoadWithStride(const T* src, npy_intp istride) {
    T temp[hn::Lanes(GetTag<T>())] = {};
    for (auto ii = 0; ii < (npy_intp)hn::Lanes(GetTag<T>()); ++ii) {
        temp[ii] = src[ii * istride];
    }
    return hn::LoadU(GetTag<T>(), temp);
}

template <typename T>
HWY_INLINE HWY_ATTR void StoreWithStride(hn::Vec<hn::ScalableTag<T>> vec,
                                         T* dst, npy_intp sdst) {
    T temp[hn::Lanes(GetTag<T>())] = {};
    hn::StoreU(vec, GetTag<T>(), temp);
    for (auto ii = 0; ii < (npy_intp)hn::Lanes(GetTag<T>()); ++ii) {
        dst[ii * sdst] = temp[ii];
    }
}

#if NPY_SIMD
template<typename T>
struct TypeTraits;

template<>
struct TypeTraits<uint32_t> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_u32;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_u32;
};

template<>
struct TypeTraits<int32_t> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_s32;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_s32;
};

template<>
struct TypeTraits<uint64_t> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_u64;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_u64;
};

template<>
struct TypeTraits<int64_t> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_s64;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_s64;
};

template<>
struct TypeTraits<float> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_f32;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_f32;
};

template<>
struct TypeTraits<double> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_f64;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_f64;
};

template <typename T>
HWY_INLINE HWY_ATTR void
simd_unary_cc_negative(const T *ip, T *op, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int UNROLL = (NPY_SIMD == 128 ? 4 : 2);
    const int vstep = hn::Lanes(GetTag<T>());
    const int wstep = vstep * UNROLL;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int u = 0; u < UNROLL; ++u) {
            auto v = hn::LoadU(GetTag<T>(), ip + u * vstep);
            auto r = op_func(v);
            hn::StoreU(r, GetTag<T>(), op + u * vstep);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        auto v = hn::LoadU(GetTag<T>(), ip);
        auto r = op_func(v);
        hn::StoreU(r, GetTag<T>(), op);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ++ip, ++op) {
        *op = op_func(*ip);
    }
}

template <typename T>
HWY_INLINE HWY_ATTR void
simd_unary_cn_negative(const T *ip, T *op, npy_intp ostride, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int UNROLL = (NPY_SIMD == 128 ? 4 : 2);
    const int vstep = hn::Lanes(GetTag<T>());
    const int wstep = vstep * UNROLL;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += ostride*wstep) {
        for (int u = 0; u < UNROLL; ++u) {
            auto v = hn::LoadU(GetTag<T>(), ip + u * vstep);
            auto r = op_func(v);
            StoreWithStride<T>(r, op + u * vstep * ostride, ostride);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += ostride*vstep) {
        auto v = hn::LoadU(GetTag<T>(), ip);
        auto r = op_func(v);
        StoreWithStride<T>(r, op, ostride);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ++ip, op += ostride) {
        *op = op_func(*ip);
    }
}

template <typename T>
HWY_INLINE HWY_ATTR void
simd_unary_nc_negative(const T *ip, npy_intp istride, T *op, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int UNROLL = (NPY_SIMD == 128 ? 4 : 2);
    const int vstep = hn::Lanes(GetTag<T>());
    const int wstep = vstep * UNROLL;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride*wstep, op += wstep) {
        for (int u = 0; u < UNROLL; ++u) {
            auto v = LoadWithStride<T>(ip + u * vstep * istride, istride);
            auto r = op_func(v);
            hn::StoreU(r, GetTag<T>(), op + u * vstep);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += istride*vstep, op += vstep) {
        auto v = LoadWithStride<T>(ip, istride);
        auto r = op_func(v);
        hn::StoreU(r, GetTag<T>(), op);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ip += istride, ++op) {
        *op = op_func(*ip);
    }
}

// X86 does better with unrolled scalar for heavy non-contiguous
#ifndef NPY_HAVE_SSE2
template <typename T>
HWY_INLINE HWY_ATTR void
simd_unary_nn_negative(const T *ip, npy_intp istride, T *op, npy_intp ostride, npy_intp len) {
    OpNegative<T> op_func;
    // non-contiguous input and output ; limit UNROLL to 2x
    constexpr int UNROLL = 2;
    const int vstep = hn::Lanes(GetTag<T>());
    const int wstep = vstep * UNROLL;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride*wstep, op += ostride*wstep) {
        for (int u = 0; u < UNROLL; ++u) {
            auto v = LoadWithStride<T>(ip + u * vstep * istride, istride);
            auto r = op_func(v);
            StoreWithStride<T>(r, op + u * vstep * ostride, ostride);
        }
    }

    for (; len >= vstep; len -= vstep, ip += istride*vstep, op += ostride*vstep) {
        auto v = LoadWithStride<T>(ip, istride);
        auto r = op_func(v);
        StoreWithStride<T>(r, op, ostride);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ip += istride, op += ostride) {
        *op = op_func(*ip);
    }
}
#endif // NPY_HAVE_SSE2

#endif // NPY_SIMD

template <typename T>
HWY_INLINE HWY_ATTR void
unary_negative(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    OpNegative<T> op_func;
    char *ip = args[0], *op = args[1];
    npy_intp istep = steps[0], ostep = steps[1], len = dimensions[0];

    bool need_scalar = true;

#if NPY_SIMD
    if constexpr (kSupportLane<T>) {
        
        if (!is_mem_overlap(ip, istep, op, ostep, len)) {
            if (IS_UNARY_CONT(T, T)) {
                // No overlap and operands are contiguous
                simd_unary_cc_negative<T>((T*)ip, (T*)op, len);
                need_scalar = false;
            }

            if constexpr (sizeof(T) > sizeof(uint16_t)){
                using Traits = TypeTraits<T>;
                if (Traits::LoadStrideFunc(istep) && Traits::StoreStrideFunc(ostep)){
                    const npy_intp istride = istep / sizeof(T);
                    const npy_intp ostride = ostep / sizeof(T);

                    if (istride == sizeof(T) && ostride != 1) {
                        // Contiguous input, non-contiguous output
                        simd_unary_cn_negative<T>((T*)ip, (T*)op, ostride, len);
                        need_scalar = false;
                    }
                    else if (istride != 1 && ostride == 1) {
                        // Non-contiguous input, contiguous output
                        simd_unary_nc_negative<T>((T*)ip, istride, (T*)op, len);
                        need_scalar = false;
                    }
                // X86 does better with unrolled scalar for heavy non-contiguous
                #ifndef NPY_HAVE_SSE2
                    else if (istride != 1 && ostride != 1) {
                        // Non-contiguous input and output
                        simd_unary_nn_negative<T>((T*)ip, istride, (T*)op, ostride, len);
                        need_scalar = false;
                    }
                #endif
                }
            }
        }
    }
#endif

    if (need_scalar) {
#ifndef NPY_DISABLE_OPTIMIZATION
    /*
     * scalar unrolls
     * 8x unroll performed best on
     *  - Apple M1 Native / arm64
     *  - Apple M1 Rosetta / SSE42
     *  - iMacPro / AVX512
     */
    constexpr int UNROLL = 8;
    for (; len >= UNROLL; len -= UNROLL, ip += istep*UNROLL, op += ostep*UNROLL) {
        for (int u = 0; u < UNROLL; ++u) {
            const T in = *((const T *)(ip + u * istep));
            *((T *)(op + u * ostep)) = op_func(in);
        }
    }
#endif  // NPY_DISABLE_OPTIMIZATION

        for (; len > 0; --len, ip += istep, op += ostep) {
            *((T *)op) = op_func(*(const T *)ip);
        }
    }

#if NPY_SIMD
    if constexpr (kSupportLane<T>) {
        npyv_cleanup();
    }
#endif

    if constexpr (std::is_floating_point_v<T>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
template <typename T>
HWY_INLINE HWY_ATTR void dispatch_negative(char **args, npy_intp const *dimensions, npy_intp const *steps) {
    using FixedType = typename np::meta::FixedWidth<T>::Type;
    unary_negative<FixedType>(args, dimensions, steps);
}

template <>
HWY_INLINE HWY_ATTR void dispatch_negative<npy_longdouble>(char **args, npy_intp const *dimensions, npy_intp const *steps) {
    unary_negative<long double>(args, dimensions, steps);
}

#define DEFINE_NEGATIVE_FUNCTION(TYPE_NAME, T) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_NAME##_negative) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    dispatch_negative<T>(args, dimensions, steps); \
}

DEFINE_NEGATIVE_FUNCTION(UBYTE,  npy_ubyte)
DEFINE_NEGATIVE_FUNCTION(USHORT, npy_ushort)
DEFINE_NEGATIVE_FUNCTION(UINT,   npy_uint)
DEFINE_NEGATIVE_FUNCTION(ULONG,  npy_ulong)
DEFINE_NEGATIVE_FUNCTION(ULONGLONG, npy_ulonglong)

DEFINE_NEGATIVE_FUNCTION(BYTE,  npy_byte)
DEFINE_NEGATIVE_FUNCTION(SHORT, npy_short)
DEFINE_NEGATIVE_FUNCTION(INT,   npy_int)
DEFINE_NEGATIVE_FUNCTION(LONG,  npy_long)
DEFINE_NEGATIVE_FUNCTION(LONGLONG, npy_longlong)

DEFINE_NEGATIVE_FUNCTION(FLOAT,      npy_float)
DEFINE_NEGATIVE_FUNCTION(DOUBLE,     npy_double)
DEFINE_NEGATIVE_FUNCTION(LONGDOUBLE, npy_longdouble)

