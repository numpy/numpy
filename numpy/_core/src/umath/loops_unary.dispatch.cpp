#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include "numpy/npy_common.h"
#include "common.hpp"
#include "simd/simd.hpp"
#include <hwy/highway.h>

namespace {
using namespace np::simd;

template <typename T>
struct OpNegative {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE  auto operator()(const V& v) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                // (v ^ signmask)
                const auto signmask = Set(static_cast<T>(-0.));
                return hn::Xor(v, signmask);
        } else {
                const auto m1 = Set(static_cast<T>(-1));
                return hn::Sub(hn::Xor(v, m1), m1);
        }
    }
#endif

    HWY_INLINE  T operator()(T a) {
        return -a;
    }
};

template <>
struct OpNegative<long double> {
    HWY_INLINE  long double operator()(long double a) {
        return -a;
    }
};

#if NPY_HWY
template <typename T>
HWY_INLINE  auto LoadWithStride(const T* src, npy_intp istride) {
    HWY_LANES_CONSTEXPR size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes);
    for (size_t ii = 0; ii < lanes; ++ii) {
        temp[ii] = src[ii * istride];
    }
    return LoadU(temp.data());
}

template <typename T>
HWY_INLINE  void StoreWithStride(Vec<T> vec, T* dst, npy_intp sdst) {
    HWY_LANES_CONSTEXPR size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes);
    StoreU(vec, temp.data());
    for (size_t ii = 0; ii < lanes; ++ii) {
        dst[ii * sdst] = temp[ii];
    }
}

template <typename T>
HWY_INLINE  void
simd_unary_cc_negative(const T *ip, T *op, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int kUnrollSize = kMaxLanes<uint8_t> == 16 ? 4 : 2;
    const int vstep = Lanes<T>();
    const int wstep = vstep * kUnrollSize;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        if constexpr (kUnrollSize >= 4) {
            auto v0 = LoadU(ip + 0 * vstep); StoreU(op_func(v0), op + 0 * vstep);
            auto v1 = LoadU(ip + 1 * vstep); StoreU(op_func(v1), op + 1 * vstep);
            auto v2 = LoadU(ip + 2 * vstep); StoreU(op_func(v2), op + 2 * vstep);
            auto v3 = LoadU(ip + 3 * vstep); StoreU(op_func(v3), op + 3 * vstep);
        } else if constexpr (kUnrollSize >= 2) {
            auto v0 = LoadU(ip + 0 * vstep); StoreU(op_func(v0), op + 0 * vstep);
            auto v1 = LoadU(ip + 1 * vstep); StoreU(op_func(v1), op + 1 * vstep);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        auto v = LoadU(ip);
        auto r = op_func(v);
        StoreU(r, op);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ++ip, ++op) {
        *op = op_func(*ip);
    }
}

template <typename T>
HWY_INLINE  void
simd_unary_cn_negative(const T *ip, T *op, npy_intp ostride, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int kUnrollSize = kMaxLanes<uint8_t> == 16 ? 4 : 2;
    const int vstep = Lanes<T>();
    const int wstep = vstep * kUnrollSize;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += ostride*wstep) {
        if constexpr (kUnrollSize >= 4) {
            auto v0 = LoadU(ip + 0 * vstep);
            auto v1 = LoadU(ip + 1 * vstep);
            auto v2 = LoadU(ip + 2 * vstep);
            auto v3 = LoadU(ip + 3 * vstep);
            StoreWithStride<T>(op_func(v0), op + 0 * vstep* ostride, ostride);
            StoreWithStride<T>(op_func(v1), op + 1 * vstep* ostride, ostride);
            StoreWithStride<T>(op_func(v2), op + 2 * vstep* ostride, ostride);
            StoreWithStride<T>(op_func(v3), op + 3 * vstep* ostride, ostride);
        } else if constexpr (kUnrollSize >= 2) {
            auto v0 = LoadU(ip + 0 * vstep);
            auto v1 = LoadU(ip + 1 * vstep);
            StoreWithStride<T>(op_func(v0), op + 0 * vstep* ostride, ostride);
            StoreWithStride<T>(op_func(v1), op + 1 * vstep* ostride, ostride);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += ostride*vstep) {
        auto v = LoadU(ip);
        auto r = op_func(v);
        StoreWithStride<T>(r, op, ostride);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ++ip, op += ostride) {
        *op = op_func(*ip);
    }
}

template <typename T>
HWY_INLINE  void
simd_unary_nc_negative(const T *ip, npy_intp istride, T *op, npy_intp len) {
    OpNegative<T> op_func;
    constexpr int kUnrollSize = kMaxLanes<uint8_t> == 16 ? 4 : 2;
    const int vstep = Lanes<T>();
    const int wstep = vstep * kUnrollSize;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride*wstep, op += wstep) {
        if constexpr (kUnrollSize >= 4) {
            auto v0 = LoadWithStride<T>(ip + 0 * vstep * istride, istride);
            auto v1 = LoadWithStride<T>(ip + 1 * vstep * istride, istride);
            auto v2 = LoadWithStride<T>(ip + 2 * vstep * istride, istride);
            auto v3 = LoadWithStride<T>(ip + 3 * vstep * istride, istride);
            StoreU(op_func(v0), op + 0 * vstep);
            StoreU(op_func(v1), op + 1 * vstep);
            StoreU(op_func(v2), op + 2 * vstep);
            StoreU(op_func(v3), op + 3 * vstep);
        } else if constexpr (kUnrollSize >= 2) {
            auto v0 = LoadWithStride<T>(ip + 0 * vstep * istride, istride);
            auto v1 = LoadWithStride<T>(ip + 1 * vstep * istride, istride);
            StoreU(op_func(v0), op + 0 * vstep);
            StoreU(op_func(v1), op + 1 * vstep);
        }
    }

    // single vector loop
    for (; len >= vstep; len -= vstep, ip += istride*vstep, op += vstep) {
        auto v = LoadWithStride<T>(ip, istride);
        auto r = op_func(v);
        StoreU(r, op);
    }

    // scalar finish up any remaining iterations
    for (; len > 0; --len, ip += istride, ++op) {
        *op = op_func(*ip);
    }
}

// X86 does better with unrolled scalar for heavy non-contiguous
#ifndef NPY_HAVE_SSE2
template <typename T>
HWY_INLINE  void
simd_unary_nn_negative(const T *ip, npy_intp istride, T *op, npy_intp ostride, npy_intp len) {
    OpNegative<T> op_func;
    // non-contiguous input and output ; limit UNROLL to 2x
    constexpr int UNROLL = 2;
    const int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;

    // unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride*wstep, op += ostride*wstep) {
        auto v0 = LoadWithStride<T>(ip + 0 * vstep * istride, istride);
        auto v1 = LoadWithStride<T>(ip + 1 * vstep * istride, istride);
        StoreWithStride<T>(op_func(v0), op + 0 * vstep * ostride, ostride);
        StoreWithStride<T>(op_func(v1), op + 1 * vstep * ostride, ostride);
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

#endif // NPY_HWY

template <typename T>
HWY_INLINE  void
unary_negative(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    OpNegative<T> op_func;
    char *ip = args[0], *op = args[1];
    npy_intp istep = steps[0], ostep = steps[1], len = dimensions[0];

    bool need_scalar = true;

#if NPY_HWY
    if constexpr (kSupportLane<T> && sizeof(long double) != sizeof(double)) {
        if (!is_mem_overlap(ip, istep, op, ostep, len)) {
            if (IS_UNARY_CONT(T, T)) {
                // No overlap and operands are contiguous
                simd_unary_cc_negative<T>((T*)ip, (T*)op, len);
                need_scalar = false;
            }

            if constexpr (sizeof(T) > sizeof(uint16_t)){
                  if (alignof(T) == sizeof(T) && istep % sizeof(T) == 0 && ostep % sizeof(T) == 0){
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
        const T in0 = *((const T *)(ip + 0 * istep)); *reinterpret_cast<T*>(op + 0 * ostep) = op_func(in0);
        const T in1 = *((const T *)(ip + 1 * istep)); *reinterpret_cast<T*>(op + 1 * ostep) = op_func(in1);
        const T in2 = *((const T *)(ip + 2 * istep)); *reinterpret_cast<T*>(op + 2 * ostep) = op_func(in2);
        const T in3 = *((const T *)(ip + 3 * istep)); *reinterpret_cast<T*>(op + 3 * ostep) = op_func(in3);
        const T in4 = *((const T *)(ip + 4 * istep)); *reinterpret_cast<T*>(op + 4 * ostep) = op_func(in4);
        const T in5 = *((const T *)(ip + 5 * istep)); *reinterpret_cast<T*>(op + 5 * ostep) = op_func(in5);
        const T in6 = *((const T *)(ip + 6 * istep)); *reinterpret_cast<T*>(op + 6 * ostep) = op_func(in6);
        const T in7 = *((const T *)(ip + 7 * istep)); *reinterpret_cast<T*>(op + 7 * ostep) = op_func(in7);
    }
#endif  // NPY_DISABLE_OPTIMIZATION

        for (; len > 0; --len, ip += istep, op += ostep) {
            *((T *)op) = op_func(*(const T *)ip);
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
template <typename T>
HWY_INLINE void dispatch_negative(char **args, npy_intp const *dimensions, npy_intp const *steps) {
    using FixedType = typename np::meta::FixedWidth<T>::Type;
    unary_negative<FixedType>(args, dimensions, steps);
}

template <>
HWY_INLINE void dispatch_negative<npy_longdouble>(char **args, npy_intp const *dimensions, npy_intp const *steps) {
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
