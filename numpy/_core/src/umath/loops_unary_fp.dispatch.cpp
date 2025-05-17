#include "loops_utils.h"
#include "loops.h"

#include <hwy/highway.h>
#include "simd/simd.hpp"
#include "common.hpp"

/**
 * Force use SSE only on x86, even if AVX2 or AVX512F are enabled
 * through the baseline, since scatter(AVX512F) and gather very costly
 * to handle non-contiguous memory access comparing with SSE for
 * such small operations that this file covers.
*/

namespace {
using namespace np::simd128;

template <typename T> struct OpRint {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Round(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_rintf(a);
        } else {
            return npy_rint(a);
        }
    }
};

template <typename T> struct OpFloor {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Floor(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_floorf(a);
        } else {
            return npy_floor(a);
        }
    }
};

template <typename T> struct OpCeil {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Ceil(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_ceilf(a);
        } else {
            return npy_ceil(a);
        }
    }
};

template <typename T> struct OpTrunc {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Trunc(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_truncf(a);
        } else {
            return npy_trunc(a);
        }
    }
};

template <typename T> struct OpSqrt {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Sqrt(v);
    }
#endif

    NPY_INLINE T operator()(T a) const ;
};
/**
 * MSVC(32-bit mode) requires a clarified contiguous loop
 * in order to use SSE, otherwise it uses a soft version of square root
 * that doesn't raise a domain error.
 */
#if defined(_MSC_VER) && defined(_M_IX86)
#include <emmintrin.h>
template <> NPY_INLINE float OpSqrt<float>::operator()(float a) const {
    __m128 aa = _mm_load_ss(&a);
    __m128 lower = _mm_sqrt_ss(aa);
    return _mm_cvtss_f32(lower);
}
template <> NPY_INLINE double OpSqrt<double>::operator()(double a) const {
    __m128d aa = _mm_load_sd(&a);
    __m128d lower = _mm_sqrt_pd(aa);
    return _mm_cvtsd_f64(lower);
}
#else
template <typename T> NPY_INLINE T OpSqrt<T>::operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
        return npy_sqrtf(a);
    } else {
        return npy_sqrt(a);
    }
}
#endif

template <typename T> struct OpAbs {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Abs(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        /* add 0 to clear -0.0 */
        return 0 + (a > 0 ? a : -a);
    }
};

template <typename T> struct OpSquare {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Mul(v, v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return a * a;
    }
};

template <typename T> struct OpRecip {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Div(Set<T>(static_cast<T>(1)), v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return static_cast<T>(1) / a;
    }
};

#if NPY_SIMDX
template <typename T>
HWY_INLINE HWY_ATTR auto LoadWithStride(const T* src, npy_intp ssrc, size_t n = Lanes<T>(), T val = 0) {
    HWY_LANES_CONSTEXPR size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes, val);
    for (size_t ii = 0; ii < lanes && ii < n; ++ii) {
        temp[ii] = src[ii * ssrc];
    }
    return LoadU(temp.data());
}

template <typename T>
HWY_INLINE HWY_ATTR void StoreWithStride(Vec<T> vec, T* dst, npy_intp sdst, size_t n = Lanes<T>()) {
    HWY_LANES_CONSTEXPR size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes);
    StoreU(vec, temp.data());
    for (size_t ii = 0; ii < lanes && ii < n; ++ii) {
        dst[ii * sdst] = temp[ii];
    }
}

/********************************************************************************
 ** Defining the SIMD kernels
 ********************************************************************************/
/** Notes:
 * - avoid the use of libmath to unify fp/domain errors
 *   for both scalars and vectors among all compilers/architectures.
 * - use intrinsic LoadNOr instead of LoadN
 *   to fill the remind lanes with 1.0 to avoid divide by zero fp
 *   exception in reciprocal.
 */
template <typename T, typename OP, int STYPE, int DTYPE, int UNROLL>
HWY_INLINE HWY_ATTR void
simd_unary_fp(const T *src, npy_intp ssrc, T *dst, npy_intp sdst, npy_intp len) {
    const OP op_func;
    const int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;

    // unrolled iterations
    for (; len >= wstep; len -= wstep, src += ssrc*wstep, dst += sdst*wstep) {
        if constexpr (UNROLL == 2) {
            Vec<T> v_src0, v_src1;
            if constexpr (STYPE == 0) {
                v_src0 = LoadU(src);
                v_src1 = LoadU(src + vstep);
            }
            else {
                v_src0 = LoadWithStride(src, ssrc);
                v_src1 = LoadWithStride(src + ssrc*vstep, ssrc);
            }

            auto v_dst0 = op_func(v_src0), v_dst1 = op_func(v_src1);
            if constexpr (DTYPE == 0) {
                StoreU(v_dst0, dst);
                StoreU(v_dst1, dst + vstep);
            }
            else {
                StoreWithStride(v_dst0, dst, sdst);
                StoreWithStride(v_dst1, dst + sdst*vstep, sdst);
            }
        }
        else if constexpr (UNROLL == 4) {
            Vec<T> v_src0, v_src1, v_src2, v_src3;
            if constexpr (STYPE == 0) {
                v_src0 = LoadU(src);
                v_src1 = LoadU(src + vstep);
                v_src2 = LoadU(src + vstep*2);
                v_src3 = LoadU(src + vstep*3);
            }
            else {
                v_src0 = LoadWithStride(src, ssrc);
                v_src1 = LoadWithStride(src + ssrc*vstep, ssrc);
                v_src2 = LoadWithStride(src + ssrc*vstep*2, ssrc);
                v_src3 = LoadWithStride(src + ssrc*vstep*3, ssrc);
            }

            auto v_dst0 = op_func(v_src0), v_dst1 = op_func(v_src1), v_dst2 = op_func(v_src2), v_dst3 = op_func(v_src3);
            if constexpr (DTYPE == 0) {
                StoreU(v_dst0, dst);
                StoreU(v_dst1, dst + vstep);
                StoreU(v_dst2, dst + vstep*2);
                StoreU(v_dst3, dst + vstep*3);
            }
            else {
                StoreWithStride(v_dst0, dst, sdst);
                StoreWithStride(v_dst1, dst + sdst*vstep, sdst);
                StoreWithStride(v_dst2, dst + sdst*vstep*2, sdst);
                StoreWithStride(v_dst3, dst + sdst*vstep*3, sdst);
            }
        }
    }

    // vector-sized iterations
    for (; len >= vstep; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        Vec<T> v_src;
        if constexpr (STYPE == 0) {
            v_src = LoadU(src);
        }
        else {
            v_src = LoadWithStride(src, ssrc);
        }

        auto v_dst = op_func(v_src);
        if constexpr (DTYPE == 0) {
            StoreU(v_dst, dst);
        }
        else {
            StoreWithStride(v_dst, dst, sdst);
        }
    }

    // last partial iteration, if needed
    if (len > 0) {
        Vec<T> v_src;
        if constexpr (STYPE == 0) {
            if constexpr (std::is_same_v<OP, OpRecip<T>>) {
                v_src = hn::LoadNOr(Set<T>(static_cast<T>(1)), _Tag<T>(), src, len);
            }
            else {
                v_src = hn::LoadN(_Tag<T>(), src, len);
            }
        }
        else {
            if constexpr (std::is_same_v<OP, OpRecip<T>>) {
                v_src = LoadWithStride(src, ssrc, len, static_cast<T>(1));
            }
            else {
                v_src = LoadWithStride(src, ssrc, len);
            }
        }

        auto v_dst = op_func(v_src);
        if constexpr (DTYPE == 0) {
            hn::StoreN(v_dst, _Tag<T>(), dst, len);
        }
        else {
            StoreWithStride(v_dst, dst, sdst, len);
        }
    }
}
#endif // NPY_SIMDX

template <typename T, typename OP, bool SKIP_SIMD>
HWY_INLINE HWY_ATTR void
unary_fp(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    const OP op_func;
    const char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];

    bool unrolled = false;
#if NPY_SIMDX
    if constexpr (!SKIP_SIMD && kSupportLane<T>) {
        if (!is_mem_overlap(src, src_step, dst, dst_step, len) && alignof(T) == sizeof(T) &&
                src_step % sizeof(T) == 0 && dst_step % sizeof(T) == 0) {
            const int lsize = sizeof(T);
            const npy_intp ssrc = src_step / lsize;
            const npy_intp sdst = dst_step / lsize;
            if (ssrc == 1 && sdst == 1) {
                simd_unary_fp<T, OP, 0, 0, 4>(reinterpret_cast<const T*>(src), 1, reinterpret_cast<T*>(dst), 1, len);
            }
            else if (sdst == 1) {
                simd_unary_fp<T, OP, 1, 0, 4>(reinterpret_cast<const T*>(src), ssrc, reinterpret_cast<T*>(dst), 1, len);
            }
            else if (ssrc == 1) {
                simd_unary_fp<T, OP, 0, 1, 2>(reinterpret_cast<const T*>(src), 1, reinterpret_cast<T*>(dst), sdst, len);
            }
            else {
                simd_unary_fp<T, OP, 1, 1, 2>(reinterpret_cast<const T*>(src), ssrc, reinterpret_cast<T*>(dst), sdst, len);
            }
            unrolled = true;
        }
    }
#endif

    // fallback to scalar implementation
    if (!unrolled) {
        for (; len > 0; --len, src += src_step, dst += dst_step) {
        #if NPY_SIMDX
            if constexpr (!SKIP_SIMD && kSupportLane<T>) {
                // to guarantee the same precision and fp/domain errors for both scalars and vectors
                simd_unary_fp<T, OP, 0, 0, 4>(reinterpret_cast<const T*>(src), 0, reinterpret_cast<T*>(dst), 0, 1);
            } else
        #endif
            {
                const T src0 = *reinterpret_cast<const T*>(src);
                *reinterpret_cast<T*>(dst) = op_func(src0);
            }
        }
    }

    if constexpr (std::is_same_v<OP, OpAbs<T>>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }

    // MSVC(32-bit mode) bug
#if defined(_MSC_VER) && defined(_M_IX86)
    if constexpr (std::is_same_v<OP, OpCeil<T>>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
#endif
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
#define DEFINE_UNARY_FP_FUNCTION(TYPE, KIND, INTR, T, SKIP_SIMD)                         \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{                                                                                        \
    unary_fp<T, Op##INTR<T>, SKIP_SIMD>(args, dimensions, steps);                        \
}
#define X86_SOFTROUND ((HWY_TARGET >= HWY_SSSE3) && (HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)))

DEFINE_UNARY_FP_FUNCTION(FLOAT, rint      , Rint  , npy_float, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(FLOAT, floor     , Floor , npy_float, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(FLOAT, ceil      , Ceil  , npy_float, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(FLOAT, trunc     , Trunc , npy_float, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(FLOAT, sqrt      , Sqrt  , npy_float, false)
DEFINE_UNARY_FP_FUNCTION(FLOAT, absolute  , Abs   , npy_float, false)
DEFINE_UNARY_FP_FUNCTION(FLOAT, square    , Square, npy_float, false)
DEFINE_UNARY_FP_FUNCTION(FLOAT, reciprocal, Recip , npy_float, HWY_ARCH_WASM)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, rint      , Rint  , npy_double, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, floor     , Floor , npy_double, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, ceil      , Ceil  , npy_double, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, trunc     , Trunc , npy_double, X86_SOFTROUND)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, sqrt      , Sqrt  , npy_double, false)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, absolute  , Abs   , npy_double, false)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, square    , Square, npy_double, false)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, reciprocal, Recip , npy_double, HWY_ARCH_WASM)
#undef DEFINE_UNARY_FP_FUNCTION
#undef X86_SOFTROUND
