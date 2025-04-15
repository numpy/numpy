#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"

#include "simd/simd.hpp"
#include "common.hpp"

namespace {
using namespace np::simd;

template <typename T> struct OpRint {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Round(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_rintf(a);
        }
        return npy_rint(a);
    }
};

template <typename T> struct OpFloor {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Floor(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_floorf(a);
        }
        return npy_floor(a);
    }
};

template <typename T> struct OpCeil {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Ceil(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_ceilf(a);
        }
        return npy_ceil(a);
    }
};

template <typename T> struct OpTrunc {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Trunc(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_truncf(a);
        }
        return npy_trunc(a);
    }
};

template <typename T> struct OpSqrt {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Sqrt(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const ;
};
#if defined(_MSC_VER) && defined(_M_IX86) && !NPY_SIMD
#include <emmintrin.h>
template <typename T> HWY_INLINE HWY_ATTR T OpSqrt<T>::operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
        __m128 aa = _mm_load_ss(&a);
        __m128 lower = _mm_sqrt_ss(aa);
        return _mm_cvtss_f32(lower);
    }
    __m128d aa = _mm_load_sd(&a);
    __m128d lower = _mm_sqrt_pd(aa);
    return _mm_cvtsd_f64(lower);
}
#else
template <typename T> HWY_INLINE HWY_ATTR T OpSqrt<T>::operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
        return npy_sqrtf(a);
    }
    return npy_sqrt(a);
}
#endif

template <typename T> struct OpAbs {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::Abs(v);
    }
#endif

    HWY_INLINE HWY_ATTR T operator()(T a) const {
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

    HWY_INLINE HWY_ATTR T operator()(T a) const {
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

    HWY_INLINE HWY_ATTR T operator()(T a) const {
        return static_cast<T>(1) / a;
    }
};

#if NPY_SIMDX
template <typename T>
HWY_INLINE HWY_ATTR auto LoadWithStride(const T* src, npy_intp ssrc, size_t n = Lanes<T>(), T val = 0) {
    constexpr size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes, val);
    for (size_t ii = 0; ii < lanes && ii < n; ++ii) {
        temp[ii] = src[ii * ssrc];
    }
    return LoadU(temp.data());
}

template <typename T>
HWY_INLINE HWY_ATTR void StoreWithStride(Vec<T> vec, T* dst, npy_intp sdst, size_t n = Lanes<T>()) {
    constexpr size_t lanes = Lanes<T>();
    std::vector<T> temp(lanes);
    StoreU(vec, temp.data());
    for (size_t ii = 0; ii < lanes && ii < n; ++ii) {
        dst[ii * sdst] = temp[ii];
    }
}

template<typename T> struct TypeTraits;
template<> struct TypeTraits<float> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_f32;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_f32;
};
#if NPY_SIMDX_F64
template<> struct TypeTraits<double> {
    static constexpr auto LoadStrideFunc = npyv_loadable_stride_f64;
    static constexpr auto StoreStrideFunc = npyv_storable_stride_f64;
};
#endif

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
                v_src = LoadNOr(Set<T>(static_cast<T>(1)), src, len);
            }
            else {
                v_src = LoadN(src, len);
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
            StoreN(v_dst, dst, len);
        }
        else {
            StoreWithStride(v_dst, dst, sdst, len);
        }
    }
}

#endif // NPY_SIMDX

template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
unary_fp(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    const OP op_func;
    const char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];

    if constexpr (kSupportLane<T>) {
        if (!is_mem_overlap(src, src_step, dst, dst_step, len) &&
                TypeTraits<T>::LoadStrideFunc(src_step) != 0 &&
                TypeTraits<T>::StoreStrideFunc(dst_step) != 0) {
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
            goto clear;
        }
    }
    // fallback to scalar implementation
    for (; len > 0; --len, src += src_step, dst += dst_step) {
        if constexpr (kSupportLane<T>) {
            simd_unary_fp<T, OP, 0, 0, 4>(reinterpret_cast<const T*>(src), 0, reinterpret_cast<T*>(dst), 0, 1);
        }
        else {
            const T src0 = *reinterpret_cast<const T*>(src);
            *reinterpret_cast<T*>(dst) = op_func(src0);
        }
    }

clear:;
    if constexpr (std::is_same_v<OP, OpAbs<T>>) {
        npy_clear_floatstatus_barrier((char*)dimensions);
    }
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
#define DEFINE_UNARY_FP_FUNCTION(TYPE, KIND, INTR, T)                                    \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{                                                                                        \
    using FixedType = typename np::meta::FixedWidth<T>::Type;                            \
    unary_fp<FixedType, Op##INTR<FixedType>>(args, dimensions, steps);                   \
}

DEFINE_UNARY_FP_FUNCTION(FLOAT, rint      , Rint  , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, floor     , Floor , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, ceil      , Ceil  , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, trunc     , Trunc , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, sqrt      , Sqrt  , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, absolute  , Abs   , npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, square    , Square, npy_float)
DEFINE_UNARY_FP_FUNCTION(FLOAT, reciprocal, Recip , npy_float)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, rint      , Rint  , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, floor     , Floor , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, ceil      , Ceil  , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, trunc     , Trunc , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, sqrt      , Sqrt  , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, absolute  , Abs   , npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, square    , Square, npy_double)
DEFINE_UNARY_FP_FUNCTION(DOUBLE, reciprocal, Recip , npy_double)
#undef DEFINE_UNARY_FP_FUNCTION
