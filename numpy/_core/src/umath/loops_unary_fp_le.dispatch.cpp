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

template <typename T> struct OpIsNaN {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::IsNaN(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return npy_isnan(a) != 0;
    }
};

template <typename T> struct OpIsInf {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::IsInf(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return npy_isinf(a) != 0;
    }
};

template <typename T> struct OpIsFinite {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::IsFinite(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return npy_isfinite(a) != 0;
    }
};

template <typename T> struct OpSignbit {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& v) const {
        return hn::IsNegative(v);
    }
#endif

    NPY_INLINE T operator()(T a) const {
        return npy_signbit(a) != 0;
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

template <typename D>
HWY_INLINE HWY_ATTR void StoreWithStride(hn::Vec<D> vec, hn::TFromD<D>* dst, npy_intp sdst, size_t n = hn::Lanes(D())) {
    HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(D());
    std::vector<hn::TFromD<D>> temp(lanes);
    hn::StoreU(vec, D(), temp.data());
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
template <typename T, typename OP, int STYPE, int DTYPE>
HWY_INLINE HWY_ATTR void
simd_unary_fp_le(const T *src, npy_intp ssrc, npy_bool *dst, npy_intp sdst, npy_intp len) {
    const OP op_func;

    // How many vectors can be packed into a u8 / bool vector?
    constexpr int PACK_FACTOR = sizeof(T);
    const int vstep = Lanes<T>();
    const int wstep = vstep * PACK_FACTOR;

    // unrolled iterations
    for (; len >= wstep; len -= wstep, src += ssrc*wstep, dst += sdst*wstep) {
        Vec<T> v0, v1, v2, v3;
        if constexpr (STYPE == 0) {
            v0 = LoadU(src);
            v1 = LoadU(src + vstep);
            v2 = LoadU(src + vstep*2);
            v3 = LoadU(src + vstep*3);
        }
        else {
            v0 = LoadWithStride(src, ssrc);
            v1 = LoadWithStride(src + ssrc*vstep, ssrc);
            v2 = LoadWithStride(src + ssrc*vstep*2, ssrc);
            v3 = LoadWithStride(src + ssrc*vstep*3, ssrc);
        }

        Vec<uint8_t> r;
        if constexpr (PACK_FACTOR == 8) {
            Vec<T> v4, v5, v6, v7;
            if constexpr (STYPE == 0) {
                v4 = LoadU(src + vstep*4);
                v5 = LoadU(src + vstep*5);
                v6 = LoadU(src + vstep*6);
                v7 = LoadU(src + vstep*7);
            }
            else {
                v4 = LoadWithStride(src + ssrc*vstep*4, ssrc);
                v5 = LoadWithStride(src + ssrc*vstep*5, ssrc);
                v6 = LoadWithStride(src + ssrc*vstep*6, ssrc);
                v7 = LoadWithStride(src + ssrc*vstep*7, ssrc);
            }

            auto m0 = hn::OrderedDemote2MasksTo(_Tag<uint32_t>(), _Tag<T>(), op_func(v0), op_func(v1));
            auto m1 = hn::OrderedDemote2MasksTo(_Tag<uint32_t>(), _Tag<T>(), op_func(v2), op_func(v3));
            auto m2 = hn::OrderedDemote2MasksTo(_Tag<uint32_t>(), _Tag<T>(), op_func(v4), op_func(v5));
            auto m3 = hn::OrderedDemote2MasksTo(_Tag<uint32_t>(), _Tag<T>(), op_func(v6), op_func(v7));
            auto m00 = hn::OrderedDemote2MasksTo(_Tag<uint16_t>(), _Tag<uint32_t>(), m0, m1);
            auto m01 = hn::OrderedDemote2MasksTo(_Tag<uint16_t>(), _Tag<uint32_t>(), m2, m3);
            r = hn::VecFromMask(_Tag<uint8_t>(), hn::OrderedDemote2MasksTo(_Tag<uint8_t>(), _Tag<uint16_t>(), m00, m01));
        }
        else {
            auto m0 = hn::OrderedDemote2MasksTo(_Tag<uint16_t>(), _Tag<T>(), op_func(v0), op_func(v1));
            auto m1 = hn::OrderedDemote2MasksTo(_Tag<uint16_t>(), _Tag<T>(), op_func(v2), op_func(v3));
            r = hn::VecFromMask(_Tag<uint8_t>(), hn::OrderedDemote2MasksTo(_Tag<uint8_t>(), _Tag<uint16_t>(), m0, m1));
        }

        if constexpr (DTYPE == 0) {
            StoreU(r, dst);
        }
        else {
            StoreWithStride<_Tag<uint8_t>>(r, dst, sdst);
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

        if constexpr (PACK_FACTOR == 4) {
            auto v_dst = hn::VecFromMask(hn::Half<hn::Half<_Tag<uint8_t>>>(), hn::DemoteMaskTo(hn::Half<hn::Half<_Tag<uint8_t>>>(), _Tag<T>(), op_func(v_src)));
            if constexpr (DTYPE == 0) {
                hn::StoreU(v_dst, hn::Half<hn::Half<_Tag<uint8_t>>>(), dst);
            }
            else {
                StoreWithStride<hn::Half<hn::Half<_Tag<uint8_t>>>>(v_dst, dst, sdst);
            }
        }
        else {
            auto v_dst = hn::VecFromMask(hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), hn::DemoteMaskTo(hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), _Tag<T>(), op_func(v_src)));
            if constexpr (DTYPE == 0) {
                hn::StoreU(v_dst, hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), dst);
            }
            else {
                StoreWithStride<hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>>(v_dst, dst, sdst);
            }
        }
    }

    // last partial iteration, if needed
    if (len > 0) {
        Vec<T> v_src;
        if constexpr (STYPE == 0) {
            v_src = hn::LoadN(_Tag<T>(), src, len);
        }
        else {
            v_src = LoadWithStride(src, ssrc, len);
        }

        if constexpr (PACK_FACTOR == 4) {
            auto v_dst = hn::VecFromMask(hn::Half<hn::Half<_Tag<uint8_t>>>(), hn::DemoteMaskTo(hn::Half<hn::Half<_Tag<uint8_t>>>(), _Tag<T>(), op_func(v_src)));
            if constexpr (DTYPE == 0) {
                hn::StoreN(v_dst, hn::Half<hn::Half<_Tag<uint8_t>>>(), dst, len);
            }
            else {
                StoreWithStride<hn::Half<hn::Half<_Tag<uint8_t>>>>(v_dst, dst, sdst, len);
            }
        }
        else {
            auto v_dst = hn::VecFromMask(hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), hn::DemoteMaskTo(hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), _Tag<T>(), op_func(v_src)));
            if constexpr (DTYPE == 0) {
                hn::StoreN(v_dst, hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>(), dst, len);
            }
            else {
                StoreWithStride<hn::Half<hn::Half<hn::Half<_Tag<uint8_t>>>>>(v_dst, dst, sdst, len);
            }
        }
    }
}
#endif // NPY_SIMDX

template <typename T, typename OP>
HWY_INLINE HWY_ATTR void
unary_fp_le(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    const OP op_func;
    const char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];

    bool unrolled = false;
#if NPY_SIMDX
    if constexpr (kSupportLane<T>) {
        if (!is_mem_overlap(src, src_step, dst, dst_step, len) && alignof(T) == sizeof(T) &&
                src_step % sizeof(T) == 0) {
            const npy_intp ssrc = src_step / sizeof(T);
            const npy_intp sdst = dst_step / sizeof(npy_bool);
            if (ssrc == 1 && sdst == 1) {
                simd_unary_fp_le<T, OP, 0, 0>(reinterpret_cast<const T*>(src), 1, reinterpret_cast<npy_bool*>(dst), 1, len);
            }
            else if (sdst == 1) {
                simd_unary_fp_le<T, OP, 1, 0>(reinterpret_cast<const T*>(src), ssrc, reinterpret_cast<npy_bool*>(dst), 1, len);
            }
            else if (ssrc == 1) {
                simd_unary_fp_le<T, OP, 0, 1>(reinterpret_cast<const T*>(src), 1, reinterpret_cast<npy_bool*>(dst), sdst, len);
            }
            else {
                simd_unary_fp_le<T, OP, 1, 1>(reinterpret_cast<const T*>(src), ssrc, reinterpret_cast<npy_bool*>(dst), sdst, len);
            }
            unrolled = true;
        }
    }
#endif

    // fallback to scalar implementation
    if (!unrolled) {
        for (; len > 0; --len, src += src_step, dst += dst_step) {
            const T src0 = *reinterpret_cast<const T*>(src);
            *reinterpret_cast<npy_bool*>(dst) = op_func(src0);
        }
    }

    npy_clear_floatstatus_barrier((char*)dimensions);
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
#define DEFINE_UNARY_FP_LE_FUNCTION(TYPE, KIND, INTR, T)                                 \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{                                                                                        \
    unary_fp_le<T, Op##INTR<T>>(args, dimensions, steps);                                \
}

DEFINE_UNARY_FP_LE_FUNCTION(FLOAT, isnan   , IsNaN   , npy_float)
DEFINE_UNARY_FP_LE_FUNCTION(FLOAT, isinf   , IsInf   , npy_float)
DEFINE_UNARY_FP_LE_FUNCTION(FLOAT, isfinite, IsFinite, npy_float)
DEFINE_UNARY_FP_LE_FUNCTION(FLOAT, signbit , Signbit , npy_float)
DEFINE_UNARY_FP_LE_FUNCTION(DOUBLE, isnan   , IsNaN   , npy_double)
DEFINE_UNARY_FP_LE_FUNCTION(DOUBLE, isinf   , IsInf   , npy_double)
DEFINE_UNARY_FP_LE_FUNCTION(DOUBLE, isfinite, IsFinite, npy_double)
DEFINE_UNARY_FP_LE_FUNCTION(DOUBLE, signbit , Signbit , npy_double)
#undef DEFINE_UNARY_FP_LE_FUNCTION
