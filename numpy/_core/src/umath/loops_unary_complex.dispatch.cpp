#include "loops_utils.h"
#include "loops.h"

#include <hwy/highway.h>
#include "simd/simd.hpp"

namespace {
using namespace np::simd;

template <typename T> struct OpCabs {
#if NPY_SIMDX
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    HWY_INLINE HWY_ATTR auto operator()(const V& a, const V& b) const {
        V inf, nan;
        if constexpr (std::is_same_v<T, float>) {
            inf = Set<T>(NPY_INFINITYF);
            nan = Set<T>(NPY_NANF);
        }
        else {
            inf = Set<T>(NPY_INFINITY);
            nan = Set<T>(NPY_NAN);
        }
        auto re = hn::Abs(a), im = hn::Abs(b);
        /*
         * If real or imag = INF, then convert it to inf + j*inf
         * Handles: inf + j*nan, nan + j*inf
         */
        auto re_infmask = hn::IsInf(re), im_infmask = hn::IsInf(im);
        im = hn::IfThenElse(re_infmask, inf, im);
        re = hn::IfThenElse(im_infmask, inf, re);
        /*
         * If real or imag = NAN, then convert it to nan + j*nan
         * Handles: x + j*nan, nan + j*x
         */
        auto re_nanmask = hn::IsNaN(re), im_nanmask = hn::IsNaN(im);
        im = hn::IfThenElse(re_nanmask, nan, im);
        re = hn::IfThenElse(im_nanmask, nan, re);

        auto larger = hn::Max(re, im), smaller = hn::Min(im, re);
        /*
         * Calculate div_mask to prevent 0./0. and inf/inf operations in div
         */
        auto zeromask = hn::Eq(larger, Set<T>(static_cast<T>(0)));
        auto infmask = hn::IsInf(smaller);
        auto div_mask = hn::ExclusiveNeither(zeromask, infmask);

        auto ratio = hn::MaskedDiv(div_mask, smaller, larger);
        auto hypot = hn::Sqrt(hn::MulAdd(ratio, ratio, Set<T>(static_cast<T>(1))));
        return hn::Mul(hypot, larger);   
    }
#endif

    NPY_INLINE T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, float>) {
            return npy_hypotf(a, b);
        } else {
            return npy_hypot(a, b);
        }
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
#endif // NPY_SIMDX

template <typename T>
HWY_INLINE HWY_ATTR void
unary_complex(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    const OpCabs<T> op_func;
    const char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];

#if NPY_SIMDX
    if constexpr (kSupportLane<T>) {
        if (!is_mem_overlap(src, src_step, dst, dst_step, len) && alignof(T) == sizeof(T) &&
                src_step % sizeof(T) == 0 && dst_step % sizeof(T) == 0) {
            const int lsize = sizeof(T);
            const npy_intp ssrc = src_step / lsize;
            const npy_intp sdst = dst_step / lsize;

            const int vstep = Lanes<T>();
            const int wstep = vstep * 2;

            const T* src_T = reinterpret_cast<const T*>(src);
            T* dst_T = reinterpret_cast<T*>(dst);

            if (ssrc == 2 && sdst == 1) {
                for (; len >= vstep; len -= vstep, src_T += wstep, dst_T += vstep) {
                    Vec<T> re, im;
                    hn::LoadInterleaved2(_Tag<T>(), src_T, re, im);
                    auto r = op_func(re, im);
                    StoreU(r, dst_T);
                }
            }
            else {
                for (; len >= vstep; len -= vstep, src_T += ssrc*vstep, dst_T += sdst*vstep) {
                    auto re = LoadWithStride(src_T, ssrc);
                    auto im = LoadWithStride(src_T + 1, ssrc);
                    auto r = op_func(re, im);
                    StoreWithStride(r, dst_T, sdst);
                }
            }
            if (len > 0) {
                auto re = LoadWithStride(src_T, ssrc, len);
                auto im = LoadWithStride(src_T + 1, ssrc, len);
                auto r = op_func(re, im);
                StoreWithStride(r, dst_T, sdst, len);
            }
            // clear the float status flags
            npy_clear_floatstatus_barrier((char*)&len);
            return;
        }
    }
#endif

    // fallback to scalar implementation
    for (; len > 0; --len, src += src_step, dst += dst_step) {
        const T src0 = *reinterpret_cast<const T*>(src);
        const T src1 = *(reinterpret_cast<const T*>(src) + 1);
        *reinterpret_cast<T*>(dst) = op_func(src0, src1);
    }
}

} // anonymous namespace

/*******************************************************************************
 ** Defining ufunc inner functions
 *******************************************************************************/
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CFLOAT_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_complex<npy_float>(args, dimensions, steps);
}
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CDOUBLE_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_complex<npy_double>(args, dimensions, steps);
}
