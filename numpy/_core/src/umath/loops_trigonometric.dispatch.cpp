#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"

#include "simd/simd.h"
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

/*
 * Vectorized approximate sine/cosine algorithms: The following code is a
 * vectorized version of the algorithm presented here:
 * https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
 * (1) Load data in registers and generate mask for elements that are within
 * range [-71476.0625f, 71476.0625f] for cosine and [-117435.992f, 117435.992f]
 * for sine.
 * (2) For elements within range, perform range reduction using
 * Cody-Waite's method: x* = x - y*PI/2, where y = rint(x*2/PI). x* \in [-PI/4,
 * PI/4].
 * (3) Map cos(x) to (+/-)sine or (+/-)cosine of x* based on the
 * quadrant k = int(y).
 * (4) For elements outside that range, Cody-Waite
 * reduction performs poorly leading to catastrophic cancellation. We compute
 * cosine by calling glibc in a scalar fashion.
 * (5) Vectorized implementation
 * has a max ULP of 1.49 and performs at least 5-7x(x86) - 2.5-3x(Power) -
 * 1-2x(Arm) faster than scalar implementations when magnitude of all elements
 * in the array < 71476.0625f (117435.992f for sine).  Worst case performance
 * is when all the elements are large leading to about 1-2% reduction in
 * performance.
 * TODO: use vectorized version of Payne-Hanek style reduction for large
 * elements or when there's no native FUSED support instead of fallback to libc
 */

#if NPY_SIMD_FMA3  // native support
typedef enum {
    SIMD_COMPUTE_SIN,
    SIMD_COMPUTE_COS
} SIMD_TRIG_OP;

const hn::ScalableTag<float> f32;
const hn::ScalableTag<int32_t> s32;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_s32 = hn::Vec<decltype(s32)>;
using opmask_t = hn::Mask<decltype(f32)>;

HWY_INLINE HWY_ATTR vec_f32
simd_range_reduction_f32(vec_f32 &x, vec_f32 &y, const vec_f32 &c1,
                         const vec_f32 &c2, const vec_f32 &c3)
{
    vec_f32 reduced_x = hn::MulAdd(y, c1, x);
    reduced_x = hn::MulAdd(y, c2, reduced_x);
    reduced_x = hn::MulAdd(y, c3, reduced_x);
    return reduced_x;
}

HWY_INLINE HWY_ATTR vec_f32
simd_cosine_poly_f32(vec_f32 &x2)
{
    const vec_f32 invf8 = hn::Set(f32, 0x1.98e616p-16f);
    const vec_f32 invf6 = hn::Set(f32, -0x1.6c06dcp-10f);
    const vec_f32 invf4 = hn::Set(f32, 0x1.55553cp-05f);
    const vec_f32 invf2 = hn::Set(f32, -0x1.000000p-01f);
    const vec_f32 invf0 = hn::Set(f32, 0x1.000000p+00f);

    vec_f32 r = hn::MulAdd(invf8, x2, invf6);
    r = hn::MulAdd(r, x2, invf4);
    r = hn::MulAdd(r, x2, invf2);
    r = hn::MulAdd(r, x2, invf0);
    return r;
}
/*
 * Approximate sine algorithm for x \in [-PI/4, PI/4]
 * Maximum ULP across all 32-bit floats = 0.647
 * Polynomial approximation based on unpublished work by T. Myklebust
 */
HWY_INLINE HWY_ATTR vec_f32
simd_sine_poly_f32(vec_f32 &x, vec_f32 &x2)
{
    const vec_f32 invf9 = hn::Set(f32, 0x1.7d3bbcp-19f);
    const vec_f32 invf7 = hn::Set(f32, -0x1.a06bbap-13f);
    const vec_f32 invf5 = hn::Set(f32, 0x1.11119ap-07f);
    const vec_f32 invf3 = hn::Set(f32, -0x1.555556p-03f);

    vec_f32 r = hn::MulAdd(invf9, x2, invf7);
    r = hn::MulAdd(r, x2, invf5);
    r = hn::MulAdd(r, x2, invf3);
    r = hn::MulAdd(r, x2, hn::Zero(f32));
    r = hn::MulAdd(r, x, x);
    return r;
}

static void HWY_ATTR SIMD_MSVC_NOINLINE
simd_sincos_f32(const float *src, npy_intp ssrc, float *dst, npy_intp sdst,
                npy_intp len, SIMD_TRIG_OP trig_op)
{
    // Load up frequently used constants
    const vec_f32 zerosf = hn::Zero(f32);
    const vec_s32 ones = hn::Set(s32, 1);
    const vec_s32 twos = hn::Set(s32, 2);
    const vec_f32 two_over_pi = hn::Set(f32, 0x1.45f306p-1f);
    const vec_f32 codyw_pio2_highf = hn::Set(f32, -0x1.921fb0p+00f);
    const vec_f32 codyw_pio2_medf = hn::Set(f32, -0x1.5110b4p-22f);
    const vec_f32 codyw_pio2_lowf = hn::Set(f32, -0x1.846988p-48f);
    const vec_f32 rint_cvt_magic = hn::Set(f32, 0x1.800000p+23f);
    // Cody-Waite's range
    float max_codi = 117435.992f;
    if (trig_op == SIMD_COMPUTE_COS) {
        max_codi = 71476.0625f;
    }
    const vec_f32 max_cody = hn::Set(f32, max_codi);

    const int lanes = hn::Lanes(f32);
    const vec_s32 src_index = hn::Mul(hn::Iota(s32, 0), hn::Set(s32, ssrc));
    const vec_s32 dst_index = hn::Mul(hn::Iota(s32, 0), hn::Set(s32, sdst));

    for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
        vec_f32 x_in;
        if (ssrc == 1) {
            x_in = hn::LoadN(f32, src, len);
        }
        else {
            x_in = hn::GatherIndexN(f32, src, src_index, len);
        }
        opmask_t nnan_mask = hn::Not(hn::IsNaN(x_in));
        // Eliminate NaN to avoid FP invalid exception
        x_in = hn::IfThenElse(nnan_mask, x_in, zerosf);
        opmask_t simd_mask = hn::Le(hn::Abs(x_in), max_cody);
        /*
         * For elements outside of this range, Cody-Waite's range reduction
         * becomes inaccurate and we will call libc to compute cosine for
         * these numbers
         */
        if (!hn::AllFalse(f32, simd_mask)) {
            vec_f32 x = hn::IfThenElse(hn::And(nnan_mask, simd_mask), x_in,
                                       zerosf);

            vec_f32 quadrant = hn::Mul(x, two_over_pi);
            // round to nearest, -0.0f -> +0.0f, and |a| must be <= 0x1.0p+22
            quadrant = hn::Add(quadrant, rint_cvt_magic);
            quadrant = hn::Sub(quadrant, rint_cvt_magic);

            // Cody-Waite's range reduction algorithm
            vec_f32 reduced_x =
                    simd_range_reduction_f32(x, quadrant, codyw_pio2_highf,
                                             codyw_pio2_medf, codyw_pio2_lowf);
            vec_f32 reduced_x2 = hn::Mul(reduced_x, reduced_x);

            // compute cosine and sine
            vec_f32 cos = simd_cosine_poly_f32(reduced_x2);
            vec_f32 sin = simd_sine_poly_f32(reduced_x, reduced_x2);

            vec_s32 iquadrant = hn::NearestInt(quadrant);
            if (trig_op == SIMD_COMPUTE_COS) {
                iquadrant = hn::Add(iquadrant, ones);
            }
            // blend sin and cos based on the quadrant
            opmask_t sine_mask = hn::RebindMask(
                    f32, hn::Eq(hn::And(iquadrant, ones), hn::Zero(s32)));
            cos = hn::IfThenElse(sine_mask, sin, cos);

            // multiply by -1 for appropriate elements
            opmask_t negate_mask = hn::RebindMask(
                    f32, hn::Eq(hn::And(iquadrant, twos), twos));
            cos = hn::MaskedSubOr(cos, negate_mask, zerosf, cos);
            cos = hn::IfThenElse(nnan_mask, cos, hn::Set(f32, NPY_NANF));

            if (sdst == 1) {
                hn::StoreN(cos, f32, dst, len);
            }
            else {
                hn::ScatterIndexN(cos, f32, dst, dst_index, len);
            }
        }
        if (!hn::AllTrue(f32, simd_mask)) {
            static_assert(hn::MaxLanes(f32) <= 64,
                          "The following fallback is not applicable for "
                          "SIMD widths larger than 2048 bits, or for scalable "
                          "SIMD in general.");
            npy_uint64 simd_maski;
            hn::StoreMaskBits(f32, simd_mask, (uint8_t *)&simd_maski);
#if HWY_IS_BIG_ENDIAN
            static_assert(hn::MaxLanes(f32) <= 8,
                          "This conversion is not supported for SIMD widths "
                          "larger than 256 bits.");
            simd_maski = ((uint8_t *)&simd_maski)[0];
#endif
            float NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) ip_fback[hn::Lanes(f32)];
            hn::Store(x_in, f32, ip_fback);

            // process elements using libc for large elements
            if (trig_op == SIMD_COMPUTE_COS) {
                for (unsigned i = 0; i < hn::Lanes(f32); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_cosf(ip_fback[i]);
                }
            }
            else {
                for (unsigned i = 0; i < hn::Lanes(f32); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_sinf(ip_fback[i]);
                }
            }
        }
        npyv_cleanup();
    }
}
#endif  // NPY_SIMD_FMA3

/* Disable SIMD code sin/cos f64 and revert to libm: see
 * https://mail.python.org/archives/list/numpy-discussion@python.org/thread/C6EYZZSR4EWGVKHAZXLE7IBILRMNVK7L/
 * for detailed discussion on this*/
#define DISPATCH_DOUBLE_FUNC(func)                                          \
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_##func)(               \
            char **args, npy_intp const *dimensions, npy_intp const *steps, \
            void *NPY_UNUSED(data))                                         \
    {                                                                       \
        UNARY_LOOP                                                          \
        {                                                                   \
            const npy_double in1 = *(npy_double *)ip1;                      \
            *(npy_double *)op1 = npy_##func(in1);                           \
        }                                                                   \
    }

DISPATCH_DOUBLE_FUNC(sin)
DISPATCH_DOUBLE_FUNC(cos)

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_sin)(char **args, npy_intp const *dimensions,
                                  npy_intp const *steps,
                                  void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3
    npy_intp len = dimensions[0];

    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len) ||
        !npyv_loadable_stride_f32(steps[0]) ||
        !npyv_storable_stride_f32(steps[1])) {
        UNARY_LOOP
        {
            simd_sincos_f32((npy_float *)ip1, 1, (npy_float *)op1, 1, 1,
                            SIMD_COMPUTE_SIN);
        }
    }
    else {
        const npy_float *src = (npy_float *)args[0];
        npy_float *dst = (npy_float *)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);

        simd_sincos_f32(src, ssrc, dst, sdst, len, SIMD_COMPUTE_SIN);
    }
#else
    UNARY_LOOP
    {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_sinf(in1);
    }
#endif
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_cos)(char **args, npy_intp const *dimensions,
                                  npy_intp const *steps,
                                  void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3
    npy_intp len = dimensions[0];

    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len) ||
        !npyv_loadable_stride_f32(steps[0]) ||
        !npyv_storable_stride_f32(steps[1])) {
        UNARY_LOOP
        {
            simd_sincos_f32((npy_float *)ip1, 1, (npy_float *)op1, 1, 1,
                            SIMD_COMPUTE_COS);
        }
    }
    else {
        const npy_float *src = (npy_float *)args[0];
        npy_float *dst = (npy_float *)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);

        simd_sincos_f32(src, ssrc, dst, sdst, len, SIMD_COMPUTE_COS);
    }
#else
    UNARY_LOOP
    {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_cosf(in1);
    }
#endif
}
