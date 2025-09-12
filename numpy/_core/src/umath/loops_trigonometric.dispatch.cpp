#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"

#include "simd/simd.h"
#include "simd/simd.hpp" 
#include <hwy/highway.h>

typedef enum {
    SIMD_COMPUTE_SIN,
    SIMD_COMPUTE_COS
} SIMD_TRIG_OP;

namespace {
using namespace np::simd;

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
#if NPY_HWY_FMA  // native support
HWY_INLINE Vec<float>
simd_range_reduction_f32(Vec<float> &x, Vec<float> &y, const Vec<float> &c1,
                         const Vec<float> &c2, const Vec<float> &c3)
{
    Vec<float> reduced_x = hn::MulAdd(y, c1, x);
    reduced_x = hn::MulAdd(y, c2, reduced_x);
    reduced_x = hn::MulAdd(y, c3, reduced_x);
    return reduced_x;
}

HWY_INLINE Vec<float>
simd_cosine_poly_f32(Vec<float> &x2)
{
    const Vec<float> invf8 = Set(float(0x1.98e616p-16f));
    const Vec<float> invf6 = Set(float(-0x1.6c06dcp-10f));
    const Vec<float> invf4 = Set(float(0x1.55553cp-05f));
    const Vec<float> invf2 = Set(float(-0x1.000000p-01f));
    const Vec<float> invf0 = Set(float(0x1.000000p+00f));

    Vec<float> r = hn::MulAdd(invf8, x2, invf6);
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
HWY_INLINE Vec<float>
simd_sine_poly_f32(Vec<float> &x, Vec<float> &x2)
{
    const Vec<float> invf9 = Set(float(0x1.7d3bbcp-19f));
    const Vec<float> invf7 = Set(float(-0x1.a06bbap-13f));
    const Vec<float> invf5 = Set(float(0x1.11119ap-07f));
    const Vec<float> invf3 = Set(float(-0x1.555556p-03f));

    Vec<float> r = hn::MulAdd(invf9, x2, invf7);
    r = hn::MulAdd(r, x2, invf5);
    r = hn::MulAdd(r, x2, invf3);
    r = hn::MulAdd(r, x2, Zero<float>());
    r = hn::MulAdd(r, x, x);
    return r;
}

static void SIMD_MSVC_NOINLINE
simd_sincos_f32(const float *src, npy_intp ssrc, float *dst, npy_intp sdst,
                npy_intp len, SIMD_TRIG_OP trig_op)
{
    // Load up frequently used constants
    const Vec<float> zerosf = Zero<float>();
    const Vec<int32_t> ones = Set(int32_t(1));
    const Vec<int32_t> twos = Set(int32_t(2));
    const Vec<float> two_over_pi      = Set(float(0x1.45f306p-1f));
    const Vec<float> codyw_pio2_highf = Set(float(-0x1.921fb0p+00f));
    const Vec<float> codyw_pio2_medf  = Set(float(-0x1.5110b4p-22f));
    const Vec<float> codyw_pio2_lowf  = Set(float(-0x1.846988p-48f));
    const Vec<float> rint_cvt_magic   = Set(float(0x1.800000p+23f));
    // Cody-Waite's range
    float max_codi = 117435.992f;
    if (trig_op == SIMD_COMPUTE_COS) {
        max_codi = 71476.0625f;
    }
    const Vec<float> max_cody = Set(float(max_codi));

    const int lanes = Lanes<float>();
    const Vec<int32_t> src_index = hn::Mul(hn::Iota(_Tag<int32_t>(), 0), Set(int32_t(ssrc)));
    const Vec<int32_t> dst_index = hn::Mul(hn::Iota(_Tag<int32_t>(), 0), Set(int32_t(sdst)));

    for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
        Vec<float> x_in = Zero<float>();
        if (ssrc == 1) {
            x_in = hn::LoadN(_Tag<float>(), src, len);
        }
        else {
            
            #if HWY_TARGET == HWY_RVV
                for (npy_intp i = 0; i < std::min(len, static_cast<npy_intp>(Lanes<int32_t>())); i++) {
                    float val = src[hn::ExtractLane(src_index, i)];
                    x_in = hn::InsertLane(x_in, i, val);
                }
            #else
                    x_in = hn::GatherIndexN(_Tag<float>(), src, src_index, len);
            #endif
        }
        Mask<float> nnan_mask = hn::Not(hn::IsNaN(x_in));
        // Eliminate NaN to avoid FP invalid exception
        x_in = hn::IfThenElse(nnan_mask, x_in, zerosf);
        Mask<float> simd_mask = hn::Le(hn::Abs(x_in), max_cody);
        /*
         * For elements outside of this range, Cody-Waite's range reduction
         * becomes inaccurate and we will call libc to compute cosine for
         * these numbers
         */
        if (!hn::AllFalse(_Tag<float>(), simd_mask)) {
            Vec<float> x = hn::IfThenElse(hn::And(nnan_mask, simd_mask), x_in,
                                       zerosf);

            Vec<float> quadrant = hn::Mul(x, two_over_pi);
            // round to nearest, -0.0f -> +0.0f, and |a| must be <= 0x1.0p+22
            quadrant = hn::Add(quadrant, rint_cvt_magic);
            quadrant = hn::Sub(quadrant, rint_cvt_magic);

            // Cody-Waite's range reduction algorithm
            Vec<float> reduced_x =
                    simd_range_reduction_f32(x, quadrant, codyw_pio2_highf,
                                             codyw_pio2_medf, codyw_pio2_lowf);
            Vec<float> reduced_x2 = hn::Mul(reduced_x, reduced_x);

            // compute cosine and sine
            Vec<float> cos = simd_cosine_poly_f32(reduced_x2);
            Vec<float> sin = simd_sine_poly_f32(reduced_x, reduced_x2);

            Vec<int32_t> iquadrant = hn::NearestInt(quadrant);
            if (trig_op == SIMD_COMPUTE_COS) {
                iquadrant = hn::Add(iquadrant, ones);
            }
            // blend sin and cos based on the quadrant
            Mask<float> sine_mask = hn::RebindMask(
                    _Tag<float>(), hn::Eq(hn::And(iquadrant, ones), Zero<int32_t>()));
            cos = hn::IfThenElse(sine_mask, sin, cos);

            // multiply by -1 for appropriate elements
            Mask<float> negate_mask = hn::RebindMask(
                    _Tag<float>(), hn::Eq(hn::And(iquadrant, twos), twos));
            cos = hn::MaskedSubOr(cos, negate_mask, zerosf, cos);
            cos = hn::IfThenElse(nnan_mask, cos, Set(float(NPY_NANF)));

            if (sdst == 1) {
                hn::StoreN(cos, _Tag<float>(), dst, len);
            }
            else {
                hn::ScatterIndexN(cos, _Tag<float>(), dst, dst_index, len);
            }
        }
        if (!hn::AllTrue(_Tag<float>(), simd_mask)) {
#if HWY_TARGET != HWY_RVV
            static_assert(hn::MaxLanes(_Tag<float>()) <= 64,
                      "The following fallback is not applicable for "
                      "SIMD widths larger than 2048 bits, or for scalable "
                      "SIMD in general.");
#endif
            npy_uint64 simd_maski = 0;
            hn::StoreMaskBits(_Tag<float>(), simd_mask, (uint8_t *)&simd_maski);
#if HWY_IS_BIG_ENDIAN
            static_assert(hn::MaxLanes(_Tag<float>()) <= 8,
                      "This conversion is not supported for SIMD widths "
                      "larger than 256 bits.");
            simd_maski = ((uint8_t *)&simd_maski)[0];
#endif
            float NPY_DECL_ALIGNED(kMaxLanes<uint8_t>) ip_fback[hn::MaxLanes(_Tag<float>())];
            hn::Store(x_in, _Tag<float>(), ip_fback);

            // process elements using libc for large elements
            if (trig_op == SIMD_COMPUTE_COS) {
                for (unsigned i = 0; i < Lanes<float>(); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_cosf(ip_fback[i]);
                }
            }
            else {
                for (unsigned i = 0; i < Lanes<float>(); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_sinf(ip_fback[i]);
                }
            }
        }
    }
}
#endif  // NPY_HWY_FMA

template <SIMD_TRIG_OP T>
HWY_INLINE void
sine_cosine(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];

#if NPY_HWY_FMA
        if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len) ||
            !(alignof(npy_float) == sizeof(npy_float) && src_step % sizeof(npy_float) == 0) ||
            !(alignof(npy_float) == sizeof(npy_float) && dst_step % sizeof(npy_float) == 0)
        ) {
            for (; len > 0; --len, src += src_step, dst += dst_step) {
                simd_sincos_f32((npy_float *)src, 1, (npy_float *)dst, 1, 1, T);
            }
        } else {
            const npy_intp ssrc = steps[0] / sizeof(npy_float);
            const npy_intp sdst = steps[1] / sizeof(npy_float);

            simd_sincos_f32((npy_float *)src, ssrc, (npy_float *)dst, sdst, len, T);
        }
#else
        for (; len > 0; --len, src += src_step, dst += dst_step) {
            const npy_float src0 = *reinterpret_cast<const npy_float*>(src);
            *reinterpret_cast<npy_float*>(dst) = (SIMD_COMPUTE_SIN==T)?npy_sinf(src0):npy_cosf(src0);
        }
#endif
}

} // anonymous namespace

/******************************************************************************************
 ** Defining ufunc inner functions
 *****************************************************************************************/
#define DEFINE_SINE_COSINE_FUNCTION(FUNC, T)                                 \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT##_##FUNC)                                 \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data)) \
{                                                                                        \
    sine_cosine<T>(args, dimensions, steps);                                \
}

DEFINE_SINE_COSINE_FUNCTION(sin, SIMD_COMPUTE_SIN)
DEFINE_SINE_COSINE_FUNCTION(cos, SIMD_COMPUTE_COS)

#undef DEFINE_SINE_COSINE_FUNCTION


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
