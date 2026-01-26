#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_MATH_H
#define _NPY_SIMD_NEON_MATH_H

/***************************
 * Elementary
 ***************************/
// Absolute
#define npyv_abs_f32 vabsq_f32
#define npyv_abs_f64 vabsq_f64

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return vmulq_f32(a, a); }
#if NPY_SIMD_F64
    NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
    { return vmulq_f64(a, a); }
#endif

// Square root
#if NPY_SIMD_F64
    #define npyv_sqrt_f32 vsqrtq_f32
    #define npyv_sqrt_f64 vsqrtq_f64
#else
    // Based on ARM doc, see https://developer.arm.com/documentation/dui0204/j/CIHDIACI
    NPY_FINLINE npyv_f32 npyv_sqrt_f32(npyv_f32 a)
    {
        const npyv_f32 one = vdupq_n_f32(1.0f);
        const npyv_f32 zero = vdupq_n_f32(0.0f);
        const npyv_u32 pinf = vdupq_n_u32(0x7f800000);
        npyv_u32 is_zero = vceqq_f32(a, zero), is_inf = vceqq_u32(vreinterpretq_u32_f32(a), pinf);
        npyv_u32 is_special = vorrq_u32(is_zero, is_inf);
        // guard against division-by-zero and infinity input to vrsqrte to avoid invalid fp error
        npyv_f32 guard_byz = vbslq_f32(is_special, one, a);
        // estimate to (1/√a)
        npyv_f32 rsqrte = vrsqrteq_f32(guard_byz);
        /**
         * Newton-Raphson iteration:
         *  x[n+1] = x[n] * (3-d * (x[n]*x[n]) )/2)
         * converges to (1/√d)if x0 is the result of VRSQRTE applied to d.
         *
         * NOTE: at least 3 iterations is needed to improve precision
         */
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        rsqrte = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrte), rsqrte), rsqrte);
        // a * (1/√a)
        npyv_f32 sqrt = vmulq_f32(a, rsqrte);
        // Handle special cases: return a for zeros and positive infinities
        return vbslq_f32(is_special, a, sqrt);
    }
#endif // NPY_SIMD_F64

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{
#if NPY_SIMD_F64
    const npyv_f32 one = vdupq_n_f32(1.0f);
    return npyv_div_f32(one, a);
#else
    npyv_f32 recipe = vrecpeq_f32(a);
    /**
     * Newton-Raphson iteration:
     *  x[n+1] = x[n] * (2-d * x[n])
     * converges to (1/d) if x0 is the result of VRECPE applied to d.
     *
     * NOTE: at least 3 iterations is needed to improve precision
     */
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    recipe = vmulq_f32(vrecpsq_f32(a, recipe), recipe);
    return recipe;
#endif
}
#if NPY_SIMD_F64
    NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
    {
        const npyv_f64 one = vdupq_n_f64(1.0);
        return npyv_div_f64(one, a);
    }
#endif // NPY_SIMD_F64

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 vmaxq_f32
#define npyv_max_f64 vmaxq_f64
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#ifdef NPY_HAVE_ASIMD
    #define npyv_maxp_f32 vmaxnmq_f32
#else
    NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
    {
        npyv_u32 nn_a = vceqq_f32(a, a);
        npyv_u32 nn_b = vceqq_f32(b, b);
        return vmaxq_f32(vbslq_f32(nn_a, a, b), vbslq_f32(nn_b, b, a));
    }
#endif
// Max, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
#define npyv_maxn_f32 vmaxq_f32
#if NPY_SIMD_F64
    #define npyv_maxp_f64 vmaxnmq_f64
    #define npyv_maxn_f64 vmaxq_f64
#endif // NPY_SIMD_F64
// Maximum, integer operations
#define npyv_max_u8 vmaxq_u8
#define npyv_max_s8 vmaxq_s8
#define npyv_max_u16 vmaxq_u16
#define npyv_max_s16 vmaxq_s16
#define npyv_max_u32 vmaxq_u32
#define npyv_max_s32 vmaxq_s32
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return vbslq_u64(npyv_cmpgt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return vbslq_s64(npyv_cmpgt_s64(a, b), a, b);
}

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 vminq_f32
#define npyv_min_f64 vminq_f64
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#ifdef NPY_HAVE_ASIMD
    #define npyv_minp_f32 vminnmq_f32
#else
    NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
    {
        npyv_u32 nn_a = vceqq_f32(a, a);
        npyv_u32 nn_b = vceqq_f32(b, b);
        return vminq_f32(vbslq_f32(nn_a, a, b), vbslq_f32(nn_b, b, a));
    }
#endif
// Min, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
#define npyv_minn_f32 vminq_f32
#if NPY_SIMD_F64
    #define npyv_minp_f64 vminnmq_f64
    #define npyv_minn_f64 vminq_f64
#endif // NPY_SIMD_F64

// Minimum, integer operations
#define npyv_min_u8 vminq_u8
#define npyv_min_s8 vminq_s8
#define npyv_min_u16 vminq_u16
#define npyv_min_s16 vminq_s16
#define npyv_min_u32 vminq_u32
#define npyv_min_s32 vminq_s32
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    return vbslq_u64(npyv_cmplt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    return vbslq_s64(npyv_cmplt_s64(a, b), a, b);
}
// reduce min/max for all data types
#if NPY_SIMD_F64
    #define npyv_reduce_max_u8 vmaxvq_u8
    #define npyv_reduce_max_s8 vmaxvq_s8
    #define npyv_reduce_max_u16 vmaxvq_u16
    #define npyv_reduce_max_s16 vmaxvq_s16
    #define npyv_reduce_max_u32 vmaxvq_u32
    #define npyv_reduce_max_s32 vmaxvq_s32

    #define npyv_reduce_max_f32 vmaxvq_f32
    #define npyv_reduce_max_f64 vmaxvq_f64
    #define npyv_reduce_maxn_f32 vmaxvq_f32
    #define npyv_reduce_maxn_f64 vmaxvq_f64
    #define npyv_reduce_maxp_f32 vmaxnmvq_f32
    #define npyv_reduce_maxp_f64 vmaxnmvq_f64

    #define npyv_reduce_min_u8 vminvq_u8
    #define npyv_reduce_min_s8 vminvq_s8
    #define npyv_reduce_min_u16 vminvq_u16
    #define npyv_reduce_min_s16 vminvq_s16
    #define npyv_reduce_min_u32 vminvq_u32
    #define npyv_reduce_min_s32 vminvq_s32

    #define npyv_reduce_min_f32 vminvq_f32
    #define npyv_reduce_min_f64 vminvq_f64
    #define npyv_reduce_minn_f32 vminvq_f32
    #define npyv_reduce_minn_f64 vminvq_f64
    #define npyv_reduce_minp_f32 vminnmvq_f32
    #define npyv_reduce_minp_f64 vminnmvq_f64
#else
    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            STYPE##x8_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint8, u8)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint8, u8)
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int8, s8)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int8, s8)
    #undef NPY_IMPL_NEON_REDUCE_MINMAX

    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            STYPE##x4_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint16, u16)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint16, u16)
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int16, s16)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int16, s16)
    #undef NPY_IMPL_NEON_REDUCE_MINMAX

    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX)                            \
        NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)             \
        {                                                                              \
            STYPE##x2_t r = vp##INTRIN##_##SFX(vget_low_##SFX(a), vget_high_##SFX(a)); \
                        r = vp##INTRIN##_##SFX(r, r);                                  \
            return (npy_##STYPE)vget_lane_##SFX(r, 0);                                 \
        }
    NPY_IMPL_NEON_REDUCE_MINMAX(min, uint32, u32)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, uint32, u32)
    NPY_IMPL_NEON_REDUCE_MINMAX(min, int32, s32)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, int32, s32)
    #undef NPY_IMPL_NEON_REDUCE_MINMAX

    #define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, INF)                            \
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                \
        {                                                                       \
            float32x2_t r = vp##INTRIN##_f32(vget_low_f32(a), vget_high_f32(a));\
                        r = vp##INTRIN##_f32(r, r);                             \
            return vget_lane_f32(r, 0);                                         \
        }                                                                       \
        NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)               \
        {                                                                       \
            npyv_b32 notnan = npyv_notnan_f32(a);                               \
            if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                          \
                return vgetq_lane_f32(a, 0);                                    \
            }                                                                   \
            a = npyv_select_f32(notnan, a,                                      \
                    npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));            \
            return npyv_reduce_##INTRIN##_f32(a);                               \
        }                                                                       \
        NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)               \
        {                                                                       \
            return npyv_reduce_##INTRIN##_f32(a);                               \
        }
    NPY_IMPL_NEON_REDUCE_MINMAX(min, 0x7f800000)
    NPY_IMPL_NEON_REDUCE_MINMAX(max, 0xff800000)
    #undef NPY_IMPL_NEON_REDUCE_MINMAX
#endif // NPY_SIMD_F64
#define NPY_IMPL_NEON_REDUCE_MINMAX(INTRIN, STYPE, SFX, OP)       \
    NPY_FINLINE STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                             \
        STYPE al = (STYPE)vget_low_##SFX(a);                      \
        STYPE ah = (STYPE)vget_high_##SFX(a);                     \
        return al OP ah ? al : ah;                                \
    }
NPY_IMPL_NEON_REDUCE_MINMAX(max, npy_uint64, u64, >)
NPY_IMPL_NEON_REDUCE_MINMAX(max, npy_int64,  s64, >)
NPY_IMPL_NEON_REDUCE_MINMAX(min, npy_uint64, u64, <)
NPY_IMPL_NEON_REDUCE_MINMAX(min, npy_int64,  s64, <)
#undef NPY_IMPL_NEON_REDUCE_MINMAX

// round to nearest integer even
NPY_FINLINE npyv_f32 npyv_rint_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_ASIMD
    return vrndnq_f32(a);
#else
    // ARMv7 NEON only supports fp to int truncate conversion.
    // a magic trick of adding 1.5 * 2^23 is used for rounding
    // to nearest even and then subtract this magic number to get
    // the integer.
    //
    const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
    const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
    const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
    const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
    npyv_u32 nnan_mask = vceqq_f32(a, a);
    // eliminate nans to avoid invalid fp errors
    npyv_f32 abs_x = vabsq_f32(vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a))));
    // round by add magic number 1.5 * 2^23
    npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
    // copysign
    round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask ));
    // a if |a| >= 2^23 or a == NaN
    npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
             mask = vandq_u32(mask, nnan_mask);
    return vbslq_f32(mask, round, a);
#endif
}
#if NPY_SIMD_F64
    #define npyv_rint_f64 vrndnq_f64
#endif // NPY_SIMD_F64

// ceil
#ifdef NPY_HAVE_ASIMD
    #define npyv_ceil_f32 vrndpq_f32
#else
    NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
    {
        const npyv_u32 one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
        const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
        const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
        const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
        const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23
        npyv_u32 nnan_mask = vceqq_f32(a, a);
        npyv_f32 x = vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a)));
        // eliminate nans to avoid invalid fp errors
        npyv_f32 abs_x = vabsq_f32(x);
        // round by add magic number 1.5 * 2^23
        npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
        // copysign
        round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask));
        npyv_f32 ceil = vaddq_f32(round, vreinterpretq_f32_u32(
            vandq_u32(vcltq_f32(round, x), one))
        );
        // respects signed zero
        ceil = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(ceil), sign_mask));
        // a if |a| >= 2^23 or a == NaN
        npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
                 mask = vandq_u32(mask, nnan_mask);
        return vbslq_f32(mask, ceil, a);
   }
#endif
#if NPY_SIMD_F64
    #define npyv_ceil_f64 vrndpq_f64
#endif // NPY_SIMD_F64

// trunc
#ifdef NPY_HAVE_ASIMD
    #define npyv_trunc_f32 vrndq_f32
#else
    NPY_FINLINE npyv_f32 npyv_trunc_f32(npyv_f32 a)
    {
        const npyv_s32 max_int = vdupq_n_s32(0x7fffffff);
        const npyv_u32 exp_mask = vdupq_n_u32(0xff000000);
        const npyv_s32 szero = vreinterpretq_s32_f32(vdupq_n_f32(-0.0f));
        const npyv_u32 sign_mask = vandq_u32(
            vreinterpretq_u32_f32(a), vreinterpretq_u32_s32(szero));

        npyv_u32 nfinite_mask = vshlq_n_u32(vreinterpretq_u32_f32(a), 1);
                 nfinite_mask = vandq_u32(nfinite_mask, exp_mask);
                 nfinite_mask = vceqq_u32(nfinite_mask, exp_mask);
        // eliminate nans/inf to avoid invalid fp errors
        npyv_f32 x = vreinterpretq_f32_u32(
            veorq_u32(nfinite_mask, vreinterpretq_u32_f32(a)));
        /**
         * On armv7, vcvtq.f32 handles special cases as follows:
         *  NaN return 0
         * +inf or +outrange return 0x80000000(-0.0f)
         * -inf or -outrange return 0x7fffffff(nan)
         */
        npyv_s32 trunci = vcvtq_s32_f32(x);
        npyv_f32 trunc = vcvtq_f32_s32(trunci);
        // respect signed zero, e.g. -0.5 -> -0.0
        trunc = vreinterpretq_f32_u32(
            vorrq_u32(vreinterpretq_u32_f32(trunc), sign_mask));
        // if overflow return a
        npyv_u32 overflow_mask = vorrq_u32(
            vceqq_s32(trunci, szero), vceqq_s32(trunci, max_int)
        );
        // a if a overflow or nonfinite
        return vbslq_f32(vorrq_u32(nfinite_mask, overflow_mask), a, trunc);
   }
#endif
#if NPY_SIMD_F64
    #define npyv_trunc_f64 vrndq_f64
#endif // NPY_SIMD_F64

// floor
#ifdef NPY_HAVE_ASIMD
    #define npyv_floor_f32 vrndmq_f32
#else
    NPY_FINLINE npyv_f32 npyv_floor_f32(npyv_f32 a)
    {
        const npyv_u32 one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
        const npyv_u32 szero = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
        const npyv_u32 sign_mask = vandq_u32(vreinterpretq_u32_f32(a), szero);
        const npyv_f32 two_power_23 = vdupq_n_f32(8388608.0); // 2^23
        const npyv_f32 two_power_23h = vdupq_n_f32(12582912.0f); // 1.5 * 2^23

        npyv_u32 nnan_mask = vceqq_f32(a, a);
        npyv_f32 x = vreinterpretq_f32_u32(vandq_u32(nnan_mask, vreinterpretq_u32_f32(a)));
        // eliminate nans to avoid invalid fp errors
        npyv_f32 abs_x = vabsq_f32(x);
        // round by add magic number 1.5 * 2^23
        npyv_f32 round = vsubq_f32(vaddq_f32(two_power_23h, abs_x), two_power_23h);
        // copysign
        round = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(round), sign_mask));
        npyv_f32 floor = vsubq_f32(round, vreinterpretq_f32_u32(
            vandq_u32(vcgtq_f32(round, x), one)
        ));
        // respects signed zero
        floor = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(floor), sign_mask));
        // a if |a| >= 2^23 or a == NaN
        npyv_u32 mask = vcleq_f32(abs_x, two_power_23);
                 mask = vandq_u32(mask, nnan_mask);
        return vbslq_f32(mask, floor, a);
   }
#endif // NPY_HAVE_ASIMD
#if NPY_SIMD_F64
    #define npyv_floor_f64 vrndmq_f64
#endif // NPY_SIMD_F64

#endif // _NPY_SIMD_NEON_MATH_H
