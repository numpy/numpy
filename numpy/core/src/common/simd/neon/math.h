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
        const npyv_f32 zero = vdupq_n_f32(0.0f);
        const npyv_u32 pinf = vdupq_n_u32(0x7f800000);
        npyv_u32 is_zero = vceqq_f32(a, zero), is_inf = vceqq_u32(vreinterpretq_u32_f32(a), pinf);
        // guard against floating-point division-by-zero error
        npyv_f32 guard_byz = vbslq_f32(is_zero, vreinterpretq_f32_u32(pinf), a);
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
        // return zero if the a is zero
        // - return zero if a is zero.
        // - return positive infinity if a is positive infinity
        return vbslq_f32(vorrq_u32(is_zero, is_inf), a, sqrt);
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
#if NPY_SIMD_F64
    #define npyv_maxp_f64 vmaxnmq_f64
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
#if NPY_SIMD_F64
    #define npyv_minp_f64 vminnmq_f64
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

// ceil
#ifdef NPY_HAVE_ASIMD
    #define npyv_ceil_f32 vrndpq_f32
#else
   NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
   {
        const npyv_s32 szero = vreinterpretq_s32_f32(vdupq_n_f32(-0.0f));
        const npyv_u32 one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
        const npyv_s32 max_int = vdupq_n_s32(0x7fffffff);
        /**
         * On armv7, vcvtq.f32 handles special cases as follows:
         *  NaN return 0
         * +inf or +outrange return 0x80000000(-0.0f)
         * -inf or -outrange return 0x7fffffff(nan)
         */
        npyv_s32 roundi = vcvtq_s32_f32(a);
        npyv_f32 round = vcvtq_f32_s32(roundi);
        npyv_f32 ceil = vaddq_f32(round, vreinterpretq_f32_u32(
            vandq_u32(vcltq_f32(round, a), one))
        );
        // respect signed zero, e.g. -0.5 -> -0.0
        npyv_f32 rzero = vreinterpretq_f32_s32(vorrq_s32(
            vreinterpretq_s32_f32(ceil),
            vandq_s32(vreinterpretq_s32_f32(a), szero)
        ));
        // if nan or overflow return a
        npyv_u32 nnan = npyv_notnan_f32(a);
        npyv_u32 overflow = vorrq_u32(
            vceqq_s32(roundi, szero), vceqq_s32(roundi, max_int)
        );
        return vbslq_f32(vbicq_u32(nnan, overflow), rzero, a);
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
        const npyv_s32 szero = vreinterpretq_s32_f32(vdupq_n_f32(-0.0f));
        const npyv_s32 max_int = vdupq_n_s32(0x7fffffff);
        /**
         * On armv7, vcvtq.f32 handles special cases as follows:
         *  NaN return 0
         * +inf or +outrange return 0x80000000(-0.0f)
         * -inf or -outrange return 0x7fffffff(nan)
         */
        npyv_s32 roundi = vcvtq_s32_f32(a);
        npyv_f32 round = vcvtq_f32_s32(roundi);
        // respect signed zero, e.g. -0.5 -> -0.0
        npyv_f32 rzero = vreinterpretq_f32_s32(vorrq_s32(
            vreinterpretq_s32_f32(round),
            vandq_s32(vreinterpretq_s32_f32(a), szero)
        ));
        // if nan or overflow return a
        npyv_u32 nnan = npyv_notnan_f32(a);
        npyv_u32 overflow = vorrq_u32(
            vceqq_s32(roundi, szero), vceqq_s32(roundi, max_int)
        );
        return vbslq_f32(vbicq_u32(nnan, overflow), rzero, a);
   }
#endif
#if NPY_SIMD_F64
    #define npyv_trunc_f64 vrndq_f64
#endif // NPY_SIMD_F64

#endif // _NPY_SIMD_NEON_MATH_H
