#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_ARITHMETIC_H
#define _NPY_SIMD_NEON_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  vaddq_u8
#define npyv_add_s8  vaddq_s8
#define npyv_add_u16 vaddq_u16
#define npyv_add_s16 vaddq_s16
#define npyv_add_u32 vaddq_u32
#define npyv_add_s32 vaddq_s32
#define npyv_add_u64 vaddq_u64
#define npyv_add_s64 vaddq_s64
#define npyv_add_f32 vaddq_f32
#define npyv_add_f64 vaddq_f64

// saturated
#define npyv_adds_u8  vqaddq_u8
#define npyv_adds_s8  vqaddq_s8
#define npyv_adds_u16 vqaddq_u16
#define npyv_adds_s16 vqaddq_s16

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  vsubq_u8
#define npyv_sub_s8  vsubq_s8
#define npyv_sub_u16 vsubq_u16
#define npyv_sub_s16 vsubq_s16
#define npyv_sub_u32 vsubq_u32
#define npyv_sub_s32 vsubq_s32
#define npyv_sub_u64 vsubq_u64
#define npyv_sub_s64 vsubq_s64
#define npyv_sub_f32 vsubq_f32
#define npyv_sub_f64 vsubq_f64

// saturated
#define npyv_subs_u8  vqsubq_u8
#define npyv_subs_s8  vqsubq_s8
#define npyv_subs_u16 vqsubq_u16
#define npyv_subs_s16 vqsubq_s16

/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  vmulq_u8
#define npyv_mul_s8  vmulq_s8
#define npyv_mul_u16 vmulq_u16
#define npyv_mul_s16 vmulq_s16
#define npyv_mul_u32 vmulq_u32
#define npyv_mul_s32 vmulq_s32
#define npyv_mul_f32 vmulq_f32
#define npyv_mul_f64 vmulq_f64

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const uint8x8_t mulc_lo = vget_low_u8(divisor.val[0]);
    // high part of unsigned multiplication
    uint16x8_t mull_lo  = vmull_u8(vget_low_u8(a), mulc_lo);
#if NPY_SIMD_F64
    uint16x8_t mull_hi  = vmull_high_u8(a, divisor.val[0]);
    // get the high unsigned bytes
    uint8x16_t mulhi    = vuzp2q_u8(vreinterpretq_u8_u16(mull_lo), vreinterpretq_u8_u16(mull_hi));
#else
    const uint8x8_t mulc_hi = vget_high_u8(divisor.val[0]);
    uint16x8_t mull_hi  = vmull_u8(vget_high_u8(a), mulc_hi);
    uint8x16_t mulhi    = vuzpq_u8(vreinterpretq_u8_u16(mull_lo), vreinterpretq_u8_u16(mull_hi)).val[1];
#endif
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    uint8x16_t q        = vsubq_u8(a, mulhi);
               q        = vshlq_u8(q, vreinterpretq_s8_u8(divisor.val[1]));
               q        = vaddq_u8(mulhi, q);
               q        = vshlq_u8(q, vreinterpretq_s8_u8(divisor.val[2]));
    return q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const int8x8_t mulc_lo = vget_low_s8(divisor.val[0]);
    // high part of signed multiplication
    int16x8_t mull_lo  = vmull_s8(vget_low_s8(a), mulc_lo);
#if NPY_SIMD_F64
    int16x8_t mull_hi  = vmull_high_s8(a, divisor.val[0]);
    // get the high unsigned bytes
    int8x16_t mulhi    = vuzp2q_s8(vreinterpretq_s8_s16(mull_lo), vreinterpretq_s8_s16(mull_hi));
#else
    const int8x8_t mulc_hi = vget_high_s8(divisor.val[0]);
    int16x8_t mull_hi  = vmull_s8(vget_high_s8(a), mulc_hi);
    int8x16_t mulhi    = vuzpq_s8(vreinterpretq_s8_s16(mull_lo), vreinterpretq_s8_s16(mull_hi)).val[1];
#endif
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    int8x16_t q        = vshlq_s8(vaddq_s8(a, mulhi), divisor.val[1]);
              q        = vsubq_s8(q, vshrq_n_s8(a, 7));
              q        = vsubq_s8(veorq_s8(q, divisor.val[2]), divisor.val[2]);
    return q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    const uint16x4_t mulc_lo = vget_low_u16(divisor.val[0]);
    // high part of unsigned multiplication
    uint32x4_t mull_lo  = vmull_u16(vget_low_u16(a), mulc_lo);
#if NPY_SIMD_F64
    uint32x4_t mull_hi  = vmull_high_u16(a, divisor.val[0]);
    // get the high unsigned bytes
    uint16x8_t mulhi    = vuzp2q_u16(vreinterpretq_u16_u32(mull_lo), vreinterpretq_u16_u32(mull_hi));
#else
    const uint16x4_t mulc_hi = vget_high_u16(divisor.val[0]);
    uint32x4_t mull_hi  = vmull_u16(vget_high_u16(a), mulc_hi);
    uint16x8_t mulhi    = vuzpq_u16(vreinterpretq_u16_u32(mull_lo), vreinterpretq_u16_u32(mull_hi)).val[1];
#endif
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    uint16x8_t q        = vsubq_u16(a, mulhi);
               q        = vshlq_u16(q, vreinterpretq_s16_u16(divisor.val[1]));
               q        = vaddq_u16(mulhi, q);
               q        = vshlq_u16(q, vreinterpretq_s16_u16(divisor.val[2]));
    return q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    const int16x4_t mulc_lo = vget_low_s16(divisor.val[0]);
    // high part of signed multiplication
    int32x4_t mull_lo  = vmull_s16(vget_low_s16(a), mulc_lo);
#if NPY_SIMD_F64
    int32x4_t mull_hi  = vmull_high_s16(a, divisor.val[0]);
    // get the high unsigned bytes
    int16x8_t mulhi    = vuzp2q_s16(vreinterpretq_s16_s32(mull_lo), vreinterpretq_s16_s32(mull_hi));
#else
    const int16x4_t mulc_hi = vget_high_s16(divisor.val[0]);
    int32x4_t mull_hi  = vmull_s16(vget_high_s16(a), mulc_hi);
    int16x8_t mulhi    = vuzpq_s16(vreinterpretq_s16_s32(mull_lo), vreinterpretq_s16_s32(mull_hi)).val[1];
#endif
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    int16x8_t q        = vshlq_s16(vaddq_s16(a, mulhi), divisor.val[1]);
              q        = vsubq_s16(q, vshrq_n_s16(a, 15));
              q        = vsubq_s16(veorq_s16(q, divisor.val[2]), divisor.val[2]);
    return q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    const uint32x2_t mulc_lo = vget_low_u32(divisor.val[0]);
    // high part of unsigned multiplication
    uint64x2_t mull_lo  = vmull_u32(vget_low_u32(a), mulc_lo);
#if NPY_SIMD_F64
    uint64x2_t mull_hi  = vmull_high_u32(a, divisor.val[0]);
    // get the high unsigned bytes
    uint32x4_t mulhi    = vuzp2q_u32(vreinterpretq_u32_u64(mull_lo), vreinterpretq_u32_u64(mull_hi));
#else
    const uint32x2_t mulc_hi = vget_high_u32(divisor.val[0]);
    uint64x2_t mull_hi  = vmull_u32(vget_high_u32(a), mulc_hi);
    uint32x4_t mulhi    = vuzpq_u32(vreinterpretq_u32_u64(mull_lo), vreinterpretq_u32_u64(mull_hi)).val[1];
#endif
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    uint32x4_t q        =  vsubq_u32(a, mulhi);
               q        =  vshlq_u32(q, vreinterpretq_s32_u32(divisor.val[1]));
               q        =  vaddq_u32(mulhi, q);
               q        =  vshlq_u32(q, vreinterpretq_s32_u32(divisor.val[2]));
    return q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    const int32x2_t mulc_lo = vget_low_s32(divisor.val[0]);
    // high part of signed multiplication
    int64x2_t mull_lo  = vmull_s32(vget_low_s32(a), mulc_lo);
#if NPY_SIMD_F64
    int64x2_t mull_hi  = vmull_high_s32(a, divisor.val[0]);
    // get the high unsigned bytes
    int32x4_t mulhi    = vuzp2q_s32(vreinterpretq_s32_s64(mull_lo), vreinterpretq_s32_s64(mull_hi));
#else
    const int32x2_t mulc_hi = vget_high_s32(divisor.val[0]);
    int64x2_t mull_hi  = vmull_s32(vget_high_s32(a), mulc_hi);
    int32x4_t mulhi    = vuzpq_s32(vreinterpretq_s32_s64(mull_lo), vreinterpretq_s32_s64(mull_hi)).val[1];
#endif
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    int32x4_t q        = vshlq_s32(vaddq_s32(a, mulhi), divisor.val[1]);
              q        = vsubq_s32(q, vshrq_n_s32(a, 31));
              q        = vsubq_s32(veorq_s32(q, divisor.val[2]), divisor.val[2]);
    return q;
}
// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    const uint64_t d = vgetq_lane_u64(divisor.val[0], 0);
    return npyv_set_u64(vgetq_lane_u64(a, 0) / d, vgetq_lane_u64(a, 1) / d);
}
// returns the high 64 bits of signed 64-bit multiplication
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    const int64_t d = vgetq_lane_s64(divisor.val[0], 0);
    return npyv_set_s64(vgetq_lane_s64(a, 0) / d, vgetq_lane_s64(a, 1) / d);
}
/***************************
 * Division
 ***************************/
#if NPY_SIMD_F64
    #define npyv_div_f32 vdivq_f32
#else
    NPY_FINLINE npyv_f32 npyv_div_f32(npyv_f32 a, npyv_f32 b)
    {
        // Based on ARM doc, see https://developer.arm.com/documentation/dui0204/j/CIHDIACI
        // estimate to 1/b
        npyv_f32 recipe = vrecpeq_f32(b);
        /**
         * Newton-Raphson iteration:
         *  x[n+1] = x[n] * (2-d * x[n])
         * converges to (1/d) if x0 is the result of VRECPE applied to d.
         *
         *  NOTE: at least 3 iterations is needed to improve precision
         */
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        recipe = vmulq_f32(vrecpsq_f32(b, recipe), recipe);
        // a/b = a*recip(b)
        return vmulq_f32(a, recipe);
    }
#endif
#define npyv_div_f64 vdivq_f64

/***************************
 * FUSED F32
 ***************************/
#ifdef NPY_HAVE_NEON_VFPV4 // FMA
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmaq_f32(c, a, b); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmaq_f32(vnegq_f32(c), a, b); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmsq_f32(c, a, b); }
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vfmsq_f32(vnegq_f32(c), a, b); }
#else
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlaq_f32(c, a, b); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlaq_f32(vnegq_f32(c), a, b); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlsq_f32(c, a, b); }
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return vmlsq_f32(vnegq_f32(c), a, b); }
#endif
// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
    const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
    return npyv_muladd_f32(a, b, npyv_xor_f32(msign, c));
}

/***************************
 * FUSED F64
 ***************************/
#if NPY_SIMD_F64
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmaq_f64(c, a, b); }
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmaq_f64(vnegq_f64(c), a, b); }
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmsq_f64(c, a, b); }
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return vfmsq_f64(vnegq_f64(c), a, b); }
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
        return npyv_muladd_f64(a, b, npyv_xor_f64(msign, c));
    }
#endif // NPY_SIMD_F64

/***************************
 * Summation
 ***************************/
// reduce sum across vector
#if NPY_SIMD_F64
    #define npyv_sum_u32 vaddvq_u32
    #define npyv_sum_u64 vaddvq_u64
    #define npyv_sum_f32 vaddvq_f32
    #define npyv_sum_f64 vaddvq_f64
#else
    NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
    {
        return vget_lane_u64(vadd_u64(vget_low_u64(a), vget_high_u64(a)),0);
    }

    NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
    {
        uint32x2_t a0 = vpadd_u32(vget_low_u32(a), vget_high_u32(a));
        return (unsigned)vget_lane_u32(vpadd_u32(a0, vget_high_u32(a)),0);
    }

    NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
    {
        float32x2_t r = vadd_f32(vget_high_f32(a), vget_low_f32(a));
        return vget_lane_f32(vpadd_f32(r, r), 0);
    }
#endif

// expand the source vector and performs sum reduce
#if NPY_SIMD_F64
    #define npyv_sumup_u8  vaddlvq_u8
    #define npyv_sumup_u16 vaddlvq_u16
#else
    NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
    {
        uint32x4_t t0 = vpaddlq_u16(vpaddlq_u8(a));
        uint32x2_t t1 = vpadd_u32(vget_low_u32(t0), vget_high_u32(t0));
        return vget_lane_u32(vpadd_u32(t1, t1), 0);
    }

    NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
    {
        uint32x4_t t0 = vpaddlq_u16(a);
        uint32x2_t t1 = vpadd_u32(vget_low_u32(t0), vget_high_u32(t0));
        return vget_lane_u32(vpadd_u32(t1, t1), 0);
    }
#endif

#endif // _NPY_SIMD_NEON_ARITHMETIC_H
