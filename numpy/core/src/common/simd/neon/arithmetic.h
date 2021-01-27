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
