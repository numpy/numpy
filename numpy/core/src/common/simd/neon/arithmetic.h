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
#ifdef __aarch64__
    #define npyv_div_f32 vdivq_f32
#else
    NPY_FINLINE float32x4_t npyv_div_f32(float32x4_t a, float32x4_t b)
    {
        float32x4_t recip = vrecpeq_f32(b);
        recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
        return vmulq_f32(a, recip);
    }
#endif
#define npyv_div_f64 vdivq_f64

#endif // _NPY_SIMD_NEON_ARITHMETIC_H
