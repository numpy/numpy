#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_MATH_H
#define _NPY_SIMD_VSX_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 vec_sqrt
#define npyv_sqrt_f64 vec_sqrt

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    return vec_div(one, a);
}
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{
    const npyv_f64 one = npyv_setall_f64(1.0);
    return vec_div(one, a);
}

// Absolute
#define npyv_abs_f32 vec_abs
#define npyv_abs_f64 vec_abs

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return vec_mul(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return vec_mul(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 vec_max
#define npyv_max_f64 vec_max
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
#define npyv_maxp_f32 vec_max
#define npyv_maxp_f64 vec_max
// Maximum, integer operations
#define npyv_max_u8 vec_max
#define npyv_max_s8 vec_max
#define npyv_max_u16 vec_max
#define npyv_max_s16 vec_max
#define npyv_max_u32 vec_max
#define npyv_max_s32 vec_max
#define npyv_max_u64 vec_max
#define npyv_max_s64 vec_max

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 vec_min
#define npyv_min_f64 vec_min
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
#define npyv_minp_f32 vec_min
#define npyv_minp_f64 vec_min
// Minimum, integer operations
#define npyv_min_u8 vec_min
#define npyv_min_s8 vec_min
#define npyv_min_u16 vec_min
#define npyv_min_s16 vec_min
#define npyv_min_u32 vec_min
#define npyv_min_s32 vec_min
#define npyv_min_u64 vec_min
#define npyv_min_s64 vec_min

// ceil
#define npyv_ceil_f32 vec_ceil
#define npyv_ceil_f64 vec_ceil

// trunc
#define npyv_trunc_f32 vec_trunc
#define npyv_trunc_f64 vec_trunc

#endif // _NPY_SIMD_VSX_MATH_H
