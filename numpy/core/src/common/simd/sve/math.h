#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_MATH_H
#define _NPY_SIMD_SVE_MATH_H

/***************************
 * Elementary
 ***************************/
// Square root
NPY_FINLINE npyv_f32
npyv_sqrt_f32(npyv_f32 a)
{
    return svsqrt_f32_x(svptrue_b32(), a);
}
NPY_FINLINE npyv_f64
npyv_sqrt_f64(npyv_f64 a)
{
    return svsqrt_f64_x(svptrue_b64(), a);
}

// Reciprocal
NPY_FINLINE npyv_f32
npyv_recip_f32(npyv_f32 a)
{
    return svdiv_f32_x(svptrue_b32(), svdup_f32(1.0f), a);
}
NPY_FINLINE npyv_f64
npyv_recip_f64(npyv_f64 a)
{
    return svdiv_f64_x(svptrue_b64(), svdup_f64(1.0f), a);
}

// Absolute
NPY_FINLINE npyv_f32
npyv_abs_f32(npyv_f32 a)
{
    return svabs_f32_x(svptrue_b32(), a);
}
NPY_FINLINE npyv_f64
npyv_abs_f64(npyv_f64 a)
{
    return svabs_f64_x(svptrue_b64(), a);
}

// Square
NPY_FINLINE npyv_f32
npyv_square_f32(npyv_f32 a)
{
    return svmul_f32_x(svptrue_b32(), a, a);
}
NPY_FINLINE npyv_f64
npyv_square_f64(npyv_f64 a)
{
    return svmul_f64_x(svptrue_b64(), a, a);
}

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32(A, B) svmax_f32_x(svptrue_b32(), A, B)
#define npyv_max_f64(A, B) svmax_f64_x(svptrue_b64(), A, B)
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the
// other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#define npyv_maxp_f32(A, B) svmaxnm_f32_x(svptrue_b32(), A, B)
#define npyv_maxp_f64(A, B) svmaxnm_f64_x(svptrue_b64(), A, B)

// Maximum, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
#define npyv_maxn_f32(A, B) svmax_f32_x(svptrue_b32(), A, B)
#define npyv_maxn_f64(A, B) svmax_f64_x(svptrue_b64(), A, B)

// Maximum, integer operations
#define npyv_max_u8(A, B) svmax_u8_x(svptrue_b8(), A, B)
#define npyv_max_u16(A, B) svmax_u16_x(svptrue_b16(), A, B)
#define npyv_max_u32(A, B) svmax_u32_x(svptrue_b32(), A, B)
#define npyv_max_u64(A, B) svmax_u64_x(svptrue_b64(), A, B)
#define npyv_max_s8(A, B) svmax_s8_x(svptrue_b8(), A, B)
#define npyv_max_s16(A, B) svmax_s16_x(svptrue_b16(), A, B)
#define npyv_max_s32(A, B) svmax_s32_x(svptrue_b32(), A, B)
#define npyv_max_s64(A, B) svmax_s64_x(svptrue_b64(), A, B)

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32(A, B) svmin_f32_x(svptrue_b32(), A, B)
#define npyv_min_f64(A, B) svmin_f64_x(svptrue_b64(), A, B)
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the
// other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#define npyv_minp_f32(A, B) svminnm_f32_x(svptrue_b32(), A, B)
#define npyv_minp_f64(A, B) svminnm_f64_x(svptrue_b64(), A, B)

// Mininum, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
#define npyv_minn_f32(A, B) svmin_f32_x(svptrue_b32(), A, B)
#define npyv_minn_f64(A, B) svmin_f64_x(svptrue_b64(), A, B)

// Minimum, integer operations
#define npyv_min_u8(A, B) svmin_u8_x(svptrue_b8(), A, B)
#define npyv_min_u16(A, B) svmin_u16_x(svptrue_b16(), A, B)
#define npyv_min_u32(A, B) svmin_u32_x(svptrue_b32(), A, B)
#define npyv_min_u64(A, B) svmin_u64_x(svptrue_b64(), A, B)
#define npyv_min_s8(A, B) svmin_s8_x(svptrue_b8(), A, B)
#define npyv_min_s16(A, B) svmin_s16_x(svptrue_b16(), A, B)
#define npyv_min_s32(A, B) svmin_s32_x(svptrue_b32(), A, B)
#define npyv_min_s64(A, B) svmin_s64_x(svptrue_b64(), A, B)

#define npyv_reduce_min_u8(A) svminv_u8(svptrue_b8(), A)
#define npyv_reduce_min_u16(A) svminv_u16(svptrue_b16(), A)
#define npyv_reduce_min_u32(A) svminv_u32(svptrue_b32(), A)
#define npyv_reduce_min_u64(A) svminv_u64(svptrue_b64(), A)
#define npyv_reduce_min_s8(A) svminv_s8(svptrue_b8(), A)
#define npyv_reduce_min_s16(A) svminv_s16(svptrue_b16(), A)
#define npyv_reduce_min_s32(A) svminv_s32(svptrue_b32(), A)
#define npyv_reduce_min_s64(A) svminv_s64(svptrue_b64(), A)
#define npyv_reduce_min_f32(A) svminv_f32(svptrue_b32(), A)
#define npyv_reduce_min_f64(A) svminv_f64(svptrue_b64(), A)
#define npyv_reduce_minn_f32(A) svminv_f32(svptrue_b32(), A)
#define npyv_reduce_minn_f64(A) svminv_f64(svptrue_b64(), A)
#define npyv_reduce_minp_f32(A) svminnmv_f32(svptrue_b32(), A)
#define npyv_reduce_minp_f64(A) svminnmv_f64(svptrue_b64(), A)

#define npyv_reduce_max_u8(A) svmaxv_u8(svptrue_b8(), A)
#define npyv_reduce_max_u16(A) svmaxv_u16(svptrue_b16(), A)
#define npyv_reduce_max_u32(A) svmaxv_u32(svptrue_b32(), A)
#define npyv_reduce_max_u64(A) svmaxv_u64(svptrue_b64(), A)
#define npyv_reduce_max_s8(A) svmaxv_s8(svptrue_b8(), A)
#define npyv_reduce_max_s16(A) svmaxv_s16(svptrue_b16(), A)
#define npyv_reduce_max_s32(A) svmaxv_s32(svptrue_b32(), A)
#define npyv_reduce_max_s64(A) svmaxv_s64(svptrue_b64(), A)
#define npyv_reduce_max_f32(A) svmaxv_f32(svptrue_b32(), A)
#define npyv_reduce_max_f64(A) svmaxv_f64(svptrue_b64(), A)
#define npyv_reduce_maxn_f32(A) svmaxv_f32(svptrue_b32(), A)
#define npyv_reduce_maxn_f64(A) svmaxv_f64(svptrue_b64(), A)
#define npyv_reduce_maxp_f32(A) svmaxnmv_f32(svptrue_b32(), A)
#define npyv_reduce_maxp_f64(A) svmaxnmv_f64(svptrue_b64(), A)

// round to nearest integer even
#define npyv_rint_f32(A) svrintn_f32_x(svptrue_b32(), A)
#define npyv_rint_f64(A) svrintn_f64_x(svptrue_b64(), A)

// ceil
#define npyv_ceil_f32(A) svrintp_f32_x(svptrue_b32(), A)
#define npyv_ceil_f64(A) svrintp_f64_x(svptrue_b64(), A)

// trunc
#define npyv_trunc_f32(A) svrintz_f32_x(svptrue_b32(), A)
#define npyv_trunc_f64(A) svrintz_f64_x(svptrue_b64(), A)

// floor
#define npyv_floor_f32(A) svrintm_f32_x(svptrue_b32(), A)
#define npyv_floor_f64(A) svrintm_f64_x(svptrue_b64(), A)

#endif  // _NPY_SIMD_SVE_MATH_H
