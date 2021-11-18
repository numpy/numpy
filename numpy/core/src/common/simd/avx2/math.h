#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_MATH_H
#define _NPY_SIMD_AVX2_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm256_sqrt_ps
#define npyv_sqrt_f64 _mm256_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm256_div_ps(_mm256_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm256_div_pd(_mm256_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm256_and_ps(
        a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm256_and_pd(
        a, _mm256_castsi256_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm256_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm256_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm256_max_ps
#define npyv_max_f64 _mm256_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    __m256 max = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(a, max, nn);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    __m256d max = _mm256_max_pd(a, b);
    return _mm256_blendv_pd(a, max, nn);
}
// Maximum, integer operations
#define npyv_max_u8 _mm256_max_epu8
#define npyv_max_s8 _mm256_max_epi8
#define npyv_max_u16 _mm256_max_epu16
#define npyv_max_s16 _mm256_max_epi16
#define npyv_max_u32 _mm256_max_epu32
#define npyv_max_s32 _mm256_max_epi32
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return _mm256_blendv_epi8(b, a, npyv_cmpgt_u64(a, b));
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
}

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm256_min_ps
#define npyv_min_f64 _mm256_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    __m256 min = _mm256_min_ps(a, b);
    return _mm256_blendv_ps(a, min, nn);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    __m256d min = _mm256_min_pd(a, b);
    return _mm256_blendv_pd(a, min, nn);
}
// Minimum, integer operations
#define npyv_min_u8 _mm256_min_epu8
#define npyv_min_s8 _mm256_min_epi8
#define npyv_min_u16 _mm256_min_epu16
#define npyv_min_s16 _mm256_min_epi16
#define npyv_min_u32 _mm256_min_epu32
#define npyv_min_s32 _mm256_min_epi32
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    return _mm256_blendv_epi8(b, a, npyv_cmplt_u64(a, b));
}
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
}

// ceil
#define npyv_ceil_f32 _mm256_ceil_ps
#define npyv_ceil_f64 _mm256_ceil_pd

// trunc
#define npyv_trunc_f32(A) _mm256_round_ps(A, _MM_FROUND_TO_ZERO)
#define npyv_trunc_f64(A) _mm256_round_pd(A, _MM_FROUND_TO_ZERO)

#endif // _NPY_SIMD_AVX2_MATH_H
