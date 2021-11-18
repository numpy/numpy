#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MATH_H
#define _NPY_SIMD_AVX512_MATH_H

/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm512_sqrt_ps
#define npyv_sqrt_f64 _mm512_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm512_div_ps(_mm512_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm512_div_pd(_mm512_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_ps(a, a, 8);
#else
    return npyv_and_f32(
        a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff))
    );
#endif
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_pd(a, a, 8);
#else
    return npyv_and_f64(
        a, _mm512_castsi512_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
#endif
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm512_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm512_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm512_max_ps
#define npyv_max_f64 _mm512_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_pd(a, nn, a, b);
}
// Maximum, integer operations
#ifdef NPY_HAVE_AVX512BW
    #define npyv_max_u8 _mm512_max_epu8
    #define npyv_max_s8 _mm512_max_epi8
    #define npyv_max_u16 _mm512_max_epu16
    #define npyv_max_s16 _mm512_max_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u8, _mm256_max_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s8, _mm256_max_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u16, _mm256_max_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s16, _mm256_max_epi16)
#endif
#define npyv_max_u32 _mm512_max_epu32
#define npyv_max_s32 _mm512_max_epi32
#define npyv_max_u64 _mm512_max_epu64
#define npyv_max_s64 _mm512_max_epi64

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm512_min_ps
#define npyv_min_f64 _mm512_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_min_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_min_pd(a, nn, a, b);
}
// Minimum, integer operations
#ifdef NPY_HAVE_AVX512BW
    #define npyv_min_u8 _mm512_min_epu8
    #define npyv_min_s8 _mm512_min_epi8
    #define npyv_min_u16 _mm512_min_epu16
    #define npyv_min_s16 _mm512_min_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u8, _mm256_min_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s8, _mm256_min_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u16, _mm256_min_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s16, _mm256_min_epi16)
#endif
#define npyv_min_u32 _mm512_min_epu32
#define npyv_min_s32 _mm512_min_epi32
#define npyv_min_u64 _mm512_min_epu64
#define npyv_min_s64 _mm512_min_epi64

// ceil
#define npyv_ceil_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_POS_INF)
#define npyv_ceil_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_POS_INF)

// trunc
#define npyv_trunc_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_ZERO)
#define npyv_trunc_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_ZERO)

#endif // _NPY_SIMD_AVX512_MATH_H
