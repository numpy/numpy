#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MATH_H
#define _NPY_SIMD_SSE_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm_sqrt_ps
#define npyv_sqrt_f64 _mm_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm_div_ps(_mm_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm_div_pd(_mm_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm_and_ps(
        a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm_and_pd(
        a, _mm_castsi128_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm_max_ps
#define npyv_max_f64 _mm_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m128 nn  = _mm_cmpord_ps(b, b);
    __m128 max = _mm_max_ps(a, b);
    return npyv_select_f32(_mm_castps_si128(nn), max, a);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128d nn  = _mm_cmpord_pd(b, b);
    __m128d max = _mm_max_pd(a, b);
    return npyv_select_f64(_mm_castpd_si128(nn), max, a);
}
// Maximum, integer operations
#ifdef NPY_HAVE_SSE41
    #define npyv_max_s8 _mm_max_epi8
    #define npyv_max_u16 _mm_max_epu16
    #define npyv_max_u32 _mm_max_epu32
    #define npyv_max_s32 _mm_max_epi32
#else
    NPY_FINLINE npyv_s8 npyv_max_s8(npyv_s8 a, npyv_s8 b)
    {
        return npyv_select_s8(npyv_cmpgt_s8(a, b), a, b);
    }
    NPY_FINLINE npyv_u16 npyv_max_u16(npyv_u16 a, npyv_u16 b)
    {
        return npyv_select_u16(npyv_cmpgt_u16(a, b), a, b);
    }
    NPY_FINLINE npyv_u32 npyv_max_u32(npyv_u32 a, npyv_u32 b)
    {
        return npyv_select_u32(npyv_cmpgt_u32(a, b), a, b);
    }
    NPY_FINLINE npyv_s32 npyv_max_s32(npyv_s32 a, npyv_s32 b)
    {
        return npyv_select_s32(npyv_cmpgt_s32(a, b), a, b);
    }
#endif
#define npyv_max_u8 _mm_max_epu8
#define npyv_max_s16 _mm_max_epi16
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return npyv_select_u64(npyv_cmpgt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return npyv_select_s64(npyv_cmpgt_s64(a, b), a, b);
}

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm_min_ps
#define npyv_min_f64 _mm_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set. 
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __m128 nn  = _mm_cmpord_ps(b, b);
    __m128 min = _mm_min_ps(a, b);
    return npyv_select_f32(_mm_castps_si128(nn), min, a);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128d nn  = _mm_cmpord_pd(b, b);
    __m128d min = _mm_min_pd(a, b);
    return npyv_select_f64(_mm_castpd_si128(nn), min, a);
}
// Minimum, integer operations
#ifdef NPY_HAVE_SSE41
    #define npyv_min_s8 _mm_min_epi8
    #define npyv_min_u16 _mm_min_epu16
    #define npyv_min_u32 _mm_min_epu32
    #define npyv_min_s32 _mm_min_epi32
#else
    NPY_FINLINE npyv_s8 npyv_min_s8(npyv_s8 a, npyv_s8 b)
    {
        return npyv_select_s8(npyv_cmplt_s8(a, b), a, b);
    }
    NPY_FINLINE npyv_u16 npyv_min_u16(npyv_u16 a, npyv_u16 b)
    {
        return npyv_select_u16(npyv_cmplt_u16(a, b), a, b);
    }
    NPY_FINLINE npyv_u32 npyv_min_u32(npyv_u32 a, npyv_u32 b)
    {
        return npyv_select_u32(npyv_cmplt_u32(a, b), a, b);
    }
    NPY_FINLINE npyv_s32 npyv_min_s32(npyv_s32 a, npyv_s32 b)
    {
        return npyv_select_s32(npyv_cmplt_s32(a, b), a, b);
    }
#endif
#define npyv_min_u8 _mm_min_epu8
#define npyv_min_s16 _mm_min_epi16
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    return npyv_select_u64(npyv_cmplt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    return npyv_select_s64(npyv_cmplt_s64(a, b), a, b);
}

#endif // _NPY_SIMD_SSE_MATH_H
