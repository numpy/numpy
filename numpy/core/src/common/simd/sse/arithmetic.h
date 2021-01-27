#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_ARITHMETIC_H
#define _NPY_SIMD_SSE_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  _mm_add_epi8
#define npyv_add_s8  _mm_add_epi8
#define npyv_add_u16 _mm_add_epi16
#define npyv_add_s16 _mm_add_epi16
#define npyv_add_u32 _mm_add_epi32
#define npyv_add_s32 _mm_add_epi32
#define npyv_add_u64 _mm_add_epi64
#define npyv_add_s64 _mm_add_epi64
#define npyv_add_f32 _mm_add_ps
#define npyv_add_f64 _mm_add_pd

// saturated
#define npyv_adds_u8  _mm_adds_epu8
#define npyv_adds_s8  _mm_adds_epi8
#define npyv_adds_u16 _mm_adds_epu16
#define npyv_adds_s16 _mm_adds_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  _mm_sub_epi8
#define npyv_sub_s8  _mm_sub_epi8
#define npyv_sub_u16 _mm_sub_epi16
#define npyv_sub_s16 _mm_sub_epi16
#define npyv_sub_u32 _mm_sub_epi32
#define npyv_sub_s32 _mm_sub_epi32
#define npyv_sub_u64 _mm_sub_epi64
#define npyv_sub_s64 _mm_sub_epi64
#define npyv_sub_f32 _mm_sub_ps
#define npyv_sub_f64 _mm_sub_pd

// saturated
#define npyv_subs_u8  _mm_subs_epu8
#define npyv_subs_s8  _mm_subs_epi8
#define npyv_subs_u16 _mm_subs_epu16
#define npyv_subs_s16 _mm_subs_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Multiplication
 ***************************/
// non-saturated
NPY_FINLINE __m128i npyv_mul_u8(__m128i a, __m128i b)
{
    const __m128i mask = _mm_set1_epi32(0xFF00FF00);
    __m128i even = _mm_mullo_epi16(a, b);
    __m128i odd  = _mm_mullo_epi16(_mm_srai_epi16(a, 8), _mm_srai_epi16(b, 8));
            odd  = _mm_slli_epi16(odd, 8);
    return npyv_select_u8(mask, odd, even);
}
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_u16 _mm_mullo_epi16
#define npyv_mul_s16 _mm_mullo_epi16

#ifdef NPY_HAVE_SSE41
    #define npyv_mul_u32 _mm_mullo_epi32
#else
    NPY_FINLINE __m128i npyv_mul_u32(__m128i a, __m128i b)
    {
        __m128i even = _mm_mul_epu32(a, b);
        __m128i odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
        __m128i low  = _mm_unpacklo_epi32(even, odd);
        __m128i high = _mm_unpackhi_epi32(even, odd);
        return _mm_unpacklo_epi64(low, high);
    }
#endif // NPY_HAVE_SSE41
#define npyv_mul_s32 npyv_mul_u32
// TODO: emulate 64-bit*/
#define npyv_mul_f32 _mm_mul_ps
#define npyv_mul_f64 _mm_mul_pd

// saturated
// TODO: after implment Packs intrins

/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm_div_ps
#define npyv_div_f64 _mm_div_pd
/***************************
 * FUSED
 ***************************/
#ifdef NPY_HAVE_FMA3
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_fmadd_ps
    #define npyv_muladd_f64 _mm_fmadd_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_fmsub_ps
    #define npyv_mulsub_f64 _mm_fmsub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_fnmadd_ps
    #define npyv_nmuladd_f64 _mm_fnmadd_pd
    // negate multiply and subtract, -(a*b) - c
    #define npyv_nmulsub_f32 _mm_fnmsub_ps
    #define npyv_nmulsub_f64 _mm_fnmsub_pd
#elif defined(NPY_HAVE_FMA4)
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm_macc_ps
    #define npyv_muladd_f64 _mm_macc_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm_msub_ps
    #define npyv_mulsub_f64 _mm_msub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm_nmacc_ps
    #define npyv_nmuladd_f64 _mm_nmacc_pd
#else
    // multiply and add, a*b + c
    NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_add_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_add_f64(npyv_mul_f64(a, b), c); }
    // multiply and subtract, a*b - c
    NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(npyv_mul_f64(a, b), c); }
    // negate multiply and add, -(a*b) + c
    NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return npyv_sub_f32(c, npyv_mul_f32(a, b)); }
    NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return npyv_sub_f64(c, npyv_mul_f64(a, b)); }
#endif // NPY_HAVE_FMA3
#ifndef NPY_HAVE_FMA3 // for FMA4 and NON-FMA3
    // negate multiply and subtract, -(a*b) - c
    NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        npyv_f32 neg_a = npyv_xor_f32(a, npyv_setall_f32(-0.0f));
        return npyv_sub_f32(npyv_mul_f32(neg_a, b), c);
    }
    NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        npyv_f64 neg_a = npyv_xor_f64(a, npyv_setall_f64(-0.0));
        return npyv_sub_f64(npyv_mul_f64(neg_a, b), c);
    }
#endif // !NPY_HAVE_FMA3

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    __m128i t = _mm_add_epi32(a, _mm_srli_si128(a, 8));
    t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
    return (unsigned)_mm_cvtsi128_si32(t);
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m128i one = _mm_add_epi64(a, _mm_unpackhi_epi64(a, a));
    return (npy_uint64)npyv128_cvtsi128_si64(one);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_SSE3
    __m128 sum_halves = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
#else
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4); 
#endif
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
#ifdef NPY_HAVE_SSE3
    return _mm_cvtsd_f64(_mm_hadd_pd(a, a));
#else
    return _mm_cvtsd_f64(_mm_add_pd(a, _mm_unpackhi_pd(a, a)));
#endif
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m128i two = _mm_sad_epu8(a, _mm_setzero_si128());
    __m128i one = _mm_add_epi16(two, _mm_unpackhi_epi64(two, two));
    return (npy_uint16)_mm_cvtsi128_si32(one);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    const __m128i even_mask = _mm_set1_epi32(0x0000FFFF);
    __m128i even = _mm_and_si128(a, even_mask);
    __m128i odd  = _mm_srli_epi32(a, 16);
    __m128i four = _mm_add_epi32(even, odd);
    return npyv_sum_u32(four);
}

#endif // _NPY_SIMD_SSE_ARITHMETIC_H


