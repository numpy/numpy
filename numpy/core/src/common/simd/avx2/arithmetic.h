#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_ARITHMETIC_H
#define _NPY_SIMD_AVX2_ARITHMETIC_H

#include "../sse/utils.h"
/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  _mm256_add_epi8
#define npyv_add_s8  _mm256_add_epi8
#define npyv_add_u16 _mm256_add_epi16
#define npyv_add_s16 _mm256_add_epi16
#define npyv_add_u32 _mm256_add_epi32
#define npyv_add_s32 _mm256_add_epi32
#define npyv_add_u64 _mm256_add_epi64
#define npyv_add_s64 _mm256_add_epi64
#define npyv_add_f32 _mm256_add_ps
#define npyv_add_f64 _mm256_add_pd

// saturated
#define npyv_adds_u8  _mm256_adds_epu8
#define npyv_adds_s8  _mm256_adds_epi8
#define npyv_adds_u16 _mm256_adds_epu16
#define npyv_adds_s16 _mm256_adds_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  _mm256_sub_epi8
#define npyv_sub_s8  _mm256_sub_epi8
#define npyv_sub_u16 _mm256_sub_epi16
#define npyv_sub_s16 _mm256_sub_epi16
#define npyv_sub_u32 _mm256_sub_epi32
#define npyv_sub_s32 _mm256_sub_epi32
#define npyv_sub_u64 _mm256_sub_epi64
#define npyv_sub_s64 _mm256_sub_epi64
#define npyv_sub_f32 _mm256_sub_ps
#define npyv_sub_f64 _mm256_sub_pd

// saturated
#define npyv_subs_u8  _mm256_subs_epu8
#define npyv_subs_s8  _mm256_subs_epi8
#define npyv_subs_u16 _mm256_subs_epu16
#define npyv_subs_s16 _mm256_subs_epi16
// TODO: rest, after implment Packs intrins

/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  npyv256_mul_u8
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_u16 _mm256_mullo_epi16
#define npyv_mul_s16 _mm256_mullo_epi16
#define npyv_mul_u32 _mm256_mullo_epi32
#define npyv_mul_s32 _mm256_mullo_epi32
#define npyv_mul_f32 _mm256_mul_ps
#define npyv_mul_f64 _mm256_mul_pd

// saturated
// TODO: after implment Packs intrins

/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm256_div_ps
#define npyv_div_f64 _mm256_div_pd

/***************************
 * FUSED
 ***************************/
#ifdef NPY_HAVE_FMA3
    // multiply and add, a*b + c
    #define npyv_muladd_f32 _mm256_fmadd_ps
    #define npyv_muladd_f64 _mm256_fmadd_pd
    // multiply and subtract, a*b - c
    #define npyv_mulsub_f32 _mm256_fmsub_ps
    #define npyv_mulsub_f64 _mm256_fmsub_pd
    // negate multiply and add, -(a*b) + c
    #define npyv_nmuladd_f32 _mm256_fnmadd_ps
    #define npyv_nmuladd_f64 _mm256_fnmadd_pd
    // negate multiply and subtract, -(a*b) - c
    #define npyv_nmulsub_f32 _mm256_fnmsub_ps
    #define npyv_nmulsub_f64 _mm256_fnmsub_pd
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
    __m256i s0 = _mm256_hadd_epi32(a, a);
            s0 = _mm256_hadd_epi32(s0, s0);
    __m128i s1 = _mm256_extracti128_si256(s0, 1);;
            s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);
    return _mm_cvtsi128_si32(s1);
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m256i two = _mm256_add_epi64(a, _mm256_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128i one = _mm_add_epi64(_mm256_castsi256_si128(two), _mm256_extracti128_si256(two, 1));
    return (npy_uint64)npyv128_cvtsi128_si64(one);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    __m256 sum_halves = _mm256_hadd_ps(a, a);
    sum_halves = _mm256_hadd_ps(sum_halves, sum_halves);
    __m128 lo = _mm256_castps256_ps128(sum_halves);
    __m128 hi = _mm256_extractf128_ps(sum_halves, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return _mm_cvtss_f32(sum);
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    __m256d sum_halves = _mm256_hadd_pd(a, a);
    __m128d lo = _mm256_castpd256_pd128(sum_halves);
    __m128d hi = _mm256_extractf128_pd(sum_halves, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(sum);
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m256i four = _mm256_sad_epu8(a, _mm256_setzero_si256());
    __m128i two  = _mm_add_epi16(_mm256_castsi256_si128(four), _mm256_extracti128_si256(four, 1));
    __m128i one  = _mm_add_epi16(two, _mm_unpackhi_epi64(two, two));
    return (npy_uint16)_mm_cvtsi128_si32(one);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    const npyv_u16 even_mask = _mm256_set1_epi32(0x0000FFFF);
    __m256i even  = _mm256_and_si256(a, even_mask);
    __m256i odd   = _mm256_srli_epi32(a, 16);
    __m256i eight = _mm256_add_epi32(even, odd);
    return npyv_sum_u32(eight);
}

#endif // _NPY_SIMD_AVX2_ARITHMETIC_H


