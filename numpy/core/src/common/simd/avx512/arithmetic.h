#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_ARITHMETIC_H
#define _NPY_SIMD_AVX512_ARITHMETIC_H

#include "../avx2/utils.h"

/***************************
 * Addition
 ***************************/
// non-saturated
#ifdef NPY_HAVE_AVX512BW
    #define npyv_add_u8  _mm512_add_epi8
    #define npyv_add_u16 _mm512_add_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_add_u8,  _mm256_add_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_add_u16, _mm256_add_epi16)
#endif
#define npyv_add_s8  npyv_add_u8
#define npyv_add_s16 npyv_add_u16
#define npyv_add_u32 _mm512_add_epi32
#define npyv_add_s32 _mm512_add_epi32
#define npyv_add_u64 _mm512_add_epi64
#define npyv_add_s64 _mm512_add_epi64
#define npyv_add_f32 _mm512_add_ps
#define npyv_add_f64 _mm512_add_pd

// saturated
#ifdef NPY_HAVE_AVX512BW
    #define npyv_adds_u8  _mm512_adds_epu8
    #define npyv_adds_s8  _mm512_adds_epi8
    #define npyv_adds_u16 _mm512_adds_epu16
    #define npyv_adds_s16 _mm512_adds_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_adds_u8,  _mm256_adds_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_adds_s8,  _mm256_adds_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_adds_u16, _mm256_adds_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_adds_s16, _mm256_adds_epi16)
#endif
// TODO: rest, after implment Packs intrins

/***************************
 * Subtraction
 ***************************/
// non-saturated
#ifdef NPY_HAVE_AVX512BW
    #define npyv_sub_u8  _mm512_sub_epi8
    #define npyv_sub_u16 _mm512_sub_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_sub_u8,  _mm256_sub_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_sub_u16, _mm256_sub_epi16)
#endif
#define npyv_sub_s8  npyv_sub_u8
#define npyv_sub_s16 npyv_sub_u16
#define npyv_sub_u32 _mm512_sub_epi32
#define npyv_sub_s32 _mm512_sub_epi32
#define npyv_sub_u64 _mm512_sub_epi64
#define npyv_sub_s64 _mm512_sub_epi64
#define npyv_sub_f32 _mm512_sub_ps
#define npyv_sub_f64 _mm512_sub_pd

// saturated
#ifdef NPY_HAVE_AVX512BW
    #define npyv_subs_u8  _mm512_subs_epu8
    #define npyv_subs_s8  _mm512_subs_epi8
    #define npyv_subs_u16 _mm512_subs_epu16
    #define npyv_subs_s16 _mm512_subs_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_subs_u8,  _mm256_subs_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_subs_s8,  _mm256_subs_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_subs_u16, _mm256_subs_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_subs_s16, _mm256_subs_epi16)
#endif
// TODO: rest, after implment Packs intrins

/***************************
 * Multiplication
 ***************************/
// non-saturated
#ifdef NPY_HAVE_AVX512BW
NPY_FINLINE __m512i npyv_mul_u8(__m512i a, __m512i b)
{
    __m512i even = _mm512_mullo_epi16(a, b);
    __m512i odd  = _mm512_mullo_epi16(_mm512_srai_epi16(a, 8), _mm512_srai_epi16(b, 8));
            odd  = _mm512_slli_epi16(odd, 8);
    return _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, even, odd);
}
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_mul_u8, npyv256_mul_u8)
#endif

#ifdef NPY_HAVE_AVX512BW
    #define npyv_mul_u16 _mm512_mullo_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_mul_u16, _mm256_mullo_epi16)
#endif
#define npyv_mul_s8  npyv_mul_u8
#define npyv_mul_s16 npyv_mul_u16
#define npyv_mul_u32 _mm512_mullo_epi32
#define npyv_mul_s32 _mm512_mullo_epi32
#define npyv_mul_f32 _mm512_mul_ps
#define npyv_mul_f64 _mm512_mul_pd

#define npyv_muladd_f32 _mm512_fmadd_ps
#define npyv_muladd_f64 _mm512_fmadd_pd

// saturated
// TODO: after implment Packs intrins

/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm512_div_ps
#define npyv_div_f64 _mm512_div_pd
NPY_FINLINE float npyv__sum_f32(__m256 a)
{
    __m128 t1 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a,1));
    __m128 t2 = _mm_movehdup_ps(t1);
    __m128 t3 = _mm_add_ps(t1, t2);
    __m128 t4 = _mm_movehl_ps(t3, t3);
    __m128 t5 = _mm_add_ss(t3, t4);
    return _mm_cvtss_f32(t5);
}

NPY_FINLINE double npyv__sum_f64(__m256d a)
{
    __m128d t1 = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a,1));
    __m128d t2 = _mm_unpackhi_pd(t1, t1);
    __m128d t3 = _mm_add_sd(t2, t1);
    return _mm_cvtsd_f64(t3);
}

NPY_FINLINE __m256 get_low_f32(__m512 a)
{
    return _mm512_castps512_ps256(a);
}

NPY_FINLINE __m256 get_high_f32(__m512 a)
{
    return _mm512_extractf32x8_ps(a, 1);
}

NPY_FINLINE __m256d get_low_f64(__m512d a)
{
    return _mm512_castpd512_pd256(a);
}

NPY_FINLINE __m256d get_high_f64(__m512d a)
{
    return _mm512_extractf64x4_pd(a, 1);
}

// Horizontal add: Calculates the sum of all vector elements.
NPY_FINLINE float npyv_sum_f32(__m512 a)
{
#if defined(__INTEL_COMPILER)
    return _mm512_reduce_add_ps(a);
#else
    return npyv__sum_f32(_mm256_add_ps(get_low_f32(a), get_high_f32(a)));
#endif
}

NPY_FINLINE double npyv_sum_f64(__m512d a)
{
#if defined(__INTEL_COMPILER)
    return _mm512_reduce_add_pd(a);
#else
    return npyv__sum_f64(_mm256_add_pd(get_low_f64(a), get_high_f64(a)));
#endif
}

#endif // _NPY_SIMD_AVX512_ARITHMETIC_H
