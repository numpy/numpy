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

// saturated
// TODO: after implment Packs intrins

/***************************
 * Division
 ***************************/
// TODO: emulate integer division
#define npyv_div_f32 _mm512_div_ps
#define npyv_div_f64 _mm512_div_pd

#endif // _NPY_SIMD_AVX512_ARITHMETIC_H
