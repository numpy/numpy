#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_ARITHMETIC_H
#define _NPY_SIMD_AVX2_ARITHMETIC_H

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

#endif // _NPY_SIMD_AVX2_ARITHMETIC_H
