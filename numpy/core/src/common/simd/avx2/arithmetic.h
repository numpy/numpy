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
#endif // _NPY_SIMD_AVX2_ARITHMETIC_H
