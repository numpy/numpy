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

#endif // _NPY_SIMD_SSE_ARITHMETIC_H
