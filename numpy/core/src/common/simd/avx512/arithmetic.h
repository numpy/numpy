#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_ARITHMETIC_H
#define _NPY_SIMD_AVX512_ARITHMETIC_H

#include "../avx2/utils.h"
#include "../sse/utils.h"
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

/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f32 _mm512_fmadd_ps
#define npyv_muladd_f64 _mm512_fmadd_pd
// multiply and subtract, a*b - c
#define npyv_mulsub_f32 _mm512_fmsub_ps
#define npyv_mulsub_f64 _mm512_fmsub_pd
// negate multiply and add, -(a*b) + c
#define npyv_nmuladd_f32 _mm512_fnmadd_ps
#define npyv_nmuladd_f64 _mm512_fnmadd_pd
// negate multiply and subtract, -(a*b) - c
#define npyv_nmulsub_f32 _mm512_fnmsub_ps
#define npyv_nmulsub_f64 _mm512_fnmsub_pd

/***************************
 * Summation: Calculates the sum of all vector elements.
 * there are three ways to implement reduce sum for AVX512:
 * 1- split(256) /add /split(128) /add /hadd /hadd /extract
 * 2- shuff(cross) /add /shuff(cross) /add /shuff /add /shuff /add /extract
 * 3- _mm512_reduce_add_ps/pd
 * The first one is been widely used by many projects
 * 
 * the second one is used by Intel Compiler, maybe because the
 * latency of hadd increased by (2-3) starting from Skylake-X which makes two
 * extra shuffles(non-cross) cheaper. check https://godbolt.org/z/s3G9Er for more info.
 * 
 * The third one is almost the same as the second one but only works for
 * intel compiler/GCC 7.1/Clang 4, we still need to support older GCC.
 ***************************/
// reduce sum across vector
#ifdef NPY_HAVE_AVX512F_REDUCE
    #define npyv_sum_u32 _mm512_reduce_add_epi32
    #define npyv_sum_u64 _mm512_reduce_add_epi64
    #define npyv_sum_f32 _mm512_reduce_add_ps
    #define npyv_sum_f64 _mm512_reduce_add_pd
#else
    NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
    {
        __m256i half = _mm256_add_epi32(npyv512_lower_si256(a), npyv512_higher_si256(a));
        __m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
        quarter = _mm_hadd_epi32(quarter, quarter);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(quarter, quarter));
    }

    NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
    {
        __m256i four = _mm256_add_epi64(npyv512_lower_si256(a), npyv512_higher_si256(a));
        __m256i two = _mm256_add_epi64(four, _mm256_shuffle_epi32(four, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128i one = _mm_add_epi64(_mm256_castsi256_si128(two), _mm256_extracti128_si256(two, 1));
        return (npy_uint64)npyv128_cvtsi128_si64(one);
    }

    NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
    {
        __m512 h64   = _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 sum32 = _mm512_add_ps(a, h64);
        __m512 h32   = _mm512_shuffle_f32x4(sum32, sum32, _MM_SHUFFLE(1, 0, 3, 2));
        __m512 sum16 = _mm512_add_ps(sum32, h32);
        __m512 h16   = _mm512_permute_ps(sum16, _MM_SHUFFLE(1, 0, 3, 2));
        __m512 sum8  = _mm512_add_ps(sum16, h16);
        __m512 h4    = _mm512_permute_ps(sum8, _MM_SHUFFLE(2, 3, 0, 1));
        __m512 sum4  = _mm512_add_ps(sum8, h4);
        return _mm_cvtss_f32(_mm512_castps512_ps128(sum4));
    }

    NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
    {
        __m512d h64   = _mm512_shuffle_f64x2(a, a, _MM_SHUFFLE(3, 2, 3, 2));
        __m512d sum32 = _mm512_add_pd(a, h64);
        __m512d h32   = _mm512_permutex_pd(sum32, _MM_SHUFFLE(1, 0, 3, 2));
        __m512d sum16 = _mm512_add_pd(sum32, h32);
        __m512d h16   = _mm512_permute_pd(sum16, _MM_SHUFFLE(2, 3, 0, 1));
        __m512d sum8  = _mm512_add_pd(sum16, h16);
        return _mm_cvtsd_f64(_mm512_castpd512_pd128(sum8));
    }
#endif

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
#ifdef NPY_HAVE_AVX512BW
    __m512i eight = _mm512_sad_epu8(a, _mm512_setzero_si512());
    __m256i four  = _mm256_add_epi16(npyv512_lower_si256(eight), npyv512_higher_si256(eight));
#else
    __m256i lo_four = _mm256_sad_epu8(npyv512_lower_si256(a), _mm256_setzero_si256());
    __m256i hi_four = _mm256_sad_epu8(npyv512_higher_si256(a), _mm256_setzero_si256());
    __m256i four    = _mm256_add_epi16(lo_four, hi_four);
#endif
    __m128i two     = _mm_add_epi16(_mm256_castsi256_si128(four), _mm256_extracti128_si256(four, 1));
    __m128i one     = _mm_add_epi16(two, _mm_unpackhi_epi64(two, two));
    return (npy_uint16)_mm_cvtsi128_si32(one);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    const npyv_u16 even_mask = _mm512_set1_epi32(0x0000FFFF);
    __m512i even = _mm512_and_si512(a, even_mask);
    __m512i odd  = _mm512_srli_epi32(a, 16);
    __m512i ff   = _mm512_add_epi32(even, odd);
    return npyv_sum_u32(ff);
}

#endif // _NPY_SIMD_AVX512_ARITHMETIC_H
