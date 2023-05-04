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
// TODO: rest, after implement Packs intrins

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
// TODO: rest, after implement Packs intrins

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
// TODO: after implement Packs intrins

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const __m128i shf1  = _mm512_castsi512_si128(divisor.val[1]);
    const __m128i shf2  = _mm512_castsi512_si128(divisor.val[2]);
#ifdef NPY_HAVE_AVX512BW
    const __m512i bmask = _mm512_set1_epi32(0x00FF00FF);
    const __m512i shf1b = _mm512_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf1));
    const __m512i shf2b = _mm512_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf2));
    // high part of unsigned multiplication
    __m512i mulhi_even  = _mm512_mullo_epi16(_mm512_and_si512(a, bmask), divisor.val[0]);
            mulhi_even  = _mm512_srli_epi16(mulhi_even, 8);
    __m512i mulhi_odd   = _mm512_mullo_epi16(_mm512_srli_epi16(a, 8), divisor.val[0]);
    __m512i mulhi       = _mm512_mask_mov_epi8(mulhi_even, 0xAAAAAAAAAAAAAAAA, mulhi_odd);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m512i q           = _mm512_sub_epi8(a, mulhi);
            q           = _mm512_and_si512(_mm512_srl_epi16(q, shf1), shf1b);
            q           = _mm512_add_epi8(mulhi, q);
            q           = _mm512_and_si512(_mm512_srl_epi16(q, shf2), shf2b);
    return  q;
#else
    const __m256i bmask = _mm256_set1_epi32(0x00FF00FF);
    const __m256i shf1b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf1));
    const __m256i shf2b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf2));
    const __m512i shf2bw= npyv512_combine_si256(shf2b, shf2b);
    const __m256i mulc  = npyv512_lower_si256(divisor.val[0]);
    //// lower 256-bit
    __m256i lo_a        = npyv512_lower_si256(a);
    // high part of unsigned multiplication
    __m256i mulhi_even  = _mm256_mullo_epi16(_mm256_and_si256(lo_a, bmask), mulc);
            mulhi_even  = _mm256_srli_epi16(mulhi_even, 8);
    __m256i mulhi_odd   = _mm256_mullo_epi16(_mm256_srli_epi16(lo_a, 8), mulc);
    __m256i mulhi       = _mm256_blendv_epi8(mulhi_odd, mulhi_even, bmask);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i lo_q        = _mm256_sub_epi8(lo_a, mulhi);
            lo_q        = _mm256_and_si256(_mm256_srl_epi16(lo_q, shf1), shf1b);
            lo_q        = _mm256_add_epi8(mulhi, lo_q);
            lo_q        = _mm256_srl_epi16(lo_q, shf2); // no sign extend

    //// higher 256-bit
    __m256i hi_a        = npyv512_higher_si256(a);
    // high part of unsigned multiplication
            mulhi_even  = _mm256_mullo_epi16(_mm256_and_si256(hi_a, bmask), mulc);
            mulhi_even  = _mm256_srli_epi16(mulhi_even, 8);
            mulhi_odd   = _mm256_mullo_epi16(_mm256_srli_epi16(hi_a, 8), mulc);
            mulhi       = _mm256_blendv_epi8(mulhi_odd, mulhi_even, bmask);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i hi_q        = _mm256_sub_epi8(hi_a, mulhi);
            hi_q        = _mm256_and_si256(_mm256_srl_epi16(hi_q, shf1), shf1b);
            hi_q        = _mm256_add_epi8(mulhi, hi_q);
            hi_q        = _mm256_srl_epi16(hi_q, shf2); // no sign extend
    return _mm512_and_si512(npyv512_combine_si256(lo_q, hi_q), shf2bw); // extend sign
#endif
}
// divide each signed 8-bit element by divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    __m512i divc_even = npyv_divc_s16(npyv_shri_s16(npyv_shli_s16(a, 8), 8), divisor);
    __m512i divc_odd  = npyv_divc_s16(npyv_shri_s16(a, 8), divisor);
            divc_odd  = npyv_shli_s16(divc_odd, 8);
#ifdef NPY_HAVE_AVX512BW
    return _mm512_mask_mov_epi8(divc_even, 0xAAAAAAAAAAAAAAAA, divc_odd);
#else
    const __m512i bmask = _mm512_set1_epi32(0x00FF00FF);
    return npyv_select_u8(bmask, divc_even, divc_odd);
#endif
}
// divide each unsigned 16-bit element by divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    const __m128i shf2 = _mm512_castsi512_si128(divisor.val[2]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    #define NPYV__DIVC_U16(RLEN, A, MULC, R)      \
        mulhi = _mm##RLEN##_mulhi_epu16(A, MULC); \
        R     = _mm##RLEN##_sub_epi16(A, mulhi);  \
        R     = _mm##RLEN##_srl_epi16(R, shf1);   \
        R     = _mm##RLEN##_add_epi16(mulhi, R);  \
        R     = _mm##RLEN##_srl_epi16(R, shf2);

#ifdef NPY_HAVE_AVX512BW
    __m512i mulhi, q;
    NPYV__DIVC_U16(512, a, divisor.val[0], q)
    return q;
#else
    const __m256i m = npyv512_lower_si256(divisor.val[0]);
    __m256i lo_a    = npyv512_lower_si256(a);
    __m256i hi_a    = npyv512_higher_si256(a);

    __m256i mulhi, lo_q, hi_q;
    NPYV__DIVC_U16(256, lo_a, m, lo_q)
    NPYV__DIVC_U16(256, hi_a, m, hi_q)
    return npyv512_combine_si256(lo_q, hi_q);
#endif
    #undef NPYV__DIVC_U16
}
// divide each signed 16-bit element by divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    #define NPYV__DIVC_S16(RLEN, A, MULC, DSIGN, R)                       \
        mulhi  = _mm##RLEN##_mulhi_epi16(A, MULC);                        \
        R = _mm##RLEN##_sra_epi16(_mm##RLEN##_add_epi16(A, mulhi), shf1); \
        R = _mm##RLEN##_sub_epi16(R, _mm##RLEN##_srai_epi16(A, 15));      \
        R = _mm##RLEN##_sub_epi16(_mm##RLEN##_xor_si##RLEN(R, DSIGN), DSIGN);

#ifdef NPY_HAVE_AVX512BW
    __m512i mulhi, q;
    NPYV__DIVC_S16(512, a, divisor.val[0], divisor.val[2], q)
    return q;
#else
    const __m256i m     = npyv512_lower_si256(divisor.val[0]);
    const __m256i dsign = npyv512_lower_si256(divisor.val[2]);
    __m256i lo_a        = npyv512_lower_si256(a);
    __m256i hi_a        = npyv512_higher_si256(a);

    __m256i mulhi, lo_q, hi_q;
    NPYV__DIVC_S16(256, lo_a, m, dsign, lo_q)
    NPYV__DIVC_S16(256, hi_a, m, dsign, hi_q)
    return npyv512_combine_si256(lo_q, hi_q);
#endif
    #undef NPYV__DIVC_S16
}
// divide each unsigned 32-bit element by divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    const __m128i shf2 = _mm512_castsi512_si128(divisor.val[2]);
    // high part of unsigned multiplication
    __m512i mulhi_even = _mm512_srli_epi64(_mm512_mul_epu32(a, divisor.val[0]), 32);
    __m512i mulhi_odd  = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), divisor.val[0]);
    __m512i mulhi      = _mm512_mask_mov_epi32(mulhi_even, 0xAAAA, mulhi_odd);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m512i q          = _mm512_sub_epi32(a, mulhi);
            q          = _mm512_srl_epi32(q, shf1);
            q          = _mm512_add_epi32(mulhi, q);
            q          = _mm512_srl_epi32(q, shf2);
    return  q;
}
// divide each signed 32-bit element by divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    // high part of signed multiplication
    __m512i mulhi_even = _mm512_srli_epi64(_mm512_mul_epi32(a, divisor.val[0]), 32);
    __m512i mulhi_odd  = _mm512_mul_epi32(_mm512_srli_epi64(a, 32), divisor.val[0]);
    __m512i mulhi      = _mm512_mask_mov_epi32(mulhi_even, 0xAAAA, mulhi_odd);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m512i q          = _mm512_sra_epi32(_mm512_add_epi32(a, mulhi), shf1);
            q          = _mm512_sub_epi32(q, _mm512_srai_epi32(a, 31));
            q          = _mm512_sub_epi32(_mm512_xor_si512(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m512i lomask = npyv_setall_s64(0xffffffff);
    __m512i a_hi   = _mm512_srli_epi64(a, 32);        // a0l, a0h, a1l, a1h
    __m512i b_hi   = _mm512_srli_epi64(b, 32);        // b0l, b0h, b1l, b1h
    // compute partial products
    __m512i w0     = _mm512_mul_epu32(a, b);          // a0l*b0l, a1l*b1l
    __m512i w1     = _mm512_mul_epu32(a, b_hi);       // a0l*b0h, a1l*b1h
    __m512i w2     = _mm512_mul_epu32(a_hi, b);       // a0h*b0l, a1h*b0l
    __m512i w3     = _mm512_mul_epu32(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // sum partial products
    __m512i w0h    = _mm512_srli_epi64(w0, 32);
    __m512i s1     = _mm512_add_epi64(w1, w0h);
    __m512i s1l    = _mm512_and_si512(s1, lomask);
    __m512i s1h    = _mm512_srli_epi64(s1, 32);

    __m512i s2     = _mm512_add_epi64(w2, s1l);
    __m512i s2h    = _mm512_srli_epi64(s2, 32);

    __m512i hi     = _mm512_add_epi64(w3, s1h);
            hi     = _mm512_add_epi64(hi, s2h);
    return hi;
}
// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    const __m128i shf2 = _mm512_castsi512_si128(divisor.val[2]);
    // high part of unsigned multiplication
    __m512i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m512i q          = _mm512_sub_epi64(a, mulhi);
            q          = _mm512_srl_epi64(q, shf1);
            q          = _mm512_add_epi64(mulhi, q);
            q          = _mm512_srl_epi64(q, shf2);
    return  q;
}
// divide each unsigned 64-bit element by a divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    const __m128i shf1 = _mm512_castsi512_si128(divisor.val[1]);
    // high part of unsigned multiplication
    __m512i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    __m512i asign      = _mm512_srai_epi64(a, 63);
    __m512i msign      = _mm512_srai_epi64(divisor.val[0], 63);
    __m512i m_asign    = _mm512_and_si512(divisor.val[0], asign);
    __m512i a_msign    = _mm512_and_si512(a, msign);
            mulhi      = _mm512_sub_epi64(mulhi, m_asign);
            mulhi      = _mm512_sub_epi64(mulhi, a_msign);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m512i q          = _mm512_sra_epi64(_mm512_add_epi64(a, mulhi), shf1);
            q          = _mm512_sub_epi64(q, asign);
            q          = _mm512_sub_epi64(_mm512_xor_si512(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
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
// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
#define npyv_muladdsub_f32 _mm512_fmaddsub_ps
#define npyv_muladdsub_f64 _mm512_fmaddsub_pd

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
