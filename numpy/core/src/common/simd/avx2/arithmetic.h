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
// TODO: rest, after implement Packs intrins

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
// TODO: rest, after implement Packs intrins

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
// TODO: after implement Packs intrins

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const __m256i bmask = _mm256_set1_epi32(0x00FF00FF);
    const __m128i shf1  = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2  = _mm256_castsi256_si128(divisor.val[2]);
    const __m256i shf1b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf1));
    const __m256i shf2b = _mm256_set1_epi8(0xFFU >> _mm_cvtsi128_si32(shf2));
    // high part of unsigned multiplication
    __m256i mulhi_even  = _mm256_mullo_epi16(_mm256_and_si256(a, bmask), divisor.val[0]);
            mulhi_even  = _mm256_srli_epi16(mulhi_even, 8);
    __m256i mulhi_odd   = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8), divisor.val[0]);
    __m256i mulhi       = _mm256_blendv_epi8(mulhi_odd, mulhi_even, bmask);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q           = _mm256_sub_epi8(a, mulhi);
            q           = _mm256_and_si256(_mm256_srl_epi16(q, shf1), shf1b);
            q           = _mm256_add_epi8(mulhi, q);
            q           = _mm256_and_si256(_mm256_srl_epi16(q, shf2), shf2b);
    return  q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const __m256i bmask = _mm256_set1_epi32(0x00FF00FF);
    // instead of _mm256_cvtepi8_epi16/_mm256_packs_epi16 to wrap around overflow
    __m256i divc_even = npyv_divc_s16(_mm256_srai_epi16(_mm256_slli_epi16(a, 8), 8), divisor);
    __m256i divc_odd  = npyv_divc_s16(_mm256_srai_epi16(a, 8), divisor);
            divc_odd  = _mm256_slli_epi16(divc_odd, 8);
    return _mm256_blendv_epi8(divc_odd, divc_even, bmask);
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // high part of unsigned multiplication
    __m256i mulhi      = _mm256_mulhi_epu16(a, divisor.val[0]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi16(a, mulhi);
            q          = _mm256_srl_epi16(q, shf1);
            q          = _mm256_add_epi16(mulhi, q);
            q          = _mm256_srl_epi16(q, shf2);
    return  q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // high part of signed multiplication
    __m256i mulhi      = _mm256_mulhi_epi16(a, divisor.val[0]);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m256i q          = _mm256_sra_epi16(_mm256_add_epi16(a, mulhi), shf1);
            q          = _mm256_sub_epi16(q, _mm256_srai_epi16(a, 15));
            q          = _mm256_sub_epi16(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // high part of unsigned multiplication
    __m256i mulhi_even = _mm256_srli_epi64(_mm256_mul_epu32(a, divisor.val[0]), 32);
    __m256i mulhi_odd  = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), divisor.val[0]);
    __m256i mulhi      = _mm256_blend_epi32(mulhi_even, mulhi_odd, 0xAA);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi32(a, mulhi);
            q          = _mm256_srl_epi32(q, shf1);
            q          = _mm256_add_epi32(mulhi, q);
            q          = _mm256_srl_epi32(q, shf2);
    return  q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // high part of signed multiplication
    __m256i mulhi_even = _mm256_srli_epi64(_mm256_mul_epi32(a, divisor.val[0]), 32);
    __m256i mulhi_odd  = _mm256_mul_epi32(_mm256_srli_epi64(a, 32), divisor.val[0]);
    __m256i mulhi      = _mm256_blend_epi32(mulhi_even, mulhi_odd, 0xAA);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m256i q          = _mm256_sra_epi32(_mm256_add_epi32(a, mulhi), shf1);
            q          = _mm256_sub_epi32(q, _mm256_srai_epi32(a, 31));
            q          = _mm256_sub_epi32(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m256i lomask = npyv_setall_s64(0xffffffff);
    __m256i a_hi   = _mm256_srli_epi64(a, 32);        // a0l, a0h, a1l, a1h
    __m256i b_hi   = _mm256_srli_epi64(b, 32);        // b0l, b0h, b1l, b1h
    // compute partial products
    __m256i w0     = _mm256_mul_epu32(a, b);          // a0l*b0l, a1l*b1l
    __m256i w1     = _mm256_mul_epu32(a, b_hi);       // a0l*b0h, a1l*b1h
    __m256i w2     = _mm256_mul_epu32(a_hi, b);       // a0h*b0l, a1h*b0l
    __m256i w3     = _mm256_mul_epu32(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // sum partial products
    __m256i w0h    = _mm256_srli_epi64(w0, 32);
    __m256i s1     = _mm256_add_epi64(w1, w0h);
    __m256i s1l    = _mm256_and_si256(s1, lomask);
    __m256i s1h    = _mm256_srli_epi64(s1, 32);

    __m256i s2     = _mm256_add_epi64(w2, s1l);
    __m256i s2h    = _mm256_srli_epi64(s2, 32);

    __m256i hi     = _mm256_add_epi64(w3, s1h);
            hi     = _mm256_add_epi64(hi, s2h);
    return hi;
}
// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    const __m128i shf2 = _mm256_castsi256_si128(divisor.val[2]);
    // high part of unsigned multiplication
    __m256i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = _mm256_sub_epi64(a, mulhi);
            q          = _mm256_srl_epi64(q, shf1);
            q          = _mm256_add_epi64(mulhi, q);
            q          = _mm256_srl_epi64(q, shf2);
    return  q;
}
// divide each unsigned 64-bit element by a divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    const __m128i shf1 = _mm256_castsi256_si128(divisor.val[1]);
    // high part of unsigned multiplication
    __m256i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    __m256i asign      = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
    __m256i msign      = _mm256_cmpgt_epi64(_mm256_setzero_si256(), divisor.val[0]);
    __m256i m_asign    = _mm256_and_si256(divisor.val[0], asign);
    __m256i a_msign    = _mm256_and_si256(a, msign);
            mulhi      = _mm256_sub_epi64(mulhi, m_asign);
            mulhi      = _mm256_sub_epi64(mulhi, a_msign);
    // q               = (a + mulhi) >> sh
    __m256i q          = _mm256_add_epi64(a, mulhi);
    // emulate arithmetic right shift
    const __m256i sigb = npyv_setall_s64(1LL << 63);
            q          = _mm256_srl_epi64(_mm256_add_epi64(q, sigb), shf1);
            q          = _mm256_sub_epi64(q, _mm256_srl_epi64(sigb, shf1));
    // q               = q - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
            q          = _mm256_sub_epi64(q, asign);
            q          = _mm256_sub_epi64(_mm256_xor_si256(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
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
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    #define npyv_muladdsub_f32 _mm256_fmaddsub_ps
    #define npyv_muladdsub_f64 _mm256_fmaddsub_pd
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
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    { return _mm256_addsub_ps(npyv_mul_f32(a, b), c); }
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    { return _mm256_addsub_pd(npyv_mul_f64(a, b), c); }

#endif // !NPY_HAVE_FMA3

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    __m256i s0 = _mm256_hadd_epi32(a, a);
            s0 = _mm256_hadd_epi32(s0, s0);
    __m128i s1 = _mm256_extracti128_si256(s0, 1);
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
