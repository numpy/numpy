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
// TODO: rest, after implement Packs intrins

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
// TODO: rest, after implement Packs intrins

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
// TODO: after implement Packs intrins

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const __m128i bmask = _mm_set1_epi32(0x00FF00FF);
    const __m128i shf1b = _mm_set1_epi8(0xFFU >> _mm_cvtsi128_si32(divisor.val[1]));
    const __m128i shf2b = _mm_set1_epi8(0xFFU >> _mm_cvtsi128_si32(divisor.val[2]));
    // high part of unsigned multiplication
    __m128i mulhi_even  = _mm_mullo_epi16(_mm_and_si128(a, bmask), divisor.val[0]);
    __m128i mulhi_odd   = _mm_mullo_epi16(_mm_srli_epi16(a, 8), divisor.val[0]);
            mulhi_even  = _mm_srli_epi16(mulhi_even, 8);
    __m128i mulhi       = npyv_select_u8(bmask, mulhi_even, mulhi_odd);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q           = _mm_sub_epi8(a, mulhi);
            q           = _mm_and_si128(_mm_srl_epi16(q, divisor.val[1]), shf1b);
            q           = _mm_add_epi8(mulhi, q);
            q           = _mm_and_si128(_mm_srl_epi16(q, divisor.val[2]), shf2b);
    return  q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const __m128i bmask = _mm_set1_epi32(0x00FF00FF);
    // instead of _mm_cvtepi8_epi16/_mm_packs_epi16 to wrap around overflow
    __m128i divc_even = npyv_divc_s16(_mm_srai_epi16(_mm_slli_epi16(a, 8), 8), divisor);
    __m128i divc_odd  = npyv_divc_s16(_mm_srai_epi16(a, 8), divisor);
            divc_odd  = _mm_slli_epi16(divc_odd, 8);
    return npyv_select_u8(bmask, divc_even, divc_odd);
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = _mm_mulhi_epu16(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = _mm_sub_epi16(a, mulhi);
            q     = _mm_srl_epi16(q, divisor.val[1]);
            q     = _mm_add_epi16(mulhi, q);
            q     = _mm_srl_epi16(q, divisor.val[2]);
    return  q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // high part of signed multiplication
    __m128i mulhi = _mm_mulhi_epi16(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m128i q     = _mm_sra_epi16(_mm_add_epi16(a, mulhi), divisor.val[1]);
            q     = _mm_sub_epi16(q, _mm_srai_epi16(a, 15));
            q     = _mm_sub_epi16(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epu32(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), divisor.val[0]);
#ifdef NPY_HAVE_SSE41
    __m128i mulhi      = _mm_blend_epi16(mulhi_even, mulhi_odd, 0xCC);
#else
    __m128i mask_13    = _mm_setr_epi32(0, -1, 0, -1);
           mulhi_odd   = _mm_and_si128(mulhi_odd, mask_13);
    __m128i mulhi      = _mm_or_si128(mulhi_even, mulhi_odd);
#endif
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q          = _mm_sub_epi32(a, mulhi);
            q          = _mm_srl_epi32(q, divisor.val[1]);
            q          = _mm_add_epi32(mulhi, q);
            q          = _mm_srl_epi32(q, divisor.val[2]);
    return  q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    __m128i asign      = _mm_srai_epi32(a, 31);
#ifdef NPY_HAVE_SSE41
    // high part of signed multiplication
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epi32(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = _mm_mul_epi32(_mm_srli_epi64(a, 32), divisor.val[0]);
    __m128i mulhi      = _mm_blend_epi16(mulhi_even, mulhi_odd, 0xCC);
#else  // not SSE4.1
    // high part of "unsigned" multiplication
    __m128i mulhi_even = _mm_srli_epi64(_mm_mul_epu32(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = _mm_mul_epu32(_mm_srli_epi64(a, 32), divisor.val[0]);
    __m128i mask_13    = _mm_setr_epi32(0, -1, 0, -1);
            mulhi_odd  = _mm_and_si128(mulhi_odd, mask_13);
    __m128i mulhi      = _mm_or_si128(mulhi_even, mulhi_odd);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    const __m128i msign= _mm_srai_epi32(divisor.val[0], 31);
    __m128i m_asign    = _mm_and_si128(divisor.val[0], asign);
    __m128i a_msign    = _mm_and_si128(a, msign);
            mulhi      = _mm_sub_epi32(mulhi, m_asign);
            mulhi      = _mm_sub_epi32(mulhi, a_msign);
#endif
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m128i q          = _mm_sra_epi32(_mm_add_epi32(a, mulhi), divisor.val[1]);
            q          = _mm_sub_epi32(q, asign);
            q          = _mm_sub_epi32(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m128i lomask = npyv_setall_s64(0xffffffff);
    __m128i a_hi   = _mm_srli_epi64(a, 32);        // a0l, a0h, a1l, a1h
    __m128i b_hi   = _mm_srli_epi64(b, 32);        // b0l, b0h, b1l, b1h
    // compute partial products
    __m128i w0     = _mm_mul_epu32(a, b);          // a0l*b0l, a1l*b1l
    __m128i w1     = _mm_mul_epu32(a, b_hi);       // a0l*b0h, a1l*b1h
    __m128i w2     = _mm_mul_epu32(a_hi, b);       // a0h*b0l, a1h*b0l
    __m128i w3     = _mm_mul_epu32(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // sum partial products
    __m128i w0h    = _mm_srli_epi64(w0, 32);
    __m128i s1     = _mm_add_epi64(w1, w0h);
    __m128i s1l    = _mm_and_si128(s1, lomask);
    __m128i s1h    = _mm_srli_epi64(s1, 32);

    __m128i s2     = _mm_add_epi64(w2, s1l);
    __m128i s2h    = _mm_srli_epi64(s2, 32);

    __m128i hi     = _mm_add_epi64(w3, s1h);
            hi     = _mm_add_epi64(hi, s2h);
    return hi;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = _mm_sub_epi64(a, mulhi);
            q     = _mm_srl_epi64(q, divisor.val[1]);
            q     = _mm_add_epi64(mulhi, q);
            q     = _mm_srl_epi64(q, divisor.val[2]);
    return  q;
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
#ifdef NPY_HAVE_SSE42
    const __m128i msign= _mm_cmpgt_epi64(_mm_setzero_si128(), divisor.val[0]);
    __m128i asign      = _mm_cmpgt_epi64(_mm_setzero_si128(), a);
#else
    const __m128i msign= _mm_srai_epi32(_mm_shuffle_epi32(divisor.val[0], _MM_SHUFFLE(3, 3, 1, 1)), 31);
    __m128i asign      = _mm_srai_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1)), 31);
#endif
    __m128i m_asign    = _mm_and_si128(divisor.val[0], asign);
    __m128i a_msign    = _mm_and_si128(a, msign);
            mulhi      = _mm_sub_epi64(mulhi, m_asign);
            mulhi      = _mm_sub_epi64(mulhi, a_msign);
    // q               = (a + mulhi) >> sh
    __m128i q          = _mm_add_epi64(a, mulhi);
    // emulate arithmetic right shift
    const __m128i sigb = npyv_setall_s64(1LL << 63);
            q          = _mm_srl_epi64(_mm_add_epi64(q, sigb), divisor.val[1]);
            q          = _mm_sub_epi64(q, _mm_srl_epi64(sigb, divisor.val[1]));
    // q               = q - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
            q          = _mm_sub_epi64(q, asign);
            q          = _mm_sub_epi64(_mm_xor_si128(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
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
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    #define npyv_muladdsub_f32 _mm_fmaddsub_ps
    #define npyv_muladdsub_f64 _mm_fmaddsub_pd
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
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    #define npyv_muladdsub_f32 _mm_maddsub_ps
    #define npyv_muladdsub_f64 _mm_maddsub_pd
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
    // multiply, add for odd elements and subtract even elements.
    // (a * b) -+ c
    NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        npyv_f32 m = npyv_mul_f32(a, b);
    #ifdef NPY_HAVE_SSE3
        return _mm_addsub_ps(m, c);
    #else
        const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
        return npyv_add_f32(m, npyv_xor_f32(msign, c));
    #endif
    }
    NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        npyv_f64 m = npyv_mul_f64(a, b);
    #ifdef NPY_HAVE_SSE3
        return _mm_addsub_pd(m, c);
    #else
        const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
        return npyv_add_f64(m, npyv_xor_f64(msign, c));
    #endif
    }
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


