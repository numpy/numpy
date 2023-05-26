#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_CVT_H
#define _NPY_SIMD_AVX512_CVT_H

// convert mask to integer vectors
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cvt_u8_b8  _mm512_movm_epi8
    #define npyv_cvt_u16_b16 _mm512_movm_epi16
#else
    #define npyv_cvt_u8_b8(BL) BL
    #define npyv_cvt_u16_b16(BL) BL
#endif
#define npyv_cvt_s8_b8  npyv_cvt_u8_b8
#define npyv_cvt_s16_b16 npyv_cvt_u16_b16

#ifdef NPY_HAVE_AVX512DQ
    #define npyv_cvt_u32_b32 _mm512_movm_epi32
    #define npyv_cvt_u64_b64 _mm512_movm_epi64
#else
    #define npyv_cvt_u32_b32(BL) _mm512_maskz_set1_epi32(BL, (int)-1)
    #define npyv_cvt_u64_b64(BL) _mm512_maskz_set1_epi64(BL, (npy_int64)-1)
#endif
#define npyv_cvt_s32_b32 npyv_cvt_u32_b32
#define npyv_cvt_s64_b64 npyv_cvt_u64_b64
#define npyv_cvt_f32_b32(BL) _mm512_castsi512_ps(npyv_cvt_u32_b32(BL))
#define npyv_cvt_f64_b64(BL) _mm512_castsi512_pd(npyv_cvt_u64_b64(BL))

// convert integer vectors to mask
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cvt_b8_u8 _mm512_movepi8_mask
    #define npyv_cvt_b16_u16 _mm512_movepi16_mask
#else
    #define npyv_cvt_b8_u8(A)  A
    #define npyv_cvt_b16_u16(A) A
#endif
#define npyv_cvt_b8_s8  npyv_cvt_b8_u8
#define npyv_cvt_b16_s16 npyv_cvt_b16_u16

#ifdef NPY_HAVE_AVX512DQ
    #define npyv_cvt_b32_u32 _mm512_movepi32_mask
    #define npyv_cvt_b64_u64 _mm512_movepi64_mask
#else
    #define npyv_cvt_b32_u32(A) _mm512_cmpneq_epu32_mask(A, _mm512_setzero_si512())
    #define npyv_cvt_b64_u64(A) _mm512_cmpneq_epu64_mask(A, _mm512_setzero_si512())
#endif
#define npyv_cvt_b32_s32 npyv_cvt_b32_u32
#define npyv_cvt_b64_s64 npyv_cvt_b64_u64
#define npyv_cvt_b32_f32(A) npyv_cvt_b32_u32(_mm512_castps_si512(A))
#define npyv_cvt_b64_f64(A) npyv_cvt_b64_u64(_mm512_castpd_si512(A))

// expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;
    __m256i lo = npyv512_lower_si256(data);
    __m256i hi = npyv512_higher_si256(data);
#ifdef NPY_HAVE_AVX512BW
    r.val[0] = _mm512_cvtepu8_epi16(lo);
    r.val[1] = _mm512_cvtepu8_epi16(hi);
#else
    __m256i loelo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo));
    __m256i loehi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1));
    __m256i hielo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi));
    __m256i hiehi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1));
    r.val[0] = npyv512_combine_si256(loelo, loehi);
    r.val[1] = npyv512_combine_si256(hielo, hiehi);
#endif
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
    __m256i lo = npyv512_lower_si256(data);
    __m256i hi = npyv512_higher_si256(data);
#ifdef NPY_HAVE_AVX512BW
    r.val[0] = _mm512_cvtepu16_epi32(lo);
    r.val[1] = _mm512_cvtepu16_epi32(hi);
#else
    __m256i loelo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo));
    __m256i loehi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo, 1));
    __m256i hielo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi));
    __m256i hiehi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi, 1));
    r.val[0] = npyv512_combine_si256(loelo, loehi);
    r.val[1] = npyv512_combine_si256(hielo, hiehi);
#endif
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
#ifdef NPY_HAVE_AVX512BW
    return _mm512_kunpackd((__mmask64)b, (__mmask64)a);
#else
    const __m512i idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    return _mm512_permutexvar_epi64(idx, npyv512_packs_epi16(a, b));
#endif
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
#ifdef NPY_HAVE_AVX512BW
    __mmask32 ab = _mm512_kunpackw((__mmask32)b, (__mmask32)a);
    __mmask32 cd = _mm512_kunpackw((__mmask32)d, (__mmask32)c);
    return npyv_pack_b8_b16(ab, cd);
#else
    const __m512i idx = _mm512_setr_epi32(
        0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    __m256i ta = npyv512_pack_lo_hi(npyv_cvt_u32_b32(a));
    __m256i tb = npyv512_pack_lo_hi(npyv_cvt_u32_b32(b));
    __m256i tc = npyv512_pack_lo_hi(npyv_cvt_u32_b32(c));
    __m256i td = npyv512_pack_lo_hi(npyv_cvt_u32_b32(d));
    __m256i ab = _mm256_packs_epi16(ta, tb);
    __m256i cd = _mm256_packs_epi16(tc, td);
    __m512i abcd = npyv512_combine_si256(ab, cd);
    return _mm512_permutexvar_epi32(idx, abcd);
#endif
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __mmask16 ab = _mm512_kunpackb((__mmask16)b, (__mmask16)a);
    __mmask16 cd = _mm512_kunpackb((__mmask16)d, (__mmask16)c);
    __mmask16 ef = _mm512_kunpackb((__mmask16)f, (__mmask16)e);
    __mmask16 gh = _mm512_kunpackb((__mmask16)h, (__mmask16)g);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// convert boolean vectors to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
#ifdef NPY_HAVE_AVX512BW_MASK
    return (npy_uint64)_cvtmask64_u64(a);
#elif defined(NPY_HAVE_AVX512BW)
    return (npy_uint64)a;
#else
    int mask_lo = _mm256_movemask_epi8(npyv512_lower_si256(a));
    int mask_hi = _mm256_movemask_epi8(npyv512_higher_si256(a));
    return (unsigned)mask_lo | ((npy_uint64)(unsigned)mask_hi << 32);
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
#ifdef NPY_HAVE_AVX512BW_MASK
    return (npy_uint32)_cvtmask32_u32(a);
#elif defined(NPY_HAVE_AVX512BW)
    return (npy_uint32)a;
#else
    __m256i pack = _mm256_packs_epi16(
        npyv512_lower_si256(a), npyv512_higher_si256(a)
    );
    return (npy_uint32)_mm256_movemask_epi8(_mm256_permute4x64_epi64(pack, _MM_SHUFFLE(3, 1, 2, 0)));
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return (npy_uint16)a; }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
#ifdef NPY_HAVE_AVX512DQ_MASK
    return _cvtmask8_u32(a);
#else
    return (npy_uint8)a;
#endif
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 _mm512_cvtps_epi32
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    __m256i lo = _mm512_cvtpd_epi32(a), hi = _mm512_cvtpd_epi32(b);
    return npyv512_combine_si256(lo, hi);
}

#endif // _NPY_SIMD_AVX512_CVT_H
