#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_CVT_H
#define _NPY_SIMD_SSE_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(BL)   BL
#define npyv_cvt_s8_b8(BL)   BL
#define npyv_cvt_u16_b16(BL) BL
#define npyv_cvt_s16_b16(BL) BL
#define npyv_cvt_u32_b32(BL) BL
#define npyv_cvt_s32_b32(BL) BL
#define npyv_cvt_u64_b64(BL) BL
#define npyv_cvt_s64_b64(BL) BL
#define npyv_cvt_f32_b32(BL) _mm_castsi128_ps(BL)
#define npyv_cvt_f64_b64(BL) _mm_castsi128_pd(BL)

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32(A) _mm_castps_si128(A)
#define npyv_cvt_b64_f64(A) _mm_castpd_si128(A)

// pack two 16-bit boolean into one 8-bit boolean vector
#define npyv_pack_b16 _mm_packs_epi16
// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
    __m128i ab = _mm_packs_epi32(a, b);
    __m128i cd = _mm_packs_epi32(c, d);
    return _mm_packs_epi16(ab, cd);
}
// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                                     npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    __m128i ab = _mm_packs_epi32(a, b);
    __m128i cd = _mm_packs_epi32(c, d);
    __m128i ef = _mm_packs_epi32(e, f);
    __m128i gh = _mm_packs_epi32(g, h);
    __m128i abcd = _mm_packs_epi32(ab, cd);
    __m128i cdef = _mm_packs_epi32(ef, gh);
    return _mm_packs_epi16(abcd, cdef);
}

#endif // _NPY_SIMD_SSE_CVT_H
