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
#define npyv_cvt_f32_b32 _mm_castsi128_ps
#define npyv_cvt_f64_b64 _mm_castsi128_pd

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32 _mm_castps_si128
#define npyv_cvt_b64_f64 _mm_castpd_si128

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{ return (npy_uint16)_mm_movemask_epi8(a); }
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    __m128i pack = _mm_packs_epi16(a, a);
    return (npy_uint8)_mm_movemask_epi8(pack);
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return (npy_uint8)_mm_movemask_ps(_mm_castsi128_ps(a)); }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{ return (npy_uint8)_mm_movemask_pd(_mm_castsi128_pd(a)); }

// expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    const __m128i z = _mm_setzero_si128();
    r.val[0] = _mm_unpacklo_epi8(data, z);
    r.val[1] = _mm_unpackhi_epi8(data, z);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    const __m128i z = _mm_setzero_si128();
    r.val[0]  = _mm_unpacklo_epi16(data, z);
    r.val[1]  = _mm_unpackhi_epi16(data, z);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return _mm_packs_epi16(a, b);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    npyv_b16 ab = _mm_packs_epi32(a, b);
    npyv_b16 cd = _mm_packs_epi32(c, d);
    return npyv_pack_b8_b16(ab, cd);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    npyv_b32 ab = _mm_packs_epi32(a, b);
    npyv_b32 cd = _mm_packs_epi32(c, d);
    npyv_b32 ef = _mm_packs_epi32(e, f);
    npyv_b32 gh = _mm_packs_epi32(g, h);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 _mm_cvtps_epi32
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i lo = _mm_cvtpd_epi32(a), hi = _mm_cvtpd_epi32(b);
    return _mm_unpacklo_epi64(lo, hi);
}

#endif // _NPY_SIMD_SSE_CVT_H
