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

// expand
NPY_FINLINE void npyv_expand_u8_u16(npyv_u8 data, npyv_u16 *low, npyv_u16 *high) {
    const __m128i z = _mm_setzero_si128();
    *low = _mm_unpacklo_epi8(data, z);
    *high = _mm_unpackhi_epi8(data, z);
}

NPY_FINLINE void npyv_expand_u16_u32(npyv_u16 data, npyv_u32 *low, npyv_u32 *high) {
    const __m128i z = _mm_setzero_si128();
    *low = _mm_unpacklo_epi16(data, z);
    *high = _mm_unpackhi_epi16(data, z);
}

#endif // _NPY_SIMD_SSE_CVT_H
