#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_CVT_H
#define _NPY_SIMD_AVX2_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_s8_b8(A)   A
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_s16_b16(A) A
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_s32_b32(A) A
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s64_b64(A) A
#define npyv_cvt_f32_b32(A) _mm256_castsi256_ps(A)
#define npyv_cvt_f64_b64(A) _mm256_castsi256_pd(A)

// convert integer types to mask types
#define npyv_cvt_b8_u8(BL)   BL
#define npyv_cvt_b8_s8(BL)   BL
#define npyv_cvt_b16_u16(BL) BL
#define npyv_cvt_b16_s16(BL) BL
#define npyv_cvt_b32_u32(BL) BL
#define npyv_cvt_b32_s32(BL) BL
#define npyv_cvt_b64_u64(BL) BL
#define npyv_cvt_b64_s64(BL) BL
#define npyv_cvt_b32_f32(BL) _mm256_castps_si256(BL)
#define npyv_cvt_b64_f64(BL) _mm256_castpd_si256(BL)

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b16(npyv_b16 a, npyv_b16 b)
{
    __m256i ab = _mm256_packs_epi16(a, b);
    return npyv256_shuffle_odd(ab);
}
// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i ab = _mm256_packs_epi32(a, b);
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i abcd = _mm256_packs_epi16(ab, cd);
    return _mm256_permutevar8x32_epi32(abcd, perm);
}
// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b16 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                                      npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    __m256i ab = _mm256_packs_epi32(a, b);
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i ef = _mm256_packs_epi32(e, f);
    __m256i gh = _mm256_packs_epi32(g, h);
    __m256i abcd = _mm256_packs_epi32(ab, cd);
    __m256i efgh = _mm256_packs_epi32(ef, gh);
    __m256i all  = npyv256_shuffle_odd(_mm256_packs_epi16(abcd, efgh));
    __m256i rev128 = _mm256_alignr_epi8(all, all, 8);
    return _mm256_unpacklo_epi16(all, rev128);
}
#endif // _NPY_SIMD_AVX2_CVT_H
