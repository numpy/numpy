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

/***************************
 * Packing
 ***************************/
// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b16(npyv_b16 a, npyv_b16 b)
{
#ifdef NPY_HAVE_AVX512BW
    return _mm512_kunpackd((__mmask64)b, (__mmask64)a);
#else
    return npyv512_shuffle_odd(npyv512_packs_epi16(a, b));
#endif
}
// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
#ifdef NPY_HAVE_AVX512BW
    __mmask32 ab = (__mmask64)_mm512_kunpackw((__mmask32)b, (__mmask32)a);
    __mmask32 cd = (__mmask64)_mm512_kunpackw((__mmask32)d, (__mmask32)c);
#else
    __m512i ai = npyv_cvt_u32_b32(a);
    __m512i bi = npyv_cvt_u32_b32(b);
    __m512i ci = npyv_cvt_u32_b32(c);
    __m512i di = npyv_cvt_u32_b32(d);
    __m512i ab = npyv512_packs_epi32(ai, bi);
    __m512i cd = npyv512_packs_epi32(ci, di);
#endif
    return npyv_pack_b16(ab, cd);
}
// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                                     npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    __mmask16 ab = _mm512_kunpackb((__mmask16)b, (__mmask16)a);
    __mmask16 cd = _mm512_kunpackb((__mmask16)d, (__mmask16)c);
    __mmask16 ef = _mm512_kunpackb((__mmask16)f, (__mmask16)e);
    __mmask16 gh = _mm512_kunpackb((__mmask16)h, (__mmask16)g);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}
#endif // _NPY_SIMD_AVX512_CVT_H
