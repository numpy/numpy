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
#define npyv_cvt_f32_b32 _mm256_castsi256_ps
#define npyv_cvt_f64_b64 _mm256_castsi256_pd

// convert integer types to mask types
#define npyv_cvt_b8_u8(BL)   BL
#define npyv_cvt_b8_s8(BL)   BL
#define npyv_cvt_b16_u16(BL) BL
#define npyv_cvt_b16_s16(BL) BL
#define npyv_cvt_b32_u32(BL) BL
#define npyv_cvt_b32_s32(BL) BL
#define npyv_cvt_b64_u64(BL) BL
#define npyv_cvt_b64_s64(BL) BL
#define npyv_cvt_b32_f32 _mm256_castps_si256
#define npyv_cvt_b64_f64 _mm256_castpd_si256

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{ return (npy_uint32)_mm256_movemask_epi8(a); }

NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    __m128i pack = _mm_packs_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    return (npy_uint16)_mm_movemask_epi8(pack);
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return (npy_uint8)_mm256_movemask_ps(_mm256_castsi256_ps(a)); }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{ return (npy_uint8)_mm256_movemask_pd(_mm256_castsi256_pd(a)); }

#endif // _NPY_SIMD_AVX2_CVT_H
