#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_REORDER_H
#define _NPY_SIMD_AVX2_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8(A, B) _mm256_permute2x128_si256(A, B, 0x20)
#define npyv_combinel_s8  npyv_combinel_u8
#define npyv_combinel_u16 npyv_combinel_u8
#define npyv_combinel_s16 npyv_combinel_u8
#define npyv_combinel_u32 npyv_combinel_u8
#define npyv_combinel_s32 npyv_combinel_u8
#define npyv_combinel_u64 npyv_combinel_u8
#define npyv_combinel_s64 npyv_combinel_u8
#define npyv_combinel_f32(A, B) _mm256_permute2f128_ps(A, B, 0x20)
#define npyv_combinel_f64(A, B) _mm256_permute2f128_pd(A, B, 0x20)

// combine higher part of two vectors
#define npyv_combineh_u8(A, B) _mm256_permute2x128_si256(A, B, 0x31)
#define npyv_combineh_s8  npyv_combineh_u8
#define npyv_combineh_u16 npyv_combineh_u8
#define npyv_combineh_s16 npyv_combineh_u8
#define npyv_combineh_u32 npyv_combineh_u8
#define npyv_combineh_s32 npyv_combineh_u8
#define npyv_combineh_u64 npyv_combineh_u8
#define npyv_combineh_s64 npyv_combineh_u8
#define npyv_combineh_f32(A, B) _mm256_permute2f128_ps(A, B, 0x31)
#define npyv_combineh_f64(A, B) _mm256_permute2f128_pd(A, B, 0x31)

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_m256ix2 npyv__combine(__m256i a, __m256i b)
{
    npyv_m256ix2 r;
    __m256i a1b0 = _mm256_permute2x128_si256(a, b, 0x21);
    r.val[0] = _mm256_blend_epi32(a, a1b0, 0xF0);
    r.val[1] = _mm256_blend_epi32(b, a1b0, 0xF);
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    __m256 a1b0 = _mm256_permute2f128_ps(a, b, 0x21);
    r.val[0] = _mm256_blend_ps(a, a1b0, 0xF0);
    r.val[1] = _mm256_blend_ps(b, a1b0, 0xF);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    __m256d a1b0 = _mm256_permute2f128_pd(a, b, 0x21);
    r.val[0] = _mm256_blend_pd(a, a1b0, 0xC);
    r.val[1] = _mm256_blend_pd(b, a1b0, 0x3);
    return r;
}
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

// interleave two vectors
#define NPYV_IMPL_AVX2_ZIP_U(T_VEC, LEN)                    \
    NPY_FINLINE T_VEC##x2 npyv_zip_u##LEN(T_VEC a, T_VEC b) \
    {                                                       \
        __m256i ab0 = _mm256_unpacklo_epi##LEN(a, b);       \
        __m256i ab1 = _mm256_unpackhi_epi##LEN(a, b);       \
        return npyv__combine(ab0, ab1);                     \
    }

NPYV_IMPL_AVX2_ZIP_U(npyv_u8,  8)
NPYV_IMPL_AVX2_ZIP_U(npyv_u16, 16)
NPYV_IMPL_AVX2_ZIP_U(npyv_u32, 32)
NPYV_IMPL_AVX2_ZIP_U(npyv_u64, 64)
#define npyv_zip_s8  npyv_zip_u8
#define npyv_zip_s16 npyv_zip_u16
#define npyv_zip_s32 npyv_zip_u32
#define npyv_zip_s64 npyv_zip_u64

NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m256 a, __m256 b)
{
    __m256 ab0 = _mm256_unpacklo_ps(a, b);
    __m256 ab1 = _mm256_unpackhi_ps(a, b);
    return npyv_combine_f32(ab0, ab1);
}
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m256d a, __m256d b)
{
    __m256d ab0 = _mm256_unpacklo_pd(a, b);
    __m256d ab1 = _mm256_unpackhi_pd(a, b);
    return npyv_combine_f64(ab0, ab1);
}

#endif // _NPY_SIMD_AVX2_REORDER_H
