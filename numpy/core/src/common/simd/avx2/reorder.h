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

// deinterleave two vectors
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
    const __m256i idx = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    );
    __m256i ab_03 = _mm256_shuffle_epi8(ab0, idx);
    __m256i ab_12 = _mm256_shuffle_epi8(ab1, idx);
    npyv_u8x2 ab_lh = npyv_combine_u8(ab_03, ab_12);
    npyv_u8x2 r;
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
#define npyv_unzip_s8 npyv_unzip_u8

NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
    const __m256i idx = _mm256_setr_epi8(
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15,
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15
    );
    __m256i ab_03 = _mm256_shuffle_epi8(ab0, idx);
    __m256i ab_12 = _mm256_shuffle_epi8(ab1, idx);
    npyv_u16x2 ab_lh = npyv_combine_u16(ab_03, ab_12);
    npyv_u16x2 r;
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
#define npyv_unzip_s16 npyv_unzip_u16

NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    const __m256i idx = npyv_set_u32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256i abl = _mm256_permutevar8x32_epi32(ab0, idx);
    __m256i abh = _mm256_permutevar8x32_epi32(ab1, idx);
    return npyv_combine_u32(abl, abh);
}
#define npyv_unzip_s32 npyv_unzip_u32

NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{
    npyv_u64x2 ab_lh = npyv_combine_u64(ab0, ab1);
    npyv_u64x2 r;
    r.val[0] = _mm256_unpacklo_epi64(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_epi64(ab_lh.val[0], ab_lh.val[1]);
    return r;
}
#define npyv_unzip_s64 npyv_unzip_u64

NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
{
    const __m256i idx = npyv_set_u32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256 abl = _mm256_permutevar8x32_ps(ab0, idx);
    __m256 abh = _mm256_permutevar8x32_ps(ab1, idx);
    return npyv_combine_f32(abl, abh);
}

NPY_FINLINE npyv_f64x2 npyv_unzip_f64(npyv_f64 ab0, npyv_f64 ab1)
{
    npyv_f64x2 ab_lh = npyv_combine_f64(ab0, ab1);
    npyv_f64x2 r;
    r.val[0] = _mm256_unpacklo_pd(ab_lh.val[0], ab_lh.val[1]);
    r.val[1] = _mm256_unpackhi_pd(ab_lh.val[0], ab_lh.val[1]);
    return r;
}

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    const __m256i idx = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    return _mm256_shuffle_epi8(a, idx);
}
#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    const __m256i idx = _mm256_setr_epi8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9,
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    return _mm256_shuffle_epi8(a, idx);
}
#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    return _mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    return _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3) \
    _mm256_shuffle_epi32(A, _MM_SHUFFLE(E3, E2, E1, E0))

#define npyv_permi128_s32 npyv_permi128_u32

#define npyv_permi128_u64(A, E0, E1) \
    _mm256_shuffle_epi32(A, _MM_SHUFFLE(((E1)<<1)+1, ((E1)<<1), ((E0)<<1)+1, ((E0)<<1)))

#define npyv_permi128_s64 npyv_permi128_u64

#define npyv_permi128_f32(A, E0, E1, E2, E3) \
    _mm256_permute_ps(A, _MM_SHUFFLE(E3, E2, E1, E0))

#define npyv_permi128_f64(A, E0, E1) \
    _mm256_permute_pd(A, ((E1)<<3) | ((E0)<<2) | ((E1)<<1) | (E0))

#endif // _NPY_SIMD_AVX2_REORDER_H
