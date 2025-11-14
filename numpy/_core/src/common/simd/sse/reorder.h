#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_REORDER_H
#define _NPY_SIMD_SSE_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8  _mm_unpacklo_epi64
#define npyv_combinel_s8  _mm_unpacklo_epi64
#define npyv_combinel_u16 _mm_unpacklo_epi64
#define npyv_combinel_s16 _mm_unpacklo_epi64
#define npyv_combinel_u32 _mm_unpacklo_epi64
#define npyv_combinel_s32 _mm_unpacklo_epi64
#define npyv_combinel_u64 _mm_unpacklo_epi64
#define npyv_combinel_s64 _mm_unpacklo_epi64
#define npyv_combinel_f32(A, B) _mm_castsi128_ps(_mm_unpacklo_epi64(_mm_castps_si128(A), _mm_castps_si128(B)))
#define npyv_combinel_f64 _mm_unpacklo_pd

// combine higher part of two vectors
#define npyv_combineh_u8  _mm_unpackhi_epi64
#define npyv_combineh_s8  _mm_unpackhi_epi64
#define npyv_combineh_u16 _mm_unpackhi_epi64
#define npyv_combineh_s16 _mm_unpackhi_epi64
#define npyv_combineh_u32 _mm_unpackhi_epi64
#define npyv_combineh_s32 _mm_unpackhi_epi64
#define npyv_combineh_u64 _mm_unpackhi_epi64
#define npyv_combineh_s64 _mm_unpackhi_epi64
#define npyv_combineh_f32(A, B) _mm_castsi128_ps(_mm_unpackhi_epi64(_mm_castps_si128(A), _mm_castps_si128(B)))
#define npyv_combineh_f64 _mm_unpackhi_pd

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_m128ix2 npyv__combine(__m128i a, __m128i b)
{
    npyv_m128ix2 r;
    r.val[0] = npyv_combinel_u8(a, b);
    r.val[1] = npyv_combineh_u8(a, b);
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m128 a, __m128 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b);
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m128d a, __m128d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b);
    r.val[1] = npyv_combineh_f64(a, b);
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
#define NPYV_IMPL_SSE_ZIP(T_VEC, SFX, INTR_SFX)            \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b) \
    {                                                      \
        T_VEC##x2 r;                                       \
        r.val[0] = _mm_unpacklo_##INTR_SFX(a, b);          \
        r.val[1] = _mm_unpackhi_##INTR_SFX(a, b);          \
        return r;                                          \
    }

NPYV_IMPL_SSE_ZIP(npyv_u8,  u8,  epi8)
NPYV_IMPL_SSE_ZIP(npyv_s8,  s8,  epi8)
NPYV_IMPL_SSE_ZIP(npyv_u16, u16, epi16)
NPYV_IMPL_SSE_ZIP(npyv_s16, s16, epi16)
NPYV_IMPL_SSE_ZIP(npyv_u32, u32, epi32)
NPYV_IMPL_SSE_ZIP(npyv_s32, s32, epi32)
NPYV_IMPL_SSE_ZIP(npyv_u64, u64, epi64)
NPYV_IMPL_SSE_ZIP(npyv_s64, s64, epi64)
NPYV_IMPL_SSE_ZIP(npyv_f32, f32, ps)
NPYV_IMPL_SSE_ZIP(npyv_f64, f64, pd)

// deinterleave two vectors
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
#ifdef NPY_HAVE_SSSE3
    const __m128i idx = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    );
    __m128i abl = _mm_shuffle_epi8(ab0, idx);
    __m128i abh = _mm_shuffle_epi8(ab1, idx);
    return npyv_combine_u8(abl, abh);
#else
    __m128i ab_083b = _mm_unpacklo_epi8(ab0, ab1);
    __m128i ab_4c6e = _mm_unpackhi_epi8(ab0, ab1);
    __m128i ab_048c = _mm_unpacklo_epi8(ab_083b, ab_4c6e);
    __m128i ab_36be = _mm_unpackhi_epi8(ab_083b, ab_4c6e);
    __m128i ab_0346 = _mm_unpacklo_epi8(ab_048c, ab_36be);
    __m128i ab_8bc8 = _mm_unpackhi_epi8(ab_048c, ab_36be);
    npyv_u8x2 r;
    r.val[0] = _mm_unpacklo_epi8(ab_0346, ab_8bc8);
    r.val[1] = _mm_unpackhi_epi8(ab_0346, ab_8bc8);
    return r;
#endif
}
#define npyv_unzip_s8 npyv_unzip_u8

NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
#ifdef NPY_HAVE_SSSE3
    const __m128i idx = _mm_setr_epi8(
        0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15
    );
    __m128i abl = _mm_shuffle_epi8(ab0, idx);
    __m128i abh = _mm_shuffle_epi8(ab1, idx);
    return npyv_combine_u16(abl, abh);
#else
    __m128i ab_0415 = _mm_unpacklo_epi16(ab0, ab1);
    __m128i ab_263f = _mm_unpackhi_epi16(ab0, ab1);
    __m128i ab_0246 = _mm_unpacklo_epi16(ab_0415, ab_263f);
    __m128i ab_135f = _mm_unpackhi_epi16(ab_0415, ab_263f);
    npyv_u16x2 r;
    r.val[0] = _mm_unpacklo_epi16(ab_0246, ab_135f);
    r.val[1] = _mm_unpackhi_epi16(ab_0246, ab_135f);
    return r;
#endif
}
#define npyv_unzip_s16 npyv_unzip_u16

NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    __m128i abl = _mm_shuffle_epi32(ab0, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i abh = _mm_shuffle_epi32(ab1, _MM_SHUFFLE(3, 1, 2, 0));
    return npyv_combine_u32(abl, abh);
}
#define npyv_unzip_s32 npyv_unzip_u32

NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{ return npyv_combine_u64(ab0, ab1); }
#define npyv_unzip_s64 npyv_unzip_u64

NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
{
    npyv_f32x2 r;
    r.val[0] = _mm_shuffle_ps(ab0, ab1, _MM_SHUFFLE(2, 0, 2, 0));
    r.val[1] = _mm_shuffle_ps(ab0, ab1, _MM_SHUFFLE(3, 1, 3, 1));
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_unzip_f64(npyv_f64 ab0, npyv_f64 ab1)
{ return npyv_combine_f64(ab0, ab1); }

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
#ifdef NPY_HAVE_SSSE3
    const __m128i idx = _mm_setr_epi8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    return _mm_shuffle_epi8(a, idx);
#else
    __m128i lo = _mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm_shufflehi_epi16(lo, _MM_SHUFFLE(0, 1, 2, 3));
#endif
}
#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
#ifdef NPY_HAVE_SSSE3
    const __m128i idx = _mm_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    return _mm_shuffle_epi8(a, idx);
#else
    __m128i rev16 = npyv_rev64_u16(a);
    // swap 8bit pairs
    return _mm_or_si128(_mm_slli_epi16(rev16, 8), _mm_srli_epi16(rev16, 8));
#endif
}
#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    return _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    return _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3) \
    _mm_shuffle_epi32(A, _MM_SHUFFLE(E3, E2, E1, E0))

#define npyv_permi128_s32 npyv_permi128_u32

#define npyv_permi128_u64(A, E0, E1) \
    _mm_shuffle_epi32(A, _MM_SHUFFLE(((E1)<<1)+1, ((E1)<<1), ((E0)<<1)+1, ((E0)<<1)))

#define npyv_permi128_s64 npyv_permi128_u64

#define npyv_permi128_f32(A, E0, E1, E2, E3) \
    _mm_shuffle_ps(A, A, _MM_SHUFFLE(E3, E2, E1, E0))

#define npyv_permi128_f64(A, E0, E1) \
    _mm_shuffle_pd(A, A, _MM_SHUFFLE2(E1, E0))

#endif // _NPY_SIMD_SSE_REORDER_H
