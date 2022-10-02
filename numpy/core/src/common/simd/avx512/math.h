#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MATH_H
#define _NPY_SIMD_AVX512_MATH_H

/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm512_sqrt_ps
#define npyv_sqrt_f64 _mm512_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm512_div_ps(_mm512_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm512_div_pd(_mm512_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_ps(a, a, 8);
#else
    return npyv_and_f32(
        a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff))
    );
#endif
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
#if 0 // def NPY_HAVE_AVX512DQ
    return _mm512_range_pd(a, a, 8);
#else
    return npyv_and_f64(
        a, _mm512_castsi512_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
#endif
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm512_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm512_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm512_max_ps
#define npyv_max_f64 _mm512_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_max_pd(a, nn, a, b);
}
// Maximum, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_max_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_max_pd(a, nn, a, b);
}
// Maximum, integer operations
#ifdef NPY_HAVE_AVX512BW
    #define npyv_max_u8 _mm512_max_epu8
    #define npyv_max_s8 _mm512_max_epi8
    #define npyv_max_u16 _mm512_max_epu16
    #define npyv_max_s16 _mm512_max_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u8, _mm256_max_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s8, _mm256_max_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_u16, _mm256_max_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_max_s16, _mm256_max_epi16)
#endif
#define npyv_max_u32 _mm512_max_epu32
#define npyv_max_s32 _mm512_max_epi32
#define npyv_max_u64 _mm512_max_epu64
#define npyv_max_s64 _mm512_max_epi64

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm512_min_ps
#define npyv_min_f64 _mm512_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_min_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(b, b, _CMP_ORD_Q);
    return _mm512_mask_min_pd(a, nn, a, b);
}
// Minimum, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __mmask16 nn = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_min_ps(a, nn, a, b);
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __mmask8 nn = _mm512_cmp_pd_mask(a, a, _CMP_ORD_Q);
    return _mm512_mask_min_pd(a, nn, a, b);
}
// Minimum, integer operations
#ifdef NPY_HAVE_AVX512BW
    #define npyv_min_u8 _mm512_min_epu8
    #define npyv_min_s8 _mm512_min_epi8
    #define npyv_min_u16 _mm512_min_epu16
    #define npyv_min_s16 _mm512_min_epi16
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u8, _mm256_min_epu8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s8, _mm256_min_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_u16, _mm256_min_epu16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_min_s16, _mm256_min_epi16)
#endif
#define npyv_min_u32 _mm512_min_epu32
#define npyv_min_s32 _mm512_min_epi32
#define npyv_min_u64 _mm512_min_epu64
#define npyv_min_s64 _mm512_min_epi64

#ifdef NPY_HAVE_AVX512F_REDUCE
    #define npyv_reduce_min_u32 _mm512_reduce_min_epu32
    #define npyv_reduce_min_s32 _mm512_reduce_min_epi32
    #define npyv_reduce_min_u64 _mm512_reduce_min_epu64
    #define npyv_reduce_min_s64 _mm512_reduce_min_epi64
    #define npyv_reduce_min_f32 _mm512_reduce_min_ps
    #define npyv_reduce_min_f64 _mm512_reduce_min_pd
    #define npyv_reduce_max_u32 _mm512_reduce_max_epu32
    #define npyv_reduce_max_s32 _mm512_reduce_max_epi32
    #define npyv_reduce_max_u64 _mm512_reduce_max_epu64
    #define npyv_reduce_max_s64 _mm512_reduce_max_epi64
    #define npyv_reduce_max_f32 _mm512_reduce_max_ps
    #define npyv_reduce_max_f64 _mm512_reduce_max_pd
#else
    // reduce min&max for 32&64-bits
    #define NPY_IMPL_AVX512_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                              \
        NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m512i a)                              \
        {                                                                                      \
            __m256i v256 = _mm256_##VINTRIN##32(npyv512_lower_si256(a),                        \
                    npyv512_higher_si256(a));                                                  \
            __m128i v128 = _mm_##VINTRIN##32(_mm256_castsi256_si128(v256),                     \
                    _mm256_extracti128_si256(v256, 1));                                        \
            __m128i v64 =  _mm_##VINTRIN##32(v128, _mm_shuffle_epi32(v128,                     \
                        (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                              \
            __m128i v32 = _mm_##VINTRIN##32(v64, _mm_shuffle_epi32(v64,                        \
                        (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                              \
            return (STYPE##32)_mm_cvtsi128_si32(v32);                                          \
        }                                                                                      \
        NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m512i a)                              \
        {                                                                                      \
            __m512i v256 = _mm512_##VINTRIN##64(a,                                             \
                    _mm512_shuffle_i64x2(a, a, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));       \
            __m512i v128 = _mm512_##VINTRIN##64(v256,                                          \
                    _mm512_shuffle_i64x2(v256, v256, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1))); \
            __m512i v64  = _mm512_##VINTRIN##64(v128,                                          \
                    _mm512_shuffle_epi32(v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));       \
            return (STYPE##64)npyv_extract0_u64(v64);                                          \
        }

    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, min_u, min_epu)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  min_s, min_epi)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, max_u, max_epu)
    NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  max_s, max_epi)
    #undef NPY_IMPL_AVX512_REDUCE_MINMAX
    // reduce min&max for ps & pd
    #define NPY_IMPL_AVX512_REDUCE_MINMAX(INTRIN)                                         \
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                          \
        {                                                                                 \
            __m256 v256 = _mm256_##INTRIN##_ps(                                           \
                    npyv512_lower_ps256(a), npyv512_higher_ps256(a));                     \
            __m128 v128 = _mm_##INTRIN##_ps(                                              \
                    _mm256_castps256_ps128(v256), _mm256_extractf128_ps(v256, 1));        \
            __m128 v64 =  _mm_##INTRIN##_ps(v128,                                         \
                    _mm_shuffle_ps(v128, v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));  \
            __m128 v32 = _mm_##INTRIN##_ps(v64,                                           \
                    _mm_shuffle_ps(v64, v64, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));    \
            return _mm_cvtss_f32(v32);                                                    \
        }                                                                                 \
        NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                         \
        {                                                                                 \
            __m256d v256 = _mm256_##INTRIN##_pd(                                          \
                    npyv512_lower_pd256(a), npyv512_higher_pd256(a));                     \
            __m128d v128 = _mm_##INTRIN##_pd(                                             \
                    _mm256_castpd256_pd128(v256), _mm256_extractf128_pd(v256, 1));        \
            __m128d v64 =  _mm_##INTRIN##_pd(v128,                                        \
                    _mm_shuffle_pd(v128, v128, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));  \
            return _mm_cvtsd_f64(v64);                                                    \
        }

    NPY_IMPL_AVX512_REDUCE_MINMAX(min)
    NPY_IMPL_AVX512_REDUCE_MINMAX(max)
    #undef NPY_IMPL_AVX512_REDUCE_MINMAX
#endif
#define NPY_IMPL_AVX512_REDUCE_MINMAX(INTRIN, INF, INF64)           \
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)       \
    {                                                               \
        npyv_b32 notnan = npyv_notnan_f32(a);                       \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                  \
            return _mm_cvtss_f32(_mm512_castps512_ps128(a));        \
        }                                                           \
        a = npyv_select_f32(notnan, a,                              \
                npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));    \
        return npyv_reduce_##INTRIN##_f32(a);                       \
    }                                                               \
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)      \
    {                                                               \
        npyv_b64 notnan = npyv_notnan_f64(a);                       \
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {                  \
            return _mm_cvtsd_f64(_mm512_castpd512_pd128(a));        \
        }                                                           \
        a = npyv_select_f64(notnan, a,                              \
                npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));  \
        return npyv_reduce_##INTRIN##_f64(a);                       \
    }                                                               \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)       \
    {                                                               \
        npyv_b32 notnan = npyv_notnan_f32(a);                       \
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                  \
            const union { npy_uint32 i; float f;} pnan = {          \
                0x7fc00000ul                                        \
            };                                                      \
            return pnan.f;                                          \
        }                                                           \
        return npyv_reduce_##INTRIN##_f32(a);                       \
    }                                                               \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)      \
    {                                                               \
        npyv_b64 notnan = npyv_notnan_f64(a);                       \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                  \
            const union { npy_uint64 i; double d;} pnan = {         \
                0x7ff8000000000000ull                               \
            };                                                      \
            return pnan.d;                                          \
        }                                                           \
        return npyv_reduce_##INTRIN##_f64(a);                       \
    }

NPY_IMPL_AVX512_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_AVX512_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
#undef NPY_IMPL_AVX512_REDUCE_MINMAX

// reduce min&max for 8&16-bits
#define NPY_IMPL_AVX512_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                               \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m512i a)                                               \
    {                                                                                                       \
        __m256i v256 = _mm256_##VINTRIN##16(npyv512_lower_si256(a), npyv512_higher_si256(a));               \
        __m128i v128 = _mm_##VINTRIN##16(_mm256_castsi256_si128(v256), _mm256_extracti128_si256(v256, 1));  \
        __m128i v64 =  _mm_##VINTRIN##16(v128, _mm_shuffle_epi32(v128,                                      \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                                                \
        __m128i v32 = _mm_##VINTRIN##16(v64, _mm_shuffle_epi32(v64,                                         \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                                \
        __m128i v16 = _mm_##VINTRIN##16(v32, _mm_shufflelo_epi16(v32,                                       \
                   (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                                \
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                           \
    }                                                                                                       \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m512i a)                                                 \
    {                                                                                                       \
        __m256i v256 = _mm256_##VINTRIN##8(npyv512_lower_si256(a), npyv512_higher_si256(a));                \
        __m128i v128 = _mm_##VINTRIN##8(_mm256_castsi256_si128(v256), _mm256_extracti128_si256(v256, 1));   \
        __m128i v64 =  _mm_##VINTRIN##8(v128, _mm_shuffle_epi32(v128,                                       \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 3, 2)));                                               \
        __m128i v32 = _mm_##VINTRIN##8(v64, _mm_shuffle_epi32(v64,                                          \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                               \
        __m128i v16 = _mm_##VINTRIN##8(v32, _mm_shufflelo_epi16(v32,                                        \
                    (_MM_PERM_ENUM)_MM_SHUFFLE(0, 0, 0, 1)));                                               \
        __m128i v8 = _mm_##VINTRIN##8(v16, _mm_srli_epi16(v16, 8));                                         \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                            \
    }
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_AVX512_REDUCE_MINMAX(npy_int,  max_s, max_epi)
#undef NPY_IMPL_AVX512_REDUCE_MINMAX

// round to nearest integer even
#define npyv_rint_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_NEAREST_INT)
#define npyv_rint_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_NEAREST_INT)

// ceil
#define npyv_ceil_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_POS_INF)
#define npyv_ceil_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_POS_INF)

// trunc
#define npyv_trunc_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_ZERO)
#define npyv_trunc_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_ZERO)

// floor
#define npyv_floor_f32(A) _mm512_roundscale_ps(A, _MM_FROUND_TO_NEG_INF)
#define npyv_floor_f64(A) _mm512_roundscale_pd(A, _MM_FROUND_TO_NEG_INF)

#endif // _NPY_SIMD_AVX512_MATH_H
