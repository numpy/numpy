#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_MATH_H
#define _NPY_SIMD_AVX2_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm256_sqrt_ps
#define npyv_sqrt_f64 _mm256_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm256_div_ps(_mm256_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm256_div_pd(_mm256_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm256_and_ps(
        a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm256_and_pd(
        a, _mm256_castsi256_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm256_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm256_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm256_max_ps
#define npyv_max_f64 _mm256_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    __m256 max = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(a, max, nn);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    __m256d max = _mm256_max_pd(a, b);
    return _mm256_blendv_pd(a, max, nn);
}
// Maximum, propagates NaNs
// If any of corresponded elements is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(a, a, _CMP_ORD_Q);
    __m256 max = _mm256_max_ps(a, b);
    return _mm256_blendv_ps(a, max, nn);
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(a, a, _CMP_ORD_Q);
    __m256d max = _mm256_max_pd(a, b);
    return _mm256_blendv_pd(a, max, nn);
}

// Maximum, integer operations
#define npyv_max_u8 _mm256_max_epu8
#define npyv_max_s8 _mm256_max_epi8
#define npyv_max_u16 _mm256_max_epu16
#define npyv_max_s16 _mm256_max_epi16
#define npyv_max_u32 _mm256_max_epu32
#define npyv_max_s32 _mm256_max_epi32
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return _mm256_blendv_epi8(b, a, npyv_cmpgt_u64(a, b));
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi64(a, b));
}

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm256_min_ps
#define npyv_min_f64 _mm256_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(b, b, _CMP_ORD_Q);
    __m256 min = _mm256_min_ps(a, b);
    return _mm256_blendv_ps(a, min, nn);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(b, b, _CMP_ORD_Q);
    __m256d min = _mm256_min_pd(a, b);
    return _mm256_blendv_pd(a, min, nn);
}
// Minimum, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __m256 nn  = _mm256_cmp_ps(a, a, _CMP_ORD_Q);
    __m256 min = _mm256_min_ps(a, b);
    return _mm256_blendv_ps(a, min, nn);
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __m256d nn  = _mm256_cmp_pd(a, a, _CMP_ORD_Q);
    __m256d min = _mm256_min_pd(a, b);
    return _mm256_blendv_pd(a, min, nn);
}
// Minimum, integer operations
#define npyv_min_u8 _mm256_min_epu8
#define npyv_min_s8 _mm256_min_epi8
#define npyv_min_u16 _mm256_min_epu16
#define npyv_min_s16 _mm256_min_epi16
#define npyv_min_u32 _mm256_min_epu32
#define npyv_min_s32 _mm256_min_epi32
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    return _mm256_blendv_epi8(b, a, npyv_cmplt_u64(a, b));
}
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
}
// reduce min&max for 32&64-bits
#define NPY_IMPL_AVX2_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                              \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m256i a)                                            \
    {                                                                                                    \
        __m128i v128 = _mm_##VINTRIN##32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));     \
        __m128i v64 =  _mm_##VINTRIN##32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));        \
        __m128i v32 = _mm_##VINTRIN##32(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));           \
        return (STYPE##32)_mm_cvtsi128_si32(v32);                                                        \
    }                                                                                                    \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m256i a)                                            \
    {                                                                                                    \
        __m256i v128 = npyv_##INTRIN##64(a, _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 0, 1)));   \
        __m256i v64  = npyv_##INTRIN##64(v128, _mm256_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));     \
        return (STYPE##64)npyv_extract0_u64(v64);                                                        \
    }
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_AVX2_REDUCE_MINMAX(npy_int,  max_s, max_epi)
#undef NPY_IMPL_AVX2_REDUCE_MINMAX

// reduce min&max for ps & pd
#define NPY_IMPL_AVX2_REDUCE_MINMAX(INTRIN, INF, INF64)                                              \
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                         \
    {                                                                                                \
        __m128 v128 = _mm_##INTRIN##_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));     \
        __m128 v64 =  _mm_##INTRIN##_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(0, 0, 3, 2)));  \
        __m128 v32 = _mm_##INTRIN##_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));      \
        return _mm_cvtss_f32(v32);                                                                   \
    }                                                                                                \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                        \
    {                                                                                                \
        __m128d v128 = _mm_##INTRIN##_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));    \
        __m128d v64 =  _mm_##INTRIN##_pd(v128, _mm_shuffle_pd(v128, v128, _MM_SHUFFLE(0, 0, 0, 1))); \
        return _mm_cvtsd_f64(v64);                                                                   \
    }                                                                                                \
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                        \
    {                                                                                                \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                        \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                                                   \
            return _mm_cvtss_f32(_mm256_castps256_ps128(a));                                         \
        }                                                                                            \
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));              \
        return npyv_reduce_##INTRIN##_f32(a);                                                        \
    }                                                                                                \
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                       \
    {                                                                                                \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                        \
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {                                                   \
            return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));                                         \
        }                                                                                            \
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));            \
        return npyv_reduce_##INTRIN##_f64(a);                                                        \
    }                                                                                                \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                                        \
    {                                                                                                \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                        \
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                                                   \
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};                             \
            return pnan.f;                                                                           \
        }                                                                                            \
        return npyv_reduce_##INTRIN##_f32(a);                                                        \
    }                                                                                                \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)                                       \
    {                                                                                                \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                        \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                                                   \
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};                   \
            return pnan.d;                                                                           \
        }                                                                                            \
        return npyv_reduce_##INTRIN##_f64(a);                                                        \
    }
NPY_IMPL_AVX2_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_AVX2_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
#undef NPY_IMPL_AVX2_REDUCE_MINMAX

// reduce min&max for 8&16-bits
#define NPY_IMPL_AVX256_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                        \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m256i a)                                        \
    {                                                                                                \
        __m128i v128 = _mm_##VINTRIN##16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)); \
        __m128i v64 =  _mm_##VINTRIN##16(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));    \
        __m128i v32 = _mm_##VINTRIN##16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));       \
        __m128i v16 = _mm_##VINTRIN##16(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));     \
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                    \
    }                                                                                                \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m256i a)                                          \
    {                                                                                                \
        __m128i v128 = _mm_##VINTRIN##8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));  \
        __m128i v64 =  _mm_##VINTRIN##8(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 0, 3, 2)));     \
        __m128i v32 = _mm_##VINTRIN##8(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));        \
        __m128i v16 = _mm_##VINTRIN##8(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));      \
        __m128i v8 = _mm_##VINTRIN##8(v16, _mm_srli_epi16(v16, 8));                                  \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                     \
    }
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_AVX256_REDUCE_MINMAX(npy_int,  max_s, max_epi)
#undef NPY_IMPL_AVX256_REDUCE_MINMAX

// round to nearest integer even
#define npyv_rint_f32(A) _mm256_round_ps(A, _MM_FROUND_TO_NEAREST_INT)
#define npyv_rint_f64(A) _mm256_round_pd(A, _MM_FROUND_TO_NEAREST_INT)

// ceil
#define npyv_ceil_f32 _mm256_ceil_ps
#define npyv_ceil_f64 _mm256_ceil_pd

// trunc
#define npyv_trunc_f32(A) _mm256_round_ps(A, _MM_FROUND_TO_ZERO)
#define npyv_trunc_f64(A) _mm256_round_pd(A, _MM_FROUND_TO_ZERO)

// floor
#define npyv_floor_f32 _mm256_floor_ps
#define npyv_floor_f64 _mm256_floor_pd

#endif // _NPY_SIMD_AVX2_MATH_H
