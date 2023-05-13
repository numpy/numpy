#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MATH_H
#define _NPY_SIMD_SSE_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm_sqrt_ps
#define npyv_sqrt_f64 _mm_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm_div_ps(_mm_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm_div_pd(_mm_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm_and_ps(
        a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm_and_pd(
        a, _mm_castsi128_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm_mul_pd(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 _mm_max_ps
#define npyv_max_f64 _mm_max_pd
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(b);
    __m128 max = _mm_max_ps(a, b);
    return npyv_select_f32(nn, max, a);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(b);
    __m128d max = _mm_max_pd(a, b);
    return npyv_select_f64(nn, max, a);
}
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(a);
    __m128 max = _mm_max_ps(a, b);
    return npyv_select_f32(nn, max, a);
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(a);
    __m128d max = _mm_max_pd(a, b);
    return npyv_select_f64(nn, max, a);
}
// Maximum, integer operations
#ifdef NPY_HAVE_SSE41
    #define npyv_max_s8 _mm_max_epi8
    #define npyv_max_u16 _mm_max_epu16
    #define npyv_max_u32 _mm_max_epu32
    #define npyv_max_s32 _mm_max_epi32
#else
    NPY_FINLINE npyv_s8 npyv_max_s8(npyv_s8 a, npyv_s8 b)
    {
        return npyv_select_s8(npyv_cmpgt_s8(a, b), a, b);
    }
    NPY_FINLINE npyv_u16 npyv_max_u16(npyv_u16 a, npyv_u16 b)
    {
        return npyv_select_u16(npyv_cmpgt_u16(a, b), a, b);
    }
    NPY_FINLINE npyv_u32 npyv_max_u32(npyv_u32 a, npyv_u32 b)
    {
        return npyv_select_u32(npyv_cmpgt_u32(a, b), a, b);
    }
    NPY_FINLINE npyv_s32 npyv_max_s32(npyv_s32 a, npyv_s32 b)
    {
        return npyv_select_s32(npyv_cmpgt_s32(a, b), a, b);
    }
#endif
#define npyv_max_u8 _mm_max_epu8
#define npyv_max_s16 _mm_max_epi16
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    return npyv_select_u64(npyv_cmpgt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    return npyv_select_s64(npyv_cmpgt_s64(a, b), a, b);
}

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 _mm_min_ps
#define npyv_min_f64 _mm_min_pd
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(b);
    __m128 min = _mm_min_ps(a, b);
    return npyv_select_f32(nn, min, a);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(b);
    __m128d min = _mm_min_pd(a, b);
    return npyv_select_f64(nn, min, a);
}
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __m128i nn = npyv_notnan_f32(a);
    __m128 min = _mm_min_ps(a, b);
    return npyv_select_f32(nn, min, a);
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __m128i nn  = npyv_notnan_f64(a);
    __m128d min = _mm_min_pd(a, b);
    return npyv_select_f64(nn, min, a);
}
// Minimum, integer operations
#ifdef NPY_HAVE_SSE41
    #define npyv_min_s8 _mm_min_epi8
    #define npyv_min_u16 _mm_min_epu16
    #define npyv_min_u32 _mm_min_epu32
    #define npyv_min_s32 _mm_min_epi32
#else
    NPY_FINLINE npyv_s8 npyv_min_s8(npyv_s8 a, npyv_s8 b)
    {
        return npyv_select_s8(npyv_cmplt_s8(a, b), a, b);
    }
    NPY_FINLINE npyv_u16 npyv_min_u16(npyv_u16 a, npyv_u16 b)
    {
        return npyv_select_u16(npyv_cmplt_u16(a, b), a, b);
    }
    NPY_FINLINE npyv_u32 npyv_min_u32(npyv_u32 a, npyv_u32 b)
    {
        return npyv_select_u32(npyv_cmplt_u32(a, b), a, b);
    }
    NPY_FINLINE npyv_s32 npyv_min_s32(npyv_s32 a, npyv_s32 b)
    {
        return npyv_select_s32(npyv_cmplt_s32(a, b), a, b);
    }
#endif
#define npyv_min_u8 _mm_min_epu8
#define npyv_min_s16 _mm_min_epi16
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    return npyv_select_u64(npyv_cmplt_u64(a, b), a, b);
}
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    return npyv_select_s64(npyv_cmplt_s64(a, b), a, b);
}

// reduce min&max for 32&64-bits
#define NPY_IMPL_SSE_REDUCE_MINMAX(STYPE, INTRIN, VINTRIN)                                     \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m128i a)                                  \
    {                                                                                          \
        __m128i v64 =  npyv_##INTRIN##32(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));    \
        __m128i v32 = npyv_##INTRIN##32(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1))); \
        return (STYPE##32)_mm_cvtsi128_si32(v32);                                              \
    }                                                                                          \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m128i a)                                  \
    {                                                                                          \
        __m128i v64  = npyv_##INTRIN##64(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));    \
        return (STYPE##64)npyv_extract0_u64(v64);                                              \
    }

NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, min_u, min_epu)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  min_s, min_epi)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, max_u, max_epu)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  max_s, max_epi)
#undef NPY_IMPL_SSE_REDUCE_MINMAX
// reduce min&max for ps & pd
#define NPY_IMPL_SSE_REDUCE_MINMAX(INTRIN, INF, INF64)                                          \
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                    \
    {                                                                                           \
        __m128 v64 =  _mm_##INTRIN##_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 3, 2)));      \
        __m128 v32 = _mm_##INTRIN##_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1))); \
        return _mm_cvtss_f32(v32);                                                              \
    }                                                                                           \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                   \
    {                                                                                           \
        __m128d v64 = _mm_##INTRIN##_pd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE(0, 0, 0, 1)));      \
        return _mm_cvtsd_f64(v64);                                                              \
    }                                                                                           \
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                   \
    {                                                                                           \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                   \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                                              \
            return _mm_cvtss_f32(a);                                                            \
        }                                                                                       \
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));         \
        return npyv_reduce_##INTRIN##_f32(a);                                                   \
    }                                                                                           \
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                  \
    {                                                                                           \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                   \
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {                                              \
            return _mm_cvtsd_f64(a);                                                            \
        }                                                                                       \
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));       \
        return npyv_reduce_##INTRIN##_f64(a);                                                   \
    }                                                                                           \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                                   \
    {                                                                                           \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                   \
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                                              \
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};                        \
            return pnan.f;                                                                      \
        }                                                                                       \
        return npyv_reduce_##INTRIN##_f32(a);                                                   \
    }                                                                                           \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)                                  \
    {                                                                                           \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                   \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                                              \
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};              \
            return pnan.d;                                                                      \
        }                                                                                       \
        return npyv_reduce_##INTRIN##_f64(a);                                                   \
    }

NPY_IMPL_SSE_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_SSE_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
#undef NPY_IMPL_SSE_REDUCE_MINMAX

// reduce min&max for 8&16-bits
#define NPY_IMPL_SSE_REDUCE_MINMAX(STYPE, INTRIN)                                                    \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m128i a)                                        \
    {                                                                                                \
        __m128i v64 =  npyv_##INTRIN##16(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));          \
        __m128i v32 = npyv_##INTRIN##16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));       \
        __m128i v16 = npyv_##INTRIN##16(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));     \
        return (STYPE##16)_mm_cvtsi128_si32(v16);                                                    \
    }                                                                                                \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m128i a)                                          \
    {                                                                                                \
        __m128i v64 =  npyv_##INTRIN##8(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 3, 2)));           \
        __m128i v32 = npyv_##INTRIN##8(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 0, 0, 1)));        \
        __m128i v16 = npyv_##INTRIN##8(v32, _mm_shufflelo_epi16(v32, _MM_SHUFFLE(0, 0, 0, 1)));      \
        __m128i v8 = npyv_##INTRIN##8(v16, _mm_srli_epi16(v16, 8));                                  \
        return (STYPE##16)_mm_cvtsi128_si32(v8);                                                     \
    }
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, min_u)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  min_s)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_uint, max_u)
NPY_IMPL_SSE_REDUCE_MINMAX(npy_int,  max_s)
#undef NPY_IMPL_SSE_REDUCE_MINMAX

// round to nearest integer even
NPY_FINLINE npyv_f32 npyv_rint_f32(npyv_f32 a)
{
#ifdef NPY_HAVE_SSE41
    return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
#else
    const __m128 szero = _mm_set1_ps(-0.0f);
    const __m128i exp_mask = _mm_set1_epi32(0xff000000);

    __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
            nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
            nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

    // eliminate nans/inf to avoid invalid fp errors
    __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
    __m128i roundi = _mm_cvtps_epi32(x);
    __m128 round = _mm_cvtepi32_ps(roundi);
    // respect signed zero
    round = _mm_or_ps(round, _mm_and_ps(a, szero));
    // if overflow return a
    __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
    // a if a overflow or nonfinite
    return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, round);
#endif
}

// round to nearest integer even
NPY_FINLINE npyv_f64 npyv_rint_f64(npyv_f64 a)
{
#ifdef NPY_HAVE_SSE41
    return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
#else
    const __m128d szero = _mm_set1_pd(-0.0);
    const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
    __m128d nan_mask = _mm_cmpunord_pd(a, a);
    // eliminate nans to avoid invalid fp errors within cmpge
    __m128d abs_x = npyv_abs_f64(_mm_xor_pd(nan_mask, a));
    // round by add magic number 2^52
    // assuming that MXCSR register is set to rounding
    __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
    // copysign
    round = _mm_or_pd(round, _mm_and_pd(a, szero));
    // a if |a| >= 2^52 or a == NaN
    __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
            mask = _mm_or_pd(mask, nan_mask);
    return npyv_select_f64(_mm_castpd_si128(mask), a, round);
#endif
}
// ceil
#ifdef NPY_HAVE_SSE41
    #define npyv_ceil_f32 _mm_ceil_ps
    #define npyv_ceil_f64 _mm_ceil_pd
#else
    NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
    {
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 szero = _mm_set1_ps(-0.0f);
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);

        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
                nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
                nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

        // eliminate nans/inf to avoid invalid fp errors
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
        __m128i roundi = _mm_cvtps_epi32(x);
        __m128 round = _mm_cvtepi32_ps(roundi);
        __m128 ceil = _mm_add_ps(round, _mm_and_ps(_mm_cmplt_ps(round, x), one));
        // respect signed zero
        ceil = _mm_or_ps(ceil, _mm_and_ps(a, szero));
        // if overflow return a
        __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
        // a if a overflow or nonfinite
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, ceil);
    }
    NPY_FINLINE npyv_f64 npyv_ceil_f64(npyv_f64 a)
    {
        const __m128d one = _mm_set1_pd(1.0);
        const __m128d szero = _mm_set1_pd(-0.0);
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        // eliminate nans to avoid invalid fp errors within cmpge
        __m128d x = _mm_xor_pd(nan_mask, a);
        __m128d abs_x = npyv_abs_f64(x);
        __m128d sign_x = _mm_and_pd(x, szero);
        // round by add magic number 2^52
        // assuming that MXCSR register is set to rounding
        __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        // copysign
        round = _mm_or_pd(round, sign_x);
        __m128d ceil = _mm_add_pd(round, _mm_and_pd(_mm_cmplt_pd(round, x), one));
        // respects sign of 0.0
        ceil = _mm_or_pd(ceil, sign_x);
        // a if |a| >= 2^52 or a == NaN
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
                mask = _mm_or_pd(mask, nan_mask);
        return npyv_select_f64(_mm_castpd_si128(mask), a, ceil);
    }
#endif

// trunc
#ifdef NPY_HAVE_SSE41
    #define npyv_trunc_f32(A) _mm_round_ps(A, _MM_FROUND_TO_ZERO)
    #define npyv_trunc_f64(A) _mm_round_pd(A, _MM_FROUND_TO_ZERO)
#else
    NPY_FINLINE npyv_f32 npyv_trunc_f32(npyv_f32 a)
    {
        const __m128 szero = _mm_set1_ps(-0.0f);
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);

        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
                nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
                nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

        // eliminate nans/inf to avoid invalid fp errors
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
        __m128i trunci = _mm_cvttps_epi32(x);
        __m128 trunc = _mm_cvtepi32_ps(trunci);
        // respect signed zero, e.g. -0.5 -> -0.0
        trunc = _mm_or_ps(trunc, _mm_and_ps(a, szero));
        // if overflow return a
        __m128i overflow_mask = _mm_cmpeq_epi32(trunci, _mm_castps_si128(szero));
        // a if a overflow or nonfinite
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, trunc);
    }
    NPY_FINLINE npyv_f64 npyv_trunc_f64(npyv_f64 a)
    {
        const __m128d one = _mm_set1_pd(1.0);
        const __m128d szero = _mm_set1_pd(-0.0);
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        // eliminate nans to avoid invalid fp errors within cmpge
        __m128d abs_x = npyv_abs_f64(_mm_xor_pd(nan_mask, a));
        // round by add magic number 2^52
        // assuming that MXCSR register is set to rounding
        __m128d abs_round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        __m128d subtrahend = _mm_and_pd(_mm_cmpgt_pd(abs_round, abs_x), one);
        __m128d trunc = _mm_sub_pd(abs_round, subtrahend);
        // copysign
        trunc = _mm_or_pd(trunc, _mm_and_pd(a, szero));
        // a if |a| >= 2^52 or a == NaN
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
               mask = _mm_or_pd(mask, nan_mask);
        return npyv_select_f64(_mm_castpd_si128(mask), a, trunc);
    }
#endif

// floor
#ifdef NPY_HAVE_SSE41
    #define npyv_floor_f32 _mm_floor_ps
    #define npyv_floor_f64 _mm_floor_pd
#else
    NPY_FINLINE npyv_f32 npyv_floor_f32(npyv_f32 a)
    {
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 szero = _mm_set1_ps(-0.0f);
        const __m128i exp_mask = _mm_set1_epi32(0xff000000);

        __m128i nfinite_mask = _mm_slli_epi32(_mm_castps_si128(a), 1);
                nfinite_mask = _mm_and_si128(nfinite_mask, exp_mask);
                nfinite_mask = _mm_cmpeq_epi32(nfinite_mask, exp_mask);

        // eliminate nans/inf to avoid invalid fp errors
        __m128 x = _mm_xor_ps(a, _mm_castsi128_ps(nfinite_mask));
        __m128i roundi = _mm_cvtps_epi32(x);
        __m128 round = _mm_cvtepi32_ps(roundi);
        __m128 floor = _mm_sub_ps(round, _mm_and_ps(_mm_cmpgt_ps(round, x), one));
        // respect signed zero
        floor = _mm_or_ps(floor, _mm_and_ps(a, szero));
        // if overflow return a
        __m128i overflow_mask = _mm_cmpeq_epi32(roundi, _mm_castps_si128(szero));
        // a if a overflow or nonfinite
        return npyv_select_f32(_mm_or_si128(nfinite_mask, overflow_mask), a, floor);
    }
    NPY_FINLINE npyv_f64 npyv_floor_f64(npyv_f64 a)
    {
        const __m128d one = _mm_set1_pd(1.0f);
        const __m128d szero = _mm_set1_pd(-0.0f);
        const __m128d two_power_52 = _mm_set1_pd(0x10000000000000);
        __m128d nan_mask = _mm_cmpunord_pd(a, a);
        // eliminate nans to avoid invalid fp errors within cmpge
        __m128d x = _mm_xor_pd(nan_mask, a);
        __m128d abs_x = npyv_abs_f64(x);
        __m128d sign_x = _mm_and_pd(x, szero);
        // round by add magic number 2^52
        // assuming that MXCSR register is set to rounding
        __m128d round = _mm_sub_pd(_mm_add_pd(two_power_52, abs_x), two_power_52);
        // copysign
        round = _mm_or_pd(round, sign_x);
        __m128d floor = _mm_sub_pd(round, _mm_and_pd(_mm_cmpgt_pd(round, x), one));
        // a if |a| >= 2^52 or a == NaN
        __m128d mask = _mm_cmpge_pd(abs_x, two_power_52);
               mask = _mm_or_pd(mask, nan_mask);
        return npyv_select_f64(_mm_castpd_si128(mask), a, floor);
    }
#endif // NPY_HAVE_SSE41

#endif // _NPY_SIMD_SSE_MATH_H
