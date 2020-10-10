#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_OPERATORS_H
#define _NPY_SIMD_SSE_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
#define npyv_shl_u16(A, C) _mm_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s16(A, C) _mm_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u32(A, C) _mm_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s32(A, C) _mm_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u64(A, C) _mm_sll_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s64(A, C) _mm_sll_epi64(A, _mm_cvtsi32_si128(C))

// left by an immediate constant
#define npyv_shli_u16 _mm_slli_epi16
#define npyv_shli_s16 _mm_slli_epi16
#define npyv_shli_u32 _mm_slli_epi32
#define npyv_shli_s32 _mm_slli_epi32
#define npyv_shli_u64 _mm_slli_epi64
#define npyv_shli_s64 _mm_slli_epi64

// right
#define npyv_shr_u16(A, C) _mm_srl_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s16(A, C) _mm_sra_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u32(A, C) _mm_srl_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s32(A, C) _mm_sra_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u64(A, C) _mm_srl_epi64(A, _mm_cvtsi32_si128(C))
NPY_FINLINE __m128i npyv_shr_s64(__m128i a, int c)
{
    const __m128i sbit = npyv_setall_s64(0x8000000000000000);
    const __m128i cv   = _mm_cvtsi32_si128(c);
    __m128i r = _mm_srl_epi64(_mm_add_epi64(a, sbit), cv);
    return _mm_sub_epi64(r, _mm_srl_epi64(sbit, cv));
}

// Right by an immediate constant
#define npyv_shri_u16 _mm_srli_epi16
#define npyv_shri_s16 _mm_srai_epi16
#define npyv_shri_u32 _mm_srli_epi32
#define npyv_shri_s32 _mm_srai_epi32
#define npyv_shri_u64 _mm_srli_epi64
#define npyv_shri_s64  npyv_shr_s64

/***************************
 * Logical
 ***************************/

// AND
#define npyv_and_u8  _mm_and_si128
#define npyv_and_s8  _mm_and_si128
#define npyv_and_u16 _mm_and_si128
#define npyv_and_s16 _mm_and_si128
#define npyv_and_u32 _mm_and_si128
#define npyv_and_s32 _mm_and_si128
#define npyv_and_u64 _mm_and_si128
#define npyv_and_s64 _mm_and_si128
#define npyv_and_f32 _mm_and_ps
#define npyv_and_f64 _mm_and_pd

// OR
#define npyv_or_u8  _mm_or_si128
#define npyv_or_s8  _mm_or_si128
#define npyv_or_u16 _mm_or_si128
#define npyv_or_s16 _mm_or_si128
#define npyv_or_u32 _mm_or_si128
#define npyv_or_s32 _mm_or_si128
#define npyv_or_u64 _mm_or_si128
#define npyv_or_s64 _mm_or_si128
#define npyv_or_f32 _mm_or_ps
#define npyv_or_f64 _mm_or_pd

// XOR
#define npyv_xor_u8  _mm_xor_si128
#define npyv_xor_s8  _mm_xor_si128
#define npyv_xor_u16 _mm_xor_si128
#define npyv_xor_s16 _mm_xor_si128
#define npyv_xor_u32 _mm_xor_si128
#define npyv_xor_s32 _mm_xor_si128
#define npyv_xor_u64 _mm_xor_si128
#define npyv_xor_s64 _mm_xor_si128
#define npyv_xor_f32 _mm_xor_ps
#define npyv_xor_f64 _mm_xor_pd

// NOT
#define npyv_not_u8(A) _mm_xor_si128(A, _mm_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#define npyv_not_f32(A) _mm_xor_ps(A, _mm_castsi128_ps(_mm_set1_epi32(-1)))
#define npyv_not_f64(A) _mm_xor_pd(A, _mm_castsi128_pd(_mm_set1_epi32(-1)))

/***************************
 * Comparison
 ***************************/

// Int Equal
#define npyv_cmpeq_u8  _mm_cmpeq_epi8
#define npyv_cmpeq_s8  _mm_cmpeq_epi8
#define npyv_cmpeq_u16 _mm_cmpeq_epi16
#define npyv_cmpeq_s16 _mm_cmpeq_epi16
#define npyv_cmpeq_u32 _mm_cmpeq_epi32
#define npyv_cmpeq_s32 _mm_cmpeq_epi32
#define npyv_cmpeq_s64  npyv_cmpeq_u64

#ifdef NPY_HAVE_SSE41
    #define npyv_cmpeq_u64 _mm_cmpeq_epi64
#else
    NPY_FINLINE __m128i npyv_cmpeq_u64(__m128i a, __m128i b)
    {
        __m128i cmpeq = _mm_cmpeq_epi32(a, b);
        __m128i cmpeq_h = _mm_srli_epi64(cmpeq, 32);
        __m128i test = _mm_and_si128(cmpeq, cmpeq_h);
        return _mm_shuffle_epi32(test, _MM_SHUFFLE(2, 2, 0, 0));
    }
#endif

// Int Not Equal
#ifdef NPY_HAVE_XOP
    #define npyv_cmpneq_u8  _mm_comneq_epi8
    #define npyv_cmpneq_u16 _mm_comneq_epi16
    #define npyv_cmpneq_u32 _mm_comneq_epi32
    #define npyv_cmpneq_u64 _mm_comneq_epi64
#else
    #define npyv_cmpneq_u8(A, B)  npyv_not_u8(npyv_cmpeq_u8(A, B))
    #define npyv_cmpneq_u16(A, B) npyv_not_u16(npyv_cmpeq_u16(A, B))
    #define npyv_cmpneq_u32(A, B) npyv_not_u32(npyv_cmpeq_u32(A, B))
    #define npyv_cmpneq_u64(A, B) npyv_not_u64(npyv_cmpeq_u64(A, B))
#endif
#define npyv_cmpneq_s8  npyv_cmpneq_u8
#define npyv_cmpneq_s16 npyv_cmpneq_u16
#define npyv_cmpneq_s32 npyv_cmpneq_u32
#define npyv_cmpneq_s64 npyv_cmpneq_u64

// signed greater than
#define npyv_cmpgt_s8  _mm_cmpgt_epi8
#define npyv_cmpgt_s16 _mm_cmpgt_epi16
#define npyv_cmpgt_s32 _mm_cmpgt_epi32

#ifdef NPY_HAVE_SSE42
    #define npyv_cmpgt_s64 _mm_cmpgt_epi64
#else
    NPY_FINLINE __m128i npyv_cmpgt_s64(__m128i a, __m128i b)
    {
        __m128i sub = _mm_sub_epi64(b, a);
        __m128i nsame_sbit = _mm_xor_si128(a, b);
        // nsame_sbit ? b : sub
        __m128i test = _mm_xor_si128(sub, _mm_and_si128(_mm_xor_si128(sub, b), nsame_sbit));
        __m128i extend_sbit = _mm_shuffle_epi32(_mm_srai_epi32(test, 31), _MM_SHUFFLE(3, 3, 1, 1));
        return  extend_sbit;
    }
#endif

// signed greater than or equal
#ifdef NPY_HAVE_XOP
    #define npyv_cmpge_s8  _mm_comge_epi8
    #define npyv_cmpge_s16 _mm_comge_epi16
    #define npyv_cmpge_s32 _mm_comge_epi32
    #define npyv_cmpge_s64 _mm_comge_epi64
#else
    #define npyv_cmpge_s8(A, B)  npyv_not_s8(_mm_cmpgt_epi8(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_s16(_mm_cmpgt_epi16(B, A))
    #define npyv_cmpge_s32(A, B) npyv_not_s32(_mm_cmpgt_epi32(B, A))
    #define npyv_cmpge_s64(A, B) npyv_not_s64(npyv_cmpgt_s64(B, A))
#endif

// unsigned greater than
#ifdef NPY_HAVE_XOP
    #define npyv_cmpgt_u8  _mm_comgt_epu8
    #define npyv_cmpgt_u16 _mm_comgt_epu16
    #define npyv_cmpgt_u32 _mm_comgt_epu32
    #define npyv_cmpgt_u64 _mm_comgt_epu64
#else
    #define NPYV_IMPL_SSE_UNSIGNED_GT(LEN, SIGN)                     \
        NPY_FINLINE __m128i npyv_cmpgt_u##LEN(__m128i a, __m128i b)  \
        {                                                            \
            const __m128i sbit = _mm_set1_epi32(SIGN);               \
            return _mm_cmpgt_epi##LEN(                               \
                _mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit)       \
            );                                                       \
        }

    NPYV_IMPL_SSE_UNSIGNED_GT(8,  0x80808080)
    NPYV_IMPL_SSE_UNSIGNED_GT(16, 0x80008000)
    NPYV_IMPL_SSE_UNSIGNED_GT(32, 0x80000000)

    NPY_FINLINE __m128i npyv_cmpgt_u64(__m128i a, __m128i b)
    {
        const __m128i sbit = npyv_setall_s64(0x8000000000000000);
        return npyv_cmpgt_s64(_mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit));
    }
#endif

// unsigned greater than or equal
#ifdef NPY_HAVE_XOP
    #define npyv_cmpge_u8  _mm_comge_epu8
    #define npyv_cmpge_u16 _mm_comge_epu16
    #define npyv_cmpge_u32 _mm_comge_epu32
    #define npyv_cmpge_u64 _mm_comge_epu64
#else
    NPY_FINLINE __m128i npyv_cmpge_u8(__m128i a, __m128i b)
    { return _mm_cmpeq_epi8(a, _mm_max_epu8(a, b)); }
    #ifdef NPY_HAVE_SSE41
        NPY_FINLINE __m128i npyv_cmpge_u16(__m128i a, __m128i b)
        { return _mm_cmpeq_epi16(a, _mm_max_epu16(a, b)); }
        NPY_FINLINE __m128i npyv_cmpge_u32(__m128i a, __m128i b)
        { return _mm_cmpeq_epi32(a, _mm_max_epu32(a, b)); }
    #else
        #define npyv_cmpge_u16(A, B) _mm_cmpeq_epi16(_mm_subs_epu16(B, A), _mm_setzero_si128())
        #define npyv_cmpge_u32(A, B) npyv_not_u32(npyv_cmpgt_u32(B, A))
    #endif
    #define npyv_cmpge_u64(A, B) npyv_not_u64(npyv_cmpgt_u64(B, A))
#endif

// less than
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)

// less than or equal
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)

// precision comparison
#define npyv_cmpeq_f32(a, b)  _mm_castps_si128(_mm_cmpeq_ps(a, b))
#define npyv_cmpeq_f64(a, b)  _mm_castpd_si128(_mm_cmpeq_pd(a, b))
#define npyv_cmpneq_f32(a, b) _mm_castps_si128(_mm_cmpneq_ps(a, b))
#define npyv_cmpneq_f64(a, b) _mm_castpd_si128(_mm_cmpneq_pd(a, b))
#define npyv_cmplt_f32(a, b)  _mm_castps_si128(_mm_cmplt_ps(a, b))
#define npyv_cmplt_f64(a, b)  _mm_castpd_si128(_mm_cmplt_pd(a, b))
#define npyv_cmple_f32(a, b)  _mm_castps_si128(_mm_cmple_ps(a, b))
#define npyv_cmple_f64(a, b)  _mm_castpd_si128(_mm_cmple_pd(a, b))
#define npyv_cmpgt_f32(a, b)  _mm_castps_si128(_mm_cmpgt_ps(a, b))
#define npyv_cmpgt_f64(a, b)  _mm_castpd_si128(_mm_cmpgt_pd(a, b))
#define npyv_cmpge_f32(a, b)  _mm_castps_si128(_mm_cmpge_ps(a, b))
#define npyv_cmpge_f64(a, b)  _mm_castpd_si128(_mm_cmpge_pd(a, b))

#endif // _NPY_SIMD_SSE_OPERATORS_H
