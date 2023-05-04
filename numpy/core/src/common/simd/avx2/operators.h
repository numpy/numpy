#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_OPERATORS_H
#define _NPY_SIMD_AVX2_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
#define npyv_shl_u16(A, C) _mm256_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s16(A, C) _mm256_sll_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u32(A, C) _mm256_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s32(A, C) _mm256_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u64(A, C) _mm256_sll_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s64(A, C) _mm256_sll_epi64(A, _mm_cvtsi32_si128(C))

// left by an immediate constant
#define npyv_shli_u16 _mm256_slli_epi16
#define npyv_shli_s16 _mm256_slli_epi16
#define npyv_shli_u32 _mm256_slli_epi32
#define npyv_shli_s32 _mm256_slli_epi32
#define npyv_shli_u64 _mm256_slli_epi64
#define npyv_shli_s64 _mm256_slli_epi64

// right
#define npyv_shr_u16(A, C) _mm256_srl_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s16(A, C) _mm256_sra_epi16(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u32(A, C) _mm256_srl_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s32(A, C) _mm256_sra_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u64(A, C) _mm256_srl_epi64(A, _mm_cvtsi32_si128(C))
NPY_FINLINE __m256i npyv_shr_s64(__m256i a, int c)
{
    const __m256i sbit = _mm256_set1_epi64x(0x8000000000000000);
    const __m128i c64  = _mm_cvtsi32_si128(c);
    __m256i r = _mm256_srl_epi64(_mm256_add_epi64(a, sbit), c64);
    return _mm256_sub_epi64(r, _mm256_srl_epi64(sbit, c64));
}

// right by an immediate constant
#define npyv_shri_u16 _mm256_srli_epi16
#define npyv_shri_s16 _mm256_srai_epi16
#define npyv_shri_u32 _mm256_srli_epi32
#define npyv_shri_s32 _mm256_srai_epi32
#define npyv_shri_u64 _mm256_srli_epi64
#define npyv_shri_s64  npyv_shr_s64

/***************************
 * Logical
 ***************************/
// AND
#define npyv_and_u8  _mm256_and_si256
#define npyv_and_s8  _mm256_and_si256
#define npyv_and_u16 _mm256_and_si256
#define npyv_and_s16 _mm256_and_si256
#define npyv_and_u32 _mm256_and_si256
#define npyv_and_s32 _mm256_and_si256
#define npyv_and_u64 _mm256_and_si256
#define npyv_and_s64 _mm256_and_si256
#define npyv_and_f32 _mm256_and_ps
#define npyv_and_f64 _mm256_and_pd
#define npyv_and_b8  _mm256_and_si256
#define npyv_and_b16 _mm256_and_si256
#define npyv_and_b32 _mm256_and_si256
#define npyv_and_b64 _mm256_and_si256

// OR
#define npyv_or_u8  _mm256_or_si256
#define npyv_or_s8  _mm256_or_si256
#define npyv_or_u16 _mm256_or_si256
#define npyv_or_s16 _mm256_or_si256
#define npyv_or_u32 _mm256_or_si256
#define npyv_or_s32 _mm256_or_si256
#define npyv_or_u64 _mm256_or_si256
#define npyv_or_s64 _mm256_or_si256
#define npyv_or_f32 _mm256_or_ps
#define npyv_or_f64 _mm256_or_pd
#define npyv_or_b8  _mm256_or_si256
#define npyv_or_b16 _mm256_or_si256
#define npyv_or_b32 _mm256_or_si256
#define npyv_or_b64 _mm256_or_si256

// XOR
#define npyv_xor_u8  _mm256_xor_si256
#define npyv_xor_s8  _mm256_xor_si256
#define npyv_xor_u16 _mm256_xor_si256
#define npyv_xor_s16 _mm256_xor_si256
#define npyv_xor_u32 _mm256_xor_si256
#define npyv_xor_s32 _mm256_xor_si256
#define npyv_xor_u64 _mm256_xor_si256
#define npyv_xor_s64 _mm256_xor_si256
#define npyv_xor_f32 _mm256_xor_ps
#define npyv_xor_f64 _mm256_xor_pd
#define npyv_xor_b8  _mm256_xor_si256
#define npyv_xor_b16 _mm256_xor_si256
#define npyv_xor_b32 _mm256_xor_si256
#define npyv_xor_b64 _mm256_xor_si256

// NOT
#define npyv_not_u8(A) _mm256_xor_si256(A, _mm256_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#define npyv_not_f32(A) _mm256_xor_ps(A, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))
#define npyv_not_f64(A) _mm256_xor_pd(A, _mm256_castsi256_pd(_mm256_set1_epi32(-1)))
#define npyv_not_b8  npyv_not_u8
#define npyv_not_b16 npyv_not_u8
#define npyv_not_b32 npyv_not_u8
#define npyv_not_b64 npyv_not_u8

// ANDC, ORC and XNOR
#define npyv_andc_u8(A, B) _mm256_andnot_si256(B, A)
#define npyv_andc_b8(A, B) _mm256_andnot_si256(B, A)
#define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
#define npyv_xnor_b8 _mm256_cmpeq_epi8

/***************************
 * Comparison
 ***************************/

// int Equal
#define npyv_cmpeq_u8  _mm256_cmpeq_epi8
#define npyv_cmpeq_s8  _mm256_cmpeq_epi8
#define npyv_cmpeq_u16 _mm256_cmpeq_epi16
#define npyv_cmpeq_s16 _mm256_cmpeq_epi16
#define npyv_cmpeq_u32 _mm256_cmpeq_epi32
#define npyv_cmpeq_s32 _mm256_cmpeq_epi32
#define npyv_cmpeq_u64 _mm256_cmpeq_epi64
#define npyv_cmpeq_s64 _mm256_cmpeq_epi64

// int Not Equal
#define npyv_cmpneq_u8(A, B) npyv_not_u8(_mm256_cmpeq_epi8(A, B))
#define npyv_cmpneq_s8 npyv_cmpneq_u8
#define npyv_cmpneq_u16(A, B) npyv_not_u16(_mm256_cmpeq_epi16(A, B))
#define npyv_cmpneq_s16 npyv_cmpneq_u16
#define npyv_cmpneq_u32(A, B) npyv_not_u32(_mm256_cmpeq_epi32(A, B))
#define npyv_cmpneq_s32 npyv_cmpneq_u32
#define npyv_cmpneq_u64(A, B) npyv_not_u64(_mm256_cmpeq_epi64(A, B))
#define npyv_cmpneq_s64 npyv_cmpneq_u64

// signed greater than
#define npyv_cmpgt_s8  _mm256_cmpgt_epi8
#define npyv_cmpgt_s16 _mm256_cmpgt_epi16
#define npyv_cmpgt_s32 _mm256_cmpgt_epi32
#define npyv_cmpgt_s64 _mm256_cmpgt_epi64

// signed greater than or equal
#define npyv_cmpge_s8(A, B)  npyv_not_s8(_mm256_cmpgt_epi8(B, A))
#define npyv_cmpge_s16(A, B) npyv_not_s16(_mm256_cmpgt_epi16(B, A))
#define npyv_cmpge_s32(A, B) npyv_not_s32(_mm256_cmpgt_epi32(B, A))
#define npyv_cmpge_s64(A, B) npyv_not_s64(_mm256_cmpgt_epi64(B, A))

// unsigned greater than
#define NPYV_IMPL_AVX2_UNSIGNED_GT(LEN, SIGN)                    \
    NPY_FINLINE __m256i npyv_cmpgt_u##LEN(__m256i a, __m256i b)  \
    {                                                            \
        const __m256i sbit = _mm256_set1_epi32(SIGN);            \
        return _mm256_cmpgt_epi##LEN(                            \
            _mm256_xor_si256(a, sbit), _mm256_xor_si256(b, sbit) \
        );                                                       \
    }

NPYV_IMPL_AVX2_UNSIGNED_GT(8,  0x80808080)
NPYV_IMPL_AVX2_UNSIGNED_GT(16, 0x80008000)
NPYV_IMPL_AVX2_UNSIGNED_GT(32, 0x80000000)

NPY_FINLINE __m256i npyv_cmpgt_u64(__m256i a, __m256i b)
{
    const __m256i sbit = _mm256_set1_epi64x(0x8000000000000000);
    return _mm256_cmpgt_epi64(_mm256_xor_si256(a, sbit), _mm256_xor_si256(b, sbit));
}

// unsigned greater than or equal
NPY_FINLINE __m256i npyv_cmpge_u8(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi8(a, _mm256_max_epu8(a, b)); }
NPY_FINLINE __m256i npyv_cmpge_u16(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi16(a, _mm256_max_epu16(a, b)); }
NPY_FINLINE __m256i npyv_cmpge_u32(__m256i a, __m256i b)
{ return _mm256_cmpeq_epi32(a, _mm256_max_epu32(a, b)); }
#define npyv_cmpge_u64(A, B) npyv_not_u64(npyv_cmpgt_u64(B, A))

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

// precision comparison (ordered)
#define npyv_cmpeq_f32(A, B)  _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_EQ_OQ))
#define npyv_cmpeq_f64(A, B)  _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_EQ_OQ))
#define npyv_cmpneq_f32(A, B) _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_NEQ_UQ))
#define npyv_cmpneq_f64(A, B) _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_NEQ_UQ))
#define npyv_cmplt_f32(A, B)  _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_LT_OQ))
#define npyv_cmplt_f64(A, B)  _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_LT_OQ))
#define npyv_cmple_f32(A, B)  _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_LE_OQ))
#define npyv_cmple_f64(A, B)  _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_LE_OQ))
#define npyv_cmpgt_f32(A, B)  _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_GT_OQ))
#define npyv_cmpgt_f64(A, B)  _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_GT_OQ))
#define npyv_cmpge_f32(A, B)  _mm256_castps_si256(_mm256_cmp_ps(A, B, _CMP_GE_OQ))
#define npyv_cmpge_f64(A, B)  _mm256_castpd_si256(_mm256_cmp_pd(A, B, _CMP_GE_OQ))

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_ORD_Q)); }
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return _mm256_castpd_si256(_mm256_cmp_pd(a, a, _CMP_ORD_Q)); }

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
#define NPYV_IMPL_AVX2_ANYALL(SFX)                \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a) \
    { return _mm256_movemask_epi8(a) != 0; }      \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a) \
    { return _mm256_movemask_epi8(a) == -1; }
NPYV_IMPL_AVX2_ANYALL(b8)
NPYV_IMPL_AVX2_ANYALL(b16)
NPYV_IMPL_AVX2_ANYALL(b32)
NPYV_IMPL_AVX2_ANYALL(b64)
#undef NPYV_IMPL_AVX2_ANYALL

#define NPYV_IMPL_AVX2_ANYALL(SFX)                     \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)      \
    {                                                  \
        return _mm256_movemask_epi8(                   \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX())     \
        ) != -1;                                       \
    }                                                  \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)      \
    {                                                  \
        return _mm256_movemask_epi8(                   \
            npyv_cmpeq_##SFX(a, npyv_zero_##SFX())     \
        ) == 0;                                        \
    }
NPYV_IMPL_AVX2_ANYALL(u8)
NPYV_IMPL_AVX2_ANYALL(s8)
NPYV_IMPL_AVX2_ANYALL(u16)
NPYV_IMPL_AVX2_ANYALL(s16)
NPYV_IMPL_AVX2_ANYALL(u32)
NPYV_IMPL_AVX2_ANYALL(s32)
NPYV_IMPL_AVX2_ANYALL(u64)
NPYV_IMPL_AVX2_ANYALL(s64)
#undef NPYV_IMPL_AVX2_ANYALL

#define NPYV_IMPL_AVX2_ANYALL(SFX, XSFX, MASK)                   \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)                \
    {                                                            \
        return _mm256_movemask_##XSFX(                           \
            _mm256_cmp_##XSFX(a, npyv_zero_##SFX(), _CMP_EQ_OQ)  \
        ) != MASK;                                               \
    }                                                            \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)                \
    {                                                            \
        return _mm256_movemask_##XSFX(                           \
            _mm256_cmp_##XSFX(a, npyv_zero_##SFX(), _CMP_EQ_OQ)  \
        ) == 0;                                                  \
    }
NPYV_IMPL_AVX2_ANYALL(f32, ps, 0xff)
NPYV_IMPL_AVX2_ANYALL(f64, pd, 0xf)
#undef NPYV_IMPL_AVX2_ANYALL

#endif // _NPY_SIMD_AVX2_OPERATORS_H
