#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_OPERATORS_H
#define _NPY_SIMD_AVX512_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
#ifdef NPY_HAVE_AVX512BW
    #define npyv_shl_u16(A, C) _mm512_sll_epi16(A, _mm_cvtsi32_si128(C))
#else
    #define NPYV_IMPL_AVX512_SHIFT(FN, INTRIN)          \
        NPY_FINLINE __m512i npyv_##FN(__m512i a, int c) \
        {                                               \
            __m256i l  = npyv512_lower_si256(a);        \
            __m256i h  = npyv512_higher_si256(a);       \
            __m128i cv = _mm_cvtsi32_si128(c);          \
            l = _mm256_##INTRIN(l, cv);                 \
            h = _mm256_##INTRIN(h, cv);                 \
            return npyv512_combine_si256(l, h);         \
        }

    NPYV_IMPL_AVX512_SHIFT(shl_u16, sll_epi16)
#endif
#define npyv_shl_s16 npyv_shl_u16
#define npyv_shl_u32(A, C) _mm512_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s32(A, C) _mm512_sll_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shl_u64(A, C) _mm512_sll_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shl_s64(A, C) _mm512_sll_epi64(A, _mm_cvtsi32_si128(C))

// left by an immediate constant
#ifdef NPY_HAVE_AVX512BW
    #define npyv_shli_u16 _mm512_slli_epi16
#else
    #define npyv_shli_u16 npyv_shl_u16
#endif
#define npyv_shli_s16  npyv_shl_u16
#define npyv_shli_u32 _mm512_slli_epi32
#define npyv_shli_s32 _mm512_slli_epi32
#define npyv_shli_u64 _mm512_slli_epi64
#define npyv_shli_s64 _mm512_slli_epi64

// right
#ifdef NPY_HAVE_AVX512BW
    #define npyv_shr_u16(A, C) _mm512_srl_epi16(A, _mm_cvtsi32_si128(C))
    #define npyv_shr_s16(A, C) _mm512_sra_epi16(A, _mm_cvtsi32_si128(C))
#else
    NPYV_IMPL_AVX512_SHIFT(shr_u16, srl_epi16)
    NPYV_IMPL_AVX512_SHIFT(shr_s16, sra_epi16)
#endif
#define npyv_shr_u32(A, C) _mm512_srl_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s32(A, C) _mm512_sra_epi32(A, _mm_cvtsi32_si128(C))
#define npyv_shr_u64(A, C) _mm512_srl_epi64(A, _mm_cvtsi32_si128(C))
#define npyv_shr_s64(A, C) _mm512_sra_epi64(A, _mm_cvtsi32_si128(C))

// right by an immediate constant
#ifdef NPY_HAVE_AVX512BW
    #define npyv_shri_u16 _mm512_srli_epi16
    #define npyv_shri_s16 _mm512_srai_epi16
#else
    #define npyv_shri_u16 npyv_shr_u16
    #define npyv_shri_s16 npyv_shr_s16
#endif
#define npyv_shri_u32 _mm512_srli_epi32
#define npyv_shri_s32 _mm512_srai_epi32
#define npyv_shri_u64 _mm512_srli_epi64
#define npyv_shri_s64 _mm512_srai_epi64

/***************************
 * Logical
 ***************************/

// AND
#define npyv_and_u8  _mm512_and_si512
#define npyv_and_s8  _mm512_and_si512
#define npyv_and_u16 _mm512_and_si512
#define npyv_and_s16 _mm512_and_si512
#define npyv_and_u32 _mm512_and_si512
#define npyv_and_s32 _mm512_and_si512
#define npyv_and_u64 _mm512_and_si512
#define npyv_and_s64 _mm512_and_si512
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_and_f32 _mm512_and_ps
    #define npyv_and_f64 _mm512_and_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_and_f32, _mm512_and_si512)
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_and_f64, _mm512_and_si512)
#endif

// OR
#define npyv_or_u8  _mm512_or_si512
#define npyv_or_s8  _mm512_or_si512
#define npyv_or_u16 _mm512_or_si512
#define npyv_or_s16 _mm512_or_si512
#define npyv_or_u32 _mm512_or_si512
#define npyv_or_s32 _mm512_or_si512
#define npyv_or_u64 _mm512_or_si512
#define npyv_or_s64 _mm512_or_si512
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_or_f32 _mm512_or_ps
    #define npyv_or_f64 _mm512_or_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_or_f32, _mm512_or_si512)
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_or_f64, _mm512_or_si512)
#endif

// XOR
#define npyv_xor_u8  _mm512_xor_si512
#define npyv_xor_s8  _mm512_xor_si512
#define npyv_xor_u16 _mm512_xor_si512
#define npyv_xor_s16 _mm512_xor_si512
#define npyv_xor_u32 _mm512_xor_si512
#define npyv_xor_s32 _mm512_xor_si512
#define npyv_xor_u64 _mm512_xor_si512
#define npyv_xor_s64 _mm512_xor_si512
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_xor_f32 _mm512_xor_ps
    #define npyv_xor_f64 _mm512_xor_pd
#else
    NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(npyv_xor_f32, _mm512_xor_si512)
    NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(npyv_xor_f64, _mm512_xor_si512)
#endif

// NOT
#define npyv_not_u8(A) _mm512_xor_si512(A, _mm512_set1_epi32(-1))
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#ifdef NPY_HAVE_AVX512DQ
    #define npyv_not_f32(A) _mm512_xor_ps(A, _mm512_castsi512_ps(_mm512_set1_epi32(-1)))
    #define npyv_not_f64(A) _mm512_xor_pd(A, _mm512_castsi512_pd(_mm512_set1_epi32(-1)))
#else
    #define npyv_not_f32(A) _mm512_castsi512_ps(npyv_not_u32(_mm512_castps_si512(A)))
    #define npyv_not_f64(A) _mm512_castsi512_pd(npyv_not_u64(_mm512_castpd_si512(A)))
#endif

/***************************
 * Comparison
 ***************************/

// int Equal
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cmpeq_u8  _mm512_cmpeq_epu8_mask
    #define npyv_cmpeq_s8  _mm512_cmpeq_epi8_mask
    #define npyv_cmpeq_u16 _mm512_cmpeq_epu16_mask
    #define npyv_cmpeq_s16 _mm512_cmpeq_epi16_mask
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpeq_u8,  _mm256_cmpeq_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpeq_u16, _mm256_cmpeq_epi16)
    #define npyv_cmpeq_s8  npyv_cmpeq_u8
    #define npyv_cmpeq_s16 npyv_cmpeq_u16
#endif
#define npyv_cmpeq_u32 _mm512_cmpeq_epu32_mask
#define npyv_cmpeq_s32 _mm512_cmpeq_epi32_mask
#define npyv_cmpeq_u64 _mm512_cmpeq_epu64_mask
#define npyv_cmpeq_s64 _mm512_cmpeq_epi64_mask

// int not equal
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cmpneq_u8  _mm512_cmpneq_epu8_mask
    #define npyv_cmpneq_s8  _mm512_cmpneq_epi8_mask
    #define npyv_cmpneq_u16 _mm512_cmpneq_epu16_mask
    #define npyv_cmpneq_s16 _mm512_cmpneq_epi16_mask
#else
    #define npyv_cmpneq_u8(A, B) npyv_not_u8(npyv_cmpeq_u8(A, B))
    #define npyv_cmpneq_u16(A, B) npyv_not_u16(npyv_cmpeq_u16(A, B))
    #define npyv_cmpneq_s8  npyv_cmpneq_u8
    #define npyv_cmpneq_s16 npyv_cmpneq_u16
#endif
#define npyv_cmpneq_u32 _mm512_cmpneq_epu32_mask
#define npyv_cmpneq_s32 _mm512_cmpneq_epi32_mask
#define npyv_cmpneq_u64 _mm512_cmpneq_epu64_mask
#define npyv_cmpneq_s64 _mm512_cmpneq_epi64_mask

// greater than
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cmpgt_u8  _mm512_cmpgt_epu8_mask
    #define npyv_cmpgt_s8  _mm512_cmpgt_epi8_mask
    #define npyv_cmpgt_u16 _mm512_cmpgt_epu16_mask
    #define npyv_cmpgt_s16 _mm512_cmpgt_epi16_mask
#else
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpgt_s8,  _mm256_cmpgt_epi8)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv_cmpgt_s16, _mm256_cmpgt_epi16)
    NPY_FINLINE __m512i npyv_cmpgt_u8(__m512i a, __m512i b)
    {
        const __m512i sbit = _mm512_set1_epi32(0x80808080);
        return npyv_cmpgt_s8(_mm512_xor_si512(a, sbit), _mm512_xor_si512(b, sbit));
    }
    NPY_FINLINE __m512i npyv_cmpgt_u16(__m512i a, __m512i b)
    {
        const __m512i sbit = _mm512_set1_epi32(0x80008000);
        return npyv_cmpgt_s16(_mm512_xor_si512(a, sbit), _mm512_xor_si512(b, sbit));
    }
#endif
#define npyv_cmpgt_u32 _mm512_cmpgt_epu32_mask
#define npyv_cmpgt_s32 _mm512_cmpgt_epi32_mask
#define npyv_cmpgt_u64 _mm512_cmpgt_epu64_mask
#define npyv_cmpgt_s64 _mm512_cmpgt_epi64_mask

// greater than or equal
#ifdef NPY_HAVE_AVX512BW
    #define npyv_cmpge_u8  _mm512_cmpge_epu8_mask
    #define npyv_cmpge_s8  _mm512_cmpge_epi8_mask
    #define npyv_cmpge_u16 _mm512_cmpge_epu16_mask
    #define npyv_cmpge_s16 _mm512_cmpge_epi16_mask
#else
    #define npyv_cmpge_u8(A, B)  npyv_not_u8(npyv_cmpgt_u8(B, A))
    #define npyv_cmpge_s8(A, B)  npyv_not_s8(npyv_cmpgt_s8(B, A))
    #define npyv_cmpge_u16(A, B) npyv_not_u16(npyv_cmpgt_u16(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_s16(npyv_cmpgt_s16(B, A))
#endif
#define npyv_cmpge_u32 _mm512_cmpge_epu32_mask
#define npyv_cmpge_s32 _mm512_cmpge_epi32_mask
#define npyv_cmpge_u64 _mm512_cmpge_epu64_mask
#define npyv_cmpge_s64 _mm512_cmpge_epi64_mask

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
#define npyv_cmpeq_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_EQ_OQ)
#define npyv_cmpeq_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_EQ_OQ)
#define npyv_cmpneq_f32(A, B) _mm512_cmp_ps_mask(A, B, _CMP_NEQ_OQ)
#define npyv_cmpneq_f64(A, B) _mm512_cmp_pd_mask(A, B, _CMP_NEQ_OQ)
#define npyv_cmplt_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_LT_OQ)
#define npyv_cmplt_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_LT_OQ)
#define npyv_cmple_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_LE_OQ)
#define npyv_cmple_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_LE_OQ)
#define npyv_cmpgt_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_GT_OQ)
#define npyv_cmpgt_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_GT_OQ)
#define npyv_cmpge_f32(A, B)  _mm512_cmp_ps_mask(A, B, _CMP_GE_OQ)
#define npyv_cmpge_f64(A, B)  _mm512_cmp_pd_mask(A, B, _CMP_GE_OQ)

#endif // _NPY_SIMD_AVX512_OPERATORS_H
