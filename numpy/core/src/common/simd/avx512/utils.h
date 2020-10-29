#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_UTILS_H
#define _NPY_SIMD_AVX512_UTILS_H

NPY_FINLINE __m512i npyv512_shuffle_odd(__m512i a)
{
    const __m512i odd_perm = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    return _mm512_permutexvar_epi64(odd_perm, a);
}

NPY_FINLINE __m512i npyv512_shuffle_odd32(__m512i a)
{
    const __m512i odd_perm = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    return _mm512_permutexvar_epi32(odd_perm, a);
}

#define npyv512_lower_si256 _mm512_castsi512_si256
#define npyv512_lower_ps256 _mm512_castps512_ps256
#define npyv512_lower_pd256 _mm512_castpd512_pd256

#define npyv512_higher_si256(A) _mm512_extracti64x4_epi64(A, 1)
#define npyv512_higher_pd256(A) _mm512_extractf64x4_pd(A, 1)

#ifdef NPY_HAVE_AVX512DQ
    #define npyv512_higher_ps256(A) _mm512_extractf32x8_ps(A, 1)
#else
    #define npyv512_higher_ps256(A) \
        _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(A), 1))
#endif

#define npyv512_combine_si256(A, B) _mm512_inserti64x4(_mm512_castsi256_si512(A), B, 1)
#define npyv512_combine_pd256(A, B) _mm512_insertf64x4(_mm512_castpd256_pd512(A), B, 1)

#ifdef NPY_HAVE_AVX512DQ
    #define npyv512_combine_ps256(A, B) _mm512_insertf32x8(_mm512_castps256_ps512(A), B, 1)
#else
    #define npyv512_combine_ps256(A, B) \
        _mm512_castsi512_ps(npyv512_combine_si256(_mm512_castps_si512(A), _mm512_castps_si512(B)))
#endif

#define NPYV_IMPL_AVX512_FROM_AVX2_1ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512i FN_NAME(__m512i a)               \
    {                                                    \
        __m256i l_a = npyv512_lower_si256(a);            \
        __m256i h_a = npyv512_higher_si256(a);           \
        __m256i l_a_i = INTRIN(l_a);                     \
        __m256i h_a_i = INTRIN(h_a);                     \
        return npyv512_combine_si256(l_a_i, h_a_i);      \
    }

#define NPYV_IMPL_AVX512_FROM_AVX2_2ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512i FN_NAME(__m512i a, __m512i b)    \
    {                                                    \
        __m256i l_a  = npyv512_lower_si256(a);           \
        __m256i h_a  = npyv512_higher_si256(a);          \
        __m256i l_b  = npyv512_lower_si256(b);           \
        __m256i h_b  = npyv512_higher_si256(b);          \
        __m256i l_a_i = INTRIN(l_a, l_b);                \
        __m256i h_a_i = INTRIN(h_a, h_b);                \
        return npyv512_combine_si256(l_a_i, h_a_i);      \
    }

#define NPYV_IMPL_AVX512_FROM_SI512_PS_2ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512 FN_NAME(__m512 a, __m512 b)           \
    {                                                        \
        return _mm512_castsi512_ps(INTRIN(                   \
            _mm512_castps_si512(a), _mm512_castps_si512(b)   \
        ));                                                  \
    }

#define NPYV_IMPL_AVX512_FROM_SI512_PD_2ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512d FN_NAME(__m512d a, __m512d b)        \
    {                                                        \
        return _mm512_castsi512_pd(INTRIN(                   \
            _mm512_castpd_si512(a), _mm512_castpd_si512(b)   \
        ));                                                  \
    }

/***************************
 * Emulate Byte And Word
 ***************************/
#ifndef NPY_HAVE_AVX512BW
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_packs_epi16,  _mm256_packs_epi16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_packs_epi32,  _mm256_packs_epi32)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_packus_epi16, _mm256_packus_epi16)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_packus_epi32, _mm256_packus_epi32)
    NPYV_IMPL_AVX512_FROM_AVX2_2ARG(npyv512_shuffle_epi8, _mm256_shuffle_epi8)
#else
    #define npyv512_packs_epi16 _mm512_packs_epi16
    #define npyv512_packs_epi32 _mm512_packs_epi32
    #define npyv512_packus_epi16 _mm512_packus_epi16
    #define npyv512_packus_epi32 _mm512_packus_epi32
    #define npyv512_shuffle_epi8 _mm512_shuffle_epi8
#endif

#endif // _NPY_SIMD_AVX512_UTILS_H
