#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_UTILS_H
#define _NPY_SIMD_AVX512_UTILS_H

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
        __m256i l_a  = npyv512_lower_si256(a);           \
        __m256i h_a  = npyv512_higher_si256(a);          \
        l_a = INTRIN(l_a);                               \
        h_a = INTRIN(h_a);                               \
        return npyv512_combine_si256(l_a, h_a);          \
    }

#define NPYV_IMPL_AVX512_FROM_AVX2_2ARG(FN_NAME, INTRIN) \
    NPY_FINLINE __m512i FN_NAME(__m512i a, __m512i b)    \
    {                                                    \
        __m256i l_a  = npyv512_lower_si256(a);           \
        __m256i h_a  = npyv512_higher_si256(a);          \
        __m256i l_b  = npyv512_lower_si256(b);           \
        __m256i h_b  = npyv512_higher_si256(b);          \
        l_a = INTRIN(l_a, l_b);                          \
        h_a = INTRIN(h_a, h_b);                          \
        return npyv512_combine_si256(l_a, h_a);          \
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

#endif // _NPY_SIMD_AVX512_UTILS_H
