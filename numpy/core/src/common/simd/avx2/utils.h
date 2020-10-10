#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_UTILS_H
#define _NPY_SIMD_AVX2_UTILS_H

#define npyv256_shuffle_odd(A)    _mm256_permute4x64_epi64(A, _MM_SHUFFLE(3, 1, 2, 0))
#define npyv256_shuffle_odd_ps(A) _mm256_castsi256_ps(npyv256_shuffle_odd(_mm256_castps_si256(A)))
#define npyv256_shuffle_odd_pd(A) _mm256_permute4x64_pd(A, _MM_SHUFFLE(3, 1, 2, 0))

NPY_FINLINE __m256i npyv256_mul_u8(__m256i a, __m256i b)
{
    const __m256i mask = _mm256_set1_epi32(0xFF00FF00);
    __m256i even = _mm256_mullo_epi16(a, b);
    __m256i odd  = _mm256_mullo_epi16(_mm256_srai_epi16(a, 8), _mm256_srai_epi16(b, 8));
            odd  = _mm256_slli_epi16(odd, 8);
    return _mm256_blendv_epi8(even, odd, mask);
}

#endif // _NPY_SIMD_AVX2_UTILS_H
