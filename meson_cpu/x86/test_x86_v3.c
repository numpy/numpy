#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #if !defined(__AVX__) || !defined(__AVX2__) || !defined(__FMA__) || \
        !defined(__BMI__) || !defined(__BMI2__) || !defined(__LZCNT__) || !defined(__F16C__)
        #error HOST/ARCH does not support x86_v3
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // to prevent optimization
    int seed = (int)argv[argc-1][0];
    volatile int result = 0;

    // AVX test
    __m256 avx_a = _mm256_set1_ps((float)seed);
    __m256 avx_b = _mm256_set1_ps(2.0f);
    __m256 avx_c = _mm256_add_ps(avx_a, avx_b);
    float avx_result = _mm256_cvtss_f32(avx_c);
    result += (int)avx_result;

    // AVX2 test
    __m256i avx2_a = _mm256_set1_epi32(seed);
    __m256i avx2_b = _mm256_set1_epi32(2);
    __m256i avx2_c = _mm256_add_epi32(avx2_a, avx2_b);
    result += _mm256_extract_epi32(avx2_c, 0);

    // FMA test
    __m256 fma_a = _mm256_set1_ps((float)seed);
    __m256 fma_b = _mm256_set1_ps(2.0f);
    __m256 fma_c = _mm256_set1_ps(3.0f);
    __m256 fma_result = _mm256_fmadd_ps(fma_a, fma_b, fma_c);
    result += (int)_mm256_cvtss_f32(fma_result);

    // BMI1 tests
    unsigned int bmi1_src = (unsigned int)seed;
    unsigned int tzcnt_result = _tzcnt_u32(bmi1_src);
    result += tzcnt_result;

    // BMI2 tests
    unsigned int bzhi_result = _bzhi_u32(bmi1_src, 17);
    result += (int)bzhi_result;

    unsigned int pdep_result = _pdep_u32(bmi1_src, 0x10101010);
    result += pdep_result;

    // LZCNT test
    unsigned int lzcnt_result = _lzcnt_u32(bmi1_src);
    result += lzcnt_result;

    // F16C tests
    __m128 f16c_src = _mm_set1_ps((float)seed);
    __m128i f16c_half = _mm_cvtps_ph(f16c_src, 0);
    __m128 f16c_restored = _mm_cvtph_ps(f16c_half);
    result += (int)_mm_cvtss_f32(f16c_restored);

    return result;
}
