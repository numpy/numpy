#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #if !defined(__AVX512F__) || !defined(__AVX512CD__) || !defined(__AVX512VL__) || \
        !defined(__AVX512BW__) || !defined(__AVX512DQ__)
        #error HOST/ARCH does not support x86_v4
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // to prevent optimization
    int seed = (int)argv[argc-1][0];
    volatile int result = 0;

    // AVX512F tests (Foundation)
    __m512 avx512f_a = _mm512_set1_ps((float)seed);
    __m512 avx512f_b = _mm512_set1_ps(2.0f);
    __m512 avx512f_c = _mm512_add_ps(avx512f_a, avx512f_b);
    float avx512f_result = _mm512_cvtss_f32(avx512f_c);
    result += (int)avx512f_result;

    // Test AVX512F mask operations
    __mmask16 k1 = _mm512_cmpeq_ps_mask(avx512f_a, avx512f_b);
    __m512 masked_result = _mm512_mask_add_ps(avx512f_a, k1, avx512f_b, avx512f_c);
    result += _mm512_mask2int(k1);

    // AVX512CD tests (Conflict Detection)
    __m512i avx512cd_a = _mm512_set1_epi32(seed);
    __m512i avx512cd_b = _mm512_conflict_epi32(avx512cd_a);
    result += _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(avx512cd_b, 0));

    __m512i avx512cd_lzcnt = _mm512_lzcnt_epi32(avx512cd_a);
    result += _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(avx512cd_lzcnt, 0));

    // AVX512VL tests (Vector Length Extensions - 128/256-bit vectors with AVX512 features)
    __m256 avx512vl_a = _mm256_set1_ps((float)seed);
    __m256 avx512vl_b = _mm256_set1_ps(2.0f);
    __mmask8 k2 = _mm256_cmp_ps_mask(avx512vl_a, avx512vl_b, _CMP_EQ_OQ);
    __m256 avx512vl_c = _mm256_mask_add_ps(avx512vl_a, k2, avx512vl_a, avx512vl_b);
    result += (int)_mm256_cvtss_f32(avx512vl_c);

    __m128 avx512vl_sm_a = _mm_set1_ps((float)seed);
    __m128 avx512vl_sm_b = _mm_set1_ps(2.0f);
    __mmask8 k3 = _mm_cmp_ps_mask(avx512vl_sm_a, avx512vl_sm_b, _CMP_EQ_OQ);
    __m128 avx512vl_sm_c = _mm_mask_add_ps(avx512vl_sm_a, k3, avx512vl_sm_a, avx512vl_sm_b);
    result += (int)_mm_cvtss_f32(avx512vl_sm_c);

    // AVX512BW tests (Byte and Word)
    __m512i avx512bw_a = _mm512_set1_epi16((short)seed);
    __m512i avx512bw_b = _mm512_set1_epi16(2);
    __mmask32 k4 = _mm512_cmpeq_epi16_mask(avx512bw_a, avx512bw_b);
    __m512i avx512bw_c = _mm512_mask_add_epi16(avx512bw_a, k4, avx512bw_a, avx512bw_b);
    result += _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(avx512bw_c, 0));

    // Test byte operations
    __m512i avx512bw_bytes_a = _mm512_set1_epi8((char)seed);
    __m512i avx512bw_bytes_b = _mm512_set1_epi8(2);
    __mmask64 k5 = _mm512_cmpeq_epi8_mask(avx512bw_bytes_a, avx512bw_bytes_b);
    result += (k5 & 1);

    // AVX512DQ tests (Doubleword and Quadword)
    __m512d avx512dq_a = _mm512_set1_pd((double)seed);
    __m512d avx512dq_b = _mm512_set1_pd(2.0);
    __mmask8 k6 = _mm512_cmpeq_pd_mask(avx512dq_a, avx512dq_b);
    __m512d avx512dq_c = _mm512_mask_add_pd(avx512dq_a, k6, avx512dq_a, avx512dq_b);
    double avx512dq_result = _mm512_cvtsd_f64(avx512dq_c);
    result += (int)avx512dq_result;

    // Test integer to/from floating point conversion
    __m512i avx512dq_back = _mm512_cvtps_epi32(masked_result);
    result += _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(avx512dq_back, 0));

    // Test 64-bit integer operations
    __m512i avx512dq_i64_a = _mm512_set1_epi64(seed);
    __m512i avx512dq_i64_b = _mm512_set1_epi64(2);
    __m512i avx512dq_i64_c = _mm512_add_epi64(avx512dq_i64_a, avx512dq_i64_b);
    result += _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(avx512dq_i64_c, 0));

    return result;
}
