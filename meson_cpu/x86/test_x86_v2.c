#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #if !defined(__SSE__) || !defined(__SSE2__) || !defined(__SSE3__) || \
        !defined(__SSSE3__) || !defined(__SSE4_1__) || !defined(__SSE4_2__) || !defined(__POPCNT__)
        #error HOST/ARCH does not support x86_v2
    #endif
#endif

#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>  // SSE4.1
#ifdef _MSC_VER
    #include <nmmintrin.h>  // SSE4.2 and POPCNT for MSVC
#else
    #include <nmmintrin.h>  // SSE4.2
    #include <popcntintrin.h>  // POPCNT
#endif

int main(int argc, char **argv)
{
    // to prevent optimization
    int seed = (int)argv[argc-1][0];
    volatile int result = 0;

    // SSE test
    __m128 a = _mm_set1_ps((float)seed);
    __m128 b = _mm_set1_ps(2.0f);
    __m128 c = _mm_add_ps(a, b);
    result += (int)_mm_cvtss_f32(c);

    // SSE2 test
    __m128i ai = _mm_set1_epi32(seed);
    __m128i bi = _mm_set1_epi32(2);
    __m128i ci = _mm_add_epi32(ai, bi);
    result += _mm_cvtsi128_si32(ci);

    // SSE3 test
    __m128 d = _mm_movehdup_ps(a);
    result += (int)_mm_cvtss_f32(d);

    // SSSE3 test
    __m128i di = _mm_abs_epi16(_mm_set1_epi16((short)seed));
    result += _mm_cvtsi128_si32(di);

    // SSE4.1 test
    __m128i ei = _mm_max_epi32(ai, bi);
    result += _mm_cvtsi128_si32(ei);

    // SSE4.2 test
    __m128i str1 = _mm_set1_epi8((char)seed);
    __m128i str2 = _mm_set1_epi8((char)(seed + 1));
    int res4_2 = _mm_cmpestra(str1, 4, str2, 4, 0);
    result += res4_2;

    // POPCNT test
    unsigned int test_val = (unsigned int)seed | 0x01234567;
    int pcnt = _mm_popcnt_u32(test_val);
    result += pcnt;

    return result;
}
