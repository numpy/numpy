#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * Unlike GCC and CLANG, Intel Compiler exposes all supported intrinsics,
     * whether or not the build options for those features are specified.
     * Therefore, we must test #definitions of CPU features when option native/host
     * is enabled via `--cpu-baseline` or through env var `CFLAGS` otherwise
     * the test will be broken and leads to enable all possible features.
     */
    #ifndef __AVX512FP16__
        #error "HOST/ARCH doesn't support AVX512FP16"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256h a = _mm256_set1_ph(2.0);
    __m512 b = _mm512_cvtxph_ps(a);
    __m512i c = _mm512_cvt_roundps_epi32(b, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    _mm512_storeu_epi32((void*)argv[argc-1], c);
    return 0;
}
