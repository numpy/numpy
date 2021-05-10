#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i aa = _mm512_abs_epi32(_mm512_loadu_si512((const __m512i*)argv[argc-1]));
    /* VL */
    __m256i a = _mm256_abs_epi64(_mm512_extracti64x4_epi64(aa, 1));
    /* DQ */
    __m512i b = _mm512_broadcast_i32x8(a);
    /* BW */
    b = _mm512_abs_epi16(b);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(b));
}
