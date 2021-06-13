#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    /* IFMA */
    a = _mm512_madd52hi_epu64(a, a, _mm512_setzero_si512());
    /* VMBI */
    a = _mm512_permutex2var_epi8(a, _mm512_setzero_si512(), a);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
