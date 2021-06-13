#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_lzcnt_epi32(_mm512_loadu_si512((const __m512i*)argv[argc-1]));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
