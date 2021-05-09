#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    __m512 b = _mm512_loadu_ps((const __m512*)argv[argc-2]);

    /* 4FMAPS */
    b = _mm512_4fmadd_ps(b, b, b, b, b, NULL);
    /* 4VNNIW */
    a = _mm512_4dpwssd_epi32(a, a, a, a, a, NULL);
    /* VPOPCNTDQ */
    a = _mm512_popcnt_epi64(a);

    a = _mm512_add_epi32(a, _mm512_castps_si512(b));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
