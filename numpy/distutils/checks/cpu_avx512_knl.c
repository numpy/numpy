#include <immintrin.h>

int main(int argc, char **argv)
{
    int base[128];
    __m512d ad = _mm512_loadu_pd((const __m512d*)argv[argc-1]);
    /* ER */
    __m512i a = _mm512_castpd_si512(_mm512_exp2a23_pd(ad));
    /* PF */
    _mm512_mask_prefetch_i64scatter_pd(base, _mm512_cmpeq_epi64_mask(a, a), a, 1, _MM_HINT_T1);
    return base[0];
}
