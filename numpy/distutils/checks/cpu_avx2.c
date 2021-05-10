#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256i a = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i*)argv[argc-1]));
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
