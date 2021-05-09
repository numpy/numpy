#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256 a = _mm256_add_ps(_mm256_loadu_ps((const float*)argv[argc-1]), _mm256_loadu_ps((const float*)argv[1]));
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
