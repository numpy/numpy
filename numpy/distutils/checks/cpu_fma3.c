#include <xmmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256 a = _mm256_loadu_ps((const float*)argv[argc-1]);
           a = _mm256_fmadd_ps(a, a, a);
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
