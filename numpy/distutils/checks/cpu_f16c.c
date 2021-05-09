#include <emmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    __m128 a  = _mm_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-1]));
    __m256 a8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-2]));
    return (int)(_mm_cvtss_f32(a) + _mm_cvtss_f32(_mm256_castps256_ps128(a8)));
}
