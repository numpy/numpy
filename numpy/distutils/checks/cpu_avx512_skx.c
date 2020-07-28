#include <immintrin.h>

int main(void)
{
    // VL
    __m256i a = _mm256_abs_epi64(_mm256_setzero_si256());
    // DQ
    __m512i b = _mm512_broadcast_i32x8(a);
    // BW
    // Test mask operations
    // https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
    __mmask64 m64 = _mm512_cmpeq_epi8_mask(_mm512_set1_epi8((char)1), _mm512_set1_epi8((char)1));
    m64 = _kor_mask64(m64, m64);
    m64 = _kxor_mask64(m64, m64);
    b = _mm512_mask_blend_epi8(m64, b, _mm512_abs_epi16(b));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(b));
}
