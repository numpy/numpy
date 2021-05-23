#ifdef _MSC_VER
    #include <nmmintrin.h>
#else
    #include <popcntintrin.h>
#endif

#include <stdlib.h>

int main(void)
{
    long long a = 0;
    int b;
    
    a = random();
    b = random();
      
#ifdef _MSC_VER
    #ifdef _M_X64
    a = _mm_popcnt_u64(a);
    #endif
    b = _mm_popcnt_u32(b);
#else
    #ifdef __x86_64__
    a = __builtin_popcountll(a);
    #endif
    b = __builtin_popcount(b);
#endif
    return (int)a + b;
}
