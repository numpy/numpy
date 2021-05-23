#ifdef _MSC_VER
    #include <nmmintrin.h>
#else
    #include <popcntintrin.h>
#endif

int main(int argc, char **argv)
{
    // To make sure popcnt instructions are generated
    // and been tested against the assembler
    unsigned long long a = *((unsigned long long*)argv[argc-1]);
    unsigned int b = *((unsigned int*)argv[argc-2]);

#if defined(_M_X64) || defined(__x86_64__)    
    a = _mm_popcnt_u64(a);
#endif
    b = _mm_popcnt_u32(b);
    return (int)a + b;
}
