#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_UTILS_H
#define _NPY_SIMD_SSE_UTILS_H

#if !defined(__x86_64__) && !defined(_M_X64)
NPY_FINLINE npy_int64 npyv128_cvtsi128_si64(__m128i a)
{
    npy_int64 NPY_DECL_ALIGNED(16) idx[2];
    _mm_store_si128((__m128i *)idx, a);
    return idx[0];
}
#else
    #define npyv128_cvtsi128_si64 _mm_cvtsi128_si64
#endif

#endif // _NPY_SIMD_SSE_UTILS_H
