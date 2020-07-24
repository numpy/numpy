#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX512_MEMORY_H
#define _NPY_SIMD_AVX512_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
#if defined(__GNUC__)
    // GCC expect pointer argument type to be `void*` instead of `const void *`,
    // which cause a massive warning.
    #define npyv__loads(PTR) _mm512_stream_load_si512((__m512i*)(PTR))
#else
    #define npyv__loads(PTR) _mm512_stream_load_si512((const __m512i*)(PTR))
#endif
#if defined(_MSC_VER) && defined(_M_IX86)
    // workaround msvc(32bit) overflow bug, reported at
    // https://developercommunity.visualstudio.com/content/problem/911872/u.html
    NPY_FINLINE __m512i npyv__loadl(const __m256i *ptr)
    {
        __m256i a = _mm256_loadu_si256(ptr);
        return _mm512_inserti64x4(_mm512_castsi256_si512(a), a, 0);
    }
#else
    #define npyv__loadl(PTR) \
        _mm512_castsi256_si512(_mm256_loadu_si256(PTR))
#endif
#define NPYV_IMPL_AVX512_MEM_INT(CTYPE, SFX)                                 \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm512_loadu_si512((const __m512i*)ptr); }                      \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm512_load_si512((const __m512i*)ptr); }                       \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return npyv__loads(ptr); }                                             \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return npyv__loadl((const __m256i *)ptr); }                            \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm512_storeu_si512((__m512i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm512_store_si512((__m512i*)ptr, vec); }                              \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm512_stream_si512((__m512i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_storeu_si256((__m256i*)ptr, npyv512_lower_si256(vec)); }        \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_storeu_si256((__m256i*)(ptr), npyv512_higher_si256(vec)); }

NPYV_IMPL_AVX512_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_AVX512_MEM_INT(npy_int8,   s8)
NPYV_IMPL_AVX512_MEM_INT(npy_uint16, u16)
NPYV_IMPL_AVX512_MEM_INT(npy_int16,  s16)
NPYV_IMPL_AVX512_MEM_INT(npy_uint32, u32)
NPYV_IMPL_AVX512_MEM_INT(npy_int32,  s32)
NPYV_IMPL_AVX512_MEM_INT(npy_uint64, u64)
NPYV_IMPL_AVX512_MEM_INT(npy_int64,  s64)

// unaligned load
#define npyv_load_f32(PTR) _mm512_loadu_ps((const __m512*)(PTR))
#define npyv_load_f64(PTR) _mm512_loadu_pd((const __m512d*)(PTR))
// aligned load
#define npyv_loada_f32(PTR) _mm512_load_ps((const __m512*)(PTR))
#define npyv_loada_f64(PTR) _mm512_load_pd((const __m512d*)(PTR))
// load lower part
#if defined(_MSC_VER) && defined(_M_IX86)
    #define npyv_loadl_f32(PTR) _mm512_castsi512_ps(npyv__loadl((const __m256i *)(PTR)))
    #define npyv_loadl_f64(PTR) _mm512_castsi512_pd(npyv__loadl((const __m256i *)(PTR)))
#else
    #define npyv_loadl_f32(PTR) _mm512_castps256_ps512(_mm256_loadu_ps(PTR))
    #define npyv_loadl_f64(PTR) _mm512_castpd256_pd512(_mm256_loadu_pd(PTR))
#endif
// stream load
#define npyv_loads_f32(PTR) _mm512_castsi512_ps(npyv__loads(PTR))
#define npyv_loads_f64(PTR) _mm512_castsi512_pd(npyv__loads(PTR))
// unaligned store
#define npyv_store_f32 _mm512_storeu_ps
#define npyv_store_f64 _mm512_storeu_pd
// aligned store
#define npyv_storea_f32 _mm512_store_ps
#define npyv_storea_f64 _mm512_store_pd
// stream store
#define npyv_stores_f32 _mm512_stream_ps
#define npyv_stores_f64 _mm512_stream_pd
// store lower part
#define npyv_storel_f32(PTR, VEC) _mm256_storeu_ps(PTR, npyv512_lower_ps256(VEC))
#define npyv_storel_f64(PTR, VEC) _mm256_storeu_pd(PTR, npyv512_lower_pd256(VEC))
// store higher part
#define npyv_storeh_f32(PTR, VEC) _mm256_storeu_ps(PTR, npyv512_higher_ps256(VEC))
#define npyv_storeh_f64(PTR, VEC) _mm256_storeu_pd(PTR, npyv512_higher_pd256(VEC))

// non-contiguous load
//// 8
NPY_FINLINE npyv_u8 npyv_loadn_u8(const npy_uint8 *ptr, int stride)
{
    const __m512i steps = npyv_set_u32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    __m512i a = _mm512_i32gather_epi32(idx, (const void*)ptr, 1);
    __m512i b = _mm512_i32gather_epi32(idx, (const void*)(ptr + stride*16), 1);
    __m512i c = _mm512_i32gather_epi32(idx, (const void*)(ptr + stride*32), 1);
    __m512i d = _mm512_i32gather_epi32(idx, (const void*)((ptr-3/*overflow guard*/)+stride*48), 1);
#ifdef NPY_HAVE_AVX512BW
    const __m512i cut32 = _mm512_set1_epi32(0xFF);
    a = _mm512_and_si512(a, cut32);
    b = _mm512_and_si512(b, cut32);
    c = _mm512_and_si512(c, cut32);
    d = _mm512_srli_epi32(d, 24);
    a = _mm512_packus_epi32(a, b);
    c = _mm512_packus_epi32(c, d);
    return npyv512_shuffle_odd32(_mm512_packus_epi16(a, c));
#else
    __m128i af = _mm512_cvtepi32_epi8(a);
    __m128i bf = _mm512_cvtepi32_epi8(b);
    __m128i cf = _mm512_cvtepi32_epi8(c);
    __m128i df = _mm512_cvtepi32_epi8(_mm512_srli_epi32(d, 24));
    return npyv512_combine_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(af), bf, 1),
        _mm256_inserti128_si256(_mm256_castsi128_si256(cf), df, 1)
    );
#endif // !NPY_HAVE_AVX512BW
}
NPY_FINLINE npyv_s8 npyv_loadn_s8(const npy_int8 *ptr, int stride)
{ return npyv_loadn_u8((const npy_uint8*)ptr, stride); }
//// 16
NPY_FINLINE npyv_u16 npyv_loadn_u16(const npy_uint16 *ptr, int stride)
{
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    __m512i a = _mm512_i32gather_epi32(idx, (const void*)ptr, 2);
    __m512i b = _mm512_i32gather_epi32(idx, (const void*)((ptr-1/*overflow guard*/)+stride*16), 2);
#ifdef NPY_HAVE_AVX512BW
    const __m512i perm = npyv_set_u16(
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63
    );
    return _mm512_permutex2var_epi16(a, perm, b);
#else
    __m256i af = _mm512_cvtepi32_epi16(a);
    __m256i bf = _mm512_cvtepi32_epi16(_mm512_srli_epi32(b, 16));
    return npyv512_combine_si256(af, bf);
#endif
}
NPY_FINLINE npyv_s16 npyv_loadn_s16(const npy_int16 *ptr, int stride)
{ return npyv_loadn_u16((const npy_uint16*)ptr, stride); }
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, int stride)
{
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    return _mm512_i32gather_epi32(idx, (const int*)ptr, 4);
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, int stride)
{ return npyv_loadn_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, int stride)
{ return _mm512_castsi512_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, int stride)
{
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i idx = _mm256_mullo_epi32(_mm256_set1_epi32(stride), steps);
    return _mm512_i32gather_epi64(idx, (const void*)ptr, 8);
}
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, int stride)
{ return npyv_loadn_u64((const npy_uint64*)ptr, stride); }
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, int stride)
{ return _mm512_castsi512_pd(npyv_loadn_u64((const npy_uint64*)ptr, stride)); }

// non-contiguous store
//// 8
NPY_FINLINE void npyv_storen_u8(npy_uint8 *ptr, int stride, npyv_u8 a)
{
    // GIT:WARN Buggy Buggy, need a fix
    // TODO: overflow guard cause small strides overlaping (-3/-2/-1/1/2/3) between [45:48]
    const __m512i steps = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    __m512i m0 = _mm512_i32gather_epi32(idx, (const void*)ptr, 1);
    __m512i m1 = _mm512_i32gather_epi32(idx, (const void*)(ptr + stride*16), 1);
    __m512i m2 = _mm512_i32gather_epi32(idx, (const void*)(ptr + stride*32), 1);
    __m512i m3 = _mm512_i32gather_epi32(idx, (const void*)((ptr-3/*overflow guard*/)+stride*48), 1);
#if 0 // def NPY_HAVE_AVX512VBMI
    // NOTE: experimental
    const __m512i perm = npyv_set_u8(
        64, 1,  2,  3,  65, 5,  6,  7,  66,  9, 10, 11, 67, 13, 14, 15,
        68, 17, 18, 19, 69, 21, 22, 23, 70, 25, 26, 27, 71, 29, 30, 31,
        72, 33, 34, 35, 73, 37, 38, 39, 74, 41, 42, 43, 75, 45, 46, 47,
        76, 49, 50, 51, 77, 53, 54, 55, 78, 57, 58, 59, 79, 61, 62, 63
    );
    const __m512i perm_ofg = _mm512_ror_epi32(perm, 8);
    __m512i a1 = _mm512_castsi128_si512(_mm512_extracti64x2_epi64(a, 1));
    __m512i a2 = _mm512_castsi128_si512(_mm512_extracti64x2_epi64(a, 2));
    __m512i a3 = _mm512_castsi128_si512(_mm512_extracti64x2_epi64(a, 3));
    __m512i s0 = _mm512_permutex2var_epi8(m0, perm, a);
    __m512i s1 = _mm512_permutex2var_epi8(m1, perm, a1);
    __m512i s2 = _mm512_permutex2var_epi8(m2, perm, a2);
    __m512i s3 = _mm512_permutex2var_epi8(_mm512_rol_epi32(m3, 8), perm_ofg, a3);
#else
    #if 0 // def NPY_HAVE_AVX512DQ
        __m512i a0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(a));
        __m512i a1 = _mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(a, 1));
        __m512i a2 = _mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(a, 2));
        __m512i a3 = _mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(a, 3));
                a3 = _mm512_slli_epi32(a3, 24);
    #else
        __m256i low  = _mm512_extracti64x4_epi64(a, 0);
        __m256i high = _mm512_extracti64x4_epi64(a, 1);
        __m512i a0 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(low));
        __m512i a1 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(low, 1));
        __m512i a2 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(high));
        __m512i a3 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(high, 1));
                a3 = _mm512_slli_epi32(a3, 24);
    #endif // NPY_HAVE_AVX512DQ
    #ifdef NPY_HAVE_AVX512BW
        __m512i s0 = _mm512_mask_blend_epi8(0x1111111111111111, m0, a0);
        __m512i s1 = _mm512_mask_blend_epi8(0x1111111111111111, m1, a1);
        __m512i s2 = _mm512_mask_blend_epi8(0x1111111111111111, m2, a2);
        __m512i s3 = _mm512_mask_blend_epi8(0x8888888888888888, m3, a3);
    #else
        const __m512i maskl = _mm512_set1_epi32(0x000000FF);
        const __m512i maskh = _mm512_set1_epi32(0xFF000000);
        __m512i s0 = npyv_select_u8(maskl, a0, m0);
        __m512i s1 = npyv_select_u8(maskl, a1, m1);
        __m512i s2 = npyv_select_u8(maskl, a2, m2);
        __m512i s3 = npyv_select_u8(maskh, a3, m3);
    #endif // NPY_HAVE_AVX512BW
#endif // AVX512VBMI
    _mm512_i32scatter_epi32((int*)ptr, idx, s0, 1);
    _mm512_i32scatter_epi32((int*)(ptr + stride*16), idx, s1, 1);
    _mm512_i32scatter_epi32((int*)((ptr-3/*overflow guard*/)+ stride*48), idx, s3, 1);
    _mm512_i32scatter_epi32((int*)(ptr + stride*32), idx, s2, 1);
}
NPY_FINLINE void npyv_storen_s8(npy_int8 *ptr, int stride, npyv_s8 a)
{ npyv_storen_u8((npy_uint8*)ptr, stride, a); }
//// 16
NPY_FINLINE void npyv_storen_u16(npy_uint16 *ptr, int stride, npyv_u16 a)
{
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    __m512i m0 = _mm512_i32gather_epi32(idx, (const void*)ptr, 2);
    __m512i m1 = _mm512_i32gather_epi32(idx, (const void*)((ptr-1/*overflow guard*/)+stride*16), 2);

    __m512i a0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a));
    __m512i a1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a, 1));
            a1 = _mm512_slli_epi32(a1, 16);
    #ifdef NPY_HAVE_AVX512BW
        __m512i s0 = _mm512_mask_blend_epi16(0x55555555, m0, a0);
        __m512i s1 = _mm512_mask_blend_epi16(0xAAAAAAAA, m1, a1);
    #else
        const __m512i mask = _mm512_set1_epi32(0x0000FFFF);
        __m512i s0 = npyv_select_u16(mask, a0, m0);
        __m512i s1 = npyv_select_u16(mask, m1, a1);
    #endif // NPY_HAVE_AVX512BW
    _mm512_i32scatter_epi32((int*)ptr, idx, s0, 2);
    _mm512_i32scatter_epi32((int*)((ptr-1/*overflow guard*/)+stride*16), idx, s1, 2);
}
NPY_FINLINE void npyv_storen_s16(npy_int16 *ptr, int stride, npyv_s16 a)
{ npyv_storen_u16((npy_uint16*)ptr, stride, a); }
//// 32
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, int stride, npyv_u32 a)
{
    const __m512i steps = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32(stride));
    _mm512_i32scatter_epi32((int*)ptr, idx, a, 4);
}
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, int stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, int stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, _mm512_castps_si512(a)); }
//// 64
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, int stride, npyv_u64 a)
{
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i idx = _mm256_mullo_epi32(_mm256_set1_epi32(stride), steps);
    _mm512_i32scatter_epi64((void*)ptr, idx, a, 8);
}
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, int stride, npyv_s64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f64(double *ptr, int stride, npyv_f64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, _mm512_castpd_si512(a)); }

#endif // _NPY_SIMD_AVX512_MEMORY_H
