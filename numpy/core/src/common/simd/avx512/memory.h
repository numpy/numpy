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

#endif // _NPY_SIMD_AVX512_MEMORY_H
