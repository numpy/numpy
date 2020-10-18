#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_MEMORY_H
#define _NPY_SIMD_AVX2_MEMORY_H

/***************************
 * load/store
 ***************************/
#define NPYV_IMPL_AVX2_MEM_INT(CTYPE, SFX)                                   \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm256_loadu_si256((const __m256i*)ptr); }                      \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm256_load_si256((const __m256i*)ptr); }                       \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return _mm256_stream_load_si256((const __m256i*)ptr); }                \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)ptr)); } \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm256_storeu_si256((__m256i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_store_si256((__m256i*)ptr, vec); }                              \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm256_stream_si256((__m256i*)ptr, vec); }                             \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storeu_si128((__m128i*)(ptr), _mm256_castsi256_si128(vec)); }      \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storeu_si128((__m128i*)(ptr), _mm256_extracti128_si256(vec, 1)); }

NPYV_IMPL_AVX2_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_AVX2_MEM_INT(npy_int8,   s8)
NPYV_IMPL_AVX2_MEM_INT(npy_uint16, u16)
NPYV_IMPL_AVX2_MEM_INT(npy_int16,  s16)
NPYV_IMPL_AVX2_MEM_INT(npy_uint32, u32)
NPYV_IMPL_AVX2_MEM_INT(npy_int32,  s32)
NPYV_IMPL_AVX2_MEM_INT(npy_uint64, u64)
NPYV_IMPL_AVX2_MEM_INT(npy_int64,  s64)

// unaligned load
#define npyv_load_f32 _mm256_loadu_ps
#define npyv_load_f64 _mm256_loadu_pd
// aligned load
#define npyv_loada_f32 _mm256_load_ps
#define npyv_loada_f64 _mm256_load_pd
// stream load
#define npyv_loads_f32(PTR) \
    _mm256_castsi256_ps(_mm256_stream_load_si256((const __m256i*)(PTR)))
#define npyv_loads_f64(PTR) \
    _mm256_castsi256_pd(_mm256_stream_load_si256((const __m256i*)(PTR)))
// load lower part
#define npyv_loadl_f32(PTR) _mm256_castps128_ps256(_mm_loadu_ps(PTR))
#define npyv_loadl_f64(PTR) _mm256_castpd128_pd256(_mm_loadu_pd(PTR))
// unaligned store
#define npyv_store_f32 _mm256_storeu_ps
#define npyv_store_f64 _mm256_storeu_pd
// aligned store
#define npyv_storea_f32 _mm256_store_ps
#define npyv_storea_f64 _mm256_store_pd
// stream store
#define npyv_stores_f32 _mm256_stream_ps
#define npyv_stores_f64 _mm256_stream_pd
// store lower part
#define npyv_storel_f32(PTR, VEC) _mm_storeu_ps(PTR, _mm256_castps256_ps128(VEC))
#define npyv_storel_f64(PTR, VEC) _mm_storeu_pd(PTR, _mm256_castpd256_pd128(VEC))
// store higher part
#define npyv_storeh_f32(PTR, VEC) _mm_storeu_ps(PTR, _mm256_extractf128_ps(VEC, 1))
#define npyv_storeh_f64(PTR, VEC) _mm_storeu_pd(PTR, _mm256_extractf128_pd(VEC, 1))

#endif // _NPY_SIMD_AVX2_MEMORY_H
