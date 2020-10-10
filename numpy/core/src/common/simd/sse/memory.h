#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MEMORY_H
#define _NPY_SIMD_SSE_MEMORY_H

/***************************
 * load/store
 ***************************/
// stream load
#ifdef NPY_HAVE_SSE41
    #define npyv__loads(PTR) _mm_stream_load_si128((__m128i *)(PTR))
#else
    #define npyv__loads(PTR) _mm_load_si128((const __m128i *)(PTR))
#endif
#define NPYV_IMPL_SSE_MEM_INT(CTYPE, SFX)                                    \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)                 \
    { return _mm_loadu_si128((const __m128i*)ptr); }                         \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)                \
    { return _mm_load_si128((const __m128i*)ptr); }                          \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)                \
    { return npyv__loads(ptr); }                                             \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)                \
    { return _mm_loadl_epi64((const __m128i*)ptr); }                         \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)            \
    { _mm_storeu_si128((__m128i*)ptr, vec); }                                \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_store_si128((__m128i*)ptr, vec); }                                 \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_stream_si128((__m128i*)ptr, vec); }                                \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storel_epi64((__m128i *)ptr, vec); }                               \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)           \
    { _mm_storel_epi64((__m128i *)ptr, _mm_unpackhi_epi64(vec, vec)); }

NPYV_IMPL_SSE_MEM_INT(npy_uint8,  u8)
NPYV_IMPL_SSE_MEM_INT(npy_int8,   s8)
NPYV_IMPL_SSE_MEM_INT(npy_uint16, u16)
NPYV_IMPL_SSE_MEM_INT(npy_int16,  s16)
NPYV_IMPL_SSE_MEM_INT(npy_uint32, u32)
NPYV_IMPL_SSE_MEM_INT(npy_int32,  s32)
NPYV_IMPL_SSE_MEM_INT(npy_uint64, u64)
NPYV_IMPL_SSE_MEM_INT(npy_int64,  s64)

// unaligned load
#define npyv_load_f32 _mm_loadu_ps
#define npyv_load_f64 _mm_loadu_pd
// aligned load
#define npyv_loada_f32 _mm_load_ps
#define npyv_loada_f64 _mm_load_pd
// load lower part
#define npyv_loadl_f32(PTR) _mm_castsi128_ps(npyv_loadl_u32((const npy_uint32*)(PTR)))
#define npyv_loadl_f64(PTR) _mm_castsi128_pd(npyv_loadl_u32((const npy_uint32*)(PTR)))
// stream load
#define npyv_loads_f32(PTR) _mm_castsi128_ps(npyv__loads(PTR))
#define npyv_loads_f64(PTR) _mm_castsi128_pd(npyv__loads(PTR))
// unaligned store
#define npyv_store_f32 _mm_storeu_ps
#define npyv_store_f64 _mm_storeu_pd
// aligned store
#define npyv_storea_f32 _mm_store_ps
#define npyv_storea_f64 _mm_store_pd
// stream store
#define npyv_stores_f32 _mm_stream_ps
#define npyv_stores_f64 _mm_stream_pd
// store lower part
#define npyv_storel_f32(PTR, VEC) _mm_storel_epi64((__m128i*)(PTR), _mm_castps_si128(VEC));
#define npyv_storel_f64(PTR, VEC) _mm_storel_epi64((__m128i*)(PTR), _mm_castpd_si128(VEC));
// store higher part
#define npyv_storeh_f32(PTR, VEC) npyv_storeh_u32((npy_uint32*)(PTR), _mm_castps_si128(VEC))
#define npyv_storeh_f64(PTR, VEC) npyv_storeh_u32((npy_uint32*)(PTR), _mm_castpd_si128(VEC))

#endif // _NPY_SIMD_SSE_MEMORY_H
