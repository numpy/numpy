#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "misc.h"

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

// non-contiguous load
// 8
NPY_FINLINE npyv_u8 npyv_loadn_u8(const npy_uint8 *ptr, int stride)
{
    return npyv_set_u8(
        ptr[stride * 0],  ptr[stride * 1],  ptr[stride * 2],  ptr[stride * 3],
        ptr[stride * 4],  ptr[stride * 5],  ptr[stride * 6],  ptr[stride * 7],
        ptr[stride * 8],  ptr[stride * 9],  ptr[stride * 10], ptr[stride * 11],
        ptr[stride * 12], ptr[stride * 13], ptr[stride * 14], ptr[stride * 15],
    );
}
NPY_FINLINE npyv_s8 npyv_loadn_s8(const npy_int8 *ptr, int stride)
{ return npyv_loadn_u8((const npy_uint8 *)ptr, stride); }
// 16
NPY_FINLINE npyv_u16 npyv_loadn_u16(const npy_uint16 *ptr, int stride)
{
    return npyv_set_u16(
        ptr[stride * 0],  ptr[stride * 1],  ptr[stride * 2],  ptr[stride * 3],
        ptr[stride * 4],  ptr[stride * 5],  ptr[stride * 6],  ptr[stride * 7]
    );
}
NPY_FINLINE npyv_s16 npyv_loadn_s16(const npy_int16 *ptr, int stride)
{ return npyv_loadn_u16((const npy_uint16 *)ptr, stride); }
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, int stride)
{
    return npyv_set_u32(
        ptr[stride * 0], ptr[stride * 1],
        ptr[stride * 2], ptr[stride * 3]
    );
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, int stride)
{ return npyv_loadn_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, int stride)
{ return _mm_castsi128_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, int stride)
{ return _mm_loadh_pd(npyv_loadl_f64(ptr), ptr + stride); }
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, int stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, int stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }

// non-contiguous store
//// 8
NPY_FINLINE void npyv_storen_u8(npy_uint8 *ptr, int stride, npyv_u8 a)
{
#ifdef NPY_HAVE_SSE41
    #define NPYV_IMPL_SSE_STOREN8(I) \
    { \
        unsigned e = (unsigned)_mm_extract_epi32(a, I/4); \
        ptr[stride*(I+0)] = (npy_uint8)e; \
        ptr[stride*(I+1)] = (npy_uint8)(e >> 8); \
        ptr[stride*(I+2)] = (npy_uint8)(e >> 16); \
        ptr[stride*(I+3)] = (npy_uint8)(e >> 24); \
    }
#else
    #define NPYV_IMPL_SSE_STOREN8(I) \
    { \
        unsigned e0 = (unsigned)_mm_extract_epi16(a, I/2); \
        unsigned e1 = (unsigned)_mm_extract_epi16(a, I/2+1); \
        ptr[stride*(I+0)] = (npy_uint8)e0; \
        ptr[stride*(I+1)] = (npy_uint8)(e0 >> 8); \
        ptr[stride*(I+2)] = (npy_uint8)e1; \
        ptr[stride*(I+3)] = (npy_uint8)(e1 >> 8); \
    }
#endif
    NPYV_IMPL_SSE_STOREN8(0)
    NPYV_IMPL_SSE_STOREN8(4)
    NPYV_IMPL_SSE_STOREN8(8)
    NPYV_IMPL_SSE_STOREN8(12)
}
NPY_FINLINE void npyv_storen_s8(npy_int8 *ptr, int stride, npyv_s8 a)
{ npyv_storen_u8((npy_uint8*)ptr, stride, a); }
//// 16
NPY_FINLINE void npyv_storen_u16(npy_uint16 *ptr, int stride, npyv_u16 a)
{
#ifdef NPY_HAVE_SSE41
    #define NPYV_IMPL_SSE_STOREN16(I) \
    { \
        unsigned e = (unsigned)_mm_extract_epi32(a, I/2); \
        ptr[stride*(I+0)] = (npy_uint16)e; \
        ptr[stride*(I+1)] = (npy_uint16)(e >> 16); \
    }
#else
    #define NPYV_IMPL_SSE_STOREN16(I) \
        ptr[stride*(I+0)] = (npy_uint16)_mm_extract_epi16(a, I); \
        ptr[stride*(I+1)] = (npy_uint16)_mm_extract_epi16(a, I+1);
#endif
    NPYV_IMPL_SSE_STOREN16(0)
    NPYV_IMPL_SSE_STOREN16(2)
    NPYV_IMPL_SSE_STOREN16(4)
    NPYV_IMPL_SSE_STOREN16(6)
}
NPY_FINLINE void npyv_storen_s16(npy_int16 *ptr, int stride, npyv_s16 a)
{ npyv_storen_u16((npy_uint16*)ptr, stride, a); }
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, int stride, npyv_s32 a)
{
    ptr[stride * 0] = _mm_cvtsi128_si32(a);
#ifdef NPY_HAVE_SSE41
    ptr[stride * 1] = _mm_extract_epi32(a, 1);
    ptr[stride * 2] = _mm_extract_epi32(a, 2);
    ptr[stride * 3] = _mm_extract_epi32(a, 3);
#else
    ptr[stride * 1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
    ptr[stride * 2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
    ptr[stride * 3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 3)));
#endif
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, int stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, int stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, _mm_castps_si128(a)); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, int stride, npyv_f64 a)
{
    _mm_storel_pd(ptr, a);
    _mm_storeh_pd(ptr + stride, a);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, int stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm_castsi128_pd(a)); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, int stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm_castsi128_pd(a)); }

#endif // _NPY_SIMD_SSE_MEMORY_H
