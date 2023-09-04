#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "misc.h"

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
/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i idx = _mm256_mullo_epi32(_mm256_set1_epi32((int)stride), steps);
    return _mm256_i32gather_epi32((const int*)ptr, idx, 4);
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_loadn_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return _mm256_castsi256_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    __m128d a0 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr));
    __m128d a2 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2)));
    __m128d a01 = _mm_loadh_pd(a0, ptr + stride);
    __m128d a23 = _mm_loadh_pd(a2, ptr + stride*3);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(a01), a23, 1);
}
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return _mm256_castpd_si256(npyv_loadn_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return _mm256_castpd_si256(npyv_loadn_f64((const double*)ptr, stride)); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{
    __m128d a0 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr));
    __m128d a2 = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2)));
    __m128d a01 = _mm_loadh_pd(a0, (const double*)(ptr + stride));
    __m128d a23 = _mm_loadh_pd(a2, (const double*)(ptr + stride*3));
    return _mm256_castpd_ps(_mm256_insertf128_pd(_mm256_castpd128_pd256(a01), a23, 1));
}
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return _mm256_castps_si256(npyv_loadn2_f32((const float*)ptr, stride)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return _mm256_castps_si256(npyv_loadn2_f32((const float*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{
    __m256d a = npyv_loadl_f64(ptr);
    return _mm256_insertf128_pd(a, _mm_loadu_pd(ptr + stride), 1);
}
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ return _mm256_castpd_si256(npyv_loadn2_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ return _mm256_castpd_si256(npyv_loadn2_f64((const double*)ptr, stride)); }

/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    __m128i a0 = _mm256_castsi256_si128(a);
    __m128i a1 = _mm256_extracti128_si256(a, 1);
    ptr[stride * 0] = _mm_cvtsi128_si32(a0);
    ptr[stride * 1] = _mm_extract_epi32(a0, 1);
    ptr[stride * 2] = _mm_extract_epi32(a0, 2);
    ptr[stride * 3] = _mm_extract_epi32(a0, 3);
    ptr[stride * 4] = _mm_cvtsi128_si32(a1);
    ptr[stride * 5] = _mm_extract_epi32(a1, 1);
    ptr[stride * 6] = _mm_extract_epi32(a1, 2);
    ptr[stride * 7] = _mm_extract_epi32(a1, 3);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, _mm256_castps_si256(a)); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
    __m128d a0 = _mm256_castpd256_pd128(a);
    __m128d a1 = _mm256_extractf128_pd(a, 1);
    _mm_storel_pd(ptr + stride * 0, a0);
    _mm_storeh_pd(ptr + stride * 1, a0);
    _mm_storel_pd(ptr + stride * 2, a1);
    _mm_storeh_pd(ptr + stride * 3, a1);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm256_castsi256_pd(a)); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm256_castsi256_pd(a)); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);
    _mm_storel_pd((double*)ptr, a0);
    _mm_storeh_pd((double*)(ptr + stride), a0);
    _mm_storel_pd((double*)(ptr + stride*2), a1);
    _mm_storeh_pd((double*)(ptr + stride*3), a1);
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, _mm256_castps_si256(a)); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    npyv_storel_u64(ptr, a);
    npyv_storeh_u64(ptr + stride, a);
}
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, _mm256_castpd_si256(a)); }

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    const __m256i vfill = _mm256_set1_epi32(fill);
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vnlane  = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    __m256i mask    = _mm256_cmpgt_epi32(vnlane, steps);
    __m256i payload = _mm256_maskload_epi32((const int*)ptr, mask);
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vnlane = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    __m256i mask   = _mm256_cmpgt_epi32(vnlane, steps);
    __m256i ret    = _mm256_maskload_epi32((const int*)ptr, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_setall_s64(fill);
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    __m256i ret     = _mm256_maskload_epi64((const long long*)ptr, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_set_s32(
        fill_lo, fill_hi, fill_lo, fill_hi,
        fill_lo, fill_hi, fill_lo, fill_hi
    );
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane  = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    __m256i mask    = _mm256_cmpgt_epi64(vnlane, steps);
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    __m256i ret     = _mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

/// 128-bit nlane
NPY_FINLINE npyv_u64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    npy_int64 m  = -((npy_int64)(nlane > 1));
    __m256i mask = npyv_set_s64(-1, -1, m, m);
    __m256i ret  = _mm256_maskload_epi64((const long long*)ptr, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_u64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    const __m256i vfill = npyv_set_s64(0, 0, fill_lo, fill_hi);
    npy_int64 m     = -((npy_int64)(nlane > 1));
    __m256i mask    = npyv_set_s64(-1, -1, m, m);
    __m256i payload = _mm256_maskload_epi64((const long long*)ptr, mask);
    __m256i ret     =_mm256_blendv_epi8(vfill, payload, mask);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const __m256i vfill = _mm256_set1_epi32(fill);
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i idx   = _mm256_mullo_epi32(_mm256_set1_epi32((int)stride), steps);
    __m256i vnlane      = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    __m256i mask        = _mm256_cmpgt_epi32(vnlane, steps);
    __m256i ret         = _mm256_mask_i32gather_epi32(vfill, (const int*)ptr, idx, mask, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }
//// 64
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_setall_s64(fill);
    const __m256i idx   = npyv_set_s64(0, 1*stride, 2*stride, 3*stride);
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    __m256i ret    = _mm256_mask_i64gather_epi64(vfill, (const long long*)ptr, idx, mask, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64
npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_set_s32(
        fill_lo, fill_hi, fill_lo, fill_hi,
        fill_lo, fill_hi, fill_lo, fill_hi
    );
    const __m256i idx   = npyv_set_s64(0, 1*stride, 2*stride, 3*stride);
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane = npyv_setall_s64(nlane > 4 ? 4 : (int)nlane);
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    __m256i ret    = _mm256_mask_i64gather_epi64(vfill, (const long long*)ptr, idx, mask, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s32(ptr, stride, nlane, 0, 0); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    __m256i a = npyv_loadl_s64(ptr);
#if defined(_MSC_VER) && defined(_M_IX86)
    __m128i fill =_mm_setr_epi32(
        (int)fill_lo, (int)(fill_lo >> 32),
        (int)fill_hi, (int)(fill_hi >> 32)
    );
#else
    __m128i fill = _mm_set_epi64x(fill_hi, fill_lo);
#endif
    __m128i b = nlane > 1 ? _mm_loadu_si128((const __m128i*)(ptr + stride)) : fill;
    __m256i ret = _mm256_inserti128_si256(a, b, 1);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m256i workaround = ret;
    ret = _mm256_or_si256(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s64(ptr, stride, nlane, 0, 0); }

/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vnlane = _mm256_set1_epi32(nlane > 8 ? 8 : (int)nlane);
    __m256i mask   = _mm256_cmpgt_epi32(vnlane, steps);
    _mm256_maskstore_epi32((int*)ptr, mask, a);
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    const __m256i steps = npyv_set_s64(0, 1, 2, 3);
    __m256i vnlane = npyv_setall_s64(nlane > 8 ? 8 : (int)nlane);
    __m256i mask   = _mm256_cmpgt_epi64(vnlane, steps);
    _mm256_maskstore_epi64((long long*)ptr, mask, a);
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, a); }

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
#ifdef _MSC_VER
   /*
    * Although this version is compatible with all other compilers,
    * there is no performance benefit in retaining the other branch.
    * However, it serves as evidence of a newly emerging bug in MSVC
    * that started to appear since v19.30.
    * For some reason, the MSVC optimizer chooses to ignore the lower store (128-bit mov)
    * and replace with full mov counting on ymmword pointer.
    *
    * For more details, please refer to the discussion on https://github.com/numpy/numpy/issues/23896.
    */
    if (nlane > 1) {
        npyv_store_s64(ptr, a);
    }
    else {
        npyv_storel_s64(ptr, a);
    }
#else
    npyv_storel_s64(ptr, a);
    if (nlane > 1) {
        npyv_storeh_s64(ptr + 2, a);
    }
#endif
}
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    __m128i a0 = _mm256_castsi256_si128(a);
    __m128i a1 = _mm256_extracti128_si256(a, 1);

    ptr[stride*0] = _mm_extract_epi32(a0, 0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        return;
    case 3:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        return;
    case 4:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        return;
    case 5:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        return;
    case 6:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        return;
    case 7:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        ptr[stride*6] = _mm_extract_epi32(a1, 2);
        return;
    default:
        ptr[stride*1] = _mm_extract_epi32(a0, 1);
        ptr[stride*2] = _mm_extract_epi32(a0, 2);
        ptr[stride*3] = _mm_extract_epi32(a0, 3);
        ptr[stride*4] = _mm_extract_epi32(a1, 0);
        ptr[stride*5] = _mm_extract_epi32(a1, 1);
        ptr[stride*6] = _mm_extract_epi32(a1, 2);
        ptr[stride*7] = _mm_extract_epi32(a1, 3);
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);

    double *dptr = (double*)ptr;
    _mm_storel_pd(dptr, a0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        _mm_storeh_pd(dptr + stride * 1, a0);
        return;
    case 3:
        _mm_storeh_pd(dptr + stride * 1, a0);
        _mm_storel_pd(dptr + stride * 2, a1);
        return;
    default:
        _mm_storeh_pd(dptr + stride * 1, a0);
        _mm_storel_pd(dptr + stride * 2, a1);
        _mm_storeh_pd(dptr + stride * 3, a1);
    }
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    __m128d a0 = _mm256_castpd256_pd128(_mm256_castsi256_pd(a));
    __m128d a1 = _mm256_extractf128_pd(_mm256_castsi256_pd(a), 1);

    _mm_storel_pd((double*)ptr, a0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        return;
    case 3:
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        _mm_storel_pd((double*)(ptr + stride * 2), a1);
        return;
    default:
        _mm_storeh_pd((double*)(ptr + stride * 1), a0);
        _mm_storel_pd((double*)(ptr + stride * 2), a1);
        _mm_storeh_pd((double*)(ptr + stride * 3), a1);
    }
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    npyv_storel_s64(ptr, a);
    if (nlane > 1) {
        npyv_storeh_s64(ptr + stride, a);
    }
}
/*****************************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via reinterpret cast
 *****************************************************************************/
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                     \
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride (load/store pair)
#define NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                \
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_AVX2_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_AVX2_MEM_INTERLEAVE(SFX, ZSFX)                             \
    NPY_FINLINE npyv_##ZSFX##x2 npyv_zip_##ZSFX(npyv_##ZSFX, npyv_##ZSFX);   \
    NPY_FINLINE npyv_##ZSFX##x2 npyv_unzip_##ZSFX(npyv_##ZSFX, npyv_##ZSFX); \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                          \
        const npyv_lanetype_##SFX *ptr                                       \
    ) {                                                                      \
        return npyv_unzip_##ZSFX(                                            \
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX)     \
        );                                                                   \
    }                                                                        \
    NPY_FINLINE void npyv_store_##SFX##x2(                                   \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                           \
    ) {                                                                      \
        npyv_##SFX##x2 zip = npyv_zip_##ZSFX(v.val[0], v.val[1]);            \
        npyv_store_##SFX(ptr, zip.val[0]);                                   \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);               \
    }

NPYV_IMPL_AVX2_MEM_INTERLEAVE(u8, u8)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s8, u8)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u16, u16)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s16, u16)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u32, u32)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s32, u32)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(u64, u64)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(s64, u64)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(f32, f32)
NPYV_IMPL_AVX2_MEM_INTERLEAVE(f64, f64)

/*********************************
 * Lookup tables
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return _mm256_i32gather_ps(table, idx, 4); }
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return _mm256_i64gather_pd(table, idx, 8); }
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_AVX2_MEMORY_H
