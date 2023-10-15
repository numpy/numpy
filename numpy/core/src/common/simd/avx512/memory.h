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
/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    return _mm512_i32gather_epi32(idx, (const __m512i*)ptr, 4);
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_loadn_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return _mm512_castsi512_ps(npyv_loadn_u32((const npy_uint32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    return _mm512_i64gather_epi64(idx, (const __m512i*)ptr, 8);
}
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_loadn_u64((const npy_uint64*)ptr, stride); }
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return _mm512_castsi512_pd(npyv_loadn_u64((const npy_uint64*)ptr, stride)); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    __m128d a = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)ptr)),
        (const double*)(ptr + stride)
    );
    __m128d b = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*2))),
        (const double*)(ptr + stride*3)
    );
    __m128d c = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*4))),
        (const double*)(ptr + stride*5)
    );
    __m128d d = _mm_loadh_pd(
        _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(ptr + stride*6))),
        (const double*)(ptr + stride*7)
    );
    return _mm512_castpd_si512(npyv512_combine_pd256(
        _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1),
        _mm256_insertf128_pd(_mm256_castpd128_pd256(c), d, 1)
    ));
}
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_loadn2_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return _mm512_castsi512_ps(npyv_loadn2_u32((const npy_uint32*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{
    __m128d a = _mm_loadu_pd(ptr);
    __m128d b = _mm_loadu_pd(ptr + stride);
    __m128d c = _mm_loadu_pd(ptr + stride * 2);
    __m128d d = _mm_loadu_pd(ptr + stride * 3);
    return npyv512_combine_pd256(
        _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1),
        _mm256_insertf128_pd(_mm256_castpd128_pd256(c), d, 1)
    );
}
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_reinterpret_u64_f64(npyv_loadn2_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_loadn2_u64((const npy_uint64*)ptr, stride); }
/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    assert(llabs(stride) <= NPY_SIMD_MAXSTORE_STRIDE32);
    const __m512i steps = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    _mm512_i32scatter_epi32((__m512i*)ptr, idx, a, 4);
}
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, _mm512_castps_si512(a)); }
//// 64
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    _mm512_i64scatter_epi64((__m512i*)ptr, idx, a, 8);
}
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, _mm512_castpd_si512(a)); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    __m256d lo = _mm512_castpd512_pd256(_mm512_castsi512_pd(a));
    __m256d hi = _mm512_extractf64x4_pd(_mm512_castsi512_pd(a), 1);
    __m128d e0 = _mm256_castpd256_pd128(lo);
    __m128d e1 = _mm256_extractf128_pd(lo, 1);
    __m128d e2 = _mm256_castpd256_pd128(hi);
    __m128d e3 = _mm256_extractf128_pd(hi, 1);
    _mm_storel_pd((double*)(ptr + stride * 0), e0);
    _mm_storeh_pd((double*)(ptr + stride * 1), e0);
    _mm_storel_pd((double*)(ptr + stride * 2), e1);
    _mm_storeh_pd((double*)(ptr + stride * 3), e1);
    _mm_storel_pd((double*)(ptr + stride * 4), e2);
    _mm_storeh_pd((double*)(ptr + stride * 5), e2);
    _mm_storel_pd((double*)(ptr + stride * 6), e3);
    _mm_storeh_pd((double*)(ptr + stride * 7), e3);
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, _mm512_castps_si512(a)); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    __m256i lo = npyv512_lower_si256(a);
    __m256i hi = npyv512_higher_si256(a);
    __m128i e0 = _mm256_castsi256_si128(lo);
    __m128i e1 = _mm256_extracti128_si256(lo, 1);
    __m128i e2 = _mm256_castsi256_si128(hi);
    __m128i e3 = _mm256_extracti128_si256(hi, 1);
    _mm_storeu_si128((__m128i*)(ptr + stride * 0), e0);
    _mm_storeu_si128((__m128i*)(ptr + stride * 1), e1);
    _mm_storeu_si128((__m128i*)(ptr + stride * 2), e2);
    _mm_storeu_si128((__m128i*)(ptr + stride * 3), e3);
}
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, _mm512_castpd_si512(a)); }

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    const __m512i vfill = _mm512_set1_epi32(fill);
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_loadu_epi32(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_maskz_loadu_epi32(mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    const __m512i vfill = npyv_setall_s64(fill);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_maskz_loadu_epi64(mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    const __m512i vfill = _mm512_set4_epi32(fill_hi, fill_lo, fill_hi, fill_lo);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane
NPY_FINLINE npyv_u64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    const __m512i vfill = _mm512_set4_epi64(fill_hi, fill_lo, fill_hi, fill_lo);
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    __m512i ret = _mm512_mask_loadu_epi64(vfill, mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
#endif
    return ret;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    __m512i ret = _mm512_maskz_loadu_epi64(mask, (const __m512i*)ptr);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
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
    const __m512i steps = npyv_set_s32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    const __m512i vfill = _mm512_set1_epi32(fill);
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_i32gather_epi32(vfill, mask, idx, (const __m512i*)ptr, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
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
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    const __m512i vfill = npyv_setall_s64(fill);
    const __mmask8 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
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
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    const __m512i vfill = _mm512_set4_epi32(fill_hi, fill_lo, fill_hi, fill_lo);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 4);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
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
    const __m512i idx = npyv_set_s64(
       0,        1,          stride,   stride+1,
       stride*2, stride*2+1, stride*3, stride*3+1
    );
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    const __m512i vfill = _mm512_set4_epi64(fill_hi, fill_lo, fill_hi, fill_lo);
    __m512i ret = _mm512_mask_i64gather_epi64(vfill, mask, idx, (const __m512i*)ptr, 8);
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m512i workaround = ret;
    ret = _mm512_or_si512(workaround, ret);
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
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    _mm512_mask_storeu_epi32((__m512i*)ptr, mask, a);
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    _mm512_mask_storeu_epi64((__m512i*)ptr, mask, a);
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    _mm512_mask_storeu_epi64((__m512i*)ptr, mask, a);
}

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    _mm512_mask_storeu_epi64((__m512i*)ptr, mask, a);
}
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXSTORE_STRIDE32);
    const __m512i steps = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    const __m512i idx = _mm512_mullo_epi32(steps, _mm512_set1_epi32((int)stride));
    const __mmask16 mask = nlane > 15 ? -1 : (1 << nlane) - 1;
    _mm512_mask_i32scatter_epi32((__m512i*)ptr, mask, idx, a, 4);
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 8);
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    const __m512i idx = npyv_set_s64(
        0*stride, 1*stride, 2*stride, 3*stride,
        4*stride, 5*stride, 6*stride, 7*stride
    );
    const __mmask8 mask = nlane > 7 ? -1 : (1 << nlane) - 1;
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 4);
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    const __m512i idx = npyv_set_s64(
        0,        1,            stride,   stride+1,
        2*stride, 2*stride+1, 3*stride, 3*stride+1
    );
    const __mmask8 mask = nlane > 3 ? -1 : (1 << (nlane*2)) - 1;
    _mm512_mask_i64scatter_epi64((__m512i*)ptr, mask, idx, a, 8);
}

/*****************************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via reinterpret cast
 *****************************************************************************/
#define NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                   \
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

NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride (pair load/store)
#define NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                              \
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

NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_AVX512_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_AVX512_MEM_INTERLEAVE(SFX, ZSFX)                           \
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

NPYV_IMPL_AVX512_MEM_INTERLEAVE(u8, u8)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s8, u8)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u16, u16)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s16, u16)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u32, u32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s32, u32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(u64, u64)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(s64, u64)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(f32, f32)
NPYV_IMPL_AVX512_MEM_INTERLEAVE(f64, f64)

/**************************************************
 * Lookup table
 *************************************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const npyv_f32 table0 = npyv_load_f32(table);
    const npyv_f32 table1 = npyv_load_f32(table + 16);
    return _mm512_permutex2var_ps(table0, idx, table1);
}
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    const npyv_f64 table0 = npyv_load_f64(table);
    const npyv_f64 table1 = npyv_load_f64(table + 8);
    return _mm512_permutex2var_pd(table0, idx, table1);
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_AVX512_MEMORY_H
