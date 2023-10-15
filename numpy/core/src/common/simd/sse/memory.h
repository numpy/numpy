#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MEMORY_H
#define _NPY_SIMD_SSE_MEMORY_H

#include "misc.h"

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
/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    __m128i a = _mm_cvtsi32_si128(*ptr);
#ifdef NPY_HAVE_SSE41
    a = _mm_insert_epi32(a, ptr[stride],   1);
    a = _mm_insert_epi32(a, ptr[stride*2], 2);
    a = _mm_insert_epi32(a, ptr[stride*3], 3);
#else
    __m128i a1 = _mm_cvtsi32_si128(ptr[stride]);
    __m128i a2 = _mm_cvtsi32_si128(ptr[stride*2]);
    __m128i a3 = _mm_cvtsi32_si128(ptr[stride*3]);
    a = _mm_unpacklo_epi32(a, a1);
    a = _mm_unpacklo_epi64(a, _mm_unpacklo_epi32(a2, a3));
#endif
    return a;
}
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_loadn_s32((const npy_int32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return _mm_castsi128_ps(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return _mm_loadh_pd(npyv_loadl_f64(ptr), ptr + stride); }
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return _mm_castpd_si128(npyv_loadn_f64((const double*)ptr, stride)); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{
    __m128d r = _mm_loadh_pd(
        npyv_loadl_f64((const double*)ptr), (const double*)(ptr + stride)
    );
    return _mm_castpd_ps(r);
}
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return _mm_castps_si128(npyv_loadn2_f32((const float*)ptr, stride)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return _mm_castps_si128(npyv_loadn2_f32((const float*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ (void)stride; return npyv_load_f64(ptr); }
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_u64(ptr); }
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_s64(ptr); }

/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
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
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, _mm_castps_si128(a)); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
    _mm_storel_pd(ptr, a);
    _mm_storeh_pd(ptr + stride, a);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm_castsi128_pd(a)); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, _mm_castsi128_pd(a)); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    _mm_storel_pd((double*)ptr, _mm_castsi128_pd(a));
    _mm_storeh_pd((double*)(ptr + stride), _mm_castsi128_pd(a));
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, _mm_castps_si128(a)); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ (void)stride; npyv_store_u64(ptr, a); }
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ (void)stride; npyv_store_s64(ptr, a); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ (void)stride; npyv_store_f64(ptr, a); }
/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    #ifndef NPY_HAVE_SSE41
        const short *wptr = (const short*)ptr;
    #endif
    const __m128i vfill = npyv_setall_s32(fill);
    __m128i a;
    switch(nlane) {
        case 2:
            a = _mm_castpd_si128(
                _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
            );
            break;
    #ifdef NPY_HAVE_SSE41
        case 1:
            a = _mm_insert_epi32(vfill, ptr[0], 0);
            break;
        case 3:
            a = _mm_loadl_epi64((const __m128i*)ptr);
            a = _mm_insert_epi32(a, ptr[2], 2);
            a = _mm_insert_epi32(a, fill, 3);
            break;
    #else
        case 1:
            a = _mm_insert_epi16(vfill, wptr[0], 0);
    a = _mm_insert_epi16(a, wptr[1], 1);
            break;
        case 3:
            a = _mm_loadl_epi64((const __m128i*)ptr);
            a = _mm_unpacklo_epi64(a, vfill);
            a = _mm_insert_epi16(a, wptr[4], 4);
            a = _mm_insert_epi16(a, wptr[5], 5);
            break;
    #endif // NPY_HAVE_SSE41
        default:
            return npyv_load_s32(ptr);
    }
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        // We use a variable marked 'volatile' to convince the compiler that
        // the entire vector is needed.
        volatile __m128i workaround = a;
        // avoid optimizing it out
        a = _mm_or_si128(workaround, a);
    #endif
    return a;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    switch(nlane) {
    case 1:
        return _mm_cvtsi32_si128(*ptr);
    case 2:
        return _mm_loadl_epi64((const __m128i*)ptr);
    case 3: {
            npyv_s32 a = _mm_loadl_epi64((const __m128i*)ptr);
        #ifdef NPY_HAVE_SSE41
            return _mm_insert_epi32(a, ptr[2], 2);
        #else
            return _mm_unpacklo_epi64(a, _mm_cvtsi32_si128(ptr[2]));
        #endif
        }
    default:
        return npyv_load_s32(ptr);
    }
}
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        const __m128i vfill = npyv_setall_s64(fill);
        npyv_s64 a = _mm_castpd_si128(
            _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
        );
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile __m128i workaround = a;
        a = _mm_or_si128(workaround, a);
    #endif
        return a;
    }
    return npyv_load_s64(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return _mm_loadl_epi64((const __m128i*)ptr);
    }
    return npyv_load_s64(ptr);
}

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        const __m128i vfill = npyv_set_s32(fill_lo, fill_hi, fill_lo, fill_hi);
        __m128i a =  _mm_castpd_si128(
            _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
        );
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile __m128i workaround = a;
        a = _mm_or_si128(workaround, a);
    #endif
        return a;
    }
    return npyv_load_s32(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{ (void)nlane; (void)fill_lo; (void)fill_hi; return npyv_load_s64(ptr); }

NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ (void)nlane; return npyv_load_s64(ptr); }

/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    __m128i vfill = npyv_setall_s32(fill);
    #ifndef NPY_HAVE_SSE41
        const short *wptr = (const short*)ptr;
    #endif
    switch(nlane) {
    #ifdef NPY_HAVE_SSE41
        case 3:
            vfill = _mm_insert_epi32(vfill, ptr[stride*2], 2);
        case 2:
            vfill = _mm_insert_epi32(vfill, ptr[stride], 1);
        case 1:
            vfill = _mm_insert_epi32(vfill, ptr[0], 0);
            break;
    #else
        case 3:
            vfill = _mm_unpacklo_epi32(_mm_cvtsi32_si128(ptr[stride*2]), vfill);
        case 2:
            vfill = _mm_unpacklo_epi64(_mm_unpacklo_epi32(
                _mm_cvtsi32_si128(*ptr), _mm_cvtsi32_si128(ptr[stride])
            ), vfill);
            break;
        case 1:
            vfill = _mm_insert_epi16(vfill, wptr[0], 0);
            vfill = _mm_insert_epi16(vfill, wptr[1], 1);
            break;
    #endif // NPY_HAVE_SSE41
    default:
        return npyv_loadn_s32(ptr, stride);
    } // switch
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile __m128i workaround = vfill;
    vfill = _mm_or_si128(workaround, vfill);
#endif
    return vfill;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    switch(nlane) {
    case 1:
        return _mm_cvtsi32_si128(ptr[0]);
    case 2:;
        {
            npyv_s32 a = _mm_cvtsi32_si128(ptr[0]);
    #ifdef NPY_HAVE_SSE41
            return _mm_insert_epi32(a, ptr[stride], 1);
    #else
            return _mm_unpacklo_epi32(a, _mm_cvtsi32_si128(ptr[stride]));
    #endif // NPY_HAVE_SSE41
        }
    case 3:
        {
            npyv_s32 a = _mm_cvtsi32_si128(ptr[0]);
    #ifdef NPY_HAVE_SSE41
            a = _mm_insert_epi32(a, ptr[stride], 1);
            a = _mm_insert_epi32(a, ptr[stride*2], 2);
            return a;
    #else
            a = _mm_unpacklo_epi32(a, _mm_cvtsi32_si128(ptr[stride]));
            a = _mm_unpacklo_epi64(a, _mm_cvtsi32_si128(ptr[stride*2]));
            return a;
    #endif // NPY_HAVE_SSE41
        }
    default:
        return npyv_loadn_s32(ptr, stride);
    }
}
//// 64
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return npyv_load_till_s64(ptr, 1, fill);
    }
    return npyv_loadn_s64(ptr, stride);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return _mm_loadl_epi64((const __m128i*)ptr);
    }
    return npyv_loadn_s64(ptr, stride);
}

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        const __m128i vfill = npyv_set_s32(0, 0, fill_lo, fill_hi);
        __m128i a = _mm_castpd_si128(
            _mm_loadl_pd(_mm_castsi128_pd(vfill), (double*)ptr)
        );
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile __m128i workaround = a;
        a = _mm_or_si128(workaround, a);
    #endif
        return a;
    }
    return npyv_loadn2_s32(ptr, stride);
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return _mm_loadl_epi64((const __m128i*)ptr);
    }
    return npyv_loadn2_s32(ptr, stride);
}

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                                  npy_int64 fill_lo, npy_int64 fill_hi)
{ assert(nlane > 0); (void)stride; (void)nlane; (void)fill_lo; (void)fill_hi; return npyv_load_s64(ptr); }

NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ assert(nlane > 0); (void)stride; (void)nlane; return npyv_load_s64(ptr); }

/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    switch(nlane) {
    case 1:
        *ptr = _mm_cvtsi128_si32(a);
        break;
    case 2:
        _mm_storel_epi64((__m128i *)ptr, a);
        break;
    case 3:
        _mm_storel_epi64((__m128i *)ptr, a);
    #ifdef NPY_HAVE_SSE41
        ptr[2] = _mm_extract_epi32(a, 2);
    #else
        ptr[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
    #endif
        break;
    default:
        npyv_store_s32(ptr, a);
    }
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        _mm_storel_epi64((__m128i *)ptr, a);
        return;
    }
    npyv_store_s64(ptr, a);
}
//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, a); }

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0); (void)nlane;
    npyv_store_s64(ptr, a);
}

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    ptr[stride*0] = _mm_cvtsi128_si32(a);
    switch(nlane) {
    case 1:
        return;
#ifdef NPY_HAVE_SSE41
    case 2:
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        return;
    case 3:
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        ptr[stride*2] = _mm_extract_epi32(a, 2);
        return;
    default:
        ptr[stride*1] = _mm_extract_epi32(a, 1);
        ptr[stride*2] = _mm_extract_epi32(a, 2);
        ptr[stride*3] = _mm_extract_epi32(a, 3);
#else
    case 2:
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        return;
    case 3:
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        ptr[stride*2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
        return;
    default:
        ptr[stride*1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 1)));
        ptr[stride*2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 2)));
        ptr[stride*3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 0, 0, 3)));
#endif
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        _mm_storel_epi64((__m128i *)ptr, a);
        return;
    }
    npyv_storen_s64(ptr, stride, a);
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    npyv_storel_s32(ptr, a);
    if (nlane > 1) {
        npyv_storeh_s32(ptr + stride, a);
    }
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ assert(nlane > 0); (void)stride; (void)nlane; npyv_store_s64(ptr, a); }

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
#define NPYV_IMPL_SSE_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
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

NPYV_IMPL_SSE_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride
#define NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                 \
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

NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_SSE_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_SSE_MEM_INTERLEAVE(SFX, ZSFX)                              \
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

NPYV_IMPL_SSE_MEM_INTERLEAVE(u8, u8)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s8, u8)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u16, u16)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s16, u16)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u32, u32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s32, u32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(u64, u64)
NPYV_IMPL_SSE_MEM_INTERLEAVE(s64, u64)
NPYV_IMPL_SSE_MEM_INTERLEAVE(f32, f32)
NPYV_IMPL_SSE_MEM_INTERLEAVE(f64, f64)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const int i0 = _mm_cvtsi128_si32(idx);
#ifdef NPY_HAVE_SSE41
    const int i1 = _mm_extract_epi32(idx, 1);
    const int i2 = _mm_extract_epi32(idx, 2);
    const int i3 = _mm_extract_epi32(idx, 3);
#else
    const int i1 = _mm_extract_epi16(idx, 2);
    const int i2 = _mm_extract_epi16(idx, 4);
    const int i3 = _mm_extract_epi16(idx, 6);
#endif
    return npyv_set_f32(table[i0], table[i1], table[i2], table[i3]);
}
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    const int i0 = _mm_cvtsi128_si32(idx);
#ifdef NPY_HAVE_SSE41
    const int i1 = _mm_extract_epi32(idx, 2);
#else
    const int i1 = _mm_extract_epi16(idx, 4);
#endif
    return npyv_set_f64(table[i0], table[i1]);
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_SSE_MEMORY_H
