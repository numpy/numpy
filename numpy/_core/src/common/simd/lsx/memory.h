#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_MEMORY_H
#define _NPY_SIMD_LSX_MEMORY_H

#include <stdint.h>
#include "misc.h"

/***************************
 * load/store
 ***************************/
#define NPYV_IMPL_LSX_MEM(SFX, CTYPE)                               \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)        \
    { return (npyv_##SFX)(__lsx_vld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)(__lsx_vld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)(__lsx_vld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)__lsx_vldrepl_d(ptr, 0); }                 \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)   \
    { __lsx_vst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lsx_vst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lsx_vst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lsx_vstelm_d(vec, ptr, 0, 0); }                             \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lsx_vstelm_d(vec, ptr, 0, 1); }

NPYV_IMPL_LSX_MEM(u8,  npy_uint8)
NPYV_IMPL_LSX_MEM(s8,  npy_int8)
NPYV_IMPL_LSX_MEM(u16, npy_uint16)
NPYV_IMPL_LSX_MEM(s16, npy_int16)
NPYV_IMPL_LSX_MEM(u32, npy_uint32)
NPYV_IMPL_LSX_MEM(s32, npy_int32)
NPYV_IMPL_LSX_MEM(u64, npy_uint64)
NPYV_IMPL_LSX_MEM(s64, npy_int64)
NPYV_IMPL_LSX_MEM(f32, float)
NPYV_IMPL_LSX_MEM(f64, double)

/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    __m128i a = __lsx_vreplgr2vr_w(*ptr);
    a = __lsx_vinsgr2vr_w(a, ptr[stride], 1);
    a = __lsx_vinsgr2vr_w(a, ptr[stride*2], 2);
    a = __lsx_vinsgr2vr_w(a, ptr[stride*3], 3);
    return a;
}
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)  //ok
{ return npyv_reinterpret_f32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return (npyv_f64)__lsx_vilvl_d((__m128i)(v2f64)__lsx_vld((ptr + stride), 0), (__m128i)(v2f64)__lsx_vld(ptr, 0)); }
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_reinterpret_u64_f64(npyv_loadn_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_reinterpret_s64_f64(npyv_loadn_f64((const double*)ptr, stride)); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return (npyv_f32)__lsx_vilvl_d(__lsx_vld((const double *)(ptr + stride), 0), __lsx_vld((const double *)ptr, 0)); }
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_f32(npyv_loadn2_f32((const float*)ptr, stride)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_reinterpret_s32_f32(npyv_loadn2_f32((const float*)ptr, stride)); }

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

    __lsx_vstelm_w(a, ptr, 0, 0);
    __lsx_vstelm_w(a, ptr + stride, 0, 1);
    __lsx_vstelm_w(a, ptr + stride*2, 0, 2);
    __lsx_vstelm_w(a, ptr + stride*3, 0, 3);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, (npyv_s32)a); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
  __lsx_vstelm_d(a, ptr, 0, 0);
  __lsx_vstelm_d(a, ptr + stride, 0, 1);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
  __lsx_vstelm_d(npyv_reinterpret_u64_u32(a), ptr, 0, 0);
  __lsx_vstelm_d(npyv_reinterpret_u64_u32(a), ptr+stride, 0, 1); // zn:TODO
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }

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
    const __m128i vfill = npyv_setall_s32(fill);
    switch(nlane) {
    case 1:
        return __lsx_vinsgr2vr_w(vfill, ptr[0], 0);
    case 2:
        return __lsx_vinsgr2vr_d(vfill, *(unsigned long *)ptr, 0);
    case 3:
        return __lsx_vinsgr2vr_w(__lsx_vld(ptr, 0), fill, 3);
    default:
        return npyv_load_s32(ptr);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    __m128i zfill = __lsx_vldi(0);
    switch(nlane) {
    case 1:
        return __lsx_vinsgr2vr_w(zfill, ptr[0], 0);
    case 2:
        return __lsx_vinsgr2vr_d(zfill, *(unsigned long *)ptr, 0);
    case 3:
        return __lsx_vinsgr2vr_w(__lsx_vld(ptr, 0), 0, 3);
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
        return __lsx_vinsgr2vr_d(vfill, ptr[0], 0);
    }
    return npyv_load_s64(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return __lsx_vinsgr2vr_d(__lsx_vld(ptr, 0), 0, 1);
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
        return (npyv_s32)__lsx_vinsgr2vr_d(vfill, *(long *)ptr, 0);
    }
    return npyv_load_s32(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return (npyv_s32)npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

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
    switch(nlane) {
        case 3:
            vfill = __lsx_vinsgr2vr_w(vfill, ptr[stride*2], 2);
        case 2:
            vfill = __lsx_vinsgr2vr_w(vfill, ptr[stride], 1);
        case 1:
            vfill = __lsx_vinsgr2vr_w(vfill, ptr[0], 0);
            break;
        default:
            return npyv_loadn_s32(ptr, stride);
    } // switch
    return vfill;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    switch(nlane) {
    case 1:
        return __lsx_vinsgr2vr_w(__lsx_vldi(0), ptr[0], 0);
    case 2:
        {
            npyv_s32 a = __lsx_vinsgr2vr_w(__lsx_vldi(0), ptr[0], 0);
            return __lsx_vinsgr2vr_w(a, ptr[stride], 1);
        }
    case 3:
        {
            npyv_s32 a = __lsx_vinsgr2vr_w(__lsx_vldi(0), ptr[0], 0);
            a = __lsx_vinsgr2vr_w(a, ptr[stride], 1);
            a = __lsx_vinsgr2vr_w(a, ptr[stride*2], 2);
            return a;
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
        const __m128i vfill = npyv_setall_s64(fill);
        return __lsx_vinsgr2vr_d(vfill, ptr[0], 0);
    }
    return npyv_loadn_s64(ptr, stride);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return __lsx_vinsgr2vr_d(__lsx_vldi(0), ptr[0], 0);
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
        return (npyv_s32)__lsx_vinsgr2vr_d(vfill, *(long *)ptr, 0);
    }
    return npyv_loadn2_s32(ptr, stride);
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return (npyv_s32)__lsx_vinsgr2vr_d(__lsx_vldi(0), *(long *)ptr, 0);
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
        __lsx_vstelm_w(a, ptr, 0, 0);
        break;
    case 2:
        __lsx_vstelm_d(a, (long *)ptr, 0, 0);
        break;
    case 3:
        __lsx_vstelm_d(a, (long *)ptr, 0, 0);
        __lsx_vstelm_w(a, ptr, 2<<2, 2);
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
        __lsx_vstelm_d(a, ptr, 0, 0);
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
    __lsx_vstelm_w(a, ptr, 0, 0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        ptr[stride*1] = __lsx_vpickve2gr_w(a, 1);
        return;
    case 3:
        ptr[stride*1] = __lsx_vpickve2gr_w(a, 1);
        ptr[stride*2] = __lsx_vpickve2gr_w(a, 2);
        return;
    default:
        ptr[stride*1] = __lsx_vpickve2gr_w(a, 1);
        ptr[stride*2] = __lsx_vpickve2gr_w(a, 2);
        ptr[stride*3] = __lsx_vpickve2gr_w(a, 3);
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        __lsx_vstelm_d(a, ptr, 0, 0);
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
#define NPYV_IMPL_LSX_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
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

NPYV_IMPL_LSX_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride
#define NPYV_IMPL_LSX_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                 \
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

NPYV_IMPL_LSX_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_LSX_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_LSX_MEM_INTERLEAVE(SFX, ZSFX)                              \
    NPY_FINLINE npyv_##SFX##x2 npyv_zip_##SFX(npyv_##SFX, npyv_##SFX);       \
    NPY_FINLINE npyv_##SFX##x2 npyv_unzip_##SFX(npyv_##SFX, npyv_##SFX);     \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                          \
        const npyv_lanetype_##SFX *ptr                                       \
    ) {                                                                      \
        return npyv_unzip_##SFX(                                             \
         npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX)        \
        );                                                                   \
    }                                                                        \
    NPY_FINLINE void npyv_store_##SFX##x2(                                   \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                           \
    ) {                                                                      \
        npyv_##SFX##x2 zip = npyv_zip_##SFX(v.val[0], v.val[1]);             \
        npyv_store_##SFX(ptr, zip.val[0]);                                   \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);               \
    }

NPYV_IMPL_LSX_MEM_INTERLEAVE(u8, uint8_t);
NPYV_IMPL_LSX_MEM_INTERLEAVE(s8, int8_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(u16, uint16_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(s16, int16_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(u32, uint32_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(s32, int32_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(u64, uint64_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(s64, int64_t)
NPYV_IMPL_LSX_MEM_INTERLEAVE(f32, float)
NPYV_IMPL_LSX_MEM_INTERLEAVE(f64, double)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const int i0 = __lsx_vpickve2gr_wu(idx, 0);
    const int i1 = __lsx_vpickve2gr_wu(idx, 1);
    const int i2 = __lsx_vpickve2gr_wu(idx, 2);
    const int i3 = __lsx_vpickve2gr_wu(idx, 3);
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
    const int i0 = __lsx_vpickve2gr_wu(idx, 0);
    const int i1 = __lsx_vpickve2gr_wu(idx, 2);
    return npyv_set_f64(table[i0], table[i1]);
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_LSX_MEMORY_H
