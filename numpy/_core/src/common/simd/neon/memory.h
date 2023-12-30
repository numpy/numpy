#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_MEMORY_H
#define _NPY_SIMD_NEON_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
// GCC requires literal type definitions for pointers types otherwise it causes ambiguous errors
#define NPYV_IMPL_NEON_MEM(SFX, CTYPE)                                           \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const npyv_lanetype_##SFX *ptr)       \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const npyv_lanetype_##SFX *ptr)      \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const npyv_lanetype_##SFX *ptr)      \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const npyv_lanetype_##SFX *ptr)      \
    {                                                                            \
        return vcombine_##SFX(                                                   \
            vld1_##SFX((const CTYPE*)ptr), vdup_n_##SFX(0)                       \
        );                                                                       \
    }                                                                            \
    NPY_FINLINE void npyv_store_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)  \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    NPY_FINLINE void npyv_storea_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    NPY_FINLINE void npyv_stores_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    NPY_FINLINE void npyv_storel_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1_##SFX((CTYPE*)ptr, vget_low_##SFX(vec)); }                            \
    NPY_FINLINE void npyv_storeh_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1_##SFX((CTYPE*)ptr, vget_high_##SFX(vec)); }

NPYV_IMPL_NEON_MEM(u8,  uint8_t)
NPYV_IMPL_NEON_MEM(s8,  int8_t)
NPYV_IMPL_NEON_MEM(u16, uint16_t)
NPYV_IMPL_NEON_MEM(s16, int16_t)
NPYV_IMPL_NEON_MEM(u32, uint32_t)
NPYV_IMPL_NEON_MEM(s32, int32_t)
NPYV_IMPL_NEON_MEM(u64, uint64_t)
NPYV_IMPL_NEON_MEM(s64, int64_t)
NPYV_IMPL_NEON_MEM(f32, float)
#if NPY_SIMD_F64
NPYV_IMPL_NEON_MEM(f64, double)
#endif
/***************************
 * Non-contiguous Load
 ***************************/
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    int32x4_t a = vdupq_n_s32(0);
    a = vld1q_lane_s32((const int32_t*)ptr,            a, 0);
    a = vld1q_lane_s32((const int32_t*)ptr + stride,   a, 1);
    a = vld1q_lane_s32((const int32_t*)ptr + stride*2, a, 2);
    a = vld1q_lane_s32((const int32_t*)ptr + stride*3, a, 3);
    return a;
}

NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    return npyv_reinterpret_u32_s32(
        npyv_loadn_s32((const npy_int32*)ptr, stride)
    );
}
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{
    return npyv_reinterpret_f32_s32(
        npyv_loadn_s32((const npy_int32*)ptr, stride)
    );
}
//// 64
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{
    return vcombine_s64(
        vld1_s64((const int64_t*)ptr), vld1_s64((const int64_t*)ptr + stride)
    );
}
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    return npyv_reinterpret_u64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    return npyv_reinterpret_f64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}
#endif

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    return vcombine_u32(
        vld1_u32((const uint32_t*)ptr), vld1_u32((const uint32_t*)ptr + stride)
    );
}
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_reinterpret_s32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return npyv_reinterpret_f32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_u64(ptr); }
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_s64(ptr); }
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ (void)stride; return npyv_load_f64(ptr); }
#endif

/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    vst1q_lane_s32((int32_t*)ptr, a, 0);
    vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
    vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
    vst1q_lane_s32((int32_t*)ptr + stride*3, a, 3);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, npyv_reinterpret_s32_u32(a)); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, npyv_reinterpret_s32_f32(a)); }
//// 64
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    vst1q_lane_s64((int64_t*)ptr, a, 0);
    vst1q_lane_s64((int64_t*)ptr + stride, a, 1);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, npyv_reinterpret_s64_u64(a)); }

#if NPY_SIMD_F64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, npyv_reinterpret_s64_f64(a)); }
#endif

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
#if NPY_SIMD_F64
    vst1q_lane_u64((uint64_t*)ptr, npyv_reinterpret_u64_u32(a), 0);
    vst1q_lane_u64((uint64_t*)(ptr + stride), npyv_reinterpret_u64_u32(a), 1);
#else
    // armhf strict to alignment
    vst1_u32((uint32_t*)ptr, vget_low_u32(a));
    vst1_u32((uint32_t*)ptr + stride, vget_high_u32(a));
#endif
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_s32(a)); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_f32(a)); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ (void)stride; npyv_store_u64(ptr, a); }
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ (void)stride; npyv_store_s64(ptr, a); }
#if NPY_SIMD_F64
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ (void)stride; npyv_store_f64(ptr, a); }
#endif
/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    npyv_s32 a;
    switch(nlane) {
    case 1:
        a = vld1q_lane_s32((const int32_t*)ptr, vdupq_n_s32(fill), 0);
        break;
    case 2:
        a = vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(fill));
        break;
    case 3:
        a = vcombine_s32(
            vld1_s32((const int32_t*)ptr),
            vld1_lane_s32((const int32_t*)ptr + 2, vdup_n_s32(fill), 0)
        );
        break;
    default:
        return npyv_load_s32(ptr);
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = a;
    a = vorrq_s32(workaround, a);
#endif
    return a;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_till_s32(ptr, nlane, 0); }
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s64 a = vcombine_s64(vld1_s64((const int64_t*)ptr), vdup_n_s64(fill));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s64 workaround = a;
        a = vorrq_s64(workaround, a);
    #endif
        return a;
    }
    return npyv_load_s64(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ return npyv_load_till_s64(ptr, nlane, 0); }

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        const int32_t NPY_DECL_ALIGNED(16) fill[2] = {fill_lo, fill_hi};
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vld1_s32(fill));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = vorrq_s32(workaround, a);
    #endif
        return a;
    }
    return npyv_load_s32(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return vreinterpretq_s32_s64(npyv_load_tillz_s64((const npy_int64*)ptr, nlane)); }

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
    int32x4_t vfill = vdupq_n_s32(fill);
    switch(nlane) {
    case 3:
        vfill = vld1q_lane_s32((const int32_t*)ptr + stride*2, vfill, 2);
    case 2:
        vfill = vld1q_lane_s32((const int32_t*)ptr + stride, vfill, 1);
    case 1:
        vfill = vld1q_lane_s32((const int32_t*)ptr, vfill, 0);
        break;
    default:
        return npyv_loadn_s32(ptr, stride);
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = vfill;
    vfill = vorrq_s32(workaround, vfill);
#endif
    return vfill;
}
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

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
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        const int32_t NPY_DECL_ALIGNED(16) fill[2] = {fill_lo, fill_hi};
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vld1_s32(fill));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = vorrq_s32(workaround, a);
    #endif
        return a;
    }
    return npyv_loadn2_s32(ptr, stride);
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(0));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = vorrq_s32(workaround, a);
    #endif
        return a;
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
        vst1q_lane_s32((int32_t*)ptr, a, 0);
        break;
    case 2:
        vst1_s32((int32_t*)ptr, vget_low_s32(a));
        break;
    case 3:
        vst1_s32((int32_t*)ptr, vget_low_s32(a));
        vst1q_lane_s32((int32_t*)ptr + 2, a, 2);
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
        vst1q_lane_s64((int64_t*)ptr, a, 0);
        return;
    }
    npyv_store_s64(ptr, a);
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        // armhf strict to alignment, may cause bus error
    #if NPY_SIMD_F64
        vst1q_lane_s64((int64_t*)ptr, npyv_reinterpret_s64_s32(a), 0);
    #else
        npyv_storel_s32(ptr, a);
    #endif
        return;
    }
    npyv_store_s32(ptr, a);
}

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
    vst1q_lane_s32((int32_t*)ptr, a, 0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        return;
    case 3:
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
        return;
    default:
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
        vst1q_lane_s32((int32_t*)ptr + stride*3, a, 3);
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        vst1q_lane_s64((int64_t*)ptr, a, 0);
        return;
    }
    npyv_storen_s64(ptr, stride, a);
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
#if NPY_SIMD_F64
    vst1q_lane_s64((int64_t*)ptr, npyv_reinterpret_s64_s32(a), 0);
    if (nlane > 1) {
        vst1q_lane_s64((int64_t*)(ptr + stride), npyv_reinterpret_s64_s32(a), 1);
    }
#else
    npyv_storel_s32(ptr, a);
    if (nlane > 1) {
        npyv_storeh_s32(ptr + stride, a);
    }
#endif
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ assert(nlane > 0); (void)stride; (void)nlane; npyv_store_s64(ptr, a); }

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
#define NPYV_IMPL_NEON_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                     \
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

NPYV_IMPL_NEON_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_NEON_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_NEON_REST_PARTIAL_TYPES(u64, s64)
#if NPY_SIMD_F64
NPYV_IMPL_NEON_REST_PARTIAL_TYPES(f64, s64)
#endif

// 128-bit/64-bit stride
#define NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                \
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

NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(u64, s64)
#if NPY_SIMD_F64
NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(f64, s64)
#endif

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_NEON_MEM_INTERLEAVE(SFX, T_PTR)                        \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                      \
        const npyv_lanetype_##SFX *ptr                                   \
    ) {                                                                  \
        return vld2q_##SFX((const T_PTR*)ptr);                           \
    }                                                                    \
    NPY_FINLINE void npyv_store_##SFX##x2(                               \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                       \
    ) {                                                                  \
        vst2q_##SFX((T_PTR*)ptr, v);                                     \
    }

NPYV_IMPL_NEON_MEM_INTERLEAVE(u8,  uint8_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(s8,  int8_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(u16, uint16_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(s16, int16_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(u32, uint32_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(s32, int32_t)
NPYV_IMPL_NEON_MEM_INTERLEAVE(f32, float)

#if NPY_SIMD_F64
    NPYV_IMPL_NEON_MEM_INTERLEAVE(f64, double)
    NPYV_IMPL_NEON_MEM_INTERLEAVE(u64, uint64_t)
    NPYV_IMPL_NEON_MEM_INTERLEAVE(s64, int64_t)
#else
    #define NPYV_IMPL_NEON_MEM_INTERLEAVE_64(SFX)                               \
        NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                         \
            const npyv_lanetype_##SFX *ptr)                                     \
        {                                                                       \
            npyv_##SFX a = npyv_load_##SFX(ptr);                                \
            npyv_##SFX b = npyv_load_##SFX(ptr + 2);                            \
            npyv_##SFX##x2 r;                                                   \
            r.val[0] = vcombine_##SFX(vget_low_##SFX(a),  vget_low_##SFX(b));   \
            r.val[1] = vcombine_##SFX(vget_high_##SFX(a), vget_high_##SFX(b));  \
            return r;                                                           \
        }                                                                       \
        NPY_FINLINE void npyv_store_##SFX##x2(                                  \
            npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v)                         \
        {                                                                       \
            npyv_store_##SFX(ptr, vcombine_##SFX(                               \
                vget_low_##SFX(v.val[0]),  vget_low_##SFX(v.val[1])));          \
            npyv_store_##SFX(ptr + 2, vcombine_##SFX(                           \
                vget_high_##SFX(v.val[0]),  vget_high_##SFX(v.val[1])));        \
        }
        NPYV_IMPL_NEON_MEM_INTERLEAVE_64(u64)
        NPYV_IMPL_NEON_MEM_INTERLEAVE_64(s64)
#endif
/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of uint32.
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    const unsigned i0 = vgetq_lane_u32(idx, 0);
    const unsigned i1 = vgetq_lane_u32(idx, 1);
    const unsigned i2 = vgetq_lane_u32(idx, 2);
    const unsigned i3 = vgetq_lane_u32(idx, 3);

    uint32x2_t low = vcreate_u32(table[i0]);
               low = vld1_lane_u32((const uint32_t*)table + i1, low, 1);
    uint32x2_t high = vcreate_u32(table[i2]);
               high = vld1_lane_u32((const uint32_t*)table + i3, high, 1);
    return vcombine_u32(low, high);
}
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return npyv_reinterpret_f32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of uint64.
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{
    const unsigned i0 = vgetq_lane_u32(vreinterpretq_u32_u64(idx), 0);
    const unsigned i1 = vgetq_lane_u32(vreinterpretq_u32_u64(idx), 2);
    return vcombine_u64(
        vld1_u64((const uint64_t*)table + i0),
        vld1_u64((const uint64_t*)table + i1)
    );
}
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return npyv_reinterpret_f64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }
#endif

#endif // _NPY_SIMD_NEON_MEMORY_H
