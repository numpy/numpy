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
    switch (stride) {
    case 2:
        return vld2q_s32((const int32_t*)ptr).val[0];
    case 3:
        return vld3q_s32((const int32_t*)ptr).val[0];
    case 4:
        return vld4q_s32((const int32_t*)ptr).val[0];
    default:;
        int32x2_t ax = vcreate_s32(*ptr);
        int32x4_t a = vcombine_s32(ax, ax);
                  a = vld1q_lane_s32((const int32_t*)ptr + stride,   a, 1);
                  a = vld1q_lane_s32((const int32_t*)ptr + stride*2, a, 2);
                  a = vld1q_lane_s32((const int32_t*)ptr + stride*3, a, 3);
        return a;
    }
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

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    switch(nlane) {
    case 1:
        return vld1q_lane_s32((const int32_t*)ptr, vdupq_n_s32(fill), 0);
    case 2:
        return vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(fill));
    case 3:
        return vcombine_s32(
            vld1_s32((const int32_t*)ptr),
            vld1_lane_s32((const int32_t*)ptr + 2, vdup_n_s32(fill), 0)
        );
    default:
        return npyv_load_s32(ptr);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_till_s32(ptr, nlane, 0); }
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return vcombine_s64(vld1_s64((const int64_t*)ptr), vdup_n_s64(fill));
    }
    return npyv_load_s64(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ return npyv_load_till_s64(ptr, nlane, 0); }

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
        return vfill;
    default:
        return npyv_loadn_s32(ptr, stride);
    }
}
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        return vcombine_s64(vld1_s64((const int64_t*)ptr), vdup_n_s64(fill));
    }
    return npyv_loadn_s64(ptr, stride);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

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
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    switch(nlane) {
    default:
        vst1q_lane_s32((int32_t*)ptr + stride*3, a, 3);
    case 3:
        vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
    case 2:
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
    case 1:
        vst1q_lane_s32((int32_t*)ptr, a, 0);
        break;
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
        } pun = {.from_##F_SFX = fill};                                                     \
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
        } pun = {.from_##F_SFX = fill};                                                     \
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

#endif // _NPY_SIMD_NEON_MEMORY_H
