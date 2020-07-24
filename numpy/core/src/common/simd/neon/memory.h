#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "misc.h"

#ifndef _NPY_SIMD_NEON_MEMORY_H
#define _NPY_SIMD_NEON_MEMORY_H

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
{ return (npyv_s8)npyv_loadn_u8((const npy_uint8 *)ptr, stride); }
// 16
NPY_FINLINE npyv_u16 npyv_loadn_u16(const npy_uint16 *ptr, int stride)
{
    return npyv_set_u16(
        ptr[stride * 0],  ptr[stride * 1],  ptr[stride * 2],  ptr[stride * 3],
        ptr[stride * 4],  ptr[stride * 5],  ptr[stride * 6],  ptr[stride * 7]
    );
}
NPY_FINLINE npyv_s16 npyv_loadn_s16(const npy_int16 *ptr, int stride)
{ return (npyv_s16)npyv_loadn_u16((const npy_uint16 *)ptr, stride); }
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, int stride)
{
    return npyv_set_u32(
        ptr[stride * 0], ptr[stride * 1],
        ptr[stride * 2], ptr[stride * 3]
    );
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, int stride)
{ return (npyv_s32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, int stride)
{ return (npyv_f32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }
//// 64
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, int stride)
{ return npyv_set_u64(ptr[0], ptr[stride]); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, int stride)
{ return npyv_set_s64(ptr[0], ptr[stride]); }
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, int stride)
{ return npyv_set_f64(ptr[0], ptr[stride]); }
#endif
// non-contiguous store
//// 8
NPY_FINLINE void npyv_storen_u8(npy_uint8 *ptr, int stride, npyv_u8 a)
{
    #define NPYV_IMPL_NEON_STOREN8(I) \
    { \
        unsigned e = vgetq_lane_u32((npyv_u32)a, I/4); \
        ptr[stride*(I+0)] = (npy_uint8)e; \
        ptr[stride*(I+1)] = (npy_uint8)(e >> 8); \
        ptr[stride*(I+2)] = (npy_uint8)(e >> 16); \
        ptr[stride*(I+3)] = (npy_uint8)(e >> 24); \
    }
    NPYV_IMPL_NEON_STOREN8(0)
    NPYV_IMPL_NEON_STOREN8(4)
    NPYV_IMPL_NEON_STOREN8(8)
    NPYV_IMPL_NEON_STOREN8(12)
}
NPY_FINLINE void npyv_storen_s8(npy_int8 *ptr, int stride, npyv_s8 a)
{ npyv_storen_u8((npy_uint8*)ptr, stride, (npyv_u8)a); }
//// 16
NPY_FINLINE void npyv_storen_u16(npy_uint16 *ptr, int stride, npyv_u16 a)
{
    #define NPYV_IMPL_NEON_STOREN16(I) \
    { \
        unsigned e = vgetq_lane_u32((npyv_u32)a, I/2); \
        ptr[stride*(I+0)] = (npy_uint16)e; \
        ptr[stride*(I+1)] = (npy_uint16)(e >> 16); \
    }
    NPYV_IMPL_NEON_STOREN16(0)
    NPYV_IMPL_NEON_STOREN16(2)
    NPYV_IMPL_NEON_STOREN16(4)
    NPYV_IMPL_NEON_STOREN16(6)
}
NPY_FINLINE void npyv_storen_s16(npy_int16 *ptr, int stride, npyv_s16 a)
{ npyv_storen_u16((npy_uint16*)ptr, stride, (npyv_u16)a); }
//// 32
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, int stride, npyv_u32 a)
{
    ptr[stride * 0] = vgetq_lane_u32(a, 0);
    ptr[stride * 1] = vgetq_lane_u32(a, 1);
    ptr[stride * 2] = vgetq_lane_u32(a, 2);
    ptr[stride * 3] = vgetq_lane_u32(a, 3);
}
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, int stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, int stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
//// 64
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, int stride, npyv_u64 a)
{
    ptr[0] = vgetq_lane_u64(a, 0);
    ptr[1] = vgetq_lane_u64(a, 1);
}
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, int stride, npyv_s64 a)
{
    ptr[0] = vgetq_lane_s64(a, 0);
    ptr[1] = vgetq_lane_s64(a, 1);
}
#if NPY_SIMD_F64
NPY_FINLINE void npyv_storen_f64(double *ptr, int stride, npyv_f64 a)
{
    ptr[0] = vgetq_lane_f64(a, 0);
    ptr[1] = vgetq_lane_f64(a, 1);
}
#endif
#endif // _NPY_SIMD_NEON_MEMORY_H
