#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_MEMORY_H
#define _NPY_SIMD_RVV_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
// GCC requires literal type definitions for pointers types otherwise it causes ambiguous errors
// uint8_t
NPY_FINLINE npyv_u8 npyv_load_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, 16); }

NPY_FINLINE npyv_u8 npyv_loada_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, 16); }

NPY_FINLINE npyv_u8 npyv_loads_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, 16); }

NPY_FINLINE npyv_u8 npyv_loadl_u8(const npyv_lanetype_u8 *ptr)
{
    return __riscv_vslideup_vx_u8m1(__riscv_vle8_v_u8m1((const uint8_t*)ptr, 8), __riscv_vmv_v_x_u8m1(0, 8), 8, 16);
}

NPY_FINLINE void npyv_store_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_storea_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_stores_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_storel_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storeh_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, __riscv_vslidedown_vx_u8m1(vec, 8, 16), 8); }


// int8_t
NPY_FINLINE npyv_s8 npyv_load_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, 16); }

NPY_FINLINE npyv_s8 npyv_loada_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, 16); }

NPY_FINLINE npyv_s8 npyv_loads_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, 16); }

NPY_FINLINE npyv_s8 npyv_loadl_s8(const npyv_lanetype_s8 *ptr)
{
    return __riscv_vslideup_vx_i8m1(__riscv_vle8_v_i8m1((const int8_t*)ptr, 8), __riscv_vmv_v_x_i8m1(0, 8), 8, 16);
}

NPY_FINLINE void npyv_store_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_storea_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_stores_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, 16); }

NPY_FINLINE void npyv_storel_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storeh_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, __riscv_vslidedown_vx_i8m1(vec, 8, 16), 8); }

// uint16_t
NPY_FINLINE npyv_u16 npyv_load_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, 8); }

NPY_FINLINE npyv_u16 npyv_loada_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, 8); }

NPY_FINLINE npyv_u16 npyv_loads_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, 8); }

NPY_FINLINE npyv_u16 npyv_loadl_u16(const npyv_lanetype_u16 *ptr)
{
    return __riscv_vslideup_vx_u16m1(
        __riscv_vle16_v_u16m1((const uint16_t*)ptr, 4), __riscv_vmv_v_x_u16m1(0, 4), 4, 8
    );
}

NPY_FINLINE void npyv_store_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storea_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_stores_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storel_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storeh_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, __riscv_vslidedown_vx_u16m1(vec, 4, 8), 4); }

// int16_t
NPY_FINLINE npyv_s16 npyv_load_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, 8); }

NPY_FINLINE npyv_s16 npyv_loada_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, 8); }

NPY_FINLINE npyv_s16 npyv_loads_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, 8); }

NPY_FINLINE npyv_s16 npyv_loadl_s16(const npyv_lanetype_s16 *ptr)
{
    return __riscv_vslideup_vx_i16m1(
        __riscv_vle16_v_i16m1((const int16_t*)ptr, 4), __riscv_vmv_v_x_i16m1(0, 4), 4, 8
    );
}

NPY_FINLINE void npyv_store_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storea_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_stores_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, 8); }

NPY_FINLINE void npyv_storel_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storeh_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, __riscv_vslidedown_vx_i16m1(vec, 4, 8), 4); }

// uint32_t
NPY_FINLINE npyv_u32 npyv_load_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, 4); }

NPY_FINLINE npyv_u32 npyv_loada_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, 4); }

NPY_FINLINE npyv_u32 npyv_loads_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, 4); }

NPY_FINLINE npyv_u32 npyv_loadl_u32(const npyv_lanetype_u32 *ptr)
{
    return __riscv_vslideup_vx_u32m1(
        __riscv_vle32_v_u32m1((const uint32_t*)ptr, 2), __riscv_vmv_v_x_u32m1(0, 2), 2, 4
    );
}

NPY_FINLINE void npyv_store_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storea_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_stores_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storel_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storeh_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, __riscv_vslidedown_vx_u32m1(vec, 2, 4), 2); }

// int32_t 
NPY_FINLINE npyv_s32 npyv_load_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, 4); }

NPY_FINLINE npyv_s32 npyv_loada_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, 4); }

NPY_FINLINE npyv_s32 npyv_loads_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, 4); }

NPY_FINLINE npyv_s32 npyv_loadl_s32(const npyv_lanetype_s32 *ptr)
{
    return __riscv_vslideup_vx_i32m1(
        __riscv_vle32_v_i32m1((const int32_t*)ptr, 2), __riscv_vmv_v_x_i32m1(0, 2), 2, 4
    );
}

NPY_FINLINE void npyv_store_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storea_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_stores_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, 4); }

NPY_FINLINE void npyv_storel_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storeh_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, __riscv_vslidedown_vx_i32m1(vec, 2, 4), 2); }

// uint64_t 
NPY_FINLINE npyv_u64 npyv_load_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, 2); }

NPY_FINLINE npyv_u64 npyv_loada_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, 2); }

NPY_FINLINE npyv_u64 npyv_loads_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, 2); }

NPY_FINLINE npyv_u64 npyv_loadl_u64(const npyv_lanetype_u64 *ptr)
{
    return __riscv_vslideup_vx_u64m1(
        __riscv_vle64_v_u64m1((const uint64_t*)ptr, 1), __riscv_vmv_v_x_u64m1(0, 1), 1, 2
    );
}

NPY_FINLINE void npyv_store_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storea_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_stores_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storel_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, 1); }

NPY_FINLINE void npyv_storeh_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, __riscv_vslidedown_vx_u64m1(vec, 1, 2), 1); }

// int64_t 
NPY_FINLINE npyv_s64 npyv_load_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, 2); }

NPY_FINLINE npyv_s64 npyv_loada_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, 2); }

NPY_FINLINE npyv_s64 npyv_loads_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, 2); }

NPY_FINLINE npyv_s64 npyv_loadl_s64(const npyv_lanetype_s64 *ptr)
{
    return __riscv_vslideup_vx_i64m1(
        __riscv_vle64_v_i64m1((const int64_t*)ptr, 1), __riscv_vmv_v_x_i64m1(0, 1), 1, 2
    );
}

NPY_FINLINE void npyv_store_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storea_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_stores_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, 2); }

NPY_FINLINE void npyv_storel_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, 1); }

NPY_FINLINE void npyv_storeh_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, __riscv_vslidedown_vx_i64m1(vec, 1, 2), 1); }

// float 
NPY_FINLINE npyv_f32 npyv_load_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, 4); }

NPY_FINLINE npyv_f32 npyv_loada_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, 4); }

NPY_FINLINE npyv_f32 npyv_loads_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, 4); }

NPY_FINLINE npyv_f32 npyv_loadl_f32(const npyv_lanetype_f32 *ptr)
{
    return __riscv_vslideup_vx_f32m1(
        __riscv_vle32_v_f32m1((const float*)ptr, 2), __riscv_vfmv_v_f_f32m1(0, 2), 2, 4
    );
}

NPY_FINLINE void npyv_store_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, 4); }

NPY_FINLINE void npyv_storea_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, 4); }

NPY_FINLINE void npyv_stores_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, 4); }

NPY_FINLINE void npyv_storel_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, 2); }

NPY_FINLINE void npyv_storeh_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, __riscv_vslidedown_vx_f32m1(vec, 2, 4), 2); }


// double 
NPY_FINLINE npyv_f64 npyv_load_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, 2); }

NPY_FINLINE npyv_f64 npyv_loada_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, 2); }

NPY_FINLINE npyv_f64 npyv_loads_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, 2); }

NPY_FINLINE npyv_f64 npyv_loadl_f64(const npyv_lanetype_f64 *ptr)
{
    return __riscv_vslideup_vx_f64m1(
        __riscv_vle64_v_f64m1((const double*)ptr, 1), __riscv_vfmv_v_f_f64m1(0, 1), 1, 2
    );
}

NPY_FINLINE void npyv_store_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, 2); }

NPY_FINLINE void npyv_storea_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, 2); }

NPY_FINLINE void npyv_stores_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, 2); }

NPY_FINLINE void npyv_storel_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, 1); }

NPY_FINLINE void npyv_storeh_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, __riscv_vslidedown_vx_f64m1(vec, 1, 2), 1); }


/***************************
 * Non-contiguous Load
 ***************************/
NPY_FINLINE npyv_s32 vld1q_lane_s32(const int32_t *a, npyv_s32 b, const int lane) {
    vbool32_t mask = __riscv_vreinterpret_v_u8m1_b32(__riscv_vmv_v_x_u8m1((uint8_t)(1 << lane), 8));
    npyv_s32 a_dup = __riscv_vmv_v_x_i32m1(a[0], 4);
    return __riscv_vmerge_vvm_i32m1(b, a_dup, mask, 4);
}

NPY_FINLINE void vst1q_lane_s32(int32_t *a, npyv_s32 b, const int lane) {
    npyv_s32 b_s = __riscv_vslidedown_vx_i32m1(b, lane, 4);
    *a = __riscv_vmv_x_s_i32m1_i32(b_s);
}

NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    npyv_s32 a = __riscv_vmv_v_x_i32m1(0, 4);
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

NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{
    return __riscv_vslideup_vx_i64m1(
        __riscv_vle64_v_i64m1((const int64_t*)ptr, 1), __riscv_vle64_v_i64m1((const int64_t*)ptr + stride, 1), 1, 2
    );
}
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    return npyv_reinterpret_u64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}

NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    return npyv_reinterpret_f64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    return __riscv_vslideup_vx_u32m1(
        __riscv_vle32_v_u32m1((const uint32_t*)ptr, 2), __riscv_vle32_v_u32m1((const uint32_t*)ptr + stride, 2), 2, 4
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

NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ (void)stride; return npyv_load_f64(ptr); }

/***************************
 * Non-contiguous Store
 ***************************/
NPY_FINLINE npyv_s32 vld1_lane_s32(const int32_t *a, npyv_s32 b, const int lane) {
    vbool32_t mask = __riscv_vreinterpret_v_u8m1_b32(__riscv_vmv_v_x_u8m1((uint8_t)(1 << lane), 8));
    npyv_s32 a_dup = __riscv_vmv_v_x_i32m1(a[0], 2);
    return __riscv_vmerge_vvm_i32m1(b, a_dup, mask, 2);
}

NPY_FINLINE npyv_u32 vld1_lane_u32(const uint32_t *a, npyv_u32 b, const int lane) {
    vbool32_t mask = __riscv_vreinterpret_v_u8m1_b32(__riscv_vmv_v_x_u8m1((uint8_t)(1 << lane), 8));
    npyv_u32 a_dup = __riscv_vmv_v_x_u32m1(a[0], 2);
    return __riscv_vmerge_vvm_u32m1(b, a_dup, mask, 2);
}

NPY_FINLINE void vst1q_lane_u64(uint64_t *a, npyv_u64 b, const int c) {
    npyv_u64 b_s = __riscv_vslidedown_vx_u64m1(b, c, 1);
    *a = __riscv_vmv_x_s_u64m1_u64(b_s);
}

NPY_FINLINE void vst1q_lane_s64(int64_t *a, npyv_s64 b, const int c) {
    npyv_s64 b_s = __riscv_vslidedown_vx_i64m1(b, c, 1);
    *a = __riscv_vmv_x_s_i64m1_i64(b_s);
}

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

NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    vst1q_lane_s64((int64_t*)ptr, a, 0);
    vst1q_lane_s64((int64_t*)ptr + stride, a, 1);
}

NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, npyv_reinterpret_s64_u64(a)); }

NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, npyv_reinterpret_s64_f64(a)); }


//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    vst1q_lane_u64((uint64_t*)ptr, npyv_reinterpret_u64_u32(a), 0);
    vst1q_lane_u64((uint64_t*)(ptr + stride), npyv_reinterpret_u64_u32(a), 1);
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

NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ (void)stride; npyv_store_f64(ptr, a); }

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
        a = vld1q_lane_s32((const int32_t*)ptr, __riscv_vmv_v_x_i32m1(fill, 4), 0);
        break;
    case 2:
        a = __riscv_vslideup_vx_i32m1(__riscv_vle32_v_i32m1((const int32_t*)ptr, 2), __riscv_vmv_v_x_i32m1(fill, 2), 2, 4);
        break;
    case 3:
        a = __riscv_vslideup_vx_i32m1(
            __riscv_vle32_v_i32m1((const int32_t*)ptr, 2),
            vld1_lane_s32((const int32_t*)ptr + 2, __riscv_vmv_v_x_i32m1(fill, 2), 0), 2, 4
        );
        break;
    default:
        return npyv_load_s32(ptr);
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = a;
    a = __riscv_vor_vv_i32m1(workaround, a, 4);
#endif
    return a;
}

// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_till_s32(ptr, nlane, 0); }

NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s64 a = __riscv_vslideup_vx_i64m1(__riscv_vle64_v_i64m1((const int64_t*)ptr, 1), __riscv_vmv_v_x_i64m1(fill, 1), 1, 2);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s64 workaround = a;
        a = __riscv_vor_vv_i64m1(workaround, a, 2);
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
        npyv_s32 a = __riscv_vslideup_vx_i32m1(__riscv_vle32_v_i32m1((const int32_t*)ptr, 2), __riscv_vle32_v_i32m1(fill, 2), 2, 4);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = __riscv_vor_vv_i32m1(workaround, a, 4);
    #endif
        return a;
    }
    return npyv_load_s32(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return __riscv_vreinterpret_v_i64m1_i32m1(npyv_load_tillz_s64((const npy_int64*)ptr, nlane)); }

//// 128-bit nlane
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{ (void)nlane; (void)fill_lo; (void)fill_hi; return npyv_load_s64(ptr); }

NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ (void)nlane; return npyv_load_s64(ptr); }

/*********************************
 * Non-contiguous partial load
 *********************************/
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    npyv_s32 vfill = __riscv_vmv_v_x_i32m1(fill, 4);
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
    vfill = __riscv_vor_vv_i32m1(workaround, vfill, 4);
#endif
    return vfill;
}

NPY_FINLINE npyv_s32 npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

NPY_FINLINE npyv_s64 npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
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
        npyv_s32 a = __riscv_vslideup_vx_i32m1(__riscv_vle32_v_i32m1((const int32_t*)ptr, 2), __riscv_vle32_v_i32m1(fill, 2), 2, 4);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = __riscv_vor_vv_i32m1(workaround, a, 4);
    #endif
        return a;
    }
    return npyv_loadn2_s32(ptr, stride);
}

NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s32 a = __riscv_vslideup_vx_i32m1(__riscv_vle32_v_i32m1((const int32_t*)ptr, 2), __riscv_vmv_v_x_i32m1(0, 2), 2, 4);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a;
        a = __riscv_vor_vv_i32m1(workaround, a, 4);
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
        __riscv_vse32_v_i32m1((int32_t*)ptr, a, 2);
        break;
    case 3:
        __riscv_vse32_v_i32m1((int32_t*)ptr, a, 2);
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
        vst1q_lane_s64((int64_t*)ptr, npyv_reinterpret_s64_s32(a), 0);
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
    vst1q_lane_s64((int64_t*)ptr, npyv_reinterpret_s64_s32(a), 0);
    if (nlane > 1) {
        vst1q_lane_s64((int64_t*)(ptr + stride), npyv_reinterpret_s64_s32(a), 1);
    }
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ assert(nlane > 0); (void)stride; (void)nlane; npyv_store_s64(ptr, a); }

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
#define NPYV_IMPL_RVV_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                     \
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

NPYV_IMPL_RVV_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride
#define NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                \
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

NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interleave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_RVV_MEM_INTERLEAVE(SFX)                                \
    NPY_FINLINE npyv_##SFX##x2 npyv_zip_##SFX(npyv_##SFX, npyv_##SFX);   \
    NPY_FINLINE npyv_##SFX##x2 npyv_unzip_##SFX(npyv_##SFX, npyv_##SFX); \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                      \
        const npyv_lanetype_##SFX *ptr                                   \
    ) {                                                                  \
        return npyv_unzip_##SFX(                                         \
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX) \
        );                                                               \
    }                                                                    \
    NPY_FINLINE void npyv_store_##SFX##x2(                               \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                       \
    ) {                                                                  \
        npyv_##SFX##x2 zip = npyv_zip_##SFX(v.val[0], v.val[1]);         \
        npyv_store_##SFX(ptr, zip.val[0]);                               \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);           \
    }

NPYV_IMPL_RVV_MEM_INTERLEAVE(u8)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s8)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u16)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s16)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u64)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s64)
NPYV_IMPL_RVV_MEM_INTERLEAVE(f32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(f64)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of uint32.
NPY_FINLINE npyv_u32 vcreate_u32(uint64_t a) {
    return __riscv_vreinterpret_v_u64m1_u32m1(
      __riscv_vreinterpret_v_i64m1_u64m1(__riscv_vreinterpret_v_u64m1_i64m1(__riscv_vmv_v_x_u64m1(a, 8))));
}

NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    const unsigned i0 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(idx, 0, 4));
    const unsigned i1 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(idx, 1, 4));
    const unsigned i2 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(idx, 2, 4));
    const unsigned i3 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(idx, 3, 4));

    npyv_u32 low = vcreate_u32(table[i0]);
               low = vld1_lane_u32((const uint32_t*)table + i1, low, 1);
    npyv_u32 high = vcreate_u32(table[i2]);
               high = vld1_lane_u32((const uint32_t*)table + i3, high, 1);
    return __riscv_vslideup_vx_u32m1(low, high, 2, 4);
}

NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }

NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return npyv_reinterpret_f32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of uint64.
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{
    const unsigned i0 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(__riscv_vreinterpret_v_u64m1_u32m1(idx), 0, 4));
    const unsigned i1 = __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(__riscv_vreinterpret_v_u64m1_u32m1(idx), 2, 4));
    return __riscv_vslideup_vx_u64m1(
        __riscv_vle64_v_u64m1((const uint64_t*)table + i0, 1),
        __riscv_vle64_v_u64m1((const uint64_t*)table + i1, 1), 1, 2
    );
}

NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }

NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return npyv_reinterpret_f64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }

#endif // _NPY_SIMD_RVV_MEMORY_H
