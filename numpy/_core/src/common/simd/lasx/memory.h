#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_MEMORY_H
#define _NPY_SIMD_LASX_MEMORY_H

#include <stdint.h>
#include "misc.h"

/***************************
 * load/store
 ***************************/
#define NPYV_IMPL_LASX_MEM(SFX, CTYPE)                                \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)          \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)         \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)         \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                     \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)         \
    { return (npyv_##SFX)__lasx_xvpermi_q(__lasx_xvld(ptr, 0), __lasx_xvldi(0), 0x22); } \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)     \
    { __lasx_xvst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)    \
    { __lasx_xvst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)    \
    { __lasx_xvst(vec, ptr, 0); }                                     \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)    \
    { __lasx_xvstelm_d(vec, ptr, 0, 0);                               \
      __lasx_xvstelm_d(vec, ptr, 8, 1); }                             \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)    \
    { __lasx_xvstelm_d(vec, ptr, 0, 2);                               \
      __lasx_xvstelm_d(vec, ptr, 8, 3); }

NPYV_IMPL_LASX_MEM(u8,  npy_uint8)
NPYV_IMPL_LASX_MEM(s8,  npy_int8)
NPYV_IMPL_LASX_MEM(u16, npy_uint16)
NPYV_IMPL_LASX_MEM(s16, npy_int16)
NPYV_IMPL_LASX_MEM(u32, npy_uint32)
NPYV_IMPL_LASX_MEM(s32, npy_int32)
NPYV_IMPL_LASX_MEM(u64, npy_uint64)
NPYV_IMPL_LASX_MEM(s64, npy_int64)
NPYV_IMPL_LASX_MEM(f32, float)
NPYV_IMPL_LASX_MEM(f64, double)

/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    __m256i a = __lasx_xvld(ptr, 0);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride], 1);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*2], 2);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*3], 3);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*4], 4);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*5], 5);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*6], 6);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*7], 7);
    return a;
}
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)  //ok
{ return npyv_reinterpret_f32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
//// 64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    __m256i a1 = __lasx_xvld(ptr, 0);
    __m256i a2 = __lasx_xvld(ptr + stride, 0);
    __m256i a3 = __lasx_xvld(ptr + stride*2, 0);
    __m256i a4 = __lasx_xvld(ptr + stride*3, 0);
    __m256i a12 = __lasx_xvilvl_d(a2, a1);
    __m256i a34 = __lasx_xvilvl_d(a4, a3);
            a1 = __lasx_xvpermi_q(a34, a12, 0x20);
    return (npyv_f64)a1;
}
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_reinterpret_u64_f64(npyv_loadn_f64((const double*)ptr, stride)); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_reinterpret_s64_f64(npyv_loadn_f64((const double*)ptr, stride)); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{
    __m256i t0 = __lasx_xvld((const double *)ptr, 0);
    __m256i t1 = __lasx_xvld((const double *)(ptr + stride), 0);
    __m256i t2 = __lasx_xvld((const double *)(ptr + stride*2), 0);
    __m256i t3 = __lasx_xvld((const double *)(ptr + stride*3), 0);
    __m256i r0 = __lasx_xvilvl_d(t1, t0);
    __m256i r1 = __lasx_xvilvl_d(t3, t2);
    return (npyv_f32)__lasx_xvpermi_q(r1, r0, 0x20);
}
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_f32(npyv_loadn2_f32((const float*)ptr, stride)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_reinterpret_s32_f32(npyv_loadn2_f32((const float*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld((ptr + stride), 0);
    return (npyv_f64)__lasx_xvpermi_q(t1, t0, 0x20);
}
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ (void)stride; return (npyv_u64)npyv_loadn2_f64((const double*)ptr, stride); }
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ (void)stride; return (npyv_s64)npyv_loadn2_f64((const double*)ptr, stride); }

/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    __lasx_xvstelm_w(a, ptr, 0, 0);
    __lasx_xvstelm_w(a, ptr + stride, 0, 1);
    __lasx_xvstelm_w(a, ptr + stride*2, 0, 2);
    __lasx_xvstelm_w(a, ptr + stride*3, 0, 3);
    __lasx_xvstelm_w(a, ptr + stride*4, 0, 4);
    __lasx_xvstelm_w(a, ptr + stride*5, 0, 5);
    __lasx_xvstelm_w(a, ptr + stride*6, 0, 6);
    __lasx_xvstelm_w(a, ptr + stride*7, 0, 7);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, (npyv_s32)a); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
  __lasx_xvstelm_d(a, ptr, 0, 0);
  __lasx_xvstelm_d(a, ptr + stride, 0, 1);
  __lasx_xvstelm_d(a, ptr + stride*2, 0, 2);
  __lasx_xvstelm_d(a, ptr + stride*3, 0, 3);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
  __lasx_xvstelm_d(npyv_reinterpret_u64_u32(a), ptr, 0, 0);
  __lasx_xvstelm_d(npyv_reinterpret_u64_u32(a), ptr+stride, 0, 1);
  __lasx_xvstelm_d(npyv_reinterpret_u64_u32(a), ptr+stride*2, 0, 2);
  __lasx_xvstelm_d(npyv_reinterpret_u64_u32(a), ptr+stride*3, 0, 3);
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    npyv_storel_u64(ptr, a);
    npyv_storeh_u64(ptr + stride, a);
}
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, (npyv_u64)a); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, (npyv_u64)a); }
/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_setall_s32(fill);
    unsigned long *long_ptr = (unsigned long *)ptr;
    __m256i t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_w(vfill, *ptr, 0);
        case 2:
            return __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
        case 3:
            t = __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
            return __lasx_xvinsgr2vr_w(t, *(ptr+2), 2);
        case 4:
            t = __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
            return __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
        case 5:
            t = __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            return __lasx_xvinsgr2vr_w(t, *(ptr+4), 4);
        case 6:
            t = __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            return __lasx_xvinsgr2vr_d(t, *(long_ptr+2), 2);
        case 7:
            t = __lasx_xvinsgr2vr_d(vfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+2), 2);
            return __lasx_xvinsgr2vr_w(t, *(ptr+6), 6);
        default:
            return npyv_load_s32(ptr);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    __m256i zfill = __lasx_xvldi(0);
    unsigned long *long_ptr = (unsigned long *)ptr;
    __m256i t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_w(zfill, *ptr, 0);
        case 2:
            return __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
        case 3:
            t = __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
            return __lasx_xvinsgr2vr_w(t, *(ptr+2), 2);
        case 4:
            t = __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
            return __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
        case 5:
            t = __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            return __lasx_xvinsgr2vr_w(t, *(ptr+4), 4);
        case 6:
            t = __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            return __lasx_xvinsgr2vr_d(t, *(long_ptr+2), 2);
        case 7:
            t = __lasx_xvinsgr2vr_d(zfill, *long_ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+1), 1);
            t = __lasx_xvinsgr2vr_d(t, *(long_ptr+2), 2);
            return __lasx_xvinsgr2vr_w(t, *(ptr+6), 6);
        default:
            return npyv_load_s32(ptr);
    }
}
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    __m256i vfill = npyv_setall_s64(fill);
    __m256i t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
        case 2:
            t = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, ptr[1], 1);
        case 3:
            t = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
            t = __lasx_xvinsgr2vr_d(t, ptr[1], 1);
            return __lasx_xvinsgr2vr_d(t, ptr[2], 2);
        default:
            return npyv_load_s64(ptr);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    __m256i zfill = __lasx_xvldi(0);
    __m256i t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_d(zfill, ptr[0], 0);
        case 2:
            t = __lasx_xvinsgr2vr_d(zfill, ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, ptr[1], 1);
        case 3:
            t = __lasx_xvinsgr2vr_d(zfill, ptr[0], 0);
            t = __lasx_xvinsgr2vr_d(t, ptr[1], 1);
            return __lasx_xvinsgr2vr_d(t, ptr[2], 2);
        default:
            return npyv_load_s64(ptr);
    }
}

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    long *long_ptr = (long *)ptr;
    const __m256i vfill = npyv_set_s32(fill_lo, fill_hi, fill_lo, fill_hi,
                                       fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_d(vfill, long_ptr[0], 0);
        case 2:
            t = __lasx_xvinsgr2vr_d(vfill, long_ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, long_ptr[1], 1);
        case 3:
            t = __lasx_xvinsgr2vr_d(vfill, long_ptr[0], 0);
            t = __lasx_xvinsgr2vr_d(t, long_ptr[1], 1);
            return __lasx_xvinsgr2vr_d(t, long_ptr[2], 2);
        default:
            return npyv_load_s32(ptr);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return (npyv_s32)npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    const __m256i vfill = npyv_set_s64(0, 0, fill_lo, fill_hi);
    __m256i t;
    switch(nlane) {
        case 1:
            t = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, ptr[1], 1);
        default:
            return npyv_load_s64(ptr);
    }
}

NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    const __m256i zfill = __lasx_xvldi(0);
    __m256i t;
    switch(nlane) {
        case 1:
            t = __lasx_xvinsgr2vr_d(zfill, ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, ptr[1], 1);
        default:
            return npyv_load_s64(ptr);
    }
}

/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    __m256i vfill = npyv_setall_s32(fill);
    switch(nlane) {
        case 7:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*6], 6);
        case 6:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*5], 5);
        case 5:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*4], 4);
        case 4:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*3], 3);
        case 3:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*2], 2);
        case 2:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride], 1);
        case 1:
            vfill = __lasx_xvinsgr2vr_w(vfill, ptr[0], 0);
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
    npyv_s32 t;
    switch(nlane) {
        case 1:
            return __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
        case 2:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            return __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
        case 3:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
            return __lasx_xvinsgr2vr_w(t, ptr[stride*2], 2);
        case 4:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*2], 2);
            return __lasx_xvinsgr2vr_w(t, ptr[stride*3], 3);
        case 5:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*2], 2);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*3], 3);
            return __lasx_xvinsgr2vr_w(t, ptr[stride*4], 4);
        case 6:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*2], 2);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*3], 3);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*4], 4);
            return __lasx_xvinsgr2vr_w(t, ptr[stride*5], 5);;
        case 7:
            t = __lasx_xvinsgr2vr_w(__lasx_xvldi(0), ptr[0], 0);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride], 1);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*2], 2);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*3], 3);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*4], 4);
            t = __lasx_xvinsgr2vr_w(t, ptr[stride*5], 5);
            return __lasx_xvinsgr2vr_w(t, ptr[stride*6], 6);
        default:
            return npyv_loadn_s32(ptr, stride);
    }
}
//// 64
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    __m256i vfill = npyv_setall_s64(fill);
    __m256i t;
    switch (nlane) {
        case 1:
            return __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
        case 2:
            t = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
            return __lasx_xvinsgr2vr_d(t, ptr[stride], 1);
        case 3:
            t = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
            t = __lasx_xvinsgr2vr_d(t, ptr[stride], 1);
            return __lasx_xvinsgr2vr_d(t, ptr[stride*2], 2);
        default:
            return npyv_loadn_s64(ptr, stride);
    }
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    return npyv_loadn_till_s64(ptr, stride, nlane, 0);
}

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    __m256i vfill = npyv_set_s32(fill_lo, fill_hi, fill_lo, fill_hi,
                                 fill_lo, fill_hi, fill_lo, fill_hi);
    npyv_s32 t;
    switch (nlane) {
        case 1:
            return (npyv_s32)__lasx_xvinsgr2vr_d(vfill, *(long *)ptr, 0);
        case 2:
            t = __lasx_xvinsgr2vr_d(vfill, *(long *)ptr, 0);
            return (npyv_s32)__lasx_xvinsgr2vr_d(t, *(long *)(ptr + stride), 1);
        case 3:
            t = __lasx_xvinsgr2vr_d(vfill, *(long *)ptr, 0);
            t = __lasx_xvinsgr2vr_d(t, *(long *)(ptr + stride), 1);
            return (npyv_s32)__lasx_xvinsgr2vr_d(t, *(long *)(ptr + stride*2), 2);
        default:
            return npyv_loadn2_s32(ptr, stride);
    }
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    return npyv_loadn2_till_s32(ptr, stride, nlane, 0, 0);
}

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                                  npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    __m256i vfill = npyv_set_s64(fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i t, t1;
    switch (nlane) {
        case 1:
            t = __lasx_xvld(ptr, 0);
            return __lasx_xvpermi_q(vfill, t, 0x20);
        default:
            t = __lasx_xvld(ptr, 0);
            t1 = __lasx_xvld(ptr + stride, 0);
            return __lasx_xvpermi_q(t1, t, 0x20);
    }
}

NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s64(ptr, stride, nlane, 0, 0); }

/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    long *long_ptr = (long *)ptr;
    switch(nlane) {
        case 1:
            __lasx_xvstelm_w(a, ptr, 0, 0);
            break;
        case 2:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            break;
        case 3:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            __lasx_xvstelm_w(a, ptr, 8, 2);
            break;
        case 4:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            __lasx_xvstelm_d(a, long_ptr, 8, 1);
            break;
        case 5:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            __lasx_xvstelm_d(a, long_ptr, 8, 1);
            __lasx_xvstelm_w(a, ptr, 16, 4);
            break;
        case 6:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            __lasx_xvstelm_d(a, long_ptr, 8, 1);
            __lasx_xvstelm_d(a, long_ptr, 16, 2);
            break;
        case 7:
            __lasx_xvstelm_d(a, long_ptr, 0, 0);
            __lasx_xvstelm_d(a, long_ptr, 8, 1);
            __lasx_xvstelm_d(a, long_ptr, 16, 2);
            __lasx_xvstelm_w(a, ptr, 24, 6);
            break;
        default:
            npyv_store_s32(ptr, a);
            break;
    }
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    switch (nlane) {
        case 1:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            break;
        case 2:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            __lasx_xvstelm_d(a, ptr, 8, 1);
            break;
        case 3:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            __lasx_xvstelm_d(a, ptr, 8, 1);
            __lasx_xvstelm_d(a, ptr, 16, 2);
            break;
        default:
            npyv_store_s64(ptr, a);
            break;
    }
}
//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, a); }

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    switch (nlane) {
        case 1:
            npyv_storel_s64(ptr, a);
            return;
        default:
            npyv_store_s64(ptr, a);
            return;
    }
}

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    __lasx_xvstelm_w(a, ptr, 0, 0);
    switch(nlane) {
        case 1:
            return;
        case 2:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            return;
        case 3:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            return;
        case 4:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3);
            return;
        case 5:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3);
            ptr[stride*4] = __lasx_xvpickve2gr_w(a, 4);
            return;
        case 6:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3);
            ptr[stride*4] = __lasx_xvpickve2gr_w(a, 4);
            ptr[stride*5] = __lasx_xvpickve2gr_w(a, 5);
            return;
        case 7:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3);
            ptr[stride*4] = __lasx_xvpickve2gr_w(a, 4);
            ptr[stride*5] = __lasx_xvpickve2gr_w(a, 5);
            ptr[stride*6] = __lasx_xvpickve2gr_w(a, 6);
            return;
        default:
            ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3);
            ptr[stride*4] = __lasx_xvpickve2gr_w(a, 4);
            ptr[stride*5] = __lasx_xvpickve2gr_w(a, 5);
            ptr[stride*6] = __lasx_xvpickve2gr_w(a, 6);
            ptr[stride*7] = __lasx_xvpickve2gr_w(a, 7);
            return;
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    __lasx_xvstelm_d(a, ptr, 0, 0);
    switch(nlane) {
        case 1:
            return;
        case 2:
            ptr[stride*1] = __lasx_xvpickve2gr_d(a, 1);
            return;
        case 3:
            ptr[stride*1] = __lasx_xvpickve2gr_d(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_d(a, 2);
            return;
        default:
            ptr[stride*1] = __lasx_xvpickve2gr_d(a, 1);
            ptr[stride*2] = __lasx_xvpickve2gr_d(a, 2);
            ptr[stride*3] = __lasx_xvpickve2gr_d(a, 3);
            return;
    }
}

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    switch (nlane) {
        case 1:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            return;
        case 2:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            __lasx_xvstelm_d(a, ptr + stride, 0, 1);
            return;
        case 3:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            __lasx_xvstelm_d(a, ptr + stride*1, 0, 1);
            __lasx_xvstelm_d(a, ptr + stride*2, 0, 2);
            return;
        default:
            __lasx_xvstelm_d(a, ptr, 0, 0);
            __lasx_xvstelm_d(a, ptr + stride*1, 0, 1);
            __lasx_xvstelm_d(a, ptr + stride*2, 0, 2);
            __lasx_xvstelm_d(a, ptr + stride*3, 0, 3);
            return;
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
#define NPYV_IMPL_LASX_MEM_INTERLEAVE(SFX, ZSFX)                              \
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

NPYV_IMPL_LASX_MEM_INTERLEAVE(u8, uint8_t);
NPYV_IMPL_LASX_MEM_INTERLEAVE(s8, int8_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u16, uint16_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s16, int16_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u32, uint32_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s32, int32_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u64, uint64_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s64, int64_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(f32, float)
NPYV_IMPL_LASX_MEM_INTERLEAVE(f64, double)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const int i0 = __lasx_xvpickve2gr_wu(idx, 0);
    const int i1 = __lasx_xvpickve2gr_wu(idx, 1);
    const int i2 = __lasx_xvpickve2gr_wu(idx, 2);
    const int i3 = __lasx_xvpickve2gr_wu(idx, 3);
    const int i4 = __lasx_xvpickve2gr_wu(idx, 4);
    const int i5 = __lasx_xvpickve2gr_wu(idx, 5);
    const int i6 = __lasx_xvpickve2gr_wu(idx, 6);
    const int i7 = __lasx_xvpickve2gr_wu(idx, 7);
    return npyv_set_f32(table[i0], table[i1], table[i2], table[i3],
                        table[i4], table[i5], table[i6], table[i7]);
}
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    const int i0 = __lasx_xvpickve2gr_wu(idx, 0);
    const int i1 = __lasx_xvpickve2gr_wu(idx, 2);
    const int i2 = __lasx_xvpickve2gr_wu(idx, 4);
    const int i3 = __lasx_xvpickve2gr_wu(idx, 6);
    return npyv_set_f64(table[i0], table[i1], table[i2], table[i3]);
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_LASX_MEMORY_H
