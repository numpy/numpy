#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_MEMORY_H
#define _NPY_SIMD_VEC_MEMORY_H

#include "misc.h"

/****************************
 * Private utilities
 ****************************/
// TODO: test load by cast
#define VSX__CAST_lOAD 0
#if VSX__CAST_lOAD
    #define npyv__load(T_VEC, PTR) (*((T_VEC*)(PTR)))
#else
    /**
     * CLANG fails to load unaligned addresses via vec_xl, vec_xst
     * so we failback to vec_vsx_ld, vec_vsx_st
     */
    #if defined (NPY_HAVE_VSX2) && ( \
        (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__)) \
    )
        #define npyv__load(T_VEC, PTR) vec_vsx_ld(0, PTR)
    #else // VX
        #define npyv__load(T_VEC, PTR) vec_xl(0, PTR)
    #endif
#endif
// unaligned store
#if defined (NPY_HAVE_VSX2) && ( \
    (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__)) \
)
    #define npyv__store(PTR, VEC) vec_vsx_st(VEC, 0, PTR)
#else // VX
    #define npyv__store(PTR, VEC) vec_xst(VEC, 0, PTR)
#endif

// aligned load/store
#if defined (NPY_HAVE_VSX)
    #define npyv__loada(PTR) vec_ld(0, PTR)
    #define npyv__storea(PTR, VEC) vec_st(VEC, 0, PTR)
#else // VX
    #define npyv__loada(PTR) vec_xl(0, PTR)
    #define npyv__storea(PTR, VEC) vec_xst(VEC, 0, PTR)
#endif

// avoid aliasing rules
NPY_FINLINE npy_uint64 *npyv__ptr2u64(const void *ptr)
{ npy_uint64 *ptr64 = (npy_uint64*)ptr; return ptr64; }

// load lower part
NPY_FINLINE npyv_u64 npyv__loadl(const void *ptr)
{
#ifdef NPY_HAVE_VSX
    #if defined(__clang__) && !defined(__IBMC__)
        // vec_promote doesn't support doubleword on clang
        return npyv_setall_u64(*npyv__ptr2u64(ptr));
    #else
        return vec_promote(*npyv__ptr2u64(ptr), 0);
    #endif
#else // VX
    return vec_load_len((const unsigned long long*)ptr, 7);
#endif
}
// store lower part
#define npyv__storel(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 0)

#define npyv__storeh(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 1)

/****************************
 * load/store
 ****************************/
#define NPYV_IMPL_VEC_MEM(SFX, DW_CAST)                                                 \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const npyv_lanetype_##SFX *ptr)              \
    { return (npyv_##SFX)npyv__load(npyv_##SFX, (const npyv_lanetype_##DW_CAST*)ptr); } \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return (npyv_##SFX)npyv__loada((const npyv_lanetype_u32*)ptr); }                  \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return npyv_loada_##SFX(ptr); }                                                   \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return (npyv_##SFX)npyv__loadl(ptr); }                                            \
    NPY_FINLINE void npyv_store_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)         \
    { npyv__store((npyv_lanetype_##DW_CAST*)ptr, (npyv_##DW_CAST)vec); }                \
    NPY_FINLINE void npyv_storea_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)        \
    { npyv__storea((npyv_lanetype_u32*)ptr, (npyv_u32)vec); }                           \
    NPY_FINLINE void npyv_stores_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)        \
    { npyv_storea_##SFX(ptr, vec); }                                                    \
    NPY_FINLINE void npyv_storel_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)        \
    { npyv__storel(ptr, vec); }                                                         \
    NPY_FINLINE void npyv_storeh_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)        \
    { npyv__storeh(ptr, vec); }

NPYV_IMPL_VEC_MEM(u8,  u8)
NPYV_IMPL_VEC_MEM(s8,  s8)
NPYV_IMPL_VEC_MEM(u16, u16)
NPYV_IMPL_VEC_MEM(s16, s16)
NPYV_IMPL_VEC_MEM(u32, u32)
NPYV_IMPL_VEC_MEM(s32, s32)
NPYV_IMPL_VEC_MEM(u64, f64)
NPYV_IMPL_VEC_MEM(s64, f64)
#if NPY_SIMD_F32
NPYV_IMPL_VEC_MEM(f32, f32)
#endif
NPYV_IMPL_VEC_MEM(f64, f64)

/***************************
 * Non-contiguous Load
 ***************************/
//// 32
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    return npyv_set_u32(
        ptr[stride * 0], ptr[stride * 1],
        ptr[stride * 2], ptr[stride * 3]
    );
}
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return (npyv_s32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }
#if NPY_SIMD_F32
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return (npyv_f32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }
#endif
//// 64
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_set_u64(ptr[0], ptr[stride]); }
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_set_s64(ptr[0], ptr[stride]); }
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return npyv_set_f64(ptr[0], ptr[stride]); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return (npyv_u32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return (npyv_s32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }
#if NPY_SIMD_F32
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return (npyv_f32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }
#endif
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
//// 32
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    ptr[stride * 0] = vec_extract(a, 0);
    ptr[stride * 1] = vec_extract(a, 1);
    ptr[stride * 2] = vec_extract(a, 2);
    ptr[stride * 3] = vec_extract(a, 3);
}
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
#if NPY_SIMD_F32
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
#endif
//// 64
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    ptr[stride * 0] = vec_extract(a, 0);
    ptr[stride * 1] = vec_extract(a, 1);
}
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, (npyv_u64)a); }
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen_u64((npy_uint64*)ptr, stride, (npyv_u64)a); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    *(npy_uint64*)ptr = vec_extract((npyv_u64)a, 0);
    *(npy_uint64*)(ptr + stride) = vec_extract((npyv_u64)a, 1);
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
#if NPY_SIMD_F32
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
#endif
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
    npyv_s32 vfill = npyv_setall_s32(fill);
#ifdef NPY_HAVE_VX
    const unsigned blane = (nlane > 4) ? 4 : nlane;
    const npyv_u32 steps = npyv_set_u32(0, 1, 2, 3);
    const npyv_u32 vlane = npyv_setall_u32(blane);
    const npyv_b32 mask  = vec_cmpgt(vlane, steps);
    npyv_s32 a = vec_load_len(ptr, blane*4-1);
    a = vec_sel(vfill, a, mask);
#else
    npyv_s32 a;
    switch(nlane) {
    case 1:
        a = vec_insert(ptr[0], vfill, 0);
        break;
    case 2:
        a = (npyv_s32)vec_insert(
            *npyv__ptr2u64(ptr), (npyv_u64)vfill, 0
        );
        break;
    case 3:
        vfill = vec_insert(ptr[2], vfill, 2);
        a = (npyv_s32)vec_insert(
            *npyv__ptr2u64(ptr), (npyv_u64)vfill, 0
        );
        break;
    default:
        return npyv_load_s32(ptr);
    }
#endif
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = a;
    a = vec_or(workaround, a);
#endif
    return a;
}
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
#ifdef NPY_HAVE_VX
    unsigned blane = (nlane > 4) ? 4 : nlane;
    return vec_load_len(ptr, blane*4-1);
#else
    return npyv_load_till_s32(ptr, nlane, 0);
#endif
}
//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s64 r = npyv_set_s64(ptr[0], fill);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s64 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
    }
    return npyv_load_s64(ptr);
}
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
#ifdef NPY_HAVE_VX
    unsigned blane = (nlane > 2) ? 2 : nlane;
    return vec_load_len((const signed long long*)ptr, blane*8-1);
#else
    return npyv_load_till_s64(ptr, nlane, 0);
#endif
}
//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s32 r = npyv_set_s32(ptr[0], ptr[1], fill_lo, fill_hi);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
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
    npyv_s32 vfill = npyv_setall_s32(fill);
    switch(nlane) {
    case 3:
        vfill = vec_insert(ptr[stride*2], vfill, 2);
    case 2:
        vfill = vec_insert(ptr[stride], vfill, 1);
    case 1:
        vfill = vec_insert(*ptr, vfill, 0);
        break;
    default:
        return npyv_loadn_s32(ptr, stride);
    } // switch
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = vfill;
    vfill = vec_or(workaround, vfill);
#endif
    return vfill;
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
    if (nlane == 1) {
        return npyv_load_till_s64(ptr, nlane, fill);
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
        npyv_s32 r = npyv_set_s32(ptr[0], ptr[1], fill_lo, fill_hi);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
    }
    return npyv_loadn2_s32(ptr, stride);
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_s32 r = (npyv_s32)npyv_set_s64(*(npy_int64*)ptr, 0);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
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
#ifdef NPY_HAVE_VX
    unsigned blane = (nlane > 4) ? 4 : nlane;
    vec_store_len(a, ptr, blane*4-1);
#else
    switch(nlane) {
    case 1:
        *ptr = vec_extract(a, 0);
        break;
    case 2:
        npyv_storel_s32(ptr, a);
        break;
    case 3:
        npyv_storel_s32(ptr, a);
        ptr[2] = vec_extract(a, 2);
        break;
    default:
        npyv_store_s32(ptr, a);
    }
#endif
}
//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
#ifdef NPY_HAVE_VX
    unsigned blane = (nlane > 2) ? 2 : nlane;
    vec_store_len(a, (signed long long*)ptr, blane*8-1);
#else
    if (nlane == 1) {
        npyv_storel_s64(ptr, a);
        return;
    }
    npyv_store_s64(ptr, a);
#endif
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, (npyv_s64)a); }

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
    ptr[stride*0] = vec_extract(a, 0);
    switch(nlane) {
    case 1:
        return;
    case 2:
        ptr[stride*1] = vec_extract(a, 1);
        return;
    case 3:
        ptr[stride*1] = vec_extract(a, 1);
        ptr[stride*2] = vec_extract(a, 2);
        return;
    default:
         ptr[stride*1] = vec_extract(a, 1);
         ptr[stride*2] = vec_extract(a, 2);
         ptr[stride*3] = vec_extract(a, 3);
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == 1) {
        npyv_storel_s64(ptr, a);
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
#define NPYV_IMPL_VEC_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
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

NPYV_IMPL_VEC_REST_PARTIAL_TYPES(u32, s32)
#if NPY_SIMD_F32
NPYV_IMPL_VEC_REST_PARTIAL_TYPES(f32, s32)
#endif
NPYV_IMPL_VEC_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_VEC_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride
#define NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                 \
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

NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(u32, s32)
#if NPY_SIMD_F32
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(f32, s32)
#endif
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_VEC_MEM_INTERLEAVE(SFX)                                \
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

NPYV_IMPL_VEC_MEM_INTERLEAVE(u8)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s8)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u16)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s16)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u32)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s32)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u64)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s64)
#if NPY_SIMD_F32
NPYV_IMPL_VEC_MEM_INTERLEAVE(f32)
#endif
NPYV_IMPL_VEC_MEM_INTERLEAVE(f64)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    const unsigned i0 = vec_extract(idx, 0);
    const unsigned i1 = vec_extract(idx, 1);
    const unsigned i2 = vec_extract(idx, 2);
    const unsigned i3 = vec_extract(idx, 3);
    npyv_u32 r = vec_promote(table[i0], 0);
             r = vec_insert(table[i1], r, 1);
             r = vec_insert(table[i2], r, 2);
             r = vec_insert(table[i3], r, 3);
    return r;
}
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return (npyv_s32)npyv_lut32_u32((const npy_uint32*)table, idx); }
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
    { return (npyv_f32)npyv_lut32_u32((const npy_uint32*)table, idx); }
#endif
// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
#ifdef NPY_HAVE_VX
    const unsigned i0 = vec_extract((npyv_u32)idx, 1);
    const unsigned i1 = vec_extract((npyv_u32)idx, 3);
#else
    const unsigned i0 = vec_extract((npyv_u32)idx, 0);
    const unsigned i1 = vec_extract((npyv_u32)idx, 2);
#endif
    npyv_f64 r = vec_promote(table[i0], 0);
             r = vec_insert(table[i1], r, 1);
    return r;
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_VEC_MEMORY_H
