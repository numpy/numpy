#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "misc.h"

#ifndef _NPY_SIMD_VSX_MEMORY_H
#define _NPY_SIMD_VSX_MEMORY_H
/****************************
 * load/store
 ****************************/
// TODO: test load by cast
#define VSX__CAST_lOAD 0
#if VSX__CAST_lOAD
    #define npyv__load(PTR, T_VEC) (*((T_VEC*)(PTR)))
#else
    /**
     * CLANG fails to load unaligned addresses via vec_xl, vec_xst
     * so we failback to vec_vsx_ld, vec_vsx_st
     */
    #if (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__))
        #define npyv__load(PTR, T_VEC) vec_vsx_ld(0, PTR)
    #else
        #define npyv__load(PTR, T_VEC) vec_xl(0, PTR)
    #endif
#endif
// unaligned load
#define npyv_load_u8(PTR)  npyv__load(PTR, npyv_u8)
#define npyv_load_s8(PTR)  npyv__load(PTR, npyv_s8)
#define npyv_load_u16(PTR) npyv__load(PTR, npyv_u16)
#define npyv_load_s16(PTR) npyv__load(PTR, npyv_s16)
#define npyv_load_u32(PTR) npyv__load(PTR, npyv_u32)
#define npyv_load_s32(PTR) npyv__load(PTR, npyv_s32)
#define npyv_load_f32(PTR) npyv__load(PTR, npyv_f32)
#define npyv_load_f64(PTR) npyv__load(PTR, npyv_f64)
#if VSX__CAST_lOAD
    #define npyv_load_u64(PTR) npyv__load(PTR, npyv_u64)
    #define npyv_load_s64(PTR) npyv__load(PTR, npyv_s64)
#else
    #define npyv_load_u64(PTR) ((npyv_u64)npyv_load_u32((const unsigned int*)PTR))
    #define npyv_load_s64(PTR) ((npyv_s64)npyv_load_s32((const unsigned int*)PTR))
#endif
// aligned load
#define npyv_loada_u8(PTR)  vec_ld(0, PTR)
#define npyv_loada_s8  npyv_loada_u8
#define npyv_loada_u16 npyv_loada_u8
#define npyv_loada_s16 npyv_loada_u8
#define npyv_loada_u32 npyv_loada_u8
#define npyv_loada_s32 npyv_loada_u8
#define npyv_loada_u64 npyv_load_u64
#define npyv_loada_s64 npyv_load_s64
#define npyv_loada_f32 npyv_loada_u8
#define npyv_loada_f64 npyv_load_f64
// stream load
#define npyv_loads_u8  npyv_loada_u8
#define npyv_loads_s8  npyv_loada_s8
#define npyv_loads_u16 npyv_loada_u16
#define npyv_loads_s16 npyv_loada_s16
#define npyv_loads_u32 npyv_loada_u32
#define npyv_loads_s32 npyv_loada_s32
#define npyv_loads_u64 npyv_loada_u64
#define npyv_loads_s64 npyv_loada_s64
#define npyv_loads_f32 npyv_loada_f32
#define npyv_loads_f64 npyv_loada_f64
// load lower part
// avoid aliasing rules
#ifdef __cplusplus
    template<typename T_PTR>
    NPY_FINLINE npy_uint64 *npyv__ptr2u64(T_PTR *ptr)
    { return npy_uint64 *ptr64 = (npy_uint64*)ptr; return ptr; }
#else
    NPY_FINLINE npy_uint64 *npyv__ptr2u64(void *ptr)
    { npy_uint64 *ptr64 = ptr; return ptr64; }
#endif // __cplusplus
#if defined(__clang__) && !defined(__IBMC__)
    // vec_promote doesn't support doubleword on clang
    #define npyv_loadl_u64(PTR) npyv_setall_u64(*npyv__ptr2u64(PTR))
#else
    #define npyv_loadl_u64(PTR) vec_promote(*npyv__ptr2u64(PTR), 0)
#endif
#define npyv_loadl_u8(PTR)  ((npyv_u8)npyv_loadl_u64(PTR))
#define npyv_loadl_s8(PTR)  ((npyv_s8)npyv_loadl_u64(PTR))
#define npyv_loadl_u16(PTR) ((npyv_u16)npyv_loadl_u64(PTR))
#define npyv_loadl_s16(PTR) ((npyv_s16)npyv_loadl_u64(PTR))
#define npyv_loadl_u32(PTR) ((npyv_u32)npyv_loadl_u64(PTR))
#define npyv_loadl_s32(PTR) ((npyv_s32)npyv_loadl_u64(PTR))
#define npyv_loadl_s64(PTR) ((npyv_s64)npyv_loadl_u64(PTR))
#define npyv_loadl_f32(PTR) ((npyv_f32)npyv_loadl_u64(PTR))
#define npyv_loadl_f64(PTR) ((npyv_f64)npyv_loadl_u64(PTR))
// unaligned store
#if (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__))
    #define npyv_store_u8(PTR, VEC) vec_vsx_st(VEC, 0, PTR)
#else
    #define npyv_store_u8(PTR, VEC) vec_xst(VEC, 0, PTR)
#endif
#define npyv_store_s8  npyv_store_u8
#define npyv_store_u16 npyv_store_u8
#define npyv_store_s16 npyv_store_u8
#define npyv_store_u32 npyv_store_u8
#define npyv_store_s32 npyv_store_u8
#define npyv_store_u64(PTR, VEC) npyv_store_u8((unsigned int*)PTR, (npyv_u32)VEC)
#define npyv_store_s64(PTR, VEC) npyv_store_u8((unsigned int*)PTR, (npyv_u32)VEC)
#define npyv_store_f32 npyv_store_u8
#define npyv_store_f64 npyv_store_u8
// aligned store
#define npyv_storea_u8(PTR, VEC)  vec_st(VEC, 0, PTR)
#define npyv_storea_s8  npyv_storea_u8
#define npyv_storea_u16 npyv_storea_u8
#define npyv_storea_s16 npyv_storea_u8
#define npyv_storea_u32 npyv_storea_u8
#define npyv_storea_s32 npyv_storea_u8
#define npyv_storea_u64 npyv_store_u64
#define npyv_storea_s64 npyv_store_s64
#define npyv_storea_f32 npyv_storea_u8
#define npyv_storea_f64 npyv_store_f64
// stream store
#define npyv_stores_u8  npyv_storea_u8
#define npyv_stores_s8  npyv_storea_s8
#define npyv_stores_u16 npyv_storea_u16
#define npyv_stores_s16 npyv_storea_s16
#define npyv_stores_u32 npyv_storea_u32
#define npyv_stores_s32 npyv_storea_s32
#define npyv_stores_u64 npyv_storea_u64
#define npyv_stores_s64 npyv_storea_s64
#define npyv_stores_f32 npyv_storea_f32
#define npyv_stores_f64 npyv_storea_f64
// store lower part
#define npyv_storel_u8(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 0)
#define npyv_storel_s8  npyv_storel_u8
#define npyv_storel_u16 npyv_storel_u8
#define npyv_storel_s16 npyv_storel_u8
#define npyv_storel_u32 npyv_storel_u8
#define npyv_storel_s32 npyv_storel_u8
#define npyv_storel_s64 npyv_storel_u8
#define npyv_storel_u64 npyv_storel_u8
#define npyv_storel_f32 npyv_storel_u8
#define npyv_storel_f64 npyv_storel_u8
// store higher part
#define npyv_storeh_u8(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 1)
#define npyv_storeh_s8  npyv_storeh_u8
#define npyv_storeh_u16 npyv_storeh_u8
#define npyv_storeh_s16 npyv_storeh_u8
#define npyv_storeh_u32 npyv_storeh_u8
#define npyv_storeh_s32 npyv_storeh_u8
#define npyv_storeh_s64 npyv_storeh_u8
#define npyv_storeh_u64 npyv_storeh_u8
#define npyv_storeh_f32 npyv_storeh_u8
#define npyv_storeh_f64 npyv_storeh_u8

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
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, int stride)
{ return npyv_set_f64(ptr[0], ptr[stride]); }

// non-contiguous store
//// 8
NPY_FINLINE void npyv_storen_u8(npy_uint8 *ptr, int stride, npyv_u8 a)
{
    #define NPYV_IMPL_VSX_STOREN8(I) \
    { \
        unsigned e = vec_extract((npyv_u32)a, I/4); \
        ptr[stride*(I+0)] = (npy_uint8)e; \
        ptr[stride*(I+1)] = (npy_uint8)(e >> 8); \
        ptr[stride*(I+2)] = (npy_uint8)(e >> 16); \
        ptr[stride*(I+3)] = (npy_uint8)(e >> 24); \
    }
    NPYV_IMPL_VSX_STOREN8(0)
    NPYV_IMPL_VSX_STOREN8(4)
    NPYV_IMPL_VSX_STOREN8(8)
    NPYV_IMPL_VSX_STOREN8(12)
}
NPY_FINLINE void npyv_storen_s8(npy_int8 *ptr, int stride, npyv_s8 a)
{ npyv_storen_u8((npy_uint8*)ptr, stride, (npyv_u8)a); }
//// 16
NPY_FINLINE void npyv_storen_u16(npy_uint16 *ptr, int stride, npyv_u16 a)
{
    #define NPYV_IMPL_VSX_STOREN16(I) \
    { \
        unsigned e = vec_extract((npyv_u32)a, I/2); \
        ptr[stride*(I+0)] = (npy_uint16)e; \
        ptr[stride*(I+1)] = (npy_uint16)(e >> 16); \
    }
    NPYV_IMPL_VSX_STOREN16(0)
    NPYV_IMPL_VSX_STOREN16(2)
    NPYV_IMPL_VSX_STOREN16(4)
    NPYV_IMPL_VSX_STOREN16(6)
}
NPY_FINLINE void npyv_storen_s16(npy_int16 *ptr, int stride, npyv_s16 a)
{ npyv_storen_u16((npy_uint16*)ptr, stride, (npyv_u16)a); }
//// 32
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, int stride, npyv_u32 a)
{
    ptr[stride * 0] = vec_extract(a, 0);
    ptr[stride * 1] = vec_extract(a, 1);
    ptr[stride * 2] = vec_extract(a, 2);
    ptr[stride * 3] = vec_extract(a, 3);
}
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, int stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, int stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
//// 64
NPY_FINLINE void npyv_storen_f64(double *ptr, int stride, npyv_f64 a)
{
    ptr[0] = vec_extract(a, 0);
    ptr[1] = vec_extract(a, 1);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, int stride, npyv_u64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, int stride, npyv_s64 a)
{ npyv_storen_f64((double*)ptr, stride, (npyv_f64)a); }

#endif // _NPY_SIMD_VSX_MEMORY_H
