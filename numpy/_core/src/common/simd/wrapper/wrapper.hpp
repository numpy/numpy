#ifndef NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_WRAPPER_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_WRAPPER_HPP_

#include "datatypes.hpp"
#include "simd/forward.inc.hpp"

#if NPY_SIMD
namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/***************************
 * Misc
 ***************************/
#define NPYV_IMPL_CPP_MISC(TLANE, SFX)              \
    template<> NPY_FINLINE size_t NLanes(TLANE)     \
    { return npyv_nlanes_##SFX; }                   \
    template<> NPY_FINLINE Vec<TLANE> Undef(TLANE)  \
    { return Vec<TLANE>(npyv_##SFX()); }            \
    template<> NPY_FINLINE Vec<TLANE> Zero(TLANE)   \
    { return Vec<TLANE>(npyv_zero_##SFX()); }       \
    template<> NPY_FINLINE Vec<TLANE> Set(TLANE v)  \
    { return Vec<TLANE>(npyv_setall_##SFX(v)); }    \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE v0,  TLANE v1,  TLANE v2,  TLANE v3,  \
        TLANE v4,  TLANE v5,  TLANE v6,  TLANE v7,  \
        TLANE v8,  TLANE v9,  TLANE v10, TLANE v11, \
        TLANE v12, TLANE v13, TLANE v14, TLANE v15, \
        TLANE v16, TLANE v17, TLANE v18, TLANE v19, \
        TLANE v20, TLANE v21, TLANE v22, TLANE v23, \
        TLANE v24, TLANE v25, TLANE v26, TLANE v27, \
        TLANE v28, TLANE v29, TLANE v30, TLANE v31, \
        TLANE v32, TLANE v33, TLANE v34, TLANE v35, \
        TLANE v36, TLANE v37, TLANE v38, TLANE v39, \
        TLANE v40, TLANE v41, TLANE v42, TLANE v43, \
        TLANE v44, TLANE v45, TLANE v46, TLANE v47, \
        TLANE v48, TLANE v49, TLANE v50, TLANE v51, \
        TLANE v52, TLANE v53, TLANE v54, TLANE v55, \
        TLANE v56, TLANE v57, TLANE v58, TLANE v59, \
        TLANE v60, TLANE v61, TLANE v62, TLANE v63  \
    ) {                                             \
        return Vec<TLANE>(npyv_set_##SFX(           \
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  \
            v8,  v9,  v10, v11, v12, v13, v14, v15, \
            v16, v17, v18, v19, v20, v21, v22, v23, \
            v24, v25, v26, v27, v28, v29, v30, v31, \
            v32, v33, v34, v35, v36, v37, v38, v39, \
            v40, v41, v42, v43, v44, v45, v46, v47, \
            v48, v49, v50, v51, v52, v53, v54, v55, \
            v56, v57, v58, v59, v60, v61, v62, v63  \
        ));                                         \
    }                                               \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE v0,  TLANE v1,  TLANE v2,  TLANE v3,  \
        TLANE v4,  TLANE v5,  TLANE v6,  TLANE v7,  \
        TLANE v8,  TLANE v9,  TLANE v10, TLANE v11, \
        TLANE v12, TLANE v13, TLANE v14, TLANE v15, \
        TLANE v16, TLANE v17, TLANE v18, TLANE v19, \
        TLANE v20, TLANE v21, TLANE v22, TLANE v23, \
        TLANE v24, TLANE v25, TLANE v26, TLANE v27, \
        TLANE v28, TLANE v29, TLANE v30, TLANE v31  \
    ) {                                             \
        return Set(                                 \
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  \
            v8,  v9,  v10, v11, v12, v13, v14, v15, \
            v16, v17, v18, v19, v20, v21, v22, v23, \
            v24, v25, v26, v27, v28, v29, v30, v31, \
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  \
            v8,  v9,  v10, v11, v12, v13, v14, v15, \
            v16, v17, v18, v19, v20, v21, v22, v23, \
            v24, v25, v26, v27, v28, v29, v30, v31  \
        );                                          \
    }                                               \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE v0,  TLANE v1,  TLANE v2,  TLANE v3,  \
        TLANE v4,  TLANE v5,  TLANE v6,  TLANE v7,  \
        TLANE v8,  TLANE v9,  TLANE v10, TLANE v11, \
        TLANE v12, TLANE v13, TLANE v14, TLANE v15  \
    ) {                                             \
        return Set(                                 \
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  \
            v8,  v9,  v10, v11, v12, v13, v14, v15, \
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  \
            v8,  v9,  v10, v11, v12, v13, v14, v15  \
        );                                          \
    }                                               \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE v0,  TLANE v1,  TLANE v2,  TLANE v3,  \
        TLANE v4,  TLANE v5,  TLANE v6,  TLANE v7   \
    ) {                                             \
        return Set(                                 \
            v0, v1, v2, v3, v4, v5, v6, v7,         \
            v0, v1, v2, v3, v4, v5, v6, v7          \
        );                                          \
    }                                               \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE a, TLANE b, TLANE c, TLANE d          \
    ) {                                             \
        return Set(a, b, c, d, a, b, c, d);         \
    }                                               \
    template<> NPY_FINLINE Vec<TLANE>               \
        Set<TLANE, TLANE>(                          \
        TLANE a, TLANE b                            \
    ) {                                             \
        return Set(a, b, a, b);                     \
    }                                               \
    template<>                                      \
    NPY_FINLINE TLANE Get0(const Vec<TLANE> &a)     \
    { return npyv_extract0_##SFX(a.val); }          \
    template<>                                      \
    NPY_FINLINE Vec2<TLANE> SetTuple(               \
        const Vec<TLANE> &a, const Vec<TLANE> &b)   \
    { return {{a, b}}; }                            \
    template<>                                      \
    NPY_FINLINE Vec3<TLANE> SetTuple(               \
        const Vec<TLANE> &a, const Vec<TLANE> &b,   \
        const Vec<TLANE> &c                         \
    )                                               \
    { return {{a, b, c}}; }                         \
    template<>                                      \
    NPY_FINLINE Vec4<TLANE> SetTuple(               \
        const Vec<TLANE> &a, const Vec<TLANE> &b,   \
        const Vec<TLANE> &c, const Vec<TLANE> &d    \
    )                                               \
    { return {{a, b, c, d}}; }                      \
    template<int Ind>                                 \
    NPY_FINLINE Vec<TLANE> GetTuple(const Vec2<TLANE> &a) \
    { return a.val[Ind]; }                                \
    template<int Ind>                                     \
    NPY_FINLINE Vec<TLANE> GetTuple(const Vec3<TLANE> &a) \
    { return a.val[Ind]; }                                \
    template<int Ind>                                     \
    NPY_FINLINE Vec<TLANE> GetTuple(const Vec4<TLANE> &a) \
    { return a.val[Ind]; }

NPYV_IMPL_CPP_MISC(uint8_t, u8)
NPYV_IMPL_CPP_MISC(int8_t,  s8)
NPYV_IMPL_CPP_MISC(uint16_t,u16)
NPYV_IMPL_CPP_MISC(int16_t, s16)
NPYV_IMPL_CPP_MISC(uint32_t,u32)
NPYV_IMPL_CPP_MISC(int32_t, s32)
NPYV_IMPL_CPP_MISC(uint64_t,u64)
NPYV_IMPL_CPP_MISC(int64_t, s64)
#if NPY_SIMD_F32
    NPYV_IMPL_CPP_MISC(float, f32)
#endif
#if NPY_SIMD_F64
    NPYV_IMPL_CPP_MISC(double, f64)
#endif
#undef NPYV_IMPL_CPP_MISC

NPY_FINLINE size_t Width()
{ return NLanes<uint8_t>(); }

// reinterpret
#define NPYV_IMPL_CPP_REINTER_(TLANEL, SFXL, TLANER, SFXR) \
    template<> NPY_FINLINE Vec<TLANEL> \
    Reinterpret<TLANEL> (const Vec<TLANER> &a, TLANEL) \
    { return Vec<TLANEL>(npyv_reinterpret_##SFXL##_##SFXR(a.val)); }

#define NPYV_IMPL_CPP_REINTER_INT(TLANE, SFX)         \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, uint8_t, u8)   \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, int8_t, s8)    \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, uint16_t, u16) \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, int16_t, s16)  \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, uint32_t, u32) \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, int32_t, s32)  \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, uint64_t, u64) \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, int64_t, s64)

#if NPY_SIMD_F32
#define NPYV_IMPL_CPP_REINTER_INT_F32(TLANE, SFX) \
    NPYV_IMPL_CPP_REINTER_INT(TLANE, SFX) \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, float, f32)
#else
#define NPYV_IMPL_CPP_REINTER_INT_F32 NPYV_IMPL_CPP_REINTER_INT
#endif

#if NPY_SIMD_F64
#define NPYV_IMPL_CPP_REINTER(TLANE, SFX) \
    NPYV_IMPL_CPP_REINTER_INT_F32(TLANE, SFX) \
    NPYV_IMPL_CPP_REINTER_(TLANE, SFX, double, f64)
#else
#define NPYV_IMPL_CPP_REINTER NPYV_IMPL_CPP_REINTER_INT_F32
#endif

NPYV_IMPL_CPP_REINTER(uint8_t, u8)
NPYV_IMPL_CPP_REINTER(int8_t, s8)
NPYV_IMPL_CPP_REINTER(uint16_t, u16)
NPYV_IMPL_CPP_REINTER(int16_t, s16)
NPYV_IMPL_CPP_REINTER(uint32_t, u32)
NPYV_IMPL_CPP_REINTER(int32_t, s32)
NPYV_IMPL_CPP_REINTER(uint64_t, u64)
NPYV_IMPL_CPP_REINTER(int64_t, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_REINTER(float, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_REINTER(double, f64)
#endif
#undef NPYV_IMPL_CPP_REINTER_INT
#undef NPYV_IMPL_CPP_REINTER_
#undef NPYV_IMPL_CPP_REINTER

#define NPYV_IMPL_CPP_SEL(TLANE, SFX)  \
    template<> NPY_FINLINE Vec<TLANE> Select \
    (const Mask<TLANE> &m, const Vec<TLANE> &a, const Vec<TLANE> &b) \
    { return Vec<TLANE>(npyv_select_##SFX(m.val, a.val, b.val)); }

NPYV_IMPL_CPP_SEL(uint8_t, u8)
NPYV_IMPL_CPP_SEL(int8_t, s8)
NPYV_IMPL_CPP_SEL(uint16_t, u16)
NPYV_IMPL_CPP_SEL(int16_t, s16)
NPYV_IMPL_CPP_SEL(uint32_t, u32)
NPYV_IMPL_CPP_SEL(int32_t, s32)
NPYV_IMPL_CPP_SEL(uint64_t, u64)
NPYV_IMPL_CPP_SEL(int64_t, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_SEL(float, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_SEL(double, f64)
#endif
#undef NPYV_IMPL_CPP_SEL

/***************************
 * Memory
 ***************************/
#define NPYV_IMPL_CPP_MEM_CONT(SFX, TLANE)                                     \
    template<> NPY_FINLINE Vec<TLANE> Load(const TLANE *ptr)                   \
    { return Vec<TLANE>(npyv_load_##SFX((const npyv_lanetype_##SFX *)ptr)); }  \
    template<> NPY_FINLINE Vec<TLANE> LoadAligned(const TLANE *ptr)            \
    { return Vec<TLANE>(npyv_loada_##SFX((const npyv_lanetype_##SFX *)ptr)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadStream(const TLANE *ptr)             \
    { return Vec<TLANE>(npyv_loads_##SFX((const npyv_lanetype_##SFX *)ptr)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadLow(const TLANE *ptr)                \
    { return Vec<TLANE>(npyv_loadl_##SFX((const npyv_lanetype_##SFX *)ptr)); } \
    template<> NPY_FINLINE Vec2<TLANE> LoadDeinter2(const TLANE *ptr)          \
    {                                                                          \
        npyv_##SFX##x2 r = npyv_load_##SFX##x2(                                \
            (const npyv_lanetype_##SFX *)ptr                                   \
        );                                                                     \
        return {{Vec<TLANE>(r.val[0]), Vec<TLANE>(r.val[1])}};                 \
    }                                                                          \
    template<> NPY_FINLINE void Store(TLANE *ptr, const Vec<TLANE> &a)         \
    { npyv_store_##SFX((npyv_lanetype_##SFX *)ptr, a.val); }                   \
    template<> NPY_FINLINE void StoreAligned(TLANE *ptr, const Vec<TLANE> &a)  \
    { npyv_storea_##SFX((npyv_lanetype_##SFX *)ptr, a.val); }                  \
    template<> NPY_FINLINE void StoreStream(TLANE *ptr, const Vec<TLANE> &a)   \
    { npyv_stores_##SFX((npyv_lanetype_##SFX *)ptr, a.val); }                  \
    template<> NPY_FINLINE void StoreLow(TLANE *ptr, const Vec<TLANE> &a)      \
    { npyv_storel_##SFX((npyv_lanetype_##SFX *)ptr, a.val); }                  \
    template<> NPY_FINLINE void StoreHigh(TLANE *ptr, const Vec<TLANE> &a)     \
    { npyv_storeh_##SFX((npyv_lanetype_##SFX *)ptr, a.val); }                  \
    template<> NPY_FINLINE void StoreInter2(TLANE *ptr, const Vec2<TLANE> &a)  \
    {                                                                          \
        npyv_##SFX##x2 v = {{a.val[0].val, a.val[1].val}};                     \
        npyv_store_##SFX##x2((npyv_lanetype_##SFX *)ptr, v);                   \
    }

NPYV_IMPL_CPP_MEM_CONT(u8, uint8_t)
NPYV_IMPL_CPP_MEM_CONT(s8, int8_t)
NPYV_IMPL_CPP_MEM_CONT(u16, uint16_t)
NPYV_IMPL_CPP_MEM_CONT(s16, int16_t)
NPYV_IMPL_CPP_MEM_CONT(u32, uint32_t)
NPYV_IMPL_CPP_MEM_CONT(s32, int32_t)
NPYV_IMPL_CPP_MEM_CONT(u64, uint64_t)
NPYV_IMPL_CPP_MEM_CONT(s64, int64_t)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_MEM_CONT(f32, float)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_MEM_CONT(f64, double)
#endif
#undef NPYV_IMPL_CPP_MEM_CONT

#define NPYV_IMPL_CPP_MEM_NCONT(SFX, TLANE) \
    template<> NPY_FINLINE Vec<TLANE> LoadTill(const TLANE *ptr, size_t len, TLANE fill) \
    { return Vec<TLANE>(npyv_load_till_##SFX((const npyv_lanetype_##SFX *)ptr, len, fill)); }  \
    template<> NPY_FINLINE Vec<TLANE> LoadTill(const TLANE *ptr, size_t len)  \
    { return Vec<TLANE>(npyv_load_tillz_##SFX((const npyv_lanetype_##SFX *)ptr, len)); }  \
    template<> NPY_FINLINE Vec<TLANE> LoadPairTill(const TLANE *ptr, size_t len, TLANE fill0, TLANE fill1) \
    { return Vec<TLANE>(npyv_load2_till_##SFX((const npyv_lanetype_##SFX *)ptr, len, fill0, fill1)); }  \
    template<> NPY_FINLINE Vec<TLANE> LoadPairTill(const TLANE *ptr, size_t len)  \
    { return Vec<TLANE>(npyv_load2_tillz_##SFX((const npyv_lanetype_##SFX *)ptr, len)); }  \
    template<> NPY_FINLINE Vec<TLANE> Loadn(const TLANE *ptr, intptr_t stride) \
    { return Vec<TLANE>(npyv_loadn_##SFX((const npyv_lanetype_##SFX *)ptr, stride)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadnTill(const TLANE *ptr, intptr_t stride, size_t len, TLANE fill) \
    { return Vec<TLANE>(npyv_loadn_till_##SFX((const npyv_lanetype_##SFX *)ptr, stride, len, fill)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadnTill(const TLANE *ptr, intptr_t stride, size_t len) \
    { return Vec<TLANE>(npyv_loadn_tillz_##SFX((const npyv_lanetype_##SFX *)ptr, stride, len)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadnPair(const TLANE *ptr, intptr_t stride) \
    { return Vec<TLANE>(npyv_loadn2_##SFX((const npyv_lanetype_##SFX *)ptr, stride)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadnPairTill(const TLANE *ptr, intptr_t stride, size_t len, TLANE fill0, TLANE fill1) \
    { return Vec<TLANE>(npyv_loadn2_till_##SFX((const npyv_lanetype_##SFX *)ptr, stride, len, fill0, fill1)); } \
    template<> NPY_FINLINE Vec<TLANE> LoadnPairTill(const TLANE *ptr, intptr_t stride, size_t len) \
    { return Vec<TLANE>(npyv_loadn2_tillz_##SFX((const npyv_lanetype_##SFX *)ptr, stride, len)); } \
    template<> NPY_FINLINE void StoreTill(TLANE *ptr, size_t len, const Vec<TLANE> &a) \
    { npyv_store_till_##SFX((npyv_lanetype_##SFX *)ptr, len, a.val); } \
    template<> NPY_FINLINE void StorePairTill(TLANE *ptr, size_t len, const Vec<TLANE> &a) \
    { npyv_store2_till_##SFX((npyv_lanetype_##SFX *)ptr, len, a.val); } \
    template<> NPY_FINLINE void Storen(TLANE *ptr, intptr_t stride, const Vec<TLANE> &a) \
    { npyv_storen_##SFX((npyv_lanetype_##SFX *)ptr, stride, a.val); } \
    template<> NPY_FINLINE void StorenTill(TLANE *ptr, intptr_t stride, size_t len, const Vec<TLANE> &a) \
    { npyv_storen_till_##SFX((npyv_lanetype_##SFX *)ptr, stride, len, a.val); } \
    template<> NPY_FINLINE void StorenPair(TLANE *ptr, intptr_t stride, const Vec<TLANE> &a) \
    { npyv_storen2_##SFX((npyv_lanetype_##SFX *)ptr, stride, a.val); } \
    template<> NPY_FINLINE void StorenPairTill(TLANE *ptr, intptr_t stride, size_t len, const Vec<TLANE> &a) \
    { npyv_storen2_till_##SFX((npyv_lanetype_##SFX *)ptr, stride, len, a.val); } \
    template<> NPY_FINLINE bool IsLoadable<TLANE>(intptr_t stride, TLANE) \
    { return static_cast<bool>(npyv_loadable_stride_##SFX(stride)); } \
    template<> NPY_FINLINE bool IsStorable<TLANE>(intptr_t stride, TLANE) \
    { return static_cast<bool>(npyv_storable_stride_##SFX(stride)); }

NPYV_IMPL_CPP_MEM_NCONT(u32, uint32_t)
NPYV_IMPL_CPP_MEM_NCONT(s32, int32_t)
NPYV_IMPL_CPP_MEM_NCONT(u64, uint64_t)
NPYV_IMPL_CPP_MEM_NCONT(s64, int64_t)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_MEM_NCONT(f32, float)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_MEM_NCONT(f64, double)
#endif
#undef NPYV_IMPL_CPP_MEM_NCONT

#define NPYV_IMPL_CPP_LUT(TLANE, TLANEU, N, SFX) \
    template<> NPY_FINLINE Vec<TLANE> \
    Lookup128(const TLANE *table, const Vec<TLANEU> &idx) \
    { return Vec<TLANE>(npyv_lut##N##_##SFX((const npyv_lanetype_##SFX *)table, idx.val)); }

NPYV_IMPL_CPP_LUT(uint32_t, uint32_t, 32, u32)
NPYV_IMPL_CPP_LUT(int32_t, uint32_t, 32, s32)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_LUT(float, uint32_t, 32, f32)
#endif
NPYV_IMPL_CPP_LUT(uint64_t, uint64_t, 16, u64)
NPYV_IMPL_CPP_LUT(int64_t, uint64_t, 16, s64)
#if NPY_SIMD_F64
NPYV_IMPL_CPP_LUT(double, uint64_t, 16, f64)
#endif
#undef NPYV_IMPL_CPP_LUT

/***************************
 * Bitwise
 ***************************/
#define NPYV_IMPL_CPP_SHIFT_(NAME, INTRIN, SFX, TLANE)       \
    template<>                                               \
    NPY_FINLINE Vec<TLANE> NAME(const Vec<TLANE> &a, int n)  \
    { return Vec<TLANE>(npyv_##INTRIN##_##SFX(a.val, n)); }  \
    template<int N>                                          \
    NPY_FINLINE Vec<TLANE> NAME##i(const Vec<TLANE> &a)      \
    { return Vec<TLANE>(npyv_##INTRIN##i_##SFX(a.val, N)); }

#define NPYV_IMPL_CPP_OP_SHIFT(SFX, TLANE)    \
    NPYV_IMPL_CPP_SHIFT_(Shl, shl, SFX, TLANE) \
    NPYV_IMPL_CPP_SHIFT_(Shr, shr, SFX, TLANE)

NPYV_IMPL_CPP_OP_SHIFT(u16, uint16_t)
NPYV_IMPL_CPP_OP_SHIFT(s16, int16_t)
NPYV_IMPL_CPP_OP_SHIFT(u32, uint32_t)
NPYV_IMPL_CPP_OP_SHIFT(s32, int32_t)
NPYV_IMPL_CPP_OP_SHIFT(u64, uint64_t)
NPYV_IMPL_CPP_OP_SHIFT(s64, int64_t)
#undef NPYV_IMPL_CPP_OP_SHIFT_
#undef NPYV_IMPL_CPP_OP_SHIFT

#define NPYV_IMPL_CPP_UNA(NAME, INTRIN, RT, IT, SFX)   \
    template<> NPY_FINLINE RT NAME(const IT &a) \
    { return RT(npyv_##INTRIN##_##SFX(a.val)); }
#define NPYV_IMPL_CPP_BIN(NAME, INTRIN, RT, IT, SFX) \
    template<> NPY_FINLINE RT NAME(const IT &a, const IT &b) \
    { return RT(npyv_##INTRIN##_##SFX(a.val, b.val)); }

#define NPYV_IMPL_CPP_OP_LOGICAL(VTYPE, SFX) \
    NPYV_IMPL_CPP_BIN(And, and,  VTYPE, VTYPE, SFX) \
    NPYV_IMPL_CPP_BIN(Or,  or,  VTYPE, VTYPE, SFX) \
    NPYV_IMPL_CPP_BIN(Xor, xor, VTYPE, VTYPE, SFX) \
    NPYV_IMPL_CPP_UNA(Not, not, VTYPE, VTYPE, SFX) \

NPYV_IMPL_CPP_OP_LOGICAL(Vec<uint8_t>, u8)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<int8_t>, s8)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<uint16_t>, u16)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<int16_t>, s16)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<uint32_t>, u32)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<int32_t>, s32)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<uint64_t>, u64)
NPYV_IMPL_CPP_OP_LOGICAL(Vec<int64_t>, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_OP_LOGICAL(Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_OP_LOGICAL(Vec<double>, f64)
#endif
NPYV_IMPL_CPP_OP_LOGICAL(Mask<uint8_t>,  b8)
#if NPY_SIMD_STRONG_MASK
NPYV_IMPL_CPP_OP_LOGICAL(Mask<uint16_t>, b16)
NPYV_IMPL_CPP_OP_LOGICAL(Mask<uint32_t>, b32)
NPYV_IMPL_CPP_OP_LOGICAL(Mask<uint64_t>, b64)
#endif
#undef NPYV_IMPL_CPP_OP_LOGICAL

/***************************
 * Comparison
 ***************************/
#define NPYV_IMPL_CPP_OP_CMP(SFX, TU, TI) \
    NPYV_IMPL_CPP_BIN(Gt, cmpgt, Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_BIN(Lt, cmplt, Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_BIN(Ge, cmpge, Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_BIN(Le, cmple, Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_BIN(Eq, cmpeq, Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_BIN(Ne, cmpneq,Mask<TU>, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_UNA(Any, any, bool, Vec<TI>, SFX) \
    NPYV_IMPL_CPP_UNA(All, all, bool, Vec<TI>, SFX)


NPYV_IMPL_CPP_OP_CMP(u8, uint8_t, uint8_t)
NPYV_IMPL_CPP_OP_CMP(s8, uint8_t, int8_t)
NPYV_IMPL_CPP_OP_CMP(u16, uint16_t, uint16_t)
NPYV_IMPL_CPP_OP_CMP(s16, uint16_t, int16_t)
NPYV_IMPL_CPP_OP_CMP(u32, uint32_t, uint32_t)
NPYV_IMPL_CPP_OP_CMP(s32, uint32_t, int32_t)
NPYV_IMPL_CPP_OP_CMP(u64, uint64_t, uint64_t)
NPYV_IMPL_CPP_OP_CMP(s64, uint64_t, int64_t)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_OP_CMP(f32, uint32_t, float)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_OP_CMP(f64, uint64_t, double)
#endif
#undef NPYV_IMPL_CPP_OP_CMP

template<> NPY_FINLINE bool Any(const Mask<uint8_t> &a)
{ return npyv_any_b8(a.val); }
template<> NPY_FINLINE bool All(const Mask<uint8_t> &a)
{ return npyv_all_b8(a.val); }
#if NPY_SIMD_STRONG_MASK
template<> NPY_FINLINE bool Any(const Mask<uint16_t> &a)
{ return npyv_any_b16(a.val); }
template<> NPY_FINLINE bool Any(const Mask<uint32_t> &a)
{ return npyv_any_b32(a.val); }
template<> NPY_FINLINE bool Any(const Mask<uint64_t> &a)
{ return npyv_any_b64(a.val); }
template<> NPY_FINLINE bool All(const Mask<uint16_t> &a)
{ return npyv_all_b16(a.val); }
template<> NPY_FINLINE bool All(const Mask<uint32_t> &a)
{ return npyv_all_b32(a.val); }
template<> NPY_FINLINE bool All(const Mask<uint64_t> &a)
{ return npyv_all_b64(a.val); }
#endif

#if NPY_SIMD_F32
NPYV_IMPL_CPP_UNA(NotNan, notnan, Mask<uint32_t>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_UNA(NotNan, notnan, Mask<uint64_t>, Vec<double>, f64)
#endif

/***************************
 * Arithmetic
 ***************************/
NPYV_IMPL_CPP_BIN(Add, add, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_BIN(Add, add, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_BIN(Add, add, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_BIN(Add, add, Vec<int16_t>, Vec<int16_t>, s16)
NPYV_IMPL_CPP_BIN(Add, add, Vec<uint32_t>, Vec<uint32_t>, u32)
NPYV_IMPL_CPP_BIN(Add, add, Vec<int32_t>, Vec<int32_t>, s32)
NPYV_IMPL_CPP_BIN(Add, add, Vec<uint64_t>, Vec<uint64_t>, u64)
NPYV_IMPL_CPP_BIN(Add, add, Vec<int64_t>, Vec<int64_t>, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_BIN(Add, add, Vec<float>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_BIN(Add, add, Vec<double>, Vec<double>, f64)
#endif

// saturated
NPYV_IMPL_CPP_BIN(Adds, adds, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_BIN(Adds, adds, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_BIN(Adds, adds, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_BIN(Adds, adds, Vec<int16_t>, Vec<int16_t>, s16)

NPYV_IMPL_CPP_BIN(Sub, sub, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<int16_t>, Vec<int16_t>, s16)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<uint32_t>, Vec<uint32_t>, u32)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<int32_t>, Vec<int32_t>, s32)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<uint64_t>, Vec<uint64_t>, u64)
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<int64_t>, Vec<int64_t>, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<float>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_BIN(Sub, sub, Vec<double>, Vec<double>, f64)
#endif

// saturated
NPYV_IMPL_CPP_BIN(Subs, subs, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_BIN(Subs, subs, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_BIN(Subs, subs, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_BIN(Subs, subs, Vec<int16_t>, Vec<int16_t>, s16)

// mask IfAdd & IfSub
#define NPYV_IMPL_CPP_MASK_OP_(NAME, INTRIN, TU, TI, SFX) \
    template<> NPY_FINLINE Vec<TI> If##NAME  \
    (const Mask<TU> &m, const Vec<TI> &a, \
     const Vec<TI> &b, const Vec<TI> &c)  \
    { return Vec<TI>(npyv_if##INTRIN##_##SFX(m.val, a.val, b.val, c.val)); }

#define NPYV_IMPL_CPP_MASK_OP(TU, TI, SFX) \
    NPYV_IMPL_CPP_MASK_OP_(Add, add, TU, TI, SFX) \
    NPYV_IMPL_CPP_MASK_OP_(Sub, sub, TU, TI, SFX)

NPYV_IMPL_CPP_MASK_OP(uint8_t, uint8_t, u8)
NPYV_IMPL_CPP_MASK_OP(uint8_t, int8_t, s8)
NPYV_IMPL_CPP_MASK_OP(uint16_t, uint16_t, u16)
NPYV_IMPL_CPP_MASK_OP(uint16_t, int16_t, s16)
NPYV_IMPL_CPP_MASK_OP(uint32_t, uint32_t, u32)
NPYV_IMPL_CPP_MASK_OP(uint32_t, int32_t, s32)
NPYV_IMPL_CPP_MASK_OP(uint64_t, uint64_t, u64)
NPYV_IMPL_CPP_MASK_OP(uint64_t, int64_t, s64)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_MASK_OP(uint32_t, float, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_MASK_OP(uint64_t, double, f64)
#endif
#undef NPYV_IMPL_CPP_MASK_OP_
#undef NPYV_IMPL_CPP_MASK_OP

NPYV_IMPL_CPP_BIN(Mul, mul, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<int16_t>, Vec<int16_t>, s16)
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<uint32_t>, Vec<uint32_t>, u32)
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<int32_t>, Vec<int32_t>, s32)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_BIN(Div, div, Vec<float>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_BIN(Mul, mul, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_BIN(Div, div, Vec<double>, Vec<double>, f64)
#endif

#define NPYV_IMPL_CPP_INTDIV(SFX, TLANE) \
    template<> NPY_FINLINE Vec3<TLANE> Divisor(TLANE d)                              \
    {                                                                                \
        npyv_##SFX##x3 r = npyv_divisor_##SFX(d);                                    \
        return {{Vec<TLANE>(r.val[0]), Vec<TLANE>(r.val[1]), Vec<TLANE>(r.val[2])}}; \
    }                                                                                \
    template<> NPY_FINLINE Vec<TLANE> Div(const Vec<TLANE> &a, const Vec3<TLANE> &b) \
    {                                                                                \
        npyv_##SFX##x3 bb = {{b.val[0].val, b.val[1].val, b.val[2].val}};            \
        return Vec<TLANE>(npyv_divc_##SFX(a.val, bb));                               \
    }

NPYV_IMPL_CPP_INTDIV(u8, uint8_t)
NPYV_IMPL_CPP_INTDIV(s8, int8_t)
NPYV_IMPL_CPP_INTDIV(u16, uint16_t)
NPYV_IMPL_CPP_INTDIV(s16, int16_t)
NPYV_IMPL_CPP_INTDIV(u32, uint32_t)
NPYV_IMPL_CPP_INTDIV(s32, int32_t)
NPYV_IMPL_CPP_INTDIV(u64, uint64_t)
NPYV_IMPL_CPP_INTDIV(s64, int64_t)
#undef NPYV_IMPL_CPP_INTDIV

#define NPYV_IMPL_CPP_TER(NAME, INTRIN, RT, IT, SFX) \
    template<> NPY_FINLINE RT NAME(const IT &a, const IT &b, const IT &c) \
    { return RT(npyv_##INTRIN##_##SFX(a.val, b.val, c.val)); }

#if NPY_SIMD_F32
NPYV_IMPL_CPP_TER(MulAdd,  muladd,  Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_TER(MulSub,  mulsub,  Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_TER(NegMulAdd, nmuladd, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_TER(NegMulSub, nmulsub, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_TER(MulAddSub, muladdsub, Vec<float>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_TER(MulAdd,  muladd,  Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_TER(MulSub,  mulsub,  Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_TER(NegMulAdd, nmuladd, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_TER(NegMulSub, nmulsub, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_TER(MulAddSub, muladdsub, Vec<double>, Vec<double>, f64)
#endif

// reduce sum across vector
template<> NPY_FINLINE uint32_t Sum(const Vec<uint32_t> &a)
{ return npyv_sum_u32(a.val); }
template<> NPY_FINLINE uint64_t Sum(const Vec<uint64_t> &a)
{ return npyv_sum_u64(a.val); }
#if NPY_SIMD_F32
template<> NPY_FINLINE float Sum(const Vec<float> &a)
{ return npyv_sum_f32(a.val); }
#endif
#if NPY_SIMD_F64
template<> NPY_FINLINE double Sum(const Vec<double> &a)
{ return npyv_sum_f64(a.val); }
#endif
// expand the source vector and performs sum reduce
template<> NPY_FINLINE uint16_t Sumup(const Vec<uint8_t> &a)
{ return npyv_sumup_u8(a.val); }
template<> NPY_FINLINE uint32_t Sumup(const Vec<uint16_t> &a)
{ return npyv_sumup_u16(a.val); }

/***************************
 * Math
 ***************************/
#if NPY_SIMD_F32
template<> NPY_FINLINE Vec<float> Round(const Vec<float> &a)
{ return Vec<float>(npyv_rint_f32(a.val)); }
template<> NPY_FINLINE Vec<float> Trunc(const Vec<float> &a)
{ return Vec<float>(npyv_trunc_f32(a.val)); }
template<> NPY_FINLINE Vec<float> Ceil(const Vec<float> &a)
{ return Vec<float>(npyv_ceil_f32(a.val)); }
template<> NPY_FINLINE Vec<float> Floor(const Vec<float> &a)
{ return Vec<float>(npyv_floor_f32(a.val)); }
template<> NPY_FINLINE Vec<int32_t> Roundi(const Vec<float> &a)
{ return Vec<int32_t>(npyv_round_s32_f32(a.val)); }
#endif
#if NPY_SIMD_F64
template<> NPY_FINLINE Vec<double> Round(const Vec<double> &a)
{ return Vec<double>(npyv_rint_f64(a.val)); }
template<> NPY_FINLINE Vec<double> Trunc(const Vec<double> &a)
{ return Vec<double>(npyv_trunc_f64(a.val)); }
template<> NPY_FINLINE Vec<double> Ceil(const Vec<double> &a)
{ return Vec<double>(npyv_ceil_f64(a.val)); }
template<> NPY_FINLINE Vec<double> Floor(const Vec<double> &a)
{ return Vec<double>(npyv_floor_f64(a.val)); }
template<> NPY_FINLINE Vec<int32_t> Roundi(const Vec<double> &a, const Vec<double> &b)
{ return Vec<int32_t>(npyv_round_s32_f64(a.val, b.val)); }
#endif

#if NPY_SIMD_F32
NPYV_IMPL_CPP_UNA(Sqrt, sqrt, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(Recip, recip, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(Abs, abs, Vec<float>, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_UNA(Sqrt, sqrt, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(Recip, recip, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(Abs, abs, Vec<double>, Vec<double>, f64)
#endif

#define NPYV_IMPL_CPP_MAXMIN(SFX, TLANE) \
    NPYV_IMPL_CPP_BIN(Max, max, Vec<TLANE>, Vec<TLANE>, SFX)  \
    NPYV_IMPL_CPP_BIN(Min, min, Vec<TLANE>, Vec<TLANE>, SFX)  \
    NPYV_IMPL_CPP_UNA(ReduceMax, reduce_max,TLANE, Vec<TLANE>, SFX)  \
    NPYV_IMPL_CPP_UNA(ReduceMin, reduce_min,TLANE, Vec<TLANE>, SFX)

NPYV_IMPL_CPP_MAXMIN(u8,  uint8_t)
NPYV_IMPL_CPP_MAXMIN(s8,  int8_t)
NPYV_IMPL_CPP_MAXMIN(u16, uint16_t)
NPYV_IMPL_CPP_MAXMIN(s16, int16_t)
NPYV_IMPL_CPP_MAXMIN(u32, uint32_t)
NPYV_IMPL_CPP_MAXMIN(s32, int32_t)
NPYV_IMPL_CPP_MAXMIN(u64, uint64_t)
NPYV_IMPL_CPP_MAXMIN(s64, int64_t)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_MAXMIN(f32, float)
NPYV_IMPL_CPP_BIN(MaxProp, maxp, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_BIN(MinProp, minp, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_BIN(MaxPropNan, maxn, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_BIN(MinPropNan, minn, Vec<float>, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(ReduceMaxProp, reduce_maxp, float, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(ReduceMinProp, reduce_minp, float, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(ReduceMaxPropNan, reduce_maxn, float, Vec<float>, f32)
NPYV_IMPL_CPP_UNA(ReduceMinPropNan, reduce_minn, float, Vec<float>, f32)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_MAXMIN(f64, double)
NPYV_IMPL_CPP_BIN(MaxProp, maxp, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_BIN(MinProp, minp, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_BIN(MaxPropNan, maxn, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_BIN(MinPropNan, minn, Vec<double>, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(ReduceMaxProp, reduce_maxp, double, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(ReduceMinProp, reduce_minp, double, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(ReduceMaxPropNan, reduce_maxn, double, Vec<double>, f64)
NPYV_IMPL_CPP_UNA(ReduceMinPropNan, reduce_minn, double, Vec<double>, f64)
#endif
#undef NPYV_IMPL_CPP_MAXMIN

/***************************
 * Reorder
 ***************************/
#define NPYV_IMPL_CPP_COMBINE_X2(SFX, TLANE)                   \
    template<> NPY_FINLINE Vec2<TLANE>                         \
    Combine(const Vec<TLANE> &a, const Vec<TLANE> &b)          \
    {                                                          \
        npyv_##SFX##x2 r = npyv_combine_##SFX(a.val, b.val);   \
        return {{Vec<TLANE>(r.val[0]), Vec<TLANE>(r.val[1])}}; \
    }                                                          \
    template<> NPY_FINLINE Vec2<TLANE>                         \
    Zip(const Vec<TLANE> &a, const Vec<TLANE> &b)              \
    {                                                          \
        npyv_##SFX##x2 r = npyv_zip_##SFX(a.val, b.val);       \
        return {{Vec<TLANE>(r.val[0]), Vec<TLANE>(r.val[1])}}; \
    }                                                          \
    template<> NPY_FINLINE Vec2<TLANE>                         \
    Unzip(const Vec<TLANE> &a, const Vec<TLANE> &b)            \
    {                                                          \
        npyv_##SFX##x2 r = npyv_unzip_##SFX(a.val, b.val);     \
        return {{Vec<TLANE>(r.val[0]), Vec<TLANE>(r.val[1])}}; \
    }

#define NPYV_IMPL_CPP_COMBINE(SFX, TLANE) \
    NPYV_IMPL_CPP_BIN(CombineLow, combinel, Vec<TLANE>, Vec<TLANE>, SFX) \
    NPYV_IMPL_CPP_BIN(CombineHigh, combineh, Vec<TLANE>, Vec<TLANE>, SFX) \
    NPYV_IMPL_CPP_COMBINE_X2(SFX, TLANE)

NPYV_IMPL_CPP_COMBINE(u8, uint8_t)
NPYV_IMPL_CPP_COMBINE(s8, int8_t)
NPYV_IMPL_CPP_COMBINE(u16, uint16_t)
NPYV_IMPL_CPP_COMBINE(s16, int16_t)
NPYV_IMPL_CPP_COMBINE(u32, uint32_t)
NPYV_IMPL_CPP_COMBINE(s32, int32_t)
NPYV_IMPL_CPP_COMBINE(u64, uint64_t)
NPYV_IMPL_CPP_COMBINE(s64, int64_t)
#if NPY_SIMD_F32
NPYV_IMPL_CPP_COMBINE(f32, float)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_CPP_COMBINE(f64, double)
#endif
#undef NPYV_IMPL_CPP_COMBINE_X2
#undef NPYV_IMPL_CPP_COMBINE

NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<uint8_t>, Vec<uint8_t>, u8)
NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<int8_t>, Vec<int8_t>, s8)
NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<uint16_t>, Vec<uint16_t>, u16)
NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<int16_t>, Vec<int16_t>, s16)
NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<uint32_t>, Vec<uint32_t>, u32)
NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<int32_t>, Vec<int32_t>, s32)
#if NPY_SIMD_F32
    NPYV_IMPL_CPP_UNA(Reverse64, rev64, Vec<float>, Vec<float>, f32)
#endif

#define NPYV_IMPL_CPP_PERMUTE128_32(SFX, TLANE)                        \
    template<int L0, int L1, int L2, int L3>                           \
    NPY_FINLINE Vec<TLANE> Permute128(const Vec<TLANE> &a)             \
    {                                                                  \
        return Vec<TLANE>(npyv_permi128_##SFX(a.val, L0, L1, L2, L3)); \
    }
#define NPYV_IMPL_CPP_PERMUTE128_64(SFX, TLANE)                        \
    template<int L0, int L1>                                           \
    NPY_FINLINE Vec<TLANE> Permute128(const Vec<TLANE> &a)             \
    {                                                                  \
        return Vec<TLANE>(npyv_permi128_##SFX(a.val, L0, L1));         \
    }

NPYV_IMPL_CPP_PERMUTE128_32(u32, uint32_t)
NPYV_IMPL_CPP_PERMUTE128_32(s32, int32_t)
NPYV_IMPL_CPP_PERMUTE128_64(u64, uint64_t)
NPYV_IMPL_CPP_PERMUTE128_64(s64, int64_t)
#if NPY_SIMD_F32
    NPYV_IMPL_CPP_PERMUTE128_32(f32, float)
#endif
#if NPY_SIMD_F64
    NPYV_IMPL_CPP_PERMUTE128_64(f64, double)
#endif
#undef NPYV_IMPL_CPP_PERMUTE128_32
#undef NPYV_IMPL_CPP_PERMUTE128_64

/***************************
 * Conversion
 ***************************/
template<> NPY_FINLINE Mask<uint8_t> ToMask(const Vec<uint8_t> &a)
{ return Mask<uint8_t>(npyv_cvt_b8_u8(a.val)); }
template<> NPY_FINLINE Mask<uint8_t> ToMask(const Vec<int8_t> &a)
{ return Mask<uint8_t>(npyv_cvt_b8_s8(a.val)); }
template<> NPY_FINLINE Mask<uint16_t> ToMask(const Vec<uint16_t> &a)
{ return Mask<uint16_t>(npyv_cvt_b16_u16(a.val)); }
template<> NPY_FINLINE Mask<uint16_t> ToMask(const Vec<int16_t> &a)
{ return Mask<uint16_t>(npyv_cvt_b16_s16(a.val)); }
template<> NPY_FINLINE Mask<uint32_t> ToMask(const Vec<uint32_t> &a)
{ return Mask<uint32_t>(npyv_cvt_b32_u32(a.val)); }
template<> NPY_FINLINE Mask<uint32_t> ToMask(const Vec<int32_t> &a)
{ return Mask<uint32_t>(npyv_cvt_b32_s32(a.val)); }
template<> NPY_FINLINE Mask<uint64_t> ToMask(const Vec<uint64_t> &a)
{ return Mask<uint64_t>(npyv_cvt_b64_u64(a.val)); }
template<> NPY_FINLINE Mask<uint64_t> ToMask(const Vec<int64_t> &a)
{ return Mask<uint64_t>(npyv_cvt_b64_s64(a.val)); }

template<> NPY_FINLINE Vec<uint8_t> ToVec(const Mask<uint8_t> &a, uint8_t)
{ return Vec<uint8_t>(npyv_cvt_u8_b8(a.val)); }
template<> NPY_FINLINE Vec<int8_t> ToVec(const Mask<int8_t> &a, int8_t)
{ return Vec<int8_t>(npyv_cvt_s8_b8(a.val)); }
template<> NPY_FINLINE Vec<uint16_t> ToVec(const Mask<uint16_t> &a, uint16_t)
{ return Vec<uint16_t>(npyv_cvt_u16_b16(a.val)); }
template<> NPY_FINLINE Vec<int16_t> ToVec(const Mask<int16_t> &a, int16_t)
{ return Vec<int16_t>(npyv_cvt_s16_b16(a.val)); }
template<> NPY_FINLINE Vec<uint32_t> ToVec(const Mask<uint32_t> &a, uint32_t)
{ return Vec<uint32_t>(npyv_cvt_u32_b32(a.val)); }
template<> NPY_FINLINE Vec<int32_t> ToVec(const Mask<int32_t> &a, int32_t)
{ return Vec<int32_t>(npyv_cvt_s32_b32(a.val)); }
template<> NPY_FINLINE Vec<uint64_t> ToVec(const Mask<uint64_t> &a, uint64_t)
{ return Vec<uint64_t>(npyv_cvt_u64_b64(a.val)); }
template<> NPY_FINLINE Vec<int64_t> ToVec(const Mask<int64_t> &a, int64_t)
{ return Vec<int64_t>(npyv_cvt_s64_b64(a.val)); }

template<> NPY_FINLINE Vec2<uint16_t> Expand(const Vec<uint8_t> &a)
{
    npyv_u16x2 r = npyv_expand_u16_u8(a.val);
    return {{Vec<uint16_t>(r.val[0]), Vec<uint16_t>(r.val[1])}};
}
template<> NPY_FINLINE Vec2<uint32_t> Expand(const Vec<uint16_t> &a)
{
    npyv_u32x2 r = npyv_expand_u32_u16(a.val);
    return {{Vec<uint32_t>(r.val[0]), Vec<uint32_t>(r.val[1])}};
}

template<> NPY_FINLINE Mask<uint8_t> Pack(const Mask<uint16_t> &a, const Mask<uint16_t> &b, uint16_t)
{ return Mask<uint8_t>(npyv_pack_b8_b16(a.val, b.val)); }

NPY_FINLINE Mask<uint8_t> Pack(const Mask<uint32_t> &a, const Mask<uint32_t> &b,
                               const Mask<uint32_t> &c, const Mask<uint32_t> &d)
{ return Mask<uint8_t>(npyv_pack_b8_b32(a.val, b.val, c.val, d.val)); }


NPY_FINLINE Mask<uint8_t> Pack(const Mask<uint64_t> &a, const Mask<uint64_t> &b,
                               const Mask<uint64_t> &c, const Mask<uint64_t> &d,
                               const Mask<uint64_t> &e, const Mask<uint64_t> &f,
                               const Mask<uint64_t> &g, const Mask<uint64_t> &h)
{ return Mask<uint8_t>(npyv_pack_b8_b64(a.val, b.val, c.val, d.val, e.val, f.val, g.val, h.val)); }

/***************************
 * Extra
 ***************************/
NPY_FINLINE void Cleanup()
{ npyv_cleanup(); }
} // namespace npy::simd_ext

#undef NPYV_IMPL_CPP_BIN
#undef NPYV_IMPL_CPP_UNA
#undef NPYV_IMPL_CPP_TER
#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_WRAPPER_WRAPPER_HPP_
