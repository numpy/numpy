#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_MEMORY_H
#define _NPY_SIMD_SVE_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
// GCC requires literal type definitions for pointers types otherwise it causes
// ambiguous errors
#define NPYV_IMPL_SVE_MEM(VL_PAT, HALF_LANE, S, W, T)                         \
    NPY_FINLINE npyv_##S##W npyv_load_##S##W(const npyv_lanetype_##S##W *ptr) \
    {                                                                         \
        return svld1_##S##W(svptrue_b##W(), (const T##_t *)ptr);              \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_loada_##S##W(                                \
            const npyv_lanetype_##S##W *ptr)                                  \
    {                                                                         \
        return svld1_##S##W(svptrue_b##W(), (const T##_t *)ptr);              \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_loadl_##S##W(                                \
            const npyv_lanetype_##S##W *ptr)                                  \
    {                                                                         \
        return svld1_##S##W(svptrue_pat_b##W(VL_PAT), (const T##_t *)ptr);    \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_loads_##S##W(                                \
            const npyv_lanetype_##S##W *ptr)                                  \
    {                                                                         \
        return svld1_##S##W(svptrue_b##W(), (const T##_t *)ptr);              \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_load_till_##S##W(                            \
            const npy_##T *ptr, npy_uintp nlane, npy_##T fill)                \
    {                                                                         \
        assert(nlane > 0);                                                    \
        if (nlane == NPY_SIMD_WIDTH / sizeof(T##_t)) {                        \
            return svld1_##S##W(svptrue_b##W(), ptr);                         \
        }                                                                     \
        else {                                                                \
            const sv##T##_t vfill = svdup_##S##W(fill);                       \
            const svbool_t mask = svwhilelt_b##W##_u32(0, nlane);             \
            return svsel_##S##W(mask, svld1_##S##W(mask, ptr), vfill);        \
        }                                                                     \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_load_tillz_##S##W(const npy_##T *ptr,        \
                                                   npy_uintp nlane)           \
    {                                                                         \
        assert(nlane > 0);                                                    \
        if (nlane == NPY_SIMD_WIDTH / sizeof(T##_t)) {                        \
            return svld1_##S##W(svptrue_b##W(), ptr);                         \
        }                                                                     \
        else {                                                                \
            const svbool_t mask = svwhilelt_b##W##_u32(0, nlane);             \
            return svld1_##S##W(mask, ptr);                                   \
        }                                                                     \
    }                                                                         \
    NPY_FINLINE void npyv_store_##S##W(npyv_lanetype_##S##W *ptr,             \
                                       npyv_##S##W vec)                       \
    {                                                                         \
        svst1_##S##W(svptrue_b##W(), (T##_t *)ptr, vec);                      \
    }                                                                         \
    NPY_FINLINE void npyv_storea_##S##W(npyv_lanetype_##S##W *ptr,            \
                                        npyv_##S##W vec)                      \
    {                                                                         \
        svst1_##S##W(svptrue_b##W(), (T##_t *)ptr, vec);                      \
    }                                                                         \
    NPY_FINLINE void npyv_stores_##S##W(npyv_lanetype_##S##W *ptr,            \
                                        npyv_##S##W vec)                      \
    {                                                                         \
        svst1_##S##W(svptrue_b##W(), (T##_t *)ptr, vec);                      \
    }                                                                         \
    NPY_FINLINE void npyv_storel_##S##W(npyv_lanetype_##S##W *ptr,            \
                                        npyv_##S##W vec)                      \
    {                                                                         \
        svst1_##S##W(svptrue_pat_b##W(VL_PAT), (T##_t *)ptr, vec);            \
    }                                                                         \
    NPY_FINLINE void npyv_storeh_##S##W(npyv_lanetype_##S##W *ptr,            \
                                        npyv_##S##W vec)                      \
    {                                                                         \
        svbool_t mask = svptrue_pat_b##W(VL_PAT);                             \
                                                                              \
        sv##T##_t tmp = svext_##S##W(vec, vec, HALF_LANE);                    \
        svst1_##S##W(mask, (T##_t *)ptr, tmp);                                \
    }                                                                         \
    NPY_FINLINE void npyv_store_till_##S##W(npy_##T *ptr, npy_uintp nlane,    \
                                            npyv_##S##W a)                    \
    {                                                                         \
        assert(nlane > 0);                                                    \
        if (nlane == NPY_SIMD_WIDTH / sizeof(T##_t)) {                        \
            svst1_##S##W(svptrue_b##W(), ptr, a);                             \
        }                                                                     \
        else {                                                                \
            const svbool_t mask = svwhilelt_b##W##_u32(0, nlane);             \
            svst1_##S##W(mask, ptr, a);                                       \
        }                                                                     \
    }

#if NPY_SIMD == 512
NPYV_IMPL_SVE_MEM(SV_VL32, 32, u, 8, uint8)
NPYV_IMPL_SVE_MEM(SV_VL16, 16, u, 16, uint16)
NPYV_IMPL_SVE_MEM(SV_VL8, 8, u, 32, uint32)
NPYV_IMPL_SVE_MEM(SV_VL4, 4, u, 64, uint64)
NPYV_IMPL_SVE_MEM(SV_VL32, 32, s, 8, int8)
NPYV_IMPL_SVE_MEM(SV_VL16, 16, s, 16, int16)
NPYV_IMPL_SVE_MEM(SV_VL8, 8, s, 32, int32)
NPYV_IMPL_SVE_MEM(SV_VL4, 4, s, 64, int64)
NPYV_IMPL_SVE_MEM(SV_VL8, 8, f, 32, float32)
NPYV_IMPL_SVE_MEM(SV_VL4, 4, f, 64, float64)
#else
#error "unsupported sve size"
#endif

/***************************
 * Non-contiguous (partial) Load/Store
 ***************************/
#define NPYV_IMPL_SVE_LOADN(S, W, T)                                          \
    NPY_FINLINE npyv_##S##W npyv_loadn_##S##W(const npy_##T *ptr,             \
                                              npy_intp stride)                \
    {                                                                         \
        assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE##W);                  \
        const svint##W##_t steps = svindex_s##W(0, 1);                        \
        const svint##W##_t idx =                                              \
                svmul_n_s##W##_x(svptrue_b##W(), steps, stride);              \
                                                                              \
        return svld1_gather_s##W##index_##S##W(svptrue_b##W(), ptr, idx);     \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_loadn_till_##S##W(                           \
            const npy_##T *ptr, npy_intp stride, npy_uintp nlane,             \
            npy_##T fill)                                                     \
    {                                                                         \
        assert(nlane > 0);                                                    \
        assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE##W);                  \
        const svint##W##_t steps = svindex_s##W(0, 1);                        \
        const svint##W##_t idx =                                              \
                svmul_n_s##W##_x(svptrue_b##W(), steps, stride);              \
                                                                              \
        if (nlane == NPY_SIMD_WIDTH / sizeof(T##_t)) {                        \
            return svld1_gather_s##W##index_##S##W(svptrue_b##W(), ptr, idx); \
        }                                                                     \
        else {                                                                \
            const svbool_t mask = svwhilelt_b##W##_u32(0, nlane);             \
            const sv##T##_t vfill = svdup_##S##W(fill);                       \
                                                                              \
            return svsel_##S##W(                                              \
                    mask, svld1_gather_s##W##index_##S##W(mask, ptr, idx),    \
                    vfill);                                                   \
        }                                                                     \
    }                                                                         \
    NPY_FINLINE npyv_##S##W npyv_loadn_tillz_##S##W(                          \
            const npy_##T *ptr, npy_intp stride, npy_uintp nlane)             \
    {                                                                         \
        return npyv_loadn_till_##S##W(ptr, stride, nlane, 0);                 \
    }                                                                         \
    NPY_FINLINE void npyv_storen_##S##W(npy_##T *ptr, npy_intp stride,        \
                                        npyv_##S##W a)                        \
    {                                                                         \
        assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE##W);                  \
        const svint##W##_t steps = svindex_s##W(0, 1);                        \
        const svint##W##_t idx =                                              \
                svmul_n_s##W##_x(svptrue_b##W(), steps, stride);              \
                                                                              \
        svst1_scatter_s##W##index_##S##W(svptrue_b##W(), ptr, idx, a);        \
    }                                                                         \
    NPY_FINLINE void npyv_storen_till_##S##W(npy_##T *ptr, npy_intp stride,   \
                                             npy_uintp nlane, npyv_##S##W a)  \
    {                                                                         \
        assert(nlane > 0);                                                    \
        assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE##W);                  \
        const svint##W##_t steps = svindex_s##W(0, 1);                        \
        const svint##W##_t idx =                                              \
                svmul_n_s##W##_x(svptrue_b##W(), steps, stride);              \
                                                                              \
        if (nlane == NPY_SIMD_WIDTH / sizeof(T##_t)) {                        \
            svst1_scatter_s##W##index_##S##W(svptrue_b##W(), ptr, idx, a);    \
        }                                                                     \
        else {                                                                \
            const svbool_t mask = svwhilelt_b##W##_u32(0, nlane);             \
                                                                              \
            svst1_scatter_s##W##index_##S##W(mask, ptr, idx, a);              \
        }                                                                     \
    }

NPYV_IMPL_SVE_LOADN(u, 32, uint32)
NPYV_IMPL_SVE_LOADN(s, 32, int32)
NPYV_IMPL_SVE_LOADN(u, 64, uint64)
NPYV_IMPL_SVE_LOADN(s, 64, int64)
NPYV_IMPL_SVE_LOADN(f, 32, float32)
NPYV_IMPL_SVE_LOADN(f, 64, float64)

/**************************************************
 * Lookup table
 *************************************************/
// uses vector as indexes into a table
// that contains 32 elements of float32.
NPY_FINLINE npyv_f32
npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const npyv_u32 t_idx =
            svand_n_u32_m(svptrue_b32(), idx, npyv_nlanes_f32 - 1);
    const npyv_f32 table0 = npyv_load_f32(table);
    const npyv_f32 table1 = npyv_load_f32(table + npyv_nlanes_f32);
    const svbool_t mask = svcmplt_n_u32(svptrue_b32(), idx, npyv_nlanes_f32);
    npyv_f32 t0 = svtbl_f32(table0, t_idx);
    npyv_f32 t1 = svtbl_f32(table1, t_idx);

    return svsel_f32(mask, t0, t1);
}
NPY_FINLINE npyv_u32
npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float *)table, idx));
}
NPY_FINLINE npyv_s32
npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{
    return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float *)table, idx));
}

// uses vector as indexes into a table
// that contains 16 elements of float64.
NPY_FINLINE npyv_f64
npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    const npyv_u64 t_idx =
            svand_n_u64_m(svptrue_b64(), idx, npyv_nlanes_f64 - 1);
    const npyv_f64 table0 = npyv_load_f64(table);
    const npyv_f64 table1 = npyv_load_f64(table + npyv_nlanes_f64);
    const svbool_t mask = svcmplt_n_u64(svptrue_b64(), idx, npyv_nlanes_f64);
    npyv_f64 t0 = svtbl_f64(table0, t_idx);
    npyv_f64 t1 = svtbl_f64(table1, t_idx);

    return svsel_f64(mask, t0, t1);
}
NPY_FINLINE npyv_u64
npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{
    return npyv_reinterpret_u64_f64(
            npyv_lut16_f64((const double *)table, idx));
}
NPY_FINLINE npyv_s64
npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{
    return npyv_reinterpret_s64_f64(
            npyv_lut16_f64((const double *)table, idx));
}

#endif  // _NPY_SIMD_SVE_MEMORY_H
