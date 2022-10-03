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
#define NPYV_IMPL_SVE_MEM(S, W, T)                                            \
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
        const svbool_t mask = svwhilelt_b##W##_u32(0, NPY_SIMD / W / 2);      \
        return svld1_##S##W(mask, (const T##_t *)ptr);                        \
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
        const svbool_t mask = svwhilelt_b##W##_u32(0, NPY_SIMD / W / 2);      \
        svst1_##S##W(mask, (T##_t *)ptr, vec);                                \
    }                                                                         \
    NPY_FINLINE void npyv_storeh_##S##W(npyv_lanetype_##S##W *ptr,            \
                                        npyv_##S##W vec)                      \
    {                                                                         \
        sv##T##_t tmp = svext_##S##W(vec, vec, NPY_SIMD / W / 2);             \
        svst1_##S##W(svptrue_pat_b##W(SV_POW2), (T##_t *)ptr, tmp);           \
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

NPYV_IMPL_SVE_MEM(u, 8,  uint8)
NPYV_IMPL_SVE_MEM(u, 16, uint16)
NPYV_IMPL_SVE_MEM(u, 32, uint32)
NPYV_IMPL_SVE_MEM(u, 64, uint64)
NPYV_IMPL_SVE_MEM(s, 8,  int8)
NPYV_IMPL_SVE_MEM(s, 16, int16)
NPYV_IMPL_SVE_MEM(s, 32, int32)
NPYV_IMPL_SVE_MEM(s, 64, int64)
NPYV_IMPL_SVE_MEM(f, 32, float32)
NPYV_IMPL_SVE_MEM(f, 64, float64)
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
    return svld1_gather_u32index_f32(svptrue_b32(), table, idx);
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
    return svld1_gather_u64index_f64(svptrue_b64(), table, idx);
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
