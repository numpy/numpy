#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

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

#endif // _NPY_SIMD_NEON_MEMORY_H
