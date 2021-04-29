/**
 * This header is used internaly by all current supported SIMD extention,
 * execpt for AVX512.
 */
#ifndef NPY_SIMD
    #error "Not a standalone header, use simd/simd.h instead"
#endif

#ifndef _NPY_SIMD_EMULATE_MASKOP_H
#define _NPY_SIMD_EMULATE_MASKOP_H

/**
 * Implements conditional addition and subtraction.
 * e.g. npyv_ifadd_f32(mask, a, b, c) -> mask ? a + b : c
 * e.g. npyv_ifsub_f32(mask, a, b, c) -> mask ? a - b : c
 */
#define NPYV_IMPL_EMULATE_MASK_ADDSUB(SFX, BSFX)              \
    NPY_FINLINE npyv_##SFX npyv_ifadd_##SFX                   \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c) \
    {                                                         \
        npyv_##SFX add = npyv_add_##SFX(a, b);                \
        return npyv_select_##SFX(m, add, c);                  \
    }                                                         \
    NPY_FINLINE npyv_##SFX npyv_ifsub_##SFX                   \
    (npyv_##BSFX m, npyv_##SFX a, npyv_##SFX b, npyv_##SFX c) \
    {                                                         \
        npyv_##SFX sub = npyv_sub_##SFX(a, b);                \
        return npyv_select_##SFX(m, sub, c);                  \
    }

NPYV_IMPL_EMULATE_MASK_ADDSUB(u8,  b8)
NPYV_IMPL_EMULATE_MASK_ADDSUB(s8,  b8)
NPYV_IMPL_EMULATE_MASK_ADDSUB(u16, b16)
NPYV_IMPL_EMULATE_MASK_ADDSUB(s16, b16)
NPYV_IMPL_EMULATE_MASK_ADDSUB(u32, b32)
NPYV_IMPL_EMULATE_MASK_ADDSUB(s32, b32)
NPYV_IMPL_EMULATE_MASK_ADDSUB(u64, b64)
NPYV_IMPL_EMULATE_MASK_ADDSUB(s64, b64)
NPYV_IMPL_EMULATE_MASK_ADDSUB(f32, b32)
#if NPY_SIMD_F64
    NPYV_IMPL_EMULATE_MASK_ADDSUB(f64, b64)
#endif

#endif // _NPY_SIMD_EMULATE_MASKOP_H
