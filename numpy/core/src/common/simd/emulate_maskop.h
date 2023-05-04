/**
 * This header is used internally by all current supported SIMD extensions,
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
#if NPY_SIMD_F32
    NPYV_IMPL_EMULATE_MASK_ADDSUB(f32, b32)
#endif
#if NPY_SIMD_F64
    NPYV_IMPL_EMULATE_MASK_ADDSUB(f64, b64)
#endif
#if NPY_SIMD_F32
    // conditional division, m ? a / b : c
    NPY_FINLINE npyv_f32
    npyv_ifdiv_f32(npyv_b32 m, npyv_f32 a, npyv_f32 b, npyv_f32 c)
    {
        const npyv_f32 one = npyv_setall_f32(1.0f);
        npyv_f32 div = npyv_div_f32(a, npyv_select_f32(m, b, one));
        return npyv_select_f32(m, div, c);
    }
    // conditional division, m ? a / b : 0
    NPY_FINLINE npyv_f32
    npyv_ifdivz_f32(npyv_b32 m, npyv_f32 a, npyv_f32 b)
    {
        const npyv_f32 zero = npyv_zero_f32();
        return npyv_ifdiv_f32(m, a, b, zero);
    }
#endif
#if NPY_SIMD_F64
    // conditional division, m ? a / b : c
    NPY_FINLINE npyv_f64
    npyv_ifdiv_f64(npyv_b64 m, npyv_f64 a, npyv_f64 b, npyv_f64 c)
    {
        const npyv_f64 one = npyv_setall_f64(1.0);
        npyv_f64 div = npyv_div_f64(a, npyv_select_f64(m, b, one));
        return npyv_select_f64(m, div, c);
    }
    // conditional division, m ? a / b : 0
    NPY_FINLINE npyv_f64
    npyv_ifdivz_f64(npyv_b64 m, npyv_f64 a, npyv_f64 b)
    {
        const npyv_f64 zero = npyv_zero_f64();
        return npyv_ifdiv_f64(m, a, b, zero);
    }
#endif

#endif // _NPY_SIMD_EMULATE_MASKOP_H
