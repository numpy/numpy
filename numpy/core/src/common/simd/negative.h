/**
 * This header is used internally by all current supported SIMD extensions,
 * except for ASIMD.
 */
#ifndef NPY_SIMD
    #error "Not a standalone header, use simd/simd.h instead"
#endif

#ifndef _NPY_SIMD_EMULATE_NEGATIVE_H
#define _NPY_SIMD_EMULATE_NEGATIVE_H

/**
 * Implements negative for int using XOR
 *   (x ^ -1) + 1
 */
#define NPYV_IMPL_NEGATIVE_INT(SFX)                                         \
    NPY_FINLINE npyv_##SFX npyv_negative_##SFX                              \
    (npyv_##SFX v)                                                          \
    {                                                                       \
        const npyv_##SFX m1 = npyv_setall_##SFX((npyv_lanetype_##SFX)-1);   \
        return npyv_sub_##SFX(npyv_xor_##SFX(v, m1), m1);                   \
    }

NPYV_IMPL_NEGATIVE_INT(u8)
NPYV_IMPL_NEGATIVE_INT(s8)
NPYV_IMPL_NEGATIVE_INT(u16)
NPYV_IMPL_NEGATIVE_INT(s16)
NPYV_IMPL_NEGATIVE_INT(u32)
NPYV_IMPL_NEGATIVE_INT(s32)
NPYV_IMPL_NEGATIVE_INT(u64)
NPYV_IMPL_NEGATIVE_INT(s64)
#undef NPYV_IMPL_NEGATIVE_INT

/**
 * Implements negative for float using XOR
 *   (v ^ signmask)
 */
#define NPYV_IMPL_NEGATIVE_FLOAT(SFX, SIGN)                                 \
    NPY_FINLINE npyv_##SFX npyv_negative_##SFX                              \
    (npyv_##SFX v)                                                          \
    {                                                                       \
        const npyv_##SFX signmask = npyv_setall_##SFX(SIGN);                \
        return npyv_xor_##SFX(v, signmask);                                 \
    }
#if NPY_SIMD_F32
NPYV_IMPL_NEGATIVE_FLOAT(f32, -0.f)
#endif
#if NPY_SIMD_F64
NPYV_IMPL_NEGATIVE_FLOAT(f64, -0.)
#endif
#undef NPYV_IMPL_NEGATIVE_FLOAT

#endif // _NPY_SIMD_EMULATE_NEGATIVE_H
