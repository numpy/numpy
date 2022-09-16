#ifndef NPY_SIMD
#error "Not a standalone header, use simd/simd.h instead"
#endif

#ifndef _NPY_SIMD_SVE_MASKOP_H
#define _NPY_SIMD_SVE_MASKOP_H

/**
 * Implements conditional addition and subtraction.
 * e.g. npyv_ifadd_f32(m, a, b, c) -> m ? a + b : c
 * e.g. npyv_ifsub_f32(m, a, b, c) -> m ? a - b : c
 */
#define NPYV_IMPL_SVE_MASK_ADDSUB(BITS, TYPE, SFX)                  \
    NPY_FINLINE npyv_##SFX##BITS npyv_ifadd_##SFX##BITS(            \
            npyv_b##BITS m, npyv_##SFX##BITS a, npyv_##SFX##BITS b, \
            npyv_##SFX##BITS c)                                     \
    {                                                               \
        sv##TYPE##BITS##_t tmp = svadd_##SFX##BITS##_m(m, a, b);    \
        return svsel_##SFX##BITS(m, tmp, c);                        \
    }                                                               \
    NPY_FINLINE npyv_##SFX##BITS npyv_ifsub_##SFX##BITS(            \
            npyv_b##BITS m, npyv_##SFX##BITS a, npyv_##SFX##BITS b, \
            npyv_##SFX##BITS c)                                     \
    {                                                               \
        sv##TYPE##BITS##_t tmp = svsub_##SFX##BITS##_m(m, a, b);    \
        return svsel_##SFX##BITS(m, tmp, c);                        \
    }

NPYV_IMPL_SVE_MASK_ADDSUB(8, uint, u)
NPYV_IMPL_SVE_MASK_ADDSUB(16, uint, u)
NPYV_IMPL_SVE_MASK_ADDSUB(32, uint, u)
NPYV_IMPL_SVE_MASK_ADDSUB(64, uint, u)
NPYV_IMPL_SVE_MASK_ADDSUB(8, int, s)
NPYV_IMPL_SVE_MASK_ADDSUB(16, int, s)
NPYV_IMPL_SVE_MASK_ADDSUB(32, int, s)
NPYV_IMPL_SVE_MASK_ADDSUB(64, int, s)
NPYV_IMPL_SVE_MASK_ADDSUB(32, float, f)
NPYV_IMPL_SVE_MASK_ADDSUB(64, float, f)

#endif  // _NPY_SIMD_SVE_MASKOP_H
