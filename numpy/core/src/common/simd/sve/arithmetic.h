#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_ARITHMETIC_H
#define _NPY_SIMD_SVE_ARITHMETIC_H

#include "operators.h"

#define NPYV_IMPL_SVE_OP(OP, S, W)                                           \
    NPY_FINLINE npyv_##S##W npyv_##OP##_##S##W(npyv_##S##W a, npyv_##S##W b) \
    {                                                                        \
        return sv##OP##_##S##W##_x(svptrue_b##W(), a, b);                    \
    }
#define NPYV_IMPL_SVE_OP2(FUNC, OP, S, W)                                      \
    NPY_FINLINE npyv_##S##W npyv_##FUNC##_##S##W(npyv_##S##W a, npyv_##S##W b) \
    {                                                                          \
        return sv##op##_##S##W##_x(svptrue_b##W(), a, b);                      \
    }
#define NPYV_IMPL_SVE_FUSED_OP(FUNC, OP, S, W)               \
    NPY_FINLINE npyv_##S##W npyv_##FUNC##_##S##W(            \
            npyv_##S##W a, npyv_##S##W b, npyv_##S##W c)     \
    {                                                        \
        return sv##OP##_##S##W##_x(svptrue_b##W(), c, a, b); \
}
#define NPYV_IMPL_SVE_FUSED_OP2(FUNC, OP, S, W)              \
    NPY_FINLINE npyv_##S##W npyv_##FUNC##_##S##W(            \
            npyv_##S##W a, npyv_##S##W b, npyv_##S##W c)     \
    {                                                        \
        return sv##OP##_##S##W##_x(svptrue_b##W(), c, b, a); \
}
#define NPYV_IMPL_SVE_SUM(FUNC, OP, S, W, T)          \
    NPY_FINLINE T npyv_##FUNC##_##S##W(npyv_##S##W a) \
    {                                                 \
        return sv##OP##_##S##W(svptrue_b##W(), a);    \
    }

/***************************
 * Addition
 ***************************/
// non-saturated
NPYV_IMPL_SVE_OP(add, u, 8)
NPYV_IMPL_SVE_OP(add, u, 16)
NPYV_IMPL_SVE_OP(add, u, 32)
NPYV_IMPL_SVE_OP(add, u, 64)
NPYV_IMPL_SVE_OP(add, s, 8)
NPYV_IMPL_SVE_OP(add, s, 16)
NPYV_IMPL_SVE_OP(add, s, 32)
NPYV_IMPL_SVE_OP(add, s, 64)
NPYV_IMPL_SVE_OP(add, f, 32)
NPYV_IMPL_SVE_OP(add, f, 64)

// saturated
#define npyv_adds_u8 svqadd_u8
#define npyv_adds_s8 svqadd_s8
#define npyv_adds_u16 svqadd_u16
#define npyv_adds_s16 svqadd_s16

/***************************
 * Subtraction
 ***************************/
// non-saturated
NPYV_IMPL_SVE_OP(sub, u, 8)
NPYV_IMPL_SVE_OP(sub, u, 16)
NPYV_IMPL_SVE_OP(sub, u, 32)
NPYV_IMPL_SVE_OP(sub, u, 64)
NPYV_IMPL_SVE_OP(sub, s, 8)
NPYV_IMPL_SVE_OP(sub, s, 16)
NPYV_IMPL_SVE_OP(sub, s, 32)
NPYV_IMPL_SVE_OP(sub, s, 64)
NPYV_IMPL_SVE_OP(sub, f, 32)
NPYV_IMPL_SVE_OP(sub, f, 64)

// saturated
#define npyv_subs_u8 svqsub_u8
#define npyv_subs_s8 svqsub_s8
#define npyv_subs_u16 svqsub_u16
#define npyv_subs_s16 svqsub_s16

/***************************
 * Multiplication
 ***************************/
// non-saturated
NPYV_IMPL_SVE_OP(mul, u, 8)
NPYV_IMPL_SVE_OP(mul, u, 16)
NPYV_IMPL_SVE_OP(mul, u, 32)
NPYV_IMPL_SVE_OP(mul, u, 64)
NPYV_IMPL_SVE_OP(mul, s, 8)
NPYV_IMPL_SVE_OP(mul, s, 16)
NPYV_IMPL_SVE_OP(mul, s, 32)
NPYV_IMPL_SVE_OP(mul, s, 64)
NPYV_IMPL_SVE_OP(mul, f, 32)
NPYV_IMPL_SVE_OP(mul, f, 64)

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by divisor
// mulhi: high part of unsigned multiplication
// floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
#define NPYV_IMPL_SVE_DIVC_U(W)                               \
    NPY_FINLINE npyv_u##W npyv_divc_u##W(                     \
        npyv_u##W a, const npyv_u##W##x3 divisor)             \
    {                                                         \
        const svbool_t mask_all = svptrue_b##W();             \
                                                              \
        svuint##W##_t mulhi =                                 \
                svmulh_u##W##_x(mask_all, a, divisor.val[0]); \
        svuint##W##_t q = svsub_u##W##_x(mask_all, a, mulhi); \
        q = svlsr_u##W##_x(mask_all, q, divisor.val[1]);      \
        q = svadd_u##W##_x(mask_all, mulhi, q);               \
        q = svlsr_u##W##_x(mask_all, q, divisor.val[2]);      \
        return q;                                             \
    }

NPYV_IMPL_SVE_DIVC_U(8)
NPYV_IMPL_SVE_DIVC_U(16)
NPYV_IMPL_SVE_DIVC_U(32)

// divide each signed 16-bit element by divisor (round towards zero)
// q               = ((a + mulhi) >> sh1) - XSIGN(a)
// trunc(a/d)      = (q ^ dsign) - dsign
#define NPYV_IMPL_SVE_DIVC_S(W)                                       \
    NPY_FINLINE npyv_s##W npyv_divc_s##W(                             \
        npyv_s##W a, const npyv_s##W##x3 divisor)                     \
    {                                                                 \
        const svbool_t mask_all = svptrue_b##W();                     \
        const svint##W##_t dsign = divisor.val[2];                    \
                                                                      \
        svint##W##_t mulhi =                                          \
                svmulh_s##W##_x(mask_all, a, divisor.val[0]);         \
        svint##W##_t q = svasr_s##W##_x(                              \
                mask_all, svadd_s##W##_x(mask_all, a, mulhi),         \
            svreinterpret_u##W##_s##W(divisor.val[1]));               \
        q = svsub_s##W##_x(mask_all, q,                               \
                svasr_n_s##W##_x(mask_all, a, W - 1));                \
        q = svsub_s##W##_x(                                           \
                mask_all, sveor_s##W##_x(mask_all, q, dsign), dsign); \
        return q;                                                     \
    }

NPYV_IMPL_SVE_DIVC_S(8)
NPYV_IMPL_SVE_DIVC_S(16)
NPYV_IMPL_SVE_DIVC_S(32)

// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    npyv_u64 mulhi = svmulh_u64_x(svptrue_b64(), a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u64 q = svsub_u64_x(svptrue_b64(), a, mulhi);

    q = svlsr_u64_x(svptrue_b64(), q, divisor.val[1]);
    q = svadd_u64_x(svptrue_b64(), mulhi, q);
    q = svlsr_x(svptrue_b64(), q, divisor.val[2]);
    return q;
}
// divide each unsigned 64-bit element by a divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // high part of unsigned multiplication
    npyv_s64 mulhi = svreinterpret_s64_u64(svmulh_u64_x(svptrue_b64(),
        svreinterpret_u64_s64(a), svreinterpret_u64_s64(divisor.val[0])));
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    npyv_s64 asign   = svasr_n_s64_x(svptrue_b64(), a, 63);
    npyv_s64 msign   = svasr_n_s64_x(svptrue_b64(), divisor.val[0], 63);
    npyv_s64 m_asign = svand_s64_x(svptrue_b64(), divisor.val[0], asign);
    npyv_s64 a_msign = svand_s64_x(svptrue_b64(), a, msign);

    mulhi = svsub_s64_x(svptrue_b64(), mulhi, m_asign);
    mulhi = svsub_s64_x(svptrue_b64(), mulhi, a_msign);

    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    npyv_s64 q = svasr_s64_x(svptrue_b64(), svadd_s64_x(svptrue_b64(),
        a, mulhi), svreinterpret_u64_s64(divisor.val[1]));

    q = svsub_s64_x(svptrue_b64(), q, asign);
    q = svsub_s64_x(svptrue_b64(), sveor_s64_x(svptrue_b64(), q,
        divisor.val[2]), divisor.val[2]);
    return  q;
}

/***************************
 * Division
 ***************************/
NPYV_IMPL_SVE_OP(div, f, 32)
NPYV_IMPL_SVE_OP(div, f, 64)

/***************************
 * FUSED F32
 ***************************/
// a*b + c
NPYV_IMPL_SVE_FUSED_OP2(muladd, mla, f, 32)
// a*b - c
NPYV_IMPL_SVE_FUSED_OP2(mulsub, nmls, f, 32)
// -(a*b) + c
NPYV_IMPL_SVE_FUSED_OP(nmuladd, mls, f, 32)  
// -(a*b) - c
NPYV_IMPL_SVE_FUSED_OP(nmulsub, nmla, f, 32)  

/***************************
 * FUSED F64
 ***************************/
// a*b + c
NPYV_IMPL_SVE_FUSED_OP2(muladd, mla, f, 64)
// a*b - c
NPYV_IMPL_SVE_FUSED_OP2(mulsub, nmls, f, 64)
// -(a*b) + c
NPYV_IMPL_SVE_FUSED_OP(nmuladd, mls, f, 64)
// -(a*b) - c
NPYV_IMPL_SVE_FUSED_OP(nmulsub, nmla, f, 64)

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPYV_IMPL_SVE_SUM(sum,   addv, u, 32, npy_uint32)
NPYV_IMPL_SVE_SUM(sum,   addv, u, 64, npy_uint64)
NPYV_IMPL_SVE_SUM(sum,   addv, f, 32, float)
NPYV_IMPL_SVE_SUM(sum,   addv, f, 64, double)
NPYV_IMPL_SVE_SUM(sumup, addv, u, 8,  npy_uint16)
NPYV_IMPL_SVE_SUM(sumup, addv, u, 16, npy_uint32)

#endif  // _NPY_SIMD_SVE_ARITHMETIC_H
