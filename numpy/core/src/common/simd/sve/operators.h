#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_OPERATORS_H
#define _NPY_SIMD_SVE_OPERATORS_H

/***************************
 * Shifting
 ***************************/
// left
#define NPYV_IMPL_SVE_SHIFT_L(S, W)                      \
    NPY_FINLINE npyv_##S##W npyv_shl_##S##W(             \
            npyv_##S##W a, npy_uintp b)                  \
    {                                                    \
        return svlsl_##S##W##_x(svptrue_b##W(), a,       \
                                       svdup_n_u##W(b)); \
    }

NPYV_IMPL_SVE_SHIFT_L(u, 8)
NPYV_IMPL_SVE_SHIFT_L(u, 16)
NPYV_IMPL_SVE_SHIFT_L(u, 32)
NPYV_IMPL_SVE_SHIFT_L(u, 64)
NPYV_IMPL_SVE_SHIFT_L(s, 8)
NPYV_IMPL_SVE_SHIFT_L(s, 16)
NPYV_IMPL_SVE_SHIFT_L(s, 32)
NPYV_IMPL_SVE_SHIFT_L(s, 64)

// left by an immediate constant
#define NPYV_IMPL_SVE_SHIFT_L_IMM(W)                    \
    NPY_FINLINE npyv_u##W npyv_shli_u##W(npyv_u##W a,   \
            npy_uint##W b)                              \
    {                                                   \
        return svlsl_n_u##W##_x(svptrue_b##W(), a, b);  \
    }                                                   \
    NPY_FINLINE npyv_s##W npyv_shli_s##W(npyv_s##W a,   \
            npy_uint##W b)                              \
    {                                                   \
        return svlsl_n_s##W##_x(svptrue_b##W(), a, b);  \
    }

NPYV_IMPL_SVE_SHIFT_L_IMM(8)
NPYV_IMPL_SVE_SHIFT_L_IMM(16)
NPYV_IMPL_SVE_SHIFT_L_IMM(32)
NPYV_IMPL_SVE_SHIFT_L_IMM(64)

// right
#define NPYV_IMPL_SVE_SHIFT_R(W)                                 \
    NPY_FINLINE npyv_u##W npyv_shr_u##W(npyv_u##W a, npy_intp b) \
    {                                                            \
        return svlsr_u##W##_x(svptrue_b##W(), a,                 \
                svdup_n_u##W(b));                                \
    }                                                            \
    NPY_FINLINE npyv_s##W npyv_shr_s##W(npyv_s##W a, npy_intp b) \
    {                                                            \
        return svasr_s##W##_x(svptrue_b##W(), a,                 \
                svdup_n_u##W(b));                                \
    }

NPYV_IMPL_SVE_SHIFT_R(8)
NPYV_IMPL_SVE_SHIFT_R(16)
NPYV_IMPL_SVE_SHIFT_R(32)
NPYV_IMPL_SVE_SHIFT_R(64)

// right by an immediate constant
#define NPYV_IMPL_SVE_SHIFT_R_IMM(W)                   \
    NPY_FINLINE npyv_u##W npyv_shri_u##W(npyv_u##W a,  \
            npy_uint##W b)                             \
    {                                                  \
        return svlsr_n_u##W##_x(svptrue_b##W(), a, b); \
    }                                                  \
    NPY_FINLINE npyv_s##W npyv_shri_s##W(npyv_s##W a,  \
            npy_uint##W b)                             \
    {                                                  \
        return svasr_n_s##W##_x(svptrue_b##W(), a, b); \
    }

NPYV_IMPL_SVE_SHIFT_R_IMM(8)
NPYV_IMPL_SVE_SHIFT_R_IMM(16)
NPYV_IMPL_SVE_SHIFT_R_IMM(32)
NPYV_IMPL_SVE_SHIFT_R_IMM(64)

/***************************
 * Logical
 ***************************/
#define NPYV_IMPL_SVE_LOGICAL(SFX0, SFX1, SFX2)                            \
    NPY_FINLINE npyv_##SFX0 npyv_and_##SFX0(npyv_##SFX0 a, npyv_##SFX0 b)  \
    {                                                                      \
        return svreinterpret_##SFX0##_##SFX2(svand_##SFX2##_x(             \
                svptrue_##SFX1(), svreinterpret_##SFX2##_##SFX0(a),        \
                svreinterpret_##SFX2##_##SFX0(b)));                        \
    }                                                                      \
    NPY_FINLINE npyv_##SFX0 npyv_or_##SFX0(npyv_##SFX0 a, npyv_##SFX0 b)   \
    {                                                                      \
        return svreinterpret_##SFX0##_##SFX2(svorr_##SFX2##_x(             \
                svptrue_##SFX1(), svreinterpret_##SFX2##_##SFX0(a),        \
                svreinterpret_##SFX2##_##SFX0(b)));                        \
    }                                                                      \
    NPY_FINLINE npyv_##SFX0 npyv_xor_##SFX0(npyv_##SFX0 a, npyv_##SFX0 b)  \
    {                                                                      \
        return svreinterpret_##SFX0##_##SFX2(sveor_##SFX2##_x(             \
                svptrue_##SFX1(), svreinterpret_##SFX2##_##SFX0(a),        \
                svreinterpret_##SFX2##_##SFX0(b)));                        \
    }                                                                      \
    NPY_FINLINE npyv_##SFX0 npyv_not_##SFX0(npyv_##SFX0 a)                 \
    {                                                                      \
        return svreinterpret_##SFX0##_##SFX2(svnot_##SFX2##_x(             \
                svptrue_##SFX1(), svreinterpret_##SFX2##_##SFX0(a)));      \
    }                                                                      \
    NPY_FINLINE npyv_##SFX0 npyv_andc_##SFX0(npyv_##SFX0 a, npyv_##SFX0 b) \
    {                                                                      \
        return svreinterpret_##SFX0##_##SFX2(svbic_##SFX2##_x(             \
                svptrue_##SFX1(), svreinterpret_##SFX2##_##SFX0(a),        \
                svreinterpret_##SFX2##_##SFX0(b)));                        \
    }
NPYV_IMPL_SVE_LOGICAL(u8, b8, u8)
NPYV_IMPL_SVE_LOGICAL(u16, b16, u16)
NPYV_IMPL_SVE_LOGICAL(u32, b32, u32)
NPYV_IMPL_SVE_LOGICAL(u64, b64, u64)
NPYV_IMPL_SVE_LOGICAL(s8, b8, u8)
NPYV_IMPL_SVE_LOGICAL(s16, b16, u16)
NPYV_IMPL_SVE_LOGICAL(s32, b32, u32)
NPYV_IMPL_SVE_LOGICAL(s64, b64, u64)
NPYV_IMPL_SVE_LOGICAL(f32, b32, u32)
NPYV_IMPL_SVE_LOGICAL(f64, b64, u64)

/***************************
 * Logical (boolean)
 ***************************/
#define NPYV_IMPL_SVE_LOGICAL_MASK(SFX)                                \
    NPY_FINLINE npyv_##SFX npyv_and_##SFX(npyv_##SFX a, npyv_##SFX b)  \
    {                                                                  \
        return svand_b_z(svptrue_##SFX(), a, b);                       \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_or_##SFX(npyv_##SFX a, npyv_##SFX b)   \
    {                                                                  \
        return svorr_b_z(svptrue_b8(), a, b);                          \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_xor_##SFX(npyv_##SFX a, npyv_##SFX b)  \
    {                                                                  \
        return sveor_b_z(svptrue_b8(), a, b);                          \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_not_##SFX(npyv_##SFX a)                \
    {                                                                  \
        return svnot_b_z(svptrue_b8(), a);                             \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_andc_##SFX(npyv_##SFX a, npyv_##SFX b) \
    {                                                                  \
      return svbic_b_z(svptrue_b8(), a, b);                            \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_orc_##SFX(npyv_##SFX a, npyv_##SFX b)  \
    {                                                                  \
      return npyv_or_##SFX(npyv_not_##SFX(b), a);                      \
    }                                                                  \
    NPY_FINLINE npyv_##SFX npyv_xnor_##SFX(npyv_##SFX a, npyv_##SFX b) \
    {                                                                  \
      return npyv_not_##SFX(npyv_xor_##SFX(a, b));                     \
    }
NPYV_IMPL_SVE_LOGICAL_MASK(b8)
NPYV_IMPL_SVE_LOGICAL_MASK(b16)
NPYV_IMPL_SVE_LOGICAL_MASK(b32)
NPYV_IMPL_SVE_LOGICAL_MASK(b64)

/***************************
 * Comparison
 ***************************/
// int equal
#define NPYV_IMPL_SVE_COMPARE(SFX, BSFX, SFX1)                \
    NPY_FINLINE npyv_##BSFX npyv_cmpeq_##SFX(SFX1 a, SFX1 b)  \
    {                                                         \
       return svcmpeq_##SFX(svptrue_##BSFX(), a, b);          \
    }                                                         \
    NPY_FINLINE npyv_##BSFX npyv_cmpneq_##SFX(SFX1 a, SFX1 b) \
    {                                                         \
        return svcmpne_##SFX(svptrue_##BSFX(), a, b);         \
    }                                                         \
    NPY_FINLINE npyv_##BSFX npyv_cmpgt_##SFX(SFX1 a, SFX1 b)  \
    {                                                         \
        return svcmpgt_##SFX(svptrue_##BSFX(), a, b);         \
    }                                                         \
    NPY_FINLINE npyv_##BSFX npyv_cmpge_##SFX(SFX1 a, SFX1 b)  \
    {                                                         \
        return svcmpge_##SFX(svptrue_##BSFX(), a, b);         \
    }                                                         \
    NPY_FINLINE npyv_##BSFX npyv_cmplt_##SFX(SFX1 a, SFX1 b)  \
    {                                                         \
        return svcmplt_##SFX(svptrue_##BSFX(), a, b);         \
    }                                                         \
    NPY_FINLINE npyv_##BSFX npyv_cmple_##SFX(SFX1 a, SFX1 b)  \
    {                                                         \
        return svcmple_##SFX(svptrue_##BSFX(), a, b);         \
    }

NPYV_IMPL_SVE_COMPARE(u8, b8, svuint8_t)
NPYV_IMPL_SVE_COMPARE(u16, b16, svuint16_t)
NPYV_IMPL_SVE_COMPARE(u32, b32, svuint32_t)
NPYV_IMPL_SVE_COMPARE(u64, b64, svuint64_t)
NPYV_IMPL_SVE_COMPARE(s8, b8, svint8_t)
NPYV_IMPL_SVE_COMPARE(s16, b16, svint16_t)
NPYV_IMPL_SVE_COMPARE(s32, b32, svint32_t)
NPYV_IMPL_SVE_COMPARE(s64, b64, svint64_t)
NPYV_IMPL_SVE_COMPARE(f32, b32, svfloat32_t)
NPYV_IMPL_SVE_COMPARE(f64, b64, svfloat64_t)

// check special cases
NPY_FINLINE npyv_b32
npyv_notnan_f32(npyv_f32 a)
{
    return svcmpeq(svptrue_b32(), a, a);
}
NPY_FINLINE npyv_b64
npyv_notnan_f64(npyv_f64 a)
{
    return svcmpeq(svptrue_b64(), a, a);
}

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
#define NPYV_IMPL_SVE_ANYALL(SFX, T)	\
  NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a) \
  { return svcntp_##SFX(svptrue_##SFX(), a) != 0; } \
  NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a) \
  { return svcntp_##SFX(svptrue_##SFX(), a) == svcnt##T(); }
NPYV_IMPL_SVE_ANYALL(b8,  b)
NPYV_IMPL_SVE_ANYALL(b16, h)
NPYV_IMPL_SVE_ANYALL(b32, w)
NPYV_IMPL_SVE_ANYALL(b64, d)
#undef NPYV_IMPL_SVE_ANYALL

#define NPYV_IMPL_SVE_ANYALL(SFX, BSFX, T)	\
  NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a) \
  { \
    return svorv_##SFX(svptrue_##BSFX(), a) != 0; \
  } \
  NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a) \
  { \
  const svbool_t cmp = svcmpne_n_##SFX(svptrue_##BSFX(), a, 0); \
  return svcntp_##BSFX(svptrue_##BSFX(), cmp) == svcnt##T(); \
  }
NPYV_IMPL_SVE_ANYALL(u8,  b8,  b)
NPYV_IMPL_SVE_ANYALL(u16, b16, h)
NPYV_IMPL_SVE_ANYALL(u32, b32, w)
NPYV_IMPL_SVE_ANYALL(u64, b64, d)
NPYV_IMPL_SVE_ANYALL(s8,  b8,  b)
NPYV_IMPL_SVE_ANYALL(s16, b16, h)
NPYV_IMPL_SVE_ANYALL(s32, b32, w)
NPYV_IMPL_SVE_ANYALL(s64, b64, d)
#undef NPYV_IMPL_SVE_ANYALL

#define NPYV_IMPL_SVE_ANYALL(SFX, BSFX, T) \
  NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a) \
{ \
  const svbool_t cmp = svcmpne_n_##SFX(svptrue_##BSFX(), a, 0); \
  return svcntp_##BSFX(svptrue_##BSFX(), cmp) != 0; \
} \
  NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a) \
{ \
  const svbool_t cmp = svcmpne_n_##SFX(svptrue_##BSFX(), a, 0); \
  return svcntp_##BSFX(svptrue_##BSFX(), cmp) == svcnt##T(); \
}
NPYV_IMPL_SVE_ANYALL(f32, b32, w)
NPYV_IMPL_SVE_ANYALL(f64, b64, d)
#undef NPYV_IMPL_SVE_ANYALL

#endif  // _NPY_SIMD_SVE_OPERATORS_H
