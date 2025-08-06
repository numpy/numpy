#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_OPERATORS_H
#define _NPY_SIMD_VEC_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// Left
#define npyv_shl_u16(A, C) vec_sl(A, npyv_setall_u16(C))
#define npyv_shl_s16(A, C) vec_sl_s16(A, npyv_setall_u16(C))
#define npyv_shl_u32(A, C) vec_sl(A, npyv_setall_u32(C))
#define npyv_shl_s32(A, C) vec_sl_s32(A, npyv_setall_u32(C))
#define npyv_shl_u64(A, C) vec_sl(A, npyv_setall_u64(C))
#define npyv_shl_s64(A, C) vec_sl_s64(A, npyv_setall_u64(C))

// Left by an immediate constant
#define npyv_shli_u16 npyv_shl_u16
#define npyv_shli_s16 npyv_shl_s16
#define npyv_shli_u32 npyv_shl_u32
#define npyv_shli_s32 npyv_shl_s32
#define npyv_shli_u64 npyv_shl_u64
#define npyv_shli_s64 npyv_shl_s64

// Right
#define npyv_shr_u16(A, C) vec_sr(A,  npyv_setall_u16(C))
#define npyv_shr_s16(A, C) vec_sra_s16(A, npyv_setall_u16(C))
#define npyv_shr_u32(A, C) vec_sr(A,  npyv_setall_u32(C))
#define npyv_shr_s32(A, C) vec_sra_s32(A, npyv_setall_u32(C))
#define npyv_shr_u64(A, C) vec_sr(A,  npyv_setall_u64(C))
#define npyv_shr_s64(A, C) vec_sra_s64(A, npyv_setall_u64(C))

// Right by an immediate constant
#define npyv_shri_u16 npyv_shr_u16
#define npyv_shri_s16 npyv_shr_s16
#define npyv_shri_u32 npyv_shr_u32
#define npyv_shri_s32 npyv_shr_s32
#define npyv_shri_u64 npyv_shr_u64
#define npyv_shri_s64 npyv_shr_s64

/***************************
 * Logical
 ***************************/
#define NPYV_IMPL_VEC_BIN_WRAP(INTRIN, SFX) \
    NPY_FINLINE npyv_##SFX npyv_##INTRIN##_##SFX(npyv_##SFX a, npyv_##SFX b) \
    { return vec_##INTRIN(a, b); }

#define NPYV_IMPL_VEC_BIN_CAST(INTRIN, SFX, CAST) \
    NPY_FINLINE npyv_##SFX npyv_##INTRIN##_##SFX(npyv_##SFX a, npyv_##SFX b) \
    { return (npyv_##SFX)vec_##INTRIN((CAST)a, (CAST)b); }

// Up to GCC 6 logical intrinsics don't support bool long long
#if defined(__GNUC__) && __GNUC__ <= 6
    #define NPYV_IMPL_VEC_BIN_B64(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, b64, npyv_u64)
#else
    #define NPYV_IMPL_VEC_BIN_B64(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, b64, npyv_b64)
#endif

// Up to clang __VEC__ 10305 logical intrinsics do not support f32 or f64
#if defined(NPY_HAVE_VX) && defined(__clang__) && __VEC__ < 10305
    #define NPYV_IMPL_VEC_BIN_F32(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, f32, npyv_u32)
    #define NPYV_IMPL_VEC_BIN_F64(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, f64, npyv_u64)
#else
    #define NPYV_IMPL_VEC_BIN_F32(INTRIN) NPYV_IMPL_VEC_BIN_WRAP(INTRIN, f32)
    #define NPYV_IMPL_VEC_BIN_F64(INTRIN) NPYV_IMPL_VEC_BIN_WRAP(INTRIN, f64)
#endif
// AND
#define npyv_and_u8  vec_and
#define npyv_and_s8  vec_and
#define npyv_and_u16 vec_and
#define npyv_and_s16 vec_and
#define npyv_and_u32 vec_and
#define npyv_and_s32 vec_and
#define npyv_and_u64 vec_and
#define npyv_and_s64 vec_and
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_BIN_F32(and)
#endif
NPYV_IMPL_VEC_BIN_F64(and)
#define npyv_and_b8  vec_and
#define npyv_and_b16 vec_and
#define npyv_and_b32 vec_and
NPYV_IMPL_VEC_BIN_B64(and)

// OR
#define npyv_or_u8  vec_or
#define npyv_or_s8  vec_or
#define npyv_or_u16 vec_or
#define npyv_or_s16 vec_or
#define npyv_or_u32 vec_or
#define npyv_or_s32 vec_or
#define npyv_or_u64 vec_or
#define npyv_or_s64 vec_or
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_BIN_F32(or)
#endif
NPYV_IMPL_VEC_BIN_F64(or)
#define npyv_or_b8  vec_or
#define npyv_or_b16 vec_or
#define npyv_or_b32 vec_or
NPYV_IMPL_VEC_BIN_B64(or)

// XOR
#define npyv_xor_u8  vec_xor
#define npyv_xor_s8  vec_xor
#define npyv_xor_u16 vec_xor
#define npyv_xor_s16 vec_xor
#define npyv_xor_u32 vec_xor
#define npyv_xor_s32 vec_xor
#define npyv_xor_u64 vec_xor
#define npyv_xor_s64 vec_xor
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_BIN_F32(xor)
#endif
NPYV_IMPL_VEC_BIN_F64(xor)
#define npyv_xor_b8  vec_xor
#define npyv_xor_b16 vec_xor
#define npyv_xor_b32 vec_xor
NPYV_IMPL_VEC_BIN_B64(xor)

// NOT
// note: we implement npyv_not_b*(boolean types) for internal use*/
#define NPYV_IMPL_VEC_NOT_INT(VEC_LEN)                                 \
    NPY_FINLINE npyv_u##VEC_LEN npyv_not_u##VEC_LEN(npyv_u##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_s##VEC_LEN npyv_not_s##VEC_LEN(npyv_s##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_b##VEC_LEN npyv_not_b##VEC_LEN(npyv_b##VEC_LEN a) \
    { return vec_nor(a, a); }

NPYV_IMPL_VEC_NOT_INT(8)
NPYV_IMPL_VEC_NOT_INT(16)
NPYV_IMPL_VEC_NOT_INT(32)

// on ppc64, up to gcc5 vec_nor doesn't support bool long long
#if defined(NPY_HAVE_VSX) && defined(__GNUC__) && __GNUC__ > 5
    NPYV_IMPL_VEC_NOT_INT(64)
#else
    NPY_FINLINE npyv_u64 npyv_not_u64(npyv_u64 a)
    { return vec_nor(a, a); }
    NPY_FINLINE npyv_s64 npyv_not_s64(npyv_s64 a)
    { return vec_nor(a, a); }
    NPY_FINLINE npyv_b64 npyv_not_b64(npyv_b64 a)
    { return (npyv_b64)vec_nor((npyv_u64)a, (npyv_u64)a); }
#endif

#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_not_f32(npyv_f32 a)
    { return vec_nor(a, a); }
#endif
NPY_FINLINE npyv_f64 npyv_not_f64(npyv_f64 a)
{ return vec_nor(a, a); }

// ANDC, ORC and XNOR
#define npyv_andc_u8 vec_andc
#define npyv_andc_b8 vec_andc
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_orc_b8 vec_orc
    #define npyv_xnor_b8 vec_eqv
#else
    #define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
    #define npyv_xnor_b8(A, B) npyv_not_b8(npyv_xor_b8(B, A))
#endif

/***************************
 * Comparison
 ***************************/

// Int Equal
#define npyv_cmpeq_u8  vec_cmpeq
#define npyv_cmpeq_s8  vec_cmpeq
#define npyv_cmpeq_u16 vec_cmpeq
#define npyv_cmpeq_s16 vec_cmpeq
#define npyv_cmpeq_u32 vec_cmpeq
#define npyv_cmpeq_s32 vec_cmpeq
#define npyv_cmpeq_u64 vec_cmpeq
#define npyv_cmpeq_s64 vec_cmpeq
#if NPY_SIMD_F32
    #define npyv_cmpeq_f32 vec_cmpeq
#endif
#define npyv_cmpeq_f64 vec_cmpeq

// Int Not Equal
#if defined(NPY_HAVE_VSX3) && (!defined(__GNUC__) || defined(vec_cmpne))
    // vec_cmpne supported by gcc since version 7
    #define npyv_cmpneq_u8  vec_cmpne
    #define npyv_cmpneq_s8  vec_cmpne
    #define npyv_cmpneq_u16 vec_cmpne
    #define npyv_cmpneq_s16 vec_cmpne
    #define npyv_cmpneq_u32 vec_cmpne
    #define npyv_cmpneq_s32 vec_cmpne
    #define npyv_cmpneq_u64 vec_cmpne
    #define npyv_cmpneq_s64 vec_cmpne
    #define npyv_cmpneq_f32 vec_cmpne
    #define npyv_cmpneq_f64 vec_cmpne
#else
    #define npyv_cmpneq_u8(A, B)  npyv_not_b8(vec_cmpeq(A, B))
    #define npyv_cmpneq_s8(A, B)  npyv_not_b8(vec_cmpeq(A, B))
    #define npyv_cmpneq_u16(A, B) npyv_not_b16(vec_cmpeq(A, B))
    #define npyv_cmpneq_s16(A, B) npyv_not_b16(vec_cmpeq(A, B))
    #define npyv_cmpneq_u32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    #define npyv_cmpneq_s32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    #define npyv_cmpneq_u64(A, B) npyv_not_b64(vec_cmpeq(A, B))
    #define npyv_cmpneq_s64(A, B) npyv_not_b64(vec_cmpeq(A, B))
    #if NPY_SIMD_F32
        #define npyv_cmpneq_f32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    #endif
    #define npyv_cmpneq_f64(A, B) npyv_not_b64(vec_cmpeq(A, B))
#endif

// Greater than
#define npyv_cmpgt_u8  vec_cmpgt
#define npyv_cmpgt_s8  vec_cmpgt
#define npyv_cmpgt_u16 vec_cmpgt
#define npyv_cmpgt_s16 vec_cmpgt
#define npyv_cmpgt_u32 vec_cmpgt
#define npyv_cmpgt_s32 vec_cmpgt
#define npyv_cmpgt_u64 vec_cmpgt
#define npyv_cmpgt_s64 vec_cmpgt
#if NPY_SIMD_F32
    #define npyv_cmpgt_f32 vec_cmpgt
#endif
#define npyv_cmpgt_f64 vec_cmpgt

// Greater than or equal
// On ppc64le, up to gcc5 vec_cmpge only supports single and double precision
#if defined(NPY_HAVE_VX) || (defined(__GNUC__) && __GNUC__ > 5)
    #define npyv_cmpge_u8  vec_cmpge
    #define npyv_cmpge_s8  vec_cmpge
    #define npyv_cmpge_u16 vec_cmpge
    #define npyv_cmpge_s16 vec_cmpge
    #define npyv_cmpge_u32 vec_cmpge
    #define npyv_cmpge_s32 vec_cmpge
    #define npyv_cmpge_u64 vec_cmpge
    #define npyv_cmpge_s64 vec_cmpge
#else
    #define npyv_cmpge_u8(A, B)  npyv_not_b8(vec_cmpgt(B, A))
    #define npyv_cmpge_s8(A, B)  npyv_not_b8(vec_cmpgt(B, A))
    #define npyv_cmpge_u16(A, B) npyv_not_b16(vec_cmpgt(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_b16(vec_cmpgt(B, A))
    #define npyv_cmpge_u32(A, B) npyv_not_b32(vec_cmpgt(B, A))
    #define npyv_cmpge_s32(A, B) npyv_not_b32(vec_cmpgt(B, A))
    #define npyv_cmpge_u64(A, B) npyv_not_b64(vec_cmpgt(B, A))
    #define npyv_cmpge_s64(A, B) npyv_not_b64(vec_cmpgt(B, A))
#endif
#if NPY_SIMD_F32
    #define npyv_cmpge_f32 vec_cmpge
#endif
#define npyv_cmpge_f64 vec_cmpge

// Less than
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)
#if NPY_SIMD_F32
    #define npyv_cmplt_f32(A, B) npyv_cmpgt_f32(B, A)
#endif
#define npyv_cmplt_f64(A, B) npyv_cmpgt_f64(B, A)

// Less than or equal
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)
#if NPY_SIMD_F32
    #define npyv_cmple_f32(A, B) npyv_cmpge_f32(B, A)
#endif
#define npyv_cmple_f64(A, B) npyv_cmpge_f64(B, A)

// check special cases
#if NPY_SIMD_F32
    NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
    { return vec_cmpeq(a, a); }
#endif
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return vec_cmpeq(a, a); }

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
#define NPYV_IMPL_VEC_ANYALL(SFX, SFX2)                       \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)             \
    { return vec_any_ne(a, (npyv_##SFX)npyv_zero_##SFX2()); } \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)             \
    { return vec_all_ne(a, (npyv_##SFX)npyv_zero_##SFX2()); }
NPYV_IMPL_VEC_ANYALL(b8,  u8)
NPYV_IMPL_VEC_ANYALL(b16, u16)
NPYV_IMPL_VEC_ANYALL(b32, u32)
NPYV_IMPL_VEC_ANYALL(b64, u64)
NPYV_IMPL_VEC_ANYALL(u8,  u8)
NPYV_IMPL_VEC_ANYALL(s8,  s8)
NPYV_IMPL_VEC_ANYALL(u16, u16)
NPYV_IMPL_VEC_ANYALL(s16, s16)
NPYV_IMPL_VEC_ANYALL(u32, u32)
NPYV_IMPL_VEC_ANYALL(s32, s32)
NPYV_IMPL_VEC_ANYALL(u64, u64)
NPYV_IMPL_VEC_ANYALL(s64, s64)
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_ANYALL(f32, f32)
#endif
NPYV_IMPL_VEC_ANYALL(f64, f64)
#undef NPYV_IMPL_VEC_ANYALL

#endif // _NPY_SIMD_VEC_OPERATORS_H
