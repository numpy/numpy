#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_OPERATORS_H
#define _NPY_SIMD_LSX_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
#define npyv_shl_u16(A, C) __lsx_vsll_h(A, npyv_setall_s16(C))
#define npyv_shl_s16(A, C) __lsx_vsll_h(A, npyv_setall_s16(C))
#define npyv_shl_u32(A, C) __lsx_vsll_w(A, npyv_setall_s32(C))
#define npyv_shl_s32(A, C) __lsx_vsll_w(A, npyv_setall_s32(C))
#define npyv_shl_u64(A, C) __lsx_vsll_d(A, npyv_setall_s64(C))
#define npyv_shl_s64(A, C) __lsx_vsll_d(A, npyv_setall_s64(C))

// left by an immediate constant
#define npyv_shli_u16 __lsx_vslli_h
#define npyv_shli_s16 __lsx_vslli_h
#define npyv_shli_u32 __lsx_vslli_w
#define npyv_shli_s32 __lsx_vslli_w
#define npyv_shli_u64 __lsx_vslli_d
#define npyv_shli_s64 __lsx_vslli_d

// right
#define npyv_shr_u16(A, C) __lsx_vsrl_h(A, npyv_setall_u16(C))
#define npyv_shr_s16(A, C) __lsx_vsra_h(A, npyv_setall_u16(C))
#define npyv_shr_u32(A, C) __lsx_vsrl_w(A, npyv_setall_u32(C))
#define npyv_shr_s32(A, C) __lsx_vsra_w(A, npyv_setall_u32(C))
#define npyv_shr_u64(A, C) __lsx_vsrl_d(A, npyv_setall_u64(C))
#define npyv_shr_s64(A, C) __lsx_vsra_d(A, npyv_setall_u64(C))

// Right by an immediate constant
#define npyv_shri_u16 __lsx_vsrli_h
#define npyv_shri_s16 __lsx_vsrai_h
#define npyv_shri_u32 __lsx_vsrli_w
#define npyv_shri_s32 __lsx_vsrai_w
#define npyv_shri_u64 __lsx_vsrli_d
#define npyv_shri_s64 __lsx_vsrai_d

/***************************
 * Logical
 ***************************/

// AND
#define npyv_and_u8  __lsx_vand_v
#define npyv_and_s8  __lsx_vand_v
#define npyv_and_u16 __lsx_vand_v
#define npyv_and_s16 __lsx_vand_v
#define npyv_and_u32 __lsx_vand_v
#define npyv_and_s32 __lsx_vand_v
#define npyv_and_u64 __lsx_vand_v
#define npyv_and_s64 __lsx_vand_v
#define npyv_and_f32(A, B)  \
        (__m128)__lsx_vand_v((__m128i)A, (__m128i)B)
#define npyv_and_f64(A, B)  \
        (__m128d)__lsx_vand_v((__m128i)A, (__m128i)B)
#define npyv_and_b8  __lsx_vand_v
#define npyv_and_b16 __lsx_vand_v
#define npyv_and_b32 __lsx_vand_v
#define npyv_and_b64 __lsx_vand_v

// OR
#define npyv_or_u8  __lsx_vor_v
#define npyv_or_s8  __lsx_vor_v
#define npyv_or_u16 __lsx_vor_v
#define npyv_or_s16 __lsx_vor_v
#define npyv_or_u32 __lsx_vor_v
#define npyv_or_s32 __lsx_vor_v
#define npyv_or_u64 __lsx_vor_v
#define npyv_or_s64 __lsx_vor_v
#define npyv_or_f32(A, B)   \
        (__m128)__lsx_vor_v((__m128i)A, (__m128i)B)
#define npyv_or_f64(A, B)   \
        (__m128d)__lsx_vor_v((__m128i)A, (__m128i)B)
#define npyv_or_b8  __lsx_vor_v
#define npyv_or_b16 __lsx_vor_v
#define npyv_or_b32 __lsx_vor_v
#define npyv_or_b64 __lsx_vor_v

// XOR
#define npyv_xor_u8  __lsx_vxor_v
#define npyv_xor_s8  __lsx_vxor_v
#define npyv_xor_u16 __lsx_vxor_v
#define npyv_xor_s16 __lsx_vxor_v
#define npyv_xor_u32 __lsx_vxor_v
#define npyv_xor_s32 __lsx_vxor_v
#define npyv_xor_u64 __lsx_vxor_v
#define npyv_xor_s64 __lsx_vxor_v
#define npyv_xor_f32(A, B)   \
        (__m128)__lsx_vxor_v((__m128i)A, (__m128i)B)
#define npyv_xor_f64(A, B)   \
        (__m128d)__lsx_vxor_v((__m128i)A, (__m128i)B)
#define npyv_xor_b8  __lsx_vxor_v
#define npyv_xor_b16 __lsx_vxor_v
#define npyv_xor_b32 __lsx_vxor_v
#define npyv_xor_b64 __lsx_vxor_v

// NOT
#define npyv_not_u8(A) __lsx_vxori_b((__m128i)A, 0xff)
#define npyv_not_s8  npyv_not_u8
#define npyv_not_u16 npyv_not_u8
#define npyv_not_s16 npyv_not_u8
#define npyv_not_u32 npyv_not_u8
#define npyv_not_s32 npyv_not_u8
#define npyv_not_u64 npyv_not_u8
#define npyv_not_s64 npyv_not_u8
#define npyv_not_f32 (__m128)npyv_not_u8
#define npyv_not_f64 (__m128d)npyv_not_u8
#define npyv_not_b8  npyv_not_u8
#define npyv_not_b16 npyv_not_u8
#define npyv_not_b32 npyv_not_u8
#define npyv_not_b64 npyv_not_u8

// ANDC, ORC and XNOR
#define npyv_andc_u8(A, B) __lsx_vandn_v(B, A)
#define npyv_andc_b8(A, B) __lsx_vandn_v(B, A)
#define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
#define npyv_xnor_b8 __lsx_vseq_b

/***************************
 * Comparison
 ***************************/

// Int Equal
#define npyv_cmpeq_u8  __lsx_vseq_b
#define npyv_cmpeq_s8  __lsx_vseq_b
#define npyv_cmpeq_u16 __lsx_vseq_h
#define npyv_cmpeq_s16 __lsx_vseq_h
#define npyv_cmpeq_u32 __lsx_vseq_w
#define npyv_cmpeq_s32 __lsx_vseq_w
#define npyv_cmpeq_u64 __lsx_vseq_d
#define npyv_cmpeq_s64 __lsx_vseq_d

// Int Not Equal
#define npyv_cmpneq_u8(A, B)  npyv_not_u8(npyv_cmpeq_u8(A, B))
#define npyv_cmpneq_u16(A, B) npyv_not_u16(npyv_cmpeq_u16(A, B))
#define npyv_cmpneq_u32(A, B) npyv_not_u32(npyv_cmpeq_u32(A, B))
#define npyv_cmpneq_u64(A, B) npyv_not_u64(npyv_cmpeq_u64(A, B))
#define npyv_cmpneq_s8  npyv_cmpneq_u8
#define npyv_cmpneq_s16 npyv_cmpneq_u16
#define npyv_cmpneq_s32 npyv_cmpneq_u32
#define npyv_cmpneq_s64 npyv_cmpneq_u64

// signed greater than
#define npyv_cmpgt_s8(A, B)  __lsx_vslt_b(B, A)
#define npyv_cmpgt_s16(A, B) __lsx_vslt_h(B, A)
#define npyv_cmpgt_s32(A, B) __lsx_vslt_w(B, A)
#define npyv_cmpgt_s64(A, B) __lsx_vslt_d(B, A)

// signed greater than or equal
#define npyv_cmpge_s8(A, B)  __lsx_vsle_b(B, A)
#define npyv_cmpge_s16(A, B) __lsx_vsle_h(B, A)
#define npyv_cmpge_s32(A, B) __lsx_vsle_w(B, A)
#define npyv_cmpge_s64(A, B) __lsx_vsle_d(B, A)

// unsigned greater than
#define npyv_cmpgt_u8(A, B)  __lsx_vslt_bu(B, A)
#define npyv_cmpgt_u16(A, B) __lsx_vslt_hu(B, A)
#define npyv_cmpgt_u32(A, B) __lsx_vslt_wu(B, A)
#define npyv_cmpgt_u64(A, B) __lsx_vslt_du(B, A)

// unsigned greater than or equal
#define npyv_cmpge_u8(A, B)  __lsx_vsle_bu(B, A)
#define npyv_cmpge_u16(A, B) __lsx_vsle_hu(B, A)
#define npyv_cmpge_u32(A, B) __lsx_vsle_wu(B, A)
#define npyv_cmpge_u64(A, B) __lsx_vsle_du(B, A)

// less than
#define npyv_cmplt_u8  __lsx_vslt_bu
#define npyv_cmplt_s8  __lsx_vslt_b
#define npyv_cmplt_u16 __lsx_vslt_hu
#define npyv_cmplt_s16 __lsx_vslt_h
#define npyv_cmplt_u32 __lsx_vslt_wu
#define npyv_cmplt_s32 __lsx_vslt_w
#define npyv_cmplt_u64 __lsx_vslt_du
#define npyv_cmplt_s64 __lsx_vslt_d

// less than or equal
#define npyv_cmple_u8  __lsx_vsle_bu
#define npyv_cmple_s8  __lsx_vsle_b
#define npyv_cmple_u16 __lsx_vsle_hu
#define npyv_cmple_s16 __lsx_vsle_h
#define npyv_cmple_u32 __lsx_vsle_wu
#define npyv_cmple_s32 __lsx_vsle_w
#define npyv_cmple_u64 __lsx_vsle_du
#define npyv_cmple_s64 __lsx_vsle_d

// precision comparison
#define npyv_cmpeq_f32  __lsx_vfcmp_ceq_s
#define npyv_cmpeq_f64  __lsx_vfcmp_ceq_d
#define npyv_cmpneq_f32 __lsx_vfcmp_cune_s
#define npyv_cmpneq_f64 __lsx_vfcmp_cune_d
#define npyv_cmplt_f32  __lsx_vfcmp_clt_s
#define npyv_cmplt_f64  __lsx_vfcmp_clt_d
#define npyv_cmple_f32  __lsx_vfcmp_cle_s
#define npyv_cmple_f64  __lsx_vfcmp_cle_d
#define npyv_cmpgt_f32(A, B) npyv_cmplt_f32(B, A)
#define npyv_cmpgt_f64(A, B) npyv_cmplt_f64(B, A)
#define npyv_cmpge_f32(A, B) npyv_cmple_f32(B, A)
#define npyv_cmpge_f64(A, B) npyv_cmple_f64(B, A)

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return __lsx_vfcmp_cor_s(a, a); }    //!nan,return:ffffffff
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return __lsx_vfcmp_cor_d(a, a); }

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
#define NPYV_IMPL_LSX_ANYALL(SFX)                   \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)   \
    { return __lsx_vmsknz_b((__m128i)a)[0] != 0; }  \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)   \
    { return __lsx_vmsknz_b((__m128i)a)[0] == 0xffff; }
NPYV_IMPL_LSX_ANYALL(b8)
NPYV_IMPL_LSX_ANYALL(b16)
NPYV_IMPL_LSX_ANYALL(b32)
NPYV_IMPL_LSX_ANYALL(b64)
#undef NPYV_IMPL_LSX_ANYALL

#define NPYV_IMPL_LSX_ANYALL(SFX, TSFX, MASK)       \
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)   \
    {                                               \
        return  __lsx_vmsknz_b(a)[0] != 0;          \
    }                                               \
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)   \
    {                                               \
        return __lsx_vmsknz_b(                      \
            __lsx_vseq_##TSFX(a, npyv_zero_##SFX()) \
        )[0] == 0;                                  \
    }
NPYV_IMPL_LSX_ANYALL(u8,  b, 0xffff)
NPYV_IMPL_LSX_ANYALL(s8,  b, 0xffff)
NPYV_IMPL_LSX_ANYALL(u16, h, 0xffff)
NPYV_IMPL_LSX_ANYALL(s16, h, 0xffff)
NPYV_IMPL_LSX_ANYALL(u32, w, 0xffff)
NPYV_IMPL_LSX_ANYALL(s32, w, 0xffff)
NPYV_IMPL_LSX_ANYALL(u64, d, 0xffff)
NPYV_IMPL_LSX_ANYALL(s64, d, 0xffff)
#undef NPYV_IMPL_LSX_ANYALL

NPY_FINLINE bool npyv_any_f32(npyv_f32 a)
{
    return __lsx_vmsknz_b(__lsx_vfcmp_ceq_s(a, npyv_zero_f32()))[0] != 0xffff;
}
NPY_FINLINE bool npyv_all_f32(npyv_f32 a)
{
    return __lsx_vmsknz_b(__lsx_vfcmp_ceq_s(a, npyv_zero_f32()))[0] == 0;
}
NPY_FINLINE bool npyv_any_f64(npyv_f64 a)
{
    return __lsx_vmsknz_b(__lsx_vfcmp_ceq_d(a, npyv_zero_f64()))[0] != 0xffff;
}
NPY_FINLINE bool npyv_all_f64(npyv_f64 a)
{
    return __lsx_vmsknz_b(__lsx_vfcmp_ceq_d(a, npyv_zero_f64()))[0] == 0;
}
#endif // _NPY_SIMD_LSX_OPERATORS_H
