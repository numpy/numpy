#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_OPERATORS_H
#define _NPY_SIMD_RVV_OPERATORS_H

/***************************
 * Shifting
 ***************************/
// left
NPY_FINLINE npyv_u16 npyv_shl_u16(npyv_u16 a, int16_t c)
{
    npyv_s16 b = npyv_setall_s16(c);
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_u16 shl = __riscv_vsll_vv_u16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vuint32m2_t a_ext = __riscv_vzext_vf2_u32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_u16 shr = __riscv_vnclipu_wv_u16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_u16m1(shr, shl, positive_mask, 8);
}

NPY_FINLINE npyv_s16 npyv_shl_s16(npyv_s16 a, int16_t c)
{
    npyv_s16 b = npyv_setall_s16(c);
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_s16 shl = __riscv_vsll_vv_i16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vint32m2_t a_ext = __riscv_vsext_vf2_i32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_s16 shr = __riscv_vnclip_wv_i16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_i16m1(shr, shl, positive_mask, 8);
}

NPY_FINLINE npyv_u32 npyv_shl_u32(npyv_u32 a, int32_t c)
{
    npyv_s32 b = npyv_setall_s32(c);
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_u32 shl = __riscv_vsll_vv_u32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vuint64m2_t a_ext = __riscv_vzext_vf2_u64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_u32 shr = __riscv_vnclipu_wv_u32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_u32m1(shr, shl, positive_mask, 4);
}

NPY_FINLINE npyv_s32 npyv_shl_s32(npyv_s32 a, int32_t c)
{
    npyv_s32 b = npyv_setall_s32(c);
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_s32 shl = __riscv_vsll_vv_i32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vint64m2_t a_ext = __riscv_vsext_vf2_i64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_s32 shr = __riscv_vnclip_wv_i32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_i32m1(shr, shl, positive_mask, 4);
}

NPY_FINLINE npyv_u64 npyv_shl_u64(npyv_u64 a, int64_t c)
{
    npyv_s64 b = npyv_setall_s64(c);
    vbool64_t positive_mask = __riscv_vmsgt_vx_i64m1_b64(b, 0, 2);
    npyv_u64 shl = __riscv_vsll_vv_u64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b), 2);
    npyv_s64 b_neg = __riscv_vneg_v_i64m1(b, 2);
    npyv_u64 shr = __riscv_vsrl_vv_u64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b_neg), 2);
    return __riscv_vmerge_vvm_u64m1(shr, shl, positive_mask, 2);
}

NPY_FINLINE npyv_s64 npyv_shl_s64(npyv_s64 a, int64_t c)
{
    npyv_s64 b = npyv_setall_s64(c);
    vbool64_t positive_mask = __riscv_vmsgt_vx_i64m1_b64(b, 0, 2);
    npyv_s64 shl = __riscv_vsll_vv_i64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b), 2);
    npyv_s64 b_neg = __riscv_vneg_v_i64m1(b, 2);
    npyv_s64 shr = __riscv_vsra_vv_i64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b_neg), 2);
    return __riscv_vmerge_vvm_i64m1(shr, shl, positive_mask, 2);
}

// left by an immediate constant
NPY_FINLINE npyv_u16 npyv_shli_u16(npyv_u16 a, const int b)
{
    return __riscv_vsll_vx_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_shli_s16(npyv_s16 a, const int b)
{
    return __riscv_vsll_vx_i16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_shli_u32(npyv_u32 a, const int b)
{
    return __riscv_vsll_vx_u32m1(a, b, 4);
}

NPY_FINLINE npyv_s32 npyv_shli_s32(npyv_s32 a, const int b)
{ 
    return __riscv_vsll_vx_i32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_shli_u64(npyv_u64 a, const int b)
{
    return __riscv_vsll_vx_u64m1(a, b, 2);
}

NPY_FINLINE npyv_s64 npyv_shli_s64(npyv_s64 a, const int b)
{
    return __riscv_vsll_vx_i64m1(a, b, 2);
}

// right
NPY_FINLINE npyv_u16 npyv_shr_u16(npyv_u16 a, int16_t c)
{
    npyv_s16 b = npyv_setall_s16(-(c));
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_u16 shl = __riscv_vsll_vv_u16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vuint32m2_t a_ext = __riscv_vzext_vf2_u32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_u16 shr = __riscv_vnclipu_wv_u16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_u16m1(shr, shl, positive_mask, 8);
}

NPY_FINLINE npyv_s16 npyv_shr_s16(npyv_s16 a, int16_t c)
{
    npyv_s16 b = npyv_setall_s16(-(c));
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_s16 shl = __riscv_vsll_vv_i16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vint32m2_t a_ext = __riscv_vsext_vf2_i32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_s16 shr = __riscv_vnclip_wv_i16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_i16m1(shr, shl, positive_mask, 8);
}

NPY_FINLINE npyv_u32 npyv_shr_u32(npyv_u32 a, int32_t c)
{
    npyv_s32 b = npyv_setall_s32(-(c));
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_u32 shl = __riscv_vsll_vv_u32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vuint64m2_t a_ext = __riscv_vzext_vf2_u64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_u32 shr = __riscv_vnclipu_wv_u32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_u32m1(shr, shl, positive_mask, 4);
}

NPY_FINLINE npyv_s32 npyv_shr_s32(npyv_s32 a, int32_t c)
{
    npyv_s32 b = npyv_setall_s32(-(c));
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_s32 shl = __riscv_vsll_vv_i32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vint64m2_t a_ext = __riscv_vsext_vf2_i64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_s32 shr = __riscv_vnclip_wv_i32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_i32m1(shr, shl, positive_mask, 4);
}

NPY_FINLINE npyv_u64 npyv_shr_u64(npyv_u64 a, int64_t c)
{
    npyv_s64 b = npyv_setall_s64(-(c));
    // implementation only works within defined range 'b' in [0, 63]
    vbool64_t positive_mask = __riscv_vmsgt_vx_i64m1_b64(b, 0, 2);
    npyv_u64 shl = __riscv_vsll_vv_u64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b), 2);
    npyv_s64 b_neg = __riscv_vneg_v_i64m1(b, 2);
    npyv_u64 shr = __riscv_vsrl_vv_u64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b_neg), 2);
    return __riscv_vmerge_vvm_u64m1(shr, shl, positive_mask, 2);
}

NPY_FINLINE npyv_s64 npyv_shr_s64(npyv_s64 a, int64_t c)
{
    npyv_s64 b = npyv_setall_s64(-(c));
    // implementation only works within defined range 'b' in [0, 63]
    vbool64_t positive_mask = __riscv_vmsgt_vx_i64m1_b64(b, 0, 2);
    npyv_s64 shl = __riscv_vsll_vv_i64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b), 2);
    npyv_s64 b_neg = __riscv_vneg_v_i64m1(b, 2);
    npyv_s64 shr = __riscv_vsra_vv_i64m1(a, __riscv_vreinterpret_v_i64m1_u64m1(b_neg), 2);
    return __riscv_vmerge_vvm_i64m1(shr, shl, positive_mask, 2);
}

// right by an immediate constant
NPY_FINLINE npyv_u16 npyv_shri_u16(npyv_u16 a, const int b) {
    const int b_half = b >> 1;
    npyv_u16 srl1 = __riscv_vsrl_vx_u16m1(a, b_half, 8);
    return __riscv_vsrl_vx_u16m1(srl1, b_half + (b & 0x1), 8);
}

NPY_FINLINE npyv_s16 npyv_shri_s16(npyv_s16 a, const int b) {
    const int imm = b - (b >> 4);
    return __riscv_vsra_vx_i16m1(a, imm, 8);
}

NPY_FINLINE npyv_u32 npyv_shri_u32(npyv_u32 a, const int b) {
    const int b_half = b >> 1;
    npyv_u32 srl1 = __riscv_vsrl_vx_u32m1(a, b_half, 4);
    return __riscv_vsrl_vx_u32m1(srl1, b_half + (b & 0x1), 4);
}

NPY_FINLINE npyv_s32 npyv_shri_s32(npyv_s32 a, const int b) {
    const int imm = b - (b >> 5);
    return __riscv_vsra_vx_i32m1(a, imm, 4);
}

NPY_FINLINE npyv_u64 npyv_shri_u64(npyv_u64 a, const int b) {
    const int b_half = b >> 1;
    npyv_u64 srl1 = __riscv_vsrl_vx_u64m1(a, b_half, 2);
    return __riscv_vsrl_vx_u64m1(srl1, b_half + (b & 0x1), 2);
}

NPY_FINLINE npyv_s64 npyv_shri_s64(npyv_s64 a, const int b) {
    const int imm = b - (b >> 6);
    return __riscv_vsra_vx_i64m1(a, imm, 2);
}

/***************************
 * Logical
 ***************************/
// AND
NPY_FINLINE npyv_u8 npyv_and_u8(npyv_u8 a, npyv_u8 b) 
{
    return __riscv_vand_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_and_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vand_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_and_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vand_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_and_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vand_vv_i16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_and_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vand_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_s32 npyv_and_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vand_vv_i32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_and_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vand_vv_u64m1(a, b, 2);
}

NPY_FINLINE npyv_s64 npyv_and_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vand_vv_i64m1(a, b, 2);
}

NPY_FINLINE npyv_f32 npyv_and_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vand_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            4
        )
    );
}

NPY_FINLINE npyv_f64 npyv_and_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vand_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            2
        )
    );
}

NPY_FINLINE npyv_u8 npyv_and_b8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vand_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_and_b16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vand_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_and_b32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vand_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_and_b64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vand_vv_u64m1(a, b, 2);
}

// OR
NPY_FINLINE npyv_u8 npyv_or_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vor_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_or_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vor_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_or_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vor_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_or_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vor_vv_i16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_or_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vor_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_s32 npyv_or_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vor_vv_i32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_or_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vor_vv_u64m1(a, b, 2);
}

NPY_FINLINE npyv_s64 npyv_or_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vor_vv_i64m1(a, b, 2);
}

NPY_FINLINE npyv_f32 npyv_or_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vor_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            4
        )
    );
}

NPY_FINLINE npyv_f64 npyv_or_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vor_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            2
        )
    );
}

NPY_FINLINE npyv_u8 npyv_or_b8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vor_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_or_b16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vor_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_or_b32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vor_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_or_b64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vor_vv_u64m1(a, b, 2);
}

// XOR
NPY_FINLINE npyv_u8 npyv_xor_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vxor_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_xor_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vxor_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_xor_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vxor_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_xor_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vxor_vv_i16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_xor_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vxor_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_s32 npyv_xor_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vxor_vv_i32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_xor_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vxor_vv_u64m1(a, b, 2);
}

NPY_FINLINE npyv_s64 npyv_xor_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vxor_vv_i64m1(a, b, 2);
}

NPY_FINLINE npyv_f32 npyv_xor_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vxor_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            4
        )
    );
}

NPY_FINLINE npyv_f64 npyv_xor_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vxor_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            2
        )
    );
}

NPY_FINLINE npyv_u8 npyv_xor_b8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vxor_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_xor_b16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vxor_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_u32 npyv_xor_b32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vxor_vv_u32m1(a, b, 4);
}

NPY_FINLINE npyv_u64 npyv_xor_b64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vxor_vv_u64m1(a, b, 2);
}

// NOT
NPY_FINLINE npyv_u8 npyv_not_u8(npyv_u8 a)
{
    return __riscv_vnot_v_u8m1(a, 16);
}

NPY_FINLINE npyv_s8 npyv_not_s8(npyv_s8 a)
{
    return __riscv_vnot_v_i8m1(a, 16);
}

NPY_FINLINE npyv_u16 npyv_not_u16(npyv_u16 a)
{
    return __riscv_vnot_v_u16m1(a, 8);
}

NPY_FINLINE npyv_s16 npyv_not_s16(npyv_s16 a)
{
    return __riscv_vnot_v_i16m1(a, 8);
}

NPY_FINLINE npyv_u32 npyv_not_u32(npyv_u32 a)
{
    return __riscv_vnot_v_u32m1(a, 4);
}

NPY_FINLINE npyv_s32 npyv_not_s32(npyv_s32 a)
{
    return __riscv_vnot_v_i32m1(a, 4);
}

NPY_FINLINE npyv_u64 npyv_not_u64(npyv_u64 a)
{
    return __riscv_vnot_v_u64m1(a, 2);
}

NPY_FINLINE npyv_s64 npyv_not_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_u64m1_i64m1(
        __riscv_vnot_v_u64m1(
            __riscv_vreinterpret_v_i64m1_u64m1(a),
            2
        )
    );
}

NPY_FINLINE npyv_f32 npyv_not_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vnot_v_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            4
        )
    );
}

NPY_FINLINE npyv_f64 npyv_not_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vnot_v_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            2
        )
    );
}

NPY_FINLINE npyv_u8 npyv_not_b8(npyv_u8 a)
{
    return __riscv_vnot_v_u8m1(a, 16);
}

NPY_FINLINE npyv_u16 npyv_not_b16(npyv_u16 a)
{
    return __riscv_vnot_v_u16m1(a, 8);
}

NPY_FINLINE npyv_u32 npyv_not_b32(npyv_u32 a)
{
    return __riscv_vnot_v_u32m1(a, 4);
}

#define npyv_not_b64  npyv_not_u64

// ANDC, ORC and XNOR
NPY_FINLINE npyv_u8 npyv_andc_u8(npyv_u8 a, npyv_u8 b) {
  return __riscv_vand_vv_u8m1(a, __riscv_vnot_v_u8m1(b, 16), 16);
}

NPY_FINLINE npyv_u8 npyv_andc_b8(npyv_u8 a, npyv_u8 b) {
  return __riscv_vand_vv_u8m1(a, __riscv_vnot_v_u8m1(b, 16), 16);
}

NPY_FINLINE npyv_u8 npyv_orc_b8(npyv_u8 a, npyv_u8 b) {
  return __riscv_vor_vv_u8m1(a, __riscv_vnot_v_u8m1(b, 16), 16);
}

NPY_FINLINE npyv_u8 npyv_xnor_b8(npyv_u8 a, npyv_u8 b) {
  vbool8_t cmp_res = __riscv_vmseq_vv_u8m1_b8(a, b, 16);
  return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

/***************************
 * Comparison
 ***************************/
// equal
NPY_FINLINE npyv_u8 npyv_cmpeq_u8(npyv_u8 a, npyv_u8 b) {
  vbool8_t cmp_res = __riscv_vmseq_vv_u8m1_b8(a, b, 16);
  return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u16 npyv_cmpeq_u16(npyv_u16 a, npyv_u16 b) {
  vbool16_t cmp_res = __riscv_vmseq_vv_u16m1_b16(a, b, 8);
  return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u32 npyv_cmpeq_u32(npyv_u32 a, npyv_u32 b) {
  vbool32_t cmp_res = __riscv_vmseq_vv_u32m1_b32(a, b, 4);
  return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u8 npyv_cmpeq_s8(npyv_s8 a, npyv_s8 b) {
  vbool8_t cmp_res = __riscv_vmseq_vv_i8m1_b8(a, b, 16);
  return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u16 npyv_cmpeq_s16(npyv_s16 a, npyv_s16 b) {
  vbool16_t cmp_res = __riscv_vmseq_vv_i16m1_b16(a, b, 8);
  return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u32 npyv_cmpeq_s32(npyv_s32 a, npyv_s32 b) {
  vbool32_t cmp_res = __riscv_vmseq_vv_i32m1_b32(a, b, 4);
  return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u32 npyv_cmpeq_f32(vfloat32m1_t a, vfloat32m1_t b) {
  vbool32_t cmp_res = __riscv_vmfeq_vv_f32m1_b32(a, b, 4);
  return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u64 npyv_cmpeq_f64(vfloat64m1_t a, vfloat64m1_t b) {
  vbool64_t cmp_res = __riscv_vmfeq_vv_f64m1_b64(a, b, 2);
  return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpeq_u64(npyv_u64 a, npyv_u64 b) {
  vbool64_t cmp_res = __riscv_vmseq_vv_u64m1_b64(a, b, 2);
  return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpeq_s64(npyv_s64 a, npyv_s64 b) {
  vbool64_t cmp_res = __riscv_vmseq_vv_i64m1_b64(a, b, 2);
  return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

// not Equal
NPY_FINLINE npyv_u8 npyv_cmpneq_u8(npyv_u8 a, npyv_u8 b) {
    vbool8_t cmp_res = __riscv_vmseq_vv_u8m1_b8(a, b, 16);
    return __riscv_vnot_v_u8m1(__riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16), 16);
}

NPY_FINLINE npyv_u8 npyv_cmpneq_s8(npyv_s8 a, npyv_s8 b) {
    vbool8_t cmp_res = __riscv_vmseq_vv_i8m1_b8(a, b, 16);
    return __riscv_vnot_v_u8m1(__riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16), 16);
}

NPY_FINLINE npyv_u16 npyv_cmpneq_u16(npyv_u16 a, npyv_u16 b) {
    vbool16_t cmp_res = __riscv_vmseq_vv_u16m1_b16(a, b, 8);
    return __riscv_vnot_v_u16m1(__riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8), 8);
}

NPY_FINLINE npyv_u16 npyv_cmpneq_s16(npyv_s16 a, npyv_s16 b) {
    vbool16_t cmp_res = __riscv_vmseq_vv_i16m1_b16(a, b, 8);
    return __riscv_vnot_v_u16m1(__riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8), 8);
}

NPY_FINLINE npyv_u32 npyv_cmpneq_u32(npyv_u32 a, npyv_u32 b) {
    vbool32_t cmp_res = __riscv_vmseq_vv_u32m1_b32(a, b, 4);
    return __riscv_vnot_v_u32m1(__riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4), 4);
}

NPY_FINLINE npyv_u32 npyv_cmpneq_s32(npyv_s32 a, npyv_s32 b) {
    vbool32_t cmp_res = __riscv_vmseq_vv_i32m1_b32(a, b, 4);
    return __riscv_vnot_v_u32m1(__riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4), 4);
}

#define npyv_cmpneq_u64(A, B) npyv_not_u64(npyv_cmpeq_u64(A, B))
#define npyv_cmpneq_s64(A, B) npyv_not_u64(npyv_cmpeq_s64(A, B))

NPY_FINLINE npyv_u32 npyv_cmpneq_f32(vfloat32m1_t a, vfloat32m1_t b) {
    vbool32_t cmp_res = __riscv_vmfeq_vv_f32m1_b32(a, b, 4);
    return  __riscv_vnot_v_u32m1(__riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4), 4);
}

#define npyv_cmpneq_f64(A, B) npyv_not_u64(__riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), __riscv_vmfeq_vv_f64m1_b64(A, B, 2), 2))

// greater than
NPY_FINLINE npyv_u8 npyv_cmpgt_u8(npyv_u8 a, npyv_u8 b) {
    vbool8_t cmp_res = __riscv_vmsgtu_vv_u8m1_b8(a, b, 16);
    return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u8 npyv_cmpgt_s8(npyv_s8 a, npyv_s8 b) {
    vbool8_t cmp_res = __riscv_vmsgt_vv_i8m1_b8(a, b, 16);
    return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u16 npyv_cmpgt_u16(npyv_u16 a, npyv_u16 b) {
    vbool16_t cmp_res = __riscv_vmsgtu_vv_u16m1_b16(a, b, 8);
    return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u16 npyv_cmpgt_s16(npyv_s16 a, npyv_s16 b) {
    vbool16_t cmp_res = __riscv_vmsgt_vv_i16m1_b16(a, b, 8);
    return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u32 npyv_cmpgt_u32(npyv_u32 a, npyv_u32 b) {
    vbool32_t cmp_res = __riscv_vmsgtu_vv_u32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u32 npyv_cmpgt_s32(npyv_s32 a, npyv_s32 b) {
    vbool32_t cmp_res = __riscv_vmsgt_vv_i32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u32 npyv_cmpgt_f32(vfloat32m1_t a, vfloat32m1_t b) {
    vbool32_t cmp_res = __riscv_vmfgt_vv_f32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u64 npyv_cmpgt_f64(vfloat64m1_t a, vfloat64m1_t b) {
    vbool64_t cmp_res = __riscv_vmfgt_vv_f64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpgt_u64(npyv_u64 a, npyv_u64 b) {
    vbool64_t cmp_res = __riscv_vmsgtu_vv_u64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpgt_s64(npyv_s64 a, npyv_s64 b) {
    vbool64_t cmp_res = __riscv_vmsgt_vv_i64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

// greater than or equal
NPY_FINLINE npyv_u8 npyv_cmpge_u8(npyv_u8 a, npyv_u8 b) {
    vbool8_t cmp_res = __riscv_vmsgeu_vv_u8m1_b8(a, b, 16);
    return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u8 npyv_cmpge_s8(npyv_s8 a, npyv_s8 b) {
    vbool8_t cmp_res = __riscv_vmsge_vv_i8m1_b8(a, b, 16);
    return __riscv_vmerge_vvm_u8m1(__riscv_vmv_v_x_u8m1(0x0, 16), __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), cmp_res, 16);
}

NPY_FINLINE npyv_u16 npyv_cmpge_u16(npyv_u16 a, npyv_u16 b) {
    vbool16_t cmp_res = __riscv_vmsgeu_vv_u16m1_b16(a, b, 8);
    return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u16 npyv_cmpge_s16(npyv_s16 a, npyv_s16 b) {
    vbool16_t cmp_res = __riscv_vmsge_vv_i16m1_b16(a, b, 8);
    return __riscv_vmerge_vvm_u16m1(__riscv_vmv_v_x_u16m1(0x0, 8), __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), cmp_res, 8);
}

NPY_FINLINE npyv_u32 npyv_cmpge_u32(npyv_u32 a, npyv_u32 b) {
    vbool32_t cmp_res = __riscv_vmsgeu_vv_u32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u32 npyv_cmpge_s32(npyv_s32 a, npyv_s32 b) {
    vbool32_t cmp_res = __riscv_vmsge_vv_i32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u32 npyv_cmpge_f32(vfloat32m1_t a, vfloat32m1_t b) {
    vbool32_t cmp_res = __riscv_vmfge_vv_f32m1_b32(a, b, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_u64 npyv_cmpge_f64(vfloat64m1_t a, vfloat64m1_t b) {
    vbool64_t cmp_res = __riscv_vmfge_vv_f64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpge_u64(npyv_u64 a, npyv_u64 b) {
    vbool64_t cmp_res = __riscv_vmsgeu_vv_u64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

NPY_FINLINE npyv_u64 npyv_cmpge_s64(npyv_s64 a, npyv_s64 b) {
    vbool64_t cmp_res = __riscv_vmsge_vv_i64m1_b64(a, b, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

// less than
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)
#define npyv_cmplt_f32(A, B) npyv_cmpgt_f32(B, A)
#define npyv_cmplt_f64(A, B) npyv_cmpgt_f64(B, A)

// less than or equal
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)
#define npyv_cmple_f32(A, B) npyv_cmpge_f32(B, A)
#define npyv_cmple_f64(A, B) npyv_cmpge_f64(B, A)

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{
    vbool32_t cmp_res = __riscv_vmfeq_vv_f32m1_b32(a, a, 4);
    return __riscv_vmerge_vvm_u32m1(__riscv_vmv_v_x_u32m1(0x0, 4), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), cmp_res, 4);
}

NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{
    vbool64_t cmp_res = __riscv_vmfeq_vv_f64m1_b64(a, a, 2);
    return __riscv_vmerge_vvm_u64m1(__riscv_vmv_v_x_u64m1(0x0, 2), __riscv_vmv_v_x_u64m1(UINT64_MAX, 2), cmp_res, 2);
}

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
NPY_FINLINE bool npyv_any_b8(npyv_u8 a)
{
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(a, __riscv_vmv_v_x_u8m1(0, 16), 16)) != 0;
}
NPY_FINLINE bool npyv_all_b8(npyv_u8 a)
{
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(a, __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), 16)) != 0;
}

NPY_FINLINE bool npyv_any_b16(npyv_b16 a)
{
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(a, __riscv_vmv_v_x_u16m1(0, 8), 8)) != 0;
}

NPY_FINLINE bool npyv_all_b16(npyv_b16 a)
{
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(a, __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), 8)) != 0;
}

NPY_FINLINE bool npyv_any_b32(npyv_b32 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(a, __riscv_vmv_v_x_u32m1(0, 4), 4)) != 0;
}

NPY_FINLINE bool npyv_all_b32(npyv_b32 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(a, __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), 4)) != 0;
}

NPY_FINLINE bool npyv_any_u8(npyv_u8 a)
{
    return npyv_any_b8(npyv_reinterpret_u8_u8(a));
}

NPY_FINLINE bool npyv_all_u8(npyv_u8 a)
{
    return npyv_all_b8(npyv_reinterpret_u8_u8(a));
}

NPY_FINLINE bool npyv_any_s8(npyv_s8 a)
{
    return npyv_any_b8(npyv_reinterpret_u8_s8(a));
}

NPY_FINLINE bool npyv_all_s8(npyv_s8 a)
{
    return npyv_all_b8(npyv_reinterpret_u8_s8(a));
}

NPY_FINLINE bool npyv_any_u16(npyv_u16 a)
{
    return npyv_any_b16(npyv_reinterpret_u16_u16(a));
}

NPY_FINLINE bool npyv_all_u16(npyv_u16 a)
{
    return npyv_all_b16(npyv_reinterpret_u16_u16(a));
}

NPY_FINLINE bool npyv_any_s16(npyv_s16 a)
{
    return npyv_any_b16(npyv_reinterpret_u16_s16(a));
}

NPY_FINLINE bool npyv_all_s16(npyv_s16 a)
{
    return npyv_all_b16(npyv_reinterpret_u16_s16(a));
}

NPY_FINLINE bool npyv_any_u32(npyv_u32 a)
{
    return npyv_any_b32(npyv_reinterpret_u32_u32(a));
}

NPY_FINLINE bool npyv_all_u32(npyv_u32 a)
{
    return npyv_all_b32(npyv_reinterpret_u32_u32(a));
}

NPY_FINLINE bool npyv_any_s32(npyv_s32 a)
{
    return npyv_any_b32(npyv_reinterpret_u32_s32(a));
}

NPY_FINLINE bool npyv_all_s32(npyv_s32 a)
{
    return npyv_all_b32(npyv_reinterpret_u32_s32(a));
}

NPY_FINLINE bool npyv_any_b64(npyv_b64 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(__riscv_vreinterpret_v_u64m1_u32m1(a), __riscv_vmv_v_x_u32m1(0, 4), 4)) != 0;
}

NPY_FINLINE bool npyv_all_b64(npyv_b64 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(__riscv_vreinterpret_v_u64m1_u32m1(a), __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), 4)) != 0;
}

#define npyv_any_u64 npyv_any_b64

NPY_FINLINE npyv_u32 vrev64q_u32(npyv_u32 a) {
    npyv_u32 vid = __riscv_vid_v_u32m1(2);
    npyv_u32 vid_slideup = __riscv_vslideup_vx_u32m1(vid, vid, 2, 4);
    npyv_u32 sub = __riscv_vslideup_vx_u32m1(__riscv_vmv_v_x_u32m1(1, 4), __riscv_vmv_v_x_u32m1(1 + 2, 4), 2, 4);
    npyv_u32 idxs = __riscv_vsub_vv_u32m1(sub, vid_slideup, 4);
    return __riscv_vrgather_vv_u32m1(a, idxs, 4);
}

NPY_FINLINE bool npyv_all_u64(npyv_u64 a)
{
    npyv_u32 a32 = __riscv_vreinterpret_v_u64m1_u32m1(a);
    a32 = __riscv_vor_vv_u32m1(a32, vrev64q_u32(a32), 4);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(a32, __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), 4))   != 0;
}

NPY_FINLINE bool npyv_any_s64(npyv_s64 a)
{
    return npyv_any_u64(__riscv_vreinterpret_v_i64m1_u64m1(a));
}

NPY_FINLINE bool npyv_all_s64(npyv_s64 a)
{
    return npyv_all_u64(__riscv_vreinterpret_v_i64m1_u64m1(a));
}

NPY_FINLINE bool npyv_any_f32(npyv_f32 a)
{
    return !npyv_all_b32(npyv_cmpeq_f32(a, npyv_zero_f32()));
}

NPY_FINLINE bool npyv_all_f32(npyv_f32 a)
{
    return !npyv_any_b32(npyv_cmpeq_f32(a, npyv_zero_f32()));
}

NPY_FINLINE bool npyv_any_f64(npyv_f64 a)
{
    return !npyv_all_b64(npyv_cmpeq_f64(a, npyv_zero_f64()));
}

NPY_FINLINE bool npyv_all_f64(npyv_f64 a)
{
    return !npyv_any_b64(npyv_cmpeq_f64(a, npyv_zero_f64()));
}

#endif // _NPY_SIMD_RVV_OPERATORS_H
