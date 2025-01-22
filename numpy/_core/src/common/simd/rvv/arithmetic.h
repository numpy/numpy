#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_ARITHMETIC_H
#define _NPY_SIMD_RVV_ARITHMETIC_H

#include <riscv_vector.h>

/***************************
 * Addition
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_add_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vadd_vv_u8m1(a, b, 16); }
NPY_FINLINE npyv_s8 npyv_add_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vadd_vv_i8m1(a, b, 16); }
NPY_FINLINE npyv_u16 npyv_add_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vadd_vv_u16m1(a, b, 8); }
NPY_FINLINE npyv_s16 npyv_add_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vadd_vv_i16m1(a, b, 8); }
NPY_FINLINE npyv_u32 npyv_add_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vadd_vv_u32m1(a, b, 4); }
NPY_FINLINE npyv_s32 npyv_add_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vadd_vv_i32m1(a, b, 4); }
NPY_FINLINE npyv_u64 npyv_add_u64(npyv_u64 a, npyv_u64 b) { return __riscv_vadd_vv_u64m1(a, b, 2); }
NPY_FINLINE npyv_s64 npyv_add_s64(npyv_s64 a, npyv_s64 b) { return __riscv_vadd_vv_i64m1(a, b, 2); }
NPY_FINLINE npyv_f32 npyv_add_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfadd_vv_f32m1(a, b, 4); }
NPY_FINLINE npyv_f64 npyv_add_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfadd_vv_f64m1(a, b, 2); }

// saturated
NPY_FINLINE npyv_u8 npyv_adds_u8(npyv_u8 a, npyv_u8 b) {
    return __riscv_vsaddu_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_adds_s8(npyv_s8 a, npyv_s8 b) {
    return __riscv_vsadd_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_adds_u16(npyv_u16 a, npyv_u16 b) {
    return __riscv_vsaddu_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_adds_s16(npyv_s16 a, npyv_s16 b) {
    return __riscv_vsadd_vv_i16m1(a, b, 8);
}

/***************************
 * Subtraction
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_sub_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vsub_vv_u8m1(a, b, 16); }
NPY_FINLINE npyv_s8 npyv_sub_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vsub_vv_i8m1(a, b, 16); }
NPY_FINLINE npyv_u16 npyv_sub_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vsub_vv_u16m1(a, b, 8); }
NPY_FINLINE npyv_s16 npyv_sub_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vsub_vv_i16m1(a, b, 8); }
NPY_FINLINE npyv_u32 npyv_sub_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vsub_vv_u32m1(a, b, 4); }
NPY_FINLINE npyv_s32 npyv_sub_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vsub_vv_i32m1(a, b, 4); }
NPY_FINLINE npyv_u64 npyv_sub_u64(npyv_u64 a, npyv_u64 b) { return __riscv_vsub_vv_u64m1(a, b, 2); }
NPY_FINLINE npyv_s64 npyv_sub_s64(npyv_s64 a, npyv_s64 b) { return __riscv_vsub_vv_i64m1(a, b, 2); }
NPY_FINLINE npyv_f32 npyv_sub_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfsub_vv_f32m1(a, b, 4); }
NPY_FINLINE npyv_f64 npyv_sub_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfsub_vv_f64m1(a, b, 2); }

// saturated
NPY_FINLINE npyv_u8 npyv_subs_u8(npyv_u8 a, npyv_u8 b) {
    return __riscv_vssubu_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_subs_s8(npyv_s8 a, npyv_s8 b) {
    return __riscv_vssub_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_subs_u16(npyv_u16 a, npyv_u16 b) {
    return __riscv_vssubu_vv_u16m1(a, b, 8);
}

NPY_FINLINE npyv_s16 npyv_subs_s16(npyv_s16 a, npyv_s16 b) {
    return __riscv_vssub_vv_i16m1(a, b, 8);
}

/***************************
 * Multiplication
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_mul_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vmul_vv_u8m1(a, b, 16); }
NPY_FINLINE npyv_s8 npyv_mul_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vmul_vv_i8m1(a, b, 16); }
NPY_FINLINE npyv_u16 npyv_mul_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vmul_vv_u16m1(a, b, 8); }
NPY_FINLINE npyv_s16 npyv_mul_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vmul_vv_i16m1(a, b, 8); }
NPY_FINLINE npyv_u32 npyv_mul_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vmul_vv_u32m1(a, b, 4); }
NPY_FINLINE npyv_s32 npyv_mul_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vmul_vv_i32m1(a, b, 4); }
NPY_FINLINE npyv_f32 npyv_mul_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfmul_vv_f32m1(a, b, 4); }
NPY_FINLINE npyv_f64 npyv_mul_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfmul_vv_f64m1(a, b, 2); }

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_s16 vmull_high_s8(npyv_s8 a, npyv_s8 b) {
    vint8mf2_t a_high = __riscv_vlmul_trunc_v_i8m1_i8mf2(__riscv_vslidedown_vx_i8m1(a, 8, 16));
    vint8mf2_t b_high = __riscv_vlmul_trunc_v_i8m1_i8mf2(__riscv_vslidedown_vx_i8m1(b, 8, 16));
    return __riscv_vwmul_vv_i16m1(a_high, b_high, 8);
}

NPY_FINLINE npyv_s32 vmull_high_s16(npyv_s16 a, npyv_s16 b) {
    vint16mf2_t a_high = __riscv_vlmul_trunc_v_i16m1_i16mf2(__riscv_vslidedown_vx_i16m1(a, 4, 8));
    vint16mf2_t b_high = __riscv_vlmul_trunc_v_i16m1_i16mf2(__riscv_vslidedown_vx_i16m1(b, 4, 8));
    return __riscv_vwmul_vv_i32m1(a_high, b_high, 4);
}

NPY_FINLINE npyv_s64 vmull_high_s32(npyv_s32 a, npyv_s32 b) {
    vint32mf2_t a_high = __riscv_vlmul_trunc_v_i32m1_i32mf2(__riscv_vslidedown_vx_i32m1(a, 2, 4));
    vint32mf2_t b_high = __riscv_vlmul_trunc_v_i32m1_i32mf2(__riscv_vslidedown_vx_i32m1(b, 2, 4));
    return __riscv_vwmul_vv_i64m1(a_high, b_high, 2);
}

NPY_FINLINE npyv_u16 vmull_high_u8(npyv_u8 a, npyv_u8 b) {
    vuint8mf2_t a_high = __riscv_vlmul_trunc_v_u8m1_u8mf2(__riscv_vslidedown_vx_u8m1(a, 8, 16));
    vuint8mf2_t b_high = __riscv_vlmul_trunc_v_u8m1_u8mf2(__riscv_vslidedown_vx_u8m1(b, 8, 16));
    return __riscv_vwmulu_vv_u16m1(a_high, b_high, 8);
}

NPY_FINLINE npyv_u32 vmull_high_u16(npyv_u16 a, npyv_u16 b) {
    vuint16mf2_t a_high = __riscv_vlmul_trunc_v_u16m1_u16mf2(__riscv_vslidedown_vx_u16m1(a, 4, 8));
    vuint16mf2_t b_high = __riscv_vlmul_trunc_v_u16m1_u16mf2(__riscv_vslidedown_vx_u16m1(b, 4, 8));
    return __riscv_vwmulu_vv_u32m1(a_high, b_high, 4);
}

NPY_FINLINE npyv_u64 vmull_high_u32(npyv_u32 a, npyv_u32 b) {
    vuint32mf2_t a_high = __riscv_vlmul_trunc_v_u32m1_u32mf2(__riscv_vslidedown_vx_u32m1(a, 2, 4));
    vuint32mf2_t b_high = __riscv_vlmul_trunc_v_u32m1_u32mf2(__riscv_vslidedown_vx_u32m1(b, 2, 4));
    return __riscv_vwmulu_vv_u64m1(a_high, b_high, 2);
}

NPY_FINLINE npyv_u8 vshlq_u8(npyv_u8 a, npyv_s8 b) {
    // implementation only works within defined range 'b' in [0, 7]
    vbool8_t positive_mask = __riscv_vmsgt_vx_i8m1_b8(b, 0, 16);
    npyv_u8 shl = __riscv_vsll_vv_u8m1(a, __riscv_vreinterpret_v_i8m1_u8m1(b), 16);
    vuint16m2_t a_ext = __riscv_vzext_vf2_u16m2(a, 16);
    npyv_s8 b_neg = __riscv_vneg_v_i8m1(b, 16);
    npyv_u8 shr = __riscv_vnclipu_wv_u8m1(a_ext, __riscv_vreinterpret_v_i8m1_u8m1(b_neg), __RISCV_VXRM_RDN, 16);
    return __riscv_vmerge_vvm_u8m1(shr, shl, positive_mask, 16);
}

NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const npyv_u8 mulc_lo = divisor.val[0];
    // high part of unsigned multiplication
    npyv_u16 mull_lo  = __riscv_vlmul_trunc_v_u16m2_u16m1(__riscv_vwmulu_vv_u16m2(a, mulc_lo, 8));
    npyv_u16 mull_hi  = vmull_high_u8(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_u8 mulhi    = vuzp2q_u8(__riscv_vreinterpret_v_u16m1_u8m1(mull_lo), __riscv_vreinterpret_v_u16m1_u8m1(mull_hi));

    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u8    q        = __riscv_vsub_vv_u8m1(a, mulhi, 16);
               q        = vshlq_u8(q, __riscv_vreinterpret_v_u8m1_i8m1(divisor.val[1]));
               q        = __riscv_vadd_vv_u8m1(mulhi, q, 16);
               q        = vshlq_u8(q, __riscv_vreinterpret_v_u8m1_i8m1(divisor.val[2]));
    return q;
}

NPY_FINLINE npyv_s8 vshlq_s8(npyv_s8 a, npyv_s8 b) {
    // implementation only works within defined range 'b' in [0, 7]
    vbool8_t positive_mask = __riscv_vmsgt_vx_i8m1_b8(b, 0, 16);
    npyv_s8 shl = __riscv_vsll_vv_i8m1(a, __riscv_vreinterpret_v_i8m1_u8m1(b), 16);
    vint16m2_t a_ext = __riscv_vsext_vf2_i16m2(a, 16);
    npyv_s8 b_neg = __riscv_vneg_v_i8m1(b, 16);
    npyv_s8 shr = __riscv_vnclip_wv_i8m1(a_ext, __riscv_vreinterpret_v_i8m1_u8m1(b_neg), __RISCV_VXRM_RDN, 16);
    return __riscv_vmerge_vvm_i8m1(shr, shl, positive_mask, 16);
}

NPY_FINLINE npyv_s8 vshrq_n_s8(npyv_s8 a, const int b) {
    const int imm = b - (b >> 3);
    return __riscv_vsra_vx_i8m1(a, imm, 16);
}

// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const npyv_s8 mulc_lo = divisor.val[0];
    // high part of signed multiplication
    npyv_s16 mull_lo  = __riscv_vlmul_trunc_v_i16m2_i16m1(__riscv_vwmul_vv_i16m2(a, mulc_lo, 8));
    npyv_s16 mull_hi  = vmull_high_s8(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_s8 mulhi    = vuzp2q_s8(__riscv_vreinterpret_v_i16m1_i8m1(mull_lo), __riscv_vreinterpret_v_i16m1_i8m1(mull_hi));
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    npyv_s8   q        = vshlq_s8(__riscv_vadd_vv_i8m1(a, mulhi, 16), divisor.val[1]);
              q        = __riscv_vsub_vv_i8m1(q, vshrq_n_s8(a, 7), 16);
              q        = __riscv_vsub_vv_i8m1(__riscv_vxor_vv_i8m1(q, divisor.val[2], 16), divisor.val[2], 16);
    return q;
}

NPY_FINLINE npyv_u16 vshlq_u16(npyv_u16 a, npyv_s16 b) {
    // implementation only works within defined range 'b' in [0, 15]
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_u16 shl = __riscv_vsll_vv_u16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vuint32m2_t a_ext = __riscv_vzext_vf2_u32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_u16 shr = __riscv_vnclipu_wv_u16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_u16m1(shr, shl, positive_mask, 8);
}

// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    const npyv_u16 mulc_lo = divisor.val[0];
    // high part of unsigned multiplication
    npyv_u32  mull_lo  = __riscv_vlmul_trunc_v_u32m2_u32m1(__riscv_vwmulu_vv_u32m2(a, mulc_lo, 4));
    npyv_u32  mull_hi  = vmull_high_u16(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_u16 mulhi    = vuzp2q_u16(__riscv_vreinterpret_v_u32m1_u16m1(mull_lo), __riscv_vreinterpret_v_u32m1_u16m1(mull_hi));

    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u16   q        = __riscv_vsub_vv_u16m1(a, mulhi, 8);
               q        = vshlq_u16(q, __riscv_vreinterpret_v_u16m1_i16m1(divisor.val[1]));
               q        = __riscv_vadd_vv_u16m1(mulhi, q, 8);
               q        = vshlq_u16(q, __riscv_vreinterpret_v_u16m1_i16m1(divisor.val[2]));
    return q;
}

NPY_FINLINE npyv_s16 vshrq_n_s16(npyv_s16 a, const int b) {
    const int imm = b - (b >> 4);
    return __riscv_vsra_vx_i16m1(a, imm, 8);
}

NPY_FINLINE npyv_s16 vshlq_s16(npyv_s16 a, npyv_s16 b) {
    // implementation only works within defined range 'b' in [0, 15]
    vbool16_t positive_mask = __riscv_vmsgt_vx_i16m1_b16(b, 0, 8);
    npyv_s16 shl = __riscv_vsll_vv_i16m1(a, __riscv_vreinterpret_v_i16m1_u16m1(b), 8);
    vint32m2_t a_ext = __riscv_vsext_vf2_i32m2(a, 8);
    npyv_s16 b_neg = __riscv_vneg_v_i16m1(b, 8);
    npyv_s16 shr = __riscv_vnclip_wv_i16m1(a_ext, __riscv_vreinterpret_v_i16m1_u16m1(b_neg), __RISCV_VXRM_RDN, 8);
    return __riscv_vmerge_vvm_i16m1(shr, shl, positive_mask, 8);
}

// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    const npyv_s16 mulc_lo = divisor.val[0];
    // high part of signed multiplication
    npyv_s32 mull_lo  = __riscv_vlmul_trunc_v_i32m2_i32m1(__riscv_vwmul_vv_i32m2(a, mulc_lo, 4));
    npyv_s32 mull_hi  = vmull_high_s16(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_s16 mulhi    = vuzp2q_s16(__riscv_vreinterpret_v_i32m1_i16m1(mull_lo), __riscv_vreinterpret_v_i32m1_i16m1(mull_hi));
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    npyv_s16  q        = vshlq_s16(__riscv_vadd_vv_i16m1(a, mulhi, 8), divisor.val[1]);
              q        = __riscv_vsub_vv_i16m1(q, vshrq_n_s16(a, 15), 8);
              q        = __riscv_vsub_vv_i16m1(__riscv_vxor_vv_i16m1(q, divisor.val[2], 8), divisor.val[2], 8);
    return q;
}

NPY_FINLINE npyv_u32 vshlq_u32(npyv_u32 a, npyv_s32 b) {
    // implementation only works within defined range 'b' in [0, 31]
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_u32 shl = __riscv_vsll_vv_u32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vuint64m2_t a_ext = __riscv_vzext_vf2_u64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_u32 shr = __riscv_vnclipu_wv_u32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_u32m1(shr, shl, positive_mask, 4);
}

NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    const npyv_u32 mulc_lo = divisor.val[0];
    // high part of unsigned multiplication
    npyv_u64 mull_lo  = __riscv_vlmul_trunc_v_u64m2_u64m1(__riscv_vwmulu_vv_u64m2(a, mulc_lo, 2));
    npyv_u64 mull_hi  = vmull_high_u32(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_u32 mulhi    = vuzp2q_u32(__riscv_vreinterpret_v_u64m1_u32m1(mull_lo), __riscv_vreinterpret_v_u64m1_u32m1(mull_hi));

    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u32   q        =  __riscv_vsub_vv_u32m1(a, mulhi, 4);
               q        =  vshlq_u32(q, __riscv_vreinterpret_v_u32m1_i32m1(divisor.val[1]));
               q        =  __riscv_vadd_vv_u32m1(mulhi, q, 4);
               q        =  vshlq_u32(q, __riscv_vreinterpret_v_u32m1_i32m1(divisor.val[2]));
    return q;
}

NPY_FINLINE npyv_s32 vshrq_n_s32(npyv_s32 a, const int b) {
    const int imm = b - (b >> 5);
    return __riscv_vsra_vx_i32m1(a, imm, 4);
}

NPY_FINLINE npyv_s32 vshlq_s32(npyv_s32 a, npyv_s32 b) {
    // implementation only works within defined range 'b' in [0, 31]
    vbool32_t positive_mask = __riscv_vmsgt_vx_i32m1_b32(b, 0, 4);
    npyv_s32 shl = __riscv_vsll_vv_i32m1(a, __riscv_vreinterpret_v_i32m1_u32m1(b), 4);
    vint64m2_t a_ext = __riscv_vsext_vf2_i64m2(a, 4);
    npyv_s32 b_neg = __riscv_vneg_v_i32m1(b, 4);
    npyv_s32 shr = __riscv_vnclip_wv_i32m1(a_ext, __riscv_vreinterpret_v_i32m1_u32m1(b_neg), __RISCV_VXRM_RDN, 4);
    return __riscv_vmerge_vvm_i32m1(shr, shl, positive_mask, 4);
}

// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    const npyv_s32 mulc_lo = divisor.val[0];
    // high part of signed multiplication
    npyv_s64 mull_lo  = __riscv_vlmul_trunc_v_i64m2_i64m1(__riscv_vwmul_vv_i64m2(a, mulc_lo, 2));
    npyv_s64 mull_hi  = vmull_high_s32(a, divisor.val[0]);
    // get the high unsigned bytes
    npyv_s32 mulhi    = vuzp2q_s32(__riscv_vreinterpret_v_i64m1_i32m1(mull_lo), __riscv_vreinterpret_v_i64m1_i32m1(mull_hi));
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    npyv_s32  q        = vshlq_s32(__riscv_vadd_vv_i32m1(a, mulhi, 4), divisor.val[1]);
              q        = __riscv_vsub_vv_i32m1(q, vshrq_n_s32(a, 31), 4);
              q        = __riscv_vsub_vv_i32m1(__riscv_vxor_vv_i32m1(q, divisor.val[2], 4), divisor.val[2], 4);
    return q;
}

NPY_FINLINE uint64_t vgetq_lane_u64(npyv_u64 a, const int b) {
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vslidedown_vx_u64m1(a, b, 2));
}

NPY_FINLINE int64_t vgetq_lane_s64(npyv_s64 a, const int b) {
    return __riscv_vmv_x_s_i64m1_i64(__riscv_vslidedown_vx_i64m1(a, b, 2));
}

// divide each unsigned 64-bit element by a divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    const uint64_t d = vgetq_lane_u64(divisor.val[0], 0);
    return npyv_set_u64(vgetq_lane_u64(a, 0) / d, vgetq_lane_u64(a, 1) / d);
}

// returns the high 64 bits of signed 64-bit multiplication
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    const int64_t d = vgetq_lane_s64(divisor.val[0], 0);
    return npyv_set_s64(vgetq_lane_s64(a, 0) / d, vgetq_lane_s64(a, 1) / d);
}

/***************************
 * Division
 ***************************/
NPY_FINLINE npyv_f32 npyv_div_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfdiv_vv_f32m1(a, b, 4); }
NPY_FINLINE npyv_f64 npyv_div_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfdiv_vv_f64m1(a, b, 2); }

/***************************
 * FUSED F32
 ***************************/
// multiply and add, a*b + c
NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfmacc_vv_f32m1(c, a, b, 4); }

// multiply and subtract, a*b - c
NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfmacc_vv_f32m1(__riscv_vfneg_v_f32m1(c, 4), a, b, 4); }

// negate multiply and add, -(a*b) + c
NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfnmsac_vv_f32m1(c, a, b, 4); }

// negate multiply and subtract, -(a*b) - c
NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfnmsac_vv_f32m1(__riscv_vfneg_v_f32m1(c, 4), a, b, 4); }

// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
    const npyv_f32 msign = npyv_set_f32(-0.0f, 0.0f, -0.0f, 0.0f);
    return npyv_muladd_f32(a, b, npyv_xor_f32(msign, c));
}

/***************************
 * FUSED F64
 ***************************/
NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfmacc_vv_f64m1(c, a, b, 2); }

NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfmacc_vv_f64m1(__riscv_vfneg_v_f64m1(c, 2), a, b, 2); }

NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfnmsac_vv_f64m1(c, a, b, 2); }

NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfnmsac_vv_f64m1(__riscv_vfneg_v_f64m1(c, 2), a, b, 2); }

NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{
    const npyv_f64 msign = npyv_set_f64(-0.0, 0.0);
    return npyv_muladd_f64(a, b, npyv_xor_f64(msign, c));
}

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a) {
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m1_u32m1(a, __riscv_vmv_v_x_u32m1(0, 4), 4));
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a) {
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m1_u64m1(a, __riscv_vmv_v_x_u64m1(0, 2), 2));
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a) {
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(0, 4), 4));
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a) {
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredosum_vs_f64m1_f64m1(a, __riscv_vfmv_v_f_f64m1(0, 2), 2));
}

NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a) {
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vwredsumu_vs_u8m1_u16m1(a, __riscv_vmv_v_x_u16m1(0, 8), 16));
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a) {
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vwredsumu_vs_u16m1_u32m1(a, __riscv_vmv_v_x_u32m1(0, 4), 8));
}

#endif // _NPY_SIMD_RVV_ARITHMETIC_H