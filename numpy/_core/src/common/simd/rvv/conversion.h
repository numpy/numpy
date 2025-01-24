#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_CVT_H
#define _NPY_SIMD_RVV_CVT_H

#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_s8_b8(A)   __riscv_vreinterpret_v_u8m1_i8m1(A)
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_s16_b16(A) __riscv_vreinterpret_v_u16m1_i16m1(A)
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_s32_b32(A) __riscv_vreinterpret_v_u32m1_i32m1(A)
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s64_b64(A) __riscv_vreinterpret_v_u64m1_i64m1(A)
#define npyv_cvt_f32_b32(A) __riscv_vreinterpret_v_u32m1_f32m1(A)
#define npyv_cvt_f64_b64(A) __riscv_vreinterpret_v_u64m1_f64m1(A)

#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   __riscv_vreinterpret_v_i8m1_u8m1(A)
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) __riscv_vreinterpret_v_i16m1_u16m1(A)
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) __riscv_vreinterpret_v_i32m1_u32m1(A)
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) __riscv_vreinterpret_v_i64m1_u64m1(A)
#define npyv_cvt_b32_f32(A) __riscv_vreinterpret_v_f32m1_u32m1(A)
#define npyv_cvt_b64_f64(A) __riscv_vreinterpret_v_f64m1_u64m1(A)

NPY_FINLINE npyv_u8 vqtbl1q_u8(npyv_u8 t, npyv_u8 idx) {
    vbool8_t mask = __riscv_vmsgeu_vx_u8m1_b8(idx, 16, 16);
    return __riscv_vmerge_vxm_u8m1(__riscv_vrgather_vv_u8m1(t, idx, 16), 0, mask, 16);
}

NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    const npyv_u8 scale = npyv_set_u8(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
    npyv_u8 seq_scale = __riscv_vand_vv_u8m1(a, scale, 16);
    const npyv_u8 byteOrder = npyv_set_u8(0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15);
    npyv_u8 v0 = vqtbl1q_u8(seq_scale, byteOrder);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vwredsumu_vs_u16m1_u32m1(__riscv_vreinterpret_v_u8m1_u16m1(v0), __riscv_vmv_v_x_u32m1(0, 4), 8));
}

NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    const npyv_u16 scale = npyv_set_u16(1, 2, 4, 8, 16, 32, 64, 128);
    npyv_u16 seq_scale = __riscv_vand_vv_u16m1(a, scale, 8);
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vredsum_vs_u16m1_u16m1(seq_scale, __riscv_vmv_v_x_u16m1(0, 8), 8));
}

NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
    const npyv_u32 scale = npyv_set_u32(1, 2, 4, 8);
    npyv_u32 seq_scale = __riscv_vand_vv_u32m1(a, scale, 4);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m1_u32m1(seq_scale, __riscv_vmv_v_x_u32m1(0, 4), 4));
}

NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    uint64_t first = __riscv_vmv_x_s_u64m1_u64(a);
    uint64_t second = __riscv_vmv_x_s_u64m1_u64(__riscv_vslidedown_vx_u64m1(a, 1, vlen));
    return ((second & 0x2) | (first & 0x1));
}

//expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    r.val[0] = __riscv_vlmul_trunc_v_u16m2_u16m1(__riscv_vzext_vf2_u16m2(data, 8));
    r.val[1] = __riscv_vlmul_trunc_v_u16m2_u16m1(__riscv_vzext_vf2_u16m2(__riscv_vslidedown_vx_u8m1(data, 8, 16), 8));
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    r.val[0] = __riscv_vlmul_trunc_v_u32m2_u32m1(__riscv_vzext_vf2_u32m2(data, 4));
    r.val[1] = __riscv_vlmul_trunc_v_u32m2_u32m1(__riscv_vzext_vf2_u32m2(__riscv_vslidedown_vx_u16m1(data, 4, 8), 4));
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    npyv_b8 a8 = __riscv_vreinterpret_v_u16m1_u8m1(a);
    npyv_b8 b8 = __riscv_vreinterpret_v_u16m1_u8m1(b);
    return vuzp1q_u8(a8, b8);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    npyv_b16 a16 = __riscv_vreinterpret_v_u32m1_u16m1(a);
    npyv_b16 b16 = __riscv_vreinterpret_v_u32m1_u16m1(b);
    npyv_b16 c16 = __riscv_vreinterpret_v_u32m1_u16m1(c);
    npyv_b16 d16 = __riscv_vreinterpret_v_u32m1_u16m1(d);

    npyv_b16 ab = vuzp1q_u16(a16, b16);
    npyv_b16 cd = vuzp1q_u16(c16, d16);

    return npyv_pack_b8_b16(ab, cd);
}

 // pack eight 64-bit boolean vectors into one 8-bit boolean vector
 NPY_FINLINE npyv_b8
 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                  npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    npyv_b32 a32 = __riscv_vreinterpret_v_u64m1_u32m1(a);
    npyv_b32 b32 = __riscv_vreinterpret_v_u64m1_u32m1(b);
    npyv_b32 c32 = __riscv_vreinterpret_v_u64m1_u32m1(c);
    npyv_b32 d32 = __riscv_vreinterpret_v_u64m1_u32m1(d);
    npyv_b32 e32 = __riscv_vreinterpret_v_u64m1_u32m1(e);
    npyv_b32 f32 = __riscv_vreinterpret_v_u64m1_u32m1(f);
    npyv_b32 g32 = __riscv_vreinterpret_v_u64m1_u32m1(g);
    npyv_b32 h32 = __riscv_vreinterpret_v_u64m1_u32m1(h);

    npyv_b32 ab = vuzp1q_u32(a32, b32);
    npyv_b32 cd = vuzp1q_u32(c32, d32);
    npyv_b32 ef = vuzp1q_u32(e32, f32);
    npyv_u32 gh = vuzp1q_u32(g32, h32);

    return npyv_pack_b8_b32(ab, cd, ef, gh);
 }

// round to nearest integer
NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    // (round-to-nearest-even)
    return __riscv_vfcvt_x_f_v_i32m1(a, vlen);
}

NPY_FINLINE npyv_s32 vmovn_s64(npyv_s64 a) {
    return __riscv_vnsra_wx_i32m1(__riscv_vlmul_ext_v_i64m1_i64m2(a), 0, 2);
}

NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    npyv_s64 lo = __riscv_vfcvt_x_f_v_i64m1(a, 2), hi = __riscv_vfcvt_x_f_v_i64m1(b, 2);
    return __riscv_vslideup_vx_i32m1(vmovn_s64(lo), vmovn_s64(hi), 2, 4);
}

#endif // _NPY_SIMD_RVV_CVT_H
