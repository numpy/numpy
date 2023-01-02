#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_CVT_H
#define _NPY_SIMD_NEON_CVT_H

// convert boolean vectors to integer vectors
#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_s8_b8   vreinterpretq_s8_u8
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_s16_b16 vreinterpretq_s16_u16
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_s32_b32 vreinterpretq_s32_u32
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s64_b64 vreinterpretq_s64_u64
#define npyv_cvt_f32_b32 vreinterpretq_f32_u32
#define npyv_cvt_f64_b64 vreinterpretq_f64_u64

// convert integer vectors to boolean vectors
#define npyv_cvt_b8_u8(BL)   BL
#define npyv_cvt_b8_s8   vreinterpretq_u8_s8
#define npyv_cvt_b16_u16(BL) BL
#define npyv_cvt_b16_s16 vreinterpretq_u16_s16
#define npyv_cvt_b32_u32(BL) BL
#define npyv_cvt_b32_s32 vreinterpretq_u32_s32
#define npyv_cvt_b64_u64(BL) BL
#define npyv_cvt_b64_s64 vreinterpretq_u64_s64
#define npyv_cvt_b32_f32 vreinterpretq_u32_f32
#define npyv_cvt_b64_f64 vreinterpretq_u64_f64

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    const npyv_u8 scale = npyv_set_u8(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
    npyv_u8 seq_scale = vandq_u8(a, scale);
#if defined(__aarch64__)
    const npyv_u8 byteOrder = {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15};
    npyv_u8 v0 = vqtbl1q_u8(seq_scale, byteOrder);
    return vaddlvq_u16(vreinterpretq_u16_u8(v0));
#else
    npyv_u64 sumh = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(seq_scale)));
    return vgetq_lane_u64(sumh, 0) + ((int)vgetq_lane_u64(sumh, 1) << 8);
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    const npyv_u16 scale = npyv_set_u16(1, 2, 4, 8, 16, 32, 64, 128);
    npyv_u16 seq_scale = vandq_u16(a, scale);
#if NPY_SIMD_F64
    return vaddvq_u16(seq_scale);
#else
    npyv_u64 sumh = vpaddlq_u32(vpaddlq_u16(seq_scale));
    return vgetq_lane_u64(sumh, 0) + vgetq_lane_u64(sumh, 1);
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
    const npyv_u32 scale = npyv_set_u32(1, 2, 4, 8);
    npyv_u32 seq_scale = vandq_u32(a, scale);
#if NPY_SIMD_F64
    return vaddvq_u32(seq_scale);
#else
    npyv_u64 sumh = vpaddlq_u32(seq_scale);
    return vgetq_lane_u64(sumh, 0) + vgetq_lane_u64(sumh, 1);
#endif
}
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
    uint64_t lo = vgetq_lane_u64(a, 0);
    uint64_t hi = vgetq_lane_u64(a, 1);
    return ((hi & 0x2) | (lo & 0x1));
}

//expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    r.val[0] = vmovl_u8(vget_low_u8(data));
    r.val[1] = vmovl_u8(vget_high_u8(data));
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    r.val[0] = vmovl_u16(vget_low_u16(data));
    r.val[1] = vmovl_u16(vget_high_u16(data));
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
#if defined(__aarch64__)
    return vuzp1q_u8((uint8x16_t)a, (uint8x16_t)b);
#else
    return vcombine_u8(vmovn_u16(a), vmovn_u16(b));
#endif
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
#if defined(__aarch64__)
    npyv_b16 ab = vuzp1q_u16((uint16x8_t)a, (uint16x8_t)b);
    npyv_b16 cd = vuzp1q_u16((uint16x8_t)c, (uint16x8_t)d);
#else
    npyv_b16 ab = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
    npyv_b16 cd = vcombine_u16(vmovn_u32(c), vmovn_u32(d));
#endif
    return npyv_pack_b8_b16(ab, cd);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
#if defined(__aarch64__)
    npyv_b32 ab = vuzp1q_u32((uint32x4_t)a, (uint32x4_t)b);
    npyv_b32 cd = vuzp1q_u32((uint32x4_t)c, (uint32x4_t)d);
    npyv_b32 ef = vuzp1q_u32((uint32x4_t)e, (uint32x4_t)f);
    npyv_u32 gh = vuzp1q_u32((uint32x4_t)g, (uint32x4_t)h);
#else
    npyv_b32 ab = vcombine_u32(vmovn_u64(a), vmovn_u64(b));
    npyv_b32 cd = vcombine_u32(vmovn_u64(c), vmovn_u64(d));
    npyv_b32 ef = vcombine_u32(vmovn_u64(e), vmovn_u64(f));
    npyv_b32 gh = vcombine_u32(vmovn_u64(g), vmovn_u64(h));
#endif
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// round to nearest integer
#if NPY_SIMD_F64
    #define npyv_round_s32_f32 vcvtnq_s32_f32
    NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
    {
        npyv_s64 lo = vcvtnq_s64_f64(a), hi = vcvtnq_s64_f64(b);
        return vcombine_s32(vmovn_s64(lo), vmovn_s64(hi));
    }
#else
    NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
    {
        // halves will be rounded up. it's very costly
        // to obey IEEE standard on arm7. tests should pass +-1 difference
        const npyv_u32 sign = vdupq_n_u32(0x80000000);
        const npyv_f32 half = vdupq_n_f32(0.5f);
        npyv_f32 sign_half = vbslq_f32(sign, a, half);
        return vcvtq_s32_f32(vaddq_f32(a, sign_half));
    }
#endif

#endif // _NPY_SIMD_NEON_CVT_H
