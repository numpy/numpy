#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_CVT_H
#define _NPY_SIMD_NEON_CVT_H

// convert boolean vectors to integer vectors
#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_s8_b8(A)   vreinterpretq_s8_u8(A)
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_s16_b16(A) vreinterpretq_s16_u16(A)
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_s32_b32(A) vreinterpretq_s32_u32(A)
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s64_b64(A) vreinterpretq_s64_u64(A)
#define npyv_cvt_f32_b32(A) vreinterpretq_f32_u32(A)
#define npyv_cvt_f64_b64(A) vreinterpretq_f64_u64(A)

// convert integer vectors to boolean vectors
#define npyv_cvt_b8_u8(BL)   BL
#define npyv_cvt_b8_s8(BL)   vreinterpretq_u8_s8(BL)
#define npyv_cvt_b16_u16(BL) BL
#define npyv_cvt_b16_s16(BL) vreinterpretq_u16_s16(BL)
#define npyv_cvt_b32_u32(BL) BL
#define npyv_cvt_b32_s32(BL) vreinterpretq_u32_s32(BL)
#define npyv_cvt_b64_u64(BL) BL
#define npyv_cvt_b64_s64(BL) vreinterpretq_u64_s64(BL)
#define npyv_cvt_b32_f32(BL) vreinterpretq_u32_f32(BL)
#define npyv_cvt_b64_f64(BL) vreinterpretq_u64_f64(BL)

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b16(npyv_b16 a, npyv_b16 b)
{ return vcombine_u8(vmovn_u16(a), vmovn_u16(b)); }
// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
    uint16x8_t ab = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
    uint16x8_t cd = vcombine_u16(vmovn_u32(c), vmovn_u32(d));
    return npyv_pack_b16(ab, cd);
}
// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                                     npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    uint32x4_t ab = vcombine_u32(vmovn_u64(a), vmovn_u64(b));
    uint32x4_t cd = vcombine_u32(vmovn_u64(c), vmovn_u64(d));
    uint32x4_t ef = vcombine_u32(vmovn_u64(e), vmovn_u64(f));
    uint32x4_t gh = vcombine_u32(vmovn_u64(g), vmovn_u64(h));
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}
#endif // _NPY_SIMD_NEON_CVT_H
