#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_CVT_H
#define _NPY_SIMD_VSX_CVT_H

// convert boolean vectors to integer vectors
#define npyv_cvt_u8_b8(BL)   ((npyv_u8)  BL)
#define npyv_cvt_s8_b8(BL)   ((npyv_s8)  BL)
#define npyv_cvt_u16_b16(BL) ((npyv_u16) BL)
#define npyv_cvt_s16_b16(BL) ((npyv_s16) BL)
#define npyv_cvt_u32_b32(BL) ((npyv_u32) BL)
#define npyv_cvt_s32_b32(BL) ((npyv_s32) BL)
#define npyv_cvt_u64_b64(BL) ((npyv_u64) BL)
#define npyv_cvt_s64_b64(BL) ((npyv_s64) BL)
#define npyv_cvt_f32_b32(BL) ((npyv_f32) BL)
#define npyv_cvt_f64_b64(BL) ((npyv_f64) BL)

// convert integer vectors to boolean vectors
#define npyv_cvt_b8_u8(A)   ((npyv_b8)  A)
#define npyv_cvt_b8_s8(A)   ((npyv_b8)  A)
#define npyv_cvt_b16_u16(A) ((npyv_b16) A)
#define npyv_cvt_b16_s16(A) ((npyv_b16) A)
#define npyv_cvt_b32_u32(A) ((npyv_b32) A)
#define npyv_cvt_b32_s32(A) ((npyv_b32) A)
#define npyv_cvt_b64_u64(A) ((npyv_b64) A)
#define npyv_cvt_b64_s64(A) ((npyv_b64) A)
#define npyv_cvt_b32_f32(A) ((npyv_b32) A)
#define npyv_cvt_b64_f64(A) ((npyv_b64) A)

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b16(npyv_b16 a, npyv_b16 b)
{ return (npyv_b8)vec_pack((npyv_u16)a, (npyv_u16)b); }
// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
    npyv_u16 ab = vec_pack((npyv_u32)a, (npyv_u32)b);
    npyv_u16 cd = vec_pack((npyv_u32)c, (npyv_u32)d);
    return (npyv_b8)vec_pack(ab, cd);
}
NPY_FINLINE npyv_b8 npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                                     npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    npyv_b32 ab = (npyv_b32)vec_pack((npyv_u32)a, (npyv_u32)b);
    npyv_b32 cd = (npyv_b32)vec_pack((npyv_u32)c, (npyv_u32)d);
    npyv_b32 ef = (npyv_b32)vec_pack((npyv_u32)e, (npyv_u32)f);
    npyv_b32 gh = (npyv_b32)vec_pack((npyv_u32)g, (npyv_u32)h);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}
#endif // _NPY_SIMD_VSX_CVT_H
