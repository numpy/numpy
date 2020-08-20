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

#endif // _NPY_SIMD_VSX_CVT_H
