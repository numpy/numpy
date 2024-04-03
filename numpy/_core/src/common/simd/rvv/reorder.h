#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_REORDER_H
#define _NPY_SIMD_RVV_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8(A, B)  vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
#define npyv_combinel_s8(A, B)  vreinterpretq_s8_u64(vzip1q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
#define npyv_combinel_u16(A, B) vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
#define npyv_combinel_s16(A, B) vreinterpretq_s16_u64(vzip1q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
#define npyv_combinel_u32(A, B) vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
#define npyv_combinel_s32(A, B) vreinterpretq_s32_u64(vzip1q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
#define npyv_combinel_u64       vzip1q_u64
#define npyv_combinel_s64       vzip1q_s64
#define npyv_combinel_f32(A, B) vreinterpretq_f32_u64(vzip1q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
#define npyv_combinel_f64       vzip1q_f64

// combine higher part of two vectors
#define npyv_combineh_u8(A, B)  vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
#define npyv_combineh_s8(A, B)  vreinterpretq_s8_u64(vzip2q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
#define npyv_combineh_u16(A, B) vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
#define npyv_combineh_s16(A, B) vreinterpretq_s16_u64(vzip2q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
#define npyv_combineh_u32(A, B) vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
#define npyv_combineh_s32(A, B) vreinterpretq_s32_u64(vzip2q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
#define npyv_combineh_u64       vzip2q_u64
#define npyv_combineh_s64       vzip2q_s64
#define npyv_combineh_f32(A, B) vreinterpretq_f32_u64(vzip2q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
#define npyv_combineh_f64       vzip2q_f64


NPY_FINLINE npyv_u8x2 npyv_combine_u8(npyv_u8 a, npyv_u8 b) {
  npyv_u8x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, u8)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, u8)(a, b);
  return r;
}
NPY_FINLINE npyv_s8x2 npyv_combine_s8(npyv_s8 a, npyv_s8 b) {
  npyv_s8x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, s8)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, s8)(a, b);
  return r;
}
NPY_FINLINE npyv_u16x2 npyv_combine_u16(npyv_u16 a, npyv_u16 b) {
  npyv_u16x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, u16)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, u16)(a, b);
  return r;
}
NPY_FINLINE npyv_s16x2 npyv_combine_s16(npyv_s16 a, npyv_s16 b) {
  npyv_s16x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, s16)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, s16)(a, b);
  return r;
}
NPY_FINLINE npyv_u32x2 npyv_combine_u32(npyv_u32 a, npyv_u32 b) {
  npyv_u32x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, u32)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, u32)(a, b);
  return r;
}
NPY_FINLINE npyv_s32x2 npyv_combine_s32(npyv_s32 a, npyv_s32 b) {
  npyv_s32x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, s32)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, s32)(a, b);
  return r;
}
NPY_FINLINE npyv_u64x2 npyv_combine_u64(npyv_u64 a, npyv_u64 b) {
  npyv_u64x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, u64)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, u64)(a, b);
  return r;
}
NPY_FINLINE npyv_s64x2 npyv_combine_s64(npyv_s64 a, npyv_s64 b) {
  npyv_s64x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, s64)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, s64)(a, b);
  return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(npyv_f32 a, npyv_f32 b) {
  npyv_f32x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, f32)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, f32)(a, b);
  return r;
}

NPY_FINLINE npyv_f64x2 npyv_combine_f64(npyv_f64 a, npyv_f64 b) {
  npyv_f64x2 r;
  r.val[0] = NPY_CAT(npyv_combinel_, f64)(a, b);
  r.val[1] = NPY_CAT(npyv_combineh_, f64)(a, b);
  return r;
}
NPY_FINLINE npyv_u8x2 npyv_zip_u8(npyv_u8 a, npyv_u8 b) {
  npyv_u8x2 r;
  r.val[0] = vzip1q_u8(a, b);
  r.val[1] = vzip2q_u8(a, b);
  return r;
}
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 a, npyv_u8 b) {
  npyv_u8x2 r;
  r.val[0] = vuzp1q_u8(a, b);
  r.val[1] = vuzp2q_u8(a, b);
  return r;
}
NPY_FINLINE npyv_s8x2 npyv_zip_s8(npyv_s8 a, npyv_s8 b) {
  npyv_s8x2 r;
  r.val[0] = vzip1q_s8(a, b);
  r.val[1] = vzip2q_s8(a, b);
  return r;
}
NPY_FINLINE npyv_s8x2 npyv_unzip_s8(npyv_s8 a, npyv_s8 b) {
  npyv_s8x2 r;
  r.val[0] = vuzp1q_s8(a, b);
  r.val[1] = vuzp2q_s8(a, b);
  return r;
}
NPY_FINLINE npyv_u16x2 npyv_zip_u16(npyv_u16 a, npyv_u16 b) {
  npyv_u16x2 r;
  r.val[0] = vzip1q_u16(a, b);
  r.val[1] = vzip2q_u16(a, b);
  return r;
}
NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 a, npyv_u16 b) {
  npyv_u16x2 r;
  r.val[0] = vuzp1q_u16(a, b);
  r.val[1] = vuzp2q_u16(a, b);
  return r;
}
NPY_FINLINE npyv_s16x2 npyv_zip_s16(npyv_s16 a, npyv_s16 b) {
  npyv_s16x2 r;
  r.val[0] = vzip1q_s16(a, b);
  r.val[1] = vzip2q_s16(a, b);
  return r;
}
NPY_FINLINE npyv_s16x2 npyv_unzip_s16(npyv_s16 a, npyv_s16 b) {
  npyv_s16x2 r;
  r.val[0] = vuzp1q_s16(a, b);
  r.val[1] = vuzp2q_s16(a, b);
  return r;
}
NPY_FINLINE npyv_u32x2 npyv_zip_u32(npyv_u32 a, npyv_u32 b) {
  npyv_u32x2 r;
  r.val[0] = vzip1q_u32(a, b);
  r.val[1] = vzip2q_u32(a, b);
  return r;
}
NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 a, npyv_u32 b) {
  npyv_u32x2 r;
  r.val[0] = vuzp1q_u32(a, b);
  r.val[1] = vuzp2q_u32(a, b);
  return r;
}
NPY_FINLINE npyv_s32x2 npyv_zip_s32(npyv_s32 a, npyv_s32 b) {
  npyv_s32x2 r;
  r.val[0] = vzip1q_s32(a, b);
  r.val[1] = vzip2q_s32(a, b);
  return r;
}
NPY_FINLINE npyv_s32x2 npyv_unzip_s32(npyv_s32 a, npyv_s32 b) {
  npyv_s32x2 r;
  r.val[0] = vuzp1q_s32(a, b);
  r.val[1] = vuzp2q_s32(a, b);
  return r;
}
NPY_FINLINE npyv_f32x2 npyv_zip_f32(npyv_f32 a, npyv_f32 b) {
  npyv_f32x2 r;
  r.val[0] = vzip1q_f32(a, b);
  r.val[1] = vzip2q_f32(a, b);
  return r;
}
NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 a, npyv_f32 b) {
  npyv_f32x2 r;
  r.val[0] = vuzp1q_f32(a, b);
  r.val[1] = vuzp2q_f32(a, b);
  return r;
}

#endif //_NPY_SIMD_RVV_REORDER_H