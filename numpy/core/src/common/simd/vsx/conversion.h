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

//expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;
    npyv_u8 zero = npyv_zero_u8();
    r.val[0] = (npyv_u16)vec_mergel(data, zero);
    r.val[1] = (npyv_u16)vec_mergeh(data, zero);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
    npyv_u16 zero = npyv_zero_u16();
    r.val[0] = (npyv_u32)vec_mergel(data, zero);
    r.val[1] = (npyv_u32)vec_mergeh(data, zero);
    return r;
}

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    const npyv_u8 qperm = npyv_set_u8(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
    return vec_extract((npyv_u32)vec_vbpermq((npyv_u8)a, qperm), 2);
}
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    const npyv_u8 qperm = npyv_setf_u8(128, 112, 96, 80, 64, 48, 32, 16, 0);
    return vec_extract((npyv_u32)vec_vbpermq((npyv_u8)a, qperm), 2);
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
    const npyv_u8 qperm = npyv_setf_u8(128, 96, 64, 32, 0);
    return vec_extract((npyv_u32)vec_vbpermq((npyv_u8)a, qperm), 2);
}
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
    npyv_u64 bit = npyv_shri_u64((npyv_u64)a, 63);
    return vec_extract(bit, 0) | (int)vec_extract(bit, 1) << 1;
}

#endif // _NPY_SIMD_VSX_CVT_H
