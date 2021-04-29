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
    r.val[0] = (npyv_u16)vec_mergeh(data, zero);
    r.val[1] = (npyv_u16)vec_mergel(data, zero);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
    npyv_u16 zero = npyv_zero_u16();
    r.val[0] = (npyv_u32)vec_mergeh(data, zero);
    r.val[1] = (npyv_u32)vec_mergel(data, zero);
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

// truncate compatible with all compilers(internal use for now)
NPY_FINLINE npyv_s32 npyv__trunc_s32_f32(npyv_f32 a)
{
#ifdef __IBMC__
    return vec_cts(a, 0);
#elif defined(__clang__)
    /**
     * old versions of CLANG doesn't support %x<n> in the inline asm template
     * which fixes register number when using any of the register constraints wa, wd, wf.
     * therefore, we count on built-in functions.
     */
    return __builtin_convertvector(a, npyv_s32);
#else // gcc
    npyv_s32 ret;
    __asm__ ("xvcvspsxws %x0,%x1" : "=wa" (ret) : "wa" (a));
    return ret;
#endif
}
NPY_FINLINE npyv_s32 npyv__trunc_s32_f64(npyv_f64 a, npyv_f64 b)
{
#ifdef __IBMC__
    const npyv_u8 seq_even = npyv_set_u8(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27);
    // unfortunately, XLC missing asm register vsx fixer
    // hopefully, xlc can optimize around big-endian compatibility
    npyv_s32 lo_even = vec_cts(a, 0);
    npyv_s32 hi_even = vec_cts(b, 0);
    return vec_perm(lo_even, hi_even, seq_even);
#else
    const npyv_u8 seq_odd = npyv_set_u8(4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31);
    #ifdef __clang__
        // __builtin_convertvector doesn't support this conversion on wide range of versions
        // fortunately, almost all versions have direct builtin of 'xvcvdpsxws'
        npyv_s32 lo_odd = __builtin_vsx_xvcvdpsxws(a);
        npyv_s32 hi_odd = __builtin_vsx_xvcvdpsxws(b);
    #else // gcc
        npyv_s32 lo_odd, hi_odd;
        __asm__ ("xvcvdpsxws %x0,%x1" : "=wa" (lo_odd) : "wa" (a));
        __asm__ ("xvcvdpsxws %x0,%x1" : "=wa" (hi_odd) : "wa" (b));
    #endif
    return vec_perm(lo_odd, hi_odd, seq_odd);
#endif
}

// round to nearest integer (assuming even)
NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
{ return npyv__trunc_s32_f32(vec_rint(a)); }

NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__trunc_s32_f64(vec_rint(a), vec_rint(b)); }

#endif // _NPY_SIMD_VSX_CVT_H
