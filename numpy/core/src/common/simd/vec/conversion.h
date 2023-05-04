#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_CVT_H
#define _NPY_SIMD_VEC_CVT_H

// convert boolean vectors to integer vectors
#define npyv_cvt_u8_b8(BL)   ((npyv_u8)  BL)
#define npyv_cvt_s8_b8(BL)   ((npyv_s8)  BL)
#define npyv_cvt_u16_b16(BL) ((npyv_u16) BL)
#define npyv_cvt_s16_b16(BL) ((npyv_s16) BL)
#define npyv_cvt_u32_b32(BL) ((npyv_u32) BL)
#define npyv_cvt_s32_b32(BL) ((npyv_s32) BL)
#define npyv_cvt_u64_b64(BL) ((npyv_u64) BL)
#define npyv_cvt_s64_b64(BL) ((npyv_s64) BL)
#if NPY_SIMD_F32
    #define npyv_cvt_f32_b32(BL) ((npyv_f32) BL)
#endif
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
#if NPY_SIMD_F32
    #define npyv_cvt_b32_f32(A) ((npyv_b32) A)
#endif
#define npyv_cvt_b64_f64(A) ((npyv_b64) A)

//expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;
#ifdef NPY_HAVE_VX
    r.val[0] = vec_unpackh(data);
    r.val[1] = vec_unpackl(data);
#else
    npyv_u8 zero = npyv_zero_u8();
    r.val[0] = (npyv_u16)vec_mergeh(data, zero);
    r.val[1] = (npyv_u16)vec_mergel(data, zero);
#endif
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
#ifdef NPY_HAVE_VX
    r.val[0] = vec_unpackh(data);
    r.val[1] = vec_unpackl(data);
#else
    npyv_u16 zero = npyv_zero_u16();
    r.val[0] = (npyv_u32)vec_mergeh(data, zero);
    r.val[1] = (npyv_u32)vec_mergel(data, zero);
#endif
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return vec_pack(a, b);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    npyv_b16 ab = vec_pack(a, b);
    npyv_b16 cd = vec_pack(c, d);
    return npyv_pack_b8_b16(ab, cd);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    npyv_b32 ab = vec_pack(a, b);
    npyv_b32 cd = vec_pack(c, d);
    npyv_b32 ef = vec_pack(e, f);
    npyv_b32 gh = vec_pack(g, h);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// convert boolean vector to integer bitfield
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX2)
    NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
    {
        const npyv_u8 qperm = npyv_set_u8(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
        npyv_u16 r = (npyv_u16)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        return vec_extract(r, 3);
    #else
        return vec_extract(r, 4);
    #endif
        // to suppress ambiguous warning: variable `r` but not used [-Wunused-but-set-variable]
	(void)r;
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
    {
        const npyv_u8 qperm = npyv_setf_u8(128, 112, 96, 80, 64, 48, 32, 16, 0);
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        return vec_extract(r, 6);
    #else
        return vec_extract(r, 8);
    #endif
	// to suppress ambiguous warning: variable `r` but not used [-Wunused-but-set-variable]
        (void)r;
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
    {
    #ifdef NPY_HAVE_VXE
        const npyv_u8 qperm = npyv_setf_u8(128, 128, 128, 128, 128, 96, 64, 32, 0);
    #else
        const npyv_u8 qperm = npyv_setf_u8(128, 96, 64, 32, 0);
    #endif
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        return vec_extract(r, 6);
    #else
        return vec_extract(r, 8);
    #endif
	// to suppress ambiguous warning: variable `r` but not used [-Wunused-but-set-variable]
        (void)r;
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
    {
    #ifdef NPY_HAVE_VXE
        const npyv_u8 qperm = npyv_setf_u8(128, 128, 128, 128, 128, 128, 128, 64, 0);
    #else
        const npyv_u8 qperm = npyv_setf_u8(128, 64, 0);
    #endif
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        return vec_extract(r, 6);
    #else
        return vec_extract(r, 8);
    #endif
	// to suppress ambiguous warning: variable `r` but not used [-Wunused-but-set-variable]
        (void)r;
    }
#else
    NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
    {
        const npyv_u8 scale = npyv_set_u8(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
        npyv_u8 seq_scale = vec_and((npyv_u8)a, scale);
        npyv_u64 sum = vec_sum2(vec_sum4(seq_scale, npyv_zero_u8()), npyv_zero_u32());
        return vec_extract(sum, 0) + ((int)vec_extract(sum, 1) << 8);
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
    {
        const npyv_u16 scale = npyv_set_u16(1, 2, 4, 8, 16, 32, 64, 128);
        npyv_u16 seq_scale = vec_and((npyv_u16)a, scale);
        npyv_u64 sum = vec_sum2(seq_scale, npyv_zero_u16());
        return vec_extract(vec_sum_u128(sum, npyv_zero_u64()), 15);
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
    {
        const npyv_u32 scale = npyv_set_u32(1, 2, 4, 8);
        npyv_u32 seq_scale = vec_and((npyv_u32)a, scale);
        return vec_extract(vec_sum_u128(seq_scale, npyv_zero_u32()), 15);
    }
    NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
    {
        const npyv_u64 scale = npyv_set_u64(1, 2);
        npyv_u64 seq_scale = vec_and((npyv_u64)a, scale);
        return vec_extract(vec_sum_u128(seq_scale, npyv_zero_u64()), 15);
    }
#endif
// truncate compatible with all compilers(internal use for now)
#if NPY_SIMD_F32
    NPY_FINLINE npyv_s32 npyv__trunc_s32_f32(npyv_f32 a)
    {
    #ifdef NPY_HAVE_VXE2
        return vec_signed(a);
    #elif defined(NPY_HAVE_VXE)
        return vec_packs(vec_signed(npyv_doublee(vec_mergeh(a,a))),
            vec_signed(npyv_doublee(vec_mergel(a, a))));
    // VSX
    #elif defined(__IBMC__)
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
#endif

NPY_FINLINE npyv_s32 npyv__trunc_s32_f64(npyv_f64 a, npyv_f64 b)
{
#ifdef NPY_HAVE_VX
    return vec_packs(vec_signed(a), vec_signed(b));
// VSX
#elif defined(__IBMC__)
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
#if NPY_SIMD_F32
    NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
    { return npyv__trunc_s32_f32(vec_rint(a)); }
#endif
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__trunc_s32_f64(vec_rint(a), vec_rint(b)); }

#endif // _NPY_SIMD_VEC_CVT_H
