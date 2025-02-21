#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_ARITHMETIC_H
#define _NPY_SIMD_LASX_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  __lasx_xvadd_b
#define npyv_add_s8  __lasx_xvadd_b
#define npyv_add_u16 __lasx_xvadd_h
#define npyv_add_s16 __lasx_xvadd_h
#define npyv_add_u32 __lasx_xvadd_w
#define npyv_add_s32 __lasx_xvadd_w
#define npyv_add_u64 __lasx_xvadd_d
#define npyv_add_s64 __lasx_xvadd_d
#define npyv_add_f32 __lasx_xvfadd_s
#define npyv_add_f64 __lasx_xvfadd_d

// saturated
#define npyv_adds_u8  __lasx_xvsadd_bu
#define npyv_adds_s8  __lasx_xvsadd_b
#define npyv_adds_u16 __lasx_xvsadd_hu
#define npyv_adds_s16 __lasx_xvsadd_h
#define npyv_adds_u32 __lasx_xvsadd_wu
#define npyv_adds_s32 __lasx_xvsadd_w
#define npyv_adds_u64 __lasx_xvsadd_du
#define npyv_adds_s64 __lasx_xvsadd_d


/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  __lasx_xvsub_b
#define npyv_sub_s8  __lasx_xvsub_b
#define npyv_sub_u16 __lasx_xvsub_h
#define npyv_sub_s16 __lasx_xvsub_h
#define npyv_sub_u32 __lasx_xvsub_w
#define npyv_sub_s32 __lasx_xvsub_w
#define npyv_sub_u64 __lasx_xvsub_d
#define npyv_sub_s64 __lasx_xvsub_d
#define npyv_sub_f32 __lasx_xvfsub_s
#define npyv_sub_f64 __lasx_xvfsub_d

// saturated
#define npyv_subs_u8  __lasx_xvssub_bu
#define npyv_subs_s8  __lasx_xvssub_b
#define npyv_subs_u16 __lasx_xvssub_hu
#define npyv_subs_s16 __lasx_xvssub_h
#define npyv_subs_u32 __lasx_xvssub_wu
#define npyv_subs_s32 __lasx_xvssub_w
#define npyv_subs_u64 __lasx_xvssub_du
#define npyv_subs_s64 __lasx_xvssub_d


/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  __lasx_xvmul_b
#define npyv_mul_s8  __lasx_xvmul_b
#define npyv_mul_u16 __lasx_xvmul_h
#define npyv_mul_s16 __lasx_xvmul_h
#define npyv_mul_u32 __lasx_xvmul_w
#define npyv_mul_s32 __lasx_xvmul_w
#define npyv_mul_f32 __lasx_xvfmul_s
#define npyv_mul_f64 __lasx_xvfmul_d


/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    // high part of unsigned multiplication
    __m256i mulhi       = __lasx_xvmuh_bu(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q           = __lasx_xvsub_b(a, mulhi);
            q           = __lasx_xvsrl_b(q, divisor.val[1]);
            q           = __lasx_xvadd_b(mulhi, q);
            q           = __lasx_xvsrl_b(q, divisor.val[2]);

    return  q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    __m256i mulhi       = __lasx_xvmuh_b(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m256i q           = __lasx_xvsra_b(__lasx_xvadd_b(a, mulhi), divisor.val[1]);
            q           = __lasx_xvsub_b(q, __lasx_xvsrai_b(a, 7));
            q           = __lasx_xvsub_b(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // high part of unsigned multiplication
    __m256i mulhi = __lasx_xvmuh_hu(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q     = __lasx_xvsub_h(a, mulhi);
            q     = __lasx_xvsrl_h(q, divisor.val[1]);
            q     = __lasx_xvadd_h(mulhi, q);
            q     = __lasx_xvsrl_h(q, divisor.val[2]);
    return  q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // high part of signed multiplication
    __m256i mulhi = __lasx_xvmuh_h(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m256i q     = __lasx_xvsra_h(__lasx_xvadd_h(a, mulhi), divisor.val[1]);
            q     = __lasx_xvsub_h(q, __lasx_xvsrai_h(a, 15));
            q     = __lasx_xvsub_h(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // high part of unsigned multiplication
    __m256i mulhi = __lasx_xvmuh_wu(a, divisor.val[0]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q          = __lasx_xvsub_w(a, mulhi);
            q          = __lasx_xvsrl_w(q, divisor.val[1]);
            q          = __lasx_xvadd_w(mulhi, q);
            q          = __lasx_xvsrl_w(q, divisor.val[2]);
    return  q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_w(a, divisor.val[0]);
    __m256i q     = __lasx_xvsra_w(__lasx_xvadd_w(a, mulhi), divisor.val[1]);
            q     = __lasx_xvsub_w(q, __lasx_xvsrai_w(a, 31));
            q     = __lasx_xvsub_w(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);;
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m256i hi     = __lasx_xvmuh_du(a, b);
    return hi;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    __m256i mulhi = __lasx_xvmuh_du(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m256i q     = __lasx_xvsub_d(a, mulhi);
            q     = __lasx_xvsrl_d(q, divisor.val[1]);
            q     = __lasx_xvadd_d(mulhi, q);
            q     = __lasx_xvsrl_d(q, divisor.val[2]);
    return  q;
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    __m256i   mulhi      = __lasx_xvmuh_d(a, divisor.val[0]);
    __m256i   q          = __lasx_xvsra_d(__lasx_xvadd_d(a, mulhi), divisor.val[1]);
              q          = __lasx_xvsub_d(q, __lasx_xvsrai_d(a, 63));
              q          = __lasx_xvsub_d(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
/***************************
 * Division
 ***************************/
#define npyv_div_f32  __lasx_xvfdiv_s
#define npyv_div_f64  __lasx_xvfdiv_d
/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f32 __lasx_xvfmadd_s
#define npyv_muladd_f64 __lasx_xvfmadd_d
// multiply and subtract, a*b - c
#define npyv_mulsub_f32 __lasx_xvfmsub_s
#define npyv_mulsub_f64 __lasx_xvfmsub_d
// negate multiply and add, -(a*b) + c equal to -(a*b - c)
#define npyv_nmuladd_f32 __lasx_xvfnmsub_s
#define npyv_nmuladd_f64 __lasx_xvfnmsub_d
// negate multiply and subtract, -(a*b) - c equal to -(a*b +c)
#define npyv_nmulsub_f32 __lasx_xvfnmadd_s
#define npyv_nmulsub_f64 __lasx_xvfnmadd_d
 // multiply, add for odd elements and subtract even elements.
 // (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
    return __lasx_xvfmadd_s(a, b, (__m256)__lasx_xvxor_v((__m256i)c, (__m256i)(v8f32){-0.0, 0.0, -0.0, 0.0,-0.0, 0.0, -0.0, 0.0}));

}
NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{
    return __lasx_xvfmadd_d(a, b, (__m256d)__lasx_xvxor_v((__m256i)c, (__m256i)(v4f64){-0.0, 0.0,-0.0, 0.0}));

}

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    __m256i t1 = __lasx_xvhaddw_du_wu(a, a);
    __m256i t2 = __lasx_xvhaddw_qu_du(t1, t1);
    __m256i t3 = __lasx_xvpickve_w(t2, 4);
            t3 = __lasx_xvsadd_wu(t2, t3);
    return __lasx_xvpickve2gr_wu(t3, 0);
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m256i t1 = __lasx_xvhaddw_qu_du(a, a);
    __m256i t2 = __lasx_xvpickve_d(t1, 2);
            t2 = __lasx_xvsadd_wu(t1, t2);
    return __lasx_xvpickve2gr_du(t2, 0);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    __m256 ft1 = __lasx_xvfadd_s(a, (__m256)__lasx_xvbsrl_v((__m256i)a, 8));
    __m256 ft2 = ft1;
           ft2 = (__m256)__lasx_xvpermi_d((__m256i)ft1, 0x4e);
           ft1 = __lasx_xvfadd_s(ft1, ft2);
    ft1 = __lasx_xvfadd_s(ft1, (__m256)__lasx_xvbsrl_v(ft1, 4));
    return ft1[0];
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    __m256d fd = __lasx_xvfadd_d(a, (__m256d)__lasx_xvbsrl_v((__m256i)a, 8));
            fd = __lasx_xvfadd_d(fd, (__m256d)__lasx_xvpermi_d((__m256i)fd, 0x02));
    return fd[0];
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m256i t1 = __lasx_xvhaddw_hu_bu((__m256i)a, (__m256i)a);
    __m256i t2 = __lasx_xvhaddw_wu_hu((__m256i)t1, (__m256i)t1);
    __m256i t3 = __lasx_xvhaddw_du_wu((__m256i)t2, (__m256i)t2);
    __m256i t4 = __lasx_xvhaddw_qu_du((__m256i)t3, (__m256i)t3);
            t4 = __lasx_xvpermi_d(t4, 0x88);
            t4 = __lasx_xvhaddw_qu_du((__m256i)t4, (__m256i)t4);
    return  __lasx_xvpickve2gr_w(t4, 0);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    __m256i t1 = __lasx_xvhaddw_wu_hu(a, a);
    __m256i t2 = __lasx_xvhaddw_du_wu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_qu_du(t2, t2);
            t3 = __lasx_xvpermi_d(t3, 0x88);
            t3 = __lasx_xvhaddw_qu_du(t3, t3);
    return __lasx_xvpickve2gr_w(t3, 0);
}

#endif // _NPY_SIMD_LASX_ARITHMETIC_H
