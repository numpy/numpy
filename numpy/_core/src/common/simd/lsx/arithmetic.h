#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_ARITHMETIC_H
#define _NPY_SIMD_LSX_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  __lsx_vadd_b
#define npyv_add_s8  __lsx_vadd_b
#define npyv_add_u16 __lsx_vadd_h
#define npyv_add_s16 __lsx_vadd_h
#define npyv_add_u32 __lsx_vadd_w
#define npyv_add_s32 __lsx_vadd_w
#define npyv_add_u64 __lsx_vadd_d
#define npyv_add_s64 __lsx_vadd_d
#define npyv_add_f32 __lsx_vfadd_s
#define npyv_add_f64 __lsx_vfadd_d

// saturated
#define npyv_adds_u8  __lsx_vsadd_bu
#define npyv_adds_s8  __lsx_vsadd_b
#define npyv_adds_u16 __lsx_vsadd_hu
#define npyv_adds_s16 __lsx_vsadd_h
#define npyv_adds_u32 __lsx_vsadd_wu
#define npyv_adds_s32 __lsx_vsadd_w
#define npyv_adds_u64 __lsx_vsadd_du
#define npyv_adds_s64 __lsx_vsadd_d


/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  __lsx_vsub_b
#define npyv_sub_s8  __lsx_vsub_b
#define npyv_sub_u16 __lsx_vsub_h
#define npyv_sub_s16 __lsx_vsub_h
#define npyv_sub_u32 __lsx_vsub_w
#define npyv_sub_s32 __lsx_vsub_w
#define npyv_sub_u64 __lsx_vsub_d
#define npyv_sub_s64 __lsx_vsub_d
#define npyv_sub_f32 __lsx_vfsub_s
#define npyv_sub_f64 __lsx_vfsub_d

// saturated
#define npyv_subs_u8  __lsx_vssub_bu
#define npyv_subs_s8  __lsx_vssub_b
#define npyv_subs_u16 __lsx_vssub_hu
#define npyv_subs_s16 __lsx_vssub_h
#define npyv_subs_u32 __lsx_vssub_wu
#define npyv_subs_s32 __lsx_vssub_w
#define npyv_subs_u64 __lsx_vssub_du
#define npyv_subs_s64 __lsx_vssub_d


/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  __lsx_vmul_b
#define npyv_mul_s8  __lsx_vmul_b
#define npyv_mul_u16 __lsx_vmul_h
#define npyv_mul_s16 __lsx_vmul_h
#define npyv_mul_u32 __lsx_vmul_w
#define npyv_mul_s32 __lsx_vmul_w
#define npyv_mul_f32 __lsx_vfmul_s
#define npyv_mul_f64 __lsx_vfmul_d


/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi       = __lsx_vmuh_bu(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q           = __lsx_vsub_b(a, mulhi);
            q           = __lsx_vsrl_b(q, divisor.val[1]);
            q           = __lsx_vadd_b(mulhi, q);
            q           = __lsx_vsrl_b(q, divisor.val[2]);

    return  q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    __m128i mulhi       = __lsx_vmuh_b(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m128i q           = __lsx_vsra_b(__lsx_vadd_b(a, mulhi), divisor.val[1]);
            q           = __lsx_vsub_b(q, __lsx_vsrai_b(a, 7));
            q           = __lsx_vsub_b(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = __lsx_vmuh_hu(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = __lsx_vsub_h(a, mulhi);
            q     = __lsx_vsrl_h(q, divisor.val[1]);
            q     = __lsx_vadd_h(mulhi, q);
            q     = __lsx_vsrl_h(q, divisor.val[2]);
    return  q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // high part of signed multiplication
    __m128i mulhi = __lsx_vmuh_h(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m128i q     = __lsx_vsra_h(__lsx_vadd_h(a, mulhi), divisor.val[1]);
            q     = __lsx_vsub_h(q, __lsx_vsrai_h(a, 15));
            q     = __lsx_vsub_h(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = __lsx_vmuh_wu(a, divisor.val[0]);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q          = __lsx_vsub_w(a, mulhi);
            q          = __lsx_vsrl_w(q, divisor.val[1]);
            q          = __lsx_vadd_w(mulhi, q);
            q          = __lsx_vsrl_w(q, divisor.val[2]);
    return  q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    __m128i mulhi = __lsx_vmuh_w(a, divisor.val[0]);
    __m128i q     = __lsx_vsra_w(__lsx_vadd_w(a, mulhi), divisor.val[1]);
            q     = __lsx_vsub_w(q, __lsx_vsrai_w(a, 31));
            q     = __lsx_vsub_w(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);;
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m128i hi     = __lsx_vmuh_du(a, b);
    return hi;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = __lsx_vmuh_du(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = __lsx_vsub_d(a, mulhi);
            q     = __lsx_vsrl_d(q, divisor.val[1]);
            q     = __lsx_vadd_d(mulhi, q);
            q     = __lsx_vsrl_d(q, divisor.val[2]);
    return  q;
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    __m128i   mulhi      = __lsx_vmuh_d(a, divisor.val[0]);
    __m128i   q          = __lsx_vsra_d(__lsx_vadd_d(a, mulhi), divisor.val[1]);
              q          = __lsx_vsub_d(q, __lsx_vsrai_d(a, 63));
              q          = __lsx_vsub_d(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
/***************************
 * Division
 ***************************/
#define npyv_div_f32  __lsx_vfdiv_s
#define npyv_div_f64  __lsx_vfdiv_d
/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f32 __lsx_vfmadd_s
#define npyv_muladd_f64 __lsx_vfmadd_d
// multiply and subtract, a*b - c
#define npyv_mulsub_f32 __lsx_vfmsub_s
#define npyv_mulsub_f64 __lsx_vfmsub_d
// negate multiply and add, -(a*b) + c equal to -(a*b - c)
#define npyv_nmuladd_f32 __lsx_vfnmsub_s
#define npyv_nmuladd_f64 __lsx_vfnmsub_d
// negate multiply and subtract, -(a*b) - c equal to -(a*b +c)
#define npyv_nmulsub_f32 __lsx_vfnmadd_s
#define npyv_nmulsub_f64 __lsx_vfnmadd_d
 // multiply, add for odd elements and subtract even elements.
 // (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
 {
    return __lsx_vfmadd_s(a, b, (__m128)__lsx_vxor_v((__m128i)c, (__m128i)(v4f32){-0.0, 0.0, -0.0, 0.0}));

 }
NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
 {
    return __lsx_vfmadd_d(a, b, (__m128d)__lsx_vxor_v((__m128i)c, (__m128i)(v2f64){-0.0, 0.0}));

 }

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    __m128i t1 = __lsx_vhaddw_du_wu(a, a);
    __m128i t2 = __lsx_vhaddw_qu_du(t1, t1);
    return __lsx_vpickve2gr_wu(t2, 0);
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m128i t = __lsx_vhaddw_qu_du(a, a);
    return __lsx_vpickve2gr_du(t, 0);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    __m128 ft = __lsx_vfadd_s(a, (__m128)__lsx_vbsrl_v((__m128i)a, 8));
    ft = __lsx_vfadd_s(ft, (__m128)__lsx_vbsrl_v(ft, 4));
    return ft[0];
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    __m128d fd = __lsx_vfadd_d(a, (__m128d)__lsx_vreplve_d((__m128i)a, 1));
    return fd[0];
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m128i first = __lsx_vhaddw_hu_bu((__m128i)a,(__m128i)a);
    __m128i second = __lsx_vhaddw_wu_hu((__m128i)first,(__m128i)first);
    __m128i third = __lsx_vhaddw_du_wu((__m128i)second,(__m128i)second);
    __m128i four = __lsx_vhaddw_qu_du((__m128i)third,(__m128i)third);
    return four[0];
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    __m128i t1 = __lsx_vhaddw_wu_hu(a, a);
    __m128i t2 = __lsx_vhaddw_du_wu(t1, t1);
    __m128i t3 = __lsx_vhaddw_qu_du(t2, t2);
    return __lsx_vpickve2gr_w(t3, 0);
}

#endif // _NPY_SIMD_LSX_ARITHMETIC_H
