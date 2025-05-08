#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_CVT_H
#define _NPY_SIMD_LASX_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(BL)   BL
#define npyv_cvt_s8_b8(BL)   BL
#define npyv_cvt_u16_b16(BL) BL
#define npyv_cvt_s16_b16(BL) BL
#define npyv_cvt_u32_b32(BL) BL
#define npyv_cvt_s32_b32(BL) BL
#define npyv_cvt_u64_b64(BL) BL
#define npyv_cvt_s64_b64(BL) BL
#define npyv_cvt_f32_b32(BL) (__m256)(BL)
#define npyv_cvt_f64_b64(BL) (__m256d)(BL)

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32(A) (__m256i)(A)
#define npyv_cvt_b64_f64(A) (__m256i)(A)

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    __m256i t1 = __lasx_xvmsknz_b(a);
    __m256i t2 = __lasx_xvpickve_w(t1, 4);
            t2 = __lasx_xvslli_w(t2, 16);
            t2 = __lasx_xvor_v(t1, t2);
    return (npy_uint32)t2[0];
}

NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    __m256i b = __lasx_xvsat_hu(a, 7);
    __m256i pack = __lasx_xvpickev_b(b, b);
    __m256i t1 = __lasx_xvmsknz_b(pack);
    __m256i t2 = __lasx_xvpickve_w(t1, 4);
            t2 = __lasx_xvslli_w(t2, 16);
            t2 = __lasx_xvor_v(t1, t2);
    return (npy_uint16)t2[0];
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
   __m256i t1 = __lasx_xvmskltz_w(a);
   __m256i t2 = __lasx_xvpickve_w(t1, 4);
           t2 = __lasx_xvslli_w(t2, 4);
           t2 = __lasx_xvor_v(t1, t2);
   v8i32 ret = (v8i32)t2;
   return ret[0];
}

NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
   __m256i t1 = __lasx_xvmskltz_d(a);
   __m256i t2 = __lasx_xvpickve_d(t1, 2);
           t2 = __lasx_xvslli_w(t2, 2);
           t2 = __lasx_xvor_v(t1, t2);
   v4i64 ret = (v4i64)t2;
   return ret[0];
}

// expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    __m256i t1 = __lasx_xvsllwil_hu_bu(data, 0);
    __m256i t2 = __lasx_xvexth_hu_bu(data);
    r.val[0] = __lasx_xvpermi_q(t2, t1, 0x20);
    r.val[1] = __lasx_xvpermi_q(t2, t1, 0x31);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    __m256i t1 = __lasx_xvsllwil_wu_hu(data, 0);
    __m256i t2 = __lasx_xvexth_wu_hu(data);
    r.val[0] = __lasx_xvpermi_q(t2, t1, 0x20);
    r.val[1] = __lasx_xvpermi_q(t2, t1, 0x31);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return __lasx_xvpermi_d(__lasx_xvpickev_b(__lasx_xvsat_h(b, 7),__lasx_xvsat_h(a, 7)), 0xd8);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    __m256i ab = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(b, 15), __lasx_xvsat_w(a, 15)), 0xd8);
    __m256i cd = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(d, 15), __lasx_xvsat_w(c, 15)), 0xd8);
    return npyv_pack_b8_b16(ab, cd);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __m256i ab = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(b, 15), __lasx_xvsat_w(a, 15)), 0xd8);
    __m256i cd = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(d, 15), __lasx_xvsat_w(c, 15)), 0xd8);
    __m256i ef = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(f, 15), __lasx_xvsat_w(e, 15)), 0xd8);
    __m256i gh = __lasx_xvpermi_d(__lasx_xvpickev_h(__lasx_xvsat_w(h, 15), __lasx_xvsat_w(g, 15)), 0xd8);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 __lasx_xvftintrne_w_s
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    return __lasx_xvpermi_d(__lasx_xvftintrne_w_d(b, a), 0xd8);
}
#endif // _NPY_SIMD_LASX_CVT_H
