#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_CVT_H
#define _NPY_SIMD_LSX_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(BL)   BL
#define npyv_cvt_s8_b8(BL)   BL
#define npyv_cvt_u16_b16(BL) BL
#define npyv_cvt_s16_b16(BL) BL
#define npyv_cvt_u32_b32(BL) BL
#define npyv_cvt_s32_b32(BL) BL
#define npyv_cvt_u64_b64(BL) BL
#define npyv_cvt_s64_b64(BL) BL
#define npyv_cvt_f32_b32(BL) (__m128)(BL)
#define npyv_cvt_f64_b64(BL) (__m128d)(BL)

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32(A) (__m128i)(A)
#define npyv_cvt_b64_f64(A) (__m128i)(A)

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{ return (npy_uint16)__lsx_vmsknz_b(a)[0]; }
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    __m128i b = __lsx_vsat_hu(a, 7);
    __m128i pack = __lsx_vpickev_b(b, b);
    return (npy_uint8)__lsx_vmsknz_b(pack)[0];
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
   __m128i b = __lsx_vmskltz_w(a);
   v4i32 ret = (v4i32)b;
   return ret[0];
}

NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
   __m128i b = __lsx_vmskltz_d(a);
   v2i64 ret = (v2i64)b;
   return ret[0];
}

// expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    r.val[0] = __lsx_vsllwil_hu_bu(data, 0);
    r.val[1] = __lsx_vexth_hu_bu(data);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    r.val[0] = __lsx_vsllwil_wu_hu(data, 0);
    r.val[1] = __lsx_vexth_wu_hu(data);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return __lsx_vpickev_b(__lsx_vsat_h(b, 7),__lsx_vsat_h(a, 7));
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    __m128i ab = __lsx_vpickev_h(__lsx_vsat_w(b, 15), __lsx_vsat_w(a, 15));
    __m128i cd = __lsx_vpickev_h(__lsx_vsat_w(d, 15), __lsx_vsat_w(c, 15));
    return npyv_pack_b8_b16(ab, cd);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __m128i ab = __lsx_vpickev_h(__lsx_vsat_w(b, 15), __lsx_vsat_w(a, 15));
    __m128i cd = __lsx_vpickev_h(__lsx_vsat_w(d, 15), __lsx_vsat_w(c, 15));
    __m128i ef = __lsx_vpickev_h(__lsx_vsat_w(f, 15), __lsx_vsat_w(e, 15));
    __m128i gh = __lsx_vpickev_h(__lsx_vsat_w(h, 15), __lsx_vsat_w(g, 15));
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 __lsx_vftintrne_w_s
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    return __lsx_vftintrne_w_d(b, a);
}
#endif // _NPY_SIMD_LSX_CVT_H
