#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_CVT_H
#define _NPY_SIMD_SVE_CVT_H

#define NPYV_IMPL_SVE_CVT_TO_B(S, W)                            \
    NPY_FINLINE npyv_b##W npyv_cvt_b##W##_##S##W(npyv_##S##W a) \
    {                                                           \
        return svcmpne_n_##S##W(svptrue_b##W(), a, 0);          \
    }

// convert mask to integer vectors
#define npyv_cvt_u8_b8(b)   svreinterpret_u8_s8(svdup_s8_z(b,  -1))
#define npyv_cvt_u16_b16(b) svreinterpret_u16_s16(svdup_s16_z(b, -1))
#define npyv_cvt_u32_b32(b) svreinterpret_u32_s32(svdup_s32_z(b, -1))
#define npyv_cvt_u64_b64(b) svreinterpret_u64_s64(svdup_s64_z(b, -1))
#define npyv_cvt_s8_b8(b)   svdup_s8_z(b, -1)
#define npyv_cvt_s16_b16(b) svdup_s16_z(b, -1)
#define npyv_cvt_s32_b32(b) svdup_s32_z(b, -1)
#define npyv_cvt_s64_b64(b) svdup_s64_z(b, -1)
#define npyv_cvt_f32_b32(b) svdup_f32_z(b, 1.0)
#define npyv_cvt_f64_b64(b) svdup_f64_z(b, 1.0)

// convert integer vectors to mask
NPYV_IMPL_SVE_CVT_TO_B(u, 8)  
NPYV_IMPL_SVE_CVT_TO_B(u, 16)  
NPYV_IMPL_SVE_CVT_TO_B(u, 32)  
NPYV_IMPL_SVE_CVT_TO_B(u, 64)  
NPYV_IMPL_SVE_CVT_TO_B(s, 8)  
NPYV_IMPL_SVE_CVT_TO_B(s, 16)  
NPYV_IMPL_SVE_CVT_TO_B(s, 32)  
NPYV_IMPL_SVE_CVT_TO_B(s, 64)  
NPYV_IMPL_SVE_CVT_TO_B(f, 32)  
NPYV_IMPL_SVE_CVT_TO_B(f, 64)  

// expand
NPY_FINLINE npyv_u16x2
npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;

    r.val[0] = svunpklo_u16(data);
    r.val[1] = svunpkhi_u16(data);
    return r;
}

NPY_FINLINE npyv_u32x2
npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;

    r.val[0] = svunpklo_u32(data);
    r.val[1] = svunpkhi_u32(data);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return svuzp1_b8(a, b);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    svbool_t p0 = svuzp1_b16(a, b);
    svbool_t p1 = svuzp1_b16(c, d);

    return svuzp1_b8(p0, p1);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    svbool_t p0 = svuzp1_b32(a, b);
    svbool_t p1 = svuzp1_b32(c, d);
    svbool_t p2 = svuzp1_b32(e, f);
    svbool_t p3 = svuzp1_b32(g, h);

    p0 = svuzp1_b16(p0, p1);
    p2 = svuzp1_b16(p2, p3);
    return svuzp1_b8(p0, p2);
}

NPY_FINLINE npy_uint64
npyv_tobits_b8(npyv_b8 a)
{
    const npy_uint64 cntb = svcntb();
    const npyv_u8 scale0 = svindex_u8(0, 1);
    const npyv_u8 scale1 = svand_n_u8_x(svptrue_b8(), scale0, (cntb - 1) >> 3);
    const npyv_u8 shift = svlsl_u8_x(svptrue_b8(),
            svdup_n_u8_z(svptrue_b8(), 1), scale1);
    const npyv_u8 byteOrder64 = svreinterpret_u8_u64(svindex_u64(0ul + (8ul<<8)
            + (16ul<<16) + (24ul<<24) + (32ul<<32) + (40ul<<40) + (48ul<<48)
            + (56ul<<56),
            1 + (1ul<<8) + (1ul<<16) + (1ul<<24) + (1ul<<32) + (1ul<<40)
            + (1ul<<48) + (1ul<<56)));
    const npyv_u8 byteOrder32 = svreinterpret_u8_u32(svindex_u32(0ul + (8ul<<8)
            + (16ul<<16) + (24ul<<24), 1 + (1ul<<8) + (1ul<<16) + (1ul<<24)));
    const npyv_u8 b = svdup_n_u8_z(a, 0xff);
    const npyv_u8 seq_scale = svand_u8_z(svptrue_b8(), b, shift);

    // SVE size of 512 and 256 are now implemented.
    assert(cntb == 64 || cntb == 32);

    if(cntb == 64) {
        const npyv_u8 v0 = svtbl_u8(seq_scale, byteOrder64);
        return svaddv_u64(svptrue_b8(), svreinterpret_u64_u8(v0));
    } else {
        const npyv_u8 v0 = svtbl_u8(seq_scale, byteOrder32);
        return svaddv_u32(svptrue_b8(), svreinterpret_u32_u8(v0));
    }
}

NPY_FINLINE npy_uint64
npyv_tobits_b16(npyv_b16 a)
{
    const uint64_t cntb = svcntb();

    switch (cntb) {
        case 64: {
            const npyv_b16 mask_zero = svpfalse();
            const npyv_u32 one = svdup_u32(1);
            npyv_b16 l = svzip1_b16(a, mask_zero);
            npyv_b16 h = svzip1_b16(a, mask_zero);
            npyv_u32 idx0 = svindex_u32(0, 1);
            npyv_u32 idx1 = svindex_u32(16, 1);
            idx0 = svlsl_u32_x(svptrue_b32(), one, idx0);
            idx1 = svlsl_u32_x(svptrue_b32(), one, idx1);
            uint64_t retVal = svorv_u32(l, idx0);
            retVal |= svorv_u32(h, idx1);
            return retVal;
        }
        case 32: {
            const npyv_u16 one = svdup_u16(1);
            npyv_u16 idx = svindex_u16(0, 1);
            idx = svlsl_u16_x(svptrue_b16(), one, idx);
            uint64_t retVal = svorv_u16(a, idx);
            return retVal;
        }
        default:
            assert(!"unsupported SVE size!");
    }

    return 0;
}

NPY_FINLINE npy_uint64
npyv_tobits_b32(npyv_b32 a)
{
    const npyv_u32 one = svdup_u32(1);
    const npyv_u32 idx = svindex_u32(0, 1);
    npyv_u32 v = svlsl_u32_x(svptrue_b32(), one, idx);
    return svorv_u32(a, v);
}

NPY_FINLINE npy_uint64
npyv_tobits_b64(npyv_b64 a)
{
    const npyv_u64 one = svdup_u64(1);
    const npyv_u64 idx = svindex_u64(0, 1);
    npyv_u64 v = svlsl_u64_x(svptrue_b64(), one, idx);
    return svorv_u64(a, v);
}

// round to nearest integer (assuming even)
NPY_FINLINE npyv_s32
npyv_round_s32_f32(npyv_f32 a)
{
    return svcvt_s32_f32_x(svptrue_b32(), svrinti_f32_x(svptrue_b32(), a));
}
NPY_FINLINE npyv_s32
npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    svfloat64_t a_rinti = svrinti_f64_x(svptrue_b64(), a);
    svfloat64_t b_rinti = svrinti_f64_x(svptrue_b64(), b);
    svint32_t a_cvt = svcvt_s32_f64_x(svptrue_b64(), a_rinti);
    svint32_t b_cvt = svcvt_s32_f64_x(svptrue_b64(), b_rinti);
    return svuzp1_s32(a_cvt, b_cvt);
}

#endif  // _NPY_SIMD_SVE_CVT_H
