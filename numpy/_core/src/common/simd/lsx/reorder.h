#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_REORDER_H
#define _NPY_SIMD_LSX_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8(A, B)  __lsx_vilvl_d(B, A)
#define npyv_combinel_s8(A, B)  __lsx_vilvl_d(B, A)
#define npyv_combinel_u16(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_s16(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_u32(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_s32(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_u64(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_s64(A, B) __lsx_vilvl_d(B, A)
#define npyv_combinel_f32(A, B) (__m128)(__lsx_vilvl_d((__m128i)B, (__m128i)A))
#define npyv_combinel_f64(A, B) (__m128d)(__lsx_vilvl_d((__m128i)B, (__m128i)A))

// combine higher part of two vectors
#define npyv_combineh_u8(A, B)  __lsx_vilvh_d(B, A)
#define npyv_combineh_s8(A, B)  __lsx_vilvh_d(B, A)
#define npyv_combineh_u16(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_s16(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_u32(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_s32(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_u64(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_s64(A, B) __lsx_vilvh_d(B, A)
#define npyv_combineh_f32(A, B) (__m128)(__lsx_vilvh_d((__m128i)B, (__m128i)A))
#define npyv_combineh_f64(A, B) (__m128d)(__lsx_vilvh_d((__m128i)B, (__m128i)A))

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_s64x2 npyv__combine(__m128i a, __m128i b)
{
    npyv_s64x2 r;
    r.val[0] = npyv_combinel_u8(a, b);
    r.val[1] = npyv_combineh_u8(a, b);
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m128 a, __m128 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b);
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m128d a, __m128d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b);
    r.val[1] = npyv_combineh_f64(a, b);
    return r;
}
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

// interleave two vectors
#define NPYV_IMPL_LSX_ZIP(T_VEC, SFX, INTR_SFX)               \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)    \
    {                                                         \
        T_VEC##x2 r;                                          \
        r.val[0] = __lsx_vilvl_##INTR_SFX(b, a);              \
        r.val[1] = __lsx_vilvh_##INTR_SFX(b, a);              \
        return r;                                             \
    }

NPYV_IMPL_LSX_ZIP(npyv_u8,  u8,  b)
NPYV_IMPL_LSX_ZIP(npyv_s8,  s8,  b)
NPYV_IMPL_LSX_ZIP(npyv_u16, u16, h)
NPYV_IMPL_LSX_ZIP(npyv_s16, s16, h)
NPYV_IMPL_LSX_ZIP(npyv_u32, u32, w)
NPYV_IMPL_LSX_ZIP(npyv_s32, s32, w)
NPYV_IMPL_LSX_ZIP(npyv_u64, u64, d)
NPYV_IMPL_LSX_ZIP(npyv_s64, s64, d)

NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m128 a, __m128 b)
{
    npyv_f32x2 r;
    r.val[0] = (__m128)(__lsx_vilvl_w((__m128i)b, (__m128i)a));
    r.val[1] = (__m128)(__lsx_vilvh_w((__m128i)b, (__m128i)a));
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m128d a, __m128d b)
{
    npyv_f64x2 r;
    r.val[0] = (__m128d)(__lsx_vilvl_d((__m128i)b, (__m128i)a));
    r.val[1] = (__m128d)(__lsx_vilvh_d((__m128i)b, (__m128i)a));
    return r;
}

// deinterleave two vectors
#define NPYV_IMPL_LSX_UNZIP(T_VEC, SFX, INTR_SFX)             \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b)  \
    {                                                         \
        T_VEC##x2 r;                                          \
        r.val[0] = __lsx_vpickev_##INTR_SFX(b, a);            \
        r.val[1] = __lsx_vpickod_##INTR_SFX(b, a);            \
        return r;                                             \
    }

NPYV_IMPL_LSX_UNZIP(npyv_u8,  u8,  b)
NPYV_IMPL_LSX_UNZIP(npyv_s8,  s8,  b)
NPYV_IMPL_LSX_UNZIP(npyv_u16, u16, h)
NPYV_IMPL_LSX_UNZIP(npyv_s16, s16, h)
NPYV_IMPL_LSX_UNZIP(npyv_u32, u32, w)
NPYV_IMPL_LSX_UNZIP(npyv_s32, s32, w)
NPYV_IMPL_LSX_UNZIP(npyv_u64, u64, d)
NPYV_IMPL_LSX_UNZIP(npyv_s64, s64, d)

NPY_FINLINE npyv_f32x2 npyv_unzip_f32(__m128 a, __m128 b)
{
    npyv_f32x2 r;
    r.val[0] = (__m128)(__lsx_vpickev_w((__m128i)b, (__m128i)a));
    r.val[1] = (__m128)(__lsx_vpickod_w((__m128i)b, (__m128i)a));
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_unzip_f64(__m128d a, __m128d b)
{
    npyv_f64x2 r;
    r.val[0] = (__m128d)(__lsx_vpickev_d((__m128i)b, (__m128i)a));
    r.val[1] = (__m128d)(__lsx_vpickod_d((__m128i)b, (__m128i)a));
    return r;
}

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    v16u8 idx = {7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};
    return __lsx_vshuf_b(a, a, (__m128i)idx);
}

#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    v8u16 idx = {3, 2, 1, 0, 7, 6, 5, 4};
    return __lsx_vshuf_h((__m128i)idx, a, a);
}

#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    v4u32 idx = {1, 0, 3, 2};
    return __lsx_vshuf_w((__m128i)idx, a, a);
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    v4i32 idx = {1, 0, 3, 2};
    return (v4f32)__lsx_vshuf_w((__m128i)idx, (__m128i)a, (__m128i)a);
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3)                    \
    npyv_set_u32(                                               \
       __lsx_vpickve2gr_wu(A, E0), __lsx_vpickve2gr_wu(A, E1),  \
       __lsx_vpickve2gr_wu(A, E2), __lsx_vpickve2gr_wu(A, E3)   \
    )
#define npyv_permi128_s32(A, E0, E1, E2, E3)                    \
    npyv_set_s32(                                               \
       __lsx_vpickve2gr_w(A, E0), __lsx_vpickve2gr_w(A, E1),    \
       __lsx_vpickve2gr_w(A, E2), __lsx_vpickve2gr_w(A, E3)     \
    )
#define npyv_permi128_u64(A, E0, E1)                            \
    npyv_set_u64(                                               \
       __lsx_vpickve2gr_du(A, E0), __lsx_vpickve2gr_du(A, E1)   \
    )
#define npyv_permi128_s64(A, E0, E1)                            \
    npyv_set_s64(                                               \
       __lsx_vpickve2gr_d(A, E0), __lsx_vpickve2gr_d(A, E1)     \
    )
#define npyv_permi128_f32(A, E0, E1, E2, E3)                    \
    (__m128)__lsx_vshuf_w((__m128i)(v4u32){E0, E1, E2, E3}, (__m128i)A, (__m128i)A)

#define npyv_permi128_f64(A, E0, E1)                            \
    (__m128d)__lsx_vshuf_d((__m128i){E0, E1}, (__m128i)A, (__m128i)A)
#endif // _NPY_SIMD_LSX_REORDER_H
