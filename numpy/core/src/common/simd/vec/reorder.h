#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_REORDER_H
#define _NPY_SIMD_VEC_REORDER_H

// combine lower part of two vectors
#define npyv__combinel(A, B) vec_mergeh((npyv_u64)(A), (npyv_u64)(B))
#define npyv_combinel_u8(A, B)  ((npyv_u8) npyv__combinel(A, B))
#define npyv_combinel_s8(A, B)  ((npyv_s8) npyv__combinel(A, B))
#define npyv_combinel_u16(A, B) ((npyv_u16)npyv__combinel(A, B))
#define npyv_combinel_s16(A, B) ((npyv_s16)npyv__combinel(A, B))
#define npyv_combinel_u32(A, B) ((npyv_u32)npyv__combinel(A, B))
#define npyv_combinel_s32(A, B) ((npyv_s32)npyv__combinel(A, B))
#define npyv_combinel_u64       vec_mergeh
#define npyv_combinel_s64       vec_mergeh
#if NPY_SIMD_F32
    #define npyv_combinel_f32(A, B) ((npyv_f32)npyv__combinel(A, B))
#endif
#define npyv_combinel_f64       vec_mergeh

// combine higher part of two vectors
#define npyv__combineh(A, B) vec_mergel((npyv_u64)(A), (npyv_u64)(B))
#define npyv_combineh_u8(A, B)  ((npyv_u8) npyv__combineh(A, B))
#define npyv_combineh_s8(A, B)  ((npyv_s8) npyv__combineh(A, B))
#define npyv_combineh_u16(A, B) ((npyv_u16)npyv__combineh(A, B))
#define npyv_combineh_s16(A, B) ((npyv_s16)npyv__combineh(A, B))
#define npyv_combineh_u32(A, B) ((npyv_u32)npyv__combineh(A, B))
#define npyv_combineh_s32(A, B) ((npyv_s32)npyv__combineh(A, B))
#define npyv_combineh_u64       vec_mergel
#define npyv_combineh_s64       vec_mergel
#if NPY_SIMD_F32
    #define npyv_combineh_f32(A, B) ((npyv_f32)npyv__combineh(A, B))
#endif
#define npyv_combineh_f64       vec_mergel

/*
 * combine: combine two vectors from lower and higher parts of two other vectors
 * zip: interleave two vectors
*/
#define NPYV_IMPL_VEC_COMBINE_ZIP(T_VEC, SFX)                  \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        return r;                                              \
    }                                                          \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)     \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = vec_mergeh(a, b);                           \
        r.val[1] = vec_mergel(a, b);                           \
        return r;                                              \
    }

NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u8,  u8)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s8,  s8)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u16, u16)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s16, s16)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u32, u32)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s32, s32)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u64, u64)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s64, s64)
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_COMBINE_ZIP(npyv_f32, f32)
#endif
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_f64, f64)

// deinterleave two vectors
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
    const npyv_u8 idx_even = npyv_set_u8(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    );
    const npyv_u8 idx_odd = npyv_set_u8(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
    );
    npyv_u8x2 r;
    r.val[0] = vec_perm(ab0, ab1, idx_even);
    r.val[1] = vec_perm(ab0, ab1, idx_odd);
    return r;
}
NPY_FINLINE npyv_s8x2 npyv_unzip_s8(npyv_s8 ab0, npyv_s8 ab1)
{
    npyv_u8x2 ru = npyv_unzip_u8((npyv_u8)ab0, (npyv_u8)ab1);
    npyv_s8x2 r;
    r.val[0] = (npyv_s8)ru.val[0];
    r.val[1] = (npyv_s8)ru.val[1];
    return r;
}
NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
    const npyv_u8 idx_even = npyv_set_u8(
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29
    );
    const npyv_u8 idx_odd = npyv_set_u8(
        2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31
    );
    npyv_u16x2 r;
    r.val[0] = vec_perm(ab0, ab1, idx_even);
    r.val[1] = vec_perm(ab0, ab1, idx_odd);
    return r;
}
NPY_FINLINE npyv_s16x2 npyv_unzip_s16(npyv_s16 ab0, npyv_s16 ab1)
{
    npyv_u16x2 ru = npyv_unzip_u16((npyv_u16)ab0, (npyv_u16)ab1);
    npyv_s16x2 r;
    r.val[0] = (npyv_s16)ru.val[0];
    r.val[1] = (npyv_s16)ru.val[1];
    return r;
}
NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    npyv_u32 m0 = vec_mergeh(ab0, ab1);
    npyv_u32 m1 = vec_mergel(ab0, ab1);
    npyv_u32 r0 = vec_mergeh(m0, m1);
    npyv_u32 r1 = vec_mergel(m0, m1);
    npyv_u32x2 r;
    r.val[0] = r0;
    r.val[1] = r1;
    return r;
}
NPY_FINLINE npyv_s32x2 npyv_unzip_s32(npyv_s32 ab0, npyv_s32 ab1)
{
    npyv_u32x2 ru = npyv_unzip_u32((npyv_u32)ab0, (npyv_u32)ab1);
    npyv_s32x2 r;
    r.val[0] = (npyv_s32)ru.val[0];
    r.val[1] = (npyv_s32)ru.val[1];
    return r;
}
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
    {
        npyv_u32x2 ru = npyv_unzip_u32((npyv_u32)ab0, (npyv_u32)ab1);
        npyv_f32x2 r;
        r.val[0] = (npyv_f32)ru.val[0];
        r.val[1] = (npyv_f32)ru.val[1];
        return r;
    }
#endif
NPY_FINLINE npyv_u64x2 npyv_unzip_u64(npyv_u64 ab0, npyv_u64 ab1)
{ return npyv_combine_u64(ab0, ab1); }
NPY_FINLINE npyv_s64x2 npyv_unzip_s64(npyv_s64 ab0, npyv_s64 ab1)
{ return npyv_combine_s64(ab0, ab1); }
NPY_FINLINE npyv_f64x2 npyv_unzip_f64(npyv_f64 ab0, npyv_f64 ab1)
{ return npyv_combine_f64(ab0, ab1); }

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
#if defined(NPY_HAVE_VSX3) && ((defined(__GNUC__) && __GNUC__ > 7) || defined(__IBMC__))
    return (npyv_u8)vec_revb((npyv_u64)a);
#elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
    npyv_u8 ret;
    __asm__ ("xxbrd %x0,%x1" : "=wa" (ret) : "wa" (a));
    return ret;
#else
    const npyv_u8 idx = npyv_set_u8(
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
    );
    return vec_perm(a, a, idx);
#endif
}
NPY_FINLINE npyv_s8 npyv_rev64_s8(npyv_s8 a)
{ return (npyv_s8)npyv_rev64_u8((npyv_u8)a); }

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    const npyv_u8 idx = npyv_set_u8(
        6, 7, 4, 5, 2, 3, 0, 1,/*64*/14, 15, 12, 13, 10, 11, 8, 9
    );
    return vec_perm(a, a, idx);
}
NPY_FINLINE npyv_s16 npyv_rev64_s16(npyv_s16 a)
{ return (npyv_s16)npyv_rev64_u16((npyv_u16)a); }

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    const npyv_u8 idx = npyv_set_u8(
        4, 5, 6, 7, 0, 1, 2, 3,/*64*/12, 13, 14, 15, 8, 9, 10, 11
    );
    return vec_perm(a, a, idx);
}
NPY_FINLINE npyv_s32 npyv_rev64_s32(npyv_s32 a)
{ return (npyv_s32)npyv_rev64_u32((npyv_u32)a); }
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
    { return (npyv_f32)npyv_rev64_u32((npyv_u32)a); }
#endif

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3)      \
    vec_perm(A, A, npyv_set_u8(                   \
        (E0<<2), (E0<<2)+1, (E0<<2)+2, (E0<<2)+3, \
        (E1<<2), (E1<<2)+1, (E1<<2)+2, (E1<<2)+3, \
        (E2<<2), (E2<<2)+1, (E2<<2)+2, (E2<<2)+3, \
        (E3<<2), (E3<<2)+1, (E3<<2)+2, (E3<<2)+3  \
    ))
#define npyv_permi128_s32 npyv_permi128_u32
#define npyv_permi128_f32 npyv_permi128_u32

#if defined(__IBMC__) || defined(vec_permi)
    #define npyv_permi128_u64(A, E0, E1) vec_permi(A, A, ((E0)<<1) | (E1))
#else
    #define npyv_permi128_u64(A, E0, E1) vec_xxpermdi(A, A, ((E0)<<1) | (E1))
#endif
#define npyv_permi128_s64 npyv_permi128_u64
#define npyv_permi128_f64 npyv_permi128_u64

#endif // _NPY_SIMD_VEC_REORDER_H
