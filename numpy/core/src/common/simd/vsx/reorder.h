#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_REORDER_H
#define _NPY_SIMD_VSX_REORDER_H

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
#define npyv_combinel_f32(A, B) ((npyv_f32)npyv__combinel(A, B))
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
#define npyv_combineh_f32(A, B) ((npyv_f32)npyv__combineh(A, B))
#define npyv_combineh_f64       vec_mergel

#if (defined(__GNUC__) && __GNUC__ >= 8) || (defined(__clang__) && __clang_major__ >= 6)
    #define npyv_shuffle_f32 vec_reve
    #define npyv_shuffle_f64 vec_reve
#else
    // missing on old versions of clang and gcc
    NPY_FINLINE npyv_f32 npyv_shuffle_f32(npyv_f32 a, npyv_f32 b, int imm8)
    {
        const npyv_u8 perm = npyv_set_u8(12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3);
        return vec_perm(a, a, perm);
    }
    #ifdef __clang__
        NPY_FINLINE npyv_f64 npyv_shuffle_f64(npyv_f64 a, npyv_f64 b, int imm8)
        { return vec_xxpermdi(a, b, 2); }
    #else
        NPY_FINLINE npyv_f64 npyv_shuffle_f64(npyv_f64 a, npyv_f64 b, int imm8)
        { __asm__("xxswapd %x0,%x1" : "=wa" (a) : "wa" (b)); return a; }
    #endif
#endif
/*
 * combine: combine two vectors from lower and higher parts of two other vectors
 * zip: interleave two vectors
*/
#define NPYV_IMPL_VSX_COMBINE_ZIP(T_VEC, SFX)                  \
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

NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u8,  u8)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s8,  s8)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u16, u16)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s16, s16)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u32, u32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s32, s32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_u64, u64)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_s64, s64)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_f32, f32)
NPYV_IMPL_VSX_COMBINE_ZIP(npyv_f64, f64)

#endif // _NPY_SIMD_VSX_REORDER_H
