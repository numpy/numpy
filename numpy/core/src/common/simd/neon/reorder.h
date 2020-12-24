#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_REORDER_H
#define _NPY_SIMD_NEON_REORDER_H

// combine lower part of two vectors
#ifdef __aarch64__
    #define npyv_combinel_u8(A, B)  vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
    #define npyv_combinel_s8(A, B)  vreinterpretq_s8_u64(vzip1q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
    #define npyv_combinel_u16(A, B) vreinterpretq_u16_u64(vzip1q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
    #define npyv_combinel_s16(A, B) vreinterpretq_s16_u64(vzip1q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
    #define npyv_combinel_u32(A, B) vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
    #define npyv_combinel_s32(A, B) vreinterpretq_s32_u64(vzip1q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
    #define npyv_combinel_u64       vzip1q_u64
    #define npyv_combinel_s64       vzip1q_s64
    #define npyv_combinel_f32(A, B) vreinterpretq_f32_u64(vzip1q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
    #define npyv_combinel_f64       vzip1q_f64
#else
    #define npyv_combinel_u8(A, B)  vcombine_u8(vget_low_u8(A), vget_low_u8(B))
    #define npyv_combinel_s8(A, B)  vcombine_s8(vget_low_s8(A), vget_low_s8(B))
    #define npyv_combinel_u16(A, B) vcombine_u16(vget_low_u16(A), vget_low_u16(B))
    #define npyv_combinel_s16(A, B) vcombine_s16(vget_low_s16(A), vget_low_s16(B))
    #define npyv_combinel_u32(A, B) vcombine_u32(vget_low_u32(A), vget_low_u32(B))
    #define npyv_combinel_s32(A, B) vcombine_s32(vget_low_s32(A), vget_low_s32(B))
    #define npyv_combinel_u64(A, B) vcombine_u64(vget_low_u64(A), vget_low_u64(B))
    #define npyv_combinel_s64(A, B) vcombine_s64(vget_low_s64(A), vget_low_s64(B))
    #define npyv_combinel_f32(A, B) vcombine_f32(vget_low_f32(A), vget_low_f32(B))
#endif

// combine higher part of two vectors
#ifdef __aarch64__
    #define npyv_combineh_u8(A, B)  vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(A), vreinterpretq_u64_u8(B)))
    #define npyv_combineh_s8(A, B)  vreinterpretq_s8_u64(vzip2q_u64(vreinterpretq_u64_s8(A), vreinterpretq_u64_s8(B)))
    #define npyv_combineh_u16(A, B) vreinterpretq_u16_u64(vzip2q_u64(vreinterpretq_u64_u16(A), vreinterpretq_u64_u16(B)))
    #define npyv_combineh_s16(A, B) vreinterpretq_s16_u64(vzip2q_u64(vreinterpretq_u64_s16(A), vreinterpretq_u64_s16(B)))
    #define npyv_combineh_u32(A, B) vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(A), vreinterpretq_u64_u32(B)))
    #define npyv_combineh_s32(A, B) vreinterpretq_s32_u64(vzip2q_u64(vreinterpretq_u64_s32(A), vreinterpretq_u64_s32(B)))
    #define npyv_combineh_u64       vzip2q_u64
    #define npyv_combineh_s64       vzip2q_s64
    #define npyv_combineh_f32(A, B) vreinterpretq_f32_u64(vzip2q_u64(vreinterpretq_u64_f32(A), vreinterpretq_u64_f32(B)))
    #define npyv_combineh_f64       vzip2q_f64
#else
    #define npyv_combineh_u8(A, B)  vcombine_u8(vget_high_u8(A), vget_high_u8(B))
    #define npyv_combineh_s8(A, B)  vcombine_s8(vget_high_s8(A), vget_high_s8(B))
    #define npyv_combineh_u16(A, B) vcombine_u16(vget_high_u16(A), vget_high_u16(B))
    #define npyv_combineh_s16(A, B) vcombine_s16(vget_high_s16(A), vget_high_s16(B))
    #define npyv_combineh_u32(A, B) vcombine_u32(vget_high_u32(A), vget_high_u32(B))
    #define npyv_combineh_s32(A, B) vcombine_s32(vget_high_s32(A), vget_high_s32(B))
    #define npyv_combineh_u64(A, B) vcombine_u64(vget_high_u64(A), vget_high_u64(B))
    #define npyv_combineh_s64(A, B) vcombine_s64(vget_high_s64(A), vget_high_s64(B))
    #define npyv_combineh_f32(A, B) vcombine_f32(vget_high_f32(A), vget_high_f32(B))
#endif

// combine two vectors from lower and higher parts of two other vectors
#define NPYV_IMPL_NEON_COMBINE(T_VEC, SFX)                     \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        return r;                                              \
    }

NPYV_IMPL_NEON_COMBINE(npyv_u8,  u8)
NPYV_IMPL_NEON_COMBINE(npyv_s8,  s8)
NPYV_IMPL_NEON_COMBINE(npyv_u16, u16)
NPYV_IMPL_NEON_COMBINE(npyv_s16, s16)
NPYV_IMPL_NEON_COMBINE(npyv_u32, u32)
NPYV_IMPL_NEON_COMBINE(npyv_s32, s32)
NPYV_IMPL_NEON_COMBINE(npyv_u64, u64)
NPYV_IMPL_NEON_COMBINE(npyv_s64, s64)
NPYV_IMPL_NEON_COMBINE(npyv_f32, f32)
#ifdef __aarch64__
NPYV_IMPL_NEON_COMBINE(npyv_f64, f64)
#endif

// interleave two vectors
#define NPYV_IMPL_NEON_ZIP(T_VEC, SFX)                       \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)   \
    {                                                        \
        T_VEC##x2 r;                                         \
        r.val[0] = vzip1q_##SFX(a, b);                       \
        r.val[1] = vzip2q_##SFX(a, b);                       \
        return r;                                            \
    }

#ifdef __aarch64__
    NPYV_IMPL_NEON_ZIP(npyv_u8,  u8)
    NPYV_IMPL_NEON_ZIP(npyv_s8,  s8)
    NPYV_IMPL_NEON_ZIP(npyv_u16, u16)
    NPYV_IMPL_NEON_ZIP(npyv_s16, s16)
    NPYV_IMPL_NEON_ZIP(npyv_u32, u32)
    NPYV_IMPL_NEON_ZIP(npyv_s32, s32)
    NPYV_IMPL_NEON_ZIP(npyv_f32, f32)
    NPYV_IMPL_NEON_ZIP(npyv_f64, f64)
#else
    #define npyv_zip_u8  vzipq_u8
    #define npyv_zip_s8  vzipq_s8
    #define npyv_zip_u16 vzipq_u16
    #define npyv_zip_s16 vzipq_s16
    #define npyv_zip_u32 vzipq_u32
    #define npyv_zip_s32 vzipq_s32
    #define npyv_zip_f32 vzipq_f32
#endif
#define npyv_zip_u64 npyv_combine_u64
#define npyv_zip_s64 npyv_combine_s64

// Reverse elements of each 64-bit lane
#define npyv_rev64_u8  vrev64q_u8
#define npyv_rev64_s8  vrev64q_s8
#define npyv_rev64_u16 vrev64q_u16
#define npyv_rev64_s16 vrev64q_s16
#define npyv_rev64_u32 vrev64q_u32
#define npyv_rev64_s32 vrev64q_s32
#define npyv_rev64_f32 vrev64q_f32

#endif // _NPY_SIMD_NEON_REORDER_H
