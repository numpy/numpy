#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_ARITHMETIC_H
#define _NPY_SIMD_VSX_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  vec_add
#define npyv_add_s8  vec_add
#define npyv_add_u16 vec_add
#define npyv_add_s16 vec_add
#define npyv_add_u32 vec_add
#define npyv_add_s32 vec_add
#define npyv_add_u64 vec_add
#define npyv_add_s64 vec_add
#define npyv_add_f32 vec_add
#define npyv_add_f64 vec_add

// saturated
#define npyv_adds_u8  vec_adds
#define npyv_adds_s8  vec_adds
#define npyv_adds_u16 vec_adds
#define npyv_adds_s16 vec_adds

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  vec_sub
#define npyv_sub_s8  vec_sub
#define npyv_sub_u16 vec_sub
#define npyv_sub_s16 vec_sub
#define npyv_sub_u32 vec_sub
#define npyv_sub_s32 vec_sub
#define npyv_sub_u64 vec_sub
#define npyv_sub_s64 vec_sub
#define npyv_sub_f32 vec_sub
#define npyv_sub_f64 vec_sub

// saturated
#define npyv_subs_u8  vec_subs
#define npyv_subs_s8  vec_subs
#define npyv_subs_u16 vec_subs
#define npyv_subs_s16 vec_subs

/***************************
 * Multiplication
 ***************************/
// non-saturated
// up to GCC 6 vec_mul only supports precisions and llong
#if defined(__GNUC__) && __GNUC__ < 7
    #define NPYV_IMPL_VSX_MUL(T_VEC, SFX, ...)              \
        NPY_FINLINE T_VEC npyv_mul_##SFX(T_VEC a, T_VEC b)  \
        {                                                   \
            const npyv_u8 ev_od = {__VA_ARGS__};            \
            return vec_perm(                                \
                (T_VEC)vec_mule(a, b),                      \
                (T_VEC)vec_mulo(a, b), ev_od                \
            );                                              \
        }

    NPYV_IMPL_VSX_MUL(npyv_u8,  u8,  0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30)
    NPYV_IMPL_VSX_MUL(npyv_s8,  s8,  0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30)
    NPYV_IMPL_VSX_MUL(npyv_u16, u16, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29)
    NPYV_IMPL_VSX_MUL(npyv_s16, s16, 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29)

    // vmuluwm can be used for unsigned or signed 32-bit integers
    #define NPYV_IMPL_VSX_MUL_32(T_VEC, SFX)                \
        NPY_FINLINE T_VEC npyv_mul_##SFX(T_VEC a, T_VEC b)  \
        {                                                   \
            T_VEC ret;                                      \
            __asm__ __volatile__(                           \
                "vmuluwm %0,%1,%2" :                        \
                "=v" (ret) : "v" (a), "v" (b)               \
            );                                              \
            return ret;                                     \
        }

    NPYV_IMPL_VSX_MUL_32(npyv_u32, u32)
    NPYV_IMPL_VSX_MUL_32(npyv_s32, s32)

#else
    #define npyv_mul_u8  vec_mul
    #define npyv_mul_s8  vec_mul
    #define npyv_mul_u16 vec_mul
    #define npyv_mul_s16 vec_mul
    #define npyv_mul_u32 vec_mul
    #define npyv_mul_s32 vec_mul
#endif
#define npyv_mul_f32 vec_mul
#define npyv_mul_f64 vec_mul

/***************************
 * Division
 ***************************/
#define npyv_div_f32 vec_div
#define npyv_div_f64 vec_div

#endif // _NPY_SIMD_VSX_ARITHMETIC_H
