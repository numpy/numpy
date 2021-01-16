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
 * Integer Division
 ***************************/
/***
 * TODO: Add support for VSX4(Power10)
 */
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const npyv_u8 mergeo_perm = {
        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
    };
    // high part of unsigned multiplication
    npyv_u16 mul_even = vec_mule(a, divisor.val[0]);
    npyv_u16 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_u8  mulhi    = (npyv_u8)vec_perm(mul_even, mul_odd, mergeo_perm);
    // floor(a/d)     = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u8 q         = vec_sub(a, mulhi);
            q         = vec_sr(q, divisor.val[1]);
            q         = vec_add(mulhi, q);
            q         = vec_sr(q, divisor.val[2]);
    return  q;
}
// divide each signed 8-bit element by a precomputed divisor
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const npyv_u8 mergeo_perm = {
        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
    };
    // high part of signed multiplication
    npyv_s16 mul_even = vec_mule(a, divisor.val[0]);
    npyv_s16 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_s8  mulhi    = (npyv_s8)vec_perm(mul_even, mul_odd, mergeo_perm);
    // q              = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)     = (q ^ dsign) - dsign
    npyv_s8 q         = vec_sra(vec_add(a, mulhi), (npyv_u8)divisor.val[1]);
            q         = vec_sub(q, vec_sra(a, npyv_setall_u8(7)));
            q         = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    const npyv_u8 mergeo_perm = {
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    };
    // high part of unsigned multiplication
    npyv_u32 mul_even = vec_mule(a, divisor.val[0]);
    npyv_u32 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_u16 mulhi    = (npyv_u16)vec_perm(mul_even, mul_odd, mergeo_perm);
    // floor(a/d)     = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    npyv_u16 q        = vec_sub(a, mulhi);
             q        = vec_sr(q, divisor.val[1]);
             q        = vec_add(mulhi, q);
             q        = vec_sr(q, divisor.val[2]);
    return   q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    const npyv_u8 mergeo_perm = {
        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
    };
    // high part of signed multiplication
    npyv_s32 mul_even = vec_mule(a, divisor.val[0]);
    npyv_s32 mul_odd  = vec_mulo(a, divisor.val[0]);
    npyv_s16 mulhi    = (npyv_s16)vec_perm(mul_even, mul_odd, mergeo_perm);
    // q              = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)     = (q ^ dsign) - dsign
    npyv_s16 q        = vec_sra(vec_add(a, mulhi), (npyv_u16)divisor.val[1]);
             q        = vec_sub(q, vec_sra(a, npyv_setall_u16(15)));
             q        = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
    return   q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
#if defined(__GNUC__) && __GNUC__ < 8
    // Doubleword integer wide multiplication supported by GCC 8+
    npyv_u64 mul_even, mul_odd;
    __asm__ ("vmulouw %0,%1,%2" : "=v" (mul_even) : "v" (a), "v" (divisor.val[0]));
    __asm__ ("vmuleuw %0,%1,%2" : "=v" (mul_odd)  : "v" (a), "v" (divisor.val[0]));
#else
    // Doubleword integer wide multiplication supported by GCC 8+
    npyv_u64 mul_even = vec_mule(a, divisor.val[0]);
    npyv_u64 mul_odd  = vec_mulo(a, divisor.val[0]);
#endif
    // high part of unsigned multiplication
    npyv_u32 mulhi    = vec_mergeo((npyv_u32)mul_even, (npyv_u32)mul_odd);
    // floor(x/d)     = (((a-mulhi) >> sh1) + mulhi) >> sh2
    npyv_u32 q        = vec_sub(a, mulhi);
             q        = vec_sr(q, divisor.val[1]);
             q        = vec_add(mulhi, q);
             q        = vec_sr(q, divisor.val[2]);
    return   q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
#if defined(__GNUC__) && __GNUC__ < 8
    // Doubleword integer wide multiplication supported by GCC8+
    npyv_s64 mul_even, mul_odd;
    __asm__ ("vmulosw %0,%1,%2" : "=v" (mul_even) : "v" (a), "v" (divisor.val[0]));
    __asm__ ("vmulesw %0,%1,%2" : "=v" (mul_odd)  : "v" (a), "v" (divisor.val[0]));
#else
    // Doubleword integer wide multiplication supported by GCC8+
    npyv_s64 mul_even = vec_mule(a, divisor.val[0]);
    npyv_s64 mul_odd  = vec_mulo(a, divisor.val[0]);
#endif
    // high part of signed multiplication
    npyv_s32 mulhi    = vec_mergeo((npyv_s32)mul_even, (npyv_s32)mul_odd);
    // q              = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)     = (q ^ dsign) - dsign
    npyv_s32 q        = vec_sra(vec_add(a, mulhi), (npyv_u32)divisor.val[1]);
             q        = vec_sub(q, vec_sra(a, npyv_setall_u32(31)));
             q        = vec_sub(vec_xor(q, divisor.val[2]), divisor.val[2]);
    return   q;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    const npy_uint64 d = vec_extract(divisor.val[0], 0);
    return npyv_set_u64(vec_extract(a, 0) / d, vec_extract(a, 1) / d);
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    npyv_b64 overflow = vec_and(vec_cmpeq(a, npyv_setall_s64(-1LL << 63)), (npyv_b64)divisor.val[1]);
    npyv_s64 d = vec_sel(divisor.val[0], npyv_setall_s64(1), overflow);
    return vec_div(a, d);
}
/***************************
 * Division
 ***************************/
#define npyv_div_f32 vec_div
#define npyv_div_f64 vec_div

/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f32 vec_madd
#define npyv_muladd_f64 vec_madd
// multiply and subtract, a*b - c
#define npyv_mulsub_f32 vec_msub
#define npyv_mulsub_f64 vec_msub
// negate multiply and add, -(a*b) + c
#define npyv_nmuladd_f32 vec_nmsub // equivalent to -(a*b - c)
#define npyv_nmuladd_f64 vec_nmsub
// negate multiply and subtract, -(a*b) - c
#define npyv_nmulsub_f32 vec_nmadd // equivalent to -(a*b + c)
#define npyv_nmulsub_f64 vec_nmadd

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    return vec_extract(vec_add(a, vec_mergel(a, a)), 0);
}

NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    const npyv_u32 rs = vec_add(a, vec_sld(a, a, 8));
    return vec_extract(vec_add(rs, vec_sld(rs, rs, 4)), 0);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    npyv_f32 sum = vec_add(a, npyv_combineh_f32(a, a));
    return vec_extract(sum, 0) + vec_extract(sum, 1);
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    return vec_extract(a, 0) + vec_extract(a, 1);
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    const npyv_u32 zero = npyv_zero_u32();
    npyv_u32 four = vec_sum4s(a, zero);
    npyv_s32 one  = vec_sums((npyv_s32)four, (npyv_s32)zero);
    return (npy_uint16)vec_extract(one, 3);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    const npyv_s32 zero = npyv_zero_s32();
    npyv_u32x2 eight = npyv_expand_u32_u16(a);
    npyv_u32   four  = vec_add(eight.val[0], eight.val[1]);
    npyv_s32   one   = vec_sums((npyv_s32)four, zero);
    return (npy_uint32)vec_extract(one, 3);
}

#endif // _NPY_SIMD_VSX_ARITHMETIC_H
