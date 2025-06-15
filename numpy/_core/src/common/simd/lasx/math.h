#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_MATH_H
#define _NPY_SIMD_LSX_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 __lasx_xvfsqrt_s
#define npyv_sqrt_f64 __lasx_xvfsqrt_d

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return __lasx_xvfrecip_s(a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return __lasx_xvfrecip_d(a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
  return (npyv_f32)__lasx_xvbitclri_w(a, 0x1F);
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
  return (npyv_f64)__lasx_xvbitclri_d(a, 0x3F);
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return __lasx_xvfmul_s(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return __lasx_xvfmul_d(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 __lasx_xvfmax_s
#define npyv_max_f64 __lasx_xvfmax_d
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
  return __lasx_xvfmax_s(a, b);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
  return __lasx_xvfmax_d(a, b);
}
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
  __m256i mask = __lasx_xvand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
  __m256 max   = __lasx_xvfmax_s(a, b);
  return npyv_select_f32(mask, max, (__m256){NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
  __m256i mask = __lasx_xvand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
  __m256d max  = __lasx_xvfmax_d(a, b);
  return npyv_select_f64(mask, max, (__m256d){NAN, NAN, NAN, NAN});
}

// Maximum, integer operations
#define npyv_max_u8  __lasx_xvmax_bu
#define npyv_max_s8  __lasx_xvmax_b
#define npyv_max_u16 __lasx_xvmax_hu
#define npyv_max_s16 __lasx_xvmax_h
#define npyv_max_u32 __lasx_xvmax_wu
#define npyv_max_s32 __lasx_xvmax_w
#define npyv_max_u64 __lasx_xvmax_du
#define npyv_max_s64 __lasx_xvmax_d

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 __lasx_xvfmin_s
#define npyv_min_f64 __lasx_xvfmin_d

// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
  return __lasx_xvfmin_s(a, b);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
  return __lasx_xvfmin_d(a, b);
}
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
  __m256i mask = __lasx_xvand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
  __m256 min   = __lasx_xvfmin_s(a, b);
  return npyv_select_f32(mask, min, (__m256){NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
  __m256i mask = __lasx_xvand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
  __m256d min  = __lasx_xvfmin_d(a, b);
  return npyv_select_f64(mask, min, (__m256d){NAN, NAN, NAN, NAN});
}

// Minimum, integer operations
#define npyv_min_u8  __lasx_xvmin_bu
#define npyv_min_s8  __lasx_xvmin_b
#define npyv_min_u16 __lasx_xvmin_hu
#define npyv_min_s16 __lasx_xvmin_h
#define npyv_min_u32 __lasx_xvmin_wu
#define npyv_min_s32 __lasx_xvmin_w
#define npyv_min_u64 __lasx_xvmin_du
#define npyv_min_s64 __lasx_xvmin_d

// reduce min&max for ps & pd
#define NPY_IMPL_LASX_REDUCE_MINMAX(INTRIN, INF, INF64)                                                                 \
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                                            \
    {                                                                                                                   \
        __m256i vector2 = {0, 0, 0, 0};                                                                                 \
        v8i32 index1 = {2, 3, 0, 0, 2, 3, 0, 0};                                                                        \
        v8i32 index2 = {1, 0, 0, 0, 1, 0, 0, 0};                                                                        \
        __m256 v64 = __lasx_xvf##INTRIN##_s(a, (__m256)__lasx_xvshuf_w((__m256i)index1, (__m256i)vector2, (__m256i)a));     \
        __m256 v32 = __lasx_xvf##INTRIN##_s(v64, (__m256)__lasx_xvshuf_w((__m256i)index2, (__m256i)vector2, (__m256i)v64)); \
        __m256i t1 = (__m256i)v32;                                                                                      \
        t1 = __lasx_xvpermi_q(t1, (__m256i)v32, 0x01);                                                                  \
        v32 = __lasx_xvf##INTRIN##_s(v32, (__m256)t1);                                                                  \
        return v32[0];                                                                                                  \
    }                                                                                                                   \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                                                           \
    {                                                                                                                   \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                                           \
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                                                                      \
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};                                                \
            return pnan.f;                                                                                              \
        }                                                                                                               \
        return npyv_reduce_##INTRIN##_f32(a);                                                                           \
    }                                                                                                                   \
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                                           \
    {                                                                                                                   \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                                           \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) {                                                                      \
            return a[0];                                                                                                \
        }                                                                                                               \
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));                                 \
        return npyv_reduce_##INTRIN##_f32(a);                                                                           \
    }                                                                                                                   \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                                           \
    {                                                                                                                   \
        __m256i index2 = {1, 0, 1, 0};                                                                                  \
        __m256d v64 = __lasx_xvf##INTRIN##_d(a, (__m256d)__lasx_xvshuf_d(index2, (__m256i){0, 0}, (__m256i)a));         \
        __m256i t1 = (__m256i)v64;                                                                                      \
        t1 = __lasx_xvpermi_q(t1, (__m256i)v64, 0x01);                                                                  \
        v64 = __lasx_xvf##INTRIN##_d(v64, (__m256d)t1);                                                                 \
        return (double)v64[0];                                                                                          \
    }                                                                                                                   \
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                                          \
    {                                                                                                                   \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                                           \
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {                                                                      \
            return a[0];                                                                                                \
        }                                                                                                               \
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));                               \
        return npyv_reduce_##INTRIN##_f64(a);                                                                           \
    }                                                                                                                   \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)                                                          \
    {                                                                                                                   \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                                           \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                                                                      \
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};                                      \
            return pnan.d;                                                                                              \
        }                                                                                                               \
        return npyv_reduce_##INTRIN##_f64(a);                                                                           \
    }

NPY_IMPL_LASX_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_LASX_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
#undef NPY_IMPL_LASX_REDUCE_MINMAX

// reduce min&max for 8&16&32&64-bits
#define NPY_IMPL_LASX_REDUCE_MINMAX(STYPE, INTRIN, TFLAG)                                            \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m256i a)                                        \
    {                                                                                                \
        __m256i vector2 = {0, 0, 0, 0};                                                              \
        v8i32 index1 = {2, 3, 0, 0, 2, 3, 0, 0};                                                     \
        __m256i v64 = npyv_##INTRIN##64(a, __lasx_xvshuf_w((__m256i)index1, (__m256i)vector2, a));   \
        __m256i t1 = v64;                                                                            \
        t1 = __lasx_xvpermi_q(t1, (__m256i)v64, 0x01);                                               \
        v64 = npyv_##INTRIN##64(v64, (__m256i)t1);                                                   \
        return (STYPE##64)__lasx_xvpickve2gr_d##TFLAG(v64, 0);                                       \
    }                                                                                                \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m256i a)                                        \
    {                                                                                                \
        __m256i vector2 = {0, 0, 0, 0};                                                              \
        v8i32 index1 = {2, 3, 0, 0, 2, 3, 0, 0};                                                     \
        v8i32 index2 = {1, 0, 0, 0, 1, 0, 0, 0};                                                     \
        __m256i v64 = npyv_##INTRIN##32(a, __lasx_xvshuf_w((__m256i)index1, (__m256i)vector2, a));     \
        __m256i v32 = npyv_##INTRIN##32(v64, __lasx_xvshuf_w((__m256i)index2, (__m256i)vector2, v64)); \
        __m256i t1 = v32;                                                                            \
        t1 = __lasx_xvpermi_q(t1, (__m256i)v32, 0x01);                                               \
        v32 = npyv_##INTRIN##32(v32, (__m256i)t1);                                                   \
        return (STYPE##32)__lasx_xvpickve2gr_w##TFLAG(v32, 0);                                       \
    }                                                                                                \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m256i a)                                        \
    {                                                                                                \
        __m256i vector2 = {0, 0, 0, 0};                                                              \
        v8i32 index1 = {2, 3, 0, 0, 2, 3, 0, 0};                                                     \
        v8i32 index2 = {1, 0, 0, 0, 1, 0, 0, 0};                                                     \
        v16i16 index3 = {1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};                            \
        __m256i v64 = npyv_##INTRIN##16(a, __lasx_xvshuf_w((__m256i)index1, (__m256i)vector2, a));     \
        __m256i v32 = npyv_##INTRIN##16(v64, __lasx_xvshuf_w((__m256i)index2, (__m256i)vector2, v64)); \
        __m256i v16 = npyv_##INTRIN##16(v32, __lasx_xvshuf_h((__m256i)index3, (__m256i)vector2, v32)); \
        __m256i t1 = v16;                                                                            \
        t1 = __lasx_xvpermi_q(t1, (__m256i)v16, 0x01);                                               \
        v16 = npyv_##INTRIN##16(v16, (__m256i)t1);                                                   \
        __m128i t2 = __lsx_vreplgr2vr_w(__lasx_xvpickve2gr_w##TFLAG(v16, 0));                        \
        return (STYPE##16)__lsx_vpickve2gr_h##TFLAG(t2, 0);                                          \
    }                                                                                                \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m256i a)                                          \
    {                                                                                                \
        __m256i val =npyv_##INTRIN##8((__m256i)a, __lasx_xvbsrl_v(a, 8));                            \
        val = npyv_##INTRIN##8(val, __lasx_xvbsrl_v(val, 4));                                        \
        val = npyv_##INTRIN##8(val, __lasx_xvbsrl_v(val, 2));                                        \
        val = npyv_##INTRIN##8(val, __lasx_xvbsrl_v(val, 1));                                        \
        __m256i t1 = val;                                                                            \
        t1 = __lasx_xvpermi_q(t1, (__m256i)val, 0x01);                                               \
        val = npyv_##INTRIN##8(val, (__m256i)t1);                                                    \
        __m128i t2 = __lsx_vreplgr2vr_w(__lasx_xvpickve2gr_w##TFLAG(val, 0));                        \
        return (STYPE##8)__lsx_vpickve2gr_b##TFLAG(t2, 0);                                        \
    }
NPY_IMPL_LASX_REDUCE_MINMAX(npy_uint, min_u, u)
NPY_IMPL_LASX_REDUCE_MINMAX(npy_int,  min_s,)
NPY_IMPL_LASX_REDUCE_MINMAX(npy_uint, max_u, u)
NPY_IMPL_LASX_REDUCE_MINMAX(npy_int,  max_s,)
#undef NPY_IMPL_LASX_REDUCE_MINMAX

// round to nearest integer even
#define npyv_rint_f32 (__m256)__lasx_xvfrintrne_s
#define npyv_rint_f64 (__m256d)__lasx_xvfrintrne_d
// ceil
#define npyv_ceil_f32 (__m256)__lasx_xvfrintrp_s
#define npyv_ceil_f64 (__m256d)__lasx_xvfrintrp_d

// trunc
#define npyv_trunc_f32 (__m256)__lasx_xvfrintrz_s
#define npyv_trunc_f64 (__m256d)__lasx_xvfrintrz_d

// floor
#define npyv_floor_f32 (__m256)__lasx_xvfrintrm_s
#define npyv_floor_f64 (__m256d)__lasx_xvfrintrm_d

#endif
