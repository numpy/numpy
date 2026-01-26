#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LSX_MATH_H
#define _NPY_SIMD_LSX_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 __lsx_vfsqrt_s
#define npyv_sqrt_f64 __lsx_vfsqrt_d

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return __lsx_vfrecip_s(a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return __lsx_vfrecip_d(a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
  return (npyv_f32)__lsx_vbitclri_w(a, 0x1F);
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
  return (npyv_f64)__lsx_vbitclri_d(a, 0x3F);
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return __lsx_vfmul_s(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return __lsx_vfmul_d(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#define npyv_max_f32 __lsx_vfmax_s
#define npyv_max_f64 __lsx_vfmax_d
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
  return __lsx_vfmax_s(a, b);
}
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
  return __lsx_vfmax_d(a, b);
}
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
  __m128i mask = __lsx_vand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
  __m128 max   = __lsx_vfmax_s(a, b);
  return npyv_select_f32(mask, max, (__m128){NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
  __m128i mask = __lsx_vand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
  __m128d max  = __lsx_vfmax_d(a, b);
  return npyv_select_f64(mask, max, (__m128d){NAN, NAN});
}

// Maximum, integer operations
#define npyv_max_u8  __lsx_vmax_bu
#define npyv_max_s8  __lsx_vmax_b
#define npyv_max_u16 __lsx_vmax_hu
#define npyv_max_s16 __lsx_vmax_h
#define npyv_max_u32 __lsx_vmax_wu
#define npyv_max_s32 __lsx_vmax_w
#define npyv_max_u64 __lsx_vmax_du
#define npyv_max_s64 __lsx_vmax_d

// Minimum, natively mapping with no guarantees to handle NaN.
#define npyv_min_f32 __lsx_vfmin_s
#define npyv_min_f64 __lsx_vfmin_d

// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
  return __lsx_vfmin_s(a, b);
}
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
  return __lsx_vfmin_d(a, b);
}
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
  __m128i mask = __lsx_vand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
  __m128 min   = __lsx_vfmin_s(a, b);
  return npyv_select_f32(mask, min, (__m128){NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
  __m128i mask = __lsx_vand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
  __m128d min  = __lsx_vfmin_d(a, b);
  return npyv_select_f64(mask, min, (__m128d){NAN, NAN});
}

// Minimum, integer operations
#define npyv_min_u8  __lsx_vmin_bu
#define npyv_min_s8  __lsx_vmin_b
#define npyv_min_u16 __lsx_vmin_hu
#define npyv_min_s16 __lsx_vmin_h
#define npyv_min_u32 __lsx_vmin_wu
#define npyv_min_s32 __lsx_vmin_w
#define npyv_min_u64 __lsx_vmin_du
#define npyv_min_s64 __lsx_vmin_d

// reduce min&max for ps & pd
#define NPY_IMPL_LSX_REDUCE_MINMAX(INTRIN, INF, INF64)                                                                  \
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                                            \
    {                                                                                                                   \
        __m128i vector2 = {0, 0};                                                                                       \
        v4i32 index1 = {2, 3, 0, 0};                                                                                    \
        v4i32 index2 = {1, 0, 0, 0};                                                                                    \
        __m128 v64 = __lsx_vf##INTRIN##_s(a, (__m128)__lsx_vshuf_w((__m128i)index1, (__m128i)vector2, (__m128i)a));     \
        __m128 v32 = __lsx_vf##INTRIN##_s(v64, (__m128)__lsx_vshuf_w((__m128i)index2, (__m128i)vector2, (__m128i)v64)); \
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
        __m128i index2 = {1, 0};                                                                                        \
        __m128d v64 = __lsx_vf##INTRIN##_d(a, (__m128d)__lsx_vshuf_d(index2, (__m128i){0, 0}, (__m128i)a));             \
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

NPY_IMPL_LSX_REDUCE_MINMAX(min, 0x7f800000, 0x7ff0000000000000)
NPY_IMPL_LSX_REDUCE_MINMAX(max, 0xff800000, 0xfff0000000000000)
#undef NPY_IMPL_LSX_REDUCE_MINMAX

// reduce min&max for 8&16&32&64-bits
#define NPY_IMPL_LSX_REDUCE_MINMAX(STYPE, INTRIN, TFLAG)                                             \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m128i a)                                        \
    {                                                                                                \
        __m128i vector2 = {0, 0};                                                                    \
        v4i32 index1 = {2, 3, 0, 0};                                                                 \
        __m128i v64 = npyv_##INTRIN##64(a, __lsx_vshuf_w((__m128i)index1, (__m128i)vector2, a));     \
        return (STYPE##64)__lsx_vpickve2gr_d##TFLAG(v64, 0);                                         \
    }                                                                                                \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m128i a)                                        \
    {                                                                                                \
        __m128i vector2 = {0, 0};                                                                    \
        v4i32 index1 = {2, 3, 0, 0};                                                                 \
        v4i32 index2 = {1, 0, 0, 0};                                                                 \
        __m128i v64 = npyv_##INTRIN##32(a, __lsx_vshuf_w((__m128i)index1, (__m128i)vector2, a));     \
        __m128i v32 = npyv_##INTRIN##32(v64, __lsx_vshuf_w((__m128i)index2, (__m128i)vector2, v64)); \
        return (STYPE##32)__lsx_vpickve2gr_w##TFLAG(v32, 0);                                         \
    }                                                                                                \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m128i a)                                        \
    {                                                                                                \
        __m128i vector2 = {0, 0};                                                                    \
        v4i32 index1 = {2, 3, 0, 0};                                                                 \
        v4i32 index2 = {1, 0, 0, 0};                                                                 \
        v8i16 index3 = {1, 0, 0, 0, 4, 5, 6, 7 };                                                    \
        __m128i v64 = npyv_##INTRIN##16(a, __lsx_vshuf_w((__m128i)index1, (__m128i)vector2, a));     \
        __m128i v32 = npyv_##INTRIN##16(v64, __lsx_vshuf_w((__m128i)index2, (__m128i)vector2, v64)); \
        __m128i v16 = npyv_##INTRIN##16(v32, __lsx_vshuf_h((__m128i)index3, (__m128i)vector2, v32)); \
        return (STYPE##16)__lsx_vpickve2gr_h##TFLAG(v16, 0);                                         \
    }                                                                                                \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m128i a)                                          \
    {                                                                                                \
        __m128i val =npyv_##INTRIN##8((__m128i)a, __lsx_vbsrl_v(a, 8));                        \
        val = npyv_##INTRIN##8(val, __lsx_vbsrl_v(val, 4));                                  \
        val = npyv_##INTRIN##8(val, __lsx_vbsrl_v(val, 2));                                  \
        val = npyv_##INTRIN##8(val, __lsx_vbsrl_v(val, 1));                                  \
        return (STYPE##8)__lsx_vpickve2gr_b##TFLAG(val, 0);                                  \
    }
NPY_IMPL_LSX_REDUCE_MINMAX(npy_uint, min_u, u)
NPY_IMPL_LSX_REDUCE_MINMAX(npy_int,  min_s,)
NPY_IMPL_LSX_REDUCE_MINMAX(npy_uint, max_u, u)
NPY_IMPL_LSX_REDUCE_MINMAX(npy_int,  max_s,)
#undef NPY_IMPL_LSX_REDUCE_MINMAX

// round to nearest integer even
#define npyv_rint_f32 (__m128)__lsx_vfrintrne_s
#define npyv_rint_f64 (__m128d)__lsx_vfrintrne_d
// ceil
#define npyv_ceil_f32 (__m128)__lsx_vfrintrp_s
#define npyv_ceil_f64 (__m128d)__lsx_vfrintrp_d

// trunc
#define npyv_trunc_f32 (__m128)__lsx_vfrintrz_s
#define npyv_trunc_f64 (__m128d)__lsx_vfrintrz_d

// floor
#define npyv_floor_f32 (__m128)__lsx_vfrintrm_s
#define npyv_floor_f64 (__m128d)__lsx_vfrintrm_d

#endif
