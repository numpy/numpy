#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_MATH_H
#define _NPY_SIMD_VEC_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#if NPY_SIMD_F32
    #define npyv_sqrt_f32 vec_sqrt
#endif
#define npyv_sqrt_f64 vec_sqrt

// Reciprocal
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
    {
        const npyv_f32 one = npyv_setall_f32(1.0f);
        return vec_div(one, a);
    }
#endif
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{
    const npyv_f64 one = npyv_setall_f64(1.0);
    return vec_div(one, a);
}

// Absolute
#if NPY_SIMD_F32
    #define npyv_abs_f32 vec_abs
#endif
#define npyv_abs_f64 vec_abs

// Square
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
    { return vec_mul(a, a); }
#endif
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return vec_mul(a, a); }

// Maximum, natively mapping with no guarantees to handle NaN.
#if NPY_SIMD_F32
    #define npyv_max_f32 vec_max
#endif
#define npyv_max_f64 vec_max
// Maximum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#if NPY_SIMD_F32
    #define npyv_maxp_f32 vec_max
#endif
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_maxp_f64 vec_max
#else
    // vfmindb & vfmaxdb appears in zarch12
    NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
    {
        npyv_b64 nn_a = npyv_notnan_f64(a);
        npyv_b64 nn_b = npyv_notnan_f64(b);
        return vec_max(vec_sel(b, a, nn_a), vec_sel(a, b, nn_b));
    }
#endif
// Maximum, integer operations
#define npyv_max_u8 vec_max
#define npyv_max_s8 vec_max
#define npyv_max_u16 vec_max
#define npyv_max_s16 vec_max
#define npyv_max_u32 vec_max
#define npyv_max_s32 vec_max
#define npyv_max_u64 vec_max
#define npyv_max_s64 vec_max

// Minimum, natively mapping with no guarantees to handle NaN.
#if NPY_SIMD_F32
    #define npyv_min_f32 vec_min
#endif
#define npyv_min_f64 vec_min
// Minimum, supports IEEE floating-point arithmetic (IEC 60559),
// - If one of the two vectors contains NaN, the equivalent element of the other vector is set
// - Only if both corresponded elements are NaN, NaN is set.
#if NPY_SIMD_F32
    #define npyv_minp_f32 vec_min
#endif
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_minp_f64 vec_min
#else
    // vfmindb & vfmaxdb appears in zarch12
    NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
    {
        npyv_b64 nn_a = npyv_notnan_f64(a);
        npyv_b64 nn_b = npyv_notnan_f64(b);
        return vec_min(vec_sel(b, a, nn_a), vec_sel(a, b, nn_b));
    }
#endif
// Minimum, integer operations
#define npyv_min_u8 vec_min
#define npyv_min_s8 vec_min
#define npyv_min_u16 vec_min
#define npyv_min_s16 vec_min
#define npyv_min_u32 vec_min
#define npyv_min_s32 vec_min
#define npyv_min_u64 vec_min
#define npyv_min_s64 vec_min

// round to nearest int even
#define npyv_rint_f64 vec_rint
// ceil
#define npyv_ceil_f64 vec_ceil
// trunc
#define npyv_trunc_f64 vec_trunc
// floor
#define npyv_floor_f64 vec_floor
#if NPY_SIMD_F32
    #define npyv_rint_f32 vec_rint
    #define npyv_ceil_f32 vec_ceil
    #define npyv_trunc_f32 vec_trunc
    #define npyv_floor_f32 vec_floor
#endif

#endif // _NPY_SIMD_VEC_MATH_H
