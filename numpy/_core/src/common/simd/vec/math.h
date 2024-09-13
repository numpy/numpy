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
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
    {
        npyv_b32 nn_a = npyv_notnan_f32(a);
        npyv_b32 nn_b = npyv_notnan_f32(b);
        npyv_f32 max = vec_max(a, b);
        return vec_sel(b, vec_sel(a, max, nn_a), nn_b);
    }
#endif
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    npyv_b64 nn_a = npyv_notnan_f64(a);
    npyv_b64 nn_b = npyv_notnan_f64(b);
    npyv_f64 max = vec_max(a, b);
    return vec_sel(b, vec_sel(a, max, nn_a), nn_b);
}

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
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
    {
        npyv_b32 nn_a = npyv_notnan_f32(a);
        npyv_b32 nn_b = npyv_notnan_f32(b);
        npyv_f32 min = vec_min(a, b);
        return vec_sel(b, vec_sel(a, min, nn_a), nn_b);
    }
#endif
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    npyv_b64 nn_a = npyv_notnan_f64(a);
    npyv_b64 nn_b = npyv_notnan_f64(b);
    npyv_f64 min = vec_min(a, b);
    return vec_sel(b, vec_sel(a, min, nn_a), nn_b);
}

// Minimum, integer operations
#define npyv_min_u8 vec_min
#define npyv_min_s8 vec_min
#define npyv_min_u16 vec_min
#define npyv_min_s16 vec_min
#define npyv_min_u32 vec_min
#define npyv_min_s32 vec_min
#define npyv_min_u64 vec_min
#define npyv_min_s64 vec_min

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                                   \
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 4));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 2));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 1));               \
        return (npy_##STYPE)vec_extract(r, 0);                          \
    }
NPY_IMPL_VEC_REDUCE_MINMAX(min, uint8, u8)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint8, u8)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int8, s8)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int8, s8)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                                   \
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 4));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 2));               \
        return (npy_##STYPE)vec_extract(r, 0);                          \
    }
NPY_IMPL_VEC_REDUCE_MINMAX(min, uint16, u16)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint16, u16)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int16, s16)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int16, s16)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                                   \
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \
                   r = vec_##INTRIN(r, vec_sld(r, r, 4));               \
        return (npy_##STYPE)vec_extract(r, 0);                          \
    }
NPY_IMPL_VEC_REDUCE_MINMAX(min, uint32, u32)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint32, u32)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int32, s32)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int32, s32)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, STYPE, SFX)                  \
    NPY_FINLINE npy_##STYPE npyv_reduce_##INTRIN##_##SFX(npyv_##SFX a)  \
    {                                                                   \
        npyv_##SFX r = vec_##INTRIN(a, vec_sld(a, a, 8));               \
        return (npy_##STYPE)vec_extract(r, 0);                          \
    	(void)r;					 	                                \
    }
NPY_IMPL_VEC_REDUCE_MINMAX(min, uint64, u64)
NPY_IMPL_VEC_REDUCE_MINMAX(max, uint64, u64)
NPY_IMPL_VEC_REDUCE_MINMAX(min, int64, s64)
NPY_IMPL_VEC_REDUCE_MINMAX(max, int64, s64)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

#if NPY_SIMD_F32
    #define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, INF)                   \
        NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)      \
        {                                                             \
            npyv_f32 r = vec_##INTRIN(a, vec_sld(a, a, 8));           \
                     r = vec_##INTRIN(r, vec_sld(r, r, 4));           \
            return vec_extract(r, 0);                                 \
        }                                                             \
        NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)     \
        {                                                             \
            return npyv_reduce_##INTRIN##_f32(a);                     \
        }                                                             \
        NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)     \
        {                                                             \
            npyv_b32 notnan = npyv_notnan_f32(a);                     \
            if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                \
                const union { npy_uint32 i; float f;}                 \
                pnan = {0x7fc00000UL};                                \
                return pnan.f;                                        \
            }                                                         \
            return npyv_reduce_##INTRIN##_f32(a);                     \
        }
    NPY_IMPL_VEC_REDUCE_MINMAX(min, 0x7f800000)
    NPY_IMPL_VEC_REDUCE_MINMAX(max, 0xff800000)
    #undef NPY_IMPL_VEC_REDUCE_MINMAX
#endif // NPY_SIMD_F32

#define NPY_IMPL_VEC_REDUCE_MINMAX(INTRIN, INF)                   \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)     \
    {                                                             \
        npyv_f64 r = vec_##INTRIN(a, vec_sld(a, a, 8));           \
        return vec_extract(r, 0);                                 \
        (void)r;                                                  \
    }                                                             \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)    \
    {                                                             \
        npyv_b64 notnan = npyv_notnan_f64(a);                     \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                \
            const union { npy_uint64 i; double f;}                \
            pnan = {0x7ff8000000000000ull};                       \
            return pnan.f;                                        \
        }                                                         \
        return npyv_reduce_##INTRIN##_f64(a);                     \
    }
NPY_IMPL_VEC_REDUCE_MINMAX(min, 0x7ff0000000000000)
NPY_IMPL_VEC_REDUCE_MINMAX(max, 0xfff0000000000000)
#undef NPY_IMPL_VEC_REDUCE_MINMAX

#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_reduce_minp_f64 npyv_reduce_min_f64
    #define npyv_reduce_maxp_f64 npyv_reduce_max_f64
#else
    NPY_FINLINE double npyv_reduce_minp_f64(npyv_f64 a)
    {
        npyv_b64 notnan = npyv_notnan_f64(a);
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {
            return vec_extract(a, 0);
        }
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(
                    npyv_setall_u64(0x7ff0000000000000)));
        return npyv_reduce_min_f64(a);
    }
    NPY_FINLINE double npyv_reduce_maxp_f64(npyv_f64 a)
    {
        npyv_b64 notnan = npyv_notnan_f64(a);
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) {
            return vec_extract(a, 0);
        }
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(
                    npyv_setall_u64(0xfff0000000000000)));
        return npyv_reduce_max_f64(a);
    }
#endif
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
