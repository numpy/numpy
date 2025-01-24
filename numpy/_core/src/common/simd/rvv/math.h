#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_MATH_H
#define _NPY_SIMD_RVV_MATH_H

#include <float.h>

/***************************
 * Elementary
 ***************************/
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfabs_v_f32m1(a, vlen);
}

NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfabs_v_f64m1(a, vlen);
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmul_vv_f32m1(a, a, vlen);
}

NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmul_vv_f64m1(a, a, vlen);
}

// Square root
NPY_FINLINE npyv_f32 npyv_sqrt_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfsqrt_v_f32m1(a, vlen);
}

NPY_FINLINE npyv_f64 npyv_sqrt_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfsqrt_v_f64m1(a, vlen);
}

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vlen);
    return __riscv_vfdiv_vv_f32m1(one, a, vlen);
}

NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    npyv_f64 one = __riscv_vfmv_v_f_f64m1(1.0, vlen);
    return __riscv_vfdiv_vv_f64m1(one, a, vlen);
}

// Maximum
NPY_FINLINE npyv_f32 npyv_max_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmax_vv_f32m1(a, b, vlen);
}

NPY_FINLINE npyv_f64 npyv_max_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmax_vv_f64m1(a, b, vlen);
}

// Maximum
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();

    vbool32_t not_nan_a = __riscv_vmfeq_vv_f32m1_b32(a, a, vlen);
    vbool32_t not_nan_b = __riscv_vmfeq_vv_f32m1_b32(b, b, vlen);

    vfloat32m1_t sel_a = __riscv_vmerge_vvm_f32m1(b, a, not_nan_a, vlen);
    vfloat32m1_t sel_b = __riscv_vmerge_vvm_f32m1(a, b, not_nan_b, vlen);

    return __riscv_vfmax_vv_f32m1(sel_a, sel_b, vlen);
}


// Max, propagates NaNs
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmax_vv_f32m1(a, b, vlen);
}

// Max, NaN-propagating
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmax_vv_f64m1(a, b, vlen);
}

// Max, NaN-suppressing 
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();

    vbool64_t a_is_nan = __riscv_vmfne_vv_f64m1_b64(a, a, vlen);
    vbool64_t b_is_nan = __riscv_vmfne_vv_f64m1_b64(b, b, vlen);

    vfloat64m1_t max = __riscv_vfmax_vv_f64m1(a, b, vlen);
    max = __riscv_vmerge_vvm_f64m1(max, b, a_is_nan, vlen);
    max = __riscv_vmerge_vvm_f64m1(max, a, b_is_nan, vlen);
    
    return max;
}

// Maximum, integer operations
NPY_FINLINE npyv_u8 npyv_max_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vmaxu_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_max_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vmax_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_max_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vmaxu_vv_u16m1(a, b, vlen);
}

NPY_FINLINE npyv_s16 npyv_max_s16(npyv_s16 a, npyv_s16 b)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vmax_vv_i16m1(a, b, vlen);
}

NPY_FINLINE npyv_u32 npyv_max_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vmaxu_vv_u32m1(a, b, vlen);
}

NPY_FINLINE npyv_s32 npyv_max_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vmax_vv_i32m1(a, b, vlen);
}

NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    vbool64_t mask = __riscv_vmsgtu_vv_u64m1_b64(a, b, vlen);
    return __riscv_vmerge_vvm_u64m1(b, a, mask, vlen);
}

NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    vbool64_t mask = __riscv_vmsgt_vv_i64m1_b64(a, b, vlen);
    return __riscv_vmerge_vvm_i64m1(b, a, mask, vlen);
}

// Minimum, natively mapping with no guarantees to handle NaN.
NPY_FINLINE npyv_f32 npyv_min_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmin_vv_f32m1(a, b, vlen);
}

NPY_FINLINE npyv_f64 npyv_min_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmin_vv_f64m1(a, b, vlen);
}

// Minimum
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();

    vbool32_t not_nan_a = __riscv_vmfeq_vv_f32m1_b32(a, a, vlen);
    vbool32_t not_nan_b = __riscv_vmfeq_vv_f32m1_b32(b, b, vlen);

    vfloat32m1_t sel_a = __riscv_vmerge_vvm_f32m1(b, a, not_nan_a, vlen);
    vfloat32m1_t sel_b = __riscv_vmerge_vvm_f32m1(a, b, not_nan_b, vlen);

    return __riscv_vfmin_vv_f32m1(sel_a, sel_b, vlen);
}

// Min, propagates NaNs
// If any of corresponded element is NaN, NaN is set.
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmin_vv_f32m1(a, b, vlen);
}

// Min, NaN-propagating
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmin_vv_f64m1(a, b, vlen);
}

// Min, NaN-suppressing
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();

    vbool64_t a_is_nan = __riscv_vmfne_vv_f64m1_b64(a, a, vlen);
    vbool64_t b_is_nan = __riscv_vmfne_vv_f64m1_b64(b, b, vlen);

    npyv_f64 min = __riscv_vfmin_vv_f64m1(a, b, vlen);
    min = __riscv_vmerge_vvm_f64m1(min, b, a_is_nan, vlen);
    min = __riscv_vmerge_vvm_f64m1(min, a, b_is_nan, vlen);
    
    return min;
}

// Minimum, integer operations
NPY_FINLINE npyv_u8 npyv_min_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vminu_vv_u8m1(a, b, 16);
}

NPY_FINLINE npyv_s8 npyv_min_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vmin_vv_i8m1(a, b, 16);
}

NPY_FINLINE npyv_u16 npyv_min_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vminu_vv_u16m1(a, b, vlen);
}

NPY_FINLINE npyv_s16 npyv_min_s16(npyv_s16 a, npyv_s16 b)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vmin_vv_i16m1(a, b, vlen);
}

NPY_FINLINE npyv_u32 npyv_min_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vminu_vv_u32m1(a, b, vlen);
}

NPY_FINLINE npyv_s32 npyv_min_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vmin_vv_i32m1(a, b, vlen);
}

NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    vbool64_t mask = __riscv_vmsltu_vv_u64m1_b64(a, b, vlen);
    return __riscv_vmerge_vvm_u64m1(b, a, mask, vlen);
}

NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    vbool64_t mask = __riscv_vmslt_vv_i64m1_b64(a, b, vlen);
    return __riscv_vmerge_vvm_i64m1(b, a, mask, vlen);
}

// reduce min/max for all data types
// Maximum reductions
NPY_FINLINE uint8_t npyv_reduce_max_u8(npyv_u8 a)
{
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(a, __riscv_vmv_v_x_u8m1(0, 16), 16));
}

NPY_FINLINE int8_t npyv_reduce_max_s8(npyv_s8 a)
{
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(a, __riscv_vmv_v_x_i8m1(INT8_MIN, 16), 16));
}

NPY_FINLINE uint16_t npyv_reduce_max_u16(npyv_u16 a)
{
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(a, __riscv_vmv_v_x_u16m1(0, 8), 8));
}

NPY_FINLINE int16_t npyv_reduce_max_s16(npyv_s16 a)
{
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(a, __riscv_vmv_v_x_i16m1(INT16_MIN, 8), 8));
}

NPY_FINLINE uint32_t npyv_reduce_max_u32(npyv_u32 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(a, __riscv_vmv_v_x_u32m1(0, 4), 4));
}

NPY_FINLINE int32_t npyv_reduce_max_s32(npyv_s32 a)
{
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(INT32_MIN, 4), 4));
}

// Floating-point maximum reductions
NPY_FINLINE float npyv_reduce_max_f32(npyv_f32 a)
{
    uint8_t mask = __riscv_vmv_x_s_u8m1_u8(__riscv_vreinterpret_v_b32_u8m1(__riscv_vmfeq_vv_f32m1_b32(a, a, 4)));
    if ((mask & 0b1111) != 0b1111) {
      return NAN;
    }
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(-FLT_MAX, 4), 4));
}

NPY_FINLINE double npyv_reduce_max_f64(npyv_f64 a)
{
    uint8_t mask = __riscv_vmv_x_s_u8m1_u8(__riscv_vreinterpret_v_b64_u8m1(__riscv_vmfeq_vv_f64m1_b64(a, a, 2)));
    if ((mask & 0b11) != 0b11) {
      return NAN;
    }
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(a, __riscv_vfmv_v_f_f64m1(-DBL_MAX, 2), 2));
}

// NaN-propagating maximum reductions
#define npyv_reduce_maxn_f32 npyv_reduce_max_f32
#define npyv_reduce_maxn_f64 npyv_reduce_max_f64

// NaN-suppressing maximum reductions
NPY_FINLINE float npyv_reduce_maxp_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();

    vbool32_t valid_mask = __riscv_vmfeq_vv_f32m1_b32(a, a, vlen);

    vbool32_t nan_mask = __riscv_vmnot_m_b32(valid_mask, vlen);

    npyv_f32 masked_a = __riscv_vfmerge_vfm_f32m1(
        a,
        -INFINITY,
        nan_mask,
        vlen
    );

    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m1_f32m1(
            masked_a,
            __riscv_vfmv_v_f_f32m1(-INFINITY, vlen),
            vlen
        )
    );
}

NPY_FINLINE double npyv_reduce_maxp_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();

    vbool64_t valid_mask = __riscv_vmfeq_vv_f64m1_b64(a, a, vlen);

    vbool64_t nan_mask = __riscv_vmnot_m_b64(valid_mask, vlen);

    npyv_f64 masked_a = __riscv_vfmerge_vfm_f64m1(
        a,
        -INFINITY,
        nan_mask,
        vlen
    );

    return __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmax_vs_f64m1_f64m1(
            masked_a,
            __riscv_vfmv_v_f_f64m1(-INFINITY, vlen),
            vlen
        )
    );
}

// Minimum reductions
NPY_FINLINE uint8_t npyv_reduce_min_u8(npyv_u8 a)
{
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(a, __riscv_vmv_v_x_u8m1(UINT8_MAX, 16), 16));
}

NPY_FINLINE int8_t npyv_reduce_min_s8(npyv_s8 a)
{
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(a, __riscv_vmv_v_x_i8m1(INT8_MAX, 16), 16));
}

NPY_FINLINE uint16_t npyv_reduce_min_u16(npyv_u16 a)
{
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(a, __riscv_vmv_v_x_u16m1(UINT16_MAX, 8), 8));
}

NPY_FINLINE int16_t npyv_reduce_min_s16(npyv_s16 a)
{
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(a, __riscv_vmv_v_x_i16m1(INT16_MAX, 8), 8));
}

NPY_FINLINE uint32_t npyv_reduce_min_u32(npyv_u32 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(a, __riscv_vmv_v_x_u32m1(UINT32_MAX, 4), 4));
}

NPY_FINLINE int32_t npyv_reduce_min_s32(npyv_s32 a)
{
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(a, __riscv_vmv_v_x_i32m1(INT32_MAX, 4), 4));
}

// Floating-point minimum reductions
NPY_FINLINE float npyv_reduce_min_f32(npyv_f32 a)
{
    uint8_t mask = __riscv_vmv_x_s_u8m1_u8(__riscv_vreinterpret_v_b32_u8m1(__riscv_vmfeq_vv_f32m1_b32(a, a, 4)));
    if ((mask & 0b1111) != 0b1111) {
      return NAN;
    }
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(a, __riscv_vfmv_v_f_f32m1(FLT_MAX, 4), 4));
}

NPY_FINLINE double npyv_reduce_min_f64(npyv_f64 a)
{
    uint8_t mask = __riscv_vmv_x_s_u8m1_u8(__riscv_vreinterpret_v_b64_u8m1(__riscv_vmfeq_vv_f64m1_b64(a, a, 2)));
    if ((mask & 0b11) != 0b11) {
      return NAN;
    }
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(a, __riscv_vfmv_v_f_f64m1(DBL_MAX, 2), 2));
}

// NaN-propagating minimum reductions
#define npyv_reduce_minn_f32 npyv_reduce_min_f32
#define npyv_reduce_minn_f64 npyv_reduce_min_f64

// NaN-suppressing minimum reductions
NPY_FINLINE float npyv_reduce_minp_f32(npyv_f32 a)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();

    vbool32_t valid_mask = __riscv_vmfeq_vv_f32m1_b32(a, a, vlen);

    vbool32_t nan_mask = __riscv_vmnot_m_b32(valid_mask, vlen);

    npyv_f32 masked_a = __riscv_vfmerge_vfm_f32m1(
        a,
        INFINITY,
        nan_mask,
        vlen
    );

    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(
            masked_a,
            __riscv_vfmv_v_f_f32m1(INFINITY, vlen),
            vlen
        )
    );
}

NPY_FINLINE double npyv_reduce_minp_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();

    vbool64_t valid_mask = __riscv_vmfeq_vv_f64m1_b64(a, a, vlen);

    vbool64_t nan_mask = __riscv_vmnot_m_b64(valid_mask, vlen);

    npyv_f64 masked_a = __riscv_vfmerge_vfm_f64m1(
        a,
        INFINITY,
        nan_mask,
        vlen
    );

    return __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(
            masked_a,
            __riscv_vfmv_v_f_f64m1(INFINITY, vlen),
            vlen
        )
    );
}

// Maximum reductions for 64-bit integers
NPY_FINLINE npy_uint64 npyv_reduce_max_u64(npyv_u64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_x_s_u64m1_u64(
        __riscv_vredmax_vs_u64m1_u64m1(
            a,
            __riscv_vmv_v_x_u64m1(0, vlen),
            vlen
        )
    );
}

NPY_FINLINE npy_int64 npyv_reduce_max_s64(npyv_s64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_x_s_i64m1_i64(
        __riscv_vredmax_vs_i64m1_i64m1(
            a,
            __riscv_vmv_v_x_i64m1(INT64_MIN, vlen),
            vlen
        )
    );
}

NPY_FINLINE npy_uint64 npyv_reduce_min_u64(npyv_u64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_x_s_u64m1_u64(
        __riscv_vredmin_vs_u64m1_u64m1(
            a,
            __riscv_vmv_v_x_u64m1(UINT64_MAX, vlen),
            vlen
        )
    );
}

NPY_FINLINE npy_int64 npyv_reduce_min_s64(npyv_s64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_x_s_i64m1_i64(
        __riscv_vredmin_vs_i64m1_i64m1(
            a,
            __riscv_vmv_v_x_i64m1(INT64_MAX, vlen),
            vlen
        )
    );
}

// round to nearest integer even
NPY_FINLINE npyv_f32 npyv_rint_f32(npyv_f32 a)
{
    return __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1_rm(a, __RISCV_FRM_RNE, 4), 4);
}

NPY_FINLINE npyv_f64 npyv_rint_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfcvt_f_x_v_f64m1(
        __riscv_vfcvt_x_f_v_i64m1_rm(a, __RISCV_FRM_RNE, vlen), 
        vlen
    );
}

// ceil
NPY_FINLINE npyv_f32 npyv_ceil_f32(npyv_f32 a)
{
    return __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1_rm(a, __RISCV_FRM_RUP, 4), 4);
}

NPY_FINLINE npyv_f64 npyv_ceil_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfcvt_f_x_v_f64m1(
        __riscv_vfcvt_x_f_v_i64m1_rm(a, __RISCV_FRM_RUP, vlen), 
        vlen
    );
}

// trunc
NPY_FINLINE npyv_f32 npyv_trunc_f32(npyv_f32 a)
{
    return __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1_rm(a, __RISCV_FRM_RTZ, 4), 4);
}

NPY_FINLINE npyv_f64 npyv_trunc_f64(npyv_f64 a)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfcvt_f_x_v_f64m1(
        __riscv_vfcvt_x_f_v_i64m1_rm(a, __RISCV_FRM_RTZ, vlen), 
        vlen
    );
}

// floor
NPY_FINLINE npyv_f32 npyv_floor_f32(npyv_f32 a)
{
    return __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1_rm(a, __RISCV_FRM_RDN, 4), 4);
}

NPY_FINLINE npyv_f64 npyv_floor_f64(npyv_f64 a)
{
    size_t vl = __riscv_vsetvlmax_e64m1();
    return __riscv_vfcvt_f_x_v_f64m1(
        __riscv_vfcvt_x_f_v_i64m1_rm(a, __RISCV_FRM_RDN, vl),
        vl
    );
}

#endif // _NPY_SIMD_RVV_MATH_H
