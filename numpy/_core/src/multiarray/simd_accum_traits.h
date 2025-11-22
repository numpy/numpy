#pragma once
/*
 * This file contians wrappers and traits for the SIMD kernels
 * using NPY_SIMD_F32 & NPY_SIMD_F64. 
 * 
 * The goal is to isolate direct uses of SIMD functionality behind a small
 * API. Allows writing in terms of T / AccumType without 
 * hard-coding SIMD details.
 * 
 * By Amelia Thurdekoos (ameliathurdekoos@gmail.com)
 * 
 * See LICENSE.txt for the license.
 * 
 */

#if NPY_SIMD_F32
struct NpySIMDF32 {
    using T = npy_float;
    using AccumType = npy_float;
    using SimdReg = npyv_f32;
    static inline int npyv_nlanes() { return npyv_nlanes_f32; }
    static inline SimdReg npyv_zero() { return npyv_zero_f32(); }
    static inline SimdReg loada(const T *p) { return npyv_loada_f32(p); }
    static inline SimdReg load(const T *p) { return npyv_load_f32(p); }
    static inline void storea(T *p, SimdReg x) { npyv_storea_f32(p, x); }
    static inline void store(T *p, SimdReg x) { npyv_store_f32(p, x); }
    static inline SimdReg load_tillz(const T *p, npy_intp n)
    {
        return npyv_load_tillz_f32(p, n);
    }
    static inline SimdReg npyv_add(SimdReg a, SimdReg b) { return npyv_add_f32(a, b); }
    static inline T npyv_sum(SimdReg v) { return npyv_sum_f32(v); }
    static inline SimdReg load_any(const T *p, int is_aligned)
    {
        return is_aligned ? loada(p) : load(p);
    }
    static inline void cleanup() { npyv_cleanup(); }
    static inline SimdReg npyv_setall(AccumType scalar)
    {
        return npyv_setall_f32(scalar);
    }
    static inline SimdReg npyv_muladd(SimdReg v_scalar, SimdReg a, SimdReg b)
    {
        return npyv_muladd_f32(v_scalar, a, b);
    }
    static inline void npyv_st(T *p, SimdReg x, int is_aligned)
    {
        is_aligned ? storea(p, x) : store(p, x);
    }
    static inline void npyv_store_till(T *data_out, npy_intp count, SimdReg z)
    {
        npyv_store_till_f32(data_out, count, z);
    }
};
#endif  // NPY_SIMD_F32
#if NPY_SIMD_F64
struct NpySIMDF64 {
    using T = npy_double;
    using AccumType = npy_double;
    using SimdReg = npyv_f64;
    static inline int npyv_nlanes() { return npyv_nlanes_f64; }
    static inline SimdReg npyv_zero() { return npyv_zero_f64(); }
    static inline SimdReg loada(const T *p) { return npyv_loada_f64(p); }
    static inline SimdReg load(const T *p) { return npyv_load_f64(p); }
    static inline void storea(T *p, SimdReg x) { npyv_storea_f64(p, x); }
    static inline void store(T *p, SimdReg x) { npyv_store_f64(p, x); }
    static inline SimdReg load_tillz(const T *p, npy_intp n)
    {
        return npyv_load_tillz_f64(p, n);
    }
    static inline SimdReg npyv_add(SimdReg a, SimdReg b) { return npyv_add_f64(a, b); }
    static inline T npyv_sum(SimdReg v) { return npyv_sum_f64(v); }
    static inline SimdReg load_any(const T *p, int is_aligned)
    {
        return is_aligned ? loada(p) : load(p);
    }
    static inline void cleanup() { npyv_cleanup(); }
    static inline SimdReg npyv_setall(AccumType scalar)
    {
        return npyv_setall_f64(scalar);
    }
    static inline SimdReg npyv_muladd(SimdReg v_scalar, SimdReg a, SimdReg b)
    {
        return npyv_muladd_f64(v_scalar, a, b);
    }
    static inline void npyv_st(T *p, SimdReg x, int is_aligned)
    {
        is_aligned ? storea(p, x) : store(p, x);
    }
    static inline void npyv_store_till(T *data_out, npy_intp count, SimdReg z)
    {
        npyv_store_till_f64(data_out, count, z);
    }
};
#endif  // NPY_SIMD_F64

/* Helper traits mapping a (T, AccumType) pair to the SIMD wrapper */
template <typename T, typename AccumType>
struct SumSIMD {};

#if NPY_SIMD_F32
/* Wrapper around the npyv float32 SIMD, provides a uniform interface between SIMDF32
 * and SIMDF64 */
template <>
struct SumSIMD<npy_float, npy_float> {
    using SimdType = NpySIMDF32;
};
#endif  // NPY_SIMD_F32
#if NPY_SIMD_F64
/* Wrapper around the npyv float64 SIMD, provides a uniform interface between SIMDF32
 * and SIMDF64 */
template <>
struct SumSIMD<npy_double, npy_double> {
    using SimdType = NpySIMDF64;
};
#endif  // NPY_SIMD_F64
