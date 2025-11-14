/*
 * This file provides optimized sum of product implementations used internally
 * by einsum.
 *
 * Copyright (c) 2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "einsum_sumprod.h"

#include "common.h"
#include "einsum_debug.h"

#include "simd/simd.h"
#include <array>
#include <iostream>
#include <numpy/arrayobject.h>
#include <numpy/halffloat.h>
#include <numpy/ndarraytypes.h> /* for NPY_NTYPES_LEGACY */
#include <numpy/npy_common.h>

// ARM/Neon don't have instructions for aligned memory access
#ifdef NPY_HAVE_NEON
#    define EINSUM_IS_ALIGNED(x) 0
#else  // NPY_HAVE_NEON
#    define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)
#endif  // NPY_HAVE_NEON

/* Converts a value from storage type T to the Temptype.
 * Centralizes norminal use and the specalized use case of npy_half_to_float
 */
template <typename T, typename Temptype>
static inline constexpr Temptype
from(T v)
{
    return (Temptype)v;
}

template <>
inline npy_float
from<npy_half, npy_float>(npy_half v)
{
    return npy_half_to_float(v);
}

/* Converts a value from storage type Temptype to the Type.
 * Centralizes norminal use and the specalized use case of npy_float_to_half
 */
template <typename T, typename Temptype>
static inline constexpr Temptype
to(T v)
{
    return (Temptype)v;
}

template <>
inline npy_half
to<npy_float, npy_half>(npy_float v)
{
    return npy_float_to_half(v);
}

#if NPY_SIMD_F32
struct NpySIMDF32 {
    using T = npy_float;
    using Temptype = npy_float;
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
    static inline SimdReg npyv_setall(Temptype scalar)
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
    using Temptype = npy_double;
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
    static inline SimdReg npyv_setall(Temptype scalar)
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

/* Helper traits mapping a (T, Temptype) pair to the SIMD wrapper */
template <typename T, typename TempType>
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

template <typename SimdType>
static inline NPY_GCC_OPT_3 typename SimdType::T
floating_point_sum_of_arr(const typename SimdType::T *data, npy_intp count)
{
    using T = typename SimdType::T;
    using SimdReg = typename SimdType::SimdReg;
    /* Use aligned instructions if possible */
    const int is_aligned = EINSUM_IS_ALIGNED(data);
    const int vstep = SimdType::npyv_nlanes();
    SimdReg v_accum = SimdType::npyv_zero();
    const npy_intp vstepx4 = vstep * 4;

    for (; count >= vstepx4; count -= vstepx4, data += vstepx4) {
        const SimdReg a0 = SimdType::load_any(data + vstep * 0, is_aligned);
        const SimdReg a1 = SimdType::load_any(data + vstep * 1, is_aligned);
        const SimdReg a2 = SimdType::load_any(data + vstep * 2, is_aligned);
        const SimdReg a3 = SimdType::load_any(data + vstep * 3, is_aligned);

        const SimdReg a01 = SimdType::npyv_add(a0, a1);
        const SimdReg a23 = SimdType::npyv_add(a2, a3);
        const SimdReg a0123 = SimdType::npyv_add(a01, a23);
        v_accum = SimdType::npyv_add(a0123, v_accum);
    }

    for (; count > 0; count -= vstep, data += vstep) {
        SimdReg a = SimdType::load_tillz(data, count);
        v_accum = SimdType::npyv_add(a, v_accum);
    }
    T accum = SimdType::npyv_sum(v_accum);
    SimdType::cleanup();
    return accum;
}

template <typename T, typename Temptype>
static inline NPY_GCC_OPT_3 Temptype
scaller_sum_of_arr(const T *data, npy_intp count)
{
    Temptype accum = 0;

#ifndef NPY_DISABLE_OPTIMIZATION
    for (; count > 4; count -= 4, data += 4) {
        const Temptype a01 = from<T, Temptype>(*data) + from<T, Temptype>(data[1]);
        const Temptype a23 = from<T, Temptype>(data[2]) + from<T, Temptype>(data[3]);
        accum += a01 + a23;
    }
#endif  // NPY_DISABLE_OPTIMIZATION

    for (; count > 0; --count, ++data) {
        accum += from<T, Temptype>(*data);
    }
    return accum;
}

/* Template where (npy_float, npy_float) || (npy_double, npy_double) will allow the SIMD
 * capable version.*/
template <typename T, typename Temptype,
          typename std::enable_if<std::is_same<T, Temptype>::value &&
                                          (std::is_same<T, npy_float>::value ||
                                           std::is_same<T, npy_double>::value),
                                  int>::type = 0>
static inline NPY_GCC_OPT_3 Temptype
sum_of_arr(T *data, npy_intp count)
{
#if (NPY_SIMD_F32 || NPY_SIMD_F64)
    return floating_point_sum_of_arr<typename SumSIMD<T, Temptype>::SimdType>(data,
                                                                              count);
#else   // !(NPY_SIMD_F32 || NPY_SIMD_F64)
    return scaller_sum_of_arr<T, Temptype>(data, count);
#endif  // (NPY_SIMD_F32 || NPY_SIMD_F64)
}

template <typename T, typename Temptype,
          typename std::enable_if<!(std::is_same<T, Temptype>::value &&
                                    (std::is_same<T, npy_float>::value ||
                                     std::is_same<T, npy_double>::value)),
                                  int>::type = 0>
static inline NPY_GCC_OPT_3 Temptype
sum_of_arr(T *data, npy_intp count)
{
    return scaller_sum_of_arr<T, Temptype>(data, count);
}

template <typename SimdType>
static inline NPY_GCC_OPT_3 void
floating_point_sum_of_products_muladd(const typename SimdType::T *data,
                                      typename SimdType::T *data_out,
                                      typename SimdType::Temptype scalar,
                                      npy_intp count)
{
    using SimdReg = typename SimdType::SimdReg;
    /* Use aligned instructions if possible */
    const int is_aligned = EINSUM_IS_ALIGNED(data) && EINSUM_IS_ALIGNED(data_out);
    const int vstep = SimdType::npyv_nlanes();
    const SimdReg v_scalar = SimdType::npyv_setall(scalar);
    const npy_intp vstepx4 = vstep * 4;

    for (; count >= vstepx4; count -= vstepx4, data += vstepx4, data_out += vstepx4) {
        const SimdReg b0 = SimdType::load_any(data + vstep * 0, is_aligned);
        const SimdReg c0 = SimdType::load_any(data_out + vstep * 0, is_aligned);
        const SimdReg b1 = SimdType::load_any(data + vstep * 1, is_aligned);
        const SimdReg c1 = SimdType::load_any(data_out + vstep * 1, is_aligned);
        const SimdReg b2 = SimdType::load_any(data + vstep * 2, is_aligned);
        const SimdReg c2 = SimdType::load_any(data_out + vstep * 2, is_aligned);
        const SimdReg b3 = SimdType::load_any(data + vstep * 3, is_aligned);
        const SimdReg c3 = SimdType::load_any(data_out + vstep * 3, is_aligned);

        const SimdReg abc0 = SimdType::npyv_muladd(v_scalar, b0, c0);
        const SimdReg abc1 = SimdType::npyv_muladd(v_scalar, b1, c1);
        const SimdReg abc2 = SimdType::npyv_muladd(v_scalar, b2, c2);
        const SimdReg abc3 = SimdType::npyv_muladd(v_scalar, b3, c3);

        SimdType::npyv_st(data_out + vstep * 0, abc0, is_aligned);
        SimdType::npyv_st(data_out + vstep * 1, abc1, is_aligned);
        SimdType::npyv_st(data_out + vstep * 2, abc2, is_aligned);
        SimdType::npyv_st(data_out + vstep * 3, abc3, is_aligned);
    }

    for (; count > 0; count -= vstep, data += vstep, data_out += vstep) {
        SimdReg a = SimdType::load_tillz(data, count);
        SimdReg b = SimdType::load_tillz(data_out, count);
        SimdReg c = SimdType::npyv_muladd(a, v_scalar, b);
        SimdType::npyv_store_till(data_out, count, c);
    }
    SimdType::cleanup();
}

template <typename T, typename Temptype>
static inline NPY_GCC_OPT_3 void
scaller_sum_of_products_muladd(const T *data, T *data_out, Temptype scalar,
                               npy_intp count)
{
#ifndef NPY_DISABLE_OPTIMIZATION
    for (; count >= 4; count -= 4, data += 4, data_out += 4) {
        const Temptype b0 = from<T, Temptype>(data[0]);
        const Temptype c0 = from<T, Temptype>(data_out[0]);
        const Temptype b1 = from<T, Temptype>(data[1]);
        const Temptype c1 = from<T, Temptype>(data_out[1]);
        const Temptype b2 = from<T, Temptype>(data[2]);
        const Temptype c2 = from<T, Temptype>(data_out[2]);
        const Temptype b3 = from<T, Temptype>(data[3]);
        const Temptype c3 = from<T, Temptype>(data_out[3]);

        const Temptype abc0 = scalar * b0 + c0;
        const Temptype abc1 = scalar * b1 + c1;
        const Temptype abc2 = scalar * b2 + c2;
        const Temptype abc3 = scalar * b3 + c3;

        data_out[0] = to<Temptype, T>(abc0);
        data_out[1] = to<Temptype, T>(abc1);
        data_out[2] = to<Temptype, T>(abc2);
        data_out[3] = to<Temptype, T>(abc3);
    }
#endif  // !NPY_DISABLE_OPTIMIZATION
    for (; count > 0; --count, ++data, ++data_out) {
        const Temptype b = from<T, Temptype>(*data);
        const Temptype c = from<T, Temptype>(*data_out);
        *data_out = to<Temptype, T>(scalar * b + c);
    }
}

/* calculate the multiply and add operation such as dataout = data*scalar+dataout*/
/* Template where (npy_float, npy_float) || (npy_double, npy_double) will allow the SIMD
 * capable version.*/
template <typename T, typename Temptype,
          typename std::enable_if<std::is_same<T, Temptype>::value &&
                                          (std::is_same<T, npy_float>::value ||
                                           std::is_same<T, npy_double>::value),
                                  int>::type = 0>
static inline NPY_GCC_OPT_3 void
sum_of_products_muladd(T *data, T *data_out, Temptype scalar, npy_intp count)
{
#if (NPY_SIMD_F32 || NPY_SIMD_F64)
    floating_point_sum_of_products_muladd<typename SumSIMD<T, Temptype>::SimdType>(
            data, data_out, scalar, count);

#else   // !(NPY_SIMD_F32 || NPY_SIMD_F64)
    scaller_sum_of_products_muladd<T, Temptype>(data, data_out, scalar, count);
#endif  // (NPY_SIMD_F32 || NPY_SIMD_F64)
}

template <typename T, typename Temptype,
          typename std::enable_if<!(std::is_same<T, Temptype>::value &&
                                    (std::is_same<T, npy_float>::value ||
                                     std::is_same<T, npy_double>::value)),
                                  int>::type = 0>
static inline NPY_GCC_OPT_3 void
sum_of_products_muladd(T *data, T *data_out, Temptype scalar, npy_intp count)

{
    scaller_sum_of_products_muladd<T, Temptype>(data, data_out, scalar, count);
}

template <typename SimdType>
static NPY_GCC_OPT_3 typename SimdType::T
floating_point_sum_of_arr_products_contig_contig_outstride0_two(
        const typename SimdType::T *data0, const typename SimdType::T *data1,
        npy_intp count)
{
    using T = typename SimdType::T;
    using SimdReg = typename SimdType::SimdReg;

    /* Use aligned instructions if possible */
    const int is_aligned = EINSUM_IS_ALIGNED(data0) && EINSUM_IS_ALIGNED(data1);
    const int vstep = SimdType::npyv_nlanes();
    SimdReg v_accum = SimdType::npyv_zero();
    const npy_intp vstepx4 = vstep * 4;

    for (; count >= vstepx4; count -= vstepx4, data0 += vstepx4, data1 += vstepx4) {
        const SimdReg a0 = SimdType::load_any(data0 + vstep * 0, is_aligned);
        const SimdReg b0 = SimdType::load_any(data1 + vstep * 0, is_aligned);
        const SimdReg a1 = SimdType::load_any(data0 + vstep * 1, is_aligned);
        const SimdReg b1 = SimdType::load_any(data1 + vstep * 1, is_aligned);
        const SimdReg a2 = SimdType::load_any(data0 + vstep * 2, is_aligned);
        const SimdReg b2 = SimdType::load_any(data1 + vstep * 2, is_aligned);
        const SimdReg a3 = SimdType::load_any(data0 + vstep * 3, is_aligned);
        const SimdReg b3 = SimdType::load_any(data1 + vstep * 3, is_aligned);

        const SimdReg ab3 = SimdType::npyv_muladd(a3, b3, v_accum);
        const SimdReg ab2 = SimdType::npyv_muladd(a2, b2, ab3);
        const SimdReg ab1 = SimdType::npyv_muladd(a1, b1, ab2);
        v_accum = SimdType::npyv_muladd(a0, b0, ab1);
    }

    for (; count > 0; count -= vstep, data0 += vstep, data1 += vstep) {
        const SimdReg a = SimdType::load_tillz(data0, count);
        const SimdReg b = SimdType::load_tillz(data1, count);
        v_accum = SimdType::npyv_muladd(a, b, v_accum);
    }

    T accum = SimdType::npyv_sum(v_accum);
    SimdType::cleanup();
    return accum;
}

template <typename T, typename Temptype>
static NPY_GCC_OPT_3 Temptype
scaller_sum_of_arr_products_contig_contig_outstride0_two(const T *data0, const T *data1,
                                                         npy_intp count)
{
    Temptype accum = 0;

#ifndef NPY_DISABLE_OPTIMIZATION
    for (; count >= 4; count -= 4, data0 += 4, data1 += 4) {
        const Temptype ab0 = from<T, Temptype>(data0[0]) * from<T, Temptype>(data1[0]);
        const Temptype ab1 = from<T, Temptype>(data0[1]) * from<T, Temptype>(data1[1]);
        const Temptype ab2 = from<T, Temptype>(data0[2]) * from<T, Temptype>(data1[2]);
        const Temptype ab3 = from<T, Temptype>(data0[3]) * from<T, Temptype>(data1[3]);

        accum += ab0 + ab1 + ab2 + ab3;
    }
#endif  // !NPY_DISABLE_OPTIMIZATION
    for (; count > 0; --count, ++data0, ++data1) {
        const Temptype a = from<T, Temptype>(*data0);
        const Temptype b = from<T, Temptype>(*data1);
        accum += a * b;
    }
    return accum;
}

/* Template where (npy_float, npy_float) || (npy_double, npy_double) will allow the SIMD
 * capable version.*/
template <typename T, typename Temptype,
          typename std::enable_if<std::is_same<T, Temptype>::value &&
                                          (std::is_same<T, npy_float>::value ||
                                           std::is_same<T, npy_double>::value),
                                  int>::type = 0>
static NPY_GCC_OPT_3 Temptype
sum_of_arr_products_contig_contig_outstride0_two(T *data0, T *data1, npy_intp count)
{
#if (NPY_SIMD_F32 || NPY_SIMD_F64)
    return floating_point_sum_of_arr_products_contig_contig_outstride0_two<
            typename SumSIMD<T, Temptype>::SimdType>(data0, data1, count);

#else   // !(NPY_SIMD_F32 || NPY_SIMD_F64)
    return scaller_sum_of_arr_products_contig_contig_outstride0_two<T, Temptype>(
            data0, data1, count);
#endif  // (NPY_SIMD_F32 || NPY_SIMD_F64)
}

template <typename T, typename Temptype,
          typename std::enable_if<!(std::is_same<T, Temptype>::value &&
                                    (std::is_same<T, npy_float>::value ||
                                     std::is_same<T, npy_double>::value)),
                                  int>::type = 0>
static NPY_GCC_OPT_3 Temptype
sum_of_arr_products_contig_contig_outstride0_two(T *data0, T *data1, npy_intp count)
{
    return scaller_sum_of_arr_products_contig_contig_outstride0_two<T, Temptype>(
            data0, data1, count);
}

template <typename T, typename Temptype>
static NPY_GCC_OPT_3 void
sum_of_products_contig_contig_outstride0_two(int nop, char **dataptr,
                                             npy_intp const *NPY_UNUSED(strides),
                                             npy_intp count)
{
    T *data0 = (T *)dataptr[0];
    T *data1 = (T *)dataptr[1];

    NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_contig_outstride0_two (%d)\n",
                          (int)count);
    Temptype accum = sum_of_arr_products_contig_contig_outstride0_two<T, Temptype>(
            data0, data1, count);
    *(T *)dataptr[2] = to<Temptype, T>(from<T, Temptype>(*(T *)dataptr[2]) + accum);
}

template <typename T, typename Temptype>
static inline NPY_GCC_OPT_3 void
sum_of_products_stride0_contig_outstride0_two(int nop, char **dataptr,
                                              npy_intp const *NPY_UNUSED(strides),
                                              npy_intp count)
{
    T *data1 = (T *)dataptr[1];
    Temptype value0 = from<T, Temptype>(*(T *)dataptr[0]);
    Temptype accum = sum_of_arr<T, Temptype>(data1, count);
    *(T *)dataptr[2] =
            to<Temptype, T>(from<T, Temptype>(*(T *)dataptr[2]) + value0 * accum);
}

/* Some extra specializations for the two operand case */
template <typename T, typename Temptype>
static inline void
sum_of_products_stride0_contig_outcontig_two(int nop, char **dataptr,
                                             npy_intp const *NPY_UNUSED(strides),
                                             npy_intp count)
{
    Temptype value0 = from<T, Temptype>(*(T *)dataptr[0]);
    T *data1 = (T *)dataptr[1];
    T *data_out = (T *)dataptr[2];

    NPY_EINSUM_DBG_PRINT1("Generic_sum_of_products_stride0_contig_outcontig_two (%d)\n",
                          (int)count);
    sum_of_products_muladd<T, Temptype>(data1, data_out, value0, count);
}

template <typename T, typename Temptype>
static inline void
sum_of_products_contig_stride0_outstride0_two(int nop, char **dataptr,
                                              npy_intp const *NPY_UNUSED(strides),
                                              npy_intp count)
{
    T *data0 = (T *)dataptr[0];
    Temptype value1 = from<T, Temptype>(*(T *)dataptr[1]);
    Temptype accum = sum_of_arr<T, Temptype>(data0, count);
    *(T *)dataptr[2] =
            to<Temptype, T>(from<T, Temptype>(*(T *)dataptr[2]) + value1 * accum);
}

template <typename T, typename Temptype>
static inline void
sum_of_products_contig_stride0_outcontig_two(int nop, char **dataptr,
                                             npy_intp const *NPY_UNUSED(strides),
                                             npy_intp count)
{
    Temptype value1 = from<T, Temptype>(*(T *)dataptr[1]);
    T *data0 = (T *)dataptr[0];
    T *data_out = (T *)dataptr[2];

    NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_stride0_outcontig_two (%d)\n",
                          (int)count);
    sum_of_products_muladd<T, Temptype>(data0, data_out, value1, count);
}

template <typename T, typename Temptype, bool Is_Complex>
static inline NPY_GCC_OPT_3 void
sum_of_products_contig_outstride0_one(int nop, char **dataptr, npy_intp const *strides,
                                      npy_intp count)
{
    NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_outstride0_one (%d)\n", (int)count);
    if constexpr (!Is_Complex) {
        T *data = (T *)dataptr[0];
        Temptype accum = sum_of_arr<T, Temptype>(data, count);

        *((T *)dataptr[1]) =
                to<Temptype, T>(accum + from<T, Temptype>(*((T *)dataptr[1])));
    }
    else {  // complex
        Temptype accum_re = 0, accum_im = 0;
        Temptype *data0 = (Temptype *)dataptr[0];
#ifndef NPY_DISABLE_OPTIMIZATION
        for (; count > 4; count -= 4, data0 += 4 * 2) {
            const Temptype re01 = data0[0] + data0[2];
            const Temptype re23 = data0[4] + data0[6];
            const Temptype im13 = data0[1] + data0[3];
            const Temptype im57 = data0[5] + data0[7];
            accum_re += re01 + re23;
            accum_im += im13 + im57;
        }
#endif  // !NPY_DISABLE_OPTIMIZATION
        for (; count > 0; --count, data0 += 2) {
            accum_re += data0[0];
            accum_im += data0[1];
        }
        ((Temptype *)dataptr[1])[0] += accum_re;
        ((Temptype *)dataptr[1])[1] += accum_im;
    }
}

/*
 *  Helper function used for all PyObject einsum sum-of-products calculations.
 */
static inline void
object_sum_of_products(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    while (count--) {
        PyObject *prod = *(PyObject **)dataptr[0];
        if (!prod) {
            prod = Py_None;  // convention is to treat nulls as None
        }
        Py_INCREF(prod);
        for (int i = 1; i < nop; ++i) {
            PyObject *curr = *(PyObject **)dataptr[i];
            if (!curr) {
                curr = Py_None;  // convention is to treat nulls as None
            }
            Py_SETREF(prod, PyNumber_Multiply(prod, curr));
            if (!prod) {
                return;
            }
        }

        PyObject *sum = PyNumber_Add(*(PyObject **)dataptr[nop], prod);
        Py_DECREF(prod);
        if (!sum) {
            return;
        }

        Py_XDECREF(*(PyObject **)dataptr[nop]);
        *(PyObject **)dataptr[nop] = sum;
        for (int i = 0; i <= nop; ++i) {
            dataptr[i] += strides[i];
        }
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_outstride0_one(int nop, char **dataptr, npy_intp const *strides,
                               npy_intp count)
{
    if constexpr (!Is_Complex) {
        Temptype accum = 0;

        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_outstride0_one(%d)\n", (int)count);

        while (count--) {
            accum += from<T, Temptype>(*(T *)data0);
            data0 += stride0;
        }

        *((T *)dataptr[1]) =
                to<Temptype, T>(accum + from<T, Temptype>(*((T *)dataptr[1])));
    }
    else {  // complex
        Temptype accum_re = 0, accum_im = 0;

        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];
        while (count--) {
            accum_re += ((Temptype *)data0)[0];
            accum_im += ((Temptype *)data0)[1];
            data0 += stride0;
        }

        ((Temptype *)dataptr[1])[0] += accum_re;
        ((Temptype *)dataptr[1])[1] += accum_im;
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_outstride0_two(int nop, char **dataptr, npy_intp const *strides,
                               npy_intp count)
{
    if constexpr (!Is_Complex) {
        Temptype accum = 0;

        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];
        char *data1 = dataptr[1];
        npy_intp stride1 = strides[1];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_outstride0_two(%d)\n", (int)count);

        while (count--) {
            accum += from<T, Temptype>(*(T *)data0) * from<T, Temptype>(*(T *)data1);
            data0 += stride0;
            data1 += stride1;
        }

        *((T *)dataptr[2]) =
                to<Temptype, T>(accum + from<T, Temptype>(*((T *)dataptr[2])));
    }
    else {  // complex
        Temptype accum_re = 0, accum_im = 0;

        while (count--) {
            Temptype re = ((Temptype *)dataptr[0])[0];
            Temptype im = ((Temptype *)dataptr[0])[1];

            Temptype tmp =
                    re * ((Temptype *)dataptr[1])[0] - im * ((Temptype *)dataptr[1])[1];
            im = re * ((Temptype *)dataptr[1])[1] + im * ((Temptype *)dataptr[1])[0];
            re = tmp;

            accum_re += re;
            accum_im += im;

            dataptr[0] += strides[0];
            dataptr[1] += strides[1];
        }

        ((Temptype *)dataptr[2])[0] += accum_re;
        ((Temptype *)dataptr[2])[1] += accum_im;
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_outstride0_three(int nop, char **dataptr, npy_intp const *strides,
                                 npy_intp count)
{
    if constexpr (!Is_Complex) {
        Temptype accum = 0;

        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];
        char *data1 = dataptr[1];
        npy_intp stride1 = strides[1];
        char *data2 = dataptr[2];
        npy_intp stride2 = strides[2];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_outstride0_three(%d)\n", (int)count);

        while (count--) {
            accum += from<T, Temptype>(*(T *)data0) * from<T, Temptype>(*(T *)data1) *
                     from<T, Temptype>(*(T *)data2);
            data0 += stride0;
            data1 += stride1;
            data2 += stride2;
        }

        *((T *)dataptr[3]) =
                to<Temptype, T>(accum + from<T, Temptype>(*((T *)dataptr[3])));
    }
    else {  // complex
        Temptype accum_re = 0, accum_im = 0;

        while (count--) {
            Temptype re = ((Temptype *)dataptr[0])[0];
            Temptype im = ((Temptype *)dataptr[0])[1];

            Temptype tmp =
                    re * ((Temptype *)dataptr[1])[0] - im * ((Temptype *)dataptr[1])[1];
            im = re * ((Temptype *)dataptr[1])[1] + im * ((Temptype *)dataptr[1])[0];
            re = tmp;

            tmp = re * ((Temptype *)dataptr[2])[0] - im * ((Temptype *)dataptr[2])[1];
            im = re * ((Temptype *)dataptr[2])[1] + im * ((Temptype *)dataptr[2])[0];
            re = tmp;

            accum_re += re;
            accum_im += im;

            dataptr[0] += strides[0];
            dataptr[1] += strides[1];
            dataptr[2] += strides[2];
        }

        ((Temptype *)dataptr[3])[0] += accum_re;
        ((Temptype *)dataptr[3])[1] += accum_im;
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_outstride0_any(int nop, char **dataptr, npy_intp const *strides,
                               npy_intp count)
{
    if constexpr (!Is_Complex) {
        Temptype accum = 0;

        while (count--) {
            Temptype temp = from<T, Temptype>(*(T *)dataptr[0]);
            int i;
            for (i = 1; i < nop; ++i) {
                temp *= from<T, Temptype>(*(T *)dataptr[i]);
            }
            accum += temp;
            for (i = 0; i < nop; ++i) {
                dataptr[i] += strides[i];
            }
        }

        *((T *)dataptr[nop]) =
                to<Temptype, T>(accum + from<T, Temptype>(*((T *)dataptr[nop])));
    }
    else {  // complex
        Temptype accum_re = 0, accum_im = 0;

        while (count--) {
            Temptype re = ((Temptype *)dataptr[0])[0];
            Temptype im = ((Temptype *)dataptr[0])[1];

            for (int i = 1; i < nop; ++i) {
                Temptype tmp = re * ((Temptype *)dataptr[i])[0] -
                               im * ((Temptype *)dataptr[i])[1];
                im = re * ((Temptype *)dataptr[i])[1] +
                     im * ((Temptype *)dataptr[i])[0];
                re = tmp;
            }

            accum_re += re;
            accum_im += im;

            for (int i = 0; i < nop; ++i) {
                dataptr[i] += strides[i];
            }
        }

        ((Temptype *)dataptr[nop])[0] += accum_re;
        ((Temptype *)dataptr[nop])[1] += accum_im;
    }
}

template <>
inline void
sum_of_products_outstride0_one<PyObject, PyObject, false, false>(
        int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_outstride0_two<PyObject, PyObject, false, false>(
        int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_outstride0_three<PyObject, PyObject, false, false>(
        int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_outstride0_any<PyObject, PyObject, false, false>(
        int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_outstride0_one<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                                npy_intp const *strides,
                                                                npy_intp count)
{
    npy_bool accum = 0;
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];

    while (count--) {
        accum = *(npy_bool *)data0 || accum;
        data0 += stride0;
    }
    *((npy_bool *)dataptr[1]) = accum || *((npy_bool *)dataptr[1]);
}

template <>
inline void
sum_of_products_outstride0_two<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                                npy_intp const *strides,
                                                                npy_intp count)
{
    npy_bool accum = 0;
    char *data0 = dataptr[0];
    char *data1 = dataptr[1];
    npy_intp stride0 = strides[0];
    npy_intp stride1 = strides[1];

    while (count--) {
        accum = ((*(npy_bool *)data0) && (*(npy_bool *)data1)) || accum;
        data0 += stride0;
        data1 += stride1;
    }

    *((npy_bool *)dataptr[2]) = accum || *((npy_bool *)dataptr[2]);
}

template <>
inline void
sum_of_products_outstride0_three<npy_bool, npy_bool, false, true>(
        int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    npy_bool accum = 0;
    char *data0 = dataptr[0];
    char *data1 = dataptr[1];
    char *data2 = dataptr[2];
    npy_intp stride0 = strides[0];
    npy_intp stride1 = strides[1];
    npy_intp stride2 = strides[2];

    while (count--) {
        accum = ((*(npy_bool *)data0) && (*(npy_bool *)data1) &&
                 (*(npy_bool *)data2)) ||
                accum;
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
    }

    *((npy_bool *)dataptr[3]) = accum || *((npy_bool *)dataptr[3]);
}

template <>
inline void
sum_of_products_outstride0_any<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                                npy_intp const *strides,
                                                                npy_intp count)
{
    npy_bool accum = 0;

    while (count--) {
        npy_bool temp = *(npy_bool *)dataptr[0];
        int i;
        for (i = 1; i < nop; ++i) {
            temp = temp && *(npy_bool *)dataptr[i];
        }
        accum = temp || accum;
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += strides[i];
        }
    }
    *((npy_bool *)dataptr[nop]) = accum || *((npy_bool *)dataptr[nop]);
}

// forward declaration
template <typename T, typename Temptype, bool Is_Complex, int Start, int End, int Step,
          bool Done = (Start == End)>
struct Sum_Of_Products_Contig_One_Stepper;

// Recursive case
template <typename T, typename Temptype, bool Is_Complex, int Start, int End, int Step>
struct Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, Start, End, Step,
                                          false> {
    static inline void apply(const T *data0, T *data_out, npy_intp count)
    {
        constexpr int I = Start;

        if (count > I) {
            if constexpr (!Is_Complex) {
                data_out[I] = to<Temptype, T>(from<T, Temptype>(data0[I]) +
                                              from<T, Temptype>(data_out[I]));
            }
            else {  // complex
                ((Temptype *)data_out + 2 * I)[0] = ((Temptype *)data0 + 2 * I)[0] +
                                                    ((Temptype *)data_out + 2 * I)[0];
                ((Temptype *)data_out + 2 * I)[1] = ((Temptype *)data0 + 2 * I)[1] +
                                                    ((Temptype *)data_out + 2 * I)[1];
            }
        }
        Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, Start + Step, End,
                                           Step>::apply(data0, data_out, count);
    };
};

// Base case
template <typename T, typename Temptype, bool Is_Complex, int Start, int End, int Step>
struct Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, Start, End, Step,
                                          true> {
    static inline void apply(const T *, T *, npy_intp) {}
};

// sum_of_products_outstride0_one template
template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_contig_one(int nop, char **dataptr, npy_intp const *NPY_UNUSED(strides),
                           npy_intp count)
{
    T *data0 = (T *)dataptr[0];
    T *data_out = (T *)dataptr[1];

    NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_one (%d)\n", (int)count);

    /* This is placed before the main loop to make small counts faster */
    if (count < 8) {
        Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, 6, -1, -1>::apply(
                data0, data_out, count);
        return;
    }
    /* Unroll the loop by 8 */
    while (count >= 8) {
        Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, 0, 8, 1>::apply(
                data0, data_out, count);
        data0 += 8;
        data_out += 8;
        count -= 8;
    }

    if (count > 0) {
        Sum_Of_Products_Contig_One_Stepper<T, Temptype, Is_Complex, 6, -1, -1>::apply(
                data0, data_out, count);
    }
    return;
}

template <typename T, typename Temptype, int NOP>
static inline void
complex_sum_of_products_contig(char **dataptr, npy_intp count)
{
    while (count--) {
        Temptype re, im, tmp;
        int i;
        re = ((Temptype *)dataptr[0])[0];
        im = ((Temptype *)dataptr[0])[1];
        for (i = 1; i < NOP; ++i) {
            tmp = re * ((Temptype *)dataptr[i])[0] - im * ((Temptype *)dataptr[i])[1];
            im = re * ((Temptype *)dataptr[i])[1] + im * ((Temptype *)dataptr[i])[0];
            re = tmp;
        }
        ((Temptype *)dataptr[NOP])[0] = re + ((Temptype *)dataptr[NOP])[0];
        ((Temptype *)dataptr[NOP])[1] = im + ((Temptype *)dataptr[NOP])[1];

        for (i = 0; i <= NOP; ++i) {
            dataptr[i] += sizeof(T);
        }
    }
}

template <typename SimdType>
static NPY_GCC_OPT_3 void
floating_point_sum_of_products_contig_two(const typename SimdType::T *data0,
                                          const typename SimdType::T *data1,
                                          typename SimdType::T *data_out,
                                          npy_intp count)
{
    using SimdReg = typename SimdType::SimdReg;
    /* Use aligned instructions if possible */
    const int is_aligned = EINSUM_IS_ALIGNED(data0) && EINSUM_IS_ALIGNED(data1) &&
                           EINSUM_IS_ALIGNED(data_out);
    const int vstep = SimdType::npyv_nlanes();
    const npy_intp vstepx4 = vstep * 4;

    for (; count >= vstepx4;
         count -= vstepx4, data0 += vstepx4, data1 += vstepx4, data_out += vstepx4) {
        const SimdReg a0 = SimdType::load_any(data0 + vstep * 0, is_aligned);
        const SimdReg b0 = SimdType::load_any(data1 + vstep * 0, is_aligned);
        const SimdReg c0 = SimdType::load_any(data_out + vstep * 0, is_aligned);
        const SimdReg a1 = SimdType::load_any(data0 + vstep * 1, is_aligned);
        const SimdReg b1 = SimdType::load_any(data1 + vstep * 1, is_aligned);
        const SimdReg c1 = SimdType::load_any(data_out + vstep * 1, is_aligned);
        const SimdReg a2 = SimdType::load_any(data0 + vstep * 2, is_aligned);
        const SimdReg b2 = SimdType::load_any(data1 + vstep * 2, is_aligned);
        const SimdReg c2 = SimdType::load_any(data_out + vstep * 2, is_aligned);
        const SimdReg a3 = SimdType::load_any(data0 + vstep * 3, is_aligned);
        const SimdReg b3 = SimdType::load_any(data1 + vstep * 3, is_aligned);
        const SimdReg c3 = SimdType::load_any(data_out + vstep * 3, is_aligned);

        const SimdReg abc0 = SimdType::npyv_muladd(a0, b0, c0);
        const SimdReg abc1 = SimdType::npyv_muladd(a1, b1, c1);
        const SimdReg abc2 = SimdType::npyv_muladd(a2, b2, c2);
        const SimdReg abc3 = SimdType::npyv_muladd(a3, b3, c3);

        SimdType::npyv_st(data_out + vstep * 0, abc0, is_aligned);
        SimdType::npyv_st(data_out + vstep * 1, abc1, is_aligned);
        SimdType::npyv_st(data_out + vstep * 2, abc2, is_aligned);
        SimdType::npyv_st(data_out + vstep * 3, abc3, is_aligned);
    }

    for (; count > 0;
         count -= vstep, data0 += vstep, data1 += vstep, data_out += vstep) {
        SimdReg a = SimdType::load_tillz(data0, count);
        SimdReg b = SimdType::load_tillz(data1, count);
        SimdReg c = SimdType::load_tillz(data_out, count);
        SimdType::npyv_store_till(data_out, count, SimdType::npyv_muladd(a, b, c));
    }
    SimdType::cleanup();
}

template <typename T, typename Temptype>
static void
scaller_sum_of_products_contig_two(const T *data0, const T *data1, npy_intp count,
                                   T *data_out)
{
#ifndef NPY_DISABLE_OPTIMIZATION
    for (; count >= 4; count -= 4, data0 += 4, data1 += 4, data_out += 4) {
        const Temptype a0 = from<T, Temptype>(data0[0]);
        const Temptype b0 = from<T, Temptype>(data1[0]);
        const Temptype c0 = from<T, Temptype>(data_out[0]);

        const Temptype a1 = from<T, Temptype>(data0[1]);
        const Temptype b1 = from<T, Temptype>(data1[1]);
        const Temptype c1 = from<T, Temptype>(data_out[1]);

        const Temptype a2 = from<T, Temptype>(data0[2]);
        const Temptype b2 = from<T, Temptype>(data1[2]);
        const Temptype c2 = from<T, Temptype>(data_out[2]);

        const Temptype a3 = from<T, Temptype>(data0[3]);
        const Temptype b3 = from<T, Temptype>(data1[3]);
        const Temptype c3 = from<T, Temptype>(data_out[3]);

        const Temptype abc0 = a0 * b0 + c0;
        const Temptype abc1 = a1 * b1 + c1;
        const Temptype abc2 = a2 * b2 + c2;
        const Temptype abc3 = a3 * b3 + c3;

        data_out[0] = to<Temptype, T>(abc0);
        data_out[1] = to<Temptype, T>(abc1);
        data_out[2] = to<Temptype, T>(abc2);
        data_out[3] = to<Temptype, T>(abc3);
    }
#endif  // !NPY_DISABLE_OPTIMIZATION
    for (; count > 0; --count, ++data0, ++data1, ++data_out) {
        const Temptype a = from<T, Temptype>(*data0);
        const Temptype b = from<T, Temptype>(*data1);
        const Temptype c = from<T, Temptype>(*data_out);
        *data_out = to<Temptype, T>(a * b + c);
    }
}

/* Template where (npy_float, npy_float) || (npy_double, npy_double) will allow the SIMD
 * capable version.*/
template <typename T, typename Temptype, bool Is_Complex, bool Is_logical,
          typename std::enable_if<std::is_same<T, Temptype>::value &&
                                          (std::is_same<T, npy_float>::value ||
                                           std::is_same<T, npy_double>::value),
                                  int>::type = 0>
static void
sum_of_products_contig_two(int nop, char **dataptr, npy_intp const *NPY_UNUSED(strides),
                           npy_intp count)
{
    if constexpr (!Is_Complex) {
        T *data0 = (T *)dataptr[0];
        T *data1 = (T *)dataptr[1];
        T *data_out = (T *)dataptr[2];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_two (%d)\n", (int)count);

#if (NPY_SIMD_F32 || NPY_SIMD_F64)

        floating_point_sum_of_products_contig_two<
                typename SumSIMD<T, Temptype>::SimdType>(data0, data1, data_out, count);

#else   // !(NPY_SIMD_F32 || NPY_SIMD_F64)
        scaller_sum_of_products_contig_two<T, Temptype>(data0, data1, count, data_out);
#endif  // (NPY_SIMD_F32 || NPY_SIMD_F64)
    }
    else {  // complex
        complex_sum_of_products_contig<T, Temptype, 2>(dataptr, count);
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical,
          typename std::enable_if<!(std::is_same<T, Temptype>::value &&
                                    (std::is_same<T, npy_float>::value ||
                                     std::is_same<T, npy_double>::value)),
                                  int>::type = 0>
static void
sum_of_products_contig_two(int nop, char **dataptr, npy_intp const *NPY_UNUSED(strides),
                           npy_intp count)
{
    if constexpr (!Is_Complex) {
        T *data0 = (T *)dataptr[0];
        T *data1 = (T *)dataptr[1];
        T *data_out = (T *)dataptr[2];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_two (%d)\n", (int)count);

        scaller_sum_of_products_contig_two<T, Temptype>(data0, data1, count, data_out);
    }
    else {  // complex
        complex_sum_of_products_contig<T, Temptype, 2>(dataptr, count);
    }
}

// forward declaration
template <typename T, typename Temptype, int Start, int End, int Step,
          bool Done = (Start == End)>
struct Sum_Of_Products_Contig_Three_Stepper;

// Recursive case
template <typename T, typename Temptype, int Start, int End, int Step>
struct Sum_Of_Products_Contig_Three_Stepper<T, Temptype, Start, End, Step, false> {
    static inline void apply(const T *data0, const T *data1, const T *data2,
                             T *data_out, npy_intp count)
    {
        constexpr int I = Start;

        if (count > I) {
            data_out[I] = to<Temptype, T>(from<T, Temptype>(data0[I]) *
                                                  from<T, Temptype>(data1[I]) *
                                                  from<T, Temptype>(data2[I]) +
                                          from<T, Temptype>(data_out[I]));
        }
        Sum_Of_Products_Contig_Three_Stepper<T, Temptype, Start + Step, End,
                                             Step>::apply(data0, data1, data2, data_out,
                                                          count);
    };
};

// Base case
template <typename T, typename Temptype, int Start, int End, int Step>
struct Sum_Of_Products_Contig_Three_Stepper<T, Temptype, Start, End, Step, true> {
    static inline void apply(const T *, const T *, const T *, T *, npy_intp) {}
};

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_contig_three(int nop, char **dataptr,
                             npy_intp const *NPY_UNUSED(strides), npy_intp count)
{
    if constexpr (!Is_Complex) {
        T *data0 = (T *)dataptr[0];
        T *data1 = (T *)dataptr[1];
        T *data2 = (T *)dataptr[2];
        T *data_out = (T *)dataptr[3];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_three (%d)\n", (int)count);

        /* Unroll the loop by 8 */
        while (count >= 8) {
            Sum_Of_Products_Contig_Three_Stepper<T, Temptype, 0, 8, 1>::apply(
                    data0, data1, data2, data_out, count);
            data0 += 8;
            data1 += 8;
            data2 += 8;
            data_out += 8;
            count -= 8;
        }
       
        if(count > 0){
             Sum_Of_Products_Contig_Three_Stepper<T, Temptype, 0, 8, 1>::apply(
                data0, data1, data2, data_out, count);
        }
       return;
    }
    else {  // complex
        complex_sum_of_products_contig<T, Temptype, 3>(dataptr, count);
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_contig_any(int nop, char **dataptr, npy_intp const *NPY_UNUSED(strides),
                           npy_intp count)
{
    NPY_EINSUM_DBG_PRINT1("sum_of_products_contig_any (%d)\n", (int)count);
    if constexpr (!Is_Complex) {
        while (count--) {
            Temptype temp = from<T, Temptype>(*(T *)dataptr[0]);
            int i;
            for (i = 1; i < nop; ++i) {
                temp *= from<T, Temptype>(*(T *)dataptr[i]);
            }
            *(T *)dataptr[nop] =
                    to<Temptype, T>(temp + from<T, Temptype>(*(T *)dataptr[i]));
            for (i = 0; i <= nop; ++i) {
                dataptr[i] += sizeof(T);
            }
        }
    }
    else {  // complex
        while (count--) {
            Temptype re, im, tmp;
            int i;
            re = ((Temptype *)dataptr[0])[0];
            im = ((Temptype *)dataptr[0])[1];
            for (i = 1; i < nop; ++i) {
                tmp = re * ((Temptype *)dataptr[i])[0] -
                      im * ((Temptype *)dataptr[i])[1];
                im = re * ((Temptype *)dataptr[i])[1] +
                     im * ((Temptype *)dataptr[i])[0];
                re = tmp;
            }
            ((Temptype *)dataptr[nop])[0] = re + ((Temptype *)dataptr[nop])[0];
            ((Temptype *)dataptr[nop])[1] = im + ((Temptype *)dataptr[nop])[1];

            for (i = 0; i <= nop; ++i) {
                dataptr[i] += sizeof(T);
            }
        }
    }
}

template <>
inline void
sum_of_products_contig_any<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                             npy_intp const *strides,
                                                             npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_contig_one<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                             npy_intp const *strides,
                                                             npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_contig_two<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                             npy_intp const *strides,
                                                             npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_contig_three<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                               npy_intp const *strides,
                                                               npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_contig_any<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                            npy_intp const *strides,
                                                            npy_intp count)
{
    while (count--) {
        npy_bool temp = *(npy_bool *)dataptr[0];
        int i;
        for (i = 1; i < nop; ++i) {
            temp = temp && *(npy_bool *)dataptr[i];
        }
        *(npy_bool *)dataptr[nop] = temp || *(npy_bool *)dataptr[i];
        for (i = 0; i <= nop; ++i) {
            dataptr[i] += sizeof(npy_bool);
        }
    }
}

static inline void
bool_sum_of_products_contig_one(char *data0, char *data_out, npy_intp count)
{
    switch (count) {
        case 7:
            ((npy_bool *)data_out)[6] =
                    ((npy_bool *)data0)[6] || ((npy_bool *)data_out)[6];
        case 6:
            ((npy_bool *)data_out)[5] =
                    ((npy_bool *)data0)[5] || ((npy_bool *)data_out)[5];
        case 5:
            ((npy_bool *)data_out)[4] =
                    ((npy_bool *)data0)[4] || ((npy_bool *)data_out)[4];
        case 4:
            ((npy_bool *)data_out)[3] =
                    ((npy_bool *)data0)[3] || ((npy_bool *)data_out)[3];
        case 3:
            ((npy_bool *)data_out)[2] =
                    ((npy_bool *)data0)[2] || ((npy_bool *)data_out)[2];
        case 2:
            ((npy_bool *)data_out)[1] =
                    ((npy_bool *)data0)[1] || ((npy_bool *)data_out)[1];
        case 1:
            ((npy_bool *)data_out)[0] =
                    ((npy_bool *)data0)[0] || ((npy_bool *)data_out)[0];
        case 0:
            return;
    }
}

template <>
inline void
sum_of_products_contig_one<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                            npy_intp const *strides,
                                                            npy_intp count)
{
    char *data0 = dataptr[0];
    char *data_out = dataptr[1];
    /* This is placed before the main loop to make small counts faster */
    if (count < 8) {
        bool_sum_of_products_contig_one(data0, data_out, count);
        return;
    }
    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;
        *((npy_bool *)data_out + 0) =
                (*((npy_bool *)data0 + 0)) || (*((npy_bool *)data_out + 0));
        *((npy_bool *)data_out + 1) =
                (*((npy_bool *)data0 + 1)) || (*((npy_bool *)data_out + 1));
        *((npy_bool *)data_out + 2) =
                (*((npy_bool *)data0 + 2)) || (*((npy_bool *)data_out + 2));
        *((npy_bool *)data_out + 3) =
                (*((npy_bool *)data0 + 3)) || (*((npy_bool *)data_out + 3));
        *((npy_bool *)data_out + 4) =
                (*((npy_bool *)data0 + 4)) || (*((npy_bool *)data_out + 4));
        *((npy_bool *)data_out + 5) =
                (*((npy_bool *)data0 + 5)) || (*((npy_bool *)data_out + 5));
        *((npy_bool *)data_out + 6) =
                (*((npy_bool *)data0 + 6)) || (*((npy_bool *)data_out + 6));
        *((npy_bool *)data_out + 7) =
                (*((npy_bool *)data0 + 7)) || (*((npy_bool *)data_out + 7));

        data0 += 8 * sizeof(npy_bool);
        data_out += 8 * sizeof(npy_bool);
    }
    if (count > 0) {
        bool_sum_of_products_contig_one(data0, data_out, count);
    }
    return;
}

static inline void
bool_sum_of_products_contig_two(char *data0, char *data1, char *data_out,
                                npy_intp count)
{
    switch (count) {
        case 7:
            ((npy_bool *)data_out)[6] =
                    (((npy_bool *)data0)[6] && ((npy_bool *)data1)[6]) ||
                    ((npy_bool *)data_out)[6];
        case 6:
            ((npy_bool *)data_out)[5] =
                    (((npy_bool *)data0)[5] && ((npy_bool *)data1)[5]) ||
                    ((npy_bool *)data_out)[5];
        case 5:
            ((npy_bool *)data_out)[4] =
                    (((npy_bool *)data0)[4] && ((npy_bool *)data1)[4]) ||
                    ((npy_bool *)data_out)[4];
        case 4:
            ((npy_bool *)data_out)[3] =
                    (((npy_bool *)data0)[3] && ((npy_bool *)data1)[3]) ||
                    ((npy_bool *)data_out)[3];
        case 3:
            ((npy_bool *)data_out)[2] =
                    (((npy_bool *)data0)[2] && ((npy_bool *)data1)[2]) ||
                    ((npy_bool *)data_out)[2];
        case 2:
            ((npy_bool *)data_out)[1] =
                    (((npy_bool *)data0)[1] && ((npy_bool *)data1)[1]) ||
                    ((npy_bool *)data_out)[1];
        case 1:
            ((npy_bool *)data_out)[0] =
                    (((npy_bool *)data0)[0] && ((npy_bool *)data1)[0]) ||
                    ((npy_bool *)data_out)[0];
        case 0:
            return;
    }
}

template <>
inline void
sum_of_products_contig_two<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                            npy_intp const *strides,
                                                            npy_intp count)
{
    char *data0 = dataptr[0];
    char *data1 = dataptr[1];
    char *data_out = dataptr[2];
    /* This is placed before the main loop to make small counts faster */
    if (count < 8) {
        bool_sum_of_products_contig_two(data0, data1, data_out, count);
        return;
    }
    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;
        *((npy_bool *)data_out + 0) =
                ((*((npy_bool *)data0 + 0)) && (*((npy_bool *)data1 + 0))) ||
                (*((npy_bool *)data_out + 0));
        *((npy_bool *)data_out + 1) =
                ((*((npy_bool *)data0 + 1)) && (*((npy_bool *)data1 + 1))) ||
                (*((npy_bool *)data_out + 1));
        *((npy_bool *)data_out + 2) =
                ((*((npy_bool *)data0 + 2)) && (*((npy_bool *)data1 + 2))) ||
                (*((npy_bool *)data_out + 2));
        *((npy_bool *)data_out + 3) =
                ((*((npy_bool *)data0 + 3)) && (*((npy_bool *)data1 + 3))) ||
                (*((npy_bool *)data_out + 3));
        *((npy_bool *)data_out + 4) =
                ((*((npy_bool *)data0 + 4)) && (*((npy_bool *)data1 + 4))) ||
                (*((npy_bool *)data_out + 4));
        *((npy_bool *)data_out + 5) =
                ((*((npy_bool *)data0 + 5)) && (*((npy_bool *)data1 + 5))) ||
                (*((npy_bool *)data_out + 5));
        *((npy_bool *)data_out + 6) =
                ((*((npy_bool *)data0 + 6)) && (*((npy_bool *)data1 + 6))) ||
                (*((npy_bool *)data_out + 6));
        *((npy_bool *)data_out + 7) =
                ((*((npy_bool *)data0 + 7)) && (*((npy_bool *)data1 + 7))) ||
                (*((npy_bool *)data_out + 7));
        data0 += 8 * sizeof(npy_bool);
        data1 += 8 * sizeof(npy_bool);
        data_out += 8 * sizeof(npy_bool);
    }

    if (count > 0) {
        bool_sum_of_products_contig_two(data0, data1, data_out, count);
    }
    return;
}

static inline void
bool_sum_of_products_contig_three(char *data0, char *data1, char *data2, char *data_out,
                                  npy_intp count)
{
    switch (count) {
        case 7:
            ((npy_bool *)data_out)[6] =
                    (((npy_bool *)data0)[6] && ((npy_bool *)data1)[6] &&
                     ((npy_bool *)data2)[6]) ||
                    ((npy_bool *)data_out)[6];
        case 6:
            ((npy_bool *)data_out)[5] =
                    (((npy_bool *)data0)[5] && ((npy_bool *)data1)[5] &&
                     ((npy_bool *)data2)[5]) ||
                    ((npy_bool *)data_out)[5];
        case 5:
            ((npy_bool *)data_out)[4] =
                    (((npy_bool *)data0)[4] && ((npy_bool *)data1)[4] &&
                     ((npy_bool *)data2)[4]) ||
                    ((npy_bool *)data_out)[4];
        case 4:
            ((npy_bool *)data_out)[3] =
                    (((npy_bool *)data0)[3] && ((npy_bool *)data1)[3] &&
                     ((npy_bool *)data2)[3]) ||
                    ((npy_bool *)data_out)[3];
        case 3:
            ((npy_bool *)data_out)[2] =
                    (((npy_bool *)data0)[2] && ((npy_bool *)data1)[2] &&
                     ((npy_bool *)data2)[2]) ||
                    ((npy_bool *)data_out)[2];
        case 2:
            ((npy_bool *)data_out)[1] =
                    (((npy_bool *)data0)[1] && ((npy_bool *)data1)[1] &&
                     ((npy_bool *)data2)[1]) ||
                    ((npy_bool *)data_out)[1];
        case 1:
            ((npy_bool *)data_out)[0] =
                    (((npy_bool *)data0)[0] && ((npy_bool *)data1)[0] &&
                     ((npy_bool *)data2)[0]) ||
                    ((npy_bool *)data_out)[0];
        case 0:
            return;
    }
}

template <>
inline void
sum_of_products_contig_three<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                              npy_intp const *strides,
                                                              npy_intp count)
{
    char *data0 = dataptr[0];
    char *data1 = dataptr[1];
    char *data2 = dataptr[2];
    char *data_out = dataptr[3];
    /* This is placed before the main loop to make small counts faster */
    if (count < 8) {
        bool_sum_of_products_contig_three(data0, data1, data2, data_out, count);
        return;
    }
    /* Unroll the loop by 8 */
    while (count >= 8) {
        count -= 8;
        *((npy_bool *)data_out + 0) =
                ((*((npy_bool *)data0 + 0)) && (*((npy_bool *)data1 + 0)) &&
                 (*((npy_bool *)data2 + 0))) ||
                (*((npy_bool *)data_out + 0));
        *((npy_bool *)data_out + 1) =
                ((*((npy_bool *)data0 + 1)) && (*((npy_bool *)data1 + 1)) &&
                 (*((npy_bool *)data2 + 1))) ||
                (*((npy_bool *)data_out + 1));
        *((npy_bool *)data_out + 2) =
                ((*((npy_bool *)data0 + 2)) && (*((npy_bool *)data1 + 2)) &&
                 (*((npy_bool *)data2 + 2))) ||
                (*((npy_bool *)data_out + 2));
        *((npy_bool *)data_out + 3) =
                ((*((npy_bool *)data0 + 3)) && (*((npy_bool *)data1 + 3)) &&
                 (*((npy_bool *)data2 + 3))) ||
                (*((npy_bool *)data_out + 3));
        *((npy_bool *)data_out + 4) =
                ((*((npy_bool *)data0 + 4)) && (*((npy_bool *)data1 + 4)) &&
                 (*((npy_bool *)data2 + 4))) ||
                (*((npy_bool *)data_out + 4));
        *((npy_bool *)data_out + 5) =
                ((*((npy_bool *)data0 + 5)) && (*((npy_bool *)data1 + 5)) &&
                 (*((npy_bool *)data2 + 5))) ||
                (*((npy_bool *)data_out + 5));
        *((npy_bool *)data_out + 6) =
                ((*((npy_bool *)data0 + 6)) && (*((npy_bool *)data1 + 6)) &&
                 (*((npy_bool *)data2 + 6))) ||
                (*((npy_bool *)data_out + 6));
        *((npy_bool *)data_out + 7) =
                ((*((npy_bool *)data0 + 7)) && (*((npy_bool *)data1 + 7)) &&
                 (*((npy_bool *)data2 + 7))) ||
                (*((npy_bool *)data_out + 7));

        data0 += 8 * sizeof(npy_bool);
        data1 += 8 * sizeof(npy_bool);
        data2 += 8 * sizeof(npy_bool);
        data_out += 8 * sizeof(npy_bool);
    }
    if (count > 0) {
        bool_sum_of_products_contig_three(data0, data1, data2, data_out, count);
    }
    return;
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_one(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
    char *data_out = dataptr[1];
    npy_intp stride_out = strides[1];

    NPY_EINSUM_DBG_PRINT1("sum_of_products_one (%d)\n", (int)count);

    if constexpr (!Is_Complex) {
        while (count--) {
            *(T *)data_out = to<Temptype, T>(from<T, Temptype>(*(T *)data0) +
                                             from<T, Temptype>(*(T *)data_out));
            data0 += stride0;
            data_out += stride_out;
        }
    }
    else {  // complex
        while (count--) {
            ((Temptype *)data_out)[0] =
                    ((Temptype *)data0)[0] + ((Temptype *)data_out)[0];
            ((Temptype *)data_out)[1] =
                    ((Temptype *)data0)[1] + ((Temptype *)data_out)[1];
            data0 += stride0;
            data_out += stride_out;
        }
    }
}

template <typename T, typename Temptype, int NOP>
static inline void
complex_sum_of_products(char **dataptr, npy_intp count, npy_intp const *strides)
{
    while (count--) {
        Temptype re, im, tmp;
        int i;
        re = ((Temptype *)dataptr[0])[0];
        im = ((Temptype *)dataptr[0])[1];
        for (i = 1; i < NOP; ++i) {
            tmp = re * ((Temptype *)dataptr[i])[0] - im * ((Temptype *)dataptr[i])[1];
            im = re * ((Temptype *)dataptr[i])[1] + im * ((Temptype *)dataptr[i])[0];
            re = tmp;
        }
        ((Temptype *)dataptr[NOP])[0] = re + ((Temptype *)dataptr[NOP])[0];
        ((Temptype *)dataptr[NOP])[1] = im + ((Temptype *)dataptr[NOP])[1];

        for (i = 0; i <= NOP; ++i) {
            dataptr[i] += strides[i];
        }
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_two(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    if constexpr (!Is_Complex) {
        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];
        char *data1 = dataptr[1];
        npy_intp stride1 = strides[1];

        char *data_out = dataptr[2];
        npy_intp stride_out = strides[2];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_two (%d)\n", (int)count);

        while (count--) {
            *(T *)data_out = to<Temptype, T>(from<T, Temptype>(*(T *)data0) *
                                                     from<T, Temptype>(*(T *)data1) +
                                             from<T, Temptype>(*(T *)data_out));
            data0 += stride0;
            data1 += stride1;
            data_out += stride_out;
        }
    }
    else {  // complex
        complex_sum_of_products<T, Temptype, 2>(dataptr, count, strides);
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_three(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    if constexpr (!Is_Complex) {
        char *data0 = dataptr[0];
        npy_intp stride0 = strides[0];
        char *data1 = dataptr[1];
        npy_intp stride1 = strides[1];
        char *data2 = dataptr[2];
        npy_intp stride2 = strides[2];

        char *data_out = dataptr[3];
        npy_intp stride_out = strides[3];

        NPY_EINSUM_DBG_PRINT1("sum_of_products_three (%d)\n", (int)count);

        while (count--) {
            *(T *)data_out = to<Temptype, T>(from<T, Temptype>(*(T *)data0) *
                                                     from<T, Temptype>(*(T *)data1) *
                                                     from<T, Temptype>(*(T *)data2) +
                                             from<T, Temptype>(*(T *)data_out));
            data0 += stride0;
            data1 += stride1;
            data2 += stride2;
            data_out += stride_out;
        }
    }
    else {  // complex
        complex_sum_of_products<T, Temptype, 3>(dataptr, count, strides);
    }
}

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
static void
sum_of_products_any(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
{
    if constexpr (!Is_Complex) {
        while (count--) {
            Temptype temp = from<T, Temptype>(*(T *)dataptr[0]);
            int i;
            for (i = 1; i < nop; ++i) {
                temp *= from<T, Temptype>(*(T *)dataptr[i]);
            }
            *(T *)dataptr[nop] =
                    to<Temptype, T>(temp + from<T, Temptype>(*(T *)dataptr[i]));
            for (i = 0; i <= nop; ++i) {
                dataptr[i] += strides[i];
            }
        }
    }
    else {
        while (count--) {
            Temptype re, im, tmp;
            int i;
            re = ((Temptype *)dataptr[0])[0];
            im = ((Temptype *)dataptr[0])[1];
            for (i = 1; i < nop; ++i) {
                tmp = re * ((Temptype *)dataptr[i])[0] -
                      im * ((Temptype *)dataptr[i])[1];
                im = re * ((Temptype *)dataptr[i])[1] +
                     im * ((Temptype *)dataptr[i])[0];
                re = tmp;
            }
            ((Temptype *)dataptr[nop])[0] = re + ((Temptype *)dataptr[nop])[0];
            ((Temptype *)dataptr[nop])[1] = im + ((Temptype *)dataptr[nop])[1];

            for (i = 0; i <= nop; ++i) {
                dataptr[i] += strides[i];
            }
        }
    }
}

template <>
inline void
sum_of_products_any<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                      npy_intp const *strides,
                                                      npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_one<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                      npy_intp const *strides,
                                                      npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_two<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                      npy_intp const *strides,
                                                      npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

template <>
inline void
sum_of_products_three<PyObject, PyObject, false, false>(int nop, char **dataptr,
                                                        npy_intp const *strides,
                                                        npy_intp count)
{
    object_sum_of_products(nop, dataptr, strides, count);
}

/* Do OR of ANDs for the boolean type */
template <>
inline void
sum_of_products_any<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                     npy_intp const *strides,
                                                     npy_intp count)
{
    npy_bool temp = *(npy_bool *)dataptr[0];
    int i;
    for (i = 1; i < nop; ++i) {
        temp = temp && *(npy_bool *)dataptr[i];
    }
    *(npy_bool *)dataptr[nop] = temp || *(npy_bool *)dataptr[i];
    for (i = 0; i <= nop; ++i) {
        dataptr[i] += strides[i];
    }
}

/* Do OR of ANDs for the boolean type */
template <>
inline void
sum_of_products_one<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                     npy_intp const *strides,
                                                     npy_intp count)
{
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
    char *data_out = dataptr[1];
    npy_intp stride_out = strides[1];
    while (count--) {
        *(npy_bool *)data_out = *(npy_bool *)data0 || *(npy_bool *)data_out;
        data0 += stride0;
        data_out += stride_out;
    }
}

/* Do OR of ANDs for the boolean type */
template <>
inline void
sum_of_products_two<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                     npy_intp const *strides,
                                                     npy_intp count)
{
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
    char *data_out = dataptr[2];
    npy_intp stride_out = strides[2];
    while (count--) {
        *(npy_bool *)data_out =
                (*(npy_bool *)data0 && *(npy_bool *)data1) || *(npy_bool *)data_out;
        data0 += stride0;
        data1 += stride1;
        data_out += stride_out;
    }
}

/* Do OR of ANDs for the boolean type */
template <>
inline void
sum_of_products_three<npy_bool, npy_bool, false, true>(int nop, char **dataptr,
                                                       npy_intp const *strides,
                                                       npy_intp count)
{
    char *data0 = dataptr[0];
    npy_intp stride0 = strides[0];
    char *data1 = dataptr[1];
    npy_intp stride1 = strides[1];
    char *data2 = dataptr[2];
    npy_intp stride2 = strides[2];
    char *data_out = dataptr[3];
    npy_intp stride_out = strides[3];

    while (count--) {
        *(npy_bool *)data_out =
                (*(npy_bool *)data0 && *(npy_bool *)data1 && *(npy_bool *)data2) ||
                *(npy_bool *)data_out;
        data0 += stride0;
        data1 += stride1;
        data2 += stride2;
        data_out += stride_out;
    }
}

inline constexpr std::array<sum_of_products_fn, NPY_NTYPES_LEGACY>
        contig_outstride0_unary_specialization_table = []() constexpr {
            std::array<sum_of_products_fn, NPY_NTYPES_LEGACY> t{};
            t[NPY_BYTE] =
                    &sum_of_products_contig_outstride0_one<npy_byte, npy_byte, false>;
            t[NPY_UBYTE] =
                    &sum_of_products_contig_outstride0_one<npy_ubyte, npy_ubyte, false>;
            t[NPY_SHORT] =
                    &sum_of_products_contig_outstride0_one<npy_short, npy_short, false>;
            t[NPY_USHORT] = &sum_of_products_contig_outstride0_one<npy_ushort,
                                                                   npy_ushort, false>;
            t[NPY_INT] =
                    &sum_of_products_contig_outstride0_one<npy_int, npy_int, false>;
            t[NPY_UINT] =
                    &sum_of_products_contig_outstride0_one<npy_uint, npy_uint, false>;
            t[NPY_LONG] =
                    &sum_of_products_contig_outstride0_one<npy_long, npy_long, false>;
            t[NPY_ULONG] =
                    &sum_of_products_contig_outstride0_one<npy_ulong, npy_ulong, false>;
            t[NPY_LONGLONG] =
                    &sum_of_products_contig_outstride0_one<npy_longlong, npy_longlong,
                                                           false>;
            t[NPY_ULONGLONG] =
                    &sum_of_products_contig_outstride0_one<npy_ulonglong, npy_ulonglong,
                                                           false>;
            t[NPY_FLOAT] =
                    &sum_of_products_contig_outstride0_one<npy_float, npy_float, false>;
            t[NPY_DOUBLE] = &sum_of_products_contig_outstride0_one<npy_double,
                                                                   npy_double, false>;
            t[NPY_LONGDOUBLE] =
                    &sum_of_products_contig_outstride0_one<npy_longdouble,
                                                           npy_longdouble, false>;
            t[NPY_CFLOAT] =
                    &sum_of_products_contig_outstride0_one<npy_cfloat, npy_float, true>;
            t[NPY_CDOUBLE] = &sum_of_products_contig_outstride0_one<npy_cdouble,
                                                                    npy_double, true>;
            t[NPY_CLONGDOUBLE] =
                    &sum_of_products_contig_outstride0_one<npy_clongdouble,
                                                           npy_longdouble, true>;
            t[NPY_HALF] =
                    &sum_of_products_contig_outstride0_one<npy_half, npy_float, false>;
            return t;
        }();

template <typename T, typename Temptype>
constexpr std::array<sum_of_products_fn, 5>
make_binary_specialization_table_row()
{
    return {&sum_of_products_stride0_contig_outstride0_two<T, Temptype>,
            &sum_of_products_stride0_contig_outcontig_two<T, Temptype>,
            &sum_of_products_contig_stride0_outstride0_two<T, Temptype>,
            &sum_of_products_contig_stride0_outcontig_two<T, Temptype>,
            &sum_of_products_contig_contig_outstride0_two<T, Temptype>};
}

inline constexpr std::array<std::array<sum_of_products_fn, 5>, NPY_NTYPES_LEGACY>
        binary_specialization_table = []() constexpr {
            std::array<std::array<sum_of_products_fn, 5>, NPY_NTYPES_LEGACY> t{};
            t[NPY_BYTE] = make_binary_specialization_table_row<npy_byte, npy_byte>();
            t[NPY_UBYTE] = make_binary_specialization_table_row<npy_ubyte, npy_ubyte>();
            t[NPY_SHORT] = make_binary_specialization_table_row<npy_short, npy_short>();
            t[NPY_USHORT] =
                    make_binary_specialization_table_row<npy_ushort, npy_ushort>();
            t[NPY_INT] = make_binary_specialization_table_row<npy_int, npy_int>();
            t[NPY_UINT] = make_binary_specialization_table_row<npy_uint, npy_uint>();
            t[NPY_LONG] = make_binary_specialization_table_row<npy_long, npy_long>();
            t[NPY_ULONG] = make_binary_specialization_table_row<npy_ulong, npy_ulong>();
            t[NPY_LONGLONG] =
                    make_binary_specialization_table_row<npy_longlong, npy_longlong>();
            t[NPY_ULONGLONG] = make_binary_specialization_table_row<npy_ulonglong,
                                                                    npy_ulonglong>();
            t[NPY_FLOAT] = make_binary_specialization_table_row<npy_float, npy_float>();
            t[NPY_DOUBLE] =
                    make_binary_specialization_table_row<npy_double, npy_double>();
            t[NPY_LONGDOUBLE] = make_binary_specialization_table_row<npy_longdouble,
                                                                     npy_longdouble>();
            t[NPY_HALF] = make_binary_specialization_table_row<npy_half, npy_float>();
            return t;
        }();

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
constexpr std::array<sum_of_products_fn, 4>
make_outstride0_specialized_table_row()
{
    return {&sum_of_products_outstride0_any<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_outstride0_one<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_outstride0_two<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_outstride0_three<T, Temptype, Is_Complex, Is_logical>};
}

inline constexpr std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY>
        outstride0_specialized_table = []() constexpr {
            std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY> t{};
            t[NPY_BOOL] = make_outstride0_specialized_table_row<npy_bool, npy_bool,
                                                                false, true>();
            t[NPY_BYTE] = make_outstride0_specialized_table_row<npy_byte, npy_byte,
                                                                false, false>();
            t[NPY_UBYTE] = make_outstride0_specialized_table_row<npy_ubyte, npy_ubyte,
                                                                 false, false>();
            t[NPY_SHORT] = make_outstride0_specialized_table_row<npy_short, npy_short,
                                                                 false, false>();
            t[NPY_USHORT] =
                    make_outstride0_specialized_table_row<npy_ushort, npy_ushort, false,
                                                          false>();
            t[NPY_INT] = make_outstride0_specialized_table_row<npy_int, npy_int, false,
                                                               false>();
            t[NPY_UINT] = make_outstride0_specialized_table_row<npy_uint, npy_uint,
                                                                false, false>();
            t[NPY_LONG] = make_outstride0_specialized_table_row<npy_long, npy_long,
                                                                false, false>();
            t[NPY_ULONG] = make_outstride0_specialized_table_row<npy_ulong, npy_ulong,
                                                                 false, false>();
            t[NPY_LONGLONG] =
                    make_outstride0_specialized_table_row<npy_longlong, npy_longlong,
                                                          false, false>();
            t[NPY_ULONGLONG] =
                    make_outstride0_specialized_table_row<npy_ulonglong, npy_ulonglong,
                                                          false, false>();
            t[NPY_FLOAT] = make_outstride0_specialized_table_row<npy_float, npy_float,
                                                                 false, false>();
            t[NPY_DOUBLE] =
                    make_outstride0_specialized_table_row<npy_double, npy_double, false,
                                                          false>();
            t[NPY_LONGDOUBLE] = make_outstride0_specialized_table_row<
                    npy_longdouble, npy_longdouble, false, false>();
            t[NPY_CFLOAT] = make_outstride0_specialized_table_row<npy_cfloat, npy_float,
                                                                  true, false>();
            t[NPY_CDOUBLE] =
                    make_outstride0_specialized_table_row<npy_cdouble, npy_double, true,
                                                          false>();
            t[NPY_CLONGDOUBLE] = make_outstride0_specialized_table_row<
                    npy_clongdouble, npy_longdouble, true, false>();
            t[NPY_OBJECT] = make_outstride0_specialized_table_row<PyObject, PyObject,
                                                                  false, false>();
            t[NPY_HALF] = make_outstride0_specialized_table_row<npy_half, npy_float,
                                                                false, false>();
            return t;
        }();

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
constexpr std::array<sum_of_products_fn, 4>
make_allcontig_specialized_table_row()
{
    return {&sum_of_products_contig_any<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_contig_one<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_contig_two<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_contig_three<T, Temptype, Is_Complex, Is_logical>};
}

inline constexpr std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY>
        allcontig_specialized_table = []() constexpr {
            std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY> t{};
            t[NPY_BOOL] = make_allcontig_specialized_table_row<npy_bool, npy_bool,
                                                               false, true>();
            t[NPY_BYTE] = make_allcontig_specialized_table_row<npy_byte, npy_byte,
                                                               false, false>();
            t[NPY_UBYTE] = make_allcontig_specialized_table_row<npy_ubyte, npy_ubyte,
                                                                false, false>();
            t[NPY_SHORT] = make_allcontig_specialized_table_row<npy_short, npy_short,
                                                                false, false>();
            t[NPY_USHORT] = make_allcontig_specialized_table_row<npy_ushort, npy_ushort,
                                                                 false, false>();
            t[NPY_INT] = make_allcontig_specialized_table_row<npy_int, npy_int, false,
                                                              false>();
            t[NPY_UINT] = make_allcontig_specialized_table_row<npy_uint, npy_uint,
                                                               false, false>();
            t[NPY_LONG] = make_allcontig_specialized_table_row<npy_long, npy_long,
                                                               false, false>();
            t[NPY_ULONG] = make_allcontig_specialized_table_row<npy_ulong, npy_ulong,
                                                                false, false>();
            t[NPY_LONGLONG] =
                    make_allcontig_specialized_table_row<npy_longlong, npy_longlong,
                                                         false, false>();
            t[NPY_ULONGLONG] =
                    make_allcontig_specialized_table_row<npy_ulonglong, npy_ulonglong,
                                                         false, false>();
            t[NPY_FLOAT] = make_allcontig_specialized_table_row<npy_float, npy_float,
                                                                false, false>();
            t[NPY_DOUBLE] = make_allcontig_specialized_table_row<npy_double, npy_double,
                                                                 false, false>();
            t[NPY_LONGDOUBLE] =
                    make_allcontig_specialized_table_row<npy_longdouble, npy_longdouble,
                                                         false, false>();
            t[NPY_CFLOAT] = make_allcontig_specialized_table_row<npy_cfloat, npy_float,
                                                                 true, false>();
            t[NPY_CDOUBLE] =
                    make_allcontig_specialized_table_row<npy_cdouble, npy_double, true,
                                                         false>();
            t[NPY_CLONGDOUBLE] =
                    make_allcontig_specialized_table_row<npy_clongdouble,
                                                         npy_longdouble, true, false>();
            t[NPY_OBJECT] = make_allcontig_specialized_table_row<PyObject, PyObject,
                                                                 false, false>();
            t[NPY_HALF] = make_allcontig_specialized_table_row<npy_half, npy_float,
                                                               false, false>();
            return t;
        }();

template <typename T, typename Temptype, bool Is_Complex, bool Is_logical>
constexpr std::array<sum_of_products_fn, 4>
make_unspecialized_table_row()
{
    return {&sum_of_products_any<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_one<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_two<T, Temptype, Is_Complex, Is_logical>,
            &sum_of_products_three<T, Temptype, Is_Complex, Is_logical>};
}

inline constexpr std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY>
        unspecialized_table = []() constexpr {
            std::array<std::array<sum_of_products_fn, 4>, NPY_NTYPES_LEGACY> t{};
            t[NPY_BOOL] =
                    make_unspecialized_table_row<npy_bool, npy_bool, false, true>();
            t[NPY_BYTE] =
                    make_unspecialized_table_row<npy_byte, npy_byte, false, false>();
            t[NPY_UBYTE] =
                    make_unspecialized_table_row<npy_ubyte, npy_ubyte, false, false>();
            t[NPY_SHORT] =
                    make_unspecialized_table_row<npy_short, npy_short, false, false>();
            t[NPY_USHORT] = make_unspecialized_table_row<npy_ushort, npy_ushort, false,
                                                         false>();
            t[NPY_INT] = make_unspecialized_table_row<npy_int, npy_int, false, false>();
            t[NPY_UINT] =
                    make_unspecialized_table_row<npy_uint, npy_uint, false, false>();
            t[NPY_LONG] =
                    make_unspecialized_table_row<npy_long, npy_long, false, false>();
            t[NPY_ULONG] =
                    make_unspecialized_table_row<npy_ulong, npy_ulong, false, false>();
            t[NPY_LONGLONG] = make_unspecialized_table_row<npy_longlong, npy_longlong,
                                                           false, false>();
            t[NPY_ULONGLONG] =
                    make_unspecialized_table_row<npy_ulonglong, npy_ulonglong, false,
                                                 false>();
            t[NPY_FLOAT] =
                    make_unspecialized_table_row<npy_float, npy_float, false, false>();
            t[NPY_DOUBLE] = make_unspecialized_table_row<npy_double, npy_double, false,
                                                         false>();
            t[NPY_LONGDOUBLE] =
                    make_unspecialized_table_row<npy_longdouble, npy_longdouble, false,
                                                 false>();
            t[NPY_CFLOAT] =
                    make_unspecialized_table_row<npy_cfloat, npy_float, true, false>();
            t[NPY_CDOUBLE] = make_unspecialized_table_row<npy_cdouble, npy_double, true,
                                                          false>();
            t[NPY_CLONGDOUBLE] =
                    make_unspecialized_table_row<npy_clongdouble, npy_longdouble, true,
                                                 false>();
            t[NPY_OBJECT] =
                    make_unspecialized_table_row<PyObject, PyObject, false, false>();
            t[NPY_HALF] =
                    make_unspecialized_table_row<npy_half, npy_float, false, false>();

            return t;
        }();

/*
 * Selects an optimized inner-loop implementation for einsum "sum of products"
 * based on:
 * - `nop`: number of input operands (not counting the output),
 * - `type_num`: NumPy dtype enum
 * - `itemsize`: byte width of one element of the dtype described by `type_num`
 * - `fixed_strides`: strides in bytes
 *
 * The dispatcher prefers highly specialized kernels in the following order:
 * 1. Contiguous input with scalar (stride-0) output.
 * 2. Special binary (nop == 2) patterns.
 * 3. Any pattern with stride-0 output.
 * 4. Fully contiguous inputs + output.
 * 5. Fallback generic unspecialized loops.
 *
 * Returns NULL when no specialization exists for the given dtype. Callers must
 * fall back to a safe generic implementation in that case.
 *
 * Architecture is based on historical einsum_sumprod c.src
 * - T: The storage type
 * - Temptype: The computation type. Small helper function `to` and `from` are used
 *            to centralizes generic use of casting between T and Temptype and
 *            handles special cases like npy_half to npy_float conversion.
 * - Is_Complex: Switches between real and complex arithmetic paths.
 * - Is_logical: Flag that whe combined with T/Temptype allow a single template
 * definition to cover numeric, complex, boolean, and object types, with explicit
 * specializations where object/boolean behavior diverges. Used to avoid conflict whne
 * npy_bool and npy_uint8 resolve to the same type.
 * - Wrappers are used to create a consistent interface between NPY_SIMD_F32 and
 * NPY_SIMD_F64
 *
 */
sum_of_products_fn
get_sum_of_products_function(int nop, int type_num, npy_intp itemsize,
                             npy_intp const *fixed_strides)
{
    int iop;
    if (type_num >= NPY_NTYPES_LEGACY) {
        return NULL;
    }

    /* contiguous reduction */
    if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0) {
        sum_of_products_fn ret = contig_outstride0_unary_specialization_table[type_num];
        if (ret != NULL) {
            return ret;
        }
    }

    /* nop of 2 has more specializations */
    if (nop == 2) {
        /* Encode the zero/contiguous strides */
        int code;
        code = (fixed_strides[0] == 0)          ? 0
               : (fixed_strides[0] == itemsize) ? 2 * 2 * 1
                                                : 8;
        code += (fixed_strides[1] == 0)          ? 0
                : (fixed_strides[1] == itemsize) ? 2 * 1
                                                 : 8;
        code += (fixed_strides[2] == 0) ? 0 : (fixed_strides[2] == itemsize) ? 1 : 8;
        if (code >= 2 && code < 7) {
            sum_of_products_fn ret = binary_specialization_table[type_num][code - 2];
            if (ret != NULL) {
                return ret;
            }
        }
    }

    /* Inner loop with an output stride of 0 */
    if (fixed_strides[nop] == 0) {
        return outstride0_specialized_table[type_num][nop <= 3 ? nop : 0];
    }

    /* Check for all contiguous */
    for (iop = 0; iop < nop + 1; ++iop) {
        if (fixed_strides[iop] != itemsize) {
            break;
        }
    }

    /* Contiguous loop */
    if (iop == nop + 1) {
        return allcontig_specialized_table[type_num][nop <= 3 ? nop : 0];
    }

    /* None of the above specializations caught it, general loops */
    return unspecialized_table[type_num][nop <= 3 ? nop : 0];
}
