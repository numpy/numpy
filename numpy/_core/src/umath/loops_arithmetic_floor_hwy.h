/**
 * Header for Highway SIMD floor divide wrapper functions.
 *
 * Declares extern "C" functions callable from C code. The implementations
 * (in loops_arithmetic_floor_hwy.dispatch.cpp) use NumPy's CPU dispatch
 * to compile per-target variants (SVE for ARM; X86_V4/V3/V2 for x86).
 * Each variant internally uses Highway's HWY_STATIC_DISPATCH for per-op
 * sub-target selection.
 *
 * Two categories of operations:
 * 1. Contig (vector-vector): both inputs are contiguous arrays
 * 2. Scalar divisor: first input is contiguous array, second is a scalar
 *
 * Supports signed (int8, int16, int32, int64) and unsigned (uint8, uint16, uint32).
 * Contig paths: s8, s16, s32, u8, u16, u32
 * Scalar divisor paths: s8, s16, s32, s64, u32
 */

#ifndef _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_
#define _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_

#include "numpy/ndarraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Availability checks — non-dispatched, defined in *_avail.c */
NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_available(int element_size);

NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_unsigned_available(int element_size);

NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_scalar_available(int element_size);

NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_scalar_unsigned_available(int element_size);

/* Signed integer contig (vector-vector) wrappers */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s32_contig(char **args, npy_intp len);

/* Signed integer scalar divisor wrappers */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s8_scalar_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s16_scalar_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s32_scalar_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s64_scalar_contig(char **args, npy_intp len);

/* Unsigned integer contig (vector-vector) wrappers */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u32_contig(char **args, npy_intp len);

/* Unsigned integer scalar divisor wrappers */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u32_scalar_contig(char **args, npy_intp len);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_ */
