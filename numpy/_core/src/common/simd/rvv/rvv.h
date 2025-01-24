#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

#define NPY_SIMD 128
#define NPY_SIMD_WIDTH 16
#define NPY_SIMD_F32 0
#define NPY_SIMD_F64 1

#ifdef NPY_HAVE_FMA3
    #define NPY_SIMD_FMA3 1 // native support
#else
    #define NPY_SIMD_FMA3 0 // fast emulated
#endif

#define NPY_SIMD_BIGENDIAN 0
#define NPY_SIMD_CMPSIGNAL 1

typedef vuint8m1_t fixed_vuint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint16m1_t fixed_vuint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint32m1_t fixed_vuint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint64m1_t fixed_vuint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint8m1_t fixed_vint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint16m1_t fixed_vint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_vint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint64m1_t fixed_vint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat32m1_t fixed_vfloat32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t fixed_vfloat64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

#define npyv_u8 fixed_vuint8m1_t
#define npyv_u16 fixed_vuint16m1_t
#define npyv_u32 fixed_vuint32m1_t
#define npyv_u64 fixed_vuint64m1_t
#define npyv_s8 fixed_vint8m1_t
#define npyv_s16 fixed_vint16m1_t
#define npyv_s32 fixed_vint32m1_t
#define npyv_s64 fixed_vint64m1_t
#define npyv_b8 fixed_vuint8m1_t
#define npyv_b16 fixed_vuint16m1_t
#define npyv_b32 fixed_vuint32m1_t
#define npyv_b64 fixed_vuint64m1_t
#define npyv_f32 fixed_vfloat32m1_t
#define npyv_f64 fixed_vfloat64m1_t


typedef struct { fixed_vuint8m1_t val[2]; } npyv_u8x2;
typedef struct { fixed_vint8m1_t val[2]; } npyv_s8x2;
typedef struct { fixed_vuint16m1_t val[2]; } npyv_u16x2;
typedef struct { fixed_vint16m1_t val[2]; } npyv_s16x2;
typedef struct { fixed_vuint32m1_t val[2]; } npyv_u32x2;
typedef struct { fixed_vint32m1_t val[2]; } npyv_s32x2;
typedef struct { fixed_vuint64m1_t val[2]; } npyv_u64x2;
typedef struct { fixed_vint64m1_t val[2]; } npyv_s64x2;
typedef struct { fixed_vfloat32m1_t val[2]; } npyv_f32x2;
typedef struct { fixed_vfloat64m1_t val[2]; } npyv_f64x2;


typedef struct { fixed_vuint8m1_t val[3]; } npyv_u8x3;
typedef struct { fixed_vint8m1_t val[3]; } npyv_s8x3;
typedef struct { fixed_vuint16m1_t val[3]; } npyv_u16x3;
typedef struct { fixed_vint16m1_t val[3]; } npyv_s16x3;
typedef struct { fixed_vuint32m1_t val[3]; } npyv_u32x3;
typedef struct { fixed_vint32m1_t val[3]; } npyv_s32x3;
typedef struct { fixed_vuint64m1_t val[3]; } npyv_u64x3;
typedef struct { fixed_vint64m1_t val[3]; } npyv_s64x3;
typedef struct { fixed_vfloat32m1_t val[3]; } npyv_f32x3;
typedef struct { fixed_vfloat64m1_t val[3]; } npyv_f64x3;

#define npyv_nlanes_u8  16
#define npyv_nlanes_s8  16
#define npyv_nlanes_u16 8
#define npyv_nlanes_s16 8
#define npyv_nlanes_u32 4
#define npyv_nlanes_s32 4
#define npyv_nlanes_u64 2
#define npyv_nlanes_s64 2
#define npyv_nlanes_f32 4
#define npyv_nlanes_f64 2

#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
