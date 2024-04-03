#ifndef _NPY_SIMD_H_
#error "Not a standalone header"
#endif

/* simd_data is defined as the union in core/src/_simd/_simd_inc.h.src.
   It includes vector data as member variables. For RISC-V Vector architecture,
   vector data size is depend on runtime environment.*/
#define NPY_SIMD 128
#define NPY_SIMD_WIDTH (NPY_SIMD / 8)


#define NPY_SIMD_F32 0
#define NPY_SIMD_F64 0
#define NPY_SIMD_FMA3 0
#define NPY_SIMD_BIGENDIAN 0
#define NPY_SIMD_CMPSIGNAL 1

typedef vuint8m1_t  npyv_u8;
typedef vint8m1_t   npyv_s8;
typedef vuint16m1_t  npyv_u16;
typedef vint16m1_t   npyv_s16;
typedef vuint32m1_t  npyv_u32;
typedef vint32m1_t   npyv_s32;
typedef vuint64m1_t  npyv_u64;
typedef vint64m1_t   npyv_s64;
#if NPY_SIMD_F64
typedef vfloat32m1_t npyv_f32;
#endif
#if NPY_SIMD_F64
typedef vfloat64m1_t npyv_f64;
#endif

typedef vuint8m1_t  npyv_b8;
typedef vuint16m1_t  npyv_b16;
typedef vuint32m1_t  npyv_b32;
typedef vuint64m1_t  npyv_b64;

typedef vuint8m1x2_t  npyv_u8x2;
typedef vint8m1x2_t   npyv_s8x2;
typedef vuint16m1x2_t  npyv_u16x2;
typedef vint16m1x2_t   npyv_s16x2;
typedef vuint32m1x2_t  npyv_u32x2;
typedef vint32m1x2_t   npyv_s32x2;
typedef vuint64m1x2_t  npyv_u64x2;
typedef vint64m1x2_t   npyv_s64x2;
typedef vfloat32m1x2_t npyv_f32x2;
#if NPY_SIMD_F64
typedef vfloat64m1x2_t npyv_f64x2;
#endif

typedef vuint8m1x3_t  npyv_u8x3;
typedef vint8m1x3_t   npyv_s8x3;
typedef vuint16m1x3_t  npyv_u16x3;
typedef vint16m1x3_t   npyv_s16x3;
typedef vuint32m1x3_t  npyv_u32x3;
typedef vint32m1x3_t   npyv_s32x3;
typedef vuint64m1x3_t  npyv_u64x3;
typedef vint64m1x3_t   npyv_s64x3;
typedef vfloat32m1x3_t npyv_f32x3;
#if NPY_SIMD_F64
typedef vfloat64m1x3_t npyv_f64x3;
#endif

#define npyv_nlanes_u8  (NPY_SIMD / 8)
#define npyv_nlanes_s8  (NPY_SIMD / 8)
#define npyv_nlanes_u16 (NPY_SIMD / 16)
#define npyv_nlanes_s16 (NPY_SIMD / 16)
#define npyv_nlanes_u32 (NPY_SIMD / 32)
#define npyv_nlanes_s32 (NPY_SIMD / 32)
#define npyv_nlanes_u64 (NPY_SIMD / 64)
#define npyv_nlanes_s64 (NPY_SIMD / 64)
#define npyv_nlanes_f32 (NPY_SIMD / 32)
#define npyv_nlanes_f64 (NPY_SIMD / 64)

#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"


