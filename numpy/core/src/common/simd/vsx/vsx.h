#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

#define NPY_SIMD 128
#define NPY_SIMD_WIDTH 16
#define NPY_SIMD_F64 1

typedef __vector unsigned char      npyv_u8;
typedef __vector signed char        npyv_s8;
typedef __vector unsigned short     npyv_u16;
typedef __vector signed short       npyv_s16;
typedef __vector unsigned int       npyv_u32;
typedef __vector signed int         npyv_s32;
typedef __vector unsigned long long npyv_u64;
typedef __vector signed long long   npyv_s64;
typedef __vector float              npyv_f32;
typedef __vector double             npyv_f64;

typedef struct { npyv_u8  val[2]; } npyv_u8x2;
typedef struct { npyv_s8  val[2]; } npyv_s8x2;
typedef struct { npyv_u16 val[2]; } npyv_u16x2;
typedef struct { npyv_s16 val[2]; } npyv_s16x2;
typedef struct { npyv_u32 val[2]; } npyv_u32x2;
typedef struct { npyv_s32 val[2]; } npyv_s32x2;
typedef struct { npyv_u64 val[2]; } npyv_u64x2;
typedef struct { npyv_s64 val[2]; } npyv_s64x2;
typedef struct { npyv_f32 val[2]; } npyv_f32x2;
typedef struct { npyv_f64 val[2]; } npyv_f64x2;

typedef struct { npyv_u8  val[3]; } npyv_u8x3;
typedef struct { npyv_s8  val[3]; } npyv_s8x3;
typedef struct { npyv_u16 val[3]; } npyv_u16x3;
typedef struct { npyv_s16 val[3]; } npyv_s16x3;
typedef struct { npyv_u32 val[3]; } npyv_u32x3;
typedef struct { npyv_s32 val[3]; } npyv_s32x3;
typedef struct { npyv_u64 val[3]; } npyv_u64x3;
typedef struct { npyv_s64 val[3]; } npyv_s64x3;
typedef struct { npyv_f32 val[3]; } npyv_f32x3;
typedef struct { npyv_f64 val[3]; } npyv_f64x3;

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

// using __bool with typdef cause ambiguous errors
#define npyv_b8  __vector __bool char
#define npyv_b16 __vector __bool short
#define npyv_b32 __vector __bool int
#define npyv_b64 __vector __bool long long

#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
