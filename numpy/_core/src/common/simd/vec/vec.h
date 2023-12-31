/**
 * branch /vec(altivec-like) provides the SIMD operations for
 * both IBM VSX(Power) and VX(ZArch).
*/
#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

#if !defined(NPY_HAVE_VX) && !defined(NPY_HAVE_VSX2)
    #error "require minimum support VX(zarch11) or VSX2(Power8/ISA2.07)"
#endif

#if defined(NPY_HAVE_VSX) && !defined(__LITTLE_ENDIAN__)
    #error "VSX support doesn't cover big-endian mode yet, only zarch."
#endif
#if defined(NPY_HAVE_VX) && defined(__LITTLE_ENDIAN__)
    #error "VX(zarch) support doesn't cover little-endian mode."
#endif

#if defined(__GNUC__) && __GNUC__ <= 7
    /**
      * GCC <= 7 produces ambiguous warning caused by -Werror=maybe-uninitialized,
      * when certain intrinsics involved. `vec_ld` is one of them but it seemed to work fine,
      * and suppressing the warning wouldn't affect its functionality.
      */
    #pragma GCC diagnostic ignored "-Wuninitialized"
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#define NPY_SIMD 128
#define NPY_SIMD_WIDTH 16
#define NPY_SIMD_F64 1
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define NPY_SIMD_F32 1
#else
    #define NPY_SIMD_F32 0
#endif
#define NPY_SIMD_FMA3 1 // native support

#ifdef NPY_HAVE_VX
    #define NPY_SIMD_BIGENDIAN 1
    #define NPY_SIMD_CMPSIGNAL 0
#else
    #define NPY_SIMD_BIGENDIAN 0
    #define NPY_SIMD_CMPSIGNAL 1
#endif

typedef __vector unsigned char      npyv_u8;
typedef __vector signed char        npyv_s8;
typedef __vector unsigned short     npyv_u16;
typedef __vector signed short       npyv_s16;
typedef __vector unsigned int       npyv_u32;
typedef __vector signed int         npyv_s32;
typedef __vector unsigned long long npyv_u64;
typedef __vector signed long long   npyv_s64;
#if NPY_SIMD_F32
typedef __vector float              npyv_f32;
#endif
typedef __vector double             npyv_f64;

typedef struct { npyv_u8  val[2]; } npyv_u8x2;
typedef struct { npyv_s8  val[2]; } npyv_s8x2;
typedef struct { npyv_u16 val[2]; } npyv_u16x2;
typedef struct { npyv_s16 val[2]; } npyv_s16x2;
typedef struct { npyv_u32 val[2]; } npyv_u32x2;
typedef struct { npyv_s32 val[2]; } npyv_s32x2;
typedef struct { npyv_u64 val[2]; } npyv_u64x2;
typedef struct { npyv_s64 val[2]; } npyv_s64x2;
#if NPY_SIMD_F32
typedef struct { npyv_f32 val[2]; } npyv_f32x2;
#endif
typedef struct { npyv_f64 val[2]; } npyv_f64x2;

typedef struct { npyv_u8  val[3]; } npyv_u8x3;
typedef struct { npyv_s8  val[3]; } npyv_s8x3;
typedef struct { npyv_u16 val[3]; } npyv_u16x3;
typedef struct { npyv_s16 val[3]; } npyv_s16x3;
typedef struct { npyv_u32 val[3]; } npyv_u32x3;
typedef struct { npyv_s32 val[3]; } npyv_s32x3;
typedef struct { npyv_u64 val[3]; } npyv_u64x3;
typedef struct { npyv_s64 val[3]; } npyv_s64x3;
#if NPY_SIMD_F32
typedef struct { npyv_f32 val[3]; } npyv_f32x3;
#endif
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

// using __bool with typedef cause ambiguous errors
#define npyv_b8  __vector __bool char
#define npyv_b16 __vector __bool short
#define npyv_b32 __vector __bool int
#define npyv_b64 __vector __bool long long

#include "utils.h"
#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
