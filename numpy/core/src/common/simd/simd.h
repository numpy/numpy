#ifndef _NPY_SIMD_H_
#define _NPY_SIMD_H_
/**
 * the NumPy C SIMD vectorization interface "NPYV" are types and functions intended
 * to simplify vectorization of code on different platforms, currently supports
 * the following SIMD extensions SSE, AVX2, AVX512, VSX and NEON.
 *
 * TODO: Add an independent sphinx doc.
*/
#include "numpy/npy_common.h"
#include "npy_cpu_dispatch.h"
#include "simd_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// lane type by intrin suffix
typedef npy_uint8  npyv_lanetype_u8;
typedef npy_int8   npyv_lanetype_s8;
typedef npy_uint16 npyv_lanetype_u16;
typedef npy_int16  npyv_lanetype_s16;
typedef npy_uint32 npyv_lanetype_u32;
typedef npy_int32  npyv_lanetype_s32;
typedef npy_uint64 npyv_lanetype_u64;
typedef npy_int64  npyv_lanetype_s64;
typedef float      npyv_lanetype_f32;
typedef double     npyv_lanetype_f64;

#if defined(NPY_HAVE_AVX512F) && !defined(NPY_SIMD_FORCE_256) && !defined(NPY_SIMD_FORCE_128)
    #include "avx512/avx512.h"
#elif defined(NPY_HAVE_AVX2) && !defined(NPY_SIMD_FORCE_128)
    #include "avx2/avx2.h"
#elif defined(NPY_HAVE_SSE2)
    #include "sse/sse.h"
#endif

// TODO: Add support for VSX(2.06) and BE Mode
#if defined(NPY_HAVE_VSX2) && defined(__LITTLE_ENDIAN__)
    #include "vsx/vsx.h"
#endif

#ifdef NPY_HAVE_NEON
    #include "neon/neon.h"
#endif

#ifndef NPY_SIMD
    #define NPY_SIMD 0
    #define NPY_SIMD_WIDTH 0
    #define NPY_SIMD_F64 0
    #define NPY_SIMD_FMA3 0
#endif

// enable emulated mask operations for all SIMD extension except for AVX512
#if !defined(NPY_HAVE_AVX512F) && NPY_SIMD && NPY_SIMD < 512
    #include "emulate_maskop.h"
#endif

/**
 * Some SIMD extensions currently(AVX2, AVX512F) require (de facto)
 * a maximum number of strides sizes when dealing with non-contiguous memory access.
 *
 * Therefore the following functions must be used to check the maximum
 * acceptable limit of strides before using any of non-contiguous load/store intrinsics.
 *
 * For instance:
 *  npy_intp ld_stride = step[0] / sizeof(float);
 *  npy_intp st_stride = step[1] / sizeof(float);
 *
 *  if (npyv_loadable_stride_f32(ld_stride) && npyv_storable_stride_f32(st_stride)) {
 *      for (;;)
 *          npyv_f32 a = npyv_loadn_f32(ld_pointer, ld_stride);
 *          // ...
 *          npyv_storen_f32(st_pointer, st_stride, a);
 *  }
 *  else {
 *      for (;;)
 *          // C scalars
 *  }
 */
#ifndef NPY_SIMD_MAXLOAD_STRIDE32
    #define NPY_SIMD_MAXLOAD_STRIDE32 0
#endif
#ifndef NPY_SIMD_MAXSTORE_STRIDE32
    #define NPY_SIMD_MAXSTORE_STRIDE32 0
#endif
#ifndef NPY_SIMD_MAXLOAD_STRIDE64
    #define NPY_SIMD_MAXLOAD_STRIDE64 0
#endif
#ifndef NPY_SIMD_MAXSTORE_STRIDE64
    #define NPY_SIMD_MAXSTORE_STRIDE64 0
#endif
#define NPYV_IMPL_MAXSTRIDE(SFX, MAXLOAD, MAXSTORE) \
    NPY_FINLINE int npyv_loadable_stride_##SFX(npy_intp stride) \
    { return MAXLOAD > 0 ? llabs(stride) <= MAXLOAD : 1; } \
    NPY_FINLINE int npyv_storable_stride_##SFX(npy_intp stride) \
    { return MAXSTORE > 0 ? llabs(stride) <= MAXSTORE : 1; }
#if NPY_SIMD
    NPYV_IMPL_MAXSTRIDE(u32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(s32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(f32, NPY_SIMD_MAXLOAD_STRIDE32, NPY_SIMD_MAXSTORE_STRIDE32)
    NPYV_IMPL_MAXSTRIDE(u64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
    NPYV_IMPL_MAXSTRIDE(s64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
#endif
#if NPY_SIMD_F64
    NPYV_IMPL_MAXSTRIDE(f64, NPY_SIMD_MAXLOAD_STRIDE64, NPY_SIMD_MAXSTORE_STRIDE64)
#endif

#ifdef __cplusplus
}
#endif
#endif // _NPY_SIMD_H_
