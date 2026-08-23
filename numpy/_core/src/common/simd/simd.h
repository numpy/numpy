#ifndef _NPY_SIMD_H_
#define _NPY_SIMD_H_

#include <stdalign.h>  /* for alignof until C23 */
/**
 * the NumPy C SIMD vectorization interface "NPYV" are types and functions intended
 * to simplify vectorization of code on different platforms, currently supports
 * the following SIMD extensions SSE, AVX2, AVX512, VSX and NEON.
 *
 * TODO: Add an independent sphinx doc.
*/
#include "numpy/npy_common.h"
#ifndef __cplusplus
    #include <stdbool.h>
#endif

#include "npy_cpu_dispatch.h"
#include "simd_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
/*
 * clang commit an aggressive optimization behaviour when flag `-ftrapping-math`
 * isn't fully supported that's present at -O1 or greater. When partially loading a
 * vector register for an operation that requires to fill up the remaining lanes
 * with certain value for example divide operation needs to fill the remaining value
 * with non-zero integer to avoid fp exception divide-by-zero.
 * clang optimizer notices that the entire register is not needed for the store
 * and optimizes out the fill of non-zero integer to the remaining
 * elements. As workaround we mark the returned register with `volatile`
 * followed by symmetric operand operation e.g. `or`
 * to convince the compiler that the entire vector is needed.
 */
#if defined(__clang__) && !defined(NPY_HAVE_CLANG_FPSTRICT)
    #define NPY_SIMD_GUARD_PARTIAL_LOAD 1
#else
    #define NPY_SIMD_GUARD_PARTIAL_LOAD 0
#endif

#if defined(_MSC_VER) && defined(_M_IX86)
/*
 * Avoid using any of the following intrinsics with MSVC 32-bit,
 * even if they are apparently work on newer versions.
 * They had bad impact on the generated instructions,
 * sometimes the compiler deal with them without the respect
 * of 32-bit mode which lead to crush due to execute 64-bit
 * instructions and other times generate bad emulated instructions.
 */
    #undef _mm512_set1_epi64
    #undef _mm256_set1_epi64x
    #undef _mm_set1_epi64x
    #undef _mm512_setr_epi64x
    #undef _mm256_setr_epi64x
    #undef _mm_setr_epi64x
    #undef _mm512_set_epi64x
    #undef _mm256_set_epi64x
    #undef _mm_set_epi64x
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

// TODO: Add support for VSX(2.06) and BE Mode for VSX
#if defined(NPY_HAVE_VX) || (defined(NPY_HAVE_VSX2) && defined(__LITTLE_ENDIAN__))
    #include "vec/vec.h"
#endif

#ifdef NPY_HAVE_NEON
    #include "neon/neon.h"
#endif

#ifdef NPY_HAVE_LSX
    #include "lsx/lsx.h"
#endif

#ifndef NPY_SIMD
    /// SIMD width in bits or 0 if there's no SIMD extension available.
    #define NPY_SIMD 0
    /// SIMD width in bytes or 0 if there's no SIMD extension available.
    #define NPY_SIMD_WIDTH 0
    /// 1 if the enabled SIMD extension supports single-precision otherwise 0.
    #define NPY_SIMD_F32 0
    /// 1 if the enabled SIMD extension supports double-precision otherwise 0.
    #define NPY_SIMD_F64 0
    /// 1 if the enabled SIMD extension supports native FMA otherwise 0.
    /// note: we still emulate(fast) FMA intrinsics even if they
    /// aren't supported but they shouldn't be used if the precision is matters.
    #define NPY_SIMD_FMA3 0
    /// 1 if the enabled SIMD extension is running on big-endian mode otherwise 0.
    #define NPY_SIMD_BIGENDIAN 0
    /// 1 if the supported comparison intrinsics(lt, le, gt, ge)
    /// raises FP invalid exception for quite NaNs.
    #define NPY_SIMD_CMPSIGNAL 0
#endif

// enable emulated mask operations for all SIMD extension except for AVX512
#if !defined(NPY_HAVE_AVX512F) && NPY_SIMD && NPY_SIMD < 512
    #include "emulate_maskop.h"
#endif

// enable integer divisor generator for all SIMD extensions
#if NPY_SIMD
    #include "intdiv.h"
#endif

/**
 * Some SIMD extensions currently(AVX2, AVX512F) require (de facto)
 * a maximum number of strides sizes when dealing with non-contiguous memory access.
 *
 * Therefore the following functions must be used to check the maximum
 * acceptable limit of strides before using any of non-contiguous load/store intrinsics.
 *
 * For instance:
 *
 *  if (npyv_loadable_stride_f32(steps[0]) && npyv_storable_stride_f32(steps[1])) {
 *      // Strides are now guaranteed to be a multiple and compatible
 *      npy_intp ld_stride = steps[0] / sizeof(float);
 *      npy_intp st_stride = steps[1] / sizeof(float);
 *      for (;;)
 *          npyv_f32 a = npyv_loadn_f32(ld_pointer, ld_stride);
 *          // ...
 *          npyv_storen_f32(st_pointer, st_stride, a);
 *  }
 *  else {
 *      for (;;)
 *          // C scalars, use byte steps/strides.
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
#define NPYV_IMPL_MAXSTRIDE(SFX, MAXLOAD, MAXSTORE)                         \
    NPY_FINLINE int                                                         \
    npyv_loadable_stride_##SFX(npy_intp stride)                             \
    {                                                                       \
        if (alignof(npyv_lanetype_##SFX) != sizeof(npyv_lanetype_##SFX) &&  \
                stride % sizeof(npyv_lanetype_##SFX) != 0) {                \
            /* stride not a multiple of itemsize, cannot handle. */         \
            return 0;                                                       \
        }                                                                   \
        stride = stride / sizeof(npyv_lanetype_##SFX);                      \
        return MAXLOAD > 0 ? llabs(stride) <= MAXLOAD : 1;                  \
    }                                                                       \
    NPY_FINLINE int                                                         \
    npyv_storable_stride_##SFX(npy_intp stride)                             \
    {                                                                       \
        if (alignof(npyv_lanetype_##SFX) != sizeof(npyv_lanetype_##SFX) &&  \
                stride % sizeof(npyv_lanetype_##SFX) != 0) {                \
            /* stride not a multiple of itemsize, cannot handle. */         \
            return 0;                                                       \
        }                                                                   \
        stride = stride / sizeof(npyv_lanetype_##SFX);                      \
        return MAXSTORE > 0 ? llabs(stride) <= MAXSTORE : 1;                \
    }
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
