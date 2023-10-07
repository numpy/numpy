/*@targets
 ** $maxopt baseline
 ** sse2 sse41
 ** vsx2
 ** neon asimd
 ** vx vxe
 **/
/**
 * Force use SSE only on x86, even if AVX2 or AVX512F are enabled
 * through the baseline, since scatter(AVX512F) and gather very costly
 * to handle non-contiguous memory access comparing with SSE for
 * such small operations that this file covers.
*/
#define NPY_SIMD_FORCE_128
#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
/**********************************************************
 ** Scalars
 **********************************************************/
#if !NPY_SIMD_F32
NPY_FINLINE float c_recip_f32(float a)
{ return 1.0f / a; }
NPY_FINLINE float c_abs_f32(float a)
{
    const float tmp = a > 0 ? a : -a;
    /* add 0 to clear -0.0 */
    return tmp + 0;
}
NPY_FINLINE float c_square_f32(float a)
{ return a * a; }
#endif // !NPY_SIMD_F32

#if !NPY_SIMD_F64
NPY_FINLINE double c_recip_f64(double a)
{ return 1.0 / a; }
NPY_FINLINE double c_abs_f64(double a)
{
    const double tmp = a > 0 ? a : -a;
    /* add 0 to clear -0.0 */
    return tmp + 0;
}
NPY_FINLINE double c_square_f64(double a)
{ return a * a; }
#endif // !NPY_SIMD_F64
/**
 * MSVC(32-bit mode) requires a clarified contiguous loop
 * in order to use SSE, otherwise it uses a soft version of square root
 * that doesn't raise a domain error.
 */
#if defined(_MSC_VER) && defined(_M_IX86) && !NPY_SIMD
    #include <emmintrin.h>
    NPY_FINLINE float c_sqrt_f32(float _a)
    {
        __m128 a = _mm_load_ss(&_a);
        __m128 lower = _mm_sqrt_ss(a);
        return _mm_cvtss_f32(lower);
    }
    NPY_FINLINE double c_sqrt_f64(double _a)
    {
        __m128d a = _mm_load_sd(&_a);
        __m128d lower = _mm_sqrt_pd(a);
        return _mm_cvtsd_f64(lower);
    }
#else
    #define c_sqrt_f32 npy_sqrtf
    #define c_sqrt_f64 npy_sqrt
#endif

#define c_ceil_f32 npy_ceilf
#define c_ceil_f64 npy_ceil

#define c_trunc_f32 npy_truncf
#define c_trunc_f64 npy_trunc

#define c_floor_f32 npy_floorf
#define c_floor_f64 npy_floor

#define c_rint_f32 npy_rintf
#define c_rint_f64 npy_rint

/********************************************************************************
 ** Defining the SIMD kernels
 ********************************************************************************/
/** Notes:
 * - avoid the use of libmath to unify fp/domain errors
 *   for both scalars and vectors among all compilers/architectures.
 * - use intrinsic npyv_load_till_* instead of npyv_load_tillz_
 *   to fill the remind lanes with 1.0 to avoid divide by zero fp
 *   exception in reciprocal.
 */
#define CONTIG  0
#define NCONTIG 1

/**begin repeat
 * #TYPE = FLOAT, DOUBLE#
 * #sfx  = f32, f64#
 * #VCHK = NPY_SIMD_F32, NPY_SIMD_F64#
 */
#if @VCHK@
/**begin repeat1
 * #kind     = rint,  floor, ceil, trunc, sqrt, absolute, square, reciprocal#
 * #intr     = rint,  floor, ceil, trunc, sqrt, abs,      square, recip#
 * #repl_0w1 = 0*7, 1#
 */
/**begin repeat2
 * #STYPE  = CONTIG, NCONTIG, CONTIG,  NCONTIG#
 * #DTYPE  = CONTIG, CONTIG,  NCONTIG, NCONTIG#
 * #unroll = 4,      4,       2,       2#
 */
static void simd_@TYPE@_@kind@_@STYPE@_@DTYPE@
(const void *_src, npy_intp ssrc, void *_dst, npy_intp sdst, npy_intp len)
{
    const npyv_lanetype_@sfx@ *src = _src;
          npyv_lanetype_@sfx@ *dst = _dst;

    const int vstep = npyv_nlanes_@sfx@;
    const int wstep = vstep * @unroll@;

    // unrolled iterations
    for (; len >= wstep; len -= wstep, src += ssrc*wstep, dst += sdst*wstep) {
        /**begin repeat3
         * #N  = 0, 1, 2, 3#
         */
        #if @unroll@ > @N@
            #if @STYPE@ == CONTIG
                npyv_@sfx@ v_src@N@ = npyv_load_@sfx@(src + vstep*@N@);
            #else
                npyv_@sfx@ v_src@N@ = npyv_loadn_@sfx@(src + ssrc*vstep*@N@, ssrc);
            #endif
            npyv_@sfx@ v_unary@N@ = npyv_@intr@_@sfx@(v_src@N@);
        #endif
        /**end repeat3**/
        /**begin repeat3
         * #N  = 0, 1, 2, 3#
         */
        #if @unroll@ > @N@
            #if @DTYPE@ == CONTIG
                npyv_store_@sfx@(dst + vstep*@N@, v_unary@N@);
            #else
                npyv_storen_@sfx@(dst + sdst*vstep*@N@, sdst, v_unary@N@);
            #endif
        #endif
        /**end repeat3**/
    }

    // vector-sized iterations
    for (; len >= vstep; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
    #if @STYPE@ == CONTIG
        npyv_@sfx@ v_src0 = npyv_load_@sfx@(src);
    #else
        npyv_@sfx@ v_src0 = npyv_loadn_@sfx@(src, ssrc);
    #endif
        npyv_@sfx@ v_unary0 = npyv_@intr@_@sfx@(v_src0);
    #if @DTYPE@ == CONTIG
        npyv_store_@sfx@(dst, v_unary0);
    #else
        npyv_storen_@sfx@(dst, sdst, v_unary0);
    #endif
    }

    // last partial iteration, if needed
    if(len > 0){
    #if @STYPE@ == CONTIG
        #if @repl_0w1@
            npyv_@sfx@ v_src0 = npyv_load_till_@sfx@(src, len, 1);
        #else
            npyv_@sfx@ v_src0 = npyv_load_tillz_@sfx@(src, len);
        #endif
    #else
        #if @repl_0w1@
            npyv_@sfx@ v_src0 = npyv_loadn_till_@sfx@(src, ssrc, len, 1);
        #else
            npyv_@sfx@ v_src0 = npyv_loadn_tillz_@sfx@(src, ssrc, len);
        #endif
    #endif
        npyv_@sfx@ v_unary0 = npyv_@intr@_@sfx@(v_src0);
    #if @DTYPE@ == CONTIG
        npyv_store_till_@sfx@(dst, len, v_unary0);
    #else
        npyv_storen_till_@sfx@(dst, sdst, len, v_unary0);
    #endif
    }

    npyv_cleanup();
}
/**end repeat2**/
/**end repeat1**/
#endif // @VCHK@
/**end repeat**/

/********************************************************************************
 ** Defining ufunc inner functions
 ********************************************************************************/
/**begin repeat
 * #TYPE = FLOAT, DOUBLE#
 * #sfx  = f32, f64#
 * #VCHK = NPY_SIMD_F32, NPY_SIMD_F64#
 */
/**begin repeat1
 * #kind  = rint, floor, ceil, trunc, sqrt, absolute, square, reciprocal#
 * #intr  = rint, floor, ceil, trunc, sqrt, abs,      square, recip#
 * #clear = 0,    0,     0,    0,     0,    1,        0,      0#
 */
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(@TYPE@_@kind@)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    const char *src = args[0]; char *dst = args[1];
    const npy_intp src_step = steps[0];
    const npy_intp dst_step = steps[1];
    npy_intp len = dimensions[0];
#if @VCHK@
    const int lsize = sizeof(npyv_lanetype_@sfx@);
    assert(len <= 1 || (src_step % lsize == 0 && dst_step % lsize == 0));
    if (is_mem_overlap(src, src_step, dst, dst_step, len)) {
        goto no_unroll;
    }
    const npy_intp ssrc = src_step / lsize;
    const npy_intp sdst = dst_step / lsize;
    if (!npyv_loadable_stride_@sfx@(ssrc) || !npyv_storable_stride_@sfx@(sdst)) {
        goto no_unroll;
    }
    if (ssrc == 1 && sdst == 1) {
        simd_@TYPE@_@kind@_CONTIG_CONTIG(src, 1, dst, 1, len);
    }
    else if (sdst == 1) {
        simd_@TYPE@_@kind@_NCONTIG_CONTIG(src, ssrc, dst, 1, len);
    }
    else if (ssrc == 1) {
        simd_@TYPE@_@kind@_CONTIG_NCONTIG(src, 1, dst, sdst, len);
    } else {
        simd_@TYPE@_@kind@_NCONTIG_NCONTIG(src, ssrc, dst, sdst, len);
    }
    goto clear;
no_unroll:
#endif // @VCHK@
    for (; len > 0; --len, src += src_step, dst += dst_step) {
    #if @VCHK@
        // to guarantee the same precision and fp/domain errors for both scalars and vectors
        simd_@TYPE@_@kind@_CONTIG_CONTIG(src, 0, dst, 0, 1);
    #else
        const npyv_lanetype_@sfx@ src0 = *(npyv_lanetype_@sfx@*)src;
        *(npyv_lanetype_@sfx@*)dst = c_@intr@_@sfx@(src0);
    #endif
    }
#if @VCHK@
clear:;
#endif
#if @clear@
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}
/**end repeat1**/
/**end repeat**/
