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
#include <hwy/highway.h>
#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"
#include "numpy/npy_math.h"
#include "simd/simd.h"
namespace hn = hwy::HWY_NAMESPACE;
/**********************************************************
 ** Scalars
 **********************************************************/
#if !NPY_SIMD_F32
NPY_FINLINE float c_recip_f32(float a) {
  return 1.0f / a;
}
NPY_FINLINE float c_abs_f32(float a) {
  const float tmp = a > 0 ? a : -a;
  /* add 0 to clear -0.0 */
  return tmp + 0;
}
NPY_FINLINE float c_square_f32(float a) {
  return a * a;
}
#endif  // !NPY_SIMD_F32

#if !NPY_SIMD_F64
NPY_FINLINE double c_recip_f64(double a) {
  return 1.0 / a;
}
NPY_FINLINE double c_abs_f64(double a) {
  const double tmp = a > 0 ? a : -a;
  /* add 0 to clear -0.0 */
  return tmp + 0;
}
NPY_FINLINE double c_square_f64(double a) {
  return a * a;
}
#endif  // !NPY_SIMD_F64
/**
 * MSVC(32-bit mode) requires a clarified contiguous loop
 * in order to use SSE, otherwise it uses a soft version of square root
 * that doesn't raise a domain error.
 */
#if defined(_MSC_VER) && defined(_M_IX86) && !NPY_SIMD
#include <emmintrin.h>
NPY_FINLINE float c_sqrt_f32(float _a) {
  __m128 a = _mm_load_ss(&_a);
  __m128 lower = _mm_sqrt_ss(a);
  return _mm_cvtss_f32(lower);
}
NPY_FINLINE double c_sqrt_f64(double _a) {
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
#define CONTIG 0
#define NCONTIG 1

#if NPY_SIMD_F32

const hn::ScalableTag<float> f32;
const hn::ScalableTag<int32_t> s32;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_s32 = hn::Vec<decltype(s32)>;
using opmask_t = hn::Mask<decltype(f32)>;

vec_f32 square(vec_f32 a) {
  return hn::Mul(a, a);
}

vec_f32 reciprocal(vec_f32 a) {
  const vec_f32 ones  = hn::Set(f32, 1);
  return hn::Div(ones, a);
}

NPY_FINLINE HWY_ATTR vec_f32
GatherIndexN(const float* src, npy_intp ssrc, npy_intp len)
{
    float temp[hn::Lanes(f32)] = { 0.0f };
    for (auto ii = 0; ii < std::min(len, (npy_intp)hn::Lanes(f32)); ++ii) {
        temp[ii] = src[ii * ssrc];
    }
    return hn::LoadU(f32, temp);
}

NPY_FINLINE HWY_ATTR void
ScatterIndexN(vec_f32 vec, float* dst, npy_intp sdst, npy_intp len)
{
    float temp[hn::Lanes(f32)];
    hn::StoreU(vec, f32, temp);
    for (auto ii = 0; ii < std::min(len, (npy_intp)hn::Lanes(f32)); ++ii) {
        dst[ii * sdst] = temp[ii];
    }
}

template <int STYPE, int DTYPE, int UNROLL, bool IS_RECIP, typename FUNC>
static void HW_FLOAT_HELPER(const void* _src,
                              npy_intp ssrc,
                              void* _dst,
                              npy_intp sdst,
                              npy_intp len,
                              FUNC fn) {
  const float* src = (const float*)_src;
  float* dst = (float*)_dst;

  const int vstep = hn::Lanes(f32);
  const int wstep = vstep * UNROLL;

  // unrolled iterations
  for (; len >= wstep; len -= wstep, src += ssrc * wstep, dst += sdst * wstep) {
    if constexpr (UNROLL > 0) {
      vec_f32 v_src0;
      if constexpr (STYPE == CONTIG) {
        v_src0 = hn::LoadN(f32, src + vstep * 0, len);
      } else {
        v_src0 = GatherIndexN(src + ssrc * vstep * 0, ssrc, len);
      }
      vec_f32 v_unary0 = fn(v_src0);

      if constexpr (DTYPE == CONTIG) {
        hn::StoreN(v_unary0, f32, dst + vstep * 0, len);
      } else {
        ScatterIndexN(v_unary0, dst + sdst * vstep * 0, sdst, len);
      }
    }

    if constexpr (UNROLL > 1) {
      vec_f32 v_src1;
      if constexpr (STYPE == CONTIG) {
        v_src1 = hn::LoadN(f32, src + vstep * 1, len);
      } else {
        v_src1 = GatherIndexN(src + ssrc * vstep * 1, ssrc, len);
      }
      vec_f32 v_unary1 = fn(v_src1);

      if constexpr (DTYPE == CONTIG) {
        hn::StoreN(v_unary1, f32, dst + vstep * 1, len);
      } else {
        ScatterIndexN(v_unary1, dst + sdst * vstep * 1, sdst, len);
      }
    }

    if constexpr (UNROLL > 2) {
      vec_f32 v_src2;
      if constexpr (STYPE == CONTIG) {
        v_src2 = hn::LoadN(f32, src + vstep * 2, len);
      } else {
        v_src2 = GatherIndexN(src + ssrc * vstep * 2, ssrc, len);
      }
      vec_f32 v_unary2 = fn(v_src2);

      if constexpr (DTYPE == CONTIG) {
        hn::StoreN(v_unary2, f32, dst + vstep * 1, len);
      } else {
        ScatterIndexN(v_unary2, dst + sdst * vstep * 2, sdst, len);
      }
    }

    if constexpr (UNROLL > 3) {
      vec_f32 v_src3;
      if constexpr (STYPE == CONTIG) {
        v_src3 = hn::LoadN(f32, src + vstep * 3, len);
      } else {
        v_src3 = GatherIndexN(src + ssrc * vstep * 3, ssrc, len);
      }
      vec_f32 v_unary3 = fn(v_src3);

      if constexpr (DTYPE == CONTIG) {
        hn::StoreN(v_unary3, f32, dst + vstep * 3, len);
      } else {
        ScatterIndexN(v_unary3, dst + sdst * vstep * 3, sdst, len);
      }
    }
  }

  // vector-sized iterations
  for (; len >= vstep; len -= vstep, src += ssrc * vstep, dst += sdst * vstep) {
    vec_f32 v_src0;
    if constexpr (STYPE == CONTIG) {
      v_src0 = hn::LoadN(f32, src, len);
    } else {
      v_src0 = GatherIndexN(src, ssrc, len);
    }
    vec_f32 v_unary0 = fn(v_src0);
    if constexpr (DTYPE == CONTIG) {
      hn::StoreN(v_unary0, f32, dst, len);
    } else {
      ScatterIndexN(v_unary0, dst, sdst, len);
    }
  }

  // last partial iteration, if needed
  if (len > 0) {
    vec_f32 v_src0;
    if (STYPE == CONTIG) {
      if (IS_RECIP) {
        v_src0 = hn::LoadN(f32, src, len);
      } else {
        v_src0 = hn::LoadN(f32, src, len);
      }
    } else {
      if (IS_RECIP) {
        v_src0 = GatherIndexN(src, ssrc, len);
      } else {
        v_src0 = GatherIndexN(src, ssrc, len);
      }
    }
    vec_f32 v_unary0 = fn(v_src0);
    if (DTYPE == CONTIG) {
      hn::StoreN(v_unary0, f32, dst, len);
      // npyv_store_till_f32(dst, len, v_unary0);
    } else {
      hn::StoreN(v_unary0, f32, dst, len);
      // npyv_storen_till_f32(dst, sdst, len, v_unary0);
    }
  }

  npyv_cleanup();
}


template <int STYPE, int DTYPE, int UNROLL, bool IS_RECIP, typename Func>
static void simd_FLOAT_HELPER
(const void *_src, npy_intp ssrc, void *_dst, npy_intp sdst, npy_intp len, Func fn)
{
    const npyv_lanetype_f32 *src = (const npyv_lanetype_f32*)_src;
          npyv_lanetype_f32 *dst = (npyv_lanetype_f32*)_dst;

    const int vstep = npyv_nlanes_f32;
    const int wstep = vstep * UNROLL;

    // unrolled iterations
    for (; len >= wstep; len -= wstep, src += ssrc*wstep, dst += sdst*wstep) {
        if(UNROLL > 0) {
          npyv_f32 v_src0;
          if (STYPE == CONTIG) {
            v_src0 = npyv_load_f32(src + vstep * 0);
          } else {
            v_src0 = npyv_loadn_f32(src + ssrc * vstep * 0, ssrc);
          }
          npyv_f32 v_unary0 = fn(v_src0);

          if (DTYPE == CONTIG) {
            npyv_store_f32(dst + vstep * 0, v_unary0);
          } else {
            npyv_storen_f32(dst + sdst * vstep * 0, sdst, v_unary0);
          }
        }

        if(UNROLL > 1) {
          npyv_f32 v_src1;
            if(STYPE == CONTIG) {
                v_src1 = npyv_load_f32(src + vstep*1);
            } else {
                v_src1 = npyv_loadn_f32(src + ssrc*vstep*1, ssrc);
            }
            npyv_f32 v_unary1 = fn(v_src1);

            if(DTYPE == CONTIG) {
                npyv_store_f32(dst + vstep*1, v_unary1);
            }else{
                npyv_storen_f32(dst + sdst*vstep*1, sdst, v_unary1);
            }
        }

        if(UNROLL > 2) {
          npyv_f32 v_src2;
            if(STYPE == CONTIG) {
                v_src2 = npyv_load_f32(src + vstep*2);
            } else {
                v_src2 = npyv_loadn_f32(src + ssrc*vstep*2, ssrc);
            }
            npyv_f32 v_unary2 = fn(v_src2);

            if(DTYPE == CONTIG) {
                npyv_store_f32(dst + vstep*2, v_unary2);
            }else{
                npyv_storen_f32(dst + sdst*vstep*2, sdst, v_unary2);
            }
        }

        if(UNROLL > 3) {
          npyv_f32 v_src3;
            if(STYPE == CONTIG) {
                v_src3 = npyv_load_f32(src + vstep*3);
            } else {
                v_src3 = npyv_loadn_f32(src + ssrc*vstep*3, ssrc);
            }
            npyv_f32 v_unary3 = fn(v_src3);

            if(DTYPE == CONTIG) {
                npyv_store_f32(dst + vstep*3, v_unary3);
            }else{
                npyv_storen_f32(dst + sdst*vstep*3, sdst, v_unary3);
            }
        }
    }

    // vector-sized iterations
    for (; len >= vstep; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
      npyv_f32 v_src0;
      if (STYPE == CONTIG) {
        v_src0 = npyv_load_f32(src);
      } else {
        v_src0 = npyv_loadn_f32(src, ssrc);
      }
      npyv_f32 v_unary0 = fn(v_src0);
      if (DTYPE == CONTIG) {
        npyv_store_f32(dst, v_unary0);
      } else {
        npyv_storen_f32(dst, sdst, v_unary0);
      }
    }

    // last partial iteration, if needed
    if (len > 0) {
      npyv_f32 v_src0;
      if (STYPE == CONTIG) {
        if (IS_RECIP) {
          v_src0 = npyv_load_till_f32(src, len, 1);
        } else {
          v_src0 = npyv_load_tillz_f32(src, len);
        }
      } else {
        if (IS_RECIP) {
          v_src0 = npyv_loadn_till_f32(src, ssrc, len, 1);
        } else {
          v_src0 = npyv_loadn_tillz_f32(src, ssrc, len);
        }
      }
      npyv_f32 v_unary0 = fn(v_src0);
      if (DTYPE == CONTIG) {
        npyv_store_till_f32(dst, len, v_unary0);
      } else {
        npyv_storen_till_f32(dst, sdst, len, v_unary0);
      }
    }

    npyv_cleanup();
}

#define SIMD_KERNERL(KIND, FUNC, IS_RECIP)                                     \
  static void simd_FLOAT_##KIND##_CONTIG_CONTIG(const void* _src,              \
                                                npy_intp ssrc, void* _dst,     \
                                                npy_intp sdst, npy_intp len) { \
    auto fn = [](npyv_f32 src){return FUNC(src);};                             \
    simd_FLOAT_HELPER<CONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,     \
                                                   len, fn);                 \
  }                                                                            \
  static void simd_FLOAT_##KIND##_NCONTIG_CONTIG(                              \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](npyv_f32 src){return FUNC(src);};                             \
    simd_FLOAT_HELPER<NCONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,    \
                                                    len, fn);                \
  }                                                                            \
  static void simd_FLOAT_##KIND##_CONTIG_NCONTIG(                              \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](npyv_f32 src){return FUNC(src);};                             \
    simd_FLOAT_HELPER<CONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst,    \
                                                    len, fn);                \
  }                                                                            \
  static void simd_FLOAT_##KIND##_NCONTIG_NCONTIG(                             \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](npyv_f32 src){return FUNC(src);};                             \
    simd_FLOAT_HELPER<NCONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst,   \
                                                     len, fn);               \
  }

#define HW_KERNERL(KIND, FUNC, IS_RECIP)                                     \
  static void simd_FLOAT_##KIND##_CONTIG_CONTIG(const void* _src,              \
                                                npy_intp ssrc, void* _dst,     \
                                                npy_intp sdst, npy_intp len) { \
    auto fn = [](vec_f32 src){return FUNC(src);};                             \
    HW_FLOAT_HELPER<CONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,     \
                                                   len, fn);                   \
  }                                                                            \
  static void simd_FLOAT_##KIND##_NCONTIG_CONTIG(                              \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](vec_f32 src){return FUNC(src);};                             \
    HW_FLOAT_HELPER<NCONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,    \
                                                    len, fn);                  \
  }                                                                            \
  static void simd_FLOAT_##KIND##_CONTIG_NCONTIG(                              \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](vec_f32 src){return FUNC(src);};                             \
    HW_FLOAT_HELPER<CONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst,    \
                                                    len, fn);                  \
  }                                                                            \
  static void simd_FLOAT_##KIND##_NCONTIG_NCONTIG(                             \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,              \
      npy_intp len) {                                                          \
    auto fn = [](vec_f32 src){return FUNC(src);};                             \
    HW_FLOAT_HELPER<NCONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst,   \
                                                     len, fn);                 \
  }

HW_KERNERL(rint, hn::Round, false)
SIMD_KERNERL(floor, npyv_floor_f32, false)
SIMD_KERNERL(ceil, npyv_ceil_f32, false)
SIMD_KERNERL(trunc, npyv_trunc_f32, false)
SIMD_KERNERL(sqrt, npyv_sqrt_f32, false)
SIMD_KERNERL(absolute, npyv_abs_f32, false)
SIMD_KERNERL(square, npyv_square_f32, false)
SIMD_KERNERL(reciprocal, npyv_recip_f32, true)
#undef SIMD_KERNERL
#endif  // NPY_SIMD_F32

#if NPY_SIMD_F64
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
template <int STYPE, int DTYPE, int UNROLL, bool IS_RECIP, typename Func>
static void simd_DOUBLE_HELPER(const void* _src,
                               npy_intp ssrc,
                               void* _dst,
                               npy_intp sdst,
                               npy_intp len,
                               Func fn) {
  const npyv_lanetype_f64* src = (const npyv_lanetype_f64*)_src;
  npyv_lanetype_f64* dst = (npyv_lanetype_f64*)_dst;

  const int vstep = npyv_nlanes_f64;
  const int wstep = vstep * UNROLL;

  // unrolled iterations
  for (; len >= wstep; len -= wstep, src += ssrc * wstep, dst += sdst * wstep) {
    if (UNROLL > 0) {
      npyv_f64 v_src0;
      if (STYPE == CONTIG) {
        v_src0 = npyv_load_f64(src + vstep * 0);
      } else {
        v_src0 = npyv_loadn_f64(src + ssrc * vstep * 0, ssrc);
      }
      npyv_f64 v_unary0 = fn(v_src0);

      if (DTYPE == CONTIG) {
        npyv_store_f64(dst + vstep * 0, v_unary0);
      } else {
        npyv_storen_f64(dst + sdst * vstep * 0, sdst, v_unary0);
      }
    }

    if (UNROLL > 1) {
      npyv_f64 v_src1;
      if (STYPE == CONTIG) {
        v_src1 = npyv_load_f64(src + vstep * 1);
      } else {
        v_src1 = npyv_loadn_f64(src + ssrc * vstep * 1, ssrc);
      }
      npyv_f64 v_unary1 = fn(v_src1);

      if (DTYPE == CONTIG) {
        npyv_store_f64(dst + vstep * 1, v_unary1);
      } else {
        npyv_storen_f64(dst + sdst * vstep * 1, sdst, v_unary1);
      }
    }

    if (UNROLL > 2) {
      npyv_f64 v_src2;
      if (STYPE == CONTIG) {
        v_src2 = npyv_load_f64(src + vstep * 2);
      } else {
        v_src2 = npyv_loadn_f64(src + ssrc * vstep * 2, ssrc);
      }
      npyv_f64 v_unary2 = fn(v_src2);

      if (DTYPE == CONTIG) {
        npyv_store_f64(dst + vstep * 2, v_unary2);
      } else {
        npyv_storen_f64(dst + sdst * vstep * 2, sdst, v_unary2);
      }
    }

    if (UNROLL > 3) {
      npyv_f64 v_src3;
      if (STYPE == CONTIG) {
        v_src3 = npyv_load_f64(src + vstep * 3);
      } else {
        v_src3 = npyv_loadn_f64(src + ssrc * vstep * 3, ssrc);
      }
      npyv_f64 v_unary3 = fn(v_src3);

      if (DTYPE == CONTIG) {
        npyv_store_f64(dst + vstep * 3, v_unary3);
      } else {
        npyv_storen_f64(dst + sdst * vstep * 3, sdst, v_unary3);
      }
    }
  }

  // vector-sized iterations
  for (; len >= vstep; len -= vstep, src += ssrc * vstep, dst += sdst * vstep) {
    npyv_f64 v_src0;
    if (STYPE == CONTIG) {
      v_src0 = npyv_load_f64(src);
    } else {
      v_src0 = npyv_loadn_f64(src, ssrc);
    }
    npyv_f64 v_unary0 = fn(v_src0);
    if (DTYPE == CONTIG) {
      npyv_store_f64(dst, v_unary0);
    } else {
      npyv_storen_f64(dst, sdst, v_unary0);
    }
  }

  // last partial iteration, if needed
  if (len > 0) {
    npyv_f64 v_src0;
    if (STYPE == CONTIG) {
      if (IS_RECIP) {
        v_src0 = npyv_load_till_f64(src, len, 1);
      } else {
        v_src0 = npyv_load_tillz_f64(src, len);
      }
    } else {
      if (IS_RECIP) {
        v_src0 = npyv_loadn_till_f64(src, ssrc, len, 1);
      } else {
        v_src0 = npyv_loadn_tillz_f64(src, ssrc, len);
      }
    }
    npyv_f64 v_unary0 = fn(v_src0);
    if (DTYPE == CONTIG) {
      npyv_store_till_f64(dst, len, v_unary0);
    } else {
      npyv_storen_till_f64(dst, sdst, len, v_unary0);
    }
  }

  npyv_cleanup();
}

#define SIMD_KERNERL(KIND, FUNC, IS_RECIP)                                    \
  static void simd_DOUBLE_##KIND##_CONTIG_CONTIG(                             \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,             \
      npy_intp len) {                                                         \
    auto fn = [](npyv_f64 src){return FUNC(src);};                            \
    simd_DOUBLE_HELPER<CONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,   \
                                                    len, fn);               \
  }                                                                           \
  static void simd_DOUBLE_##KIND##_NCONTIG_CONTIG(                            \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,             \
      npy_intp len) {                                                         \
    auto fn = [](npyv_f64 src){return FUNC(src);};                            \
    simd_DOUBLE_HELPER<NCONTIG, CONTIG, 4, IS_RECIP>(_src, ssrc, _dst, sdst,  \
                                                     len, fn);              \
  }                                                                           \
  static void simd_DOUBLE_##KIND##_CONTIG_NCONTIG(                            \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,             \
      npy_intp len) {                                                         \
    auto fn = [](npyv_f64 src){return FUNC(src);};                            \
    simd_DOUBLE_HELPER<CONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst,  \
                                                     len, fn);              \
  }                                                                           \
  static void simd_DOUBLE_##KIND##_NCONTIG_NCONTIG(                           \
      const void* _src, npy_intp ssrc, void* _dst, npy_intp sdst,             \
      npy_intp len) {                                                         \
    auto fn = [](npyv_f64 src){return FUNC(src);};                            \
    simd_DOUBLE_HELPER<NCONTIG, NCONTIG, 2, IS_RECIP>(_src, ssrc, _dst, sdst, \
                                                      len, fn);             \
  }

SIMD_KERNERL(rint, npyv_rint_f64, false)
SIMD_KERNERL(floor, npyv_floor_f64, false)
SIMD_KERNERL(ceil, npyv_ceil_f64, false)
SIMD_KERNERL(trunc, npyv_trunc_f64, false)
SIMD_KERNERL(sqrt, npyv_sqrt_f64, false)
SIMD_KERNERL(absolute, npyv_abs_f64, false)
SIMD_KERNERL(square, npyv_square_f64, false)
SIMD_KERNERL(reciprocal, npyv_recip_f64, true)
#undef SIMD_KERNERL
#endif  // NPY_SIMD_F64
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
 * #intr  = rint, floor, ceil, trunc, sqrt, absolute, square, reciprocal#
 * #clear = 0,    0,     0,    0,     0,    1,        0,      0#
 */

#ifdef NPY_SIMD_F32
#define FLOAT_FUNC(KIND, INTR, CLEAR)                                         \
  NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_##KIND)(                    \
      char** args, npy_intp const* dimensions, npy_intp const* steps,         \
      void* NPY_UNUSED(func)) {                                               \
    const char* src = args[0];                                                \
    char* dst = args[1];                                                      \
    const npy_intp src_step = steps[0];                                       \
    const npy_intp dst_step = steps[1];                                       \
    npy_intp len = dimensions[0];                                             \
    const int lsize = sizeof(npyv_lanetype_f32);                              \
    assert(len <= 1 || (src_step % lsize == 0 && dst_step % lsize == 0));     \
    if (is_mem_overlap(src, src_step, dst, dst_step, len)) {                  \
      for (; len > 0; --len, src += src_step, dst += dst_step) {              \
        /*to guarantee the same precision and fp/domain errors for both       \
         * scalars and vectors*/                                                                \
        simd_FLOAT_##KIND##_CONTIG_CONTIG(src, 0, dst, 0, 1);                 \
      }                                                                       \
    }                                                                         \
    const npy_intp ssrc = src_step / lsize;                                   \
    const npy_intp sdst = dst_step / lsize;                                   \
    if (!npyv_loadable_stride_f32(ssrc) || !npyv_storable_stride_f32(sdst)) { \
      for (; len > 0; --len, src += src_step, dst += dst_step) {              \
        /*to guarantee the same precision and fp/domain errors for both       \
         * scalars and vectors*/                                                                \
        simd_FLOAT_##KIND##_CONTIG_CONTIG(src, 0, dst, 0, 1);                 \
      }                                                                       \
    }                                                                         \
    if (ssrc == 1 && sdst == 1) {                                             \
      simd_FLOAT_##KIND##_CONTIG_CONTIG(src, 1, dst, 1, len);                 \
    } else if (sdst == 1) {                                                   \
      simd_FLOAT_##KIND##_NCONTIG_CONTIG(src, ssrc, dst, 1, len);             \
    } else if (ssrc == 1) {                                                   \
      simd_FLOAT_##KIND##_CONTIG_NCONTIG(src, 1, dst, sdst, len);             \
    } else {                                                                  \
      simd_FLOAT_##KIND##_NCONTIG_NCONTIG(src, ssrc, dst, sdst, len);         \
    }                                                                         \
    if (CLEAR) {                                                              \
      npy_clear_floatstatus_barrier((char*)dimensions);                       \
    }                                                                         \
  }
#else
#define FLOAT_FUNC(KIND, INTR, CLEAR)                                 \
  NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_##KIND)(            \
      char** args, npy_intp const* dimensions, npy_intp const* steps, \
      void* NPY_UNUSED(func)) {                                       \
    const char* src = args[0];                                        \
    char* dst = args[1];                                              \
    const npy_intp src_step = steps[0];                               \
    const npy_intp dst_step = steps[1];                               \
    npy_intp len = dimensions[0];                                     \
    for (; len > 0; --len, src += src_step, dst += dst_step) {        \
      const npyv_lanetype_f32 src0 = *(npyv_lanetype_f32*)src;        \
      *(npyv_lanetype_f32*)dst = c_##INTR##_f32(src0);                \
    }                                                                 \
    if (CLEAR) {                                                      \
      npy_clear_floatstatus_barrier((char*)dimensions);               \
    }                                                                 \
  }
#endif  // NPY_SIMD_F32

FLOAT_FUNC(rint, rint, 0)
FLOAT_FUNC(floor, floor, 0)
FLOAT_FUNC(ceil, ceil, 0)
FLOAT_FUNC(trunc, trunc, 0)
FLOAT_FUNC(sqrt, sqrt, 0)
FLOAT_FUNC(absolute, abs, 1)
FLOAT_FUNC(square, square, 0)
FLOAT_FUNC(reciprocal, recip, 0)

#ifdef NPY_SIMD_F64
#define DOUBLE_FUNC(KIND, INTR, CLEAR)                                        \
  NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_##KIND)(                   \
      char** args, npy_intp const* dimensions, npy_intp const* steps,         \
      void* NPY_UNUSED(func)) {                                               \
    const char* src = args[0];                                                \
    char* dst = args[1];                                                      \
    const npy_intp src_step = steps[0];                                       \
    const npy_intp dst_step = steps[1];                                       \
    npy_intp len = dimensions[0];                                             \
    const int lsize = sizeof(npyv_lanetype_f64);                              \
    assert(len <= 1 || (src_step % lsize == 0 && dst_step % lsize == 0));     \
    if (is_mem_overlap(src, src_step, dst, dst_step, len)) {                  \
      for (; len > 0; --len, src += src_step, dst += dst_step) {              \
        /*to guarantee the same precision and fp/domain errors for both       \
         * scalars and vectors*/                                                                \
        simd_DOUBLE_##KIND##_CONTIG_CONTIG(src, 0, dst, 0, 1);                \
      }                                                                       \
    }                                                                         \
    const npy_intp ssrc = src_step / lsize;                                   \
    const npy_intp sdst = dst_step / lsize;                                   \
    if (!npyv_loadable_stride_f64(ssrc) || !npyv_storable_stride_f64(sdst)) { \
      for (; len > 0; --len, src += src_step, dst += dst_step) {              \
        /*to guarantee the same precision and fp/domain errors for both       \
         * scalars and vectors*/                                                                \
        simd_DOUBLE_##KIND##_CONTIG_CONTIG(src, 0, dst, 0, 1);                \
      }                                                                       \
    }                                                                         \
    if (ssrc == 1 && sdst == 1) {                                             \
      simd_DOUBLE_##KIND##_CONTIG_CONTIG(src, 1, dst, 1, len);                \
    } else if (sdst == 1) {                                                   \
      simd_DOUBLE_##KIND##_NCONTIG_CONTIG(src, ssrc, dst, 1, len);            \
    } else if (ssrc == 1) {                                                   \
      simd_DOUBLE_##KIND##_CONTIG_NCONTIG(src, 1, dst, sdst, len);            \
    } else {                                                                  \
      simd_DOUBLE_##KIND##_NCONTIG_NCONTIG(src, ssrc, dst, sdst, len);        \
    }                                                                         \
    if (CLEAR) {                                                              \
      npy_clear_floatstatus_barrier((char*)dimensions);                       \
    }                                                                         \
  }
#else
#define DOUBLE_FUNC(KIND, INTR, CLEAR)                                \
  NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_##KIND)(           \
      char** args, npy_intp const* dimensions, npy_intp const* steps, \
      void* NPY_UNUSED(func)) {                                       \
    const char* src = args[0];                                        \
    char* dst = args[1];                                              \
    const npy_intp src_step = steps[0];                               \
    const npy_intp dst_step = steps[1];                               \
    npy_intp len = dimensions[0];                                     \
    for (; len > 0; --len, src += src_step, dst += dst_step) {        \
      const npyv_lanetype_f64 src0 = *(npyv_lanetype_f64*)src;        \
      *(npyv_lanetype_f64*)dst = c_##INTR##_f64(src0);                \
    }                                                                 \
    if (CLEAR) {                                                      \
      npy_clear_floatstatus_barrier((char*)dimensions);               \
    }                                                                 \
  }
#endif  // NPY_SIMD_F64

DOUBLE_FUNC(rint, rint, 0)
DOUBLE_FUNC(floor, floor, 0)
DOUBLE_FUNC(ceil, ceil, 0)
DOUBLE_FUNC(trunc, trunc, 0)
DOUBLE_FUNC(sqrt, sqrt, 0)
DOUBLE_FUNC(absolute, abs, 1)
DOUBLE_FUNC(square, square, 0)
DOUBLE_FUNC(reciprocal, recip, 0)
