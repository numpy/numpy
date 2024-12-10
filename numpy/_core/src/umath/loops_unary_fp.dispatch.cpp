#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"
#include "fast_loop_macros.h"
#include "loops_utils.h"
#include <hwy/highway.h>
#include "hwy/print-inl.h"
#include <hwy/aligned_allocator.h>

namespace hn = hwy::HWY_NAMESPACE;
HWY_BEFORE_NAMESPACE();
template <typename T>
struct OpRound {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
#if HWY_ARCH_X86_64 && HWY_TARGET >= HWY_SSSE3
   hn::ScalableTag<T> d;
   auto infmask = hn::IsInf(a);
   auto nanmask = hn::IsNaN(a);
   auto mask = hn::Or(infmask, nanmask);
   auto b = hn::IfThenElse(mask, hn::Set(d, 0), a);
   b = hn::Round(b);
   return hn::IfThenElse(mask, a, b);
#else
    return hn::Round(a);
#endif
  }
};

template <typename T>
struct OpFloor {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
#if HWY_ARCH_X86_64 && HWY_TARGET >= HWY_SSSE3
   hn::ScalableTag<T> d;
   auto infmask = hn::IsInf(a);
   auto nanmask = hn::IsNaN(a);
   auto mask = hn::Or(infmask, nanmask);
   auto b = hn::IfThenElse(mask, hn::Set(d, 0), a);
   b = hn::Floor(b);
   return hn::IfThenElse(mask, a, b);
#else
    return hn::Floor(a);
#endif
  }
};

template <typename T>
struct OpCeil {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
#if HWY_ARCH_X86_64 && HWY_TARGET >= HWY_SSSE3
   hn::ScalableTag<T> d;
   auto infmask = hn::IsInf(a);
   auto nanmask = hn::IsNaN(a);
   auto mask = hn::Or(infmask, nanmask);
   auto b = hn::IfThenElse(mask, hn::Set(d, 0), a);
   b = hn::Ceil(b);
   return hn::IfThenElse(mask, a, b);
#else
    return hn::Ceil(a);
#endif
  }
};

template <typename T>
struct OpTrunc {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
#if HWY_ARCH_X86_64 && HWY_TARGET >= HWY_SSSE3
   hn::ScalableTag<T> d;
   auto infmask = hn::IsInf(a);
   auto nanmask = hn::IsNaN(a);
   auto mask = hn::Or(infmask, nanmask);
   auto b = hn::IfThenElse(mask, hn::Set(d, 0), a);
   b = hn::Trunc(b);
   return hn::IfThenElse(mask, a, b);
#else
    return hn::Trunc(a);
#endif
  }
};

template <typename T>
struct OpSqrt {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
    return hn::Sqrt(a);
  }
};

template <typename T>
struct OpSquare {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
    return hn::Mul(a, a);
  }
};

template <typename T>
struct OpAbs {
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
    return hn::Abs(a);
  }
};

template <typename T>
struct OpReciprocal {
  hn::ScalableTag<T> d;
  HWY_INLINE hn::VFromD<hn::ScalableTag<T>> operator()(
      hn::VFromD<hn::ScalableTag<T>> a) {
    return hn::Div(hn::Set(d, T(1.0)), a);
  }
};

template <typename T, typename OP>
HWY_INLINE void Super(char** args,
                    npy_intp const* dimensions,
                    npy_intp const* steps,
                    bool IS_RECIP) {
  const T* HWY_RESTRICT input_array = (const T*)args[0];
  T* HWY_RESTRICT output_array = (T*)args[1];
  const size_t size = dimensions[0];
  const hn::ScalableTag<T> d;
  auto one = hn::Set(d, 1);
  OP op;
  if (is_mem_overlap(input_array, steps[0], output_array, steps[1], size)) {
    const int lsize = sizeof(input_array[0]);
    const npy_intp ssrc = steps[0] / lsize;
    const npy_intp sdst = steps[1] / lsize;
    for (size_t len = size; 0 < len;
         len--, input_array += ssrc, output_array += sdst) {
      hn::Vec<hn::ScalableTag<T>> in;
      if (IS_RECIP) {
        in = hn::LoadNOr(one, d, input_array, 1);
      } else {
        in = hn::LoadN(d, input_array, 1);
      }
      auto x = op(in);
      hn::StoreN(x, d, output_array, 1);
    }
  } else if (IS_UNARY_CONT(input_array[0], output_array[0])) {
    const int vstep = hn::Lanes(d);
    const int wstep = vstep * 4;
    size_t len = size;
    for (; len >= wstep;
         len -= wstep, input_array += wstep, output_array += wstep) {
      const auto in0 = hn::LoadU(d, input_array + vstep * 0);
      auto x0 = op(in0);
      const auto in1 = hn::LoadU(d, input_array + vstep * 1);
      auto x1 = op(in1);
      const auto in2 = hn::LoadU(d, input_array + vstep * 2);
      auto x2 = op(in2);
      const auto in3 = hn::LoadU(d, input_array + vstep * 3);
      auto x3 = op(in3);
      hn::StoreU(x0, d, output_array + vstep * 0);
      hn::StoreU(x1, d, output_array + vstep * 1);
      hn::StoreU(x2, d, output_array + vstep * 2);
      hn::StoreU(x3, d, output_array + vstep * 3);
    }
    for (; len >= vstep;
         len -= vstep, input_array += vstep, output_array += vstep) {
      const auto in = hn::LoadU(d, input_array);
      auto x = op(in);
      hn::StoreU(x, d, output_array);
    }
    if (len) {
      hn::Vec<hn::ScalableTag<T>> in;
      if (IS_RECIP) {
        in = hn::LoadNOr(one, d, input_array, len);
      } else {
        in = hn::LoadN(d, input_array, len);
      }
      auto x = op(in);
      hn::StoreN(x, d, output_array, len);
    }
  } else {
    using TI = hwy::MakeSigned<T>;
    const hn::Rebind<TI, hn::ScalableTag<T>> di;

    const int lsize = sizeof(input_array[0]);
    const npy_intp ssrc = steps[0] / lsize;
    const npy_intp sdst = steps[1] / lsize;
    auto load_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));
    auto store_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));
    size_t full = size & -hn::Lanes(d);
    size_t remainder = size - full;
    if (sdst == 1 && ssrc != 1) {
      for (size_t i = 0; i < full; i += hn::Lanes(d)) {
        const auto in = hn::GatherIndex(d, input_array + i * ssrc, load_index);
        auto x = op(in);
        hn::StoreU(x, d, output_array + i);
      }
    } else if (sdst != 1 && ssrc == 1) {
      for (size_t i = 0; i < full; i += hn::Lanes(d)) {
        const auto in = hn::LoadU(d, input_array + i);
        auto x = op(in);
        hn::ScatterIndex(x, d, output_array + i * sdst, store_index);
      }
    } else {
      for (size_t i = 0; i < full; i += hn::Lanes(d)) {
        const auto in = hn::GatherIndex(d, input_array + i * ssrc, load_index);
        auto x = op(in);
        hn::ScatterIndex(x, d, output_array + i * sdst, store_index);
      }
    }
    if (remainder) {
      hn::Vec<hn::ScalableTag<T>> in;
      if (IS_RECIP) {
        if (ssrc == 1) {
          in = hn::LoadNOr(one, d, input_array + full, remainder);
        } else {
          in = hn::GatherIndexNOr(one, d, input_array + full * ssrc, load_index,
                                  remainder);
        }
      } else {
        if (ssrc == 1) {
          in = hn::LoadN(d, input_array + full, remainder);
        } else {
          in = hn::GatherIndexN(d, input_array + full * ssrc, load_index,
                                remainder);
        }
      }
      auto x = op(in);
      if (sdst == 1) {
        hn::StoreN(x, d, output_array + full, remainder);
      } else {
        hn::ScatterIndexN(x, d, output_array + full * sdst, store_index,
                          remainder);
      }
    }
  }
}


extern "C" {
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpRound<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpRound<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpFloor<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpFloor<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpCeil<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpCeil<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpTrunc<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpTrunc<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpSqrt<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpSqrt<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double, OpSquare<npy_double>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float, OpSquare<npy_float>>(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_double, OpAbs<npy_double>>(args, dimensions, steps, false);
  npy_clear_floatstatus_barrier((char*)dimensions);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_float, OpAbs<npy_float>>(args, dimensions, steps, false);
  npy_clear_floatstatus_barrier((char*)dimensions);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_double, OpReciprocal<npy_double>>(args, dimensions, steps, true);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_float, OpReciprocal<npy_float>>(args, dimensions, steps, true);
}
}
HWY_AFTER_NAMESPACE();