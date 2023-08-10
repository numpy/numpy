/*@targets
 * $maxopt $keep_baseline avx512_skx asimd
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "absolute.dispatch.cpp"  // this file
#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"

namespace numpy {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
template <typename T>
HWY_ATTR void SuperAbsolute(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  const T* HWY_RESTRICT input_array = (const T*) args[0];
  T* HWY_RESTRICT output_array = (T*) args[1];
  const size_t size = dimensions[0];
  const hn::ScalableTag<T> d;
  
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto in = hn::Load(d, input_array + i);
    auto x = hn::Abs(in);
    hn::Store(x, d, output_array + i);
  }
}

HWY_ATTR void INT_SuperAbsolute(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  SuperAbsolute<npy_int>(args, dimensions, steps);
}

HWY_ATTR void DOUBLE_SuperAbsolute(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  SuperAbsolute<npy_double>(args, dimensions, steps);
}

HWY_ATTR void FLOAT_SuperAbsolute(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  SuperAbsolute<npy_float>(args, dimensions, steps);
}

}
}

#if HWY_ONCE
namespace numpy {

HWY_EXPORT(INT_SuperAbsolute);
HWY_EXPORT(FLOAT_SuperAbsolute);
HWY_EXPORT(DOUBLE_SuperAbsolute);

extern "C" {

NPY_NO_EXPORT void
INT_absolute(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  static auto dispatcher = HWY_STATIC_DISPATCH(INT_SuperAbsolute);
  return dispatcher(args, dimensions, steps);
}

NPY_NO_EXPORT void
DOUBLE_absolute(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  static auto dispatcher = HWY_STATIC_DISPATCH(DOUBLE_SuperAbsolute);
  return dispatcher(args, dimensions, steps);
}

NPY_NO_EXPORT void
FLOAT_absolute(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  static auto dispatcher = HWY_STATIC_DISPATCH(FLOAT_SuperAbsolute);
  return dispatcher(args, dimensions, steps);
}

} // extern "C"
} // numpy
#endif


