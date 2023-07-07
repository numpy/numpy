#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "absolute.cpp"  // this file
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
using T = npy_int;

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
HWY_ATTR void SuperAbsolute(const T* HWY_RESTRICT input_array,
                T* HWY_RESTRICT output_array,
                const size_t size) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto in = hn::Load(d, input_array + i);
    auto x = hn::Abs(in);
    hn::Store(x, d, output_array + i);
  }
}

}
}

#if HWY_ONCE
namespace numpy {

HWY_EXPORT(SuperAbsolute);

extern "C" {
NPY_NO_EXPORT void
INT_absolute(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_int *ip1 = (npy_int*) args[0];
    npy_int *op1 = (npy_int*) args[1];
    // npy_intp is1 = steps[0];
    // npy_intp os1 = steps[1];
    npy_intp n = dimensions[0];
    // npy_intp i;
  // This must reside outside of HWY_NAMESPACE because it references (calls the
  // appropriate one from) the per-target implementations there.
  // For static dispatch, use HWY_STATIC_DISPATCH.
  static auto dispatcher = HWY_DYNAMIC_DISPATCH(SuperAbsolute);
  return dispatcher(ip1, op1, n);
}
}

}
#endif


