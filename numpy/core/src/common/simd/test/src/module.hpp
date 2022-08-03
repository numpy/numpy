#ifndef NUMPY_CORE_SRC_COMMON_SIMD_TEST_MODULE_H_
#define NUMPY_CORE_SRC_COMMON_SIMD_TEST_MODULE_H_

#include <Python.h>

#include "npy_cpu_dispatch.h"

namespace np {
#ifndef NPY_DISABLE_OPTIMIZATION
    #include "intrinsics.dispatch.h"
#endif
NPY_CPU_DISPATCH_DECLARE(PyObject *SimdExtention, ());
} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_SIMD_TEST_MODULE_H_
