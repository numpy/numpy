#ifndef NUMPY_CORE_SRC_COMMON_NPY_NPY_FPMATH_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NPY_FPMATH_H_

#include "npy_config.h"

#include "numpy/npy_os.h"
#include "numpy/npy_cpu.h"
#include "numpy/npy_common.h"

#if !(defined(HAVE_LDOUBLE_IEEE_QUAD_BE) || \
      defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
      defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE) || \
      defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE))
    #error No long double representation defined
#endif

/* for back-compat, also keep old name for double-double */
#ifdef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE
    #define HAVE_LDOUBLE_DOUBLE_DOUBLE_LE
#endif
#ifdef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE
    #define HAVE_LDOUBLE_DOUBLE_DOUBLE_BE
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NPY_FPMATH_H_ */
