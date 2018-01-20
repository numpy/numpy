#ifndef _NPY_NPY_FPMATH_H_
#define _NPY_NPY_FPMATH_H_

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
      defined(HAVE_LDOUBLE_DOUBLE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_DOUBLE_DOUBLE_LE))
    #error No long double representation defined
#endif

#endif
