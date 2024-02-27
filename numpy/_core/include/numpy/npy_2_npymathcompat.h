/* This header is designed to be copy-pasted into downstream packages, since it provides
   a compatibility layer between the old C struct complex types and the new native C99
   complex types. The new macros are in libnpymath/npy_math.h, which is why it is included here. */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_

#ifdef NPYMATH_USE_NUMPY_TYPES

#include <numpy/npy_common.h>

#endif


#include <libnpymath/npy_math.h>
#include <libnpymath/halffloat.h>

#endif /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_ */
