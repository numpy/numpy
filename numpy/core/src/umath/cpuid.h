#ifndef _NPY_PRIVATE__CPUID_H_
#define _NPY_PRIVATE__CPUID_H_

#include <numpy/ndarraytypes.h>  /* for NPY_NO_EXPORT */

NPY_NO_EXPORT int
npy_cpu_supports(const char * feature);

#endif
