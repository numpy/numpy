/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */
#ifndef _NPY_CBLAS64__H_
#define _NPY_CBLAS64__H_

#include <stddef.h>

#include "npy_cblas.h"

/* Allow the use in C++ code.  */
#ifdef __cplusplus
extern "C"
{
#endif

#define BLASINT npy_int64
#define BLASNAME(name) name##64_

#include "npy_cblas_base.h"

#undef BLASINT
#undef BLASNAME

#ifdef __cplusplus
}
#endif

#endif
