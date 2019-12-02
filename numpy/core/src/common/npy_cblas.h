/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */
#ifndef _NPY_CBLAS_H_
#define _NPY_CBLAS_H_

#include <stddef.h>

/* Allow the use in C++ code.  */
#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Enumerated and derived types
 */
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#define CBLAS_INDEX size_t  /* this may vary between platforms */

#define BLASINT int
#define BLASNAME(name) name

#include "npy_cblas_base.h"

#undef BLASINT
#undef BLASNAME

#ifdef __cplusplus
}
#endif

#endif
