/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_

#include <stddef.h>
#include "numpy/npy_common.h"
#include "npy_blas_config.h"

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

#define CBLAS_INT   BLAS_INT

#define BLASNAME(name) CBLAS_FUNC(name)
#define BLASINT CBLAS_INT

#include "npy_cblas_base.h"

#undef BLASINT
#undef BLASNAME

/*
 * Define a chunksize for CBLAS.
 *
 * The chunksize is the greatest power of two less than BLAS_INT_MAX.
 */
#if NPY_MAX_INTP > BLAS_INT_MAX
# define NPY_CBLAS_CHUNK  (BLAS_INT_MAX / 2 + 1)
#else
# define NPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif


#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_ */
