#ifndef NUMPY_CORE_SRC_COMMON_NPY_BLAS_CONFIG_H_
#define NUMPY_CORE_SRC_COMMON_NPY_BLAS_CONFIG_H_

#include <stddef.h>
#include "numpy/npy_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef ACCELERATE_NEW_LAPACK
    #if __MAC_OS_X_VERSION_MAX_ALLOWED < 130300
        #ifdef HAVE_BLAS_ILP64
            #error "Accelerate ILP64 support is only available with macOS 13.3 SDK or later"
        #endif
    #else
        #define NO_APPEND_FORTRAN
        #ifdef HAVE_BLAS_ILP64
            #define BLAS_SYMBOL_SUFFIX $NEWLAPACK$ILP64
        #else
            #define BLAS_SYMBOL_PREFIX accelerate_
        #endif
    #endif
#endif

#ifdef NO_APPEND_FORTRAN
#define BLAS_FORTRAN_SUFFIX
#else
#define BLAS_FORTRAN_SUFFIX _
#endif

#ifndef BLAS_SYMBOL_PREFIX
#define BLAS_SYMBOL_PREFIX
#endif

#ifndef BLAS_SYMBOL_SUFFIX
#define BLAS_SYMBOL_SUFFIX
#endif

#define BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2) prefix ## name ## suffix ## suffix2
#define BLAS_FUNC_EXPAND(name,prefix,suffix,suffix2) BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2)

#define CBLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,,BLAS_SYMBOL_SUFFIX)
#define BLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,BLAS_FORTRAN_SUFFIX,BLAS_SYMBOL_SUFFIX)

#ifdef HAVE_BLAS_ILP64
#define BLAS_INT npy_int64
#define BLAS_INT_MAX NPY_MAX_INT64
#else
#define BLAS_INT int
#define BLAS_INT_MAX INT_MAX
#endif

/*
 * Convert NumPy stride to BLAS stride. Returns 0 if conversion cannot be done
 * (BLAS won't handle negative or zero strides the way we want).
 */
static inline BLAS_INT
blas_stride(npy_intp stride, unsigned itemsize)
{
    /*
     * Should probably check pointer alignment also, but this may cause
     * problems if we require complex to be 16 byte aligned.
     */
    if (stride > 0 && (stride % itemsize) == 0) {
        stride /= itemsize;
        if (stride <= BLAS_INT_MAX) {
            return stride;
        }
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif // #ifndef NUMPY_CORE_SRC_COMMON_NPY_BLAS_CONFIG_H_