#include "numpy/numpyconfig.h" // for NPY_VISIBILITY_HIDDEN

#include <stdbool.h>

/* 
 * NPY_BLAS_CHECK_FPE_SUPPORT controls whether we need a runtime check
 * for floating-point error (FPE) support in BLAS.
 * The known culprit right now is SVM likely only on mac, but that is not
 * quite clear.
 * This checks always on all ARM (it is a small check overall).
 */
#if defined(__aarch64__)
#define NPY_BLAS_CHECK_FPE_SUPPORT 1
#else
#define NPY_BLAS_CHECK_FPE_SUPPORT 0
#endif

/* Initialize BLAS environment, if needed
 */
NPY_VISIBILITY_HIDDEN int
npy_blas_init(void);

/* Runtime check if BLAS supports floating-point errors.
 * true  - BLAS supports FPE and one can rely on them to indicate errors
 * false - BLAS does not support FPE.  Special handling needed for FPE state
 */
NPY_VISIBILITY_HIDDEN bool
npy_blas_supports_fpe(void);

/* If BLAS supports FPE, exactly the same as npy_get_floatstatus_barrier().
 * Otherwise, we can't rely on FPE state and need special handling.
 */
NPY_VISIBILITY_HIDDEN int
npy_get_floatstatus_after_blas(void);
