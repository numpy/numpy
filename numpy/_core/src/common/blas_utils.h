#include <stdbool.h>

#include "numpy/numpyconfig.h" // for NPY_VISIBILITY_HIDDEN

/* NPY_BLAS_CHECK_FPE_SUPPORT controls whether we need a runtime check
 * for floating-point error (FPE) support in BLAS.
 */
#if defined(__APPLE__) && defined(__aarch64__) && defined(ACCELERATE_NEW_LAPACK)
#define NPY_BLAS_CHECK_FPE_SUPPORT 1
#else
#define NPY_BLAS_CHECK_FPE_SUPPORT 0
#endif

/* Initialize BLAS environment, if needed
 */
NPY_VISIBILITY_HIDDEN void
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
