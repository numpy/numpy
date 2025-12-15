#include "numpy/npy_math.h"     // npy_get_floatstatus_barrier
#include "numpy/numpyconfig.h"  // NPY_VISIBILITY_HIDDEN
#include "blas_utils.h"
#include "npy_cblas.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if NPY_BLAS_CHECK_FPE_SUPPORT
/*
 * Static variable to cache runtime check of BLAS FPE support.
 * Will always be false (ignore all FPE) when accelerate is the compiled backend
 */
  #if defined(ACCELERATE_NEW_LAPACK)
static bool blas_supports_fpe = false;
  #else
static bool blas_supports_fpe = true;
  #endif // ACCELERATE_NEW_LAPACK

#endif // NPY_BLAS_CHECK_FPE_SUPPORT


NPY_VISIBILITY_HIDDEN bool
npy_blas_supports_fpe(void)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    return blas_supports_fpe;
#else
    return true;
#endif
}

NPY_VISIBILITY_HIDDEN bool
npy_set_blas_supports_fpe(bool value)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    blas_supports_fpe = (bool)value;
    return blas_supports_fpe;
#endif
    return true;  // ignore input not set up on this platform
}

NPY_VISIBILITY_HIDDEN int
npy_get_floatstatus_after_blas(void)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    if (!blas_supports_fpe){
        // BLAS does not support FPE and we need to return FPE state.
        // Instead of clearing and then grabbing state, just return
        // that no flags are set.
        return 0;
    }
#endif
    char *param = NULL;
    return npy_get_floatstatus_barrier(param);
}
