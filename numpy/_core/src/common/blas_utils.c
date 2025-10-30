#include <Python.h>

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
 */
 static bool blas_supports_fpe = true;

/*
 * ARM Scalable Matrix Extension (SME) raises all floating-point error flags
 * when it's used regardless of values or operations.  As a consequence,
 * when SME is used, all FPE state is lost and special handling is needed.
 *
 * For NumPy, SME is not currently used directly, but can be used via
 * BLAS / LAPACK libraries.  This function does a runtime check for whether
 * BLAS / LAPACK can use SME and special handling around FPE is required.
 *
 * This may be an Accelerate bug (at least OpenBLAS consider it that way)
 * but when we find an ARM system with SVE we do a runtime check for whether
 * FPEs are spuriously given.
 */
static inline int
set_BLAS_causes_spurious_FPEs(void)
{
    // These are all small, so just work on stack to not worry about error
    // handling.
    double *x = PyMem_Malloc(20*20*3*sizeof(double));
    if (x == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    double *y = x + 20*20;
    double *res = y + 20*20;

    npy_clear_floatstatus_barrier((char *)x);

    CBLAS_FUNC(cblas_dgemm)(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 20, 20, 20, 1.,
        x, 20, y, 20, 0., res, 20);
    PyMem_Free(x);

    int fpe_status = npy_get_floatstatus_barrier((char *)x);
    // Entries were all zero, so we shouldn't see any FPEs
    blas_supports_fpe = fpe_status != 0;
    return 0;
}

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

NPY_VISIBILITY_HIDDEN int
npy_blas_init(void)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    return set_BLAS_causes_spurious_FPEs();
#endif
    return 0;
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
