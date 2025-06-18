#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "numpy/numpyconfig.h"  // NPY_VISIBILITY_HIDDEN
#include "numpy/npy_math.h"     // npy_get_floatstatus_barrier
#include "blas_utils.h"

#if NPY_BLAS_CHECK_FPE_SUPPORT

/* Return whether we're running on macOS 15.4 or later
 */
static inline bool
is_macOS_version_15_4_or_later(void){
#if !defined(__APPLE__)
    return false;
#else
    char *osProductVersion = NULL;
    size_t size = 0;
    bool ret = false;

    // Query how large OS version string should be
    if(-1 == sysctlbyname("kern.osproductversion", NULL, &size, NULL, 0)){
        goto cleanup;
    }

    osProductVersion = malloc(size + 1);

    // Get the OS version string
    if(-1 == sysctlbyname("kern.osproductversion", osProductVersion, &size, NULL, 0)){
        goto cleanup;
    }

    osProductVersion[size] = '\0';

    // Parse the version string
    int major = 0, minor = 0;
    if(2 > sscanf(osProductVersion, "%d.%d", &major, &minor)) {
        goto cleanup;
    }

    if(major > 15 || (major == 15 && minor >= 4)) {
        ret = true;
    }

cleanup:
    if(osProductVersion){
        free(osProductVersion);
    }

    return ret;
#endif
}

/* ARM Scalable Matrix Extension (SME) raises all floating-point error flags
 * when it's used regardless of values or operations.  As a consequence,
 * when SME is used, all FPE state is lost and special handling is needed.
 *
 * For NumPy, SME is not currently used directly, but can be used via
 * BLAS / LAPACK libraries.  This function does a runtime check for whether
 * BLAS / LAPACK can use SME and special handling around FPE is required.
 */
static inline bool
BLAS_can_use_ARM_SME(void)
{
#if defined(__APPLE__) && defined(__aarch64__) && defined(ACCELERATE_NEW_LAPACK)
    // ARM SME can be used by Apple's Accelerate framework for BLAS / LAPACK
    // - macOS 15.4+
    // - Apple silicon M4+

    // Does OS / Accelerate support ARM SME?
    if(!is_macOS_version_15_4_or_later()){
        return false;
    }

    // Does hardware support SME?
    int has_SME = 0;
    size_t size = sizeof(has_SME);
    if(-1 == sysctlbyname("hw.optional.arm.FEAT_SME", &has_SME, &size, NULL, 0)){
        return false;
    }

    if(has_SME){
        return true;
    }
#endif

    // default assume SME is not used
    return false;
}

/* Static variable to cache runtime check of BLAS FPE support.
 */
static bool blas_supports_fpe = true;

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

NPY_VISIBILITY_HIDDEN void
npy_blas_init(void)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    blas_supports_fpe = !BLAS_can_use_ARM_SME();
#endif
}

NPY_VISIBILITY_HIDDEN int
npy_get_floatstatus_after_blas(void)
{
#if NPY_BLAS_CHECK_FPE_SUPPORT
    if(!blas_supports_fpe){
        // BLAS does not support FPE and we need to return FPE state.
        // Instead of clearing and then grabbing state, just return
        // that no flags are set.
        return 0;
    }
#endif
    char *param = NULL;
    return npy_get_floatstatus_barrier(param);
}
