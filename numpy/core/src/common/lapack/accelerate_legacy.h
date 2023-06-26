#include <sys/cdefs.h>

/*
 * The macros and additional headers here expose Accelerate's legacy
 * BLAS / LAPACK APIs with unique names.  Each API will be suffixed
 * with '$LEGACY' for use in source code.  Those will be hooked up
 * to the legacy binary symbols.
 *
 * Examples:
 * - dgemm
 *     source code: dgemm$LEGACY(...)
 *     binary symbol: _dgemm_
 * - cblas_dgemm
 *     source code: cblas_dgemm$LEGACY(...)
 *     binary symbol: _cblas_dgemm
 */

#define __TEMPLATE_FUNC(func)       __CONCAT(func,$LEGACY)
#define __TEMPLATE_ALIAS(sym)       __asm("_" __STRING(sym))

#include "lapack/accelerate_legacy_blas.h"
#include "lapack/accelerate_legacy_cblas.h"
#include "lapack/accelerate_legacy_lapack.h"

#undef __TEMPLATE_FUNC
#undef __TEMPLATE_ALIAS
