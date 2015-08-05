/* This is a collection of ugly hacks to circumvent a bug in
 * Apple Accelerate framework's SGEMV subroutine.
 *
 * See: https://github.com/numpy/numpy/issues/4007
 *
 * SGEMV in Accelerate framework will segfault on MacOS X version 10.9
 * (aka Mavericks) if arrays are not aligned to 32 byte boundaries
 * and the CPU supports AVX instructions. This can produce segfaults
 * in np.dot.
 *
 * This patch overshadows the symbols cblas_sgemv, sgemv_ and sgemv
 * exported by Accelerate to produce the correct behavior. The MacOS X
 * version and CPU specs are checked on module import. If Mavericks and
 * AVX are detected the call to SGEMV is emulated with a call to SGEMM
 * if the arrays are not 32 byte aligned. If the exported symbols cannot
 * be overshadowed on module import, a fatal error is produced and the
 * process aborts. All the fixes are in a self-contained C file
 * and do not alter the multiarray C code. The patch is not applied
 * unless NumPy is configured to link with Apple's Accelerate
 * framework.
 *
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"

#include <string.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>

/* ----------------------------------------------------------------- */
/* Original cblas_sgemv */

#define VECLIB_FILE "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/vecLib"

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
extern void cblas_xerbla(int info, const char *rout, const char *form, ...);

typedef void cblas_sgemv_t(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float  *A, const int lda,
                 const float  *X, const int incX,
                 const float beta, float  *Y, const int incY);

typedef void cblas_sgemm_t(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const float alpha, const float  *A, const int lda,
                 const float  *B, const int ldb,
                 const float beta, float  *C, const int incC);

typedef void fortran_sgemv_t( const char* trans, const int* m, const int* n,
             const float* alpha, const float* A, const int* ldA,
             const float* X, const int* incX,
             const float* beta, float* Y, const int* incY );

static void *veclib = NULL;
static cblas_sgemv_t *accelerate_cblas_sgemv = NULL;
static cblas_sgemm_t *accelerate_cblas_sgemm = NULL;
static fortran_sgemv_t *accelerate_sgemv = NULL;
static int AVX_and_10_9 = 0;

/* Dynamic check for AVX support
 * __builtin_cpu_supports("avx") is available in gcc 4.8,
 * but clang and icc do not currently support it. */
#define cpu_supports_avx()\
(system("sysctl -n machdep.cpu.features | grep -q AVX") == 0)

/* Check if we are using MacOS X version 10.9 */
#define using_mavericks()\
(system("sw_vers -productVersion | grep -q 10\\.9\\.") == 0)

__attribute__((destructor))
static void unloadlib(void)
{
   if (veclib) dlclose(veclib);
}

__attribute__((constructor))
static void loadlib()
/* automatically executed on module import */
{
    char errormsg[1024];
    int AVX, MAVERICKS;
    memset((void*)errormsg, 0, sizeof(errormsg));
    /* check if the CPU supports AVX */
    AVX = cpu_supports_avx();
    /* check if the OS is MacOS X Mavericks */
    MAVERICKS = using_mavericks();
    /* we need the workaround when the CPU supports
     * AVX and the OS version is Mavericks */
    AVX_and_10_9 = AVX && MAVERICKS;
    /* load vecLib */
    veclib = dlopen(VECLIB_FILE, RTLD_LOCAL | RTLD_FIRST);
    if (!veclib) {
        veclib = NULL;
        snprintf(errormsg, sizeof(errormsg),
                 "Failed to open vecLib from location '%s'.", VECLIB_FILE);
        Py_FatalError(errormsg); /* calls abort() and dumps core */
    }
    /* resolve Fortran SGEMV from Accelerate */
    accelerate_sgemv = (fortran_sgemv_t*) dlsym(veclib, "sgemv_");
    if (!accelerate_sgemv) {
        unloadlib();
        Py_FatalError("Failed to resolve symbol 'sgemv_'.");
    }
    /* resolve cblas_sgemv from Accelerate */
    accelerate_cblas_sgemv = (cblas_sgemv_t*) dlsym(veclib, "cblas_sgemv");
    if (!accelerate_cblas_sgemv) {
        unloadlib();
        Py_FatalError("Failed to resolve symbol 'cblas_sgemv'.");
    }
    /* resolve cblas_sgemm from Accelerate */
    accelerate_cblas_sgemm = (cblas_sgemm_t*) dlsym(veclib, "cblas_sgemm");
    if (!accelerate_cblas_sgemm) {
        unloadlib();
        Py_FatalError("Failed to resolve symbol 'cblas_sgemm'.");
    }
}

/* ----------------------------------------------------------------- */
/* Fortran SGEMV override */

void sgemv_( const char* trans, const int* m, const int* n,
             const float* alpha, const float* A, const int* ldA,
             const float* X, const int* incX,
             const float* beta, float* Y, const int* incY )
{
    /* It is safe to use the original SGEMV if we are not using AVX on Mavericks
     * or the input arrays A, X and Y are all aligned on 32 byte boundaries. */
    #define BADARRAY(x) (((npy_intp)(void*)x) % 32)
    const int use_sgemm = AVX_and_10_9 && (BADARRAY(A) || BADARRAY(X) || BADARRAY(Y));
    if (!use_sgemm) {
        accelerate_sgemv(trans,m,n,alpha,A,ldA,X,incX,beta,Y,incY);
        return;
    }

    /* Arrays are misaligned, the CPU supports AVX, and we are running
     * Mavericks.
     *
     * Emulation of SGEMV with SGEMM:
     *
     * SGEMV allows vectors to be strided. SGEMM requires all arrays to be
     * contiguous along the leading dimension. To emulate striding in SGEMV
     * with the leading dimension arguments in SGEMM we compute
     *
     *    Y = alpha * op(A) @ X + beta * Y
     *
     * as
     *
     *    Y.T = alpha * X.T @ op(A).T + beta * Y.T
     *
     * Because Fortran uses column major order and X.T and Y.T are row vectors,
     * the leading dimensions of X.T and Y.T in SGEMM become equal to the
     * strides of the the column vectors X and Y in SGEMV. */

    switch (*trans) {
        case 'T':
        case 't':
        case 'C':
        case 'c':
            accelerate_cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                1, *n, *m, *alpha, X, *incX, A, *ldA, *beta, Y, *incY );
            break;
        case 'N':
        case 'n':
            accelerate_cblas_sgemm( CblasColMajor, CblasNoTrans, CblasTrans,
                1, *m, *n, *alpha, X, *incX, A, *ldA, *beta, Y, *incY );
            break;
        default:
            cblas_xerbla(1, "SGEMV", "Illegal transpose setting: %c\n", *trans);
    }
}

/* ----------------------------------------------------------------- */
/* Override for an alias symbol for sgemv_ in Accelerate */

void sgemv (char *trans,
            const int *m, const int *n,
            const float *alpha,
            const float *A, const int *lda,
            const float *B, const int *incB,
            const float *beta,
            float *C, const int *incC)
{
    sgemv_(trans,m,n,alpha,A,lda,B,incB,beta,C,incC);
}

/* ----------------------------------------------------------------- */
/* cblas_sgemv override, based on Netlib CBLAS code */

void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float  *A, const int lda,
                 const float  *X, const int incX, const float beta,
                 float  *Y, const int incY)
{
   char TA;
   if (order == CblasColMajor)
   {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else
      {
         cblas_xerbla(2, "cblas_sgemv","Illegal TransA setting, %d\n", TransA);
      }
      sgemv_(&TA, &M, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
   }
   else if (order == CblasRowMajor)
   {
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans) TA = 'N';
      else
      {
         cblas_xerbla(2, "cblas_sgemv", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      sgemv_(&TA, &N, &M, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
   }
   else
      cblas_xerbla(1, "cblas_sgemv", "Illegal Order setting, %d\n", order);
}
