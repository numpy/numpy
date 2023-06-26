#include <stdarg.h>
// Order of includes matters here.
// npy_cblas.h and Accelerate both define CBLAS enums.  Accelerate
// has an option to not define them though (CBLAS_ENUM_DEFINED_H).
#include "npy_cblas.h"
#include "lapack/accelerate_legacy.h"

// Don't include CBLAS enums to avoid redefinition errors
#define CBLAS_ENUM_DEFINED_H
#include <Accelerate/Accelerate.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define EXPORT_ACCELERATE_WRAPPER __attribute__((visibility("default")))

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_sdsdot
(
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    const float *Y,
    const CBLAS_INT incY
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_sdsdot(N, alpha, X, incX, Y, incY);
    } else {
        ret = cblas_sdsdot$LEGACY(N, alpha, X, incX, Y, incY);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_dsdot
(
    const CBLAS_INT N,
    const float *X,
    const CBLAS_INT incX,
    const float *Y,
    const CBLAS_INT incY
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_dsdot(N, X, incX, Y, incY);
    } else {
        ret = cblas_dsdot$LEGACY(N, X, incX, Y, incY);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_sdot
(
    const CBLAS_INT N,
    const float  *X,
    const CBLAS_INT incX,
    const float  *Y,
    const CBLAS_INT incY
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_sdot(N, X, incX, Y, incY);
    } else {
        ret = cblas_sdot$LEGACY(N, X, incX, Y, incY);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_ddot
(
    const CBLAS_INT N,
    const double *X,
    const CBLAS_INT incX,
    const double *Y,
    const CBLAS_INT incY
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_ddot(N, X, incX, Y, incY);
    } else {
        ret = cblas_ddot$LEGACY(N, X, incX, Y, incY);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cdotu_sub
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *dotu
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cdotu_sub(N, X, incX, Y, incY, dotu);
    } else {
        cblas_cdotu_sub$LEGACY(N, X, incX, Y, incY, dotu);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cdotc_sub
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *dotc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cdotc_sub(N, X, incX, Y, incY, dotc);
    } else {
        cblas_cdotc_sub$LEGACY(N, X, incX, Y, incY, dotc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zdotu_sub
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *dotu
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zdotu_sub(N, X, incX, Y, incY, dotu);
    } else {
        cblas_zdotu_sub$LEGACY(N, X, incX, Y, incY, dotu);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zdotc_sub
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *dotc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zdotc_sub(N, X, incX, Y, incY, dotc);
    } else {
        cblas_zdotc_sub$LEGACY(N, X, incX, Y, incY, dotc);
    }
}

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_snrm2
(
    const CBLAS_INT N,
    const float *X,
    const CBLAS_INT incX
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_snrm2(N, X, incX);
    } else {
        ret = cblas_snrm2$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_sasum
(
    const CBLAS_INT N,
    const float *X,
    const CBLAS_INT incX
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_sasum(N, X, incX);
    } else {
        ret = cblas_sasum$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_dnrm2
(
    const CBLAS_INT N,
    const double *X,
    const CBLAS_INT incX
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_dnrm2(N, X, incX);
    } else {
        ret = cblas_dnrm2$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_dasum
(
    const CBLAS_INT N,
    const double *X,
    const CBLAS_INT incX
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_dasum(N, X, incX);
    } else {
        ret = cblas_dasum$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_scnrm2
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_scnrm2(N, X, incX);
    } else {
        ret = cblas_scnrm2$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
float
accelerate_cblas_scasum
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX
){
    float ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_scasum(N, X, incX);
    } else {
        ret = cblas_scasum$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_dznrm2
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_dznrm2(N, X, incX);
    } else {
        ret = cblas_dznrm2$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
double
accelerate_cblas_dzasum
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX
){
    double ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_dzasum(N, X, incX);
    } else {
        ret = cblas_dzasum$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
CBLAS_INDEX
accelerate_cblas_isamax
(
    const CBLAS_INT N,
    const float  *X,
    const CBLAS_INT incX
){
    CBLAS_INDEX ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_isamax(N, X, incX);
    } else {
        ret = cblas_isamax$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
CBLAS_INDEX
accelerate_cblas_idamax
(
    const CBLAS_INT N,
    const double *X,
    const CBLAS_INT incX
){
    CBLAS_INDEX ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_idamax(N, X, incX);
    } else {
        ret = cblas_idamax$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
CBLAS_INDEX
accelerate_cblas_icamax
(
    const CBLAS_INT N,
    const void   *X,
    const CBLAS_INT incX
){
    CBLAS_INDEX ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_icamax(N, X, incX);
    } else {
        ret = cblas_icamax$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
CBLAS_INDEX
accelerate_cblas_izamax
(
    const CBLAS_INT N,
    const void   *X,
    const CBLAS_INT incX
){
    CBLAS_INDEX ret;
    if(__builtin_available(macos 13.3, *)){
        ret = cblas_izamax(N, X, incX);
    } else {
        ret = cblas_izamax$LEGACY(N, X, incX);
    }
    return ret;
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sswap
(
    const CBLAS_INT N,
    float *X,
    const CBLAS_INT incX,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sswap(N, X, incX, Y, incY);
    } else {
        cblas_sswap$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_scopy
(
    const CBLAS_INT N,
    const float *X,
    const CBLAS_INT incX,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_scopy(N, X, incX, Y, incY);
    } else {
        cblas_scopy$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_saxpy
(
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_saxpy(N, alpha, X, incX, Y, incY);
    } else {
        cblas_saxpy$LEGACY(N, alpha, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dswap
(
    const CBLAS_INT N,
    double *X,
    const CBLAS_INT incX,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dswap(N, X, incX, Y, incY);
    } else {
        cblas_dswap$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dcopy
(
    const CBLAS_INT N,
    const double *X,
    const CBLAS_INT incX,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dcopy(N, X, incX, Y, incY);
    } else {
        cblas_dcopy$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_daxpy
(
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_daxpy(N, alpha, X, incX, Y, incY);
    } else {
        cblas_daxpy$LEGACY(N, alpha, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cswap
(
    const CBLAS_INT N,
    void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cswap(N, X, incX, Y, incY);
    } else {
        cblas_cswap$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ccopy
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ccopy(N, X, incX, Y, incY);
    } else {
        cblas_ccopy$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_caxpy
(
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_caxpy(N, alpha, X, incX, Y, incY);
    } else {
        cblas_caxpy$LEGACY(N, alpha, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zswap
(
    const CBLAS_INT N,
    void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zswap(N, X, incX, Y, incY);
    } else {
        cblas_zswap$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zcopy
(
    const CBLAS_INT N,
    const void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zcopy(N, X, incX, Y, incY);
    } else {
        cblas_zcopy$LEGACY(N, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zaxpy
(
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zaxpy(N, alpha, X, incX, Y, incY);
    } else {
        cblas_zaxpy$LEGACY(N, alpha, X, incX, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_srotg
(
    float *a,
    float *b,
    float *c,
    float *s
){
    if(__builtin_available(macos 13.3, *)){
        cblas_srotg(a, b, c, s);
    } else {
        cblas_srotg$LEGACY(a, b, c, s);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_srotmg
(
    float *d1,
    float *d2,
    float *b1,
    const float b2,
    float *P
){
    if(__builtin_available(macos 13.3, *)){
        cblas_srotmg(d1, d2, b1, b2, P);
    } else {
        cblas_srotmg$LEGACY(d1, d2, b1, b2, P);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_srot
(
    const CBLAS_INT N,
    float *X,
    const CBLAS_INT incX,
    float *Y,
    const CBLAS_INT incY,
    const float c,
    const float s
){
    if(__builtin_available(macos 13.3, *)){
        cblas_srot(N, X, incX, Y, incY, c, s);
    } else {
        cblas_srot$LEGACY(N, X, incX, Y, incY, c, s);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_srotm
(
    const CBLAS_INT N,
    float *X,
    const CBLAS_INT incX,
    float *Y,
    const CBLAS_INT incY,
    const float *P
){
    if(__builtin_available(macos 13.3, *)){
        cblas_srotm(N, X, incX, Y, incY, P);
    } else {
        cblas_srotm$LEGACY(N, X, incX, Y, incY, P);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_drotg
(
    double *a,
    double *b,
    double *c,
    double *s
){
    if(__builtin_available(macos 13.3, *)){
        cblas_drotg(a, b, c, s);
    } else {
        cblas_drotg$LEGACY(a, b, c, s);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_drotmg
(
    double *d1,
    double *d2,
    double *b1,
    const double b2,
    double *P
){
    if(__builtin_available(macos 13.3, *)){
        cblas_drotmg(d1, d2, b1, b2, P);
    } else {
        cblas_drotmg$LEGACY(d1, d2, b1, b2, P);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_drot
(
    const CBLAS_INT N,
    double *X,
    const CBLAS_INT incX,
    double *Y,
    const CBLAS_INT incY,
    const double c,
    const double  s
){
    if(__builtin_available(macos 13.3, *)){
        cblas_drot(N, X, incX, Y, incY, c, s);
    } else {
        cblas_drot$LEGACY(N, X, incX, Y, incY, c, s);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_drotm
(
    const CBLAS_INT N,
    double *X,
    const CBLAS_INT incX,
    double *Y,
    const CBLAS_INT incY,
    const double *P
){
    if(__builtin_available(macos 13.3, *)){
        cblas_drotm(N, X, incX, Y, incY, P);
    } else {
        cblas_drotm$LEGACY(N, X, incX, Y, incY, P);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sscal
(
    const CBLAS_INT N,
    const float alpha,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sscal(N, alpha, X, incX);
    } else {
        cblas_sscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dscal
(
    const CBLAS_INT N,
    const double alpha,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dscal(N, alpha, X, incX);
    } else {
        cblas_dscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cscal
(
    const CBLAS_INT N,
    const void *alpha,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cscal(N, alpha, X, incX);
    } else {
        cblas_cscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zscal
(
    const CBLAS_INT N,
    const void *alpha,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zscal(N, alpha, X, incX);
    } else {
        cblas_zscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_csscal
(
    const CBLAS_INT N,
    const float alpha,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_csscal(N, alpha, X, incX);
    } else {
        cblas_csscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zdscal
(
    const CBLAS_INT N,
    const double alpha,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zdscal(N, alpha, X, incX);
    } else {
        cblas_zdscal$LEGACY(N, alpha, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sgemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *X,
    const CBLAS_INT incX,
    const float beta,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_sgemv$LEGACY(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sgbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT KL,
    const CBLAS_INT KU,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *X,
    const CBLAS_INT incX,
    const float beta,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_sgbmv$LEGACY(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_strmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const float *A,
    const CBLAS_INT lda,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_strmv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_strmv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_stbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float *A,
    const CBLAS_INT lda,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_stbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_stbmv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_stpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const float *Ap,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_stpmv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_stpmv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_strsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const float *A,
    const CBLAS_INT lda,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_strsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_strsv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_stbsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float *A,
    const CBLAS_INT lda,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_stbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_stbsv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_stpsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const float *Ap,
    float *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_stpsv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_stpsv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dgemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *X,
    const CBLAS_INT incX,
    const double beta,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_dgemv$LEGACY(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dgbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT KL,
    const CBLAS_INT KU,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *X,
    const CBLAS_INT incX,
    const double beta,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_dgbmv$LEGACY(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtrmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const double *A,
    const CBLAS_INT lda,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_dtrmv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double *A,
    const CBLAS_INT lda,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_dtbmv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const double *Ap,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtpmv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_dtpmv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtrsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const double *A,
    const CBLAS_INT lda,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_dtrsv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtbsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double *A,
    const CBLAS_INT lda,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_dtbsv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtpsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const double *Ap,
    double *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtpsv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_dtpsv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cgemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_cgemv$LEGACY(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cgbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT KL,
    const CBLAS_INT KU,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_cgbmv$LEGACY(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctrmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_ctrmv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_ctbmv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *Ap,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctpmv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_ctpmv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctrsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_ctrsv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctbsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_ctbsv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctpsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *Ap,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctpsv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_ctpsv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zgemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_zgemv$LEGACY(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zgbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT KL,
    const CBLAS_INT KU,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zgbmv(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_zgbmv$LEGACY(order, TransA, M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztrmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztrmv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_ztrmv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztbmv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_ztbmv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *Ap,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztpmv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_ztpmv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztrsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    } else {
        cblas_ztrsv$LEGACY(order, Uplo, TransA, Diag, N, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztbsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *A,
    const CBLAS_INT lda,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztbsv(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    } else {
        cblas_ztbsv$LEGACY(order, Uplo, TransA, Diag, N, K, A, lda, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztpsv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT N,
    const void *Ap,
    void *X,
    const CBLAS_INT incX
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztpsv(order, Uplo, TransA, Diag, N, Ap, X, incX);
    } else {
        cblas_ztpsv$LEGACY(order, Uplo, TransA, Diag, N, Ap, X, incX);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssymv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *X,
    const CBLAS_INT incX,
    const float beta,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_ssymv$LEGACY(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *X,
    const CBLAS_INT incX,
    const float beta,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_ssbmv$LEGACY(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sspmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *Ap,
    const float *X,
    const CBLAS_INT incX,
    const float beta,
    float *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        cblas_sspmv$LEGACY(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sger
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    const float *Y,
    const CBLAS_INT incY,
    float *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sger(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_sger$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssyr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    float *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssyr(order, Uplo, N, alpha, X, incX, A, lda);
    } else {
        cblas_ssyr$LEGACY(order, Uplo, N, alpha, X, incX, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sspr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    float *Ap
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sspr(order, Uplo, N, alpha, X, incX, Ap);
    } else {
        cblas_sspr$LEGACY(order, Uplo, N, alpha, X, incX, Ap);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssyr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    const float *Y,
    const CBLAS_INT incY,
    float *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_ssyr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sspr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const float *X,
    const CBLAS_INT incX,
    const float *Y,
    const CBLAS_INT incY,
    float *A
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sspr2(order, Uplo, N, alpha, X, incX, Y, incY, A);
    } else {
        cblas_sspr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsymv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *X,
    const CBLAS_INT incX,
    const double beta,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsymv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_dsymv$LEGACY(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *X,
    const CBLAS_INT incX,
    const double beta,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_dsbmv$LEGACY(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dspmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *Ap,
    const double *X,
    const CBLAS_INT incX,
    const double beta,
    double *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dspmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        cblas_dspmv$LEGACY(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dger
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    const double *Y,
    const CBLAS_INT incY,
    double *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_dger$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsyr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    double *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsyr(order, Uplo, N, alpha, X, incX, A, lda);
    } else {
        cblas_dsyr$LEGACY(order, Uplo, N, alpha, X, incX, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dspr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    double *Ap
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dspr(order, Uplo, N, alpha, X, incX, Ap);
    } else {
        cblas_dspr$LEGACY(order, Uplo, N, alpha, X, incX, Ap);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsyr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    const double *Y,
    const CBLAS_INT incY,
    double *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsyr2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_dsyr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dspr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const double *X,
    const CBLAS_INT incX,
    const double *Y,
    const CBLAS_INT incY,
    double *A
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dspr2(order, Uplo, N, alpha, X, incX, Y, incY, A);
    } else {
        cblas_dspr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_chemv$LEGACY(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_chbmv$LEGACY(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *Ap,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        cblas_chpmv$LEGACY(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cgeru
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cgeru(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_cgeru$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cgerc
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cgerc(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_cgerc$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cher
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const void *X,
    const CBLAS_INT incX,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cher(order, Uplo, N, alpha, X, incX, A, lda);
    } else {
        cblas_cher$LEGACY(order, Uplo, N, alpha, X, incX, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chpr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const float alpha,
    const void *X,
    const CBLAS_INT incX,
    void *A
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chpr(order, Uplo, N, alpha, X, incX, A);
    } else {
        cblas_chpr$LEGACY(order, Uplo, N, alpha, X, incX, A);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cher2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_cher2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chpr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *Ap
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap);
    } else {
        cblas_chpr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, Ap);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhemv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhemv(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_zhemv$LEGACY(order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhbmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhbmv(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    } else {
        cblas_zhbmv$LEGACY(order, Uplo, N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhpmv
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *Ap,
    const void *X,
    const CBLAS_INT incX,
    const void *beta,
    void *Y,
    const CBLAS_INT incY
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhpmv(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    } else {
        cblas_zhpmv$LEGACY(order, Uplo, N, alpha, Ap, X, incX, beta, Y, incY);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zgeru
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zgeru(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_zgeru$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zgerc
(
    const enum CBLAS_ORDER order,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zgerc(order, M, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_zgerc$LEGACY(order, M, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zher
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const void *X,
    const CBLAS_INT incX,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zher(order, Uplo, N, alpha, X, incX, A, lda);
    } else {
        cblas_zher$LEGACY(order, Uplo, N, alpha, X, incX, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhpr
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const double alpha,
    const void *X,
    const CBLAS_INT incX,
    void *A
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhpr(order, Uplo, N, alpha, X, incX, A);
    } else {
        cblas_zhpr$LEGACY(order, Uplo, N, alpha, X, incX, A);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zher2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *A,
    const CBLAS_INT lda
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zher2(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    } else {
        cblas_zher2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, A, lda);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhpr2
(
    const enum CBLAS_ORDER order,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT N,
    const void *alpha,
    const void *X,
    const CBLAS_INT incX,
    const void *Y,
    const CBLAS_INT incY,
    void *Ap
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhpr2(order, Uplo, N, alpha, X, incX, Y, incY, Ap);
    } else {
        cblas_zhpr2$LEGACY(order, Uplo, N, alpha, X, incX, Y, incY, Ap);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_sgemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *B,
    const CBLAS_INT ldb,
    const float beta,
    float *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_sgemm$LEGACY(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssymm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *B,
    const CBLAS_INT ldb,
    const float beta,
    float *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_ssymm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssyrk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float beta,
    float *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_ssyrk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ssyr2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    const float *B,
    const CBLAS_INT ldb,
    const float beta,
    float *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ssyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_ssyr2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_strmm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    float *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_strmm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_strsm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const float alpha,
    const float *A,
    const CBLAS_INT lda,
    float *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_strsm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dgemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *B,
    const CBLAS_INT ldb,
    const double beta,
    double *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_dgemm$LEGACY(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsymm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *B,
    const CBLAS_INT ldb,
    const double beta,
    double *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_dsymm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsyrk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double beta,
    double *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_dsyrk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dsyr2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    const double *B,
    const CBLAS_INT ldb,
    const double beta,
    double *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_dsyr2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtrmm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    double *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_dtrmm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_dtrsm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const double alpha,
    const double *A,
    const CBLAS_INT lda,
    double *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_dtrsm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cgemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_cgemm$LEGACY(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_csymm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_csymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_csymm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_csyrk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_csyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_csyrk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_csyr2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_csyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_csyr2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctrmm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    void *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_ctrmm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ctrsm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    void *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ctrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_ctrsm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zgemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_zgemm$LEGACY(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zsymm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zsymm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_zsymm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zsyrk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_zsyrk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zsyr2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zsyr2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_zsyr2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztrmm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    void *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_ztrmm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_ztrsm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_DIAG Diag,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    void *B,
    const CBLAS_INT ldb
){
    if(__builtin_available(macos 13.3, *)){
        cblas_ztrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    } else {
        cblas_ztrsm$LEGACY(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_chemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_chemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_chemm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cherk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const float alpha,
    const void *A,
    const CBLAS_INT lda,
    const float beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_cherk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_cher2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const float beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_cher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_cher2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zhemm
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_SIDE Side,
    const enum CBLAS_UPLO Uplo,
    const CBLAS_INT M,
    const CBLAS_INT N,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const void *beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zhemm(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_zhemm$LEGACY(Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zherk
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const double alpha,
    const void *A,
    const CBLAS_INT lda,
    const double beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    } else {
        cblas_zherk$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
    }
}

EXPORT_ACCELERATE_WRAPPER
void
accelerate_cblas_zher2k
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_UPLO Uplo,
    const enum CBLAS_TRANSPOSE Trans,
    const CBLAS_INT N,
    const CBLAS_INT K,
    const void *alpha,
    const void *A,
    const CBLAS_INT lda,
    const void *B,
    const CBLAS_INT ldb,
    const double beta,
    void *C,
    const CBLAS_INT ldc
){
    if(__builtin_available(macos 13.3, *)){
        cblas_zher2k(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_zher2k$LEGACY(Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// void
// accelerate_cblas_xerbla
// (
//     CBLAS_INT p,
//     const char *rout,
//     const char *form,
//     ...
// ){
//     va_list va;
//     va_start(va, form);
//     if(__builtin_available(macos 13.3, *)){
//         cblas_xerbla(p, rout, form, ...);
//     } else {
//         cblas_xerbla$LEGACY(p, rout, form, ...);
//     }
// }

#undef EXPORT_ACCELERATE_WRAPPER

#ifdef __cplusplus
}
#endif
