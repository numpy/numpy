#include "npy_blas_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef BLAS_INT         fortran_int;

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

typedef float             fortran_real;
typedef double            fortran_doublereal;
typedef f2c_complex       fortran_complex;
typedef f2c_doublecomplex fortran_doublecomplex;

void
BLAS_FUNC(sgeev)(char *jobvl, char *jobvr, fortran_int *n,
             float a[], fortran_int *lda, float wr[], float wi[],
             float vl[], fortran_int *ldvl, float vr[], fortran_int *ldvr,
             float work[], fortran_int lwork[],
             fortran_int *info);
void
BLAS_FUNC(dgeev)(char *jobvl, char *jobvr, fortran_int *n,
             double a[], fortran_int *lda, double wr[], double wi[],
             double vl[], fortran_int *ldvl, double vr[], fortran_int *ldvr,
             double work[], fortran_int lwork[],
             fortran_int *info);
void
BLAS_FUNC(cgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_complex a[], fortran_int *lda,
             f2c_complex w[],
             f2c_complex vl[], fortran_int *ldvl,
             f2c_complex vr[], fortran_int *ldvr,
             f2c_complex work[], fortran_int *lwork,
             float rwork[],
             fortran_int *info);
void
BLAS_FUNC(zgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_doublecomplex a[], fortran_int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], fortran_int *ldvl,
             f2c_doublecomplex vr[], fortran_int *ldvr,
             f2c_doublecomplex work[], fortran_int *lwork,
             double rwork[],
             fortran_int *info);

void
BLAS_FUNC(ssyevd)(char *jobz, char *uplo, fortran_int *n,
              float a[], fortran_int *lda, float w[], float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *liwork,
              fortran_int *info);
void
BLAS_FUNC(dsyevd)(char *jobz, char *uplo, fortran_int *n,
              double a[], fortran_int *lda, double w[], double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *liwork,
              fortran_int *info);
void
BLAS_FUNC(cheevd)(char *jobz, char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float w[], f2c_complex work[],
              fortran_int *lwork, float rwork[], fortran_int *lrwork, fortran_int iwork[],
              fortran_int *liwork,
              fortran_int *info);
void
BLAS_FUNC(zheevd)(char *jobz, char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double w[], f2c_doublecomplex work[],
              fortran_int *lwork, double rwork[], fortran_int *lrwork, fortran_int iwork[],
              fortran_int *liwork,
              fortran_int *info);

void
BLAS_FUNC(sgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              float a[], fortran_int *lda, float b[], fortran_int *ldb,
              float s[], float *rcond, fortran_int *rank,
              float work[], fortran_int *lwork, fortran_int iwork[],
              fortran_int *info);
void
BLAS_FUNC(dgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              double a[], fortran_int *lda, double b[], fortran_int *ldb,
              double s[], double *rcond, fortran_int *rank,
              double work[], fortran_int *lwork, fortran_int iwork[],
              fortran_int *info);
void
BLAS_FUNC(cgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              float s[], float *rcond, fortran_int *rank,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[],
              fortran_int *info);
void
BLAS_FUNC(zgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              double s[], double *rcond, fortran_int *rank,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[],
              fortran_int *info);

void
BLAS_FUNC(dgeqrf)(fortran_int *m, fortran_int *n, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);
void
BLAS_FUNC(zgeqrf)(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex tau[], f2c_doublecomplex work[],
              fortran_int *lwork, fortran_int *info);

void
BLAS_FUNC(dorgqr)(fortran_int *m, fortran_int *n, fortran_int *k, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);
void
BLAS_FUNC(zungqr)(fortran_int *m, fortran_int *n, fortran_int *k, f2c_doublecomplex a[],
              fortran_int *lda, f2c_doublecomplex tau[],
              f2c_doublecomplex work[], fortran_int *lwork, fortran_int *info);

void
BLAS_FUNC(sgesv)(fortran_int *n, fortran_int *nrhs,
             float a[], fortran_int *lda,
             fortran_int ipiv[],
             float b[], fortran_int *ldb,
             fortran_int *info);
void
BLAS_FUNC(dgesv)(fortran_int *n, fortran_int *nrhs,
             double a[], fortran_int *lda,
             fortran_int ipiv[],
             double b[], fortran_int *ldb,
             fortran_int *info);
void
BLAS_FUNC(cgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_complex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_complex b[], fortran_int *ldb,
             fortran_int *info);
void
BLAS_FUNC(zgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_doublecomplex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_doublecomplex b[], fortran_int *ldb,
             fortran_int *info);

fortran_int
BLAS_FUNC(sgetrf)(fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
fortran_int
BLAS_FUNC(dgetrf)(fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
fortran_int
BLAS_FUNC(cgetrf)(fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
fortran_int
BLAS_FUNC(zgetrf)(fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);

void
BLAS_FUNC(spotrf)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(dpotrf)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(cpotrf)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(zpotrf)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);

void
BLAS_FUNC(sgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda, float s[], float u[],
              fortran_int *ldu, float vt[], fortran_int *ldvt, float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
void
BLAS_FUNC(dgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda, double s[], double u[],
              fortran_int *ldu, double vt[], fortran_int *ldvt, double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
void
BLAS_FUNC(cgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float s[], f2c_complex u[], fortran_int *ldu,
              f2c_complex vt[], fortran_int *ldvt,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[], fortran_int *info);
void
BLAS_FUNC(zgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double s[], f2c_doublecomplex u[], fortran_int *ldu,
              f2c_doublecomplex vt[], fortran_int *ldvt,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[], fortran_int *info);

void
BLAS_FUNC(spotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              float a[], fortran_int *lda,
              float b[], fortran_int *ldb,
              fortran_int *info);
void
BLAS_FUNC(dpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              double a[], fortran_int *lda,
              double b[], fortran_int *ldb,
              fortran_int *info);
void
BLAS_FUNC(cpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              fortran_int *info);
void
BLAS_FUNC(zpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              fortran_int *info);

void
BLAS_FUNC(spotri)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(dpotri)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(cpotri)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
void
BLAS_FUNC(zpotri)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);

fortran_int
BLAS_FUNC(scopy)(fortran_int *n,
             float *sx, fortran_int *incx,
             float *sy, fortran_int *incy);
fortran_int
BLAS_FUNC(dcopy)(fortran_int *n,
             double *sx, fortran_int *incx,
             double *sy, fortran_int *incy);
fortran_int
BLAS_FUNC(ccopy)(fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
fortran_int
BLAS_FUNC(zcopy)(fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);

float
BLAS_FUNC(sdot)(fortran_int *n,
            float *sx, fortran_int *incx,
            float *sy, fortran_int *incy);
double
BLAS_FUNC(ddot)(fortran_int *n,
            double *sx, fortran_int *incx,
            double *sy, fortran_int *incy);
void
BLAS_FUNC(cdotu)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
void
BLAS_FUNC(zdotu)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);
void
BLAS_FUNC(cdotc)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
void
BLAS_FUNC(zdotc)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);

void
BLAS_FUNC(sgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             float *alpha,
             float *a, fortran_int *lda,
             float *b, fortran_int *ldb,
             float *beta,
             float *c, fortran_int *ldc);
void
BLAS_FUNC(dgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             double *alpha,
             double *a, fortran_int *lda,
             double *b, fortran_int *ldb,
             double *beta,
             double *c, fortran_int *ldc);
void
BLAS_FUNC(cgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_complex *alpha,
             f2c_complex *a, fortran_int *lda,
             f2c_complex *b, fortran_int *ldb,
             f2c_complex *beta,
             f2c_complex *c, fortran_int *ldc);
void
BLAS_FUNC(zgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, fortran_int *lda,
             f2c_doublecomplex *b, fortran_int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, fortran_int *ldc);

#ifdef __cplusplus
}
#endif