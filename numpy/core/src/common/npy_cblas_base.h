/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_

float  BLASNAME(cblas_sdsdot)(const BLASINT N, const float alpha, const float *X,
                              const BLASINT incX, const float *Y, const BLASINT incY);
double BLASNAME(cblas_dsdot)(const BLASINT N, const float *X, const BLASINT incX, const float *Y,
                             const BLASINT incY);
float  BLASNAME(cblas_sdot)(const BLASINT N, const float  *X, const BLASINT incX,
                            const float  *Y, const BLASINT incY);
double BLASNAME(cblas_ddot)(const BLASINT N, const double *X, const BLASINT incX,
                            const double *Y, const BLASINT incY);

/*
 * Functions having prefixes Z and C only
 */
void   BLASNAME(cblas_cdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotu);
void   BLASNAME(cblas_cdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotc);

void   BLASNAME(cblas_zdotu_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotu);
void   BLASNAME(cblas_zdotc_sub)(const BLASINT N, const void *X, const BLASINT incX,
                                 const void *Y, const BLASINT incY, void *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
float  BLASNAME(cblas_snrm2)(const BLASINT N, const float *X, const BLASINT incX);
float  BLASNAME(cblas_sasum)(const BLASINT N, const float *X, const BLASINT incX);

double BLASNAME(cblas_dnrm2)(const BLASINT N, const double *X, const BLASINT incX);
double BLASNAME(cblas_dasum)(const BLASINT N, const double *X, const BLASINT incX);

float  BLASNAME(cblas_scnrm2)(const BLASINT N, const void *X, const BLASINT incX);
float  BLASNAME(cblas_scasum)(const BLASINT N, const void *X, const BLASINT incX);

double BLASNAME(cblas_dznrm2)(const BLASINT N, const void *X, const BLASINT incX);
double BLASNAME(cblas_dzasum)(const BLASINT N, const void *X, const BLASINT incX);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX BLASNAME(cblas_isamax)(const BLASINT N, const float  *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_idamax)(const BLASINT N, const double *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_icamax)(const BLASINT N, const void   *X, const BLASINT incX);
CBLAS_INDEX BLASNAME(cblas_izamax)(const BLASINT N, const void   *X, const BLASINT incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void BLASNAME(cblas_sswap)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);
void BLASNAME(cblas_scopy)(const BLASINT N, const float *X, const BLASINT incX,
                           float *Y, const BLASINT incY);
void BLASNAME(cblas_saxpy)(const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, float *Y, const BLASINT incY);

void BLASNAME(cblas_dswap)(const BLASINT N, double *X, const BLASINT incX,
                           double *Y, const BLASINT incY);
void BLASNAME(cblas_dcopy)(const BLASINT N, const double *X, const BLASINT incX,
                           double *Y, const BLASINT incY);
void BLASNAME(cblas_daxpy)(const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, double *Y, const BLASINT incY);

void BLASNAME(cblas_cswap)(const BLASINT N, void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_ccopy)(const BLASINT N, const void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_caxpy)(const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, void *Y, const BLASINT incY);

void BLASNAME(cblas_zswap)(const BLASINT N, void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_zcopy)(const BLASINT N, const void *X, const BLASINT incX,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_zaxpy)(const BLASINT N, const void *alpha, const void *X,
                           const BLASINT incX, void *Y, const BLASINT incY);


/*
 * Routines with S and D prefix only
 */
void BLASNAME(cblas_srotg)(float *a, float *b, float *c, float *s);
void BLASNAME(cblas_srotmg)(float *d1, float *d2, float *b1, const float b2, float *P);
void BLASNAME(cblas_srot)(const BLASINT N, float *X, const BLASINT incX,
                          float *Y, const BLASINT incY, const float c, const float s);
void BLASNAME(cblas_srotm)(const BLASINT N, float *X, const BLASINT incX,
                           float *Y, const BLASINT incY, const float *P);

void BLASNAME(cblas_drotg)(double *a, double *b, double *c, double *s);
void BLASNAME(cblas_drotmg)(double *d1, double *d2, double *b1, const double b2, double *P);
void BLASNAME(cblas_drot)(const BLASINT N, double *X, const BLASINT incX,
                          double *Y, const BLASINT incY, const double c, const double  s);
void BLASNAME(cblas_drotm)(const BLASINT N, double *X, const BLASINT incX,
                           double *Y, const BLASINT incY, const double *P);


/*
 * Routines with S D C Z CS and ZD prefixes
 */
void BLASNAME(cblas_sscal)(const BLASINT N, const float alpha, float *X, const BLASINT incX);
void BLASNAME(cblas_dscal)(const BLASINT N, const double alpha, double *X, const BLASINT incX);
void BLASNAME(cblas_cscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_zscal)(const BLASINT N, const void *alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_csscal)(const BLASINT N, const float alpha, void *X, const BLASINT incX);
void BLASNAME(cblas_zdscal)(const BLASINT N, const double alpha, void *X, const BLASINT incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLASNAME(cblas_sgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *X, const BLASINT incX, const float beta,
                           float *Y, const BLASINT incY);
void BLASNAME(cblas_sgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const float alpha,
                           const float *A, const BLASINT lda, const float *X,
                           const BLASINT incX, const float beta, float *Y, const BLASINT incY);
void BLASNAME(cblas_strmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
void BLASNAME(cblas_stbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
void BLASNAME(cblas_stpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *Ap, float *X, const BLASINT incX);
void BLASNAME(cblas_strsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *A, const BLASINT lda, float *X,
                           const BLASINT incX);
void BLASNAME(cblas_stbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const float *A, const BLASINT lda,
                           float *X, const BLASINT incX);
void BLASNAME(cblas_stpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const float *Ap, float *X, const BLASINT incX);

void BLASNAME(cblas_dgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *X, const BLASINT incX, const double beta,
                           double *Y, const BLASINT incY);
void BLASNAME(cblas_dgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const double alpha,
                           const double *A, const BLASINT lda, const double *X,
                           const BLASINT incX, const double beta, double *Y, const BLASINT incY);
void BLASNAME(cblas_dtrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
void BLASNAME(cblas_dtbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
void BLASNAME(cblas_dtpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);
void BLASNAME(cblas_dtrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *A, const BLASINT lda, double *X,
                           const BLASINT incX);
void BLASNAME(cblas_dtbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const double *A, const BLASINT lda,
                           double *X, const BLASINT incX);
void BLASNAME(cblas_dtpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const double *Ap, double *X, const BLASINT incX);

void BLASNAME(cblas_cgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_cgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_ctrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ctbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ctpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);
void BLASNAME(cblas_ctrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);
void BLASNAME(cblas_ctbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ctpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);

void BLASNAME(cblas_zgemv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *X, const BLASINT incX, const void *beta,
                           void *Y, const BLASINT incY);
void BLASNAME(cblas_zgbmv)(const enum CBLAS_ORDER order,
                           const enum CBLAS_TRANSPOSE TransA, const BLASINT M, const BLASINT N,
                           const BLASINT KL, const BLASINT KU, const void *alpha,
                           const void *A, const BLASINT lda, const void *X,
                           const BLASINT incX, const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_ztrmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ztbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ztpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);
void BLASNAME(cblas_ztrsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *A, const BLASINT lda, void *X,
                           const BLASINT incX);
void BLASNAME(cblas_ztbsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const BLASINT K, const void *A, const BLASINT lda,
                           void *X, const BLASINT incX);
void BLASNAME(cblas_ztpsv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                           const BLASINT N, const void *Ap, void *X, const BLASINT incX);


/*
 * Routines with S and D prefixes only
 */
void BLASNAME(cblas_ssymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
void BLASNAME(cblas_ssbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
void BLASNAME(cblas_sspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *Ap,
                           const float *X, const BLASINT incX,
                           const float beta, float *Y, const BLASINT incY);
void BLASNAME(cblas_sger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const float alpha, const float *X, const BLASINT incX,
                          const float *Y, const BLASINT incY, float *A, const BLASINT lda);
void BLASNAME(cblas_ssyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *A, const BLASINT lda);
void BLASNAME(cblas_sspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const float *X,
                          const BLASINT incX, float *Ap);
void BLASNAME(cblas_ssyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A,
                           const BLASINT lda);
void BLASNAME(cblas_sspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const float alpha, const float *X,
                           const BLASINT incX, const float *Y, const BLASINT incY, float *A);

void BLASNAME(cblas_dsymv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
void BLASNAME(cblas_dsbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
void BLASNAME(cblas_dspmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *Ap,
                           const double *X, const BLASINT incX,
                           const double beta, double *Y, const BLASINT incY);
void BLASNAME(cblas_dger)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                          const double alpha, const double *X, const BLASINT incX,
                          const double *Y, const BLASINT incY, double *A, const BLASINT lda);
void BLASNAME(cblas_dsyr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *A, const BLASINT lda);
void BLASNAME(cblas_dspr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const double *X,
                          const BLASINT incX, double *Ap);
void BLASNAME(cblas_dsyr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A,
                           const BLASINT lda);
void BLASNAME(cblas_dspr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const double alpha, const double *X,
                           const BLASINT incX, const double *Y, const BLASINT incY, double *A);


/*
 * Routines with C and Z prefixes only
 */
void BLASNAME(cblas_chemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_chbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_chpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_cgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_cgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_cher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);
void BLASNAME(cblas_chpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const float alpha, const void *X,
                          const BLASINT incX, void *A);
void BLASNAME(cblas_cher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_chpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *Ap);

void BLASNAME(cblas_zhemv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_zhbmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_zhpmv)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                           const BLASINT N, const void *alpha, const void *Ap,
                           const void *X, const BLASINT incX,
                           const void *beta, void *Y, const BLASINT incY);
void BLASNAME(cblas_zgeru)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_zgerc)(const enum CBLAS_ORDER order, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_zher)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X, const BLASINT incX,
                          void *A, const BLASINT lda);
void BLASNAME(cblas_zhpr)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                          const BLASINT N, const double alpha, const void *X,
                          const BLASINT incX, void *A);
void BLASNAME(cblas_zher2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *A, const BLASINT lda);
void BLASNAME(cblas_zhpr2)(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const BLASINT N,
                           const void *alpha, const void *X, const BLASINT incX,
                           const void *Y, const BLASINT incY, void *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLASNAME(cblas_sgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const float alpha, const float *A,
                           const BLASINT lda, const float *B, const BLASINT ldb,
                           const float beta, float *C, const BLASINT ldc);
void BLASNAME(cblas_ssymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           const float *B, const BLASINT ldb, const float beta,
                           float *C, const BLASINT ldc);
void BLASNAME(cblas_ssyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const float alpha, const float *A, const BLASINT lda,
                           const float beta, float *C, const BLASINT ldc);
void BLASNAME(cblas_ssyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const float alpha, const float *A, const BLASINT lda,
                            const float *B, const BLASINT ldb, const float beta,
                            float *C, const BLASINT ldc);
void BLASNAME(cblas_strmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);
void BLASNAME(cblas_strsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const float alpha, const float *A, const BLASINT lda,
                           float *B, const BLASINT ldb);

void BLASNAME(cblas_dgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const double alpha, const double *A,
                           const BLASINT lda, const double *B, const BLASINT ldb,
                           const double beta, double *C, const BLASINT ldc);
void BLASNAME(cblas_dsymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           const double *B, const BLASINT ldb, const double beta,
                           double *C, const BLASINT ldc);
void BLASNAME(cblas_dsyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const double *A, const BLASINT lda,
                           const double beta, double *C, const BLASINT ldc);
void BLASNAME(cblas_dsyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const double alpha, const double *A, const BLASINT lda,
                            const double *B, const BLASINT ldb, const double beta,
                            double *C, const BLASINT ldc);
void BLASNAME(cblas_dtrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);
void BLASNAME(cblas_dtrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const double alpha, const double *A, const BLASINT lda,
                           double *B, const BLASINT ldb);

void BLASNAME(cblas_cgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_csymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);
void BLASNAME(cblas_csyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_csyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const void *beta,
                            void *C, const BLASINT ldc);
void BLASNAME(cblas_ctrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);
void BLASNAME(cblas_ctrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);

void BLASNAME(cblas_zgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                           const BLASINT K, const void *alpha, const void *A,
                           const BLASINT lda, const void *B, const BLASINT ldb,
                           const void *beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_zsymm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);
void BLASNAME(cblas_zsyrk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_zsyr2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const void *beta,
                            void *C, const BLASINT ldc);
void BLASNAME(cblas_ztrmm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);
void BLASNAME(cblas_ztrsm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                           const enum CBLAS_DIAG Diag, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           void *B, const BLASINT ldb);


/*
 * Routines with prefixes C and Z only
 */
void BLASNAME(cblas_chemm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);
void BLASNAME(cblas_cherk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const float alpha, const void *A, const BLASINT lda,
                           const float beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_cher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const float beta,
                            void *C, const BLASINT ldc);

void BLASNAME(cblas_zhemm)(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                           const enum CBLAS_UPLO Uplo, const BLASINT M, const BLASINT N,
                           const void *alpha, const void *A, const BLASINT lda,
                           const void *B, const BLASINT ldb, const void *beta,
                           void *C, const BLASINT ldc);
void BLASNAME(cblas_zherk)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                           const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                           const double alpha, const void *A, const BLASINT lda,
                           const double beta, void *C, const BLASINT ldc);
void BLASNAME(cblas_zher2k)(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                            const enum CBLAS_TRANSPOSE Trans, const BLASINT N, const BLASINT K,
                            const void *alpha, const void *A, const BLASINT lda,
                            const void *B, const BLASINT ldb, const double beta,
                            void *C, const BLASINT ldc);

void BLASNAME(cblas_xerbla)(BLASINT p, const char *rout, const char *form, ...);

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CBLAS_BASE_H_ */
