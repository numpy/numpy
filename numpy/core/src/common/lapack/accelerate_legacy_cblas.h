#if !defined(__TEMPLATE_FUNC) || !defined(__TEMPLATE_ALIAS)
#   error "Please use top-level lapack/accelerate_legacy.h header"
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
// CBLAS_INDEX may already be defined.
// we'll save it and restore it at end
#pragma push_macro("CBLAS_INDEX")
#undef CBLAS_INDEX
#define CBLAS_INDEX int
  
int __TEMPLATE_FUNC(cblas_errprn)(int __ierr, int __info, char *__form, ...)
__TEMPLATE_ALIAS(cblas_errprn);
void __TEMPLATE_FUNC(cblas_xerbla)(int __p, char *__rout, char *__form, ...)
__TEMPLATE_ALIAS(cblas_xerbla);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
float  __TEMPLATE_FUNC(cblas_sdsdot)(const int __N, const float __alpha, const float *__X,
                    const int __incX, const float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_sdsdot);

double __TEMPLATE_FUNC(cblas_dsdot)(const int __N, const float *__X, const int __incX,
                   const float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_dsdot);
float  __TEMPLATE_FUNC(cblas_sdot)(const int __N, const float *__X, const int __incX,
                  const float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_sdot);
double __TEMPLATE_FUNC(cblas_ddot)(const int __N, const double *__X, const int __incX,
                  const double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_ddot);
/*
 * Functions having prefixes Z and C only
 */
void   __TEMPLATE_FUNC(cblas_cdotu_sub)(const int __N, const void *__X, const int __incX,
                       const void *__Y, const int __incY, void *__dotu)
__TEMPLATE_ALIAS(cblas_cdotu_sub);
void   __TEMPLATE_FUNC(cblas_cdotc_sub)(const int __N, const void *__X, const int __incX,
                       const void *__Y, const int __incY, void *__dotc)
__TEMPLATE_ALIAS(cblas_cdotc_sub);

void   __TEMPLATE_FUNC(cblas_zdotu_sub)(const int __N, const void *__X, const int __incX,
                       const void *__Y, const int __incY, void *__dotu)
__TEMPLATE_ALIAS(cblas_zdotu_sub);
void   __TEMPLATE_FUNC(cblas_zdotc_sub)(const int __N, const void *__X, const int __incX,
                       const void *__Y, const int __incY, void *__dotc)
__TEMPLATE_ALIAS(cblas_zdotc_sub);


/*
 * Functions having prefixes S D SC DZ
 */
float  __TEMPLATE_FUNC(cblas_snrm2)(const int __N, const float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_snrm2);
float  __TEMPLATE_FUNC(cblas_sasum)(const int __N, const float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_sasum);

double __TEMPLATE_FUNC(cblas_dnrm2)(const int __N, const double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dnrm2);
double __TEMPLATE_FUNC(cblas_dasum)(const int __N, const double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dasum);

float  __TEMPLATE_FUNC(cblas_scnrm2)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_scnrm2);
float  __TEMPLATE_FUNC(cblas_scasum)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_scasum);

double __TEMPLATE_FUNC(cblas_dznrm2)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dznrm2);
double __TEMPLATE_FUNC(cblas_dzasum)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dzasum);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX __TEMPLATE_FUNC(cblas_isamax)(const int __N, const float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_isamax);
CBLAS_INDEX __TEMPLATE_FUNC(cblas_idamax)(const int __N, const double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_idamax);
CBLAS_INDEX __TEMPLATE_FUNC(cblas_icamax)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_icamax);
CBLAS_INDEX __TEMPLATE_FUNC(cblas_izamax)(const int __N, const void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_izamax);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void __TEMPLATE_FUNC(cblas_sswap)(const int __N, float *__X, const int __incX, float *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_sswap);
void __TEMPLATE_FUNC(cblas_scopy)(const int __N, const float *__X, const int __incX, float *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_scopy);
void __TEMPLATE_FUNC(cblas_saxpy)(const int __N, const float __alpha, const float *__X,
                 const int __incX, float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_saxpy);
void __TEMPLATE_FUNC(catlas_saxpby)(const int __N, const float __alpha, const float *__X,
                   const int __incX, const float __beta, float *__Y, const int __incY)
__TEMPLATE_ALIAS(catlas_saxpby);
void __TEMPLATE_FUNC(catlas_sset)(const int __N, const float __alpha, float *__X,
                 const int __incX)
__TEMPLATE_ALIAS(catlas_sset);

void __TEMPLATE_FUNC(cblas_dswap)(const int __N, double *__X, const int __incX, double *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_dswap);
void __TEMPLATE_FUNC(cblas_dcopy)(const int __N, const double *__X, const int __incX,
                 double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_dcopy);
void __TEMPLATE_FUNC(cblas_daxpy)(const int __N, const double __alpha, const double *__X,
                 const int __incX, double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_daxpy);
void __TEMPLATE_FUNC(catlas_daxpby)(const int __N, const double __alpha, const double *__X,
                   const int __incX, const double __beta, double *__Y, const int __incY)
__TEMPLATE_ALIAS(catlas_daxpby);
void __TEMPLATE_FUNC(catlas_dset)(const int __N, const double __alpha, double *__X,
                 const int __incX)
__TEMPLATE_ALIAS(catlas_dset);

void __TEMPLATE_FUNC(cblas_cswap)(const int __N, void *__X, const int __incX, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_cswap);
void __TEMPLATE_FUNC(cblas_ccopy)(const int __N, const void *__X, const int __incX, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_ccopy);
void __TEMPLATE_FUNC(cblas_caxpy)(const int __N, const void *__alpha, const void *__X,
                 const int __incX, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_caxpy);
void __TEMPLATE_FUNC(catlas_caxpby)(const int __N, const void *__alpha, const void *__X,
                   const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(catlas_caxpby);
void __TEMPLATE_FUNC(catlas_cset)(const int __N, const void *__alpha, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(catlas_cset);

void __TEMPLATE_FUNC(cblas_zswap)(const int __N, void *__X, const int __incX, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_zswap);
void __TEMPLATE_FUNC(cblas_zcopy)(const int __N, const void *__X, const int __incX, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_zcopy);
void __TEMPLATE_FUNC(cblas_zaxpy)(const int __N, const void *__alpha, const void *__X,
                 const int __incX, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_zaxpy);
void __TEMPLATE_FUNC(catlas_zaxpby)(const int __N, const void *__alpha, const void *__X,
                   const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(catlas_zaxpby);
void __TEMPLATE_FUNC(catlas_zset)(const int __N, const void *__alpha, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(catlas_zset);


/*
 * Routines with S and D prefix only
 */
void __TEMPLATE_FUNC(cblas_srotg)(float *__a, float *__b, float *__c, float *__s)
__TEMPLATE_ALIAS(cblas_srotg);
void __TEMPLATE_FUNC(cblas_srotmg)(float *__d1, float *__d2, float *__b1, const float __b2,
                  float *__P)
__TEMPLATE_ALIAS(cblas_srotmg);
void __TEMPLATE_FUNC(cblas_srot)(const int __N, float *__X, const int __incX, float *__Y,
                const int __incY, const float __c, const float __s)
__TEMPLATE_ALIAS(cblas_srot);
void __TEMPLATE_FUNC(cblas_srotm)(const int __N, float *__X, const int __incX, float *__Y,
                 const int __incY, const float *__P)
__TEMPLATE_ALIAS(cblas_srotm);

void __TEMPLATE_FUNC(cblas_drotg)(double *__a, double *__b, double *__c, double *__s)
__TEMPLATE_ALIAS(cblas_drotg);
void __TEMPLATE_FUNC(cblas_drotmg)(double *__d1, double *__d2, double *__b1, const double __b2,
                  double *__P)
__TEMPLATE_ALIAS(cblas_drotmg);
void __TEMPLATE_FUNC(cblas_drot)(const int __N, double *__X, const int __incX, double *__Y,
                const int __incY, const double __c, const double __s)
__TEMPLATE_ALIAS(cblas_drot);
void __TEMPLATE_FUNC(cblas_drotm)(const int __N, double *__X, const int __incX, double *__Y,
                 const int __incY, const double *__P)
__TEMPLATE_ALIAS(cblas_drotm);


/*
 * Routines with S D C Z CS and ZD prefixes
 */
void __TEMPLATE_FUNC(cblas_sscal)(const int __N, const float __alpha, float *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_sscal);
void __TEMPLATE_FUNC(cblas_dscal)(const int __N, const double __alpha, double *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_dscal);
void __TEMPLATE_FUNC(cblas_cscal)(const int __N, const void *__alpha, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_cscal);
void __TEMPLATE_FUNC(cblas_zscal)(const int __N, const void *__alpha, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_zscal);
void __TEMPLATE_FUNC(cblas_csscal)(const int __N, const float __alpha, void *__X,
                  const int __incX)
__TEMPLATE_ALIAS(cblas_csscal);
void __TEMPLATE_FUNC(cblas_zdscal)(const int __N, const double __alpha, void *__X,
                  const int __incX)
__TEMPLATE_ALIAS(cblas_zdscal);

/*
 * Extra reference routines provided by ATLAS, but not mandated by the standard
 */
void __TEMPLATE_FUNC(cblas_crotg)(void *__a, void *__b, void *__c, void *__s)
__TEMPLATE_ALIAS(cblas_crotg);
void __TEMPLATE_FUNC(cblas_zrotg)(void *__a, void *__b, void *__c, void *__s)
__TEMPLATE_ALIAS(cblas_zrotg);
void __TEMPLATE_FUNC(cblas_csrot)(const int __N, void *__X, const int __incX, void *__Y,
                 const int __incY, const float __c, const float __s)
__TEMPLATE_ALIAS(cblas_csrot);
void __TEMPLATE_FUNC(cblas_zdrot)(const int __N, void *__X, const int __incX, void *__Y,
                 const int __incY, const double __c, const double __s)
__TEMPLATE_ALIAS(cblas_zdrot);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void __TEMPLATE_FUNC(cblas_sgemv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const float __alpha, const float *__A, const int __lda,
                 const float *__X, const int __incX, const float __beta, float *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_sgemv);
void __TEMPLATE_FUNC(cblas_sgbmv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const int __KL, const int __KU, const float __alpha, const float *__A,
                 const int __lda, const float *__X, const int __incX,
                 const float __beta, float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_sgbmv);
void __TEMPLATE_FUNC(cblas_strmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const float *__A, const int __lda, float *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_strmv);
void __TEMPLATE_FUNC(cblas_stbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const float *__A, const int __lda,
                 float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_stbmv);
void __TEMPLATE_FUNC(cblas_stpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const float *__Ap, float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_stpmv);
void __TEMPLATE_FUNC(cblas_strsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const float *__A, const int __lda, float *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_strsv);
void __TEMPLATE_FUNC(cblas_stbsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const float *__A, const int __lda,
                 float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_stbsv);
void __TEMPLATE_FUNC(cblas_stpsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const float *__Ap, float *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_stpsv);

void __TEMPLATE_FUNC(cblas_dgemv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const double __alpha, const double *__A, const int __lda,
                 const double *__X, const int __incX, const double __beta, double *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_dgemv);
void __TEMPLATE_FUNC(cblas_dgbmv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const int __KL, const int __KU, const double __alpha,
                 const double *__A, const int __lda, const double *__X,
                 const int __incX, const double __beta, double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_dgbmv);
void __TEMPLATE_FUNC(cblas_dtrmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const double *__A, const int __lda, double *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_dtrmv);
void __TEMPLATE_FUNC(cblas_dtbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const double *__A, const int __lda,
                 double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dtbmv);
void __TEMPLATE_FUNC(cblas_dtpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const double *__Ap, double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dtpmv);
void __TEMPLATE_FUNC(cblas_dtrsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const double *__A, const int __lda, double *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_dtrsv);
void __TEMPLATE_FUNC(cblas_dtbsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const double *__A, const int __lda,
                 double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dtbsv);
void __TEMPLATE_FUNC(cblas_dtpsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const double *__Ap, double *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_dtpsv);

void __TEMPLATE_FUNC(cblas_cgemv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__X,
                 const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_cgemv);
void __TEMPLATE_FUNC(cblas_cgbmv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const int __KL, const int __KU, const void *__alpha, const void *__A,
                 const int __lda, const void *__X, const int __incX, const void *__beta,
                 void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_cgbmv);
void __TEMPLATE_FUNC(cblas_ctrmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__A, const int __lda, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_ctrmv);
void __TEMPLATE_FUNC(cblas_ctbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const void *__A, const int __lda,
                 void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ctbmv);
void __TEMPLATE_FUNC(cblas_ctpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__Ap, void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ctpmv);
void __TEMPLATE_FUNC(cblas_ctrsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__A, const int __lda, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_ctrsv);
void __TEMPLATE_FUNC(cblas_ctbsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const void *__A, const int __lda,
                 void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ctbsv);
void __TEMPLATE_FUNC(cblas_ctpsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__Ap, void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ctpsv);

void __TEMPLATE_FUNC(cblas_zgemv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__X,
                 const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_zgemv);
void __TEMPLATE_FUNC(cblas_zgbmv)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA, const int __M, const int __N,
                 const int __KL, const int __KU, const void *__alpha, const void *__A,
                 const int __lda, const void *__X, const int __incX, const void *__beta,
                 void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_zgbmv);
void __TEMPLATE_FUNC(cblas_ztrmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__A, const int __lda, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_ztrmv);
void __TEMPLATE_FUNC(cblas_ztbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const void *__A, const int __lda,
                 void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ztbmv);
void __TEMPLATE_FUNC(cblas_ztpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__Ap, void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ztpmv);
void __TEMPLATE_FUNC(cblas_ztrsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__A, const int __lda, void *__X,
                 const int __incX)
__TEMPLATE_ALIAS(cblas_ztrsv);
void __TEMPLATE_FUNC(cblas_ztbsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const int __K, const void *__A, const int __lda,
                 void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ztbsv);
void __TEMPLATE_FUNC(cblas_ztpsv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag,
                 const int __N, const void *__Ap, void *__X, const int __incX)
__TEMPLATE_ALIAS(cblas_ztpsv);


/*
 * Routines with S and D prefixes only
 */
void __TEMPLATE_FUNC(cblas_ssymv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const float __alpha, const float *__A, const int __lda,
                 const float *__X, const int __incX, const float __beta, float *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_ssymv);
void __TEMPLATE_FUNC(cblas_ssbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const int __K, const float __alpha, const float *__A,
                 const int __lda, const float *__X, const int __incX,
                 const float __beta, float *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_ssbmv);
void __TEMPLATE_FUNC(cblas_sspmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const float __alpha, const float *__Ap,
                 const float *__X, const int __incX, const float __beta, float *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_sspmv);
void __TEMPLATE_FUNC(cblas_sger)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                const float __alpha, const float *__X, const int __incX,
                const float *__Y, const int __incY, float *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_sger);
void __TEMPLATE_FUNC(cblas_ssyr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const float __alpha, const float *__X, const int __incX,
                float *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_ssyr);
void __TEMPLATE_FUNC(cblas_sspr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const float __alpha, const float *__X, const int __incX,
                float *__Ap)
__TEMPLATE_ALIAS(cblas_sspr);
void __TEMPLATE_FUNC(cblas_ssyr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const float __alpha, const float *__X, const int __incX,
                 const float *__Y, const int __incY, float *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_ssyr2);
void __TEMPLATE_FUNC(cblas_sspr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const float __alpha, const float *__X, const int __incX,
                 const float *__Y, const int __incY, float *__A)
__TEMPLATE_ALIAS(cblas_sspr2);

void __TEMPLATE_FUNC(cblas_dsymv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const double __alpha, const double *__A,
                 const int __lda, const double *__X, const int __incX,
                 const double __beta, double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_dsymv);
void __TEMPLATE_FUNC(cblas_dsbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const int __K, const double __alpha, const double *__A,
                 const int __lda, const double *__X, const int __incX,
                 const double __beta, double *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_dsbmv);
void __TEMPLATE_FUNC(cblas_dspmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const double __alpha, const double *__Ap,
                 const double *__X, const int __incX, const double __beta, double *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_dspmv);
void __TEMPLATE_FUNC(cblas_dger)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                const double __alpha, const double *__X, const int __incX,
                const double *__Y, const int __incY, double *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_dger);
void __TEMPLATE_FUNC(cblas_dsyr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const double __alpha, const double *__X,
                const int __incX, double *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_dsyr);
void __TEMPLATE_FUNC(cblas_dspr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const double __alpha, const double *__X,
                const int __incX, double *__Ap)
__TEMPLATE_ALIAS(cblas_dspr);
void __TEMPLATE_FUNC(cblas_dsyr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const double __alpha, const double *__X,
                 const int __incX, const double *__Y, const int __incY, double *__A,
                 const int __lda)
__TEMPLATE_ALIAS(cblas_dsyr2);
void __TEMPLATE_FUNC(cblas_dspr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const double __alpha, const double *__X,
                 const int __incX, const double *__Y, const int __incY, double *__A)
__TEMPLATE_ALIAS(cblas_dspr2);


/*
 * Routines with C and Z prefixes only
 */
void __TEMPLATE_FUNC(cblas_chemv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__A, const int __lda,
                 const void *__X, const int __incX, const void *__beta, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_chemv);
void __TEMPLATE_FUNC(cblas_chbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const int __K, const void *__alpha, const void *__A,
                 const int __lda, const void *__X, const int __incX, const void *__beta,
                 void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_chbmv);
void __TEMPLATE_FUNC(cblas_chpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__Ap, const void *__X,
                 const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_chpmv);
void __TEMPLATE_FUNC(cblas_cgeru)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                 const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_cgeru);
void __TEMPLATE_FUNC(cblas_cgerc)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                 const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_cgerc);
void __TEMPLATE_FUNC(cblas_cher)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const float __alpha, const void *__X, const int __incX,
                void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_cher);
void __TEMPLATE_FUNC(cblas_chpr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const float __alpha, const void *__X, const int __incX,
                void *__A)
__TEMPLATE_ALIAS(cblas_chpr);
void __TEMPLATE_FUNC(cblas_cher2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_cher2);
void __TEMPLATE_FUNC(cblas_chpr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__Ap)
__TEMPLATE_ALIAS(cblas_chpr2);

void __TEMPLATE_FUNC(cblas_zhemv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__A, const int __lda,
                 const void *__X, const int __incX, const void *__beta, void *__Y,
                 const int __incY)
__TEMPLATE_ALIAS(cblas_zhemv);
void __TEMPLATE_FUNC(cblas_zhbmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const int __K, const void *__alpha, const void *__A,
                 const int __lda, const void *__X, const int __incX, const void *__beta,
                 void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_zhbmv);
void __TEMPLATE_FUNC(cblas_zhpmv)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__Ap, const void *__X,
                 const int __incX, const void *__beta, void *__Y, const int __incY)
__TEMPLATE_ALIAS(cblas_zhpmv);
void __TEMPLATE_FUNC(cblas_zgeru)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                 const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_zgeru);
void __TEMPLATE_FUNC(cblas_zgerc)(const enum CBLAS_ORDER __Order, const int __M, const int __N,
                 const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_zgerc);
void __TEMPLATE_FUNC(cblas_zher)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const double __alpha, const void *__X, const int __incX,
                void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_zher);
void __TEMPLATE_FUNC(cblas_zhpr)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                const int __N, const double __alpha, const void *__X, const int __incX,
                void *__A)
__TEMPLATE_ALIAS(cblas_zhpr);
void __TEMPLATE_FUNC(cblas_zher2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__A, const int __lda)
__TEMPLATE_ALIAS(cblas_zher2);
void __TEMPLATE_FUNC(cblas_zhpr2)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const int __N, const void *__alpha, const void *__X, const int __incX,
                 const void *__Y, const int __incY, void *__Ap)
__TEMPLATE_ALIAS(cblas_zhpr2);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void __TEMPLATE_FUNC(cblas_sgemm)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const float __alpha, const float *__A, const int __lda,
                 const float *__B, const int __ldb, const float __beta, float *__C,
                 const int __ldc)
__TEMPLATE_ALIAS(cblas_sgemm);
void __TEMPLATE_FUNC(cblas_ssymm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const float __alpha, const float *__A, const int __lda,
                 const float *__B, const int __ldb, const float __beta, float *__C,
                 const int __ldc)
__TEMPLATE_ALIAS(cblas_ssymm);
void __TEMPLATE_FUNC(cblas_ssyrk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const float __alpha, const float *__A, const int __lda,
                 const float __beta, float *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_ssyrk);
void __TEMPLATE_FUNC(cblas_ssyr2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const float __alpha, const float *__A, const int __lda,
                  const float *__B, const int __ldb, const float __beta, float *__C,
                  const int __ldc)
__TEMPLATE_ALIAS(cblas_ssyr2k);
void __TEMPLATE_FUNC(cblas_strmm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const float __alpha, const float *__A, const int __lda, float *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_strmm);
void __TEMPLATE_FUNC(cblas_strsm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const float __alpha, const float *__A, const int __lda, float *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_strsm);

void __TEMPLATE_FUNC(cblas_dgemm)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const double __alpha, const double *__A,
                 const int __lda, const double *__B, const int __ldb,
                 const double __beta, double *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_dgemm);
void __TEMPLATE_FUNC(cblas_dsymm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const double __alpha, const double *__A, const int __lda,
                 const double *__B, const int __ldb, const double __beta, double *__C,
                 const int __ldc)
__TEMPLATE_ALIAS(cblas_dsymm);
void __TEMPLATE_FUNC(cblas_dsyrk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const double __alpha, const double *__A, const int __lda,
                 const double __beta, double *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_dsyrk);
void __TEMPLATE_FUNC(cblas_dsyr2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const double __alpha, const double *__A, const int __lda,
                  const double *__B, const int __ldb, const double __beta, double *__C,
                  const int __ldc)
__TEMPLATE_ALIAS(cblas_dsyr2k);
void __TEMPLATE_FUNC(cblas_dtrmm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const double __alpha, const double *__A, const int __lda, double *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_dtrmm);
void __TEMPLATE_FUNC(cblas_dtrsm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const double __alpha, const double *__A, const int __lda, double *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_dtrsm);

void __TEMPLATE_FUNC(cblas_cgemm)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const void *__alpha, const void *__A, const int __lda,
                 const void *__B, const int __ldb, const void *__beta, void *__C,
                 const int __ldc)
__TEMPLATE_ALIAS(cblas_cgemm);
void __TEMPLATE_FUNC(cblas_csymm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__B,
                 const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_csymm);
void __TEMPLATE_FUNC(cblas_csyrk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const void *__alpha, const void *__A, const int __lda,
                 const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_csyrk);
void __TEMPLATE_FUNC(cblas_csyr2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const void *__alpha, const void *__A, const int __lda, const void *__B,
                  const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_csyr2k);
void __TEMPLATE_FUNC(cblas_ctrmm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, void *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_ctrmm);
void __TEMPLATE_FUNC(cblas_ctrsm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, void *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_ctrsm);

void __TEMPLATE_FUNC(cblas_zgemm)(const enum CBLAS_ORDER __Order,
                 const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N,
                 const int __K, const void *__alpha, const void *__A, const int __lda,
                 const void *__B, const int __ldb, const void *__beta, void *__C,
                 const int __ldc)
__TEMPLATE_ALIAS(cblas_zgemm);
void __TEMPLATE_FUNC(cblas_zsymm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__B,
                 const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zsymm);
void __TEMPLATE_FUNC(cblas_zsyrk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const void *__alpha, const void *__A, const int __lda,
                 const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zsyrk);
void __TEMPLATE_FUNC(cblas_zsyr2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const void *__alpha, const void *__A, const int __lda, const void *__B,
                  const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zsyr2k);
void __TEMPLATE_FUNC(cblas_ztrmm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, void *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_ztrmm);
void __TEMPLATE_FUNC(cblas_ztrsm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_DIAG __Diag, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, void *__B,
                 const int __ldb)
__TEMPLATE_ALIAS(cblas_ztrsm);


/*
 * Routines with prefixes C and Z only
 */
void __TEMPLATE_FUNC(cblas_chemm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__B,
                 const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_chemm);
void __TEMPLATE_FUNC(cblas_cherk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const float __alpha, const void *__A, const int __lda,
                 const float __beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_cherk);
void __TEMPLATE_FUNC(cblas_cher2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const void *__alpha, const void *__A, const int __lda, const void *__B,
                  const int __ldb, const float __beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_cher2k);
void __TEMPLATE_FUNC(cblas_zhemm)(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side,
                 const enum CBLAS_UPLO __Uplo, const int __M, const int __N,
                 const void *__alpha, const void *__A, const int __lda, const void *__B,
                 const int __ldb, const void *__beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zhemm);
void __TEMPLATE_FUNC(cblas_zherk)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                 const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                 const double __alpha, const void *__A, const int __lda,
                 const double __beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zherk);
void __TEMPLATE_FUNC(cblas_zher2k)(const enum CBLAS_ORDER __Order, const enum CBLAS_UPLO __Uplo,
                  const enum CBLAS_TRANSPOSE __Trans, const int __N, const int __K,
                  const void *__alpha, const void *__A, const int __lda, const void *__B,
                  const int __ldb, const double __beta, void *__C, const int __ldc)
__TEMPLATE_ALIAS(cblas_zher2k);

typedef void (*__TEMPLATE_FUNC(BLASParamErrorProc))(const char *funcName, const char *paramName,
                                   const int *paramPos,  const int *paramValue);

void __TEMPLATE_FUNC(SetBLASParamErrorProc)(__TEMPLATE_FUNC(BLASParamErrorProc) __ErrorProc)
__TEMPLATE_ALIAS(SetBLASParamErrorProc);

// restore CBLAS_INDEX
#pragma pop_macro("CBLAS_INDEX")
  
#ifdef __cplusplus
}
#endif
