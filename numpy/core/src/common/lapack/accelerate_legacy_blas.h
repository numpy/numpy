// Guard against using header directly
#if !defined(__TEMPLATE_FUNC) || !defined(__TEMPLATE_ALIAS)
# error "Please use top-level lapack/accelerate_legacy.h header"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =====================================
 * Prototypes for level 1 BLAS functions
 * =====================================
 */

int __TEMPLATE_FUNC(isamax)(int *n, float *sx, int *incx)
__TEMPLATE_ALIAS(isamax_);

int __TEMPLATE_FUNC(idamax)(int *n, double *dx, int *incx)
__TEMPLATE_ALIAS(idamax_);

int __TEMPLATE_FUNC(icamax)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(icamax_);

int __TEMPLATE_FUNC(izamax)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(izamax_);

double __TEMPLATE_FUNC(sasum)(int *n, float *sx, int *incx)
__TEMPLATE_ALIAS(sasum_);

double __TEMPLATE_FUNC(dasum)(int *n, double *dx, int *incx)
__TEMPLATE_ALIAS(dasum_);

double __TEMPLATE_FUNC(scasum)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(scasum_);

double __TEMPLATE_FUNC(dzasum)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(dzasum_);

int __TEMPLATE_FUNC(saxpy)(int *n, float *da,
           float *sx, int *incx,
           float *sy, int *incy)
__TEMPLATE_ALIAS(saxpy_);

int __TEMPLATE_FUNC(daxpy)(int *n, double *da,
           double *dx, int *incx,
           double *dy, int *incy)
__TEMPLATE_ALIAS(daxpy_);

int __TEMPLATE_FUNC(caxpy)(int *n, void *ca,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(caxpy_);

int __TEMPLATE_FUNC(zaxpy)(int *n, void *ca,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(zaxpy_);

int __TEMPLATE_FUNC(scopy)(int *n,
           float *sx, int *incx,
           float *sy, int *incy)
__TEMPLATE_ALIAS(scopy_);

int __TEMPLATE_FUNC(dcopy)(int *n,
           double *dx, int *incx,
           double *dy, int *incy)
__TEMPLATE_ALIAS(dcopy_);

int __TEMPLATE_FUNC(ccopy)(int *n,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(ccopy_);

int __TEMPLATE_FUNC(zcopy)(int *n,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(zcopy_);

double __TEMPLATE_FUNC(sdot)(int *n,
                 float *sx, int *incx,
                 float *sy, int *incy)
__TEMPLATE_ALIAS(sdot_);

double __TEMPLATE_FUNC(ddot)(int *n,
                 double *dx, int *incx,
                 double *dy, int *incy)
__TEMPLATE_ALIAS(ddot_);

double __TEMPLATE_FUNC(sdsdot)(int *n, float *sb,
                   float *sx, int *incx,
                   float *sy, int *incy)
__TEMPLATE_ALIAS(sdsdot_);

double __TEMPLATE_FUNC(dsdot)(int *n,
                  float *sx, int *incx,
                  float *sy, int *incy)
__TEMPLATE_ALIAS(dsdot_);

void __TEMPLATE_FUNC(cdotc)(void * ret_val, int *n,
            void *cx, int *incx,
            void *cy, int *incy)
__TEMPLATE_ALIAS(cdotc_);

void __TEMPLATE_FUNC(zdotc)(void * ret_val, int *n,
            void *cx, int *incx,
            void *cy, int *incy)
__TEMPLATE_ALIAS(zdotc_);

void __TEMPLATE_FUNC(cdotu)(void * ret_val, int *n,
            void *cx, int *incx,
            void *cy, int *incy)
__TEMPLATE_ALIAS(cdotu_);

void __TEMPLATE_FUNC(zdotu)(void * ret_val, int *n,
            void *cx, int *incx,
            void *cy, int *incy)
__TEMPLATE_ALIAS(zdotu_);

double __TEMPLATE_FUNC(snrm2)(int *n, float *x, int *incx)
__TEMPLATE_ALIAS(snrm2_);

double __TEMPLATE_FUNC(dnrm2)(int *n, double *x, int *incx)
__TEMPLATE_ALIAS(dnrm2_);

double __TEMPLATE_FUNC(scnrm2)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(scnrm2_);

double __TEMPLATE_FUNC(dznrm2)(int *n, void *cx, int *incx)
__TEMPLATE_ALIAS(dznrm2_);

int __TEMPLATE_FUNC(srot)(int *n,
          float *sx, int *incx,
          float *sy, int *incy,
          float *c, float *s)
__TEMPLATE_ALIAS(srot_);

int __TEMPLATE_FUNC(drot)(int *n,
          double *dx, int *incx,
          double *dy, int *incy,
          double *c, double *s)
__TEMPLATE_ALIAS(drot_);

int __TEMPLATE_FUNC(csrot)(int *n,
           void *cx, int *incx,
           void *cy, int *incy,
           float *c, float *s)
__TEMPLATE_ALIAS(csrot_);

int __TEMPLATE_FUNC(zdrot)(int *n,
           void *cx, int *incx,
           void *cy, int *incy,
           double *c, double *s)
__TEMPLATE_ALIAS(zdrot_);

int __TEMPLATE_FUNC(srotg)(float *sa, float *sb,
           float *c, float *s)
__TEMPLATE_ALIAS(srotg_);

int __TEMPLATE_FUNC(drotg)(double *da, double *db,
           double *c, double *s)
__TEMPLATE_ALIAS(drotg_);

int __TEMPLATE_FUNC(crotg)(void *ca, void *cb, float *c, void *cs)
__TEMPLATE_ALIAS(crotg_);

int __TEMPLATE_FUNC(zrotg)(void *ca, void *cb,
           double *c, void *cs)
__TEMPLATE_ALIAS(zrotg_);

int __TEMPLATE_FUNC(srotm)(int *n,
           float *sx, int *incx,
           float *sy, int *incy,
           float *param)
__TEMPLATE_ALIAS(srotm_);

int __TEMPLATE_FUNC(drotm)(int *n,
           double *dx, int *incx,
           double *dy, int *incy,
           double *dparam)
__TEMPLATE_ALIAS(drotm_);

int __TEMPLATE_FUNC(srotmg)(float *sd1, float *sd2,
            float *sx1, float *sy1,
            float *param)
__TEMPLATE_ALIAS(srotmg_);

int __TEMPLATE_FUNC(drotmg)(double *dd1, double *dd2,
            double *dx1, double *dy1,
            double *dparam)
__TEMPLATE_ALIAS(drotmg_);

int __TEMPLATE_FUNC(sscal)(int *n, float *sa,
           float *sx, int *incx)
__TEMPLATE_ALIAS(sscal_);

int __TEMPLATE_FUNC(dscal)(int *n, double *da,
           double *dx, int *incx)
__TEMPLATE_ALIAS(dscal_);

int __TEMPLATE_FUNC(cscal)(int *n, void *ca,
           void *cx, int *incx)
__TEMPLATE_ALIAS(cscal_);

int __TEMPLATE_FUNC(zscal)(int *n, void *ca,
           void *cx, int *incx)
__TEMPLATE_ALIAS(zscal_);

int __TEMPLATE_FUNC(csscal)(int *n, float *sa,
            void *cx, int *incx)
__TEMPLATE_ALIAS(csscal_);

int __TEMPLATE_FUNC(zdscal)(int *n, double *sa,
            void *cx, int *incx)
__TEMPLATE_ALIAS(zdscal_);

int __TEMPLATE_FUNC(sswap)(int *n,
           float *sx, int *incx,
           float *sy, int *incy)
__TEMPLATE_ALIAS(sswap_);

int __TEMPLATE_FUNC(dswap)(int *n,
           double *dx, int *incx,
           double *dy, int *incy)
__TEMPLATE_ALIAS(dswap_);

int __TEMPLATE_FUNC(cswap)(int *n,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(cswap_);

int __TEMPLATE_FUNC(zswap)(int *n,
           void *cx, int *incx,
           void *cy, int *incy)
__TEMPLATE_ALIAS(zswap_);

/*
 * =====================================
 * Prototypes for level 2 BLAS functions
 * =====================================
 */

int __TEMPLATE_FUNC(sgemv)(char *trans, int *m, int *n, float *alpha,
           float *a, int *lda, float *x, int *incx, float *beta, float *y,
           int *incy)
__TEMPLATE_ALIAS(sgemv_);

int __TEMPLATE_FUNC(dgemv)(char *trans, int *m, int *n, double *alpha,
           double *a, int *lda, double *x, int *incx, double *beta, double *y,
           int *incy)
__TEMPLATE_ALIAS(dgemv_);

int __TEMPLATE_FUNC(cgemv)(char *trans, int *m, int *n, void *alpha, void *a,
           int *lda, void *x, int *incx, void *beta, void *y,
           int *incy)
__TEMPLATE_ALIAS(cgemv_);

int __TEMPLATE_FUNC(zgemv)(char *trans, int *m, int *n, void *alpha, void *a,
           int *lda, void *x, int *incx, void *beta, void *y,
           int *incy)
__TEMPLATE_ALIAS(zgemv_);

int __TEMPLATE_FUNC(sgbmv)(char *trans, int *m, int *n, int *kl,
           int *ku, float *alpha, float *a, int *lda, float *x, int *
           incx, float *beta, float *y, int *incy)
__TEMPLATE_ALIAS(sgbmv_);

int __TEMPLATE_FUNC(dgbmv)(char *trans, int *m, int *n, int *kl,
           int *ku, double *alpha, double *a, int *lda, double *x, int *
           incx, double *beta, double *y, int *incy)
__TEMPLATE_ALIAS(dgbmv_);

int __TEMPLATE_FUNC(cgbmv)(char *trans, int *m, int *n, int *kl,
           int *ku, void *alpha, void *a, int *lda, void *x,
           int *incx, void *beta, void *y, int *incy)
__TEMPLATE_ALIAS(cgbmv_);

int __TEMPLATE_FUNC(zgbmv)(char *trans, int *m, int *n, int *kl,
           int *ku, void *alpha, void *a, int *lda, void *x,
           int *incx, void *beta, void *y, int *incy)
__TEMPLATE_ALIAS(zgbmv_);

int __TEMPLATE_FUNC(ssymv)(char *uplo, int *n, float *alpha, float *a,
           int *lda, float *x, int *incx, float *beta, float *y, int *
           incy)
__TEMPLATE_ALIAS(ssymv_);

int __TEMPLATE_FUNC(dsymv)(char *uplo, int *n, double *alpha, double *a,
           int *lda, double *x, int *incx, double *beta, double *y, int *
           incy)
__TEMPLATE_ALIAS(dsymv_);

int __TEMPLATE_FUNC(chemv)(char *uplo, int *n, void *alpha, void *
           a, int *lda, void *x, int *incx, void *beta, void *y,
           int *incy)
__TEMPLATE_ALIAS(chemv_);

int __TEMPLATE_FUNC(zhemv)(char *uplo, int *n, void *alpha, void *
           a, int *lda, void *x, int *incx, void *beta, void *y,
           int *incy)
__TEMPLATE_ALIAS(zhemv_);

int __TEMPLATE_FUNC(ssbmv)(char *uplo, int *n, int *k, float *alpha,
           float *a, int *lda, float *x, int *incx, float *beta, float *y,
           int *incy)
__TEMPLATE_ALIAS(ssbmv_);

int __TEMPLATE_FUNC(dsbmv)(char *uplo, int *n, int *k, double *alpha,
           double *a, int *lda, double *x, int *incx, double *beta, double *y,
           int *incy)
__TEMPLATE_ALIAS(dsbmv_);

int __TEMPLATE_FUNC(chbmv)(char *uplo, int *n, int *k, void *
           alpha, void *a, int *lda, void *x, int *incx, void *
           beta, void *y, int *incy)
__TEMPLATE_ALIAS(chbmv_);

int __TEMPLATE_FUNC(zhbmv)(char *uplo, int *n, int *k, void *
           alpha, void *a, int *lda, void *x, int *incx, void *
           beta, void *y, int *incy)
__TEMPLATE_ALIAS(zhbmv_);

int __TEMPLATE_FUNC(sspmv)(char *uplo, int *n, float *alpha, float *ap,
           float *x, int *incx, float *beta, float *y, int *incy)
__TEMPLATE_ALIAS(sspmv_);

int __TEMPLATE_FUNC(dspmv)(char *uplo, int *n, double *alpha, double *ap,
           double *x, int *incx, double *beta, double *y, int *incy)
__TEMPLATE_ALIAS(dspmv_);

int __TEMPLATE_FUNC(chpmv)(char *uplo, int *n, void *alpha, void *
           ap, void *x, int *incx, void *beta, void *y, int *
           incy)
__TEMPLATE_ALIAS(chpmv_);

int __TEMPLATE_FUNC(zhpmv)(char *uplo, int *n, void *alpha, void *
           ap, void *x, int *incx, void *beta, void *y, int *
           incy)
__TEMPLATE_ALIAS(zhpmv_);

int __TEMPLATE_FUNC(strmv)(char *uplo, char *trans, char *diag, int *n,
           float *a, int *lda, float *x, int *incx)
__TEMPLATE_ALIAS(strmv_);

int __TEMPLATE_FUNC(dtrmv)(char *uplo, char *trans, char *diag, int *n,
           double *a, int *lda, double *x, int *incx)
__TEMPLATE_ALIAS(dtrmv_);

int __TEMPLATE_FUNC(ctrmv)(char *uplo, char *trans, char *diag, int *n,
           void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ctrmv_);

int __TEMPLATE_FUNC(ztrmv)(char *uplo, char *trans, char *diag, int *n,
           void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ztrmv_);

int __TEMPLATE_FUNC(stbmv)(char *uplo, char *trans, char *diag, int *n,
           int *k, float *a, int *lda, float *x, int *incx)
__TEMPLATE_ALIAS(stbmv_);

int __TEMPLATE_FUNC(dtbmv)(char *uplo, char *trans, char *diag, int *n,
           int *k, double *a, int *lda, double *x, int *incx)
__TEMPLATE_ALIAS(dtbmv_);

int __TEMPLATE_FUNC(ctbmv)(char *uplo, char *trans, char *diag, int *n,
           int *k, void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ctbmv_);

int __TEMPLATE_FUNC(ztbmv)(char *uplo, char *trans, char *diag, int *n,
           int *k, void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ztbmv_);

int __TEMPLATE_FUNC(stpmv)(char *uplo, char *trans, char *diag, int *n,
           float *ap, float *x, int *incx)
__TEMPLATE_ALIAS(stpmv_);

int __TEMPLATE_FUNC(dtpmv)(char *uplo, char *trans, char *diag, int *n,
           double *ap, double *x, int *incx)
__TEMPLATE_ALIAS(dtpmv_);

int __TEMPLATE_FUNC(ctpmv)(char *uplo, char *trans, char *diag, int *n,
           void *ap, void *x, int *incx)
__TEMPLATE_ALIAS(ctpmv_);

int __TEMPLATE_FUNC(ztpmv)(char *uplo, char *trans, char *diag, int *n,
           void *ap, void *x, int *incx)
__TEMPLATE_ALIAS(ztpmv_);

int __TEMPLATE_FUNC(strsv)(char *uplo, char *trans, char *diag, int *n,
           float *a, int *lda, float *x, int *incx)
__TEMPLATE_ALIAS(strsv_);

int __TEMPLATE_FUNC(dtrsv)(char *uplo, char *trans, char *diag, int *n,
           double *a, int *lda, double *x, int *incx)
__TEMPLATE_ALIAS(dtrsv_);

int __TEMPLATE_FUNC(ctrsv)(char *uplo, char *trans, char *diag, int *n,
           void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ctrsv_);

int __TEMPLATE_FUNC(ztrsv)(char *uplo, char *trans, char *diag, int *n,
           void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ztrsv_);

int __TEMPLATE_FUNC(stbsv)(char *uplo, char *trans, char *diag, int *n,
           int *k, float *a, int *lda, float *x, int *incx)
__TEMPLATE_ALIAS(stbsv_);

int __TEMPLATE_FUNC(dtbsv)(char *uplo, char *trans, char *diag, int *n,
           int *k, double *a, int *lda, double *x, int *incx)
__TEMPLATE_ALIAS(dtbsv_);

int __TEMPLATE_FUNC(ctbsv)(char *uplo, char *trans, char *diag, int *n,
           int *k, void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ctbsv_);

int __TEMPLATE_FUNC(ztbsv)(char *uplo, char *trans, char *diag, int *n,
           int *k, void *a, int *lda, void *x, int *incx)
__TEMPLATE_ALIAS(ztbsv_);

int __TEMPLATE_FUNC(stpsv)(char *uplo, char *trans, char *diag, int *n,
           float *ap, float *x, int *incx)
__TEMPLATE_ALIAS(stpsv_);

int __TEMPLATE_FUNC(dtpsv)(char *uplo, char *trans, char *diag, int *n,
           double *ap, double *x, int *incx)
__TEMPLATE_ALIAS(dtpsv_);

int __TEMPLATE_FUNC(ctpsv)(char *uplo, char *trans, char *diag, int *n,
           void *ap, void *x, int *incx)
__TEMPLATE_ALIAS(ctpsv_);

int __TEMPLATE_FUNC(ztpsv)(char *uplo, char *trans, char *diag, int *n,
           void *ap, void *x, int *incx)
__TEMPLATE_ALIAS(ztpsv_);

int __TEMPLATE_FUNC(sger)(int *m, int *n, float *alpha, float *x,
          int *incx, float *y, int *incy, float *a, int *lda)
__TEMPLATE_ALIAS(sger_);

int __TEMPLATE_FUNC(dger)(int *m, int *n, double *alpha, double *x,
          int *incx, double *y, int *incy, double *a, int *lda)
__TEMPLATE_ALIAS(dger_);

int __TEMPLATE_FUNC(cgerc)(int *m, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(cgerc_);

int __TEMPLATE_FUNC(zgerc)(int *m, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(zgerc_);

int __TEMPLATE_FUNC(cgeru)(int *m, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(cgeru_);

int __TEMPLATE_FUNC(zgeru)(int *m, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(zgeru_);

int __TEMPLATE_FUNC(ssyr)(char *uplo, int *n, float *alpha, float *x,
          int *incx, float *a, int *lda)
__TEMPLATE_ALIAS(ssyr_);

int __TEMPLATE_FUNC(dsyr)(char *uplo, int *n, double *alpha, double *x,
          int *incx, double *a, int *lda)
__TEMPLATE_ALIAS(dsyr_);

int __TEMPLATE_FUNC(cher)(char *uplo, int *n, float *alpha, void *x,
          int *incx, void *a, int *lda)
__TEMPLATE_ALIAS(cher_);

int __TEMPLATE_FUNC(zher)(char *uplo, int *n, double *alpha, void *x,
          int *incx, void *a, int *lda)
__TEMPLATE_ALIAS(zher_);

int __TEMPLATE_FUNC(ssyr2)(char *uplo, int *n, float *alpha, float *x,
           int *incx, float *y, int *incy, float *a, int *lda)
__TEMPLATE_ALIAS(ssyr2_);

int __TEMPLATE_FUNC(dsyr2)(char *uplo, int *n, double *alpha, double *x,
           int *incx, double *y, int *incy, double *a, int *lda)
__TEMPLATE_ALIAS(dsyr2_);

int __TEMPLATE_FUNC(cher2)(char *uplo, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(cher2_);

int __TEMPLATE_FUNC(zher2)(char *uplo, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *a, int *lda)
__TEMPLATE_ALIAS(zher2_);

int __TEMPLATE_FUNC(sspr)(char *uplo, int *n, float *alpha, float *x, int *incx, float *ap)
__TEMPLATE_ALIAS(sspr_);

int __TEMPLATE_FUNC(dspr)(char *uplo, int *n, double *alpha, double *x, int *incx, double *ap)
__TEMPLATE_ALIAS(dspr_);

int __TEMPLATE_FUNC(chpr)(char *uplo, int *n, float *alpha, void *x,
          int *incx, void *ap)
__TEMPLATE_ALIAS(chpr_);

int __TEMPLATE_FUNC(zhpr)(char *uplo, int *n, double *alpha, void *x,
          int *incx, void *ap)
__TEMPLATE_ALIAS(zhpr_);

int __TEMPLATE_FUNC(sspr2)(char *uplo, int *n, float *alpha, float *x,
           int *incx, float *y, int *incy, float *ap)
__TEMPLATE_ALIAS(sspr2_);

int __TEMPLATE_FUNC(dspr2)(char *uplo, int *n, double *alpha, double *x,
           int *incx, double *y, int *incy, double *ap)
__TEMPLATE_ALIAS(dspr2_);

int __TEMPLATE_FUNC(chpr2)(char *uplo, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *ap)
__TEMPLATE_ALIAS(chpr2_);

int __TEMPLATE_FUNC(zhpr2)(char *uplo, int *n, void *alpha, void *
           x, int *incx, void *y, int *incy, void *ap)
__TEMPLATE_ALIAS(zhpr2_);

/*
 * =====================================
 * Prototypes for level 3 BLAS functions
 * =====================================
 */

int __TEMPLATE_FUNC(sgemm)(char *transa, char *transb, int *m, int *n, int *k,
           float *alpha, float *a, int *lda, float *b, int *ldb,
           float *beta, float *c__, int *ldc)
__TEMPLATE_ALIAS(sgemm_);

int __TEMPLATE_FUNC(dgemm)(char *transa, char *transb, int *m, int *n, int *k,
           double *alpha, double *a, int *lda, double *b,
           int *ldb, double *beta, double *c__, int *ldc)
__TEMPLATE_ALIAS(dgemm_);

int __TEMPLATE_FUNC(cgemm)(char *transa, char *transb, int *m, int *n,
           int *k, void *alpha, void *a, int *lda, void *b,
           int *ldb, void *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(cgemm_);

int __TEMPLATE_FUNC(zgemm)(char *transa, char *transb, int *m, int *
           n, int *k, void *alpha, void *a, int *lda,
           void *b, int *ldb, void *beta, void *
           c__, int *ldc)
__TEMPLATE_ALIAS(zgemm_);

int __TEMPLATE_FUNC(ssymm)(char *side, char *uplo, int *m, int *n, float *alpha,
           float *a, int *lda, float *b, int *ldb, float *beta,
           float *c__, int *ldc)
__TEMPLATE_ALIAS(ssymm_);

int __TEMPLATE_FUNC(dsymm)(char *side, char *uplo, int *m, int *n,
           double *alpha, double *a, int *lda, double *b,
           int *ldb, double *beta, double *c__, int *ldc)
__TEMPLATE_ALIAS(dsymm_);

int __TEMPLATE_FUNC(csymm)(char *side, char *uplo, int *m, int *n,
           void *alpha, void *a, int *lda, void *b, int *ldb,
           void *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(csymm_);

int __TEMPLATE_FUNC(zsymm)(char *side, char *uplo, int *m, int *n,
           void *alpha, void *a, int *lda, void *
           b, int *ldb, void *beta, void *c__, int *
           ldc)
__TEMPLATE_ALIAS(zsymm_);

int __TEMPLATE_FUNC(chemm)(char *side, char *uplo, int *m, int *n,
           void *alpha, void *a, int *lda, void *b, int *ldb,
           void *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(chemm_);

int __TEMPLATE_FUNC(zhemm)(char *side, char *uplo, int *m, int *n,
           void *alpha, void *a, int *lda, void *
           b, int *ldb, void *beta, void *c__, int *
           ldc)
__TEMPLATE_ALIAS(zhemm_);

int __TEMPLATE_FUNC(strmm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, float *alpha, float *a, int *lda,
           float *b, int *ldb)
__TEMPLATE_ALIAS(strmm_);

int __TEMPLATE_FUNC(dtrmm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, double *alpha, double *a, int *
           lda, double *b, int *ldb)
__TEMPLATE_ALIAS(dtrmm_);

int __TEMPLATE_FUNC(ctrmm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, void *alpha, void *a, int *lda,
           void *b, int *ldb)
__TEMPLATE_ALIAS(ctrmm_);

int __TEMPLATE_FUNC(ztrmm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, void *alpha, void *a,
           int *lda, void *b, int *ldb)
__TEMPLATE_ALIAS(ztrmm_);

int __TEMPLATE_FUNC(strsm)(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, float *alpha, float *a, int *lda, float *b, int *ldb)
__TEMPLATE_ALIAS(strsm_);

int __TEMPLATE_FUNC(dtrsm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, double *alpha, double *a, int *
           lda, double *b, int *ldb)
__TEMPLATE_ALIAS(dtrsm_);

int __TEMPLATE_FUNC(ctrsm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, void *alpha, void *a, int *lda,
           void *b, int *ldb)
__TEMPLATE_ALIAS(ctrsm_);

int __TEMPLATE_FUNC(ztrsm)(char *side, char *uplo, char *transa, char *diag,
           int *m, int *n, void *alpha, void *a,
           int *lda, void *b, int *ldb)
__TEMPLATE_ALIAS(ztrsm_);

int __TEMPLATE_FUNC(ssyrk)(char *uplo, char *trans, int *n, int *k, float *alpha,
           float *a, int *lda, float *beta, float *c__, int *ldc)
__TEMPLATE_ALIAS(ssyrk_);

int __TEMPLATE_FUNC(dsyrk)(char *uplo, char *trans, int *n, int *k,
           double *alpha, double *a, int *lda, double *beta,
           double *c__, int *ldc)
__TEMPLATE_ALIAS(dsyrk_);

int __TEMPLATE_FUNC(csyrk)(char *uplo, char *trans, int *n, int *k,
           void *alpha, void *a, int *lda, void *beta, void *c__,
           int *ldc)
__TEMPLATE_ALIAS(csyrk_);

int __TEMPLATE_FUNC(zsyrk)(char *uplo, char *trans, int *n, int *k,
           void *alpha, void *a, int *lda, void *
           beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(zsyrk_);

int __TEMPLATE_FUNC(cherk)(char *uplo, char *trans, int *n, int *k,
           float *alpha, void *a, int *lda, float *beta, void *c__,
           int *ldc)
__TEMPLATE_ALIAS(cherk_);

int __TEMPLATE_FUNC(zherk)(char *uplo, char *trans, int *n, int *k,
           double *alpha, void *a, int *lda, double *beta,
           void *c__, int *ldc)
__TEMPLATE_ALIAS(zherk_);

int __TEMPLATE_FUNC(ssyr2k)(char *uplo, char *trans, int *n, int *k, float *alpha,
            float *a, int *lda, float *b, int *ldb, float *beta,
            float *c__, int *ldc)
__TEMPLATE_ALIAS(ssyr2k_);

int __TEMPLATE_FUNC(dsyr2k)(char *uplo, char *trans, int *n, int *k,
            double *alpha, double *a, int *lda, double *b,
            int *ldb, double *beta, double *c__, int *ldc)
__TEMPLATE_ALIAS(dsyr2k_);

int __TEMPLATE_FUNC(csyr2k)(char *uplo, char *trans, int *n, int *k,
            void *alpha, void *a, int *lda, void *b, int *ldb,
            void *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(csyr2k_);

int __TEMPLATE_FUNC(zsyr2k)(char *uplo, char *trans, int *n, int *k,
            void *alpha, void *a, int *lda, void *
            b, int *ldb, void *beta, void *c__, int *
            ldc)
__TEMPLATE_ALIAS(zsyr2k_);

int __TEMPLATE_FUNC(cher2k)(char *uplo, char *trans, int *n, int *k,
            void *alpha, void *a, int *lda, void *b, int *ldb,
            float *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(cher2k_);

int __TEMPLATE_FUNC(zher2k)(char *uplo, char *trans, int *n, int *k,
            void *alpha, void *a, int *lda, void *
            b, int *ldb, double *beta, void *c__, int *ldc)
__TEMPLATE_ALIAS(zher2k_);


#ifdef __cplusplus
}
#endif
