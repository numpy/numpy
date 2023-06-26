// Guard against using header directly
#if !defined(__TEMPLATE_FUNC) || !defined(__TEMPLATE_ALIAS)
#   error "Please use top-level lapack/accelerate_legacy.h header"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef __CLPK_TYPES_DEFINED
#define __CLPK_TYPES_DEFINED

#if defined(__LP64__) /* In LP64 match sizes with the 32 bit ABI */
    typedef int                 __CLPK_integer;
    typedef int                 __CLPK_logical;
    typedef float               __CLPK_real;
    typedef double              __CLPK_doublereal;
    
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"
    typedef __CLPK_logical      (*__CLPK_L_fp)();
#pragma clang diagnostic pop
    
    typedef int                 __CLPK_ftnlen;
#else
    typedef long int            __CLPK_integer;
    typedef long int            __CLPK_logical;
    typedef float               __CLPK_real;
    typedef double              __CLPK_doublereal;
    
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"
    typedef __CLPK_logical      (*__CLPK_L_fp)();
#pragma clang diagnostic pop
    
    typedef long int            __CLPK_ftnlen;
#endif

typedef struct { __CLPK_real r, i; } __CLPK_complex;
typedef struct { __CLPK_doublereal r, i; } __CLPK_doublecomplex;

#endif //#ifndef __CLPK_TYPES_DEFINED

int __TEMPLATE_FUNC(cbdsqr)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__ncvt,
        __CLPK_integer *__nru, __CLPK_integer *__ncc, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__vt, __CLPK_integer *__ldvt,
        __CLPK_complex *__u, __CLPK_integer *__ldu, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cbdsqr);

int __TEMPLATE_FUNC(cgbbrd)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__ncc, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__q, __CLPK_integer *__ldq,
        __CLPK_complex *__pt, __CLPK_integer *__ldpt, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbbrd);

int __TEMPLATE_FUNC(cgbcon)(char *__norm, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbcon);

int __TEMPLATE_FUNC(cgbequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbequ);

int __TEMPLATE_FUNC(cgbequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbequb);

int __TEMPLATE_FUNC(cgbrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__afb, __CLPK_integer *__ldafb,
        __CLPK_integer *__ipiv, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbrfs);

int __TEMPLATE_FUNC(cgbsv)(__CLPK_integer *__n, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_integer *__nrhs, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbsv);

int __TEMPLATE_FUNC(cgbsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_complex *__afb,
        __CLPK_integer *__ldafb, __CLPK_integer *__ipiv, char *__equed,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbsvx);

int __TEMPLATE_FUNC(cgbtf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbtf2);

int __TEMPLATE_FUNC(cgbtrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbtrf);

int __TEMPLATE_FUNC(cgbtrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgbtrs);

int __TEMPLATE_FUNC(cgebak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__scale,
        __CLPK_integer *__m, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgebak);

int __TEMPLATE_FUNC(cgebal)(char *__job, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgebal);

int __TEMPLATE_FUNC(cgebd2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_complex *__tauq, __CLPK_complex *__taup, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgebd2);

int __TEMPLATE_FUNC(cgebrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_complex *__tauq, __CLPK_complex *__taup, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgebrd);

int __TEMPLATE_FUNC(cgecon)(char *__norm, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgecon);

int __TEMPLATE_FUNC(cgeequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeequ);

int __TEMPLATE_FUNC(cgeequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeequb);

int __TEMPLATE_FUNC(cgees)(char *__jobvs, char *__sort, __CLPK_L_fp __select,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_complex *__w, __CLPK_complex *__vs,
        __CLPK_integer *__ldvs, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgees);

int __TEMPLATE_FUNC(cgeesx)(char *__jobvs, char *__sort, __CLPK_L_fp __select, char *__sense,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_complex *__w, __CLPK_complex *__vs,
        __CLPK_integer *__ldvs, __CLPK_real *__rconde, __CLPK_real *__rcondv,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeesx);

int __TEMPLATE_FUNC(cgeev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__w,
        __CLPK_complex *__vl, __CLPK_integer *__ldvl, __CLPK_complex *__vr,
        __CLPK_integer *__ldvr, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeev);

int __TEMPLATE_FUNC(cgeevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__w, __CLPK_complex *__vl, __CLPK_integer *__ldvl,
        __CLPK_complex *__vr, __CLPK_integer *__ldvr, __CLPK_integer *__ilo,
        __CLPK_integer *__ihi, __CLPK_real *__scale, __CLPK_real *__abnrm,
        __CLPK_real *__rconde, __CLPK_real *__rcondv, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeevx);

int __TEMPLATE_FUNC(cgegs)(char *__jobvsl, char *__jobvsr, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__alpha, __CLPK_complex *__beta,
        __CLPK_complex *__vsl, __CLPK_integer *__ldvsl, __CLPK_complex *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgegs);

int __TEMPLATE_FUNC(cgegv)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__alpha, __CLPK_complex *__beta,
        __CLPK_complex *__vl, __CLPK_integer *__ldvl, __CLPK_complex *__vr,
        __CLPK_integer *__ldvr, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgegv);

int __TEMPLATE_FUNC(cgehd2)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgehd2);

int __TEMPLATE_FUNC(cgehrd)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgehrd);

int __TEMPLATE_FUNC(cgelq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelq2);

int __TEMPLATE_FUNC(cgelqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelqf);

int __TEMPLATE_FUNC(cgels)(char *__trans, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgels);

int __TEMPLATE_FUNC(cgelsd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_real *__s, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelsd);

int __TEMPLATE_FUNC(cgelss)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_real *__s, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelss);

int __TEMPLATE_FUNC(cgelsx)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelsx);

int __TEMPLATE_FUNC(cgelsy)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgelsy);

int __TEMPLATE_FUNC(cgeql2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeql2);

int __TEMPLATE_FUNC(cgeqlf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeqlf);

int __TEMPLATE_FUNC(cgeqp3)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeqp3);

int __TEMPLATE_FUNC(cgeqpf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeqpf);

int __TEMPLATE_FUNC(cgeqr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeqr2);

int __TEMPLATE_FUNC(cgeqrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgeqrf);

int __TEMPLATE_FUNC(cgerfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgerfs);

int __TEMPLATE_FUNC(cgerq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgerq2);

int __TEMPLATE_FUNC(cgerqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgerqf);

int __TEMPLATE_FUNC(cgesc2)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__rhs, __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_real *__scale)
__TEMPLATE_ALIAS(cgesc2);

int __TEMPLATE_FUNC(cgesdd)(char *__jobz, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__s,
        __CLPK_complex *__u, __CLPK_integer *__ldu, __CLPK_complex *__vt,
        __CLPK_integer *__ldvt, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgesdd);

int __TEMPLATE_FUNC(cgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgesv);

int __TEMPLATE_FUNC(cgesvd)(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_complex *__u, __CLPK_integer *__ldu,
        __CLPK_complex *__vt, __CLPK_integer *__ldvt, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgesvd);

int __TEMPLATE_FUNC(cgesvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        char *__equed, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgesvx);

int __TEMPLATE_FUNC(cgetc2)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgetc2);

int __TEMPLATE_FUNC(cgetf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgetf2);

int __TEMPLATE_FUNC(cgetrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgetrf);

int __TEMPLATE_FUNC(cgetri)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgetri);

int __TEMPLATE_FUNC(cgetrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgetrs);

int __TEMPLATE_FUNC(cggbak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__lscale,
        __CLPK_real *__rscale, __CLPK_integer *__m, __CLPK_complex *__v,
        __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggbak);

int __TEMPLATE_FUNC(cggbal)(char *__job, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__lscale,
        __CLPK_real *__rscale, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggbal);

int __TEMPLATE_FUNC(cgges)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_integer *__sdim,
        __CLPK_complex *__alpha, __CLPK_complex *__beta, __CLPK_complex *__vsl,
        __CLPK_integer *__ldvsl, __CLPK_complex *__vsr, __CLPK_integer *__ldvsr,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgges);

int __TEMPLATE_FUNC(cggesx)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        char *__sense, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__sdim, __CLPK_complex *__alpha, __CLPK_complex *__beta,
        __CLPK_complex *__vsl, __CLPK_integer *__ldvsl, __CLPK_complex *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_real *__rconde, __CLPK_real *__rcondv,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggesx);

int __TEMPLATE_FUNC(cggev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__alpha, __CLPK_complex *__beta,
        __CLPK_complex *__vl, __CLPK_integer *__ldvl, __CLPK_complex *__vr,
        __CLPK_integer *__ldvr, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggev);

int __TEMPLATE_FUNC(cggevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__alpha,
        __CLPK_complex *__beta, __CLPK_complex *__vl, __CLPK_integer *__ldvl,
        __CLPK_complex *__vr, __CLPK_integer *__ldvr, __CLPK_integer *__ilo,
        __CLPK_integer *__ihi, __CLPK_real *__lscale, __CLPK_real *__rscale,
        __CLPK_real *__abnrm, __CLPK_real *__bbnrm, __CLPK_real *__rconde,
        __CLPK_real *__rcondv, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork, __CLPK_integer *__iwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggevx);

int __TEMPLATE_FUNC(cggglm)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__d__, __CLPK_complex *__x,
        __CLPK_complex *__y, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggglm);

int __TEMPLATE_FUNC(cgghrd)(char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_complex *__z__,
        __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgghrd);

int __TEMPLATE_FUNC(cgglse)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__p,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__c__, __CLPK_complex *__d__,
        __CLPK_complex *__x, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgglse);

int __TEMPLATE_FUNC(cggqrf)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__taua,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__taub,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggqrf);

int __TEMPLATE_FUNC(cggrqf)(__CLPK_integer *__m, __CLPK_integer *__p, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__taua,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__taub,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggrqf);

int __TEMPLATE_FUNC(cggsvd)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__p, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__alpha,
        __CLPK_real *__beta, __CLPK_complex *__u, __CLPK_integer *__ldu,
        __CLPK_complex *__v, __CLPK_integer *__ldv, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggsvd);

int __TEMPLATE_FUNC(cggsvp)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_real *__tola, __CLPK_real *__tolb, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_complex *__u, __CLPK_integer *__ldu,
        __CLPK_complex *__v, __CLPK_integer *__ldv, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_integer *__iwork, __CLPK_real *__rwork,
        __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cggsvp);

int __TEMPLATE_FUNC(cgtcon)(char *__norm, __CLPK_integer *__n, __CLPK_complex *__dl,
        __CLPK_complex *__d__, __CLPK_complex *__du, __CLPK_complex *__du2,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgtcon);

int __TEMPLATE_FUNC(cgtrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__dl, __CLPK_complex *__d__, __CLPK_complex *__du,
        __CLPK_complex *__dlf, __CLPK_complex *__df, __CLPK_complex *__duf,
        __CLPK_complex *__du2, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgtrfs);

int __TEMPLATE_FUNC(cgtsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_complex *__dl,
        __CLPK_complex *__d__, __CLPK_complex *__du, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgtsv);

int __TEMPLATE_FUNC(cgtsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__dl, __CLPK_complex *__d__,
        __CLPK_complex *__du, __CLPK_complex *__dlf, __CLPK_complex *__df,
        __CLPK_complex *__duf, __CLPK_complex *__du2, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgtsvx);

int __TEMPLATE_FUNC(cgttrf)(__CLPK_integer *__n, __CLPK_complex *__dl, __CLPK_complex *__d__,
        __CLPK_complex *__du, __CLPK_complex *__du2, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgttrf);

int __TEMPLATE_FUNC(cgttrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__dl, __CLPK_complex *__d__, __CLPK_complex *__du,
        __CLPK_complex *__du2, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cgttrs);

int __TEMPLATE_FUNC(cgtts2)(__CLPK_integer *__itrans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__dl, __CLPK_complex *__d__,
        __CLPK_complex *__du, __CLPK_complex *__du2, __CLPK_integer *__ipiv,
        __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(cgtts2);

int __TEMPLATE_FUNC(chbev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbev);

int __TEMPLATE_FUNC(chbevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbevd);

int __TEMPLATE_FUNC(chbevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_complex *__work,
        __CLPK_real *__rwork, __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbevx);

int __TEMPLATE_FUNC(chbgst)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__bb, __CLPK_integer *__ldbb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbgst);

int __TEMPLATE_FUNC(chbgv)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbgv);

int __TEMPLATE_FUNC(chbgvd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbgvd);

int __TEMPLATE_FUNC(chbgvx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__bb, __CLPK_integer *__ldbb,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_complex *__work,
        __CLPK_real *__rwork, __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbgvx);

int __TEMPLATE_FUNC(chbtrd)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chbtrd);

int __TEMPLATE_FUNC(checon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__anorm,
        __CLPK_real *__rcond, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(checon);

int __TEMPLATE_FUNC(cheequb)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cheequb);

int __TEMPLATE_FUNC(cheev)(char *__jobz, char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__w, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cheev);

int __TEMPLATE_FUNC(cheevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__w,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cheevd);

int __TEMPLATE_FUNC(cheevr)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cheevr);

int __TEMPLATE_FUNC(cheevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cheevx);

int __TEMPLATE_FUNC(chegs2)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chegs2);

int __TEMPLATE_FUNC(chegst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chegst);

int __TEMPLATE_FUNC(chegv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__w,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chegv);

int __TEMPLATE_FUNC(chegvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__w,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chegvd);

int __TEMPLATE_FUNC(chegvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chegvx);

int __TEMPLATE_FUNC(cherfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cherfs);

int __TEMPLATE_FUNC(chesv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chesv);

int __TEMPLATE_FUNC(chesvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chesvx);

int __TEMPLATE_FUNC(chetd2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_complex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetd2);

int __TEMPLATE_FUNC(chetf2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetf2);

int __TEMPLATE_FUNC(chetrd)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_complex *__tau, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetrd);

int __TEMPLATE_FUNC(chetrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetrf);

int __TEMPLATE_FUNC(chetri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetri);

int __TEMPLATE_FUNC(chetrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chetrs);

int __TEMPLATE_FUNC(chfrk)(char *__transr, char *__uplo, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__alpha, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__beta,
        __CLPK_complex *__c__)
__TEMPLATE_ALIAS(chfrk);

int __TEMPLATE_FUNC(chgeqz)(char *__job, char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_complex *__h__,
        __CLPK_integer *__ldh, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_complex *__alpha, __CLPK_complex *__beta, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chgeqz);

void __TEMPLATE_FUNC(chla_transtype_)(char *__ret_val, __CLPK_ftnlen __ret_val_len,
        __CLPK_integer *__trans)
__TEMPLATE_ALIAS(chla_transtype_);

int __TEMPLATE_FUNC(chpcon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpcon);

int __TEMPLATE_FUNC(chpev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_real *__w, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpev);

int __TEMPLATE_FUNC(chpevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_real *__w, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork, __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpevd);

int __TEMPLATE_FUNC(chpevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_real *__vl, __CLPK_real *__vu,
        __CLPK_integer *__il, __CLPK_integer *__iu, __CLPK_real *__abstol,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpevx);

int __TEMPLATE_FUNC(chpgst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_complex *__bp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpgst);

int __TEMPLATE_FUNC(chpgv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__ap, __CLPK_complex *__bp,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpgv);

int __TEMPLATE_FUNC(chpgvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__ap, __CLPK_complex *__bp,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpgvd);

int __TEMPLATE_FUNC(chpgvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__ap, __CLPK_complex *__bp,
        __CLPK_real *__vl, __CLPK_real *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_real *__abstol, __CLPK_integer *__m,
        __CLPK_real *__w, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpgvx);

int __TEMPLATE_FUNC(chprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_complex *__afp, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chprfs);

int __TEMPLATE_FUNC(chpsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpsv);

int __TEMPLATE_FUNC(chpsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__ap, __CLPK_complex *__afp,
        __CLPK_integer *__ipiv, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chpsvx);

int __TEMPLATE_FUNC(chptrd)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_complex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chptrd);

int __TEMPLATE_FUNC(chptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chptrf);

int __TEMPLATE_FUNC(chptri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chptri);

int __TEMPLATE_FUNC(chptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chptrs);

int __TEMPLATE_FUNC(chsein)(char *__side, char *__eigsrc, char *__initv,
        __CLPK_logical *__select, __CLPK_integer *__n, __CLPK_complex *__h__,
        __CLPK_integer *__ldh, __CLPK_complex *__w, __CLPK_complex *__vl,
        __CLPK_integer *__ldvl, __CLPK_complex *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_complex *__work,
        __CLPK_real *__rwork, __CLPK_integer *__ifaill,
        __CLPK_integer *__ifailr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chsein);

int __TEMPLATE_FUNC(chseqr)(char *__job, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_complex *__h__,
        __CLPK_integer *__ldh, __CLPK_complex *__w, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(chseqr);

int __TEMPLATE_FUNC(clabrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__tauq, __CLPK_complex *__taup,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_complex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(clabrd);

int __TEMPLATE_FUNC(clacgv)(__CLPK_integer *__n, __CLPK_complex *__x,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(clacgv);

int __TEMPLATE_FUNC(clacn2)(__CLPK_integer *__n, __CLPK_complex *__v, __CLPK_complex *__x,
        __CLPK_real *__est, __CLPK_integer *__kase,
        __CLPK_integer *__isave)
__TEMPLATE_ALIAS(clacn2);

int __TEMPLATE_FUNC(clacon)(__CLPK_integer *__n, __CLPK_complex *__v, __CLPK_complex *__x,
        __CLPK_real *__est,
        __CLPK_integer *__kase)
__TEMPLATE_ALIAS(clacon);

int __TEMPLATE_FUNC(clacp2)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(clacp2);

int __TEMPLATE_FUNC(clacpy)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(clacpy);

int __TEMPLATE_FUNC(clacrm)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__rwork)
__TEMPLATE_ALIAS(clacrm);

int __TEMPLATE_FUNC(clacrt)(__CLPK_integer *__n, __CLPK_complex *__cx, __CLPK_integer *__incx,
        __CLPK_complex *__cy, __CLPK_integer *__incy, __CLPK_complex *__c__,
        __CLPK_complex *__s)
__TEMPLATE_ALIAS(clacrt);

void __TEMPLATE_FUNC(cladiv)(__CLPK_complex *__ret_val, __CLPK_complex *__x,
        __CLPK_complex *__y)
__TEMPLATE_ALIAS(cladiv);

int __TEMPLATE_FUNC(claed0)(__CLPK_integer *__qsiz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__q, __CLPK_integer *__ldq,
        __CLPK_complex *__qstore, __CLPK_integer *__ldqs, __CLPK_real *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claed0);

int __TEMPLATE_FUNC(claed7)(__CLPK_integer *__n, __CLPK_integer *__cutpnt,
        __CLPK_integer *__qsiz, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm, __CLPK_real *__d__,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_real *__rho,
        __CLPK_integer *__indxq, __CLPK_real *__qstore, __CLPK_integer *__qptr,
        __CLPK_integer *__prmptr, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_real *__givnum, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claed7);

int __TEMPLATE_FUNC(claed8)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__qsiz,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_real *__d__,
        __CLPK_real *__rho, __CLPK_integer *__cutpnt, __CLPK_real *__z__,
        __CLPK_real *__dlamda, __CLPK_complex *__q2, __CLPK_integer *__ldq2,
        __CLPK_real *__w, __CLPK_integer *__indxp, __CLPK_integer *__indx,
        __CLPK_integer *__indxq, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_real *__givnum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claed8);

int __TEMPLATE_FUNC(claein)(__CLPK_logical *__rightv, __CLPK_logical *__noinit,
        __CLPK_integer *__n, __CLPK_complex *__h__, __CLPK_integer *__ldh,
        __CLPK_complex *__w, __CLPK_complex *__v, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_real *__rwork, __CLPK_real *__eps3,
        __CLPK_real *__smlnum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claein);

int __TEMPLATE_FUNC(claesy)(__CLPK_complex *__a, __CLPK_complex *__b, __CLPK_complex *__c__,
        __CLPK_complex *__rt1, __CLPK_complex *__rt2, __CLPK_complex *__evscal,
        __CLPK_complex *__cs1,
        __CLPK_complex *__sn1)
__TEMPLATE_ALIAS(claesy);

int __TEMPLATE_FUNC(claev2)(__CLPK_complex *__a, __CLPK_complex *__b, __CLPK_complex *__c__,
        __CLPK_real *__rt1, __CLPK_real *__rt2, __CLPK_real *__cs1,
        __CLPK_complex *__sn1)
__TEMPLATE_ALIAS(claev2);

int __TEMPLATE_FUNC(clag2z)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__sa,
        __CLPK_integer *__ldsa, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clag2z);

int __TEMPLATE_FUNC(clags2)(__CLPK_logical *__upper, __CLPK_real *__a1, __CLPK_complex *__a2,
        __CLPK_real *__a3, __CLPK_real *__b1, __CLPK_complex *__b2,
        __CLPK_real *__b3, __CLPK_real *__csu, __CLPK_complex *__snu,
        __CLPK_real *__csv, __CLPK_complex *__snv, __CLPK_real *__csq,
        __CLPK_complex *__snq)
__TEMPLATE_ALIAS(clags2);

int __TEMPLATE_FUNC(clagtm)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__alpha, __CLPK_complex *__dl, __CLPK_complex *__d__,
        __CLPK_complex *__du, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__beta, __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(clagtm);

int __TEMPLATE_FUNC(clahef)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_complex *__w, __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clahef);

int __TEMPLATE_FUNC(clahqr)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__h__, __CLPK_integer *__ldh, __CLPK_complex *__w,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_complex *__z__,
        __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clahqr);

int __TEMPLATE_FUNC(clahr2)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__t, __CLPK_integer *__ldt, __CLPK_complex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(clahr2);

int __TEMPLATE_FUNC(clahrd)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__t, __CLPK_integer *__ldt, __CLPK_complex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(clahrd);

int __TEMPLATE_FUNC(claic1)(__CLPK_integer *__job, __CLPK_integer *__j, __CLPK_complex *__x,
        __CLPK_real *__sest, __CLPK_complex *__w, __CLPK_complex *__gamma,
        __CLPK_real *__sestpr, __CLPK_complex *__s,
        __CLPK_complex *__c__)
__TEMPLATE_ALIAS(claic1);

int __TEMPLATE_FUNC(clals0)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__nrhs,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__bx,
        __CLPK_integer *__ldbx, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_real *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_real *__poles, __CLPK_real *__difl,
        __CLPK_real *__difr, __CLPK_real *__z__, __CLPK_integer *__k,
        __CLPK_real *__c__, __CLPK_real *__s, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clals0);

int __TEMPLATE_FUNC(clalsa)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__bx, __CLPK_integer *__ldbx,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__vt,
        __CLPK_integer *__k, __CLPK_real *__difl, __CLPK_real *__difr,
        __CLPK_real *__z__, __CLPK_real *__poles, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_integer *__perm, __CLPK_real *__givnum, __CLPK_real *__c__,
        __CLPK_real *__s, __CLPK_real *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clalsa);

int __TEMPLATE_FUNC(clalsd)(char *__uplo, __CLPK_integer *__smlsiz, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clalsd);

__CLPK_doublereal __TEMPLATE_FUNC(clangb)(char *__norm, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_complex *__ab,
        __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clangb);

__CLPK_doublereal __TEMPLATE_FUNC(clange)(char *__norm, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clange);

__CLPK_doublereal __TEMPLATE_FUNC(clangt)(char *__norm, __CLPK_integer *__n,
        __CLPK_complex *__dl, __CLPK_complex *__d__,
        __CLPK_complex *__du)
__TEMPLATE_ALIAS(clangt);

__CLPK_doublereal __TEMPLATE_FUNC(clanhb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clanhb);

__CLPK_doublereal __TEMPLATE_FUNC(clanhe)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clanhe);

__CLPK_doublereal __TEMPLATE_FUNC(clanhf)(char *__norm, char *__transr, char *__uplo,
        __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clanhf);

__CLPK_doublereal __TEMPLATE_FUNC(clanhp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clanhp);

__CLPK_doublereal __TEMPLATE_FUNC(clanhs)(char *__norm, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clanhs);

__CLPK_doublereal __TEMPLATE_FUNC(clanht)(char *__norm, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_complex *__e)
__TEMPLATE_ALIAS(clanht);

__CLPK_doublereal __TEMPLATE_FUNC(clansb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clansb);

__CLPK_doublereal __TEMPLATE_FUNC(clansp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clansp);

__CLPK_doublereal __TEMPLATE_FUNC(clansy)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clansy);

__CLPK_doublereal __TEMPLATE_FUNC(clantb)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__ab,
        __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clantb);

__CLPK_doublereal __TEMPLATE_FUNC(clantp)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clantp);

__CLPK_doublereal __TEMPLATE_FUNC(clantr)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clantr);

int __TEMPLATE_FUNC(clapll)(__CLPK_integer *__n, __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_complex *__y, __CLPK_integer *__incy,
        __CLPK_real *__ssmin)
__TEMPLATE_ALIAS(clapll);

int __TEMPLATE_FUNC(clapmt)(__CLPK_logical *__forwrd, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_integer *__k)
__TEMPLATE_ALIAS(clapmt);

int __TEMPLATE_FUNC(claqgb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqgb);

int __TEMPLATE_FUNC(claqge)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqge);

int __TEMPLATE_FUNC(claqhb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_real *__s,
        __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqhb);

int __TEMPLATE_FUNC(claqhe)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, char *__equed)
__TEMPLATE_ALIAS(claqhe);

int __TEMPLATE_FUNC(claqhp)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqhp);

int __TEMPLATE_FUNC(claqp2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_complex *__tau, __CLPK_real *__vn1, __CLPK_real *__vn2,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(claqp2);

int __TEMPLATE_FUNC(claqps)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_integer *__nb, __CLPK_integer *__kb, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_complex *__tau,
        __CLPK_real *__vn1, __CLPK_real *__vn2, __CLPK_complex *__auxv,
        __CLPK_complex *__f,
        __CLPK_integer *__ldf)
__TEMPLATE_ALIAS(claqps);

int __TEMPLATE_FUNC(claqr0)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__h__, __CLPK_integer *__ldh, __CLPK_complex *__w,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claqr0);

int __TEMPLATE_FUNC(claqr1)(__CLPK_integer *__n, __CLPK_complex *__h__, __CLPK_integer *__ldh,
        __CLPK_complex *__s1, __CLPK_complex *__s2,
        __CLPK_complex *__v)
__TEMPLATE_ALIAS(claqr1);

int __TEMPLATE_FUNC(claqr2)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_complex *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__ns, __CLPK_integer *__nd,
        __CLPK_complex *__sh, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__nh, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_integer *__nv, __CLPK_complex *__wv, __CLPK_integer *__ldwv,
        __CLPK_complex *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(claqr2);

int __TEMPLATE_FUNC(claqr3)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_complex *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__ns, __CLPK_integer *__nd,
        __CLPK_complex *__sh, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__nh, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_integer *__nv, __CLPK_complex *__wv, __CLPK_integer *__ldwv,
        __CLPK_complex *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(claqr3);

int __TEMPLATE_FUNC(claqr4)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__h__, __CLPK_integer *__ldh, __CLPK_complex *__w,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(claqr4);

int __TEMPLATE_FUNC(claqr5)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__kacc22, __CLPK_integer *__n, __CLPK_integer *__ktop,
        __CLPK_integer *__kbot, __CLPK_integer *__nshfts, __CLPK_complex *__s,
        __CLPK_complex *__h__, __CLPK_integer *__ldh, __CLPK_integer *__iloz,
        __CLPK_integer *__ihiz, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__v, __CLPK_integer *__ldv, __CLPK_complex *__u,
        __CLPK_integer *__ldu, __CLPK_integer *__nv, __CLPK_complex *__wv,
        __CLPK_integer *__ldwv, __CLPK_integer *__nh, __CLPK_complex *__wh,
        __CLPK_integer *__ldwh)
__TEMPLATE_ALIAS(claqr5);

int __TEMPLATE_FUNC(claqsb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_real *__s,
        __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqsb);

int __TEMPLATE_FUNC(claqsp)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(claqsp);

int __TEMPLATE_FUNC(claqsy)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, char *__equed)
__TEMPLATE_ALIAS(claqsy);

int __TEMPLATE_FUNC(clar1v)(__CLPK_integer *__n, __CLPK_integer *__b1, __CLPK_integer *__bn,
        __CLPK_real *__lambda, __CLPK_real *__d__, __CLPK_real *__l,
        __CLPK_real *__ld, __CLPK_real *__lld, __CLPK_real *__pivmin,
        __CLPK_real *__gaptol, __CLPK_complex *__z__, __CLPK_logical *__wantnc,
        __CLPK_integer *__negcnt, __CLPK_real *__ztz, __CLPK_real *__mingma,
        __CLPK_integer *__r__, __CLPK_integer *__isuppz, __CLPK_real *__nrminv,
        __CLPK_real *__resid, __CLPK_real *__rqcorr,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(clar1v);

int __TEMPLATE_FUNC(clar2v)(__CLPK_integer *__n, __CLPK_complex *__x, __CLPK_complex *__y,
        __CLPK_complex *__z__, __CLPK_integer *__incx, __CLPK_real *__c__,
        __CLPK_complex *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(clar2v);

int __TEMPLATE_FUNC(clarcm)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__rwork)
__TEMPLATE_ALIAS(clarcm);

int __TEMPLATE_FUNC(clarf)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__v, __CLPK_integer *__incv, __CLPK_complex *__tau,
        __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(clarf);

int __TEMPLATE_FUNC(clarfb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__v, __CLPK_integer *__ldv, __CLPK_complex *__t,
        __CLPK_integer *__ldt, __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_complex *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(clarfb);

int __TEMPLATE_FUNC(clarfg)(__CLPK_integer *__n, __CLPK_complex *__alpha, __CLPK_complex *__x,
        __CLPK_integer *__incx,
        __CLPK_complex *__tau)
__TEMPLATE_ALIAS(clarfg);

int __TEMPLATE_FUNC(clarfp)(__CLPK_integer *__n, __CLPK_complex *__alpha, __CLPK_complex *__x,
        __CLPK_integer *__incx,
        __CLPK_complex *__tau)
__TEMPLATE_ALIAS(clarfp);

int __TEMPLATE_FUNC(clarft)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_complex *__tau, __CLPK_complex *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(clarft);

int __TEMPLATE_FUNC(clarfx)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__v, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(clarfx);

int __TEMPLATE_FUNC(clargv)(__CLPK_integer *__n, __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_complex *__y, __CLPK_integer *__incy, __CLPK_real *__c__,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(clargv);

int __TEMPLATE_FUNC(clarnv)(__CLPK_integer *__idist, __CLPK_integer *__iseed,
        __CLPK_integer *__n,
        __CLPK_complex *__x)
__TEMPLATE_ALIAS(clarnv);

int __TEMPLATE_FUNC(clarrv)(__CLPK_integer *__n, __CLPK_real *__vl, __CLPK_real *__vu,
        __CLPK_real *__d__, __CLPK_real *__l, __CLPK_real *__pivmin,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_integer *__dol,
        __CLPK_integer *__dou, __CLPK_real *__minrgp, __CLPK_real *__rtol1,
        __CLPK_real *__rtol2, __CLPK_real *__w, __CLPK_real *__werr,
        __CLPK_real *__wgap, __CLPK_integer *__iblock, __CLPK_integer *__indexw,
        __CLPK_real *__gers, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clarrv);

int __TEMPLATE_FUNC(clarscl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_complex *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(clarscl2);

int __TEMPLATE_FUNC(clartg)(__CLPK_complex *__f, __CLPK_complex *__g, __CLPK_real *__cs,
        __CLPK_complex *__sn,
        __CLPK_complex *__r__)
__TEMPLATE_ALIAS(clartg);

int __TEMPLATE_FUNC(clartv)(__CLPK_integer *__n, __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_complex *__y, __CLPK_integer *__incy, __CLPK_real *__c__,
        __CLPK_complex *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(clartv);

int __TEMPLATE_FUNC(clarz)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__l, __CLPK_complex *__v, __CLPK_integer *__incv,
        __CLPK_complex *__tau, __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(clarz);

int __TEMPLATE_FUNC(clarzb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_complex *__t, __CLPK_integer *__ldt, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(clarzb);

int __TEMPLATE_FUNC(clarzt)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_complex *__v, __CLPK_integer *__ldv,
        __CLPK_complex *__tau, __CLPK_complex *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(clarzt);

int __TEMPLATE_FUNC(clascl)(char *__type__, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_real *__cfrom, __CLPK_real *__cto, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clascl);

int __TEMPLATE_FUNC(clascl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_complex *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(clascl2);

int __TEMPLATE_FUNC(claset)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__alpha, __CLPK_complex *__beta, __CLPK_complex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(claset);

int __TEMPLATE_FUNC(clasr)(char *__side, char *__pivot, char *__direct, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__c__, __CLPK_real *__s,
        __CLPK_complex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(clasr);

int __TEMPLATE_FUNC(classq)(__CLPK_integer *__n, __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_real *__scale,
        __CLPK_real *__sumsq)
__TEMPLATE_ALIAS(classq);

int __TEMPLATE_FUNC(claswp)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__k1, __CLPK_integer *__k2, __CLPK_integer *__ipiv,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(claswp);

int __TEMPLATE_FUNC(clasyf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_complex *__w, __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clasyf);

int __TEMPLATE_FUNC(clatbs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_integer *__kd, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__x, __CLPK_real *__scale,
        __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clatbs);

int __TEMPLATE_FUNC(clatdf)(__CLPK_integer *__ijob, __CLPK_integer *__n, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_complex *__rhs, __CLPK_real *__rdsum,
        __CLPK_real *__rdscal, __CLPK_integer *__ipiv,
        __CLPK_integer *__jpiv)
__TEMPLATE_ALIAS(clatdf);

int __TEMPLATE_FUNC(clatps)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_complex *__ap, __CLPK_complex *__x,
        __CLPK_real *__scale, __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clatps);

int __TEMPLATE_FUNC(clatrd)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__e,
        __CLPK_complex *__tau, __CLPK_complex *__w,
        __CLPK_integer *__ldw)
__TEMPLATE_ALIAS(clatrd);

int __TEMPLATE_FUNC(clatrs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__x, __CLPK_real *__scale, __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clatrs);

int __TEMPLATE_FUNC(clatrz)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__l,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(clatrz);

int __TEMPLATE_FUNC(clatzm)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__v, __CLPK_integer *__incv, __CLPK_complex *__tau,
        __CLPK_complex *__c1, __CLPK_complex *__c2, __CLPK_integer *__ldc,
        __CLPK_complex *__work)
__TEMPLATE_ALIAS(clatzm);

int __TEMPLATE_FUNC(clauu2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clauu2);

int __TEMPLATE_FUNC(clauum)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(clauum);

int __TEMPLATE_FUNC(cpbcon)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_real *__anorm,
        __CLPK_real *__rcond, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbcon);

int __TEMPLATE_FUNC(cpbequ)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab, __CLPK_real *__s,
        __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbequ);

int __TEMPLATE_FUNC(cpbrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_complex *__afb, __CLPK_integer *__ldafb, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbrfs);

int __TEMPLATE_FUNC(cpbstf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbstf);

int __TEMPLATE_FUNC(cpbsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbsv);

int __TEMPLATE_FUNC(cpbsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__afb, __CLPK_integer *__ldafb,
        char *__equed, __CLPK_real *__s, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbsvx);

int __TEMPLATE_FUNC(cpbtf2)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbtf2);

int __TEMPLATE_FUNC(cpbtrf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbtrf);

int __TEMPLATE_FUNC(cpbtrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpbtrs);

int __TEMPLATE_FUNC(cpftrf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpftrf);

int __TEMPLATE_FUNC(cpftri)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpftri);

int __TEMPLATE_FUNC(cpftrs)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpftrs);

int __TEMPLATE_FUNC(cpocon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpocon);

int __TEMPLATE_FUNC(cpoequ)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpoequ);

int __TEMPLATE_FUNC(cpoequb)(__CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpoequb);

int __TEMPLATE_FUNC(cporfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__af,
        __CLPK_integer *__ldaf, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cporfs);

int __TEMPLATE_FUNC(cposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cposv);

int __TEMPLATE_FUNC(cposvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__af, __CLPK_integer *__ldaf, char *__equed,
        __CLPK_real *__s, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cposvx);

int __TEMPLATE_FUNC(cpotf2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpotf2);

int __TEMPLATE_FUNC(cpotrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpotrf);

int __TEMPLATE_FUNC(cpotri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpotri);

int __TEMPLATE_FUNC(cpotrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpotrs);

int __TEMPLATE_FUNC(cppcon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__anorm, __CLPK_real *__rcond, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cppcon);

int __TEMPLATE_FUNC(cppequ)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cppequ);

int __TEMPLATE_FUNC(cpprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_complex *__afp, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpprfs);

int __TEMPLATE_FUNC(cppsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cppsv);

int __TEMPLATE_FUNC(cppsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__ap, __CLPK_complex *__afp,
        char *__equed, __CLPK_real *__s, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cppsvx);

int __TEMPLATE_FUNC(cpptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpptrf);

int __TEMPLATE_FUNC(cpptri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpptri);

int __TEMPLATE_FUNC(cpptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpptrs);

int __TEMPLATE_FUNC(cpstf2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_real *__tol, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpstf2);

int __TEMPLATE_FUNC(cpstrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_real *__tol, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpstrf);

int __TEMPLATE_FUNC(cptcon)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_complex *__e,
        __CLPK_real *__anorm, __CLPK_real *__rcond, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cptcon);

int __TEMPLATE_FUNC(cpteqr)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpteqr);

int __TEMPLATE_FUNC(cptrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__d__, __CLPK_complex *__e, __CLPK_real *__df,
        __CLPK_complex *__ef, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cptrfs);

int __TEMPLATE_FUNC(cptsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__d__,
        __CLPK_complex *__e, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cptsv);

int __TEMPLATE_FUNC(cptsvx)(char *__fact, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__d__, __CLPK_complex *__e, __CLPK_real *__df,
        __CLPK_complex *__ef, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cptsvx);

int __TEMPLATE_FUNC(cpttrf)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_complex *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpttrf);

int __TEMPLATE_FUNC(cpttrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__d__, __CLPK_complex *__e, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cpttrs);

int __TEMPLATE_FUNC(cptts2)(__CLPK_integer *__iuplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__d__, __CLPK_complex *__e,
        __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(cptts2);

int __TEMPLATE_FUNC(crot)(__CLPK_integer *__n, __CLPK_complex *__cx, __CLPK_integer *__incx,
        __CLPK_complex *__cy, __CLPK_integer *__incy, __CLPK_real *__c__,
        __CLPK_complex *__s)
__TEMPLATE_ALIAS(crot);

int __TEMPLATE_FUNC(cspcon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cspcon);

int __TEMPLATE_FUNC(cspmv)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__alpha,
        __CLPK_complex *__ap, __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_complex *__beta, __CLPK_complex *__y,
        __CLPK_integer *__incy)
__TEMPLATE_ALIAS(cspmv);

int __TEMPLATE_FUNC(cspr)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__alpha,
        __CLPK_complex *__x, __CLPK_integer *__incx,
        __CLPK_complex *__ap)
__TEMPLATE_ALIAS(cspr);

int __TEMPLATE_FUNC(csprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_complex *__afp, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csprfs);

int __TEMPLATE_FUNC(cspsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cspsv);

int __TEMPLATE_FUNC(cspsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__ap, __CLPK_complex *__afp,
        __CLPK_integer *__ipiv, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cspsvx);

int __TEMPLATE_FUNC(csptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csptrf);

int __TEMPLATE_FUNC(csptri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csptri);

int __TEMPLATE_FUNC(csptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__ap, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csptrs);

int __TEMPLATE_FUNC(csrscl)(__CLPK_integer *__n, __CLPK_real *__sa, __CLPK_complex *__sx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(csrscl);

int __TEMPLATE_FUNC(cstedc)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_complex *__work, __CLPK_integer *__lwork, __CLPK_real *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cstedc);

int __TEMPLATE_FUNC(cstegr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_complex *__z__, __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cstegr);

int __TEMPLATE_FUNC(cstein)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_integer *__iblock,
        __CLPK_integer *__isplit, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cstein);

int __TEMPLATE_FUNC(cstemr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__nzc, __CLPK_integer *__isuppz,
        __CLPK_logical *__tryrac, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cstemr);

int __TEMPLATE_FUNC(csteqr)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csteqr);

int __TEMPLATE_FUNC(csycon)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__anorm,
        __CLPK_real *__rcond, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csycon);

int __TEMPLATE_FUNC(csyequb)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csyequb);

int __TEMPLATE_FUNC(csymv)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__alpha,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__x,
        __CLPK_integer *__incx, __CLPK_complex *__beta, __CLPK_complex *__y,
        __CLPK_integer *__incy)
__TEMPLATE_ALIAS(csymv);

int __TEMPLATE_FUNC(csyr)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__alpha,
        __CLPK_complex *__x, __CLPK_integer *__incx, __CLPK_complex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(csyr);

int __TEMPLATE_FUNC(csyrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csyrfs);

int __TEMPLATE_FUNC(csysv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csysv);

int __TEMPLATE_FUNC(csysvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csysvx);

int __TEMPLATE_FUNC(csytf2)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csytf2);

int __TEMPLATE_FUNC(csytrf)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csytrf);

int __TEMPLATE_FUNC(csytri)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csytri);

int __TEMPLATE_FUNC(csytrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(csytrs);

int __TEMPLATE_FUNC(ctbcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_complex *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__rcond, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctbcon);

int __TEMPLATE_FUNC(ctbrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctbrfs);

int __TEMPLATE_FUNC(ctbtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_complex *__ab,
        __CLPK_integer *__ldab, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctbtrs);

int __TEMPLATE_FUNC(ctfsm)(char *__transr, char *__side, char *__uplo, char *__trans,
        char *__diag, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__alpha, __CLPK_complex *__a, __CLPK_complex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(ctfsm);

int __TEMPLATE_FUNC(ctftri)(char *__transr, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctftri);

int __TEMPLATE_FUNC(ctfttp)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__arf, __CLPK_complex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctfttp);

int __TEMPLATE_FUNC(ctfttr)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__arf, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctfttr);

int __TEMPLATE_FUNC(ctgevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_complex *__s, __CLPK_integer *__lds,
        __CLPK_complex *__p, __CLPK_integer *__ldp, __CLPK_complex *__vl,
        __CLPK_integer *__ldvl, __CLPK_complex *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgevc);

int __TEMPLATE_FUNC(ctgex2)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__j1,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgex2);

int __TEMPLATE_FUNC(ctgexc)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__q,
        __CLPK_integer *__ldq, __CLPK_complex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgexc);

int __TEMPLATE_FUNC(ctgsen)(__CLPK_integer *__ijob, __CLPK_logical *__wantq,
        __CLPK_logical *__wantz, __CLPK_logical *__select, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__alpha, __CLPK_complex *__beta,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_complex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__m, __CLPK_real *__pl,
        __CLPK_real *__pr, __CLPK_real *__dif, __CLPK_complex *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgsen);

int __TEMPLATE_FUNC(ctgsja)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_real *__tola,
        __CLPK_real *__tolb, __CLPK_real *__alpha, __CLPK_real *__beta,
        __CLPK_complex *__u, __CLPK_integer *__ldu, __CLPK_complex *__v,
        __CLPK_integer *__ldv, __CLPK_complex *__q, __CLPK_integer *__ldq,
        __CLPK_complex *__work, __CLPK_integer *__ncycle,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgsja);

int __TEMPLATE_FUNC(ctgsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__vl,
        __CLPK_integer *__ldvl, __CLPK_complex *__vr, __CLPK_integer *__ldvr,
        __CLPK_real *__s, __CLPK_real *__dif, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgsna);

int __TEMPLATE_FUNC(ctgsy2)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__d__, __CLPK_integer *__ldd,
        __CLPK_complex *__e, __CLPK_integer *__lde, __CLPK_complex *__f,
        __CLPK_integer *__ldf, __CLPK_real *__scale, __CLPK_real *__rdsum,
        __CLPK_real *__rdscal,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgsy2);

int __TEMPLATE_FUNC(ctgsyl)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__d__, __CLPK_integer *__ldd,
        __CLPK_complex *__e, __CLPK_integer *__lde, __CLPK_complex *__f,
        __CLPK_integer *__ldf, __CLPK_real *__scale, __CLPK_real *__dif,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctgsyl);

int __TEMPLATE_FUNC(ctpcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_real *__rcond, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctpcon);

int __TEMPLATE_FUNC(ctprfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__ap, __CLPK_complex *__b,
        __CLPK_integer *__ldb, __CLPK_complex *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_complex *__work,
        __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctprfs);

int __TEMPLATE_FUNC(ctptri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctptri);

int __TEMPLATE_FUNC(ctptrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__ap, __CLPK_complex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctptrs);

int __TEMPLATE_FUNC(ctpttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__ap, __CLPK_complex *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctpttf);

int __TEMPLATE_FUNC(ctpttr)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctpttr);

int __TEMPLATE_FUNC(ctrcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_real *__rcond,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrcon);

int __TEMPLATE_FUNC(ctrevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_complex *__vl, __CLPK_integer *__ldvl, __CLPK_complex *__vr,
        __CLPK_integer *__ldvr, __CLPK_integer *__mm, __CLPK_integer *__m,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrevc);

int __TEMPLATE_FUNC(ctrexc)(char *__compq, __CLPK_integer *__n, __CLPK_complex *__t,
        __CLPK_integer *__ldt, __CLPK_complex *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrexc);

int __TEMPLATE_FUNC(ctrrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb, __CLPK_complex *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_complex *__work, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrrfs);

int __TEMPLATE_FUNC(ctrsen)(char *__job, char *__compq, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_complex *__q, __CLPK_integer *__ldq, __CLPK_complex *__w,
        __CLPK_integer *__m, __CLPK_real *__s, __CLPK_real *__sep,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrsen);

int __TEMPLATE_FUNC(ctrsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_complex *__t, __CLPK_integer *__ldt,
        __CLPK_complex *__vl, __CLPK_integer *__ldvl, __CLPK_complex *__vr,
        __CLPK_integer *__ldvr, __CLPK_real *__s, __CLPK_real *__sep,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_complex *__work,
        __CLPK_integer *__ldwork, __CLPK_real *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrsna);

int __TEMPLATE_FUNC(ctrsyl)(char *__trana, char *__tranb, __CLPK_integer *__isgn,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_complex *__c__, __CLPK_integer *__ldc, __CLPK_real *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrsyl);

int __TEMPLATE_FUNC(ctrti2)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrti2);

int __TEMPLATE_FUNC(ctrtri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrtri);

int __TEMPLATE_FUNC(ctrtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrtrs);

int __TEMPLATE_FUNC(ctrttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrttf);

int __TEMPLATE_FUNC(ctrttp)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctrttp);

int __TEMPLATE_FUNC(ctzrqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctzrqf);

int __TEMPLATE_FUNC(ctzrzf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ctzrzf);

int __TEMPLATE_FUNC(cung2l)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cung2l);

int __TEMPLATE_FUNC(cung2r)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cung2r);

int __TEMPLATE_FUNC(cungbr)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__tau, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungbr);

int __TEMPLATE_FUNC(cunghr)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunghr);

int __TEMPLATE_FUNC(cungl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungl2);

int __TEMPLATE_FUNC(cunglq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunglq);

int __TEMPLATE_FUNC(cungql)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungql);

int __TEMPLATE_FUNC(cungqr)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungqr);

int __TEMPLATE_FUNC(cungr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungr2);

int __TEMPLATE_FUNC(cungrq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungrq);

int __TEMPLATE_FUNC(cungtr)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cungtr);

int __TEMPLATE_FUNC(cunm2l)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunm2l);

int __TEMPLATE_FUNC(cunm2r)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunm2r);

int __TEMPLATE_FUNC(cunmbr)(char *__vect, char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmbr);

int __TEMPLATE_FUNC(cunmhr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__c__, __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmhr);

int __TEMPLATE_FUNC(cunml2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunml2);

int __TEMPLATE_FUNC(cunmlq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmlq);

int __TEMPLATE_FUNC(cunmql)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmql);

int __TEMPLATE_FUNC(cunmqr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmqr);

int __TEMPLATE_FUNC(cunmr2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmr2);

int __TEMPLATE_FUNC(cunmr3)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__c__, __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmr3);

int __TEMPLATE_FUNC(cunmrq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_complex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__tau, __CLPK_complex *__c__,
        __CLPK_integer *__ldc, __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmrq);

int __TEMPLATE_FUNC(cunmrz)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_complex *__a, __CLPK_integer *__lda, __CLPK_complex *__tau,
        __CLPK_complex *__c__, __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmrz);

int __TEMPLATE_FUNC(cunmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__a, __CLPK_integer *__lda,
        __CLPK_complex *__tau, __CLPK_complex *__c__, __CLPK_integer *__ldc,
        __CLPK_complex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cunmtr);

int __TEMPLATE_FUNC(cupgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_complex *__ap,
        __CLPK_complex *__tau, __CLPK_complex *__q, __CLPK_integer *__ldq,
        __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cupgtr);

int __TEMPLATE_FUNC(cupmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_complex *__ap, __CLPK_complex *__tau,
        __CLPK_complex *__c__, __CLPK_integer *__ldc, __CLPK_complex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(cupmtr);

int __TEMPLATE_FUNC(dbdsdc)(char *__uplo, char *__compq, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublereal *__q, __CLPK_integer *__iq,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dbdsdc);

int __TEMPLATE_FUNC(dbdsqr)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__ncvt,
        __CLPK_integer *__nru, __CLPK_integer *__ncc, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__vt, __CLPK_integer *__ldvt,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dbdsqr);

int __TEMPLATE_FUNC(ddisna)(char *__job, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__sep,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ddisna);

int __TEMPLATE_FUNC(dgbbrd)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__ncc, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__pt,
        __CLPK_integer *__ldpt, __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbbrd);

int __TEMPLATE_FUNC(dgbcon)(char *__norm, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbcon);

int __TEMPLATE_FUNC(dgbequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__rowcnd, __CLPK_doublereal *__colcnd,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbequ);

int __TEMPLATE_FUNC(dgbequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__rowcnd, __CLPK_doublereal *__colcnd,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbequb);

int __TEMPLATE_FUNC(dgbrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__afb,
        __CLPK_integer *__ldafb, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbrfs);

int __TEMPLATE_FUNC(dgbsv)(__CLPK_integer *__n, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbsv);

int __TEMPLATE_FUNC(dgbsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__afb, __CLPK_integer *__ldafb,
        __CLPK_integer *__ipiv, char *__equed, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbsvx);

int __TEMPLATE_FUNC(dgbtf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbtf2);

int __TEMPLATE_FUNC(dgbtrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbtrf);

int __TEMPLATE_FUNC(dgbtrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgbtrs);

int __TEMPLATE_FUNC(dgebak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__scale, __CLPK_integer *__m, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgebak);

int __TEMPLATE_FUNC(dgebal)(char *__job, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgebal);

int __TEMPLATE_FUNC(dgebd2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tauq, __CLPK_doublereal *__taup,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgebd2);

int __TEMPLATE_FUNC(dgebrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tauq, __CLPK_doublereal *__taup,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgebrd);

int __TEMPLATE_FUNC(dgecon)(char *__norm, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgecon);

int __TEMPLATE_FUNC(dgeequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeequ);

int __TEMPLATE_FUNC(dgeequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeequb);

int __TEMPLATE_FUNC(dgees)(char *__jobvs, char *__sort, __CLPK_L_fp __select,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_doublereal *__wr,
        __CLPK_doublereal *__wi, __CLPK_doublereal *__vs,
        __CLPK_integer *__ldvs, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgees);

int __TEMPLATE_FUNC(dgeesx)(char *__jobvs, char *__sort, __CLPK_L_fp __select, char *__sense,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_doublereal *__wr,
        __CLPK_doublereal *__wi, __CLPK_doublereal *__vs,
        __CLPK_integer *__ldvs, __CLPK_doublereal *__rconde,
        __CLPK_doublereal *__rcondv, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeesx);

int __TEMPLATE_FUNC(dgeev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__wr,
        __CLPK_doublereal *__wi, __CLPK_doublereal *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeev);

int __TEMPLATE_FUNC(dgeevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr, __CLPK_integer *__ilo,
        __CLPK_integer *__ihi, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__abnrm, __CLPK_doublereal *__rconde,
        __CLPK_doublereal *__rcondv, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeevx);

int __TEMPLATE_FUNC(dgegs)(char *__jobvsl, char *__jobvsr, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__alphar,
        __CLPK_doublereal *__alphai, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__vsl, __CLPK_integer *__ldvsl,
        __CLPK_doublereal *__vsr, __CLPK_integer *__ldvsr,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgegs);

int __TEMPLATE_FUNC(dgegv)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__alphar,
        __CLPK_doublereal *__alphai, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgegv);

int __TEMPLATE_FUNC(dgehd2)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgehd2);

int __TEMPLATE_FUNC(dgehrd)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgehrd);

int __TEMPLATE_FUNC(dgejsv)(char *__joba, char *__jobu, char *__jobv, char *__jobr,
        char *__jobt, char *__jobp, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__sva,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgejsv);

int __TEMPLATE_FUNC(dgelq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelq2);

int __TEMPLATE_FUNC(dgelqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelqf);

int __TEMPLATE_FUNC(dgels)(char *__trans, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgels);

int __TEMPLATE_FUNC(dgelsd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__s,
        __CLPK_doublereal *__rcond, __CLPK_integer *__rank,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelsd);

int __TEMPLATE_FUNC(dgelss)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__s,
        __CLPK_doublereal *__rcond, __CLPK_integer *__rank,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelss);

int __TEMPLATE_FUNC(dgelsx)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt,
        __CLPK_doublereal *__rcond, __CLPK_integer *__rank,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelsx);

int __TEMPLATE_FUNC(dgelsy)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt,
        __CLPK_doublereal *__rcond, __CLPK_integer *__rank,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgelsy);

int __TEMPLATE_FUNC(dgeql2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeql2);

int __TEMPLATE_FUNC(dgeqlf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeqlf);

int __TEMPLATE_FUNC(dgeqp3)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeqp3);

int __TEMPLATE_FUNC(dgeqpf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeqpf);

int __TEMPLATE_FUNC(dgeqr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeqr2);

int __TEMPLATE_FUNC(dgeqrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgeqrf);

int __TEMPLATE_FUNC(dgerfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgerfs);

int __TEMPLATE_FUNC(dgerq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgerq2);

int __TEMPLATE_FUNC(dgerqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgerqf);

int __TEMPLATE_FUNC(dgesc2)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__rhs, __CLPK_integer *__ipiv,
        __CLPK_integer *__jpiv,
        __CLPK_doublereal *__scale)
__TEMPLATE_ALIAS(dgesc2);

int __TEMPLATE_FUNC(dgesdd)(char *__jobz, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgesdd);

int __TEMPLATE_FUNC(dgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgesv);

int __TEMPLATE_FUNC(dgesvd)(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublereal *__u, __CLPK_integer *__ldu,
        __CLPK_doublereal *__vt, __CLPK_integer *__ldvt,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgesvd);

int __TEMPLATE_FUNC(dgesvj)(char *__joba, char *__jobu, char *__jobv, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__sva, __CLPK_integer *__mv, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgesvj);

int __TEMPLATE_FUNC(dgesvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        char *__equed, __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgesvx);

int __TEMPLATE_FUNC(dgetc2)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgetc2);

int __TEMPLATE_FUNC(dgetf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgetf2);

int __TEMPLATE_FUNC(dgetrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgetrf);

int __TEMPLATE_FUNC(dgetri)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgetri);

int __TEMPLATE_FUNC(dgetrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgetrs);

int __TEMPLATE_FUNC(dggbak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_integer *__m, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggbak);

int __TEMPLATE_FUNC(dggbal)(char *__job, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggbal);

int __TEMPLATE_FUNC(dgges)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_integer *__sdim,
        __CLPK_doublereal *__alphar, __CLPK_doublereal *__alphai,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__vsl,
        __CLPK_integer *__ldvsl, __CLPK_doublereal *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgges);

int __TEMPLATE_FUNC(dggesx)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        char *__sense, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__sdim, __CLPK_doublereal *__alphar,
        __CLPK_doublereal *__alphai, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__vsl, __CLPK_integer *__ldvsl,
        __CLPK_doublereal *__vsr, __CLPK_integer *__ldvsr,
        __CLPK_doublereal *__rconde, __CLPK_doublereal *__rcondv,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggesx);

int __TEMPLATE_FUNC(dggev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__alphar,
        __CLPK_doublereal *__alphai, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggev);

int __TEMPLATE_FUNC(dggevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__alphar, __CLPK_doublereal *__alphai,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_doublereal *__abnrm, __CLPK_doublereal *__bbnrm,
        __CLPK_doublereal *__rconde, __CLPK_doublereal *__rcondv,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggevx);

int __TEMPLATE_FUNC(dggglm)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__d__, __CLPK_doublereal *__x,
        __CLPK_doublereal *__y, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggglm);

int __TEMPLATE_FUNC(dgghrd)(char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__z__,
        __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgghrd);

int __TEMPLATE_FUNC(dgglse)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__p,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__x,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgglse);

int __TEMPLATE_FUNC(dggqrf)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__taua, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__taub,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggqrf);

int __TEMPLATE_FUNC(dggrqf)(__CLPK_integer *__m, __CLPK_integer *__p, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__taua, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__taub,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggrqf);

int __TEMPLATE_FUNC(dggsvd)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__p, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggsvd);

int __TEMPLATE_FUNC(dggsvp)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__tola, __CLPK_doublereal *__tolb,
        __CLPK_integer *__k, __CLPK_integer *__l, __CLPK_doublereal *__u,
        __CLPK_integer *__ldu, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_integer *__iwork,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dggsvp);

int __TEMPLATE_FUNC(dgsvj0)(char *__jobv, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__sva, __CLPK_integer *__mv, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv, __CLPK_doublereal *__eps,
        __CLPK_doublereal *__sfmin, __CLPK_doublereal *__tol,
        __CLPK_integer *__nsweep, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgsvj0);

int __TEMPLATE_FUNC(dgsvj1)(char *__jobv, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__n1, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__sva,
        __CLPK_integer *__mv, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_doublereal *__eps, __CLPK_doublereal *__sfmin,
        __CLPK_doublereal *__tol, __CLPK_integer *__nsweep,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgsvj1);

int __TEMPLATE_FUNC(dgtcon)(char *__norm, __CLPK_integer *__n, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgtcon);

int __TEMPLATE_FUNC(dgtrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__dl, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__du, __CLPK_doublereal *__dlf,
        __CLPK_doublereal *__df, __CLPK_doublereal *__duf,
        __CLPK_doublereal *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgtrfs);

int __TEMPLATE_FUNC(dgtsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgtsv);

int __TEMPLATE_FUNC(dgtsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__dlf, __CLPK_doublereal *__df,
        __CLPK_doublereal *__duf, __CLPK_doublereal *__du2,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgtsvx);

int __TEMPLATE_FUNC(dgttrf)(__CLPK_integer *__n, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__du2, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgttrf);

int __TEMPLATE_FUNC(dgttrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__dl, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__du, __CLPK_doublereal *__du2,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dgttrs);

int __TEMPLATE_FUNC(dgtts2)(__CLPK_integer *__itrans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(dgtts2);

int __TEMPLATE_FUNC(dhgeqz)(char *__job, char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_doublereal *__h__,
        __CLPK_integer *__ldh, __CLPK_doublereal *__t, __CLPK_integer *__ldt,
        __CLPK_doublereal *__alphar, __CLPK_doublereal *__alphai,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dhgeqz);

int __TEMPLATE_FUNC(dhsein)(char *__side, char *__eigsrc, char *__initv,
        __CLPK_logical *__select, __CLPK_integer *__n, __CLPK_doublereal *__h__,
        __CLPK_integer *__ldh, __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_doublereal *__work,
        __CLPK_integer *__ifaill, __CLPK_integer *__ifailr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dhsein);

int __TEMPLATE_FUNC(dhseqr)(char *__job, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_doublereal *__h__,
        __CLPK_integer *__ldh, __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dhseqr);


        __CLPK_logical __TEMPLATE_FUNC(disnan)(__CLPK_doublereal *__din)
__TEMPLATE_ALIAS(disnan);

int __TEMPLATE_FUNC(dlabad)(__CLPK_doublereal *__small,
        __CLPK_doublereal *__large)
__TEMPLATE_ALIAS(dlabad);

int __TEMPLATE_FUNC(dlabrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__tauq,
        __CLPK_doublereal *__taup, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(dlabrd);

int __TEMPLATE_FUNC(dlacn2)(__CLPK_integer *__n, __CLPK_doublereal *__v, __CLPK_doublereal *__x,
        __CLPK_integer *__isgn, __CLPK_doublereal *__est,
        __CLPK_integer *__kase,
        __CLPK_integer *__isave)
__TEMPLATE_ALIAS(dlacn2);

int __TEMPLATE_FUNC(dlacon)(__CLPK_integer *__n, __CLPK_doublereal *__v, __CLPK_doublereal *__x,
        __CLPK_integer *__isgn, __CLPK_doublereal *__est,
        __CLPK_integer *__kase)
__TEMPLATE_ALIAS(dlacon);

int __TEMPLATE_FUNC(dlacpy)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(dlacpy);

int __TEMPLATE_FUNC(dladiv)(__CLPK_doublereal *__a, __CLPK_doublereal *__b,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__p,
        __CLPK_doublereal *__q)
__TEMPLATE_ALIAS(dladiv);

int __TEMPLATE_FUNC(dlae2)(__CLPK_doublereal *__a, __CLPK_doublereal *__b,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rt1,
        __CLPK_doublereal *__rt2)
__TEMPLATE_ALIAS(dlae2);

int __TEMPLATE_FUNC(dlaebz)(__CLPK_integer *__ijob, __CLPK_integer *__nitmax,
        __CLPK_integer *__n, __CLPK_integer *__mmax, __CLPK_integer *__minp,
        __CLPK_integer *__nbmin, __CLPK_doublereal *__abstol,
        __CLPK_doublereal *__reltol, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__e2, __CLPK_integer *__nval,
        __CLPK_doublereal *__ab, __CLPK_doublereal *__c__,
        __CLPK_integer *__mout, __CLPK_integer *__nab,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaebz);

int __TEMPLATE_FUNC(dlaed0)(__CLPK_integer *__icompq, __CLPK_integer *__qsiz,
        __CLPK_integer *__n, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__qstore, __CLPK_integer *__ldqs,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed0);

int __TEMPLATE_FUNC(dlaed1)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_integer *__indxq,
        __CLPK_doublereal *__rho, __CLPK_integer *__cutpnt,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed1);

int __TEMPLATE_FUNC(dlaed2)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__n1,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__indxq, __CLPK_doublereal *__rho,
        __CLPK_doublereal *__z__, __CLPK_doublereal *__dlamda,
        __CLPK_doublereal *__w, __CLPK_doublereal *__q2, __CLPK_integer *__indx,
        __CLPK_integer *__indxc, __CLPK_integer *__indxp,
        __CLPK_integer *__coltyp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed2);

int __TEMPLATE_FUNC(dlaed3)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__n1,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__rho, __CLPK_doublereal *__dlamda,
        __CLPK_doublereal *__q2, __CLPK_integer *__indx, __CLPK_integer *__ctot,
        __CLPK_doublereal *__w, __CLPK_doublereal *__s,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed3);

int __TEMPLATE_FUNC(dlaed4)(__CLPK_integer *__n, __CLPK_integer *__i__,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__delta, __CLPK_doublereal *__rho,
        __CLPK_doublereal *__dlam,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed4);

int __TEMPLATE_FUNC(dlaed5)(__CLPK_integer *__i__, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__z__, __CLPK_doublereal *__delta,
        __CLPK_doublereal *__rho,
        __CLPK_doublereal *__dlam)
__TEMPLATE_ALIAS(dlaed5);

int __TEMPLATE_FUNC(dlaed6)(__CLPK_integer *__kniter, __CLPK_logical *__orgati,
        __CLPK_doublereal *__rho, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__z__, __CLPK_doublereal *__finit,
        __CLPK_doublereal *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed6);

int __TEMPLATE_FUNC(dlaed7)(__CLPK_integer *__icompq, __CLPK_integer *__n,
        __CLPK_integer *__qsiz, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__indxq, __CLPK_doublereal *__rho,
        __CLPK_integer *__cutpnt, __CLPK_doublereal *__qstore,
        __CLPK_integer *__qptr, __CLPK_integer *__prmptr,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_doublereal *__givnum,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed7);

int __TEMPLATE_FUNC(dlaed8)(__CLPK_integer *__icompq, __CLPK_integer *__k, __CLPK_integer *__n,
        __CLPK_integer *__qsiz, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_integer *__indxq,
        __CLPK_doublereal *__rho, __CLPK_integer *__cutpnt,
        __CLPK_doublereal *__z__, __CLPK_doublereal *__dlamda,
        __CLPK_doublereal *__q2, __CLPK_integer *__ldq2, __CLPK_doublereal *__w,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_doublereal *__givnum,
        __CLPK_integer *__indxp, __CLPK_integer *__indx,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed8);

int __TEMPLATE_FUNC(dlaed9)(__CLPK_integer *__k, __CLPK_integer *__kstart,
        __CLPK_integer *__kstop, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__rho,
        __CLPK_doublereal *__dlamda, __CLPK_doublereal *__w,
        __CLPK_doublereal *__s, __CLPK_integer *__lds,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaed9);

int __TEMPLATE_FUNC(dlaeda)(__CLPK_integer *__n, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm,
        __CLPK_integer *__prmptr, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_doublereal *__givnum, __CLPK_doublereal *__q,
        __CLPK_integer *__qptr, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__ztemp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaeda);

int __TEMPLATE_FUNC(dlaein)(__CLPK_logical *__rightv, __CLPK_logical *__noinit,
        __CLPK_integer *__n, __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_doublereal *__vr, __CLPK_doublereal *__vi,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__work, __CLPK_doublereal *__eps3,
        __CLPK_doublereal *__smlnum, __CLPK_doublereal *__bignum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaein);

int __TEMPLATE_FUNC(dlaev2)(__CLPK_doublereal *__a, __CLPK_doublereal *__b,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rt1,
        __CLPK_doublereal *__rt2, __CLPK_doublereal *__cs1,
        __CLPK_doublereal *__sn1)
__TEMPLATE_ALIAS(dlaev2);

int __TEMPLATE_FUNC(dlaexc)(__CLPK_logical *__wantq, __CLPK_integer *__n,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_doublereal *__q,
        __CLPK_integer *__ldq, __CLPK_integer *__j1, __CLPK_integer *__n1,
        __CLPK_integer *__n2, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaexc);

int __TEMPLATE_FUNC(dlag2)(__CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__safmin, __CLPK_doublereal *__scale1,
        __CLPK_doublereal *__scale2, __CLPK_doublereal *__wr1,
        __CLPK_doublereal *__wr2,
        __CLPK_doublereal *__wi)
__TEMPLATE_ALIAS(dlag2);

int __TEMPLATE_FUNC(dlag2s)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_real *__sa, __CLPK_integer *__ldsa,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlag2s);

int __TEMPLATE_FUNC(dlags2)(__CLPK_logical *__upper, __CLPK_doublereal *__a1,
        __CLPK_doublereal *__a2, __CLPK_doublereal *__a3,
        __CLPK_doublereal *__b1, __CLPK_doublereal *__b2,
        __CLPK_doublereal *__b3, __CLPK_doublereal *__csu,
        __CLPK_doublereal *__snu, __CLPK_doublereal *__csv,
        __CLPK_doublereal *__snv, __CLPK_doublereal *__csq,
        __CLPK_doublereal *__snq)
__TEMPLATE_ALIAS(dlags2);

int __TEMPLATE_FUNC(dlagtf)(__CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_doublereal *__lambda, __CLPK_doublereal *__b,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__tol,
        __CLPK_doublereal *__d__, __CLPK_integer *__in,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlagtf);

int __TEMPLATE_FUNC(dlagtm)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__dl,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__du,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(dlagtm);

int __TEMPLATE_FUNC(dlagts)(__CLPK_integer *__job, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_doublereal *__b, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__d__, __CLPK_integer *__in, __CLPK_doublereal *__y,
        __CLPK_doublereal *__tol,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlagts);

int __TEMPLATE_FUNC(dlagv2)(__CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__alphar, __CLPK_doublereal *__alphai,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__csl,
        __CLPK_doublereal *__snl, __CLPK_doublereal *__csr,
        __CLPK_doublereal *__snr)
__TEMPLATE_ALIAS(dlagv2);

int __TEMPLATE_FUNC(dlahqr)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlahqr);

int __TEMPLATE_FUNC(dlahr2)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_doublereal *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(dlahr2);

int __TEMPLATE_FUNC(dlahrd)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_doublereal *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(dlahrd);

int __TEMPLATE_FUNC(dlaic1)(__CLPK_integer *__job, __CLPK_integer *__j, __CLPK_doublereal *__x,
        __CLPK_doublereal *__sest, __CLPK_doublereal *__w,
        __CLPK_doublereal *__gamma, __CLPK_doublereal *__sestpr,
        __CLPK_doublereal *__s,
        __CLPK_doublereal *__c__)
__TEMPLATE_ALIAS(dlaic1);

__CLPK_logical __TEMPLATE_FUNC(dlaisnan)(__CLPK_doublereal *__din1,
        __CLPK_doublereal *__din2)
__TEMPLATE_ALIAS(dlaisnan);

int __TEMPLATE_FUNC(dlaln2)(__CLPK_logical *__ltrans, __CLPK_integer *__na,
        __CLPK_integer *__nw, __CLPK_doublereal *__smin,
        __CLPK_doublereal *__ca, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__d1, __CLPK_doublereal *__d2,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__wr,
        __CLPK_doublereal *__wi, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__xnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaln2);

int __TEMPLATE_FUNC(dlals0)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__bx,
        __CLPK_integer *__ldbx, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_doublereal *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_doublereal *__poles,
        __CLPK_doublereal *__difl, __CLPK_doublereal *__difr,
        __CLPK_doublereal *__z__, __CLPK_integer *__k, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__s, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlals0);

int __TEMPLATE_FUNC(dlalsa)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__bx, __CLPK_integer *__ldbx,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__vt,
        __CLPK_integer *__k, __CLPK_doublereal *__difl,
        __CLPK_doublereal *__difr, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__poles, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_integer *__perm, __CLPK_doublereal *__givnum,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlalsa);

int __TEMPLATE_FUNC(dlalsd)(char *__uplo, __CLPK_integer *__smlsiz, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__rcond, __CLPK_integer *__rank,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlalsd);

int __TEMPLATE_FUNC(dlamrg)(__CLPK_integer *__n1, __CLPK_integer *__n2, __CLPK_doublereal *__a,
        __CLPK_integer *__dtrd1, __CLPK_integer *__dtrd2,
        __CLPK_integer *__index)
__TEMPLATE_ALIAS(dlamrg);

__CLPK_integer __TEMPLATE_FUNC(dlaneg)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__lld, __CLPK_doublereal *__sigma,
        __CLPK_doublereal *__pivmin,
        __CLPK_integer *__r__)
__TEMPLATE_ALIAS(dlaneg);

__CLPK_doublereal __TEMPLATE_FUNC(dlangb)(char *__norm, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlangb);

__CLPK_doublereal __TEMPLATE_FUNC(dlange)(char *__norm, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlange);

__CLPK_doublereal __TEMPLATE_FUNC(dlangt)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublereal *__dl, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__du)
__TEMPLATE_ALIAS(dlangt);

__CLPK_doublereal __TEMPLATE_FUNC(dlanhs)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlanhs);

__CLPK_doublereal __TEMPLATE_FUNC(dlansb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlansb);

__CLPK_doublereal __TEMPLATE_FUNC(dlansf)(char *__norm, char *__transr, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlansf);

__CLPK_doublereal __TEMPLATE_FUNC(dlansp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlansp);

__CLPK_doublereal __TEMPLATE_FUNC(dlanst)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e)
__TEMPLATE_ALIAS(dlanst);

__CLPK_doublereal __TEMPLATE_FUNC(dlansy)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlansy);

__CLPK_doublereal __TEMPLATE_FUNC(dlantb)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlantb);

__CLPK_doublereal __TEMPLATE_FUNC(dlantp)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlantp);

__CLPK_doublereal __TEMPLATE_FUNC(dlantr)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlantr);

int __TEMPLATE_FUNC(dlanv2)(__CLPK_doublereal *__a, __CLPK_doublereal *__b,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__rt1r, __CLPK_doublereal *__rt1i,
        __CLPK_doublereal *__rt2r, __CLPK_doublereal *__rt2i,
        __CLPK_doublereal *__cs,
        __CLPK_doublereal *__sn)
__TEMPLATE_ALIAS(dlanv2);

int __TEMPLATE_FUNC(dlapll)(__CLPK_integer *__n, __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__y, __CLPK_integer *__incy,
        __CLPK_doublereal *__ssmin)
__TEMPLATE_ALIAS(dlapll);

int __TEMPLATE_FUNC(dlapmt)(__CLPK_logical *__forwrd, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_integer *__k)
__TEMPLATE_ALIAS(dlapmt);

__CLPK_doublereal __TEMPLATE_FUNC(dlapy2)(__CLPK_doublereal *__x,
        __CLPK_doublereal *__y)
__TEMPLATE_ALIAS(dlapy2);

__CLPK_doublereal __TEMPLATE_FUNC(dlapy3)(__CLPK_doublereal *__x, __CLPK_doublereal *__y,
        __CLPK_doublereal *__z__)
__TEMPLATE_ALIAS(dlapy3);

int __TEMPLATE_FUNC(dlaqgb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__rowcnd, __CLPK_doublereal *__colcnd,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(dlaqgb);

int __TEMPLATE_FUNC(dlaqge)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(dlaqge);

int __TEMPLATE_FUNC(dlaqp2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__vn1,
        __CLPK_doublereal *__vn2,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlaqp2);

int __TEMPLATE_FUNC(dlaqps)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_integer *__nb, __CLPK_integer *__kb, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__vn1, __CLPK_doublereal *__vn2,
        __CLPK_doublereal *__auxv, __CLPK_doublereal *__f,
        __CLPK_integer *__ldf)
__TEMPLATE_ALIAS(dlaqps);

int __TEMPLATE_FUNC(dlaqr0)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaqr0);

int __TEMPLATE_FUNC(dlaqr1)(__CLPK_integer *__n, __CLPK_doublereal *__h__,
        __CLPK_integer *__ldh, __CLPK_doublereal *__sr1,
        __CLPK_doublereal *__si1, __CLPK_doublereal *__sr2,
        __CLPK_doublereal *__si2,
        __CLPK_doublereal *__v)
__TEMPLATE_ALIAS(dlaqr1);

int __TEMPLATE_FUNC(dlaqr2)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz, __CLPK_integer *__ns,
        __CLPK_integer *__nd, __CLPK_doublereal *__sr, __CLPK_doublereal *__si,
        __CLPK_doublereal *__v, __CLPK_integer *__ldv, __CLPK_integer *__nh,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_integer *__nv,
        __CLPK_doublereal *__wv, __CLPK_integer *__ldwv,
        __CLPK_doublereal *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(dlaqr2);

int __TEMPLATE_FUNC(dlaqr3)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz, __CLPK_integer *__ns,
        __CLPK_integer *__nd, __CLPK_doublereal *__sr, __CLPK_doublereal *__si,
        __CLPK_doublereal *__v, __CLPK_integer *__ldv, __CLPK_integer *__nh,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_integer *__nv,
        __CLPK_doublereal *__wv, __CLPK_integer *__ldwv,
        __CLPK_doublereal *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(dlaqr3);

int __TEMPLATE_FUNC(dlaqr4)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__h__, __CLPK_integer *__ldh,
        __CLPK_doublereal *__wr, __CLPK_doublereal *__wi,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaqr4);

int __TEMPLATE_FUNC(dlaqr5)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__kacc22, __CLPK_integer *__n, __CLPK_integer *__ktop,
        __CLPK_integer *__kbot, __CLPK_integer *__nshfts,
        __CLPK_doublereal *__sr, __CLPK_doublereal *__si,
        __CLPK_doublereal *__h__, __CLPK_integer *__ldh, __CLPK_integer *__iloz,
        __CLPK_integer *__ihiz, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__v, __CLPK_integer *__ldv, __CLPK_doublereal *__u,
        __CLPK_integer *__ldu, __CLPK_integer *__nv, __CLPK_doublereal *__wv,
        __CLPK_integer *__ldwv, __CLPK_integer *__nh, __CLPK_doublereal *__wh,
        __CLPK_integer *__ldwh)
__TEMPLATE_ALIAS(dlaqr5);

int __TEMPLATE_FUNC(dlaqsb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(dlaqsb);

int __TEMPLATE_FUNC(dlaqsp)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(dlaqsp);

int __TEMPLATE_FUNC(dlaqsy)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(dlaqsy);

int __TEMPLATE_FUNC(dlaqtr)(__CLPK_logical *__ltran, __CLPK_logical *__l__CLPK_real,
        __CLPK_integer *__n, __CLPK_doublereal *__t, __CLPK_integer *__ldt,
        __CLPK_doublereal *__b, __CLPK_doublereal *__w,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__x,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlaqtr);

int __TEMPLATE_FUNC(dlar1v)(__CLPK_integer *__n, __CLPK_integer *__b1, __CLPK_integer *__bn,
        __CLPK_doublereal *__lambda, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__l, __CLPK_doublereal *__ld,
        __CLPK_doublereal *__lld, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__gaptol, __CLPK_doublereal *__z__,
        __CLPK_logical *__wantnc, __CLPK_integer *__negcnt,
        __CLPK_doublereal *__ztz, __CLPK_doublereal *__mingma,
        __CLPK_integer *__r__, __CLPK_integer *__isuppz,
        __CLPK_doublereal *__nrminv, __CLPK_doublereal *__resid,
        __CLPK_doublereal *__rqcorr,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlar1v);

int __TEMPLATE_FUNC(dlar2v)(__CLPK_integer *__n, __CLPK_doublereal *__x, __CLPK_doublereal *__y,
        __CLPK_doublereal *__z__, __CLPK_integer *__incx,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(dlar2v);

int __TEMPLATE_FUNC(dlarf)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__v, __CLPK_integer *__incv,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlarf);

int __TEMPLATE_FUNC(dlarfb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__v, __CLPK_integer *__ldv, __CLPK_doublereal *__t,
        __CLPK_integer *__ldt, __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(dlarfb);

int __TEMPLATE_FUNC(dlarfg)(__CLPK_integer *__n, __CLPK_doublereal *__alpha,
        __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__tau)
__TEMPLATE_ALIAS(dlarfg);

int __TEMPLATE_FUNC(dlarfp)(__CLPK_integer *__n, __CLPK_doublereal *__alpha,
        __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__tau)
__TEMPLATE_ALIAS(dlarfp);

int __TEMPLATE_FUNC(dlarft)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(dlarft);

int __TEMPLATE_FUNC(dlarfx)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__v, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlarfx);

int __TEMPLATE_FUNC(dlargv)(__CLPK_integer *__n, __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__y, __CLPK_integer *__incy,
        __CLPK_doublereal *__c__,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(dlargv);

int __TEMPLATE_FUNC(dlarnv)(__CLPK_integer *__idist, __CLPK_integer *__iseed,
        __CLPK_integer *__n,
        __CLPK_doublereal *__x)
__TEMPLATE_ALIAS(dlarnv);

int __TEMPLATE_FUNC(dlarra)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__e2,
        __CLPK_doublereal *__spltol, __CLPK_doublereal *__tnrm,
        __CLPK_integer *__nsplit, __CLPK_integer *__isplit,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarra);

int __TEMPLATE_FUNC(dlarrb)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__lld, __CLPK_integer *__ifirst,
        __CLPK_integer *__ilast, __CLPK_doublereal *__rtol1,
        __CLPK_doublereal *__rtol2, __CLPK_integer *__offset,
        __CLPK_doublereal *__w, __CLPK_doublereal *__wgap,
        __CLPK_doublereal *__werr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__spdiam, __CLPK_integer *__twist,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrb);

int __TEMPLATE_FUNC(dlarrc)(char *__jobt, __CLPK_integer *__n, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__pivmin,
        __CLPK_integer *__eigcnt, __CLPK_integer *__lcnt,
        __CLPK_integer *__rcnt,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrc);

int __TEMPLATE_FUNC(dlarrd)(char *__range, char *__order, __CLPK_integer *__n,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__gers,
        __CLPK_doublereal *__reltol, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__e2,
        __CLPK_doublereal *__pivmin, __CLPK_integer *__nsplit,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_doublereal *__werr, __CLPK_doublereal *__wl,
        __CLPK_doublereal *__wu, __CLPK_integer *__iblock,
        __CLPK_integer *__indexw, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrd);

int __TEMPLATE_FUNC(dlarre)(char *__range, __CLPK_integer *__n, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__e2, __CLPK_doublereal *__rtol1,
        __CLPK_doublereal *__rtol2, __CLPK_doublereal *__spltol,
        __CLPK_integer *__nsplit, __CLPK_integer *__isplit, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__werr,
        __CLPK_doublereal *__wgap, __CLPK_integer *__iblock,
        __CLPK_integer *__indexw, __CLPK_doublereal *__gers,
        __CLPK_doublereal *__pivmin, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarre);

int __TEMPLATE_FUNC(dlarrf)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__l, __CLPK_doublereal *__ld,
        __CLPK_integer *__clstrt, __CLPK_integer *__clend,
        __CLPK_doublereal *__w, __CLPK_doublereal *__wgap,
        __CLPK_doublereal *__werr, __CLPK_doublereal *__spdiam,
        __CLPK_doublereal *__clgapl, __CLPK_doublereal *__clgapr,
        __CLPK_doublereal *__pivmin, __CLPK_doublereal *__sigma,
        __CLPK_doublereal *__dplus, __CLPK_doublereal *__lplus,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrf);

int __TEMPLATE_FUNC(dlarrj)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e2, __CLPK_integer *__ifirst,
        __CLPK_integer *__ilast, __CLPK_doublereal *__rtol,
        __CLPK_integer *__offset, __CLPK_doublereal *__w,
        __CLPK_doublereal *__werr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__spdiam,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrj);

int __TEMPLATE_FUNC(dlarrk)(__CLPK_integer *__n, __CLPK_integer *__iw, __CLPK_doublereal *__gl,
        __CLPK_doublereal *__gu, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e2, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__reltol, __CLPK_doublereal *__w,
        __CLPK_doublereal *__werr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrk);

int __TEMPLATE_FUNC(dlarrr)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrr);

int __TEMPLATE_FUNC(dlarrv)(__CLPK_integer *__n, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__l, __CLPK_doublereal *__pivmin,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_integer *__dol,
        __CLPK_integer *__dou, __CLPK_doublereal *__minrgp,
        __CLPK_doublereal *__rtol1, __CLPK_doublereal *__rtol2,
        __CLPK_doublereal *__w, __CLPK_doublereal *__werr,
        __CLPK_doublereal *__wgap, __CLPK_integer *__iblock,
        __CLPK_integer *__indexw, __CLPK_doublereal *__gers,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlarrv);

int __TEMPLATE_FUNC(dlarscl2)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(dlarscl2);

int __TEMPLATE_FUNC(dlartg)(__CLPK_doublereal *__f, __CLPK_doublereal *__g,
        __CLPK_doublereal *__cs, __CLPK_doublereal *__sn,
        __CLPK_doublereal *__r__)
__TEMPLATE_ALIAS(dlartg);

int __TEMPLATE_FUNC(dlartv)(__CLPK_integer *__n, __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__y, __CLPK_integer *__incy,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(dlartv);

int __TEMPLATE_FUNC(dlaruv)(__CLPK_integer *__iseed, __CLPK_integer *__n,
        __CLPK_doublereal *__x)
__TEMPLATE_ALIAS(dlaruv);

int __TEMPLATE_FUNC(dlarz)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__l, __CLPK_doublereal *__v, __CLPK_integer *__incv,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlarz);

int __TEMPLATE_FUNC(dlarzb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_doublereal *__t, __CLPK_integer *__ldt, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(dlarzb);

int __TEMPLATE_FUNC(dlarzt)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__v, __CLPK_integer *__ldv,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(dlarzt);

int __TEMPLATE_FUNC(dlas2)(__CLPK_doublereal *__f, __CLPK_doublereal *__g,
        __CLPK_doublereal *__h__, __CLPK_doublereal *__ssmin,
        __CLPK_doublereal *__ssmax)
__TEMPLATE_ALIAS(dlas2);

int __TEMPLATE_FUNC(dlascl)(char *__type__, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_doublereal *__cfrom, __CLPK_doublereal *__cto,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlascl);

int __TEMPLATE_FUNC(dlascl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(dlascl2);

int __TEMPLATE_FUNC(dlasd0)(__CLPK_integer *__n, __CLPK_integer *__sqre,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__vt,
        __CLPK_integer *__ldvt, __CLPK_integer *__smlsiz,
        __CLPK_integer *__iwork, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd0);

int __TEMPLATE_FUNC(dlasd1)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__alpha,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__u,
        __CLPK_integer *__ldu, __CLPK_doublereal *__vt, __CLPK_integer *__ldvt,
        __CLPK_integer *__idxq, __CLPK_integer *__iwork,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd1);

int __TEMPLATE_FUNC(dlasd2)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_integer *__k, __CLPK_doublereal *__d__, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublereal *__dsigma,
        __CLPK_doublereal *__u2, __CLPK_integer *__ldu2,
        __CLPK_doublereal *__vt2, __CLPK_integer *__ldvt2,
        __CLPK_integer *__idxp, __CLPK_integer *__idx, __CLPK_integer *__idxc,
        __CLPK_integer *__idxq, __CLPK_integer *__coltyp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd2);

int __TEMPLATE_FUNC(dlasd3)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_integer *__k, __CLPK_doublereal *__d__, __CLPK_doublereal *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__dsigma,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__u2,
        __CLPK_integer *__ldu2, __CLPK_doublereal *__vt, __CLPK_integer *__ldvt,
        __CLPK_doublereal *__vt2, __CLPK_integer *__ldvt2,
        __CLPK_integer *__idxc, __CLPK_integer *__ctot,
        __CLPK_doublereal *__z__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd3);

int __TEMPLATE_FUNC(dlasd4)(__CLPK_integer *__n, __CLPK_integer *__i__,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__delta, __CLPK_doublereal *__rho,
        __CLPK_doublereal *__sigma, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd4);

int __TEMPLATE_FUNC(dlasd5)(__CLPK_integer *__i__, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__z__, __CLPK_doublereal *__delta,
        __CLPK_doublereal *__rho, __CLPK_doublereal *__dsigma,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlasd5);

int __TEMPLATE_FUNC(dlasd6)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__vf, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_integer *__idxq, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_doublereal *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_doublereal *__poles,
        __CLPK_doublereal *__difl, __CLPK_doublereal *__difr,
        __CLPK_doublereal *__z__, __CLPK_integer *__k, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__s, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd6);

int __TEMPLATE_FUNC(dlasd7)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__k,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__zw, __CLPK_doublereal *__vf,
        __CLPK_doublereal *__vfw, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vlw, __CLPK_doublereal *__alpha,
        __CLPK_doublereal *__beta, __CLPK_doublereal *__dsigma,
        __CLPK_integer *__idx, __CLPK_integer *__idxp, __CLPK_integer *__idxq,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_doublereal *__givnum, __CLPK_integer *__ldgnum,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd7);

int __TEMPLATE_FUNC(dlasd8)(__CLPK_integer *__icompq, __CLPK_integer *__k,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__vf, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__difl, __CLPK_doublereal *__difr,
        __CLPK_integer *__lddifr, __CLPK_doublereal *__dsigma,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasd8);

int __TEMPLATE_FUNC(dlasda)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__sqre, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__u, __CLPK_integer *__ldu,
        __CLPK_doublereal *__vt, __CLPK_integer *__k, __CLPK_doublereal *__difl,
        __CLPK_doublereal *__difr, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__poles, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_integer *__perm, __CLPK_doublereal *__givnum,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasda);

int __TEMPLATE_FUNC(dlasdq)(char *__uplo, __CLPK_integer *__sqre, __CLPK_integer *__n,
        __CLPK_integer *__ncvt, __CLPK_integer *__nru, __CLPK_integer *__ncc,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vt, __CLPK_integer *__ldvt, __CLPK_doublereal *__u,
        __CLPK_integer *__ldu, __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasdq);

int __TEMPLATE_FUNC(dlasdt)(__CLPK_integer *__n, __CLPK_integer *__lvl, __CLPK_integer *__nd,
        __CLPK_integer *__inode, __CLPK_integer *__ndiml,
        __CLPK_integer *__ndimr,
        __CLPK_integer *__msub)
__TEMPLATE_ALIAS(dlasdt);

int __TEMPLATE_FUNC(dlaset)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(dlaset);

int __TEMPLATE_FUNC(dlasq1)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasq1);

int __TEMPLATE_FUNC(dlasq2)(__CLPK_integer *__n, __CLPK_doublereal *__z__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasq2);

int __TEMPLATE_FUNC(dlasq3)(__CLPK_integer *__i0, __CLPK_integer *__n0,
        __CLPK_doublereal *__z__, __CLPK_integer *__pp,
        __CLPK_doublereal *__dmin__, __CLPK_doublereal *__sigma,
        __CLPK_doublereal *__desig, __CLPK_doublereal *__qmax,
        __CLPK_integer *__nfail, __CLPK_integer *__iter, __CLPK_integer *__ndiv,
        __CLPK_logical *__ieee, __CLPK_integer *__ttype,
        __CLPK_doublereal *__dmin1, __CLPK_doublereal *__dmin2,
        __CLPK_doublereal *__dn, __CLPK_doublereal *__dn1,
        __CLPK_doublereal *__dn2, __CLPK_doublereal *__g,
        __CLPK_doublereal *__tau)
__TEMPLATE_ALIAS(dlasq3);

int __TEMPLATE_FUNC(dlasq4)(__CLPK_integer *__i0, __CLPK_integer *__n0,
        __CLPK_doublereal *__z__, __CLPK_integer *__pp, __CLPK_integer *__n0in,
        __CLPK_doublereal *__dmin__, __CLPK_doublereal *__dmin1,
        __CLPK_doublereal *__dmin2, __CLPK_doublereal *__dn,
        __CLPK_doublereal *__dn1, __CLPK_doublereal *__dn2,
        __CLPK_doublereal *__tau, __CLPK_integer *__ttype,
        __CLPK_doublereal *__g)
__TEMPLATE_ALIAS(dlasq4);

int __TEMPLATE_FUNC(dlasq5)(__CLPK_integer *__i0, __CLPK_integer *__n0,
        __CLPK_doublereal *__z__, __CLPK_integer *__pp,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__dmin__,
        __CLPK_doublereal *__dmin1, __CLPK_doublereal *__dmin2,
        __CLPK_doublereal *__dn, __CLPK_doublereal *__dnm1,
        __CLPK_doublereal *__dnm2,
        __CLPK_logical *__ieee)
__TEMPLATE_ALIAS(dlasq5);

int __TEMPLATE_FUNC(dlasq6)(__CLPK_integer *__i0, __CLPK_integer *__n0,
        __CLPK_doublereal *__z__, __CLPK_integer *__pp,
        __CLPK_doublereal *__dmin__, __CLPK_doublereal *__dmin1,
        __CLPK_doublereal *__dmin2, __CLPK_doublereal *__dn,
        __CLPK_doublereal *__dnm1,
        __CLPK_doublereal *__dnm2)
__TEMPLATE_ALIAS(dlasq6);

int __TEMPLATE_FUNC(dlasr)(char *__side, char *__pivot, char *__direct, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublereal *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(dlasr);

int __TEMPLATE_FUNC(dlasrt)(char *__id, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasrt);

int __TEMPLATE_FUNC(dlassq)(__CLPK_integer *__n, __CLPK_doublereal *__x, __CLPK_integer *__incx,
        __CLPK_doublereal *__scale,
        __CLPK_doublereal *__sumsq)
__TEMPLATE_ALIAS(dlassq);

int __TEMPLATE_FUNC(dlasv2)(__CLPK_doublereal *__f, __CLPK_doublereal *__g,
        __CLPK_doublereal *__h__, __CLPK_doublereal *__ssmin,
        __CLPK_doublereal *__ssmax, __CLPK_doublereal *__snr,
        __CLPK_doublereal *__csr, __CLPK_doublereal *__snl,
        __CLPK_doublereal *__csl)
__TEMPLATE_ALIAS(dlasv2);

int __TEMPLATE_FUNC(dlaswp)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__k1, __CLPK_integer *__k2, __CLPK_integer *__ipiv,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(dlaswp);

int __TEMPLATE_FUNC(dlasy2)(__CLPK_logical *__ltranl, __CLPK_logical *__ltranr,
        __CLPK_integer *__isgn, __CLPK_integer *__n1, __CLPK_integer *__n2,
        __CLPK_doublereal *__tl, __CLPK_integer *__ldtl,
        __CLPK_doublereal *__tr, __CLPK_integer *__ldtr, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__xnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasy2);

int __TEMPLATE_FUNC(dlasyf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__w, __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlasyf);

int __TEMPLATE_FUNC(dlat2s)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_real *__sa, __CLPK_integer *__ldsa,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlat2s);

int __TEMPLATE_FUNC(dlatbs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_integer *__kd, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__x,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlatbs);

int __TEMPLATE_FUNC(dlatdf)(__CLPK_integer *__ijob, __CLPK_integer *__n,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__rhs, __CLPK_doublereal *__rdsum,
        __CLPK_doublereal *__rdscal, __CLPK_integer *__ipiv,
        __CLPK_integer *__jpiv)
__TEMPLATE_ALIAS(dlatdf);

int __TEMPLATE_FUNC(dlatps)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_doublereal *__ap, __CLPK_doublereal *__x,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlatps);

int __TEMPLATE_FUNC(dlatrd)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__w,
        __CLPK_integer *__ldw)
__TEMPLATE_ALIAS(dlatrd);

int __TEMPLATE_FUNC(dlatrs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__x, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlatrs);

int __TEMPLATE_FUNC(dlatrz)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__l,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlatrz);

int __TEMPLATE_FUNC(dlatzm)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__v, __CLPK_integer *__incv,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__c1,
        __CLPK_doublereal *__c2, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(dlatzm);

int __TEMPLATE_FUNC(dlauu2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlauu2);

int __TEMPLATE_FUNC(dlauum)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dlauum);

int __TEMPLATE_FUNC(dopgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dopgtr);

int __TEMPLATE_FUNC(dopmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__ap, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dopmtr);

int __TEMPLATE_FUNC(dorg2l)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorg2l);

int __TEMPLATE_FUNC(dorg2r)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorg2r);

int __TEMPLATE_FUNC(dorgbr)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgbr);

int __TEMPLATE_FUNC(dorghr)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorghr);

int __TEMPLATE_FUNC(dorgl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgl2);

int __TEMPLATE_FUNC(dorglq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorglq);

int __TEMPLATE_FUNC(dorgql)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgql);

int __TEMPLATE_FUNC(dorgqr)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgqr);

int __TEMPLATE_FUNC(dorgr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgr2);

int __TEMPLATE_FUNC(dorgrq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgrq);

int __TEMPLATE_FUNC(dorgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorgtr);

int __TEMPLATE_FUNC(dorm2l)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorm2l);

int __TEMPLATE_FUNC(dorm2r)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorm2r);

int __TEMPLATE_FUNC(dormbr)(char *__vect, char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormbr);

int __TEMPLATE_FUNC(dormhr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormhr);

int __TEMPLATE_FUNC(dorml2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dorml2);

int __TEMPLATE_FUNC(dormlq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormlq);

int __TEMPLATE_FUNC(dormql)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormql);

int __TEMPLATE_FUNC(dormqr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormqr);

int __TEMPLATE_FUNC(dormr2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormr2);

int __TEMPLATE_FUNC(dormr3)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormr3);

int __TEMPLATE_FUNC(dormrq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormrq);

int __TEMPLATE_FUNC(dormrz)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormrz);

int __TEMPLATE_FUNC(dormtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dormtr);

int __TEMPLATE_FUNC(dpbcon)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbcon);

int __TEMPLATE_FUNC(dpbequ)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbequ);

int __TEMPLATE_FUNC(dpbrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__afb, __CLPK_integer *__ldafb,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbrfs);

int __TEMPLATE_FUNC(dpbstf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbstf);

int __TEMPLATE_FUNC(dpbsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbsv);

int __TEMPLATE_FUNC(dpbsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__afb,
        __CLPK_integer *__ldafb, char *__equed, __CLPK_doublereal *__s,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbsvx);

int __TEMPLATE_FUNC(dpbtf2)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbtf2);

int __TEMPLATE_FUNC(dpbtrf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbtrf);

int __TEMPLATE_FUNC(dpbtrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpbtrs);

int __TEMPLATE_FUNC(dpftrf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpftrf);

int __TEMPLATE_FUNC(dpftri)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpftri);

int __TEMPLATE_FUNC(dpftrs)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpftrs);

int __TEMPLATE_FUNC(dpocon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpocon);

int __TEMPLATE_FUNC(dpoequ)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpoequ);

int __TEMPLATE_FUNC(dpoequb)(__CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpoequb);

int __TEMPLATE_FUNC(dporfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__af,
        __CLPK_integer *__ldaf, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dporfs);

int __TEMPLATE_FUNC(dposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dposv);

int __TEMPLATE_FUNC(dposvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__af, __CLPK_integer *__ldaf, char *__equed,
        __CLPK_doublereal *__s, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dposvx);

int __TEMPLATE_FUNC(dpotf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpotf2);

int __TEMPLATE_FUNC(dpotrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpotrf);

int __TEMPLATE_FUNC(dpotri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpotri);

int __TEMPLATE_FUNC(dpotrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpotrs);

int __TEMPLATE_FUNC(dppcon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dppcon);

int __TEMPLATE_FUNC(dppequ)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dppequ);

int __TEMPLATE_FUNC(dpprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__afp,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpprfs);

int __TEMPLATE_FUNC(dppsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dppsv);

int __TEMPLATE_FUNC(dppsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__afp, char *__equed, __CLPK_doublereal *__s,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dppsvx);

int __TEMPLATE_FUNC(dpptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpptrf);

int __TEMPLATE_FUNC(dpptri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpptri);

int __TEMPLATE_FUNC(dpptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpptrs);

int __TEMPLATE_FUNC(dpstf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_doublereal *__tol, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpstf2);

int __TEMPLATE_FUNC(dpstrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_doublereal *__tol, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpstrf);

int __TEMPLATE_FUNC(dptcon)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dptcon);

int __TEMPLATE_FUNC(dpteqr)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpteqr);

int __TEMPLATE_FUNC(dptrfs)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__df, __CLPK_doublereal *__ef,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dptrfs);

int __TEMPLATE_FUNC(dptsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dptsv);

int __TEMPLATE_FUNC(dptsvx)(char *__fact, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__df, __CLPK_doublereal *__ef,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dptsvx);

int __TEMPLATE_FUNC(dpttrf)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpttrf);

int __TEMPLATE_FUNC(dpttrs)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dpttrs);

int __TEMPLATE_FUNC(dptts2)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(dptts2);

int __TEMPLATE_FUNC(drscl)(__CLPK_integer *__n, __CLPK_doublereal *__sa,
        __CLPK_doublereal *__sx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(drscl);

int __TEMPLATE_FUNC(dsbev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbev);

int __TEMPLATE_FUNC(dsbevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbevd);

int __TEMPLATE_FUNC(dsbevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbevx);

int __TEMPLATE_FUNC(dsbgst)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__bb, __CLPK_integer *__ldbb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbgst);

int __TEMPLATE_FUNC(dsbgv)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__bb, __CLPK_integer *__ldbb,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbgv);

int __TEMPLATE_FUNC(dsbgvd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__bb, __CLPK_integer *__ldbb,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbgvd);

int __TEMPLATE_FUNC(dsbgvx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__bb, __CLPK_integer *__ldbb,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbgvx);

int __TEMPLATE_FUNC(dsbtrd)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsbtrd);

int __TEMPLATE_FUNC(dsfrk)(char *__transr, char *__uplo, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__alpha, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__c__)
__TEMPLATE_ALIAS(dsfrk);

int __TEMPLATE_FUNC(dsgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__work, __CLPK_real *__swork, __CLPK_integer *__iter,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsgesv);

int __TEMPLATE_FUNC(dspcon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspcon);

int __TEMPLATE_FUNC(dspev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__w,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspev);

int __TEMPLATE_FUNC(dspevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__w,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspevd);

int __TEMPLATE_FUNC(dspevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspevx);

int __TEMPLATE_FUNC(dspgst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__bp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspgst);

int __TEMPLATE_FUNC(dspgv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__ap, __CLPK_doublereal *__bp,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspgv);

int __TEMPLATE_FUNC(dspgvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__ap, __CLPK_doublereal *__bp,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspgvd);

int __TEMPLATE_FUNC(dspgvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__ap, __CLPK_doublereal *__bp,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspgvx);

int __TEMPLATE_FUNC(dsposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__work, __CLPK_real *__swork, __CLPK_integer *__iter,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsposv);

int __TEMPLATE_FUNC(dsprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__afp,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsprfs);

int __TEMPLATE_FUNC(dspsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspsv);

int __TEMPLATE_FUNC(dspsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__afp, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dspsvx);

int __TEMPLATE_FUNC(dsptrd)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsptrd);

int __TEMPLATE_FUNC(dsptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsptrf);

int __TEMPLATE_FUNC(dsptri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsptri);

int __TEMPLATE_FUNC(dsptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__ap, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsptrs);

int __TEMPLATE_FUNC(dstebz)(char *__range, char *__order, __CLPK_integer *__n,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e, __CLPK_integer *__m,
        __CLPK_integer *__nsplit, __CLPK_doublereal *__w,
        __CLPK_integer *__iblock, __CLPK_integer *__isplit,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstebz);

int __TEMPLATE_FUNC(dstedc)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstedc);

int __TEMPLATE_FUNC(dstegr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstegr);

int __TEMPLATE_FUNC(dstein)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_integer *__iblock, __CLPK_integer *__isplit,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstein);

int __TEMPLATE_FUNC(dstemr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_doublereal *__z__, __CLPK_integer *__ldz, __CLPK_integer *__nzc,
        __CLPK_integer *__isuppz, __CLPK_logical *__tryrac,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstemr);

int __TEMPLATE_FUNC(dsteqr)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsteqr);

int __TEMPLATE_FUNC(dsterf)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsterf);

int __TEMPLATE_FUNC(dstev)(char *__jobz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstev);

int __TEMPLATE_FUNC(dstevd)(char *__jobz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstevd);

int __TEMPLATE_FUNC(dstevr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstevr);

int __TEMPLATE_FUNC(dstevx)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dstevx);

int __TEMPLATE_FUNC(dsycon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsycon);

int __TEMPLATE_FUNC(dsyequb)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyequb);

int __TEMPLATE_FUNC(dsyev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__w,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyev);

int __TEMPLATE_FUNC(dsyevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__w,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyevd);

int __TEMPLATE_FUNC(dsyevr)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyevr);

int __TEMPLATE_FUNC(dsyevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyevx);

int __TEMPLATE_FUNC(dsygs2)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsygs2);

int __TEMPLATE_FUNC(dsygst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsygst);

int __TEMPLATE_FUNC(dsygv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__w,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsygv);

int __TEMPLATE_FUNC(dsygvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__w,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsygvd);

int __TEMPLATE_FUNC(dsygvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsygvx);

int __TEMPLATE_FUNC(dsyrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsyrfs);

int __TEMPLATE_FUNC(dsysv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsysv);

int __TEMPLATE_FUNC(dsysvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsysvx);

int __TEMPLATE_FUNC(dsytd2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytd2);

int __TEMPLATE_FUNC(dsytf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytf2);

int __TEMPLATE_FUNC(dsytrd)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__tau, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytrd);

int __TEMPLATE_FUNC(dsytrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytrf);

int __TEMPLATE_FUNC(dsytri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytri);

int __TEMPLATE_FUNC(dsytrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dsytrs);

int __TEMPLATE_FUNC(dtbcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublereal *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtbcon);

int __TEMPLATE_FUNC(dtbrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtbrfs);

int __TEMPLATE_FUNC(dtbtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_doublereal *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtbtrs);

int __TEMPLATE_FUNC(dtfsm)(char *__transr, char *__side, char *__uplo, char *__trans,
        char *__diag, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__a,
        __CLPK_doublereal *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(dtfsm);

int __TEMPLATE_FUNC(dtftri)(char *__transr, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtftri);

int __TEMPLATE_FUNC(dtfttp)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__arf, __CLPK_doublereal *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtfttp);

int __TEMPLATE_FUNC(dtfttr)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__arf, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtfttr);

int __TEMPLATE_FUNC(dtgevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublereal *__s, __CLPK_integer *__lds,
        __CLPK_doublereal *__p, __CLPK_integer *__ldp, __CLPK_doublereal *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgevc);

int __TEMPLATE_FUNC(dtgex2)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__j1, __CLPK_integer *__n1, __CLPK_integer *__n2,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgex2);

int __TEMPLATE_FUNC(dtgexc)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgexc);

int __TEMPLATE_FUNC(dtgsen)(__CLPK_integer *__ijob, __CLPK_logical *__wantq,
        __CLPK_logical *__wantz, __CLPK_logical *__select, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__alphar,
        __CLPK_doublereal *__alphai, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__m, __CLPK_doublereal *__pl,
        __CLPK_doublereal *__pr, __CLPK_doublereal *__dif,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgsen);

int __TEMPLATE_FUNC(dtgsja)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__tola, __CLPK_doublereal *__tolb,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublereal *__u, __CLPK_integer *__ldu, __CLPK_doublereal *__v,
        __CLPK_integer *__ldv, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__work, __CLPK_integer *__ncycle,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgsja);

int __TEMPLATE_FUNC(dtgsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublereal *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__s, __CLPK_doublereal *__dif, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgsna);

int __TEMPLATE_FUNC(dtgsy2)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__d__, __CLPK_integer *__ldd,
        __CLPK_doublereal *__e, __CLPK_integer *__lde, __CLPK_doublereal *__f,
        __CLPK_integer *__ldf, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__rdsum, __CLPK_doublereal *__rdscal,
        __CLPK_integer *__iwork, __CLPK_integer *__pq,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgsy2);

int __TEMPLATE_FUNC(dtgsyl)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__d__, __CLPK_integer *__ldd,
        __CLPK_doublereal *__e, __CLPK_integer *__lde, __CLPK_doublereal *__f,
        __CLPK_integer *__ldf, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__dif, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtgsyl);

int __TEMPLATE_FUNC(dtpcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__rcond,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtpcon);

int __TEMPLATE_FUNC(dtprfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ap, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtprfs);

int __TEMPLATE_FUNC(dtptri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtptri);

int __TEMPLATE_FUNC(dtptrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__ap, __CLPK_doublereal *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtptrs);

int __TEMPLATE_FUNC(dtpttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__ap, __CLPK_doublereal *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtpttf);

int __TEMPLATE_FUNC(dtpttr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__ap,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtpttr);

int __TEMPLATE_FUNC(dtrcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrcon);

int __TEMPLATE_FUNC(dtrevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublereal *__t, __CLPK_integer *__ldt,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrevc);

int __TEMPLATE_FUNC(dtrexc)(char *__compq, __CLPK_integer *__n, __CLPK_doublereal *__t,
        __CLPK_integer *__ldt, __CLPK_doublereal *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrexc);

int __TEMPLATE_FUNC(dtrrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb, __CLPK_doublereal *__x,
        __CLPK_integer *__ldx, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrrfs);

int __TEMPLATE_FUNC(dtrsen)(char *__job, char *__compq, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublereal *__t, __CLPK_integer *__ldt,
        __CLPK_doublereal *__q, __CLPK_integer *__ldq, __CLPK_doublereal *__wr,
        __CLPK_doublereal *__wi, __CLPK_integer *__m, __CLPK_doublereal *__s,
        __CLPK_doublereal *__sep, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrsen);

int __TEMPLATE_FUNC(dtrsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublereal *__t, __CLPK_integer *__ldt,
        __CLPK_doublereal *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublereal *__vr, __CLPK_integer *__ldvr, __CLPK_doublereal *__s,
        __CLPK_doublereal *__sep, __CLPK_integer *__mm, __CLPK_integer *__m,
        __CLPK_doublereal *__work, __CLPK_integer *__ldwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrsna);

int __TEMPLATE_FUNC(dtrsyl)(char *__trana, char *__tranb, __CLPK_integer *__isgn,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrsyl);

int __TEMPLATE_FUNC(dtrti2)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrti2);

int __TEMPLATE_FUNC(dtrtri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrtri);

int __TEMPLATE_FUNC(dtrtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrtrs);

int __TEMPLATE_FUNC(dtrttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda, __CLPK_doublereal *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrttf);

int __TEMPLATE_FUNC(dtrttp)(char *__uplo, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtrttp);

int __TEMPLATE_FUNC(dtzrqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtzrqf);

int __TEMPLATE_FUNC(dtzrzf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__tau,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(dtzrzf);

__CLPK_doublereal __TEMPLATE_FUNC(dzsum1)(__CLPK_integer *__n, __CLPK_doublecomplex *__cx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(dzsum1);

__CLPK_integer __TEMPLATE_FUNC(icmax1)(__CLPK_integer *__n, __CLPK_complex *__cx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(icmax1);

__CLPK_integer __TEMPLATE_FUNC(ieeeck)(__CLPK_integer *__ispec, __CLPK_real *__zero,
        __CLPK_real *__one)
__TEMPLATE_ALIAS(ieeeck);

__CLPK_integer __TEMPLATE_FUNC(ilaclc)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilaclc);

__CLPK_integer __TEMPLATE_FUNC(ilaclr)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_complex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilaclr);

__CLPK_integer __TEMPLATE_FUNC(iladiag)(char *__diag)
__TEMPLATE_ALIAS(iladiag);

__CLPK_integer __TEMPLATE_FUNC(iladlc)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(iladlc);

__CLPK_integer __TEMPLATE_FUNC(iladlr)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(iladlr);

__CLPK_integer __TEMPLATE_FUNC(ilaenv)(__CLPK_integer *__ispec, char *__name__, char *__opts,
        __CLPK_integer *__n1, __CLPK_integer *__n2, __CLPK_integer *__n3,
        __CLPK_integer *__n4)
__TEMPLATE_ALIAS(ilaenv);

__CLPK_integer __TEMPLATE_FUNC(ilaprec)(char *__prec)
__TEMPLATE_ALIAS(ilaprec);

__CLPK_integer __TEMPLATE_FUNC(ilaslc)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilaslc);

__CLPK_integer __TEMPLATE_FUNC(ilaslr)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilaslr);

__CLPK_integer __TEMPLATE_FUNC(ilatrans)(char *__trans)
__TEMPLATE_ALIAS(ilatrans);

__CLPK_integer __TEMPLATE_FUNC(ilauplo)(char *__uplo)
__TEMPLATE_ALIAS(ilauplo);

int __TEMPLATE_FUNC(ilaver)(__CLPK_integer *__vers_major__, __CLPK_integer *__vers_minor__,
        __CLPK_integer *__vers_patch__)
__TEMPLATE_ALIAS(ilaver);

__CLPK_integer __TEMPLATE_FUNC(ilazlc)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilazlc);

__CLPK_integer __TEMPLATE_FUNC(ilazlr)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(ilazlr);

__CLPK_integer __TEMPLATE_FUNC(iparmq)(__CLPK_integer *__ispec, char *__name__, char *__opts,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(iparmq);

__CLPK_integer __TEMPLATE_FUNC(izmax1)(__CLPK_integer *__n, __CLPK_doublecomplex *__cx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(izmax1);

__CLPK_logical __TEMPLATE_FUNC(lsamen)(__CLPK_integer *__n, char *__ca,
        char *__cb)
__TEMPLATE_ALIAS(lsamen);

__CLPK_integer __TEMPLATE_FUNC(smaxloc)(__CLPK_real *__a,
        __CLPK_integer *__dimm)
__TEMPLATE_ALIAS(smaxloc);

int __TEMPLATE_FUNC(sbdsdc)(char *__uplo, char *__compq, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__u,
        __CLPK_integer *__ldu, __CLPK_real *__vt, __CLPK_integer *__ldvt,
        __CLPK_real *__q, __CLPK_integer *__iq, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sbdsdc);

int __TEMPLATE_FUNC(sbdsqr)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__ncvt,
        __CLPK_integer *__nru, __CLPK_integer *__ncc, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__vt, __CLPK_integer *__ldvt,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sbdsqr);

__CLPK_doublereal __TEMPLATE_FUNC(scsum1)(__CLPK_integer *__n, __CLPK_complex *__cx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(scsum1);

int __TEMPLATE_FUNC(sdisna)(char *__job, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__sep,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sdisna);

int __TEMPLATE_FUNC(sgbbrd)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__ncc, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_real *__ab, __CLPK_integer *__ldab, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_real *__pt, __CLPK_integer *__ldpt, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbbrd);

int __TEMPLATE_FUNC(sgbcon)(char *__norm, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbcon);

int __TEMPLATE_FUNC(sgbequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbequ);

int __TEMPLATE_FUNC(sgbequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbequb);

int __TEMPLATE_FUNC(sgbrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__afb, __CLPK_integer *__ldafb,
        __CLPK_integer *__ipiv, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbrfs);

int __TEMPLATE_FUNC(sgbsv)(__CLPK_integer *__n, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_integer *__nrhs, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbsv);

int __TEMPLATE_FUNC(sgbsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_real *__ab, __CLPK_integer *__ldab, __CLPK_real *__afb,
        __CLPK_integer *__ldafb, __CLPK_integer *__ipiv, char *__equed,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbsvx);

int __TEMPLATE_FUNC(sgbtf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbtf2);

int __TEMPLATE_FUNC(sgbtrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbtrf);

int __TEMPLATE_FUNC(sgbtrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgbtrs);

int __TEMPLATE_FUNC(sgebak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__scale,
        __CLPK_integer *__m, __CLPK_real *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgebak);

int __TEMPLATE_FUNC(sgebal)(char *__job, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgebal);

int __TEMPLATE_FUNC(sgebd2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__tauq, __CLPK_real *__taup, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgebd2);

int __TEMPLATE_FUNC(sgebrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__tauq, __CLPK_real *__taup, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgebrd);

int __TEMPLATE_FUNC(sgecon)(char *__norm, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgecon);

int __TEMPLATE_FUNC(sgeequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeequ);

int __TEMPLATE_FUNC(sgeequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeequb);

int __TEMPLATE_FUNC(sgees)(char *__jobvs, char *__sort, __CLPK_L_fp __select,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_real *__wr, __CLPK_real *__wi,
        __CLPK_real *__vs, __CLPK_integer *__ldvs, __CLPK_real *__work,
        __CLPK_integer *__lwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgees);

int __TEMPLATE_FUNC(sgeesx)(char *__jobvs, char *__sort, __CLPK_L_fp __select, char *__sense,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_real *__wr, __CLPK_real *__wi,
        __CLPK_real *__vs, __CLPK_integer *__ldvs, __CLPK_real *__rconde,
        __CLPK_real *__rcondv, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeesx);

int __TEMPLATE_FUNC(sgeev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__wr, __CLPK_real *__wi,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeev);

int __TEMPLATE_FUNC(sgeevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__wr, __CLPK_real *__wi, __CLPK_real *__vl,
        __CLPK_integer *__ldvl, __CLPK_real *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__scale,
        __CLPK_real *__abnrm, __CLPK_real *__rconde, __CLPK_real *__rcondv,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeevx);

int __TEMPLATE_FUNC(sgegs)(char *__jobvsl, char *__jobvsr, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__alphar, __CLPK_real *__alphai,
        __CLPK_real *__beta, __CLPK_real *__vsl, __CLPK_integer *__ldvsl,
        __CLPK_real *__vsr, __CLPK_integer *__ldvsr, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgegs);

int __TEMPLATE_FUNC(sgegv)(char *__jobvl, char *__jobvr, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__alphar, __CLPK_real *__alphai, __CLPK_real *__beta,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgegv);

int __TEMPLATE_FUNC(sgehd2)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgehd2);

int __TEMPLATE_FUNC(sgehrd)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgehrd);

int __TEMPLATE_FUNC(sgejsv)(char *__joba, char *__jobu, char *__jobv, char *__jobr,
        char *__jobt, char *__jobp, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__sva,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgejsv);

int __TEMPLATE_FUNC(sgelq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelq2);

int __TEMPLATE_FUNC(sgelqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelqf);

int __TEMPLATE_FUNC(sgels)(char *__trans, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgels);

int __TEMPLATE_FUNC(sgelsd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__s, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelsd);

int __TEMPLATE_FUNC(sgelss)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__s, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelss);

int __TEMPLATE_FUNC(sgelsx)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelsx);

int __TEMPLATE_FUNC(sgelsy)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_integer *__jpvt, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgelsy);

int __TEMPLATE_FUNC(sgeql2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeql2);

int __TEMPLATE_FUNC(sgeqlf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeqlf);

int __TEMPLATE_FUNC(sgeqp3)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeqp3);

int __TEMPLATE_FUNC(sgeqpf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeqpf);

int __TEMPLATE_FUNC(sgeqr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeqr2);

int __TEMPLATE_FUNC(sgeqrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgeqrf);

int __TEMPLATE_FUNC(sgerfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgerfs);

int __TEMPLATE_FUNC(sgerq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgerq2);

int __TEMPLATE_FUNC(sgerqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgerqf);

int __TEMPLATE_FUNC(sgesc2)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__rhs, __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_real *__scale)
__TEMPLATE_ALIAS(sgesc2);

int __TEMPLATE_FUNC(sgesdd)(char *__jobz, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__s,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__vt,
        __CLPK_integer *__ldvt, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgesdd);

int __TEMPLATE_FUNC(sgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgesv);

int __TEMPLATE_FUNC(sgesvd)(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__vt, __CLPK_integer *__ldvt, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgesvd);

int __TEMPLATE_FUNC(sgesvj)(char *__joba, char *__jobu, char *__jobv, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__sva, __CLPK_integer *__mv, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgesvj);

int __TEMPLATE_FUNC(sgesvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        char *__equed, __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgesvx);

int __TEMPLATE_FUNC(sgetc2)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgetc2);

int __TEMPLATE_FUNC(sgetf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgetf2);

int __TEMPLATE_FUNC(sgetrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgetrf);

int __TEMPLATE_FUNC(sgetri)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgetri);

int __TEMPLATE_FUNC(sgetrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgetrs);

int __TEMPLATE_FUNC(sggbak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__lscale,
        __CLPK_real *__rscale, __CLPK_integer *__m, __CLPK_real *__v,
        __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggbak);

int __TEMPLATE_FUNC(sggbal)(char *__job, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__lscale,
        __CLPK_real *__rscale, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggbal);

int __TEMPLATE_FUNC(sgges)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_integer *__sdim,
        __CLPK_real *__alphar, __CLPK_real *__alphai, __CLPK_real *__beta,
        __CLPK_real *__vsl, __CLPK_integer *__ldvsl, __CLPK_real *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgges);

int __TEMPLATE_FUNC(sggesx)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        char *__sense, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__sdim, __CLPK_real *__alphar, __CLPK_real *__alphai,
        __CLPK_real *__beta, __CLPK_real *__vsl, __CLPK_integer *__ldvsl,
        __CLPK_real *__vsr, __CLPK_integer *__ldvsr, __CLPK_real *__rconde,
        __CLPK_real *__rcondv, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggesx);

int __TEMPLATE_FUNC(sggev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__alphar, __CLPK_real *__alphai, __CLPK_real *__beta,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggev);

int __TEMPLATE_FUNC(sggevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__alphar,
        __CLPK_real *__alphai, __CLPK_real *__beta, __CLPK_real *__vl,
        __CLPK_integer *__ldvl, __CLPK_real *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__lscale,
        __CLPK_real *__rscale, __CLPK_real *__abnrm, __CLPK_real *__bbnrm,
        __CLPK_real *__rconde, __CLPK_real *__rcondv, __CLPK_real *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggevx);

int __TEMPLATE_FUNC(sggglm)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__d__, __CLPK_real *__x,
        __CLPK_real *__y, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggglm);

int __TEMPLATE_FUNC(sgghrd)(char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__z__,
        __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgghrd);

int __TEMPLATE_FUNC(sgglse)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__p,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__c__, __CLPK_real *__d__,
        __CLPK_real *__x, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgglse);

int __TEMPLATE_FUNC(sggqrf)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__taua,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__taub,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggqrf);

int __TEMPLATE_FUNC(sggrqf)(__CLPK_integer *__m, __CLPK_integer *__p, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__taua,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__taub,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggrqf);

int __TEMPLATE_FUNC(sggsvd)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__p, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__alpha,
        __CLPK_real *__beta, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__v, __CLPK_integer *__ldv, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggsvd);

int __TEMPLATE_FUNC(sggsvp)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__tola, __CLPK_real *__tolb, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__v, __CLPK_integer *__ldv, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_integer *__iwork, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sggsvp);

int __TEMPLATE_FUNC(sgsvj0)(char *__jobv, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__d__,
        __CLPK_real *__sva, __CLPK_integer *__mv, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_real *__eps, __CLPK_real *__sfmin,
        __CLPK_real *__tol, __CLPK_integer *__nsweep, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgsvj0);

int __TEMPLATE_FUNC(sgsvj1)(char *__jobv, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__n1, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__d__, __CLPK_real *__sva, __CLPK_integer *__mv,
        __CLPK_real *__v, __CLPK_integer *__ldv, __CLPK_real *__eps,
        __CLPK_real *__sfmin, __CLPK_real *__tol, __CLPK_integer *__nsweep,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgsvj1);

int __TEMPLATE_FUNC(sgtcon)(char *__norm, __CLPK_integer *__n, __CLPK_real *__dl,
        __CLPK_real *__d__, __CLPK_real *__du, __CLPK_real *__du2,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgtcon);

int __TEMPLATE_FUNC(sgtrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__dl, __CLPK_real *__d__, __CLPK_real *__du,
        __CLPK_real *__dlf, __CLPK_real *__df, __CLPK_real *__duf,
        __CLPK_real *__du2, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgtrfs);

int __TEMPLATE_FUNC(sgtsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__dl,
        __CLPK_real *__d__, __CLPK_real *__du, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgtsv);

int __TEMPLATE_FUNC(sgtsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__dl, __CLPK_real *__d__,
        __CLPK_real *__du, __CLPK_real *__dlf, __CLPK_real *__df,
        __CLPK_real *__duf, __CLPK_real *__du2, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgtsvx);

int __TEMPLATE_FUNC(sgttrf)(__CLPK_integer *__n, __CLPK_real *__dl, __CLPK_real *__d__,
        __CLPK_real *__du, __CLPK_real *__du2, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgttrf);

int __TEMPLATE_FUNC(sgttrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__dl, __CLPK_real *__d__, __CLPK_real *__du,
        __CLPK_real *__du2, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sgttrs);

int __TEMPLATE_FUNC(sgtts2)(__CLPK_integer *__itrans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__dl, __CLPK_real *__d__,
        __CLPK_real *__du, __CLPK_real *__du2, __CLPK_integer *__ipiv,
        __CLPK_real *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(sgtts2);

int __TEMPLATE_FUNC(shgeqz)(char *__job, char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__h__,
        __CLPK_integer *__ldh, __CLPK_real *__t, __CLPK_integer *__ldt,
        __CLPK_real *__alphar, __CLPK_real *__alphai, __CLPK_real *__beta,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(shgeqz);

int __TEMPLATE_FUNC(shsein)(char *__side, char *__eigsrc, char *__initv,
        __CLPK_logical *__select, __CLPK_integer *__n, __CLPK_real *__h__,
        __CLPK_integer *__ldh, __CLPK_real *__wr, __CLPK_real *__wi,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_integer *__mm, __CLPK_integer *__m,
        __CLPK_real *__work, __CLPK_integer *__ifaill, __CLPK_integer *__ifailr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(shsein);

int __TEMPLATE_FUNC(shseqr)(char *__job, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_real *__h__,
        __CLPK_integer *__ldh, __CLPK_real *__wr, __CLPK_real *__wi,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(shseqr);


        __CLPK_logical __TEMPLATE_FUNC(sisnan)(__CLPK_real *__sin__)
__TEMPLATE_ALIAS(sisnan);

int __TEMPLATE_FUNC(slabad)(__CLPK_real *__small,
        __CLPK_real *__large)
__TEMPLATE_ALIAS(slabad);

int __TEMPLATE_FUNC(slabrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__tauq, __CLPK_real *__taup,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(slabrd);

int __TEMPLATE_FUNC(slacn2)(__CLPK_integer *__n, __CLPK_real *__v, __CLPK_real *__x,
        __CLPK_integer *__isgn, __CLPK_real *__est, __CLPK_integer *__kase,
        __CLPK_integer *__isave)
__TEMPLATE_ALIAS(slacn2);

int __TEMPLATE_FUNC(slacon)(__CLPK_integer *__n, __CLPK_real *__v, __CLPK_real *__x,
        __CLPK_integer *__isgn, __CLPK_real *__est,
        __CLPK_integer *__kase)
__TEMPLATE_ALIAS(slacon);

int __TEMPLATE_FUNC(slacpy)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(slacpy);

int __TEMPLATE_FUNC(sladiv)(__CLPK_real *__a, __CLPK_real *__b, __CLPK_real *__c__,
        __CLPK_real *__d__, __CLPK_real *__p,
        __CLPK_real *__q)
__TEMPLATE_ALIAS(sladiv);

int __TEMPLATE_FUNC(slae2)(__CLPK_real *__a, __CLPK_real *__b, __CLPK_real *__c__,
        __CLPK_real *__rt1,
        __CLPK_real *__rt2)
__TEMPLATE_ALIAS(slae2);

int __TEMPLATE_FUNC(slaebz)(__CLPK_integer *__ijob, __CLPK_integer *__nitmax,
        __CLPK_integer *__n, __CLPK_integer *__mmax, __CLPK_integer *__minp,
        __CLPK_integer *__nbmin, __CLPK_real *__abstol, __CLPK_real *__reltol,
        __CLPK_real *__pivmin, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__e2, __CLPK_integer *__nval, __CLPK_real *__ab,
        __CLPK_real *__c__, __CLPK_integer *__mout, __CLPK_integer *__nab,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaebz);

int __TEMPLATE_FUNC(slaed0)(__CLPK_integer *__icompq, __CLPK_integer *__qsiz,
        __CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__qstore,
        __CLPK_integer *__ldqs, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed0);

int __TEMPLATE_FUNC(slaed1)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_integer *__indxq, __CLPK_real *__rho,
        __CLPK_integer *__cutpnt, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed1);

int __TEMPLATE_FUNC(slaed2)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__n1,
        __CLPK_real *__d__, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__indxq, __CLPK_real *__rho, __CLPK_real *__z__,
        __CLPK_real *__dlamda, __CLPK_real *__w, __CLPK_real *__q2,
        __CLPK_integer *__indx, __CLPK_integer *__indxc,
        __CLPK_integer *__indxp, __CLPK_integer *__coltyp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed2);

int __TEMPLATE_FUNC(slaed3)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__n1,
        __CLPK_real *__d__, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_real *__rho, __CLPK_real *__dlamda, __CLPK_real *__q2,
        __CLPK_integer *__indx, __CLPK_integer *__ctot, __CLPK_real *__w,
        __CLPK_real *__s,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed3);

int __TEMPLATE_FUNC(slaed4)(__CLPK_integer *__n, __CLPK_integer *__i__, __CLPK_real *__d__,
        __CLPK_real *__z__, __CLPK_real *__delta, __CLPK_real *__rho,
        __CLPK_real *__dlam,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed4);

int __TEMPLATE_FUNC(slaed5)(__CLPK_integer *__i__, __CLPK_real *__d__, __CLPK_real *__z__,
        __CLPK_real *__delta, __CLPK_real *__rho,
        __CLPK_real *__dlam)
__TEMPLATE_ALIAS(slaed5);

int __TEMPLATE_FUNC(slaed6)(__CLPK_integer *__kniter, __CLPK_logical *__orgati,
        __CLPK_real *__rho, __CLPK_real *__d__, __CLPK_real *__z__,
        __CLPK_real *__finit, __CLPK_real *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed6);

int __TEMPLATE_FUNC(slaed7)(__CLPK_integer *__icompq, __CLPK_integer *__n,
        __CLPK_integer *__qsiz, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm, __CLPK_real *__d__,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_integer *__indxq,
        __CLPK_real *__rho, __CLPK_integer *__cutpnt, __CLPK_real *__qstore,
        __CLPK_integer *__qptr, __CLPK_integer *__prmptr,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_real *__givnum, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed7);

int __TEMPLATE_FUNC(slaed8)(__CLPK_integer *__icompq, __CLPK_integer *__k, __CLPK_integer *__n,
        __CLPK_integer *__qsiz, __CLPK_real *__d__, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_integer *__indxq, __CLPK_real *__rho,
        __CLPK_integer *__cutpnt, __CLPK_real *__z__, __CLPK_real *__dlamda,
        __CLPK_real *__q2, __CLPK_integer *__ldq2, __CLPK_real *__w,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_real *__givnum,
        __CLPK_integer *__indxp, __CLPK_integer *__indx,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed8);

int __TEMPLATE_FUNC(slaed9)(__CLPK_integer *__k, __CLPK_integer *__kstart,
        __CLPK_integer *__kstop, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__rho,
        __CLPK_real *__dlamda, __CLPK_real *__w, __CLPK_real *__s,
        __CLPK_integer *__lds,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaed9);

int __TEMPLATE_FUNC(slaeda)(__CLPK_integer *__n, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm,
        __CLPK_integer *__prmptr, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_real *__givnum, __CLPK_real *__q, __CLPK_integer *__qptr,
        __CLPK_real *__z__, __CLPK_real *__ztemp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaeda);

int __TEMPLATE_FUNC(slaein)(__CLPK_logical *__rightv, __CLPK_logical *__noinit,
        __CLPK_integer *__n, __CLPK_real *__h__, __CLPK_integer *__ldh,
        __CLPK_real *__wr, __CLPK_real *__wi, __CLPK_real *__vr,
        __CLPK_real *__vi, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__work, __CLPK_real *__eps3, __CLPK_real *__smlnum,
        __CLPK_real *__bignum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaein);

int __TEMPLATE_FUNC(slaev2)(__CLPK_real *__a, __CLPK_real *__b, __CLPK_real *__c__,
        __CLPK_real *__rt1, __CLPK_real *__rt2, __CLPK_real *__cs1,
        __CLPK_real *__sn1)
__TEMPLATE_ALIAS(slaev2);

int __TEMPLATE_FUNC(slaexc)(__CLPK_logical *__wantq, __CLPK_integer *__n, __CLPK_real *__t,
        __CLPK_integer *__ldt, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__j1, __CLPK_integer *__n1, __CLPK_integer *__n2,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaexc);

int __TEMPLATE_FUNC(slag2)(__CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__safmin, __CLPK_real *__scale1,
        __CLPK_real *__scale2, __CLPK_real *__wr1, __CLPK_real *__wr2,
        __CLPK_real *__wi)
__TEMPLATE_ALIAS(slag2);

int __TEMPLATE_FUNC(slag2d)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__sa,
        __CLPK_integer *__ldsa, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slag2d);

int __TEMPLATE_FUNC(slags2)(__CLPK_logical *__upper, __CLPK_real *__a1, __CLPK_real *__a2,
        __CLPK_real *__a3, __CLPK_real *__b1, __CLPK_real *__b2,
        __CLPK_real *__b3, __CLPK_real *__csu, __CLPK_real *__snu,
        __CLPK_real *__csv, __CLPK_real *__snv, __CLPK_real *__csq,
        __CLPK_real *__snq)
__TEMPLATE_ALIAS(slags2);

int __TEMPLATE_FUNC(slagtf)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_real *__lambda,
        __CLPK_real *__b, __CLPK_real *__c__, __CLPK_real *__tol,
        __CLPK_real *__d__, __CLPK_integer *__in,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slagtf);

int __TEMPLATE_FUNC(slagtm)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__alpha, __CLPK_real *__dl, __CLPK_real *__d__,
        __CLPK_real *__du, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__beta, __CLPK_real *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(slagtm);

int __TEMPLATE_FUNC(slagts)(__CLPK_integer *__job, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_real *__b, __CLPK_real *__c__, __CLPK_real *__d__,
        __CLPK_integer *__in, __CLPK_real *__y, __CLPK_real *__tol,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slagts);

int __TEMPLATE_FUNC(slagv2)(__CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__alphar, __CLPK_real *__alphai,
        __CLPK_real *__beta, __CLPK_real *__csl, __CLPK_real *__snl,
        __CLPK_real *__csr,
        __CLPK_real *__snr)
__TEMPLATE_ALIAS(slagv2);

int __TEMPLATE_FUNC(slahqr)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__h__, __CLPK_integer *__ldh, __CLPK_real *__wr,
        __CLPK_real *__wi, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slahqr);

int __TEMPLATE_FUNC(slahr2)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__t, __CLPK_integer *__ldt, __CLPK_real *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(slahr2);

int __TEMPLATE_FUNC(slahrd)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__t, __CLPK_integer *__ldt, __CLPK_real *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(slahrd);

int __TEMPLATE_FUNC(slaic1)(__CLPK_integer *__job, __CLPK_integer *__j, __CLPK_real *__x,
        __CLPK_real *__sest, __CLPK_real *__w, __CLPK_real *__gamma,
        __CLPK_real *__sestpr, __CLPK_real *__s,
        __CLPK_real *__c__)
__TEMPLATE_ALIAS(slaic1);

__CLPK_logical __TEMPLATE_FUNC(slaisnan)(__CLPK_real *__sin1,
        __CLPK_real *__sin2)
__TEMPLATE_ALIAS(slaisnan);

int __TEMPLATE_FUNC(slaln2)(__CLPK_logical *__ltrans, __CLPK_integer *__na,
        __CLPK_integer *__nw, __CLPK_real *__smin, __CLPK_real *__ca,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__d1,
        __CLPK_real *__d2, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__wr, __CLPK_real *__wi, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__scale, __CLPK_real *__xnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaln2);

int __TEMPLATE_FUNC(slals0)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__nrhs,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__bx,
        __CLPK_integer *__ldbx, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_real *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_real *__poles, __CLPK_real *__difl,
        __CLPK_real *__difr, __CLPK_real *__z__, __CLPK_integer *__k,
        __CLPK_real *__c__, __CLPK_real *__s, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slals0);

int __TEMPLATE_FUNC(slalsa)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__bx, __CLPK_integer *__ldbx,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__vt,
        __CLPK_integer *__k, __CLPK_real *__difl, __CLPK_real *__difr,
        __CLPK_real *__z__, __CLPK_real *__poles, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_integer *__perm, __CLPK_real *__givnum, __CLPK_real *__c__,
        __CLPK_real *__s, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slalsa);

int __TEMPLATE_FUNC(slalsd)(char *__uplo, __CLPK_integer *__smlsiz, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__rcond,
        __CLPK_integer *__rank, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slalsd);

int __TEMPLATE_FUNC(slamrg)(__CLPK_integer *__n1, __CLPK_integer *__n2, __CLPK_real *__a,
        __CLPK_integer *__strd1, __CLPK_integer *__strd2,
        __CLPK_integer *__index)
__TEMPLATE_ALIAS(slamrg);

__CLPK_integer __TEMPLATE_FUNC(slaneg)(__CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__lld, __CLPK_real *__sigma, __CLPK_real *__pivmin,
        __CLPK_integer *__r__)
__TEMPLATE_ALIAS(slaneg);

__CLPK_doublereal __TEMPLATE_FUNC(slangb)(char *__norm, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_real *__ab,
        __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slangb);

__CLPK_doublereal __TEMPLATE_FUNC(slange)(char *__norm, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slange);

__CLPK_doublereal __TEMPLATE_FUNC(slangt)(char *__norm, __CLPK_integer *__n, __CLPK_real *__dl,
        __CLPK_real *__d__,
        __CLPK_real *__du)
__TEMPLATE_ALIAS(slangt);

__CLPK_doublereal __TEMPLATE_FUNC(slanhs)(char *__norm, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slanhs);

__CLPK_doublereal __TEMPLATE_FUNC(slansb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slansb);

__CLPK_doublereal __TEMPLATE_FUNC(slansf)(char *__norm, char *__transr, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slansf);

__CLPK_doublereal __TEMPLATE_FUNC(slansp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__ap,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slansp);

__CLPK_doublereal __TEMPLATE_FUNC(slanst)(char *__norm, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e)
__TEMPLATE_ALIAS(slanst);

__CLPK_doublereal __TEMPLATE_FUNC(slansy)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slansy);

__CLPK_doublereal __TEMPLATE_FUNC(slantb)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__ab,
        __CLPK_integer *__ldab,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slantb);

__CLPK_doublereal __TEMPLATE_FUNC(slantp)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slantp);

__CLPK_doublereal __TEMPLATE_FUNC(slantr)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slantr);

int __TEMPLATE_FUNC(slanv2)(__CLPK_real *__a, __CLPK_real *__b, __CLPK_real *__c__,
        __CLPK_real *__d__, __CLPK_real *__rt1r, __CLPK_real *__rt1i,
        __CLPK_real *__rt2r, __CLPK_real *__rt2i, __CLPK_real *__cs,
        __CLPK_real *__sn)
__TEMPLATE_ALIAS(slanv2);

int __TEMPLATE_FUNC(slapll)(__CLPK_integer *__n, __CLPK_real *__x, __CLPK_integer *__incx,
        __CLPK_real *__y, __CLPK_integer *__incy,
        __CLPK_real *__ssmin)
__TEMPLATE_ALIAS(slapll);

int __TEMPLATE_FUNC(slapmt)(__CLPK_logical *__forwrd, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_integer *__k)
__TEMPLATE_ALIAS(slapmt);

__CLPK_doublereal __TEMPLATE_FUNC(slapy2)(__CLPK_real *__x,
        __CLPK_real *__y)
__TEMPLATE_ALIAS(slapy2);

__CLPK_doublereal __TEMPLATE_FUNC(slapy3)(__CLPK_real *__x, __CLPK_real *__y,
        __CLPK_real *__z__)
__TEMPLATE_ALIAS(slapy3);

int __TEMPLATE_FUNC(slaqgb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__r__, __CLPK_real *__c__, __CLPK_real *__rowcnd,
        __CLPK_real *__colcnd, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(slaqgb);

int __TEMPLATE_FUNC(slaqge)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__r__, __CLPK_real *__c__,
        __CLPK_real *__rowcnd, __CLPK_real *__colcnd, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(slaqge);

int __TEMPLATE_FUNC(slaqp2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_real *__tau, __CLPK_real *__vn1, __CLPK_real *__vn2,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slaqp2);

int __TEMPLATE_FUNC(slaqps)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_integer *__nb, __CLPK_integer *__kb, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt, __CLPK_real *__tau,
        __CLPK_real *__vn1, __CLPK_real *__vn2, __CLPK_real *__auxv,
        __CLPK_real *__f,
        __CLPK_integer *__ldf)
__TEMPLATE_ALIAS(slaqps);

int __TEMPLATE_FUNC(slaqr0)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__h__, __CLPK_integer *__ldh, __CLPK_real *__wr,
        __CLPK_real *__wi, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaqr0);

int __TEMPLATE_FUNC(slaqr1)(__CLPK_integer *__n, __CLPK_real *__h__, __CLPK_integer *__ldh,
        __CLPK_real *__sr1, __CLPK_real *__si1, __CLPK_real *__sr2,
        __CLPK_real *__si2,
        __CLPK_real *__v)
__TEMPLATE_ALIAS(slaqr1);

int __TEMPLATE_FUNC(slaqr2)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_real *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__ns, __CLPK_integer *__nd,
        __CLPK_real *__sr, __CLPK_real *__si, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_integer *__nh, __CLPK_real *__t,
        __CLPK_integer *__ldt, __CLPK_integer *__nv, __CLPK_real *__wv,
        __CLPK_integer *__ldwv, __CLPK_real *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(slaqr2);

int __TEMPLATE_FUNC(slaqr3)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_real *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__ns, __CLPK_integer *__nd,
        __CLPK_real *__sr, __CLPK_real *__si, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_integer *__nh, __CLPK_real *__t,
        __CLPK_integer *__ldt, __CLPK_integer *__nv, __CLPK_real *__wv,
        __CLPK_integer *__ldwv, __CLPK_real *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(slaqr3);

int __TEMPLATE_FUNC(slaqr4)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__h__, __CLPK_integer *__ldh, __CLPK_real *__wr,
        __CLPK_real *__wi, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaqr4);

int __TEMPLATE_FUNC(slaqr5)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__kacc22, __CLPK_integer *__n, __CLPK_integer *__ktop,
        __CLPK_integer *__kbot, __CLPK_integer *__nshfts, __CLPK_real *__sr,
        __CLPK_real *__si, __CLPK_real *__h__, __CLPK_integer *__ldh,
        __CLPK_integer *__iloz, __CLPK_integer *__ihiz, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_real *__v, __CLPK_integer *__ldv,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_integer *__nv,
        __CLPK_real *__wv, __CLPK_integer *__ldwv, __CLPK_integer *__nh,
        __CLPK_real *__wh,
        __CLPK_integer *__ldwh)
__TEMPLATE_ALIAS(slaqr5);

int __TEMPLATE_FUNC(slaqsb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab, __CLPK_real *__s,
        __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(slaqsb);

int __TEMPLATE_FUNC(slaqsp)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        char *__equed)
__TEMPLATE_ALIAS(slaqsp);

int __TEMPLATE_FUNC(slaqsy)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, char *__equed)
__TEMPLATE_ALIAS(slaqsy);

int __TEMPLATE_FUNC(slaqtr)(__CLPK_logical *__ltran, __CLPK_logical *__l__CLPK_real,
        __CLPK_integer *__n, __CLPK_real *__t, __CLPK_integer *__ldt,
        __CLPK_real *__b, __CLPK_real *__w, __CLPK_real *__scale,
        __CLPK_real *__x, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slaqtr);

int __TEMPLATE_FUNC(slar1v)(__CLPK_integer *__n, __CLPK_integer *__b1, __CLPK_integer *__bn,
        __CLPK_real *__lambda, __CLPK_real *__d__, __CLPK_real *__l,
        __CLPK_real *__ld, __CLPK_real *__lld, __CLPK_real *__pivmin,
        __CLPK_real *__gaptol, __CLPK_real *__z__, __CLPK_logical *__wantnc,
        __CLPK_integer *__negcnt, __CLPK_real *__ztz, __CLPK_real *__mingma,
        __CLPK_integer *__r__, __CLPK_integer *__isuppz, __CLPK_real *__nrminv,
        __CLPK_real *__resid, __CLPK_real *__rqcorr,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slar1v);

int __TEMPLATE_FUNC(slar2v)(__CLPK_integer *__n, __CLPK_real *__x, __CLPK_real *__y,
        __CLPK_real *__z__, __CLPK_integer *__incx, __CLPK_real *__c__,
        __CLPK_real *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(slar2v);

int __TEMPLATE_FUNC(slarf)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__v, __CLPK_integer *__incv, __CLPK_real *__tau,
        __CLPK_real *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slarf);

int __TEMPLATE_FUNC(slarfb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__v, __CLPK_integer *__ldv, __CLPK_real *__t,
        __CLPK_integer *__ldt, __CLPK_real *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(slarfb);

int __TEMPLATE_FUNC(slarfg)(__CLPK_integer *__n, __CLPK_real *__alpha, __CLPK_real *__x,
        __CLPK_integer *__incx,
        __CLPK_real *__tau)
__TEMPLATE_ALIAS(slarfg);

int __TEMPLATE_FUNC(slarfp)(__CLPK_integer *__n, __CLPK_real *__alpha, __CLPK_real *__x,
        __CLPK_integer *__incx,
        __CLPK_real *__tau)
__TEMPLATE_ALIAS(slarfp);

int __TEMPLATE_FUNC(slarft)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__v, __CLPK_integer *__ldv,
        __CLPK_real *__tau, __CLPK_real *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(slarft);

int __TEMPLATE_FUNC(slarfx)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__v, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slarfx);

int __TEMPLATE_FUNC(slargv)(__CLPK_integer *__n, __CLPK_real *__x, __CLPK_integer *__incx,
        __CLPK_real *__y, __CLPK_integer *__incy, __CLPK_real *__c__,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(slargv);

int __TEMPLATE_FUNC(slarnv)(__CLPK_integer *__idist, __CLPK_integer *__iseed,
        __CLPK_integer *__n,
        __CLPK_real *__x)
__TEMPLATE_ALIAS(slarnv);

int __TEMPLATE_FUNC(slarra)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__e2, __CLPK_real *__spltol, __CLPK_real *__tnrm,
        __CLPK_integer *__nsplit, __CLPK_integer *__isplit,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarra);

int __TEMPLATE_FUNC(slarrb)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__lld,
        __CLPK_integer *__ifirst, __CLPK_integer *__ilast, __CLPK_real *__rtol1,
        __CLPK_real *__rtol2, __CLPK_integer *__offset, __CLPK_real *__w,
        __CLPK_real *__wgap, __CLPK_real *__werr, __CLPK_real *__work,
        __CLPK_integer *__iwork, __CLPK_real *__pivmin, __CLPK_real *__spdiam,
        __CLPK_integer *__twist,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrb);

int __TEMPLATE_FUNC(slarrc)(char *__jobt, __CLPK_integer *__n, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__pivmin, __CLPK_integer *__eigcnt, __CLPK_integer *__lcnt,
        __CLPK_integer *__rcnt,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrc);

int __TEMPLATE_FUNC(slarrd)(char *__range, char *__order, __CLPK_integer *__n,
        __CLPK_real *__vl, __CLPK_real *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_real *__gers, __CLPK_real *__reltol,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__e2,
        __CLPK_real *__pivmin, __CLPK_integer *__nsplit,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__werr, __CLPK_real *__wl, __CLPK_real *__wu,
        __CLPK_integer *__iblock, __CLPK_integer *__indexw, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrd);

int __TEMPLATE_FUNC(slarre)(char *__range, __CLPK_integer *__n, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__e2,
        __CLPK_real *__rtol1, __CLPK_real *__rtol2, __CLPK_real *__spltol,
        __CLPK_integer *__nsplit, __CLPK_integer *__isplit, __CLPK_integer *__m,
        __CLPK_real *__w, __CLPK_real *__werr, __CLPK_real *__wgap,
        __CLPK_integer *__iblock, __CLPK_integer *__indexw, __CLPK_real *__gers,
        __CLPK_real *__pivmin, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarre);

int __TEMPLATE_FUNC(slarrf)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__l,
        __CLPK_real *__ld, __CLPK_integer *__clstrt, __CLPK_integer *__clend,
        __CLPK_real *__w, __CLPK_real *__wgap, __CLPK_real *__werr,
        __CLPK_real *__spdiam, __CLPK_real *__clgapl, __CLPK_real *__clgapr,
        __CLPK_real *__pivmin, __CLPK_real *__sigma, __CLPK_real *__dplus,
        __CLPK_real *__lplus, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrf);

int __TEMPLATE_FUNC(slarrj)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e2,
        __CLPK_integer *__ifirst, __CLPK_integer *__ilast, __CLPK_real *__rtol,
        __CLPK_integer *__offset, __CLPK_real *__w, __CLPK_real *__werr,
        __CLPK_real *__work, __CLPK_integer *__iwork, __CLPK_real *__pivmin,
        __CLPK_real *__spdiam,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrj);

int __TEMPLATE_FUNC(slarrk)(__CLPK_integer *__n, __CLPK_integer *__iw, __CLPK_real *__gl,
        __CLPK_real *__gu, __CLPK_real *__d__, __CLPK_real *__e2,
        __CLPK_real *__pivmin, __CLPK_real *__reltol, __CLPK_real *__w,
        __CLPK_real *__werr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrk);

int __TEMPLATE_FUNC(slarrr)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrr);

int __TEMPLATE_FUNC(slarrv)(__CLPK_integer *__n, __CLPK_real *__vl, __CLPK_real *__vu,
        __CLPK_real *__d__, __CLPK_real *__l, __CLPK_real *__pivmin,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_integer *__dol,
        __CLPK_integer *__dou, __CLPK_real *__minrgp, __CLPK_real *__rtol1,
        __CLPK_real *__rtol2, __CLPK_real *__w, __CLPK_real *__werr,
        __CLPK_real *__wgap, __CLPK_integer *__iblock, __CLPK_integer *__indexw,
        __CLPK_real *__gers, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slarrv);

int __TEMPLATE_FUNC(slarscl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(slarscl2);

int __TEMPLATE_FUNC(slartg)(__CLPK_real *__f, __CLPK_real *__g, __CLPK_real *__cs,
        __CLPK_real *__sn,
        __CLPK_real *__r__)
__TEMPLATE_ALIAS(slartg);

int __TEMPLATE_FUNC(slartv)(__CLPK_integer *__n, __CLPK_real *__x, __CLPK_integer *__incx,
        __CLPK_real *__y, __CLPK_integer *__incy, __CLPK_real *__c__,
        __CLPK_real *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(slartv);

int __TEMPLATE_FUNC(slaruv)(__CLPK_integer *__iseed, __CLPK_integer *__n,
        __CLPK_real *__x)
__TEMPLATE_ALIAS(slaruv);

int __TEMPLATE_FUNC(slarz)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__l, __CLPK_real *__v, __CLPK_integer *__incv,
        __CLPK_real *__tau, __CLPK_real *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slarz);

int __TEMPLATE_FUNC(slarzb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_real *__v, __CLPK_integer *__ldv,
        __CLPK_real *__t, __CLPK_integer *__ldt, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(slarzb);

int __TEMPLATE_FUNC(slarzt)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__v, __CLPK_integer *__ldv,
        __CLPK_real *__tau, __CLPK_real *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(slarzt);

int __TEMPLATE_FUNC(slas2)(__CLPK_real *__f, __CLPK_real *__g, __CLPK_real *__h__,
        __CLPK_real *__ssmin,
        __CLPK_real *__ssmax)
__TEMPLATE_ALIAS(slas2);

int __TEMPLATE_FUNC(slascl)(char *__type__, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_real *__cfrom, __CLPK_real *__cto, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slascl);

int __TEMPLATE_FUNC(slascl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(slascl2);

int __TEMPLATE_FUNC(slasd0)(__CLPK_integer *__n, __CLPK_integer *__sqre, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__vt, __CLPK_integer *__ldvt, __CLPK_integer *__smlsiz,
        __CLPK_integer *__iwork, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd0);

int __TEMPLATE_FUNC(slasd1)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_real *__d__, __CLPK_real *__alpha, __CLPK_real *__beta,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__vt,
        __CLPK_integer *__ldvt, __CLPK_integer *__idxq, __CLPK_integer *__iwork,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd1);

int __TEMPLATE_FUNC(slasd2)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_integer *__k, __CLPK_real *__d__, __CLPK_real *__z__,
        __CLPK_real *__alpha, __CLPK_real *__beta, __CLPK_real *__u,
        __CLPK_integer *__ldu, __CLPK_real *__vt, __CLPK_integer *__ldvt,
        __CLPK_real *__dsigma, __CLPK_real *__u2, __CLPK_integer *__ldu2,
        __CLPK_real *__vt2, __CLPK_integer *__ldvt2, __CLPK_integer *__idxp,
        __CLPK_integer *__idx, __CLPK_integer *__idxc, __CLPK_integer *__idxq,
        __CLPK_integer *__coltyp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd2);

int __TEMPLATE_FUNC(slasd3)(__CLPK_integer *__nl, __CLPK_integer *__nr, __CLPK_integer *__sqre,
        __CLPK_integer *__k, __CLPK_real *__d__, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_real *__dsigma, __CLPK_real *__u,
        __CLPK_integer *__ldu, __CLPK_real *__u2, __CLPK_integer *__ldu2,
        __CLPK_real *__vt, __CLPK_integer *__ldvt, __CLPK_real *__vt2,
        __CLPK_integer *__ldvt2, __CLPK_integer *__idxc, __CLPK_integer *__ctot,
        __CLPK_real *__z__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd3);

int __TEMPLATE_FUNC(slasd4)(__CLPK_integer *__n, __CLPK_integer *__i__, __CLPK_real *__d__,
        __CLPK_real *__z__, __CLPK_real *__delta, __CLPK_real *__rho,
        __CLPK_real *__sigma, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd4);

int __TEMPLATE_FUNC(slasd5)(__CLPK_integer *__i__, __CLPK_real *__d__, __CLPK_real *__z__,
        __CLPK_real *__delta, __CLPK_real *__rho, __CLPK_real *__dsigma,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slasd5);

int __TEMPLATE_FUNC(slasd6)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_real *__d__,
        __CLPK_real *__vf, __CLPK_real *__vl, __CLPK_real *__alpha,
        __CLPK_real *__beta, __CLPK_integer *__idxq, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_real *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_real *__poles, __CLPK_real *__difl,
        __CLPK_real *__difr, __CLPK_real *__z__, __CLPK_integer *__k,
        __CLPK_real *__c__, __CLPK_real *__s, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd6);

int __TEMPLATE_FUNC(slasd7)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__k,
        __CLPK_real *__d__, __CLPK_real *__z__, __CLPK_real *__zw,
        __CLPK_real *__vf, __CLPK_real *__vfw, __CLPK_real *__vl,
        __CLPK_real *__vlw, __CLPK_real *__alpha, __CLPK_real *__beta,
        __CLPK_real *__dsigma, __CLPK_integer *__idx, __CLPK_integer *__idxp,
        __CLPK_integer *__idxq, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_real *__givnum,
        __CLPK_integer *__ldgnum, __CLPK_real *__c__, __CLPK_real *__s,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd7);

int __TEMPLATE_FUNC(slasd8)(__CLPK_integer *__icompq, __CLPK_integer *__k, __CLPK_real *__d__,
        __CLPK_real *__z__, __CLPK_real *__vf, __CLPK_real *__vl,
        __CLPK_real *__difl, __CLPK_real *__difr, __CLPK_integer *__lddifr,
        __CLPK_real *__dsigma, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasd8);

int __TEMPLATE_FUNC(slasda)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__sqre, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__vt, __CLPK_integer *__k, __CLPK_real *__difl,
        __CLPK_real *__difr, __CLPK_real *__z__, __CLPK_real *__poles,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_integer *__ldgcol, __CLPK_integer *__perm, __CLPK_real *__givnum,
        __CLPK_real *__c__, __CLPK_real *__s, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasda);

int __TEMPLATE_FUNC(slasdq)(char *__uplo, __CLPK_integer *__sqre, __CLPK_integer *__n,
        __CLPK_integer *__ncvt, __CLPK_integer *__nru, __CLPK_integer *__ncc,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vt,
        __CLPK_integer *__ldvt, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasdq);

int __TEMPLATE_FUNC(slasdt)(__CLPK_integer *__n, __CLPK_integer *__lvl, __CLPK_integer *__nd,
        __CLPK_integer *__inode, __CLPK_integer *__ndiml,
        __CLPK_integer *__ndimr,
        __CLPK_integer *__msub)
__TEMPLATE_ALIAS(slasdt);

int __TEMPLATE_FUNC(slaset)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__alpha, __CLPK_real *__beta, __CLPK_real *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(slaset);

int __TEMPLATE_FUNC(slasq1)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasq1);

int __TEMPLATE_FUNC(slasq2)(__CLPK_integer *__n, __CLPK_real *__z__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasq2);

int __TEMPLATE_FUNC(slasq3)(__CLPK_integer *__i0, __CLPK_integer *__n0, __CLPK_real *__z__,
        __CLPK_integer *__pp, __CLPK_real *__dmin__, __CLPK_real *__sigma,
        __CLPK_real *__desig, __CLPK_real *__qmax, __CLPK_integer *__nfail,
        __CLPK_integer *__iter, __CLPK_integer *__ndiv, __CLPK_logical *__ieee,
        __CLPK_integer *__ttype, __CLPK_real *__dmin1, __CLPK_real *__dmin2,
        __CLPK_real *__dn, __CLPK_real *__dn1, __CLPK_real *__dn2,
        __CLPK_real *__g,
        __CLPK_real *__tau)
__TEMPLATE_ALIAS(slasq3);

int __TEMPLATE_FUNC(slasq4)(__CLPK_integer *__i0, __CLPK_integer *__n0, __CLPK_real *__z__,
        __CLPK_integer *__pp, __CLPK_integer *__n0in, __CLPK_real *__dmin__,
        __CLPK_real *__dmin1, __CLPK_real *__dmin2, __CLPK_real *__dn,
        __CLPK_real *__dn1, __CLPK_real *__dn2, __CLPK_real *__tau,
        __CLPK_integer *__ttype,
        __CLPK_real *__g)
__TEMPLATE_ALIAS(slasq4);

int __TEMPLATE_FUNC(slasq5)(__CLPK_integer *__i0, __CLPK_integer *__n0, __CLPK_real *__z__,
        __CLPK_integer *__pp, __CLPK_real *__tau, __CLPK_real *__dmin__,
        __CLPK_real *__dmin1, __CLPK_real *__dmin2, __CLPK_real *__dn,
        __CLPK_real *__dnm1, __CLPK_real *__dnm2,
        __CLPK_logical *__ieee)
__TEMPLATE_ALIAS(slasq5);

int __TEMPLATE_FUNC(slasq6)(__CLPK_integer *__i0, __CLPK_integer *__n0, __CLPK_real *__z__,
        __CLPK_integer *__pp, __CLPK_real *__dmin__, __CLPK_real *__dmin1,
        __CLPK_real *__dmin2, __CLPK_real *__dn, __CLPK_real *__dnm1,
        __CLPK_real *__dnm2)
__TEMPLATE_ALIAS(slasq6);

int __TEMPLATE_FUNC(slasr)(char *__side, char *__pivot, char *__direct, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__c__, __CLPK_real *__s,
        __CLPK_real *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(slasr);

int __TEMPLATE_FUNC(slasrt)(char *__id, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasrt);

int __TEMPLATE_FUNC(slassq)(__CLPK_integer *__n, __CLPK_real *__x, __CLPK_integer *__incx,
        __CLPK_real *__scale,
        __CLPK_real *__sumsq)
__TEMPLATE_ALIAS(slassq);

int __TEMPLATE_FUNC(slasv2)(__CLPK_real *__f, __CLPK_real *__g, __CLPK_real *__h__,
        __CLPK_real *__ssmin, __CLPK_real *__ssmax, __CLPK_real *__snr,
        __CLPK_real *__csr, __CLPK_real *__snl,
        __CLPK_real *__csl)
__TEMPLATE_ALIAS(slasv2);

int __TEMPLATE_FUNC(slaswp)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__k1, __CLPK_integer *__k2, __CLPK_integer *__ipiv,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(slaswp);

int __TEMPLATE_FUNC(slasy2)(__CLPK_logical *__ltranl, __CLPK_logical *__ltranr,
        __CLPK_integer *__isgn, __CLPK_integer *__n1, __CLPK_integer *__n2,
        __CLPK_real *__tl, __CLPK_integer *__ldtl, __CLPK_real *__tr,
        __CLPK_integer *__ldtr, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__scale, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__xnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasy2);

int __TEMPLATE_FUNC(slasyf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_real *__w, __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slasyf);

int __TEMPLATE_FUNC(slatbs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_integer *__kd, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__x, __CLPK_real *__scale,
        __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slatbs);

int __TEMPLATE_FUNC(slatdf)(__CLPK_integer *__ijob, __CLPK_integer *__n, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_real *__rhs, __CLPK_real *__rdsum,
        __CLPK_real *__rdscal, __CLPK_integer *__ipiv,
        __CLPK_integer *__jpiv)
__TEMPLATE_ALIAS(slatdf);

int __TEMPLATE_FUNC(slatps)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_real *__ap, __CLPK_real *__x,
        __CLPK_real *__scale, __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slatps);

int __TEMPLATE_FUNC(slatrd)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__e,
        __CLPK_real *__tau, __CLPK_real *__w,
        __CLPK_integer *__ldw)
__TEMPLATE_ALIAS(slatrd);

int __TEMPLATE_FUNC(slatrs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__x, __CLPK_real *__scale, __CLPK_real *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slatrs);

int __TEMPLATE_FUNC(slatrz)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__l,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slatrz);

int __TEMPLATE_FUNC(slatzm)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__v, __CLPK_integer *__incv, __CLPK_real *__tau,
        __CLPK_real *__c1, __CLPK_real *__c2, __CLPK_integer *__ldc,
        __CLPK_real *__work)
__TEMPLATE_ALIAS(slatzm);

int __TEMPLATE_FUNC(slauu2)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slauu2);

int __TEMPLATE_FUNC(slauum)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(slauum);

int __TEMPLATE_FUNC(sopgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__tau, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sopgtr);

int __TEMPLATE_FUNC(sopmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__ap, __CLPK_real *__tau,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sopmtr);

int __TEMPLATE_FUNC(sorg2l)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorg2l);

int __TEMPLATE_FUNC(sorg2r)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorg2r);

int __TEMPLATE_FUNC(sorgbr)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__tau, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgbr);

int __TEMPLATE_FUNC(sorghr)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorghr);

int __TEMPLATE_FUNC(sorgl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgl2);

int __TEMPLATE_FUNC(sorglq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorglq);

int __TEMPLATE_FUNC(sorgql)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgql);

int __TEMPLATE_FUNC(sorgqr)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgqr);

int __TEMPLATE_FUNC(sorgr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgr2);

int __TEMPLATE_FUNC(sorgrq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgrq);

int __TEMPLATE_FUNC(sorgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorgtr);

int __TEMPLATE_FUNC(sorm2l)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorm2l);

int __TEMPLATE_FUNC(sorm2r)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorm2r);

int __TEMPLATE_FUNC(sormbr)(char *__vect, char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormbr);

int __TEMPLATE_FUNC(sormhr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormhr);

int __TEMPLATE_FUNC(sorml2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sorml2);

int __TEMPLATE_FUNC(sormlq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormlq);

int __TEMPLATE_FUNC(sormql)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormql);

int __TEMPLATE_FUNC(sormqr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormqr);

int __TEMPLATE_FUNC(sormr2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormr2);

int __TEMPLATE_FUNC(sormr3)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormr3);

int __TEMPLATE_FUNC(sormrq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormrq);

int __TEMPLATE_FUNC(sormrz)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormrz);

int __TEMPLATE_FUNC(sormtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__tau, __CLPK_real *__c__, __CLPK_integer *__ldc,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sormtr);

int __TEMPLATE_FUNC(spbcon)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab, __CLPK_real *__anorm,
        __CLPK_real *__rcond, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbcon);

int __TEMPLATE_FUNC(spbequ)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab, __CLPK_real *__s,
        __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbequ);

int __TEMPLATE_FUNC(spbrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__afb, __CLPK_integer *__ldafb, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbrfs);

int __TEMPLATE_FUNC(spbstf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbstf);

int __TEMPLATE_FUNC(spbsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbsv);

int __TEMPLATE_FUNC(spbsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__afb, __CLPK_integer *__ldafb,
        char *__equed, __CLPK_real *__s, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbsvx);

int __TEMPLATE_FUNC(spbtf2)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbtf2);

int __TEMPLATE_FUNC(spbtrf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbtrf);

int __TEMPLATE_FUNC(spbtrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spbtrs);

int __TEMPLATE_FUNC(spftrf)(char *__transr, char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spftrf);

int __TEMPLATE_FUNC(spftri)(char *__transr, char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spftri);

int __TEMPLATE_FUNC(spftrs)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spftrs);

int __TEMPLATE_FUNC(spocon)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spocon);

int __TEMPLATE_FUNC(spoequ)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spoequ);

int __TEMPLATE_FUNC(spoequb)(__CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spoequb);

int __TEMPLATE_FUNC(sporfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__af,
        __CLPK_integer *__ldaf, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sporfs);

int __TEMPLATE_FUNC(sposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sposv);

int __TEMPLATE_FUNC(sposvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__af, __CLPK_integer *__ldaf, char *__equed,
        __CLPK_real *__s, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sposvx);

int __TEMPLATE_FUNC(spotf2)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spotf2);

int __TEMPLATE_FUNC(spotrf)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spotrf);

int __TEMPLATE_FUNC(spotri)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spotri);

int __TEMPLATE_FUNC(spotrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spotrs);

int __TEMPLATE_FUNC(sppcon)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__anorm, __CLPK_real *__rcond, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sppcon);

int __TEMPLATE_FUNC(sppequ)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__s, __CLPK_real *__scond, __CLPK_real *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sppequ);

int __TEMPLATE_FUNC(spprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_real *__afp, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spprfs);

int __TEMPLATE_FUNC(sppsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sppsv);

int __TEMPLATE_FUNC(sppsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__ap, __CLPK_real *__afp,
        char *__equed, __CLPK_real *__s, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__rcond, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sppsvx);

int __TEMPLATE_FUNC(spptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spptrf);

int __TEMPLATE_FUNC(spptri)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spptri);

int __TEMPLATE_FUNC(spptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spptrs);

int __TEMPLATE_FUNC(spstf2)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_real *__tol, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spstf2);

int __TEMPLATE_FUNC(spstrf)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_real *__tol, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spstrf);

int __TEMPLATE_FUNC(sptcon)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__anorm, __CLPK_real *__rcond, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sptcon);

int __TEMPLATE_FUNC(spteqr)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spteqr);

int __TEMPLATE_FUNC(sptrfs)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__df, __CLPK_real *__ef,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sptrfs);

int __TEMPLATE_FUNC(sptsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sptsv);

int __TEMPLATE_FUNC(sptsvx)(char *__fact, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__df,
        __CLPK_real *__ef, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sptsvx);

int __TEMPLATE_FUNC(spttrf)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spttrf);

int __TEMPLATE_FUNC(spttrs)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(spttrs);

int __TEMPLATE_FUNC(sptts2)(__CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(sptts2);

int __TEMPLATE_FUNC(srscl)(__CLPK_integer *__n, __CLPK_real *__sa, __CLPK_real *__sx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(srscl);

int __TEMPLATE_FUNC(ssbev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbev);

int __TEMPLATE_FUNC(ssbevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbevd);

int __TEMPLATE_FUNC(ssbevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbevx);

int __TEMPLATE_FUNC(ssbgst)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbgst);

int __TEMPLATE_FUNC(ssbgv)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbgv);

int __TEMPLATE_FUNC(ssbgvd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbgvd);

int __TEMPLATE_FUNC(ssbgvx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__bb, __CLPK_integer *__ldbb,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbgvx);

int __TEMPLATE_FUNC(ssbtrd)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssbtrd);

int __TEMPLATE_FUNC(ssfrk)(char *__transr, char *__uplo, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_real *__alpha, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__beta,
        __CLPK_real *__c__)
__TEMPLATE_ALIAS(ssfrk);

int __TEMPLATE_FUNC(sspcon)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__ipiv, __CLPK_real *__anorm, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspcon);

int __TEMPLATE_FUNC(sspev)(char *__jobz, char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspev);

int __TEMPLATE_FUNC(sspevd)(char *__jobz, char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspevd);

int __TEMPLATE_FUNC(sspevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__ap, __CLPK_real *__vl, __CLPK_real *__vu,
        __CLPK_integer *__il, __CLPK_integer *__iu, __CLPK_real *__abstol,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspevx);

int __TEMPLATE_FUNC(sspgst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__ap, __CLPK_real *__bp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspgst);

int __TEMPLATE_FUNC(sspgv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__ap, __CLPK_real *__bp,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspgv);

int __TEMPLATE_FUNC(sspgvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__ap, __CLPK_real *__bp,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspgvd);

int __TEMPLATE_FUNC(sspgvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__ap, __CLPK_real *__bp,
        __CLPK_real *__vl, __CLPK_real *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_real *__abstol, __CLPK_integer *__m,
        __CLPK_real *__w, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspgvx);

int __TEMPLATE_FUNC(ssprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_real *__afp, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssprfs);

int __TEMPLATE_FUNC(sspsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspsv);

int __TEMPLATE_FUNC(sspsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__ap, __CLPK_real *__afp,
        __CLPK_integer *__ipiv, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__rcond,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sspsvx);

int __TEMPLATE_FUNC(ssptrd)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssptrd);

int __TEMPLATE_FUNC(ssptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssptrf);

int __TEMPLATE_FUNC(ssptri)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__ipiv, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssptri);

int __TEMPLATE_FUNC(ssptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__ap, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssptrs);

int __TEMPLATE_FUNC(sstebz)(char *__range, char *__order, __CLPK_integer *__n,
        __CLPK_real *__vl, __CLPK_real *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_real *__abstol, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_integer *__m, __CLPK_integer *__nsplit,
        __CLPK_real *__w, __CLPK_integer *__iblock, __CLPK_integer *__isplit,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstebz);

int __TEMPLATE_FUNC(sstedc)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstedc);

int __TEMPLATE_FUNC(sstegr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstegr);

int __TEMPLATE_FUNC(sstein)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_integer *__iblock,
        __CLPK_integer *__isplit, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstein);

int __TEMPLATE_FUNC(sstemr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_integer *__m, __CLPK_real *__w, __CLPK_real *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__nzc, __CLPK_integer *__isuppz,
        __CLPK_logical *__tryrac, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstemr);

int __TEMPLATE_FUNC(ssteqr)(char *__compz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssteqr);

int __TEMPLATE_FUNC(ssterf)(__CLPK_integer *__n, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssterf);

int __TEMPLATE_FUNC(sstev)(char *__jobz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstev);

int __TEMPLATE_FUNC(sstevd)(char *__jobz, __CLPK_integer *__n, __CLPK_real *__d__,
        __CLPK_real *__e, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstevd);

int __TEMPLATE_FUNC(sstevr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstevr);

int __TEMPLATE_FUNC(sstevx)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_real *__d__, __CLPK_real *__e, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(sstevx);

int __TEMPLATE_FUNC(ssycon)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__anorm,
        __CLPK_real *__rcond, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssycon);

int __TEMPLATE_FUNC(ssyequb)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__s, __CLPK_real *__scond,
        __CLPK_real *__amax, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyequb);

int __TEMPLATE_FUNC(ssyev)(char *__jobz, char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__w, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyev);

int __TEMPLATE_FUNC(ssyevd)(char *__jobz, char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__w, __CLPK_real *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyevd);

int __TEMPLATE_FUNC(ssyevr)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyevr);

int __TEMPLATE_FUNC(ssyevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyevx);

int __TEMPLATE_FUNC(ssygs2)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssygs2);

int __TEMPLATE_FUNC(ssygst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssygst);

int __TEMPLATE_FUNC(ssygv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__w,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssygv);

int __TEMPLATE_FUNC(ssygvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__w,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssygvd);

int __TEMPLATE_FUNC(ssygvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__vl,
        __CLPK_real *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_real *__abstol, __CLPK_integer *__m, __CLPK_real *__w,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_real *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssygvx);

int __TEMPLATE_FUNC(ssyrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssyrfs);

int __TEMPLATE_FUNC(ssysv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssysv);

int __TEMPLATE_FUNC(ssysvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__af, __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__rcond, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssysvx);

int __TEMPLATE_FUNC(ssytd2)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytd2);

int __TEMPLATE_FUNC(ssytf2)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytf2);

int __TEMPLATE_FUNC(ssytrd)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__d__, __CLPK_real *__e,
        __CLPK_real *__tau, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytrd);

int __TEMPLATE_FUNC(ssytrf)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytrf);

int __TEMPLATE_FUNC(ssytri)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytri);

int __TEMPLATE_FUNC(ssytrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ssytrs);

int __TEMPLATE_FUNC(stbcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_real *__ab, __CLPK_integer *__ldab,
        __CLPK_real *__rcond, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stbcon);

int __TEMPLATE_FUNC(stbrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__x, __CLPK_integer *__ldx, __CLPK_real *__ferr,
        __CLPK_real *__berr, __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stbrfs);

int __TEMPLATE_FUNC(stbtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs, __CLPK_real *__ab,
        __CLPK_integer *__ldab, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stbtrs);

int __TEMPLATE_FUNC(stfsm)(char *__transr, char *__side, char *__uplo, char *__trans,
        char *__diag, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_real *__alpha, __CLPK_real *__a, __CLPK_real *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(stfsm);

int __TEMPLATE_FUNC(stftri)(char *__transr, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_real *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stftri);

int __TEMPLATE_FUNC(stfttp)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__arf, __CLPK_real *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stfttp);

int __TEMPLATE_FUNC(stfttr)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__arf, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stfttr);

int __TEMPLATE_FUNC(stgevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_real *__s, __CLPK_integer *__lds,
        __CLPK_real *__p, __CLPK_integer *__ldp, __CLPK_real *__vl,
        __CLPK_integer *__ldvl, __CLPK_real *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgevc);

int __TEMPLATE_FUNC(stgex2)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__j1, __CLPK_integer *__n1, __CLPK_integer *__n2,
        __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgex2);

int __TEMPLATE_FUNC(stgexc)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__q,
        __CLPK_integer *__ldq, __CLPK_real *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgexc);

int __TEMPLATE_FUNC(stgsen)(__CLPK_integer *__ijob, __CLPK_logical *__wantq,
        __CLPK_logical *__wantz, __CLPK_logical *__select, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__alphar, __CLPK_real *__alphai,
        __CLPK_real *__beta, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_real *__z__, __CLPK_integer *__ldz, __CLPK_integer *__m,
        __CLPK_real *__pl, __CLPK_real *__pr, __CLPK_real *__dif,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgsen);

int __TEMPLATE_FUNC(stgsja)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__tola,
        __CLPK_real *__tolb, __CLPK_real *__alpha, __CLPK_real *__beta,
        __CLPK_real *__u, __CLPK_integer *__ldu, __CLPK_real *__v,
        __CLPK_integer *__ldv, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_real *__work, __CLPK_integer *__ncycle,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgsja);

int __TEMPLATE_FUNC(stgsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__vl,
        __CLPK_integer *__ldvl, __CLPK_real *__vr, __CLPK_integer *__ldvr,
        __CLPK_real *__s, __CLPK_real *__dif, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgsna);

int __TEMPLATE_FUNC(stgsy2)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__d__, __CLPK_integer *__ldd,
        __CLPK_real *__e, __CLPK_integer *__lde, __CLPK_real *__f,
        __CLPK_integer *__ldf, __CLPK_real *__scale, __CLPK_real *__rdsum,
        __CLPK_real *__rdscal, __CLPK_integer *__iwork, __CLPK_integer *__pq,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgsy2);

int __TEMPLATE_FUNC(stgsyl)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__c__,
        __CLPK_integer *__ldc, __CLPK_real *__d__, __CLPK_integer *__ldd,
        __CLPK_real *__e, __CLPK_integer *__lde, __CLPK_real *__f,
        __CLPK_integer *__ldf, __CLPK_real *__scale, __CLPK_real *__dif,
        __CLPK_real *__work, __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stgsyl);

int __TEMPLATE_FUNC(stpcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_real *__ap, __CLPK_real *__rcond, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stpcon);

int __TEMPLATE_FUNC(stprfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__ap, __CLPK_real *__b,
        __CLPK_integer *__ldb, __CLPK_real *__x, __CLPK_integer *__ldx,
        __CLPK_real *__ferr, __CLPK_real *__berr, __CLPK_real *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stprfs);

int __TEMPLATE_FUNC(stptri)(char *__uplo, char *__diag, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stptri);

int __TEMPLATE_FUNC(stptrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__ap, __CLPK_real *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stptrs);

int __TEMPLATE_FUNC(stpttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_real *__ap, __CLPK_real *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stpttf);

int __TEMPLATE_FUNC(stpttr)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__ap,
        __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stpttr);

int __TEMPLATE_FUNC(strcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_real *__a, __CLPK_integer *__lda, __CLPK_real *__rcond,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strcon);

int __TEMPLATE_FUNC(strevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_real *__t, __CLPK_integer *__ldt,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_integer *__mm, __CLPK_integer *__m,
        __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strevc);

int __TEMPLATE_FUNC(strexc)(char *__compq, __CLPK_integer *__n, __CLPK_real *__t,
        __CLPK_integer *__ldt, __CLPK_real *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst, __CLPK_real *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strexc);

int __TEMPLATE_FUNC(strrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb, __CLPK_real *__x,
        __CLPK_integer *__ldx, __CLPK_real *__ferr, __CLPK_real *__berr,
        __CLPK_real *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strrfs);

int __TEMPLATE_FUNC(strsen)(char *__job, char *__compq, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_real *__t, __CLPK_integer *__ldt,
        __CLPK_real *__q, __CLPK_integer *__ldq, __CLPK_real *__wr,
        __CLPK_real *__wi, __CLPK_integer *__m, __CLPK_real *__s,
        __CLPK_real *__sep, __CLPK_real *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strsen);

int __TEMPLATE_FUNC(strsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_real *__t, __CLPK_integer *__ldt,
        __CLPK_real *__vl, __CLPK_integer *__ldvl, __CLPK_real *__vr,
        __CLPK_integer *__ldvr, __CLPK_real *__s, __CLPK_real *__sep,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_real *__work,
        __CLPK_integer *__ldwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strsna);

int __TEMPLATE_FUNC(strsyl)(char *__trana, char *__tranb, __CLPK_integer *__isgn,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_real *__c__, __CLPK_integer *__ldc, __CLPK_real *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strsyl);

int __TEMPLATE_FUNC(strti2)(char *__uplo, char *__diag, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strti2);

int __TEMPLATE_FUNC(strtri)(char *__uplo, char *__diag, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strtri);

int __TEMPLATE_FUNC(strtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strtrs);

int __TEMPLATE_FUNC(strttf)(char *__transr, char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strttf);

int __TEMPLATE_FUNC(strttp)(char *__uplo, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(strttp);

int __TEMPLATE_FUNC(stzrqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stzrqf);

int __TEMPLATE_FUNC(stzrzf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_real *__a,
        __CLPK_integer *__lda, __CLPK_real *__tau, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(stzrzf);

int __TEMPLATE_FUNC(zbdsqr)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__ncvt,
        __CLPK_integer *__nru, __CLPK_integer *__ncc, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublecomplex *__u,
        __CLPK_integer *__ldu, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zbdsqr);

int __TEMPLATE_FUNC(zcgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublecomplex *__work, __CLPK_complex *__swork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iter,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zcgesv);

int __TEMPLATE_FUNC(zcposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublecomplex *__work, __CLPK_complex *__swork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iter,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zcposv);

int __TEMPLATE_FUNC(zdrscl)(__CLPK_integer *__n, __CLPK_doublereal *__sa,
        __CLPK_doublecomplex *__sx,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(zdrscl);

int __TEMPLATE_FUNC(zgbbrd)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__ncc, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__pt, __CLPK_integer *__ldpt,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbbrd);

int __TEMPLATE_FUNC(zgbcon)(char *__norm, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbcon);

int __TEMPLATE_FUNC(zgbequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbequ);

int __TEMPLATE_FUNC(zgbequb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbequb);

int __TEMPLATE_FUNC(zgbrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublecomplex *__afb, __CLPK_integer *__ldafb,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbrfs);

int __TEMPLATE_FUNC(zgbsv)(__CLPK_integer *__n, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbsv);

int __TEMPLATE_FUNC(zgbsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublecomplex *__afb, __CLPK_integer *__ldafb,
        __CLPK_integer *__ipiv, char *__equed, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbsvx);

int __TEMPLATE_FUNC(zgbtf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbtf2);

int __TEMPLATE_FUNC(zgbtrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbtrf);

int __TEMPLATE_FUNC(zgbtrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgbtrs);

int __TEMPLATE_FUNC(zgebak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__scale, __CLPK_integer *__m,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgebak);

int __TEMPLATE_FUNC(zgebal)(char *__job, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgebal);

int __TEMPLATE_FUNC(zgebd2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tauq, __CLPK_doublecomplex *__taup,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgebd2);

int __TEMPLATE_FUNC(zgebrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tauq, __CLPK_doublecomplex *__taup,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgebrd);

int __TEMPLATE_FUNC(zgecon)(char *__norm, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgecon);

int __TEMPLATE_FUNC(zgeequ)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeequ);

int __TEMPLATE_FUNC(zgeequb)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublereal *__rowcnd, __CLPK_doublereal *__colcnd,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeequb);

int __TEMPLATE_FUNC(zgees)(char *__jobvs, char *__sort, __CLPK_L_fp __select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_doublecomplex *__w,
        __CLPK_doublecomplex *__vs, __CLPK_integer *__ldvs,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgees);

int __TEMPLATE_FUNC(zgeesx)(char *__jobvs, char *__sort, __CLPK_L_fp __select, char *__sense,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__sdim, __CLPK_doublecomplex *__w,
        __CLPK_doublecomplex *__vs, __CLPK_integer *__ldvs,
        __CLPK_doublereal *__rconde, __CLPK_doublereal *__rcondv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeesx);

int __TEMPLATE_FUNC(zgeev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublecomplex *__vr,
        __CLPK_integer *__ldvr, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeev);

int __TEMPLATE_FUNC(zgeevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublecomplex *__vr,
        __CLPK_integer *__ldvr, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__abnrm,
        __CLPK_doublereal *__rconde, __CLPK_doublereal *__rcondv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeevx);

int __TEMPLATE_FUNC(zgegs)(char *__jobvsl, char *__jobvsr, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__vsl, __CLPK_integer *__ldvsl,
        __CLPK_doublecomplex *__vsr, __CLPK_integer *__ldvsr,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgegs);

int __TEMPLATE_FUNC(zgegv)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgegv);

int __TEMPLATE_FUNC(zgehd2)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgehd2);

int __TEMPLATE_FUNC(zgehrd)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgehrd);

int __TEMPLATE_FUNC(zgelq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelq2);

int __TEMPLATE_FUNC(zgelqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelqf);

int __TEMPLATE_FUNC(zgels)(char *__trans, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgels);

int __TEMPLATE_FUNC(zgelsd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__s, __CLPK_doublereal *__rcond,
        __CLPK_integer *__rank, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelsd);

int __TEMPLATE_FUNC(zgelss)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__s, __CLPK_doublereal *__rcond,
        __CLPK_integer *__rank, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelss);

int __TEMPLATE_FUNC(zgelsx)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__jpvt, __CLPK_doublereal *__rcond,
        __CLPK_integer *__rank, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelsx);

int __TEMPLATE_FUNC(zgelsy)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__jpvt, __CLPK_doublereal *__rcond,
        __CLPK_integer *__rank, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgelsy);

int __TEMPLATE_FUNC(zgeql2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeql2);

int __TEMPLATE_FUNC(zgeqlf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeqlf);

int __TEMPLATE_FUNC(zgeqp3)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeqp3);

int __TEMPLATE_FUNC(zgeqpf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeqpf);

int __TEMPLATE_FUNC(zgeqr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeqr2);

int __TEMPLATE_FUNC(zgeqrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgeqrf);

int __TEMPLATE_FUNC(zgerfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__af, __CLPK_integer *__ldaf,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgerfs);

int __TEMPLATE_FUNC(zgerq2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgerq2);

int __TEMPLATE_FUNC(zgerqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgerqf);

int __TEMPLATE_FUNC(zgesc2)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__rhs,
        __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_doublereal *__scale)
__TEMPLATE_ALIAS(zgesc2);

int __TEMPLATE_FUNC(zgesdd)(char *__jobz, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublecomplex *__u,
        __CLPK_integer *__ldu, __CLPK_doublecomplex *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgesdd);

int __TEMPLATE_FUNC(zgesv)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgesv);

int __TEMPLATE_FUNC(zgesvd)(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublecomplex *__u,
        __CLPK_integer *__ldu, __CLPK_doublecomplex *__vt,
        __CLPK_integer *__ldvt, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgesvd);

int __TEMPLATE_FUNC(zgesvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv, char *__equed,
        __CLPK_doublereal *__r__, __CLPK_doublereal *__c__,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgesvx);

int __TEMPLATE_FUNC(zgetc2)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv, __CLPK_integer *__jpiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgetc2);

int __TEMPLATE_FUNC(zgetf2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgetf2);

int __TEMPLATE_FUNC(zgetrf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgetrf);

int __TEMPLATE_FUNC(zgetri)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgetri);

int __TEMPLATE_FUNC(zgetrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgetrs);

int __TEMPLATE_FUNC(zggbak)(char *__job, char *__side, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_integer *__m, __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggbak);

int __TEMPLATE_FUNC(zggbal)(char *__job, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggbal);

int __TEMPLATE_FUNC(zgges)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__sdim, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__beta, __CLPK_doublecomplex *__vsl,
        __CLPK_integer *__ldvsl, __CLPK_doublecomplex *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgges);

int __TEMPLATE_FUNC(zggesx)(char *__jobvsl, char *__jobvsr, char *__sort, __CLPK_L_fp __selctg,
        char *__sense, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__sdim, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__beta, __CLPK_doublecomplex *__vsl,
        __CLPK_integer *__ldvsl, __CLPK_doublecomplex *__vsr,
        __CLPK_integer *__ldvsr, __CLPK_doublereal *__rconde,
        __CLPK_doublereal *__rcondv, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggesx);

int __TEMPLATE_FUNC(zggev)(char *__jobvl, char *__jobvr, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggev);

int __TEMPLATE_FUNC(zggevx)(char *__balanc, char *__jobvl, char *__jobvr, char *__sense,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublereal *__lscale, __CLPK_doublereal *__rscale,
        __CLPK_doublereal *__abnrm, __CLPK_doublereal *__bbnrm,
        __CLPK_doublereal *__rconde, __CLPK_doublereal *__rcondv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_logical *__bwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggevx);

int __TEMPLATE_FUNC(zggglm)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__x,
        __CLPK_doublecomplex *__y, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggglm);

int __TEMPLATE_FUNC(zgghrd)(char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgghrd);

int __TEMPLATE_FUNC(zgglse)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__p,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_doublecomplex *__d__,
        __CLPK_doublecomplex *__x, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgglse);

int __TEMPLATE_FUNC(zggqrf)(__CLPK_integer *__n, __CLPK_integer *__m, __CLPK_integer *__p,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__taua, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__taub,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggqrf);

int __TEMPLATE_FUNC(zggrqf)(__CLPK_integer *__m, __CLPK_integer *__p, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__taua, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__taub,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggrqf);

int __TEMPLATE_FUNC(zggsvd)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__p, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublecomplex *__u, __CLPK_integer *__ldu,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggsvd);

int __TEMPLATE_FUNC(zggsvp)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__tola, __CLPK_doublereal *__tolb,
        __CLPK_integer *__k, __CLPK_integer *__l, __CLPK_doublecomplex *__u,
        __CLPK_integer *__ldu, __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__iwork, __CLPK_doublereal *__rwork,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zggsvp);

int __TEMPLATE_FUNC(zgtcon)(char *__norm, __CLPK_integer *__n, __CLPK_doublecomplex *__dl,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__du,
        __CLPK_doublecomplex *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgtcon);

int __TEMPLATE_FUNC(zgtrfs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__dl, __CLPK_doublecomplex *__d__,
        __CLPK_doublecomplex *__du, __CLPK_doublecomplex *__dlf,
        __CLPK_doublecomplex *__df, __CLPK_doublecomplex *__duf,
        __CLPK_doublecomplex *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgtrfs);

int __TEMPLATE_FUNC(zgtsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__dl, __CLPK_doublecomplex *__d__,
        __CLPK_doublecomplex *__du, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgtsv);

int __TEMPLATE_FUNC(zgtsvx)(char *__fact, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__dl,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__du,
        __CLPK_doublecomplex *__dlf, __CLPK_doublecomplex *__df,
        __CLPK_doublecomplex *__duf, __CLPK_doublecomplex *__du2,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgtsvx);

int __TEMPLATE_FUNC(zgttrf)(__CLPK_integer *__n, __CLPK_doublecomplex *__dl,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__du,
        __CLPK_doublecomplex *__du2, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgttrf);

int __TEMPLATE_FUNC(zgttrs)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__dl, __CLPK_doublecomplex *__d__,
        __CLPK_doublecomplex *__du, __CLPK_doublecomplex *__du2,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zgttrs);

int __TEMPLATE_FUNC(zgtts2)(__CLPK_integer *__itrans, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__dl,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__du,
        __CLPK_doublecomplex *__du2, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(zgtts2);

int __TEMPLATE_FUNC(zhbev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbev);

int __TEMPLATE_FUNC(zhbevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__lrwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbevd);

int __TEMPLATE_FUNC(zhbevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__vl, __CLPK_doublereal *__vu,
        __CLPK_integer *__il, __CLPK_integer *__iu, __CLPK_doublereal *__abstol,
        __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbevx);

int __TEMPLATE_FUNC(zhbgst)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__bb,
        __CLPK_integer *__ldbb, __CLPK_doublecomplex *__x,
        __CLPK_integer *__ldx, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbgst);

int __TEMPLATE_FUNC(zhbgv)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__bb,
        __CLPK_integer *__ldbb, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbgv);

int __TEMPLATE_FUNC(zhbgvd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__bb,
        __CLPK_integer *__ldbb, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__lrwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbgvd);

int __TEMPLATE_FUNC(zhbgvx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__ka, __CLPK_integer *__kb, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__bb,
        __CLPK_integer *__ldbb, __CLPK_doublecomplex *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__vl, __CLPK_doublereal *__vu,
        __CLPK_integer *__il, __CLPK_integer *__iu, __CLPK_doublereal *__abstol,
        __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbgvx);

int __TEMPLATE_FUNC(zhbtrd)(char *__vect, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__q,
        __CLPK_integer *__ldq, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhbtrd);

int __TEMPLATE_FUNC(zhecon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhecon);

int __TEMPLATE_FUNC(zheequb)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zheequb);

int __TEMPLATE_FUNC(zheev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zheev);

int __TEMPLATE_FUNC(zheevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zheevd);

int __TEMPLATE_FUNC(zheevr)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__lrwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zheevr);

int __TEMPLATE_FUNC(zheevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zheevx);

int __TEMPLATE_FUNC(zhegs2)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhegs2);

int __TEMPLATE_FUNC(zhegst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhegst);

int __TEMPLATE_FUNC(zhegv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhegv);

int __TEMPLATE_FUNC(zhegvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhegvd);

int __TEMPLATE_FUNC(zhegvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork, __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhegvx);

int __TEMPLATE_FUNC(zherfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__af, __CLPK_integer *__ldaf,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zherfs);

int __TEMPLATE_FUNC(zhesv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhesv);

int __TEMPLATE_FUNC(zhesvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhesvx);

int __TEMPLATE_FUNC(zhetd2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetd2);

int __TEMPLATE_FUNC(zhetf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetf2);

int __TEMPLATE_FUNC(zhetrd)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetrd);

int __TEMPLATE_FUNC(zhetrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetrf);

int __TEMPLATE_FUNC(zhetri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetri);

int __TEMPLATE_FUNC(zhetrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhetrs);

int __TEMPLATE_FUNC(zhfrk)(char *__transr, char *__uplo, char *__trans, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublereal *__alpha,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__beta,
        __CLPK_doublecomplex *__c__)
__TEMPLATE_ALIAS(zhfrk);

int __TEMPLATE_FUNC(zhgeqz)(char *__job, char *__compq, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhgeqz);

int __TEMPLATE_FUNC(zhpcon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpcon);

int __TEMPLATE_FUNC(zhpev)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpev);

int __TEMPLATE_FUNC(zhpevd)(char *__jobz, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__lrwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpevd);

int __TEMPLATE_FUNC(zhpevx)(char *__jobz, char *__range, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpevx);

int __TEMPLATE_FUNC(zhpgst)(__CLPK_integer *__itype, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__bp,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpgst);

int __TEMPLATE_FUNC(zhpgv)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__bp, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpgv);

int __TEMPLATE_FUNC(zhpgvd)(__CLPK_integer *__itype, char *__jobz, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__bp, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_doublereal *__rwork, __CLPK_integer *__lrwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpgvd);

int __TEMPLATE_FUNC(zhpgvx)(__CLPK_integer *__itype, char *__jobz, char *__range, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__bp, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_integer *__il, __CLPK_integer *__iu,
        __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpgvx);

int __TEMPLATE_FUNC(zhprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__afp,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhprfs);

int __TEMPLATE_FUNC(zhpsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpsv);

int __TEMPLATE_FUNC(zhpsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__afp, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhpsvx);

int __TEMPLATE_FUNC(zhptrd)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhptrd);

int __TEMPLATE_FUNC(zhptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhptrf);

int __TEMPLATE_FUNC(zhptri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhptri);

int __TEMPLATE_FUNC(zhptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhptrs);

int __TEMPLATE_FUNC(zhsein)(char *__side, char *__eigsrc, char *__initv,
        __CLPK_logical *__select, __CLPK_integer *__n,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__vl,
        __CLPK_integer *__ldvl, __CLPK_doublecomplex *__vr,
        __CLPK_integer *__ldvr, __CLPK_integer *__mm, __CLPK_integer *__m,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__ifaill, __CLPK_integer *__ifailr,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhsein);

int __TEMPLATE_FUNC(zhseqr)(char *__job, char *__compz, __CLPK_integer *__n,
        __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zhseqr);

int __TEMPLATE_FUNC(zlabrd)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__tauq, __CLPK_doublecomplex *__taup,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublecomplex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(zlabrd);

int __TEMPLATE_FUNC(zlacgv)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(zlacgv);

int __TEMPLATE_FUNC(zlacn2)(__CLPK_integer *__n, __CLPK_doublecomplex *__v,
        __CLPK_doublecomplex *__x, __CLPK_doublereal *__est,
        __CLPK_integer *__kase,
        __CLPK_integer *__isave)
__TEMPLATE_ALIAS(zlacn2);

int __TEMPLATE_FUNC(zlacon)(__CLPK_integer *__n, __CLPK_doublecomplex *__v,
        __CLPK_doublecomplex *__x, __CLPK_doublereal *__est,
        __CLPK_integer *__kase)
__TEMPLATE_ALIAS(zlacon);

int __TEMPLATE_FUNC(zlacp2)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(zlacp2);

int __TEMPLATE_FUNC(zlacpy)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(zlacpy);

int __TEMPLATE_FUNC(zlacrm)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__rwork)
__TEMPLATE_ALIAS(zlacrm);

int __TEMPLATE_FUNC(zlacrt)(__CLPK_integer *__n, __CLPK_doublecomplex *__cx,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__cy,
        __CLPK_integer *__incy, __CLPK_doublecomplex *__c__,
        __CLPK_doublecomplex *__s)
__TEMPLATE_ALIAS(zlacrt);

void __TEMPLATE_FUNC(zladiv)(__CLPK_doublecomplex *__ret_val, __CLPK_doublecomplex *__x,
        __CLPK_doublecomplex *__y)
__TEMPLATE_ALIAS(zladiv);

int __TEMPLATE_FUNC(zlaed0)(__CLPK_integer *__qsiz, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__qstore, __CLPK_integer *__ldqs,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaed0);

int __TEMPLATE_FUNC(zlaed7)(__CLPK_integer *__n, __CLPK_integer *__cutpnt,
        __CLPK_integer *__qsiz, __CLPK_integer *__tlvls,
        __CLPK_integer *__curlvl, __CLPK_integer *__curpbm,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__q,
        __CLPK_integer *__ldq, __CLPK_doublereal *__rho,
        __CLPK_integer *__indxq, __CLPK_doublereal *__qstore,
        __CLPK_integer *__qptr, __CLPK_integer *__prmptr,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_doublereal *__givnum,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaed7);

int __TEMPLATE_FUNC(zlaed8)(__CLPK_integer *__k, __CLPK_integer *__n, __CLPK_integer *__qsiz,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__rho,
        __CLPK_integer *__cutpnt, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__dlamda, __CLPK_doublecomplex *__q2,
        __CLPK_integer *__ldq2, __CLPK_doublereal *__w, __CLPK_integer *__indxp,
        __CLPK_integer *__indx, __CLPK_integer *__indxq, __CLPK_integer *__perm,
        __CLPK_integer *__givptr, __CLPK_integer *__givcol,
        __CLPK_doublereal *__givnum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaed8);

int __TEMPLATE_FUNC(zlaein)(__CLPK_logical *__rightv, __CLPK_logical *__noinit,
        __CLPK_integer *__n, __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__v,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__rwork, __CLPK_doublereal *__eps3,
        __CLPK_doublereal *__smlnum,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaein);

int __TEMPLATE_FUNC(zlaesy)(__CLPK_doublecomplex *__a, __CLPK_doublecomplex *__b,
        __CLPK_doublecomplex *__c__, __CLPK_doublecomplex *__rt1,
        __CLPK_doublecomplex *__rt2, __CLPK_doublecomplex *__evscal,
        __CLPK_doublecomplex *__cs1,
        __CLPK_doublecomplex *__sn1)
__TEMPLATE_ALIAS(zlaesy);

int __TEMPLATE_FUNC(zlaev2)(__CLPK_doublecomplex *__a, __CLPK_doublecomplex *__b,
        __CLPK_doublecomplex *__c__, __CLPK_doublereal *__rt1,
        __CLPK_doublereal *__rt2, __CLPK_doublereal *__cs1,
        __CLPK_doublecomplex *__sn1)
__TEMPLATE_ALIAS(zlaev2);

int __TEMPLATE_FUNC(zlag2c)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__sa, __CLPK_integer *__ldsa,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlag2c);

int __TEMPLATE_FUNC(zlags2)(__CLPK_logical *__upper, __CLPK_doublereal *__a1,
        __CLPK_doublecomplex *__a2, __CLPK_doublereal *__a3,
        __CLPK_doublereal *__b1, __CLPK_doublecomplex *__b2,
        __CLPK_doublereal *__b3, __CLPK_doublereal *__csu,
        __CLPK_doublecomplex *__snu, __CLPK_doublereal *__csv,
        __CLPK_doublecomplex *__snv, __CLPK_doublereal *__csq,
        __CLPK_doublecomplex *__snq)
__TEMPLATE_ALIAS(zlags2);

int __TEMPLATE_FUNC(zlagtm)(char *__trans, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__alpha, __CLPK_doublecomplex *__dl,
        __CLPK_doublecomplex *__d__, __CLPK_doublecomplex *__du,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__beta, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(zlagtm);

int __TEMPLATE_FUNC(zlahef)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__w,
        __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlahef);

int __TEMPLATE_FUNC(zlahqr)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_integer *__iloz,
        __CLPK_integer *__ihiz, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlahqr);

int __TEMPLATE_FUNC(zlahr2)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__t,
        __CLPK_integer *__ldt, __CLPK_doublecomplex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(zlahr2);

int __TEMPLATE_FUNC(zlahrd)(__CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__nb,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__t,
        __CLPK_integer *__ldt, __CLPK_doublecomplex *__y,
        __CLPK_integer *__ldy)
__TEMPLATE_ALIAS(zlahrd);

int __TEMPLATE_FUNC(zlaic1)(__CLPK_integer *__job, __CLPK_integer *__j,
        __CLPK_doublecomplex *__x, __CLPK_doublereal *__sest,
        __CLPK_doublecomplex *__w, __CLPK_doublecomplex *__gamma,
        __CLPK_doublereal *__sestpr, __CLPK_doublecomplex *__s,
        __CLPK_doublecomplex *__c__)
__TEMPLATE_ALIAS(zlaic1);

int __TEMPLATE_FUNC(zlals0)(__CLPK_integer *__icompq, __CLPK_integer *__nl,
        __CLPK_integer *__nr, __CLPK_integer *__sqre, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__bx, __CLPK_integer *__ldbx,
        __CLPK_integer *__perm, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_doublereal *__givnum, __CLPK_integer *__ldgnum,
        __CLPK_doublereal *__poles, __CLPK_doublereal *__difl,
        __CLPK_doublereal *__difr, __CLPK_doublereal *__z__,
        __CLPK_integer *__k, __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlals0);

int __TEMPLATE_FUNC(zlalsa)(__CLPK_integer *__icompq, __CLPK_integer *__smlsiz,
        __CLPK_integer *__n, __CLPK_integer *__nrhs, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__bx,
        __CLPK_integer *__ldbx, __CLPK_doublereal *__u, __CLPK_integer *__ldu,
        __CLPK_doublereal *__vt, __CLPK_integer *__k, __CLPK_doublereal *__difl,
        __CLPK_doublereal *__difr, __CLPK_doublereal *__z__,
        __CLPK_doublereal *__poles, __CLPK_integer *__givptr,
        __CLPK_integer *__givcol, __CLPK_integer *__ldgcol,
        __CLPK_integer *__perm, __CLPK_doublereal *__givnum,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlalsa);

int __TEMPLATE_FUNC(zlalsd)(char *__uplo, __CLPK_integer *__smlsiz, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublereal *__rcond,
        __CLPK_integer *__rank, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlalsd);

__CLPK_doublereal __TEMPLATE_FUNC(zlangb)(char *__norm, __CLPK_integer *__n,
        __CLPK_integer *__kl, __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlangb);

__CLPK_doublereal __TEMPLATE_FUNC(zlange)(char *__norm, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlange);

__CLPK_doublereal __TEMPLATE_FUNC(zlangt)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublecomplex *__dl, __CLPK_doublecomplex *__d__,
        __CLPK_doublecomplex *__du)
__TEMPLATE_ALIAS(zlangt);

__CLPK_doublereal __TEMPLATE_FUNC(zlanhb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlanhb);

__CLPK_doublereal __TEMPLATE_FUNC(zlanhe)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlanhe);

__CLPK_doublereal __TEMPLATE_FUNC(zlanhf)(char *__norm, char *__transr, char *__uplo,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlanhf);

__CLPK_doublereal __TEMPLATE_FUNC(zlanhp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlanhp);

__CLPK_doublereal __TEMPLATE_FUNC(zlanhs)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlanhs);

__CLPK_doublereal __TEMPLATE_FUNC(zlanht)(char *__norm, __CLPK_integer *__n,
        __CLPK_doublereal *__d__,
        __CLPK_doublecomplex *__e)
__TEMPLATE_ALIAS(zlanht);

__CLPK_doublereal __TEMPLATE_FUNC(zlansb)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlansb);

__CLPK_doublereal __TEMPLATE_FUNC(zlansp)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlansp);

__CLPK_doublereal __TEMPLATE_FUNC(zlansy)(char *__norm, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlansy);

__CLPK_doublereal __TEMPLATE_FUNC(zlantb)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlantb);

__CLPK_doublereal __TEMPLATE_FUNC(zlantp)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlantp);

__CLPK_doublereal __TEMPLATE_FUNC(zlantr)(char *__norm, char *__uplo, char *__diag,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlantr);

int __TEMPLATE_FUNC(zlapll)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__y,
        __CLPK_integer *__incy,
        __CLPK_doublereal *__ssmin)
__TEMPLATE_ALIAS(zlapll);

int __TEMPLATE_FUNC(zlapmt)(__CLPK_logical *__forwrd, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_integer *__k)
__TEMPLATE_ALIAS(zlapmt);

int __TEMPLATE_FUNC(zlaqgb)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__kl,
        __CLPK_integer *__ku, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqgb);

int __TEMPLATE_FUNC(zlaqge)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__r__,
        __CLPK_doublereal *__c__, __CLPK_doublereal *__rowcnd,
        __CLPK_doublereal *__colcnd, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqge);

int __TEMPLATE_FUNC(zlaqhb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqhb);

int __TEMPLATE_FUNC(zlaqhe)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqhe);

int __TEMPLATE_FUNC(zlaqhp)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqhp);

int __TEMPLATE_FUNC(zlaqp2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__jpvt, __CLPK_doublecomplex *__tau,
        __CLPK_doublereal *__vn1, __CLPK_doublereal *__vn2,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlaqp2);

int __TEMPLATE_FUNC(zlaqps)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__offset,
        __CLPK_integer *__nb, __CLPK_integer *__kb, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__jpvt,
        __CLPK_doublecomplex *__tau, __CLPK_doublereal *__vn1,
        __CLPK_doublereal *__vn2, __CLPK_doublecomplex *__auxv,
        __CLPK_doublecomplex *__f,
        __CLPK_integer *__ldf)
__TEMPLATE_ALIAS(zlaqps);

int __TEMPLATE_FUNC(zlaqr0)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_integer *__iloz,
        __CLPK_integer *__ihiz, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaqr0);

int __TEMPLATE_FUNC(zlaqr1)(__CLPK_integer *__n, __CLPK_doublecomplex *__h__,
        __CLPK_integer *__ldh, __CLPK_doublecomplex *__s1,
        __CLPK_doublecomplex *__s2,
        __CLPK_doublecomplex *__v)
__TEMPLATE_ALIAS(zlaqr1);

int __TEMPLATE_FUNC(zlaqr2)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_doublecomplex *__h__,
        __CLPK_integer *__ldh, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ns, __CLPK_integer *__nd, __CLPK_doublecomplex *__sh,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv, __CLPK_integer *__nh,
        __CLPK_doublecomplex *__t, __CLPK_integer *__ldt, __CLPK_integer *__nv,
        __CLPK_doublecomplex *__wv, __CLPK_integer *__ldwv,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(zlaqr2);

int __TEMPLATE_FUNC(zlaqr3)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ktop, __CLPK_integer *__kbot,
        __CLPK_integer *__nw, __CLPK_doublecomplex *__h__,
        __CLPK_integer *__ldh, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ns, __CLPK_integer *__nd, __CLPK_doublecomplex *__sh,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv, __CLPK_integer *__nh,
        __CLPK_doublecomplex *__t, __CLPK_integer *__ldt, __CLPK_integer *__nv,
        __CLPK_doublecomplex *__wv, __CLPK_integer *__ldwv,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork)
__TEMPLATE_ALIAS(zlaqr3);

int __TEMPLATE_FUNC(zlaqr4)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__h__, __CLPK_integer *__ldh,
        __CLPK_doublecomplex *__w, __CLPK_integer *__iloz,
        __CLPK_integer *__ihiz, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlaqr4);

int __TEMPLATE_FUNC(zlaqr5)(__CLPK_logical *__wantt, __CLPK_logical *__wantz,
        __CLPK_integer *__kacc22, __CLPK_integer *__n, __CLPK_integer *__ktop,
        __CLPK_integer *__kbot, __CLPK_integer *__nshfts,
        __CLPK_doublecomplex *__s, __CLPK_doublecomplex *__h__,
        __CLPK_integer *__ldh, __CLPK_integer *__iloz, __CLPK_integer *__ihiz,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__u, __CLPK_integer *__ldu, __CLPK_integer *__nv,
        __CLPK_doublecomplex *__wv, __CLPK_integer *__ldwv,
        __CLPK_integer *__nh, __CLPK_doublecomplex *__wh,
        __CLPK_integer *__ldwh)
__TEMPLATE_ALIAS(zlaqr5);

int __TEMPLATE_FUNC(zlaqsb)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqsb);

int __TEMPLATE_FUNC(zlaqsp)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqsp);

int __TEMPLATE_FUNC(zlaqsy)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        char *__equed)
__TEMPLATE_ALIAS(zlaqsy);

int __TEMPLATE_FUNC(zlar1v)(__CLPK_integer *__n, __CLPK_integer *__b1, __CLPK_integer *__bn,
        __CLPK_doublereal *__lambda, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__l, __CLPK_doublereal *__ld,
        __CLPK_doublereal *__lld, __CLPK_doublereal *__pivmin,
        __CLPK_doublereal *__gaptol, __CLPK_doublecomplex *__z__,
        __CLPK_logical *__wantnc, __CLPK_integer *__negcnt,
        __CLPK_doublereal *__ztz, __CLPK_doublereal *__mingma,
        __CLPK_integer *__r__, __CLPK_integer *__isuppz,
        __CLPK_doublereal *__nrminv, __CLPK_doublereal *__resid,
        __CLPK_doublereal *__rqcorr,
        __CLPK_doublereal *__work)
__TEMPLATE_ALIAS(zlar1v);

int __TEMPLATE_FUNC(zlar2v)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_doublecomplex *__y, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__incx, __CLPK_doublereal *__c__,
        __CLPK_doublecomplex *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(zlar2v);

int __TEMPLATE_FUNC(zlarcm)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__rwork)
__TEMPLATE_ALIAS(zlarcm);

int __TEMPLATE_FUNC(zlarf)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__v, __CLPK_integer *__incv,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlarf);

int __TEMPLATE_FUNC(zlarfb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(zlarfb);

int __TEMPLATE_FUNC(zlarfg)(__CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__x, __CLPK_integer *__incx,
        __CLPK_doublecomplex *__tau)
__TEMPLATE_ALIAS(zlarfg);

int __TEMPLATE_FUNC(zlarfp)(__CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__x, __CLPK_integer *__incx,
        __CLPK_doublecomplex *__tau)
__TEMPLATE_ALIAS(zlarfp);

int __TEMPLATE_FUNC(zlarft)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(zlarft);

int __TEMPLATE_FUNC(zlarfx)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__v, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlarfx);

int __TEMPLATE_FUNC(zlargv)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__y,
        __CLPK_integer *__incy, __CLPK_doublereal *__c__,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(zlargv);

int __TEMPLATE_FUNC(zlarnv)(__CLPK_integer *__idist, __CLPK_integer *__iseed,
        __CLPK_integer *__n,
        __CLPK_doublecomplex *__x)
__TEMPLATE_ALIAS(zlarnv);

int __TEMPLATE_FUNC(zlarrv)(__CLPK_integer *__n, __CLPK_doublereal *__vl,
        __CLPK_doublereal *__vu, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__l, __CLPK_doublereal *__pivmin,
        __CLPK_integer *__isplit, __CLPK_integer *__m, __CLPK_integer *__dol,
        __CLPK_integer *__dou, __CLPK_doublereal *__minrgp,
        __CLPK_doublereal *__rtol1, __CLPK_doublereal *__rtol2,
        __CLPK_doublereal *__w, __CLPK_doublereal *__werr,
        __CLPK_doublereal *__wgap, __CLPK_integer *__iblock,
        __CLPK_integer *__indexw, __CLPK_doublereal *__gers,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__isuppz, __CLPK_doublereal *__work,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlarrv);

int __TEMPLATE_FUNC(zlarscl2)(__CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(zlarscl2);

int __TEMPLATE_FUNC(zlartg)(__CLPK_doublecomplex *__f, __CLPK_doublecomplex *__g,
        __CLPK_doublereal *__cs, __CLPK_doublecomplex *__sn,
        __CLPK_doublecomplex *__r__)
__TEMPLATE_ALIAS(zlartg);

int __TEMPLATE_FUNC(zlartv)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__y,
        __CLPK_integer *__incy, __CLPK_doublereal *__c__,
        __CLPK_doublecomplex *__s,
        __CLPK_integer *__incc)
__TEMPLATE_ALIAS(zlartv);

int __TEMPLATE_FUNC(zlarz)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__l, __CLPK_doublecomplex *__v, __CLPK_integer *__incv,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlarz);

int __TEMPLATE_FUNC(zlarzb)(char *__side, char *__trans, char *__direct, char *__storev,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__ldwork)
__TEMPLATE_ALIAS(zlarzb);

int __TEMPLATE_FUNC(zlarzt)(char *__direct, char *__storev, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__t,
        __CLPK_integer *__ldt)
__TEMPLATE_ALIAS(zlarzt);

int __TEMPLATE_FUNC(zlascl)(char *__type__, __CLPK_integer *__kl, __CLPK_integer *__ku,
        __CLPK_doublereal *__cfrom, __CLPK_doublereal *__cto,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlascl);

int __TEMPLATE_FUNC(zlascl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublecomplex *__x,
        __CLPK_integer *__ldx)
__TEMPLATE_ALIAS(zlascl2);

int __TEMPLATE_FUNC(zlaset)(char *__uplo, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(zlaset);

int __TEMPLATE_FUNC(zlasr)(char *__side, char *__pivot, char *__direct, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__c__, __CLPK_doublereal *__s,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(zlasr);

int __TEMPLATE_FUNC(zlassq)(__CLPK_integer *__n, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__sumsq)
__TEMPLATE_ALIAS(zlassq);

int __TEMPLATE_FUNC(zlaswp)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__k1, __CLPK_integer *__k2,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__incx)
__TEMPLATE_ALIAS(zlaswp);

int __TEMPLATE_FUNC(zlasyf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_integer *__kb, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__w,
        __CLPK_integer *__ldw,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlasyf);

int __TEMPLATE_FUNC(zlat2c)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_complex *__sa, __CLPK_integer *__ldsa,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlat2c);

int __TEMPLATE_FUNC(zlatbs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__x,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlatbs);

int __TEMPLATE_FUNC(zlatdf)(__CLPK_integer *__ijob, __CLPK_integer *__n,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublecomplex *__rhs, __CLPK_doublereal *__rdsum,
        __CLPK_doublereal *__rdscal, __CLPK_integer *__ipiv,
        __CLPK_integer *__jpiv)
__TEMPLATE_ALIAS(zlatdf);

int __TEMPLATE_FUNC(zlatps)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__x, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlatps);

int __TEMPLATE_FUNC(zlatrd)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nb,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__w,
        __CLPK_integer *__ldw)
__TEMPLATE_ALIAS(zlatrd);

int __TEMPLATE_FUNC(zlatrs)(char *__uplo, char *__trans, char *__diag, char *__normin,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__x, __CLPK_doublereal *__scale,
        __CLPK_doublereal *__cnorm,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlatrs);

int __TEMPLATE_FUNC(zlatrz)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__l,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlatrz);

int __TEMPLATE_FUNC(zlatzm)(char *__side, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__v, __CLPK_integer *__incv,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c1,
        __CLPK_doublecomplex *__c2, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work)
__TEMPLATE_ALIAS(zlatzm);

int __TEMPLATE_FUNC(zlauu2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlauu2);

int __TEMPLATE_FUNC(zlauum)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zlauum);

int __TEMPLATE_FUNC(zpbcon)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbcon);

int __TEMPLATE_FUNC(zpbequ)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbequ);

int __TEMPLATE_FUNC(zpbrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__afb,
        __CLPK_integer *__ldafb, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbrfs);

int __TEMPLATE_FUNC(zpbstf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbstf);

int __TEMPLATE_FUNC(zpbsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbsv);

int __TEMPLATE_FUNC(zpbsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublecomplex *__afb, __CLPK_integer *__ldafb, char *__equed,
        __CLPK_doublereal *__s, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbsvx);

int __TEMPLATE_FUNC(zpbtf2)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbtf2);

int __TEMPLATE_FUNC(zpbtrf)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbtrf);

int __TEMPLATE_FUNC(zpbtrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__kd,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpbtrs);

int __TEMPLATE_FUNC(zpftrf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpftrf);

int __TEMPLATE_FUNC(zpftri)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpftri);

int __TEMPLATE_FUNC(zpftrs)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpftrs);

int __TEMPLATE_FUNC(zpocon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpocon);

int __TEMPLATE_FUNC(zpoequ)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpoequ);

int __TEMPLATE_FUNC(zpoequb)(__CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpoequb);

int __TEMPLATE_FUNC(zporfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__af, __CLPK_integer *__ldaf,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zporfs);

int __TEMPLATE_FUNC(zposv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zposv);

int __TEMPLATE_FUNC(zposvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__af,
        __CLPK_integer *__ldaf, char *__equed, __CLPK_doublereal *__s,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zposvx);

int __TEMPLATE_FUNC(zpotf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpotf2);

int __TEMPLATE_FUNC(zpotrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpotrf);

int __TEMPLATE_FUNC(zpotri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpotri);

int __TEMPLATE_FUNC(zpotrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpotrs);

int __TEMPLATE_FUNC(zppcon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zppcon);

int __TEMPLATE_FUNC(zppequ)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublereal *__s, __CLPK_doublereal *__scond,
        __CLPK_doublereal *__amax,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zppequ);

int __TEMPLATE_FUNC(zpprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__afp,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpprfs);

int __TEMPLATE_FUNC(zppsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zppsv);

int __TEMPLATE_FUNC(zppsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__afp, char *__equed, __CLPK_doublereal *__s,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zppsvx);

int __TEMPLATE_FUNC(zpptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpptrf);

int __TEMPLATE_FUNC(zpptri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpptri);

int __TEMPLATE_FUNC(zpptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpptrs);

int __TEMPLATE_FUNC(zpstf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_doublereal *__tol, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpstf2);

int __TEMPLATE_FUNC(zpstrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__piv, __CLPK_integer *__rank,
        __CLPK_doublereal *__tol, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpstrf);

int __TEMPLATE_FUNC(zptcon)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublecomplex *__e, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zptcon);

int __TEMPLATE_FUNC(zpteqr)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpteqr);

int __TEMPLATE_FUNC(zptrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__e,
        __CLPK_doublereal *__df, __CLPK_doublecomplex *__ef,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zptrfs);

int __TEMPLATE_FUNC(zptsv)(__CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__e,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zptsv);

int __TEMPLATE_FUNC(zptsvx)(char *__fact, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__e,
        __CLPK_doublereal *__df, __CLPK_doublecomplex *__ef,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zptsvx);

int __TEMPLATE_FUNC(zpttrf)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublecomplex *__e,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpttrf);

int __TEMPLATE_FUNC(zpttrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublereal *__d__, __CLPK_doublecomplex *__e,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zpttrs);

int __TEMPLATE_FUNC(zptts2)(__CLPK_integer *__iuplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublereal *__d__,
        __CLPK_doublecomplex *__e, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(zptts2);

int __TEMPLATE_FUNC(zrot)(__CLPK_integer *__n, __CLPK_doublecomplex *__cx,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__cy,
        __CLPK_integer *__incy, __CLPK_doublereal *__c__,
        __CLPK_doublecomplex *__s)
__TEMPLATE_ALIAS(zrot);

int __TEMPLATE_FUNC(zspcon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublereal *__anorm,
        __CLPK_doublereal *__rcond, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zspcon);

int __TEMPLATE_FUNC(zspmv)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__x,
        __CLPK_integer *__incx, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__y,
        __CLPK_integer *__incy)
__TEMPLATE_ALIAS(zspmv);

int __TEMPLATE_FUNC(zspr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__x, __CLPK_integer *__incx,
        __CLPK_doublecomplex *__ap)
__TEMPLATE_ALIAS(zspr);

int __TEMPLATE_FUNC(zsprfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__afp,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsprfs);

int __TEMPLATE_FUNC(zspsv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zspsv);

int __TEMPLATE_FUNC(zspsvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__afp, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zspsvx);

int __TEMPLATE_FUNC(zsptrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsptrf);

int __TEMPLATE_FUNC(zsptri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsptri);

int __TEMPLATE_FUNC(zsptrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ap, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsptrs);

int __TEMPLATE_FUNC(zstedc)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__lrwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zstedc);

int __TEMPLATE_FUNC(zstegr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_doublereal *__abstol, __CLPK_integer *__m,
        __CLPK_doublereal *__w, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_integer *__isuppz,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork, __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zstegr);

int __TEMPLATE_FUNC(zstein)(__CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_integer *__iblock, __CLPK_integer *__isplit,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_doublereal *__work, __CLPK_integer *__iwork,
        __CLPK_integer *__ifail,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zstein);

int __TEMPLATE_FUNC(zstemr)(char *__jobz, char *__range, __CLPK_integer *__n,
        __CLPK_doublereal *__d__, __CLPK_doublereal *__e,
        __CLPK_doublereal *__vl, __CLPK_doublereal *__vu, __CLPK_integer *__il,
        __CLPK_integer *__iu, __CLPK_integer *__m, __CLPK_doublereal *__w,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__nzc, __CLPK_integer *__isuppz,
        __CLPK_logical *__tryrac, __CLPK_doublereal *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zstemr);

int __TEMPLATE_FUNC(zsteqr)(char *__compz, __CLPK_integer *__n, __CLPK_doublereal *__d__,
        __CLPK_doublereal *__e, __CLPK_doublecomplex *__z__,
        __CLPK_integer *__ldz, __CLPK_doublereal *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsteqr);

int __TEMPLATE_FUNC(zsycon)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublereal *__anorm, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsycon);

int __TEMPLATE_FUNC(zsyequb)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublereal *__s,
        __CLPK_doublereal *__scond, __CLPK_doublereal *__amax,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsyequb);

int __TEMPLATE_FUNC(zsymv)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__x, __CLPK_integer *__incx,
        __CLPK_doublecomplex *__beta, __CLPK_doublecomplex *__y,
        __CLPK_integer *__incy)
__TEMPLATE_ALIAS(zsymv);

int __TEMPLATE_FUNC(zsyr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__alpha,
        __CLPK_doublecomplex *__x, __CLPK_integer *__incx,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda)
__TEMPLATE_ALIAS(zsyr);

int __TEMPLATE_FUNC(zsyrfs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__af, __CLPK_integer *__ldaf,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsyrfs);

int __TEMPLATE_FUNC(zsysv)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsysv);

int __TEMPLATE_FUNC(zsysvx)(char *__fact, char *__uplo, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__af,
        __CLPK_integer *__ldaf, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__rcond, __CLPK_doublereal *__ferr,
        __CLPK_doublereal *__berr, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsysvx);

int __TEMPLATE_FUNC(zsytf2)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsytf2);

int __TEMPLATE_FUNC(zsytrf)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsytrf);

int __TEMPLATE_FUNC(zsytri)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_integer *__ipiv,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsytri);

int __TEMPLATE_FUNC(zsytrs)(char *__uplo, __CLPK_integer *__n, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__ipiv, __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zsytrs);

int __TEMPLATE_FUNC(ztbcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_doublecomplex *__ab,
        __CLPK_integer *__ldab, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztbcon);

int __TEMPLATE_FUNC(ztbrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztbrfs);

int __TEMPLATE_FUNC(ztbtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__kd, __CLPK_integer *__nrhs,
        __CLPK_doublecomplex *__ab, __CLPK_integer *__ldab,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztbtrs);

int __TEMPLATE_FUNC(ztfsm)(char *__transr, char *__side, char *__uplo, char *__trans,
        char *__diag, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__a,
        __CLPK_doublecomplex *__b,
        __CLPK_integer *__ldb)
__TEMPLATE_ALIAS(ztfsm);

int __TEMPLATE_FUNC(ztftri)(char *__transr, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztftri);

int __TEMPLATE_FUNC(ztfttp)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__arf, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztfttp);

int __TEMPLATE_FUNC(ztfttr)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__arf, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztfttr);

int __TEMPLATE_FUNC(ztgevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__s, __CLPK_integer *__lds,
        __CLPK_doublecomplex *__p, __CLPK_integer *__ldp,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgevc);

int __TEMPLATE_FUNC(ztgex2)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__j1,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgex2);

int __TEMPLATE_FUNC(ztgexc)(__CLPK_logical *__wantq, __CLPK_logical *__wantz,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgexc);

int __TEMPLATE_FUNC(ztgsen)(__CLPK_integer *__ijob, __CLPK_logical *__wantq,
        __CLPK_logical *__wantz, __CLPK_logical *__select, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__alpha, __CLPK_doublecomplex *__beta,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__z__, __CLPK_integer *__ldz, __CLPK_integer *__m,
        __CLPK_doublereal *__pl, __CLPK_doublereal *__pr,
        __CLPK_doublereal *__dif, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__liwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgsen);

int __TEMPLATE_FUNC(ztgsja)(char *__jobu, char *__jobv, char *__jobq, __CLPK_integer *__m,
        __CLPK_integer *__p, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_integer *__l, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublereal *__tola, __CLPK_doublereal *__tolb,
        __CLPK_doublereal *__alpha, __CLPK_doublereal *__beta,
        __CLPK_doublecomplex *__u, __CLPK_integer *__ldu,
        __CLPK_doublecomplex *__v, __CLPK_integer *__ldv,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__work, __CLPK_integer *__ncycle,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgsja);

int __TEMPLATE_FUNC(ztgsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__s, __CLPK_doublereal *__dif, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork, __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgsna);

int __TEMPLATE_FUNC(ztgsy2)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__d__, __CLPK_integer *__ldd,
        __CLPK_doublecomplex *__e, __CLPK_integer *__lde,
        __CLPK_doublecomplex *__f, __CLPK_integer *__ldf,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__rdsum,
        __CLPK_doublereal *__rdscal,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgsy2);

int __TEMPLATE_FUNC(ztgsyl)(char *__trans, __CLPK_integer *__ijob, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__d__, __CLPK_integer *__ldd,
        __CLPK_doublecomplex *__e, __CLPK_integer *__lde,
        __CLPK_doublecomplex *__f, __CLPK_integer *__ldf,
        __CLPK_doublereal *__scale, __CLPK_doublereal *__dif,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__iwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztgsyl);

int __TEMPLATE_FUNC(ztpcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublereal *__rcond,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztpcon);

int __TEMPLATE_FUNC(ztprfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztprfs);

int __TEMPLATE_FUNC(ztptri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztptri);

int __TEMPLATE_FUNC(ztptrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztptrs);

int __TEMPLATE_FUNC(ztpttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__ap, __CLPK_doublecomplex *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztpttf);

int __TEMPLATE_FUNC(ztpttr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztpttr);

int __TEMPLATE_FUNC(ztrcon)(char *__norm, char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__rcond, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrcon);

int __TEMPLATE_FUNC(ztrevc)(char *__side, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_integer *__mm, __CLPK_integer *__m, __CLPK_doublecomplex *__work,
        __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrevc);

int __TEMPLATE_FUNC(ztrexc)(char *__compq, __CLPK_integer *__n, __CLPK_doublecomplex *__t,
        __CLPK_integer *__ldt, __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_integer *__ifst, __CLPK_integer *__ilst,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrexc);

int __TEMPLATE_FUNC(ztrrfs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__x, __CLPK_integer *__ldx,
        __CLPK_doublereal *__ferr, __CLPK_doublereal *__berr,
        __CLPK_doublecomplex *__work, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrrfs);

int __TEMPLATE_FUNC(ztrsen)(char *__job, char *__compq, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__q, __CLPK_integer *__ldq,
        __CLPK_doublecomplex *__w, __CLPK_integer *__m, __CLPK_doublereal *__s,
        __CLPK_doublereal *__sep, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrsen);

int __TEMPLATE_FUNC(ztrsna)(char *__job, char *__howmny, __CLPK_logical *__select,
        __CLPK_integer *__n, __CLPK_doublecomplex *__t, __CLPK_integer *__ldt,
        __CLPK_doublecomplex *__vl, __CLPK_integer *__ldvl,
        __CLPK_doublecomplex *__vr, __CLPK_integer *__ldvr,
        __CLPK_doublereal *__s, __CLPK_doublereal *__sep, __CLPK_integer *__mm,
        __CLPK_integer *__m, __CLPK_doublecomplex *__work,
        __CLPK_integer *__ldwork, __CLPK_doublereal *__rwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrsna);

int __TEMPLATE_FUNC(ztrsyl)(char *__trana, char *__tranb, __CLPK_integer *__isgn,
        __CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublereal *__scale,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrsyl);

int __TEMPLATE_FUNC(ztrti2)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrti2);

int __TEMPLATE_FUNC(ztrtri)(char *__uplo, char *__diag, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrtri);

int __TEMPLATE_FUNC(ztrtrs)(char *__uplo, char *__trans, char *__diag, __CLPK_integer *__n,
        __CLPK_integer *__nrhs, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__b, __CLPK_integer *__ldb,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrtrs);

int __TEMPLATE_FUNC(ztrttf)(char *__transr, char *__uplo, __CLPK_integer *__n,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__arf,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrttf);

int __TEMPLATE_FUNC(ztrttp)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__ap,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztrttp);

int __TEMPLATE_FUNC(ztzrqf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztzrqf);

int __TEMPLATE_FUNC(ztzrzf)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(ztzrzf);

int __TEMPLATE_FUNC(zung2l)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zung2l);

int __TEMPLATE_FUNC(zung2r)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zung2r);

int __TEMPLATE_FUNC(zungbr)(char *__vect, __CLPK_integer *__m, __CLPK_integer *__n,
        __CLPK_integer *__k, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungbr);

int __TEMPLATE_FUNC(zunghr)(__CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunghr);

int __TEMPLATE_FUNC(zungl2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungl2);

int __TEMPLATE_FUNC(zunglq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunglq);

int __TEMPLATE_FUNC(zungql)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungql);

int __TEMPLATE_FUNC(zungqr)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungqr);

int __TEMPLATE_FUNC(zungr2)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungr2);

int __TEMPLATE_FUNC(zungrq)(__CLPK_integer *__m, __CLPK_integer *__n, __CLPK_integer *__k,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungrq);

int __TEMPLATE_FUNC(zungtr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zungtr);

int __TEMPLATE_FUNC(zunm2l)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunm2l);

int __TEMPLATE_FUNC(zunm2r)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunm2r);

int __TEMPLATE_FUNC(zunmbr)(char *__vect, char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmbr);

int __TEMPLATE_FUNC(zunmhr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__ilo, __CLPK_integer *__ihi,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmhr);

int __TEMPLATE_FUNC(zunml2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunml2);

int __TEMPLATE_FUNC(zunmlq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmlq);

int __TEMPLATE_FUNC(zunmql)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmql);

int __TEMPLATE_FUNC(zunmqr)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmqr);

int __TEMPLATE_FUNC(zunmr2)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmr2);

int __TEMPLATE_FUNC(zunmr3)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmr3);

int __TEMPLATE_FUNC(zunmrq)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_doublecomplex *__a,
        __CLPK_integer *__lda, __CLPK_doublecomplex *__tau,
        __CLPK_doublecomplex *__c__, __CLPK_integer *__ldc,
        __CLPK_doublecomplex *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmrq);

int __TEMPLATE_FUNC(zunmrz)(char *__side, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_integer *__k, __CLPK_integer *__l,
        __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmrz);

int __TEMPLATE_FUNC(zunmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__a, __CLPK_integer *__lda,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublecomplex *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zunmtr);

int __TEMPLATE_FUNC(zupgtr)(char *__uplo, __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__q,
        __CLPK_integer *__ldq, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zupgtr);

int __TEMPLATE_FUNC(zupmtr)(char *__side, char *__uplo, char *__trans, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublecomplex *__ap,
        __CLPK_doublecomplex *__tau, __CLPK_doublecomplex *__c__,
        __CLPK_integer *__ldc, __CLPK_doublecomplex *__work,
        __CLPK_integer *__info)
__TEMPLATE_ALIAS(zupmtr);

int __TEMPLATE_FUNC(dlamc1)(__CLPK_integer *__beta, __CLPK_integer *__t, __CLPK_logical *__rnd,
        __CLPK_logical *__ieee1)
__TEMPLATE_ALIAS(dlamc1);

int __TEMPLATE_FUNC(ilaver)(__CLPK_integer *__vers_major__, __CLPK_integer *__vers_minor__,
        __CLPK_integer *__vers_patch__)
__TEMPLATE_ALIAS(ilaver);

__CLPK_doublereal __TEMPLATE_FUNC(slamch)(char *__cmach)
__TEMPLATE_ALIAS(slamch);

int __TEMPLATE_FUNC(slamc1)(__CLPK_integer *__beta, __CLPK_integer *__t, __CLPK_logical *__rnd,
        __CLPK_logical *__ieee1)
__TEMPLATE_ALIAS(slamc1);

int __TEMPLATE_FUNC(slamc2)(__CLPK_integer *__beta, __CLPK_integer *__t, __CLPK_logical *__rnd,
        __CLPK_real *__eps, __CLPK_integer *__emin, __CLPK_real *__rmin,
        __CLPK_integer *__emax,
        __CLPK_real *__rmax)
__TEMPLATE_ALIAS(slamc2);

__CLPK_doublereal __TEMPLATE_FUNC(slamc3)(__CLPK_real *__a,
        __CLPK_real *__b)
__TEMPLATE_ALIAS(slamc3);

int __TEMPLATE_FUNC(slamc4)(__CLPK_integer *__emin, __CLPK_real *__start,
        __CLPK_integer *__base)
__TEMPLATE_ALIAS(slamc4);

int __TEMPLATE_FUNC(slamc5)(__CLPK_integer *__beta, __CLPK_integer *__p, __CLPK_integer *__emin,
        __CLPK_logical *__ieee, __CLPK_integer *__emax,
        __CLPK_real *__rmax)
__TEMPLATE_ALIAS(slamc5);


__CLPK_doublereal __TEMPLATE_FUNC(dlamch)(char *__cmach)
__TEMPLATE_ALIAS(dlamch);

int __TEMPLATE_FUNC(dlamc1)(__CLPK_integer *__beta, __CLPK_integer *__t, __CLPK_logical *__rnd,
        __CLPK_logical *__ieee1)
__TEMPLATE_ALIAS(dlamc1);

int __TEMPLATE_FUNC(dlamc2)(__CLPK_integer *__beta, __CLPK_integer *__t, __CLPK_logical *__rnd,
        __CLPK_doublereal *__eps, __CLPK_integer *__emin,
        __CLPK_doublereal *__rmin, __CLPK_integer *__emax,
        __CLPK_doublereal *__rmax)
__TEMPLATE_ALIAS(dlamc2);

__CLPK_doublereal __TEMPLATE_FUNC(dlamc3)(__CLPK_doublereal *__a,
        __CLPK_doublereal *__b)
__TEMPLATE_ALIAS(dlamc3);

int __TEMPLATE_FUNC(dlamc4)(__CLPK_integer *__emin, __CLPK_doublereal *__start,
        __CLPK_integer *__base)
__TEMPLATE_ALIAS(dlamc4);

int __TEMPLATE_FUNC(dlamc5)(__CLPK_integer *__beta, __CLPK_integer *__p, __CLPK_integer *__emin,
        __CLPK_logical *__ieee, __CLPK_integer *__emax,
        __CLPK_doublereal *__rmax)
__TEMPLATE_ALIAS(dlamc5);

__CLPK_integer __TEMPLATE_FUNC(ilaenv)(__CLPK_integer *__ispec, char *__name__, char *__opts,
        __CLPK_integer *__n1, __CLPK_integer *__n2, __CLPK_integer *__n3,
        __CLPK_integer *__n4)
__TEMPLATE_ALIAS(ilaenv);


#ifdef __cplusplus
}
#endif
