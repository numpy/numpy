/*
NOTE: This is generated code. Look in Misc/lapack_lite for information on
      remaking this file.
*/
#include "f2c.h"

#ifdef HAVE_CONFIG
#include "config.h"
#else
extern doublereal dlamch_(char *);
#define EPSILON dlamch_("Epsilon")
#define SAFEMINIMUM dlamch_("Safe minimum")
#define PRECISION dlamch_("Precision")
#define BASE dlamch_("Base")
#endif

extern doublereal dlapy2_(doublereal *x, doublereal *y);

/*
f2c knows the exact rules for precedence, and so omits parentheses where not
strictly necessary. Since this is generated code, we don't really care if
it's readable, and we know what is written is correct. So don't warn about
them.
*/
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#endif


/* Table of constant values */

static integer c__9 = 9;
static integer c__0 = 0;
static real c_b15 = 1.f;
static integer c__1 = 1;
static real c_b29 = 0.f;
static doublereal c_b94 = -.125;
static real c_b151 = -1.f;
static integer c_n1 = -1;
static integer c__3 = 3;
static integer c__2 = 2;
static integer c__8 = 8;
static integer c__4 = 4;
static integer c__65 = 65;
static integer c__15 = 15;
static logical c_false = FALSE_;
static integer c__10 = 10;
static integer c__11 = 11;
static real c_b2489 = 2.f;
static logical c_true = TRUE_;

/* Subroutine */ int sbdsdc_(char *uplo, char *compq, integer *n, real *d__,
	real *e, real *u, integer *ldu, real *vt, integer *ldvt, real *q,
	integer *iq, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double r_sign(real *, real *), log(doublereal);

    /* Local variables */
    static integer i__, j, k;
    static real p, r__;
    static integer z__, ic, ii, kk;
    static real cs;
    static integer is, iu;
    static real sn;
    static integer nm1;
    static real eps;
    static integer ivt, difl, difr, ierr, perm, mlvl, sqre;
    extern logical lsame_(char *, char *);
    static integer poles;
    extern /* Subroutine */ int slasr_(char *, char *, char *, integer *,
	    integer *, real *, real *, real *, integer *);
    static integer iuplo, nsize, start;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), sswap_(integer *, real *, integer *, real *, integer *
	    ), slasd0_(integer *, integer *, real *, real *, real *, integer *
	    , real *, integer *, integer *, integer *, real *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int slasda_(integer *, integer *, integer *,
	    integer *, real *, real *, real *, integer *, real *, integer *,
	    real *, real *, real *, real *, integer *, integer *, integer *,
	    integer *, real *, real *, real *, real *, integer *, integer *),
	    xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    static integer givcol;
    extern /* Subroutine */ int slasdq_(char *, integer *, integer *, integer
	    *, integer *, integer *, real *, real *, real *, integer *, real *
	    , integer *, real *, integer *, real *, integer *);
    static integer icompq;
    extern /* Subroutine */ int slaset_(char *, integer *, integer *, real *,
	    real *, real *, integer *), slartg_(real *, real *, real *
	    , real *, real *);
    static real orgnrm;
    static integer givnum;
    extern doublereal slanst_(char *, integer *, real *, real *);
    static integer givptr, qstart, smlsiz, wstart, smlszp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       December 1, 1999


    Purpose
    =======

    SBDSDC computes the singular value decomposition (SVD) of a real
    N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
    using a divide and conquer method, where S is a diagonal matrix
    with non-negative diagonal elements (the singular values of B), and
    U and VT are orthogonal matrices of left and right singular vectors,
    respectively. SBDSDC can be used to compute all singular values,
    and optionally, singular vectors or singular vectors in compact form.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.  See SLASD3 for details.

    The code currently call SLASDQ if singular values only are desired.
    However, it can be slightly modified to compute singular values
    using the divide and conquer method.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  B is upper bidiagonal.
            = 'L':  B is lower bidiagonal.

    COMPQ   (input) CHARACTER*1
            Specifies whether singular vectors are to be computed
            as follows:
            = 'N':  Compute singular values only;
            = 'P':  Compute singular values and compute singular
                    vectors in compact form;
            = 'I':  Compute singular values and singular vectors.

    N       (input) INTEGER
            The order of the matrix B.  N >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the n diagonal elements of the bidiagonal matrix B.
            On exit, if INFO=0, the singular values of B.

    E       (input/output) REAL array, dimension (N)
            On entry, the elements of E contain the offdiagonal
            elements of the bidiagonal matrix whose SVD is desired.
            On exit, E has been destroyed.

    U       (output) REAL array, dimension (LDU,N)
            If  COMPQ = 'I', then:
               On exit, if INFO = 0, U contains the left singular vectors
               of the bidiagonal matrix.
            For other values of COMPQ, U is not referenced.

    LDU     (input) INTEGER
            The leading dimension of the array U.  LDU >= 1.
            If singular vectors are desired, then LDU >= max( 1, N ).

    VT      (output) REAL array, dimension (LDVT,N)
            If  COMPQ = 'I', then:
               On exit, if INFO = 0, VT' contains the right singular
               vectors of the bidiagonal matrix.
            For other values of COMPQ, VT is not referenced.

    LDVT    (input) INTEGER
            The leading dimension of the array VT.  LDVT >= 1.
            If singular vectors are desired, then LDVT >= max( 1, N ).

    Q       (output) REAL array, dimension (LDQ)
            If  COMPQ = 'P', then:
               On exit, if INFO = 0, Q and IQ contain the left
               and right singular vectors in a compact form,
               requiring O(N log N) space instead of 2*N**2.
               In particular, Q contains all the REAL data in
               LDQ >= N*(11 + 2*SMLSIZ + 8*INT(LOG_2(N/(SMLSIZ+1))))
               words of memory, where SMLSIZ is returned by ILAENV and
               is equal to the maximum size of the subproblems at the
               bottom of the computation tree (usually about 25).
            For other values of COMPQ, Q is not referenced.

    IQ      (output) INTEGER array, dimension (LDIQ)
            If  COMPQ = 'P', then:
               On exit, if INFO = 0, Q and IQ contain the left
               and right singular vectors in a compact form,
               requiring O(N log N) space instead of 2*N**2.
               In particular, IQ contains all INTEGER data in
               LDIQ >= N*(3 + 3*INT(LOG_2(N/(SMLSIZ+1))))
               words of memory, where SMLSIZ is returned by ILAENV and
               is equal to the maximum size of the subproblems at the
               bottom of the computation tree (usually about 25).
            For other values of COMPQ, IQ is not referenced.

    WORK    (workspace) REAL array, dimension (LWORK)
            If COMPQ = 'N' then LWORK >= (4 * N).
            If COMPQ = 'P' then LWORK >= (6 * N).
            If COMPQ = 'I' then LWORK >= (3 * N**2 + 4 * N).

    IWORK   (workspace) INTEGER array, dimension (8*N)

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The algorithm failed to compute an singular value.
                  The update process of divide and conquer failed.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --q;
    --iq;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    iuplo = 0;
    if (lsame_(uplo, "U")) {
	iuplo = 1;
    }
    if (lsame_(uplo, "L")) {
	iuplo = 2;
    }
    if (lsame_(compq, "N")) {
	icompq = 0;
    } else if (lsame_(compq, "P")) {
	icompq = 1;
    } else if (lsame_(compq, "I")) {
	icompq = 2;
    } else {
	icompq = -1;
    }
    if (iuplo == 0) {
	*info = -1;
    } else if (icompq < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ldu < 1 || icompq == 2 && *ldu < *n) {
	*info = -7;
    } else if (*ldvt < 1 || icompq == 2 && *ldvt < *n) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SBDSDC", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    smlsiz = ilaenv_(&c__9, "SBDSDC", " ", &c__0, &c__0, &c__0, &c__0, (
	    ftnlen)6, (ftnlen)1);
    if (*n == 1) {
	if (icompq == 1) {
	    q[1] = r_sign(&c_b15, &d__[1]);
	    q[smlsiz * *n + 1] = 1.f;
	} else if (icompq == 2) {
	    u[u_dim1 + 1] = r_sign(&c_b15, &d__[1]);
	    vt[vt_dim1 + 1] = 1.f;
	}
	d__[1] = dabs(d__[1]);
	return 0;
    }
    nm1 = *n - 1;

/*
       If matrix lower bidiagonal, rotate to be upper bidiagonal
       by applying Givens rotations on the left
*/

    wstart = 1;
    qstart = 3;
    if (icompq == 1) {
	scopy_(n, &d__[1], &c__1, &q[1], &c__1);
	i__1 = *n - 1;
	scopy_(&i__1, &e[1], &c__1, &q[*n + 1], &c__1);
    }
    if (iuplo == 2) {
	qstart = 5;
	wstart = (*n << 1) - 1;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (icompq == 1) {
		q[i__ + (*n << 1)] = cs;
		q[i__ + *n * 3] = sn;
	    } else if (icompq == 2) {
		work[i__] = cs;
		work[nm1 + i__] = -sn;
	    }
/* L10: */
	}
    }

/*     If ICOMPQ = 0, use SLASDQ to compute the singular values. */

    if (icompq == 0) {
	slasdq_("U", &c__0, n, &c__0, &c__0, &c__0, &d__[1], &e[1], &vt[
		vt_offset], ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		wstart], info);
	goto L40;
    }

/*
       If N is smaller than the minimum divide size SMLSIZ, then solve
       the problem with another solver.
*/

    if (*n <= smlsiz) {
	if (icompq == 2) {
	    slaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	    slaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
	    slasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &vt[vt_offset]
		    , ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		    wstart], info);
	} else if (icompq == 1) {
	    iu = 1;
	    ivt = iu + *n;
	    slaset_("A", n, n, &c_b29, &c_b15, &q[iu + (qstart - 1) * *n], n);
	    slaset_("A", n, n, &c_b29, &c_b15, &q[ivt + (qstart - 1) * *n], n);
	    slasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &q[ivt + (
		    qstart - 1) * *n], n, &q[iu + (qstart - 1) * *n], n, &q[
		    iu + (qstart - 1) * *n], n, &work[wstart], info);
	}
	goto L40;
    }

    if (icompq == 2) {
	slaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	slaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
    }

/*     Scale. */

    orgnrm = slanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.f) {
	return 0;
    }
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, n, &c__1, &d__[1], n, &ierr);
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &nm1, &c__1, &e[1], &nm1, &
	    ierr);

    eps = slamch_("Epsilon");

    mlvl = (integer) (log((real) (*n) / (real) (smlsiz + 1)) / log(2.f)) + 1;
    smlszp = smlsiz + 1;

    if (icompq == 1) {
	iu = 1;
	ivt = smlsiz + 1;
	difl = ivt + smlszp;
	difr = difl + mlvl;
	z__ = difr + (mlvl << 1);
	ic = z__ + mlvl;
	is = ic + 1;
	poles = is + 1;
	givnum = poles + (mlvl << 1);

	k = 1;
	givptr = 2;
	perm = 3;
	givcol = perm + mlvl;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = d__[i__], dabs(r__1)) < eps) {
	    d__[i__] = r_sign(&eps, &d__[i__]);
	}
/* L20: */
    }

    start = 1;
    sqre = 0;

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = e[i__], dabs(r__1)) < eps || i__ == nm1) {

/*
          Subproblem found. First determine its size and then
          apply divide and conquer on it.
*/

	    if (i__ < nm1) {

/*        A subproblem with E(I) small for I < NM1. */

		nsize = i__ - start + 1;
	    } else if ((r__1 = e[i__], dabs(r__1)) >= eps) {

/*        A subproblem with E(NM1) not too small but I = NM1. */

		nsize = *n - start + 1;
	    } else {

/*
          A subproblem with E(NM1) small. This implies an
          1-by-1 subproblem at D(N). Solve this 1-by-1 problem
          first.
*/

		nsize = i__ - start + 1;
		if (icompq == 2) {
		    u[*n + *n * u_dim1] = r_sign(&c_b15, &d__[*n]);
		    vt[*n + *n * vt_dim1] = 1.f;
		} else if (icompq == 1) {
		    q[*n + (qstart - 1) * *n] = r_sign(&c_b15, &d__[*n]);
		    q[*n + (smlsiz + qstart - 1) * *n] = 1.f;
		}
		d__[*n] = (r__1 = d__[*n], dabs(r__1));
	    }
	    if (icompq == 2) {
		slasd0_(&nsize, &sqre, &d__[start], &e[start], &u[start +
			start * u_dim1], ldu, &vt[start + start * vt_dim1],
			ldvt, &smlsiz, &iwork[1], &work[wstart], info);
	    } else {
		slasda_(&icompq, &smlsiz, &nsize, &sqre, &d__[start], &e[
			start], &q[start + (iu + qstart - 2) * *n], n, &q[
			start + (ivt + qstart - 2) * *n], &iq[start + k * *n],
			 &q[start + (difl + qstart - 2) * *n], &q[start + (
			difr + qstart - 2) * *n], &q[start + (z__ + qstart -
			2) * *n], &q[start + (poles + qstart - 2) * *n], &iq[
			start + givptr * *n], &iq[start + givcol * *n], n, &
			iq[start + perm * *n], &q[start + (givnum + qstart -
			2) * *n], &q[start + (ic + qstart - 2) * *n], &q[
			start + (is + qstart - 2) * *n], &work[wstart], &
			iwork[1], info);
		if (*info != 0) {
		    return 0;
		}
	    }
	    start = i__ + 1;
	}
/* L30: */
    }

/*     Unscale */

    slascl_("G", &c__0, &c__0, &c_b15, &orgnrm, n, &c__1, &d__[1], n, &ierr);
L40:

/*     Use Selection Sort to minimize swaps of singular vectors */

    i__1 = *n;
    for (ii = 2; ii <= i__1; ++ii) {
	i__ = ii - 1;
	kk = i__;
	p = d__[i__];
	i__2 = *n;
	for (j = ii; j <= i__2; ++j) {
	    if (d__[j] > p) {
		kk = j;
		p = d__[j];
	    }
/* L50: */
	}
	if (kk != i__) {
	    d__[kk] = d__[i__];
	    d__[i__] = p;
	    if (icompq == 1) {
		iq[i__] = kk;
	    } else if (icompq == 2) {
		sswap_(n, &u[i__ * u_dim1 + 1], &c__1, &u[kk * u_dim1 + 1], &
			c__1);
		sswap_(n, &vt[i__ + vt_dim1], ldvt, &vt[kk + vt_dim1], ldvt);
	    }
	} else if (icompq == 1) {
	    iq[i__] = i__;
	}
/* L60: */
    }

/*     If ICOMPQ = 1, use IQ(N,1) as the indicator for UPLO */

    if (icompq == 1) {
	if (iuplo == 1) {
	    iq[*n] = 1;
	} else {
	    iq[*n] = 0;
	}
    }

/*
       If B is lower bidiagonal, update U by those Givens rotations
       which rotated B to be upper bidiagonal
*/

    if (iuplo == 2 && icompq == 2) {
	slasr_("L", "V", "B", n, n, &work[1], &work[*n], &u[u_offset], ldu);
    }

    return 0;

/*     End of SBDSDC */

} /* sbdsdc_ */

/* Subroutine */ int sbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, real *d__, real *e, real *vt, integer *ldvt, real *
	u, integer *ldu, real *c__, integer *ldc, real *work, integer *info)
{
    /* System generated locals */
    integer c_dim1, c_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1,
	    i__2;
    real r__1, r__2, r__3, r__4;
    doublereal d__1;

    /* Builtin functions */
    double pow_dd(doublereal *, doublereal *), sqrt(doublereal), r_sign(real *
	    , real *);

    /* Local variables */
    static real f, g, h__;
    static integer i__, j, m;
    static real r__, cs;
    static integer ll;
    static real sn, mu;
    static integer nm1, nm12, nm13, lll;
    static real eps, sll, tol, abse;
    static integer idir;
    static real abss;
    static integer oldm;
    static real cosl;
    static integer isub, iter;
    static real unfl, sinl, cosr, smin, smax, sinr;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *), slas2_(real *, real *, real *, real *,
	     real *);
    extern logical lsame_(char *, char *);
    static real oldcs;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static integer oldll;
    static real shift, sigmn, oldsn;
    static integer maxit;
    static real sminl;
    extern /* Subroutine */ int slasr_(char *, char *, char *, integer *,
	    integer *, real *, real *, real *, integer *);
    static real sigmx;
    static logical lower;
    extern /* Subroutine */ int sswap_(integer *, real *, integer *, real *,
	    integer *), slasq1_(integer *, real *, real *, real *, integer *),
	     slasv2_(real *, real *, real *, real *, real *, real *, real *,
	    real *, real *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real sminoa;
    extern /* Subroutine */ int slartg_(real *, real *, real *, real *, real *
	    );
    static real thresh;
    static logical rotate;
    static real sminlo, tolmul;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SBDSQR computes the singular value decomposition (SVD) of a real
    N-by-N (upper or lower) bidiagonal matrix B:  B = Q * S * P' (P'
    denotes the transpose of P), where S is a diagonal matrix with
    non-negative diagonal elements (the singular values of B), and Q
    and P are orthogonal matrices.

    The routine computes S, and optionally computes U * Q, P' * VT,
    or Q' * C, for given real input matrices U, VT, and C.

    See "Computing  Small Singular Values of Bidiagonal Matrices With
    Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
    LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
    no. 5, pp. 873-912, Sept 1990) and
    "Accurate singular values and differential qd algorithms," by
    B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
    Department, University of California at Berkeley, July 1992
    for a detailed description of the algorithm.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  B is upper bidiagonal;
            = 'L':  B is lower bidiagonal.

    N       (input) INTEGER
            The order of the matrix B.  N >= 0.

    NCVT    (input) INTEGER
            The number of columns of the matrix VT. NCVT >= 0.

    NRU     (input) INTEGER
            The number of rows of the matrix U. NRU >= 0.

    NCC     (input) INTEGER
            The number of columns of the matrix C. NCC >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the n diagonal elements of the bidiagonal matrix B.
            On exit, if INFO=0, the singular values of B in decreasing
            order.

    E       (input/output) REAL array, dimension (N)
            On entry, the elements of E contain the
            offdiagonal elements of the bidiagonal matrix whose SVD
            is desired. On normal exit (INFO = 0), E is destroyed.
            If the algorithm does not converge (INFO > 0), D and E
            will contain the diagonal and superdiagonal elements of a
            bidiagonal matrix orthogonally equivalent to the one given
            as input. E(N) is used for workspace.

    VT      (input/output) REAL array, dimension (LDVT, NCVT)
            On entry, an N-by-NCVT matrix VT.
            On exit, VT is overwritten by P' * VT.
            VT is not referenced if NCVT = 0.

    LDVT    (input) INTEGER
            The leading dimension of the array VT.
            LDVT >= max(1,N) if NCVT > 0; LDVT >= 1 if NCVT = 0.

    U       (input/output) REAL array, dimension (LDU, N)
            On entry, an NRU-by-N matrix U.
            On exit, U is overwritten by U * Q.
            U is not referenced if NRU = 0.

    LDU     (input) INTEGER
            The leading dimension of the array U.  LDU >= max(1,NRU).

    C       (input/output) REAL array, dimension (LDC, NCC)
            On entry, an N-by-NCC matrix C.
            On exit, C is overwritten by Q' * C.
            C is not referenced if NCC = 0.

    LDC     (input) INTEGER
            The leading dimension of the array C.
            LDC >= max(1,N) if NCC > 0; LDC >=1 if NCC = 0.

    WORK    (workspace) REAL array, dimension (4*N)

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  If INFO = -i, the i-th argument had an illegal value
            > 0:  the algorithm did not converge; D and E contain the
                  elements of a bidiagonal matrix which is orthogonally
                  similar to the input matrix B;  if INFO = i, i
                  elements of E have not converged to zero.

    Internal Parameters
    ===================

    TOLMUL  REAL, default = max(10,min(100,EPS**(-1/8)))
            TOLMUL controls the convergence criterion of the QR loop.
            If it is positive, TOLMUL*EPS is the desired relative
               precision in the computed singular values.
            If it is negative, abs(TOLMUL*EPS*sigma_max) is the
               desired absolute accuracy in the computed singular
               values (corresponds to relative accuracy
               abs(TOLMUL*EPS) in the largest singular value.
            abs(TOLMUL) should be between 1 and 1/EPS, and preferably
               between 10 (for fast convergence) and .1/EPS
               (for there to be some accuracy in the results).
            Default is to lose at either one eighth or 2 of the
               available decimal digits in each computed singular value
               (whichever is smaller).

    MAXITR  INTEGER, default = 6
            MAXITR controls the maximum number of passes of the
            algorithm through its inner loop. The algorithms stops
            (and so fails to converge) if the number of passes
            through the inner loop exceeds MAXITR*N**2.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    lower = lsame_(uplo, "L");
    if (! lsame_(uplo, "U") && ! lower) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ncvt < 0) {
	*info = -3;
    } else if (*nru < 0) {
	*info = -4;
    } else if (*ncc < 0) {
	*info = -5;
    } else if (*ncvt == 0 && *ldvt < 1 || *ncvt > 0 && *ldvt < max(1,*n)) {
	*info = -9;
    } else if (*ldu < max(1,*nru)) {
	*info = -11;
    } else if (*ncc == 0 && *ldc < 1 || *ncc > 0 && *ldc < max(1,*n)) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SBDSQR", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }
    if (*n == 1) {
	goto L160;
    }

/*     ROTATE is true if any singular vectors desired, false otherwise */

    rotate = *ncvt > 0 || *nru > 0 || *ncc > 0;

/*     If no singular vectors desired, use qd algorithm */

    if (! rotate) {
	slasq1_(n, &d__[1], &e[1], &work[1], info);
	return 0;
    }

    nm1 = *n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;

/*     Get machine constants */

    eps = slamch_("Epsilon");
    unfl = slamch_("Safe minimum");

/*
       If matrix lower bidiagonal, rotate to be upper bidiagonal
       by applying Givens rotations on the left
*/

    if (lower) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    work[i__] = cs;
	    work[nm1 + i__] = sn;
/* L10: */
	}

/*        Update singular vectors if desired */

	if (*nru > 0) {
	    slasr_("R", "V", "F", nru, n, &work[1], &work[*n], &u[u_offset],
		    ldu);
	}
	if (*ncc > 0) {
	    slasr_("L", "V", "F", n, ncc, &work[1], &work[*n], &c__[c_offset],
		     ldc);
	}
    }

/*
       Compute singular values to relative accuracy TOL
       (By setting TOL to be negative, algorithm will compute
       singular values to absolute accuracy ABS(TOL)*norm(input matrix))

   Computing MAX
   Computing MIN
*/
    d__1 = (doublereal) eps;
    r__3 = 100.f, r__4 = pow_dd(&d__1, &c_b94);
    r__1 = 10.f, r__2 = dmin(r__3,r__4);
    tolmul = dmax(r__1,r__2);
    tol = tolmul * eps;

/*     Compute approximate maximum, minimum singular values */

    smax = 0.f;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	r__2 = smax, r__3 = (r__1 = d__[i__], dabs(r__1));
	smax = dmax(r__2,r__3);
/* L20: */
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	r__2 = smax, r__3 = (r__1 = e[i__], dabs(r__1));
	smax = dmax(r__2,r__3);
/* L30: */
    }
    sminl = 0.f;
    if (tol >= 0.f) {

/*        Relative accuracy desired */

	sminoa = dabs(d__[1]);
	if (sminoa == 0.f) {
	    goto L50;
	}
	mu = sminoa;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    mu = (r__2 = d__[i__], dabs(r__2)) * (mu / (mu + (r__1 = e[i__ -
		    1], dabs(r__1))));
	    sminoa = dmin(sminoa,mu);
	    if (sminoa == 0.f) {
		goto L50;
	    }
/* L40: */
	}
L50:
	sminoa /= sqrt((real) (*n));
/* Computing MAX */
	r__1 = tol * sminoa, r__2 = *n * 6 * *n * unfl;
	thresh = dmax(r__1,r__2);
    } else {

/*
          Absolute accuracy desired

   Computing MAX
*/
	r__1 = dabs(tol) * smax, r__2 = *n * 6 * *n * unfl;
	thresh = dmax(r__1,r__2);
    }

/*
       Prepare for main iteration loop for the singular values
       (MAXIT is the maximum number of passes through the inner
       loop permitted before nonconvergence signalled.)
*/

    maxit = *n * 6 * *n;
    iter = 0;
    oldll = -1;
    oldm = -1;

/*     M points to last element of unconverged part of matrix */

    m = *n;

/*     Begin main iteration loop */

L60:

/*     Check for convergence or exceeding iteration count */

    if (m <= 1) {
	goto L160;
    }
    if (iter > maxit) {
	goto L200;
    }

/*     Find diagonal block of matrix to work on */

    if (tol < 0.f && (r__1 = d__[m], dabs(r__1)) <= thresh) {
	d__[m] = 0.f;
    }
    smax = (r__1 = d__[m], dabs(r__1));
    smin = smax;
    i__1 = m - 1;
    for (lll = 1; lll <= i__1; ++lll) {
	ll = m - lll;
	abss = (r__1 = d__[ll], dabs(r__1));
	abse = (r__1 = e[ll], dabs(r__1));
	if (tol < 0.f && abss <= thresh) {
	    d__[ll] = 0.f;
	}
	if (abse <= thresh) {
	    goto L80;
	}
	smin = dmin(smin,abss);
/* Computing MAX */
	r__1 = max(smax,abss);
	smax = dmax(r__1,abse);
/* L70: */
    }
    ll = 0;
    goto L90;
L80:
    e[ll] = 0.f;

/*     Matrix splits since E(LL) = 0 */

    if (ll == m - 1) {

/*        Convergence of bottom singular value, return to top of loop */

	--m;
	goto L60;
    }
L90:
    ++ll;

/*     E(LL) through E(M-1) are nonzero, E(LL-1) is zero */

    if (ll == m - 1) {

/*        2 by 2 block, handle separately */

	slasv2_(&d__[m - 1], &e[m - 1], &d__[m], &sigmn, &sigmx, &sinr, &cosr,
		 &sinl, &cosl);
	d__[m - 1] = sigmx;
	e[m - 1] = 0.f;
	d__[m] = sigmn;

/*        Compute singular vectors, if desired */

	if (*ncvt > 0) {
	    srot_(ncvt, &vt[m - 1 + vt_dim1], ldvt, &vt[m + vt_dim1], ldvt, &
		    cosr, &sinr);
	}
	if (*nru > 0) {
	    srot_(nru, &u[(m - 1) * u_dim1 + 1], &c__1, &u[m * u_dim1 + 1], &
		    c__1, &cosl, &sinl);
	}
	if (*ncc > 0) {
	    srot_(ncc, &c__[m - 1 + c_dim1], ldc, &c__[m + c_dim1], ldc, &
		    cosl, &sinl);
	}
	m += -2;
	goto L60;
    }

/*
       If working on new submatrix, choose shift direction
       (from larger end diagonal element towards smaller)
*/

    if (ll > oldm || m < oldll) {
	if ((r__1 = d__[ll], dabs(r__1)) >= (r__2 = d__[m], dabs(r__2))) {

/*           Chase bulge from top (big end) to bottom (small end) */

	    idir = 1;
	} else {

/*           Chase bulge from bottom (big end) to top (small end) */

	    idir = 2;
	}
    }

/*     Apply convergence tests */

    if (idir == 1) {

/*
          Run convergence test in forward direction
          First apply standard test to bottom of matrix
*/

	if ((r__2 = e[m - 1], dabs(r__2)) <= dabs(tol) * (r__1 = d__[m], dabs(
		r__1)) || tol < 0.f && (r__3 = e[m - 1], dabs(r__3)) <=
		thresh) {
	    e[m - 1] = 0.f;
	    goto L60;
	}

	if (tol >= 0.f) {

/*
             If relative accuracy desired,
             apply convergence criterion forward
*/

	    mu = (r__1 = d__[ll], dabs(r__1));
	    sminl = mu;
	    i__1 = m - 1;
	    for (lll = ll; lll <= i__1; ++lll) {
		if ((r__1 = e[lll], dabs(r__1)) <= tol * mu) {
		    e[lll] = 0.f;
		    goto L60;
		}
		sminlo = sminl;
		mu = (r__2 = d__[lll + 1], dabs(r__2)) * (mu / (mu + (r__1 =
			e[lll], dabs(r__1))));
		sminl = dmin(sminl,mu);
/* L100: */
	    }
	}

    } else {

/*
          Run convergence test in backward direction
          First apply standard test to top of matrix
*/

	if ((r__2 = e[ll], dabs(r__2)) <= dabs(tol) * (r__1 = d__[ll], dabs(
		r__1)) || tol < 0.f && (r__3 = e[ll], dabs(r__3)) <= thresh) {
	    e[ll] = 0.f;
	    goto L60;
	}

	if (tol >= 0.f) {

/*
             If relative accuracy desired,
             apply convergence criterion backward
*/

	    mu = (r__1 = d__[m], dabs(r__1));
	    sminl = mu;
	    i__1 = ll;
	    for (lll = m - 1; lll >= i__1; --lll) {
		if ((r__1 = e[lll], dabs(r__1)) <= tol * mu) {
		    e[lll] = 0.f;
		    goto L60;
		}
		sminlo = sminl;
		mu = (r__2 = d__[lll], dabs(r__2)) * (mu / (mu + (r__1 = e[
			lll], dabs(r__1))));
		sminl = dmin(sminl,mu);
/* L110: */
	    }
	}
    }
    oldll = ll;
    oldm = m;

/*
       Compute shift.  First, test if shifting would ruin relative
       accuracy, and if so set the shift to zero.

   Computing MAX
*/
    r__1 = eps, r__2 = tol * .01f;
    if (tol >= 0.f && *n * tol * (sminl / smax) <= dmax(r__1,r__2)) {

/*        Use a zero shift to avoid loss of relative accuracy */

	shift = 0.f;
    } else {

/*        Compute the shift from 2-by-2 block at end of matrix */

	if (idir == 1) {
	    sll = (r__1 = d__[ll], dabs(r__1));
	    slas2_(&d__[m - 1], &e[m - 1], &d__[m], &shift, &r__);
	} else {
	    sll = (r__1 = d__[m], dabs(r__1));
	    slas2_(&d__[ll], &e[ll], &d__[ll + 1], &shift, &r__);
	}

/*        Test if shift negligible, and if so set to zero */

	if (sll > 0.f) {
/* Computing 2nd power */
	    r__1 = shift / sll;
	    if (r__1 * r__1 < eps) {
		shift = 0.f;
	    }
	}
    }

/*     Increment iteration count */

    iter = iter + m - ll;

/*     If SHIFT = 0, do simplified QR iteration */

    if (shift == 0.f) {
	if (idir == 1) {

/*
             Chase bulge from top to bottom
             Save cosines and sines for later singular vector updates
*/

	    cs = 1.f;
	    oldcs = 1.f;
	    i__1 = m - 1;
	    for (i__ = ll; i__ <= i__1; ++i__) {
		r__1 = d__[i__] * cs;
		slartg_(&r__1, &e[i__], &cs, &sn, &r__);
		if (i__ > ll) {
		    e[i__ - 1] = oldsn * r__;
		}
		r__1 = oldcs * r__;
		r__2 = d__[i__ + 1] * sn;
		slartg_(&r__1, &r__2, &oldcs, &oldsn, &d__[i__]);
		work[i__ - ll + 1] = cs;
		work[i__ - ll + 1 + nm1] = sn;
		work[i__ - ll + 1 + nm12] = oldcs;
		work[i__ - ll + 1 + nm13] = oldsn;
/* L120: */
	    }
	    h__ = d__[m] * cs;
	    d__[m] = h__ * oldcs;
	    e[m - 1] = h__ * oldsn;

/*           Update singular vectors */

	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "F", &i__1, ncvt, &work[1], &work[*n], &vt[
			ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		slasr_("R", "V", "F", nru, &i__1, &work[nm12 + 1], &work[nm13
			+ 1], &u[ll * u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "F", &i__1, ncc, &work[nm12 + 1], &work[nm13
			+ 1], &c__[ll + c_dim1], ldc);
	    }

/*           Test convergence */

	    if ((r__1 = e[m - 1], dabs(r__1)) <= thresh) {
		e[m - 1] = 0.f;
	    }

	} else {

/*
             Chase bulge from bottom to top
             Save cosines and sines for later singular vector updates
*/

	    cs = 1.f;
	    oldcs = 1.f;
	    i__1 = ll + 1;
	    for (i__ = m; i__ >= i__1; --i__) {
		r__1 = d__[i__] * cs;
		slartg_(&r__1, &e[i__ - 1], &cs, &sn, &r__);
		if (i__ < m) {
		    e[i__] = oldsn * r__;
		}
		r__1 = oldcs * r__;
		r__2 = d__[i__ - 1] * sn;
		slartg_(&r__1, &r__2, &oldcs, &oldsn, &d__[i__]);
		work[i__ - ll] = cs;
		work[i__ - ll + nm1] = -sn;
		work[i__ - ll + nm12] = oldcs;
		work[i__ - ll + nm13] = -oldsn;
/* L130: */
	    }
	    h__ = d__[ll] * cs;
	    d__[ll] = h__ * oldcs;
	    e[ll] = h__ * oldsn;

/*           Update singular vectors */

	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[
			nm13 + 1], &vt[ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		slasr_("R", "V", "B", nru, &i__1, &work[1], &work[*n], &u[ll *
			 u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "B", &i__1, ncc, &work[1], &work[*n], &c__[
			ll + c_dim1], ldc);
	    }

/*           Test convergence */

	    if ((r__1 = e[ll], dabs(r__1)) <= thresh) {
		e[ll] = 0.f;
	    }
	}
    } else {

/*        Use nonzero shift */

	if (idir == 1) {

/*
             Chase bulge from top to bottom
             Save cosines and sines for later singular vector updates
*/

	    f = ((r__1 = d__[ll], dabs(r__1)) - shift) * (r_sign(&c_b15, &d__[
		    ll]) + shift / d__[ll]);
	    g = e[ll];
	    i__1 = m - 1;
	    for (i__ = ll; i__ <= i__1; ++i__) {
		slartg_(&f, &g, &cosr, &sinr, &r__);
		if (i__ > ll) {
		    e[i__ - 1] = r__;
		}
		f = cosr * d__[i__] + sinr * e[i__];
		e[i__] = cosr * e[i__] - sinr * d__[i__];
		g = sinr * d__[i__ + 1];
		d__[i__ + 1] = cosr * d__[i__ + 1];
		slartg_(&f, &g, &cosl, &sinl, &r__);
		d__[i__] = r__;
		f = cosl * e[i__] + sinl * d__[i__ + 1];
		d__[i__ + 1] = cosl * d__[i__ + 1] - sinl * e[i__];
		if (i__ < m - 1) {
		    g = sinl * e[i__ + 1];
		    e[i__ + 1] = cosl * e[i__ + 1];
		}
		work[i__ - ll + 1] = cosr;
		work[i__ - ll + 1 + nm1] = sinr;
		work[i__ - ll + 1 + nm12] = cosl;
		work[i__ - ll + 1 + nm13] = sinl;
/* L140: */
	    }
	    e[m - 1] = f;

/*           Update singular vectors */

	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "F", &i__1, ncvt, &work[1], &work[*n], &vt[
			ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		slasr_("R", "V", "F", nru, &i__1, &work[nm12 + 1], &work[nm13
			+ 1], &u[ll * u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "F", &i__1, ncc, &work[nm12 + 1], &work[nm13
			+ 1], &c__[ll + c_dim1], ldc);
	    }

/*           Test convergence */

	    if ((r__1 = e[m - 1], dabs(r__1)) <= thresh) {
		e[m - 1] = 0.f;
	    }

	} else {

/*
             Chase bulge from bottom to top
             Save cosines and sines for later singular vector updates
*/

	    f = ((r__1 = d__[m], dabs(r__1)) - shift) * (r_sign(&c_b15, &d__[
		    m]) + shift / d__[m]);
	    g = e[m - 1];
	    i__1 = ll + 1;
	    for (i__ = m; i__ >= i__1; --i__) {
		slartg_(&f, &g, &cosr, &sinr, &r__);
		if (i__ < m) {
		    e[i__] = r__;
		}
		f = cosr * d__[i__] + sinr * e[i__ - 1];
		e[i__ - 1] = cosr * e[i__ - 1] - sinr * d__[i__];
		g = sinr * d__[i__ - 1];
		d__[i__ - 1] = cosr * d__[i__ - 1];
		slartg_(&f, &g, &cosl, &sinl, &r__);
		d__[i__] = r__;
		f = cosl * e[i__ - 1] + sinl * d__[i__ - 1];
		d__[i__ - 1] = cosl * d__[i__ - 1] - sinl * e[i__ - 1];
		if (i__ > ll + 1) {
		    g = sinl * e[i__ - 2];
		    e[i__ - 2] = cosl * e[i__ - 2];
		}
		work[i__ - ll] = cosr;
		work[i__ - ll + nm1] = -sinr;
		work[i__ - ll + nm12] = cosl;
		work[i__ - ll + nm13] = -sinl;
/* L150: */
	    }
	    e[ll] = f;

/*           Test convergence */

	    if ((r__1 = e[ll], dabs(r__1)) <= thresh) {
		e[ll] = 0.f;
	    }

/*           Update singular vectors if desired */

	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[
			nm13 + 1], &vt[ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		slasr_("R", "V", "B", nru, &i__1, &work[1], &work[*n], &u[ll *
			 u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		slasr_("L", "V", "B", &i__1, ncc, &work[1], &work[*n], &c__[
			ll + c_dim1], ldc);
	    }
	}
    }

/*     QR iteration finished, go back and check convergence */

    goto L60;

/*     All singular values converged, so make them positive */

L160:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] < 0.f) {
	    d__[i__] = -d__[i__];

/*           Change sign of singular vectors, if desired */

	    if (*ncvt > 0) {
		sscal_(ncvt, &c_b151, &vt[i__ + vt_dim1], ldvt);
	    }
	}
/* L170: */
    }

/*
       Sort the singular values into decreasing order (insertion sort on
       singular values, but only one transposition per singular vector)
*/

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Scan for smallest D(I) */

	isub = 1;
	smin = d__[1];
	i__2 = *n + 1 - i__;
	for (j = 2; j <= i__2; ++j) {
	    if (d__[j] <= smin) {
		isub = j;
		smin = d__[j];
	    }
/* L180: */
	}
	if (isub != *n + 1 - i__) {

/*           Swap singular values and vectors */

	    d__[isub] = d__[*n + 1 - i__];
	    d__[*n + 1 - i__] = smin;
	    if (*ncvt > 0) {
		sswap_(ncvt, &vt[isub + vt_dim1], ldvt, &vt[*n + 1 - i__ +
			vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		sswap_(nru, &u[isub * u_dim1 + 1], &c__1, &u[(*n + 1 - i__) *
			u_dim1 + 1], &c__1);
	    }
	    if (*ncc > 0) {
		sswap_(ncc, &c__[isub + c_dim1], ldc, &c__[*n + 1 - i__ +
			c_dim1], ldc);
	    }
	}
/* L190: */
    }
    goto L220;

/*     Maximum number of iterations exceeded, failure to converge */

L200:
    *info = 0;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (e[i__] != 0.f) {
	    ++(*info);
	}
/* L210: */
    }
L220:
    return 0;

/*     End of SBDSQR */

} /* sbdsqr_ */

/* Subroutine */ int sgebak_(char *job, char *side, integer *n, integer *ilo,
	integer *ihi, real *scale, integer *m, real *v, integer *ldv, integer
	*info)
{
    /* System generated locals */
    integer v_dim1, v_offset, i__1;

    /* Local variables */
    static integer i__, k;
    static real s;
    static integer ii;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static logical leftv;
    extern /* Subroutine */ int sswap_(integer *, real *, integer *, real *,
	    integer *), xerbla_(char *, integer *);
    static logical rightv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SGEBAK forms the right or left eigenvectors of a real general matrix
    by backward transformation on the computed eigenvectors of the
    balanced matrix output by SGEBAL.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            Specifies the type of backward transformation required:
            = 'N', do nothing, return immediately;
            = 'P', do backward transformation for permutation only;
            = 'S', do backward transformation for scaling only;
            = 'B', do backward transformations for both permutation and
                   scaling.
            JOB must be the same as the argument JOB supplied to SGEBAL.

    SIDE    (input) CHARACTER*1
            = 'R':  V contains right eigenvectors;
            = 'L':  V contains left eigenvectors.

    N       (input) INTEGER
            The number of rows of the matrix V.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            The integers ILO and IHI determined by SGEBAL.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    SCALE   (input) REAL array, dimension (N)
            Details of the permutation and scaling factors, as returned
            by SGEBAL.

    M       (input) INTEGER
            The number of columns of the matrix V.  M >= 0.

    V       (input/output) REAL array, dimension (LDV,M)
            On entry, the matrix of right or left eigenvectors to be
            transformed, as returned by SHSEIN or STREVC.
            On exit, V is overwritten by the transformed eigenvectors.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    =====================================================================


       Decode and Test the input parameters
*/

    /* Parameter adjustments */
    --scale;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;

    /* Function Body */
    rightv = lsame_(side, "R");
    leftv = lsame_(side, "L");

    *info = 0;
    if (! lsame_(job, "N") && ! lsame_(job, "P") && ! lsame_(job, "S")
	    && ! lsame_(job, "B")) {
	*info = -1;
    } else if (! rightv && ! leftv) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -4;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -5;
    } else if (*m < 0) {
	*info = -7;
    } else if (*ldv < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEBAK", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    if (*m == 0) {
	return 0;
    }
    if (lsame_(job, "N")) {
	return 0;
    }

    if (*ilo == *ihi) {
	goto L30;
    }

/*     Backward balance */

    if (lsame_(job, "S") || lsame_(job, "B")) {

	if (rightv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = scale[i__];
		sscal_(m, &s, &v[i__ + v_dim1], ldv);
/* L10: */
	    }
	}

	if (leftv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = 1.f / scale[i__];
		sscal_(m, &s, &v[i__ + v_dim1], ldv);
/* L20: */
	    }
	}

    }

/*
       Backward permutation

       For  I = ILO-1 step -1 until 1,
                IHI+1 step 1 until N do --
*/

L30:
    if (lsame_(job, "P") || lsame_(job, "B")) {
	if (rightv) {
	    i__1 = *n;
	    for (ii = 1; ii <= i__1; ++ii) {
		i__ = ii;
		if (i__ >= *ilo && i__ <= *ihi) {
		    goto L40;
		}
		if (i__ < *ilo) {
		    i__ = *ilo - ii;
		}
		k = scale[i__];
		if (k == i__) {
		    goto L40;
		}
		sswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L40:
		;
	    }
	}

	if (leftv) {
	    i__1 = *n;
	    for (ii = 1; ii <= i__1; ++ii) {
		i__ = ii;
		if (i__ >= *ilo && i__ <= *ihi) {
		    goto L50;
		}
		if (i__ < *ilo) {
		    i__ = *ilo - ii;
		}
		k = scale[i__];
		if (k == i__) {
		    goto L50;
		}
		sswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L50:
		;
	    }
	}
    }

    return 0;

/*     End of SGEBAK */

} /* sgebak_ */

/* Subroutine */ int sgebal_(char *job, integer *n, real *a, integer *lda,
	integer *ilo, integer *ihi, real *scale, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1, r__2;

    /* Local variables */
    static real c__, f, g;
    static integer i__, j, k, l, m;
    static real r__, s, ca, ra;
    static integer ica, ira, iexc;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sswap_(integer *, real *, integer *, real *, integer *);
    static real sfmin1, sfmin2, sfmax1, sfmax2;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    static logical noconv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SGEBAL balances a general real matrix A.  This involves, first,
    permuting A by a similarity transformation to isolate eigenvalues
    in the first 1 to ILO-1 and last IHI+1 to N elements on the
    diagonal; and second, applying a diagonal similarity transformation
    to rows and columns ILO to IHI to make the rows and columns as
    close in norm as possible.  Both steps are optional.

    Balancing may reduce the 1-norm of the matrix, and improve the
    accuracy of the computed eigenvalues and/or eigenvectors.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            Specifies the operations to be performed on A:
            = 'N':  none:  simply set ILO = 1, IHI = N, SCALE(I) = 1.0
                    for i = 1,...,N;
            = 'P':  permute only;
            = 'S':  scale only;
            = 'B':  both permute and scale.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the input matrix A.
            On exit,  A is overwritten by the balanced matrix.
            If JOB = 'N', A is not referenced.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    ILO     (output) INTEGER
    IHI     (output) INTEGER
            ILO and IHI are set to integers such that on exit
            A(i,j) = 0 if i > j and j = 1,...,ILO-1 or I = IHI+1,...,N.
            If JOB = 'N' or 'S', ILO = 1 and IHI = N.

    SCALE   (output) REAL array, dimension (N)
            Details of the permutations and scaling factors applied to
            A.  If P(j) is the index of the row and column interchanged
            with row and column j and D(j) is the scaling factor
            applied to row and column j, then
            SCALE(j) = P(j)    for j = 1,...,ILO-1
                     = D(j)    for j = ILO,...,IHI
                     = P(j)    for j = IHI+1,...,N.
            The order in which the interchanges are made is N to IHI+1,
            then 1 to ILO-1.

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The permutations consist of row and column interchanges which put
    the matrix in the form

               ( T1   X   Y  )
       P A P = (  0   B   Z  )
               (  0   0   T2 )

    where T1 and T2 are upper triangular matrices whose eigenvalues lie
    along the diagonal.  The column indices ILO and IHI mark the starting
    and ending columns of the submatrix B. Balancing consists of applying
    a diagonal similarity transformation inv(D) * B * D to make the
    1-norms of each row of B and its corresponding column nearly equal.
    The output matrix is

       ( T1     X*D          Y    )
       (  0  inv(D)*B*D  inv(D)*Z ).
       (  0      0           T2   )

    Information about the permutations P and the diagonal matrix D is
    returned in the vector SCALE.

    This subroutine is based on the EISPACK routine BALANC.

    Modified by Tzu-Yi Chen, Computer Science Division, University of
      California at Berkeley, USA

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --scale;

    /* Function Body */
    *info = 0;
    if (! lsame_(job, "N") && ! lsame_(job, "P") && ! lsame_(job, "S")
	    && ! lsame_(job, "B")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEBAL", &i__1);
	return 0;
    }

    k = 1;
    l = *n;

    if (*n == 0) {
	goto L210;
    }

    if (lsame_(job, "N")) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scale[i__] = 1.f;
/* L10: */
	}
	goto L210;
    }

    if (lsame_(job, "S")) {
	goto L120;
    }

/*     Permutation to isolate eigenvalues if possible */

    goto L50;

/*     Row and column exchange. */

L20:
    scale[m] = (real) j;
    if (j == m) {
	goto L30;
    }

    sswap_(&l, &a[j * a_dim1 + 1], &c__1, &a[m * a_dim1 + 1], &c__1);
    i__1 = *n - k + 1;
    sswap_(&i__1, &a[j + k * a_dim1], lda, &a[m + k * a_dim1], lda);

L30:
    switch (iexc) {
	case 1:  goto L40;
	case 2:  goto L80;
    }

/*     Search for rows isolating an eigenvalue and push them down. */

L40:
    if (l == 1) {
	goto L210;
    }
    --l;

L50:
    for (j = l; j >= 1; --j) {

	i__1 = l;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ == j) {
		goto L60;
	    }
	    if (a[j + i__ * a_dim1] != 0.f) {
		goto L70;
	    }
L60:
	    ;
	}

	m = l;
	iexc = 1;
	goto L20;
L70:
	;
    }

    goto L90;

/*     Search for columns isolating an eigenvalue and push them left. */

L80:
    ++k;

L90:
    i__1 = l;
    for (j = k; j <= i__1; ++j) {

	i__2 = l;
	for (i__ = k; i__ <= i__2; ++i__) {
	    if (i__ == j) {
		goto L100;
	    }
	    if (a[i__ + j * a_dim1] != 0.f) {
		goto L110;
	    }
L100:
	    ;
	}

	m = k;
	iexc = 2;
	goto L20;
L110:
	;
    }

L120:
    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	scale[i__] = 1.f;
/* L130: */
    }

    if (lsame_(job, "P")) {
	goto L210;
    }

/*
       Balance the submatrix in rows K to L.

       Iterative loop for norm reduction
*/

    sfmin1 = slamch_("S") / slamch_("P");
    sfmax1 = 1.f / sfmin1;
    sfmin2 = sfmin1 * 8.f;
    sfmax2 = 1.f / sfmin2;
L140:
    noconv = FALSE_;

    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	c__ = 0.f;
	r__ = 0.f;

	i__2 = l;
	for (j = k; j <= i__2; ++j) {
	    if (j == i__) {
		goto L150;
	    }
	    c__ += (r__1 = a[j + i__ * a_dim1], dabs(r__1));
	    r__ += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
L150:
	    ;
	}
	ica = isamax_(&l, &a[i__ * a_dim1 + 1], &c__1);
	ca = (r__1 = a[ica + i__ * a_dim1], dabs(r__1));
	i__2 = *n - k + 1;
	ira = isamax_(&i__2, &a[i__ + k * a_dim1], lda);
	ra = (r__1 = a[i__ + (ira + k - 1) * a_dim1], dabs(r__1));

/*        Guard against zero C or R due to underflow. */

	if (c__ == 0.f || r__ == 0.f) {
	    goto L200;
	}
	g = r__ / 8.f;
	f = 1.f;
	s = c__ + r__;
L160:
/* Computing MAX */
	r__1 = max(f,c__);
/* Computing MIN */
	r__2 = min(r__,g);
	if (c__ >= g || dmax(r__1,ca) >= sfmax2 || dmin(r__2,ra) <= sfmin2) {
	    goto L170;
	}
	f *= 8.f;
	c__ *= 8.f;
	ca *= 8.f;
	r__ /= 8.f;
	g /= 8.f;
	ra /= 8.f;
	goto L160;

L170:
	g = c__ / 8.f;
L180:
/* Computing MIN */
	r__1 = min(f,c__), r__1 = min(r__1,g);
	if (g < r__ || dmax(r__,ra) >= sfmax2 || dmin(r__1,ca) <= sfmin2) {
	    goto L190;
	}
	f /= 8.f;
	c__ /= 8.f;
	g /= 8.f;
	ca /= 8.f;
	r__ *= 8.f;
	ra *= 8.f;
	goto L180;

/*        Now balance. */

L190:
	if (c__ + r__ >= s * .95f) {
	    goto L200;
	}
	if (f < 1.f && scale[i__] < 1.f) {
	    if (f * scale[i__] <= sfmin1) {
		goto L200;
	    }
	}
	if (f > 1.f && scale[i__] > 1.f) {
	    if (scale[i__] >= sfmax1 / f) {
		goto L200;
	    }
	}
	g = 1.f / f;
	scale[i__] *= f;
	noconv = TRUE_;

	i__2 = *n - k + 1;
	sscal_(&i__2, &g, &a[i__ + k * a_dim1], lda);
	sscal_(&l, &f, &a[i__ * a_dim1 + 1], &c__1);

L200:
	;
    }

    if (noconv) {
	goto L140;
    }

L210:
    *ilo = k;
    *ihi = l;

    return 0;

/*     End of SGEBAL */

} /* sgebal_ */

/* Subroutine */ int sgebd2_(integer *m, integer *n, real *a, integer *lda,
	real *d__, real *e, real *tauq, real *taup, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__;
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *), slarfg_(integer *, real *, real *,
	    integer *, real *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SGEBD2 reduces a real general m by n matrix A to upper or lower
    bidiagonal form B by an orthogonal transformation: Q' * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit,
            if m >= n, the diagonal and the first superdiagonal are
              overwritten with the upper bidiagonal matrix B; the
              elements below the diagonal, with the array TAUQ, represent
              the orthogonal matrix Q as a product of elementary
              reflectors, and the elements above the first superdiagonal,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors;
            if m < n, the diagonal and the first subdiagonal are
              overwritten with the lower bidiagonal matrix B; the
              elements below the first subdiagonal, with the array TAUQ,
              represent the orthogonal matrix Q as a product of
              elementary reflectors, and the elements above the diagonal,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) REAL array, dimension (min(M,N))
            The diagonal elements of the bidiagonal matrix B:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (min(M,N)-1)
            The off-diagonal elements of the bidiagonal matrix B:
            if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
            if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.

    TAUQ    (output) REAL array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    WORK    (workspace) REAL array, dimension (max(M,N))

    INFO    (output) INTEGER
            = 0: successful exit.
            < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

    If m >= n,

       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors;
    v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
    u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n,

       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors;
    v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
    u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    The contents of A on exit are illustrated by the following examples:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
      (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
      (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
      (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
      (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
      (  v1  v2  v3  v4  v5 )

    where d and e denote diagonal and off-diagonal elements of B, vi
    denotes an element of the vector defining H(i), and ui an element of
    the vector defining G(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info < 0) {
	i__1 = -(*info);
	xerbla_("SGEBD2", &i__1);
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ *
		    a_dim1], &c__1, &tauq[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;

/*           Apply H(i) to A(i:m,i+1:n) from the left */

	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    slarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &tauq[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = d__[i__];

	    if (i__ < *n) {

/*
                Generate elementary reflector G(i) to annihilate
                A(i,i+2:n)
*/

		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, &a[i__ + (i__ + 1) * a_dim1], &a[i__ + min(
			i__3,*n) * a_dim1], lda, &taup[i__]);
		e[i__] = a[i__ + (i__ + 1) * a_dim1];
		a[i__ + (i__ + 1) * a_dim1] = 1.f;

/*              Apply G(i) to A(i+1:m,i+1:n) from the right */

		i__2 = *m - i__;
		i__3 = *n - i__;
		slarf_("Right", &i__2, &i__3, &a[i__ + (i__ + 1) * a_dim1],
			lda, &taup[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		a[i__ + (i__ + 1) * a_dim1] = e[i__];
	    } else {
		taup[i__] = 0.f;
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector G(i) to annihilate A(i,i+1:n) */

	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) *
		    a_dim1], lda, &taup[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;

/*           Apply G(i) to A(i+1:m,i:n) from the right */

	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
/* Computing MIN */
	    i__4 = i__ + 1;
	    slarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &taup[
		    i__], &a[min(i__4,*m) + i__ * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = d__[i__];

	    if (i__ < *m) {

/*
                Generate elementary reflector H(i) to annihilate
                A(i+2:m,i)
*/

		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*m) +
			i__ * a_dim1], &c__1, &tauq[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.f;

/*              Apply H(i) to A(i+1:m,i+1:n) from the left */

		i__2 = *m - i__;
		i__3 = *n - i__;
		slarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &
			c__1, &tauq[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		a[i__ + 1 + i__ * a_dim1] = e[i__];
	    } else {
		tauq[i__] = 0.f;
	    }
/* L20: */
	}
    }
    return 0;

/*     End of SGEBD2 */

} /* sgebd2_ */

/* Subroutine */ int sgebrd_(integer *m, integer *n, real *a, integer *lda,
	real *d__, real *e, real *tauq, real *taup, real *work, integer *
	lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, nb, nx;
    static real ws;
    static integer nbmin, iinfo;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer minmn;
    extern /* Subroutine */ int sgebd2_(integer *, integer *, real *, integer
	    *, real *, real *, real *, real *, real *, integer *), slabrd_(
	    integer *, integer *, integer *, real *, integer *, real *, real *
	    , real *, real *, real *, integer *, real *, integer *), xerbla_(
	    char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static integer ldwrkx, ldwrky, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SGEBRD reduces a general real M-by-N matrix A to upper or lower
    bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N general matrix to be reduced.
            On exit,
            if m >= n, the diagonal and the first superdiagonal are
              overwritten with the upper bidiagonal matrix B; the
              elements below the diagonal, with the array TAUQ, represent
              the orthogonal matrix Q as a product of elementary
              reflectors, and the elements above the first superdiagonal,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors;
            if m < n, the diagonal and the first subdiagonal are
              overwritten with the lower bidiagonal matrix B; the
              elements below the first subdiagonal, with the array TAUQ,
              represent the orthogonal matrix Q as a product of
              elementary reflectors, and the elements above the diagonal,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) REAL array, dimension (min(M,N))
            The diagonal elements of the bidiagonal matrix B:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (min(M,N)-1)
            The off-diagonal elements of the bidiagonal matrix B:
            if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
            if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.

    TAUQ    (output) REAL array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,M,N).
            For optimum performance LWORK >= (M+N)*NB, where NB
            is the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

    If m >= n,

       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors;
    v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
    u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n,

       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors;
    v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
    u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
    tauq is stored in TAUQ(i) and taup in TAUP(i).

    The contents of A on exit are illustrated by the following examples:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
      (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
      (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
      (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
      (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
      (  v1  v2  v3  v4  v5 )

    where d and e denote diagonal and off-diagonal elements of B, vi
    denotes an element of the vector defining H(i), and ui an element of
    the vector defining G(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    --work;

    /* Function Body */
    *info = 0;
/* Computing MAX */
    i__1 = 1, i__2 = ilaenv_(&c__1, "SGEBRD", " ", m, n, &c_n1, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = max(i__1,i__2);
    lwkopt = (*m + *n) * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = max(1,*m);
	if (*lwork < max(i__1,*n) && ! lquery) {
	    *info = -10;
	}
    }
    if (*info < 0) {
	i__1 = -(*info);
	xerbla_("SGEBRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    minmn = min(*m,*n);
    if (minmn == 0) {
	work[1] = 1.f;
	return 0;
    }

    ws = (real) max(*m,*n);
    ldwrkx = *m;
    ldwrky = *n;

    if (nb > 1 && nb < minmn) {

/*
          Set the crossover point NX.

   Computing MAX
*/
	i__1 = nb, i__2 = ilaenv_(&c__3, "SGEBRD", " ", m, n, &c_n1, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);

/*        Determine when to switch from blocked to unblocked code. */

	if (nx < minmn) {
	    ws = (real) ((*m + *n) * nb);
	    if ((real) (*lwork) < ws) {

/*
                Not enough work space for the optimal NB, consider using
                a smaller block size.
*/

		nbmin = ilaenv_(&c__2, "SGEBRD", " ", m, n, &c_n1, &c_n1, (
			ftnlen)6, (ftnlen)1);
		if (*lwork >= (*m + *n) * nbmin) {
		    nb = *lwork / (*m + *n);
		} else {
		    nb = 1;
		    nx = minmn;
		}
	    }
	}
    } else {
	nx = minmn;
    }

    i__1 = minmn - nx;
    i__2 = nb;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {

/*
          Reduce rows and columns i:i+nb-1 to bidiagonal form and return
          the matrices X and Y which are needed to update the unreduced
          part of the matrix
*/

	i__3 = *m - i__ + 1;
	i__4 = *n - i__ + 1;
	slabrd_(&i__3, &i__4, &nb, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[
		i__], &tauq[i__], &taup[i__], &work[1], &ldwrkx, &work[ldwrkx
		* nb + 1], &ldwrky);

/*
          Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
          of the form  A := A - V*Y' - X*U'
*/

	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	sgemm_("No transpose", "Transpose", &i__3, &i__4, &nb, &c_b151, &a[
		i__ + nb + i__ * a_dim1], lda, &work[ldwrkx * nb + nb + 1], &
		ldwrky, &c_b15, &a[i__ + nb + (i__ + nb) * a_dim1], lda);
	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	sgemm_("No transpose", "No transpose", &i__3, &i__4, &nb, &c_b151, &
		work[nb + 1], &ldwrkx, &a[i__ + (i__ + nb) * a_dim1], lda, &
		c_b15, &a[i__ + nb + (i__ + nb) * a_dim1], lda);

/*        Copy diagonal and off-diagonal elements of B back into A */

	if (*m >= *n) {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + j * a_dim1] = d__[j];
		a[j + (j + 1) * a_dim1] = e[j];
/* L10: */
	    }
	} else {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + j * a_dim1] = d__[j];
		a[j + 1 + j * a_dim1] = e[j];
/* L20: */
	    }
	}
/* L30: */
    }

/*     Use unblocked code to reduce the remainder of the matrix */

    i__2 = *m - i__ + 1;
    i__1 = *n - i__ + 1;
    sgebd2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__], &
	    tauq[i__], &taup[i__], &work[1], &iinfo);
    work[1] = ws;
    return 0;

/*     End of SGEBRD */

} /* sgebrd_ */

/* Subroutine */ int sgeev_(char *jobvl, char *jobvr, integer *n, real *a,
	integer *lda, real *wr, real *wi, real *vl, integer *ldvl, real *vr,
	integer *ldvr, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3, i__4;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, k;
    static real r__, cs, sn;
    static integer ihi;
    static real scl;
    static integer ilo;
    static real dum[1], eps;
    static integer ibal;
    static char side[1];
    static integer maxb;
    static real anrm;
    static integer ierr, itau, iwrk, nout;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *);
    extern doublereal snrm2_(integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    extern doublereal slapy2_(real *, real *);
    extern /* Subroutine */ int slabad_(real *, real *);
    static logical scalea;
    static real cscale;
    extern /* Subroutine */ int sgebak_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *), sgebal_(char *, integer *, real *, integer *,
	    integer *, integer *, real *, integer *);
    extern doublereal slamch_(char *), slange_(char *, integer *,
	    integer *, real *, integer *, real *);
    extern /* Subroutine */ int sgehrd_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *), xerbla_(char
	    *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical select[1];
    static real bignum;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slacpy_(char *, integer *, integer *, real *,
	    integer *, real *, integer *), slartg_(real *, real *,
	    real *, real *, real *), sorghr_(integer *, integer *, integer *,
	    real *, integer *, real *, real *, integer *, integer *), shseqr_(
	    char *, char *, integer *, integer *, integer *, real *, integer *
	    , real *, real *, real *, integer *, real *, integer *, integer *), strevc_(char *, char *, logical *, integer *,
	    real *, integer *, real *, integer *, real *, integer *, integer *
	    , integer *, real *, integer *);
    static integer minwrk, maxwrk;
    static logical wantvl;
    static real smlnum;
    static integer hswork;
    static logical lquery, wantvr;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       December 8, 1999


    Purpose
    =======

    SGEEV computes for an N-by-N real nonsymmetric matrix A, the
    eigenvalues and, optionally, the left and/or right eigenvectors.

    The right eigenvector v(j) of A satisfies
                     A * v(j) = lambda(j) * v(j)
    where lambda(j) is its eigenvalue.
    The left eigenvector u(j) of A satisfies
                  u(j)**H * A = lambda(j) * u(j)**H
    where u(j)**H denotes the conjugate transpose of u(j).

    The computed eigenvectors are normalized to have Euclidean norm
    equal to 1 and largest component real.

    Arguments
    =========

    JOBVL   (input) CHARACTER*1
            = 'N': left eigenvectors of A are not computed;
            = 'V': left eigenvectors of A are computed.

    JOBVR   (input) CHARACTER*1
            = 'N': right eigenvectors of A are not computed;
            = 'V': right eigenvectors of A are computed.

    N       (input) INTEGER
            The order of the matrix A. N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the N-by-N matrix A.
            On exit, A has been overwritten.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    WR      (output) REAL array, dimension (N)
    WI      (output) REAL array, dimension (N)
            WR and WI contain the real and imaginary parts,
            respectively, of the computed eigenvalues.  Complex
            conjugate pairs of eigenvalues appear consecutively
            with the eigenvalue having the positive imaginary part
            first.

    VL      (output) REAL array, dimension (LDVL,N)
            If JOBVL = 'V', the left eigenvectors u(j) are stored one
            after another in the columns of VL, in the same order
            as their eigenvalues.
            If JOBVL = 'N', VL is not referenced.
            If the j-th eigenvalue is real, then u(j) = VL(:,j),
            the j-th column of VL.
            If the j-th and (j+1)-st eigenvalues form a complex
            conjugate pair, then u(j) = VL(:,j) + i*VL(:,j+1) and
            u(j+1) = VL(:,j) - i*VL(:,j+1).

    LDVL    (input) INTEGER
            The leading dimension of the array VL.  LDVL >= 1; if
            JOBVL = 'V', LDVL >= N.

    VR      (output) REAL array, dimension (LDVR,N)
            If JOBVR = 'V', the right eigenvectors v(j) are stored one
            after another in the columns of VR, in the same order
            as their eigenvalues.
            If JOBVR = 'N', VR is not referenced.
            If the j-th eigenvalue is real, then v(j) = VR(:,j),
            the j-th column of VR.
            If the j-th and (j+1)-st eigenvalues form a complex
            conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
            v(j+1) = VR(:,j) - i*VR(:,j+1).

    LDVR    (input) INTEGER
            The leading dimension of the array VR.  LDVR >= 1; if
            JOBVR = 'V', LDVR >= N.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,3*N), and
            if JOBVL = 'V' or JOBVR = 'V', LWORK >= 4*N.  For good
            performance, LWORK must generally be larger.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = i, the QR algorithm failed to compute all the
                  eigenvalues, and no eigenvectors have been computed;
                  elements i+1:N of WR and WI contain eigenvalues which
                  have converged.

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --wr;
    --wi;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    wantvl = lsame_(jobvl, "V");
    wantvr = lsame_(jobvr, "V");
    if (! wantvl && ! lsame_(jobvl, "N")) {
	*info = -1;
    } else if (! wantvr && ! lsame_(jobvr, "N")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldvl < 1 || wantvl && *ldvl < *n) {
	*info = -9;
    } else if (*ldvr < 1 || wantvr && *ldvr < *n) {
	*info = -11;
    }

/*
       Compute workspace
        (Note: Comments in the code beginning "Workspace:" describe the
         minimal amount of workspace needed at that point in the code,
         as well as the preferred amount for good performance.
         NB refers to the optimal block size for the immediately
         following subroutine, as returned by ILAENV.
         HSWORK refers to the workspace preferred by SHSEQR, as
         calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
         the worst case.)
*/

    minwrk = 1;
    if (*info == 0 && (*lwork >= 1 || lquery)) {
	maxwrk = (*n << 1) + *n * ilaenv_(&c__1, "SGEHRD", " ", n, &c__1, n, &
		c__0, (ftnlen)6, (ftnlen)1);
	if (! wantvl && ! wantvr) {
/* Computing MAX */
	    i__1 = 1, i__2 = *n * 3;
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "SHSEQR", "EN", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "SHSEQR", "EN", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = *n << 1;
	    hswork = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + 1, i__1 = max(i__1,i__2), i__2 = *n +
		    hswork;
	    maxwrk = max(i__1,i__2);
	} else {
/* Computing MAX */
	    i__1 = 1, i__2 = *n << 2;
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = (*n << 1) + (*n - 1) * ilaenv_(&c__1, "SOR"
		    "GHR", " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "SHSEQR", "SV", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "SHSEQR", "SV", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = *n << 1;
	    hswork = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + 1, i__1 = max(i__1,i__2), i__2 = *n +
		    hswork;
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n << 2;
	    maxwrk = max(i__1,i__2);
	}
	work[1] = (real) maxwrk;
    }
    if (*lwork < minwrk && ! lquery) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEEV ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Get machine constants */

    eps = slamch_("P");
    smlnum = slamch_("S");
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = 1.f / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = slange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0.f && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	slascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }

/*
       Balance the matrix
       (Workspace: need N)
*/

    ibal = 1;
    sgebal_("B", n, &a[a_offset], lda, &ilo, &ihi, &work[ibal], &ierr);

/*
       Reduce to upper Hessenberg form
       (Workspace: need 3*N, prefer 2*N+N*NB)
*/

    itau = ibal + *n;
    iwrk = itau + *n;
    i__1 = *lwork - iwrk + 1;
    sgehrd_(n, &ilo, &ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1,
	     &ierr);

    if (wantvl) {

/*
          Want left eigenvectors
          Copy Householder vectors to VL
*/

	*(unsigned char *)side = 'L';
	slacpy_("L", n, n, &a[a_offset], lda, &vl[vl_offset], ldvl)
		;

/*
          Generate orthogonal matrix in VL
          (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
*/

	i__1 = *lwork - iwrk + 1;
	sorghr_(n, &ilo, &ihi, &vl[vl_offset], ldvl, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VL
          (Workspace: need N+1, prefer N+HSWORK (see comments) )
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	shseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vl[vl_offset], ldvl, &work[iwrk], &i__1, info);

	if (wantvr) {

/*
             Want left and right eigenvectors
             Copy Schur vectors to VR
*/

	    *(unsigned char *)side = 'B';
	    slacpy_("F", n, n, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr);
	}

    } else if (wantvr) {

/*
          Want right eigenvectors
          Copy Householder vectors to VR
*/

	*(unsigned char *)side = 'R';
	slacpy_("L", n, n, &a[a_offset], lda, &vr[vr_offset], ldvr)
		;

/*
          Generate orthogonal matrix in VR
          (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
*/

	i__1 = *lwork - iwrk + 1;
	sorghr_(n, &ilo, &ihi, &vr[vr_offset], ldvr, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VR
          (Workspace: need N+1, prefer N+HSWORK (see comments) )
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	shseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vr[vr_offset], ldvr, &work[iwrk], &i__1, info);

    } else {

/*
          Compute eigenvalues only
          (Workspace: need N+1, prefer N+HSWORK (see comments) )
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	shseqr_("E", "N", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vr[vr_offset], ldvr, &work[iwrk], &i__1, info);
    }

/*     If INFO > 0 from SHSEQR, then quit */

    if (*info > 0) {
	goto L50;
    }

    if (wantvl || wantvr) {

/*
          Compute left and/or right eigenvectors
          (Workspace: need 4*N)
*/

	strevc_(side, "B", select, n, &a[a_offset], lda, &vl[vl_offset], ldvl,
		 &vr[vr_offset], ldvr, n, &nout, &work[iwrk], &ierr);
    }

    if (wantvl) {

/*
          Undo balancing of left eigenvectors
          (Workspace: need N)
*/

	sgebak_("B", "L", n, &ilo, &ihi, &work[ibal], n, &vl[vl_offset], ldvl,
		 &ierr);

/*        Normalize left eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.f) {
		scl = 1.f / snrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		sscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.f) {
		r__1 = snrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		r__2 = snrm2_(n, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		scl = 1.f / slapy2_(&r__1, &r__2);
		sscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
		sscal_(n, &scl, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
/* Computing 2nd power */
		    r__1 = vl[k + i__ * vl_dim1];
/* Computing 2nd power */
		    r__2 = vl[k + (i__ + 1) * vl_dim1];
		    work[iwrk + k - 1] = r__1 * r__1 + r__2 * r__2;
/* L10: */
		}
		k = isamax_(n, &work[iwrk], &c__1);
		slartg_(&vl[k + i__ * vl_dim1], &vl[k + (i__ + 1) * vl_dim1],
			&cs, &sn, &r__);
		srot_(n, &vl[i__ * vl_dim1 + 1], &c__1, &vl[(i__ + 1) *
			vl_dim1 + 1], &c__1, &cs, &sn);
		vl[k + (i__ + 1) * vl_dim1] = 0.f;
	    }
/* L20: */
	}
    }

    if (wantvr) {

/*
          Undo balancing of right eigenvectors
          (Workspace: need N)
*/

	sgebak_("B", "R", n, &ilo, &ihi, &work[ibal], n, &vr[vr_offset], ldvr,
		 &ierr);

/*        Normalize right eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.f) {
		scl = 1.f / snrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		sscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.f) {
		r__1 = snrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		r__2 = snrm2_(n, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		scl = 1.f / slapy2_(&r__1, &r__2);
		sscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
		sscal_(n, &scl, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
/* Computing 2nd power */
		    r__1 = vr[k + i__ * vr_dim1];
/* Computing 2nd power */
		    r__2 = vr[k + (i__ + 1) * vr_dim1];
		    work[iwrk + k - 1] = r__1 * r__1 + r__2 * r__2;
/* L30: */
		}
		k = isamax_(n, &work[iwrk], &c__1);
		slartg_(&vr[k + i__ * vr_dim1], &vr[k + (i__ + 1) * vr_dim1],
			&cs, &sn, &r__);
		srot_(n, &vr[i__ * vr_dim1 + 1], &c__1, &vr[(i__ + 1) *
			vr_dim1 + 1], &c__1, &cs, &sn);
		vr[k + (i__ + 1) * vr_dim1] = 0.f;
	    }
/* L40: */
	}
    }

/*     Undo scaling if necessary */

L50:
    if (scalea) {
	i__1 = *n - *info;
/* Computing MAX */
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	slascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[*info +
		1], &i__2, &ierr);
	i__1 = *n - *info;
/* Computing MAX */
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	slascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[*info +
		1], &i__2, &ierr);
	if (*info > 0) {
	    i__1 = ilo - 1;
	    slascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[1],
		    n, &ierr);
	    i__1 = ilo - 1;
	    slascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[1],
		    n, &ierr);
	}
    }

    work[1] = (real) maxwrk;
    return 0;

/*     End of SGEEV */

} /* sgeev_ */

/* Subroutine */ int sgehd2_(integer *n, integer *ilo, integer *ihi, real *a,
	integer *lda, real *tau, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__;
    static real aii;
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *), slarfg_(integer *, real *, real *,
	    integer *, real *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SGEHD2 reduces a real general matrix A to upper Hessenberg form H by
    an orthogonal similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to SGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= max(1,N).

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the n by n general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the orthogonal matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) REAL array, dimension (N)

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrix Q is represented as a product of (ihi-ilo) elementary
    reflectors

       Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
    exit in A(i+2:ihi,i), and tau in TAU(i).

    The contents of A are illustrated by the following example, with
    n = 7, ilo = 2 and ihi = 6:

    on entry,                        on exit,

    ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
    (                         a )    (                          a )

    where a denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEHD2", &i__1);
	return 0;
    }

    i__1 = *ihi - 1;
    for (i__ = *ilo; i__ <= i__1; ++i__) {

/*        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i) */

	i__2 = *ihi - i__;
/* Computing MIN */
	i__3 = i__ + 2;
	slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		a_dim1], &c__1, &tau[i__]);
	aii = a[i__ + 1 + i__ * a_dim1];
	a[i__ + 1 + i__ * a_dim1] = 1.f;

/*        Apply H(i) to A(1:ihi,i+1:ihi) from the right */

	i__2 = *ihi - i__;
	slarf_("Right", ihi, &i__2, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[(i__ + 1) * a_dim1 + 1], lda, &work[1]);

/*        Apply H(i) to A(i+1:ihi,i+1:n) from the left */

	i__2 = *ihi - i__;
	i__3 = *n - i__;
	slarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &work[1]);

	a[i__ + 1 + i__ * a_dim1] = aii;
/* L10: */
    }

    return 0;

/*     End of SGEHD2 */

} /* sgehd2_ */

/* Subroutine */ int sgehrd_(integer *n, integer *ilo, integer *ihi, real *a,
	integer *lda, real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__;
    static real t[4160]	/* was [65][64] */;
    static integer ib;
    static real ei;
    static integer nb, nh, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *), sgehd2_(integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *), slarfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *,
	    real *, integer *, real *, integer *, real *, integer *, real *,
	    integer *), slahrd_(integer *,
	    integer *, integer *, real *, integer *, real *, real *, integer *
	    , real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SGEHRD reduces a real general matrix A to upper Hessenberg form H by
    an orthogonal similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to SGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the N-by-N general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the orthogonal matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
            zero.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,N).
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrix Q is represented as a product of (ihi-ilo) elementary
    reflectors

       Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
    exit in A(i+2:ihi,i), and tau in TAU(i).

    The contents of A are illustrated by the following example, with
    n = 7, ilo = 2 and ihi = 6:

    on entry,                        on exit,

    ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
    (                         a )    (                          a )

    where a denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
/* Computing MIN */
    i__1 = 64, i__2 = ilaenv_(&c__1, "SGEHRD", " ", n, ilo, ihi, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = min(i__1,i__2);
    lwkopt = *n * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEHRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Set elements 1:ILO-1 and IHI:N-1 of TAU to zero */

    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	tau[i__] = 0.f;
/* L10: */
    }
    i__1 = *n - 1;
    for (i__ = max(1,*ihi); i__ <= i__1; ++i__) {
	tau[i__] = 0.f;
/* L20: */
    }

/*     Quick return if possible */

    nh = *ihi - *ilo + 1;
    if (nh <= 1) {
	work[1] = 1.f;
	return 0;
    }

/*
       Determine the block size.

   Computing MIN
*/
    i__1 = 64, i__2 = ilaenv_(&c__1, "SGEHRD", " ", n, ilo, ihi, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = min(i__1,i__2);
    nbmin = 2;
    iws = 1;
    if (nb > 1 && nb < nh) {

/*
          Determine when to cross over from blocked to unblocked code
          (last block is always handled by unblocked code).

   Computing MAX
*/
	i__1 = nb, i__2 = ilaenv_(&c__3, "SGEHRD", " ", n, ilo, ihi, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < nh) {

/*           Determine if workspace is large enough for blocked code. */

	    iws = *n * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  determine the
                minimum value of NB, and reduce NB or force use of
                unblocked code.

   Computing MAX
*/
		i__1 = 2, i__2 = ilaenv_(&c__2, "SGEHRD", " ", n, ilo, ihi, &
			c_n1, (ftnlen)6, (ftnlen)1);
		nbmin = max(i__1,i__2);
		if (*lwork >= *n * nbmin) {
		    nb = *lwork / *n;
		} else {
		    nb = 1;
		}
	    }
	}
    }
    ldwork = *n;

    if (nb < nbmin || nb >= nh) {

/*        Use unblocked code below */

	i__ = *ilo;

    } else {

/*        Use blocked code */

	i__1 = *ihi - 1 - nx;
	i__2 = nb;
	for (i__ = *ilo; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__3 = nb, i__4 = *ihi - i__;
	    ib = min(i__3,i__4);

/*
             Reduce columns i:i+ib-1 to Hessenberg form, returning the
             matrices V and T of the block reflector H = I - V*T*V'
             which performs the reduction, and also the matrix Y = A*V*T
*/

	    slahrd_(ihi, &i__, &ib, &a[i__ * a_dim1 + 1], lda, &tau[i__], t, &
		    c__65, &work[1], &ldwork);

/*
             Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
             right, computing  A := A - Y * V'. V(i+ib,ib-1) must be set
             to 1.
*/

	    ei = a[i__ + ib + (i__ + ib - 1) * a_dim1];
	    a[i__ + ib + (i__ + ib - 1) * a_dim1] = 1.f;
	    i__3 = *ihi - i__ - ib + 1;
	    sgemm_("No transpose", "Transpose", ihi, &i__3, &ib, &c_b151, &
		    work[1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda, &
		    c_b15, &a[(i__ + ib) * a_dim1 + 1], lda);
	    a[i__ + ib + (i__ + ib - 1) * a_dim1] = ei;

/*
             Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
             left
*/

	    i__3 = *ihi - i__;
	    i__4 = *n - i__ - ib + 1;
	    slarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
		    i__4, &ib, &a[i__ + 1 + i__ * a_dim1], lda, t, &c__65, &a[
		    i__ + 1 + (i__ + ib) * a_dim1], lda, &work[1], &ldwork);
/* L30: */
	}
    }

/*     Use unblocked code to reduce the rest of the matrix */

    sgehd2_(n, &i__, ihi, &a[a_offset], lda, &tau[1], &work[1], &iinfo);
    work[1] = (real) iws;

    return 0;

/*     End of SGEHRD */

} /* sgehrd_ */

/* Subroutine */ int sgelq2_(integer *m, integer *n, real *a, integer *lda,
	real *tau, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, k;
    static real aii;
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *), slarfg_(integer *, real *, real *,
	    integer *, real *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SGELQ2 computes an LQ factorization of a real m by n matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m by min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) REAL array, dimension (M)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(k) . . . H(2) H(1), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
    and tau in TAU(i).

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGELQ2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i,i+1:n) */

	i__2 = *n - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) * a_dim1]
		, lda, &tau[i__]);
	if (i__ < *m) {

/*           Apply H(i) to A(i+1:m,i:n) from the right */

	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    slarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
/* L10: */
    }
    return 0;

/*     End of SGELQ2 */

} /* sgelq2_ */

/* Subroutine */ int sgelqf_(integer *m, integer *n, real *a, integer *lda,
	real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int sgelq2_(integer *, integer *, real *, integer
	    *, real *, real *, integer *), slarfb_(char *, char *, char *,
	    char *, integer *, integer *, integer *, real *, integer *, real *
	    , integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SGELQF computes an LQ factorization of a real M-by-N matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,M).
            For optimum performance LWORK >= M*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(k) . . . H(2) H(1), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
    and tau in TAU(i).

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nb = ilaenv_(&c__1, "SGELQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *m * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGELQF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    nx = 0;
    iws = *m;
    if (nb > 1 && nb < k) {

/*
          Determine when to cross over from blocked to unblocked code.

   Computing MAX
*/
	i__1 = 0, i__2 = ilaenv_(&c__3, "SGELQF", " ", m, n, &c_n1, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < k) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  reduce NB and
                determine the minimum value of NB.
*/

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "SGELQF", " ", m, n, &c_n1, &
			c_n1, (ftnlen)6, (ftnlen)1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < k && nx < k) {

/*        Use blocked code initially */

	i__1 = k - nx;
	i__2 = nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__3 = k - i__ + 1;
	    ib = min(i__3,nb);

/*
             Compute the LQ factorization of the current block
             A(i:i+ib-1,i:n)
*/

	    i__3 = *n - i__ + 1;
	    sgelq2_(&ib, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *m) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *n - i__ + 1;
		slarft_("Forward", "Rowwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i+ib:m,i:n) from the right */

		i__3 = *m - i__ - ib + 1;
		i__4 = *n - i__ + 1;
		slarfb_("Right", "No transpose", "Forward", "Rowwise", &i__3,
			&i__4, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[ib +
			1], &ldwork);
	    }
/* L10: */
	}
    } else {
	i__ = 1;
    }

/*     Use unblocked code to factor the last or only block. */

    if (i__ <= k) {
	i__2 = *m - i__ + 1;
	i__1 = *n - i__ + 1;
	sgelq2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1] = (real) iws;
    return 0;

/*     End of SGELQF */

} /* sgelqf_ */

/* Subroutine */ int sgeqr2_(integer *m, integer *n, real *a, integer *lda,
	real *tau, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, k;
    static real aii;
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *), slarfg_(integer *, real *, real *,
	    integer *, real *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SGEQR2 computes a QR factorization of a real m by n matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) REAL array, dimension (N)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEQR2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	i__2 = *m - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ * a_dim1]
		, &c__1, &tau[i__]);
	if (i__ < *n) {

/*           Apply H(i) to A(i:m,i+1:n) from the left */

	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    slarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
/* L10: */
    }
    return 0;

/*     End of SGEQR2 */

} /* sgeqr2_ */

/* Subroutine */ int sgeqrf_(integer *m, integer *n, real *a, integer *lda,
	real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int sgeqr2_(integer *, integer *, real *, integer
	    *, real *, real *, integer *), slarfb_(char *, char *, char *,
	    char *, integer *, integer *, integer *, real *, integer *, real *
	    , integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,N).
            For optimum performance LWORK >= N*NB, where NB is
            the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nb = ilaenv_(&c__1, "SGEQRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *n * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    nx = 0;
    iws = *n;
    if (nb > 1 && nb < k) {

/*
          Determine when to cross over from blocked to unblocked code.

   Computing MAX
*/
	i__1 = 0, i__2 = ilaenv_(&c__3, "SGEQRF", " ", m, n, &c_n1, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < k) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  reduce NB and
                determine the minimum value of NB.
*/

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "SGEQRF", " ", m, n, &c_n1, &
			c_n1, (ftnlen)6, (ftnlen)1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < k && nx < k) {

/*        Use blocked code initially */

	i__1 = k - nx;
	i__2 = nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__3 = k - i__ + 1;
	    ib = min(i__3,nb);

/*
             Compute the QR factorization of the current block
             A(i:m,i:i+ib-1)
*/

	    i__3 = *m - i__ + 1;
	    sgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *n) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *m - i__ + 1;
		slarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i:m,i+ib:n) from the left */

		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		slarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
			i__4, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &work[ib
			+ 1], &ldwork);
	    }
/* L10: */
	}
    } else {
	i__ = 1;
    }

/*     Use unblocked code to factor the last or only block. */

    if (i__ <= k) {
	i__2 = *m - i__ + 1;
	i__1 = *n - i__ + 1;
	sgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1] = (real) iws;
    return 0;

/*     End of SGEQRF */

} /* sgeqrf_ */

/* Subroutine */ int sgesdd_(char *jobz, integer *m, integer *n, real *a,
	integer *lda, real *s, real *u, integer *ldu, real *vt, integer *ldvt,
	 real *work, integer *lwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1,
	    i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, ie, il, ir, iu, blk;
    static real dum[1], eps;
    static integer ivt, iscl;
    static real anrm;
    static integer idum[1], ierr, itau;
    extern logical lsame_(char *, char *);
    static integer chunk;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer minmn, wrkbl, itaup, itauq, mnthr;
    static logical wntqa;
    static integer nwork;
    static logical wntqn, wntqo, wntqs;
    static integer bdspac;
    extern /* Subroutine */ int sbdsdc_(char *, char *, integer *, real *,
	    real *, real *, integer *, real *, integer *, real *, integer *,
	    real *, integer *, integer *), sgebrd_(integer *,
	    integer *, real *, integer *, real *, real *, real *, real *,
	    real *, integer *, integer *);
    extern doublereal slamch_(char *), slange_(char *, integer *,
	    integer *, real *, integer *, real *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static real bignum;
    extern /* Subroutine */ int sgelqf_(integer *, integer *, real *, integer
	    *, real *, real *, integer *, integer *), slascl_(char *, integer
	    *, integer *, real *, real *, integer *, integer *, real *,
	    integer *, integer *), sgeqrf_(integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *), slacpy_(char
	    *, integer *, integer *, real *, integer *, real *, integer *), slaset_(char *, integer *, integer *, real *, real *,
	    real *, integer *), sorgbr_(char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, integer *
	    );
    static integer ldwrkl;
    extern /* Subroutine */ int sormbr_(char *, char *, char *, integer *,
	    integer *, integer *, real *, integer *, real *, real *, integer *
	    , real *, integer *, integer *);
    static integer ldwrkr, minwrk, ldwrku, maxwrk;
    extern /* Subroutine */ int sorglq_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *);
    static integer ldwkvt;
    static real smlnum;
    static logical wntqas;
    extern /* Subroutine */ int sorgqr_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *);
    static logical lquery;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SGESDD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and right singular
    vectors.  If singular vectors are desired, it uses a
    divide-and-conquer algorithm.

    The SVD is written

         A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**T, not V.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========

    JOBZ    (input) CHARACTER*1
            Specifies options for computing all or part of the matrix U:
            = 'A':  all M columns of U and all N rows of V**T are
                    returned in the arrays U and VT;
            = 'S':  the first min(M,N) columns of U and the first
                    min(M,N) rows of V**T are returned in the arrays U
                    and VT;
            = 'O':  If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**T are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**T are overwritten
                    in the array VT;
            = 'N':  no columns of U or rows of V**T are computed.

    M       (input) INTEGER
            The number of rows of the input matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the input matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
            if JOBZ = 'O',  A is overwritten with the first N columns
                            of U (the left singular vectors, stored
                            columnwise) if M >= N;
                            A is overwritten with the first M rows
                            of V**T (the right singular vectors, stored
                            rowwise) otherwise.
            if JOBZ .ne. 'O', the contents of A are destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    S       (output) REAL array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    U       (output) REAL array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
            UCOL = min(M,N) if JOBZ = 'S'.
            If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
            orthogonal matrix U;
            if JOBZ = 'S', U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
            if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.

    LDU     (input) INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.

    VT      (output) REAL array, dimension (LDVT,N)
            If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
            N-by-N orthogonal matrix V**T;
            if JOBZ = 'S', VT contains the first min(M,N) rows of
            V**T (the right singular vectors, stored rowwise);
            if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.

    LDVT    (input) INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
            if JOBZ = 'S', LDVT >= min(M,N).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK;

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= 1.
            If JOBZ = 'N',
              LWORK >= 3*min(M,N) + max(max(M,N),6*min(M,N)).
            If JOBZ = 'O',
              LWORK >= 3*min(M,N)*min(M,N) +
                       max(max(M,N),5*min(M,N)*min(M,N)+4*min(M,N)).
            If JOBZ = 'S' or 'A'
              LWORK >= 3*min(M,N)*min(M,N) +
                       max(max(M,N),4*min(M,N)*min(M,N)+4*min(M,N)).
            For good performance, LWORK should generally be larger.
            If LWORK < 0 but other input arguments are legal, WORK(1)
            returns the optimal LWORK.

    IWORK   (workspace) INTEGER array, dimension (8*min(M,N))

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  SBDSDC did not converge, updating process failed.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    mnthr = (integer) (minmn * 11.f / 6.f);
    wntqa = lsame_(jobz, "A");
    wntqs = lsame_(jobz, "S");
    wntqas = wntqa || wntqs;
    wntqo = lsame_(jobz, "O");
    wntqn = lsame_(jobz, "N");
    minwrk = 1;
    maxwrk = 1;
    lquery = *lwork == -1;

    if (! (wntqa || wntqs || wntqo || wntqn)) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldu < 1 || wntqas && *ldu < *m || wntqo && *m < *n && *ldu < *
	    m) {
	*info = -8;
    } else if (*ldvt < 1 || wntqa && *ldvt < *n || wntqs && *ldvt < minmn ||
	    wntqo && *m >= *n && *ldvt < *n) {
	*info = -10;
    }

/*
       Compute workspace
        (Note: Comments in the code beginning "Workspace:" describe the
         minimal amount of workspace needed at that point in the code,
         as well as the preferred amount for good performance.
         NB refers to the optimal block size for the immediately
         following subroutine, as returned by ILAENV.)
*/

    if (*info == 0 && *m > 0 && *n > 0) {
	if (*m >= *n) {

/*           Compute space needed for SBDSDC */

	    if (wntqn) {
		bdspac = *n * 7;
	    } else {
		bdspac = *n * 3 * *n + (*n << 2);
	    }
	    if (*m >= mnthr) {
		if (wntqn) {

/*                 Path 1 (M much larger than N, JOBZ='N') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "SGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *n;
		} else if (wntqo) {

/*                 Path 2 (M much larger than N, JOBZ='O') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "SGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "SORGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "QLN", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*n << 1) * *n;
		    minwrk = bdspac + (*n << 1) * *n + *n * 3;
		} else if (wntqs) {

/*                 Path 3 (M much larger than N, JOBZ='S') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "SGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "SORGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "QLN", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    minwrk = bdspac + *n * *n + *n * 3;
		} else if (wntqa) {

/*                 Path 4 (M much larger than N, JOBZ='A') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "SGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *m * ilaenv_(&c__1, "SORGQR",
			    " ", m, m, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "QLN", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    minwrk = bdspac + *n * *n + *n * 3;
		}
	    } else {

/*              Path 5 (M at least N, but not much larger) */

		wrkbl = *n * 3 + (*m + *n) * ilaenv_(&c__1, "SGEBRD", " ", m,
			n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		if (wntqn) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqo) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
/* Computing MAX */
		    i__1 = *m, i__2 = *n * *n + bdspac;
		    minwrk = *n * 3 + max(i__1,i__2);
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		}
	    }
	} else {

/*           Compute space needed for SBDSDC */

	    if (wntqn) {
		bdspac = *m * 7;
	    } else {
		bdspac = *m * 3 * *m + (*m << 2);
	    }
	    if (*n >= mnthr) {
		if (wntqn) {

/*                 Path 1t (N much larger than M, JOBZ='N') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "SGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *m;
		} else if (wntqo) {

/*                 Path 2t (N much larger than M, JOBZ='O') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "SGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "SORGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*m << 1) * *m;
		    minwrk = bdspac + (*m << 1) * *m + *m * 3;
		} else if (wntqs) {

/*                 Path 3t (N much larger than M, JOBZ='S') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "SGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "SORGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    minwrk = bdspac + *m * *m + *m * 3;
		} else if (wntqa) {

/*                 Path 4t (N much larger than M, JOBZ='A') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "SGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *n * ilaenv_(&c__1, "SORGLQ",
			    " ", n, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1,
			    "SGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)6, (
			    ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", m, m, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    minwrk = bdspac + *m * *m + *m * 3;
		}
	    } else {

/*              Path 5t (N greater than M, but not much larger) */

		wrkbl = *m * 3 + (*m + *n) * ilaenv_(&c__1, "SGEBRD", " ", m,
			n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		if (wntqn) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqo) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", m, n, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
/* Computing MAX */
		    i__1 = *n, i__2 = *m * *m + bdspac;
		    minwrk = *m * 3 + max(i__1,i__2);
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", m, n, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "QLN", m, m, n, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "SORMBR"
			    , "PRT", n, n, m, &c_n1, (ftnlen)6, (ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		}
	    }
	}
	work[1] = (real) maxwrk;
    }

    if (*lwork < minwrk && ! lquery) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGESDD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	if (*lwork >= 1) {
	    work[1] = 1.f;
	}
	return 0;
    }

/*     Get machine constants */

    eps = slamch_("P");
    smlnum = sqrt(slamch_("S")) / eps;
    bignum = 1.f / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = slange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0.f && anrm < smlnum) {
	iscl = 1;
	slascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	slascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
		ierr);
    }

    if (*m >= *n) {

/*
          A has at least as many rows as columns. If A has sufficiently
          more rows than columns, first reduce using the QR
          decomposition (if sufficient workspace available)
*/

	if (*m >= mnthr) {

	    if (wntqn) {

/*
                Path 1 (M much larger than N, JOBZ='N')
                No singular vectors to be computed
*/

		itau = 1;
		nwork = itau + *n;

/*
                Compute A=Q*R
                (Workspace: need 2*N, prefer N+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out below R */

		i__1 = *n - 1;
		i__2 = *n - 1;
		slaset_("L", &i__1, &i__2, &c_b29, &c_b29, &a[a_dim1 + 2],
			lda);
		ie = 1;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in A
                (Workspace: need 4*N, prefer 3*N+2*N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *n;

/*
                Perform bidiagonal SVD, computing singular values only
                (Workspace: need N+BDSPAC)
*/

		sbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);

	    } else if (wntqo) {

/*
                Path 2 (M much larger than N, JOBZ = 'O')
                N left singular vectors to be overwritten on A and
                N right singular vectors to be computed in VT
*/

		ir = 1;

/*              WORK(IR) is LDWRKR by N */

		if (*lwork >= *lda * *n + *n * *n + *n * 3 + bdspac) {
		    ldwrkr = *lda;
		} else {
		    ldwrkr = (*lwork - *n * *n - *n * 3 - bdspac) / *n;
		}
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*
                Compute A=Q*R
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		slacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		slaset_("L", &i__1, &i__2, &c_b29, &c_b29, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in VT, copying result to WORK(IR)
                (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              WORK(IU) is N by N */

		iu = nwork;
		nwork = iu + *n * *n;

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in WORK(IU) and computing right
                singular vectors of bidiagonal matrix in VT
                (Workspace: need N+N*N+BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite WORK(IU) by left singular vectors of R
                and VT by right singular vectors of R
                (Workspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], n, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*
                Multiply Q in A by left singular vectors of R in
                WORK(IU), storing result in WORK(IR) and copying to A
                (Workspace: need 2*N*N, prefer N*N+M*N)
*/

		i__1 = *m;
		i__2 = ldwrkr;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrkr);
		    sgemm_("N", "N", &chunk, n, n, &c_b15, &a[i__ + a_dim1],
			    lda, &work[iu], n, &c_b29, &work[ir], &ldwrkr);
		    slacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
			    a_dim1], lda);
/* L10: */
		}

	    } else if (wntqs) {

/*
                Path 3 (M much larger than N, JOBZ='S')
                N left singular vectors to be computed in U and
                N right singular vectors to be computed in VT
*/

		ir = 1;

/*              WORK(IR) is N by N */

		ldwrkr = *n;
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*
                Compute A=Q*R
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		slacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		slaset_("L", &i__2, &i__1, &c_b29, &c_b29, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in WORK(IR)
                (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagoal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need N+BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite U by left singular vectors of R and VT
                by right singular vectors of R
                (Workspace: need N*N+3*N, prefer N*N+2*N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		i__2 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in A by left singular vectors of R in
                WORK(IR), storing result in U
                (Workspace: need N*N)
*/

		slacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		sgemm_("N", "N", m, n, n, &c_b15, &a[a_offset], lda, &work[ir]
			, &ldwrkr, &c_b29, &u[u_offset], ldu);

	    } else if (wntqa) {

/*
                Path 4 (M much larger than N, JOBZ='A')
                M left singular vectors to be computed in U and
                N right singular vectors to be computed in VT
*/

		iu = 1;

/*              WORK(IU) is N by N */

		ldwrku = *n;
		itau = iu + ldwrku * *n;
		nwork = itau + *n;

/*
                Compute A=Q*R, copying result to U
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		slacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Generate Q in U
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/
		i__2 = *lwork - nwork + 1;
		sorgqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork],
			 &i__2, &ierr);

/*              Produce R in A, zeroing out other entries */

		i__2 = *n - 1;
		i__1 = *n - 1;
		slaset_("L", &i__2, &i__1, &c_b29, &c_b29, &a[a_dim1 + 2],
			lda);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in A
                (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in WORK(IU) and computing right
                singular vectors of bidiagonal matrix in VT
                (Workspace: need N+N*N+BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite WORK(IU) by left singular vectors of R and VT
                by right singular vectors of R
                (Workspace: need N*N+3*N, prefer N*N+2*N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);
		i__2 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in U by left singular vectors of R in
                WORK(IU), storing result in A
                (Workspace: need N*N)
*/

		sgemm_("N", "N", m, n, n, &c_b15, &u[u_offset], ldu, &work[iu]
			, &ldwrku, &c_b29, &a[a_offset], lda);

/*              Copy left singular vectors of A from A to U */

		slacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);

	    }

	} else {

/*
             M .LT. MNTHR

             Path 5 (M at least N, but not much larger)
             Reduce to bidiagonal form without QR decomposition
*/

	    ie = 1;
	    itauq = ie + *n;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*
             Bidiagonalize A
             (Workspace: need 3*N+M, prefer 3*N+(M+N)*NB)
*/

	    i__2 = *lwork - nwork + 1;
	    sgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Perform bidiagonal SVD, only computing singular values
                (Workspace: need N+BDSPAC)
*/

		sbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		if (*lwork >= *m * *n + *n * 3 + bdspac) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		    nwork = iu + ldwrku * *n;
		    slaset_("F", m, n, &c_b29, &c_b29, &work[iu], &ldwrku);
		} else {

/*                 WORK( IU ) is N by N */

		    ldwrku = *n;
		    nwork = iu + ldwrku * *n;

/*                 WORK(IR) is LDWRKR by N */

		    ir = nwork;
		    ldwrkr = (*lwork - *n * *n - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in WORK(IU) and computing right
                singular vectors of bidiagonal matrix in VT
                (Workspace: need N+N*N+BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], &ldwrku, &
			vt[vt_offset], ldvt, dum, idum, &work[nwork], &iwork[
			1], info);

/*
                Overwrite VT by right singular vectors of A
                (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

		if (*lwork >= *m * *n + *n * 3 + bdspac) {

/*
                   Overwrite WORK(IU) by left singular vectors of A
                   (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		    i__2 = *lwork - nwork + 1;
		    sormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			    ierr);

/*                 Copy left singular vectors of A from WORK(IU) to A */

		    slacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {

/*
                   Generate Q in A
                   (Workspace: need N*N+2*N, prefer N*N+N+N*NB)
*/

		    i__2 = *lwork - nwork + 1;
		    sorgbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
			    work[nwork], &i__2, &ierr);

/*
                   Multiply Q in A by left singular vectors of
                   bidiagonal matrix in WORK(IU), storing result in
                   WORK(IR) and copying to A
                   (Workspace: need 2*N*N, prefer N*N+M*N)
*/

		    i__2 = *m;
		    i__1 = ldwrkr;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *m - i__ + 1;
			chunk = min(i__3,ldwrkr);
			sgemm_("N", "N", &chunk, n, n, &c_b15, &a[i__ +
				a_dim1], lda, &work[iu], &ldwrku, &c_b29, &
				work[ir], &ldwrkr);
			slacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
				a_dim1], lda);
/* L20: */
		    }
		}

	    } else if (wntqs) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need N+BDSPAC)
*/

		slaset_("F", m, n, &c_b29, &c_b29, &u[u_offset], ldu);
		sbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite U by left singular vectors of A and VT
                by right singular vectors of A
                (Workspace: need 3*N, prefer 2*N+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need N+BDSPAC)
*/

		slaset_("F", m, m, &c_b29, &c_b29, &u[u_offset], ldu);
		sbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*              Set the right corner of U to identity matrix */

		i__1 = *m - *n;
		i__2 = *m - *n;
		slaset_("F", &i__1, &i__2, &c_b29, &c_b15, &u[*n + 1 + (*n +
			1) * u_dim1], ldu);

/*
                Overwrite U by left singular vectors of A and VT
                by right singular vectors of A
                (Workspace: need N*N+2*N+M, prefer N*N+2*N+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    } else {

/*
          A has more columns than rows. If A has sufficiently more
          columns than rows, first reduce using the LQ decomposition (if
          sufficient workspace available)
*/

	if (*n >= mnthr) {

	    if (wntqn) {

/*
                Path 1t (N much larger than M, JOBZ='N')
                No singular vectors to be computed
*/

		itau = 1;
		nwork = itau + *m;

/*
                Compute A=L*Q
                (Workspace: need 2*M, prefer M+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out above L */

		i__1 = *m - 1;
		i__2 = *m - 1;
		slaset_("U", &i__1, &i__2, &c_b29, &c_b29, &a[(a_dim1 << 1) +
			1], lda);
		ie = 1;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in A
                (Workspace: need 4*M, prefer 3*M+2*M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *m;

/*
                Perform bidiagonal SVD, computing singular values only
                (Workspace: need M+BDSPAC)
*/

		sbdsdc_("U", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);

	    } else if (wntqo) {

/*
                Path 2t (N much larger than M, JOBZ='O')
                M right singular vectors to be overwritten on A and
                M left singular vectors to be computed in U
*/

		ivt = 1;

/*              IVT is M by M */

		il = ivt + *m * *m;
		if (*lwork >= *m * *n + *m * *m + *m * 3 + bdspac) {

/*                 WORK(IL) is M by N */

		    ldwrkl = *m;
		    chunk = *n;
		} else {
		    ldwrkl = *m;
		    chunk = (*lwork - *m * *m) / *m;
		}
		itau = il + ldwrkl * *m;
		nwork = itau + *m;

/*
                Compute A=L*Q
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy L to WORK(IL), zeroing about above it */

		slacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		slaset_("U", &i__1, &i__2, &c_b29, &c_b29, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in WORK(IL)
                (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U, and computing right singular
                vectors of bidiagonal matrix in WORK(IVT)
                (Workspace: need M+M*M+BDSPAC)
*/

		sbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], m, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite U by left singular vectors of L and WORK(IVT)
                by right singular vectors of L
                (Workspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &work[ivt], m, &work[nwork], &i__1, &ierr);

/*
                Multiply right singular vectors of L in WORK(IVT) by Q
                in A, storing result in WORK(IL) and copying to A
                (Workspace: need 2*M*M, prefer M*M+M*N)
*/

		i__1 = *n;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    sgemm_("N", "N", m, &blk, m, &c_b15, &work[ivt], m, &a[
			    i__ * a_dim1 + 1], lda, &c_b29, &work[il], &
			    ldwrkl);
		    slacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1
			    + 1], lda);
/* L30: */
		}

	    } else if (wntqs) {

/*
                Path 3t (N much larger than M, JOBZ='S')
                M right singular vectors to be computed in VT and
                M left singular vectors to be computed in U
*/

		il = 1;

/*              WORK(IL) is M by M */

		ldwrkl = *m;
		itau = il + ldwrkl * *m;
		nwork = itau + *m;

/*
                Compute A=L*Q
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy L to WORK(IL), zeroing out above it */

		slacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		slaset_("U", &i__2, &i__1, &c_b29, &c_b29, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in WORK(IU), copying result to U
                (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need M+BDSPAC)
*/

		sbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite U by left singular vectors of L and VT
                by right singular vectors of L
                (Workspace: need M*M+3*M, prefer M*M+2*M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);
		i__2 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply right singular vectors of L in WORK(IL) by
                Q in A, storing result in VT
                (Workspace: need M*M)
*/

		slacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		sgemm_("N", "N", m, n, m, &c_b15, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b29, &vt[vt_offset], ldvt);

	    } else if (wntqa) {

/*
                Path 4t (N much larger than M, JOBZ='A')
                N right singular vectors to be computed in VT and
                M left singular vectors to be computed in U
*/

		ivt = 1;

/*              WORK(IVT) is M by M */

		ldwkvt = *m;
		itau = ivt + ldwkvt * *m;
		nwork = itau + *m;

/*
                Compute A=L*Q, copying result to VT
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		slacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Generate Q in VT
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sorglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__2, &ierr);

/*              Produce L in A, zeroing out other entries */

		i__2 = *m - 1;
		i__1 = *m - 1;
		slaset_("U", &i__2, &i__1, &c_b29, &c_b29, &a[(a_dim1 << 1) +
			1], lda);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in A
                (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in WORK(IVT)
                (Workspace: need M+M*M+BDSPAC)
*/

		sbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
			, info);

/*
                Overwrite U by left singular vectors of L and WORK(IVT)
                by right singular vectors of L
                (Workspace: need M*M+3*M, prefer M*M+2*M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);
		i__2 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply right singular vectors of L in WORK(IVT) by
                Q in VT, storing result in A
                (Workspace: need M*M)
*/

		sgemm_("N", "N", m, n, m, &c_b15, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b29, &a[a_offset], lda);

/*              Copy right singular vectors of A from A to VT */

		slacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

	    }

	} else {

/*
             N .LT. MNTHR

             Path 5t (N greater than M, but not much larger)
             Reduce to bidiagonal form without LQ decomposition
*/

	    ie = 1;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*
             Bidiagonalize A
             (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB)
*/

	    i__2 = *lwork - nwork + 1;
	    sgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Perform bidiagonal SVD, only computing singular values
                (Workspace: need M+BDSPAC)
*/

		sbdsdc_("L", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3 + bdspac) {

/*                 WORK( IVT ) is M by N */

		    slaset_("F", m, n, &c_b29, &c_b29, &work[ivt], &ldwkvt);
		    nwork = ivt + ldwkvt * *n;
		} else {

/*                 WORK( IVT ) is M by M */

		    nwork = ivt + ldwkvt * *m;
		    il = nwork;

/*                 WORK(IL) is M by CHUNK */

		    chunk = (*lwork - *m * *m - *m * 3) / *m;
		}

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in WORK(IVT)
                (Workspace: need M*M+BDSPAC)
*/

		sbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
			, info);

/*
                Overwrite U by left singular vectors of A
                (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		i__2 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *m * 3 + bdspac) {

/*
                   Overwrite WORK(IVT) by left singular vectors of A
                   (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		    i__2 = *lwork - nwork + 1;
		    sormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2,
			    &ierr);

/*                 Copy right singular vectors of A from WORK(IVT) to A */

		    slacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda);
		} else {

/*
                   Generate P**T in A
                   (Workspace: need M*M+2*M, prefer M*M+M+M*NB)
*/

		    i__2 = *lwork - nwork + 1;
		    sorgbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
			    work[nwork], &i__2, &ierr);

/*
                   Multiply Q in A by right singular vectors of
                   bidiagonal matrix in WORK(IVT), storing result in
                   WORK(IL) and copying to A
                   (Workspace: need 2*M*M, prefer M*M+M*N)
*/

		    i__2 = *n;
		    i__1 = chunk;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *n - i__ + 1;
			blk = min(i__3,chunk);
			sgemm_("N", "N", m, &blk, m, &c_b15, &work[ivt], &
				ldwkvt, &a[i__ * a_dim1 + 1], lda, &c_b29, &
				work[il], m);
			slacpy_("F", m, &blk, &work[il], m, &a[i__ * a_dim1 +
				1], lda);
/* L40: */
		    }
		}
	    } else if (wntqs) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need M+BDSPAC)
*/

		slaset_("F", m, n, &c_b29, &c_b29, &vt[vt_offset], ldvt);
		sbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*
                Overwrite U by left singular vectors of A and VT
                by right singular vectors of A
                (Workspace: need 3*M, prefer 2*M+M*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in U and computing right singular
                vectors of bidiagonal matrix in VT
                (Workspace: need M+BDSPAC)
*/

		slaset_("F", n, n, &c_b29, &c_b29, &vt[vt_offset], ldvt);
		sbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);

/*              Set the right corner of VT to identity matrix */

		i__1 = *n - *m;
		i__2 = *n - *m;
		slaset_("F", &i__1, &i__2, &c_b29, &c_b15, &vt[*m + 1 + (*m +
			1) * vt_dim1], ldvt);

/*
                Overwrite U by left singular vectors of A and VT
                by right singular vectors of A
                (Workspace: need 2*M+N, prefer 2*M+N*NB)
*/

		i__1 = *lwork - nwork + 1;
		sormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		sormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    }

/*     Undo scaling if necessary */

    if (iscl == 1) {
	if (anrm > bignum) {
	    slascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (anrm < smlnum) {
	    slascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
    }

/*     Return optimal workspace in WORK(1) */

    work[1] = (real) maxwrk;

    return 0;

/*     End of SGESDD */

} /* sgesdd_ */

/* Subroutine */ int sgesv_(integer *n, integer *nrhs, real *a, integer *lda,
	integer *ipiv, real *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern /* Subroutine */ int xerbla_(char *, integer *), sgetrf_(
	    integer *, integer *, real *, integer *, integer *, integer *),
	    sgetrs_(char *, integer *, integer *, real *, integer *, integer *
	    , real *, integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SGESV computes the solution to a real system of linear equations
       A * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.

    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as
       A = P * L * U,
    where P is a permutation matrix, L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    =========

    N       (input) INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (output) INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).

    B       (input/output) REAL array, dimension (LDB,NRHS)
            On entry, the N-by-NRHS matrix of right hand side matrix B.
            On exit, if INFO = 0, the N-by-NRHS solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, U(i,i) is exactly zero.  The factorization
                  has been completed, but the factor U is exactly
                  singular, so the solution could not be computed.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGESV ", &i__1);
	return 0;
    }

/*     Compute the LU factorization of A. */

    sgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);
    if (*info == 0) {

/*        Solve the system A*X = B, overwriting B with X. */

	sgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &b[
		b_offset], ldb, info);
    }
    return 0;

/*     End of SGESV */

} /* sgesv_ */

/* Subroutine */ int sgetf2_(integer *m, integer *n, real *a, integer *lda,
	integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;

    /* Local variables */
    static integer j, jp;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *), sscal_(integer *
	    , real *, real *, integer *), sswap_(integer *, real *, integer *,
	     real *, integer *), xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1992


    Purpose
    =======

    SGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGETF2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

    i__1 = min(*m,*n);
    for (j = 1; j <= i__1; ++j) {

/*        Find pivot and test for singularity. */

	i__2 = *m - j + 1;
	jp = j - 1 + isamax_(&i__2, &a[j + j * a_dim1], &c__1);
	ipiv[j] = jp;
	if (a[jp + j * a_dim1] != 0.f) {

/*           Apply the interchange to columns 1:N. */

	    if (jp != j) {
		sswap_(n, &a[j + a_dim1], lda, &a[jp + a_dim1], lda);
	    }

/*           Compute elements J+1:M of J-th column. */

	    if (j < *m) {
		i__2 = *m - j;
		r__1 = 1.f / a[j + j * a_dim1];
		sscal_(&i__2, &r__1, &a[j + 1 + j * a_dim1], &c__1);
	    }

	} else if (*info == 0) {

	    *info = j;
	}

	if (j < min(*m,*n)) {

/*           Update trailing submatrix. */

	    i__2 = *m - j;
	    i__3 = *n - j;
	    sger_(&i__2, &i__3, &c_b151, &a[j + 1 + j * a_dim1], &c__1, &a[j
		    + (j + 1) * a_dim1], lda, &a[j + 1 + (j + 1) * a_dim1],
		    lda);
	}
/* L10: */
    }
    return 0;

/*     End of SGETF2 */

} /* sgetf2_ */

/* Subroutine */ int sgetrf_(integer *m, integer *n, real *a, integer *lda,
	integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    static integer i__, j, jb, nb, iinfo;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *), strsm_(char *, char *, char *,
	     char *, integer *, integer *, real *, real *, integer *, real *,
	    integer *), sgetf2_(integer *,
	    integer *, real *, integer *, integer *, integer *), xerbla_(char
	    *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slaswp_(integer *, real *, integer *, integer
	    *, integer *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGETRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "SGETRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    if (nb <= 1 || nb >= min(*m,*n)) {

/*        Use unblocked code. */

	sgetf2_(m, n, &a[a_offset], lda, &ipiv[1], info);
    } else {

/*        Use blocked code. */

	i__1 = min(*m,*n);
	i__2 = nb;
	for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
/* Computing MIN */
	    i__3 = min(*m,*n) - j + 1;
	    jb = min(i__3,nb);

/*
             Factor diagonal and subdiagonal blocks and test for exact
             singularity.
*/

	    i__3 = *m - j + 1;
	    sgetf2_(&i__3, &jb, &a[j + j * a_dim1], lda, &ipiv[j], &iinfo);

/*           Adjust INFO and the pivot indices. */

	    if (*info == 0 && iinfo > 0) {
		*info = iinfo + j - 1;
	    }
/* Computing MIN */
	    i__4 = *m, i__5 = j + jb - 1;
	    i__3 = min(i__4,i__5);
	    for (i__ = j; i__ <= i__3; ++i__) {
		ipiv[i__] = j - 1 + ipiv[i__];
/* L10: */
	    }

/*           Apply interchanges to columns 1:J-1. */

	    i__3 = j - 1;
	    i__4 = j + jb - 1;
	    slaswp_(&i__3, &a[a_offset], lda, &j, &i__4, &ipiv[1], &c__1);

	    if (j + jb <= *n) {

/*              Apply interchanges to columns J+JB:N. */

		i__3 = *n - j - jb + 1;
		i__4 = j + jb - 1;
		slaswp_(&i__3, &a[(j + jb) * a_dim1 + 1], lda, &j, &i__4, &
			ipiv[1], &c__1);

/*              Compute block row of U. */

		i__3 = *n - j - jb + 1;
		strsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
			c_b15, &a[j + j * a_dim1], lda, &a[j + (j + jb) *
			a_dim1], lda);
		if (j + jb <= *m) {

/*                 Update trailing submatrix. */

		    i__3 = *m - j - jb + 1;
		    i__4 = *n - j - jb + 1;
		    sgemm_("No transpose", "No transpose", &i__3, &i__4, &jb,
			    &c_b151, &a[j + jb + j * a_dim1], lda, &a[j + (j
			    + jb) * a_dim1], lda, &c_b15, &a[j + jb + (j + jb)
			     * a_dim1], lda);
		}
	    }
/* L20: */
	}
    }
    return 0;

/*     End of SGETRF */

} /* sgetrf_ */

/* Subroutine */ int sgetrs_(char *trans, integer *n, integer *nrhs, real *a,
	integer *lda, integer *ipiv, real *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *,
	    integer *, integer *, real *, real *, integer *, real *, integer *
	    ), xerbla_(char *, integer *);
    static logical notran;
    extern /* Subroutine */ int slaswp_(integer *, real *, integer *, integer
	    *, integer *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SGETRS solves a system of linear equations
       A * X = B  or  A' * X = B
    with a general N-by-N matrix A using the LU factorization computed
    by SGETRF.

    Arguments
    =========

    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B  (No transpose)
            = 'T':  A'* X = B  (Transpose)
            = 'C':  A'* X = B  (Conjugate transpose = Transpose)

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) REAL array, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U
            as computed by SGETRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from SGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) REAL array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    notran = lsame_(trans, "N");
    if (! notran && ! lsame_(trans, "T") && ! lsame_(
	    trans, "C")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGETRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

    if (notran) {

/*
          Solve A * X = B.

          Apply row interchanges to the right hand sides.
*/

	slaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);

/*        Solve L*X = B, overwriting B with X. */

	strsm_("Left", "Lower", "No transpose", "Unit", n, nrhs, &c_b15, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	strsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b15, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A' * X = B.

          Solve U'*X = B, overwriting B with X.
*/

	strsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b15, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	strsm_("Left", "Lower", "Transpose", "Unit", n, nrhs, &c_b15, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Apply row interchanges to the solution vectors. */

	slaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return 0;

/*     End of SGETRS */

} /* sgetrs_ */

/* Subroutine */ int shseqr_(char *job, char *compz, integer *n, integer *ilo,
	 integer *ihi, real *h__, integer *ldh, real *wr, real *wi, real *z__,
	 integer *ldz, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3[2], i__4,
	    i__5;
    real r__1, r__2;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__, j, k, l;
    static real s[225]	/* was [15][15] */, v[16];
    static integer i1, i2, ii, nh, nr, ns, nv;
    static real vv[16];
    static integer itn;
    static real tau;
    static integer its;
    static real ulp, tst1;
    static integer maxb;
    static real absw;
    static integer ierr;
    static real unfl, temp, ovfl;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static integer itemp;
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *);
    static logical initz, wantt;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    static logical wantz;
    extern doublereal slapy2_(real *, real *);
    extern /* Subroutine */ int slabad_(real *, real *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarfg_(integer *, real *, real *, integer *,
	    real *);
    extern integer isamax_(integer *, real *, integer *);
    extern doublereal slanhs_(char *, integer *, real *, integer *, real *);
    extern /* Subroutine */ int slahqr_(logical *, logical *, integer *,
	    integer *, integer *, real *, integer *, real *, real *, integer *
	    , integer *, real *, integer *, integer *), slacpy_(char *,
	    integer *, integer *, real *, integer *, real *, integer *), slaset_(char *, integer *, integer *, real *, real *,
	    real *, integer *), slarfx_(char *, integer *, integer *,
	    real *, real *, real *, integer *, real *);
    static real smlnum;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SHSEQR computes the eigenvalues of a real upper Hessenberg matrix H
    and, optionally, the matrices T and Z from the Schur decomposition
    H = Z T Z**T, where T is an upper quasi-triangular matrix (the Schur
    form), and Z is the orthogonal matrix of Schur vectors.

    Optionally Z may be postmultiplied into an input orthogonal matrix Q,
    so that this routine can give the Schur factorization of a matrix A
    which has been reduced to the Hessenberg form H by the orthogonal
    matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            = 'E':  compute eigenvalues only;
            = 'S':  compute eigenvalues and the Schur form T.

    COMPZ   (input) CHARACTER*1
            = 'N':  no Schur vectors are computed;
            = 'I':  Z is initialized to the unit matrix and the matrix Z
                    of Schur vectors of H is returned;
            = 'V':  Z must contain an orthogonal matrix Q on entry, and
                    the product Q*Z is returned.

    N       (input) INTEGER
            The order of the matrix H.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that H is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to SGEBAL, and then passed to SGEHRD
            when the matrix output by SGEBAL is reduced to Hessenberg
            form. Otherwise ILO and IHI should be set to 1 and N
            respectively.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    H       (input/output) REAL array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if JOB = 'S', H contains the upper quasi-triangular
            matrix T from the Schur decomposition (the Schur form);
            2-by-2 diagonal blocks (corresponding to complex conjugate
            pairs of eigenvalues) are returned in standard form, with
            H(i,i) = H(i+1,i+1) and H(i+1,i)*H(i,i+1) < 0. If JOB = 'E',
            the contents of H are unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    WR      (output) REAL array, dimension (N)
    WI      (output) REAL array, dimension (N)
            The real and imaginary parts, respectively, of the computed
            eigenvalues. If two eigenvalues are computed as a complex
            conjugate pair, they are stored in consecutive elements of
            WR and WI, say the i-th and (i+1)th, with WI(i) > 0 and
            WI(i+1) < 0. If JOB = 'S', the eigenvalues are stored in the
            same order as on the diagonal of the Schur form returned in
            H, with WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2
            diagonal block, WI(i) = sqrt(H(i+1,i)*H(i,i+1)) and
            WI(i+1) = -WI(i).

    Z       (input/output) REAL array, dimension (LDZ,N)
            If COMPZ = 'N': Z is not referenced.
            If COMPZ = 'I': on entry, Z need not be set, and on exit, Z
            contains the orthogonal matrix Z of the Schur vectors of H.
            If COMPZ = 'V': on entry Z must contain an N-by-N matrix Q,
            which is assumed to be equal to the unit matrix except for
            the submatrix Z(ILO:IHI,ILO:IHI); on exit Z contains Q*Z.
            Normally Q is the orthogonal matrix generated by SORGHR after
            the call to SGEHRD which formed the Hessenberg matrix H.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.
            LDZ >= max(1,N) if COMPZ = 'I' or 'V'; LDZ >= 1 otherwise.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,N).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, SHSEQR failed to compute all of the
                  eigenvalues in a total of 30*(IHI-ILO+1) iterations;
                  elements 1:ilo-1 and i+1:n of WR and WI contain those
                  eigenvalues which have been successfully computed.

    =====================================================================


       Decode and test the input parameters
*/

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    /* Function Body */
    wantt = lsame_(job, "S");
    initz = lsame_(compz, "I");
    wantz = initz || lsame_(compz, "V");

    *info = 0;
    work[1] = (real) max(1,*n);
    lquery = *lwork == -1;
    if (! lsame_(job, "E") && ! wantt) {
	*info = -1;
    } else if (! lsame_(compz, "N") && ! wantz) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -4;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -5;
    } else if (*ldh < max(1,*n)) {
	*info = -7;
    } else if (*ldz < 1 || wantz && *ldz < max(1,*n)) {
	*info = -11;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SHSEQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Initialize Z, if necessary */

    if (initz) {
	slaset_("Full", n, n, &c_b29, &c_b15, &z__[z_offset], ldz);
    }

/*     Store the eigenvalues isolated by SGEBAL. */

    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.f;
/* L10: */
    }
    i__1 = *n;
    for (i__ = *ihi + 1; i__ <= i__1; ++i__) {
	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.f;
/* L20: */
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }
    if (*ilo == *ihi) {
	wr[*ilo] = h__[*ilo + *ilo * h_dim1];
	wi[*ilo] = 0.f;
	return 0;
    }

/*
       Set rows and columns ILO to IHI to zero below the first
       subdiagonal.
*/

    i__1 = *ihi - 2;
    for (j = *ilo; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = j + 2; i__ <= i__2; ++i__) {
	    h__[i__ + j * h_dim1] = 0.f;
/* L30: */
	}
/* L40: */
    }
    nh = *ihi - *ilo + 1;

/*
       Determine the order of the multi-shift QR algorithm to be used.

   Writing concatenation
*/
    i__3[0] = 1, a__1[0] = job;
    i__3[1] = 1, a__1[1] = compz;
    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
    ns = ilaenv_(&c__4, "SHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
/* Writing concatenation */
    i__3[0] = 1, a__1[0] = job;
    i__3[1] = 1, a__1[1] = compz;
    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
    maxb = ilaenv_(&c__8, "SHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if (ns <= 2 || ns > nh || maxb >= nh) {

/*        Use the standard double-shift algorithm */

	slahqr_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1], &wi[
		1], ilo, ihi, &z__[z_offset], ldz, info);
	return 0;
    }
    maxb = max(3,maxb);
/* Computing MIN */
    i__1 = min(ns,maxb);
    ns = min(i__1,15);

/*
       Now 2 < NS <= MAXB < NH.

       Set machine-dependent constants for the stopping criterion.
       If norm(H) <= sqrt(OVFL), overflow should not occur.
*/

    unfl = slamch_("Safe minimum");
    ovfl = 1.f / unfl;
    slabad_(&unfl, &ovfl);
    ulp = slamch_("Precision");
    smlnum = unfl * (nh / ulp);

/*
       I1 and I2 are the indices of the first row and last column of H
       to which transformations must be applied. If eigenvalues only are
       being computed, I1 and I2 are set inside the main loop.
*/

    if (wantt) {
	i1 = 1;
	i2 = *n;
    }

/*     ITN is the total number of multiple-shift QR iterations allowed. */

    itn = nh * 30;

/*
       The main loop begins here. I is the loop index and decreases from
       IHI to ILO in steps of at most MAXB. Each iteration of the loop
       works with the active submatrix in rows and columns L to I.
       Eigenvalues I+1 to IHI have already converged. Either L = ILO or
       H(L,L-1) is negligible so that the matrix splits.
*/

    i__ = *ihi;
L50:
    l = *ilo;
    if (i__ < *ilo) {
	goto L170;
    }

/*
       Perform multiple-shift QR iterations on rows and columns ILO to I
       until a submatrix of order at most MAXB splits off at the bottom
       because a subdiagonal element has become negligible.
*/

    i__1 = itn;
    for (its = 0; its <= i__1; ++its) {

/*        Look for a single small subdiagonal element. */

	i__2 = l + 1;
	for (k = i__; k >= i__2; --k) {
	    tst1 = (r__1 = h__[k - 1 + (k - 1) * h_dim1], dabs(r__1)) + (r__2
		    = h__[k + k * h_dim1], dabs(r__2));
	    if (tst1 == 0.f) {
		i__4 = i__ - l + 1;
		tst1 = slanhs_("1", &i__4, &h__[l + l * h_dim1], ldh, &work[1]
			);
	    }
/* Computing MAX */
	    r__2 = ulp * tst1;
	    if ((r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)) <= dmax(r__2,
		    smlnum)) {
		goto L70;
	    }
/* L60: */
	}
L70:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible. */

	    h__[l + (l - 1) * h_dim1] = 0.f;
	}

/*        Exit from loop if a submatrix of order <= MAXB has split off. */

	if (l >= i__ - maxb + 1) {
	    goto L160;
	}

/*
          Now the active submatrix is in rows and columns L to I. If
          eigenvalues only are being computed, only the active submatrix
          need be transformed.
*/

	if (! wantt) {
	    i1 = l;
	    i2 = i__;
	}

	if (its == 20 || its == 30) {

/*           Exceptional shifts. */

	    i__2 = i__;
	    for (ii = i__ - ns + 1; ii <= i__2; ++ii) {
		wr[ii] = ((r__1 = h__[ii + (ii - 1) * h_dim1], dabs(r__1)) + (
			r__2 = h__[ii + ii * h_dim1], dabs(r__2))) * 1.5f;
		wi[ii] = 0.f;
/* L80: */
	    }
	} else {

/*           Use eigenvalues of trailing submatrix of order NS as shifts. */

	    slacpy_("Full", &ns, &ns, &h__[i__ - ns + 1 + (i__ - ns + 1) *
		    h_dim1], ldh, s, &c__15);
	    slahqr_(&c_false, &c_false, &ns, &c__1, &ns, s, &c__15, &wr[i__ -
		    ns + 1], &wi[i__ - ns + 1], &c__1, &ns, &z__[z_offset],
		    ldz, &ierr);
	    if (ierr > 0) {

/*
                If SLAHQR failed to compute all NS eigenvalues, use the
                unconverged diagonal elements as the remaining shifts.
*/

		i__2 = ierr;
		for (ii = 1; ii <= i__2; ++ii) {
		    wr[i__ - ns + ii] = s[ii + ii * 15 - 16];
		    wi[i__ - ns + ii] = 0.f;
/* L90: */
		}
	    }
	}

/*
          Form the first column of (G-w(1)) (G-w(2)) . . . (G-w(ns))
          where G is the Hessenberg submatrix H(L:I,L:I) and w is
          the vector of shifts (stored in WR and WI). The result is
          stored in the local array V.
*/

	v[0] = 1.f;
	i__2 = ns + 1;
	for (ii = 2; ii <= i__2; ++ii) {
	    v[ii - 1] = 0.f;
/* L100: */
	}
	nv = 1;
	i__2 = i__;
	for (j = i__ - ns + 1; j <= i__2; ++j) {
	    if (wi[j] >= 0.f) {
		if (wi[j] == 0.f) {

/*                 real shift */

		    i__4 = nv + 1;
		    scopy_(&i__4, v, &c__1, vv, &c__1);
		    i__4 = nv + 1;
		    r__1 = -wr[j];
		    sgemv_("No transpose", &i__4, &nv, &c_b15, &h__[l + l *
			    h_dim1], ldh, vv, &c__1, &r__1, v, &c__1);
		    ++nv;
		} else if (wi[j] > 0.f) {

/*                 complex conjugate pair of shifts */

		    i__4 = nv + 1;
		    scopy_(&i__4, v, &c__1, vv, &c__1);
		    i__4 = nv + 1;
		    r__1 = wr[j] * -2.f;
		    sgemv_("No transpose", &i__4, &nv, &c_b15, &h__[l + l *
			    h_dim1], ldh, v, &c__1, &r__1, vv, &c__1);
		    i__4 = nv + 1;
		    itemp = isamax_(&i__4, vv, &c__1);
/* Computing MAX */
		    r__2 = (r__1 = vv[itemp - 1], dabs(r__1));
		    temp = 1.f / dmax(r__2,smlnum);
		    i__4 = nv + 1;
		    sscal_(&i__4, &temp, vv, &c__1);
		    absw = slapy2_(&wr[j], &wi[j]);
		    temp = temp * absw * absw;
		    i__4 = nv + 2;
		    i__5 = nv + 1;
		    sgemv_("No transpose", &i__4, &i__5, &c_b15, &h__[l + l *
			    h_dim1], ldh, vv, &c__1, &temp, v, &c__1);
		    nv += 2;
		}

/*
                Scale V(1:NV) so that max(abs(V(i))) = 1. If V is zero,
                reset it to the unit vector.
*/

		itemp = isamax_(&nv, v, &c__1);
		temp = (r__1 = v[itemp - 1], dabs(r__1));
		if (temp == 0.f) {
		    v[0] = 1.f;
		    i__4 = nv;
		    for (ii = 2; ii <= i__4; ++ii) {
			v[ii - 1] = 0.f;
/* L110: */
		    }
		} else {
		    temp = dmax(temp,smlnum);
		    r__1 = 1.f / temp;
		    sscal_(&nv, &r__1, v, &c__1);
		}
	    }
/* L120: */
	}

/*        Multiple-shift QR step */

	i__2 = i__ - 1;
	for (k = l; k <= i__2; ++k) {

/*
             The first iteration of this loop determines a reflection G
             from the vector V and applies it from left and right to H,
             thus creating a nonzero bulge below the subdiagonal.

             Each subsequent iteration determines a reflection G to
             restore the Hessenberg form in the (K-1)th column, and thus
             chases the bulge one step toward the bottom of the active
             submatrix. NR is the order of G.

   Computing MIN
*/
	    i__4 = ns + 1, i__5 = i__ - k + 1;
	    nr = min(i__4,i__5);
	    if (k > l) {
		scopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    slarfg_(&nr, v, &v[1], &c__1, &tau);
	    if (k > l) {
		h__[k + (k - 1) * h_dim1] = v[0];
		i__4 = i__;
		for (ii = k + 1; ii <= i__4; ++ii) {
		    h__[ii + (k - 1) * h_dim1] = 0.f;
/* L130: */
		}
	    }
	    v[0] = 1.f;

/*
             Apply G from the left to transform the rows of the matrix in
             columns K to I2.
*/

	    i__4 = i2 - k + 1;
	    slarfx_("Left", &nr, &i__4, v, &tau, &h__[k + k * h_dim1], ldh, &
		    work[1]);

/*
             Apply G from the right to transform the columns of the
             matrix in rows I1 to min(K+NR,I).

   Computing MIN
*/
	    i__5 = k + nr;
	    i__4 = min(i__5,i__) - i1 + 1;
	    slarfx_("Right", &i__4, &nr, v, &tau, &h__[i1 + k * h_dim1], ldh,
		    &work[1]);

	    if (wantz) {

/*              Accumulate transformations in the matrix Z */

		slarfx_("Right", &nh, &nr, v, &tau, &z__[*ilo + k * z_dim1],
			ldz, &work[1]);
	    }
/* L140: */
	}

/* L150: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L160:

/*
       A submatrix of order <= MAXB in rows and columns L to I has split
       off. Use the double-shift QR algorithm to handle it.
*/

    slahqr_(&wantt, &wantz, n, &l, &i__, &h__[h_offset], ldh, &wr[1], &wi[1],
	    ilo, ihi, &z__[z_offset], ldz, info);
    if (*info > 0) {
	return 0;
    }

/*
       Decrement number of remaining iterations, and return to start of
       the main loop with a new value of I.
*/

    itn -= its;
    i__ = l - 1;
    goto L50;

L170:
    work[1] = (real) max(1,*n);
    return 0;

/*     End of SHSEQR */

} /* shseqr_ */

/* Subroutine */ int slabad_(real *small, real *large)
{
    /* Builtin functions */
    double r_lg10(real *), sqrt(doublereal);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLABAD takes as input the values computed by SLAMCH for underflow and
    overflow, and returns the square root of each of these values if the
    log of LARGE is sufficiently large.  This subroutine is intended to
    identify machines with a large exponent range, such as the Crays, and
    redefine the underflow and overflow limits to be the square roots of
    the values computed by SLAMCH.  This subroutine is needed because
    SLAMCH does not compensate for poor arithmetic in the upper half of
    the exponent range, as is found on a Cray.

    Arguments
    =========

    SMALL   (input/output) REAL
            On entry, the underflow threshold as computed by SLAMCH.
            On exit, if LOG10(LARGE) is sufficiently large, the square
            root of SMALL, otherwise unchanged.

    LARGE   (input/output) REAL
            On entry, the overflow threshold as computed by SLAMCH.
            On exit, if LOG10(LARGE) is sufficiently large, the square
            root of LARGE, otherwise unchanged.

    =====================================================================


       If it looks like we're on a Cray, take the square root of
       SMALL and LARGE to avoid overflow and underflow problems.
*/

    if (r_lg10(large) > 2e3f) {
	*small = sqrt(*small);
	*large = sqrt(*large);
    }

    return 0;

/*     End of SLABAD */

} /* slabad_ */

/* Subroutine */ int slabrd_(integer *m, integer *n, integer *nb, real *a,
	integer *lda, real *d__, real *e, real *tauq, real *taup, real *x,
	integer *ldx, real *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;

    /* Local variables */
    static integer i__;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sgemv_(char *, integer *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *), slarfg_(
	    integer *, real *, real *, integer *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLABRD reduces the first NB rows and columns of a real general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by SGEBRD

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.

    N       (input) INTEGER
            The number of columns in the matrix A.

    NB      (input) INTEGER
            The number of leading rows and columns of A to be reduced.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit, the first NB rows and columns of the matrix are
            overwritten; the rest of the array is unchanged.
            If m >= n, elements on and below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors; and
              elements above the diagonal in the first NB rows, with the
              array TAUP, represent the orthogonal matrix P as a product
              of elementary reflectors.
            If m < n, elements below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors, and
              elements on and above the diagonal in the first NB rows,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) REAL array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    E       (output) REAL array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    TAUQ    (output) REAL array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) REAL array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    X       (output) REAL array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    LDX     (input) INTEGER
            The leading dimension of the array X. LDX >= M.

    Y       (output) REAL array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    LDY     (output) INTEGER
            The leading dimension of the array Y. LDY >= N.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

       Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors.

    If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
    A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    The elements of the vectors v and u together form the m-by-nb matrix
    V and the nb-by-n matrix U' which are needed, with X and Y, to apply
    the transformation to the unreduced part of the matrix, using a block
    update of the form:  A := A - V*Y' - X*U'.

    The contents of A on exit are illustrated by the following examples
    with nb = 2:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 )
      (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 )
      (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )

    where a denotes an element of the original matrix which is unchanged,
    vi denotes an element of the vector defining H(i), and ui an element
    of the vector defining G(i).

    =====================================================================


       Quick return if possible
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:m,i) */

	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[i__ + a_dim1],
		    lda, &y[i__ + y_dim1], ldy, &c_b15, &a[i__ + i__ * a_dim1]
		    , &c__1);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &x[i__ + x_dim1],
		    ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b15, &a[i__ + i__ *
		    a_dim1], &c__1);

/*           Generate reflection Q(i) to annihilate A(i+1:m,i) */

	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ *
		    a_dim1], &c__1, &tauq[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *n) {
		a[i__ + i__ * a_dim1] = 1.f;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + (i__ + 1) *
			 a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b29,
			&y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + a_dim1],
			lda, &a[i__ + i__ * a_dim1], &c__1, &c_b29, &y[i__ *
			y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b15, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &x[i__ + x_dim1],
			ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b29, &y[i__ *
			y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b151, &a[(i__ + 1) *
			a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b15,
			 &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		sscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);

/*              Update A(i,i+1:n) */

		i__2 = *n - i__;
		sgemv_("No transpose", &i__2, &i__, &c_b151, &y[i__ + 1 +
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b15, &a[i__ +
			(i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b151, &a[(i__ + 1) *
			a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b15, &a[
			i__ + (i__ + 1) * a_dim1], lda);

/*              Generate reflection P(i) to annihilate A(i,i+2:n) */

		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, &a[i__ + (i__ + 1) * a_dim1], &a[i__ + min(
			i__3,*n) * a_dim1], lda, &taup[i__]);
		e[i__] = a[i__ + (i__ + 1) * a_dim1];
		a[i__ + (i__ + 1) * a_dim1] = 1.f;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		sgemv_("No transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + (
			i__ + 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1],
			 lda, &c_b29, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		sgemv_("Transpose", &i__2, &i__, &c_b15, &y[i__ + 1 + y_dim1],
			 ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &c_b29, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		sgemv_("No transpose", &i__2, &i__, &c_b151, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b15, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		sgemv_("No transpose", &i__2, &i__3, &c_b15, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b29, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b15, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		sscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i,i:n) */

	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &y[i__ + y_dim1],
		    ldy, &a[i__ + a_dim1], lda, &c_b15, &a[i__ + i__ * a_dim1]
		    , lda);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    sgemv_("Transpose", &i__2, &i__3, &c_b151, &a[i__ * a_dim1 + 1],
		    lda, &x[i__ + x_dim1], ldx, &c_b15, &a[i__ + i__ * a_dim1]
		    , lda);

/*           Generate reflection P(i) to annihilate A(i,i+1:n) */

	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) *
		    a_dim1], lda, &taup[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.f;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + i__
			* a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b29, &
			x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &y[i__ + y_dim1],
			ldy, &a[i__ + i__ * a_dim1], lda, &c_b29, &x[i__ *
			x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b15, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b15, &a[i__ * a_dim1
			+ 1], lda, &a[i__ + i__ * a_dim1], lda, &c_b29, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b15, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		sscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);

/*              Update A(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[i__ + 1 +
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b15, &a[i__ +
			1 + i__ * a_dim1], &c__1);
		i__2 = *m - i__;
		sgemv_("No transpose", &i__2, &i__, &c_b151, &x[i__ + 1 +
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b15, &a[
			i__ + 1 + i__ * a_dim1], &c__1);

/*              Generate reflection Q(i) to annihilate A(i+2:m,i) */

		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*m) +
			i__ * a_dim1], &c__1, &tauq[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.f;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + (i__ +
			1) * a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1,
			&c_b29, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + a_dim1]
			, lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b29, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b15, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		sgemv_("Transpose", &i__2, &i__, &c_b15, &x[i__ + 1 + x_dim1],
			 ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b29, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		sgemv_("Transpose", &i__, &i__2, &c_b151, &a[(i__ + 1) *
			a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b15,
			 &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		sscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    }
/* L20: */
	}
    }
    return 0;

/*     End of SLABRD */

} /* slabrd_ */

/* Subroutine */ int slacpy_(char *uplo, integer *m, integer *n, real *a,
	integer *lda, real *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLACPY copies all or part of a two-dimensional matrix A to another
    matrix B.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies the part of the matrix A to be copied to B.
            = 'U':      Upper triangular part
            = 'L':      Lower triangular part
            Otherwise:  All of the matrix A

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input) REAL array, dimension (LDA,N)
            The m by n matrix A.  If UPLO = 'U', only the upper triangle
            or trapezoid is accessed; if UPLO = 'L', only the lower
            triangle or trapezoid is accessed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (output) REAL array, dimension (LDB,N)
            On exit, B = A in the locations specified by UPLO.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,M).

    =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (lsame_(uplo, "U")) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = min(j,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L10: */
	    }
/* L20: */
	}
    } else if (lsame_(uplo, "L")) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L30: */
	    }
/* L40: */
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L50: */
	    }
/* L60: */
	}
    }
    return 0;

/*     End of SLACPY */

} /* slacpy_ */

/* Subroutine */ int sladiv_(real *a, real *b, real *c__, real *d__, real *p,
	real *q)
{
    static real e, f;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLADIV performs complex division in  real arithmetic

                          a + i*b
               p + i*q = ---------
                          c + i*d

    The algorithm is due to Robert L. Smith and can be found
    in D. Knuth, The art of Computer Programming, Vol.2, p.195

    Arguments
    =========

    A       (input) REAL
    B       (input) REAL
    C       (input) REAL
    D       (input) REAL
            The scalars a, b, c, and d in the above expression.

    P       (output) REAL
    Q       (output) REAL
            The scalars p and q in the above expression.

    =====================================================================
*/


    if (dabs(*d__) < dabs(*c__)) {
	e = *d__ / *c__;
	f = *c__ + *d__ * e;
	*p = (*a + *b * e) / f;
	*q = (*b - *a * e) / f;
    } else {
	e = *c__ / *d__;
	f = *d__ + *c__ * e;
	*p = (*b + *a * e) / f;
	*q = (-(*a) + *b * e) / f;
    }

    return 0;

/*     End of SLADIV */

} /* sladiv_ */

/* Subroutine */ int slae2_(real *a, real *b, real *c__, real *rt1, real *rt2)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real ab, df, tb, sm, rt, adf, acmn, acmx;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLAE2  computes the eigenvalues of a 2-by-2 symmetric matrix
       [  A   B  ]
       [  B   C  ].
    On return, RT1 is the eigenvalue of larger absolute value, and RT2
    is the eigenvalue of smaller absolute value.

    Arguments
    =========

    A       (input) REAL
            The (1,1) element of the 2-by-2 matrix.

    B       (input) REAL
            The (1,2) and (2,1) elements of the 2-by-2 matrix.

    C       (input) REAL
            The (2,2) element of the 2-by-2 matrix.

    RT1     (output) REAL
            The eigenvalue of larger absolute value.

    RT2     (output) REAL
            The eigenvalue of smaller absolute value.

    Further Details
    ===============

    RT1 is accurate to a few ulps barring over/underflow.

    RT2 may be inaccurate if there is massive cancellation in the
    determinant A*C-B*B; higher precision or correctly rounded or
    correctly truncated arithmetic would be needed to compute RT2
    accurately in all cases.

    Overflow is possible only if RT1 is within a factor of 5 of overflow.
    Underflow is harmless if the input data is 0 or exceeds
       underflow_threshold / macheps.

   =====================================================================


       Compute the eigenvalues
*/

    sm = *a + *c__;
    df = *a - *c__;
    adf = dabs(df);
    tb = *b + *b;
    ab = dabs(tb);
    if (dabs(*a) > dabs(*c__)) {
	acmx = *a;
	acmn = *c__;
    } else {
	acmx = *c__;
	acmn = *a;
    }
    if (adf > ab) {
/* Computing 2nd power */
	r__1 = ab / adf;
	rt = adf * sqrt(r__1 * r__1 + 1.f);
    } else if (adf < ab) {
/* Computing 2nd power */
	r__1 = adf / ab;
	rt = ab * sqrt(r__1 * r__1 + 1.f);
    } else {

/*        Includes case AB=ADF=0 */

	rt = ab * sqrt(2.f);
    }
    if (sm < 0.f) {
	*rt1 = (sm - rt) * .5f;

/*
          Order of execution important.
          To get fully accurate smaller eigenvalue,
          next line needs to be executed in higher precision.
*/

	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else if (sm > 0.f) {
	*rt1 = (sm + rt) * .5f;

/*
          Order of execution important.
          To get fully accurate smaller eigenvalue,
          next line needs to be executed in higher precision.
*/

	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else {

/*        Includes case RT1 = RT2 = 0 */

	*rt1 = rt * .5f;
	*rt2 = rt * -.5f;
    }
    return 0;

/*     End of SLAE2 */

} /* slae2_ */

/* Subroutine */ int slaed0_(integer *icompq, integer *qsiz, integer *n, real
	*d__, real *e, real *q, integer *ldq, real *qstore, integer *ldqs,
	real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, qstore_dim1, qstore_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, k, iq, lgn, msd2, smm1, spm1, spm2;
    static real temp;
    static integer curr;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer iperm, indxq, iwrem;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    static integer iqptr, tlvls;
    extern /* Subroutine */ int slaed1_(integer *, real *, real *, integer *,
	    integer *, real *, integer *, real *, integer *, integer *),
	    slaed7_(integer *, integer *, integer *, integer *, integer *,
	    integer *, real *, real *, integer *, integer *, real *, integer *
	    , real *, integer *, integer *, integer *, integer *, integer *,
	    real *, real *, integer *, integer *);
    static integer igivcl;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static integer igivnm, submat;
    extern /* Subroutine */ int slacpy_(char *, integer *, integer *, real *,
	    integer *, real *, integer *);
    static integer curprb, subpbs, igivpt, curlvl, matsiz, iprmpt, smlsiz;
    extern /* Subroutine */ int ssteqr_(char *, integer *, real *, real *,
	    real *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLAED0 computes all eigenvalues and corresponding eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    Arguments
    =========

    ICOMPQ  (input) INTEGER
            = 0:  Compute eigenvalues only.
            = 1:  Compute eigenvectors of original dense symmetric matrix
                  also.  On entry, Q contains the orthogonal matrix used
                  to reduce the original matrix to tridiagonal form.
            = 2:  Compute eigenvalues and eigenvectors of tridiagonal
                  matrix.

    QSIZ   (input) INTEGER
           The dimension of the orthogonal matrix used to reduce
           the full matrix to tridiagonal form.  QSIZ >= N if ICOMPQ = 1.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D      (input/output) REAL array, dimension (N)
           On entry, the main diagonal of the tridiagonal matrix.
           On exit, its eigenvalues.

    E      (input) REAL array, dimension (N-1)
           The off-diagonal elements of the tridiagonal matrix.
           On exit, E has been destroyed.

    Q      (input/output) REAL array, dimension (LDQ, N)
           On entry, Q must contain an N-by-N orthogonal matrix.
           If ICOMPQ = 0    Q is not referenced.
           If ICOMPQ = 1    On entry, Q is a subset of the columns of the
                            orthogonal matrix used to reduce the full
                            matrix to tridiagonal form corresponding to
                            the subset of the full matrix which is being
                            decomposed at this time.
           If ICOMPQ = 2    On entry, Q will be the identity matrix.
                            On exit, Q contains the eigenvectors of the
                            tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  If eigenvectors are
           desired, then  LDQ >= max(1,N).  In any case,  LDQ >= 1.

    QSTORE (workspace) REAL array, dimension (LDQS, N)
           Referenced only when ICOMPQ = 1.  Used to store parts of
           the eigenvector matrix when the updating matrix multiplies
           take place.

    LDQS   (input) INTEGER
           The leading dimension of the array QSTORE.  If ICOMPQ = 1,
           then  LDQS >= max(1,N).  In any case,  LDQS >= 1.

    WORK   (workspace) REAL array,
           If ICOMPQ = 0 or 1, the dimension of WORK must be at least
                       1 + 3*N + 2*N*lg N + 2*N**2
                       ( lg( N ) = smallest integer k
                                   such that 2^k >= N )
           If ICOMPQ = 2, the dimension of WORK must be at least
                       4*N + N**2.

    IWORK  (workspace) INTEGER array,
           If ICOMPQ = 0 or 1, the dimension of IWORK must be at least
                          6 + 6*N + 5*N*lg N.
                          ( lg( N ) = smallest integer k
                                      such that 2^k >= N )
           If ICOMPQ = 2, the dimension of IWORK must be at least
                          3 + 5*N.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    qstore_dim1 = *ldqs;
    qstore_offset = 1 + qstore_dim1;
    qstore -= qstore_offset;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 2) {
	*info = -1;
    } else if (*icompq == 1 && *qsiz < max(0,*n)) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -7;
    } else if (*ldqs < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED0", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "SLAED0", " ", &c__0, &c__0, &c__0, &c__0, (
	    ftnlen)6, (ftnlen)1);

/*
       Determine the size and placement of the submatrices, and save in
       the leading elements of IWORK.
*/

    iwork[1] = *n;
    subpbs = 1;
    tlvls = 0;
L10:
    if (iwork[subpbs] > smlsiz) {
	for (j = subpbs; j >= 1; --j) {
	    iwork[j * 2] = (iwork[j] + 1) / 2;
	    iwork[(j << 1) - 1] = iwork[j] / 2;
/* L20: */
	}
	++tlvls;
	subpbs <<= 1;
	goto L10;
    }
    i__1 = subpbs;
    for (j = 2; j <= i__1; ++j) {
	iwork[j] += iwork[j - 1];
/* L30: */
    }

/*
       Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
       using rank-1 modifications (cuts).
*/

    spm1 = subpbs - 1;
    i__1 = spm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	submat = iwork[i__] + 1;
	smm1 = submat - 1;
	d__[smm1] -= (r__1 = e[smm1], dabs(r__1));
	d__[submat] -= (r__1 = e[smm1], dabs(r__1));
/* L40: */
    }

    indxq = (*n << 2) + 3;
    if (*icompq != 2) {

/*
          Set up workspaces for eigenvalues only/accumulate new vectors
          routine
*/

	temp = log((real) (*n)) / log(2.f);
	lgn = (integer) temp;
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	iprmpt = indxq + *n + 1;
	iperm = iprmpt + *n * lgn;
	iqptr = iperm + *n * lgn;
	igivpt = iqptr + *n + 2;
	igivcl = igivpt + *n * lgn;

	igivnm = 1;
	iq = igivnm + (*n << 1) * lgn;
/* Computing 2nd power */
	i__1 = *n;
	iwrem = iq + i__1 * i__1 + 1;

/*        Initialize pointers */

	i__1 = subpbs;
	for (i__ = 0; i__ <= i__1; ++i__) {
	    iwork[iprmpt + i__] = 1;
	    iwork[igivpt + i__] = 1;
/* L50: */
	}
	iwork[iqptr] = 1;
    }

/*
       Solve each submatrix eigenproblem at the bottom of the divide and
       conquer tree.
*/

    curr = 0;
    i__1 = spm1;
    for (i__ = 0; i__ <= i__1; ++i__) {
	if (i__ == 0) {
	    submat = 1;
	    matsiz = iwork[1];
	} else {
	    submat = iwork[i__] + 1;
	    matsiz = iwork[i__ + 1] - iwork[i__];
	}
	if (*icompq == 2) {
	    ssteqr_("I", &matsiz, &d__[submat], &e[submat], &q[submat +
		    submat * q_dim1], ldq, &work[1], info);
	    if (*info != 0) {
		goto L130;
	    }
	} else {
	    ssteqr_("I", &matsiz, &d__[submat], &e[submat], &work[iq - 1 +
		    iwork[iqptr + curr]], &matsiz, &work[1], info);
	    if (*info != 0) {
		goto L130;
	    }
	    if (*icompq == 1) {
		sgemm_("N", "N", qsiz, &matsiz, &matsiz, &c_b15, &q[submat *
			q_dim1 + 1], ldq, &work[iq - 1 + iwork[iqptr + curr]],
			 &matsiz, &c_b29, &qstore[submat * qstore_dim1 + 1],
			ldqs);
	    }
/* Computing 2nd power */
	    i__2 = matsiz;
	    iwork[iqptr + curr + 1] = iwork[iqptr + curr] + i__2 * i__2;
	    ++curr;
	}
	k = 1;
	i__2 = iwork[i__ + 1];
	for (j = submat; j <= i__2; ++j) {
	    iwork[indxq + j] = k;
	    ++k;
/* L60: */
	}
/* L70: */
    }

/*
       Successively merge eigensystems of adjacent submatrices
       into eigensystem for the corresponding larger matrix.

       while ( SUBPBS > 1 )
*/

    curlvl = 1;
L80:
    if (subpbs > 1) {
	spm2 = subpbs - 2;
	i__1 = spm2;
	for (i__ = 0; i__ <= i__1; i__ += 2) {
	    if (i__ == 0) {
		submat = 1;
		matsiz = iwork[2];
		msd2 = iwork[1];
		curprb = 0;
	    } else {
		submat = iwork[i__] + 1;
		matsiz = iwork[i__ + 2] - iwork[i__];
		msd2 = matsiz / 2;
		++curprb;
	    }

/*
       Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
       into an eigensystem of size MATSIZ.
       SLAED1 is used only for the full eigensystem of a tridiagonal
       matrix.
       SLAED7 handles the cases in which eigenvalues only or eigenvalues
       and eigenvectors of a full symmetric matrix (which was reduced to
       tridiagonal form) are desired.
*/

	    if (*icompq == 2) {
		slaed1_(&matsiz, &d__[submat], &q[submat + submat * q_dim1],
			ldq, &iwork[indxq + submat], &e[submat + msd2 - 1], &
			msd2, &work[1], &iwork[subpbs + 1], info);
	    } else {
		slaed7_(icompq, &matsiz, qsiz, &tlvls, &curlvl, &curprb, &d__[
			submat], &qstore[submat * qstore_dim1 + 1], ldqs, &
			iwork[indxq + submat], &e[submat + msd2 - 1], &msd2, &
			work[iq], &iwork[iqptr], &iwork[iprmpt], &iwork[iperm]
			, &iwork[igivpt], &iwork[igivcl], &work[igivnm], &
			work[iwrem], &iwork[subpbs + 1], info);
	    }
	    if (*info != 0) {
		goto L130;
	    }
	    iwork[i__ / 2 + 1] = iwork[i__ + 2];
/* L90: */
	}
	subpbs /= 2;
	++curlvl;
	goto L80;
    }

/*
       end while

       Re-merge the eigenvalues/vectors which were deflated at the final
       merge step.
*/

    if (*icompq == 1) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    j = iwork[indxq + i__];
	    work[i__] = d__[j];
	    scopy_(qsiz, &qstore[j * qstore_dim1 + 1], &c__1, &q[i__ * q_dim1
		    + 1], &c__1);
/* L100: */
	}
	scopy_(n, &work[1], &c__1, &d__[1], &c__1);
    } else if (*icompq == 2) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    j = iwork[indxq + i__];
	    work[i__] = d__[j];
	    scopy_(n, &q[j * q_dim1 + 1], &c__1, &work[*n * i__ + 1], &c__1);
/* L110: */
	}
	scopy_(n, &work[1], &c__1, &d__[1], &c__1);
	slacpy_("A", n, n, &work[*n + 1], n, &q[q_offset], ldq);
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    j = iwork[indxq + i__];
	    work[i__] = d__[j];
/* L120: */
	}
	scopy_(n, &work[1], &c__1, &d__[1], &c__1);
    }
    goto L140;

L130:
    *info = submat * (*n + 1) + submat + matsiz - 1;

L140:
    return 0;

/*     End of SLAED0 */

} /* slaed0_ */

/* Subroutine */ int slaed1_(integer *n, real *d__, real *q, integer *ldq,
	integer *indxq, real *rho, integer *cutpnt, real *work, integer *
	iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Local variables */
    static integer i__, k, n1, n2, is, iw, iz, iq2, cpp1, indx, indxc, indxp;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), slaed2_(integer *, integer *, integer *, real *, real
	    *, integer *, integer *, real *, real *, real *, real *, real *,
	    integer *, integer *, integer *, integer *, integer *), slaed3_(
	    integer *, integer *, integer *, real *, real *, integer *, real *
	    , real *, real *, integer *, integer *, real *, real *, integer *)
	    ;
    static integer idlmda;
    extern /* Subroutine */ int xerbla_(char *, integer *), slamrg_(
	    integer *, integer *, real *, integer *, integer *, integer *);
    static integer coltyp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLAED1 computes the updated eigensystem of a diagonal
    matrix after modification by a rank-one symmetric matrix.  This
    routine is used only for the eigenproblem which requires all
    eigenvalues and eigenvectors of a tridiagonal matrix.  SLAED7 handles
    the case in which eigenvalues only or eigenvalues and eigenvectors
    of a full symmetric matrix (which was reduced to tridiagonal form)
    are desired.

      T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out)

       where Z = Q'u, u is a vector of length N with ones in the
       CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.

       The eigenvectors of the original matrix are stored in Q, and the
       eigenvalues are in D.  The algorithm consists of three stages:

          The first stage consists of deflating the size of the problem
          when there are multiple eigenvalues or if there is a zero in
          the Z vector.  For each such occurence the dimension of the
          secular equation problem is reduced by one.  This stage is
          performed by the routine SLAED2.

          The second stage consists of calculating the updated
          eigenvalues. This is done by finding the roots of the secular
          equation via the routine SLAED4 (as called by SLAED3).
          This routine also calculates the eigenvectors of the current
          problem.

          The final stage consists of computing the updated eigenvectors
          directly using the updated eigenvalues.  The eigenvectors for
          the current problem are multiplied with the eigenvectors from
          the overall problem.

    Arguments
    =========

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D      (input/output) REAL array, dimension (N)
           On entry, the eigenvalues of the rank-1-perturbed matrix.
           On exit, the eigenvalues of the repaired matrix.

    Q      (input/output) REAL array, dimension (LDQ,N)
           On entry, the eigenvectors of the rank-1-perturbed matrix.
           On exit, the eigenvectors of the repaired tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    INDXQ  (input/output) INTEGER array, dimension (N)
           On entry, the permutation which separately sorts the two
           subproblems in D into ascending order.
           On exit, the permutation which will reintegrate the
           subproblems back into sorted order,
           i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.

    RHO    (input) REAL
           The subdiagonal entry used to create the rank-1 modification.

    CUTPNT (input) INTEGER
           The location of the last eigenvalue in the leading sub-matrix.
           min(1,N) <= CUTPNT <= N/2.

    WORK   (workspace) REAL array, dimension (4*N + N**2)

    IWORK  (workspace) INTEGER array, dimension (4*N)

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -1;
    } else if (*ldq < max(1,*n)) {
	*info = -4;
    } else /* if(complicated condition) */ {
/* Computing MIN */
	i__1 = 1, i__2 = *n / 2;
	if (min(i__1,i__2) > *cutpnt || *n / 2 < *cutpnt) {
	    *info = -7;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED1", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*
       The following values are integer pointers which indicate
       the portion of the workspace
       used by a particular array in SLAED2 and SLAED3.
*/

    iz = 1;
    idlmda = iz + *n;
    iw = idlmda + *n;
    iq2 = iw + *n;

    indx = 1;
    indxc = indx + *n;
    coltyp = indxc + *n;
    indxp = coltyp + *n;


/*
       Form the z-vector which consists of the last row of Q_1 and the
       first row of Q_2.
*/

    scopy_(cutpnt, &q[*cutpnt + q_dim1], ldq, &work[iz], &c__1);
    cpp1 = *cutpnt + 1;
    i__1 = *n - *cutpnt;
    scopy_(&i__1, &q[cpp1 + cpp1 * q_dim1], ldq, &work[iz + *cutpnt], &c__1);

/*     Deflate eigenvalues. */

    slaed2_(&k, n, cutpnt, &d__[1], &q[q_offset], ldq, &indxq[1], rho, &work[
	    iz], &work[idlmda], &work[iw], &work[iq2], &iwork[indx], &iwork[
	    indxc], &iwork[indxp], &iwork[coltyp], info);

    if (*info != 0) {
	goto L20;
    }

/*     Solve Secular Equation. */

    if (k != 0) {
	is = (iwork[coltyp] + iwork[coltyp + 1]) * *cutpnt + (iwork[coltyp +
		1] + iwork[coltyp + 2]) * (*n - *cutpnt) + iq2;
	slaed3_(&k, n, cutpnt, &d__[1], &q[q_offset], ldq, rho, &work[idlmda],
		 &work[iq2], &iwork[indxc], &iwork[coltyp], &work[iw], &work[
		is], info);
	if (*info != 0) {
	    goto L20;
	}

/*     Prepare the INDXQ sorting permutation. */

	n1 = k;
	n2 = *n - k;
	slamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &indxq[1]);
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    indxq[i__] = i__;
/* L10: */
	}
    }

L20:
    return 0;

/*     End of SLAED1 */

} /* slaed1_ */

/* Subroutine */ int slaed2_(integer *k, integer *n, integer *n1, real *d__,
	real *q, integer *ldq, integer *indxq, real *rho, real *z__, real *
	dlamda, real *w, real *q2, integer *indx, integer *indxc, integer *
	indxp, integer *coltyp, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real c__;
    static integer i__, j;
    static real s, t;
    static integer k2, n2, ct, nj, pj, js, iq1, iq2, n1p1;
    static real eps, tau, tol;
    static integer psm[4], imax, jmax, ctot[4];
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *), sscal_(integer *, real *, real *,
	    integer *), scopy_(integer *, real *, integer *, real *, integer *
	    );
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slamrg_(integer *, integer *, real *, integer
	    *, integer *, integer *), slacpy_(char *, integer *, integer *,
	    real *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLAED2 merges the two sets of eigenvalues together into a single
    sorted set.  Then it tries to deflate the size of the problem.
    There are two ways in which deflation can occur:  when two or more
    eigenvalues are close together or if there is a tiny entry in the
    Z vector.  For each such occurrence the order of the related secular
    equation problem is reduced by one.

    Arguments
    =========

    K      (output) INTEGER
           The number of non-deflated eigenvalues, and the order of the
           related secular equation. 0 <= K <=N.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    N1     (input) INTEGER
           The location of the last eigenvalue in the leading sub-matrix.
           min(1,N) <= N1 <= N/2.

    D      (input/output) REAL array, dimension (N)
           On entry, D contains the eigenvalues of the two submatrices to
           be combined.
           On exit, D contains the trailing (N-K) updated eigenvalues
           (those which were deflated) sorted into increasing order.

    Q      (input/output) REAL array, dimension (LDQ, N)
           On entry, Q contains the eigenvectors of two submatrices in
           the two square blocks with corners at (1,1), (N1,N1)
           and (N1+1, N1+1), (N,N).
           On exit, Q contains the trailing (N-K) updated eigenvectors
           (those which were deflated) in its last N-K columns.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    INDXQ  (input/output) INTEGER array, dimension (N)
           The permutation which separately sorts the two sub-problems
           in D into ascending order.  Note that elements in the second
           half of this permutation must first have N1 added to their
           values. Destroyed on exit.

    RHO    (input/output) REAL
           On entry, the off-diagonal element associated with the rank-1
           cut which originally split the two submatrices which are now
           being recombined.
           On exit, RHO has been modified to the value required by
           SLAED3.

    Z      (input) REAL array, dimension (N)
           On entry, Z contains the updating vector (the last
           row of the first sub-eigenvector matrix and the first row of
           the second sub-eigenvector matrix).
           On exit, the contents of Z have been destroyed by the updating
           process.

    DLAMDA (output) REAL array, dimension (N)
           A copy of the first K eigenvalues which will be used by
           SLAED3 to form the secular equation.

    W      (output) REAL array, dimension (N)
           The first k values of the final deflation-altered z-vector
           which will be passed to SLAED3.

    Q2     (output) REAL array, dimension (N1**2+(N-N1)**2)
           A copy of the first K eigenvectors which will be used by
           SLAED3 in a matrix multiply (SGEMM) to solve for the new
           eigenvectors.

    INDX   (workspace) INTEGER array, dimension (N)
           The permutation used to sort the contents of DLAMDA into
           ascending order.

    INDXC  (output) INTEGER array, dimension (N)
           The permutation used to arrange the columns of the deflated
           Q matrix into three groups:  the first group contains non-zero
           elements only at and above N1, the second contains
           non-zero elements only below N1, and the third is dense.

    INDXP  (workspace) INTEGER array, dimension (N)
           The permutation used to place deflated values of D at the end
           of the array.  INDXP(1:K) points to the nondeflated D-values
           and INDXP(K+1:N) points to the deflated eigenvalues.

    COLTYP (workspace/output) INTEGER array, dimension (N)
           During execution, a label which will indicate which of the
           following types a column in the Q2 matrix is:
           1 : non-zero in the upper half only;
           2 : dense;
           3 : non-zero in the lower half only;
           4 : deflated.
           On exit, COLTYP(i) is the number of columns of type i,
           for i=1 to 4 only.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --z__;
    --dlamda;
    --w;
    --q2;
    --indx;
    --indxc;
    --indxp;
    --coltyp;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -2;
    } else if (*ldq < max(1,*n)) {
	*info = -6;
    } else /* if(complicated condition) */ {
/* Computing MIN */
	i__1 = 1, i__2 = *n / 2;
	if (min(i__1,i__2) > *n1 || *n / 2 < *n1) {
	    *info = -3;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    n2 = *n - *n1;
    n1p1 = *n1 + 1;

    if (*rho < 0.f) {
	sscal_(&n2, &c_b151, &z__[n1p1], &c__1);
    }

/*
       Normalize z so that norm(z) = 1.  Since z is the concatenation of
       two normalized vectors, norm2(z) = sqrt(2).
*/

    t = 1.f / sqrt(2.f);
    sscal_(n, &t, &z__[1], &c__1);

/*     RHO = ABS( norm(z)**2 * RHO ) */

    *rho = (r__1 = *rho * 2.f, dabs(r__1));

/*     Sort the eigenvalues into increasing order */

    i__1 = *n;
    for (i__ = n1p1; i__ <= i__1; ++i__) {
	indxq[i__] += *n1;
/* L10: */
    }

/*     re-integrate the deflated parts from the last pass */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = d__[indxq[i__]];
/* L20: */
    }
    slamrg_(n1, &n2, &dlamda[1], &c__1, &c__1, &indxc[1]);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	indx[i__] = indxq[indxc[i__]];
/* L30: */
    }

/*     Calculate the allowable deflation tolerance */

    imax = isamax_(n, &z__[1], &c__1);
    jmax = isamax_(n, &d__[1], &c__1);
    eps = slamch_("Epsilon");
/* Computing MAX */
    r__3 = (r__1 = d__[jmax], dabs(r__1)), r__4 = (r__2 = z__[imax], dabs(
	    r__2));
    tol = eps * 8.f * dmax(r__3,r__4);

/*
       If the rank-1 modifier is small enough, no more needs to be done
       except to reorganize Q so that its columns correspond with the
       elements in D.
*/

    if (*rho * (r__1 = z__[imax], dabs(r__1)) <= tol) {
	*k = 0;
	iq2 = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__ = indx[j];
	    scopy_(n, &q[i__ * q_dim1 + 1], &c__1, &q2[iq2], &c__1);
	    dlamda[j] = d__[i__];
	    iq2 += *n;
/* L40: */
	}
	slacpy_("A", n, n, &q2[1], n, &q[q_offset], ldq);
	scopy_(n, &dlamda[1], &c__1, &d__[1], &c__1);
	goto L190;
    }

/*
       If there are multiple eigenvalues then the problem deflates.  Here
       the number of equal eigenvalues are found.  As each equal
       eigenvalue is found, an elementary reflector is computed to rotate
       the corresponding eigensubspace so that the corresponding
       components of Z are zero in this new basis.
*/

    i__1 = *n1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	coltyp[i__] = 1;
/* L50: */
    }
    i__1 = *n;
    for (i__ = n1p1; i__ <= i__1; ++i__) {
	coltyp[i__] = 3;
/* L60: */
    }


    *k = 0;
    k2 = *n + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	nj = indx[j];
	if (*rho * (r__1 = z__[nj], dabs(r__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    coltyp[nj] = 4;
	    indxp[k2] = nj;
	    if (j == *n) {
		goto L100;
	    }
	} else {
	    pj = nj;
	    goto L80;
	}
/* L70: */
    }
L80:
    ++j;
    nj = indx[j];
    if (j > *n) {
	goto L100;
    }
    if (*rho * (r__1 = z__[nj], dabs(r__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	coltyp[nj] = 4;
	indxp[k2] = nj;
    } else {

/*        Check if eigenvalues are close enough to allow deflation. */

	s = z__[pj];
	c__ = z__[nj];

/*
          Find sqrt(a**2+b**2) without overflow or
          destructive underflow.
*/

	tau = slapy2_(&c__, &s);
	t = d__[nj] - d__[pj];
	c__ /= tau;
	s = -s / tau;
	if ((r__1 = t * c__ * s, dabs(r__1)) <= tol) {

/*           Deflation is possible. */

	    z__[nj] = tau;
	    z__[pj] = 0.f;
	    if (coltyp[nj] != coltyp[pj]) {
		coltyp[nj] = 2;
	    }
	    coltyp[pj] = 4;
	    srot_(n, &q[pj * q_dim1 + 1], &c__1, &q[nj * q_dim1 + 1], &c__1, &
		    c__, &s);
/* Computing 2nd power */
	    r__1 = c__;
/* Computing 2nd power */
	    r__2 = s;
	    t = d__[pj] * (r__1 * r__1) + d__[nj] * (r__2 * r__2);
/* Computing 2nd power */
	    r__1 = s;
/* Computing 2nd power */
	    r__2 = c__;
	    d__[nj] = d__[pj] * (r__1 * r__1) + d__[nj] * (r__2 * r__2);
	    d__[pj] = t;
	    --k2;
	    i__ = 1;
L90:
	    if (k2 + i__ <= *n) {
		if (d__[pj] < d__[indxp[k2 + i__]]) {
		    indxp[k2 + i__ - 1] = indxp[k2 + i__];
		    indxp[k2 + i__] = pj;
		    ++i__;
		    goto L90;
		} else {
		    indxp[k2 + i__ - 1] = pj;
		}
	    } else {
		indxp[k2 + i__ - 1] = pj;
	    }
	    pj = nj;
	} else {
	    ++(*k);
	    dlamda[*k] = d__[pj];
	    w[*k] = z__[pj];
	    indxp[*k] = pj;
	    pj = nj;
	}
    }
    goto L80;
L100:

/*     Record the last eigenvalue. */

    ++(*k);
    dlamda[*k] = d__[pj];
    w[*k] = z__[pj];
    indxp[*k] = pj;

/*
       Count up the total number of the various types of columns, then
       form a permutation which positions the four column types into
       four uniform groups (although one or more of these groups may be
       empty).
*/

    for (j = 1; j <= 4; ++j) {
	ctot[j - 1] = 0;
/* L110: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	ct = coltyp[j];
	++ctot[ct - 1];
/* L120: */
    }

/*     PSM(*) = Position in SubMatrix (of types 1 through 4) */

    psm[0] = 1;
    psm[1] = ctot[0] + 1;
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    *k = *n - ctot[3];

/*
       Fill out the INDXC array so that the permutation which it induces
       will place all type-1 columns first, all type-2 columns next,
       then all type-3's, and finally all type-4's.
*/

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	js = indxp[j];
	ct = coltyp[js];
	indx[psm[ct - 1]] = js;
	indxc[psm[ct - 1]] = j;
	++psm[ct - 1];
/* L130: */
    }

/*
       Sort the eigenvalues and corresponding eigenvectors into DLAMDA
       and Q2 respectively.  The eigenvalues/vectors which were not
       deflated go into the first K slots of DLAMDA and Q2 respectively,
       while those which were deflated go into the last N - K slots.
*/

    i__ = 1;
    iq1 = 1;
    iq2 = (ctot[0] + ctot[1]) * *n1 + 1;
    i__1 = ctot[0];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	scopy_(n1, &q[js * q_dim1 + 1], &c__1, &q2[iq1], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq1 += *n1;
/* L140: */
    }

    i__1 = ctot[1];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	scopy_(n1, &q[js * q_dim1 + 1], &c__1, &q2[iq1], &c__1);
	scopy_(&n2, &q[*n1 + 1 + js * q_dim1], &c__1, &q2[iq2], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq1 += *n1;
	iq2 += n2;
/* L150: */
    }

    i__1 = ctot[2];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	scopy_(&n2, &q[*n1 + 1 + js * q_dim1], &c__1, &q2[iq2], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq2 += n2;
/* L160: */
    }

    iq1 = iq2;
    i__1 = ctot[3];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	scopy_(n, &q[js * q_dim1 + 1], &c__1, &q2[iq2], &c__1);
	iq2 += *n;
	z__[i__] = d__[js];
	++i__;
/* L170: */
    }

/*
       The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively.
*/

    slacpy_("A", n, &ctot[3], &q2[iq1], n, &q[(*k + 1) * q_dim1 + 1], ldq);
    i__1 = *n - *k;
    scopy_(&i__1, &z__[*k + 1], &c__1, &d__[*k + 1], &c__1);

/*     Copy CTOT into COLTYP for referencing in SLAED3. */

    for (j = 1; j <= 4; ++j) {
	coltyp[j] = ctot[j - 1];
/* L180: */
    }

L190:
    return 0;

/*     End of SLAED2 */

} /* slaed2_ */

/* Subroutine */ int slaed3_(integer *k, integer *n, integer *n1, real *d__,
	real *q, integer *ldq, real *rho, real *dlamda, real *q2, integer *
	indx, integer *ctot, real *w, real *s, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static integer i__, j, n2, n12, ii, n23, iq2;
    static real temp;
    extern doublereal snrm2_(integer *, real *, integer *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *), scopy_(integer *, real *,
	    integer *, real *, integer *), slaed4_(integer *, integer *, real
	    *, real *, real *, real *, real *, integer *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int xerbla_(char *, integer *), slacpy_(
	    char *, integer *, integer *, real *, integer *, real *, integer *
	    ), slaset_(char *, integer *, integer *, real *, real *,
	    real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       June 30, 1999


    Purpose
    =======

    SLAED3 finds the roots of the secular equation, as defined by the
    values in D, W, and RHO, between 1 and K.  It makes the
    appropriate calls to SLAED4 and then updates the eigenvectors by
    multiplying the matrix of eigenvectors of the pair of eigensystems
    being combined by the matrix of eigenvectors of the K-by-K system
    which is solved here.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========

    K       (input) INTEGER
            The number of terms in the rational function to be solved by
            SLAED4.  K >= 0.

    N       (input) INTEGER
            The number of rows and columns in the Q matrix.
            N >= K (deflation may result in N>K).

    N1      (input) INTEGER
            The location of the last eigenvalue in the leading submatrix.
            min(1,N) <= N1 <= N/2.

    D       (output) REAL array, dimension (N)
            D(I) contains the updated eigenvalues for
            1 <= I <= K.

    Q       (output) REAL array, dimension (LDQ,N)
            Initially the first K columns are used as workspace.
            On output the columns 1 to K contain
            the updated eigenvectors.

    LDQ     (input) INTEGER
            The leading dimension of the array Q.  LDQ >= max(1,N).

    RHO     (input) REAL
            The value of the parameter in the rank one update equation.
            RHO >= 0 required.

    DLAMDA  (input/output) REAL array, dimension (K)
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation. May be changed on output by
            having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
            Cray-2, or Cray C-90, as described above.

    Q2      (input) REAL array, dimension (LDQ2, N)
            The first K columns of this matrix contain the non-deflated
            eigenvectors for the split problem.

    INDX    (input) INTEGER array, dimension (N)
            The permutation used to arrange the columns of the deflated
            Q matrix into three groups (see SLAED2).
            The rows of the eigenvectors found by SLAED4 must be likewise
            permuted before the matrix multiply can take place.

    CTOT    (input) INTEGER array, dimension (4)
            A count of the total number of the various types of columns
            in Q, as described in INDX.  The fourth column type is any
            column which has been deflated.

    W       (input/output) REAL array, dimension (K)
            The first K elements of this array contain the components
            of the deflation-adjusted updating vector. Destroyed on
            output.

    S       (workspace) REAL array, dimension (N1 + 1)*K
            Will contain the eigenvectors of the repaired matrix which
            will be multiplied by the previously accumulated eigenvectors
            to update the system.

    LDS     (input) INTEGER
            The leading dimension of S.  LDS >= max(1,K).

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dlamda;
    --q2;
    --indx;
    --ctot;
    --w;
    --s;

    /* Function Body */
    *info = 0;

    if (*k < 0) {
	*info = -1;
    } else if (*n < *k) {
	*info = -2;
    } else if (*ldq < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED3", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 0) {
	return 0;
    }

/*
       Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
       be computed with high relative accuracy (barring over/underflow).
       This is a problem on machines without a guard digit in
       add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
       The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
       which on any of these machines zeros out the bottommost
       bit of DLAMDA(I) if it is 1; this makes the subsequent
       subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
       occurs. On binary machines with a guard digit (almost all
       machines) it does not change DLAMDA(I) at all. On hexadecimal
       and decimal machines with a guard digit, it slightly
       changes the bottommost bits of DLAMDA(I). It does not account
       for hexadecimal or decimal machines without guard digits
       (we know of none). We use a subroutine call to compute
       2*DLAMBDA(I) to prevent optimizing compilers from eliminating
       this code.
*/

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = slamc3_(&dlamda[i__], &dlamda[i__]) - dlamda[i__];
/* L10: */
    }

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	slaed4_(k, &j, &dlamda[1], &w[1], &q[j * q_dim1 + 1], rho, &d__[j],
		info);

/*        If the zero finder fails, the computation is terminated. */

	if (*info != 0) {
	    goto L120;
	}
/* L20: */
    }

    if (*k == 1) {
	goto L110;
    }
    if (*k == 2) {
	i__1 = *k;
	for (j = 1; j <= i__1; ++j) {
	    w[1] = q[j * q_dim1 + 1];
	    w[2] = q[j * q_dim1 + 2];
	    ii = indx[1];
	    q[j * q_dim1 + 1] = w[ii];
	    ii = indx[2];
	    q[j * q_dim1 + 2] = w[ii];
/* L30: */
	}
	goto L110;
    }

/*     Compute updated W. */

    scopy_(k, &w[1], &c__1, &s[1], &c__1);

/*     Initialize W(I) = Q(I,I) */

    i__1 = *ldq + 1;
    scopy_(k, &q[q_offset], &i__1, &w[1], &c__1);
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L40: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L50: */
	}
/* L60: */
    }
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__1 = sqrt(-w[i__]);
	w[i__] = r_sign(&r__1, &s[i__]);
/* L70: */
    }

/*     Compute eigenvectors of the modified rank-1 modification. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s[i__] = w[i__] / q[i__ + j * q_dim1];
/* L80: */
	}
	temp = snrm2_(k, &s[1], &c__1);
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ii = indx[i__];
	    q[i__ + j * q_dim1] = s[ii] / temp;
/* L90: */
	}
/* L100: */
    }

/*     Compute the updated eigenvectors. */

L110:

    n2 = *n - *n1;
    n12 = ctot[1] + ctot[2];
    n23 = ctot[2] + ctot[3];

    slacpy_("A", &n23, k, &q[ctot[1] + 1 + q_dim1], ldq, &s[1], &n23);
    iq2 = *n1 * n12 + 1;
    if (n23 != 0) {
	sgemm_("N", "N", &n2, k, &n23, &c_b15, &q2[iq2], &n2, &s[1], &n23, &
		c_b29, &q[*n1 + 1 + q_dim1], ldq);
    } else {
	slaset_("A", &n2, k, &c_b29, &c_b29, &q[*n1 + 1 + q_dim1], ldq);
    }

    slacpy_("A", &n12, k, &q[q_offset], ldq, &s[1], &n12);
    if (n12 != 0) {
	sgemm_("N", "N", n1, k, &n12, &c_b15, &q2[1], n1, &s[1], &n12, &c_b29,
		 &q[q_offset], ldq);
    } else {
	slaset_("A", n1, k, &c_b29, &c_b29, &q[q_dim1 + 1], ldq);
    }


L120:
    return 0;

/*     End of SLAED3 */

} /* slaed3_ */

/* Subroutine */ int slaed4_(integer *n, integer *i__, real *d__, real *z__,
	real *delta, real *rho, real *dlam, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real a, b, c__;
    static integer j;
    static real w;
    static integer ii;
    static real dw, zz[3];
    static integer ip1;
    static real del, eta, phi, eps, tau, psi;
    static integer iim1, iip1;
    static real dphi, dpsi;
    static integer iter;
    static real temp, prew, temp1, dltlb, dltub, midpt;
    static integer niter;
    static logical swtch;
    extern /* Subroutine */ int slaed5_(integer *, real *, real *, real *,
	    real *, real *), slaed6_(integer *, logical *, real *, real *,
	    real *, real *, real *, integer *);
    static logical swtch3;
    extern doublereal slamch_(char *);
    static logical orgati;
    static real erretm, rhoinv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       December 23, 1999


    Purpose
    =======

    This subroutine computes the I-th updated eigenvalue of a symmetric
    rank-one modification to a diagonal matrix whose elements are
    given in the array d, and that

               D(i) < D(j)  for  i < j

    and that RHO > 0.  This is arranged by the calling routine, and is
    no loss in generality.  The rank-one modified system is thus

               diag( D )  +  RHO *  Z * Z_transpose.

    where we assume the Euclidean norm of Z is 1.

    The method consists of approximating the rational functions in the
    secular equation by simpler interpolating rational functions.

    Arguments
    =========

    N      (input) INTEGER
           The length of all arrays.

    I      (input) INTEGER
           The index of the eigenvalue to be computed.  1 <= I <= N.

    D      (input) REAL array, dimension (N)
           The original eigenvalues.  It is assumed that they are in
           order, D(I) < D(J)  for I < J.

    Z      (input) REAL array, dimension (N)
           The components of the updating vector.

    DELTA  (output) REAL array, dimension (N)
           If N .ne. 1, DELTA contains (D(j) - lambda_I) in its  j-th
           component.  If N = 1, then DELTA(1) = 1.  The vector DELTA
           contains the information necessary to construct the
           eigenvectors.

    RHO    (input) REAL
           The scalar in the symmetric updating formula.

    DLAM   (output) REAL
           The computed lambda_I, the I-th updated eigenvalue.

    INFO   (output) INTEGER
           = 0:  successful exit
           > 0:  if INFO = 1, the updating process failed.

    Internal Parameters
    ===================

    Logical variable ORGATI (origin-at-i?) is used for distinguishing
    whether D(i) or D(i+1) is treated as the origin.

              ORGATI = .true.    origin at i
              ORGATI = .false.   origin at i+1

     Logical variable SWTCH3 (switch-for-3-poles?) is for noting
     if we are working with THREE poles!

     MAXIT is the maximum number of iterations allowed for each
     eigenvalue.

    Further Details
    ===============

    Based on contributions by
       Ren-Cang Li, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Since this routine is called in an inner loop, we do no argument
       checking.

       Quick return for N=1 and 2.
*/

    /* Parameter adjustments */
    --delta;
    --z__;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n == 1) {

/*         Presumably, I=1 upon entry */

	*dlam = d__[1] + *rho * z__[1] * z__[1];
	delta[1] = 1.f;
	return 0;
    }
    if (*n == 2) {
	slaed5_(i__, &d__[1], &z__[1], &delta[1], rho, dlam);
	return 0;
    }

/*     Compute machine epsilon */

    eps = slamch_("Epsilon");
    rhoinv = 1.f / *rho;

/*     The case I = N */

    if (*i__ == *n) {

/*        Initialize some basic variables */

	ii = *n - 1;
	niter = 1;

/*        Calculate initial guess */

	midpt = *rho / 2.f;

/*
          If ||Z||_2 is not one, then TEMP should be set to
          RHO * ||Z||_2^2 / TWO
*/

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - midpt;
/* L10: */
	}

	psi = 0.f;
	i__1 = *n - 2;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / delta[j];
/* L20: */
	}

	c__ = rhoinv + psi;
	w = c__ + z__[ii] * z__[ii] / delta[ii] + z__[*n] * z__[*n] / delta[*
		n];

	if (w <= 0.f) {
	    temp = z__[*n - 1] * z__[*n - 1] / (d__[*n] - d__[*n - 1] + *rho)
		    + z__[*n] * z__[*n] / *rho;
	    if (c__ <= temp) {
		tau = *rho;
	    } else {
		del = d__[*n] - d__[*n - 1];
		a = -c__ * del + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n]
			;
		b = z__[*n] * z__[*n] * del;
		if (a < 0.f) {
		    tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
		} else {
		    tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
		}
	    }

/*
             It can be proved that
                 D(N)+RHO/2 <= LAMBDA(N) < D(N)+TAU <= D(N)+RHO
*/

	    dltlb = midpt;
	    dltub = *rho;
	} else {
	    del = d__[*n] - d__[*n - 1];
	    a = -c__ * del + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n];
	    b = z__[*n] * z__[*n] * del;
	    if (a < 0.f) {
		tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
	    } else {
		tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
	    }

/*
             It can be proved that
                 D(N) < D(N)+TAU < LAMBDA(N) < D(N)+RHO/2
*/

	    dltlb = 0.f;
	    dltub = midpt;
	}

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - tau;
/* L30: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L40: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / delta[*n];
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    *dlam = d__[*i__] + tau;
	    goto L250;
	}

	if (w <= 0.f) {
	    dltlb = dmax(dltlb,tau);
	} else {
	    dltub = dmin(dltub,tau);
	}

/*        Calculate the new step */

	++niter;
	c__ = w - delta[*n - 1] * dpsi - delta[*n] * dphi;
	a = (delta[*n - 1] + delta[*n]) * w - delta[*n - 1] * delta[*n] * (
		dpsi + dphi);
	b = delta[*n - 1] * delta[*n] * w;
	if (c__ < 0.f) {
	    c__ = dabs(c__);
	}
	if (c__ == 0.f) {
/*
            ETA = B/A
             ETA = RHO - TAU
*/
	    eta = dltub - tau;
	} else if (a >= 0.f) {
	    eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) / (
		    c__ * 2.f);
	} else {
	    eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
		    r__1))));
	}

/*
          Note, eta should be positive if w is negative, and
          eta should be negative otherwise. However,
          if for some reason caused by roundoff, eta*w > 0,
          we simply use one Newton step instead. This way
          will guarantee eta*w < 0.
*/

	if (w * eta > 0.f) {
	    eta = -w / (dpsi + dphi);
	}
	temp = tau + eta;
	if (temp > dltub || temp < dltlb) {
	    if (w < 0.f) {
		eta = (dltub - tau) / 2.f;
	    } else {
		eta = (dltlb - tau) / 2.f;
	    }
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
/* L50: */
	}

	tau += eta;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L60: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / delta[*n];
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 30; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		*dlam = d__[*i__] + tau;
		goto L250;
	    }

	    if (w <= 0.f) {
		dltlb = dmax(dltlb,tau);
	    } else {
		dltub = dmin(dltub,tau);
	    }

/*           Calculate the new step */

	    c__ = w - delta[*n - 1] * dpsi - delta[*n] * dphi;
	    a = (delta[*n - 1] + delta[*n]) * w - delta[*n - 1] * delta[*n] *
		    (dpsi + dphi);
	    b = delta[*n - 1] * delta[*n] * w;
	    if (a >= 0.f) {
		eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }

/*
             Note, eta should be positive if w is negative, and
             eta should be negative otherwise. However,
             if for some reason caused by roundoff, eta*w > 0,
             we simply use one Newton step instead. This way
             will guarantee eta*w < 0.
*/

	    if (w * eta > 0.f) {
		eta = -w / (dpsi + dphi);
	    }
	    temp = tau + eta;
	    if (temp > dltub || temp < dltlb) {
		if (w < 0.f) {
		    eta = (dltub - tau) / 2.f;
		} else {
		    eta = (dltlb - tau) / 2.f;
		}
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
/* L70: */
	    }

	    tau += eta;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = ii;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / delta[j];
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L80: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    temp = z__[*n] / delta[*n];
	    phi = z__[*n] * temp;
	    dphi = temp * temp;
	    erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) *
		    (dpsi + dphi);

	    w = rhoinv + phi + psi;
/* L90: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	*dlam = d__[*i__] + tau;
	goto L250;

/*        End for the case I = N */

    } else {

/*        The case for I < N */

	niter = 1;
	ip1 = *i__ + 1;

/*        Calculate initial guess */

	del = d__[ip1] - d__[*i__];
	midpt = del / 2.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - midpt;
/* L100: */
	}

	psi = 0.f;
	i__1 = *i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / delta[j];
/* L110: */
	}

	phi = 0.f;
	i__1 = *i__ + 2;
	for (j = *n; j >= i__1; --j) {
	    phi += z__[j] * z__[j] / delta[j];
/* L120: */
	}
	c__ = rhoinv + psi + phi;
	w = c__ + z__[*i__] * z__[*i__] / delta[*i__] + z__[ip1] * z__[ip1] /
		delta[ip1];

	if (w > 0.f) {

/*
             d(i)< the ith eigenvalue < (d(i)+d(i+1))/2

             We choose d(i) as origin.
*/

	    orgati = TRUE_;
	    a = c__ * del + z__[*i__] * z__[*i__] + z__[ip1] * z__[ip1];
	    b = z__[*i__] * z__[*i__] * del;
	    if (a > 0.f) {
		tau = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    }
	    dltlb = 0.f;
	    dltub = midpt;
	} else {

/*
             (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)

             We choose d(i+1) as origin.
*/

	    orgati = FALSE_;
	    a = c__ * del - z__[*i__] * z__[*i__] - z__[ip1] * z__[ip1];
	    b = z__[ip1] * z__[ip1] * del;
	    if (a < 0.f) {
		tau = b * 2.f / (a - sqrt((r__1 = a * a + b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = -(a + sqrt((r__1 = a * a + b * 4.f * c__, dabs(r__1))))
			/ (c__ * 2.f);
	    }
	    dltlb = -midpt;
	    dltub = 0.f;
	}

	if (orgati) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] = d__[j] - d__[*i__] - tau;
/* L130: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] = d__[j] - d__[ip1] - tau;
/* L140: */
	    }
	}
	if (orgati) {
	    ii = *i__;
	} else {
	    ii = *i__ + 1;
	}
	iim1 = ii - 1;
	iip1 = ii + 1;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L150: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / delta[j];
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L160: */
	}

	w = rhoinv + phi + psi;

/*
          W is the value of the secular function with
          its ii-th element removed.
*/

	swtch3 = FALSE_;
	if (orgati) {
	    if (w < 0.f) {
		swtch3 = TRUE_;
	    }
	} else {
	    if (w > 0.f) {
		swtch3 = TRUE_;
	    }
	}
	if (ii == 1 || ii == *n) {
	    swtch3 = FALSE_;
	}

	temp = z__[ii] / delta[ii];
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w += temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f
		+ dabs(tau) * dw;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    if (orgati) {
		*dlam = d__[*i__] + tau;
	    } else {
		*dlam = d__[ip1] + tau;
	    }
	    goto L250;
	}

	if (w <= 0.f) {
	    dltlb = dmax(dltlb,tau);
	} else {
	    dltub = dmin(dltub,tau);
	}

/*        Calculate the new step */

	++niter;
	if (! swtch3) {
	    if (orgati) {
/* Computing 2nd power */
		r__1 = z__[*i__] / delta[*i__];
		c__ = w - delta[ip1] * dw - (d__[*i__] - d__[ip1]) * (r__1 *
			r__1);
	    } else {
/* Computing 2nd power */
		r__1 = z__[ip1] / delta[ip1];
		c__ = w - delta[*i__] * dw - (d__[ip1] - d__[*i__]) * (r__1 *
			r__1);
	    }
	    a = (delta[*i__] + delta[ip1]) * w - delta[*i__] * delta[ip1] *
		    dw;
	    b = delta[*i__] * delta[ip1] * w;
	    if (c__ == 0.f) {
		if (a == 0.f) {
		    if (orgati) {
			a = z__[*i__] * z__[*i__] + delta[ip1] * delta[ip1] *
				(dpsi + dphi);
		    } else {
			a = z__[ip1] * z__[ip1] + delta[*i__] * delta[*i__] *
				(dpsi + dphi);
		    }
		}
		eta = b / a;
	    } else if (a <= 0.f) {
		eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }
	} else {

/*           Interpolation using THREE most relevant poles */

	    temp = rhoinv + psi + phi;
	    if (orgati) {
		temp1 = z__[iim1] / delta[iim1];
		temp1 *= temp1;
		c__ = temp - delta[iip1] * (dpsi + dphi) - (d__[iim1] - d__[
			iip1]) * temp1;
		zz[0] = z__[iim1] * z__[iim1];
		zz[2] = delta[iip1] * delta[iip1] * (dpsi - temp1 + dphi);
	    } else {
		temp1 = z__[iip1] / delta[iip1];
		temp1 *= temp1;
		c__ = temp - delta[iim1] * (dpsi + dphi) - (d__[iip1] - d__[
			iim1]) * temp1;
		zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
		zz[2] = z__[iip1] * z__[iip1];
	    }
	    zz[1] = z__[ii] * z__[ii];
	    slaed6_(&niter, &orgati, &c__, &delta[iim1], zz, &w, &eta, info);
	    if (*info != 0) {
		goto L250;
	    }
	}

/*
          Note, eta should be positive if w is negative, and
          eta should be negative otherwise. However,
          if for some reason caused by roundoff, eta*w > 0,
          we simply use one Newton step instead. This way
          will guarantee eta*w < 0.
*/

	if (w * eta >= 0.f) {
	    eta = -w / dw;
	}
	temp = tau + eta;
	if (temp > dltub || temp < dltlb) {
	    if (w < 0.f) {
		eta = (dltub - tau) / 2.f;
	    } else {
		eta = (dltlb - tau) / 2.f;
	    }
	}

	prew = w;

/* L170: */
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
/* L180: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / delta[j];
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L190: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / delta[j];
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L200: */
	}

	temp = z__[ii] / delta[ii];
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w = rhoinv + phi + psi + temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f
		+ (r__1 = tau + eta, dabs(r__1)) * dw;

	swtch = FALSE_;
	if (orgati) {
	    if (-w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	} else {
	    if (w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	}

	tau += eta;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 30; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		if (orgati) {
		    *dlam = d__[*i__] + tau;
		} else {
		    *dlam = d__[ip1] + tau;
		}
		goto L250;
	    }

	    if (w <= 0.f) {
		dltlb = dmax(dltlb,tau);
	    } else {
		dltub = dmin(dltub,tau);
	    }

/*           Calculate the new step */

	    if (! swtch3) {
		if (! swtch) {
		    if (orgati) {
/* Computing 2nd power */
			r__1 = z__[*i__] / delta[*i__];
			c__ = w - delta[ip1] * dw - (d__[*i__] - d__[ip1]) * (
				r__1 * r__1);
		    } else {
/* Computing 2nd power */
			r__1 = z__[ip1] / delta[ip1];
			c__ = w - delta[*i__] * dw - (d__[ip1] - d__[*i__]) *
				(r__1 * r__1);
		    }
		} else {
		    temp = z__[ii] / delta[ii];
		    if (orgati) {
			dpsi += temp * temp;
		    } else {
			dphi += temp * temp;
		    }
		    c__ = w - delta[*i__] * dpsi - delta[ip1] * dphi;
		}
		a = (delta[*i__] + delta[ip1]) * w - delta[*i__] * delta[ip1]
			* dw;
		b = delta[*i__] * delta[ip1] * w;
		if (c__ == 0.f) {
		    if (a == 0.f) {
			if (! swtch) {
			    if (orgati) {
				a = z__[*i__] * z__[*i__] + delta[ip1] *
					delta[ip1] * (dpsi + dphi);
			    } else {
				a = z__[ip1] * z__[ip1] + delta[*i__] * delta[
					*i__] * (dpsi + dphi);
			    }
			} else {
			    a = delta[*i__] * delta[*i__] * dpsi + delta[ip1]
				    * delta[ip1] * dphi;
			}
		    }
		    eta = b / a;
		} else if (a <= 0.f) {
		    eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1))
			    )) / (c__ * 2.f);
		} else {
		    eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__,
			    dabs(r__1))));
		}
	    } else {

/*              Interpolation using THREE most relevant poles */

		temp = rhoinv + psi + phi;
		if (swtch) {
		    c__ = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
		    zz[0] = delta[iim1] * delta[iim1] * dpsi;
		    zz[2] = delta[iip1] * delta[iip1] * dphi;
		} else {
		    if (orgati) {
			temp1 = z__[iim1] / delta[iim1];
			temp1 *= temp1;
			c__ = temp - delta[iip1] * (dpsi + dphi) - (d__[iim1]
				- d__[iip1]) * temp1;
			zz[0] = z__[iim1] * z__[iim1];
			zz[2] = delta[iip1] * delta[iip1] * (dpsi - temp1 +
				dphi);
		    } else {
			temp1 = z__[iip1] / delta[iip1];
			temp1 *= temp1;
			c__ = temp - delta[iim1] * (dpsi + dphi) - (d__[iip1]
				- d__[iim1]) * temp1;
			zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi -
				temp1));
			zz[2] = z__[iip1] * z__[iip1];
		    }
		}
		slaed6_(&niter, &orgati, &c__, &delta[iim1], zz, &w, &eta,
			info);
		if (*info != 0) {
		    goto L250;
		}
	    }

/*
             Note, eta should be positive if w is negative, and
             eta should be negative otherwise. However,
             if for some reason caused by roundoff, eta*w > 0,
             we simply use one Newton step instead. This way
             will guarantee eta*w < 0.
*/

	    if (w * eta >= 0.f) {
		eta = -w / dw;
	    }
	    temp = tau + eta;
	    if (temp > dltub || temp < dltlb) {
		if (w < 0.f) {
		    eta = (dltub - tau) / 2.f;
		} else {
		    eta = (dltlb - tau) / 2.f;
		}
	    }

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
/* L210: */
	    }

	    tau += eta;
	    prew = w;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = iim1;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / delta[j];
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L220: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    dphi = 0.f;
	    phi = 0.f;
	    i__1 = iip1;
	    for (j = *n; j >= i__1; --j) {
		temp = z__[j] / delta[j];
		phi += z__[j] * temp;
		dphi += temp * temp;
		erretm += phi;
/* L230: */
	    }

	    temp = z__[ii] / delta[ii];
	    dw = dpsi + dphi + temp * temp;
	    temp = z__[ii] * temp;
	    w = rhoinv + phi + psi + temp;
	    erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) *
		    3.f + dabs(tau) * dw;
	    if (w * prew > 0.f && dabs(w) > dabs(prew) / 10.f) {
		swtch = ! swtch;
	    }

/* L240: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	if (orgati) {
	    *dlam = d__[*i__] + tau;
	} else {
	    *dlam = d__[ip1] + tau;
	}

    }

L250:

    return 0;

/*     End of SLAED4 */

} /* slaed4_ */

/* Subroutine */ int slaed5_(integer *i__, real *d__, real *z__, real *delta,
	real *rho, real *dlam)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real b, c__, w, del, tau, temp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       September 30, 1994


    Purpose
    =======

    This subroutine computes the I-th eigenvalue of a symmetric rank-one
    modification of a 2-by-2 diagonal matrix

               diag( D )  +  RHO *  Z * transpose(Z) .

    The diagonal elements in the array D are assumed to satisfy

               D(i) < D(j)  for  i < j .

    We also assume RHO > 0 and that the Euclidean norm of the vector
    Z is one.

    Arguments
    =========

    I      (input) INTEGER
           The index of the eigenvalue to be computed.  I = 1 or I = 2.

    D      (input) REAL array, dimension (2)
           The original eigenvalues.  We assume D(1) < D(2).

    Z      (input) REAL array, dimension (2)
           The components of the updating vector.

    DELTA  (output) REAL array, dimension (2)
           The vector DELTA contains the information necessary
           to construct the eigenvectors.

    RHO    (input) REAL
           The scalar in the symmetric updating formula.

    DLAM   (output) REAL
           The computed lambda_I, the I-th updated eigenvalue.

    Further Details
    ===============

    Based on contributions by
       Ren-Cang Li, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================
*/


    /* Parameter adjustments */
    --delta;
    --z__;
    --d__;

    /* Function Body */
    del = d__[2] - d__[1];
    if (*i__ == 1) {
	w = *rho * 2.f * (z__[2] * z__[2] - z__[1] * z__[1]) / del + 1.f;
	if (w > 0.f) {
	    b = del + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[1] * z__[1] * del;

/*           B > ZERO, always */

	    tau = c__ * 2.f / (b + sqrt((r__1 = b * b - c__ * 4.f, dabs(r__1))
		    ));
	    *dlam = d__[1] + tau;
	    delta[1] = -z__[1] / tau;
	    delta[2] = z__[2] / (del - tau);
	} else {
	    b = -del + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[2] * z__[2] * del;
	    if (b > 0.f) {
		tau = c__ * -2.f / (b + sqrt(b * b + c__ * 4.f));
	    } else {
		tau = (b - sqrt(b * b + c__ * 4.f)) / 2.f;
	    }
	    *dlam = d__[2] + tau;
	    delta[1] = -z__[1] / (del + tau);
	    delta[2] = -z__[2] / tau;
	}
	temp = sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
	delta[1] /= temp;
	delta[2] /= temp;
    } else {

/*     Now I=2 */

	b = -del + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	c__ = *rho * z__[2] * z__[2] * del;
	if (b > 0.f) {
	    tau = (b + sqrt(b * b + c__ * 4.f)) / 2.f;
	} else {
	    tau = c__ * 2.f / (-b + sqrt(b * b + c__ * 4.f));
	}
	*dlam = d__[2] + tau;
	delta[1] = -z__[1] / (del + tau);
	delta[2] = -z__[2] / tau;
	temp = sqrt(delta[1] * delta[1] + delta[2] * delta[2]);
	delta[1] /= temp;
	delta[2] /= temp;
    }
    return 0;

/*     End OF SLAED5 */

} /* slaed5_ */

/* Subroutine */ int slaed6_(integer *kniter, logical *orgati, real *rho,
	real *d__, real *z__, real *finit, real *tau, integer *info)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    double sqrt(doublereal), log(doublereal), pow_ri(real *, integer *);

    /* Local variables */
    static real a, b, c__, f;
    static integer i__;
    static real fc, df, ddf, eta, eps, base;
    static integer iter;
    static real temp, temp1, temp2, temp3, temp4;
    static logical scale;
    static integer niter;
    static real small1, small2, sminv1, sminv2, dscale[3], sclfac;
    extern doublereal slamch_(char *);
    static real zscale[3], erretm, sclinv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       June 30, 1999


    Purpose
    =======

    SLAED6 computes the positive or negative root (closest to the origin)
    of
                     z(1)        z(2)        z(3)
    f(x) =   rho + --------- + ---------- + ---------
                    d(1)-x      d(2)-x      d(3)-x

    It is assumed that

          if ORGATI = .true. the root is between d(2) and d(3);
          otherwise it is between d(1) and d(2)

    This routine will be called by SLAED4 when necessary. In most cases,
    the root sought is the smallest in magnitude, though it might not be
    in some extremely rare situations.

    Arguments
    =========

    KNITER       (input) INTEGER
                 Refer to SLAED4 for its significance.

    ORGATI       (input) LOGICAL
                 If ORGATI is true, the needed root is between d(2) and
                 d(3); otherwise it is between d(1) and d(2).  See
                 SLAED4 for further details.

    RHO          (input) REAL
                 Refer to the equation f(x) above.

    D            (input) REAL array, dimension (3)
                 D satisfies d(1) < d(2) < d(3).

    Z            (input) REAL array, dimension (3)
                 Each of the elements in z must be positive.

    FINIT        (input) REAL
                 The value of f at 0. It is more accurate than the one
                 evaluated inside this routine (if someone wants to do
                 so).

    TAU          (output) REAL
                 The root of the equation f(x).

    INFO         (output) INTEGER
                 = 0: successful exit
                 > 0: if INFO = 1, failure to converge

    Further Details
    ===============

    Based on contributions by
       Ren-Cang Li, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================
*/

    /* Parameter adjustments */
    --z__;
    --d__;

    /* Function Body */

    *info = 0;

    niter = 1;
    *tau = 0.f;
    if (*kniter == 2) {
	if (*orgati) {
	    temp = (d__[3] - d__[2]) / 2.f;
	    c__ = *rho + z__[1] / (d__[1] - d__[2] - temp);
	    a = c__ * (d__[2] + d__[3]) + z__[2] + z__[3];
	    b = c__ * d__[2] * d__[3] + z__[2] * d__[3] + z__[3] * d__[2];
	} else {
	    temp = (d__[1] - d__[2]) / 2.f;
	    c__ = *rho + z__[3] / (d__[3] - d__[2] - temp);
	    a = c__ * (d__[1] + d__[2]) + z__[1] + z__[2];
	    b = c__ * d__[1] * d__[2] + z__[1] * d__[2] + z__[2] * d__[1];
	}
/* Computing MAX */
	r__1 = dabs(a), r__2 = dabs(b), r__1 = max(r__1,r__2), r__2 = dabs(
		c__);
	temp = dmax(r__1,r__2);
	a /= temp;
	b /= temp;
	c__ /= temp;
	if (c__ == 0.f) {
	    *tau = b / a;
	} else if (a <= 0.f) {
	    *tau = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) / (
		    c__ * 2.f);
	} else {
	    *tau = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
		    r__1))));
	}
	temp = *rho + z__[1] / (d__[1] - *tau) + z__[2] / (d__[2] - *tau) +
		z__[3] / (d__[3] - *tau);
	if (dabs(*finit) <= dabs(temp)) {
	    *tau = 0.f;
	}
    }

/*
       On first call to routine, get machine parameters for
       possible scaling to avoid overflow
*/

    if (first) {
	eps = slamch_("Epsilon");
	base = slamch_("Base");
	i__1 = (integer) (log(slamch_("SafMin")) / log(base) / 3.f)
		;
	small1 = pow_ri(&base, &i__1);
	sminv1 = 1.f / small1;
	small2 = small1 * small1;
	sminv2 = sminv1 * sminv1;
	first = FALSE_;
    }

/*
       Determine if scaling of inputs necessary to avoid overflow
       when computing 1/TEMP**3
*/

    if (*orgati) {
/* Computing MIN */
	r__3 = (r__1 = d__[2] - *tau, dabs(r__1)), r__4 = (r__2 = d__[3] - *
		tau, dabs(r__2));
	temp = dmin(r__3,r__4);
    } else {
/* Computing MIN */
	r__3 = (r__1 = d__[1] - *tau, dabs(r__1)), r__4 = (r__2 = d__[2] - *
		tau, dabs(r__2));
	temp = dmin(r__3,r__4);
    }
    scale = FALSE_;
    if (temp <= small1) {
	scale = TRUE_;
	if (temp <= small2) {

/*        Scale up by power of radix nearest 1/SAFMIN**(2/3) */

	    sclfac = sminv2;
	    sclinv = small2;
	} else {

/*        Scale up by power of radix nearest 1/SAFMIN**(1/3) */

	    sclfac = sminv1;
	    sclinv = small1;
	}

/*        Scaling up safe because D, Z, TAU scaled elsewhere to be O(1) */

	for (i__ = 1; i__ <= 3; ++i__) {
	    dscale[i__ - 1] = d__[i__] * sclfac;
	    zscale[i__ - 1] = z__[i__] * sclfac;
/* L10: */
	}
	*tau *= sclfac;
    } else {

/*        Copy D and Z to DSCALE and ZSCALE */

	for (i__ = 1; i__ <= 3; ++i__) {
	    dscale[i__ - 1] = d__[i__];
	    zscale[i__ - 1] = z__[i__];
/* L20: */
	}
    }

    fc = 0.f;
    df = 0.f;
    ddf = 0.f;
    for (i__ = 1; i__ <= 3; ++i__) {
	temp = 1.f / (dscale[i__ - 1] - *tau);
	temp1 = zscale[i__ - 1] * temp;
	temp2 = temp1 * temp;
	temp3 = temp2 * temp;
	fc += temp1 / dscale[i__ - 1];
	df += temp2;
	ddf += temp3;
/* L30: */
    }
    f = *finit + *tau * fc;

    if (dabs(f) <= 0.f) {
	goto L60;
    }

/*
          Iteration begins

       It is not hard to see that

             1) Iterations will go up monotonically
                if FINIT < 0;

             2) Iterations will go down monotonically
                if FINIT > 0.
*/

    iter = niter + 1;

    for (niter = iter; niter <= 20; ++niter) {

	if (*orgati) {
	    temp1 = dscale[1] - *tau;
	    temp2 = dscale[2] - *tau;
	} else {
	    temp1 = dscale[0] - *tau;
	    temp2 = dscale[1] - *tau;
	}
	a = (temp1 + temp2) * f - temp1 * temp2 * df;
	b = temp1 * temp2 * f;
	c__ = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
/* Computing MAX */
	r__1 = dabs(a), r__2 = dabs(b), r__1 = max(r__1,r__2), r__2 = dabs(
		c__);
	temp = dmax(r__1,r__2);
	a /= temp;
	b /= temp;
	c__ /= temp;
	if (c__ == 0.f) {
	    eta = b / a;
	} else if (a <= 0.f) {
	    eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) / (
		    c__ * 2.f);
	} else {
	    eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
		    r__1))));
	}
	if (f * eta >= 0.f) {
	    eta = -f / df;
	}

	temp = eta + *tau;
	if (*orgati) {
	    if (eta > 0.f && temp >= dscale[2]) {
		eta = (dscale[2] - *tau) / 2.f;
	    }
	    if (eta < 0.f && temp <= dscale[1]) {
		eta = (dscale[1] - *tau) / 2.f;
	    }
	} else {
	    if (eta > 0.f && temp >= dscale[1]) {
		eta = (dscale[1] - *tau) / 2.f;
	    }
	    if (eta < 0.f && temp <= dscale[0]) {
		eta = (dscale[0] - *tau) / 2.f;
	    }
	}
	*tau += eta;

	fc = 0.f;
	erretm = 0.f;
	df = 0.f;
	ddf = 0.f;
	for (i__ = 1; i__ <= 3; ++i__) {
	    temp = 1.f / (dscale[i__ - 1] - *tau);
	    temp1 = zscale[i__ - 1] * temp;
	    temp2 = temp1 * temp;
	    temp3 = temp2 * temp;
	    temp4 = temp1 / dscale[i__ - 1];
	    fc += temp4;
	    erretm += dabs(temp4);
	    df += temp2;
	    ddf += temp3;
/* L40: */
	}
	f = *finit + *tau * fc;
	erretm = (dabs(*finit) + dabs(*tau) * erretm) * 8.f + dabs(*tau) * df;
	if (dabs(f) <= eps * erretm) {
	    goto L60;
	}
/* L50: */
    }
    *info = 1;
L60:

/*     Undo scaling */

    if (scale) {
	*tau *= sclinv;
    }
    return 0;

/*     End of SLAED6 */

} /* slaed6_ */

/* Subroutine */ int slaed7_(integer *icompq, integer *n, integer *qsiz,
	integer *tlvls, integer *curlvl, integer *curpbm, real *d__, real *q,
	integer *ldq, integer *indxq, real *rho, integer *cutpnt, real *
	qstore, integer *qptr, integer *prmptr, integer *perm, integer *
	givptr, integer *givcol, real *givnum, real *work, integer *iwork,
	integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, k, n1, n2, is, iw, iz, iq2, ptr, ldq2, indx, curr,
	    indxc;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer indxp;
    extern /* Subroutine */ int slaed8_(integer *, integer *, integer *,
	    integer *, real *, real *, integer *, integer *, real *, integer *
	    , real *, real *, real *, integer *, real *, integer *, integer *,
	     integer *, real *, integer *, integer *, integer *), slaed9_(
	    integer *, integer *, integer *, integer *, real *, real *,
	    integer *, real *, real *, real *, real *, integer *, integer *),
	    slaeda_(integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, real *, real *, integer *, real *
	    , real *, integer *);
    static integer idlmda;
    extern /* Subroutine */ int xerbla_(char *, integer *), slamrg_(
	    integer *, integer *, real *, integer *, integer *, integer *);
    static integer coltyp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLAED7 computes the updated eigensystem of a diagonal
    matrix after modification by a rank-one symmetric matrix. This
    routine is used only for the eigenproblem which requires all
    eigenvalues and optionally eigenvectors of a dense symmetric matrix
    that has been reduced to tridiagonal form.  SLAED1 handles
    the case in which all eigenvalues and eigenvectors of a symmetric
    tridiagonal matrix are desired.

      T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out)

       where Z = Q'u, u is a vector of length N with ones in the
       CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.

       The eigenvectors of the original matrix are stored in Q, and the
       eigenvalues are in D.  The algorithm consists of three stages:

          The first stage consists of deflating the size of the problem
          when there are multiple eigenvalues or if there is a zero in
          the Z vector.  For each such occurence the dimension of the
          secular equation problem is reduced by one.  This stage is
          performed by the routine SLAED8.

          The second stage consists of calculating the updated
          eigenvalues. This is done by finding the roots of the secular
          equation via the routine SLAED4 (as called by SLAED9).
          This routine also calculates the eigenvectors of the current
          problem.

          The final stage consists of computing the updated eigenvectors
          directly using the updated eigenvalues.  The eigenvectors for
          the current problem are multiplied with the eigenvectors from
          the overall problem.

    Arguments
    =========

    ICOMPQ  (input) INTEGER
            = 0:  Compute eigenvalues only.
            = 1:  Compute eigenvectors of original dense symmetric matrix
                  also.  On entry, Q contains the orthogonal matrix used
                  to reduce the original matrix to tridiagonal form.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    QSIZ   (input) INTEGER
           The dimension of the orthogonal matrix used to reduce
           the full matrix to tridiagonal form.  QSIZ >= N if ICOMPQ = 1.

    TLVLS  (input) INTEGER
           The total number of merging levels in the overall divide and
           conquer tree.

    CURLVL (input) INTEGER
           The current level in the overall merge routine,
           0 <= CURLVL <= TLVLS.

    CURPBM (input) INTEGER
           The current problem in the current level in the overall
           merge routine (counting from upper left to lower right).

    D      (input/output) REAL array, dimension (N)
           On entry, the eigenvalues of the rank-1-perturbed matrix.
           On exit, the eigenvalues of the repaired matrix.

    Q      (input/output) REAL array, dimension (LDQ, N)
           On entry, the eigenvectors of the rank-1-perturbed matrix.
           On exit, the eigenvectors of the repaired tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    INDXQ  (output) INTEGER array, dimension (N)
           The permutation which will reintegrate the subproblem just
           solved back into sorted order, i.e., D( INDXQ( I = 1, N ) )
           will be in ascending order.

    RHO    (input) REAL
           The subdiagonal element used to create the rank-1
           modification.

    CUTPNT (input) INTEGER
           Contains the location of the last eigenvalue in the leading
           sub-matrix.  min(1,N) <= CUTPNT <= N.

    QSTORE (input/output) REAL array, dimension (N**2+1)
           Stores eigenvectors of submatrices encountered during
           divide and conquer, packed together. QPTR points to
           beginning of the submatrices.

    QPTR   (input/output) INTEGER array, dimension (N+2)
           List of indices pointing to beginning of submatrices stored
           in QSTORE. The submatrices are numbered starting at the
           bottom left of the divide and conquer tree, from left to
           right and bottom to top.

    PRMPTR (input) INTEGER array, dimension (N lg N)
           Contains a list of pointers which indicate where in PERM a
           level's permutation is stored.  PRMPTR(i+1) - PRMPTR(i)
           indicates the size of the permutation and also the size of
           the full, non-deflated problem.

    PERM   (input) INTEGER array, dimension (N lg N)
           Contains the permutations (from deflation and sorting) to be
           applied to each eigenblock.

    GIVPTR (input) INTEGER array, dimension (N lg N)
           Contains a list of pointers which indicate where in GIVCOL a
           level's Givens rotations are stored.  GIVPTR(i+1) - GIVPTR(i)
           indicates the number of Givens rotations.

    GIVCOL (input) INTEGER array, dimension (2, N lg N)
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation.

    GIVNUM (input) REAL array, dimension (2, N lg N)
           Each number indicates the S value to be used in the
           corresponding Givens rotation.

    WORK   (workspace) REAL array, dimension (3*N+QSIZ*N)

    IWORK  (workspace) INTEGER array, dimension (4*N)

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --qstore;
    --qptr;
    --prmptr;
    --perm;
    --givptr;
    givcol -= 3;
    givnum -= 3;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*icompq == 1 && *qsiz < *n) {
	*info = -4;
    } else if (*ldq < max(1,*n)) {
	*info = -9;
    } else if (min(1,*n) > *cutpnt || *n < *cutpnt) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED7", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*
       The following values are for bookkeeping purposes only.  They are
       integer pointers which indicate the portion of the workspace
       used by a particular array in SLAED8 and SLAED9.
*/

    if (*icompq == 1) {
	ldq2 = *qsiz;
    } else {
	ldq2 = *n;
    }

    iz = 1;
    idlmda = iz + *n;
    iw = idlmda + *n;
    iq2 = iw + *n;
    is = iq2 + *n * ldq2;

    indx = 1;
    indxc = indx + *n;
    coltyp = indxc + *n;
    indxp = coltyp + *n;

/*
       Form the z-vector which consists of the last row of Q_1 and the
       first row of Q_2.
*/

    ptr = pow_ii(&c__2, tlvls) + 1;
    i__1 = *curlvl - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *tlvls - i__;
	ptr += pow_ii(&c__2, &i__2);
/* L10: */
    }
    curr = ptr + *curpbm;
    slaeda_(n, tlvls, curlvl, curpbm, &prmptr[1], &perm[1], &givptr[1], &
	    givcol[3], &givnum[3], &qstore[1], &qptr[1], &work[iz], &work[iz
	    + *n], info);

/*
       When solving the final problem, we no longer need the stored data,
       so we will overwrite the data from this level onto the previously
       used storage space.
*/

    if (*curlvl == *tlvls) {
	qptr[curr] = 1;
	prmptr[curr] = 1;
	givptr[curr] = 1;
    }

/*     Sort and Deflate eigenvalues. */

    slaed8_(icompq, &k, n, qsiz, &d__[1], &q[q_offset], ldq, &indxq[1], rho,
	    cutpnt, &work[iz], &work[idlmda], &work[iq2], &ldq2, &work[iw], &
	    perm[prmptr[curr]], &givptr[curr + 1], &givcol[(givptr[curr] << 1)
	     + 1], &givnum[(givptr[curr] << 1) + 1], &iwork[indxp], &iwork[
	    indx], info);
    prmptr[curr + 1] = prmptr[curr] + *n;
    givptr[curr + 1] += givptr[curr];

/*     Solve Secular Equation. */

    if (k != 0) {
	slaed9_(&k, &c__1, &k, n, &d__[1], &work[is], &k, rho, &work[idlmda],
		&work[iw], &qstore[qptr[curr]], &k, info);
	if (*info != 0) {
	    goto L30;
	}
	if (*icompq == 1) {
	    sgemm_("N", "N", qsiz, &k, &k, &c_b15, &work[iq2], &ldq2, &qstore[
		    qptr[curr]], &k, &c_b29, &q[q_offset], ldq);
	}
/* Computing 2nd power */
	i__1 = k;
	qptr[curr + 1] = qptr[curr] + i__1 * i__1;

/*     Prepare the INDXQ sorting permutation. */

	n1 = k;
	n2 = *n - k;
	slamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &indxq[1]);
    } else {
	qptr[curr + 1] = qptr[curr];
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    indxq[i__] = i__;
/* L20: */
	}
    }

L30:
    return 0;

/*     End of SLAED7 */

} /* slaed7_ */

/* Subroutine */ int slaed8_(integer *icompq, integer *k, integer *n, integer
	*qsiz, real *d__, real *q, integer *ldq, integer *indxq, real *rho,
	integer *cutpnt, real *z__, real *dlamda, real *q2, integer *ldq2,
	real *w, integer *perm, integer *givptr, integer *givcol, real *
	givnum, integer *indxp, integer *indx, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, q2_dim1, q2_offset, i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real c__;
    static integer i__, j;
    static real s, t;
    static integer k2, n1, n2, jp, n1p1;
    static real eps, tau, tol;
    static integer jlam, imax, jmax;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *), sscal_(integer *, real *, real *,
	    integer *), scopy_(integer *, real *, integer *, real *, integer *
	    );
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slamrg_(integer *, integer *, real *, integer
	    *, integer *, integer *), slacpy_(char *, integer *, integer *,
	    real *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       September 30, 1994


    Purpose
    =======

    SLAED8 merges the two sets of eigenvalues together into a single
    sorted set.  Then it tries to deflate the size of the problem.
    There are two ways in which deflation can occur:  when two or more
    eigenvalues are close together or if there is a tiny element in the
    Z vector.  For each such occurrence the order of the related secular
    equation problem is reduced by one.

    Arguments
    =========

    ICOMPQ  (input) INTEGER
            = 0:  Compute eigenvalues only.
            = 1:  Compute eigenvectors of original dense symmetric matrix
                  also.  On entry, Q contains the orthogonal matrix used
                  to reduce the original matrix to tridiagonal form.

    K      (output) INTEGER
           The number of non-deflated eigenvalues, and the order of the
           related secular equation.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    QSIZ   (input) INTEGER
           The dimension of the orthogonal matrix used to reduce
           the full matrix to tridiagonal form.  QSIZ >= N if ICOMPQ = 1.

    D      (input/output) REAL array, dimension (N)
           On entry, the eigenvalues of the two submatrices to be
           combined.  On exit, the trailing (N-K) updated eigenvalues
           (those which were deflated) sorted into increasing order.

    Q      (input/output) REAL array, dimension (LDQ,N)
           If ICOMPQ = 0, Q is not referenced.  Otherwise,
           on entry, Q contains the eigenvectors of the partially solved
           system which has been previously updated in matrix
           multiplies with other partially solved eigensystems.
           On exit, Q contains the trailing (N-K) updated eigenvectors
           (those which were deflated) in its last N-K columns.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    INDXQ  (input) INTEGER array, dimension (N)
           The permutation which separately sorts the two sub-problems
           in D into ascending order.  Note that elements in the second
           half of this permutation must first have CUTPNT added to
           their values in order to be accurate.

    RHO    (input/output) REAL
           On entry, the off-diagonal element associated with the rank-1
           cut which originally split the two submatrices which are now
           being recombined.
           On exit, RHO has been modified to the value required by
           SLAED3.

    CUTPNT (input) INTEGER
           The location of the last eigenvalue in the leading
           sub-matrix.  min(1,N) <= CUTPNT <= N.

    Z      (input) REAL array, dimension (N)
           On entry, Z contains the updating vector (the last row of
           the first sub-eigenvector matrix and the first row of the
           second sub-eigenvector matrix).
           On exit, the contents of Z are destroyed by the updating
           process.

    DLAMDA (output) REAL array, dimension (N)
           A copy of the first K eigenvalues which will be used by
           SLAED3 to form the secular equation.

    Q2     (output) REAL array, dimension (LDQ2,N)
           If ICOMPQ = 0, Q2 is not referenced.  Otherwise,
           a copy of the first K eigenvectors which will be used by
           SLAED7 in a matrix multiply (SGEMM) to update the new
           eigenvectors.

    LDQ2   (input) INTEGER
           The leading dimension of the array Q2.  LDQ2 >= max(1,N).

    W      (output) REAL array, dimension (N)
           The first k values of the final deflation-altered z-vector and
           will be passed to SLAED3.

    PERM   (output) INTEGER array, dimension (N)
           The permutations (from deflation and sorting) to be applied
           to each eigenblock.

    GIVPTR (output) INTEGER
           The number of Givens rotations which took place in this
           subproblem.

    GIVCOL (output) INTEGER array, dimension (2, N)
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation.

    GIVNUM (output) REAL array, dimension (2, N)
           Each number indicates the S value to be used in the
           corresponding Givens rotation.

    INDXP  (workspace) INTEGER array, dimension (N)
           The permutation used to place deflated values of D at the end
           of the array.  INDXP(1:K) points to the nondeflated D-values
           and INDXP(K+1:N) points to the deflated eigenvalues.

    INDX   (workspace) INTEGER array, dimension (N)
           The permutation used to sort the contents of D into ascending
           order.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --z__;
    --dlamda;
    q2_dim1 = *ldq2;
    q2_offset = 1 + q2_dim1;
    q2 -= q2_offset;
    --w;
    --perm;
    givcol -= 3;
    givnum -= 3;
    --indxp;
    --indx;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -3;
    } else if (*icompq == 1 && *qsiz < *n) {
	*info = -4;
    } else if (*ldq < max(1,*n)) {
	*info = -7;
    } else if (*cutpnt < min(1,*n) || *cutpnt > *n) {
	*info = -10;
    } else if (*ldq2 < max(1,*n)) {
	*info = -14;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED8", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    n1 = *cutpnt;
    n2 = *n - n1;
    n1p1 = n1 + 1;

    if (*rho < 0.f) {
	sscal_(&n2, &c_b151, &z__[n1p1], &c__1);
    }

/*     Normalize z so that norm(z) = 1 */

    t = 1.f / sqrt(2.f);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	indx[j] = j;
/* L10: */
    }
    sscal_(n, &t, &z__[1], &c__1);
    *rho = (r__1 = *rho * 2.f, dabs(r__1));

/*     Sort the eigenvalues into increasing order */

    i__1 = *n;
    for (i__ = *cutpnt + 1; i__ <= i__1; ++i__) {
	indxq[i__] += *cutpnt;
/* L20: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = d__[indxq[i__]];
	w[i__] = z__[indxq[i__]];
/* L30: */
    }
    i__ = 1;
    j = *cutpnt + 1;
    slamrg_(&n1, &n2, &dlamda[1], &c__1, &c__1, &indx[1]);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = dlamda[indx[i__]];
	z__[i__] = w[indx[i__]];
/* L40: */
    }

/*     Calculate the allowable deflation tolerence */

    imax = isamax_(n, &z__[1], &c__1);
    jmax = isamax_(n, &d__[1], &c__1);
    eps = slamch_("Epsilon");
    tol = eps * 8.f * (r__1 = d__[jmax], dabs(r__1));

/*
       If the rank-1 modifier is small enough, no more needs to be done
       except to reorganize Q so that its columns correspond with the
       elements in D.
*/

    if (*rho * (r__1 = z__[imax], dabs(r__1)) <= tol) {
	*k = 0;
	if (*icompq == 0) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		perm[j] = indxq[indx[j]];
/* L50: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		perm[j] = indxq[indx[j]];
		scopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1
			+ 1], &c__1);
/* L60: */
	    }
	    slacpy_("A", qsiz, n, &q2[q2_dim1 + 1], ldq2, &q[q_dim1 + 1], ldq);
	}
	return 0;
    }

/*
       If there are multiple eigenvalues then the problem deflates.  Here
       the number of equal eigenvalues are found.  As each equal
       eigenvalue is found, an elementary reflector is computed to rotate
       the corresponding eigensubspace so that the corresponding
       components of Z are zero in this new basis.
*/

    *k = 0;
    *givptr = 0;
    k2 = *n + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (*rho * (r__1 = z__[j], dabs(r__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    indxp[k2] = j;
	    if (j == *n) {
		goto L110;
	    }
	} else {
	    jlam = j;
	    goto L80;
	}
/* L70: */
    }
L80:
    ++j;
    if (j > *n) {
	goto L100;
    }
    if (*rho * (r__1 = z__[j], dabs(r__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	indxp[k2] = j;
    } else {

/*        Check if eigenvalues are close enough to allow deflation. */

	s = z__[jlam];
	c__ = z__[j];

/*
          Find sqrt(a**2+b**2) without overflow or
          destructive underflow.
*/

	tau = slapy2_(&c__, &s);
	t = d__[j] - d__[jlam];
	c__ /= tau;
	s = -s / tau;
	if ((r__1 = t * c__ * s, dabs(r__1)) <= tol) {

/*           Deflation is possible. */

	    z__[j] = tau;
	    z__[jlam] = 0.f;

/*           Record the appropriate Givens rotation */

	    ++(*givptr);
	    givcol[(*givptr << 1) + 1] = indxq[indx[jlam]];
	    givcol[(*givptr << 1) + 2] = indxq[indx[j]];
	    givnum[(*givptr << 1) + 1] = c__;
	    givnum[(*givptr << 1) + 2] = s;
	    if (*icompq == 1) {
		srot_(qsiz, &q[indxq[indx[jlam]] * q_dim1 + 1], &c__1, &q[
			indxq[indx[j]] * q_dim1 + 1], &c__1, &c__, &s);
	    }
	    t = d__[jlam] * c__ * c__ + d__[j] * s * s;
	    d__[j] = d__[jlam] * s * s + d__[j] * c__ * c__;
	    d__[jlam] = t;
	    --k2;
	    i__ = 1;
L90:
	    if (k2 + i__ <= *n) {
		if (d__[jlam] < d__[indxp[k2 + i__]]) {
		    indxp[k2 + i__ - 1] = indxp[k2 + i__];
		    indxp[k2 + i__] = jlam;
		    ++i__;
		    goto L90;
		} else {
		    indxp[k2 + i__ - 1] = jlam;
		}
	    } else {
		indxp[k2 + i__ - 1] = jlam;
	    }
	    jlam = j;
	} else {
	    ++(*k);
	    w[*k] = z__[jlam];
	    dlamda[*k] = d__[jlam];
	    indxp[*k] = jlam;
	    jlam = j;
	}
    }
    goto L80;
L100:

/*     Record the last eigenvalue. */

    ++(*k);
    w[*k] = z__[jlam];
    dlamda[*k] = d__[jlam];
    indxp[*k] = jlam;

L110:

/*
       Sort the eigenvalues and corresponding eigenvectors into DLAMDA
       and Q2 respectively.  The eigenvalues/vectors which were not
       deflated go into the first K slots of DLAMDA and Q2 respectively,
       while those which were deflated go into the last N - K slots.
*/

    if (*icompq == 0) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    jp = indxp[j];
	    dlamda[j] = d__[jp];
	    perm[j] = indxq[indx[jp]];
/* L120: */
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    jp = indxp[j];
	    dlamda[j] = d__[jp];
	    perm[j] = indxq[indx[jp]];
	    scopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1]
		    , &c__1);
/* L130: */
	}
    }

/*
       The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively.
*/

    if (*k < *n) {
	if (*icompq == 0) {
	    i__1 = *n - *k;
	    scopy_(&i__1, &dlamda[*k + 1], &c__1, &d__[*k + 1], &c__1);
	} else {
	    i__1 = *n - *k;
	    scopy_(&i__1, &dlamda[*k + 1], &c__1, &d__[*k + 1], &c__1);
	    i__1 = *n - *k;
	    slacpy_("A", qsiz, &i__1, &q2[(*k + 1) * q2_dim1 + 1], ldq2, &q[(*
		    k + 1) * q_dim1 + 1], ldq);
	}
    }

    return 0;

/*     End of SLAED8 */

} /* slaed8_ */

/* Subroutine */ int slaed9_(integer *k, integer *kstart, integer *kstop,
	integer *n, real *d__, real *q, integer *ldq, real *rho, real *dlamda,
	 real *w, real *s, integer *lds, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, s_dim1, s_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static integer i__, j;
    static real temp;
    extern doublereal snrm2_(integer *, real *, integer *);
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), slaed4_(integer *, integer *, real *, real *, real *,
	    real *, real *, integer *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       September 30, 1994


    Purpose
    =======

    SLAED9 finds the roots of the secular equation, as defined by the
    values in D, Z, and RHO, between KSTART and KSTOP.  It makes the
    appropriate calls to SLAED4 and then stores the new matrix of
    eigenvectors for use in calculating the next level of Z vectors.

    Arguments
    =========

    K       (input) INTEGER
            The number of terms in the rational function to be solved by
            SLAED4.  K >= 0.

    KSTART  (input) INTEGER
    KSTOP   (input) INTEGER
            The updated eigenvalues Lambda(I), KSTART <= I <= KSTOP
            are to be computed.  1 <= KSTART <= KSTOP <= K.

    N       (input) INTEGER
            The number of rows and columns in the Q matrix.
            N >= K (delation may result in N > K).

    D       (output) REAL array, dimension (N)
            D(I) contains the updated eigenvalues
            for KSTART <= I <= KSTOP.

    Q       (workspace) REAL array, dimension (LDQ,N)

    LDQ     (input) INTEGER
            The leading dimension of the array Q.  LDQ >= max( 1, N ).

    RHO     (input) REAL
            The value of the parameter in the rank one update equation.
            RHO >= 0 required.

    DLAMDA  (input) REAL array, dimension (K)
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation.

    W       (input) REAL array, dimension (K)
            The first K elements of this array contain the components
            of the deflation-adjusted updating vector.

    S       (output) REAL array, dimension (LDS, K)
            Will contain the eigenvectors of the repaired matrix which
            will be stored for subsequent Z vector calculation and
            multiplied by the previously accumulated eigenvectors
            to update the system.

    LDS     (input) INTEGER
            The leading dimension of S.  LDS >= max( 1, K ).

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dlamda;
    --w;
    s_dim1 = *lds;
    s_offset = 1 + s_dim1;
    s -= s_offset;

    /* Function Body */
    *info = 0;

    if (*k < 0) {
	*info = -1;
    } else if (*kstart < 1 || *kstart > max(1,*k)) {
	*info = -2;
    } else if (max(1,*kstop) < *kstart || *kstop > max(1,*k)) {
	*info = -3;
    } else if (*n < *k) {
	*info = -4;
    } else if (*ldq < max(1,*k)) {
	*info = -7;
    } else if (*lds < max(1,*k)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED9", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 0) {
	return 0;
    }

/*
       Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
       be computed with high relative accuracy (barring over/underflow).
       This is a problem on machines without a guard digit in
       add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
       The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
       which on any of these machines zeros out the bottommost
       bit of DLAMDA(I) if it is 1; this makes the subsequent
       subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
       occurs. On binary machines with a guard digit (almost all
       machines) it does not change DLAMDA(I) at all. On hexadecimal
       and decimal machines with a guard digit, it slightly
       changes the bottommost bits of DLAMDA(I). It does not account
       for hexadecimal or decimal machines without guard digits
       (we know of none). We use a subroutine call to compute
       2*DLAMBDA(I) to prevent optimizing compilers from eliminating
       this code.
*/

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = slamc3_(&dlamda[i__], &dlamda[i__]) - dlamda[i__];
/* L10: */
    }

    i__1 = *kstop;
    for (j = *kstart; j <= i__1; ++j) {
	slaed4_(k, &j, &dlamda[1], &w[1], &q[j * q_dim1 + 1], rho, &d__[j],
		info);

/*        If the zero finder fails, the computation is terminated. */

	if (*info != 0) {
	    goto L120;
	}
/* L20: */
    }

    if (*k == 1 || *k == 2) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = *k;
	    for (j = 1; j <= i__2; ++j) {
		s[j + i__ * s_dim1] = q[j + i__ * q_dim1];
/* L30: */
	    }
/* L40: */
	}
	goto L120;
    }

/*     Compute updated W. */

    scopy_(k, &w[1], &c__1, &s[s_offset], &c__1);

/*     Initialize W(I) = Q(I,I) */

    i__1 = *ldq + 1;
    scopy_(k, &q[q_offset], &i__1, &w[1], &c__1);
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L50: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L60: */
	}
/* L70: */
    }
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__1 = sqrt(-w[i__]);
	w[i__] = r_sign(&r__1, &s[i__ + s_dim1]);
/* L80: */
    }

/*     Compute eigenvectors of the modified rank-1 modification. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    q[i__ + j * q_dim1] = w[i__] / q[i__ + j * q_dim1];
/* L90: */
	}
	temp = snrm2_(k, &q[j * q_dim1 + 1], &c__1);
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s[i__ + j * s_dim1] = q[i__ + j * q_dim1] / temp;
/* L100: */
	}
/* L110: */
    }

L120:
    return 0;

/*     End of SLAED9 */

} /* slaed9_ */

/* Subroutine */ int slaeda_(integer *n, integer *tlvls, integer *curlvl,
	integer *curpbm, integer *prmptr, integer *perm, integer *givptr,
	integer *givcol, real *givnum, real *q, integer *qptr, real *z__,
	real *ztemp, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, k, mid, ptr, curr;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *);
    static integer bsiz1, bsiz2, psiz1, psiz2, zptr1;
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *), scopy_(integer *, real *, integer *, real *, integer *),
	    xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLAEDA computes the Z vector corresponding to the merge step in the
    CURLVLth step of the merge process with TLVLS steps for the CURPBMth
    problem.

    Arguments
    =========

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    TLVLS  (input) INTEGER
           The total number of merging levels in the overall divide and
           conquer tree.

    CURLVL (input) INTEGER
           The current level in the overall merge routine,
           0 <= curlvl <= tlvls.

    CURPBM (input) INTEGER
           The current problem in the current level in the overall
           merge routine (counting from upper left to lower right).

    PRMPTR (input) INTEGER array, dimension (N lg N)
           Contains a list of pointers which indicate where in PERM a
           level's permutation is stored.  PRMPTR(i+1) - PRMPTR(i)
           indicates the size of the permutation and incidentally the
           size of the full, non-deflated problem.

    PERM   (input) INTEGER array, dimension (N lg N)
           Contains the permutations (from deflation and sorting) to be
           applied to each eigenblock.

    GIVPTR (input) INTEGER array, dimension (N lg N)
           Contains a list of pointers which indicate where in GIVCOL a
           level's Givens rotations are stored.  GIVPTR(i+1) - GIVPTR(i)
           indicates the number of Givens rotations.

    GIVCOL (input) INTEGER array, dimension (2, N lg N)
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation.

    GIVNUM (input) REAL array, dimension (2, N lg N)
           Each number indicates the S value to be used in the
           corresponding Givens rotation.

    Q      (input) REAL array, dimension (N**2)
           Contains the square eigenblocks from previous levels, the
           starting positions for blocks are given by QPTR.

    QPTR   (input) INTEGER array, dimension (N+2)
           Contains a list of pointers which indicate where in Q an
           eigenblock is stored.  SQRT( QPTR(i+1) - QPTR(i) ) indicates
           the size of the block.

    Z      (output) REAL array, dimension (N)
           On output this vector contains the updating vector (the last
           row of the first sub-eigenvector matrix and the first row of
           the second sub-eigenvector matrix).

    ZTEMP  (workspace) REAL array, dimension (N)

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --ztemp;
    --z__;
    --qptr;
    --q;
    givnum -= 3;
    givcol -= 3;
    --givptr;
    --perm;
    --prmptr;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -1;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAEDA", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine location of first number in second half. */

    mid = *n / 2 + 1;

/*     Gather last/first rows of appropriate eigenblocks into center of Z */

    ptr = 1;

/*
       Determine location of lowest level subproblem in the full storage
       scheme
*/

    i__1 = *curlvl - 1;
    curr = ptr + *curpbm * pow_ii(&c__2, curlvl) + pow_ii(&c__2, &i__1) - 1;

/*
       Determine size of these matrices.  We add HALF to the value of
       the SQRT in case the machine underestimates one of these square
       roots.
*/

    bsiz1 = (integer) (sqrt((real) (qptr[curr + 1] - qptr[curr])) + .5f);
    bsiz2 = (integer) (sqrt((real) (qptr[curr + 2] - qptr[curr + 1])) + .5f);
    i__1 = mid - bsiz1 - 1;
    for (k = 1; k <= i__1; ++k) {
	z__[k] = 0.f;
/* L10: */
    }
    scopy_(&bsiz1, &q[qptr[curr] + bsiz1 - 1], &bsiz1, &z__[mid - bsiz1], &
	    c__1);
    scopy_(&bsiz2, &q[qptr[curr + 1]], &bsiz2, &z__[mid], &c__1);
    i__1 = *n;
    for (k = mid + bsiz2; k <= i__1; ++k) {
	z__[k] = 0.f;
/* L20: */
    }

/*
       Loop thru remaining levels 1 -> CURLVL applying the Givens
       rotations and permutation and then multiplying the center matrices
       against the current Z.
*/

    ptr = pow_ii(&c__2, tlvls) + 1;
    i__1 = *curlvl - 1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *curlvl - k;
	i__3 = *curlvl - k - 1;
	curr = ptr + *curpbm * pow_ii(&c__2, &i__2) + pow_ii(&c__2, &i__3) -
		1;
	psiz1 = prmptr[curr + 1] - prmptr[curr];
	psiz2 = prmptr[curr + 2] - prmptr[curr + 1];
	zptr1 = mid - psiz1;

/*       Apply Givens at CURR and CURR+1 */

	i__2 = givptr[curr + 1] - 1;
	for (i__ = givptr[curr]; i__ <= i__2; ++i__) {
	    srot_(&c__1, &z__[zptr1 + givcol[(i__ << 1) + 1] - 1], &c__1, &
		    z__[zptr1 + givcol[(i__ << 1) + 2] - 1], &c__1, &givnum[(
		    i__ << 1) + 1], &givnum[(i__ << 1) + 2]);
/* L30: */
	}
	i__2 = givptr[curr + 2] - 1;
	for (i__ = givptr[curr + 1]; i__ <= i__2; ++i__) {
	    srot_(&c__1, &z__[mid - 1 + givcol[(i__ << 1) + 1]], &c__1, &z__[
		    mid - 1 + givcol[(i__ << 1) + 2]], &c__1, &givnum[(i__ <<
		    1) + 1], &givnum[(i__ << 1) + 2]);
/* L40: */
	}
	psiz1 = prmptr[curr + 1] - prmptr[curr];
	psiz2 = prmptr[curr + 2] - prmptr[curr + 1];
	i__2 = psiz1 - 1;
	for (i__ = 0; i__ <= i__2; ++i__) {
	    ztemp[i__ + 1] = z__[zptr1 + perm[prmptr[curr] + i__] - 1];
/* L50: */
	}
	i__2 = psiz2 - 1;
	for (i__ = 0; i__ <= i__2; ++i__) {
	    ztemp[psiz1 + i__ + 1] = z__[mid + perm[prmptr[curr + 1] + i__] -
		    1];
/* L60: */
	}

/*
          Multiply Blocks at CURR and CURR+1

          Determine size of these matrices.  We add HALF to the value of
          the SQRT in case the machine underestimates one of these
          square roots.
*/

	bsiz1 = (integer) (sqrt((real) (qptr[curr + 1] - qptr[curr])) + .5f);
	bsiz2 = (integer) (sqrt((real) (qptr[curr + 2] - qptr[curr + 1])) +
		.5f);
	if (bsiz1 > 0) {
	    sgemv_("T", &bsiz1, &bsiz1, &c_b15, &q[qptr[curr]], &bsiz1, &
		    ztemp[1], &c__1, &c_b29, &z__[zptr1], &c__1);
	}
	i__2 = psiz1 - bsiz1;
	scopy_(&i__2, &ztemp[bsiz1 + 1], &c__1, &z__[zptr1 + bsiz1], &c__1);
	if (bsiz2 > 0) {
	    sgemv_("T", &bsiz2, &bsiz2, &c_b15, &q[qptr[curr + 1]], &bsiz2, &
		    ztemp[psiz1 + 1], &c__1, &c_b29, &z__[mid], &c__1);
	}
	i__2 = psiz2 - bsiz2;
	scopy_(&i__2, &ztemp[psiz1 + bsiz2 + 1], &c__1, &z__[mid + bsiz2], &
		c__1);

	i__2 = *tlvls - k;
	ptr += pow_ii(&c__2, &i__2);
/* L70: */
    }

    return 0;

/*     End of SLAEDA */

} /* slaeda_ */

/* Subroutine */ int slaev2_(real *a, real *b, real *c__, real *rt1, real *
	rt2, real *cs1, real *sn1)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real ab, df, cs, ct, tb, sm, tn, rt, adf, acs;
    static integer sgn1, sgn2;
    static real acmn, acmx;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
       [  A   B  ]
       [  B   C  ].
    On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
    eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
    eigenvector for RT1, giving the decomposition

       [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
       [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].

    Arguments
    =========

    A       (input) REAL
            The (1,1) element of the 2-by-2 matrix.

    B       (input) REAL
            The (1,2) element and the conjugate of the (2,1) element of
            the 2-by-2 matrix.

    C       (input) REAL
            The (2,2) element of the 2-by-2 matrix.

    RT1     (output) REAL
            The eigenvalue of larger absolute value.

    RT2     (output) REAL
            The eigenvalue of smaller absolute value.

    CS1     (output) REAL
    SN1     (output) REAL
            The vector (CS1, SN1) is a unit right eigenvector for RT1.

    Further Details
    ===============

    RT1 is accurate to a few ulps barring over/underflow.

    RT2 may be inaccurate if there is massive cancellation in the
    determinant A*C-B*B; higher precision or correctly rounded or
    correctly truncated arithmetic would be needed to compute RT2
    accurately in all cases.

    CS1 and SN1 are accurate to a few ulps barring over/underflow.

    Overflow is possible only if RT1 is within a factor of 5 of overflow.
    Underflow is harmless if the input data is 0 or exceeds
       underflow_threshold / macheps.

   =====================================================================


       Compute the eigenvalues
*/

    sm = *a + *c__;
    df = *a - *c__;
    adf = dabs(df);
    tb = *b + *b;
    ab = dabs(tb);
    if (dabs(*a) > dabs(*c__)) {
	acmx = *a;
	acmn = *c__;
    } else {
	acmx = *c__;
	acmn = *a;
    }
    if (adf > ab) {
/* Computing 2nd power */
	r__1 = ab / adf;
	rt = adf * sqrt(r__1 * r__1 + 1.f);
    } else if (adf < ab) {
/* Computing 2nd power */
	r__1 = adf / ab;
	rt = ab * sqrt(r__1 * r__1 + 1.f);
    } else {

/*        Includes case AB=ADF=0 */

	rt = ab * sqrt(2.f);
    }
    if (sm < 0.f) {
	*rt1 = (sm - rt) * .5f;
	sgn1 = -1;

/*
          Order of execution important.
          To get fully accurate smaller eigenvalue,
          next line needs to be executed in higher precision.
*/

	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else if (sm > 0.f) {
	*rt1 = (sm + rt) * .5f;
	sgn1 = 1;

/*
          Order of execution important.
          To get fully accurate smaller eigenvalue,
          next line needs to be executed in higher precision.
*/

	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else {

/*        Includes case RT1 = RT2 = 0 */

	*rt1 = rt * .5f;
	*rt2 = rt * -.5f;
	sgn1 = 1;
    }

/*     Compute the eigenvector */

    if (df >= 0.f) {
	cs = df + rt;
	sgn2 = 1;
    } else {
	cs = df - rt;
	sgn2 = -1;
    }
    acs = dabs(cs);
    if (acs > ab) {
	ct = -tb / cs;
	*sn1 = 1.f / sqrt(ct * ct + 1.f);
	*cs1 = ct * *sn1;
    } else {
	if (ab == 0.f) {
	    *cs1 = 1.f;
	    *sn1 = 0.f;
	} else {
	    tn = -cs / tb;
	    *cs1 = 1.f / sqrt(tn * tn + 1.f);
	    *sn1 = tn * *cs1;
	}
    }
    if (sgn1 == sgn2) {
	tn = *cs1;
	*cs1 = -(*sn1);
	*sn1 = tn;
    }
    return 0;

/*     End of SLAEV2 */

} /* slaev2_ */

/* Subroutine */ int slahqr_(logical *wantt, logical *wantz, integer *n,
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *
	info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static integer i__, j, k, l, m;
    static real s, v[3];
    static integer i1, i2;
    static real t1, t2, t3, v1, v2, v3, h00, h10, h11, h12, h21, h22, h33,
	    h44;
    static integer nh;
    static real cs;
    static integer nr;
    static real sn;
    static integer nz;
    static real ave, h33s, h44s;
    static integer itn, its;
    static real ulp, sum, tst1, h43h34, disc, unfl, ovfl, work[1];
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *), scopy_(integer *, real *, integer *,
	    real *, integer *), slanv2_(real *, real *, real *, real *, real *
	    , real *, real *, real *, real *, real *), slabad_(real *, real *)
	    ;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int slarfg_(integer *, real *, real *, integer *,
	    real *);
    extern doublereal slanhs_(char *, integer *, real *, integer *, real *);
    static real smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLAHQR is an auxiliary routine called by SHSEQR to update the
    eigenvalues and Schur decomposition already computed by SHSEQR, by
    dealing with the Hessenberg submatrix in rows and columns ILO to IHI.

    Arguments
    =========

    WANTT   (input) LOGICAL
            = .TRUE. : the full Schur form T is required;
            = .FALSE.: only eigenvalues are required.

    WANTZ   (input) LOGICAL
            = .TRUE. : the matrix of Schur vectors Z is required;
            = .FALSE.: Schur vectors are not required.

    N       (input) INTEGER
            The order of the matrix H.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that H is already upper quasi-triangular in
            rows and columns IHI+1:N, and that H(ILO,ILO-1) = 0 (unless
            ILO = 1). SLAHQR works primarily with the Hessenberg
            submatrix in rows and columns ILO to IHI, but applies
            transformations to all of H if WANTT is .TRUE..
            1 <= ILO <= max(1,IHI); IHI <= N.

    H       (input/output) REAL array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if WANTT is .TRUE., H is upper quasi-triangular in
            rows and columns ILO:IHI, with any 2-by-2 diagonal blocks in
            standard form. If WANTT is .FALSE., the contents of H are
            unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    WR      (output) REAL array, dimension (N)
    WI      (output) REAL array, dimension (N)
            The real and imaginary parts, respectively, of the computed
            eigenvalues ILO to IHI are stored in the corresponding
            elements of WR and WI. If two eigenvalues are computed as a
            complex conjugate pair, they are stored in consecutive
            elements of WR and WI, say the i-th and (i+1)th, with
            WI(i) > 0 and WI(i+1) < 0. If WANTT is .TRUE., the
            eigenvalues are stored in the same order as on the diagonal
            of the Schur form returned in H, with WR(i) = H(i,i), and, if
            H(i:i+1,i:i+1) is a 2-by-2 diagonal block,
            WI(i) = sqrt(H(i+1,i)*H(i,i+1)) and WI(i+1) = -WI(i).

    ILOZ    (input) INTEGER
    IHIZ    (input) INTEGER
            Specify the rows of Z to which transformations must be
            applied if WANTZ is .TRUE..
            1 <= ILOZ <= ILO; IHI <= IHIZ <= N.

    Z       (input/output) REAL array, dimension (LDZ,N)
            If WANTZ is .TRUE., on entry Z must contain the current
            matrix Z of transformations accumulated by SHSEQR, and on
            exit Z has been updated; transformations are applied only to
            the submatrix Z(ILOZ:IHIZ,ILO:IHI).
            If WANTZ is .FALSE., Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            > 0: SLAHQR failed to compute all the eigenvalues ILO to IHI
                 in a total of 30*(IHI-ILO+1) iterations; if INFO = i,
                 elements i+1:ihi of WR and WI contain those eigenvalues
                 which have been successfully computed.

    Further Details
    ===============

    2-96 Based on modifications by
       David Day, Sandia National Laboratory, USA

    =====================================================================
*/


    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    if (*ilo == *ihi) {
	wr[*ilo] = h__[*ilo + *ilo * h_dim1];
	wi[*ilo] = 0.f;
	return 0;
    }

    nh = *ihi - *ilo + 1;
    nz = *ihiz - *iloz + 1;

/*
       Set machine-dependent constants for the stopping criterion.
       If norm(H) <= sqrt(OVFL), overflow should not occur.
*/

    unfl = slamch_("Safe minimum");
    ovfl = 1.f / unfl;
    slabad_(&unfl, &ovfl);
    ulp = slamch_("Precision");
    smlnum = unfl * (nh / ulp);

/*
       I1 and I2 are the indices of the first row and last column of H
       to which transformations must be applied. If eigenvalues only are
       being computed, I1 and I2 are set inside the main loop.
*/

    if (*wantt) {
	i1 = 1;
	i2 = *n;
    }

/*     ITN is the total number of QR iterations allowed. */

    itn = nh * 30;

/*
       The main loop begins here. I is the loop index and decreases from
       IHI to ILO in steps of 1 or 2. Each iteration of the loop works
       with the active submatrix in rows and columns L to I.
       Eigenvalues I+1 to IHI have already converged. Either L = ILO or
       H(L,L-1) is negligible so that the matrix splits.
*/

    i__ = *ihi;
L10:
    l = *ilo;
    if (i__ < *ilo) {
	goto L150;
    }

/*
       Perform QR iterations on rows and columns ILO to I until a
       submatrix of order 1 or 2 splits off at the bottom because a
       subdiagonal element has become negligible.
*/

    i__1 = itn;
    for (its = 0; its <= i__1; ++its) {

/*        Look for a single small subdiagonal element. */

	i__2 = l + 1;
	for (k = i__; k >= i__2; --k) {
	    tst1 = (r__1 = h__[k - 1 + (k - 1) * h_dim1], dabs(r__1)) + (r__2
		    = h__[k + k * h_dim1], dabs(r__2));
	    if (tst1 == 0.f) {
		i__3 = i__ - l + 1;
		tst1 = slanhs_("1", &i__3, &h__[l + l * h_dim1], ldh, work);
	    }
/* Computing MAX */
	    r__2 = ulp * tst1;
	    if ((r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)) <= dmax(r__2,
		    smlnum)) {
		goto L30;
	    }
/* L20: */
	}
L30:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible */

	    h__[l + (l - 1) * h_dim1] = 0.f;
	}

/*        Exit from loop if a submatrix of order 1 or 2 has split off. */

	if (l >= i__ - 1) {
	    goto L140;
	}

/*
          Now the active submatrix is in rows and columns L to I. If
          eigenvalues only are being computed, only the active submatrix
          need be transformed.
*/

	if (! (*wantt)) {
	    i1 = l;
	    i2 = i__;
	}

	if (its == 10 || its == 20) {

/*           Exceptional shift. */

	    s = (r__1 = h__[i__ + (i__ - 1) * h_dim1], dabs(r__1)) + (r__2 =
		    h__[i__ - 1 + (i__ - 2) * h_dim1], dabs(r__2));
	    h44 = s * .75f + h__[i__ + i__ * h_dim1];
	    h33 = h44;
	    h43h34 = s * -.4375f * s;
	} else {

/*
             Prepare to use Francis' double shift
             (i.e. 2nd degree generalized Rayleigh quotient)
*/

	    h44 = h__[i__ + i__ * h_dim1];
	    h33 = h__[i__ - 1 + (i__ - 1) * h_dim1];
	    h43h34 = h__[i__ + (i__ - 1) * h_dim1] * h__[i__ - 1 + i__ *
		    h_dim1];
	    s = h__[i__ - 1 + (i__ - 2) * h_dim1] * h__[i__ - 1 + (i__ - 2) *
		    h_dim1];
	    disc = (h33 - h44) * .5f;
	    disc = disc * disc + h43h34;
	    if (disc > 0.f) {

/*              Real roots: use Wilkinson's shift twice */

		disc = sqrt(disc);
		ave = (h33 + h44) * .5f;
		if (dabs(h33) - dabs(h44) > 0.f) {
		    h33 = h33 * h44 - h43h34;
		    h44 = h33 / (r_sign(&disc, &ave) + ave);
		} else {
		    h44 = r_sign(&disc, &ave) + ave;
		}
		h33 = h44;
		h43h34 = 0.f;
	    }
	}

/*        Look for two consecutive small subdiagonal elements. */

	i__2 = l;
	for (m = i__ - 2; m >= i__2; --m) {
/*
             Determine the effect of starting the double-shift QR
             iteration at row M, and see if this would make H(M,M-1)
             negligible.
*/

	    h11 = h__[m + m * h_dim1];
	    h22 = h__[m + 1 + (m + 1) * h_dim1];
	    h21 = h__[m + 1 + m * h_dim1];
	    h12 = h__[m + (m + 1) * h_dim1];
	    h44s = h44 - h11;
	    h33s = h33 - h11;
	    v1 = (h33s * h44s - h43h34) / h21 + h12;
	    v2 = h22 - h11 - h33s - h44s;
	    v3 = h__[m + 2 + (m + 1) * h_dim1];
	    s = dabs(v1) + dabs(v2) + dabs(v3);
	    v1 /= s;
	    v2 /= s;
	    v3 /= s;
	    v[0] = v1;
	    v[1] = v2;
	    v[2] = v3;
	    if (m == l) {
		goto L50;
	    }
	    h00 = h__[m - 1 + (m - 1) * h_dim1];
	    h10 = h__[m + (m - 1) * h_dim1];
	    tst1 = dabs(v1) * (dabs(h00) + dabs(h11) + dabs(h22));
	    if (dabs(h10) * (dabs(v2) + dabs(v3)) <= ulp * tst1) {
		goto L50;
	    }
/* L40: */
	}
L50:

/*        Double-shift QR step */

	i__2 = i__ - 1;
	for (k = m; k <= i__2; ++k) {

/*
             The first iteration of this loop determines a reflection G
             from the vector V and applies it from left and right to H,
             thus creating a nonzero bulge below the subdiagonal.

             Each subsequent iteration determines a reflection G to
             restore the Hessenberg form in the (K-1)th column, and thus
             chases the bulge one step toward the bottom of the active
             submatrix. NR is the order of G.

   Computing MIN
*/
	    i__3 = 3, i__4 = i__ - k + 1;
	    nr = min(i__3,i__4);
	    if (k > m) {
		scopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    slarfg_(&nr, v, &v[1], &c__1, &t1);
	    if (k > m) {
		h__[k + (k - 1) * h_dim1] = v[0];
		h__[k + 1 + (k - 1) * h_dim1] = 0.f;
		if (k < i__ - 1) {
		    h__[k + 2 + (k - 1) * h_dim1] = 0.f;
		}
	    } else if (m > l) {
		h__[k + (k - 1) * h_dim1] = -h__[k + (k - 1) * h_dim1];
	    }
	    v2 = v[1];
	    t2 = t1 * v2;
	    if (nr == 3) {
		v3 = v[2];
		t3 = t1 * v3;

/*
                Apply G from the left to transform the rows of the matrix
                in columns K to I2.
*/

		i__3 = i2;
		for (j = k; j <= i__3; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1]
			    + v3 * h__[k + 2 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
		    h__[k + 2 + j * h_dim1] -= sum * t3;
/* L60: */
		}

/*
                Apply G from the right to transform the columns of the
                matrix in rows I1 to min(K+3,I).

   Computing MIN
*/
		i__4 = k + 3;
		i__3 = min(i__4,i__);
		for (j = i1; j <= i__3; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			     + v3 * h__[j + (k + 2) * h_dim1];
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
		    h__[j + (k + 2) * h_dim1] -= sum * t3;
/* L70: */
		}

		if (*wantz) {

/*                 Accumulate transformations in the matrix Z */

		    i__3 = *ihiz;
		    for (j = *iloz; j <= i__3; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) *
				z_dim1] + v3 * z__[j + (k + 2) * z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
			z__[j + (k + 2) * z_dim1] -= sum * t3;
/* L80: */
		    }
		}
	    } else if (nr == 2) {

/*
                Apply G from the left to transform the rows of the matrix
                in columns K to I2.
*/

		i__3 = i2;
		for (j = k; j <= i__3; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
/* L90: */
		}

/*
                Apply G from the right to transform the columns of the
                matrix in rows I1 to min(K+3,I).
*/

		i__3 = i__;
		for (j = i1; j <= i__3; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			    ;
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
/* L100: */
		}

		if (*wantz) {

/*                 Accumulate transformations in the matrix Z */

		    i__3 = *ihiz;
		    for (j = *iloz; j <= i__3; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) *
				z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
/* L110: */
		    }
		}
	    }
/* L120: */
	}

/* L130: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L140:

    if (l == i__) {

/*        H(I,I-1) is negligible: one eigenvalue has converged. */

	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.f;
    } else if (l == i__ - 1) {

/*
          H(I-1,I-2) is negligible: a pair of eigenvalues have converged.

          Transform the 2-by-2 submatrix to standard Schur form,
          and compute and store the eigenvalues.
*/

	slanv2_(&h__[i__ - 1 + (i__ - 1) * h_dim1], &h__[i__ - 1 + i__ *
		h_dim1], &h__[i__ + (i__ - 1) * h_dim1], &h__[i__ + i__ *
		h_dim1], &wr[i__ - 1], &wi[i__ - 1], &wr[i__], &wi[i__], &cs,
		&sn);

	if (*wantt) {

/*           Apply the transformation to the rest of H. */

	    if (i2 > i__) {
		i__1 = i2 - i__;
		srot_(&i__1, &h__[i__ - 1 + (i__ + 1) * h_dim1], ldh, &h__[
			i__ + (i__ + 1) * h_dim1], ldh, &cs, &sn);
	    }
	    i__1 = i__ - i1 - 1;
	    srot_(&i__1, &h__[i1 + (i__ - 1) * h_dim1], &c__1, &h__[i1 + i__ *
		     h_dim1], &c__1, &cs, &sn);
	}
	if (*wantz) {

/*           Apply the transformation to Z. */

	    srot_(&nz, &z__[*iloz + (i__ - 1) * z_dim1], &c__1, &z__[*iloz +
		    i__ * z_dim1], &c__1, &cs, &sn);
	}
    }

/*
       Decrement number of remaining iterations, and return to start of
       the main loop with new value of I.
*/

    itn -= its;
    i__ = l - 1;
    goto L10;

L150:
    return 0;

/*     End of SLAHQR */

} /* slahqr_ */

/* Subroutine */ int slahrd_(integer *n, integer *k, integer *nb, real *a,
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    real r__1;

    /* Local variables */
    static integer i__;
    static real ei;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sgemv_(char *, integer *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *), scopy_(
	    integer *, real *, integer *, real *, integer *), saxpy_(integer *
	    , real *, real *, integer *, real *, integer *), strmv_(char *,
	    char *, char *, integer *, real *, integer *, real *, integer *), slarfg_(integer *, real *, real *,
	    integer *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLAHRD reduces the first NB columns of a real general n-by-(n-k+1)
    matrix A so that elements below the k-th subdiagonal are zero. The
    reduction is performed by an orthogonal similarity transformation
    Q' * A * Q. The routine returns the matrices V and T which determine
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V * T.

    This is an auxiliary routine called by SGEHRD.

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.

    K       (input) INTEGER
            The offset for the reduction. Elements below the k-th
            subdiagonal in the first NB columns are reduced to zero.

    NB      (input) INTEGER
            The number of columns to be reduced.

    A       (input/output) REAL array, dimension (LDA,N-K+1)
            On entry, the n-by-(n-k+1) general matrix A.
            On exit, the elements on and above the k-th subdiagonal in
            the first NB columns are overwritten with the corresponding
            elements of the reduced matrix; the elements below the k-th
            subdiagonal, with the array TAU, represent the matrix Q as a
            product of elementary reflectors. The other columns of A are
            unchanged. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) REAL array, dimension (NB)
            The scalar factors of the elementary reflectors. See Further
            Details.

    T       (output) REAL array, dimension (LDT,NB)
            The upper triangular matrix T.

    LDT     (input) INTEGER
            The leading dimension of the array T.  LDT >= NB.

    Y       (output) REAL array, dimension (LDY,NB)
            The n-by-nb matrix Y.

    LDY     (input) INTEGER
            The leading dimension of the array Y. LDY >= N.

    Further Details
    ===============

    The matrix Q is represented as a product of nb elementary reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in
    A(i+k+1:n,i), and tau in TAU(i).

    The elements of the vectors v together form the (n-k+1)-by-nb matrix
    V which is needed, with T and Y, to apply the transformation to the
    unreduced part of the matrix, using an update of the form:
    A := (I - V*T*V') * (A - Y*V').

    The contents of A on exit are illustrated by the following example
    with n = 7, k = 3 and nb = 2:

       ( a   h   a   a   a )
       ( a   h   a   a   a )
       ( a   h   a   a   a )
       ( h   h   a   a   a )
       ( v1  h   a   a   a )
       ( v1  v2  a   a   a )
       ( v1  v2  a   a   a )

    where a denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    =====================================================================


       Quick return if possible
*/

    /* Parameter adjustments */
    --tau;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*n <= 1) {
	return 0;
    }

    i__1 = *nb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ > 1) {

/*
             Update A(1:n,i)

             Compute i-th column of A - Y * V'
*/

	    i__2 = i__ - 1;
	    sgemv_("No transpose", n, &i__2, &c_b151, &y[y_offset], ldy, &a[*
		    k + i__ - 1 + a_dim1], lda, &c_b15, &a[i__ * a_dim1 + 1],
		    &c__1);

/*
             Apply I - V * T' * V' to this column (call it b) from the
             left, using the last column of T as workspace

             Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
                      ( V2 )             ( b2 )

             where V1 is unit lower triangular

             w := V1' * b1
*/

	    i__2 = i__ - 1;
	    scopy_(&i__2, &a[*k + 1 + i__ * a_dim1], &c__1, &t[*nb * t_dim1 +
		    1], &c__1);
	    i__2 = i__ - 1;
	    strmv_("Lower", "Transpose", "Unit", &i__2, &a[*k + 1 + a_dim1],
		    lda, &t[*nb * t_dim1 + 1], &c__1);

/*           w := w + V2'*b2 */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[*k + i__ + a_dim1],
		    lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b15, &t[*nb *
		    t_dim1 + 1], &c__1);

/*           w := T'*w */

	    i__2 = i__ - 1;
	    strmv_("Upper", "Transpose", "Non-unit", &i__2, &t[t_offset], ldt,
		     &t[*nb * t_dim1 + 1], &c__1);

/*           b2 := b2 - V2*w */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[*k + i__ +
		    a_dim1], lda, &t[*nb * t_dim1 + 1], &c__1, &c_b15, &a[*k
		    + i__ + i__ * a_dim1], &c__1);

/*           b1 := b1 - V1*w */

	    i__2 = i__ - 1;
	    strmv_("Lower", "No transpose", "Unit", &i__2, &a[*k + 1 + a_dim1]
		    , lda, &t[*nb * t_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    saxpy_(&i__2, &c_b151, &t[*nb * t_dim1 + 1], &c__1, &a[*k + 1 +
		    i__ * a_dim1], &c__1);

	    a[*k + i__ - 1 + (i__ - 1) * a_dim1] = ei;
	}

/*
          Generate the elementary reflector H(i) to annihilate
          A(k+i+1:n,i)
*/

	i__2 = *n - *k - i__ + 1;
/* Computing MIN */
	i__3 = *k + i__ + 1;
	slarfg_(&i__2, &a[*k + i__ + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		a_dim1], &c__1, &tau[i__]);
	ei = a[*k + i__ + i__ * a_dim1];
	a[*k + i__ + i__ * a_dim1] = 1.f;

/*        Compute  Y(1:n,i) */

	i__2 = *n - *k - i__ + 1;
	sgemv_("No transpose", n, &i__2, &c_b15, &a[(i__ + 1) * a_dim1 + 1],
		lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b29, &y[i__ *
		y_dim1 + 1], &c__1);
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[*k + i__ + a_dim1], lda,
		&a[*k + i__ + i__ * a_dim1], &c__1, &c_b29, &t[i__ * t_dim1 +
		1], &c__1);
	i__2 = i__ - 1;
	sgemv_("No transpose", n, &i__2, &c_b151, &y[y_offset], ldy, &t[i__ *
		t_dim1 + 1], &c__1, &c_b15, &y[i__ * y_dim1 + 1], &c__1);
	sscal_(n, &tau[i__], &y[i__ * y_dim1 + 1], &c__1);

/*        Compute T(1:i,i) */

	i__2 = i__ - 1;
	r__1 = -tau[i__];
	sscal_(&i__2, &r__1, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	strmv_("Upper", "No transpose", "Non-unit", &i__2, &t[t_offset], ldt,
		&t[i__ * t_dim1 + 1], &c__1)
		;
	t[i__ + i__ * t_dim1] = tau[i__];

/* L10: */
    }
    a[*k + *nb + *nb * a_dim1] = ei;

    return 0;

/*     End of SLAHRD */

} /* slahrd_ */

/* Subroutine */ int slaln2_(logical *ltrans, integer *na, integer *nw, real *
	smin, real *ca, real *a, integer *lda, real *d1, real *d2, real *b,
	integer *ldb, real *wr, real *wi, real *x, integer *ldx, real *scale,
	real *xnorm, integer *info)
{
    /* Initialized data */

    static logical cswap[4] = { FALSE_,FALSE_,TRUE_,TRUE_ };
    static logical rswap[4] = { FALSE_,TRUE_,FALSE_,TRUE_ };
    static integer ipivot[16]	/* was [4][4] */ = { 1,2,3,4,2,1,4,3,3,4,1,2,
	    4,3,2,1 };

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, x_dim1, x_offset;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    static real equiv_0[4], equiv_1[4];

    /* Local variables */
    static integer j;
#define ci (equiv_0)
#define cr (equiv_1)
    static real bi1, bi2, br1, br2, xi1, xi2, xr1, xr2, ci21, ci22, cr21,
	    cr22, li21, csi, ui11, lr21, ui12, ui22;
#define civ (equiv_0)
    static real csr, ur11, ur12, ur22;
#define crv (equiv_1)
    static real bbnd, cmax, ui11r, ui12s, temp, ur11r, ur12s, u22abs;
    static integer icmax;
    static real bnorm, cnorm, smini;
    extern doublereal slamch_(char *);
    static real bignum;
    extern /* Subroutine */ int sladiv_(real *, real *, real *, real *, real *
	    , real *);
    static real smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLALN2 solves a system of the form  (ca A - w D ) X = s B
    or (ca A' - w D) X = s B   with possible scaling ("s") and
    perturbation of A.  (A' means A-transpose.)

    A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
    real diagonal matrix, w is a real or complex value, and X and B are
    NA x 1 matrices -- real if w is real, complex if w is complex.  NA
    may be 1 or 2.

    If w is complex, X and B are represented as NA x 2 matrices,
    the first column of each being the real part and the second
    being the imaginary part.

    "s" is a scaling factor (.LE. 1), computed by SLALN2, which is
    so chosen that X can be computed without overflow.  X is further
    scaled if necessary to assure that norm(ca A - w D)*norm(X) is less
    than overflow.

    If both singular values of (ca A - w D) are less than SMIN,
    SMIN*identity will be used instead of (ca A - w D).  If only one
    singular value is less than SMIN, one element of (ca A - w D) will be
    perturbed enough to make the smallest singular value roughly SMIN.
    If both singular values are at least SMIN, (ca A - w D) will not be
    perturbed.  In any case, the perturbation will be at most some small
    multiple of max( SMIN, ulp*norm(ca A - w D) ).  The singular values
    are computed by infinity-norm approximations, and thus will only be
    correct to a factor of 2 or so.

    Note: all input quantities are assumed to be smaller than overflow
    by a reasonable factor.  (See BIGNUM.)

    Arguments
    ==========

    LTRANS  (input) LOGICAL
            =.TRUE.:  A-transpose will be used.
            =.FALSE.: A will be used (not transposed.)

    NA      (input) INTEGER
            The size of the matrix A.  It may (only) be 1 or 2.

    NW      (input) INTEGER
            1 if "w" is real, 2 if "w" is complex.  It may only be 1
            or 2.

    SMIN    (input) REAL
            The desired lower bound on the singular values of A.  This
            should be a safe distance away from underflow or overflow,
            say, between (underflow/machine precision) and  (machine
            precision * overflow ).  (See BIGNUM and ULP.)

    CA      (input) REAL
            The coefficient c, which A is multiplied by.

    A       (input) REAL array, dimension (LDA,NA)
            The NA x NA matrix A.

    LDA     (input) INTEGER
            The leading dimension of A.  It must be at least NA.

    D1      (input) REAL
            The 1,1 element in the diagonal matrix D.

    D2      (input) REAL
            The 2,2 element in the diagonal matrix D.  Not used if NW=1.

    B       (input) REAL array, dimension (LDB,NW)
            The NA x NW matrix B (right-hand side).  If NW=2 ("w" is
            complex), column 1 contains the real part of B and column 2
            contains the imaginary part.

    LDB     (input) INTEGER
            The leading dimension of B.  It must be at least NA.

    WR      (input) REAL
            The real part of the scalar "w".

    WI      (input) REAL
            The imaginary part of the scalar "w".  Not used if NW=1.

    X       (output) REAL array, dimension (LDX,NW)
            The NA x NW matrix X (unknowns), as computed by SLALN2.
            If NW=2 ("w" is complex), on exit, column 1 will contain
            the real part of X and column 2 will contain the imaginary
            part.

    LDX     (input) INTEGER
            The leading dimension of X.  It must be at least NA.

    SCALE   (output) REAL
            The scale factor that B must be multiplied by to insure
            that overflow does not occur when computing X.  Thus,
            (ca A - w D) X  will be SCALE*B, not B (ignoring
            perturbations of A.)  It will be at most 1.

    XNORM   (output) REAL
            The infinity-norm of X, when X is regarded as an NA x NW
            real matrix.

    INFO    (output) INTEGER
            An error flag.  It will be set to zero if no error occurs,
            a negative number if an argument is in error, or a positive
            number if  ca A - w D  had to be perturbed.
            The possible values are:
            = 0: No error occurred, and (ca A - w D) did not have to be
                   perturbed.
            = 1: (ca A - w D) had to be perturbed to make its smallest
                 (or only) singular value greater than SMIN.
            NOTE: In the interests of speed, this routine does not
                  check the inputs for errors.

   =====================================================================
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;

    /* Function Body */

/*     Compute BIGNUM */

    smlnum = 2.f * slamch_("Safe minimum");
    bignum = 1.f / smlnum;
    smini = dmax(*smin,smlnum);

/*     Don't check for input errors */

    *info = 0;

/*     Standard Initializations */

    *scale = 1.f;

    if (*na == 1) {

/*        1 x 1  (i.e., scalar) system   C X = B */

	if (*nw == 1) {

/*
             Real 1x1 system.

             C = ca A - w D
*/

	    csr = *ca * a[a_dim1 + 1] - *wr * *d1;
	    cnorm = dabs(csr);

/*           If | C | < SMINI, use C = SMINI */

	    if (cnorm < smini) {
		csr = smini;
		cnorm = smini;
		*info = 1;
	    }

/*           Check scaling for  X = B / C */

	    bnorm = (r__1 = b[b_dim1 + 1], dabs(r__1));
	    if (cnorm < 1.f && bnorm > 1.f) {
		if (bnorm > bignum * cnorm) {
		    *scale = 1.f / bnorm;
		}
	    }

/*           Compute X */

	    x[x_dim1 + 1] = b[b_dim1 + 1] * *scale / csr;
	    *xnorm = (r__1 = x[x_dim1 + 1], dabs(r__1));
	} else {

/*
             Complex 1x1 system (w is complex)

             C = ca A - w D
*/

	    csr = *ca * a[a_dim1 + 1] - *wr * *d1;
	    csi = -(*wi) * *d1;
	    cnorm = dabs(csr) + dabs(csi);

/*           If | C | < SMINI, use C = SMINI */

	    if (cnorm < smini) {
		csr = smini;
		csi = 0.f;
		cnorm = smini;
		*info = 1;
	    }

/*           Check scaling for  X = B / C */

	    bnorm = (r__1 = b[b_dim1 + 1], dabs(r__1)) + (r__2 = b[(b_dim1 <<
		    1) + 1], dabs(r__2));
	    if (cnorm < 1.f && bnorm > 1.f) {
		if (bnorm > bignum * cnorm) {
		    *scale = 1.f / bnorm;
		}
	    }

/*           Compute X */

	    r__1 = *scale * b[b_dim1 + 1];
	    r__2 = *scale * b[(b_dim1 << 1) + 1];
	    sladiv_(&r__1, &r__2, &csr, &csi, &x[x_dim1 + 1], &x[(x_dim1 << 1)
		     + 1]);
	    *xnorm = (r__1 = x[x_dim1 + 1], dabs(r__1)) + (r__2 = x[(x_dim1 <<
		     1) + 1], dabs(r__2));
	}

    } else {

/*
          2x2 System

          Compute the real part of  C = ca A - w D  (or  ca A' - w D )
*/

	cr[0] = *ca * a[a_dim1 + 1] - *wr * *d1;
	cr[3] = *ca * a[(a_dim1 << 1) + 2] - *wr * *d2;
	if (*ltrans) {
	    cr[2] = *ca * a[a_dim1 + 2];
	    cr[1] = *ca * a[(a_dim1 << 1) + 1];
	} else {
	    cr[1] = *ca * a[a_dim1 + 2];
	    cr[2] = *ca * a[(a_dim1 << 1) + 1];
	}

	if (*nw == 1) {

/*
             Real 2x2 system  (w is real)

             Find the largest element in C
*/

	    cmax = 0.f;
	    icmax = 0;

	    for (j = 1; j <= 4; ++j) {
		if ((r__1 = crv[j - 1], dabs(r__1)) > cmax) {
		    cmax = (r__1 = crv[j - 1], dabs(r__1));
		    icmax = j;
		}
/* L10: */
	    }

/*           If norm(C) < SMINI, use SMINI*identity. */

	    if (cmax < smini) {
/* Computing MAX */
		r__3 = (r__1 = b[b_dim1 + 1], dabs(r__1)), r__4 = (r__2 = b[
			b_dim1 + 2], dabs(r__2));
		bnorm = dmax(r__3,r__4);
		if (smini < 1.f && bnorm > 1.f) {
		    if (bnorm > bignum * smini) {
			*scale = 1.f / bnorm;
		    }
		}
		temp = *scale / smini;
		x[x_dim1 + 1] = temp * b[b_dim1 + 1];
		x[x_dim1 + 2] = temp * b[b_dim1 + 2];
		*xnorm = temp * bnorm;
		*info = 1;
		return 0;
	    }

/*           Gaussian elimination with complete pivoting. */

	    ur11 = crv[icmax - 1];
	    cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
	    ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
	    cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
	    ur11r = 1.f / ur11;
	    lr21 = ur11r * cr21;
	    ur22 = cr22 - ur12 * lr21;

/*           If smaller pivot < SMINI, use SMINI */

	    if (dabs(ur22) < smini) {
		ur22 = smini;
		*info = 1;
	    }
	    if (rswap[icmax - 1]) {
		br1 = b[b_dim1 + 2];
		br2 = b[b_dim1 + 1];
	    } else {
		br1 = b[b_dim1 + 1];
		br2 = b[b_dim1 + 2];
	    }
	    br2 -= lr21 * br1;
/* Computing MAX */
	    r__2 = (r__1 = br1 * (ur22 * ur11r), dabs(r__1)), r__3 = dabs(br2)
		    ;
	    bbnd = dmax(r__2,r__3);
	    if (bbnd > 1.f && dabs(ur22) < 1.f) {
		if (bbnd >= bignum * dabs(ur22)) {
		    *scale = 1.f / bbnd;
		}
	    }

	    xr2 = br2 * *scale / ur22;
	    xr1 = *scale * br1 * ur11r - xr2 * (ur11r * ur12);
	    if (cswap[icmax - 1]) {
		x[x_dim1 + 1] = xr2;
		x[x_dim1 + 2] = xr1;
	    } else {
		x[x_dim1 + 1] = xr1;
		x[x_dim1 + 2] = xr2;
	    }
/* Computing MAX */
	    r__1 = dabs(xr1), r__2 = dabs(xr2);
	    *xnorm = dmax(r__1,r__2);

/*           Further scaling if  norm(A) norm(X) > overflow */

	    if (*xnorm > 1.f && cmax > 1.f) {
		if (*xnorm > bignum / cmax) {
		    temp = cmax / bignum;
		    x[x_dim1 + 1] = temp * x[x_dim1 + 1];
		    x[x_dim1 + 2] = temp * x[x_dim1 + 2];
		    *xnorm = temp * *xnorm;
		    *scale = temp * *scale;
		}
	    }
	} else {

/*
             Complex 2x2 system  (w is complex)

             Find the largest element in C
*/

	    ci[0] = -(*wi) * *d1;
	    ci[1] = 0.f;
	    ci[2] = 0.f;
	    ci[3] = -(*wi) * *d2;
	    cmax = 0.f;
	    icmax = 0;

	    for (j = 1; j <= 4; ++j) {
		if ((r__1 = crv[j - 1], dabs(r__1)) + (r__2 = civ[j - 1],
			dabs(r__2)) > cmax) {
		    cmax = (r__1 = crv[j - 1], dabs(r__1)) + (r__2 = civ[j -
			    1], dabs(r__2));
		    icmax = j;
		}
/* L20: */
	    }

/*           If norm(C) < SMINI, use SMINI*identity. */

	    if (cmax < smini) {
/* Computing MAX */
		r__5 = (r__1 = b[b_dim1 + 1], dabs(r__1)) + (r__2 = b[(b_dim1
			<< 1) + 1], dabs(r__2)), r__6 = (r__3 = b[b_dim1 + 2],
			 dabs(r__3)) + (r__4 = b[(b_dim1 << 1) + 2], dabs(
			r__4));
		bnorm = dmax(r__5,r__6);
		if (smini < 1.f && bnorm > 1.f) {
		    if (bnorm > bignum * smini) {
			*scale = 1.f / bnorm;
		    }
		}
		temp = *scale / smini;
		x[x_dim1 + 1] = temp * b[b_dim1 + 1];
		x[x_dim1 + 2] = temp * b[b_dim1 + 2];
		x[(x_dim1 << 1) + 1] = temp * b[(b_dim1 << 1) + 1];
		x[(x_dim1 << 1) + 2] = temp * b[(b_dim1 << 1) + 2];
		*xnorm = temp * bnorm;
		*info = 1;
		return 0;
	    }

/*           Gaussian elimination with complete pivoting. */

	    ur11 = crv[icmax - 1];
	    ui11 = civ[icmax - 1];
	    cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
	    ci21 = civ[ipivot[(icmax << 2) - 3] - 1];
	    ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
	    ui12 = civ[ipivot[(icmax << 2) - 2] - 1];
	    cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
	    ci22 = civ[ipivot[(icmax << 2) - 1] - 1];
	    if (icmax == 1 || icmax == 4) {

/*              Code when off-diagonals of pivoted C are real */

		if (dabs(ur11) > dabs(ui11)) {
		    temp = ui11 / ur11;
/* Computing 2nd power */
		    r__1 = temp;
		    ur11r = 1.f / (ur11 * (r__1 * r__1 + 1.f));
		    ui11r = -temp * ur11r;
		} else {
		    temp = ur11 / ui11;
/* Computing 2nd power */
		    r__1 = temp;
		    ui11r = -1.f / (ui11 * (r__1 * r__1 + 1.f));
		    ur11r = -temp * ui11r;
		}
		lr21 = cr21 * ur11r;
		li21 = cr21 * ui11r;
		ur12s = ur12 * ur11r;
		ui12s = ur12 * ui11r;
		ur22 = cr22 - ur12 * lr21;
		ui22 = ci22 - ur12 * li21;
	    } else {

/*              Code when diagonals of pivoted C are real */

		ur11r = 1.f / ur11;
		ui11r = 0.f;
		lr21 = cr21 * ur11r;
		li21 = ci21 * ur11r;
		ur12s = ur12 * ur11r;
		ui12s = ui12 * ur11r;
		ur22 = cr22 - ur12 * lr21 + ui12 * li21;
		ui22 = -ur12 * li21 - ui12 * lr21;
	    }
	    u22abs = dabs(ur22) + dabs(ui22);

/*           If smaller pivot < SMINI, use SMINI */

	    if (u22abs < smini) {
		ur22 = smini;
		ui22 = 0.f;
		*info = 1;
	    }
	    if (rswap[icmax - 1]) {
		br2 = b[b_dim1 + 1];
		br1 = b[b_dim1 + 2];
		bi2 = b[(b_dim1 << 1) + 1];
		bi1 = b[(b_dim1 << 1) + 2];
	    } else {
		br1 = b[b_dim1 + 1];
		br2 = b[b_dim1 + 2];
		bi1 = b[(b_dim1 << 1) + 1];
		bi2 = b[(b_dim1 << 1) + 2];
	    }
	    br2 = br2 - lr21 * br1 + li21 * bi1;
	    bi2 = bi2 - li21 * br1 - lr21 * bi1;
/* Computing MAX */
	    r__1 = (dabs(br1) + dabs(bi1)) * (u22abs * (dabs(ur11r) + dabs(
		    ui11r))), r__2 = dabs(br2) + dabs(bi2);
	    bbnd = dmax(r__1,r__2);
	    if (bbnd > 1.f && u22abs < 1.f) {
		if (bbnd >= bignum * u22abs) {
		    *scale = 1.f / bbnd;
		    br1 = *scale * br1;
		    bi1 = *scale * bi1;
		    br2 = *scale * br2;
		    bi2 = *scale * bi2;
		}
	    }

	    sladiv_(&br2, &bi2, &ur22, &ui22, &xr2, &xi2);
	    xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
	    xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
	    if (cswap[icmax - 1]) {
		x[x_dim1 + 1] = xr2;
		x[x_dim1 + 2] = xr1;
		x[(x_dim1 << 1) + 1] = xi2;
		x[(x_dim1 << 1) + 2] = xi1;
	    } else {
		x[x_dim1 + 1] = xr1;
		x[x_dim1 + 2] = xr2;
		x[(x_dim1 << 1) + 1] = xi1;
		x[(x_dim1 << 1) + 2] = xi2;
	    }
/* Computing MAX */
	    r__1 = dabs(xr1) + dabs(xi1), r__2 = dabs(xr2) + dabs(xi2);
	    *xnorm = dmax(r__1,r__2);

/*           Further scaling if  norm(A) norm(X) > overflow */

	    if (*xnorm > 1.f && cmax > 1.f) {
		if (*xnorm > bignum / cmax) {
		    temp = cmax / bignum;
		    x[x_dim1 + 1] = temp * x[x_dim1 + 1];
		    x[x_dim1 + 2] = temp * x[x_dim1 + 2];
		    x[(x_dim1 << 1) + 1] = temp * x[(x_dim1 << 1) + 1];
		    x[(x_dim1 << 1) + 2] = temp * x[(x_dim1 << 1) + 2];
		    *xnorm = temp * *xnorm;
		    *scale = temp * *scale;
		}
	    }
	}
    }

    return 0;

/*     End of SLALN2 */

} /* slaln2_ */

#undef crv
#undef civ
#undef cr
#undef ci


/* Subroutine */ int slamrg_(integer *n1, integer *n2, real *a, integer *
	strd1, integer *strd2, integer *index)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, ind1, ind2, n1sv, n2sv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLAMRG will create a permutation list which will merge the elements
    of A (which is composed of two independently sorted sets) into a
    single set which is sorted in ascending order.

    Arguments
    =========

    N1     (input) INTEGER
    N2     (input) INTEGER
           These arguements contain the respective lengths of the two
           sorted lists to be merged.

    A      (input) REAL array, dimension (N1+N2)
           The first N1 elements of A contain a list of numbers which
           are sorted in either ascending or descending order.  Likewise
           for the final N2 elements.

    STRD1  (input) INTEGER
    STRD2  (input) INTEGER
           These are the strides to be taken through the array A.
           Allowable strides are 1 and -1.  They indicate whether a
           subset of A is sorted in ascending (STRDx = 1) or descending
           (STRDx = -1) order.

    INDEX  (output) INTEGER array, dimension (N1+N2)
           On exit this array will contain a permutation such that
           if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be
           sorted in ascending order.

    =====================================================================
*/


    /* Parameter adjustments */
    --index;
    --a;

    /* Function Body */
    n1sv = *n1;
    n2sv = *n2;
    if (*strd1 > 0) {
	ind1 = 1;
    } else {
	ind1 = *n1;
    }
    if (*strd2 > 0) {
	ind2 = *n1 + 1;
    } else {
	ind2 = *n1 + *n2;
    }
    i__ = 1;
/*     while ( (N1SV > 0) & (N2SV > 0) ) */
L10:
    if (n1sv > 0 && n2sv > 0) {
	if (a[ind1] <= a[ind2]) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *strd1;
	    --n1sv;
	} else {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *strd2;
	    --n2sv;
	}
	goto L10;
    }
/*     end while */
    if (n1sv == 0) {
	i__1 = n2sv;
	for (n1sv = 1; n1sv <= i__1; ++n1sv) {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *strd2;
/* L20: */
	}
    } else {
/*     N2SV .EQ. 0 */
	i__1 = n1sv;
	for (n2sv = 1; n2sv <= i__1; ++n2sv) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *strd1;
/* L30: */
	}
    }

    return 0;

/*     End of SLAMRG */

} /* slamrg_ */

doublereal slange_(char *norm, integer *m, integer *n, real *a, integer *lda,
	real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *,
	    real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLANGE  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    real matrix A.

    Description
    ===========

    SLANGE returns the value

       SLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.

    Arguments
    =========

    NORM    (input) CHARACTER*1
            Specifies the value to be returned in SLANGE as described
            above.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            SLANGE is set to zero.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            SLANGE is set to zero.

    A       (input) REAL array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(M,1).

    WORK    (workspace) REAL array, dimension (LWORK),
            where LWORK >= M when NORM = 'I'; otherwise, WORK is not
            referenced.

   =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    /* Function Body */
    if (min(*m,*n) == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		value = dmax(r__2,r__3);
/* L10: */
	    }
/* L20: */
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {

/*        Find norm1(A). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.f;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
/* L30: */
	    }
	    value = dmax(value,sum);
/* L40: */
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.f;
/* L50: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
/* L60: */
	    }
/* L70: */
	}
	value = 0.f;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__1 = value, r__2 = work[i__];
	    value = dmax(r__1,r__2);
/* L80: */
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    slassq_(m, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of SLANGE */

} /* slange_ */

doublereal slanhs_(char *norm, integer *n, real *a, integer *lda, real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *,
	    real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLANHS  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    Hessenberg matrix A.

    Description
    ===========

    SLANHS returns the value

       SLANHS = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.

    Arguments
    =========

    NORM    (input) CHARACTER*1
            Specifies the value to be returned in SLANHS as described
            above.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, SLANHS is
            set to zero.

    A       (input) REAL array, dimension (LDA,N)
            The n by n upper Hessenberg matrix A; the part of A below the
            first sub-diagonal is not referenced.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(N,1).

    WORK    (workspace) REAL array, dimension (LWORK),
            where LWORK >= N when NORM = 'I'; otherwise, WORK is not
            referenced.

   =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    /* Function Body */
    if (*n == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		value = dmax(r__2,r__3);
/* L10: */
	    }
/* L20: */
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {

/*        Find norm1(A). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.f;
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
/* L30: */
	    }
	    value = dmax(value,sum);
/* L40: */
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.f;
/* L50: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
/* L60: */
	    }
/* L70: */
	}
	value = 0.f;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__1 = value, r__2 = work[i__];
	    value = dmax(r__1,r__2);
/* L80: */
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    slassq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of SLANHS */

} /* slanhs_ */

doublereal slanst_(char *norm, integer *n, real *d__, real *e)
{
    /* System generated locals */
    integer i__1;
    real ret_val, r__1, r__2, r__3, r__4, r__5;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__;
    static real sum, scale;
    extern logical lsame_(char *, char *);
    static real anorm;
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *,
	    real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLANST  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    real symmetric tridiagonal matrix A.

    Description
    ===========

    SLANST returns the value

       SLANST = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.

    Arguments
    =========

    NORM    (input) CHARACTER*1
            Specifies the value to be returned in SLANST as described
            above.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, SLANST is
            set to zero.

    D       (input) REAL array, dimension (N)
            The diagonal elements of A.

    E       (input) REAL array, dimension (N-1)
            The (n-1) sub-diagonal or super-diagonal elements of A.

    =====================================================================
*/


    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    if (*n <= 0) {
	anorm = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	anorm = (r__1 = d__[*n], dabs(r__1));
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__2 = anorm, r__3 = (r__1 = d__[i__], dabs(r__1));
	    anorm = dmax(r__2,r__3);
/* Computing MAX */
	    r__2 = anorm, r__3 = (r__1 = e[i__], dabs(r__1));
	    anorm = dmax(r__2,r__3);
/* L10: */
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1' || lsame_(norm, "I")) {

/*        Find norm1(A). */

	if (*n == 1) {
	    anorm = dabs(d__[1]);
	} else {
/* Computing MAX */
	    r__3 = dabs(d__[1]) + dabs(e[1]), r__4 = (r__1 = e[*n - 1], dabs(
		    r__1)) + (r__2 = d__[*n], dabs(r__2));
	    anorm = dmax(r__3,r__4);
	    i__1 = *n - 1;
	    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MAX */
		r__4 = anorm, r__5 = (r__1 = d__[i__], dabs(r__1)) + (r__2 =
			e[i__], dabs(r__2)) + (r__3 = e[i__ - 1], dabs(r__3));
		anorm = dmax(r__4,r__5);
/* L20: */
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	if (*n > 1) {
	    i__1 = *n - 1;
	    slassq_(&i__1, &e[1], &c__1, &scale, &sum);
	    sum *= 2;
	}
	slassq_(n, &d__[1], &c__1, &scale, &sum);
	anorm = scale * sqrt(sum);
    }

    ret_val = anorm;
    return ret_val;

/*     End of SLANST */

} /* slanst_ */

doublereal slansy_(char *norm, char *uplo, integer *n, real *a, integer *lda,
	real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, absa, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *,
	    real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLANSY  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    real symmetric matrix A.

    Description
    ===========

    SLANSY returns the value

       SLANSY = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'
                (
                ( normI(A),         NORM = 'I' or 'i'
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

    where  norm1  denotes the  one norm of a matrix (maximum column sum),
    normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
    normF  denotes the  Frobenius norm of a matrix (square root of sum of
    squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.

    Arguments
    =========

    NORM    (input) CHARACTER*1
            Specifies the value to be returned in SLANSY as described
            above.

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is to be referenced.
            = 'U':  Upper triangular part of A is referenced
            = 'L':  Lower triangular part of A is referenced

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, SLANSY is
            set to zero.

    A       (input) REAL array, dimension (LDA,N)
            The symmetric matrix A.  If UPLO = 'U', the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced.  If UPLO = 'L', the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(N,1).

    WORK    (workspace) REAL array, dimension (LWORK),
            where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise,
            WORK is not referenced.

   =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    /* Function Body */
    if (*n == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(
			    r__1));
		    value = dmax(r__2,r__3);
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(
			    r__1));
		    value = dmax(r__2,r__3);
/* L30: */
		}
/* L40: */
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*        Find normI(A) ( = norm1(A), since A is symmetric). */

	value = 0.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.f;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		    sum += absa;
		    work[i__] += absa;
/* L50: */
		}
		work[j] = sum + (r__1 = a[j + j * a_dim1], dabs(r__1));
/* L60: */
	    }
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		r__1 = value, r__2 = work[i__];
		value = dmax(r__1,r__2);
/* L70: */
	    }
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] = 0.f;
/* L80: */
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = work[j] + (r__1 = a[j + j * a_dim1], dabs(r__1));
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		    sum += absa;
		    work[i__] += absa;
/* L90: */
		}
		value = dmax(value,sum);
/* L100: */
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		i__2 = j - 1;
		slassq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L110: */
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		slassq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, &scale, &sum);
/* L120: */
	    }
	}
	sum *= 2;
	i__1 = *lda + 1;
	slassq_(n, &a[a_offset], &i__1, &scale, &sum);
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of SLANSY */

} /* slansy_ */

/* Subroutine */ int slanv2_(real *a, real *b, real *c__, real *d__, real *
	rt1r, real *rt1i, real *rt2r, real *rt2i, real *cs, real *sn)
{
    /* System generated locals */
    real r__1, r__2;

    /* Builtin functions */
    double r_sign(real *, real *), sqrt(doublereal);

    /* Local variables */
    static real p, z__, aa, bb, cc, dd, cs1, sn1, sab, sac, eps, tau, temp,
	    scale, bcmax, bcmis, sigma;
    extern doublereal slapy2_(real *, real *), slamch_(char *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
    matrix in standard form:

         [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
         [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]

    where either
    1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or
    2) AA = DD and BB*CC < 0, so that AA + or - sqrt(BB*CC) are complex
    conjugate eigenvalues.

    Arguments
    =========

    A       (input/output) REAL
    B       (input/output) REAL
    C       (input/output) REAL
    D       (input/output) REAL
            On entry, the elements of the input matrix.
            On exit, they are overwritten by the elements of the
            standardised Schur form.

    RT1R    (output) REAL
    RT1I    (output) REAL
    RT2R    (output) REAL
    RT2I    (output) REAL
            The real and imaginary parts of the eigenvalues. If the
            eigenvalues are a complex conjugate pair, RT1I > 0.

    CS      (output) REAL
    SN      (output) REAL
            Parameters of the rotation matrix.

    Further Details
    ===============

    Modified by V. Sima, Research Institute for Informatics, Bucharest,
    Romania, to reduce the risk of cancellation errors,
    when computing real eigenvalues, and to ensure, if possible, that
    abs(RT1R) >= abs(RT2R).

    =====================================================================
*/


    eps = slamch_("P");
    if (*c__ == 0.f) {
	*cs = 1.f;
	*sn = 0.f;
	goto L10;

    } else if (*b == 0.f) {

/*        Swap rows and columns */

	*cs = 0.f;
	*sn = 1.f;
	temp = *d__;
	*d__ = *a;
	*a = temp;
	*b = -(*c__);
	*c__ = 0.f;
	goto L10;
    } else if (*a - *d__ == 0.f && r_sign(&c_b15, b) != r_sign(&c_b15, c__)) {
	*cs = 1.f;
	*sn = 0.f;
	goto L10;
    } else {

	temp = *a - *d__;
	p = temp * .5f;
/* Computing MAX */
	r__1 = dabs(*b), r__2 = dabs(*c__);
	bcmax = dmax(r__1,r__2);
/* Computing MIN */
	r__1 = dabs(*b), r__2 = dabs(*c__);
	bcmis = dmin(r__1,r__2) * r_sign(&c_b15, b) * r_sign(&c_b15, c__);
/* Computing MAX */
	r__1 = dabs(p);
	scale = dmax(r__1,bcmax);
	z__ = p / scale * p + bcmax / scale * bcmis;

/*
          If Z is of the order of the machine accuracy, postpone the
          decision on the nature of eigenvalues
*/

	if (z__ >= eps * 4.f) {

/*           Real eigenvalues. Compute A and D. */

	    r__1 = sqrt(scale) * sqrt(z__);
	    z__ = p + r_sign(&r__1, &p);
	    *a = *d__ + z__;
	    *d__ -= bcmax / z__ * bcmis;

/*           Compute B and the rotation matrix */

	    tau = slapy2_(c__, &z__);
	    *cs = z__ / tau;
	    *sn = *c__ / tau;
	    *b -= *c__;
	    *c__ = 0.f;
	} else {

/*
             Complex eigenvalues, or real (almost) equal eigenvalues.
             Make diagonal elements equal.
*/

	    sigma = *b + *c__;
	    tau = slapy2_(&sigma, &temp);
	    *cs = sqrt((dabs(sigma) / tau + 1.f) * .5f);
	    *sn = -(p / (tau * *cs)) * r_sign(&c_b15, &sigma);

/*
             Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
                     [ CC  DD ]   [ C  D ] [ SN  CS ]
*/

	    aa = *a * *cs + *b * *sn;
	    bb = -(*a) * *sn + *b * *cs;
	    cc = *c__ * *cs + *d__ * *sn;
	    dd = -(*c__) * *sn + *d__ * *cs;

/*
             Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
                     [ C  D ]   [-SN  CS ] [ CC  DD ]
*/

	    *a = aa * *cs + cc * *sn;
	    *b = bb * *cs + dd * *sn;
	    *c__ = -aa * *sn + cc * *cs;
	    *d__ = -bb * *sn + dd * *cs;

	    temp = (*a + *d__) * .5f;
	    *a = temp;
	    *d__ = temp;

	    if (*c__ != 0.f) {
		if (*b != 0.f) {
		    if (r_sign(&c_b15, b) == r_sign(&c_b15, c__)) {

/*                    Real eigenvalues: reduce to upper triangular form */

			sab = sqrt((dabs(*b)));
			sac = sqrt((dabs(*c__)));
			r__1 = sab * sac;
			p = r_sign(&r__1, c__);
			tau = 1.f / sqrt((r__1 = *b + *c__, dabs(r__1)));
			*a = temp + p;
			*d__ = temp - p;
			*b -= *c__;
			*c__ = 0.f;
			cs1 = sab * tau;
			sn1 = sac * tau;
			temp = *cs * cs1 - *sn * sn1;
			*sn = *cs * sn1 + *sn * cs1;
			*cs = temp;
		    }
		} else {
		    *b = -(*c__);
		    *c__ = 0.f;
		    temp = *cs;
		    *cs = -(*sn);
		    *sn = temp;
		}
	    }
	}

    }

L10:

/*     Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I). */

    *rt1r = *a;
    *rt2r = *d__;
    if (*c__ == 0.f) {
	*rt1i = 0.f;
	*rt2i = 0.f;
    } else {
	*rt1i = sqrt((dabs(*b))) * sqrt((dabs(*c__)));
	*rt2i = -(*rt1i);
    }
    return 0;

/*     End of SLANV2 */

} /* slanv2_ */

doublereal slapy2_(real *x, real *y)
{
    /* System generated locals */
    real ret_val, r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real w, z__, xabs, yabs;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary
    overflow.

    Arguments
    =========

    X       (input) REAL
    Y       (input) REAL
            X and Y specify the values x and y.

    =====================================================================
*/


    xabs = dabs(*x);
    yabs = dabs(*y);
    w = dmax(xabs,yabs);
    z__ = dmin(xabs,yabs);
    if (z__ == 0.f) {
	ret_val = w;
    } else {
/* Computing 2nd power */
	r__1 = z__ / w;
	ret_val = w * sqrt(r__1 * r__1 + 1.f);
    }
    return ret_val;

/*     End of SLAPY2 */

} /* slapy2_ */

doublereal slapy3_(real *x, real *y, real *z__)
{
    /* System generated locals */
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real w, xabs, yabs, zabs;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause
    unnecessary overflow.

    Arguments
    =========

    X       (input) REAL
    Y       (input) REAL
    Z       (input) REAL
            X, Y and Z specify the values x, y and z.

    =====================================================================
*/


    xabs = dabs(*x);
    yabs = dabs(*y);
    zabs = dabs(*z__);
/* Computing MAX */
    r__1 = max(xabs,yabs);
    w = dmax(r__1,zabs);
    if (w == 0.f) {
	ret_val = 0.f;
    } else {
/* Computing 2nd power */
	r__1 = xabs / w;
/* Computing 2nd power */
	r__2 = yabs / w;
/* Computing 2nd power */
	r__3 = zabs / w;
	ret_val = w * sqrt(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    }
    return ret_val;

/*     End of SLAPY3 */

} /* slapy3_ */

/* Subroutine */ int slarf_(char *side, integer *m, integer *n, real *v,
	integer *incv, real *tau, real *c__, integer *ldc, real *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    real r__1;

    /* Local variables */
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLARF applies a real elementary reflector H to a real m by n matrix
    C, from either the left or the right. H is represented in the form

          H = I - tau * v * v'

    where tau is a real scalar and v is a real vector.

    If tau = 0, then H is taken to be the unit matrix.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': form  H * C
            = 'R': form  C * H

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    V       (input) REAL array, dimension
                       (1 + (M-1)*abs(INCV)) if SIDE = 'L'
                    or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
            The vector v in the representation of H. V is not used if
            TAU = 0.

    INCV    (input) INTEGER
            The increment between elements of v. INCV <> 0.

    TAU     (input) REAL
            The value tau in the representation of H.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) REAL array, dimension
                           (N) if SIDE = 'L'
                        or (M) if SIDE = 'R'

    =====================================================================
*/


    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (lsame_(side, "L")) {

/*        Form  H * C */

	if (*tau != 0.f) {

/*           w := C' * v */

	    sgemv_("Transpose", m, n, &c_b15, &c__[c_offset], ldc, &v[1],
		    incv, &c_b29, &work[1], &c__1);

/*           C := C - v * w' */

	    r__1 = -(*tau);
	    sger_(m, n, &r__1, &v[1], incv, &work[1], &c__1, &c__[c_offset],
		    ldc);
	}
    } else {

/*        Form  C * H */

	if (*tau != 0.f) {

/*           w := C * v */

	    sgemv_("No transpose", m, n, &c_b15, &c__[c_offset], ldc, &v[1],
		    incv, &c_b29, &work[1], &c__1);

/*           C := C - w * v' */

	    r__1 = -(*tau);
	    sger_(m, n, &r__1, &work[1], &c__1, &v[1], incv, &c__[c_offset],
		    ldc);
	}
    }
    return 0;

/*     End of SLARF */

} /* slarf_ */

/* Subroutine */ int slarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, real *v, integer *ldv,
	real *t, integer *ldt, real *c__, integer *ldc, real *work, integer *
	ldwork)
{
    /* System generated locals */
    integer c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1,
	    work_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *), scopy_(integer *, real *,
	    integer *, real *, integer *), strmm_(char *, char *, char *,
	    char *, integer *, integer *, real *, real *, integer *, real *,
	    integer *);
    static char transt[1];


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLARFB applies a real block reflector H or its transpose H' to a
    real m by n matrix C, from either the left or the right.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply H or H' from the Left
            = 'R': apply H or H' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply H (No transpose)
            = 'T': apply H' (Transpose)

    DIRECT  (input) CHARACTER*1
            Indicates how H is formed from a product of elementary
            reflectors
            = 'F': H = H(1) H(2) . . . H(k) (Forward)
            = 'B': H = H(k) . . . H(2) H(1) (Backward)

    STOREV  (input) CHARACTER*1
            Indicates how the vectors which define the elementary
            reflectors are stored:
            = 'C': Columnwise
            = 'R': Rowwise

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    K       (input) INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    V       (input) REAL array, dimension
                                  (LDV,K) if STOREV = 'C'
                                  (LDV,M) if STOREV = 'R' and SIDE = 'L'
                                  (LDV,N) if STOREV = 'R' and SIDE = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
            if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
            if STOREV = 'R', LDV >= K.

    T       (input) REAL array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C or H'*C or C*H or C*H'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) REAL array, dimension (LDWORK,K)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK.
            If SIDE = 'L', LDWORK >= max(1,N);
            if SIDE = 'R', LDWORK >= max(1,M).

    =====================================================================


       Quick return if possible
*/

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    work_dim1 = *ldwork;
    work_offset = 1 + work_dim1;
    work -= work_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (lsame_(trans, "N")) {
	*(unsigned char *)transt = 'T';
    } else {
	*(unsigned char *)transt = 'N';
    }

    if (lsame_(storev, "C")) {

	if (lsame_(direct, "F")) {

/*
             Let  V =  ( V1 )    (first K rows)
                       ( V2 )
             where  V1  is unit lower triangular.
*/

	    if (lsame_(side, "L")) {

/*
                Form  H * C  or  H' * C  where  C = ( C1 )
                                                    ( C2 )

                W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK)

                W := C1'
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
/* L10: */
		}

/*              W := W * V1 */

		strmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b15,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2 */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "No transpose", n, k, &i__1, &c_b15, &
			    c__[*k + 1 + c_dim1], ldc, &v[*k + 1 + v_dim1],
			    ldv, &c_b15, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		strmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2 * W' */

		    i__1 = *m - *k;
		    sgemm_("No transpose", "Transpose", &i__1, n, k, &c_b151,
			    &v[*k + 1 + v_dim1], ldv, &work[work_offset],
			    ldwork, &c_b15, &c__[*k + 1 + c_dim1], ldc);
		}

/*              W := W * V1' */

		strmm_("Right", "Lower", "Transpose", "Unit", n, k, &c_b15, &
			v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
/* L20: */
		    }
/* L30: */
		}

	    } else if (lsame_(side, "R")) {

/*
                Form  C * H  or  C * H'  where  C = ( C1  C2 )

                W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)

                W := C1
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L40: */
		}

/*              W := W * V1 */

		strmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b15,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2 */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b15, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k +
			    1 + v_dim1], ldv, &c_b15, &work[work_offset],
			    ldwork);
		}

/*              W := W * T  or  W * T' */

		strmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C2 := C2 - W * V2' */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, &i__1, k, &c_b151,
			    &work[work_offset], ldwork, &v[*k + 1 + v_dim1],
			    ldv, &c_b15, &c__[(*k + 1) * c_dim1 + 1], ldc);
		}

/*              W := W * V1' */

		strmm_("Right", "Lower", "Transpose", "Unit", m, k, &c_b15, &
			v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
/* L50: */
		    }
/* L60: */
		}
	    }

	} else {

/*
             Let  V =  ( V1 )
                       ( V2 )    (last K rows)
             where  V2  is unit upper triangular.
*/

	    if (lsame_(side, "L")) {

/*
                Form  H * C  or  H' * C  where  C = ( C1 )
                                                    ( C2 )

                W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK)

                W := C2'
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
/* L70: */
		}

/*              W := W * V2 */

		strmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b15,
			 &v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1 */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "No transpose", n, k, &i__1, &c_b15, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b15, &
			    work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		strmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1 * W' */

		    i__1 = *m - *k;
		    sgemm_("No transpose", "Transpose", &i__1, n, k, &c_b151,
			    &v[v_offset], ldv, &work[work_offset], ldwork, &
			    c_b15, &c__[c_offset], ldc)
			    ;
		}

/*              W := W * V2' */

		strmm_("Right", "Upper", "Transpose", "Unit", n, k, &c_b15, &
			v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[*m - *k + j + i__ * c_dim1] -= work[i__ + j *
				work_dim1];
/* L80: */
		    }
/* L90: */
		}

	    } else if (lsame_(side, "R")) {

/*
                Form  C * H  or  C * H'  where  C = ( C1  C2 )

                W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)

                W := C2
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L100: */
		}

/*              W := W * V2 */

		strmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b15,
			 &v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1 */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b15, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b15, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		strmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C1 := C1 - W * V1' */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, &i__1, k, &c_b151,
			    &work[work_offset], ldwork, &v[v_offset], ldv, &
			    c_b15, &c__[c_offset], ldc)
			    ;
		}

/*              W := W * V2' */

		strmm_("Right", "Upper", "Transpose", "Unit", m, k, &c_b15, &
			v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);

/*              C2 := C2 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (*n - *k + j) * c_dim1] -= work[i__ + j *
				work_dim1];
/* L110: */
		    }
/* L120: */
		}
	    }
	}

    } else if (lsame_(storev, "R")) {

	if (lsame_(direct, "F")) {

/*
             Let  V =  ( V1  V2 )    (V1: first K columns)
             where  V1  is unit upper triangular.
*/

	    if (lsame_(side, "L")) {

/*
                Form  H * C  or  H' * C  where  C = ( C1 )
                                                    ( C2 )

                W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK)

                W := C1'
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
/* L130: */
		}

/*              W := W * V1' */

		strmm_("Right", "Upper", "Transpose", "Unit", n, k, &c_b15, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2' */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", n, k, &i__1, &c_b15, &
			    c__[*k + 1 + c_dim1], ldc, &v[(*k + 1) * v_dim1 +
			    1], ldv, &c_b15, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		strmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2' * W' */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", &i__1, n, k, &c_b151, &v[
			    (*k + 1) * v_dim1 + 1], ldv, &work[work_offset],
			    ldwork, &c_b15, &c__[*k + 1 + c_dim1], ldc);
		}

/*              W := W * V1 */

		strmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b15,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
/* L140: */
		    }
/* L150: */
		}

	    } else if (lsame_(side, "R")) {

/*
                Form  C * H  or  C * H'  where  C = ( C1  C2 )

                W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK)

                W := C1
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L160: */
		}

/*              W := W * V1' */

		strmm_("Right", "Upper", "Transpose", "Unit", m, k, &c_b15, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2' */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, k, &i__1, &c_b15, &
			    c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k + 1) *
			    v_dim1 + 1], ldv, &c_b15, &work[work_offset],
			    ldwork);
		}

/*              W := W * T  or  W * T' */

		strmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C2 := C2 - W * V2 */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, &i__1, k, &
			    c_b151, &work[work_offset], ldwork, &v[(*k + 1) *
			    v_dim1 + 1], ldv, &c_b15, &c__[(*k + 1) * c_dim1
			    + 1], ldc);
		}

/*              W := W * V1 */

		strmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b15,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
/* L170: */
		    }
/* L180: */
		}

	    }

	} else {

/*
             Let  V =  ( V1  V2 )    (V2: last K columns)
             where  V2  is unit lower triangular.
*/

	    if (lsame_(side, "L")) {

/*
                Form  H * C  or  H' * C  where  C = ( C1 )
                                                    ( C2 )

                W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK)

                W := C2'
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
/* L190: */
		}

/*              W := W * V2' */

		strmm_("Right", "Lower", "Transpose", "Unit", n, k, &c_b15, &
			v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[work_offset]
			, ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1' */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", n, k, &i__1, &c_b15, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b15, &
			    work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		strmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1' * W' */

		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", &i__1, n, k, &c_b151, &v[
			    v_offset], ldv, &work[work_offset], ldwork, &
			    c_b15, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		strmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b15,
			 &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[*m - *k + j + i__ * c_dim1] -= work[i__ + j *
				work_dim1];
/* L200: */
		    }
/* L210: */
		}

	    } else if (lsame_(side, "R")) {

/*
                Form  C * H  or  C * H'  where  C = ( C1  C2 )

                W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK)

                W := C2
*/

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L220: */
		}

/*              W := W * V2' */

		strmm_("Right", "Lower", "Transpose", "Unit", m, k, &c_b15, &
			v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[work_offset]
			, ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1' */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, k, &i__1, &c_b15, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b15, &
			    work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		strmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b15, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C1 := C1 - W * V1 */

		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, &i__1, k, &
			    c_b151, &work[work_offset], ldwork, &v[v_offset],
			    ldv, &c_b15, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		strmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b15,
			 &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (*n - *k + j) * c_dim1] -= work[i__ + j *
				work_dim1];
/* L230: */
		    }
/* L240: */
		}

	    }

	}
    }

    return 0;

/*     End of SLARFB */

} /* slarfb_ */

/* Subroutine */ int slarfg_(integer *n, real *alpha, real *x, integer *incx,
	real *tau)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    double r_sign(real *, real *);

    /* Local variables */
    static integer j, knt;
    static real beta;
    extern doublereal snrm2_(integer *, real *, integer *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static real xnorm;
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    static real safmin, rsafmn;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLARFG generates a real elementary reflector H of order n, such
    that

          H * ( alpha ) = ( beta ),   H' * H = I.
              (   x   )   (   0  )

    where alpha and beta are scalars, and x is an (n-1)-element real
    vector. H is represented in the form

          H = I - tau * ( 1 ) * ( 1 v' ) ,
                        ( v )

    where tau is a real scalar and v is a real (n-1)-element
    vector.

    If the elements of x are all zero, then tau = 0 and H is taken to be
    the unit matrix.

    Otherwise  1 <= tau <= 2.

    Arguments
    =========

    N       (input) INTEGER
            The order of the elementary reflector.

    ALPHA   (input/output) REAL
            On entry, the value alpha.
            On exit, it is overwritten with the value beta.

    X       (input/output) REAL array, dimension
                           (1+(N-2)*abs(INCX))
            On entry, the vector x.
            On exit, it is overwritten with the vector v.

    INCX    (input) INTEGER
            The increment between elements of X. INCX > 0.

    TAU     (output) REAL
            The value tau.

    =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 1) {
	*tau = 0.f;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = snrm2_(&i__1, &x[1], incx);

    if (xnorm == 0.f) {

/*        H  =  I */

	*tau = 0.f;
    } else {

/*        general case */

	r__1 = slapy2_(alpha, &xnorm);
	beta = -r_sign(&r__1, alpha);
	safmin = slamch_("S") / slamch_("E");
	if (dabs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

	    rsafmn = 1.f / safmin;
	    knt = 0;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    sscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    *alpha *= rsafmn;
	    if (dabs(beta) < safmin) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = snrm2_(&i__1, &x[1], incx);
	    r__1 = slapy2_(alpha, &xnorm);
	    beta = -r_sign(&r__1, alpha);
	    *tau = (beta - *alpha) / beta;
	    i__1 = *n - 1;
	    r__1 = 1.f / (*alpha - beta);
	    sscal_(&i__1, &r__1, &x[1], incx);

/*           If ALPHA is subnormal, it may lose relative accuracy */

	    *alpha = beta;
	    i__1 = knt;
	    for (j = 1; j <= i__1; ++j) {
		*alpha *= safmin;
/* L20: */
	    }
	} else {
	    *tau = (beta - *alpha) / beta;
	    i__1 = *n - 1;
	    r__1 = 1.f / (*alpha - beta);
	    sscal_(&i__1, &r__1, &x[1], incx);
	    *alpha = beta;
	}
    }

    return 0;

/*     End of SLARFG */

} /* slarfg_ */

/* Subroutine */ int slarft_(char *direct, char *storev, integer *n, integer *
	k, real *v, integer *ldv, real *tau, real *t, integer *ldt)
{
    /* System generated locals */
    integer t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3;
    real r__1;

    /* Local variables */
    static integer i__, j;
    static real vii;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *), strmv_(char *, char *, char *, integer *, real *,
	    integer *, real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLARFT forms the triangular factor T of a real block reflector H
    of order n, which is defined as a product of k elementary reflectors.

    If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;

    If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.

    If STOREV = 'C', the vector which defines the elementary reflector
    H(i) is stored in the i-th column of the array V, and

       H  =  I - V * T * V'

    If STOREV = 'R', the vector which defines the elementary reflector
    H(i) is stored in the i-th row of the array V, and

       H  =  I - V' * T * V

    Arguments
    =========

    DIRECT  (input) CHARACTER*1
            Specifies the order in which the elementary reflectors are
            multiplied to form the block reflector:
            = 'F': H = H(1) H(2) . . . H(k) (Forward)
            = 'B': H = H(k) . . . H(2) H(1) (Backward)

    STOREV  (input) CHARACTER*1
            Specifies how the vectors which define the elementary
            reflectors are stored (see also Further Details):
            = 'C': columnwise
            = 'R': rowwise

    N       (input) INTEGER
            The order of the block reflector H. N >= 0.

    K       (input) INTEGER
            The order of the triangular factor T (= the number of
            elementary reflectors). K >= 1.

    V       (input/output) REAL array, dimension
                                 (LDV,K) if STOREV = 'C'
                                 (LDV,N) if STOREV = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i).

    T       (output) REAL array, dimension (LDT,K)
            The k by k triangular factor T of the block reflector.
            If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is
            lower triangular. The rest of the array is not used.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    Further Details
    ===============

    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3. The elements equal to 1 are not stored; the corresponding
    array elements are modified but restored on exit. The rest of the
    array is not used.

    DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':

                 V = (  1       )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1    )                     (     1 v2 v2 v2 )
                     ( v1 v2  1 )                     (        1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

    DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1    )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (     1 v3 )
                     (        1 )

    =====================================================================


       Quick return if possible
*/

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    /* Function Body */
    if (*n == 0) {
	return 0;
    }

    if (lsame_(direct, "F")) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (tau[i__] == 0.f) {

/*              H(i)  =  I */

		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t[j + i__ * t_dim1] = 0.f;
/* L10: */
		}
	    } else {

/*              general case */

		vii = v[i__ + i__ * v_dim1];
		v[i__ + i__ * v_dim1] = 1.f;
		if (lsame_(storev, "C")) {

/*                 T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i) */

		    i__2 = *n - i__ + 1;
		    i__3 = i__ - 1;
		    r__1 = -tau[i__];
		    sgemv_("Transpose", &i__2, &i__3, &r__1, &v[i__ + v_dim1],
			     ldv, &v[i__ + i__ * v_dim1], &c__1, &c_b29, &t[
			    i__ * t_dim1 + 1], &c__1);
		} else {

/*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:n) * V(i,i:n)' */

		    i__2 = i__ - 1;
		    i__3 = *n - i__ + 1;
		    r__1 = -tau[i__];
		    sgemv_("No transpose", &i__2, &i__3, &r__1, &v[i__ *
			    v_dim1 + 1], ldv, &v[i__ + i__ * v_dim1], ldv, &
			    c_b29, &t[i__ * t_dim1 + 1], &c__1);
		}
		v[i__ + i__ * v_dim1] = vii;

/*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i) */

		i__2 = i__ - 1;
		strmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t[i__ * t_dim1 + 1], &c__1);
		t[i__ + i__ * t_dim1] = tau[i__];
	    }
/* L20: */
	}
    } else {
	for (i__ = *k; i__ >= 1; --i__) {
	    if (tau[i__] == 0.f) {

/*              H(i)  =  I */

		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    t[j + i__ * t_dim1] = 0.f;
/* L30: */
		}
	    } else {

/*              general case */

		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			vii = v[*n - *k + i__ + i__ * v_dim1];
			v[*n - *k + i__ + i__ * v_dim1] = 1.f;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(1:n-k+i,i+1:k)' * V(1:n-k+i,i)
*/

			i__1 = *n - *k + i__;
			i__2 = *k - i__;
			r__1 = -tau[i__];
			sgemv_("Transpose", &i__1, &i__2, &r__1, &v[(i__ + 1)
				* v_dim1 + 1], ldv, &v[i__ * v_dim1 + 1], &
				c__1, &c_b29, &t[i__ + 1 + i__ * t_dim1], &
				c__1);
			v[*n - *k + i__ + i__ * v_dim1] = vii;
		    } else {
			vii = v[i__ + (*n - *k + i__) * v_dim1];
			v[i__ + (*n - *k + i__) * v_dim1] = 1.f;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(i+1:k,1:n-k+i) * V(i,1:n-k+i)'
*/

			i__1 = *k - i__;
			i__2 = *n - *k + i__;
			r__1 = -tau[i__];
			sgemv_("No transpose", &i__1, &i__2, &r__1, &v[i__ +
				1 + v_dim1], ldv, &v[i__ + v_dim1], ldv, &
				c_b29, &t[i__ + 1 + i__ * t_dim1], &c__1);
			v[i__ + (*n - *k + i__) * v_dim1] = vii;
		    }

/*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i) */

		    i__1 = *k - i__;
		    strmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__
			    + 1 + (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ *
			     t_dim1], &c__1)
			    ;
		}
		t[i__ + i__ * t_dim1] = tau[i__];
	    }
/* L40: */
	}
    }
    return 0;

/*     End of SLARFT */

} /* slarft_ */

/* Subroutine */ int slarfx_(char *side, integer *m, integer *n, real *v,
	real *tau, real *c__, integer *ldc, real *work)
{
    /* System generated locals */
    integer c_dim1, c_offset, i__1;
    real r__1;

    /* Local variables */
    static integer j;
    static real t1, t2, t3, t4, t5, t6, t7, t8, t9, v1, v2, v3, v4, v5, v6,
	    v7, v8, v9, t10, v10, sum;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLARFX applies a real elementary reflector H to a real m by n
    matrix C, from either the left or the right. H is represented in the
    form

          H = I - tau * v * v'

    where tau is a real scalar and v is a real vector.

    If tau = 0, then H is taken to be the unit matrix

    This version uses inline code if H has order < 11.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': form  H * C
            = 'R': form  C * H

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    V       (input) REAL array, dimension (M) if SIDE = 'L'
                                       or (N) if SIDE = 'R'
            The vector v in the representation of H.

    TAU     (input) REAL
            The value tau in the representation of H.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= (1,M).

    WORK    (workspace) REAL array, dimension
                        (N) if SIDE = 'L'
                        or (M) if SIDE = 'R'
            WORK is not referenced if H has order < 11.

    =====================================================================
*/


    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (*tau == 0.f) {
	return 0;
    }
    if (lsame_(side, "L")) {

/*        Form  H * C, where H has order m. */

	switch (*m) {
	    case 1:  goto L10;
	    case 2:  goto L30;
	    case 3:  goto L50;
	    case 4:  goto L70;
	    case 5:  goto L90;
	    case 6:  goto L110;
	    case 7:  goto L130;
	    case 8:  goto L150;
	    case 9:  goto L170;
	    case 10:  goto L190;
	}

/*
          Code for general M

          w := C'*v
*/

	sgemv_("Transpose", m, n, &c_b15, &c__[c_offset], ldc, &v[1], &c__1, &
		c_b29, &work[1], &c__1);

/*        C := C - tau * v * w' */

	r__1 = -(*tau);
	sger_(m, n, &r__1, &v[1], &c__1, &work[1], &c__1, &c__[c_offset], ldc)
		;
	goto L410;
L10:

/*        Special code for 1 x 1 Householder */

	t1 = 1.f - *tau * v[1] * v[1];
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    c__[j * c_dim1 + 1] = t1 * c__[j * c_dim1 + 1];
/* L20: */
	}
	goto L410;
L30:

/*        Special code for 2 x 2 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
/* L40: */
	}
	goto L410;
L50:

/*        Special code for 3 x 3 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
/* L60: */
	}
	goto L410;
L70:

/*        Special code for 4 x 4 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
/* L80: */
	}
	goto L410;
L90:

/*        Special code for 5 x 5 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
/* L100: */
	}
	goto L410;
L110:

/*        Special code for 6 x 6 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
/* L120: */
	}
	goto L410;
L130:

/*        Special code for 7 x 7 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
/* L140: */
	}
	goto L410;
L150:

/*        Special code for 8 x 8 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
/* L160: */
	}
	goto L410;
L170:

/*        Special code for 9 x 9 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8] + v9 * c__[j *
		    c_dim1 + 9];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
	    c__[j * c_dim1 + 9] -= sum * t9;
/* L180: */
	}
	goto L410;
L190:

/*        Special code for 10 x 10 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	v10 = v[10];
	t10 = *tau * v10;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8] + v9 * c__[j *
		    c_dim1 + 9] + v10 * c__[j * c_dim1 + 10];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
	    c__[j * c_dim1 + 9] -= sum * t9;
	    c__[j * c_dim1 + 10] -= sum * t10;
/* L200: */
	}
	goto L410;
    } else {

/*        Form  C * H, where H has order n. */

	switch (*n) {
	    case 1:  goto L210;
	    case 2:  goto L230;
	    case 3:  goto L250;
	    case 4:  goto L270;
	    case 5:  goto L290;
	    case 6:  goto L310;
	    case 7:  goto L330;
	    case 8:  goto L350;
	    case 9:  goto L370;
	    case 10:  goto L390;
	}

/*
          Code for general N

          w := C * v
*/

	sgemv_("No transpose", m, n, &c_b15, &c__[c_offset], ldc, &v[1], &
		c__1, &c_b29, &work[1], &c__1);

/*        C := C - tau * w * v' */

	r__1 = -(*tau);
	sger_(m, n, &r__1, &work[1], &c__1, &v[1], &c__1, &c__[c_offset], ldc)
		;
	goto L410;
L210:

/*        Special code for 1 x 1 Householder */

	t1 = 1.f - *tau * v[1] * v[1];
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    c__[j + c_dim1] = t1 * c__[j + c_dim1];
/* L220: */
	}
	goto L410;
L230:

/*        Special code for 2 x 2 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
/* L240: */
	}
	goto L410;
L250:

/*        Special code for 3 x 3 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
/* L260: */
	}
	goto L410;
L270:

/*        Special code for 4 x 4 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
/* L280: */
	}
	goto L410;
L290:

/*        Special code for 5 x 5 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
/* L300: */
	}
	goto L410;
L310:

/*        Special code for 6 x 6 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
/* L320: */
	}
	goto L410;
L330:

/*        Special code for 7 x 7 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
/* L340: */
	}
	goto L410;
L350:

/*        Special code for 8 x 8 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
/* L360: */
	}
	goto L410;
L370:

/*        Special code for 9 x 9 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)] + v9 * c__[
		    j + c_dim1 * 9];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
	    c__[j + c_dim1 * 9] -= sum * t9;
/* L380: */
	}
	goto L410;
L390:

/*        Special code for 10 x 10 Householder */

	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	v10 = v[10];
	t10 = *tau * v10;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)] + v9 * c__[
		    j + c_dim1 * 9] + v10 * c__[j + c_dim1 * 10];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
	    c__[j + c_dim1 * 9] -= sum * t9;
	    c__[j + c_dim1 * 10] -= sum * t10;
/* L400: */
	}
	goto L410;
    }
L410:
    return 0;

/*     End of SLARFX */

} /* slarfx_ */

/* Subroutine */ int slartg_(real *f, real *g, real *cs, real *sn, real *r__)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */
    double log(doublereal), pow_ri(real *, integer *), sqrt(doublereal);

    /* Local variables */
    static integer i__;
    static real f1, g1, eps, scale;
    static integer count;
    static real safmn2, safmx2;
    extern doublereal slamch_(char *);
    static real safmin;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLARTG generate a plane rotation so that

       [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1.
       [ -SN  CS  ]     [ G ]     [ 0 ]

    This is a slower, more accurate version of the BLAS1 routine SROTG,
    with the following other differences:
       F and G are unchanged on return.
       If G=0, then CS=1 and SN=0.
       If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any
          floating point operations (saves work in SBDSQR when
          there are zeros on the diagonal).

    If F exceeds G in magnitude, CS will be positive.

    Arguments
    =========

    F       (input) REAL
            The first component of vector to be rotated.

    G       (input) REAL
            The second component of vector to be rotated.

    CS      (output) REAL
            The cosine of the rotation.

    SN      (output) REAL
            The sine of the rotation.

    R       (output) REAL
            The nonzero component of the rotated vector.

    =====================================================================
*/


    if (first) {
	first = FALSE_;
	safmin = slamch_("S");
	eps = slamch_("E");
	r__1 = slamch_("B");
	i__1 = (integer) (log(safmin / eps) / log(slamch_("B")) /
		2.f);
	safmn2 = pow_ri(&r__1, &i__1);
	safmx2 = 1.f / safmn2;
    }
    if (*g == 0.f) {
	*cs = 1.f;
	*sn = 0.f;
	*r__ = *f;
    } else if (*f == 0.f) {
	*cs = 0.f;
	*sn = 1.f;
	*r__ = *g;
    } else {
	f1 = *f;
	g1 = *g;
/* Computing MAX */
	r__1 = dabs(f1), r__2 = dabs(g1);
	scale = dmax(r__1,r__2);
	if (scale >= safmx2) {
	    count = 0;
L10:
	    ++count;
	    f1 *= safmn2;
	    g1 *= safmn2;
/* Computing MAX */
	    r__1 = dabs(f1), r__2 = dabs(g1);
	    scale = dmax(r__1,r__2);
	    if (scale >= safmx2) {
		goto L10;
	    }
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmx2;
/* L20: */
	    }
	} else if (scale <= safmn2) {
	    count = 0;
L30:
	    ++count;
	    f1 *= safmx2;
	    g1 *= safmx2;
/* Computing MAX */
	    r__1 = dabs(f1), r__2 = dabs(g1);
	    scale = dmax(r__1,r__2);
	    if (scale <= safmn2) {
		goto L30;
	    }
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmn2;
/* L40: */
	    }
	} else {
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = sqrt(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	}
	if (dabs(*f) > dabs(*g) && *cs < 0.f) {
	    *cs = -(*cs);
	    *sn = -(*sn);
	    *r__ = -(*r__);
	}
    }
    return 0;

/*     End of SLARTG */

} /* slartg_ */

/* Subroutine */ int slas2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax)
{
    /* System generated locals */
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real c__, fa, ga, ha, as, at, au, fhmn, fhmx;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SLAS2  computes the singular values of the 2-by-2 matrix
       [  F   G  ]
       [  0   H  ].
    On return, SSMIN is the smaller singular value and SSMAX is the
    larger singular value.

    Arguments
    =========

    F       (input) REAL
            The (1,1) element of the 2-by-2 matrix.

    G       (input) REAL
            The (1,2) element of the 2-by-2 matrix.

    H       (input) REAL
            The (2,2) element of the 2-by-2 matrix.

    SSMIN   (output) REAL
            The smaller singular value.

    SSMAX   (output) REAL
            The larger singular value.

    Further Details
    ===============

    Barring over/underflow, all output quantities are correct to within
    a few units in the last place (ulps), even in the absence of a guard
    digit in addition/subtraction.

    In IEEE arithmetic, the code works correctly if one matrix element is
    infinite.

    Overflow will not occur unless the largest singular value itself
    overflows, or is within a few ulps of overflow. (On machines with
    partial overflow, like the Cray, overflow may occur if the largest
    singular value is within a factor of 2 of overflow.)

    Underflow is harmless if underflow is gradual. Otherwise, results
    may correspond to a matrix modified by perturbations of size near
    the underflow threshold.

    ====================================================================
*/


    fa = dabs(*f);
    ga = dabs(*g);
    ha = dabs(*h__);
    fhmn = dmin(fa,ha);
    fhmx = dmax(fa,ha);
    if (fhmn == 0.f) {
	*ssmin = 0.f;
	if (fhmx == 0.f) {
	    *ssmax = ga;
	} else {
/* Computing 2nd power */
	    r__1 = dmin(fhmx,ga) / dmax(fhmx,ga);
	    *ssmax = dmax(fhmx,ga) * sqrt(r__1 * r__1 + 1.f);
	}
    } else {
	if (ga < fhmx) {
	    as = fhmn / fhmx + 1.f;
	    at = (fhmx - fhmn) / fhmx;
/* Computing 2nd power */
	    r__1 = ga / fhmx;
	    au = r__1 * r__1;
	    c__ = 2.f / (sqrt(as * as + au) + sqrt(at * at + au));
	    *ssmin = fhmn * c__;
	    *ssmax = fhmx / c__;
	} else {
	    au = fhmx / ga;
	    if (au == 0.f) {

/*
                Avoid possible harmful underflow if exponent range
                asymmetric (true SSMIN may not underflow even if
                AU underflows)
*/

		*ssmin = fhmn * fhmx / ga;
		*ssmax = ga;
	    } else {
		as = fhmn / fhmx + 1.f;
		at = (fhmx - fhmn) / fhmx;
/* Computing 2nd power */
		r__1 = as * au;
/* Computing 2nd power */
		r__2 = at * au;
		c__ = 1.f / (sqrt(r__1 * r__1 + 1.f) + sqrt(r__2 * r__2 + 1.f)
			);
		*ssmin = fhmn * c__ * au;
		*ssmin += *ssmin;
		*ssmax = ga / (c__ + c__);
	    }
	}
    }
    return 0;

/*     End of SLAS2 */

} /* slas2_ */

/* Subroutine */ int slascl_(char *type__, integer *kl, integer *ku, real *
	cfrom, real *cto, integer *m, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    static integer i__, j, k1, k2, k3, k4;
    static real mul, cto1;
    static logical done;
    static real ctoc;
    extern logical lsame_(char *, char *);
    static integer itype;
    static real cfrom1;
    extern doublereal slamch_(char *);
    static real cfromc;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real bignum, smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLASCL multiplies the M by N real matrix A by the real scalar
    CTO/CFROM.  This is done without over/underflow as long as the final
    result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
    A may be full, upper triangular, lower triangular, upper Hessenberg,
    or banded.

    Arguments
    =========

    TYPE    (input) CHARACTER*1
            TYPE indices the storage type of the input matrix.
            = 'G':  A is a full matrix.
            = 'L':  A is a lower triangular matrix.
            = 'U':  A is an upper triangular matrix.
            = 'H':  A is an upper Hessenberg matrix.
            = 'B':  A is a symmetric band matrix with lower bandwidth KL
                    and upper bandwidth KU and with the only the lower
                    half stored.
            = 'Q':  A is a symmetric band matrix with lower bandwidth KL
                    and upper bandwidth KU and with the only the upper
                    half stored.
            = 'Z':  A is a band matrix with lower bandwidth KL and upper
                    bandwidth KU.

    KL      (input) INTEGER
            The lower bandwidth of A.  Referenced only if TYPE = 'B',
            'Q' or 'Z'.

    KU      (input) INTEGER
            The upper bandwidth of A.  Referenced only if TYPE = 'B',
            'Q' or 'Z'.

    CFROM   (input) REAL
    CTO     (input) REAL
            The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
            without over/underflow if the final result CTO*A(I,J)/CFROM
            can be represented without over/underflow.  CFROM must be
            nonzero.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,M)
            The matrix to be multiplied by CTO/CFROM.  See TYPE for the
            storage type.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    INFO    (output) INTEGER
            0  - successful exit
            <0 - if INFO = -i, the i-th argument had an illegal value.

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;

    if (lsame_(type__, "G")) {
	itype = 0;
    } else if (lsame_(type__, "L")) {
	itype = 1;
    } else if (lsame_(type__, "U")) {
	itype = 2;
    } else if (lsame_(type__, "H")) {
	itype = 3;
    } else if (lsame_(type__, "B")) {
	itype = 4;
    } else if (lsame_(type__, "Q")) {
	itype = 5;
    } else if (lsame_(type__, "Z")) {
	itype = 6;
    } else {
	itype = -1;
    }

    if (itype == -1) {
	*info = -1;
    } else if (*cfrom == 0.f) {
	*info = -4;
    } else if (*m < 0) {
	*info = -6;
    } else if (*n < 0 || itype == 4 && *n != *m || itype == 5 && *n != *m) {
	*info = -7;
    } else if (itype <= 3 && *lda < max(1,*m)) {
	*info = -9;
    } else if (itype >= 4) {
/* Computing MAX */
	i__1 = *m - 1;
	if (*kl < 0 || *kl > max(i__1,0)) {
	    *info = -2;
	} else /* if(complicated condition) */ {
/* Computing MAX */
	    i__1 = *n - 1;
	    if (*ku < 0 || *ku > max(i__1,0) || (itype == 4 || itype == 5) &&
		    *kl != *ku) {
		*info = -3;
	    } else if (itype == 4 && *lda < *kl + 1 || itype == 5 && *lda < *
		    ku + 1 || itype == 6 && *lda < (*kl << 1) + *ku + 1) {
		*info = -9;
	    }
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASCL", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *m == 0) {
	return 0;
    }

/*     Get machine parameters */

    smlnum = slamch_("S");
    bignum = 1.f / smlnum;

    cfromc = *cfrom;
    ctoc = *cto;

L10:
    cfrom1 = cfromc * smlnum;
    cto1 = ctoc / bignum;
    if (dabs(cfrom1) > dabs(ctoc) && ctoc != 0.f) {
	mul = smlnum;
	done = FALSE_;
	cfromc = cfrom1;
    } else if (dabs(cto1) > dabs(cfromc)) {
	mul = bignum;
	done = FALSE_;
	ctoc = cto1;
    } else {
	mul = ctoc / cfromc;
	done = TRUE_;
    }

    if (itype == 0) {

/*        Full matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L20: */
	    }
/* L30: */
	}

    } else if (itype == 1) {

/*        Lower triangular matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L40: */
	    }
/* L50: */
	}

    } else if (itype == 2) {

/*        Upper triangular matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = min(j,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L60: */
	    }
/* L70: */
	}

    } else if (itype == 3) {

/*        Upper Hessenberg matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = j + 1;
	    i__2 = min(i__3,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L80: */
	    }
/* L90: */
	}

    } else if (itype == 4) {

/*        Lower half of a symmetric band matrix */

	k3 = *kl + 1;
	k4 = *n + 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = k3, i__4 = k4 - j;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L100: */
	    }
/* L110: */
	}

    } else if (itype == 5) {

/*        Upper half of a symmetric band matrix */

	k1 = *ku + 2;
	k3 = *ku + 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    i__2 = k1 - j;
	    i__3 = k3;
	    for (i__ = max(i__2,1); i__ <= i__3; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L120: */
	    }
/* L130: */
	}

    } else if (itype == 6) {

/*        Band matrix */

	k1 = *kl + *ku + 2;
	k2 = *kl + 1;
	k3 = (*kl << 1) + *ku + 1;
	k4 = *kl + *ku + 1 + *m;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    i__3 = k1 - j;
/* Computing MIN */
	    i__4 = k3, i__5 = k4 - j;
	    i__2 = min(i__4,i__5);
	    for (i__ = max(i__3,k2); i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
/* L140: */
	    }
/* L150: */
	}

    }

    if (! done) {
	goto L10;
    }

    return 0;

/*     End of SLASCL */

} /* slascl_ */

/* Subroutine */ int slasd0_(integer *n, integer *sqre, real *d__, real *e,
	real *u, integer *ldu, real *vt, integer *ldvt, integer *smlsiz,
	integer *iwork, real *work, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, m, i1, ic, lf, nd, ll, nl, nr, im1, ncc, nlf, nrf,
	    iwk, lvl, ndb1, nlp1, nrp1;
    static real beta;
    static integer idxq, nlvl;
    static real alpha;
    static integer inode, ndiml, idxqc, ndimr, itemp, sqrei;
    extern /* Subroutine */ int slasd1_(integer *, integer *, integer *, real
	    *, real *, real *, real *, integer *, real *, integer *, integer *
	    , integer *, real *, integer *), xerbla_(char *, integer *), slasdq_(char *, integer *, integer *, integer *, integer
	    *, integer *, real *, real *, real *, integer *, real *, integer *
	    , real *, integer *, real *, integer *), slasdt_(integer *
	    , integer *, integer *, integer *, integer *, integer *, integer *
	    );


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    Using a divide and conquer approach, SLASD0 computes the singular
    value decomposition (SVD) of a real upper bidiagonal N-by-M
    matrix B with diagonal D and offdiagonal E, where M = N + SQRE.
    The algorithm computes orthogonal matrices U and VT such that
    B = U * S * VT. The singular values S are overwritten on D.

    A related subroutine, SLASDA, computes only the singular values,
    and optionally, the singular vectors in compact form.

    Arguments
    =========

    N      (input) INTEGER
           On entry, the row dimension of the upper bidiagonal matrix.
           This is also the dimension of the main diagonal array D.

    SQRE   (input) INTEGER
           Specifies the column dimension of the bidiagonal matrix.
           = 0: The bidiagonal matrix has column dimension M = N;
           = 1: The bidiagonal matrix has column dimension M = N+1;

    D      (input/output) REAL array, dimension (N)
           On entry D contains the main diagonal of the bidiagonal
           matrix.
           On exit D, if INFO = 0, contains its singular values.

    E      (input) REAL array, dimension (M-1)
           Contains the subdiagonal entries of the bidiagonal matrix.
           On exit, E has been destroyed.

    U      (output) REAL array, dimension at least (LDQ, N)
           On exit, U contains the left singular vectors.

    LDU    (input) INTEGER
           On entry, leading dimension of U.

    VT     (output) REAL array, dimension at least (LDVT, M)
           On exit, VT' contains the right singular vectors.

    LDVT   (input) INTEGER
           On entry, leading dimension of VT.

    SMLSIZ (input) INTEGER
           On entry, maximum size of the subproblems at the
           bottom of the computation tree.

    IWORK  INTEGER work array.
           Dimension must be at least (8 * N)

    WORK   REAL work array.
           Dimension must be at least (3 * M**2 + 2 * M)

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --iwork;
    --work;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -1;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -2;
    }

    m = *n + *sqre;

    if (*ldu < *n) {
	*info = -6;
    } else if (*ldvt < m) {
	*info = -8;
    } else if (*smlsiz < 3) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD0", &i__1);
	return 0;
    }

/*     If the input matrix is too small, call SLASDQ to find the SVD. */

    if (*n <= *smlsiz) {
	slasdq_("U", sqre, n, &m, n, &c__0, &d__[1], &e[1], &vt[vt_offset],
		ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[1], info);
	return 0;
    }

/*     Set up the computation tree. */

    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;
    idxq = ndimr + *n;
    iwk = idxq + *n;
    slasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr],
	    smlsiz);

/*
       For the nodes on bottom level of the tree, solve
       their subproblems by SLASDQ.
*/

    ndb1 = (nd + 1) / 2;
    ncc = 0;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {

/*
       IC : center row of each node
       NL : number of rows of left  subproblem
       NR : number of rows of right subproblem
       NLF: starting row of the left   subproblem
       NRF: starting row of the right  subproblem
*/

	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nlp1 = nl + 1;
	nr = iwork[ndimr + i1];
	nrp1 = nr + 1;
	nlf = ic - nl;
	nrf = ic + 1;
	sqrei = 1;
	slasdq_("U", &sqrei, &nl, &nlp1, &nl, &ncc, &d__[nlf], &e[nlf], &vt[
		nlf + nlf * vt_dim1], ldvt, &u[nlf + nlf * u_dim1], ldu, &u[
		nlf + nlf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + nlf - 2;
	i__2 = nl;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j] = j;
/* L10: */
	}
	if (i__ == nd) {
	    sqrei = *sqre;
	} else {
	    sqrei = 1;
	}
	nrp1 = nr + sqrei;
	slasdq_("U", &sqrei, &nr, &nrp1, &nr, &ncc, &d__[nrf], &e[nrf], &vt[
		nrf + nrf * vt_dim1], ldvt, &u[nrf + nrf * u_dim1], ldu, &u[
		nrf + nrf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + ic;
	i__2 = nr;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j - 1] = j;
/* L20: */
	}
/* L30: */
    }

/*     Now conquer each subproblem bottom-up. */

    for (lvl = nlvl; lvl >= 1; --lvl) {

/*
          Find the first node LF and last node LL on the
          current level LVL.
*/

	if (lvl == 1) {
	    lf = 1;
	    ll = 1;
	} else {
	    i__1 = lvl - 1;
	    lf = pow_ii(&c__2, &i__1);
	    ll = (lf << 1) - 1;
	}
	i__1 = ll;
	for (i__ = lf; i__ <= i__1; ++i__) {
	    im1 = i__ - 1;
	    ic = iwork[inode + im1];
	    nl = iwork[ndiml + im1];
	    nr = iwork[ndimr + im1];
	    nlf = ic - nl;
	    if (*sqre == 0 && i__ == ll) {
		sqrei = *sqre;
	    } else {
		sqrei = 1;
	    }
	    idxqc = idxq + nlf - 1;
	    alpha = d__[ic];
	    beta = e[ic];
	    slasd1_(&nl, &nr, &sqrei, &d__[nlf], &alpha, &beta, &u[nlf + nlf *
		     u_dim1], ldu, &vt[nlf + nlf * vt_dim1], ldvt, &iwork[
		    idxqc], &iwork[iwk], &work[1], info);
	    if (*info != 0) {
		return 0;
	    }
/* L40: */
	}
/* L50: */
    }

    return 0;

/*     End of SLASD0 */

} /* slasd0_ */

/* Subroutine */ int slasd1_(integer *nl, integer *nr, integer *sqre, real *
	d__, real *alpha, real *beta, real *u, integer *ldu, real *vt,
	integer *ldvt, integer *idxq, integer *iwork, real *work, integer *
	info)
{
    /* System generated locals */
    integer u_dim1, u_offset, vt_dim1, vt_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    static integer i__, k, m, n, n1, n2, iq, iz, iu2, ldq, idx, ldu2, ivt2,
	    idxc, idxp, ldvt2;
    extern /* Subroutine */ int slasd2_(integer *, integer *, integer *,
	    integer *, real *, real *, real *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *, real *, integer *,
	    integer *, integer *, integer *, integer *, integer *, integer *),
	     slasd3_(integer *, integer *, integer *, integer *, real *, real
	    *, integer *, real *, real *, integer *, real *, integer *, real *
	    , integer *, real *, integer *, integer *, integer *, real *,
	    integer *);
    static integer isigma;
    extern /* Subroutine */ int xerbla_(char *, integer *), slascl_(
	    char *, integer *, integer *, real *, real *, integer *, integer *
	    , real *, integer *, integer *), slamrg_(integer *,
	    integer *, real *, integer *, integer *, integer *);
    static real orgnrm;
    static integer coltyp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLASD1 computes the SVD of an upper bidiagonal N-by-M matrix B,
    where N = NL + NR + 1 and M = N + SQRE. SLASD1 is called from SLASD0.

    A related subroutine SLASD7 handles the case in which the singular
    values (and the singular vectors in factored form) are desired.

    SLASD1 computes the SVD as follows:

                  ( D1(in)  0    0     0 )
      B = U(in) * (   Z1'   a   Z2'    b ) * VT(in)
                  (   0     0   D2(in) 0 )

        = U(out) * ( D(out) 0) * VT(out)

    where Z' = (Z1' a Z2' b) = u' VT', and u is a vector of dimension M
    with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
    elsewhere; and the entry b is empty if SQRE = 0.

    The left singular vectors of the original matrix are stored in U, and
    the transpose of the right singular vectors are stored in VT, and the
    singular values are in D.  The algorithm consists of three stages:

       The first stage consists of deflating the size of the problem
       when there are multiple singular values or when there are zeros in
       the Z vector.  For each such occurence the dimension of the
       secular equation problem is reduced by one.  This stage is
       performed by the routine SLASD2.

       The second stage consists of calculating the updated
       singular values. This is done by finding the square roots of the
       roots of the secular equation via the routine SLASD4 (as called
       by SLASD3). This routine also calculates the singular vectors of
       the current problem.

       The final stage consists of computing the updated singular vectors
       directly using the updated singular values.  The singular vectors
       for the current problem are multiplied with the singular vectors
       from the overall problem.

    Arguments
    =========

    NL     (input) INTEGER
           The row dimension of the upper block.  NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block.  NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has row dimension N = NL + NR + 1,
           and column dimension M = N + SQRE.

    D      (input/output) REAL array,
                          dimension (N = NL+NR+1).
           On entry D(1:NL,1:NL) contains the singular values of the
           upper block; and D(NL+2:N) contains the singular values of
           the lower block. On exit D(1:N) contains the singular values
           of the modified matrix.

    ALPHA  (input) REAL
           Contains the diagonal element associated with the added row.

    BETA   (input) REAL
           Contains the off-diagonal element associated with the added
           row.

    U      (input/output) REAL array, dimension(LDU,N)
           On entry U(1:NL, 1:NL) contains the left singular vectors of
           the upper block; U(NL+2:N, NL+2:N) contains the left singular
           vectors of the lower block. On exit U contains the left
           singular vectors of the bidiagonal matrix.

    LDU    (input) INTEGER
           The leading dimension of the array U.  LDU >= max( 1, N ).

    VT     (input/output) REAL array, dimension(LDVT,M)
           where M = N + SQRE.
           On entry VT(1:NL+1, 1:NL+1)' contains the right singular
           vectors of the upper block; VT(NL+2:M, NL+2:M)' contains
           the right singular vectors of the lower block. On exit
           VT' contains the right singular vectors of the
           bidiagonal matrix.

    LDVT   (input) INTEGER
           The leading dimension of the array VT.  LDVT >= max( 1, M ).

    IDXQ  (output) INTEGER array, dimension(N)
           This contains the permutation which will reintegrate the
           subproblem just solved back into sorted order, i.e.
           D( IDXQ( I = 1, N ) ) will be in ascending order.

    IWORK  (workspace) INTEGER array, dimension( 4 * N )

    WORK   (workspace) REAL array, dimension( 3*M**2 + 2*M )

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --idxq;
    --iwork;
    --work;

    /* Function Body */
    *info = 0;

    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD1", &i__1);
	return 0;
    }

    n = *nl + *nr + 1;
    m = n + *sqre;

/*
       The following values are for bookkeeping purposes only.  They are
       integer pointers which indicate the portion of the workspace
       used by a particular array in SLASD2 and SLASD3.
*/

    ldu2 = n;
    ldvt2 = m;

    iz = 1;
    isigma = iz + m;
    iu2 = isigma + n;
    ivt2 = iu2 + ldu2 * n;
    iq = ivt2 + ldvt2 * m;

    idx = 1;
    idxc = idx + n;
    coltyp = idxc + n;
    idxp = coltyp + n;

/*
       Scale.

   Computing MAX
*/
    r__1 = dabs(*alpha), r__2 = dabs(*beta);
    orgnrm = dmax(r__1,r__2);
    d__[*nl + 1] = 0.f;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = d__[i__], dabs(r__1)) > orgnrm) {
	    orgnrm = (r__1 = d__[i__], dabs(r__1));
	}
/* L10: */
    }
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &n, &c__1, &d__[1], &n, info);
    *alpha /= orgnrm;
    *beta /= orgnrm;

/*     Deflate singular values. */

    slasd2_(nl, nr, sqre, &k, &d__[1], &work[iz], alpha, beta, &u[u_offset],
	    ldu, &vt[vt_offset], ldvt, &work[isigma], &work[iu2], &ldu2, &
	    work[ivt2], &ldvt2, &iwork[idxp], &iwork[idx], &iwork[idxc], &
	    idxq[1], &iwork[coltyp], info);

/*     Solve Secular Equation and update singular vectors. */

    ldq = k;
    slasd3_(nl, nr, sqre, &k, &d__[1], &work[iq], &ldq, &work[isigma], &u[
	    u_offset], ldu, &work[iu2], &ldu2, &vt[vt_offset], ldvt, &work[
	    ivt2], &ldvt2, &iwork[idxc], &iwork[coltyp], &work[iz], info);
    if (*info != 0) {
	return 0;
    }

/*     Unscale. */

    slascl_("G", &c__0, &c__0, &c_b15, &orgnrm, &n, &c__1, &d__[1], &n, info);

/*     Prepare the IDXQ sorting permutation. */

    n1 = k;
    n2 = n - k;
    slamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &idxq[1]);

    return 0;

/*     End of SLASD1 */

} /* slasd1_ */

/* Subroutine */ int slasd2_(integer *nl, integer *nr, integer *sqre, integer
	*k, real *d__, real *z__, real *alpha, real *beta, real *u, integer *
	ldu, real *vt, integer *ldvt, real *dsigma, real *u2, integer *ldu2,
	real *vt2, integer *ldvt2, integer *idxp, integer *idx, integer *idxc,
	 integer *idxq, integer *coltyp, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1, vt_offset,
	    vt2_dim1, vt2_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    static real c__;
    static integer i__, j, m, n;
    static real s;
    static integer k2;
    static real z1;
    static integer ct, jp;
    static real eps, tau, tol;
    static integer psm[4], nlp1, nlp2, idxi, idxj, ctot[4];
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *);
    static integer idxjp, jprev;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), slamrg_(
	    integer *, integer *, real *, integer *, integer *, integer *);
    static real hlftol;
    extern /* Subroutine */ int slacpy_(char *, integer *, integer *, real *,
	    integer *, real *, integer *), slaset_(char *, integer *,
	    integer *, real *, real *, real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       October 31, 1999


    Purpose
    =======

    SLASD2 merges the two sets of singular values together into a single
    sorted set.  Then it tries to deflate the size of the problem.
    There are two ways in which deflation can occur:  when two or more
    singular values are close together or if there is a tiny entry in the
    Z vector.  For each such occurrence the order of the related secular
    equation problem is reduced by one.

    SLASD2 is called from SLASD1.

    Arguments
    =========

    NL     (input) INTEGER
           The row dimension of the upper block.  NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block.  NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has N = NL + NR + 1 rows and
           M = N + SQRE >= N columns.

    K      (output) INTEGER
           Contains the dimension of the non-deflated matrix,
           This is the order of the related secular equation. 1 <= K <=N.

    D      (input/output) REAL array, dimension(N)
           On entry D contains the singular values of the two submatrices
           to be combined.  On exit D contains the trailing (N-K) updated
           singular values (those which were deflated) sorted into
           increasing order.

    ALPHA  (input) REAL
           Contains the diagonal element associated with the added row.

    BETA   (input) REAL
           Contains the off-diagonal element associated with the added
           row.

    U      (input/output) REAL array, dimension(LDU,N)
           On entry U contains the left singular vectors of two
           submatrices in the two square blocks with corners at (1,1),
           (NL, NL), and (NL+2, NL+2), (N,N).
           On exit U contains the trailing (N-K) updated left singular
           vectors (those which were deflated) in its last N-K columns.

    LDU    (input) INTEGER
           The leading dimension of the array U.  LDU >= N.

    Z      (output) REAL array, dimension(N)
           On exit Z contains the updating row vector in the secular
           equation.

    DSIGMA (output) REAL array, dimension (N)
           Contains a copy of the diagonal elements (K-1 singular values
           and one zero) in the secular equation.

    U2     (output) REAL array, dimension(LDU2,N)
           Contains a copy of the first K-1 left singular vectors which
           will be used by SLASD3 in a matrix multiply (SGEMM) to solve
           for the new left singular vectors. U2 is arranged into four
           blocks. The first block contains a column with 1 at NL+1 and
           zero everywhere else; the second block contains non-zero
           entries only at and above NL; the third contains non-zero
           entries only below NL+1; and the fourth is dense.

    LDU2   (input) INTEGER
           The leading dimension of the array U2.  LDU2 >= N.

    VT     (input/output) REAL array, dimension(LDVT,M)
           On entry VT' contains the right singular vectors of two
           submatrices in the two square blocks with corners at (1,1),
           (NL+1, NL+1), and (NL+2, NL+2), (M,M).
           On exit VT' contains the trailing (N-K) updated right singular
           vectors (those which were deflated) in its last N-K columns.
           In case SQRE =1, the last row of VT spans the right null
           space.

    LDVT   (input) INTEGER
           The leading dimension of the array VT.  LDVT >= M.

    VT2    (output) REAL array, dimension(LDVT2,N)
           VT2' contains a copy of the first K right singular vectors
           which will be used by SLASD3 in a matrix multiply (SGEMM) to
           solve for the new right singular vectors. VT2 is arranged into
           three blocks. The first block contains a row that corresponds
           to the special 0 diagonal element in SIGMA; the second block
           contains non-zeros only at and before NL +1; the third block
           contains non-zeros only at and after  NL +2.

    LDVT2  (input) INTEGER
           The leading dimension of the array VT2.  LDVT2 >= M.

    IDXP   (workspace) INTEGER array, dimension(N)
           This will contain the permutation used to place deflated
           values of D at the end of the array. On output IDXP(2:K)
           points to the nondeflated D-values and IDXP(K+1:N)
           points to the deflated singular values.

    IDX    (workspace) INTEGER array, dimension(N)
           This will contain the permutation used to sort the contents of
           D into ascending order.

    IDXC   (output) INTEGER array, dimension(N)
           This will contain the permutation used to arrange the columns
           of the deflated U matrix into three groups:  the first group
           contains non-zero entries only at and above NL, the second
           contains non-zero entries only below NL+2, and the third is
           dense.

    COLTYP (workspace/output) INTEGER array, dimension(N)
           As workspace, this will contain a label which will indicate
           which of the following types a column in the U2 matrix or a
           row in the VT2 matrix is:
           1 : non-zero in the upper half only
           2 : non-zero in the lower half only
           3 : dense
           4 : deflated

           On exit, it is an array of dimension 4, with COLTYP(I) being
           the dimension of the I-th type columns.

    IDXQ   (input) INTEGER array, dimension(N)
           This contains the permutation which separately sorts the two
           sub-problems in D into ascending order.  Note that entries in
           the first hlaf of this permutation must first be moved one
           position backward; and entries in the second half
           must first have NL+1 added to their values.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --z__;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --dsigma;
    u2_dim1 = *ldu2;
    u2_offset = 1 + u2_dim1;
    u2 -= u2_offset;
    vt2_dim1 = *ldvt2;
    vt2_offset = 1 + vt2_dim1;
    vt2 -= vt2_offset;
    --idxp;
    --idx;
    --idxc;
    --idxq;
    --coltyp;

    /* Function Body */
    *info = 0;

    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre != 1 && *sqre != 0) {
	*info = -3;
    }

    n = *nl + *nr + 1;
    m = n + *sqre;

    if (*ldu < n) {
	*info = -10;
    } else if (*ldvt < m) {
	*info = -12;
    } else if (*ldu2 < n) {
	*info = -15;
    } else if (*ldvt2 < m) {
	*info = -17;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD2", &i__1);
	return 0;
    }

    nlp1 = *nl + 1;
    nlp2 = *nl + 2;

/*
       Generate the first part of the vector Z; and move the singular
       values in the first part of D one position backward.
*/

    z1 = *alpha * vt[nlp1 + nlp1 * vt_dim1];
    z__[1] = z1;
    for (i__ = *nl; i__ >= 1; --i__) {
	z__[i__ + 1] = *alpha * vt[i__ + nlp1 * vt_dim1];
	d__[i__ + 1] = d__[i__];
	idxq[i__ + 1] = idxq[i__] + 1;
/* L10: */
    }

/*     Generate the second part of the vector Z. */

    i__1 = m;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	z__[i__] = *beta * vt[i__ + nlp2 * vt_dim1];
/* L20: */
    }

/*     Initialize some reference arrays. */

    i__1 = nlp1;
    for (i__ = 2; i__ <= i__1; ++i__) {
	coltyp[i__] = 1;
/* L30: */
    }
    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	coltyp[i__] = 2;
/* L40: */
    }

/*     Sort the singular values into increasing order */

    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	idxq[i__] += nlp1;
/* L50: */
    }

/*
       DSIGMA, IDXC, IDXC, and the first column of U2
       are used as storage space.
*/

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	dsigma[i__] = d__[idxq[i__]];
	u2[i__ + u2_dim1] = z__[idxq[i__]];
	idxc[i__] = coltyp[idxq[i__]];
/* L60: */
    }

    slamrg_(nl, nr, &dsigma[2], &c__1, &c__1, &idx[2]);

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	idxi = idx[i__] + 1;
	d__[i__] = dsigma[idxi];
	z__[i__] = u2[idxi + u2_dim1];
	coltyp[i__] = idxc[idxi];
/* L70: */
    }

/*     Calculate the allowable deflation tolerance */

    eps = slamch_("Epsilon");
/* Computing MAX */
    r__1 = dabs(*alpha), r__2 = dabs(*beta);
    tol = dmax(r__1,r__2);
/* Computing MAX */
    r__2 = (r__1 = d__[n], dabs(r__1));
    tol = eps * 8.f * dmax(r__2,tol);

/*
       There are 2 kinds of deflation -- first a value in the z-vector
       is small, second two (or more) singular values are very close
       together (their difference is small).

       If the value in the z-vector is small, we simply permute the
       array so that the corresponding singular value is moved to the
       end.

       If two values in the D-vector are close, we perform a two-sided
       rotation designed to make one of the corresponding z-vector
       entries zero, and then permute the array so that the deflated
       singular value is moved to the end.

       If there are multiple singular values then the problem deflates.
       Here the number of equal singular values are found.  As each equal
       singular value is found, an elementary reflector is computed to
       rotate the corresponding singular subspace so that the
       corresponding components of Z are zero in this new basis.
*/

    *k = 1;
    k2 = n + 1;
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	if ((r__1 = z__[j], dabs(r__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    idxp[k2] = j;
	    coltyp[j] = 4;
	    if (j == n) {
		goto L120;
	    }
	} else {
	    jprev = j;
	    goto L90;
	}
/* L80: */
    }
L90:
    j = jprev;
L100:
    ++j;
    if (j > n) {
	goto L110;
    }
    if ((r__1 = z__[j], dabs(r__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	idxp[k2] = j;
	coltyp[j] = 4;
    } else {

/*        Check if singular values are close enough to allow deflation. */

	if ((r__1 = d__[j] - d__[jprev], dabs(r__1)) <= tol) {

/*           Deflation is possible. */

	    s = z__[jprev];
	    c__ = z__[j];

/*
             Find sqrt(a**2+b**2) without overflow or
             destructive underflow.
*/

	    tau = slapy2_(&c__, &s);
	    c__ /= tau;
	    s = -s / tau;
	    z__[j] = tau;
	    z__[jprev] = 0.f;

/*
             Apply back the Givens rotation to the left and right
             singular vector matrices.
*/

	    idxjp = idxq[idx[jprev] + 1];
	    idxj = idxq[idx[j] + 1];
	    if (idxjp <= nlp1) {
		--idxjp;
	    }
	    if (idxj <= nlp1) {
		--idxj;
	    }
	    srot_(&n, &u[idxjp * u_dim1 + 1], &c__1, &u[idxj * u_dim1 + 1], &
		    c__1, &c__, &s);
	    srot_(&m, &vt[idxjp + vt_dim1], ldvt, &vt[idxj + vt_dim1], ldvt, &
		    c__, &s);
	    if (coltyp[j] != coltyp[jprev]) {
		coltyp[j] = 3;
	    }
	    coltyp[jprev] = 4;
	    --k2;
	    idxp[k2] = jprev;
	    jprev = j;
	} else {
	    ++(*k);
	    u2[*k + u2_dim1] = z__[jprev];
	    dsigma[*k] = d__[jprev];
	    idxp[*k] = jprev;
	    jprev = j;
	}
    }
    goto L100;
L110:

/*     Record the last singular value. */

    ++(*k);
    u2[*k + u2_dim1] = z__[jprev];
    dsigma[*k] = d__[jprev];
    idxp[*k] = jprev;

L120:

/*
       Count up the total number of the various types of columns, then
       form a permutation which positions the four column types into
       four groups of uniform structure (although one or more of these
       groups may be empty).
*/

    for (j = 1; j <= 4; ++j) {
	ctot[j - 1] = 0;
/* L130: */
    }
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	ct = coltyp[j];
	++ctot[ct - 1];
/* L140: */
    }

/*     PSM(*) = Position in SubMatrix (of types 1 through 4) */

    psm[0] = 2;
    psm[1] = ctot[0] + 2;
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];

/*
       Fill out the IDXC array so that the permutation which it induces
       will place all type-1 columns first, all type-2 columns next,
       then all type-3's, and finally all type-4's, starting from the
       second column. This applies similarly to the rows of VT.
*/

    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	ct = coltyp[jp];
	idxc[psm[ct - 1]] = j;
	++psm[ct - 1];
/* L150: */
    }

/*
       Sort the singular values and corresponding singular vectors into
       DSIGMA, U2, and VT2 respectively.  The singular values/vectors
       which were not deflated go into the first K slots of DSIGMA, U2,
       and VT2 respectively, while those which were deflated go into the
       last N - K slots, except that the first column/row will be treated
       separately.
*/

    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	dsigma[j] = d__[jp];
	idxj = idxq[idx[idxp[idxc[j]]] + 1];
	if (idxj <= nlp1) {
	    --idxj;
	}
	scopy_(&n, &u[idxj * u_dim1 + 1], &c__1, &u2[j * u2_dim1 + 1], &c__1);
	scopy_(&m, &vt[idxj + vt_dim1], ldvt, &vt2[j + vt2_dim1], ldvt2);
/* L160: */
    }

/*     Determine DSIGMA(1), DSIGMA(2) and Z(1) */

    dsigma[1] = 0.f;
    hlftol = tol / 2.f;
    if (dabs(dsigma[2]) <= hlftol) {
	dsigma[2] = hlftol;
    }
    if (m > n) {
	z__[1] = slapy2_(&z1, &z__[m]);
	if (z__[1] <= tol) {
	    c__ = 1.f;
	    s = 0.f;
	    z__[1] = tol;
	} else {
	    c__ = z1 / z__[1];
	    s = z__[m] / z__[1];
	}
    } else {
	if (dabs(z1) <= tol) {
	    z__[1] = tol;
	} else {
	    z__[1] = z1;
	}
    }

/*     Move the rest of the updating row to Z. */

    i__1 = *k - 1;
    scopy_(&i__1, &u2[u2_dim1 + 2], &c__1, &z__[2], &c__1);

/*
       Determine the first column of U2, the first row of VT2 and the
       last row of VT.
*/

    slaset_("A", &n, &c__1, &c_b29, &c_b29, &u2[u2_offset], ldu2);
    u2[nlp1 + u2_dim1] = 1.f;
    if (m > n) {
	i__1 = nlp1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    vt[m + i__ * vt_dim1] = -s * vt[nlp1 + i__ * vt_dim1];
	    vt2[i__ * vt2_dim1 + 1] = c__ * vt[nlp1 + i__ * vt_dim1];
/* L170: */
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[i__ * vt2_dim1 + 1] = s * vt[m + i__ * vt_dim1];
	    vt[m + i__ * vt_dim1] = c__ * vt[m + i__ * vt_dim1];
/* L180: */
	}
    } else {
	scopy_(&m, &vt[nlp1 + vt_dim1], ldvt, &vt2[vt2_dim1 + 1], ldvt2);
    }
    if (m > n) {
	scopy_(&m, &vt[m + vt_dim1], ldvt, &vt2[m + vt2_dim1], ldvt2);
    }

/*
       The deflated singular values and their corresponding vectors go
       into the back of D, U, and V respectively.
*/

    if (n > *k) {
	i__1 = n - *k;
	scopy_(&i__1, &dsigma[*k + 1], &c__1, &d__[*k + 1], &c__1);
	i__1 = n - *k;
	slacpy_("A", &n, &i__1, &u2[(*k + 1) * u2_dim1 + 1], ldu2, &u[(*k + 1)
		 * u_dim1 + 1], ldu);
	i__1 = n - *k;
	slacpy_("A", &i__1, &m, &vt2[*k + 1 + vt2_dim1], ldvt2, &vt[*k + 1 +
		vt_dim1], ldvt);
    }

/*     Copy CTOT into COLTYP for referencing in SLASD3. */

    for (j = 1; j <= 4; ++j) {
	coltyp[j] = ctot[j - 1];
/* L190: */
    }

    return 0;

/*     End of SLASD2 */

} /* slasd2_ */

/* Subroutine */ int slasd3_(integer *nl, integer *nr, integer *sqre, integer
	*k, real *d__, real *q, integer *ldq, real *dsigma, real *u, integer *
	ldu, real *u2, integer *ldu2, real *vt, integer *ldvt, real *vt2,
	integer *ldvt2, integer *idxc, integer *ctot, real *z__, integer *
	info)
{
    /* System generated locals */
    integer q_dim1, q_offset, u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1,
	    vt_offset, vt2_dim1, vt2_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static integer i__, j, m, n, jc;
    static real rho;
    static integer nlp1, nlp2, nrp1;
    static real temp;
    extern doublereal snrm2_(integer *, real *, integer *);
    static integer ctemp;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer ktemp;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int slasd4_(integer *, integer *, real *, real *,
	    real *, real *, real *, real *, integer *), xerbla_(char *,
	    integer *), slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *), slacpy_(char *, integer *, integer *, real *, integer *,
	    real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       October 31, 1999


    Purpose
    =======

    SLASD3 finds all the square roots of the roots of the secular
    equation, as defined by the values in D and Z.  It makes the
    appropriate calls to SLASD4 and then updates the singular
    vectors by matrix multiplication.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    SLASD3 is called from SLASD1.

    Arguments
    =========

    NL     (input) INTEGER
           The row dimension of the upper block.  NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block.  NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has N = NL + NR + 1 rows and
           M = N + SQRE >= N columns.

    K      (input) INTEGER
           The size of the secular equation, 1 =< K = < N.

    D      (output) REAL array, dimension(K)
           On exit the square roots of the roots of the secular equation,
           in ascending order.

    Q      (workspace) REAL array,
                       dimension at least (LDQ,K).

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= K.

    DSIGMA (input) REAL array, dimension(K)
           The first K elements of this array contain the old roots
           of the deflated updating problem.  These are the poles
           of the secular equation.

    U      (input) REAL array, dimension (LDU, N)
           The last N - K columns of this matrix contain the deflated
           left singular vectors.

    LDU    (input) INTEGER
           The leading dimension of the array U.  LDU >= N.

    U2     (input) REAL array, dimension (LDU2, N)
           The first K columns of this matrix contain the non-deflated
           left singular vectors for the split problem.

    LDU2   (input) INTEGER
           The leading dimension of the array U2.  LDU2 >= N.

    VT     (input) REAL array, dimension (LDVT, M)
           The last M - K columns of VT' contain the deflated
           right singular vectors.

    LDVT   (input) INTEGER
           The leading dimension of the array VT.  LDVT >= N.

    VT2    (input) REAL array, dimension (LDVT2, N)
           The first K columns of VT2' contain the non-deflated
           right singular vectors for the split problem.

    LDVT2  (input) INTEGER
           The leading dimension of the array VT2.  LDVT2 >= N.

    IDXC   (input) INTEGER array, dimension ( N )
           The permutation used to arrange the columns of U (and rows of
           VT) into three groups:  the first group contains non-zero
           entries only at and above (or before) NL +1; the second
           contains non-zero entries only at and below (or after) NL+2;
           and the third is dense. The first column of U and the row of
           VT are treated separately, however.

           The rows of the singular vectors found by SLASD4
           must be likewise permuted before the matrix multiplies can
           take place.

    CTOT   (input) INTEGER array, dimension ( 4 )
           A count of the total number of the various types of columns
           in U (or rows in VT), as described in IDXC. The fourth column
           type is any column which has been deflated.

    Z      (input) REAL array, dimension (K)
           The first K elements of this array contain the components
           of the deflation-adjusted updating row vector.

    INFO   (output) INTEGER
           = 0:  successful exit.
           < 0:  if INFO = -i, the i-th argument had an illegal value.
           > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dsigma;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    u2_dim1 = *ldu2;
    u2_offset = 1 + u2_dim1;
    u2 -= u2_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    vt2_dim1 = *ldvt2;
    vt2_offset = 1 + vt2_dim1;
    vt2 -= vt2_offset;
    --idxc;
    --ctot;
    --z__;

    /* Function Body */
    *info = 0;

    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre != 1 && *sqre != 0) {
	*info = -3;
    }

    n = *nl + *nr + 1;
    m = n + *sqre;
    nlp1 = *nl + 1;
    nlp2 = *nl + 2;

    if (*k < 1 || *k > n) {
	*info = -4;
    } else if (*ldq < *k) {
	*info = -7;
    } else if (*ldu < n) {
	*info = -10;
    } else if (*ldu2 < n) {
	*info = -12;
    } else if (*ldvt < m) {
	*info = -14;
    } else if (*ldvt2 < m) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD3", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 1) {
	d__[1] = dabs(z__[1]);
	scopy_(&m, &vt2[vt2_dim1 + 1], ldvt2, &vt[vt_dim1 + 1], ldvt);
	if (z__[1] > 0.f) {
	    scopy_(&n, &u2[u2_dim1 + 1], &c__1, &u[u_dim1 + 1], &c__1);
	} else {
	    i__1 = n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		u[i__ + u_dim1] = -u2[i__ + u2_dim1];
/* L10: */
	    }
	}
	return 0;
    }

/*
       Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
       be computed with high relative accuracy (barring over/underflow).
       This is a problem on machines without a guard digit in
       add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
       The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
       which on any of these machines zeros out the bottommost
       bit of DSIGMA(I) if it is 1; this makes the subsequent
       subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
       occurs. On binary machines with a guard digit (almost all
       machines) it does not change DSIGMA(I) at all. On hexadecimal
       and decimal machines with a guard digit, it slightly
       changes the bottommost bits of DSIGMA(I). It does not account
       for hexadecimal or decimal machines without guard digits
       (we know of none). We use a subroutine call to compute
       2*DLAMBDA(I) to prevent optimizing compilers from eliminating
       this code.
*/

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = slamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
/* L20: */
    }

/*     Keep a copy of Z. */

    scopy_(k, &z__[1], &c__1, &q[q_offset], &c__1);

/*     Normalize Z. */

    rho = snrm2_(k, &z__[1], &c__1);
    slascl_("G", &c__0, &c__0, &rho, &c_b15, k, &c__1, &z__[1], k, info);
    rho *= rho;

/*     Find the new singular values. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	slasd4_(k, &j, &dsigma[1], &z__[1], &u[j * u_dim1 + 1], &rho, &d__[j],
		 &vt[j * vt_dim1 + 1], info);

/*        If the zero finder fails, the computation is terminated. */

	if (*info != 0) {
	    return 0;
	}
/* L30: */
    }

/*     Compute updated Z. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__[i__] = u[i__ + *k * u_dim1] * vt[i__ + *k * vt_dim1];
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j]) / (dsigma[i__] + dsigma[j]);
/* L40: */
	}
	i__2 = *k - 1;
	for (j = i__; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j + 1]) / (dsigma[i__] + dsigma[j + 1]);
/* L50: */
	}
	r__2 = sqrt((r__1 = z__[i__], dabs(r__1)));
	z__[i__] = r_sign(&r__2, &q[i__ + q_dim1]);
/* L60: */
    }

/*
       Compute left singular vectors of the modified diagonal matrix,
       and store related information for the right singular vectors.
*/

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	vt[i__ * vt_dim1 + 1] = z__[1] / u[i__ * u_dim1 + 1] / vt[i__ *
		vt_dim1 + 1];
	u[i__ * u_dim1 + 1] = -1.f;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    vt[j + i__ * vt_dim1] = z__[j] / u[j + i__ * u_dim1] / vt[j + i__
		    * vt_dim1];
	    u[j + i__ * u_dim1] = dsigma[j] * vt[j + i__ * vt_dim1];
/* L70: */
	}
	temp = snrm2_(k, &u[i__ * u_dim1 + 1], &c__1);
	q[i__ * q_dim1 + 1] = u[i__ * u_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[j + i__ * q_dim1] = u[jc + i__ * u_dim1] / temp;
/* L80: */
	}
/* L90: */
    }

/*     Update the left singular vector matrix. */

    if (*k == 2) {
	sgemm_("N", "N", &n, k, k, &c_b15, &u2[u2_offset], ldu2, &q[q_offset],
		 ldq, &c_b29, &u[u_offset], ldu);
	goto L100;
    }
    if (ctot[1] > 0) {
	sgemm_("N", "N", nl, k, &ctot[1], &c_b15, &u2[(u2_dim1 << 1) + 1],
		ldu2, &q[q_dim1 + 2], ldq, &c_b29, &u[u_dim1 + 1], ldu);
	if (ctot[3] > 0) {
	    ktemp = ctot[1] + 2 + ctot[2];
	    sgemm_("N", "N", nl, k, &ctot[3], &c_b15, &u2[ktemp * u2_dim1 + 1]
		    , ldu2, &q[ktemp + q_dim1], ldq, &c_b15, &u[u_dim1 + 1],
		    ldu);
	}
    } else if (ctot[3] > 0) {
	ktemp = ctot[1] + 2 + ctot[2];
	sgemm_("N", "N", nl, k, &ctot[3], &c_b15, &u2[ktemp * u2_dim1 + 1],
		ldu2, &q[ktemp + q_dim1], ldq, &c_b29, &u[u_dim1 + 1], ldu);
    } else {
	slacpy_("F", nl, k, &u2[u2_offset], ldu2, &u[u_offset], ldu);
    }
    scopy_(k, &q[q_dim1 + 1], ldq, &u[nlp1 + u_dim1], ldu);
    ktemp = ctot[1] + 2;
    ctemp = ctot[2] + ctot[3];
    sgemm_("N", "N", nr, k, &ctemp, &c_b15, &u2[nlp2 + ktemp * u2_dim1], ldu2,
	     &q[ktemp + q_dim1], ldq, &c_b29, &u[nlp2 + u_dim1], ldu);

/*     Generate the right singular vectors. */

L100:
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp = snrm2_(k, &vt[i__ * vt_dim1 + 1], &c__1);
	q[i__ + q_dim1] = vt[i__ * vt_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[i__ + j * q_dim1] = vt[jc + i__ * vt_dim1] / temp;
/* L110: */
	}
/* L120: */
    }

/*     Update the right singular vector matrix. */

    if (*k == 2) {
	sgemm_("N", "N", k, &m, k, &c_b15, &q[q_offset], ldq, &vt2[vt2_offset]
		, ldvt2, &c_b29, &vt[vt_offset], ldvt);
	return 0;
    }
    ktemp = ctot[1] + 1;
    sgemm_("N", "N", k, &nlp1, &ktemp, &c_b15, &q[q_dim1 + 1], ldq, &vt2[
	    vt2_dim1 + 1], ldvt2, &c_b29, &vt[vt_dim1 + 1], ldvt);
    ktemp = ctot[1] + 2 + ctot[2];
    if (ktemp <= *ldvt2) {
	sgemm_("N", "N", k, &nlp1, &ctot[3], &c_b15, &q[ktemp * q_dim1 + 1],
		ldq, &vt2[ktemp + vt2_dim1], ldvt2, &c_b15, &vt[vt_dim1 + 1],
		ldvt);
    }

    ktemp = ctot[1] + 1;
    nrp1 = *nr + *sqre;
    if (ktemp > 1) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    q[i__ + ktemp * q_dim1] = q[i__ + q_dim1];
/* L130: */
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[ktemp + i__ * vt2_dim1] = vt2[i__ * vt2_dim1 + 1];
/* L140: */
	}
    }
    ctemp = ctot[2] + 1 + ctot[3];
    sgemm_("N", "N", k, &nrp1, &ctemp, &c_b15, &q[ktemp * q_dim1 + 1], ldq, &
	    vt2[ktemp + nlp2 * vt2_dim1], ldvt2, &c_b29, &vt[nlp2 * vt_dim1 +
	    1], ldvt);

    return 0;

/*     End of SLASD3 */

} /* slasd3_ */

/* Subroutine */ int slasd4_(integer *n, integer *i__, real *d__, real *z__,
	real *delta, real *rho, real *sigma, real *work, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real a, b, c__;
    static integer j;
    static real w, dd[3];
    static integer ii;
    static real dw, zz[3];
    static integer ip1;
    static real eta, phi, eps, tau, psi;
    static integer iim1, iip1;
    static real dphi, dpsi;
    static integer iter;
    static real temp, prew, sg2lb, sg2ub, temp1, temp2, dtiim, delsq, dtiip;
    static integer niter;
    static real dtisq;
    static logical swtch;
    static real dtnsq;
    extern /* Subroutine */ int slaed6_(integer *, logical *, real *, real *,
	    real *, real *, real *, integer *);
    static real delsq2;
    extern /* Subroutine */ int slasd5_(integer *, real *, real *, real *,
	    real *, real *, real *);
    static real dtnsq1;
    static logical swtch3;
    extern doublereal slamch_(char *);
    static logical orgati;
    static real erretm, dtipsq, rhoinv;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       October 31, 1999


    Purpose
    =======

    This subroutine computes the square root of the I-th updated
    eigenvalue of a positive symmetric rank-one modification to
    a positive diagonal matrix whose entries are given as the squares
    of the corresponding entries in the array d, and that

           0 <= D(i) < D(j)  for  i < j

    and that RHO > 0. This is arranged by the calling routine, and is
    no loss in generality.  The rank-one modified system is thus

           diag( D ) * diag( D ) +  RHO *  Z * Z_transpose.

    where we assume the Euclidean norm of Z is 1.

    The method consists of approximating the rational functions in the
    secular equation by simpler interpolating rational functions.

    Arguments
    =========

    N      (input) INTEGER
           The length of all arrays.

    I      (input) INTEGER
           The index of the eigenvalue to be computed.  1 <= I <= N.

    D      (input) REAL array, dimension ( N )
           The original eigenvalues.  It is assumed that they are in
           order, 0 <= D(I) < D(J)  for I < J.

    Z      (input) REAL array, dimension ( N )
           The components of the updating vector.

    DELTA  (output) REAL array, dimension ( N )
           If N .ne. 1, DELTA contains (D(j) - sigma_I) in its  j-th
           component.  If N = 1, then DELTA(1) = 1.  The vector DELTA
           contains the information necessary to construct the
           (singular) eigenvectors.

    RHO    (input) REAL
           The scalar in the symmetric updating formula.

    SIGMA  (output) REAL
           The computed lambda_I, the I-th updated eigenvalue.

    WORK   (workspace) REAL array, dimension ( N )
           If N .ne. 1, WORK contains (D(j) + sigma_I) in its  j-th
           component.  If N = 1, then WORK( 1 ) = 1.

    INFO   (output) INTEGER
           = 0:  successful exit
           > 0:  if INFO = 1, the updating process failed.

    Internal Parameters
    ===================

    Logical variable ORGATI (origin-at-i?) is used for distinguishing
    whether D(i) or D(i+1) is treated as the origin.

              ORGATI = .true.    origin at i
              ORGATI = .false.   origin at i+1

    Logical variable SWTCH3 (switch-for-3-poles?) is for noting
    if we are working with THREE poles!

    MAXIT is the maximum number of iterations allowed for each
    eigenvalue.

    Further Details
    ===============

    Based on contributions by
       Ren-Cang Li, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================


       Since this routine is called in an inner loop, we do no argument
       checking.

       Quick return for N=1 and 2.
*/

    /* Parameter adjustments */
    --work;
    --delta;
    --z__;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n == 1) {

/*        Presumably, I=1 upon entry */

	*sigma = sqrt(d__[1] * d__[1] + *rho * z__[1] * z__[1]);
	delta[1] = 1.f;
	work[1] = 1.f;
	return 0;
    }
    if (*n == 2) {
	slasd5_(i__, &d__[1], &z__[1], &delta[1], rho, sigma, &work[1]);
	return 0;
    }

/*     Compute machine epsilon */

    eps = slamch_("Epsilon");
    rhoinv = 1.f / *rho;

/*     The case I = N */

    if (*i__ == *n) {

/*        Initialize some basic variables */

	ii = *n - 1;
	niter = 1;

/*        Calculate initial guess */

	temp = *rho / 2.f;

/*
          If ||Z||_2 is not one, then TEMP should be set to
          RHO * ||Z||_2^2 / TWO
*/

	temp1 = temp / (d__[*n] + sqrt(d__[*n] * d__[*n] + temp));
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*n] + temp1;
	    delta[j] = d__[j] - d__[*n] - temp1;
/* L10: */
	}

	psi = 0.f;
	i__1 = *n - 2;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (delta[j] * work[j]);
/* L20: */
	}

	c__ = rhoinv + psi;
	w = c__ + z__[ii] * z__[ii] / (delta[ii] * work[ii]) + z__[*n] * z__[*
		n] / (delta[*n] * work[*n]);

	if (w <= 0.f) {
	    temp1 = sqrt(d__[*n] * d__[*n] + *rho);
	    temp = z__[*n - 1] * z__[*n - 1] / ((d__[*n - 1] + temp1) * (d__[*
		    n] - d__[*n - 1] + *rho / (d__[*n] + temp1))) + z__[*n] *
		    z__[*n] / *rho;

/*
             The following TAU is to approximate
             SIGMA_n^2 - D( N )*D( N )
*/

	    if (c__ <= temp) {
		tau = *rho;
	    } else {
		delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
		a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*
			n];
		b = z__[*n] * z__[*n] * delsq;
		if (a < 0.f) {
		    tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
		} else {
		    tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
		}
	    }

/*
             It can be proved that
                 D(N)^2+RHO/2 <= SIGMA_n^2 < D(N)^2+TAU <= D(N)^2+RHO
*/

	} else {
	    delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
	    a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n];
	    b = z__[*n] * z__[*n] * delsq;

/*
             The following TAU is to approximate
             SIGMA_n^2 - D( N )*D( N )
*/

	    if (a < 0.f) {
		tau = b * 2.f / (sqrt(a * a + b * 4.f * c__) - a);
	    } else {
		tau = (a + sqrt(a * a + b * 4.f * c__)) / (c__ * 2.f);
	    }

/*
             It can be proved that
             D(N)^2 < D(N)^2+TAU < SIGMA(N)^2 < D(N)^2+RHO/2
*/

	}

/*        The following ETA is to approximate SIGMA_n - D( N ) */

	eta = tau / (d__[*n] + sqrt(d__[*n] * d__[*n] + tau));

	*sigma = d__[*n] + eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*i__] - eta;
	    work[j] = d__[j] + d__[*i__] + eta;
/* L30: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (delta[j] * work[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L40: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / (delta[*n] * work[*n]);
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    goto L240;
	}

/*        Calculate the new step */

	++niter;
	dtnsq1 = work[*n - 1] * delta[*n - 1];
	dtnsq = work[*n] * delta[*n];
	c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
	b = dtnsq * dtnsq1 * w;
	if (c__ < 0.f) {
	    c__ = dabs(c__);
	}
	if (c__ == 0.f) {
	    eta = *rho - *sigma * *sigma;
	} else if (a >= 0.f) {
	    eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) / (
		    c__ * 2.f);
	} else {
	    eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
		    r__1))));
	}

/*
          Note, eta should be positive if w is negative, and
          eta should be negative otherwise. However,
          if for some reason caused by roundoff, eta*w > 0,
          we simply use one Newton step instead. This way
          will guarantee eta*w < 0.
*/

	if (w * eta > 0.f) {
	    eta = -w / (dpsi + dphi);
	}
	temp = eta - dtnsq;
	if (temp > *rho) {
	    eta = *rho + dtnsq;
	}

	tau += eta;
	eta /= *sigma + sqrt(eta + *sigma * *sigma);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
	    work[j] += eta;
/* L50: */
	}

	*sigma += eta;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L60: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	temp = z__[*n] / (work[*n] * delta[*n]);
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) * (
		dpsi + dphi);

	w = rhoinv + phi + psi;

/*        Main loop to update the values of the array   DELTA */

	iter = niter + 1;

	for (niter = iter; niter <= 20; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		goto L240;
	    }

/*           Calculate the new step */

	    dtnsq1 = work[*n - 1] * delta[*n - 1];
	    dtnsq = work[*n] * delta[*n];
	    c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	    a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
	    b = dtnsq1 * dtnsq * w;
	    if (a >= 0.f) {
		eta = (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }

/*
             Note, eta should be positive if w is negative, and
             eta should be negative otherwise. However,
             if for some reason caused by roundoff, eta*w > 0,
             we simply use one Newton step instead. This way
             will guarantee eta*w < 0.
*/

	    if (w * eta > 0.f) {
		eta = -w / (dpsi + dphi);
	    }
	    temp = eta - dtnsq;
	    if (temp <= 0.f) {
		eta /= 2.f;
	    }

	    tau += eta;
	    eta /= *sigma + sqrt(eta + *sigma * *sigma);
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
		work[j] += eta;
/* L70: */
	    }

	    *sigma += eta;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = ii;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L80: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    temp = z__[*n] / (work[*n] * delta[*n]);
	    phi = z__[*n] * temp;
	    dphi = temp * temp;
	    erretm = (-phi - psi) * 8.f + erretm - phi + rhoinv + dabs(tau) *
		    (dpsi + dphi);

	    w = rhoinv + phi + psi;
/* L90: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;
	goto L240;

/*        End for the case I = N */

    } else {

/*        The case for I < N */

	niter = 1;
	ip1 = *i__ + 1;

/*        Calculate initial guess */

	delsq = (d__[ip1] - d__[*i__]) * (d__[ip1] + d__[*i__]);
	delsq2 = delsq / 2.f;
	temp = delsq2 / (d__[*i__] + sqrt(d__[*i__] * d__[*i__] + delsq2));
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*i__] + temp;
	    delta[j] = d__[j] - d__[*i__] - temp;
/* L100: */
	}

	psi = 0.f;
	i__1 = *i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (work[j] * delta[j]);
/* L110: */
	}

	phi = 0.f;
	i__1 = *i__ + 2;
	for (j = *n; j >= i__1; --j) {
	    phi += z__[j] * z__[j] / (work[j] * delta[j]);
/* L120: */
	}
	c__ = rhoinv + psi + phi;
	w = c__ + z__[*i__] * z__[*i__] / (work[*i__] * delta[*i__]) + z__[
		ip1] * z__[ip1] / (work[ip1] * delta[ip1]);

	if (w > 0.f) {

/*
             d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2

             We choose d(i) as origin.
*/

	    orgati = TRUE_;
	    sg2lb = 0.f;
	    sg2ub = delsq2;
	    a = c__ * delsq + z__[*i__] * z__[*i__] + z__[ip1] * z__[ip1];
	    b = z__[*i__] * z__[*i__] * delsq;
	    if (a > 0.f) {
		tau = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    }

/*
             TAU now is an estimation of SIGMA^2 - D( I )^2. The
             following, however, is the corresponding estimation of
             SIGMA - D( I ).
*/

	    eta = tau / (d__[*i__] + sqrt(d__[*i__] * d__[*i__] + tau));
	} else {

/*
             (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2

             We choose d(i+1) as origin.
*/

	    orgati = FALSE_;
	    sg2lb = -delsq2;
	    sg2ub = 0.f;
	    a = c__ * delsq - z__[*i__] * z__[*i__] - z__[ip1] * z__[ip1];
	    b = z__[ip1] * z__[ip1] * delsq;
	    if (a < 0.f) {
		tau = b * 2.f / (a - sqrt((r__1 = a * a + b * 4.f * c__, dabs(
			r__1))));
	    } else {
		tau = -(a + sqrt((r__1 = a * a + b * 4.f * c__, dabs(r__1))))
			/ (c__ * 2.f);
	    }

/*
             TAU now is an estimation of SIGMA^2 - D( IP1 )^2. The
             following, however, is the corresponding estimation of
             SIGMA - D( IP1 ).
*/

	    eta = tau / (d__[ip1] + sqrt((r__1 = d__[ip1] * d__[ip1] + tau,
		    dabs(r__1))));
	}

	if (orgati) {
	    ii = *i__;
	    *sigma = d__[*i__] + eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] = d__[j] + d__[*i__] + eta;
		delta[j] = d__[j] - d__[*i__] - eta;
/* L130: */
	    }
	} else {
	    ii = *i__ + 1;
	    *sigma = d__[ip1] + eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] = d__[j] + d__[ip1] + eta;
		delta[j] = d__[j] - d__[ip1] - eta;
/* L140: */
	    }
	}
	iim1 = ii - 1;
	iip1 = ii + 1;

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L150: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L160: */
	}

	w = rhoinv + phi + psi;

/*
          W is the value of the secular function with
          its ii-th element removed.
*/

	swtch3 = FALSE_;
	if (orgati) {
	    if (w < 0.f) {
		swtch3 = TRUE_;
	    }
	} else {
	    if (w > 0.f) {
		swtch3 = TRUE_;
	    }
	}
	if (ii == 1 || ii == *n) {
	    swtch3 = FALSE_;
	}

	temp = z__[ii] / (work[ii] * delta[ii]);
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w += temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f
		+ dabs(tau) * dw;

/*        Test for convergence */

	if (dabs(w) <= eps * erretm) {
	    goto L240;
	}

	if (w <= 0.f) {
	    sg2lb = dmax(sg2lb,tau);
	} else {
	    sg2ub = dmin(sg2ub,tau);
	}

/*        Calculate the new step */

	++niter;
	if (! swtch3) {
	    dtipsq = work[ip1] * delta[ip1];
	    dtisq = work[*i__] * delta[*i__];
	    if (orgati) {
/* Computing 2nd power */
		r__1 = z__[*i__] / dtisq;
		c__ = w - dtipsq * dw + delsq * (r__1 * r__1);
	    } else {
/* Computing 2nd power */
		r__1 = z__[ip1] / dtipsq;
		c__ = w - dtisq * dw - delsq * (r__1 * r__1);
	    }
	    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
	    b = dtipsq * dtisq * w;
	    if (c__ == 0.f) {
		if (a == 0.f) {
		    if (orgati) {
			a = z__[*i__] * z__[*i__] + dtipsq * dtipsq * (dpsi +
				dphi);
		    } else {
			a = z__[ip1] * z__[ip1] + dtisq * dtisq * (dpsi +
				dphi);
		    }
		}
		eta = b / a;
	    } else if (a <= 0.f) {
		eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1)))) /
			 (c__ * 2.f);
	    } else {
		eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__, dabs(
			r__1))));
	    }
	} else {

/*           Interpolation using THREE most relevant poles */

	    dtiim = work[iim1] * delta[iim1];
	    dtiip = work[iip1] * delta[iip1];
	    temp = rhoinv + psi + phi;
	    if (orgati) {
		temp1 = z__[iim1] / dtiim;
		temp1 *= temp1;
		c__ = temp - dtiip * (dpsi + dphi) - (d__[iim1] - d__[iip1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		zz[0] = z__[iim1] * z__[iim1];
		if (dpsi < temp1) {
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
		}
	    } else {
		temp1 = z__[iip1] / dtiip;
		temp1 *= temp1;
		c__ = temp - dtiim * (dpsi + dphi) - (d__[iip1] - d__[iim1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		if (dphi < temp1) {
		    zz[0] = dtiim * dtiim * dpsi;
		} else {
		    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
		}
		zz[2] = z__[iip1] * z__[iip1];
	    }
	    zz[1] = z__[ii] * z__[ii];
	    dd[0] = dtiim;
	    dd[1] = delta[ii] * work[ii];
	    dd[2] = dtiip;
	    slaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
	    if (*info != 0) {
		goto L240;
	    }
	}

/*
          Note, eta should be positive if w is negative, and
          eta should be negative otherwise. However,
          if for some reason caused by roundoff, eta*w > 0,
          we simply use one Newton step instead. This way
          will guarantee eta*w < 0.
*/

	if (w * eta >= 0.f) {
	    eta = -w / dw;
	}
	if (orgati) {
	    temp1 = work[*i__] * delta[*i__];
	    temp = eta - temp1;
	} else {
	    temp1 = work[ip1] * delta[ip1];
	    temp = eta - temp1;
	}
	if (temp > sg2ub || temp < sg2lb) {
	    if (w < 0.f) {
		eta = (sg2ub - tau) / 2.f;
	    } else {
		eta = (sg2lb - tau) / 2.f;
	    }
	}

	tau += eta;
	eta /= *sigma + sqrt(*sigma * *sigma + eta);

	prew = w;

	*sigma += eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] += eta;
	    delta[j] -= eta;
/* L170: */
	}

/*        Evaluate PSI and the derivative DPSI */

	dpsi = 0.f;
	psi = 0.f;
	erretm = 0.f;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
/* L180: */
	}
	erretm = dabs(erretm);

/*        Evaluate PHI and the derivative DPHI */

	dphi = 0.f;
	phi = 0.f;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
/* L190: */
	}

	temp = z__[ii] / (work[ii] * delta[ii]);
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w = rhoinv + phi + psi + temp;
	erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) * 3.f
		+ dabs(tau) * dw;

	if (w <= 0.f) {
	    sg2lb = dmax(sg2lb,tau);
	} else {
	    sg2ub = dmin(sg2ub,tau);
	}

	swtch = FALSE_;
	if (orgati) {
	    if (-w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	} else {
	    if (w > dabs(prew) / 10.f) {
		swtch = TRUE_;
	    }
	}

/*        Main loop to update the values of the array   DELTA and WORK */

	iter = niter + 1;

	for (niter = iter; niter <= 20; ++niter) {

/*           Test for convergence */

	    if (dabs(w) <= eps * erretm) {
		goto L240;
	    }

/*           Calculate the new step */

	    if (! swtch3) {
		dtipsq = work[ip1] * delta[ip1];
		dtisq = work[*i__] * delta[*i__];
		if (! swtch) {
		    if (orgati) {
/* Computing 2nd power */
			r__1 = z__[*i__] / dtisq;
			c__ = w - dtipsq * dw + delsq * (r__1 * r__1);
		    } else {
/* Computing 2nd power */
			r__1 = z__[ip1] / dtipsq;
			c__ = w - dtisq * dw - delsq * (r__1 * r__1);
		    }
		} else {
		    temp = z__[ii] / (work[ii] * delta[ii]);
		    if (orgati) {
			dpsi += temp * temp;
		    } else {
			dphi += temp * temp;
		    }
		    c__ = w - dtisq * dpsi - dtipsq * dphi;
		}
		a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
		b = dtipsq * dtisq * w;
		if (c__ == 0.f) {
		    if (a == 0.f) {
			if (! swtch) {
			    if (orgati) {
				a = z__[*i__] * z__[*i__] + dtipsq * dtipsq *
					(dpsi + dphi);
			    } else {
				a = z__[ip1] * z__[ip1] + dtisq * dtisq * (
					dpsi + dphi);
			    }
			} else {
			    a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
			}
		    }
		    eta = b / a;
		} else if (a <= 0.f) {
		    eta = (a - sqrt((r__1 = a * a - b * 4.f * c__, dabs(r__1))
			    )) / (c__ * 2.f);
		} else {
		    eta = b * 2.f / (a + sqrt((r__1 = a * a - b * 4.f * c__,
			    dabs(r__1))));
		}
	    } else {

/*              Interpolation using THREE most relevant poles */

		dtiim = work[iim1] * delta[iim1];
		dtiip = work[iip1] * delta[iip1];
		temp = rhoinv + psi + phi;
		if (swtch) {
		    c__ = temp - dtiim * dpsi - dtiip * dphi;
		    zz[0] = dtiim * dtiim * dpsi;
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    if (orgati) {
			temp1 = z__[iim1] / dtiim;
			temp1 *= temp1;
			temp2 = (d__[iim1] - d__[iip1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiip * (dpsi + dphi) - temp2;
			zz[0] = z__[iim1] * z__[iim1];
			if (dpsi < temp1) {
			    zz[2] = dtiip * dtiip * dphi;
			} else {
			    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
			}
		    } else {
			temp1 = z__[iip1] / dtiip;
			temp1 *= temp1;
			temp2 = (d__[iip1] - d__[iim1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiim * (dpsi + dphi) - temp2;
			if (dphi < temp1) {
			    zz[0] = dtiim * dtiim * dpsi;
			} else {
			    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
			}
			zz[2] = z__[iip1] * z__[iip1];
		    }
		}
		dd[0] = dtiim;
		dd[1] = delta[ii] * work[ii];
		dd[2] = dtiip;
		slaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
		if (*info != 0) {
		    goto L240;
		}
	    }

/*
             Note, eta should be positive if w is negative, and
             eta should be negative otherwise. However,
             if for some reason caused by roundoff, eta*w > 0,
             we simply use one Newton step instead. This way
             will guarantee eta*w < 0.
*/

	    if (w * eta >= 0.f) {
		eta = -w / dw;
	    }
	    if (orgati) {
		temp1 = work[*i__] * delta[*i__];
		temp = eta - temp1;
	    } else {
		temp1 = work[ip1] * delta[ip1];
		temp = eta - temp1;
	    }
	    if (temp > sg2ub || temp < sg2lb) {
		if (w < 0.f) {
		    eta = (sg2ub - tau) / 2.f;
		} else {
		    eta = (sg2lb - tau) / 2.f;
		}
	    }

	    tau += eta;
	    eta /= *sigma + sqrt(*sigma * *sigma + eta);

	    *sigma += eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] += eta;
		delta[j] -= eta;
/* L200: */
	    }

	    prew = w;

/*           Evaluate PSI and the derivative DPSI */

	    dpsi = 0.f;
	    psi = 0.f;
	    erretm = 0.f;
	    i__1 = iim1;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
/* L210: */
	    }
	    erretm = dabs(erretm);

/*           Evaluate PHI and the derivative DPHI */

	    dphi = 0.f;
	    phi = 0.f;
	    i__1 = iip1;
	    for (j = *n; j >= i__1; --j) {
		temp = z__[j] / (work[j] * delta[j]);
		phi += z__[j] * temp;
		dphi += temp * temp;
		erretm += phi;
/* L220: */
	    }

	    temp = z__[ii] / (work[ii] * delta[ii]);
	    dw = dpsi + dphi + temp * temp;
	    temp = z__[ii] * temp;
	    w = rhoinv + phi + psi + temp;
	    erretm = (phi - psi) * 8.f + erretm + rhoinv * 2.f + dabs(temp) *
		    3.f + dabs(tau) * dw;
	    if (w * prew > 0.f && dabs(w) > dabs(prew) / 10.f) {
		swtch = ! swtch;
	    }

	    if (w <= 0.f) {
		sg2lb = dmax(sg2lb,tau);
	    } else {
		sg2ub = dmin(sg2ub,tau);
	    }

/* L230: */
	}

/*        Return with INFO = 1, NITER = MAXIT and not converged */

	*info = 1;

    }

L240:
    return 0;

/*     End of SLASD4 */

} /* slasd4_ */

/* Subroutine */ int slasd5_(integer *i__, real *d__, real *z__, real *delta,
	real *rho, real *dsigma, real *work)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real b, c__, w, del, tau, delsq;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       June 30, 1999


    Purpose
    =======

    This subroutine computes the square root of the I-th eigenvalue
    of a positive symmetric rank-one modification of a 2-by-2 diagonal
    matrix

               diag( D ) * diag( D ) +  RHO *  Z * transpose(Z) .

    The diagonal entries in the array D are assumed to satisfy

               0 <= D(i) < D(j)  for  i < j .

    We also assume RHO > 0 and that the Euclidean norm of the vector
    Z is one.

    Arguments
    =========

    I      (input) INTEGER
           The index of the eigenvalue to be computed.  I = 1 or I = 2.

    D      (input) REAL array, dimension ( 2 )
           The original eigenvalues.  We assume 0 <= D(1) < D(2).

    Z      (input) REAL array, dimension ( 2 )
           The components of the updating vector.

    DELTA  (output) REAL array, dimension ( 2 )
           Contains (D(j) - lambda_I) in its  j-th component.
           The vector DELTA contains the information necessary
           to construct the eigenvectors.

    RHO    (input) REAL
           The scalar in the symmetric updating formula.

    DSIGMA (output) REAL
           The computed lambda_I, the I-th updated eigenvalue.

    WORK   (workspace) REAL array, dimension ( 2 )
           WORK contains (D(j) + sigma_I) in its  j-th component.

    Further Details
    ===============

    Based on contributions by
       Ren-Cang Li, Computer Science Division, University of California
       at Berkeley, USA

    =====================================================================
*/


    /* Parameter adjustments */
    --work;
    --delta;
    --z__;
    --d__;

    /* Function Body */
    del = d__[2] - d__[1];
    delsq = del * (d__[2] + d__[1]);
    if (*i__ == 1) {
	w = *rho * 4.f * (z__[2] * z__[2] / (d__[1] + d__[2] * 3.f) - z__[1] *
		 z__[1] / (d__[1] * 3.f + d__[2])) / del + 1.f;
	if (w > 0.f) {
	    b = delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[1] * z__[1] * delsq;

/*
             B > ZERO, always

             The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 )
*/

	    tau = c__ * 2.f / (b + sqrt((r__1 = b * b - c__ * 4.f, dabs(r__1))
		    ));

/*           The following TAU is DSIGMA - D( 1 ) */

	    tau /= d__[1] + sqrt(d__[1] * d__[1] + tau);
	    *dsigma = d__[1] + tau;
	    delta[1] = -tau;
	    delta[2] = del - tau;
	    work[1] = d__[1] * 2.f + tau;
	    work[2] = d__[1] + tau + d__[2];
/*
             DELTA( 1 ) = -Z( 1 ) / TAU
             DELTA( 2 ) = Z( 2 ) / ( DEL-TAU )
*/
	} else {
	    b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[2] * z__[2] * delsq;

/*           The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	    if (b > 0.f) {
		tau = c__ * -2.f / (b + sqrt(b * b + c__ * 4.f));
	    } else {
		tau = (b - sqrt(b * b + c__ * 4.f)) / 2.f;
	    }

/*           The following TAU is DSIGMA - D( 2 ) */

	    tau /= d__[2] + sqrt((r__1 = d__[2] * d__[2] + tau, dabs(r__1)));
	    *dsigma = d__[2] + tau;
	    delta[1] = -(del + tau);
	    delta[2] = -tau;
	    work[1] = d__[1] + tau + d__[2];
	    work[2] = d__[2] * 2.f + tau;
/*
             DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
             DELTA( 2 ) = -Z( 2 ) / TAU
*/
	}
/*
          TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
          DELTA( 1 ) = DELTA( 1 ) / TEMP
          DELTA( 2 ) = DELTA( 2 ) / TEMP
*/
    } else {

/*        Now I=2 */

	b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	c__ = *rho * z__[2] * z__[2] * delsq;

/*        The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	if (b > 0.f) {
	    tau = (b + sqrt(b * b + c__ * 4.f)) / 2.f;
	} else {
	    tau = c__ * 2.f / (-b + sqrt(b * b + c__ * 4.f));
	}

/*        The following TAU is DSIGMA - D( 2 ) */

	tau /= d__[2] + sqrt(d__[2] * d__[2] + tau);
	*dsigma = d__[2] + tau;
	delta[1] = -(del + tau);
	delta[2] = -tau;
	work[1] = d__[1] + tau + d__[2];
	work[2] = d__[2] * 2.f + tau;
/*
          DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
          DELTA( 2 ) = -Z( 2 ) / TAU
          TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
          DELTA( 1 ) = DELTA( 1 ) / TEMP
          DELTA( 2 ) = DELTA( 2 ) / TEMP
*/
    }
    return 0;

/*     End of SLASD5 */

} /* slasd5_ */

/* Subroutine */ int slasd6_(integer *icompq, integer *nl, integer *nr,
	integer *sqre, real *d__, real *vf, real *vl, real *alpha, real *beta,
	 integer *idxq, integer *perm, integer *givptr, integer *givcol,
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, givnum_dim1, givnum_offset,
	    poles_dim1, poles_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    static integer i__, m, n, n1, n2, iw, idx, idxc, idxp, ivfw, ivlw;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), slasd7_(integer *, integer *, integer *, integer *,
	    integer *, real *, real *, real *, real *, real *, real *, real *,
	     real *, real *, real *, integer *, integer *, integer *, integer
	    *, integer *, integer *, integer *, real *, integer *, real *,
	    real *, integer *), slasd8_(integer *, integer *, real *, real *,
	    real *, real *, real *, real *, integer *, real *, real *,
	    integer *);
    static integer isigma;
    extern /* Subroutine */ int xerbla_(char *, integer *), slascl_(
	    char *, integer *, integer *, real *, real *, integer *, integer *
	    , real *, integer *, integer *), slamrg_(integer *,
	    integer *, real *, integer *, integer *, integer *);
    static real orgnrm;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLASD6 computes the SVD of an updated upper bidiagonal matrix B
    obtained by merging two smaller ones by appending a row. This
    routine is used only for the problem which requires all singular
    values and optionally singular vector matrices in factored form.
    B is an N-by-M matrix with N = NL + NR + 1 and M = N + SQRE.
    A related subroutine, SLASD1, handles the case in which all singular
    values and singular vectors of the bidiagonal matrix are desired.

    SLASD6 computes the SVD as follows:

                  ( D1(in)  0    0     0 )
      B = U(in) * (   Z1'   a   Z2'    b ) * VT(in)
                  (   0     0   D2(in) 0 )

        = U(out) * ( D(out) 0) * VT(out)

    where Z' = (Z1' a Z2' b) = u' VT', and u is a vector of dimension M
    with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
    elsewhere; and the entry b is empty if SQRE = 0.

    The singular values of B can be computed using D1, D2, the first
    components of all the right singular vectors of the lower block, and
    the last components of all the right singular vectors of the upper
    block. These components are stored and updated in VF and VL,
    respectively, in SLASD6. Hence U and VT are not explicitly
    referenced.

    The singular values are stored in D. The algorithm consists of two
    stages:

          The first stage consists of deflating the size of the problem
          when there are multiple singular values or if there is a zero
          in the Z vector. For each such occurence the dimension of the
          secular equation problem is reduced by one. This stage is
          performed by the routine SLASD7.

          The second stage consists of calculating the updated
          singular values. This is done by finding the roots of the
          secular equation via the routine SLASD4 (as called by SLASD8).
          This routine also updates VF and VL and computes the distances
          between the updated singular values and the old singular
          values.

    SLASD6 is called from SLASDA.

    Arguments
    =========

    ICOMPQ (input) INTEGER
           Specifies whether singular vectors are to be computed in
           factored form:
           = 0: Compute singular values only.
           = 1: Compute singular vectors in factored form as well.

    NL     (input) INTEGER
           The row dimension of the upper block.  NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block.  NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has row dimension N = NL + NR + 1,
           and column dimension M = N + SQRE.

    D      (input/output) REAL array, dimension ( NL+NR+1 ).
           On entry D(1:NL,1:NL) contains the singular values of the
           upper block, and D(NL+2:N) contains the singular values
           of the lower block. On exit D(1:N) contains the singular
           values of the modified matrix.

    VF     (input/output) REAL array, dimension ( M )
           On entry, VF(1:NL+1) contains the first components of all
           right singular vectors of the upper block; and VF(NL+2:M)
           contains the first components of all right singular vectors
           of the lower block. On exit, VF contains the first components
           of all right singular vectors of the bidiagonal matrix.

    VL     (input/output) REAL array, dimension ( M )
           On entry, VL(1:NL+1) contains the  last components of all
           right singular vectors of the upper block; and VL(NL+2:M)
           contains the last components of all right singular vectors of
           the lower block. On exit, VL contains the last components of
           all right singular vectors of the bidiagonal matrix.

    ALPHA  (input) REAL
           Contains the diagonal element associated with the added row.

    BETA   (input) REAL
           Contains the off-diagonal element associated with the added
           row.

    IDXQ   (output) INTEGER array, dimension ( N )
           This contains the permutation which will reintegrate the
           subproblem just solved back into sorted order, i.e.
           D( IDXQ( I = 1, N ) ) will be in ascending order.

    PERM   (output) INTEGER array, dimension ( N )
           The permutations (from deflation and sorting) to be applied
           to each block. Not referenced if ICOMPQ = 0.

    GIVPTR (output) INTEGER
           The number of Givens rotations which took place in this
           subproblem. Not referenced if ICOMPQ = 0.

    GIVCOL (output) INTEGER array, dimension ( LDGCOL, 2 )
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation. Not referenced if ICOMPQ = 0.

    LDGCOL (input) INTEGER
           leading dimension of GIVCOL, must be at least N.

    GIVNUM (output) REAL array, dimension ( LDGNUM, 2 )
           Each number indicates the C or S value to be used in the
           corresponding Givens rotation. Not referenced if ICOMPQ = 0.

    LDGNUM (input) INTEGER
           The leading dimension of GIVNUM and POLES, must be at least N.

    POLES  (output) REAL array, dimension ( LDGNUM, 2 )
           On exit, POLES(1,*) is an array containing the new singular
           values obtained from solving the secular equation, and
           POLES(2,*) is an array containing the poles in the secular
           equation. Not referenced if ICOMPQ = 0.

    DIFL   (output) REAL array, dimension ( N )
           On exit, DIFL(I) is the distance between I-th updated
           (undeflated) singular value and the I-th (undeflated) old
           singular value.

    DIFR   (output) REAL array,
                    dimension ( LDGNUM, 2 ) if ICOMPQ = 1 and
                    dimension ( N ) if ICOMPQ = 0.
           On exit, DIFR(I, 1) is the distance between I-th updated
           (undeflated) singular value and the I+1-th (undeflated) old
           singular value.

           If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
           normalizing factors for the right singular vector matrix.

           See SLASD8 for details on DIFL and DIFR.

    Z      (output) REAL array, dimension ( M )
           The first elements of this array contain the components
           of the deflation-adjusted updating row vector.

    K      (output) INTEGER
           Contains the dimension of the non-deflated matrix,
           This is the order of the related secular equation. 1 <= K <=N.

    C      (output) REAL
           C contains garbage if SQRE =0 and the C-value of a Givens
           rotation related to the right null space if SQRE = 1.

    S      (output) REAL
           S contains garbage if SQRE =0 and the S-value of a Givens
           rotation related to the right null space if SQRE = 1.

    WORK   (workspace) REAL array, dimension ( 4 * M )

    IWORK  (workspace) INTEGER array, dimension ( 3 * N )

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --vf;
    --vl;
    --idxq;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    poles_dim1 = *ldgnum;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    --difl;
    --difr;
    --z__;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    n = *nl + *nr + 1;
    m = n + *sqre;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldgcol < n) {
	*info = -14;
    } else if (*ldgnum < n) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD6", &i__1);
	return 0;
    }

/*
       The following values are for bookkeeping purposes only.  They are
       integer pointers which indicate the portion of the workspace
       used by a particular array in SLASD7 and SLASD8.
*/

    isigma = 1;
    iw = isigma + n;
    ivfw = iw + m;
    ivlw = ivfw + m;

    idx = 1;
    idxc = idx + n;
    idxp = idxc + n;

/*
       Scale.

   Computing MAX
*/
    r__1 = dabs(*alpha), r__2 = dabs(*beta);
    orgnrm = dmax(r__1,r__2);
    d__[*nl + 1] = 0.f;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = d__[i__], dabs(r__1)) > orgnrm) {
	    orgnrm = (r__1 = d__[i__], dabs(r__1));
	}
/* L10: */
    }
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &n, &c__1, &d__[1], &n, info);
    *alpha /= orgnrm;
    *beta /= orgnrm;

/*     Sort and Deflate singular values. */

    slasd7_(icompq, nl, nr, sqre, k, &d__[1], &z__[1], &work[iw], &vf[1], &
	    work[ivfw], &vl[1], &work[ivlw], alpha, beta, &work[isigma], &
	    iwork[idx], &iwork[idxp], &idxq[1], &perm[1], givptr, &givcol[
	    givcol_offset], ldgcol, &givnum[givnum_offset], ldgnum, c__, s,
	    info);

/*     Solve Secular Equation, compute DIFL, DIFR, and update VF, VL. */

    slasd8_(icompq, k, &d__[1], &z__[1], &vf[1], &vl[1], &difl[1], &difr[1],
	    ldgnum, &work[isigma], &work[iw], info);

/*     Save the poles if ICOMPQ = 1. */

    if (*icompq == 1) {
	scopy_(k, &d__[1], &c__1, &poles[poles_dim1 + 1], &c__1);
	scopy_(k, &work[isigma], &c__1, &poles[(poles_dim1 << 1) + 1], &c__1);
    }

/*     Unscale. */

    slascl_("G", &c__0, &c__0, &c_b15, &orgnrm, &n, &c__1, &d__[1], &n, info);

/*     Prepare the IDXQ sorting permutation. */

    n1 = *k;
    n2 = n - *k;
    slamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &idxq[1]);

    return 0;

/*     End of SLASD6 */

} /* slasd6_ */

/* Subroutine */ int slasd7_(integer *icompq, integer *nl, integer *nr,
	integer *sqre, integer *k, real *d__, real *z__, real *zw, real *vf,
	real *vfw, real *vl, real *vlw, real *alpha, real *beta, real *dsigma,
	 integer *idx, integer *idxp, integer *idxq, integer *perm, integer *
	givptr, integer *givcol, integer *ldgcol, real *givnum, integer *
	ldgnum, real *c__, real *s, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, givnum_dim1, givnum_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    static integer i__, j, m, n, k2;
    static real z1;
    static integer jp;
    static real eps, tau, tol;
    static integer nlp1, nlp2, idxi, idxj;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *,
	    integer *, real *, real *);
    static integer idxjp, jprev;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), slamrg_(
	    integer *, integer *, real *, integer *, integer *, integer *);
    static real hlftol;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       June 30, 1999


    Purpose
    =======

    SLASD7 merges the two sets of singular values together into a single
    sorted set. Then it tries to deflate the size of the problem. There
    are two ways in which deflation can occur:  when two or more singular
    values are close together or if there is a tiny entry in the Z
    vector. For each such occurrence the order of the related
    secular equation problem is reduced by one.

    SLASD7 is called from SLASD6.

    Arguments
    =========

    ICOMPQ  (input) INTEGER
            Specifies whether singular vectors are to be computed
            in compact form, as follows:
            = 0: Compute singular values only.
            = 1: Compute singular vectors of upper
                 bidiagonal matrix in compact form.

    NL     (input) INTEGER
           The row dimension of the upper block. NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block. NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has
           N = NL + NR + 1 rows and
           M = N + SQRE >= N columns.

    K      (output) INTEGER
           Contains the dimension of the non-deflated matrix, this is
           the order of the related secular equation. 1 <= K <=N.

    D      (input/output) REAL array, dimension ( N )
           On entry D contains the singular values of the two submatrices
           to be combined. On exit D contains the trailing (N-K) updated
           singular values (those which were deflated) sorted into
           increasing order.

    Z      (output) REAL array, dimension ( M )
           On exit Z contains the updating row vector in the secular
           equation.

    ZW     (workspace) REAL array, dimension ( M )
           Workspace for Z.

    VF     (input/output) REAL array, dimension ( M )
           On entry, VF(1:NL+1) contains the first components of all
           right singular vectors of the upper block; and VF(NL+2:M)
           contains the first components of all right singular vectors
           of the lower block. On exit, VF contains the first components
           of all right singular vectors of the bidiagonal matrix.

    VFW    (workspace) REAL array, dimension ( M )
           Workspace for VF.

    VL     (input/output) REAL array, dimension ( M )
           On entry, VL(1:NL+1) contains the  last components of all
           right singular vectors of the upper block; and VL(NL+2:M)
           contains the last components of all right singular vectors
           of the lower block. On exit, VL contains the last components
           of all right singular vectors of the bidiagonal matrix.

    VLW    (workspace) REAL array, dimension ( M )
           Workspace for VL.

    ALPHA  (input) REAL
           Contains the diagonal element associated with the added row.

    BETA   (input) REAL
           Contains the off-diagonal element associated with the added
           row.

    DSIGMA (output) REAL array, dimension ( N )
           Contains a copy of the diagonal elements (K-1 singular values
           and one zero) in the secular equation.

    IDX    (workspace) INTEGER array, dimension ( N )
           This will contain the permutation used to sort the contents of
           D into ascending order.

    IDXP   (workspace) INTEGER array, dimension ( N )
           This will contain the permutation used to place deflated
           values of D at the end of the array. On output IDXP(2:K)
           points to the nondeflated D-values and IDXP(K+1:N)
           points to the deflated singular values.

    IDXQ   (input) INTEGER array, dimension ( N )
           This contains the permutation which separately sorts the two
           sub-problems in D into ascending order.  Note that entries in
           the first half of this permutation must first be moved one
           position backward; and entries in the second half
           must first have NL+1 added to their values.

    PERM   (output) INTEGER array, dimension ( N )
           The permutations (from deflation and sorting) to be applied
           to each singular block. Not referenced if ICOMPQ = 0.

    GIVPTR (output) INTEGER
           The number of Givens rotations which took place in this
           subproblem. Not referenced if ICOMPQ = 0.

    GIVCOL (output) INTEGER array, dimension ( LDGCOL, 2 )
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation. Not referenced if ICOMPQ = 0.

    LDGCOL (input) INTEGER
           The leading dimension of GIVCOL, must be at least N.

    GIVNUM (output) REAL array, dimension ( LDGNUM, 2 )
           Each number indicates the C or S value to be used in the
           corresponding Givens rotation. Not referenced if ICOMPQ = 0.

    LDGNUM (input) INTEGER
           The leading dimension of GIVNUM, must be at least N.

    C      (output) REAL
           C contains garbage if SQRE =0 and the C-value of a Givens
           rotation related to the right null space if SQRE = 1.

    S      (output) REAL
           S contains garbage if SQRE =0 and the S-value of a Givens
           rotation related to the right null space if SQRE = 1.

    INFO   (output) INTEGER
           = 0:  successful exit.
           < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --z__;
    --zw;
    --vf;
    --vfw;
    --vl;
    --vlw;
    --dsigma;
    --idx;
    --idxp;
    --idxq;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;

    /* Function Body */
    *info = 0;
    n = *nl + *nr + 1;
    m = n + *sqre;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldgcol < n) {
	*info = -22;
    } else if (*ldgnum < n) {
	*info = -24;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD7", &i__1);
	return 0;
    }

    nlp1 = *nl + 1;
    nlp2 = *nl + 2;
    if (*icompq == 1) {
	*givptr = 0;
    }

/*
       Generate the first part of the vector Z and move the singular
       values in the first part of D one position backward.
*/

    z1 = *alpha * vl[nlp1];
    vl[nlp1] = 0.f;
    tau = vf[nlp1];
    for (i__ = *nl; i__ >= 1; --i__) {
	z__[i__ + 1] = *alpha * vl[i__];
	vl[i__] = 0.f;
	vf[i__ + 1] = vf[i__];
	d__[i__ + 1] = d__[i__];
	idxq[i__ + 1] = idxq[i__] + 1;
/* L10: */
    }
    vf[1] = tau;

/*     Generate the second part of the vector Z. */

    i__1 = m;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	z__[i__] = *beta * vf[i__];
	vf[i__] = 0.f;
/* L20: */
    }

/*     Sort the singular values into increasing order */

    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	idxq[i__] += nlp1;
/* L30: */
    }

/*     DSIGMA, IDXC, IDXC, and ZW are used as storage space. */

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	dsigma[i__] = d__[idxq[i__]];
	zw[i__] = z__[idxq[i__]];
	vfw[i__] = vf[idxq[i__]];
	vlw[i__] = vl[idxq[i__]];
/* L40: */
    }

    slamrg_(nl, nr, &dsigma[2], &c__1, &c__1, &idx[2]);

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	idxi = idx[i__] + 1;
	d__[i__] = dsigma[idxi];
	z__[i__] = zw[idxi];
	vf[i__] = vfw[idxi];
	vl[i__] = vlw[idxi];
/* L50: */
    }

/*     Calculate the allowable deflation tolerence */

    eps = slamch_("Epsilon");
/* Computing MAX */
    r__1 = dabs(*alpha), r__2 = dabs(*beta);
    tol = dmax(r__1,r__2);
/* Computing MAX */
    r__2 = (r__1 = d__[n], dabs(r__1));
    tol = eps * 64.f * dmax(r__2,tol);

/*
       There are 2 kinds of deflation -- first a value in the z-vector
       is small, second two (or more) singular values are very close
       together (their difference is small).

       If the value in the z-vector is small, we simply permute the
       array so that the corresponding singular value is moved to the
       end.

       If two values in the D-vector are close, we perform a two-sided
       rotation designed to make one of the corresponding z-vector
       entries zero, and then permute the array so that the deflated
       singular value is moved to the end.

       If there are multiple singular values then the problem deflates.
       Here the number of equal singular values are found.  As each equal
       singular value is found, an elementary reflector is computed to
       rotate the corresponding singular subspace so that the
       corresponding components of Z are zero in this new basis.
*/

    *k = 1;
    k2 = n + 1;
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	if ((r__1 = z__[j], dabs(r__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    idxp[k2] = j;
	    if (j == n) {
		goto L100;
	    }
	} else {
	    jprev = j;
	    goto L70;
	}
/* L60: */
    }
L70:
    j = jprev;
L80:
    ++j;
    if (j > n) {
	goto L90;
    }
    if ((r__1 = z__[j], dabs(r__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	idxp[k2] = j;
    } else {

/*        Check if singular values are close enough to allow deflation. */

	if ((r__1 = d__[j] - d__[jprev], dabs(r__1)) <= tol) {

/*           Deflation is possible. */

	    *s = z__[jprev];
	    *c__ = z__[j];

/*
             Find sqrt(a**2+b**2) without overflow or
             destructive underflow.
*/

	    tau = slapy2_(c__, s);
	    z__[j] = tau;
	    z__[jprev] = 0.f;
	    *c__ /= tau;
	    *s = -(*s) / tau;

/*           Record the appropriate Givens rotation */

	    if (*icompq == 1) {
		++(*givptr);
		idxjp = idxq[idx[jprev] + 1];
		idxj = idxq[idx[j] + 1];
		if (idxjp <= nlp1) {
		    --idxjp;
		}
		if (idxj <= nlp1) {
		    --idxj;
		}
		givcol[*givptr + (givcol_dim1 << 1)] = idxjp;
		givcol[*givptr + givcol_dim1] = idxj;
		givnum[*givptr + (givnum_dim1 << 1)] = *c__;
		givnum[*givptr + givnum_dim1] = *s;
	    }
	    srot_(&c__1, &vf[jprev], &c__1, &vf[j], &c__1, c__, s);
	    srot_(&c__1, &vl[jprev], &c__1, &vl[j], &c__1, c__, s);
	    --k2;
	    idxp[k2] = jprev;
	    jprev = j;
	} else {
	    ++(*k);
	    zw[*k] = z__[jprev];
	    dsigma[*k] = d__[jprev];
	    idxp[*k] = jprev;
	    jprev = j;
	}
    }
    goto L80;
L90:

/*     Record the last singular value. */

    ++(*k);
    zw[*k] = z__[jprev];
    dsigma[*k] = d__[jprev];
    idxp[*k] = jprev;

L100:

/*
       Sort the singular values into DSIGMA. The singular values which
       were not deflated go into the first K slots of DSIGMA, except
       that DSIGMA(1) is treated separately.
*/

    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	dsigma[j] = d__[jp];
	vfw[j] = vf[jp];
	vlw[j] = vl[jp];
/* L110: */
    }
    if (*icompq == 1) {
	i__1 = n;
	for (j = 2; j <= i__1; ++j) {
	    jp = idxp[j];
	    perm[j] = idxq[idx[jp] + 1];
	    if (perm[j] <= nlp1) {
		--perm[j];
	    }
/* L120: */
	}
    }

/*
       The deflated singular values go back into the last N - K slots of
       D.
*/

    i__1 = n - *k;
    scopy_(&i__1, &dsigma[*k + 1], &c__1, &d__[*k + 1], &c__1);

/*
       Determine DSIGMA(1), DSIGMA(2), Z(1), VF(1), VL(1), VF(M), and
       VL(M).
*/

    dsigma[1] = 0.f;
    hlftol = tol / 2.f;
    if (dabs(dsigma[2]) <= hlftol) {
	dsigma[2] = hlftol;
    }
    if (m > n) {
	z__[1] = slapy2_(&z1, &z__[m]);
	if (z__[1] <= tol) {
	    *c__ = 1.f;
	    *s = 0.f;
	    z__[1] = tol;
	} else {
	    *c__ = z1 / z__[1];
	    *s = -z__[m] / z__[1];
	}
	srot_(&c__1, &vf[m], &c__1, &vf[1], &c__1, c__, s);
	srot_(&c__1, &vl[m], &c__1, &vl[1], &c__1, c__, s);
    } else {
	if (dabs(z1) <= tol) {
	    z__[1] = tol;
	} else {
	    z__[1] = z1;
	}
    }

/*     Restore Z, VF, and VL. */

    i__1 = *k - 1;
    scopy_(&i__1, &zw[2], &c__1, &z__[2], &c__1);
    i__1 = n - 1;
    scopy_(&i__1, &vfw[2], &c__1, &vf[2], &c__1);
    i__1 = n - 1;
    scopy_(&i__1, &vlw[2], &c__1, &vl[2], &c__1);

    return 0;

/*     End of SLASD7 */

} /* slasd7_ */

/* Subroutine */ int slasd8_(integer *icompq, integer *k, real *d__, real *
	z__, real *vf, real *vl, real *difl, real *difr, integer *lddifr,
	real *dsigma, real *work, integer *info)
{
    /* System generated locals */
    integer difr_dim1, difr_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static integer i__, j;
    static real dj, rho;
    static integer iwk1, iwk2, iwk3;
    static real temp;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    static integer iwk2i, iwk3i;
    extern doublereal snrm2_(integer *, real *, integer *);
    static real diflj, difrj, dsigj;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int slasd4_(integer *, integer *, real *, real *,
	    real *, real *, real *, real *, integer *), xerbla_(char *,
	    integer *);
    static real dsigjp;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *), slaset_(char *, integer *, integer *, real *, real *,
	    real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       June 30, 1999


    Purpose
    =======

    SLASD8 finds the square roots of the roots of the secular equation,
    as defined by the values in DSIGMA and Z. It makes the appropriate
    calls to SLASD4, and stores, for each  element in D, the distance
    to its two nearest poles (elements in DSIGMA). It also updates
    the arrays VF and VL, the first and last components of all the
    right singular vectors of the original bidiagonal matrix.

    SLASD8 is called from SLASD6.

    Arguments
    =========

    ICOMPQ  (input) INTEGER
            Specifies whether singular vectors are to be computed in
            factored form in the calling routine:
            = 0: Compute singular values only.
            = 1: Compute singular vectors in factored form as well.

    K       (input) INTEGER
            The number of terms in the rational function to be solved
            by SLASD4.  K >= 1.

    D       (output) REAL array, dimension ( K )
            On output, D contains the updated singular values.

    Z       (input) REAL array, dimension ( K )
            The first K elements of this array contain the components
            of the deflation-adjusted updating row vector.

    VF      (input/output) REAL array, dimension ( K )
            On entry, VF contains  information passed through DBEDE8.
            On exit, VF contains the first K components of the first
            components of all right singular vectors of the bidiagonal
            matrix.

    VL      (input/output) REAL array, dimension ( K )
            On entry, VL contains  information passed through DBEDE8.
            On exit, VL contains the first K components of the last
            components of all right singular vectors of the bidiagonal
            matrix.

    DIFL    (output) REAL array, dimension ( K )
            On exit, DIFL(I) = D(I) - DSIGMA(I).

    DIFR    (output) REAL array,
                     dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and
                     dimension ( K ) if ICOMPQ = 0.
            On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not
            defined and will not be referenced.

            If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
            normalizing factors for the right singular vector matrix.

    LDDIFR  (input) INTEGER
            The leading dimension of DIFR, must be at least K.

    DSIGMA  (input) REAL array, dimension ( K )
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation.

    WORK    (workspace) REAL array, dimension at least 3 * K

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --z__;
    --vf;
    --vl;
    --difl;
    difr_dim1 = *lddifr;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    --dsigma;
    --work;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*k < 1) {
	*info = -2;
    } else if (*lddifr < *k) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD8", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 1) {
	d__[1] = dabs(z__[1]);
	difl[1] = d__[1];
	if (*icompq == 1) {
	    difl[2] = 1.f;
	    difr[(difr_dim1 << 1) + 1] = 1.f;
	}
	return 0;
    }

/*
       Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
       be computed with high relative accuracy (barring over/underflow).
       This is a problem on machines without a guard digit in
       add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
       The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
       which on any of these machines zeros out the bottommost
       bit of DSIGMA(I) if it is 1; this makes the subsequent
       subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
       occurs. On binary machines with a guard digit (almost all
       machines) it does not change DSIGMA(I) at all. On hexadecimal
       and decimal machines with a guard digit, it slightly
       changes the bottommost bits of DSIGMA(I). It does not account
       for hexadecimal or decimal machines without guard digits
       (we know of none). We use a subroutine call to compute
       2*DLAMBDA(I) to prevent optimizing compilers from eliminating
       this code.
*/

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = slamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
/* L10: */
    }

/*     Book keeping. */

    iwk1 = 1;
    iwk2 = iwk1 + *k;
    iwk3 = iwk2 + *k;
    iwk2i = iwk2 - 1;
    iwk3i = iwk3 - 1;

/*     Normalize Z. */

    rho = snrm2_(k, &z__[1], &c__1);
    slascl_("G", &c__0, &c__0, &rho, &c_b15, k, &c__1, &z__[1], k, info);
    rho *= rho;

/*     Initialize WORK(IWK3). */

    slaset_("A", k, &c__1, &c_b15, &c_b15, &work[iwk3], k);

/*
       Compute the updated singular values, the arrays DIFL, DIFR,
       and the updated Z.
*/

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	slasd4_(k, &j, &dsigma[1], &z__[1], &work[iwk1], &rho, &d__[j], &work[
		iwk2], info);

/*        If the root finder fails, the computation is terminated. */

	if (*info != 0) {
	    return 0;
	}
	work[iwk3i + j] = work[iwk3i + j] * work[j] * work[iwk2i + j];
	difl[j] = -work[j];
	difr[j + difr_dim1] = -work[j + 1];
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i +
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L20: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i +
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L30: */
	}
/* L40: */
    }

/*     Compute updated Z. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__2 = sqrt((r__1 = work[iwk3i + i__], dabs(r__1)));
	z__[i__] = r_sign(&r__2, &z__[i__]);
/* L50: */
    }

/*     Update VF and VL. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	diflj = difl[j];
	dj = d__[j];
	dsigj = -dsigma[j];
	if (j < *k) {
	    difrj = -difr[j + difr_dim1];
	    dsigjp = -dsigma[j + 1];
	}
	work[j] = -z__[j] / diflj / (dsigma[j] + dj);
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (slamc3_(&dsigma[i__], &dsigj) - diflj) / (
		    dsigma[i__] + dj);
/* L60: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (slamc3_(&dsigma[i__], &dsigjp) + difrj) /
		    (dsigma[i__] + dj);
/* L70: */
	}
	temp = snrm2_(k, &work[1], &c__1);
	work[iwk2i + j] = sdot_(k, &work[1], &c__1, &vf[1], &c__1) / temp;
	work[iwk3i + j] = sdot_(k, &work[1], &c__1, &vl[1], &c__1) / temp;
	if (*icompq == 1) {
	    difr[j + (difr_dim1 << 1)] = temp;
	}
/* L80: */
    }

    scopy_(k, &work[iwk2], &c__1, &vf[1], &c__1);
    scopy_(k, &work[iwk3], &c__1, &vl[1], &c__1);

    return 0;

/*     End of SLASD8 */

} /* slasd8_ */

/* Subroutine */ int slasda_(integer *icompq, integer *smlsiz, integer *n,
	integer *sqre, real *d__, real *e, real *u, integer *ldu, real *vt,
	integer *k, real *difl, real *difr, real *z__, real *poles, integer *
	givptr, integer *givcol, integer *ldgcol, integer *perm, real *givnum,
	 real *c__, real *s, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, perm_dim1, perm_offset, difl_dim1,
	    difl_offset, difr_dim1, difr_offset, givnum_dim1, givnum_offset,
	    poles_dim1, poles_offset, u_dim1, u_offset, vt_dim1, vt_offset,
	    z_dim1, z_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, m, i1, ic, lf, nd, ll, nl, vf, nr, vl, im1, ncc,
	    nlf, nrf, vfi, iwk, vli, lvl, nru, ndb1, nlp1, lvl2, nrp1;
    static real beta;
    static integer idxq, nlvl;
    static real alpha;
    static integer inode, ndiml, ndimr, idxqi, itemp, sqrei;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), slasd6_(integer *, integer *, integer *, integer *,
	    real *, real *, real *, real *, real *, integer *, integer *,
	    integer *, integer *, integer *, real *, integer *, real *, real *
	    , real *, real *, integer *, real *, real *, real *, integer *,
	    integer *);
    static integer nwork1, nwork2;
    extern /* Subroutine */ int xerbla_(char *, integer *), slasdq_(
	    char *, integer *, integer *, integer *, integer *, integer *,
	    real *, real *, real *, integer *, real *, integer *, real *,
	    integer *, real *, integer *), slasdt_(integer *, integer
	    *, integer *, integer *, integer *, integer *, integer *),
	    slaset_(char *, integer *, integer *, real *, real *, real *,
	    integer *);
    static integer smlszp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    Using a divide and conquer approach, SLASDA computes the singular
    value decomposition (SVD) of a real upper bidiagonal N-by-M matrix
    B with diagonal D and offdiagonal E, where M = N + SQRE. The
    algorithm computes the singular values in the SVD B = U * S * VT.
    The orthogonal matrices U and VT are optionally computed in
    compact form.

    A related subroutine, SLASD0, computes the singular values and
    the singular vectors in explicit form.

    Arguments
    =========

    ICOMPQ (input) INTEGER
           Specifies whether singular vectors are to be computed
           in compact form, as follows
           = 0: Compute singular values only.
           = 1: Compute singular vectors of upper bidiagonal
                matrix in compact form.

    SMLSIZ (input) INTEGER
           The maximum size of the subproblems at the bottom of the
           computation tree.

    N      (input) INTEGER
           The row dimension of the upper bidiagonal matrix. This is
           also the dimension of the main diagonal array D.

    SQRE   (input) INTEGER
           Specifies the column dimension of the bidiagonal matrix.
           = 0: The bidiagonal matrix has column dimension M = N;
           = 1: The bidiagonal matrix has column dimension M = N + 1.

    D      (input/output) REAL array, dimension ( N )
           On entry D contains the main diagonal of the bidiagonal
           matrix. On exit D, if INFO = 0, contains its singular values.

    E      (input) REAL array, dimension ( M-1 )
           Contains the subdiagonal entries of the bidiagonal matrix.
           On exit, E has been destroyed.

    U      (output) REAL array,
           dimension ( LDU, SMLSIZ ) if ICOMPQ = 1, and not referenced
           if ICOMPQ = 0. If ICOMPQ = 1, on exit, U contains the left
           singular vector matrices of all subproblems at the bottom
           level.

    LDU    (input) INTEGER, LDU = > N.
           The leading dimension of arrays U, VT, DIFL, DIFR, POLES,
           GIVNUM, and Z.

    VT     (output) REAL array,
           dimension ( LDU, SMLSIZ+1 ) if ICOMPQ = 1, and not referenced
           if ICOMPQ = 0. If ICOMPQ = 1, on exit, VT' contains the right
           singular vector matrices of all subproblems at the bottom
           level.

    K      (output) INTEGER array,
           dimension ( N ) if ICOMPQ = 1 and dimension 1 if ICOMPQ = 0.
           If ICOMPQ = 1, on exit, K(I) is the dimension of the I-th
           secular equation on the computation tree.

    DIFL   (output) REAL array, dimension ( LDU, NLVL ),
           where NLVL = floor(log_2 (N/SMLSIZ))).

    DIFR   (output) REAL array,
                    dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1 and
                    dimension ( N ) if ICOMPQ = 0.
           If ICOMPQ = 1, on exit, DIFL(1:N, I) and DIFR(1:N, 2 * I - 1)
           record distances between singular values on the I-th
           level and singular values on the (I -1)-th level, and
           DIFR(1:N, 2 * I ) contains the normalizing factors for
           the right singular vector matrix. See SLASD8 for details.

    Z      (output) REAL array,
                    dimension ( LDU, NLVL ) if ICOMPQ = 1 and
                    dimension ( N ) if ICOMPQ = 0.
           The first K elements of Z(1, I) contain the components of
           the deflation-adjusted updating row vector for subproblems
           on the I-th level.

    POLES  (output) REAL array,
           dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1, and not referenced
           if ICOMPQ = 0. If ICOMPQ = 1, on exit, POLES(1, 2*I - 1) and
           POLES(1, 2*I) contain  the new and old singular values
           involved in the secular equations on the I-th level.

    GIVPTR (output) INTEGER array,
           dimension ( N ) if ICOMPQ = 1, and not referenced if
           ICOMPQ = 0. If ICOMPQ = 1, on exit, GIVPTR( I ) records
           the number of Givens rotations performed on the I-th
           problem on the computation tree.

    GIVCOL (output) INTEGER array,
           dimension ( LDGCOL, 2 * NLVL ) if ICOMPQ = 1, and not
           referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I,
           GIVCOL(1, 2 *I - 1) and GIVCOL(1, 2 *I) record the locations
           of Givens rotations performed on the I-th level on the
           computation tree.

    LDGCOL (input) INTEGER, LDGCOL = > N.
           The leading dimension of arrays GIVCOL and PERM.

    PERM   (output) INTEGER array,
           dimension ( LDGCOL, NLVL ) if ICOMPQ = 1, and not referenced
           if ICOMPQ = 0. If ICOMPQ = 1, on exit, PERM(1, I) records
           permutations done on the I-th level of the computation tree.

    GIVNUM (output) REAL array,
           dimension ( LDU,  2 * NLVL ) if ICOMPQ = 1, and not
           referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I,
           GIVNUM(1, 2 *I - 1) and GIVNUM(1, 2 *I) record the C- and S-
           values of Givens rotations performed on the I-th level on
           the computation tree.

    C      (output) REAL array,
           dimension ( N ) if ICOMPQ = 1, and dimension 1 if ICOMPQ = 0.
           If ICOMPQ = 1 and the I-th subproblem is not square, on exit,
           C( I ) contains the C-value of a Givens rotation related to
           the right null space of the I-th subproblem.

    S      (output) REAL array, dimension ( N ) if
           ICOMPQ = 1, and dimension 1 if ICOMPQ = 0. If ICOMPQ = 1
           and the I-th subproblem is not square, on exit, S( I )
           contains the S-value of a Givens rotation related to
           the right null space of the I-th subproblem.

    WORK   (workspace) REAL array, dimension
           (6 * N + (SMLSIZ + 1)*(SMLSIZ + 1)).

    IWORK  (workspace) INTEGER array.
           Dimension must be at least (7 * N).

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an singular value did not converge

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    givnum_dim1 = *ldu;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    poles_dim1 = *ldu;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    z_dim1 = *ldu;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    difr_dim1 = *ldu;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    difl_dim1 = *ldu;
    difl_offset = 1 + difl_dim1;
    difl -= difl_offset;
    vt_dim1 = *ldu;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    --k;
    --givptr;
    perm_dim1 = *ldgcol;
    perm_offset = 1 + perm_dim1;
    perm -= perm_offset;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    --c__;
    --s;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*smlsiz < 3) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldu < *n + *sqre) {
	*info = -8;
    } else if (*ldgcol < *n) {
	*info = -17;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASDA", &i__1);
	return 0;
    }

    m = *n + *sqre;

/*     If the input matrix is too small, call SLASDQ to find the SVD. */

    if (*n <= *smlsiz) {
	if (*icompq == 0) {
	    slasdq_("U", sqre, n, &c__0, &c__0, &c__0, &d__[1], &e[1], &vt[
		    vt_offset], ldu, &u[u_offset], ldu, &u[u_offset], ldu, &
		    work[1], info);
	} else {
	    slasdq_("U", sqre, n, &m, n, &c__0, &d__[1], &e[1], &vt[vt_offset]
		    , ldu, &u[u_offset], ldu, &u[u_offset], ldu, &work[1],
		    info);
	}
	return 0;
    }

/*     Book-keeping and  set up the computation tree. */

    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;
    idxq = ndimr + *n;
    iwk = idxq + *n;

    ncc = 0;
    nru = 0;

    smlszp = *smlsiz + 1;
    vf = 1;
    vl = vf + m;
    nwork1 = vl + m;
    nwork2 = nwork1 + smlszp * smlszp;

    slasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr],
	    smlsiz);

/*
       for the nodes on bottom level of the tree, solve
       their subproblems by SLASDQ.
*/

    ndb1 = (nd + 1) / 2;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {

/*
          IC : center row of each node
          NL : number of rows of left  subproblem
          NR : number of rows of right subproblem
          NLF: starting row of the left   subproblem
          NRF: starting row of the right  subproblem
*/

	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nlp1 = nl + 1;
	nr = iwork[ndimr + i1];
	nlf = ic - nl;
	nrf = ic + 1;
	idxqi = idxq + nlf - 2;
	vfi = vf + nlf - 1;
	vli = vl + nlf - 1;
	sqrei = 1;
	if (*icompq == 0) {
	    slaset_("A", &nlp1, &nlp1, &c_b29, &c_b15, &work[nwork1], &smlszp);
	    slasdq_("U", &sqrei, &nl, &nlp1, &nru, &ncc, &d__[nlf], &e[nlf], &
		    work[nwork1], &smlszp, &work[nwork2], &nl, &work[nwork2],
		    &nl, &work[nwork2], info);
	    itemp = nwork1 + nl * smlszp;
	    scopy_(&nlp1, &work[nwork1], &c__1, &work[vfi], &c__1);
	    scopy_(&nlp1, &work[itemp], &c__1, &work[vli], &c__1);
	} else {
	    slaset_("A", &nl, &nl, &c_b29, &c_b15, &u[nlf + u_dim1], ldu);
	    slaset_("A", &nlp1, &nlp1, &c_b29, &c_b15, &vt[nlf + vt_dim1],
		    ldu);
	    slasdq_("U", &sqrei, &nl, &nlp1, &nl, &ncc, &d__[nlf], &e[nlf], &
		    vt[nlf + vt_dim1], ldu, &u[nlf + u_dim1], ldu, &u[nlf +
		    u_dim1], ldu, &work[nwork1], info);
	    scopy_(&nlp1, &vt[nlf + vt_dim1], &c__1, &work[vfi], &c__1);
	    scopy_(&nlp1, &vt[nlf + nlp1 * vt_dim1], &c__1, &work[vli], &c__1)
		    ;
	}
	if (*info != 0) {
	    return 0;
	}
	i__2 = nl;
	for (j = 1; j <= i__2; ++j) {
	    iwork[idxqi + j] = j;
/* L10: */
	}
	if (i__ == nd && *sqre == 0) {
	    sqrei = 0;
	} else {
	    sqrei = 1;
	}
	idxqi += nlp1;
	vfi += nlp1;
	vli += nlp1;
	nrp1 = nr + sqrei;
	if (*icompq == 0) {
	    slaset_("A", &nrp1, &nrp1, &c_b29, &c_b15, &work[nwork1], &smlszp);
	    slasdq_("U", &sqrei, &nr, &nrp1, &nru, &ncc, &d__[nrf], &e[nrf], &
		    work[nwork1], &smlszp, &work[nwork2], &nr, &work[nwork2],
		    &nr, &work[nwork2], info);
	    itemp = nwork1 + (nrp1 - 1) * smlszp;
	    scopy_(&nrp1, &work[nwork1], &c__1, &work[vfi], &c__1);
	    scopy_(&nrp1, &work[itemp], &c__1, &work[vli], &c__1);
	} else {
	    slaset_("A", &nr, &nr, &c_b29, &c_b15, &u[nrf + u_dim1], ldu);
	    slaset_("A", &nrp1, &nrp1, &c_b29, &c_b15, &vt[nrf + vt_dim1],
		    ldu);
	    slasdq_("U", &sqrei, &nr, &nrp1, &nr, &ncc, &d__[nrf], &e[nrf], &
		    vt[nrf + vt_dim1], ldu, &u[nrf + u_dim1], ldu, &u[nrf +
		    u_dim1], ldu, &work[nwork1], info);
	    scopy_(&nrp1, &vt[nrf + vt_dim1], &c__1, &work[vfi], &c__1);
	    scopy_(&nrp1, &vt[nrf + nrp1 * vt_dim1], &c__1, &work[vli], &c__1)
		    ;
	}
	if (*info != 0) {
	    return 0;
	}
	i__2 = nr;
	for (j = 1; j <= i__2; ++j) {
	    iwork[idxqi + j] = j;
/* L20: */
	}
/* L30: */
    }

/*     Now conquer each subproblem bottom-up. */

    j = pow_ii(&c__2, &nlvl);
    for (lvl = nlvl; lvl >= 1; --lvl) {
	lvl2 = (lvl << 1) - 1;

/*
          Find the first node LF and last node LL on
          the current level LVL.
*/

	if (lvl == 1) {
	    lf = 1;
	    ll = 1;
	} else {
	    i__1 = lvl - 1;
	    lf = pow_ii(&c__2, &i__1);
	    ll = (lf << 1) - 1;
	}
	i__1 = ll;
	for (i__ = lf; i__ <= i__1; ++i__) {
	    im1 = i__ - 1;
	    ic = iwork[inode + im1];
	    nl = iwork[ndiml + im1];
	    nr = iwork[ndimr + im1];
	    nlf = ic - nl;
	    nrf = ic + 1;
	    if (i__ == ll) {
		sqrei = *sqre;
	    } else {
		sqrei = 1;
	    }
	    vfi = vf + nlf - 1;
	    vli = vl + nlf - 1;
	    idxqi = idxq + nlf - 1;
	    alpha = d__[ic];
	    beta = e[ic];
	    if (*icompq == 0) {
		slasd6_(icompq, &nl, &nr, &sqrei, &d__[nlf], &work[vfi], &
			work[vli], &alpha, &beta, &iwork[idxqi], &perm[
			perm_offset], &givptr[1], &givcol[givcol_offset],
			ldgcol, &givnum[givnum_offset], ldu, &poles[
			poles_offset], &difl[difl_offset], &difr[difr_offset],
			 &z__[z_offset], &k[1], &c__[1], &s[1], &work[nwork1],
			 &iwork[iwk], info);
	    } else {
		--j;
		slasd6_(icompq, &nl, &nr, &sqrei, &d__[nlf], &work[vfi], &
			work[vli], &alpha, &beta, &iwork[idxqi], &perm[nlf +
			lvl * perm_dim1], &givptr[j], &givcol[nlf + lvl2 *
			givcol_dim1], ldgcol, &givnum[nlf + lvl2 *
			givnum_dim1], ldu, &poles[nlf + lvl2 * poles_dim1], &
			difl[nlf + lvl * difl_dim1], &difr[nlf + lvl2 *
			difr_dim1], &z__[nlf + lvl * z_dim1], &k[j], &c__[j],
			&s[j], &work[nwork1], &iwork[iwk], info);
	    }
	    if (*info != 0) {
		return 0;
	    }
/* L40: */
	}
/* L50: */
    }

    return 0;

/*     End of SLASDA */

} /* slasda_ */

/* Subroutine */ int slasdq_(char *uplo, integer *sqre, integer *n, integer *
	ncvt, integer *nru, integer *ncc, real *d__, real *e, real *vt,
	integer *ldvt, real *u, integer *ldu, real *c__, integer *ldc, real *
	work, integer *info)
{
    /* System generated locals */
    integer c_dim1, c_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1,
	    i__2;

    /* Local variables */
    static integer i__, j;
    static real r__, cs, sn;
    static integer np1, isub;
    static real smin;
    static integer sqre1;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int slasr_(char *, char *, char *, integer *,
	    integer *, real *, real *, real *, integer *);
    static integer iuplo;
    extern /* Subroutine */ int sswap_(integer *, real *, integer *, real *,
	    integer *), xerbla_(char *, integer *), slartg_(real *,
	    real *, real *, real *, real *);
    static logical rotate;
    extern /* Subroutine */ int sbdsqr_(char *, integer *, integer *, integer
	    *, integer *, real *, real *, real *, integer *, real *, integer *
	    , real *, integer *, real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASDQ computes the singular value decomposition (SVD) of a real
    (upper or lower) bidiagonal matrix with diagonal D and offdiagonal
    E, accumulating the transformations if desired. Letting B denote
    the input bidiagonal matrix, the algorithm computes orthogonal
    matrices Q and P such that B = Q * S * P' (P' denotes the transpose
    of P). The singular values S are overwritten on D.

    The input matrix U  is changed to U  * Q  if desired.
    The input matrix VT is changed to P' * VT if desired.
    The input matrix C  is changed to Q' * C  if desired.

    See "Computing  Small Singular Values of Bidiagonal Matrices With
    Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
    LAPACK Working Note #3, for a detailed description of the algorithm.

    Arguments
    =========

    UPLO  (input) CHARACTER*1
          On entry, UPLO specifies whether the input bidiagonal matrix
          is upper or lower bidiagonal, and wether it is square are
          not.
             UPLO = 'U' or 'u'   B is upper bidiagonal.
             UPLO = 'L' or 'l'   B is lower bidiagonal.

    SQRE  (input) INTEGER
          = 0: then the input matrix is N-by-N.
          = 1: then the input matrix is N-by-(N+1) if UPLU = 'U' and
               (N+1)-by-N if UPLU = 'L'.

          The bidiagonal matrix has
          N = NL + NR + 1 rows and
          M = N + SQRE >= N columns.

    N     (input) INTEGER
          On entry, N specifies the number of rows and columns
          in the matrix. N must be at least 0.

    NCVT  (input) INTEGER
          On entry, NCVT specifies the number of columns of
          the matrix VT. NCVT must be at least 0.

    NRU   (input) INTEGER
          On entry, NRU specifies the number of rows of
          the matrix U. NRU must be at least 0.

    NCC   (input) INTEGER
          On entry, NCC specifies the number of columns of
          the matrix C. NCC must be at least 0.

    D     (input/output) REAL array, dimension (N)
          On entry, D contains the diagonal entries of the
          bidiagonal matrix whose SVD is desired. On normal exit,
          D contains the singular values in ascending order.

    E     (input/output) REAL array.
          dimension is (N-1) if SQRE = 0 and N if SQRE = 1.
          On entry, the entries of E contain the offdiagonal entries
          of the bidiagonal matrix whose SVD is desired. On normal
          exit, E will contain 0. If the algorithm does not converge,
          D and E will contain the diagonal and superdiagonal entries
          of a bidiagonal matrix orthogonally equivalent to the one
          given as input.

    VT    (input/output) REAL array, dimension (LDVT, NCVT)
          On entry, contains a matrix which on exit has been
          premultiplied by P', dimension N-by-NCVT if SQRE = 0
          and (N+1)-by-NCVT if SQRE = 1 (not referenced if NCVT=0).

    LDVT  (input) INTEGER
          On entry, LDVT specifies the leading dimension of VT as
          declared in the calling (sub) program. LDVT must be at
          least 1. If NCVT is nonzero LDVT must also be at least N.

    U     (input/output) REAL array, dimension (LDU, N)
          On entry, contains a  matrix which on exit has been
          postmultiplied by Q, dimension NRU-by-N if SQRE = 0
          and NRU-by-(N+1) if SQRE = 1 (not referenced if NRU=0).

    LDU   (input) INTEGER
          On entry, LDU  specifies the leading dimension of U as
          declared in the calling (sub) program. LDU must be at
          least max( 1, NRU ) .

    C     (input/output) REAL array, dimension (LDC, NCC)
          On entry, contains an N-by-NCC matrix which on exit
          has been premultiplied by Q'  dimension N-by-NCC if SQRE = 0
          and (N+1)-by-NCC if SQRE = 1 (not referenced if NCC=0).

    LDC   (input) INTEGER
          On entry, LDC  specifies the leading dimension of C as
          declared in the calling (sub) program. LDC must be at
          least 1. If NCC is nonzero, LDC must also be at least N.

    WORK  (workspace) REAL array, dimension (4*N)
          Workspace. Only referenced if one of NCVT, NRU, or NCC is
          nonzero, and if N is at least 2.

    INFO  (output) INTEGER
          On exit, a value of 0 indicates a successful exit.
          If INFO < 0, argument number -INFO is illegal.
          If INFO > 0, the algorithm did not converge, and INFO
          specifies how many superdiagonals did not converge.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    iuplo = 0;
    if (lsame_(uplo, "U")) {
	iuplo = 1;
    }
    if (lsame_(uplo, "L")) {
	iuplo = 2;
    }
    if (iuplo == 0) {
	*info = -1;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ncvt < 0) {
	*info = -4;
    } else if (*nru < 0) {
	*info = -5;
    } else if (*ncc < 0) {
	*info = -6;
    } else if (*ncvt == 0 && *ldvt < 1 || *ncvt > 0 && *ldvt < max(1,*n)) {
	*info = -10;
    } else if (*ldu < max(1,*nru)) {
	*info = -12;
    } else if (*ncc == 0 && *ldc < 1 || *ncc > 0 && *ldc < max(1,*n)) {
	*info = -14;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASDQ", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }

/*     ROTATE is true if any singular vectors desired, false otherwise */

    rotate = *ncvt > 0 || *nru > 0 || *ncc > 0;
    np1 = *n + 1;
    sqre1 = *sqre;

/*
       If matrix non-square upper bidiagonal, rotate to be lower
       bidiagonal.  The rotations are on the right.
*/

    if (iuplo == 1 && sqre1 == 1) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (rotate) {
		work[i__] = cs;
		work[*n + i__] = sn;
	    }
/* L10: */
	}
	slartg_(&d__[*n], &e[*n], &cs, &sn, &r__);
	d__[*n] = r__;
	e[*n] = 0.f;
	if (rotate) {
	    work[*n] = cs;
	    work[*n + *n] = sn;
	}
	iuplo = 2;
	sqre1 = 0;

/*        Update singular vectors if desired. */

	if (*ncvt > 0) {
	    slasr_("L", "V", "F", &np1, ncvt, &work[1], &work[np1], &vt[
		    vt_offset], ldvt);
	}
    }

/*
       If matrix lower bidiagonal, rotate to be upper bidiagonal
       by applying Givens rotations on the left.
*/

    if (iuplo == 2) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (rotate) {
		work[i__] = cs;
		work[*n + i__] = sn;
	    }
/* L20: */
	}

/*
          If matrix (N+1)-by-N lower bidiagonal, one additional
          rotation is needed.
*/

	if (sqre1 == 1) {
	    slartg_(&d__[*n], &e[*n], &cs, &sn, &r__);
	    d__[*n] = r__;
	    if (rotate) {
		work[*n] = cs;
		work[*n + *n] = sn;
	    }
	}

/*        Update singular vectors if desired. */

	if (*nru > 0) {
	    if (sqre1 == 0) {
		slasr_("R", "V", "F", nru, n, &work[1], &work[np1], &u[
			u_offset], ldu);
	    } else {
		slasr_("R", "V", "F", nru, &np1, &work[1], &work[np1], &u[
			u_offset], ldu);
	    }
	}
	if (*ncc > 0) {
	    if (sqre1 == 0) {
		slasr_("L", "V", "F", n, ncc, &work[1], &work[np1], &c__[
			c_offset], ldc);
	    } else {
		slasr_("L", "V", "F", &np1, ncc, &work[1], &work[np1], &c__[
			c_offset], ldc);
	    }
	}
    }

/*
       Call SBDSQR to compute the SVD of the reduced real
       N-by-N upper bidiagonal matrix.
*/

    sbdsqr_("U", n, ncvt, nru, ncc, &d__[1], &e[1], &vt[vt_offset], ldvt, &u[
	    u_offset], ldu, &c__[c_offset], ldc, &work[1], info);

/*
       Sort the singular values into ascending order (insertion sort on
       singular values, but only one transposition per singular vector)
*/

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Scan for smallest D(I). */

	isub = i__;
	smin = d__[i__];
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    if (d__[j] < smin) {
		isub = j;
		smin = d__[j];
	    }
/* L30: */
	}
	if (isub != i__) {

/*           Swap singular values and vectors. */

	    d__[isub] = d__[i__];
	    d__[i__] = smin;
	    if (*ncvt > 0) {
		sswap_(ncvt, &vt[isub + vt_dim1], ldvt, &vt[i__ + vt_dim1],
			ldvt);
	    }
	    if (*nru > 0) {
		sswap_(nru, &u[isub * u_dim1 + 1], &c__1, &u[i__ * u_dim1 + 1]
			, &c__1);
	    }
	    if (*ncc > 0) {
		sswap_(ncc, &c__[isub + c_dim1], ldc, &c__[i__ + c_dim1], ldc)
			;
	    }
	}
/* L40: */
    }

    return 0;

/*     End of SLASDQ */

} /* slasdq_ */

/* Subroutine */ int slasdt_(integer *n, integer *lvl, integer *nd, integer *
	inode, integer *ndiml, integer *ndimr, integer *msub)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    static integer i__, il, ir, maxn;
    static real temp;
    static integer nlvl, llst, ncrnt;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASDT creates a tree of subproblems for bidiagonal divide and
    conquer.

    Arguments
    =========

     N      (input) INTEGER
            On entry, the number of diagonal elements of the
            bidiagonal matrix.

     LVL    (output) INTEGER
            On exit, the number of levels on the computation tree.

     ND     (output) INTEGER
            On exit, the number of nodes on the tree.

     INODE  (output) INTEGER array, dimension ( N )
            On exit, centers of subproblems.

     NDIML  (output) INTEGER array, dimension ( N )
            On exit, row dimensions of left children.

     NDIMR  (output) INTEGER array, dimension ( N )
            On exit, row dimensions of right children.

     MSUB   (input) INTEGER.
            On entry, the maximum row dimension each subproblem at the
            bottom of the tree can be of.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Huan Ren, Computer Science Division, University of
       California at Berkeley, USA

    =====================================================================


       Find the number of levels on the tree.
*/

    /* Parameter adjustments */
    --ndimr;
    --ndiml;
    --inode;

    /* Function Body */
    maxn = max(1,*n);
    temp = log((real) maxn / (real) (*msub + 1)) / log(2.f);
    *lvl = (integer) temp + 1;

    i__ = *n / 2;
    inode[1] = i__ + 1;
    ndiml[1] = i__;
    ndimr[1] = *n - i__ - 1;
    il = 0;
    ir = 1;
    llst = 1;
    i__1 = *lvl - 1;
    for (nlvl = 1; nlvl <= i__1; ++nlvl) {

/*
          Constructing the tree at (NLVL+1)-st level. The number of
          nodes created on this level is LLST * 2.
*/

	i__2 = llst - 1;
	for (i__ = 0; i__ <= i__2; ++i__) {
	    il += 2;
	    ir += 2;
	    ncrnt = llst + i__;
	    ndiml[il] = ndiml[ncrnt] / 2;
	    ndimr[il] = ndiml[ncrnt] - ndiml[il] - 1;
	    inode[il] = inode[ncrnt] - ndimr[il] - 1;
	    ndiml[ir] = ndimr[ncrnt] / 2;
	    ndimr[ir] = ndimr[ncrnt] - ndiml[ir] - 1;
	    inode[ir] = inode[ncrnt] + ndiml[ir] + 1;
/* L10: */
	}
	llst <<= 1;
/* L20: */
    }
    *nd = (llst << 1) - 1;

    return 0;

/*     End of SLASDT */

} /* slasdt_ */

/* Subroutine */ int slaset_(char *uplo, integer *m, integer *n, real *alpha,
	real *beta, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLASET initializes an m-by-n matrix A to BETA on the diagonal and
    ALPHA on the offdiagonals.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies the part of the matrix A to be set.
            = 'U':      Upper triangular part is set; the strictly lower
                        triangular part of A is not changed.
            = 'L':      Lower triangular part is set; the strictly upper
                        triangular part of A is not changed.
            Otherwise:  All of the matrix A is set.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    ALPHA   (input) REAL
            The constant to which the offdiagonal elements are to be set.

    BETA    (input) REAL
            The constant to which the diagonal elements are to be set.

    A       (input/output) REAL array, dimension (LDA,N)
            On exit, the leading m-by-n submatrix of A is set as follows:

            if UPLO = 'U', A(i,j) = ALPHA, 1<=i<=j-1, 1<=j<=n,
            if UPLO = 'L', A(i,j) = ALPHA, j+1<=i<=m, 1<=j<=n,
            otherwise,     A(i,j) = ALPHA, 1<=i<=m, 1<=j<=n, i.ne.j,

            and, for all UPLO, A(i,i) = BETA, 1<=i<=min(m,n).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

   =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    if (lsame_(uplo, "U")) {

/*
          Set the strictly upper triangular or trapezoidal part of the
          array to ALPHA.
*/

	i__1 = *n;
	for (j = 2; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = j - 1;
	    i__2 = min(i__3,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
/* L10: */
	    }
/* L20: */
	}

    } else if (lsame_(uplo, "L")) {

/*
          Set the strictly lower triangular or trapezoidal part of the
          array to ALPHA.
*/

	i__1 = min(*m,*n);
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j + 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
/* L30: */
	    }
/* L40: */
	}

    } else {

/*        Set the leading m-by-n submatrix to ALPHA. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
/* L50: */
	    }
/* L60: */
	}
    }

/*     Set the first min(M,N) diagonal elements to BETA. */

    i__1 = min(*m,*n);
    for (i__ = 1; i__ <= i__1; ++i__) {
	a[i__ + i__ * a_dim1] = *beta;
/* L70: */
    }

    return 0;

/*     End of SLASET */

} /* slaset_ */

/* Subroutine */ int slasq1_(integer *n, real *d__, real *e, real *work,
	integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    real r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__;
    static real eps;
    extern /* Subroutine */ int slas2_(real *, real *, real *, real *, real *)
	    ;
    static real scale;
    static integer iinfo;
    static real sigmn, sigmx;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *), slasq2_(integer *, real *, integer *);
    extern doublereal slamch_(char *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *), slascl_(
	    char *, integer *, integer *, real *, real *, integer *, integer *
	    , real *, integer *, integer *), slasrt_(char *, integer *
	    , real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASQ1 computes the singular values of a real N-by-N bidiagonal
    matrix with diagonal D and off-diagonal E. The singular values
    are computed to high relative accuracy, in the absence of
    denormalization, underflow and overflow. The algorithm was first
    presented in

    "Accurate singular values and differential qd algorithms" by K. V.
    Fernando and B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230,
    1994,

    and the present implementation is described in "An implementation of
    the dqds Algorithm (Positive Case)", LAPACK Working Note.

    Arguments
    =========

    N     (input) INTEGER
          The number of rows and columns in the matrix. N >= 0.

    D     (input/output) REAL array, dimension (N)
          On entry, D contains the diagonal elements of the
          bidiagonal matrix whose SVD is desired. On normal exit,
          D contains the singular values in decreasing order.

    E     (input/output) REAL array, dimension (N)
          On entry, elements E(1:N-1) contain the off-diagonal elements
          of the bidiagonal matrix whose SVD is desired.
          On exit, E is overwritten.

    WORK  (workspace) REAL array, dimension (4*N)

    INFO  (output) INTEGER
          = 0: successful exit
          < 0: if INFO = -i, the i-th argument had an illegal value
          > 0: the algorithm failed
               = 1, a split was marked by a positive value in E
               = 2, current block of Z not diagonalized after 30*N
                    iterations (in inner while loop)
               = 3, termination criterion of outer while loop not met
                    (program created more than N unreduced blocks)

    =====================================================================
*/


    /* Parameter adjustments */
    --work;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -2;
	i__1 = -(*info);
	xerbla_("SLASQ1", &i__1);
	return 0;
    } else if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	d__[1] = dabs(d__[1]);
	return 0;
    } else if (*n == 2) {
	slas2_(&d__[1], &e[1], &d__[2], &sigmn, &sigmx);
	d__[1] = sigmx;
	d__[2] = sigmn;
	return 0;
    }

/*     Estimate the largest singular value. */

    sigmx = 0.f;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = (r__1 = d__[i__], dabs(r__1));
/* Computing MAX */
	r__2 = sigmx, r__3 = (r__1 = e[i__], dabs(r__1));
	sigmx = dmax(r__2,r__3);
/* L10: */
    }
    d__[*n] = (r__1 = d__[*n], dabs(r__1));

/*     Early return if SIGMX is zero (matrix is already diagonal). */

    if (sigmx == 0.f) {
	slasrt_("D", n, &d__[1], &iinfo);
	return 0;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	r__1 = sigmx, r__2 = d__[i__];
	sigmx = dmax(r__1,r__2);
/* L20: */
    }

/*
       Copy D and E into WORK (in the Z format) and scale (squaring the
       input data makes scaling by a power of the radix pointless).
*/

    eps = slamch_("Precision");
    safmin = slamch_("Safe minimum");
    scale = sqrt(eps / safmin);
    scopy_(n, &d__[1], &c__1, &work[1], &c__2);
    i__1 = *n - 1;
    scopy_(&i__1, &e[1], &c__1, &work[2], &c__2);
    i__1 = (*n << 1) - 1;
    i__2 = (*n << 1) - 1;
    slascl_("G", &c__0, &c__0, &sigmx, &scale, &i__1, &c__1, &work[1], &i__2,
	    &iinfo);

/*     Compute the q's and e's. */

    i__1 = (*n << 1) - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	r__1 = work[i__];
	work[i__] = r__1 * r__1;
/* L30: */
    }
    work[*n * 2] = 0.f;

    slasq2_(n, &work[1], info);

    if (*info == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] = sqrt(work[i__]);
/* L40: */
	}
	slascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &d__[1], n, &
		iinfo);
    }

    return 0;

/*     End of SLASQ1 */

} /* slasq1_ */

/* Subroutine */ int slasq2_(integer *n, real *z__, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real d__, e;
    static integer k;
    static real s, t;
    static integer i0, i4, n0, pp;
    static real eps, tol;
    static integer ipn4;
    static real tol2;
    static logical ieee;
    static integer nbig;
    static real dmin__, emin, emax;
    static integer ndiv, iter;
    static real qmin, temp, qmax, zmax;
    static integer splt, nfail;
    static real desig, trace, sigma;
    static integer iinfo;
    extern /* Subroutine */ int slasq3_(integer *, integer *, real *, integer
	    *, real *, real *, real *, real *, integer *, integer *, integer *
	    , logical *);
    extern doublereal slamch_(char *);
    static integer iwhila, iwhilb;
    static real oldemn, safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slasrt_(char *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASQ2 computes all the eigenvalues of the symmetric positive
    definite tridiagonal matrix associated with the qd array Z to high
    relative accuracy are computed to high relative accuracy, in the
    absence of denormalization, underflow and overflow.

    To see the relation of Z to the tridiagonal matrix, let L be a
    unit lower bidiagonal matrix with subdiagonals Z(2,4,6,,..) and
    let U be an upper bidiagonal matrix with 1's above and diagonal
    Z(1,3,5,,..). The tridiagonal is L*U or, if you prefer, the
    symmetric tridiagonal to which it is similar.

    Note : SLASQ2 defines a logical variable, IEEE, which is true
    on machines which follow ieee-754 floating-point standard in their
    handling of infinities and NaNs, and false otherwise. This variable
    is passed to SLASQ3.

    Arguments
    =========

    N     (input) INTEGER
          The number of rows and columns in the matrix. N >= 0.

    Z     (workspace) REAL array, dimension ( 4*N )
          On entry Z holds the qd array. On exit, entries 1 to N hold
          the eigenvalues in decreasing order, Z( 2*N+1 ) holds the
          trace, and Z( 2*N+2 ) holds the sum of the eigenvalues. If
          N > 2, then Z( 2*N+3 ) holds the iteration count, Z( 2*N+4 )
          holds NDIVS/NIN^2, and Z( 2*N+5 ) holds the percentage of
          shifts that failed.

    INFO  (output) INTEGER
          = 0: successful exit
          < 0: if the i-th argument is a scalar and had an illegal
               value, then INFO = -i, if the i-th argument is an
               array and the j-entry had an illegal value, then
               INFO = -(i*100+j)
          > 0: the algorithm failed
                = 1, a split was marked by a positive value in E
                = 2, current block of Z not diagonalized after 30*N
                     iterations (in inner while loop)
                = 3, termination criterion of outer while loop not met
                     (program created more than N unreduced blocks)

    Further Details
    ===============
    Local Variables: I0:N0 defines a current unreduced segment of Z.
    The shifts are accumulated in SIGMA. Iteration count is in ITER.
    Ping-pong is controlled by PP (alternates between 0 and 1).

    =====================================================================


       Test the input arguments.
       (in case SLASQ2 is not called by SLASQ1)
*/

    /* Parameter adjustments */
    --z__;

    /* Function Body */
    *info = 0;
    eps = slamch_("Precision");
    safmin = slamch_("Safe minimum");
    tol = eps * 100.f;
/* Computing 2nd power */
    r__1 = tol;
    tol2 = r__1 * r__1;

    if (*n < 0) {
	*info = -1;
	xerbla_("SLASQ2", &c__1);
	return 0;
    } else if (*n == 0) {
	return 0;
    } else if (*n == 1) {

/*        1-by-1 case. */

	if (z__[1] < 0.f) {
	    *info = -201;
	    xerbla_("SLASQ2", &c__2);
	}
	return 0;
    } else if (*n == 2) {

/*        2-by-2 case. */

	if (z__[2] < 0.f || z__[3] < 0.f) {
	    *info = -2;
	    xerbla_("SLASQ2", &c__2);
	    return 0;
	} else if (z__[3] > z__[1]) {
	    d__ = z__[3];
	    z__[3] = z__[1];
	    z__[1] = d__;
	}
	z__[5] = z__[1] + z__[2] + z__[3];
	if (z__[2] > z__[3] * tol2) {
	    t = (z__[1] - z__[3] + z__[2]) * .5f;
	    s = z__[3] * (z__[2] / t);
	    if (s <= t) {
		s = z__[3] * (z__[2] / (t * (sqrt(s / t + 1.f) + 1.f)));
	    } else {
		s = z__[3] * (z__[2] / (t + sqrt(t) * sqrt(t + s)));
	    }
	    t = z__[1] + (s + z__[2]);
	    z__[3] *= z__[1] / t;
	    z__[1] = t;
	}
	z__[2] = z__[3];
	z__[6] = z__[2] + z__[1];
	return 0;
    }

/*     Check for negative data and compute sums of q's and e's. */

    z__[*n * 2] = 0.f;
    emin = z__[2];
    qmax = 0.f;
    zmax = 0.f;
    d__ = 0.f;
    e = 0.f;

    i__1 = *n - 1 << 1;
    for (k = 1; k <= i__1; k += 2) {
	if (z__[k] < 0.f) {
	    *info = -(k + 200);
	    xerbla_("SLASQ2", &c__2);
	    return 0;
	} else if (z__[k + 1] < 0.f) {
	    *info = -(k + 201);
	    xerbla_("SLASQ2", &c__2);
	    return 0;
	}
	d__ += z__[k];
	e += z__[k + 1];
/* Computing MAX */
	r__1 = qmax, r__2 = z__[k];
	qmax = dmax(r__1,r__2);
/* Computing MIN */
	r__1 = emin, r__2 = z__[k + 1];
	emin = dmin(r__1,r__2);
/* Computing MAX */
	r__1 = max(qmax,zmax), r__2 = z__[k + 1];
	zmax = dmax(r__1,r__2);
/* L10: */
    }
    if (z__[(*n << 1) - 1] < 0.f) {
	*info = -((*n << 1) + 199);
	xerbla_("SLASQ2", &c__2);
	return 0;
    }
    d__ += z__[(*n << 1) - 1];
/* Computing MAX */
    r__1 = qmax, r__2 = z__[(*n << 1) - 1];
    qmax = dmax(r__1,r__2);
    zmax = dmax(qmax,zmax);

/*     Check for diagonality. */

    if (e == 0.f) {
	i__1 = *n;
	for (k = 2; k <= i__1; ++k) {
	    z__[k] = z__[(k << 1) - 1];
/* L20: */
	}
	slasrt_("D", n, &z__[1], &iinfo);
	z__[(*n << 1) - 1] = d__;
	return 0;
    }

    trace = d__ + e;

/*     Check for zero data. */

    if (trace == 0.f) {
	z__[(*n << 1) - 1] = 0.f;
	return 0;
    }

/*     Check whether the machine is IEEE conformable. */

    ieee = ilaenv_(&c__10, "SLASQ2", "N", &c__1, &c__2, &c__3, &c__4, (ftnlen)
	    6, (ftnlen)1) == 1 && ilaenv_(&c__11, "SLASQ2", "N", &c__1, &c__2,
	     &c__3, &c__4, (ftnlen)6, (ftnlen)1) == 1;

/*     Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...). */

    for (k = *n << 1; k >= 2; k += -2) {
	z__[k * 2] = 0.f;
	z__[(k << 1) - 1] = z__[k];
	z__[(k << 1) - 2] = 0.f;
	z__[(k << 1) - 3] = z__[k - 1];
/* L30: */
    }

    i0 = 1;
    n0 = *n;

/*     Reverse the qd-array, if warranted. */

    if (z__[(i0 << 2) - 3] * 1.5f < z__[(n0 << 2) - 3]) {
	ipn4 = i0 + n0 << 2;
	i__1 = i0 + n0 - 1 << 1;
	for (i4 = i0 << 2; i4 <= i__1; i4 += 4) {
	    temp = z__[i4 - 3];
	    z__[i4 - 3] = z__[ipn4 - i4 - 3];
	    z__[ipn4 - i4 - 3] = temp;
	    temp = z__[i4 - 1];
	    z__[i4 - 1] = z__[ipn4 - i4 - 5];
	    z__[ipn4 - i4 - 5] = temp;
/* L40: */
	}
    }

/*     Initial split checking via dqd and Li's test. */

    pp = 0;

    for (k = 1; k <= 2; ++k) {

	d__ = z__[(n0 << 2) + pp - 3];
	i__1 = (i0 << 2) + pp;
	for (i4 = (n0 - 1 << 2) + pp; i4 >= i__1; i4 += -4) {
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.f;
		d__ = z__[i4 - 3];
	    } else {
		d__ = z__[i4 - 3] * (d__ / (d__ + z__[i4 - 1]));
	    }
/* L50: */
	}

/*        dqd maps Z to ZZ plus Li's test. */

	emin = z__[(i0 << 2) + pp + 1];
	d__ = z__[(i0 << 2) + pp - 3];
	i__1 = (n0 - 1 << 2) + pp;
	for (i4 = (i0 << 2) + pp; i4 <= i__1; i4 += 4) {
	    z__[i4 - (pp << 1) - 2] = d__ + z__[i4 - 1];
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.f;
		z__[i4 - (pp << 1) - 2] = d__;
		z__[i4 - (pp << 1)] = 0.f;
		d__ = z__[i4 + 1];
	    } else if (safmin * z__[i4 + 1] < z__[i4 - (pp << 1) - 2] &&
		    safmin * z__[i4 - (pp << 1) - 2] < z__[i4 + 1]) {
		temp = z__[i4 + 1] / z__[i4 - (pp << 1) - 2];
		z__[i4 - (pp << 1)] = z__[i4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[i4 - (pp << 1)] = z__[i4 + 1] * (z__[i4 - 1] / z__[i4 - (
			pp << 1) - 2]);
		d__ = z__[i4 + 1] * (d__ / z__[i4 - (pp << 1) - 2]);
	    }
/* Computing MIN */
	    r__1 = emin, r__2 = z__[i4 - (pp << 1)];
	    emin = dmin(r__1,r__2);
/* L60: */
	}
	z__[(n0 << 2) - pp - 2] = d__;

/*        Now find qmax. */

	qmax = z__[(i0 << 2) - pp - 2];
	i__1 = (n0 << 2) - pp - 2;
	for (i4 = (i0 << 2) - pp + 2; i4 <= i__1; i4 += 4) {
/* Computing MAX */
	    r__1 = qmax, r__2 = z__[i4];
	    qmax = dmax(r__1,r__2);
/* L70: */
	}

/*        Prepare for the next iteration on K. */

	pp = 1 - pp;
/* L80: */
    }

    iter = 2;
    nfail = 0;
    ndiv = n0 - i0 << 1;

    i__1 = *n + 1;
    for (iwhila = 1; iwhila <= i__1; ++iwhila) {
	if (n0 < 1) {
	    goto L150;
	}

/*
          While array unfinished do

          E(N0) holds the value of SIGMA when submatrix in I0:N0
          splits from the rest of the array, but is negated.
*/

	desig = 0.f;
	if (n0 == *n) {
	    sigma = 0.f;
	} else {
	    sigma = -z__[(n0 << 2) - 1];
	}
	if (sigma < 0.f) {
	    *info = 1;
	    return 0;
	}

/*
          Find last unreduced submatrix's top index I0, find QMAX and
          EMIN. Find Gershgorin-type bound if Q's much greater than E's.
*/

	emax = 0.f;
	if (n0 > i0) {
	    emin = (r__1 = z__[(n0 << 2) - 5], dabs(r__1));
	} else {
	    emin = 0.f;
	}
	qmin = z__[(n0 << 2) - 3];
	qmax = qmin;
	for (i4 = n0 << 2; i4 >= 8; i4 += -4) {
	    if (z__[i4 - 5] <= 0.f) {
		goto L100;
	    }
	    if (qmin >= emax * 4.f) {
/* Computing MIN */
		r__1 = qmin, r__2 = z__[i4 - 3];
		qmin = dmin(r__1,r__2);
/* Computing MAX */
		r__1 = emax, r__2 = z__[i4 - 5];
		emax = dmax(r__1,r__2);
	    }
/* Computing MAX */
	    r__1 = qmax, r__2 = z__[i4 - 7] + z__[i4 - 5];
	    qmax = dmax(r__1,r__2);
/* Computing MIN */
	    r__1 = emin, r__2 = z__[i4 - 5];
	    emin = dmin(r__1,r__2);
/* L90: */
	}
	i4 = 4;

L100:
	i0 = i4 / 4;

/*        Store EMIN for passing to SLASQ3. */

	z__[(n0 << 2) - 1] = emin;

/*
          Put -(initial shift) into DMIN.

   Computing MAX
*/
	r__1 = 0.f, r__2 = qmin - sqrt(qmin) * 2.f * sqrt(emax);
	dmin__ = -dmax(r__1,r__2);

/*        Now I0:N0 is unreduced. PP = 0 for ping, PP = 1 for pong. */

	pp = 0;

	nbig = (n0 - i0 + 1) * 30;
	i__2 = nbig;
	for (iwhilb = 1; iwhilb <= i__2; ++iwhilb) {
	    if (i0 > n0) {
		goto L130;
	    }

/*           While submatrix unfinished take a good dqds step. */

	    slasq3_(&i0, &n0, &z__[1], &pp, &dmin__, &sigma, &desig, &qmax, &
		    nfail, &iter, &ndiv, &ieee);

	    pp = 1 - pp;

/*           When EMIN is very small check for splits. */

	    if (pp == 0 && n0 - i0 >= 3) {
		if (z__[n0 * 4] <= tol2 * qmax || z__[(n0 << 2) - 1] <= tol2 *
			 sigma) {
		    splt = i0 - 1;
		    qmax = z__[(i0 << 2) - 3];
		    emin = z__[(i0 << 2) - 1];
		    oldemn = z__[i0 * 4];
		    i__3 = n0 - 3 << 2;
		    for (i4 = i0 << 2; i4 <= i__3; i4 += 4) {
			if (z__[i4] <= tol2 * z__[i4 - 3] || z__[i4 - 1] <=
				tol2 * sigma) {
			    z__[i4 - 1] = -sigma;
			    splt = i4 / 4;
			    qmax = 0.f;
			    emin = z__[i4 + 3];
			    oldemn = z__[i4 + 4];
			} else {
/* Computing MAX */
			    r__1 = qmax, r__2 = z__[i4 + 1];
			    qmax = dmax(r__1,r__2);
/* Computing MIN */
			    r__1 = emin, r__2 = z__[i4 - 1];
			    emin = dmin(r__1,r__2);
/* Computing MIN */
			    r__1 = oldemn, r__2 = z__[i4];
			    oldemn = dmin(r__1,r__2);
			}
/* L110: */
		    }
		    z__[(n0 << 2) - 1] = emin;
		    z__[n0 * 4] = oldemn;
		    i0 = splt + 1;
		}
	    }

/* L120: */
	}

	*info = 2;
	return 0;

/*        end IWHILB */

L130:

/* L140: */
	;
    }

    *info = 3;
    return 0;

/*     end IWHILA */

L150:

/*     Move q's to the front. */

    i__1 = *n;
    for (k = 2; k <= i__1; ++k) {
	z__[k] = z__[(k << 2) - 3];
/* L160: */
    }

/*     Sort and compute sum of eigenvalues. */

    slasrt_("D", n, &z__[1], &iinfo);

    e = 0.f;
    for (k = *n; k >= 1; --k) {
	e += z__[k];
/* L170: */
    }

/*     Store trace, sum(eigenvalues) and information on performance. */

    z__[(*n << 1) + 1] = trace;
    z__[(*n << 1) + 2] = e;
    z__[(*n << 1) + 3] = (real) iter;
/* Computing 2nd power */
    i__1 = *n;
    z__[(*n << 1) + 4] = (real) ndiv / (real) (i__1 * i__1);
    z__[(*n << 1) + 5] = nfail * 100.f / (real) iter;
    return 0;

/*     End of SLASQ2 */

} /* slasq2_ */

/* Subroutine */ int slasq3_(integer *i0, integer *n0, real *z__, integer *pp,
	 real *dmin__, real *sigma, real *desig, real *qmax, integer *nfail,
	integer *iter, integer *ndiv, logical *ieee)
{
    /* Initialized data */

    static integer ttype = 0;
    static real dmin1 = 0.f;
    static real dmin2 = 0.f;
    static real dn = 0.f;
    static real dn1 = 0.f;
    static real dn2 = 0.f;
    static real tau = 0.f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real s, t;
    static integer j4, nn;
    static real eps, tol;
    static integer n0in, ipn4;
    static real tol2, temp;
    extern /* Subroutine */ int slasq4_(integer *, integer *, real *, integer
	    *, integer *, real *, real *, real *, real *, real *, real *,
	    real *, integer *), slasq5_(integer *, integer *, real *, integer
	    *, real *, real *, real *, real *, real *, real *, real *,
	    logical *), slasq6_(integer *, integer *, real *, integer *, real
	    *, real *, real *, real *, real *, real *);
    extern doublereal slamch_(char *);
    static real safmin;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       May 17, 2000


    Purpose
    =======

    SLASQ3 checks for deflation, computes a shift (TAU) and calls dqds.
    In case of failure it changes shifts, and tries again until output
    is positive.

    Arguments
    =========

    I0     (input) INTEGER
           First index.

    N0     (input) INTEGER
           Last index.

    Z      (input) REAL array, dimension ( 4*N )
           Z holds the qd array.

    PP     (input) INTEGER
           PP=0 for ping, PP=1 for pong.

    DMIN   (output) REAL
           Minimum value of d.

    SIGMA  (output) REAL
           Sum of shifts used in current segment.

    DESIG  (input/output) REAL
           Lower order part of SIGMA

    QMAX   (input) REAL
           Maximum value of q.

    NFAIL  (output) INTEGER
           Number of times shift was too big.

    ITER   (output) INTEGER
           Number of iterations.

    NDIV   (output) INTEGER
           Number of divisions.

    TTYPE  (output) INTEGER
           Shift type.

    IEEE   (input) LOGICAL
           Flag for IEEE or non IEEE arithmetic (passed to SLASQ5).

    =====================================================================
*/

    /* Parameter adjustments */
    --z__;

    /* Function Body */

    n0in = *n0;
    eps = slamch_("Precision");
    safmin = slamch_("Safe minimum");
    tol = eps * 100.f;
/* Computing 2nd power */
    r__1 = tol;
    tol2 = r__1 * r__1;

/*     Check for deflation. */

L10:

    if (*n0 < *i0) {
	return 0;
    }
    if (*n0 == *i0) {
	goto L20;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0 == *i0 + 1) {
	goto L40;
    }

/*     Check whether E(N0-1) is negligible, 1 eigenvalue. */

    if (z__[nn - 5] > tol2 * (*sigma + z__[nn - 3]) && z__[nn - (*pp << 1) -
	    4] > tol2 * z__[nn - 7]) {
	goto L30;
    }

L20:

    z__[(*n0 << 2) - 3] = z__[(*n0 << 2) + *pp - 3] + *sigma;
    --(*n0);
    goto L10;

/*     Check  whether E(N0-2) is negligible, 2 eigenvalues. */

L30:

    if (z__[nn - 9] > tol2 * *sigma && z__[nn - (*pp << 1) - 8] > tol2 * z__[
	    nn - 11]) {
	goto L50;
    }

L40:

    if (z__[nn - 3] > z__[nn - 7]) {
	s = z__[nn - 3];
	z__[nn - 3] = z__[nn - 7];
	z__[nn - 7] = s;
    }
    if (z__[nn - 5] > z__[nn - 3] * tol2) {
	t = (z__[nn - 7] - z__[nn - 3] + z__[nn - 5]) * .5f;
	s = z__[nn - 3] * (z__[nn - 5] / t);
	if (s <= t) {
	    s = z__[nn - 3] * (z__[nn - 5] / (t * (sqrt(s / t + 1.f) + 1.f)));
	} else {
	    s = z__[nn - 3] * (z__[nn - 5] / (t + sqrt(t) * sqrt(t + s)));
	}
	t = z__[nn - 7] + (s + z__[nn - 5]);
	z__[nn - 3] *= z__[nn - 7] / t;
	z__[nn - 7] = t;
    }
    z__[(*n0 << 2) - 7] = z__[nn - 7] + *sigma;
    z__[(*n0 << 2) - 3] = z__[nn - 3] + *sigma;
    *n0 += -2;
    goto L10;

L50:

/*     Reverse the qd-array, if warranted. */

    if (*dmin__ <= 0.f || *n0 < n0in) {
	if (z__[(*i0 << 2) + *pp - 3] * 1.5f < z__[(*n0 << 2) + *pp - 3]) {
	    ipn4 = *i0 + *n0 << 2;
	    i__1 = *i0 + *n0 - 1 << 1;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		temp = z__[j4 - 3];
		z__[j4 - 3] = z__[ipn4 - j4 - 3];
		z__[ipn4 - j4 - 3] = temp;
		temp = z__[j4 - 2];
		z__[j4 - 2] = z__[ipn4 - j4 - 2];
		z__[ipn4 - j4 - 2] = temp;
		temp = z__[j4 - 1];
		z__[j4 - 1] = z__[ipn4 - j4 - 5];
		z__[ipn4 - j4 - 5] = temp;
		temp = z__[j4];
		z__[j4] = z__[ipn4 - j4 - 4];
		z__[ipn4 - j4 - 4] = temp;
/* L60: */
	    }
	    if (*n0 - *i0 <= 4) {
		z__[(*n0 << 2) + *pp - 1] = z__[(*i0 << 2) + *pp - 1];
		z__[(*n0 << 2) - *pp] = z__[(*i0 << 2) - *pp];
	    }
/* Computing MIN */
	    r__1 = dmin2, r__2 = z__[(*n0 << 2) + *pp - 1];
	    dmin2 = dmin(r__1,r__2);
/* Computing MIN */
	    r__1 = z__[(*n0 << 2) + *pp - 1], r__2 = z__[(*i0 << 2) + *pp - 1]
		    , r__1 = min(r__1,r__2), r__2 = z__[(*i0 << 2) + *pp + 3];
	    z__[(*n0 << 2) + *pp - 1] = dmin(r__1,r__2);
/* Computing MIN */
	    r__1 = z__[(*n0 << 2) - *pp], r__2 = z__[(*i0 << 2) - *pp], r__1 =
		     min(r__1,r__2), r__2 = z__[(*i0 << 2) - *pp + 4];
	    z__[(*n0 << 2) - *pp] = dmin(r__1,r__2);
/* Computing MAX */
	    r__1 = *qmax, r__2 = z__[(*i0 << 2) + *pp - 3], r__1 = max(r__1,
		    r__2), r__2 = z__[(*i0 << 2) + *pp + 1];
	    *qmax = dmax(r__1,r__2);
	    *dmin__ = -0.f;
	}
    }

/*
   L70:

   Computing MIN
*/
    r__1 = z__[(*n0 << 2) + *pp - 1], r__2 = z__[(*n0 << 2) + *pp - 9], r__1 =
	     min(r__1,r__2), r__2 = dmin2 + z__[(*n0 << 2) - *pp];
    if (*dmin__ < 0.f || safmin * *qmax < dmin(r__1,r__2)) {

/*        Choose a shift. */

	slasq4_(i0, n0, &z__[1], pp, &n0in, dmin__, &dmin1, &dmin2, &dn, &dn1,
		 &dn2, &tau, &ttype);

/*        Call dqds until DMIN > 0. */

L80:

	slasq5_(i0, n0, &z__[1], pp, &tau, dmin__, &dmin1, &dmin2, &dn, &dn1,
		&dn2, ieee);

	*ndiv += *n0 - *i0 + 2;
	++(*iter);

/*        Check status. */

	if (*dmin__ >= 0.f && dmin1 > 0.f) {

/*           Success. */

	    goto L100;

	} else if (*dmin__ < 0.f && dmin1 > 0.f && z__[(*n0 - 1 << 2) - *pp] <
		 tol * (*sigma + dn1) && dabs(dn) < tol * *sigma) {

/*           Convergence hidden by negative DN. */

	    z__[(*n0 - 1 << 2) - *pp + 2] = 0.f;
	    *dmin__ = 0.f;
	    goto L100;
	} else if (*dmin__ < 0.f) {

/*           TAU too big. Select new TAU and try again. */

	    ++(*nfail);
	    if (ttype < -22) {

/*              Failed twice. Play it safe. */

		tau = 0.f;
	    } else if (dmin1 > 0.f) {

/*              Late failure. Gives excellent shift. */

		tau = (tau + *dmin__) * (1.f - eps * 2.f);
		ttype += -11;
	    } else {

/*              Early failure. Divide by 4. */

		tau *= .25f;
		ttype += -12;
	    }
	    goto L80;
	} else if (*dmin__ != *dmin__) {

/*           NaN. */

	    tau = 0.f;
	    goto L80;
	} else {

/*           Possible underflow. Play it safe. */

	    goto L90;
	}
    }

/*     Risk of underflow. */

L90:
    slasq6_(i0, n0, &z__[1], pp, dmin__, &dmin1, &dmin2, &dn, &dn1, &dn2);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    tau = 0.f;

L100:
    if (tau < *sigma) {
	*desig += tau;
	t = *sigma + *desig;
	*desig -= t - *sigma;
    } else {
	t = *sigma + tau;
	*desig = *sigma - (t - tau) + *desig;
    }
    *sigma = t;

    return 0;

/*     End of SLASQ3 */

} /* slasq3_ */

/* Subroutine */ int slasq4_(integer *i0, integer *n0, real *z__, integer *pp,
	 integer *n0in, real *dmin__, real *dmin1, real *dmin2, real *dn,
	real *dn1, real *dn2, real *tau, integer *ttype)
{
    /* Initialized data */

    static real g = 0.f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real s, a2, b1, b2;
    static integer i4, nn, np;
    static real gam, gap1, gap2;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASQ4 computes an approximation TAU to the smallest eigenvalue
    using values of d from the previous transform.

    I0    (input) INTEGER
          First index.

    N0    (input) INTEGER
          Last index.

    Z     (input) REAL array, dimension ( 4*N )
          Z holds the qd array.

    PP    (input) INTEGER
          PP=0 for ping, PP=1 for pong.

    NOIN  (input) INTEGER
          The value of N0 at start of EIGTEST.

    DMIN  (input) REAL
          Minimum value of d.

    DMIN1 (input) REAL
          Minimum value of d, excluding D( N0 ).

    DMIN2 (input) REAL
          Minimum value of d, excluding D( N0 ) and D( N0-1 ).

    DN    (input) REAL
          d(N)

    DN1   (input) REAL
          d(N-1)

    DN2   (input) REAL
          d(N-2)

    TAU   (output) REAL
          This is the shift.

    TTYPE (output) INTEGER
          Shift type.

    Further Details
    ===============
    CNST1 = 9/16

    =====================================================================
*/

    /* Parameter adjustments */
    --z__;

    /* Function Body */

/*
       A negative DMIN forces the shift to take that absolute value
       TTYPE records the type of shift.
*/

    if (*dmin__ <= 0.f) {
	*tau = -(*dmin__);
	*ttype = -1;
	return 0;
    }

    nn = (*n0 << 2) + *pp;
    if (*n0in == *n0) {

/*        No eigenvalues deflated. */

	if (*dmin__ == *dn || *dmin__ == *dn1) {

	    b1 = sqrt(z__[nn - 3]) * sqrt(z__[nn - 5]);
	    b2 = sqrt(z__[nn - 7]) * sqrt(z__[nn - 9]);
	    a2 = z__[nn - 7] + z__[nn - 5];

/*           Cases 2 and 3. */

	    if (*dmin__ == *dn && *dmin1 == *dn1) {
		gap2 = *dmin2 - a2 - *dmin2 * .25f;
		if (gap2 > 0.f && gap2 > b2) {
		    gap1 = a2 - *dn - b2 / gap2 * b2;
		} else {
		    gap1 = a2 - *dn - (b1 + b2);
		}
		if (gap1 > 0.f && gap1 > b1) {
/* Computing MAX */
		    r__1 = *dn - b1 / gap1 * b1, r__2 = *dmin__ * .5f;
		    s = dmax(r__1,r__2);
		    *ttype = -2;
		} else {
		    s = 0.f;
		    if (*dn > b1) {
			s = *dn - b1;
		    }
		    if (a2 > b1 + b2) {
/* Computing MIN */
			r__1 = s, r__2 = a2 - (b1 + b2);
			s = dmin(r__1,r__2);
		    }
/* Computing MAX */
		    r__1 = s, r__2 = *dmin__ * .333f;
		    s = dmax(r__1,r__2);
		    *ttype = -3;
		}
	    } else {

/*              Case 4. */

		*ttype = -4;
		s = *dmin__ * .25f;
		if (*dmin__ == *dn) {
		    gam = *dn;
		    a2 = 0.f;
		    if (z__[nn - 5] > z__[nn - 7]) {
			return 0;
		    }
		    b2 = z__[nn - 5] / z__[nn - 7];
		    np = nn - 9;
		} else {
		    np = nn - (*pp << 1);
		    b2 = z__[np - 2];
		    gam = *dn1;
		    if (z__[np - 4] > z__[np - 2]) {
			return 0;
		    }
		    a2 = z__[np - 4] / z__[np - 2];
		    if (z__[nn - 9] > z__[nn - 11]) {
			return 0;
		    }
		    b2 = z__[nn - 9] / z__[nn - 11];
		    np = nn - 13;
		}

/*              Approximate contribution to norm squared from I < NN-1. */

		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = np; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.f) {
			goto L20;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (dmax(b2,b1) * 100.f < a2 || .563f < a2) {
			goto L20;
		    }
/* L10: */
		}
L20:
		a2 *= 1.05f;

/*              Rayleigh quotient residual bound. */

		if (a2 < .563f) {
		    s = gam * (1.f - sqrt(a2)) / (a2 + 1.f);
		}
	    }
	} else if (*dmin__ == *dn2) {

/*           Case 5. */

	    *ttype = -5;
	    s = *dmin__ * .25f;

/*           Compute contribution to norm squared from I > NN-2. */

	    np = nn - (*pp << 1);
	    b1 = z__[np - 2];
	    b2 = z__[np - 6];
	    gam = *dn2;
	    if (z__[np - 8] > b2 || z__[np - 4] > b1) {
		return 0;
	    }
	    a2 = z__[np - 8] / b2 * (z__[np - 4] / b1 + 1.f);

/*           Approximate contribution to norm squared from I < NN-2. */

	    if (*n0 - *i0 > 2) {
		b2 = z__[nn - 13] / z__[nn - 15];
		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = nn - 17; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.f) {
			goto L40;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (dmax(b2,b1) * 100.f < a2 || .563f < a2) {
			goto L40;
		    }
/* L30: */
		}
L40:
		a2 *= 1.05f;
	    }

	    if (a2 < .563f) {
		s = gam * (1.f - sqrt(a2)) / (a2 + 1.f);
	    }
	} else {

/*           Case 6, no information to guide us. */

	    if (*ttype == -6) {
		g += (1.f - g) * .333f;
	    } else if (*ttype == -18) {
		g = .083250000000000005f;
	    } else {
		g = .25f;
	    }
	    s = g * *dmin__;
	    *ttype = -6;
	}

    } else if (*n0in == *n0 + 1) {

/*        One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN. */

	if (*dmin1 == *dn1 && *dmin2 == *dn2) {

/*           Cases 7 and 8. */

	    *ttype = -7;
	    s = *dmin1 * .333f;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.f) {
		goto L60;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		a2 = b1;
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (dmax(b1,a2) * 100.f < b2) {
		    goto L60;
		}
/* L50: */
	    }
L60:
	    b2 = sqrt(b2 * 1.05f);
/* Computing 2nd power */
	    r__1 = b2;
	    a2 = *dmin1 / (r__1 * r__1 + 1.f);
	    gap2 = *dmin2 * .5f - a2;
	    if (gap2 > 0.f && gap2 > b2 * a2) {
/* Computing MAX */
		r__1 = s, r__2 = a2 * (1.f - a2 * 1.01f * (b2 / gap2) * b2);
		s = dmax(r__1,r__2);
	    } else {
/* Computing MAX */
		r__1 = s, r__2 = a2 * (1.f - b2 * 1.01f);
		s = dmax(r__1,r__2);
		*ttype = -8;
	    }
	} else {

/*           Case 9. */

	    s = *dmin1 * .25f;
	    if (*dmin1 == *dn1) {
		s = *dmin1 * .5f;
	    }
	    *ttype = -9;
	}

    } else if (*n0in == *n0 + 2) {

/*
          Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.

          Cases 10 and 11.
*/

	if (*dmin2 == *dn2 && z__[nn - 5] * 2.f < z__[nn - 7]) {
	    *ttype = -10;
	    s = *dmin2 * .333f;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.f) {
		goto L80;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (b1 * 100.f < b2) {
		    goto L80;
		}
/* L70: */
	    }
L80:
	    b2 = sqrt(b2 * 1.05f);
/* Computing 2nd power */
	    r__1 = b2;
	    a2 = *dmin2 / (r__1 * r__1 + 1.f);
	    gap2 = z__[nn - 7] + z__[nn - 9] - sqrt(z__[nn - 11]) * sqrt(z__[
		    nn - 9]) - a2;
	    if (gap2 > 0.f && gap2 > b2 * a2) {
/* Computing MAX */
		r__1 = s, r__2 = a2 * (1.f - a2 * 1.01f * (b2 / gap2) * b2);
		s = dmax(r__1,r__2);
	    } else {
/* Computing MAX */
		r__1 = s, r__2 = a2 * (1.f - b2 * 1.01f);
		s = dmax(r__1,r__2);
	    }
	} else {
	    s = *dmin2 * .25f;
	    *ttype = -11;
	}
    } else if (*n0in > *n0 + 2) {

/*        Case 12, more than two eigenvalues deflated. No information. */

	s = 0.f;
	*ttype = -12;
    }

    *tau = s;
    return 0;

/*     End of SLASQ4 */

} /* slasq4_ */

/* Subroutine */ int slasq5_(integer *i0, integer *n0, real *z__, integer *pp,
	 real *tau, real *dmin__, real *dmin1, real *dmin2, real *dn, real *
	dnm1, real *dnm2, logical *ieee)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    static real d__;
    static integer j4, j4p2;
    static real emin, temp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       May 17, 2000


    Purpose
    =======

    SLASQ5 computes one dqds transform in ping-pong form, one
    version for IEEE machines another for non IEEE machines.

    Arguments
    =========

    I0    (input) INTEGER
          First index.

    N0    (input) INTEGER
          Last index.

    Z     (input) REAL array, dimension ( 4*N )
          Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
          an extra argument.

    PP    (input) INTEGER
          PP=0 for ping, PP=1 for pong.

    TAU   (input) REAL
          This is the shift.

    DMIN  (output) REAL
          Minimum value of d.

    DMIN1 (output) REAL
          Minimum value of d, excluding D( N0 ).

    DMIN2 (output) REAL
          Minimum value of d, excluding D( N0 ) and D( N0-1 ).

    DN    (output) REAL
          d(N0), the last value of d.

    DNM1  (output) REAL
          d(N0-1).

    DNM2  (output) REAL
          d(N0-2).

    IEEE  (input) LOGICAL
          Flag for IEEE or non IEEE arithmetic.

    =====================================================================
*/


    /* Parameter adjustments */
    --z__;

    /* Function Body */
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }

    j4 = (*i0 << 2) + *pp - 3;
    emin = z__[j4 + 4];
    d__ = z__[j4] - *tau;
    *dmin__ = d__;
    *dmin1 = -z__[j4];

    if (*ieee) {

/*        Code for IEEE arithmetic. */

	if (*pp == 0) {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 2] = d__ + z__[j4 - 1];
		temp = z__[j4 + 1] / z__[j4 - 2];
		d__ = d__ * temp - *tau;
		*dmin__ = dmin(*dmin__,d__);
		z__[j4] = z__[j4 - 1] * temp;
/* Computing MIN */
		r__1 = z__[j4];
		emin = dmin(r__1,emin);
/* L10: */
	    }
	} else {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 3] = d__ + z__[j4];
		temp = z__[j4 + 2] / z__[j4 - 3];
		d__ = d__ * temp - *tau;
		*dmin__ = dmin(*dmin__,d__);
		z__[j4 - 1] = z__[j4] * temp;
/* Computing MIN */
		r__1 = z__[j4 - 1];
		emin = dmin(r__1,emin);
/* L20: */
	    }
	}

/*        Unroll last two steps. */

	*dnm2 = d__;
	*dmin2 = *dmin__;
	j4 = (*n0 - 2 << 2) - *pp;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm2 + z__[j4p2];
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	*dmin__ = dmin(*dmin__,*dnm1);

	*dmin1 = *dmin__;
	j4 += 4;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm1 + z__[j4p2];
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	*dmin__ = dmin(*dmin__,*dn);

    } else {

/*        Code for non IEEE arithmetic. */

	if (*pp == 0) {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 2] = d__ + z__[j4 - 1];
		if (d__ < 0.f) {
		    return 0;
		} else {
		    z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
		    d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		}
		*dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
		r__1 = emin, r__2 = z__[j4];
		emin = dmin(r__1,r__2);
/* L30: */
	    }
	} else {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 3] = d__ + z__[j4];
		if (d__ < 0.f) {
		    return 0;
		} else {
		    z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
		    d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		}
		*dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
		r__1 = emin, r__2 = z__[j4 - 1];
		emin = dmin(r__1,r__2);
/* L40: */
	    }
	}

/*        Unroll last two steps. */

	*dnm2 = d__;
	*dmin2 = *dmin__;
	j4 = (*n0 - 2 << 2) - *pp;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm2 + z__[j4p2];
	if (*dnm2 < 0.f) {
	    return 0;
	} else {
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	}
	*dmin__ = dmin(*dmin__,*dnm1);

	*dmin1 = *dmin__;
	j4 += 4;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm1 + z__[j4p2];
	if (*dnm1 < 0.f) {
	    return 0;
	} else {
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	}
	*dmin__ = dmin(*dmin__,*dn);

    }

    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;

/*     End of SLASQ5 */

} /* slasq5_ */

/* Subroutine */ int slasq6_(integer *i0, integer *n0, real *z__, integer *pp,
	 real *dmin__, real *dmin1, real *dmin2, real *dn, real *dnm1, real *
	dnm2)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    static real d__;
    static integer j4, j4p2;
    static real emin, temp;
    extern doublereal slamch_(char *);
    static real safmin;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    SLASQ6 computes one dqd (shift equal to zero) transform in
    ping-pong form, with protection against underflow and overflow.

    Arguments
    =========

    I0    (input) INTEGER
          First index.

    N0    (input) INTEGER
          Last index.

    Z     (input) REAL array, dimension ( 4*N )
          Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
          an extra argument.

    PP    (input) INTEGER
          PP=0 for ping, PP=1 for pong.

    DMIN  (output) REAL
          Minimum value of d.

    DMIN1 (output) REAL
          Minimum value of d, excluding D( N0 ).

    DMIN2 (output) REAL
          Minimum value of d, excluding D( N0 ) and D( N0-1 ).

    DN    (output) REAL
          d(N0), the last value of d.

    DNM1  (output) REAL
          d(N0-1).

    DNM2  (output) REAL
          d(N0-2).

    =====================================================================
*/


    /* Parameter adjustments */
    --z__;

    /* Function Body */
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }

    safmin = slamch_("Safe minimum");
    j4 = (*i0 << 2) + *pp - 3;
    emin = z__[j4 + 4];
    d__ = z__[j4];
    *dmin__ = d__;

    if (*pp == 0) {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 2] = d__ + z__[j4 - 1];
	    if (z__[j4 - 2] == 0.f) {
		z__[j4] = 0.f;
		d__ = z__[j4 + 1];
		*dmin__ = d__;
		emin = 0.f;
	    } else if (safmin * z__[j4 + 1] < z__[j4 - 2] && safmin * z__[j4
		    - 2] < z__[j4 + 1]) {
		temp = z__[j4 + 1] / z__[j4 - 2];
		z__[j4] = z__[j4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
		d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]);
	    }
	    *dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
	    r__1 = emin, r__2 = z__[j4];
	    emin = dmin(r__1,r__2);
/* L10: */
	}
    } else {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 3] = d__ + z__[j4];
	    if (z__[j4 - 3] == 0.f) {
		z__[j4 - 1] = 0.f;
		d__ = z__[j4 + 2];
		*dmin__ = d__;
		emin = 0.f;
	    } else if (safmin * z__[j4 + 2] < z__[j4 - 3] && safmin * z__[j4
		    - 3] < z__[j4 + 2]) {
		temp = z__[j4 + 2] / z__[j4 - 3];
		z__[j4 - 1] = z__[j4] * temp;
		d__ *= temp;
	    } else {
		z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
		d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]);
	    }
	    *dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
	    r__1 = emin, r__2 = z__[j4 - 1];
	    emin = dmin(r__1,r__2);
/* L20: */
	}
    }

/*     Unroll last two steps. */

    *dnm2 = d__;
    *dmin2 = *dmin__;
    j4 = (*n0 - 2 << 2) - *pp;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm2 + z__[j4p2];
    if (z__[j4 - 2] == 0.f) {
	z__[j4] = 0.f;
	*dnm1 = z__[j4p2 + 2];
	*dmin__ = *dnm1;
	emin = 0.f;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] <
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dnm1 = *dnm2 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]);
    }
    *dmin__ = dmin(*dmin__,*dnm1);

    *dmin1 = *dmin__;
    j4 += 4;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm1 + z__[j4p2];
    if (z__[j4 - 2] == 0.f) {
	z__[j4] = 0.f;
	*dn = z__[j4p2 + 2];
	*dmin__ = *dn;
	emin = 0.f;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] <
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dn = *dnm1 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]);
    }
    *dmin__ = dmin(*dmin__,*dn);

    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;

/*     End of SLASQ6 */

} /* slasq6_ */

/* Subroutine */ int slasr_(char *side, char *pivot, char *direct, integer *m,
	 integer *n, real *c__, real *s, real *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, info;
    static real temp;
    extern logical lsame_(char *, char *);
    static real ctemp, stemp;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLASR   performs the transformation

       A := P*A,   when SIDE = 'L' or 'l'  (  Left-hand side )

       A := A*P',  when SIDE = 'R' or 'r'  ( Right-hand side )

    where A is an m by n real matrix and P is an orthogonal matrix,
    consisting of a sequence of plane rotations determined by the
    parameters PIVOT and DIRECT as follows ( z = m when SIDE = 'L' or 'l'
    and z = n when SIDE = 'R' or 'r' ):

    When  DIRECT = 'F' or 'f'  ( Forward sequence ) then

       P = P( z - 1 )*...*P( 2 )*P( 1 ),

    and when DIRECT = 'B' or 'b'  ( Backward sequence ) then

       P = P( 1 )*P( 2 )*...*P( z - 1 ),

    where  P( k ) is a plane rotation matrix for the following planes:

       when  PIVOT = 'V' or 'v'  ( Variable pivot ),
          the plane ( k, k + 1 )

       when  PIVOT = 'T' or 't'  ( Top pivot ),
          the plane ( 1, k + 1 )

       when  PIVOT = 'B' or 'b'  ( Bottom pivot ),
          the plane ( k, z )

    c( k ) and s( k )  must contain the  cosine and sine that define the
    matrix  P( k ).  The two by two plane rotation part of the matrix
    P( k ), R( k ), is assumed to be of the form

       R( k ) = (  c( k )  s( k ) ).
                ( -s( k )  c( k ) )

    This version vectorises across rows of the array A when SIDE = 'L'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            Specifies whether the plane rotation matrix P is applied to
            A on the left or the right.
            = 'L':  Left, compute A := P*A
            = 'R':  Right, compute A:= A*P'

    DIRECT  (input) CHARACTER*1
            Specifies whether P is a forward or backward sequence of
            plane rotations.
            = 'F':  Forward, P = P( z - 1 )*...*P( 2 )*P( 1 )
            = 'B':  Backward, P = P( 1 )*P( 2 )*...*P( z - 1 )

    PIVOT   (input) CHARACTER*1
            Specifies the plane for which P(k) is a plane rotation
            matrix.
            = 'V':  Variable pivot, the plane (k,k+1)
            = 'T':  Top pivot, the plane (1,k+1)
            = 'B':  Bottom pivot, the plane (k,z)

    M       (input) INTEGER
            The number of rows of the matrix A.  If m <= 1, an immediate
            return is effected.

    N       (input) INTEGER
            The number of columns of the matrix A.  If n <= 1, an
            immediate return is effected.

    C, S    (input) REAL arrays, dimension
                    (M-1) if SIDE = 'L'
                    (N-1) if SIDE = 'R'
            c(k) and s(k) contain the cosine and sine that define the
            matrix P(k).  The two by two plane rotation part of the
            matrix P(k), R(k), is assumed to be of the form
            R( k ) = (  c( k )  s( k ) ).
                     ( -s( k )  c( k ) )

    A       (input/output) REAL array, dimension (LDA,N)
            The m by n matrix A.  On exit, A is overwritten by P*A if
            SIDE = 'R' or by A*P' if SIDE = 'L'.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    --c__;
    --s;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! (lsame_(side, "L") || lsame_(side, "R"))) {
	info = 1;
    } else if (! (lsame_(pivot, "V") || lsame_(pivot,
	    "T") || lsame_(pivot, "B"))) {
	info = 2;
    } else if (! (lsame_(direct, "F") || lsame_(direct,
	    "B"))) {
	info = 3;
    } else if (*m < 0) {
	info = 4;
    } else if (*n < 0) {
	info = 5;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("SLASR ", &info);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }
    if (lsame_(side, "L")) {

/*        Form  P * A */

	if (lsame_(pivot, "V")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp *
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j
				    + i__ * a_dim1];
/* L10: */
			}
		    }
/* L20: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp *
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j
				    + i__ * a_dim1];
/* L30: */
			}
		    }
/* L40: */
		}
	    }
	} else if (lsame_(pivot, "T")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
/* L50: */
			}
		    }
/* L60: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
/* L70: */
			}
		    }
/* L80: */
		}
	    }
	} else if (lsame_(pivot, "B")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ *
				    a_dim1] - stemp * temp;
/* L90: */
			}
		    }
/* L100: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ *
				    a_dim1] - stemp * temp;
/* L110: */
			}
		    }
/* L120: */
		}
	    }
	}
    } else if (lsame_(side, "R")) {

/*        Form A * P' */

	if (lsame_(pivot, "V")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
/* L130: */
			}
		    }
/* L140: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
/* L150: */
			}
		    }
/* L160: */
		}
	    }
	} else if (lsame_(pivot, "T")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ +
				    a_dim1];
/* L170: */
			}
		    }
/* L180: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ +
				    a_dim1];
/* L190: */
			}
		    }
/* L200: */
		}
	    }
	} else if (lsame_(pivot, "B")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n *
				    a_dim1] - stemp * temp;
/* L210: */
			}
		    }
/* L220: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n *
				    a_dim1] - stemp * temp;
/* L230: */
			}
		    }
/* L240: */
		}
	    }
	}
    }

    return 0;

/*     End of SLASR */

} /* slasr_ */

/* Subroutine */ int slasrt_(char *id, integer *n, real *d__, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    static integer i__, j;
    static real d1, d2, d3;
    static integer dir;
    static real tmp;
    static integer endd;
    extern logical lsame_(char *, char *);
    static integer stack[64]	/* was [2][32] */;
    static real dmnmx;
    static integer start;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static integer stkpnt;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    Sort the numbers in D in increasing order (if ID = 'I') or
    in decreasing order (if ID = 'D' ).

    Use Quick Sort, reverting to Insertion sort on arrays of
    size <= 20. Dimension of STACK limits N to about 2**32.

    Arguments
    =========

    ID      (input) CHARACTER*1
            = 'I': sort D in increasing order;
            = 'D': sort D in decreasing order.

    N       (input) INTEGER
            The length of the array D.

    D       (input/output) REAL array, dimension (N)
            On entry, the array to be sorted.
            On exit, D has been sorted into increasing order
            (D(1) <= ... <= D(N) ) or into decreasing order
            (D(1) >= ... >= D(N) ), depending on ID.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input paramters.
*/

    /* Parameter adjustments */
    --d__;

    /* Function Body */
    *info = 0;
    dir = -1;
    if (lsame_(id, "D")) {
	dir = 0;
    } else if (lsame_(id, "I")) {
	dir = 1;
    }
    if (dir == -1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASRT", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 1) {
	return 0;
    }

    stkpnt = 1;
    stack[0] = 1;
    stack[1] = *n;
L10:
    start = stack[(stkpnt << 1) - 2];
    endd = stack[(stkpnt << 1) - 1];
    --stkpnt;
    if (endd - start <= 20 && endd - start > 0) {

/*        Do Insertion sort on D( START:ENDD ) */

	if (dir == 0) {

/*           Sort into decreasing order */

	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] > d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L30;
		    }
/* L20: */
		}
L30:
		;
	    }

	} else {

/*           Sort into increasing order */

	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] < d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L50;
		    }
/* L40: */
		}
L50:
		;
	    }

	}

    } else if (endd - start > 20) {

/*
          Partition D( START:ENDD ) and stack parts, largest one first

          Choose partition entry as median of 3
*/

	d1 = d__[start];
	d2 = d__[endd];
	i__ = (start + endd) / 2;
	d3 = d__[i__];
	if (d1 < d2) {
	    if (d3 < d1) {
		dmnmx = d1;
	    } else if (d3 < d2) {
		dmnmx = d3;
	    } else {
		dmnmx = d2;
	    }
	} else {
	    if (d3 < d2) {
		dmnmx = d2;
	    } else if (d3 < d1) {
		dmnmx = d3;
	    } else {
		dmnmx = d1;
	    }
	}

	if (dir == 0) {

/*           Sort into decreasing order */

	    i__ = start - 1;
	    j = endd + 1;
L60:
L70:
	    --j;
	    if (d__[j] < dmnmx) {
		goto L70;
	    }
L80:
	    ++i__;
	    if (d__[i__] > dmnmx) {
		goto L80;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L60;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	} else {

/*           Sort into increasing order */

	    i__ = start - 1;
	    j = endd + 1;
L90:
L100:
	    --j;
	    if (d__[j] > dmnmx) {
		goto L100;
	    }
L110:
	    ++i__;
	    if (d__[i__] < dmnmx) {
		goto L110;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L90;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	}
    }
    if (stkpnt > 0) {
	goto L10;
    }
    return 0;

/*     End of SLASRT */

} /* slasrt_ */

/* Subroutine */ int slassq_(integer *n, real *x, integer *incx, real *scale,
	real *sumsq)
{
    /* System generated locals */
    integer i__1, i__2;
    real r__1;

    /* Local variables */
    static integer ix;
    static real absxi;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLASSQ  returns the values  scl  and  smsq  such that

       ( scl**2 )*smsq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,

    where  x( i ) = X( 1 + ( i - 1 )*INCX ). The value of  sumsq  is
    assumed to be non-negative and  scl  returns the value

       scl = max( scale, abs( x( i ) ) ).

    scale and sumsq must be supplied in SCALE and SUMSQ and
    scl and smsq are overwritten on SCALE and SUMSQ respectively.

    The routine makes only one pass through the vector x.

    Arguments
    =========

    N       (input) INTEGER
            The number of elements to be used from the vector X.

    X       (input) REAL array, dimension (N)
            The vector for which a scaled sum of squares is computed.
               x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n.

    INCX    (input) INTEGER
            The increment between successive values of the vector X.
            INCX > 0.

    SCALE   (input/output) REAL
            On entry, the value  scale  in the equation above.
            On exit, SCALE is overwritten with  scl , the scaling factor
            for the sum of squares.

    SUMSQ   (input/output) REAL
            On entry, the value  sumsq  in the equation above.
            On exit, SUMSQ is overwritten with  smsq , the basic sum of
            squares from which  scl  has been factored out.

   =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n > 0) {
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    if (x[ix] != 0.f) {
		absxi = (r__1 = x[ix], dabs(r__1));
		if (*scale < absxi) {
/* Computing 2nd power */
		    r__1 = *scale / absxi;
		    *sumsq = *sumsq * (r__1 * r__1) + 1;
		    *scale = absxi;
		} else {
/* Computing 2nd power */
		    r__1 = absxi / *scale;
		    *sumsq += r__1 * r__1;
		}
	    }
/* L10: */
	}
    }
    return 0;

/*     End of SLASSQ */

} /* slassq_ */

/* Subroutine */ int slasv2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax, real *snr, real *csr, real *snl, real *csl)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static real a, d__, l, m, r__, s, t, fa, ga, ha, ft, gt, ht, mm, tt, clt,
	    crt, slt, srt;
    static integer pmax;
    static real temp;
    static logical swap;
    static real tsign;
    static logical gasmal;
    extern doublereal slamch_(char *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLASV2 computes the singular value decomposition of a 2-by-2
    triangular matrix
       [  F   G  ]
       [  0   H  ].
    On return, abs(SSMAX) is the larger singular value, abs(SSMIN) is the
    smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
    right singular vectors for abs(SSMAX), giving the decomposition

       [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
       [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].

    Arguments
    =========

    F       (input) REAL
            The (1,1) element of the 2-by-2 matrix.

    G       (input) REAL
            The (1,2) element of the 2-by-2 matrix.

    H       (input) REAL
            The (2,2) element of the 2-by-2 matrix.

    SSMIN   (output) REAL
            abs(SSMIN) is the smaller singular value.

    SSMAX   (output) REAL
            abs(SSMAX) is the larger singular value.

    SNL     (output) REAL
    CSL     (output) REAL
            The vector (CSL, SNL) is a unit left singular vector for the
            singular value abs(SSMAX).

    SNR     (output) REAL
    CSR     (output) REAL
            The vector (CSR, SNR) is a unit right singular vector for the
            singular value abs(SSMAX).

    Further Details
    ===============

    Any input parameter may be aliased with any output parameter.

    Barring over/underflow and assuming a guard digit in subtraction, all
    output quantities are correct to within a few units in the last
    place (ulps).

    In IEEE arithmetic, the code works correctly if one matrix element is
    infinite.

    Overflow will not occur unless the largest singular value itself
    overflows or is within a few ulps of overflow. (On machines with
    partial overflow, like the Cray, overflow may occur if the largest
    singular value is within a factor of 2 of overflow.)

    Underflow is harmless if underflow is gradual. Otherwise, results
    may correspond to a matrix modified by perturbations of size near
    the underflow threshold.

   =====================================================================
*/


    ft = *f;
    fa = dabs(ft);
    ht = *h__;
    ha = dabs(*h__);

/*
       PMAX points to the maximum absolute element of matrix
         PMAX = 1 if F largest in absolute values
         PMAX = 2 if G largest in absolute values
         PMAX = 3 if H largest in absolute values
*/

    pmax = 1;
    swap = ha > fa;
    if (swap) {
	pmax = 3;
	temp = ft;
	ft = ht;
	ht = temp;
	temp = fa;
	fa = ha;
	ha = temp;

/*        Now FA .ge. HA */

    }
    gt = *g;
    ga = dabs(gt);
    if (ga == 0.f) {

/*        Diagonal matrix */

	*ssmin = ha;
	*ssmax = fa;
	clt = 1.f;
	crt = 1.f;
	slt = 0.f;
	srt = 0.f;
    } else {
	gasmal = TRUE_;
	if (ga > fa) {
	    pmax = 2;
	    if (fa / ga < slamch_("EPS")) {

/*              Case of very large GA */

		gasmal = FALSE_;
		*ssmax = ga;
		if (ha > 1.f) {
		    *ssmin = fa / (ga / ha);
		} else {
		    *ssmin = fa / ga * ha;
		}
		clt = 1.f;
		slt = ht / gt;
		srt = 1.f;
		crt = ft / gt;
	    }
	}
	if (gasmal) {

/*           Normal case */

	    d__ = fa - ha;
	    if (d__ == fa) {

/*              Copes with infinite F or H */

		l = 1.f;
	    } else {
		l = d__ / fa;
	    }

/*           Note that 0 .le. L .le. 1 */

	    m = gt / ft;

/*           Note that abs(M) .le. 1/macheps */

	    t = 2.f - l;

/*           Note that T .ge. 1 */

	    mm = m * m;
	    tt = t * t;
	    s = sqrt(tt + mm);

/*           Note that 1 .le. S .le. 1 + 1/macheps */

	    if (l == 0.f) {
		r__ = dabs(m);
	    } else {
		r__ = sqrt(l * l + mm);
	    }

/*           Note that 0 .le. R .le. 1 + 1/macheps */

	    a = (s + r__) * .5f;

/*           Note that 1 .le. A .le. 1 + abs(M) */

	    *ssmin = ha / a;
	    *ssmax = fa * a;
	    if (mm == 0.f) {

/*              Note that M is very tiny */

		if (l == 0.f) {
		    t = r_sign(&c_b2489, &ft) * r_sign(&c_b15, &gt);
		} else {
		    t = gt / r_sign(&d__, &ft) + m / t;
		}
	    } else {
		t = (m / (s + t) + m / (r__ + l)) * (a + 1.f);
	    }
	    l = sqrt(t * t + 4.f);
	    crt = 2.f / l;
	    srt = t / l;
	    clt = (crt + srt * m) / a;
	    slt = ht / ft * srt / a;
	}
    }
    if (swap) {
	*csl = srt;
	*snl = crt;
	*csr = slt;
	*snr = clt;
    } else {
	*csl = clt;
	*snl = slt;
	*csr = crt;
	*snr = srt;
    }

/*     Correct signs of SSMAX and SSMIN */

    if (pmax == 1) {
	tsign = r_sign(&c_b15, csr) * r_sign(&c_b15, csl) * r_sign(&c_b15, f);
    }
    if (pmax == 2) {
	tsign = r_sign(&c_b15, snr) * r_sign(&c_b15, csl) * r_sign(&c_b15, g);
    }
    if (pmax == 3) {
	tsign = r_sign(&c_b15, snr) * r_sign(&c_b15, snl) * r_sign(&c_b15,
		h__);
    }
    *ssmax = r_sign(ssmax, &tsign);
    r__1 = tsign * r_sign(&c_b15, f) * r_sign(&c_b15, h__);
    *ssmin = r_sign(ssmin, &r__1);
    return 0;

/*     End of SLASV2 */

} /* slasv2_ */

/* Subroutine */ int slaswp_(integer *n, real *a, integer *lda, integer *k1,
	integer *k2, integer *ipiv, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, k, i1, i2, n32, ip, ix, ix0, inc;
    static real temp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.

    Arguments
    =========

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the matrix of column dimension N to which the row
            interchanges will be applied.
            On exit, the permuted matrix.

    LDA     (input) INTEGER
            The leading dimension of the array A.

    K1      (input) INTEGER
            The first element of IPIV for which a row interchange will
            be done.

    K2      (input) INTEGER
            The last element of IPIV for which a row interchange will
            be done.

    IPIV    (input) INTEGER array, dimension (M*abs(INCX))
            The vector of pivot indices.  Only the elements in positions
            K1 through K2 of IPIV are accessed.
            IPIV(K) = L implies rows K and L are to be interchanged.

    INCX    (input) INTEGER
            The increment between successive values of IPIV.  If IPIV
            is negative, the pivots are applied in reverse order.

    Further Details
    ===============

    Modified by
     R. C. Whaley, Computer Science Dept., Univ. of Tenn., Knoxville, USA

   =====================================================================


       Interchange row I with row IPIV(I) for each of rows K1 through K2.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

    /* Function Body */
    if (*incx > 0) {
	ix0 = *k1;
	i1 = *k1;
	i2 = *k2;
	inc = 1;
    } else if (*incx < 0) {
	ix0 = (1 - *k2) * *incx + 1;
	i1 = *k2;
	i2 = *k1;
	inc = -1;
    } else {
	return 0;
    }

    n32 = *n / 32 << 5;
    if (n32 != 0) {
	i__1 = n32;
	for (j = 1; j <= i__1; j += 32) {
	    ix = ix0;
	    i__2 = i2;
	    i__3 = inc;
	    for (i__ = i1; i__3 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__3)
		    {
		ip = ipiv[ix];
		if (ip != i__) {
		    i__4 = j + 31;
		    for (k = j; k <= i__4; ++k) {
			temp = a[i__ + k * a_dim1];
			a[i__ + k * a_dim1] = a[ip + k * a_dim1];
			a[ip + k * a_dim1] = temp;
/* L10: */
		    }
		}
		ix += *incx;
/* L20: */
	    }
/* L30: */
	}
    }
    if (n32 != *n) {
	++n32;
	ix = ix0;
	i__1 = i2;
	i__3 = inc;
	for (i__ = i1; i__3 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__3) {
	    ip = ipiv[ix];
	    if (ip != i__) {
		i__2 = *n;
		for (k = n32; k <= i__2; ++k) {
		    temp = a[i__ + k * a_dim1];
		    a[i__ + k * a_dim1] = a[ip + k * a_dim1];
		    a[ip + k * a_dim1] = temp;
/* L40: */
		}
	    }
	    ix += *incx;
/* L50: */
	}
    }

    return 0;

/*     End of SLASWP */

} /* slaswp_ */

/* Subroutine */ int slatrd_(char *uplo, integer *n, integer *nb, real *a,
	integer *lda, real *e, real *tau, real *w, integer *ldw)
{
    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, iw;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    static real alpha;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sgemv_(char *, integer *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *), saxpy_(
	    integer *, real *, real *, integer *, real *, integer *), ssymv_(
	    char *, integer *, real *, real *, integer *, real *, integer *,
	    real *, real *, integer *), slarfg_(integer *, real *,
	    real *, integer *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SLATRD reduces NB rows and columns of a real symmetric matrix A to
    symmetric tridiagonal form by an orthogonal similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = 'U', SLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = 'L', SLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by SSYTRD.

    Arguments
    =========

    UPLO    (input) CHARACTER
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored:
            = 'U': Upper triangular
            = 'L': Lower triangular

    N       (input) INTEGER
            The order of the matrix A.

    NB      (input) INTEGER
            The number of rows and columns to be reduced.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit:
            if UPLO = 'U', the last NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements above the diagonal
              with the array TAU, represent the orthogonal matrix Q as a
              product of elementary reflectors;
            if UPLO = 'L', the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  orthogonal matrix Q as a
              product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= (1,N).

    E       (output) REAL array, dimension (N-1)
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    TAU     (output) REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
            See Further Details.

    W       (output) REAL array, dimension (LDW,NB)
            The n-by-nb matrix W required to update the unreduced part
            of A.

    LDW     (input) INTEGER
            The leading dimension of the array W. LDW >= max(1,N).

    Further Details
    ===============

    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n) H(n-1) . . . H(n-nb+1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
    and tau in TAU(i).

    The elements of the vectors v together form the n-by-nb matrix V
    which is needed, with W, to apply the transformation to the unreduced
    part of the matrix, using a symmetric rank-2k update of the form:
    A := A - V*W' - W*V'.

    The contents of A on exit are illustrated by the following examples
    with n = 5 and nb = 2:

    if UPLO = 'U':                       if UPLO = 'L':

      (  a   a   a   v4  v5 )              (  d                  )
      (      a   a   v4  v5 )              (  1   d              )
      (          a   1   v5 )              (  v1  1   a          )
      (              d   1  )              (  v1  v2  a   a      )
      (                  d  )              (  v1  v2  a   a   a  )

    where d denotes a diagonal element of the reduced matrix, a denotes
    an element of the original matrix that is unchanged, and vi denotes
    an element of the vector defining H(i).

    =====================================================================


       Quick return if possible
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --e;
    --tau;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }

    if (lsame_(uplo, "U")) {

/*        Reduce last NB columns of upper triangle */

	i__1 = *n - *nb + 1;
	for (i__ = *n; i__ >= i__1; --i__) {
	    iw = i__ - *n + *nb;
	    if (i__ < *n) {

/*              Update A(1:i,i) */

		i__2 = *n - i__;
		sgemv_("No transpose", &i__, &i__2, &c_b151, &a[(i__ + 1) *
			a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, &
			c_b15, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		sgemv_("No transpose", &i__, &i__2, &c_b151, &w[(iw + 1) *
			w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b15, &a[i__ * a_dim1 + 1], &c__1);
	    }
	    if (i__ > 1) {

/*
                Generate elementary reflector H(i) to annihilate
                A(1:i-2,i)
*/

		i__2 = i__ - 1;
		slarfg_(&i__2, &a[i__ - 1 + i__ * a_dim1], &a[i__ * a_dim1 +
			1], &c__1, &tau[i__ - 1]);
		e[i__ - 1] = a[i__ - 1 + i__ * a_dim1];
		a[i__ - 1 + i__ * a_dim1] = 1.f;

/*              Compute W(1:i-1,i) */

		i__2 = i__ - 1;
		ssymv_("Upper", &i__2, &c_b15, &a[a_offset], lda, &a[i__ *
			a_dim1 + 1], &c__1, &c_b29, &w[iw * w_dim1 + 1], &
			c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    sgemv_("Transpose", &i__2, &i__3, &c_b15, &w[(iw + 1) *
			    w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1], &c__1, &
			    c_b29, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[(i__ + 1)
			     * a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b15, &w[iw * w_dim1 + 1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[(i__ + 1) *
			    a_dim1 + 1], lda, &a[i__ * a_dim1 + 1], &c__1, &
			    c_b29, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    sgemv_("No transpose", &i__2, &i__3, &c_b151, &w[(iw + 1)
			    * w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b15, &w[iw * w_dim1 + 1], &c__1);
		}
		i__2 = i__ - 1;
		sscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		alpha = tau[i__ - 1] * -.5f * sdot_(&i__2, &w[iw * w_dim1 + 1]
			, &c__1, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		saxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw *
			w_dim1 + 1], &c__1);
	    }

/* L10: */
	}
    } else {

/*        Reduce first NB columns of lower triangle */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:n,i) */

	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[i__ + a_dim1],
		    lda, &w[i__ + w_dim1], ldw, &c_b15, &a[i__ + i__ * a_dim1]
		    , &c__1);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b151, &w[i__ + w_dim1],
		    ldw, &a[i__ + a_dim1], lda, &c_b15, &a[i__ + i__ * a_dim1]
		    , &c__1);
	    if (i__ < *n) {

/*
                Generate elementary reflector H(i) to annihilate
                A(i+2:n,i)
*/

		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) +
			i__ * a_dim1], &c__1, &tau[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.f;

/*              Compute W(i+1:n,i) */

		i__2 = *n - i__;
		ssymv_("Lower", &i__2, &c_b15, &a[i__ + 1 + (i__ + 1) *
			a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b29, &w[i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &w[i__ + 1 + w_dim1]
			, ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b29, &w[
			i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[i__ + 1 +
			a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b15, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + a_dim1]
			, lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b29, &w[
			i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &w[i__ + 1 +
			w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b15, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		sscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		alpha = tau[i__] * -.5f * sdot_(&i__2, &w[i__ + 1 + i__ *
			w_dim1], &c__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
		i__2 = *n - i__;
		saxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
	    }

/* L20: */
	}
    }

    return 0;

/*     End of SLATRD */

} /* slatrd_ */

/* Subroutine */ int slauu2_(char *uplo, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__;
    static real aii;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sgemv_(char *, integer *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLAUU2 computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array A.

    If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
    overwriting the factor U in A.
    If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
    overwriting the factor L in A.

    This is the unblocked form of the algorithm, calling Level 2 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the triangular factor stored in the array A
            is upper or lower triangular:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the triangular factor U or L.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = 'U', the upper triangle of A is
            overwritten with the upper triangle of the product U * U';
            if UPLO = 'L', the lower triangle of A is overwritten with
            the lower triangle of the product L' * L.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAUU2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (upper) {

/*        Compute the product U * U'. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    aii = a[i__ + i__ * a_dim1];
	    if (i__ < *n) {
		i__2 = *n - i__ + 1;
		a[i__ + i__ * a_dim1] = sdot_(&i__2, &a[i__ + i__ * a_dim1],
			lda, &a[i__ + i__ * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		sgemv_("No transpose", &i__2, &i__3, &c_b15, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			aii, &a[i__ * a_dim1 + 1], &c__1);
	    } else {
		sscal_(&i__, &aii, &a[i__ * a_dim1 + 1], &c__1);
	    }
/* L10: */
	}

    } else {

/*        Compute the product L' * L. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    aii = a[i__ + i__ * a_dim1];
	    if (i__ < *n) {
		i__2 = *n - i__ + 1;
		a[i__ + i__ * a_dim1] = sdot_(&i__2, &a[i__ + i__ * a_dim1], &
			c__1, &a[i__ + i__ * a_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b15, &a[i__ + 1 + a_dim1]
			, lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &aii, &a[
			i__ + a_dim1], lda);
	    } else {
		sscal_(&i__, &aii, &a[i__ + a_dim1], lda);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of SLAUU2 */

} /* slauu2_ */

/* Subroutine */ int slauum_(char *uplo, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, ib, nb;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static logical upper;
    extern /* Subroutine */ int strmm_(char *, char *, char *, char *,
	    integer *, integer *, real *, real *, integer *, real *, integer *
	    ), ssyrk_(char *, char *, integer
	    *, integer *, real *, real *, integer *, real *, real *, integer *
	    ), slauu2_(char *, integer *, real *, integer *,
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SLAUUM computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array A.

    If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
    overwriting the factor U in A.
    If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
    overwriting the factor L in A.

    This is the blocked form of the algorithm, calling Level 3 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the triangular factor stored in the array A
            is upper or lower triangular:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the triangular factor U or L.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = 'U', the upper triangle of A is
            overwritten with the upper triangle of the product U * U';
            if UPLO = 'L', the lower triangle of A is overwritten with
            the lower triangle of the product L' * L.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAUUM", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "SLAUUM", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	slauu2_(uplo, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code */

	if (upper) {

/*           Compute the product U * U'. */

	    i__1 = *n;
	    i__2 = nb;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);
		i__3 = i__ - 1;
		strmm_("Right", "Upper", "Transpose", "Non-unit", &i__3, &ib,
			&c_b15, &a[i__ + i__ * a_dim1], lda, &a[i__ * a_dim1
			+ 1], lda)
			;
		slauu2_("Upper", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    sgemm_("No transpose", "Transpose", &i__3, &ib, &i__4, &
			    c_b15, &a[(i__ + ib) * a_dim1 + 1], lda, &a[i__ +
			    (i__ + ib) * a_dim1], lda, &c_b15, &a[i__ *
			    a_dim1 + 1], lda);
		    i__3 = *n - i__ - ib + 1;
		    ssyrk_("Upper", "No transpose", &ib, &i__3, &c_b15, &a[
			    i__ + (i__ + ib) * a_dim1], lda, &c_b15, &a[i__ +
			    i__ * a_dim1], lda);
		}
/* L10: */
	    }
	} else {

/*           Compute the product L' * L. */

	    i__2 = *n;
	    i__1 = nb;
	    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);
		i__3 = i__ - 1;
		strmm_("Left", "Lower", "Transpose", "Non-unit", &ib, &i__3, &
			c_b15, &a[i__ + i__ * a_dim1], lda, &a[i__ + a_dim1],
			lda);
		slauu2_("Lower", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    sgemm_("Transpose", "No transpose", &ib, &i__3, &i__4, &
			    c_b15, &a[i__ + ib + i__ * a_dim1], lda, &a[i__ +
			    ib + a_dim1], lda, &c_b15, &a[i__ + a_dim1], lda);
		    i__3 = *n - i__ - ib + 1;
		    ssyrk_("Lower", "Transpose", &ib, &i__3, &c_b15, &a[i__ +
			    ib + i__ * a_dim1], lda, &c_b15, &a[i__ + i__ *
			    a_dim1], lda);
		}
/* L20: */
	    }
	}
    }

    return 0;

/*     End of SLAUUM */

} /* slauum_ */

/* Subroutine */ int sorg2r_(integer *m, integer *n, integer *k, real *a,
	integer *lda, real *tau, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1;

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    slarf_(char *, integer *, integer *, real *, integer *, real *,
	    real *, integer *, real *), xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SORG2R generates an m by n real matrix Q with orthonormal columns,
    which is defined as the first n columns of a product of k elementary
    reflectors of order m

          Q  =  H(1) H(2) . . . H(k)

    as returned by SGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by SGEQRF in the first k columns of its array
            argument A.
            On exit, the m-by-n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF.

    WORK    (workspace) REAL array, dimension (N)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument has an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORG2R", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

/*     Initialise columns k+1:n to columns of the unit matrix */

    i__1 = *n;
    for (j = *k + 1; j <= i__1; ++j) {
	i__2 = *m;
	for (l = 1; l <= i__2; ++l) {
	    a[l + j * a_dim1] = 0.f;
/* L10: */
	}
	a[j + j * a_dim1] = 1.f;
/* L20: */
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i) to A(i:m,i:n) from the left */

	if (i__ < *n) {
	    a[i__ + i__ * a_dim1] = 1.f;
	    i__1 = *m - i__ + 1;
	    i__2 = *n - i__;
	    slarf_("Left", &i__1, &i__2, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	}
	if (i__ < *m) {
	    i__1 = *m - i__;
	    r__1 = -tau[i__];
	    sscal_(&i__1, &r__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
	}
	a[i__ + i__ * a_dim1] = 1.f - tau[i__];

/*        Set A(1:i-1,i) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    a[l + i__ * a_dim1] = 0.f;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of SORG2R */

} /* sorg2r_ */

/* Subroutine */ int sorgbr_(char *vect, integer *m, integer *n, integer *k,
	real *a, integer *lda, real *tau, real *work, integer *lwork, integer
	*info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, nb, mn;
    extern logical lsame_(char *, char *);
    static integer iinfo;
    static logical wantq;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int sorglq_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *), sorgqr_(
	    integer *, integer *, integer *, real *, integer *, real *, real *
	    , integer *, integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORGBR generates one of the real orthogonal matrices Q or P**T
    determined by SGEBRD when reducing a real matrix A to bidiagonal
    form: A = Q * B * P**T.  Q and P**T are defined as products of
    elementary reflectors H(i) or G(i) respectively.

    If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
    is of order M:
    if m >= k, Q = H(1) H(2) . . . H(k) and SORGBR returns the first n
    columns of Q, where m >= n >= k;
    if m < k, Q = H(1) H(2) . . . H(m-1) and SORGBR returns Q as an
    M-by-M matrix.

    If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T
    is of order N:
    if k < n, P**T = G(k) . . . G(2) G(1) and SORGBR returns the first m
    rows of P**T, where n >= m >= k;
    if k >= n, P**T = G(n-1) . . . G(2) G(1) and SORGBR returns P**T as
    an N-by-N matrix.

    Arguments
    =========

    VECT    (input) CHARACTER*1
            Specifies whether the matrix Q or the matrix P**T is
            required, as defined in the transformation applied by SGEBRD:
            = 'Q':  generate Q;
            = 'P':  generate P**T.

    M       (input) INTEGER
            The number of rows of the matrix Q or P**T to be returned.
            M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q or P**T to be returned.
            N >= 0.
            If VECT = 'Q', M >= N >= min(M,K);
            if VECT = 'P', N >= M >= min(N,K).

    K       (input) INTEGER
            If VECT = 'Q', the number of columns in the original M-by-K
            matrix reduced by SGEBRD.
            If VECT = 'P', the number of rows in the original K-by-N
            matrix reduced by SGEBRD.
            K >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by SGEBRD.
            On exit, the M-by-N matrix Q or P**T.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    TAU     (input) REAL array, dimension
                                  (min(M,K)) if VECT = 'Q'
                                  (min(N,K)) if VECT = 'P'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i), which determines Q or P**T, as
            returned by SGEBRD in its array argument TAUQ or TAUP.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= max(1,min(M,N)).
            For optimum performance LWORK >= min(M,N)*NB, where NB
            is the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    wantq = lsame_(vect, "Q");
    mn = min(*m,*n);
    lquery = *lwork == -1;
    if (! wantq && ! lsame_(vect, "P")) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0 || wantq && (*n > *m || *n < min(*m,*k)) || ! wantq && (
	    *m > *n || *m < min(*n,*k))) {
	*info = -3;
    } else if (*k < 0) {
	*info = -4;
    } else if (*lda < max(1,*m)) {
	*info = -6;
    } else if (*lwork < max(1,mn) && ! lquery) {
	*info = -9;
    }

    if (*info == 0) {
	if (wantq) {
	    nb = ilaenv_(&c__1, "SORGQR", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	} else {
	    nb = ilaenv_(&c__1, "SORGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	}
	lwkopt = max(1,mn) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	work[1] = 1.f;
	return 0;
    }

    if (wantq) {

/*
          Form Q, determined by a call to SGEBRD to reduce an m-by-k
          matrix
*/

	if (*m >= *k) {

/*           If m >= k, assume m >= n >= k */

	    sorgqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*
             If m < k, assume m = n

             Shift the vectors which define the elementary reflectors one
             column to the right, and set the first row and column of Q
             to those of the unit matrix
*/

	    for (j = *m; j >= 2; --j) {
		a[j * a_dim1 + 1] = 0.f;
		i__1 = *m;
		for (i__ = j + 1; i__ <= i__1; ++i__) {
		    a[i__ + j * a_dim1] = a[i__ + (j - 1) * a_dim1];
/* L10: */
		}
/* L20: */
	    }
	    a[a_dim1 + 1] = 1.f;
	    i__1 = *m;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.f;
/* L30: */
	    }
	    if (*m > 1) {

/*              Form Q(2:m,2:m) */

		i__1 = *m - 1;
		i__2 = *m - 1;
		i__3 = *m - 1;
		sorgqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    } else {

/*
          Form P', determined by a call to SGEBRD to reduce a k-by-n
          matrix
*/

	if (*k < *n) {

/*           If k < n, assume k <= m <= n */

	    sorglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*
             If k >= n, assume m = n

             Shift the vectors which define the elementary reflectors one
             row downward, and set the first row and column of P' to
             those of the unit matrix
*/

	    a[a_dim1 + 1] = 1.f;
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.f;
/* L40: */
	    }
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		for (i__ = j - 1; i__ >= 2; --i__) {
		    a[i__ + j * a_dim1] = a[i__ - 1 + j * a_dim1];
/* L50: */
		}
		a[j * a_dim1 + 1] = 0.f;
/* L60: */
	    }
	    if (*n > 1) {

/*              Form P'(2:n,2:n) */

		i__1 = *n - 1;
		i__2 = *n - 1;
		i__3 = *n - 1;
		sorglq_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORGBR */

} /* sorgbr_ */

/* Subroutine */ int sorghr_(integer *n, integer *ilo, integer *ihi, real *a,
	integer *lda, real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, nb, nh, iinfo;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int sorgqr_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *, integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORGHR generates a real orthogonal matrix Q which is defined as the
    product of IHI-ILO elementary reflectors of order N, as returned by
    SGEHRD:

    Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix Q. N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            ILO and IHI must have the same values as in the previous call
            of SGEHRD. Q is equal to the unit matrix except in the
            submatrix Q(ilo+1:ihi,ilo+1:ihi).
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by SGEHRD.
            On exit, the N-by-N orthogonal matrix Q.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,N).

    TAU     (input) REAL array, dimension (N-1)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEHRD.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= IHI-ILO.
            For optimum performance LWORK >= (IHI-ILO)*NB, where NB is
            the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nh = *ihi - *ilo;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,nh) && ! lquery) {
	*info = -8;
    }

    if (*info == 0) {
	nb = ilaenv_(&c__1, "SORGQR", " ", &nh, &nh, &nh, &c_n1, (ftnlen)6, (
		ftnlen)1);
	lwkopt = max(1,nh) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGHR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1] = 1.f;
	return 0;
    }

/*
       Shift the vectors which define the elementary reflectors one
       column to the right, and set the first ilo and the last n-ihi
       rows and columns to those of the unit matrix
*/

    i__1 = *ilo + 1;
    for (j = *ihi; j >= i__1; --j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.f;
/* L10: */
	}
	i__2 = *ihi;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = a[i__ + (j - 1) * a_dim1];
/* L20: */
	}
	i__2 = *n;
	for (i__ = *ihi + 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.f;
/* L30: */
	}
/* L40: */
    }
    i__1 = *ilo;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.f;
/* L50: */
	}
	a[j + j * a_dim1] = 1.f;
/* L60: */
    }
    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.f;
/* L70: */
	}
	a[j + j * a_dim1] = 1.f;
/* L80: */
    }

    if (nh > 0) {

/*        Generate Q(ilo+1:ihi,ilo+1:ihi) */

	sorgqr_(&nh, &nh, &nh, &a[*ilo + 1 + (*ilo + 1) * a_dim1], lda, &tau[*
		ilo], &work[1], lwork, &iinfo);
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORGHR */

} /* sorghr_ */

/* Subroutine */ int sorgl2_(integer *m, integer *n, integer *k, real *a,
	integer *lda, real *tau, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1;

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    slarf_(char *, integer *, integer *, real *, integer *, real *,
	    real *, integer *, real *), xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORGL2 generates an m by n real matrix Q with orthonormal rows,
    which is defined as the first m rows of a product of k elementary
    reflectors of order n

          Q  =  H(k) . . . H(2) H(1)

    as returned by SGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by SGELQF in the first k rows of its array argument A.
            On exit, the m-by-n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGELQF.

    WORK    (workspace) REAL array, dimension (M)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument has an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*k < 0 || *k > *m) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGL2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m <= 0) {
	return 0;
    }

    if (*k < *m) {

/*        Initialise rows k+1:m to rows of the unit matrix */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (l = *k + 1; l <= i__2; ++l) {
		a[l + j * a_dim1] = 0.f;
/* L10: */
	    }
	    if (j > *k && j <= *m) {
		a[j + j * a_dim1] = 1.f;
	    }
/* L20: */
	}
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i) to A(i:m,i:n) from the right */

	if (i__ < *n) {
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.f;
		i__1 = *m - i__;
		i__2 = *n - i__ + 1;
		slarf_("Right", &i__1, &i__2, &a[i__ + i__ * a_dim1], lda, &
			tau[i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    i__1 = *n - i__;
	    r__1 = -tau[i__];
	    sscal_(&i__1, &r__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	a[i__ + i__ * a_dim1] = 1.f - tau[i__];

/*        Set A(i,1:i-1) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    a[i__ + l * a_dim1] = 0.f;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of SORGL2 */

} /* sorgl2_ */

/* Subroutine */ int sorglq_(integer *m, integer *n, integer *k, real *a,
	integer *lda, real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int sorgl2_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *), slarfb_(char *, char *,
	    char *, char *, integer *, integer *, integer *, real *, integer *
	    , real *, integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORGLQ generates an M-by-N real matrix Q with orthonormal rows,
    which is defined as the first M rows of a product of K elementary
    reflectors of order N

          Q  =  H(k) . . . H(2) H(1)

    as returned by SGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by SGELQF in the first k rows of its array argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGELQF.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= max(1,M).
            For optimum performance LWORK >= M*NB, where NB is
            the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument has an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nb = ilaenv_(&c__1, "SORGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*m) * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*k < 0 || *k > *m) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m <= 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    nx = 0;
    iws = *m;
    if (nb > 1 && nb < *k) {

/*
          Determine when to cross over from blocked to unblocked code.

   Computing MAX
*/
	i__1 = 0, i__2 = ilaenv_(&c__3, "SORGLQ", " ", m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < *k) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  reduce NB and
                determine the minimum value of NB.
*/

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "SORGLQ", " ", m, n, k, &c_n1,
			 (ftnlen)6, (ftnlen)1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < *k && nx < *k) {

/*
          Use blocked code after the last block.
          The first kk rows are handled by the block method.
*/

	ki = (*k - nx - 1) / nb * nb;
/* Computing MIN */
	i__1 = *k, i__2 = ki + nb;
	kk = min(i__1,i__2);

/*        Set A(kk+1:m,1:kk) to zero. */

	i__1 = kk;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = kk + 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = 0.f;
/* L10: */
	    }
/* L20: */
	}
    } else {
	kk = 0;
    }

/*     Use unblocked code for the last or only block. */

    if (kk < *m) {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	sorgl2_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
		tau[kk + 1], &work[1], &iinfo);
    }

    if (kk > 0) {

/*        Use blocked code */

	i__1 = -nb;
	for (i__ = ki + 1; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = nb, i__3 = *k - i__ + 1;
	    ib = min(i__2,i__3);
	    if (i__ + ib <= *m) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__2 = *n - i__ + 1;
		slarft_("Forward", "Rowwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i+ib:m,i:n) from the right */

		i__2 = *m - i__ - ib + 1;
		i__3 = *n - i__ + 1;
		slarfb_("Right", "Transpose", "Forward", "Rowwise", &i__2, &
			i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[ib +
			1], &ldwork);
	    }

/*           Apply H' to columns i:n of current block */

	    i__2 = *n - i__ + 1;
	    sorgl2_(&ib, &i__2, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set columns 1:i-1 of current block to zero */

	    i__2 = i__ - 1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + ib - 1;
		for (l = i__; l <= i__3; ++l) {
		    a[l + j * a_dim1] = 0.f;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1] = (real) iws;
    return 0;

/*     End of SORGLQ */

} /* sorglq_ */

/* Subroutine */ int sorgqr_(integer *m, integer *n, integer *k, real *a,
	integer *lda, real *tau, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int sorg2r_(integer *, integer *, integer *, real
	    *, integer *, real *, real *, integer *), slarfb_(char *, char *,
	    char *, char *, integer *, integer *, integer *, real *, integer *
	    , real *, integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORGQR generates an M-by-N real matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by SGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by SGEQRF in the first k columns of its array
            argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF.

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= max(1,N).
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument has an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nb = ilaenv_(&c__1, "SORGQR", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*n) * nb;
    work[1] = (real) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    nx = 0;
    iws = *n;
    if (nb > 1 && nb < *k) {

/*
          Determine when to cross over from blocked to unblocked code.

   Computing MAX
*/
	i__1 = 0, i__2 = ilaenv_(&c__3, "SORGQR", " ", m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < *k) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  reduce NB and
                determine the minimum value of NB.
*/

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "SORGQR", " ", m, n, k, &c_n1,
			 (ftnlen)6, (ftnlen)1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < *k && nx < *k) {

/*
          Use blocked code after the last block.
          The first kk columns are handled by the block method.
*/

	ki = (*k - nx - 1) / nb * nb;
/* Computing MIN */
	i__1 = *k, i__2 = ki + nb;
	kk = min(i__1,i__2);

/*        Set A(1:kk,kk+1:n) to zero. */

	i__1 = *n;
	for (j = kk + 1; j <= i__1; ++j) {
	    i__2 = kk;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = 0.f;
/* L10: */
	    }
/* L20: */
	}
    } else {
	kk = 0;
    }

/*     Use unblocked code for the last or only block. */

    if (kk < *n) {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	sorg2r_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
		tau[kk + 1], &work[1], &iinfo);
    }

    if (kk > 0) {

/*        Use blocked code */

	i__1 = -nb;
	for (i__ = ki + 1; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
/* Computing MIN */
	    i__2 = nb, i__3 = *k - i__ + 1;
	    ib = min(i__2,i__3);
	    if (i__ + ib <= *n) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__2 = *m - i__ + 1;
		slarft_("Forward", "Columnwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i:m,i+ib:n) from the left */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__ - ib + 1;
		slarfb_("Left", "No transpose", "Forward", "Columnwise", &
			i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &
			work[ib + 1], &ldwork);
	    }

/*           Apply H to rows i:m of current block */

	    i__2 = *m - i__ + 1;
	    sorg2r_(&i__2, &ib, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set rows 1:i-1 of current block to zero */

	    i__2 = i__ + ib - 1;
	    for (j = i__; j <= i__2; ++j) {
		i__3 = i__ - 1;
		for (l = 1; l <= i__3; ++l) {
		    a[l + j * a_dim1] = 0.f;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1] = (real) iws;
    return 0;

/*     End of SORGQR */

} /* sorgqr_ */

/* Subroutine */ int sorm2l_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    /* Local variables */
    static integer i__, i1, i2, i3, mi, ni, nq;
    static real aii;
    static logical left;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SORM2L overwrites the general real m by n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'T', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'T',

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by SGEQLF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'T': apply Q' (Transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQLF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) REAL array, dimension
                                     (N) if SIDE = 'L',
                                     (M) if SIDE = 'R'

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");

/*     NQ is the order of Q */

    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORM2L", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }

    if (left && notran || ! left && ! notran) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }

    if (left) {
	ni = *n;
    } else {
	mi = *m;
    }

    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {

/*           H(i) is applied to C(1:m-k+i,1:n) */

	    mi = *m - *k + i__;
	} else {

/*           H(i) is applied to C(1:m,1:n-k+i) */

	    ni = *n - *k + i__;
	}

/*        Apply H(i) */

	aii = a[nq - *k + i__ + i__ * a_dim1];
	a[nq - *k + i__ + i__ * a_dim1] = 1.f;
	slarf_(side, &mi, &ni, &a[i__ * a_dim1 + 1], &c__1, &tau[i__], &c__[
		c_offset], ldc, &work[1]);
	a[nq - *k + i__ + i__ * a_dim1] = aii;
/* L10: */
    }
    return 0;

/*     End of SORM2L */

} /* sorm2l_ */

/* Subroutine */ int sorm2r_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static real aii;
    static logical left;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SORM2R overwrites the general real m by n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'T', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'T',

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by SGEQRF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'T': apply Q' (Transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) REAL array, dimension
                                     (N) if SIDE = 'L',
                                     (M) if SIDE = 'R'

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");

/*     NQ is the order of Q */

    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORM2R", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }

    if (left && ! notran || ! left && notran) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }

    if (left) {
	ni = *n;
	jc = 1;
    } else {
	mi = *m;
	ic = 1;
    }

    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {

/*           H(i) is applied to C(i:m,1:n) */

	    mi = *m - i__ + 1;
	    ic = i__;
	} else {

/*           H(i) is applied to C(1:m,i:n) */

	    ni = *n - i__ + 1;
	    jc = i__;
	}

/*        Apply H(i) */

	aii = a[i__ + i__ * a_dim1];
	a[i__ + i__ * a_dim1] = 1.f;
	slarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], &c__1, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
/* L10: */
    }
    return 0;

/*     End of SORM2R */

} /* sorm2r_ */

/* Subroutine */ int sormbr_(char *vect, char *side, char *trans, integer *m,
	integer *n, integer *k, real *a, integer *lda, real *tau, real *c__,
	integer *ldc, real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2];
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i1, i2, nb, mi, ni, nq, nw;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer iinfo;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical notran, applyq;
    static char transt[1];
    extern /* Subroutine */ int sormlq_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *, integer *);
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int sormqr_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    If VECT = 'Q', SORMBR overwrites the general real M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**T * C       C * Q**T

    If VECT = 'P', SORMBR overwrites the general real M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      P * C          C * P
    TRANS = 'T':      P**T * C       C * P**T

    Here Q and P**T are the orthogonal matrices determined by SGEBRD when
    reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
    P**T are defined as products of elementary reflectors H(i) and G(i)
    respectively.

    Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
    order of the orthogonal matrix Q or P**T that is applied.

    If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
    if nq >= k, Q = H(1) H(2) . . . H(k);
    if nq < k, Q = H(1) H(2) . . . H(nq-1).

    If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
    if k < nq, P = G(1) G(2) . . . G(k);
    if k >= nq, P = G(1) G(2) . . . G(nq-1).

    Arguments
    =========

    VECT    (input) CHARACTER*1
            = 'Q': apply Q or Q**T;
            = 'P': apply P or P**T.

    SIDE    (input) CHARACTER*1
            = 'L': apply Q, Q**T, P or P**T from the Left;
            = 'R': apply Q, Q**T, P or P**T from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q  or P;
            = 'T':  Transpose, apply Q**T or P**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            If VECT = 'Q', the number of columns in the original
            matrix reduced by SGEBRD.
            If VECT = 'P', the number of rows in the original
            matrix reduced by SGEBRD.
            K >= 0.

    A       (input) REAL array, dimension
                                  (LDA,min(nq,K)) if VECT = 'Q'
                                  (LDA,nq)        if VECT = 'P'
            The vectors which define the elementary reflectors H(i) and
            G(i), whose products determine the matrices Q and P, as
            returned by SGEBRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If VECT = 'Q', LDA >= max(1,nq);
            if VECT = 'P', LDA >= max(1,min(nq,K)).

    TAU     (input) REAL array, dimension (min(nq,K))
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i) which determines Q or P, as returned
            by SGEBRD in the array argument TAUQ or TAUP.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q
            or P*C or P**T*C or C*P or C*P**T.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    applyq = lsame_(vect, "Q");
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;

/*     NQ is the order of Q or P and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! applyq && ! lsame_(vect, "P")) {
	*info = -1;
    } else if (! left && ! lsame_(side, "R")) {
	*info = -2;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*k < 0) {
	*info = -6;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__2 = min(nq,*k);
	if (applyq && *lda < max(1,nq) || ! applyq && *lda < max(i__1,i__2)) {
	    *info = -8;
	} else if (*ldc < max(1,*m)) {
	    *info = -11;
	} else if (*lwork < max(1,nw) && ! lquery) {
	    *info = -13;
	}
    }

    if (*info == 0) {
	if (applyq) {
	    if (left) {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *m - 1;
		i__2 = *m - 1;
		nb = ilaenv_(&c__1, "SORMQR", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "SORMQR", ch__1, m, &i__1, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	} else {
	    if (left) {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *m - 1;
		i__2 = *m - 1;
		nb = ilaenv_(&c__1, "SORMLQ", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "SORMLQ", ch__1, m, &i__1, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    work[1] = 1.f;
    if (*m == 0 || *n == 0) {
	return 0;
    }

    if (applyq) {

/*        Apply Q */

	if (nq >= *k) {

/*           Q was determined by a call to SGEBRD with nq >= k */

	    sormqr_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           Q was determined by a call to SGEBRD with nq < k */

	    if (left) {
		mi = *m - 1;
		ni = *n;
		i1 = 2;
		i2 = 1;
	    } else {
		mi = *m;
		ni = *n - 1;
		i1 = 1;
		i2 = 2;
	    }
	    i__1 = nq - 1;
	    sormqr_(side, trans, &mi, &ni, &i__1, &a[a_dim1 + 2], lda, &tau[1]
		    , &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
	}
    } else {

/*        Apply P */

	if (notran) {
	    *(unsigned char *)transt = 'T';
	} else {
	    *(unsigned char *)transt = 'N';
	}
	if (nq > *k) {

/*           P was determined by a call to SGEBRD with nq > k */

	    sormlq_(side, transt, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           P was determined by a call to SGEBRD with nq <= k */

	    if (left) {
		mi = *m - 1;
		ni = *n;
		i1 = 2;
		i2 = 1;
	    } else {
		mi = *m;
		ni = *n - 1;
		i1 = 1;
		i2 = 2;
	    }
	    i__1 = nq - 1;
	    sormlq_(side, transt, &mi, &ni, &i__1, &a[(a_dim1 << 1) + 1], lda,
		     &tau[1], &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &
		    iinfo);
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORMBR */

} /* sormbr_ */

/* Subroutine */ int sorml2_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static real aii;
    static logical left;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int slarf_(char *, integer *, integer *, real *,
	    integer *, real *, real *, integer *, real *), xerbla_(
	    char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SORML2 overwrites the general real m by n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'T', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'T',

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by SGELQF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'T': apply Q' (Transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGELQF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) REAL array, dimension
                                     (N) if SIDE = 'L',
                                     (M) if SIDE = 'R'

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");

/*     NQ is the order of Q */

    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,*k)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORML2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }

    if (left && notran || ! left && ! notran) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }

    if (left) {
	ni = *n;
	jc = 1;
    } else {
	mi = *m;
	ic = 1;
    }

    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {

/*           H(i) is applied to C(i:m,1:n) */

	    mi = *m - i__ + 1;
	    ic = i__;
	} else {

/*           H(i) is applied to C(1:m,i:n) */

	    ni = *n - i__ + 1;
	    jc = i__;
	}

/*        Apply H(i) */

	aii = a[i__ + i__ * a_dim1];
	a[i__ + i__ * a_dim1] = 1.f;
	slarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], lda, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
/* L10: */
    }
    return 0;

/*     End of SORML2 */

} /* sorml2_ */

/* Subroutine */ int sormlq_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2], i__4,
	    i__5;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__;
    static real t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int sorml2_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *), slarfb_(char *, char *, char *, char *
	    , integer *, integer *, integer *, real *, integer *, real *,
	    integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static logical notran;
    static integer ldwork;
    static char transt[1];
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORMLQ overwrites the general real M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**T * C       C * Q**T

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by SGELQF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**T from the Left;
            = 'R': apply Q or Q**T from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGELQF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;

/*     NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,*k)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {

/*
          Determine the block size.  NB may be at most NBMAX, where NBMAX
          is used to define the local array T.

   Computing MIN
   Writing concatenation
*/
	i__3[0] = 1, a__1[0] = side;
	i__3[1] = 1, a__1[1] = trans;
	s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	i__1 = 64, i__2 = ilaenv_(&c__1, "SORMLQ", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	iws = nw * nb;
	if (*lwork < iws) {
	    nb = *lwork / ldwork;
/*
   Computing MAX
   Writing concatenation
*/
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	    i__1 = 2, i__2 = ilaenv_(&c__2, "SORMLQ", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	sorml2_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if (left && notran || ! left && ! notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}

	if (left) {
	    ni = *n;
	    jc = 1;
	} else {
	    mi = *m;
	    ic = 1;
	}

	if (notran) {
	    *(unsigned char *)transt = 'T';
	} else {
	    *(unsigned char *)transt = 'N';
	}

	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);

/*
             Form the triangular factor of the block reflector
             H = H(i) H(i+1) . . . H(i+ib-1)
*/

	    i__4 = nq - i__ + 1;
	    slarft_("Forward", "Rowwise", &i__4, &ib, &a[i__ + i__ * a_dim1],
		    lda, &tau[i__], t, &c__65);
	    if (left) {

/*              H or H' is applied to C(i:m,1:n) */

		mi = *m - i__ + 1;
		ic = i__;
	    } else {

/*              H or H' is applied to C(1:m,i:n) */

		ni = *n - i__ + 1;
		jc = i__;
	    }

/*           Apply H or H' */

	    slarfb_(side, transt, "Forward", "Rowwise", &mi, &ni, &ib, &a[i__
		    + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc * c_dim1],
		    ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORMLQ */

} /* sormlq_ */

/* Subroutine */ int sormql_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2], i__4,
	    i__5;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__;
    static real t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int sorm2l_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *), slarfb_(char *, char *, char *, char *
	    , integer *, integer *, integer *, real *, integer *, real *,
	    integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static logical notran;
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORMQL overwrites the general real M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**T * C       C * Q**T

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by SGEQLF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**T from the Left;
            = 'R': apply Q or Q**T from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQLF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;

/*     NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {

/*
          Determine the block size.  NB may be at most NBMAX, where NBMAX
          is used to define the local array T.

   Computing MIN
   Writing concatenation
*/
	i__3[0] = 1, a__1[0] = side;
	i__3[1] = 1, a__1[1] = trans;
	s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	i__1 = 64, i__2 = ilaenv_(&c__1, "SORMQL", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMQL", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	iws = nw * nb;
	if (*lwork < iws) {
	    nb = *lwork / ldwork;
/*
   Computing MAX
   Writing concatenation
*/
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	    i__1 = 2, i__2 = ilaenv_(&c__2, "SORMQL", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	sorm2l_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if (left && notran || ! left && ! notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}

	if (left) {
	    ni = *n;
	} else {
	    mi = *m;
	}

	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);

/*
             Form the triangular factor of the block reflector
             H = H(i+ib-1) . . . H(i+1) H(i)
*/

	    i__4 = nq - *k + i__ + ib - 1;
	    slarft_("Backward", "Columnwise", &i__4, &ib, &a[i__ * a_dim1 + 1]
		    , lda, &tau[i__], t, &c__65);
	    if (left) {

/*              H or H' is applied to C(1:m-k+i+ib-1,1:n) */

		mi = *m - *k + i__ + ib - 1;
	    } else {

/*              H or H' is applied to C(1:m,1:n-k+i+ib-1) */

		ni = *n - *k + i__ + ib - 1;
	    }

/*           Apply H or H' */

	    slarfb_(side, trans, "Backward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ * a_dim1 + 1], lda, t, &c__65, &c__[c_offset], ldc, &
		    work[1], &ldwork);
/* L10: */
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORMQL */

} /* sormql_ */

/* Subroutine */ int sormqr_(char *side, char *trans, integer *m, integer *n,
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2], i__4,
	    i__5;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__;
    static real t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int sorm2r_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *), slarfb_(char *, char *, char *, char *
	    , integer *, integer *, integer *, real *, integer *, real *,
	    integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slarft_(char *, char *, integer *, integer *,
	    real *, integer *, real *, real *, integer *);
    static logical notran;
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORMQR overwrites the general real M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**T * C       C * Q**T

    where Q is a real orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by SGEQRF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**T from the Left;
            = 'R': apply Q or Q**T from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;

/*     NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {

/*
          Determine the block size.  NB may be at most NBMAX, where NBMAX
          is used to define the local array T.

   Computing MIN
   Writing concatenation
*/
	i__3[0] = 1, a__1[0] = side;
	i__3[1] = 1, a__1[1] = trans;
	s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	i__1 = 64, i__2 = ilaenv_(&c__1, "SORMQR", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	iws = nw * nb;
	if (*lwork < iws) {
	    nb = *lwork / ldwork;
/*
   Computing MAX
   Writing concatenation
*/
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
	    i__1 = 2, i__2 = ilaenv_(&c__2, "SORMQR", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	sorm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if (left && ! notran || ! left && notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}

	if (left) {
	    ni = *n;
	    jc = 1;
	} else {
	    mi = *m;
	    ic = 1;
	}

	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);

/*
             Form the triangular factor of the block reflector
             H = H(i) H(i+1) . . . H(i+ib-1)
*/

	    i__4 = nq - i__ + 1;
	    slarft_("Forward", "Columnwise", &i__4, &ib, &a[i__ + i__ *
		    a_dim1], lda, &tau[i__], t, &c__65)
		    ;
	    if (left) {

/*              H or H' is applied to C(i:m,1:n) */

		mi = *m - i__ + 1;
		ic = i__;
	    } else {

/*              H or H' is applied to C(1:m,i:n) */

		ni = *n - i__ + 1;
		jc = i__;
	    }

/*           Apply H or H' */

	    slarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc *
		    c_dim1], ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORMQR */

} /* sormqr_ */

/* Subroutine */ int sormtr_(char *side, char *uplo, char *trans, integer *m,
	integer *n, real *a, integer *lda, real *tau, real *c__, integer *ldc,
	 real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1[2], i__2, i__3;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i1, i2, nb, mi, ni, nq, nw;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer iinfo;
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int sormql_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *, integer *);
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int sormqr_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SORMTR overwrites the general real M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**T * C       C * Q**T

    where Q is a real orthogonal matrix of order nq, with nq = m if
    SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
    nq-1 elementary reflectors, as returned by SSYTRD:

    if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);

    if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**T from the Left;
            = 'R': apply Q or Q**T from the Right.

    UPLO    (input) CHARACTER*1
            = 'U': Upper triangle of A contains elementary reflectors
                   from SSYTRD;
            = 'L': Lower triangle of A contains elementary reflectors
                   from SSYTRD.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    A       (input) REAL array, dimension
                                 (LDA,M) if SIDE = 'L'
                                 (LDA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by SSYTRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.

    TAU     (input) REAL array, dimension
                                 (M-1) if SIDE = 'L'
                                 (N-1) if SIDE = 'R'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SSYTRD.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;

/*     NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (! lsame_(trans, "N") && ! lsame_(trans,
	    "T")) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {
	if (upper) {
	    if (left) {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *m - 1;
		i__3 = *m - 1;
		nb = ilaenv_(&c__1, "SORMQL", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "SORMQL", ch__1, m, &i__2, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	} else {
	    if (left) {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *m - 1;
		i__3 = *m - 1;
		nb = ilaenv_(&c__1, "SORMQR", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "SORMQR", ch__1, m, &i__2, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("SORMTR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || nq == 1) {
	work[1] = 1.f;
	return 0;
    }

    if (left) {
	mi = *m - 1;
	ni = *n;
    } else {
	mi = *m;
	ni = *n - 1;
    }

    if (upper) {

/*        Q was determined by a call to SSYTRD with UPLO = 'U' */

	i__2 = nq - 1;
	sormql_(side, trans, &mi, &ni, &i__2, &a[(a_dim1 << 1) + 1], lda, &
		tau[1], &c__[c_offset], ldc, &work[1], lwork, &iinfo);
    } else {

/*        Q was determined by a call to SSYTRD with UPLO = 'L' */

	if (left) {
	    i1 = 2;
	    i2 = 1;
	} else {
	    i1 = 1;
	    i2 = 2;
	}
	i__2 = nq - 1;
	sormqr_(side, trans, &mi, &ni, &i__2, &a[a_dim1 + 2], lda, &tau[1], &
		c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORMTR */

} /* sormtr_ */

/* Subroutine */ int spotf2_(char *uplo, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer j;
    static real ajj;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    sgemv_(char *, integer *, integer *, real *, real *, integer *,
	    real *, integer *, real *, real *, integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    SPOTF2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
       A = U' * U ,  if UPLO = 'U', or
       A = L  * L',  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U'*U  or A = L*L'.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTF2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (upper) {

/*        Compute the Cholesky factorization A = U'*U. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute U(J,J) and test for non-positive-definiteness. */

	    i__2 = j - 1;
	    ajj = a[j + j * a_dim1] - sdot_(&i__2, &a[j * a_dim1 + 1], &c__1,
		    &a[j * a_dim1 + 1], &c__1);
	    if (ajj <= 0.f) {
		a[j + j * a_dim1] = ajj;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    a[j + j * a_dim1] = ajj;

/*           Compute elements J+1:N of row J. */

	    if (j < *n) {
		i__2 = j - 1;
		i__3 = *n - j;
		sgemv_("Transpose", &i__2, &i__3, &c_b151, &a[(j + 1) *
			a_dim1 + 1], lda, &a[j * a_dim1 + 1], &c__1, &c_b15, &
			a[j + (j + 1) * a_dim1], lda);
		i__2 = *n - j;
		r__1 = 1.f / ajj;
		sscal_(&i__2, &r__1, &a[j + (j + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness. */

	    i__2 = j - 1;
	    ajj = a[j + j * a_dim1] - sdot_(&i__2, &a[j + a_dim1], lda, &a[j
		    + a_dim1], lda);
	    if (ajj <= 0.f) {
		a[j + j * a_dim1] = ajj;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    a[j + j * a_dim1] = ajj;

/*           Compute elements J+1:N of column J. */

	    if (j < *n) {
		i__2 = *n - j;
		i__3 = j - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b151, &a[j + 1 +
			a_dim1], lda, &a[j + a_dim1], lda, &c_b15, &a[j + 1 +
			j * a_dim1], &c__1);
		i__2 = *n - j;
		r__1 = 1.f / ajj;
		sscal_(&i__2, &r__1, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }
    goto L40;

L30:
    *info = j;

L40:
    return 0;

/*     End of SPOTF2 */

} /* spotf2_ */

/* Subroutine */ int spotrf_(char *uplo, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer j, jb, nb;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static logical upper;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *,
	    integer *, integer *, real *, real *, integer *, real *, integer *
	    ), ssyrk_(char *, char *, integer
	    *, integer *, real *, real *, integer *, real *, real *, integer *
	    ), spotf2_(char *, integer *, real *, integer *,
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
       A = U**T * U,  if UPLO = 'U', or
       A = L  * L**T,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**T*U or A = L*L**T.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "SPOTRF", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);
    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code. */

	spotf2_(uplo, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code. */

	if (upper) {

/*           Compute the Cholesky factorization A = U'*U. */

	    i__1 = *n;
	    i__2 = nb;
	    for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*
                Update and factorize the current diagonal block and test
                for non-positive-definiteness.

   Computing MIN
*/
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		ssyrk_("Upper", "Transpose", &jb, &i__3, &c_b151, &a[j *
			a_dim1 + 1], lda, &c_b15, &a[j + j * a_dim1], lda);
		spotf2_("Upper", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block row. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    sgemm_("Transpose", "No transpose", &jb, &i__3, &i__4, &
			    c_b151, &a[j * a_dim1 + 1], lda, &a[(j + jb) *
			    a_dim1 + 1], lda, &c_b15, &a[j + (j + jb) *
			    a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    strsm_("Left", "Upper", "Transpose", "Non-unit", &jb, &
			    i__3, &c_b15, &a[j + j * a_dim1], lda, &a[j + (j
			    + jb) * a_dim1], lda);
		}
/* L10: */
	    }

	} else {

/*           Compute the Cholesky factorization A = L*L'. */

	    i__2 = *n;
	    i__1 = nb;
	    for (j = 1; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*
                Update and factorize the current diagonal block and test
                for non-positive-definiteness.

   Computing MIN
*/
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		ssyrk_("Lower", "No transpose", &jb, &i__3, &c_b151, &a[j +
			a_dim1], lda, &c_b15, &a[j + j * a_dim1], lda);
		spotf2_("Lower", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block column. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    sgemm_("No transpose", "Transpose", &i__3, &jb, &i__4, &
			    c_b151, &a[j + jb + a_dim1], lda, &a[j + a_dim1],
			    lda, &c_b15, &a[j + jb + j * a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    strsm_("Right", "Lower", "Transpose", "Non-unit", &i__3, &
			    jb, &c_b15, &a[j + j * a_dim1], lda, &a[j + jb +
			    j * a_dim1], lda);
		}
/* L20: */
	    }
	}
    }
    goto L40;

L30:
    *info = *info + j - 1;

L40:
    return 0;

/*     End of SPOTRF */

} /* spotrf_ */

/* Subroutine */ int spotri_(char *uplo, integer *n, real *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), slauum_(
	    char *, integer *, real *, integer *, integer *), strtri_(
	    char *, char *, integer *, real *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SPOTRI computes the inverse of a real symmetric positive definite
    matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
    computed by SPOTRF.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the triangular factor U or L from the Cholesky
            factorization A = U**T*U or A = L*L**T, as computed by
            SPOTRF.
            On exit, the upper or lower triangle of the (symmetric)
            inverse of A, overwriting the input factor U or L.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the (i,i) element of the factor U or L is
                  zero, and the inverse could not be computed.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Invert the triangular Cholesky factor U or L. */

    strtri_(uplo, "Non-unit", n, &a[a_offset], lda, info);
    if (*info > 0) {
	return 0;
    }

/*     Form inv(U)*inv(U)' or inv(L)'*inv(L). */

    slauum_(uplo, n, &a[a_offset], lda, info);

    return 0;

/*     End of SPOTRI */

} /* spotri_ */

/* Subroutine */ int spotrs_(char *uplo, integer *n, integer *nrhs, real *a,
	integer *lda, real *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *,
	    integer *, integer *, real *, real *, integer *, real *, integer *
	    ), xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    SPOTRS solves a system of linear equations A*X = B with a symmetric
    positive definite matrix A using the Cholesky factorization
    A = U**T*U or A = L*L**T computed by SPOTRF.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) REAL array, dimension (LDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**T*U or A = L*L**T, as computed by SPOTRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input/output) REAL array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

    if (upper) {

/*
          Solve A*X = B where A = U'*U.

          Solve U'*X = B, overwriting B with X.
*/

	strsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b15, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	strsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b15, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A*X = B where A = L*L'.

          Solve L*X = B, overwriting B with X.
*/

	strsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b15, &
		a[a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	strsm_("Left", "Lower", "Transpose", "Non-unit", n, nrhs, &c_b15, &a[
		a_offset], lda, &b[b_offset], ldb);
    }

    return 0;

/*     End of SPOTRS */

} /* spotrs_ */

/* Subroutine */ int sstedc_(char *compz, integer *n, real *d__, real *e,
	real *z__, integer *ldz, real *work, integer *lwork, integer *iwork,
	integer *liwork, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, k, m;
    static real p;
    static integer ii, end, lgn;
    static real eps, tiny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);
    static integer lwmin, start;
    extern /* Subroutine */ int sswap_(integer *, real *, integer *, real *,
	    integer *), slaed0_(integer *, integer *, integer *, real *, real
	    *, real *, integer *, real *, integer *, real *, integer *,
	    integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *), slacpy_(char *, integer *, integer *, real *, integer *,
	    real *, integer *), slaset_(char *, integer *, integer *,
	    real *, real *, real *, integer *);
    static integer liwmin, icompz;
    static real orgnrm;
    extern doublereal slanst_(char *, integer *, real *, real *);
    extern /* Subroutine */ int ssterf_(integer *, real *, real *, integer *),
	     slasrt_(char *, integer *, real *, integer *);
    static logical lquery;
    static integer smlsiz;
    extern /* Subroutine */ int ssteqr_(char *, integer *, real *, real *,
	    real *, integer *, real *, integer *);
    static integer storez, strtrw;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SSTEDC computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.
    The eigenvectors of a full or band real symmetric matrix can also be
    found if SSYTRD or SSPTRD or SSBTRD has been used to reduce this
    matrix to tridiagonal form.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.  See SLAED3 for details.

    Arguments
    =========

    COMPZ   (input) CHARACTER*1
            = 'N':  Compute eigenvalues only.
            = 'I':  Compute eigenvectors of tridiagonal matrix also.
            = 'V':  Compute eigenvectors of original dense symmetric
                    matrix also.  On entry, Z contains the orthogonal
                    matrix used to reduce the original matrix to
                    tridiagonal form.

    N       (input) INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) REAL array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    Z       (input/output) REAL array, dimension (LDZ,N)
            On entry, if COMPZ = 'V', then Z contains the orthogonal
            matrix used in the reduction to tridiagonal form.
            On exit, if INFO = 0, then if COMPZ = 'V', Z contains the
            orthonormal eigenvectors of the original symmetric matrix,
            and if COMPZ = 'I', Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.
            If  COMPZ = 'N', then Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.  LDZ >= 1.
            If eigenvectors are desired, then LDZ >= max(1,N).

    WORK    (workspace/output) REAL array,
                                           dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If COMPZ = 'N' or N <= 1 then LWORK must be at least 1.
            If COMPZ = 'V' and N > 1 then LWORK must be at least
                           ( 1 + 3*N + 2*N*lg N + 3*N**2 ),
                           where lg( N ) = smallest integer k such
                           that 2**k >= N.
            If COMPZ = 'I' and N > 1 then LWORK must be at least
                           ( 1 + 4*N + N**2 ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If COMPZ = 'N' or N <= 1 then LIWORK must be at least 1.
            If COMPZ = 'V' and N > 1 then LIWORK must be at least
                           ( 6 + 6*N + 5*N*lg N ).
            If COMPZ = 'I' and N > 1 then LIWORK must be at least
                           ( 3 + 5*N ).

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1 || *liwork == -1;

    if (lsame_(compz, "N")) {
	icompz = 0;
    } else if (lsame_(compz, "V")) {
	icompz = 1;
    } else if (lsame_(compz, "I")) {
	icompz = 2;
    } else {
	icompz = -1;
    }
    if (*n <= 1 || icompz <= 0) {
	liwmin = 1;
	lwmin = 1;
    } else {
	lgn = (integer) (log((real) (*n)) / log(2.f));
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	if (icompz == 1) {
/* Computing 2nd power */
	    i__1 = *n;
	    lwmin = *n * 3 + 1 + (*n << 1) * lgn + i__1 * i__1 * 3;
	    liwmin = *n * 6 + 6 + *n * 5 * lgn;
	} else if (icompz == 2) {
/* Computing 2nd power */
	    i__1 = *n;
	    lwmin = (*n << 2) + 1 + i__1 * i__1;
	    liwmin = *n * 5 + 3;
	}
    }
    if (icompz < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldz < 1 || icompz > 0 && *ldz < max(1,*n)) {
	*info = -6;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -8;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -10;
    }

    if (*info == 0) {
	work[1] = (real) lwmin;
	iwork[1] = liwmin;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSTEDC", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    if (*n == 1) {
	if (icompz != 0) {
	    z__[z_dim1 + 1] = 1.f;
	}
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "SSTEDC", " ", &c__0, &c__0, &c__0, &c__0, (
	    ftnlen)6, (ftnlen)1);

/*
       If the following conditional clause is removed, then the routine
       will use the Divide and Conquer routine to compute only the
       eigenvalues, which requires (3N + 3N**2) real workspace and
       (2 + 5N + 2N lg(N)) integer workspace.
       Since on many architectures SSTERF is much faster than any other
       algorithm for finding eigenvalues only, it is used here
       as the default.

       If COMPZ = 'N', use SSTERF to compute the eigenvalues.
*/

    if (icompz == 0) {
	ssterf_(n, &d__[1], &e[1], info);
	return 0;
    }

/*
       If N is smaller than the minimum divide size (SMLSIZ+1), then
       solve the problem with another solver.
*/

    if (*n <= smlsiz) {
	if (icompz == 0) {
	    ssterf_(n, &d__[1], &e[1], info);
	    return 0;
	} else if (icompz == 2) {
	    ssteqr_("I", n, &d__[1], &e[1], &z__[z_offset], ldz, &work[1],
		    info);
	    return 0;
	} else {
	    ssteqr_("V", n, &d__[1], &e[1], &z__[z_offset], ldz, &work[1],
		    info);
	    return 0;
	}
    }

/*
       If COMPZ = 'V', the Z matrix must be stored elsewhere for later
       use.
*/

    if (icompz == 1) {
	storez = *n * *n + 1;
    } else {
	storez = 1;
    }

    if (icompz == 2) {
	slaset_("Full", n, n, &c_b29, &c_b15, &z__[z_offset], ldz);
    }

/*     Scale. */

    orgnrm = slanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.f) {
	return 0;
    }

    eps = slamch_("Epsilon");

    start = 1;

/*     while ( START <= N ) */

L10:
    if (start <= *n) {

/*
       Let END be the position of the next subdiagonal entry such that
       E( END ) <= TINY or END = N if no such subdiagonal exists.  The
       matrix identified by the elements between START and END
       constitutes an independent sub-problem.
*/

	end = start;
L20:
	if (end < *n) {
	    tiny = eps * sqrt((r__1 = d__[end], dabs(r__1))) * sqrt((r__2 =
		    d__[end + 1], dabs(r__2)));
	    if ((r__1 = e[end], dabs(r__1)) > tiny) {
		++end;
		goto L20;
	    }
	}

/*        (Sub) Problem determined.  Compute its size and solve it. */

	m = end - start + 1;
	if (m == 1) {
	    start = end + 1;
	    goto L10;
	}
	if (m > smlsiz) {
	    *info = smlsiz;

/*           Scale. */

	    orgnrm = slanst_("M", &m, &d__[start], &e[start]);
	    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &m, &c__1, &d__[start]
		    , &m, info);
	    i__1 = m - 1;
	    i__2 = m - 1;
	    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &i__1, &c__1, &e[
		    start], &i__2, info);

	    if (icompz == 1) {
		strtrw = 1;
	    } else {
		strtrw = start;
	    }
	    slaed0_(&icompz, n, &m, &d__[start], &e[start], &z__[strtrw +
		    start * z_dim1], ldz, &work[1], n, &work[storez], &iwork[
		    1], info);
	    if (*info != 0) {
		*info = (*info / (m + 1) + start - 1) * (*n + 1) + *info % (m
			+ 1) + start - 1;
		return 0;
	    }

/*           Scale back. */

	    slascl_("G", &c__0, &c__0, &c_b15, &orgnrm, &m, &c__1, &d__[start]
		    , &m, info);

	} else {
	    if (icompz == 1) {

/*
       Since QR won't update a Z matrix which is larger than the
       length of D, we must solve the sub-problem in a workspace and
       then multiply back into Z.
*/

		ssteqr_("I", &m, &d__[start], &e[start], &work[1], &m, &work[
			m * m + 1], info);
		slacpy_("A", n, &m, &z__[start * z_dim1 + 1], ldz, &work[
			storez], n);
		sgemm_("N", "N", n, &m, &m, &c_b15, &work[storez], ldz, &work[
			1], &m, &c_b29, &z__[start * z_dim1 + 1], ldz);
	    } else if (icompz == 2) {
		ssteqr_("I", &m, &d__[start], &e[start], &z__[start + start *
			z_dim1], ldz, &work[1], info);
	    } else {
		ssterf_(&m, &d__[start], &e[start], info);
	    }
	    if (*info != 0) {
		*info = start * (*n + 1) + end;
		return 0;
	    }
	}

	start = end + 1;
	goto L10;
    }

/*
       endwhile

       If the problem split any number of times, then the eigenvalues
       will not be properly ordered.  Here we permute the eigenvalues
       (and the associated eigenvectors) into ascending order.
*/

    if (m != *n) {
	if (icompz == 0) {

/*        Use Quick Sort */

	    slasrt_("I", n, &d__[1], info);

	} else {

/*        Use Selection Sort to minimize swaps of eigenvectors */

	    i__1 = *n;
	    for (ii = 2; ii <= i__1; ++ii) {
		i__ = ii - 1;
		k = i__;
		p = d__[i__];
		i__2 = *n;
		for (j = ii; j <= i__2; ++j) {
		    if (d__[j] < p) {
			k = j;
			p = d__[j];
		    }
/* L30: */
		}
		if (k != i__) {
		    d__[k] = d__[i__];
		    d__[i__] = p;
		    sswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1
			    + 1], &c__1);
		}
/* L40: */
	    }
	}
    }

    work[1] = (real) lwmin;
    iwork[1] = liwmin;

    return 0;

/*     End of SSTEDC */

} /* sstedc_ */

/* Subroutine */ int ssteqr_(char *compz, integer *n, real *d__, real *e,
	real *z__, integer *ldz, real *work, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static real b, c__, f, g;
    static integer i__, j, k, l, m;
    static real p, r__, s;
    static integer l1, ii, mm, lm1, mm1, nm1;
    static real rt1, rt2, eps;
    static integer lsv;
    static real tst, eps2;
    static integer lend, jtot;
    extern /* Subroutine */ int slae2_(real *, real *, real *, real *, real *)
	    ;
    extern logical lsame_(char *, char *);
    static real anorm;
    extern /* Subroutine */ int slasr_(char *, char *, char *, integer *,
	    integer *, real *, real *, real *, integer *), sswap_(integer *, real *, integer *, real *, integer *);
    static integer lendm1, lendp1;
    extern /* Subroutine */ int slaev2_(real *, real *, real *, real *, real *
	    , real *, real *);
    extern doublereal slapy2_(real *, real *);
    static integer iscale;
    extern doublereal slamch_(char *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real safmax;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    static integer lendsv;
    extern /* Subroutine */ int slartg_(real *, real *, real *, real *, real *
	    ), slaset_(char *, integer *, integer *, real *, real *, real *,
	    integer *);
    static real ssfmin;
    static integer nmaxit, icompz;
    static real ssfmax;
    extern doublereal slanst_(char *, integer *, real *, real *);
    extern /* Subroutine */ int slasrt_(char *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    SSTEQR computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the implicit QL or QR method.
    The eigenvectors of a full or band symmetric matrix can also be found
    if SSYTRD or SSPTRD or SSBTRD has been used to reduce this matrix to
    tridiagonal form.

    Arguments
    =========

    COMPZ   (input) CHARACTER*1
            = 'N':  Compute eigenvalues only.
            = 'V':  Compute eigenvalues and eigenvectors of the original
                    symmetric matrix.  On entry, Z must contain the
                    orthogonal matrix used to reduce the original matrix
                    to tridiagonal form.
            = 'I':  Compute eigenvalues and eigenvectors of the
                    tridiagonal matrix.  Z is initialized to the identity
                    matrix.

    N       (input) INTEGER
            The order of the matrix.  N >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) REAL array, dimension (N-1)
            On entry, the (n-1) subdiagonal elements of the tridiagonal
            matrix.
            On exit, E has been destroyed.

    Z       (input/output) REAL array, dimension (LDZ, N)
            On entry, if  COMPZ = 'V', then Z contains the orthogonal
            matrix used in the reduction to tridiagonal form.
            On exit, if INFO = 0, then if  COMPZ = 'V', Z contains the
            orthonormal eigenvectors of the original symmetric matrix,
            and if COMPZ = 'I', Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.
            If COMPZ = 'N', then Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.  LDZ >= 1, and if
            eigenvectors are desired, then  LDZ >= max(1,N).

    WORK    (workspace) REAL array, dimension (max(1,2*N-2))
            If COMPZ = 'N', then WORK is not referenced.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  the algorithm has failed to find all the eigenvalues in
                  a total of 30*N iterations; if INFO = i, then i
                  elements of E have not converged to zero; on exit, D
                  and E contain the elements of a symmetric tridiagonal
                  matrix which is orthogonally similar to the original
                  matrix.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    /* Function Body */
    *info = 0;

    if (lsame_(compz, "N")) {
	icompz = 0;
    } else if (lsame_(compz, "V")) {
	icompz = 1;
    } else if (lsame_(compz, "I")) {
	icompz = 2;
    } else {
	icompz = -1;
    }
    if (icompz < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldz < 1 || icompz > 0 && *ldz < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSTEQR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (icompz == 2) {
	    z__[z_dim1 + 1] = 1.f;
	}
	return 0;
    }

/*     Determine the unit roundoff and over/underflow thresholds. */

    eps = slamch_("E");
/* Computing 2nd power */
    r__1 = eps;
    eps2 = r__1 * r__1;
    safmin = slamch_("S");
    safmax = 1.f / safmin;
    ssfmax = sqrt(safmax) / 3.f;
    ssfmin = sqrt(safmin) / eps2;

/*
       Compute the eigenvalues and eigenvectors of the tridiagonal
       matrix.
*/

    if (icompz == 2) {
	slaset_("Full", n, n, &c_b29, &c_b15, &z__[z_offset], ldz);
    }

    nmaxit = *n * 30;
    jtot = 0;

/*
       Determine where the matrix splits and choose QL or QR iteration
       for each block, according to whether top or bottom diagonal
       element is smaller.
*/

    l1 = 1;
    nm1 = *n - 1;

L10:
    if (l1 > *n) {
	goto L160;
    }
    if (l1 > 1) {
	e[l1 - 1] = 0.f;
    }
    if (l1 <= nm1) {
	i__1 = nm1;
	for (m = l1; m <= i__1; ++m) {
	    tst = (r__1 = e[m], dabs(r__1));
	    if (tst == 0.f) {
		goto L30;
	    }
	    if (tst <= sqrt((r__1 = d__[m], dabs(r__1))) * sqrt((r__2 = d__[m
		    + 1], dabs(r__2))) * eps) {
		e[m] = 0.f;
		goto L30;
	    }
/* L20: */
	}
    }
    m = *n;

L30:
    l = l1;
    lsv = l;
    lend = m;
    lendsv = lend;
    l1 = m + 1;
    if (lend == l) {
	goto L10;
    }

/*     Scale submatrix in rows and columns L to LEND */

    i__1 = lend - l + 1;
    anorm = slanst_("I", &i__1, &d__[l], &e[l]);
    iscale = 0;
    if (anorm == 0.f) {
	goto L10;
    }
    if (anorm > ssfmax) {
	iscale = 1;
	i__1 = lend - l + 1;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &e[l], n,
		info);
    } else if (anorm < ssfmin) {
	iscale = 2;
	i__1 = lend - l + 1;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &e[l], n,
		info);
    }

/*     Choose between QL and QR iteration */

    if ((r__1 = d__[lend], dabs(r__1)) < (r__2 = d__[l], dabs(r__2))) {
	lend = lsv;
	l = lendsv;
    }

    if (lend > l) {

/*
          QL Iteration

          Look for small subdiagonal element.
*/

L40:
	if (l != lend) {
	    lendm1 = lend - 1;
	    i__1 = lendm1;
	    for (m = l; m <= i__1; ++m) {
/* Computing 2nd power */
		r__2 = (r__1 = e[m], dabs(r__1));
		tst = r__2 * r__2;
		if (tst <= eps2 * (r__1 = d__[m], dabs(r__1)) * (r__2 = d__[m
			+ 1], dabs(r__2)) + safmin) {
		    goto L60;
		}
/* L50: */
	    }
	}

	m = lend;

L60:
	if (m < lend) {
	    e[m] = 0.f;
	}
	p = d__[l];
	if (m == l) {
	    goto L80;
	}

/*
          If remaining matrix is 2-by-2, use SLAE2 or SLAEV2
          to compute its eigensystem.
*/

	if (m == l + 1) {
	    if (icompz > 0) {
		slaev2_(&d__[l], &e[l], &d__[l + 1], &rt1, &rt2, &c__, &s);
		work[l] = c__;
		work[*n - 1 + l] = s;
		slasr_("R", "V", "B", n, &c__2, &work[l], &work[*n - 1 + l], &
			z__[l * z_dim1 + 1], ldz);
	    } else {
		slae2_(&d__[l], &e[l], &d__[l + 1], &rt1, &rt2);
	    }
	    d__[l] = rt1;
	    d__[l + 1] = rt2;
	    e[l] = 0.f;
	    l += 2;
	    if (l <= lend) {
		goto L40;
	    }
	    goto L140;
	}

	if (jtot == nmaxit) {
	    goto L140;
	}
	++jtot;

/*        Form shift. */

	g = (d__[l + 1] - p) / (e[l] * 2.f);
	r__ = slapy2_(&g, &c_b15);
	g = d__[m] - p + e[l] / (g + r_sign(&r__, &g));

	s = 1.f;
	c__ = 1.f;
	p = 0.f;

/*        Inner loop */

	mm1 = m - 1;
	i__1 = l;
	for (i__ = mm1; i__ >= i__1; --i__) {
	    f = s * e[i__];
	    b = c__ * e[i__];
	    slartg_(&g, &f, &c__, &s, &r__);
	    if (i__ != m - 1) {
		e[i__ + 1] = r__;
	    }
	    g = d__[i__ + 1] - p;
	    r__ = (d__[i__] - g) * s + c__ * 2.f * b;
	    p = s * r__;
	    d__[i__ + 1] = g + p;
	    g = c__ * r__ - b;

/*           If eigenvectors are desired, then save rotations. */

	    if (icompz > 0) {
		work[i__] = c__;
		work[*n - 1 + i__] = -s;
	    }

/* L70: */
	}

/*        If eigenvectors are desired, then apply saved rotations. */

	if (icompz > 0) {
	    mm = m - l + 1;
	    slasr_("R", "V", "B", n, &mm, &work[l], &work[*n - 1 + l], &z__[l
		    * z_dim1 + 1], ldz);
	}

	d__[l] -= p;
	e[l] = g;
	goto L40;

/*        Eigenvalue found. */

L80:
	d__[l] = p;

	++l;
	if (l <= lend) {
	    goto L40;
	}
	goto L140;

    } else {

/*
          QR Iteration

          Look for small superdiagonal element.
*/

L90:
	if (l != lend) {
	    lendp1 = lend + 1;
	    i__1 = lendp1;
	    for (m = l; m >= i__1; --m) {
/* Computing 2nd power */
		r__2 = (r__1 = e[m - 1], dabs(r__1));
		tst = r__2 * r__2;
		if (tst <= eps2 * (r__1 = d__[m], dabs(r__1)) * (r__2 = d__[m
			- 1], dabs(r__2)) + safmin) {
		    goto L110;
		}
/* L100: */
	    }
	}

	m = lend;

L110:
	if (m > lend) {
	    e[m - 1] = 0.f;
	}
	p = d__[l];
	if (m == l) {
	    goto L130;
	}

/*
          If remaining matrix is 2-by-2, use SLAE2 or SLAEV2
          to compute its eigensystem.
*/

	if (m == l - 1) {
	    if (icompz > 0) {
		slaev2_(&d__[l - 1], &e[l - 1], &d__[l], &rt1, &rt2, &c__, &s)
			;
		work[m] = c__;
		work[*n - 1 + m] = s;
		slasr_("R", "V", "F", n, &c__2, &work[m], &work[*n - 1 + m], &
			z__[(l - 1) * z_dim1 + 1], ldz);
	    } else {
		slae2_(&d__[l - 1], &e[l - 1], &d__[l], &rt1, &rt2);
	    }
	    d__[l - 1] = rt1;
	    d__[l] = rt2;
	    e[l - 1] = 0.f;
	    l += -2;
	    if (l >= lend) {
		goto L90;
	    }
	    goto L140;
	}

	if (jtot == nmaxit) {
	    goto L140;
	}
	++jtot;

/*        Form shift. */

	g = (d__[l - 1] - p) / (e[l - 1] * 2.f);
	r__ = slapy2_(&g, &c_b15);
	g = d__[m] - p + e[l - 1] / (g + r_sign(&r__, &g));

	s = 1.f;
	c__ = 1.f;
	p = 0.f;

/*        Inner loop */

	lm1 = l - 1;
	i__1 = lm1;
	for (i__ = m; i__ <= i__1; ++i__) {
	    f = s * e[i__];
	    b = c__ * e[i__];
	    slartg_(&g, &f, &c__, &s, &r__);
	    if (i__ != m) {
		e[i__ - 1] = r__;
	    }
	    g = d__[i__] - p;
	    r__ = (d__[i__ + 1] - g) * s + c__ * 2.f * b;
	    p = s * r__;
	    d__[i__] = g + p;
	    g = c__ * r__ - b;

/*           If eigenvectors are desired, then save rotations. */

	    if (icompz > 0) {
		work[i__] = c__;
		work[*n - 1 + i__] = s;
	    }

/* L120: */
	}

/*        If eigenvectors are desired, then apply saved rotations. */

	if (icompz > 0) {
	    mm = l - m + 1;
	    slasr_("R", "V", "F", n, &mm, &work[m], &work[*n - 1 + m], &z__[m
		    * z_dim1 + 1], ldz);
	}

	d__[l] -= p;
	e[lm1] = g;
	goto L90;

/*        Eigenvalue found. */

L130:
	d__[l] = p;

	--l;
	if (l >= lend) {
	    goto L90;
	}
	goto L140;

    }

/*     Undo scaling if necessary */

L140:
    if (iscale == 1) {
	i__1 = lendsv - lsv + 1;
	slascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
	i__1 = lendsv - lsv;
	slascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &e[lsv], n,
		info);
    } else if (iscale == 2) {
	i__1 = lendsv - lsv + 1;
	slascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
	i__1 = lendsv - lsv;
	slascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &e[lsv], n,
		info);
    }

/*
       Check for no convergence to an eigenvalue after a total
       of N*MAXIT iterations.
*/

    if (jtot < nmaxit) {
	goto L10;
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (e[i__] != 0.f) {
	    ++(*info);
	}
/* L150: */
    }
    goto L190;

/*     Order eigenvalues and eigenvectors. */

L160:
    if (icompz == 0) {

/*        Use Quick Sort */

	slasrt_("I", n, &d__[1], info);

    } else {

/*        Use Selection Sort to minimize swaps of eigenvectors */

	i__1 = *n;
	for (ii = 2; ii <= i__1; ++ii) {
	    i__ = ii - 1;
	    k = i__;
	    p = d__[i__];
	    i__2 = *n;
	    for (j = ii; j <= i__2; ++j) {
		if (d__[j] < p) {
		    k = j;
		    p = d__[j];
		}
/* L170: */
	    }
	    if (k != i__) {
		d__[k] = d__[i__];
		d__[i__] = p;
		sswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1 + 1],
			 &c__1);
	    }
/* L180: */
	}
    }

L190:
    return 0;

/*     End of SSTEQR */

} /* ssteqr_ */

/* Subroutine */ int ssterf_(integer *n, real *d__, real *e, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    static real c__;
    static integer i__, l, m;
    static real p, r__, s;
    static integer l1;
    static real bb, rt1, rt2, eps, rte;
    static integer lsv;
    static real eps2, oldc;
    static integer lend, jtot;
    extern /* Subroutine */ int slae2_(real *, real *, real *, real *, real *)
	    ;
    static real gamma, alpha, sigma, anorm;
    extern doublereal slapy2_(real *, real *);
    static integer iscale;
    static real oldgam;
    extern doublereal slamch_(char *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real safmax;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    static integer lendsv;
    static real ssfmin;
    static integer nmaxit;
    static real ssfmax;
    extern doublereal slanst_(char *, integer *, real *, real *);
    extern /* Subroutine */ int slasrt_(char *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SSTERF computes all eigenvalues of a symmetric tridiagonal matrix
    using the Pal-Walker-Kahan variant of the QL or QR algorithm.

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix.  N >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the n diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) REAL array, dimension (N-1)
            On entry, the (n-1) subdiagonal elements of the tridiagonal
            matrix.
            On exit, E has been destroyed.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  the algorithm failed to find all of the eigenvalues in
                  a total of 30*N iterations; if INFO = i, then i
                  elements of E have not converged to zero.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("SSTERF", &i__1);
	return 0;
    }
    if (*n <= 1) {
	return 0;
    }

/*     Determine the unit roundoff for this environment. */

    eps = slamch_("E");
/* Computing 2nd power */
    r__1 = eps;
    eps2 = r__1 * r__1;
    safmin = slamch_("S");
    safmax = 1.f / safmin;
    ssfmax = sqrt(safmax) / 3.f;
    ssfmin = sqrt(safmin) / eps2;

/*     Compute the eigenvalues of the tridiagonal matrix. */

    nmaxit = *n * 30;
    sigma = 0.f;
    jtot = 0;

/*
       Determine where the matrix splits and choose QL or QR iteration
       for each block, according to whether top or bottom diagonal
       element is smaller.
*/

    l1 = 1;

L10:
    if (l1 > *n) {
	goto L170;
    }
    if (l1 > 1) {
	e[l1 - 1] = 0.f;
    }
    i__1 = *n - 1;
    for (m = l1; m <= i__1; ++m) {
	if ((r__3 = e[m], dabs(r__3)) <= sqrt((r__1 = d__[m], dabs(r__1))) *
		sqrt((r__2 = d__[m + 1], dabs(r__2))) * eps) {
	    e[m] = 0.f;
	    goto L30;
	}
/* L20: */
    }
    m = *n;

L30:
    l = l1;
    lsv = l;
    lend = m;
    lendsv = lend;
    l1 = m + 1;
    if (lend == l) {
	goto L10;
    }

/*     Scale submatrix in rows and columns L to LEND */

    i__1 = lend - l + 1;
    anorm = slanst_("I", &i__1, &d__[l], &e[l]);
    iscale = 0;
    if (anorm > ssfmax) {
	iscale = 1;
	i__1 = lend - l + 1;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &e[l], n,
		info);
    } else if (anorm < ssfmin) {
	iscale = 2;
	i__1 = lend - l + 1;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	slascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &e[l], n,
		info);
    }

    i__1 = lend - 1;
    for (i__ = l; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	r__1 = e[i__];
	e[i__] = r__1 * r__1;
/* L40: */
    }

/*     Choose between QL and QR iteration */

    if ((r__1 = d__[lend], dabs(r__1)) < (r__2 = d__[l], dabs(r__2))) {
	lend = lsv;
	l = lendsv;
    }

    if (lend >= l) {

/*
          QL Iteration

          Look for small subdiagonal element.
*/

L50:
	if (l != lend) {
	    i__1 = lend - 1;
	    for (m = l; m <= i__1; ++m) {
		if ((r__2 = e[m], dabs(r__2)) <= eps2 * (r__1 = d__[m] * d__[
			m + 1], dabs(r__1))) {
		    goto L70;
		}
/* L60: */
	    }
	}
	m = lend;

L70:
	if (m < lend) {
	    e[m] = 0.f;
	}
	p = d__[l];
	if (m == l) {
	    goto L90;
	}

/*
          If remaining matrix is 2 by 2, use SLAE2 to compute its
          eigenvalues.
*/

	if (m == l + 1) {
	    rte = sqrt(e[l]);
	    slae2_(&d__[l], &rte, &d__[l + 1], &rt1, &rt2);
	    d__[l] = rt1;
	    d__[l + 1] = rt2;
	    e[l] = 0.f;
	    l += 2;
	    if (l <= lend) {
		goto L50;
	    }
	    goto L150;
	}

	if (jtot == nmaxit) {
	    goto L150;
	}
	++jtot;

/*        Form shift. */

	rte = sqrt(e[l]);
	sigma = (d__[l + 1] - p) / (rte * 2.f);
	r__ = slapy2_(&sigma, &c_b15);
	sigma = p - rte / (sigma + r_sign(&r__, &sigma));

	c__ = 1.f;
	s = 0.f;
	gamma = d__[m] - sigma;
	p = gamma * gamma;

/*        Inner loop */

	i__1 = l;
	for (i__ = m - 1; i__ >= i__1; --i__) {
	    bb = e[i__];
	    r__ = p + bb;
	    if (i__ != m - 1) {
		e[i__ + 1] = s * r__;
	    }
	    oldc = c__;
	    c__ = p / r__;
	    s = bb / r__;
	    oldgam = gamma;
	    alpha = d__[i__];
	    gamma = c__ * (alpha - sigma) - s * oldgam;
	    d__[i__ + 1] = oldgam + (alpha - gamma);
	    if (c__ != 0.f) {
		p = gamma * gamma / c__;
	    } else {
		p = oldc * bb;
	    }
/* L80: */
	}

	e[l] = s * p;
	d__[l] = sigma + gamma;
	goto L50;

/*        Eigenvalue found. */

L90:
	d__[l] = p;

	++l;
	if (l <= lend) {
	    goto L50;
	}
	goto L150;

    } else {

/*
          QR Iteration

          Look for small superdiagonal element.
*/

L100:
	i__1 = lend + 1;
	for (m = l; m >= i__1; --m) {
	    if ((r__2 = e[m - 1], dabs(r__2)) <= eps2 * (r__1 = d__[m] * d__[
		    m - 1], dabs(r__1))) {
		goto L120;
	    }
/* L110: */
	}
	m = lend;

L120:
	if (m > lend) {
	    e[m - 1] = 0.f;
	}
	p = d__[l];
	if (m == l) {
	    goto L140;
	}

/*
          If remaining matrix is 2 by 2, use SLAE2 to compute its
          eigenvalues.
*/

	if (m == l - 1) {
	    rte = sqrt(e[l - 1]);
	    slae2_(&d__[l], &rte, &d__[l - 1], &rt1, &rt2);
	    d__[l] = rt1;
	    d__[l - 1] = rt2;
	    e[l - 1] = 0.f;
	    l += -2;
	    if (l >= lend) {
		goto L100;
	    }
	    goto L150;
	}

	if (jtot == nmaxit) {
	    goto L150;
	}
	++jtot;

/*        Form shift. */

	rte = sqrt(e[l - 1]);
	sigma = (d__[l - 1] - p) / (rte * 2.f);
	r__ = slapy2_(&sigma, &c_b15);
	sigma = p - rte / (sigma + r_sign(&r__, &sigma));

	c__ = 1.f;
	s = 0.f;
	gamma = d__[m] - sigma;
	p = gamma * gamma;

/*        Inner loop */

	i__1 = l - 1;
	for (i__ = m; i__ <= i__1; ++i__) {
	    bb = e[i__];
	    r__ = p + bb;
	    if (i__ != m) {
		e[i__ - 1] = s * r__;
	    }
	    oldc = c__;
	    c__ = p / r__;
	    s = bb / r__;
	    oldgam = gamma;
	    alpha = d__[i__ + 1];
	    gamma = c__ * (alpha - sigma) - s * oldgam;
	    d__[i__] = oldgam + (alpha - gamma);
	    if (c__ != 0.f) {
		p = gamma * gamma / c__;
	    } else {
		p = oldc * bb;
	    }
/* L130: */
	}

	e[l - 1] = s * p;
	d__[l] = sigma + gamma;
	goto L100;

/*        Eigenvalue found. */

L140:
	d__[l] = p;

	--l;
	if (l >= lend) {
	    goto L100;
	}
	goto L150;

    }

/*     Undo scaling if necessary */

L150:
    if (iscale == 1) {
	i__1 = lendsv - lsv + 1;
	slascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
    }
    if (iscale == 2) {
	i__1 = lendsv - lsv + 1;
	slascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
    }

/*
       Check for no convergence to an eigenvalue after a total
       of N*MAXIT iterations.
*/

    if (jtot < nmaxit) {
	goto L10;
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (e[i__] != 0.f) {
	    ++(*info);
	}
/* L160: */
    }
    goto L180;

/*     Sort eigenvalues in increasing order. */

L170:
    slasrt_("I", n, &d__[1], info);

L180:
    return 0;

/*     End of SSTERF */

} /* ssterf_ */

/* Subroutine */ int ssyevd_(char *jobz, char *uplo, integer *n, real *a,
	integer *lda, real *w, real *work, integer *lwork, integer *iwork,
	integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real eps;
    static integer inde;
    static real anrm, rmin, rmax;
    static integer lopt;
    static real sigma;
    extern logical lsame_(char *, char *);
    static integer iinfo;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static integer lwmin, liopt;
    static logical lower, wantz;
    static integer indwk2, llwrk2, iscale;
    extern doublereal slamch_(char *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real bignum;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    static integer indtau;
    extern /* Subroutine */ int sstedc_(char *, integer *, real *, real *,
	    real *, integer *, real *, integer *, integer *, integer *,
	    integer *), slacpy_(char *, integer *, integer *, real *,
	    integer *, real *, integer *);
    static integer indwrk, liwmin;
    extern /* Subroutine */ int ssterf_(integer *, real *, real *, integer *);
    extern doublereal slansy_(char *, char *, integer *, real *, integer *,
	    real *);
    static integer llwork;
    static real smlnum;
    static logical lquery;
    extern /* Subroutine */ int sormtr_(char *, char *, char *, integer *,
	    integer *, real *, integer *, real *, real *, integer *, real *,
	    integer *, integer *), ssytrd_(char *,
	    integer *, real *, integer *, real *, real *, real *, real *,
	    integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SSYEVD computes all eigenvalues and, optionally, eigenvectors of a
    real symmetric matrix A. If eigenvectors are desired, it uses a
    divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Because of large use of BLAS of level 3, SSYEVD needs N**2 more
    workspace than SSYEVX.

    Arguments
    =========

    JOBZ    (input) CHARACTER*1
            = 'N':  Compute eigenvalues only;
            = 'V':  Compute eigenvalues and eigenvectors.

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA, N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = 'V', then if INFO = 0, A contains the
            orthonormal eigenvectors of the matrix A.
            If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
            or the upper triangle (if UPLO='U') of A, including the
            diagonal, is destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    W       (output) REAL array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    WORK    (workspace/output) REAL array,
                                           dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If N <= 1,               LWORK must be at least 1.
            If JOBZ = 'N' and N > 1, LWORK must be at least 2*N+1.
            If JOBZ = 'V' and N > 1, LWORK must be at least
                                                  1 + 6*N + 2*N**2.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If N <= 1,                LIWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LIWORK must be at least 1.
            If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the algorithm failed to converge; i
                  off-diagonal elements of an intermediate tridiagonal
                  form did not converge to zero.

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --work;
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");
    lquery = *lwork == -1 || *liwork == -1;

    *info = 0;
    if (*n <= 1) {
	liwmin = 1;
	lwmin = 1;
	lopt = lwmin;
	liopt = liwmin;
    } else {
	if (wantz) {
	    liwmin = *n * 5 + 3;
/* Computing 2nd power */
	    i__1 = *n;
	    lwmin = *n * 6 + 1 + (i__1 * i__1 << 1);
	} else {
	    liwmin = 1;
	    lwmin = (*n << 1) + 1;
	}
	lopt = lwmin;
	liopt = liwmin;
    }
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -8;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -10;
    }

    if (*info == 0) {
	work[1] = (real) lopt;
	iwork[1] = liopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYEVD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	w[1] = a[a_dim1 + 1];
	if (wantz) {
	    a[a_dim1 + 1] = 1.f;
	}
	return 0;
    }

/*     Get machine constants. */

    safmin = slamch_("Safe minimum");
    eps = slamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1.f / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = slansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    iscale = 0;
    if (anrm > 0.f && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	slascl_(uplo, &c__0, &c__0, &c_b15, &sigma, n, n, &a[a_offset], lda,
		info);
    }

/*     Call SSYTRD to reduce symmetric matrix to tridiagonal form. */

    inde = 1;
    indtau = inde + *n;
    indwrk = indtau + *n;
    llwork = *lwork - indwrk + 1;
    indwk2 = indwrk + *n * *n;
    llwrk2 = *lwork - indwk2 + 1;

    ssytrd_(uplo, n, &a[a_offset], lda, &w[1], &work[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);
    lopt = (*n << 1) + work[indwrk];

/*
       For eigenvalues only, call SSTERF.  For eigenvectors, first call
       SSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call SORMTR to multiply it by the
       Householder transformations stored in A.
*/

    if (! wantz) {
	ssterf_(n, &w[1], &work[inde], info);
    } else {
	sstedc_("I", n, &w[1], &work[inde], &work[indwrk], n, &work[indwk2], &
		llwrk2, &iwork[1], liwork, info);
	sormtr_("L", uplo, "N", n, n, &a[a_offset], lda, &work[indtau], &work[
		indwrk], n, &work[indwk2], &llwrk2, &iinfo);
	slacpy_("A", n, n, &work[indwrk], n, &a[a_offset], lda);
/*
   Computing MAX
   Computing 2nd power
*/
	i__3 = *n;
	i__1 = lopt, i__2 = *n * 6 + 1 + (i__3 * i__3 << 1);
	lopt = max(i__1,i__2);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	r__1 = 1.f / sigma;
	sscal_(n, &r__1, &w[1], &c__1);
    }

    work[1] = (real) lopt;
    iwork[1] = liopt;

    return 0;

/*     End of SSYEVD */

} /* ssyevd_ */

/* Subroutine */ int ssytd2_(char *uplo, integer *n, real *a, integer *lda,
	real *d__, real *e, real *tau, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__;
    static real taui;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    extern /* Subroutine */ int ssyr2_(char *, integer *, real *, real *,
	    integer *, real *, integer *, real *, integer *);
    static real alpha;
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int saxpy_(integer *, real *, real *, integer *,
	    real *, integer *), ssymv_(char *, integer *, real *, real *,
	    integer *, real *, integer *, real *, real *, integer *),
	    xerbla_(char *, integer *), slarfg_(integer *, real *,
	    real *, integer *, real *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    SSYTD2 reduces a real symmetric matrix A to symmetric tridiagonal
    form T by an orthogonal similarity transformation: Q' * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = 'U', the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the orthogonal
            matrix Q as a product of elementary reflectors; if UPLO
            = 'L', the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the orthogonal matrix Q as a product
            of elementary reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    D       (output) REAL array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n-1) . . . H(2) H(1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples
    with n = 5:

    if UPLO = 'U':                       if UPLO = 'L':

      (  d   e   v2  v3  v4 )              (  d                  )
      (      d   e   v3  v4 )              (  e   d              )
      (          d   e   v4 )              (  v1  e   d          )
      (              d   e  )              (  v1  v2  e   d      )
      (                  d  )              (  v1  v2  v3  e   d  )

    where d and e denote diagonal and off-diagonal elements of T, and vi
    denotes an element of the vector defining H(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tau;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYTD2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

    if (upper) {

/*        Reduce the upper triangle of A */

	for (i__ = *n - 1; i__ >= 1; --i__) {

/*
             Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(1:i-1,i+1)
*/

	    slarfg_(&i__, &a[i__ + (i__ + 1) * a_dim1], &a[(i__ + 1) * a_dim1
		    + 1], &c__1, &taui);
	    e[i__] = a[i__ + (i__ + 1) * a_dim1];

	    if (taui != 0.f) {

/*              Apply H(i) from both sides to A(1:i,1:i) */

		a[i__ + (i__ + 1) * a_dim1] = 1.f;

/*              Compute  x := tau * A * v  storing x in TAU(1:i) */

		ssymv_(uplo, &i__, &taui, &a[a_offset], lda, &a[(i__ + 1) *
			a_dim1 + 1], &c__1, &c_b29, &tau[1], &c__1)
			;

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		alpha = taui * -.5f * sdot_(&i__, &tau[1], &c__1, &a[(i__ + 1)
			 * a_dim1 + 1], &c__1);
		saxpy_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &tau[
			1], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		ssyr2_(uplo, &i__, &c_b151, &a[(i__ + 1) * a_dim1 + 1], &c__1,
			 &tau[1], &c__1, &a[a_offset], lda);

		a[i__ + (i__ + 1) * a_dim1] = e[i__];
	    }
	    d__[i__ + 1] = a[i__ + 1 + (i__ + 1) * a_dim1];
	    tau[i__] = taui;
/* L10: */
	}
	d__[1] = a[a_dim1 + 1];
    } else {

/*        Reduce the lower triangle of A */

	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*
             Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(i+2:n,i)
*/

	    i__2 = *n - i__;
/* Computing MIN */
	    i__3 = i__ + 2;
	    slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		     a_dim1], &c__1, &taui);
	    e[i__] = a[i__ + 1 + i__ * a_dim1];

	    if (taui != 0.f) {

/*              Apply H(i) from both sides to A(i+1:n,i+1:n) */

		a[i__ + 1 + i__ * a_dim1] = 1.f;

/*              Compute  x := tau * A * v  storing y in TAU(i:n-1) */

		i__2 = *n - i__;
		ssymv_(uplo, &i__2, &taui, &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b29, &tau[
			i__], &c__1);

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		i__2 = *n - i__;
		alpha = taui * -.5f * sdot_(&i__2, &tau[i__], &c__1, &a[i__ +
			1 + i__ * a_dim1], &c__1);
		i__2 = *n - i__;
		saxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
			i__], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		i__2 = *n - i__;
		ssyr2_(uplo, &i__2, &c_b151, &a[i__ + 1 + i__ * a_dim1], &
			c__1, &tau[i__], &c__1, &a[i__ + 1 + (i__ + 1) *
			a_dim1], lda);

		a[i__ + 1 + i__ * a_dim1] = e[i__];
	    }
	    d__[i__] = a[i__ + i__ * a_dim1];
	    tau[i__] = taui;
/* L20: */
	}
	d__[*n] = a[*n + *n * a_dim1];
    }

    return 0;

/*     End of SSYTD2 */

} /* ssytd2_ */

/* Subroutine */ int ssytrd_(char *uplo, integer *n, real *a, integer *lda,
	real *d__, real *e, real *tau, real *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, j, nb, kk, nx, iws;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    static logical upper;
    extern /* Subroutine */ int ssytd2_(char *, integer *, real *, integer *,
	    real *, real *, real *, integer *), ssyr2k_(char *, char *
	    , integer *, integer *, real *, real *, integer *, real *,
	    integer *, real *, real *, integer *), xerbla_(
	    char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slatrd_(char *, integer *, integer *, real *,
	    integer *, real *, real *, real *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    SSYTRD reduces a real symmetric matrix A to real symmetric
    tridiagonal form T by an orthogonal similarity transformation:
    Q**T * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = 'U', the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the orthogonal
            matrix Q as a product of elementary reflectors; if UPLO
            = 'L', the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the orthogonal matrix Q as a product
            of elementary reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    D       (output) REAL array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= 1.
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n-1) . . . H(2) H(1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples
    with n = 5:

    if UPLO = 'U':                       if UPLO = 'L':

      (  d   e   v2  v3  v4 )              (  d                  )
      (      d   e   v3  v4 )              (  e   d              )
      (          d   e   v4 )              (  v1  e   d          )
      (              d   e  )              (  v1  v2  e   d      )
      (                  d  )              (  v1  v2  v3  e   d  )

    where d and e denote diagonal and off-diagonal elements of T, and vi
    denotes an element of the vector defining H(i).

    =====================================================================


       Test the input parameters
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*lwork < 1 && ! lquery) {
	*info = -9;
    }

    if (*info == 0) {

/*        Determine the block size. */

	nb = ilaenv_(&c__1, "SSYTRD", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6,
		 (ftnlen)1);
	lwkopt = *n * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYTRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1] = 1.f;
	return 0;
    }

    nx = *n;
    iws = 1;
    if (nb > 1 && nb < *n) {

/*
          Determine when to cross over from blocked to unblocked code
          (last block is always handled by unblocked code).

   Computing MAX
*/
	i__1 = nb, i__2 = ilaenv_(&c__3, "SSYTRD", uplo, n, &c_n1, &c_n1, &
		c_n1, (ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);
	if (nx < *n) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*
                Not enough workspace to use optimal NB:  determine the
                minimum value of NB, and reduce NB or force use of
                unblocked code by setting NX = N.

   Computing MAX
*/
		i__1 = *lwork / ldwork;
		nb = max(i__1,1);
		nbmin = ilaenv_(&c__2, "SSYTRD", uplo, n, &c_n1, &c_n1, &c_n1,
			 (ftnlen)6, (ftnlen)1);
		if (nb < nbmin) {
		    nx = *n;
		}
	    }
	} else {
	    nx = *n;
	}
    } else {
	nb = 1;
    }

    if (upper) {

/*
          Reduce the upper triangle of A.
          Columns 1:kk are handled by the unblocked method.
*/

	kk = *n - (*n - nx + nb - 1) / nb * nb;
	i__1 = kk + 1;
	i__2 = -nb;
	for (i__ = *n - nb + 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
		i__2) {

/*
             Reduce columns i:i+nb-1 to tridiagonal form and form the
             matrix W which is needed to update the unreduced part of
             the matrix
*/

	    i__3 = i__ + nb - 1;
	    slatrd_(uplo, &i__3, &nb, &a[a_offset], lda, &e[1], &tau[1], &
		    work[1], &ldwork);

/*
             Update the unreduced submatrix A(1:i-1,1:i-1), using an
             update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = i__ - 1;
	    ssyr2k_(uplo, "No transpose", &i__3, &nb, &c_b151, &a[i__ *
		    a_dim1 + 1], lda, &work[1], &ldwork, &c_b15, &a[a_offset],
		     lda);

/*
             Copy superdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j - 1 + j * a_dim1] = e[j - 1];
		d__[j] = a[j + j * a_dim1];
/* L10: */
	    }
/* L20: */
	}

/*        Use unblocked code to reduce the last or only block */

	ssytd2_(uplo, &kk, &a[a_offset], lda, &d__[1], &e[1], &tau[1], &iinfo);
    } else {

/*        Reduce the lower triangle of A */

	i__2 = *n - nx;
	i__1 = nb;
	for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {

/*
             Reduce columns i:i+nb-1 to tridiagonal form and form the
             matrix W which is needed to update the unreduced part of
             the matrix
*/

	    i__3 = *n - i__ + 1;
	    slatrd_(uplo, &i__3, &nb, &a[i__ + i__ * a_dim1], lda, &e[i__], &
		    tau[i__], &work[1], &ldwork);

/*
             Update the unreduced submatrix A(i+ib:n,i+ib:n), using
             an update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = *n - i__ - nb + 1;
	    ssyr2k_(uplo, "No transpose", &i__3, &nb, &c_b151, &a[i__ + nb +
		    i__ * a_dim1], lda, &work[nb + 1], &ldwork, &c_b15, &a[
		    i__ + nb + (i__ + nb) * a_dim1], lda);

/*
             Copy subdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + 1 + j * a_dim1] = e[j];
		d__[j] = a[j + j * a_dim1];
/* L30: */
	    }
/* L40: */
	}

/*        Use unblocked code to reduce the last or only block */

	i__1 = *n - i__ + 1;
	ssytd2_(uplo, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__],
		&tau[i__], &iinfo);
    }

    work[1] = (real) lwkopt;
    return 0;

/*     End of SSYTRD */

} /* ssytrd_ */

/* Subroutine */ int strevc_(char *side, char *howmny, logical *select,
	integer *n, real *t, integer *ldt, real *vl, integer *ldvl, real *vr,
	integer *ldvr, integer *mm, integer *m, real *work, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, k;
    static real x[4]	/* was [2][2] */;
    static integer j1, j2, n2, ii, ki, ip, is;
    static real wi, wr, rec, ulp, beta, emax;
    static logical pair, allv;
    static integer ierr;
    static real unfl, ovfl, smin;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    static logical over;
    static real vmax;
    static integer jnxt;
    static real scale;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static real remax;
    static logical leftv;
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *,
	    real *, integer *, real *, integer *, real *, real *, integer *);
    static logical bothv;
    static real vcrit;
    static logical somev;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    static real xnorm;
    extern /* Subroutine */ int saxpy_(integer *, real *, real *, integer *,
	    real *, integer *), slaln2_(logical *, integer *, integer *, real
	    *, real *, real *, integer *, real *, real *, real *, integer *,
	    real *, real *, real *, integer *, real *, real *, integer *),
	    slabad_(real *, real *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real bignum;
    extern integer isamax_(integer *, real *, integer *);
    static logical rightv;
    static real smlnum;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    STREVC computes some or all of the right and/or left eigenvectors of
    a real upper quasi-triangular matrix T.

    The right eigenvector x and the left eigenvector y of T corresponding
    to an eigenvalue w are defined by:

                 T*x = w*x,     y'*T = w*y'

    where y' denotes the conjugate transpose of the vector y.

    If all eigenvectors are requested, the routine may either return the
    matrices X and/or Y of right or left eigenvectors of T, or the
    products Q*X and/or Q*Y, where Q is an input orthogonal
    matrix. If T was obtained from the real-Schur factorization of an
    original matrix A = Q*T*Q', then Q*X and Q*Y are the matrices of
    right or left eigenvectors of A.

    T must be in Schur canonical form (as returned by SHSEQR), that is,
    block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
    2-by-2 diagonal block has its diagonal elements equal and its
    off-diagonal elements of opposite sign.  Corresponding to each 2-by-2
    diagonal block is a complex conjugate pair of eigenvalues and
    eigenvectors; only one eigenvector of the pair is computed, namely
    the one corresponding to the eigenvalue with positive imaginary part.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'R':  compute right eigenvectors only;
            = 'L':  compute left eigenvectors only;
            = 'B':  compute both right and left eigenvectors.

    HOWMNY  (input) CHARACTER*1
            = 'A':  compute all right and/or left eigenvectors;
            = 'B':  compute all right and/or left eigenvectors,
                    and backtransform them using the input matrices
                    supplied in VR and/or VL;
            = 'S':  compute selected right and/or left eigenvectors,
                    specified by the logical array SELECT.

    SELECT  (input/output) LOGICAL array, dimension (N)
            If HOWMNY = 'S', SELECT specifies the eigenvectors to be
            computed.
            If HOWMNY = 'A' or 'B', SELECT is not referenced.
            To select the real eigenvector corresponding to a real
            eigenvalue w(j), SELECT(j) must be set to .TRUE..  To select
            the complex eigenvector corresponding to a complex conjugate
            pair w(j) and w(j+1), either SELECT(j) or SELECT(j+1) must be
            set to .TRUE.; then on exit SELECT(j) is .TRUE. and
            SELECT(j+1) is .FALSE..

    N       (input) INTEGER
            The order of the matrix T. N >= 0.

    T       (input) REAL array, dimension (LDT,N)
            The upper quasi-triangular matrix T in Schur canonical form.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= max(1,N).

    VL      (input/output) REAL array, dimension (LDVL,MM)
            On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
            contain an N-by-N matrix Q (usually the orthogonal matrix Q
            of Schur vectors returned by SHSEQR).
            On exit, if SIDE = 'L' or 'B', VL contains:
            if HOWMNY = 'A', the matrix Y of left eigenvectors of T;
                             VL has the same quasi-lower triangular form
                             as T'. If T(i,i) is a real eigenvalue, then
                             the i-th column VL(i) of VL  is its
                             corresponding eigenvector. If T(i:i+1,i:i+1)
                             is a 2-by-2 block whose eigenvalues are
                             complex-conjugate eigenvalues of T, then
                             VL(i)+sqrt(-1)*VL(i+1) is the complex
                             eigenvector corresponding to the eigenvalue
                             with positive real part.
            if HOWMNY = 'B', the matrix Q*Y;
            if HOWMNY = 'S', the left eigenvectors of T specified by
                             SELECT, stored consecutively in the columns
                             of VL, in the same order as their
                             eigenvalues.
            A complex eigenvector corresponding to a complex eigenvalue
            is stored in two consecutive columns, the first holding the
            real part, and the second the imaginary part.
            If SIDE = 'R', VL is not referenced.

    LDVL    (input) INTEGER
            The leading dimension of the array VL.  LDVL >= max(1,N) if
            SIDE = 'L' or 'B'; LDVL >= 1 otherwise.

    VR      (input/output) REAL array, dimension (LDVR,MM)
            On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
            contain an N-by-N matrix Q (usually the orthogonal matrix Q
            of Schur vectors returned by SHSEQR).
            On exit, if SIDE = 'R' or 'B', VR contains:
            if HOWMNY = 'A', the matrix X of right eigenvectors of T;
                             VR has the same quasi-upper triangular form
                             as T. If T(i,i) is a real eigenvalue, then
                             the i-th column VR(i) of VR  is its
                             corresponding eigenvector. If T(i:i+1,i:i+1)
                             is a 2-by-2 block whose eigenvalues are
                             complex-conjugate eigenvalues of T, then
                             VR(i)+sqrt(-1)*VR(i+1) is the complex
                             eigenvector corresponding to the eigenvalue
                             with positive real part.
            if HOWMNY = 'B', the matrix Q*X;
            if HOWMNY = 'S', the right eigenvectors of T specified by
                             SELECT, stored consecutively in the columns
                             of VR, in the same order as their
                             eigenvalues.
            A complex eigenvector corresponding to a complex eigenvalue
            is stored in two consecutive columns, the first holding the
            real part and the second the imaginary part.
            If SIDE = 'L', VR is not referenced.

    LDVR    (input) INTEGER
            The leading dimension of the array VR.  LDVR >= max(1,N) if
            SIDE = 'R' or 'B'; LDVR >= 1 otherwise.

    MM      (input) INTEGER
            The number of columns in the arrays VL and/or VR. MM >= M.

    M       (output) INTEGER
            The number of columns in the arrays VL and/or VR actually
            used to store the eigenvectors.
            If HOWMNY = 'A' or 'B', M is set to N.
            Each selected real eigenvector occupies one column and each
            selected complex eigenvector occupies two columns.

    WORK    (workspace) REAL array, dimension (3*N)

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The algorithm used in this program is basically backward (forward)
    substitution, with scaling to make the the code robust against
    possible overflow.

    Each eigenvector is normalized so that the element of largest
    magnitude has magnitude 1; here the magnitude of a complex number
    (x,y) is taken to be |x| + |y|.

    =====================================================================


       Decode and test the input parameters
*/

    /* Parameter adjustments */
    --select;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;

    /* Function Body */
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;

    allv = lsame_(howmny, "A");
    over = lsame_(howmny, "B");
    somev = lsame_(howmny, "S");

    *info = 0;
    if (! rightv && ! leftv) {
	*info = -1;
    } else if (! allv && ! over && ! somev) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ldt < max(1,*n)) {
	*info = -6;
    } else if (*ldvl < 1 || leftv && *ldvl < *n) {
	*info = -8;
    } else if (*ldvr < 1 || rightv && *ldvr < *n) {
	*info = -10;
    } else {

/*
          Set M to the number of columns required to store the selected
          eigenvectors, standardize the array SELECT if necessary, and
          test MM.
*/

	if (somev) {
	    *m = 0;
	    pair = FALSE_;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (pair) {
		    pair = FALSE_;
		    select[j] = FALSE_;
		} else {
		    if (j < *n) {
			if (t[j + 1 + j * t_dim1] == 0.f) {
			    if (select[j]) {
				++(*m);
			    }
			} else {
			    pair = TRUE_;
			    if (select[j] || select[j + 1]) {
				select[j] = TRUE_;
				*m += 2;
			    }
			}
		    } else {
			if (select[*n]) {
			    ++(*m);
			}
		    }
		}
/* L10: */
	    }
	} else {
	    *m = *n;
	}

	if (*mm < *m) {
	    *info = -11;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STREVC", &i__1);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     Set the constants to control overflow. */

    unfl = slamch_("Safe minimum");
    ovfl = 1.f / unfl;
    slabad_(&unfl, &ovfl);
    ulp = slamch_("Precision");
    smlnum = unfl * (*n / ulp);
    bignum = (1.f - ulp) / smlnum;

/*
       Compute 1-norm of each column of strictly upper triangular
       part of T to control overflow in triangular solver.
*/

    work[1] = 0.f;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	work[j] = 0.f;
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[j] += (r__1 = t[i__ + j * t_dim1], dabs(r__1));
/* L20: */
	}
/* L30: */
    }

/*
       Index IP is used to specify the real or complex eigenvalue:
         IP = 0, real eigenvalue,
              1, first of conjugate complex pair: (wr,wi)
             -1, second of conjugate complex pair: (wr,wi)
*/

    n2 = *n << 1;

    if (rightv) {

/*        Compute right eigenvectors. */

	ip = 0;
	is = *m;
	for (ki = *n; ki >= 1; --ki) {

	    if (ip == 1) {
		goto L130;
	    }
	    if (ki == 1) {
		goto L40;
	    }
	    if (t[ki + (ki - 1) * t_dim1] == 0.f) {
		goto L40;
	    }
	    ip = -1;

L40:
	    if (somev) {
		if (ip == 0) {
		    if (! select[ki]) {
			goto L130;
		    }
		} else {
		    if (! select[ki - 1]) {
			goto L130;
		    }
		}
	    }

/*           Compute the KI-th eigenvalue (WR,WI). */

	    wr = t[ki + ki * t_dim1];
	    wi = 0.f;
	    if (ip != 0) {
		wi = sqrt((r__1 = t[ki + (ki - 1) * t_dim1], dabs(r__1))) *
			sqrt((r__2 = t[ki - 1 + ki * t_dim1], dabs(r__2)));
	    }
/* Computing MAX */
	    r__1 = ulp * (dabs(wr) + dabs(wi));
	    smin = dmax(r__1,smlnum);

	    if (ip == 0) {

/*              Real right eigenvector */

		work[ki + *n] = 1.f;

/*              Form right-hand side */

		i__1 = ki - 1;
		for (k = 1; k <= i__1; ++k) {
		    work[k + *n] = -t[k + ki * t_dim1];
/* L50: */
		}

/*
                Solve the upper quasi-triangular system:
                   (T(1:KI-1,1:KI-1) - WR)*X = SCALE*WORK.
*/

		jnxt = ki - 1;
		for (j = ki - 1; j >= 1; --j) {
		    if (j > jnxt) {
			goto L60;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j - 1;
		    if (j > 1) {
			if (t[j + (j - 1) * t_dim1] != 0.f) {
			    j1 = j - 1;
			    jnxt = j - 2;
			}
		    }

		    if (j1 == j2) {

/*                    1-by-1 diagonal block */

			slaln2_(&c_false, &c__1, &c__1, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &c_b29, x, &c__2, &scale, &xnorm,
				&ierr);

/*
                      Scale X(1,1) to avoid overflow when updating
                      the right-hand side.
*/

			if (xnorm > 1.f) {
			    if (work[j] > bignum / xnorm) {
				x[0] /= xnorm;
				scale /= xnorm;
			    }
			}

/*                    Scale if necessary */

			if (scale != 1.f) {
			    sscal_(&ki, &scale, &work[*n + 1], &c__1);
			}
			work[j + *n] = x[0];

/*                    Update right-hand side */

			i__1 = j - 1;
			r__1 = -x[0];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				*n + 1], &c__1);

		    } else {

/*                    2-by-2 diagonal block */

			slaln2_(&c_false, &c__2, &c__1, &smin, &c_b15, &t[j -
				1 + (j - 1) * t_dim1], ldt, &c_b15, &c_b15, &
				work[j - 1 + *n], n, &wr, &c_b29, x, &c__2, &
				scale, &xnorm, &ierr);

/*
                      Scale X(1,1) and X(2,1) to avoid overflow when
                      updating the right-hand side.
*/

			if (xnorm > 1.f) {
/* Computing MAX */
			    r__1 = work[j - 1], r__2 = work[j];
			    beta = dmax(r__1,r__2);
			    if (beta > bignum / xnorm) {
				x[0] /= xnorm;
				x[1] /= xnorm;
				scale /= xnorm;
			    }
			}

/*                    Scale if necessary */

			if (scale != 1.f) {
			    sscal_(&ki, &scale, &work[*n + 1], &c__1);
			}
			work[j - 1 + *n] = x[0];
			work[j + *n] = x[1];

/*                    Update right-hand side */

			i__1 = j - 2;
			r__1 = -x[0];
			saxpy_(&i__1, &r__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[*n + 1], &c__1);
			i__1 = j - 2;
			r__1 = -x[1];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				*n + 1], &c__1);
		    }
L60:
		    ;
		}

/*              Copy the vector x or Q*x to VR and normalize. */

		if (! over) {
		    scopy_(&ki, &work[*n + 1], &c__1, &vr[is * vr_dim1 + 1], &
			    c__1);

		    ii = isamax_(&ki, &vr[is * vr_dim1 + 1], &c__1);
		    remax = 1.f / (r__1 = vr[ii + is * vr_dim1], dabs(r__1));
		    sscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);

		    i__1 = *n;
		    for (k = ki + 1; k <= i__1; ++k) {
			vr[k + is * vr_dim1] = 0.f;
/* L70: */
		    }
		} else {
		    if (ki > 1) {
			i__1 = ki - 1;
			sgemv_("N", n, &i__1, &c_b15, &vr[vr_offset], ldvr, &
				work[*n + 1], &c__1, &work[ki + *n], &vr[ki *
				vr_dim1 + 1], &c__1);
		    }

		    ii = isamax_(n, &vr[ki * vr_dim1 + 1], &c__1);
		    remax = 1.f / (r__1 = vr[ii + ki * vr_dim1], dabs(r__1));
		    sscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
		}

	    } else {

/*
                Complex right eigenvector.

                Initial solve
                  [ (T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I* WI)]*X = 0.
                  [ (T(KI,KI-1)   T(KI,KI)   )               ]
*/

		if ((r__1 = t[ki - 1 + ki * t_dim1], dabs(r__1)) >= (r__2 = t[
			ki + (ki - 1) * t_dim1], dabs(r__2))) {
		    work[ki - 1 + *n] = 1.f;
		    work[ki + n2] = wi / t[ki - 1 + ki * t_dim1];
		} else {
		    work[ki - 1 + *n] = -wi / t[ki + (ki - 1) * t_dim1];
		    work[ki + n2] = 1.f;
		}
		work[ki + *n] = 0.f;
		work[ki - 1 + n2] = 0.f;

/*              Form right-hand side */

		i__1 = ki - 2;
		for (k = 1; k <= i__1; ++k) {
		    work[k + *n] = -work[ki - 1 + *n] * t[k + (ki - 1) *
			    t_dim1];
		    work[k + n2] = -work[ki + n2] * t[k + ki * t_dim1];
/* L80: */
		}

/*
                Solve upper quasi-triangular system:
                (T(1:KI-2,1:KI-2) - (WR+i*WI))*X = SCALE*(WORK+i*WORK2)
*/

		jnxt = ki - 2;
		for (j = ki - 2; j >= 1; --j) {
		    if (j > jnxt) {
			goto L90;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j - 1;
		    if (j > 1) {
			if (t[j + (j - 1) * t_dim1] != 0.f) {
			    j1 = j - 1;
			    jnxt = j - 2;
			}
		    }

		    if (j1 == j2) {

/*                    1-by-1 diagonal block */

			slaln2_(&c_false, &c__1, &c__2, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &wi, x, &c__2, &scale, &xnorm, &
				ierr);

/*
                      Scale X(1,1) and X(1,2) to avoid overflow when
                      updating the right-hand side.
*/

			if (xnorm > 1.f) {
			    if (work[j] > bignum / xnorm) {
				x[0] /= xnorm;
				x[2] /= xnorm;
				scale /= xnorm;
			    }
			}

/*                    Scale if necessary */

			if (scale != 1.f) {
			    sscal_(&ki, &scale, &work[*n + 1], &c__1);
			    sscal_(&ki, &scale, &work[n2 + 1], &c__1);
			}
			work[j + *n] = x[0];
			work[j + n2] = x[2];

/*                    Update the right-hand side */

			i__1 = j - 1;
			r__1 = -x[0];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				*n + 1], &c__1);
			i__1 = j - 1;
			r__1 = -x[2];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				n2 + 1], &c__1);

		    } else {

/*                    2-by-2 diagonal block */

			slaln2_(&c_false, &c__2, &c__2, &smin, &c_b15, &t[j -
				1 + (j - 1) * t_dim1], ldt, &c_b15, &c_b15, &
				work[j - 1 + *n], n, &wr, &wi, x, &c__2, &
				scale, &xnorm, &ierr);

/*
                      Scale X to avoid overflow when updating
                      the right-hand side.
*/

			if (xnorm > 1.f) {
/* Computing MAX */
			    r__1 = work[j - 1], r__2 = work[j];
			    beta = dmax(r__1,r__2);
			    if (beta > bignum / xnorm) {
				rec = 1.f / xnorm;
				x[0] *= rec;
				x[2] *= rec;
				x[1] *= rec;
				x[3] *= rec;
				scale *= rec;
			    }
			}

/*                    Scale if necessary */

			if (scale != 1.f) {
			    sscal_(&ki, &scale, &work[*n + 1], &c__1);
			    sscal_(&ki, &scale, &work[n2 + 1], &c__1);
			}
			work[j - 1 + *n] = x[0];
			work[j + *n] = x[1];
			work[j - 1 + n2] = x[2];
			work[j + n2] = x[3];

/*                    Update the right-hand side */

			i__1 = j - 2;
			r__1 = -x[0];
			saxpy_(&i__1, &r__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[*n + 1], &c__1);
			i__1 = j - 2;
			r__1 = -x[1];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				*n + 1], &c__1);
			i__1 = j - 2;
			r__1 = -x[2];
			saxpy_(&i__1, &r__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[n2 + 1], &c__1);
			i__1 = j - 2;
			r__1 = -x[3];
			saxpy_(&i__1, &r__1, &t[j * t_dim1 + 1], &c__1, &work[
				n2 + 1], &c__1);
		    }
L90:
		    ;
		}

/*              Copy the vector x or Q*x to VR and normalize. */

		if (! over) {
		    scopy_(&ki, &work[*n + 1], &c__1, &vr[(is - 1) * vr_dim1
			    + 1], &c__1);
		    scopy_(&ki, &work[n2 + 1], &c__1, &vr[is * vr_dim1 + 1], &
			    c__1);

		    emax = 0.f;
		    i__1 = ki;
		    for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			r__3 = emax, r__4 = (r__1 = vr[k + (is - 1) * vr_dim1]
				, dabs(r__1)) + (r__2 = vr[k + is * vr_dim1],
				dabs(r__2));
			emax = dmax(r__3,r__4);
/* L100: */
		    }

		    remax = 1.f / emax;
		    sscal_(&ki, &remax, &vr[(is - 1) * vr_dim1 + 1], &c__1);
		    sscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);

		    i__1 = *n;
		    for (k = ki + 1; k <= i__1; ++k) {
			vr[k + (is - 1) * vr_dim1] = 0.f;
			vr[k + is * vr_dim1] = 0.f;
/* L110: */
		    }

		} else {

		    if (ki > 2) {
			i__1 = ki - 2;
			sgemv_("N", n, &i__1, &c_b15, &vr[vr_offset], ldvr, &
				work[*n + 1], &c__1, &work[ki - 1 + *n], &vr[(
				ki - 1) * vr_dim1 + 1], &c__1);
			i__1 = ki - 2;
			sgemv_("N", n, &i__1, &c_b15, &vr[vr_offset], ldvr, &
				work[n2 + 1], &c__1, &work[ki + n2], &vr[ki *
				vr_dim1 + 1], &c__1);
		    } else {
			sscal_(n, &work[ki - 1 + *n], &vr[(ki - 1) * vr_dim1
				+ 1], &c__1);
			sscal_(n, &work[ki + n2], &vr[ki * vr_dim1 + 1], &
				c__1);
		    }

		    emax = 0.f;
		    i__1 = *n;
		    for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			r__3 = emax, r__4 = (r__1 = vr[k + (ki - 1) * vr_dim1]
				, dabs(r__1)) + (r__2 = vr[k + ki * vr_dim1],
				dabs(r__2));
			emax = dmax(r__3,r__4);
/* L120: */
		    }
		    remax = 1.f / emax;
		    sscal_(n, &remax, &vr[(ki - 1) * vr_dim1 + 1], &c__1);
		    sscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
		}
	    }

	    --is;
	    if (ip != 0) {
		--is;
	    }
L130:
	    if (ip == 1) {
		ip = 0;
	    }
	    if (ip == -1) {
		ip = 1;
	    }
/* L140: */
	}
    }

    if (leftv) {

/*        Compute left eigenvectors. */

	ip = 0;
	is = 1;
	i__1 = *n;
	for (ki = 1; ki <= i__1; ++ki) {

	    if (ip == -1) {
		goto L250;
	    }
	    if (ki == *n) {
		goto L150;
	    }
	    if (t[ki + 1 + ki * t_dim1] == 0.f) {
		goto L150;
	    }
	    ip = 1;

L150:
	    if (somev) {
		if (! select[ki]) {
		    goto L250;
		}
	    }

/*           Compute the KI-th eigenvalue (WR,WI). */

	    wr = t[ki + ki * t_dim1];
	    wi = 0.f;
	    if (ip != 0) {
		wi = sqrt((r__1 = t[ki + (ki + 1) * t_dim1], dabs(r__1))) *
			sqrt((r__2 = t[ki + 1 + ki * t_dim1], dabs(r__2)));
	    }
/* Computing MAX */
	    r__1 = ulp * (dabs(wr) + dabs(wi));
	    smin = dmax(r__1,smlnum);

	    if (ip == 0) {

/*              Real left eigenvector. */

		work[ki + *n] = 1.f;

/*              Form right-hand side */

		i__2 = *n;
		for (k = ki + 1; k <= i__2; ++k) {
		    work[k + *n] = -t[ki + k * t_dim1];
/* L160: */
		}

/*
                Solve the quasi-triangular system:
                   (T(KI+1:N,KI+1:N) - WR)'*X = SCALE*WORK
*/

		vmax = 1.f;
		vcrit = bignum;

		jnxt = ki + 1;
		i__2 = *n;
		for (j = ki + 1; j <= i__2; ++j) {
		    if (j < jnxt) {
			goto L170;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j + 1;
		    if (j < *n) {
			if (t[j + 1 + j * t_dim1] != 0.f) {
			    j2 = j + 1;
			    jnxt = j + 2;
			}
		    }

		    if (j1 == j2) {

/*
                      1-by-1 diagonal block

                      Scale if necessary to avoid overflow when forming
                      the right-hand side.
*/

			if (work[j] > vcrit) {
			    rec = 1.f / vmax;
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + *n], &c__1);
			    vmax = 1.f;
			    vcrit = bignum;
			}

			i__3 = j - ki - 1;
			work[j + *n] -= sdot_(&i__3, &t[ki + 1 + j * t_dim1],
				&c__1, &work[ki + 1 + *n], &c__1);

/*                    Solve (T(J,J)-WR)'*X = WORK */

			slaln2_(&c_false, &c__1, &c__1, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &c_b29, x, &c__2, &scale, &xnorm,
				&ierr);

/*                    Scale if necessary */

			if (scale != 1.f) {
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + *n], &c__1);
			}
			work[j + *n] = x[0];
/* Computing MAX */
			r__2 = (r__1 = work[j + *n], dabs(r__1));
			vmax = dmax(r__2,vmax);
			vcrit = bignum / vmax;

		    } else {

/*
                      2-by-2 diagonal block

                      Scale if necessary to avoid overflow when forming
                      the right-hand side.

   Computing MAX
*/
			r__1 = work[j], r__2 = work[j + 1];
			beta = dmax(r__1,r__2);
			if (beta > vcrit) {
			    rec = 1.f / vmax;
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + *n], &c__1);
			    vmax = 1.f;
			    vcrit = bignum;
			}

			i__3 = j - ki - 1;
			work[j + *n] -= sdot_(&i__3, &t[ki + 1 + j * t_dim1],
				&c__1, &work[ki + 1 + *n], &c__1);

			i__3 = j - ki - 1;
			work[j + 1 + *n] -= sdot_(&i__3, &t[ki + 1 + (j + 1) *
				 t_dim1], &c__1, &work[ki + 1 + *n], &c__1);

/*
                      Solve
                        [T(J,J)-WR   T(J,J+1)     ]'* X = SCALE*( WORK1 )
                        [T(J+1,J)    T(J+1,J+1)-WR]             ( WORK2 )
*/

			slaln2_(&c_true, &c__2, &c__1, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &c_b29, x, &c__2, &scale, &xnorm,
				&ierr);

/*                    Scale if necessary */

			if (scale != 1.f) {
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + *n], &c__1);
			}
			work[j + *n] = x[0];
			work[j + 1 + *n] = x[1];

/* Computing MAX */
			r__3 = (r__1 = work[j + *n], dabs(r__1)), r__4 = (
				r__2 = work[j + 1 + *n], dabs(r__2)), r__3 =
				max(r__3,r__4);
			vmax = dmax(r__3,vmax);
			vcrit = bignum / vmax;

		    }
L170:
		    ;
		}

/*              Copy the vector x or Q*x to VL and normalize. */

		if (! over) {
		    i__2 = *n - ki + 1;
		    scopy_(&i__2, &work[ki + *n], &c__1, &vl[ki + is *
			    vl_dim1], &c__1);

		    i__2 = *n - ki + 1;
		    ii = isamax_(&i__2, &vl[ki + is * vl_dim1], &c__1) + ki -
			    1;
		    remax = 1.f / (r__1 = vl[ii + is * vl_dim1], dabs(r__1));
		    i__2 = *n - ki + 1;
		    sscal_(&i__2, &remax, &vl[ki + is * vl_dim1], &c__1);

		    i__2 = ki - 1;
		    for (k = 1; k <= i__2; ++k) {
			vl[k + is * vl_dim1] = 0.f;
/* L180: */
		    }

		} else {

		    if (ki < *n) {
			i__2 = *n - ki;
			sgemv_("N", n, &i__2, &c_b15, &vl[(ki + 1) * vl_dim1
				+ 1], ldvl, &work[ki + 1 + *n], &c__1, &work[
				ki + *n], &vl[ki * vl_dim1 + 1], &c__1);
		    }

		    ii = isamax_(n, &vl[ki * vl_dim1 + 1], &c__1);
		    remax = 1.f / (r__1 = vl[ii + ki * vl_dim1], dabs(r__1));
		    sscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);

		}

	    } else {

/*
                Complex left eigenvector.

                 Initial solve:
                   ((T(KI,KI)    T(KI,KI+1) )' - (WR - I* WI))*X = 0.
                   ((T(KI+1,KI) T(KI+1,KI+1))                )
*/

		if ((r__1 = t[ki + (ki + 1) * t_dim1], dabs(r__1)) >= (r__2 =
			t[ki + 1 + ki * t_dim1], dabs(r__2))) {
		    work[ki + *n] = wi / t[ki + (ki + 1) * t_dim1];
		    work[ki + 1 + n2] = 1.f;
		} else {
		    work[ki + *n] = 1.f;
		    work[ki + 1 + n2] = -wi / t[ki + 1 + ki * t_dim1];
		}
		work[ki + 1 + *n] = 0.f;
		work[ki + n2] = 0.f;

/*              Form right-hand side */

		i__2 = *n;
		for (k = ki + 2; k <= i__2; ++k) {
		    work[k + *n] = -work[ki + *n] * t[ki + k * t_dim1];
		    work[k + n2] = -work[ki + 1 + n2] * t[ki + 1 + k * t_dim1]
			    ;
/* L190: */
		}

/*
                Solve complex quasi-triangular system:
                ( T(KI+2,N:KI+2,N) - (WR-i*WI) )*X = WORK1+i*WORK2
*/

		vmax = 1.f;
		vcrit = bignum;

		jnxt = ki + 2;
		i__2 = *n;
		for (j = ki + 2; j <= i__2; ++j) {
		    if (j < jnxt) {
			goto L200;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j + 1;
		    if (j < *n) {
			if (t[j + 1 + j * t_dim1] != 0.f) {
			    j2 = j + 1;
			    jnxt = j + 2;
			}
		    }

		    if (j1 == j2) {

/*
                      1-by-1 diagonal block

                      Scale if necessary to avoid overflow when
                      forming the right-hand side elements.
*/

			if (work[j] > vcrit) {
			    rec = 1.f / vmax;
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + *n], &c__1);
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + n2], &c__1);
			    vmax = 1.f;
			    vcrit = bignum;
			}

			i__3 = j - ki - 2;
			work[j + *n] -= sdot_(&i__3, &t[ki + 2 + j * t_dim1],
				&c__1, &work[ki + 2 + *n], &c__1);
			i__3 = j - ki - 2;
			work[j + n2] -= sdot_(&i__3, &t[ki + 2 + j * t_dim1],
				&c__1, &work[ki + 2 + n2], &c__1);

/*                    Solve (T(J,J)-(WR-i*WI))*(X11+i*X12)= WK+I*WK2 */

			r__1 = -wi;
			slaln2_(&c_false, &c__1, &c__2, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &r__1, x, &c__2, &scale, &xnorm, &
				ierr);

/*                    Scale if necessary */

			if (scale != 1.f) {
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + *n], &c__1);
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + n2], &c__1);
			}
			work[j + *n] = x[0];
			work[j + n2] = x[2];
/* Computing MAX */
			r__3 = (r__1 = work[j + *n], dabs(r__1)), r__4 = (
				r__2 = work[j + n2], dabs(r__2)), r__3 = max(
				r__3,r__4);
			vmax = dmax(r__3,vmax);
			vcrit = bignum / vmax;

		    } else {

/*
                      2-by-2 diagonal block

                      Scale if necessary to avoid overflow when forming
                      the right-hand side elements.

   Computing MAX
*/
			r__1 = work[j], r__2 = work[j + 1];
			beta = dmax(r__1,r__2);
			if (beta > vcrit) {
			    rec = 1.f / vmax;
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + *n], &c__1);
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &rec, &work[ki + n2], &c__1);
			    vmax = 1.f;
			    vcrit = bignum;
			}

			i__3 = j - ki - 2;
			work[j + *n] -= sdot_(&i__3, &t[ki + 2 + j * t_dim1],
				&c__1, &work[ki + 2 + *n], &c__1);

			i__3 = j - ki - 2;
			work[j + n2] -= sdot_(&i__3, &t[ki + 2 + j * t_dim1],
				&c__1, &work[ki + 2 + n2], &c__1);

			i__3 = j - ki - 2;
			work[j + 1 + *n] -= sdot_(&i__3, &t[ki + 2 + (j + 1) *
				 t_dim1], &c__1, &work[ki + 2 + *n], &c__1);

			i__3 = j - ki - 2;
			work[j + 1 + n2] -= sdot_(&i__3, &t[ki + 2 + (j + 1) *
				 t_dim1], &c__1, &work[ki + 2 + n2], &c__1);

/*
                      Solve 2-by-2 complex linear equation
                        ([T(j,j)   T(j,j+1)  ]'-(wr-i*wi)*I)*X = SCALE*B
                        ([T(j+1,j) T(j+1,j+1)]             )
*/

			r__1 = -wi;
			slaln2_(&c_true, &c__2, &c__2, &smin, &c_b15, &t[j +
				j * t_dim1], ldt, &c_b15, &c_b15, &work[j + *
				n], n, &wr, &r__1, x, &c__2, &scale, &xnorm, &
				ierr);

/*                    Scale if necessary */

			if (scale != 1.f) {
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + *n], &c__1);
			    i__3 = *n - ki + 1;
			    sscal_(&i__3, &scale, &work[ki + n2], &c__1);
			}
			work[j + *n] = x[0];
			work[j + n2] = x[2];
			work[j + 1 + *n] = x[1];
			work[j + 1 + n2] = x[3];
/* Computing MAX */
			r__1 = dabs(x[0]), r__2 = dabs(x[2]), r__1 = max(r__1,
				r__2), r__2 = dabs(x[1]), r__1 = max(r__1,
				r__2), r__2 = dabs(x[3]), r__1 = max(r__1,
				r__2);
			vmax = dmax(r__1,vmax);
			vcrit = bignum / vmax;

		    }
L200:
		    ;
		}

/*
                Copy the vector x or Q*x to VL and normalize.

   L210:
*/
		if (! over) {
		    i__2 = *n - ki + 1;
		    scopy_(&i__2, &work[ki + *n], &c__1, &vl[ki + is *
			    vl_dim1], &c__1);
		    i__2 = *n - ki + 1;
		    scopy_(&i__2, &work[ki + n2], &c__1, &vl[ki + (is + 1) *
			    vl_dim1], &c__1);

		    emax = 0.f;
		    i__2 = *n;
		    for (k = ki; k <= i__2; ++k) {
/* Computing MAX */
			r__3 = emax, r__4 = (r__1 = vl[k + is * vl_dim1],
				dabs(r__1)) + (r__2 = vl[k + (is + 1) *
				vl_dim1], dabs(r__2));
			emax = dmax(r__3,r__4);
/* L220: */
		    }
		    remax = 1.f / emax;
		    i__2 = *n - ki + 1;
		    sscal_(&i__2, &remax, &vl[ki + is * vl_dim1], &c__1);
		    i__2 = *n - ki + 1;
		    sscal_(&i__2, &remax, &vl[ki + (is + 1) * vl_dim1], &c__1)
			    ;

		    i__2 = ki - 1;
		    for (k = 1; k <= i__2; ++k) {
			vl[k + is * vl_dim1] = 0.f;
			vl[k + (is + 1) * vl_dim1] = 0.f;
/* L230: */
		    }
		} else {
		    if (ki < *n - 1) {
			i__2 = *n - ki - 1;
			sgemv_("N", n, &i__2, &c_b15, &vl[(ki + 2) * vl_dim1
				+ 1], ldvl, &work[ki + 2 + *n], &c__1, &work[
				ki + *n], &vl[ki * vl_dim1 + 1], &c__1);
			i__2 = *n - ki - 1;
			sgemv_("N", n, &i__2, &c_b15, &vl[(ki + 2) * vl_dim1
				+ 1], ldvl, &work[ki + 2 + n2], &c__1, &work[
				ki + 1 + n2], &vl[(ki + 1) * vl_dim1 + 1], &
				c__1);
		    } else {
			sscal_(n, &work[ki + *n], &vl[ki * vl_dim1 + 1], &
				c__1);
			sscal_(n, &work[ki + 1 + n2], &vl[(ki + 1) * vl_dim1
				+ 1], &c__1);
		    }

		    emax = 0.f;
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
/* Computing MAX */
			r__3 = emax, r__4 = (r__1 = vl[k + ki * vl_dim1],
				dabs(r__1)) + (r__2 = vl[k + (ki + 1) *
				vl_dim1], dabs(r__2));
			emax = dmax(r__3,r__4);
/* L240: */
		    }
		    remax = 1.f / emax;
		    sscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);
		    sscal_(n, &remax, &vl[(ki + 1) * vl_dim1 + 1], &c__1);

		}

	    }

	    ++is;
	    if (ip != 0) {
		++is;
	    }
L250:
	    if (ip == -1) {
		ip = 0;
	    }
	    if (ip == 1) {
		ip = -1;
	    }

/* L260: */
	}

    }

    return 0;

/*     End of STREVC */

} /* strevc_ */

/* Subroutine */ int strti2_(char *uplo, char *diag, integer *n, real *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer j;
    static real ajj;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static logical upper;
    extern /* Subroutine */ int strmv_(char *, char *, char *, integer *,
	    real *, integer *, real *, integer *),
	    xerbla_(char *, integer *);
    static logical nounit;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    STRTI2 computes the inverse of a real upper or lower triangular
    matrix.

    This is the Level 2 BLAS version of the algorithm.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the matrix A is upper or lower triangular.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    DIAG    (input) CHARACTER*1
            Specifies whether or not the matrix A is unit triangular.
            = 'N':  Non-unit triangular
            = 'U':  Unit triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the triangular matrix A.  If UPLO = 'U', the
            leading n by n upper triangular part of the array A contains
            the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n by n lower triangular part of the array A contains
            the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.  If DIAG = 'U', the
            diagonal elements of A are also not referenced and are
            assumed to be 1.

            On exit, the (triangular) inverse of the original matrix, in
            the same storage format.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STRTI2", &i__1);
	return 0;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		a[j + j * a_dim1] = 1.f / a[j + j * a_dim1];
		ajj = -a[j + j * a_dim1];
	    } else {
		ajj = -1.f;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    strmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a[j * a_dim1 + 1], &c__1);
	    i__2 = j - 1;
	    sscal_(&i__2, &ajj, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		a[j + j * a_dim1] = 1.f / a[j + j * a_dim1];
		ajj = -a[j + j * a_dim1];
	    } else {
		ajj = -1.f;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		strmv_("Lower", "No transpose", diag, &i__1, &a[j + 1 + (j +
			1) * a_dim1], lda, &a[j + 1 + j * a_dim1], &c__1);
		i__1 = *n - j;
		sscal_(&i__1, &ajj, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of STRTI2 */

} /* strti2_ */

/* Subroutine */ int strtri_(char *uplo, char *diag, integer *n, real *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, i__1, i__2[2], i__3, i__4, i__5;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer j, jb, nb, nn;
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int strmm_(char *, char *, char *, char *,
	    integer *, integer *, real *, real *, integer *, real *, integer *
	    ), strsm_(char *, char *, char *,
	    char *, integer *, integer *, real *, real *, integer *, real *,
	    integer *), strti2_(char *, char *
	    , integer *, real *, integer *, integer *),
	    xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical nounit;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    STRTRI computes the inverse of a real upper or lower triangular
    matrix A.

    This is the Level 3 BLAS version of the algorithm.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  A is upper triangular;
            = 'L':  A is lower triangular.

    DIAG    (input) CHARACTER*1
            = 'N':  A is non-unit triangular;
            = 'U':  A is unit triangular.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the triangular matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of the array A contains
            the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of the array A contains
            the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.  If DIAG = 'U', the
            diagonal elements of A are also not referenced and are
            assumed to be 1.
            On exit, the (triangular) inverse of the original matrix, in
            the same storage format.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value
            > 0: if INFO = i, A(i,i) is exactly zero.  The triangular
                 matrix is singular and its inverse can not be computed.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STRTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Check for singularity if non-unit. */

    if (nounit) {
	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    if (a[*info + *info * a_dim1] == 0.f) {
		return 0;
	    }
/* L10: */
	}
	*info = 0;
    }

/*
       Determine the block size for this environment.

   Writing concatenation
*/
    i__2[0] = 1, a__1[0] = uplo;
    i__2[1] = 1, a__1[1] = diag;
    s_cat(ch__1, a__1, i__2, &c__2, (ftnlen)2);
    nb = ilaenv_(&c__1, "STRTRI", ch__1, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	strti2_(uplo, diag, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code */

	if (upper) {

/*           Compute inverse of upper triangular matrix */

	    i__1 = *n;
	    i__3 = nb;
	    for (j = 1; i__3 < 0 ? j >= i__1 : j <= i__1; j += i__3) {
/* Computing MIN */
		i__4 = nb, i__5 = *n - j + 1;
		jb = min(i__4,i__5);

/*              Compute rows 1:j-1 of current block column */

		i__4 = j - 1;
		strmm_("Left", "Upper", "No transpose", diag, &i__4, &jb, &
			c_b15, &a[a_offset], lda, &a[j * a_dim1 + 1], lda);
		i__4 = j - 1;
		strsm_("Right", "Upper", "No transpose", diag, &i__4, &jb, &
			c_b151, &a[j + j * a_dim1], lda, &a[j * a_dim1 + 1],
			lda);

/*              Compute inverse of current diagonal block */

		strti2_("Upper", diag, &jb, &a[j + j * a_dim1], lda, info);
/* L20: */
	    }
	} else {

/*           Compute inverse of lower triangular matrix */

	    nn = (*n - 1) / nb * nb + 1;
	    i__3 = -nb;
	    for (j = nn; i__3 < 0 ? j >= 1 : j <= 1; j += i__3) {
/* Computing MIN */
		i__1 = nb, i__4 = *n - j + 1;
		jb = min(i__1,i__4);
		if (j + jb <= *n) {

/*                 Compute rows j+jb:n of current block column */

		    i__1 = *n - j - jb + 1;
		    strmm_("Left", "Lower", "No transpose", diag, &i__1, &jb,
			    &c_b15, &a[j + jb + (j + jb) * a_dim1], lda, &a[j
			    + jb + j * a_dim1], lda);
		    i__1 = *n - j - jb + 1;
		    strsm_("Right", "Lower", "No transpose", diag, &i__1, &jb,
			     &c_b151, &a[j + j * a_dim1], lda, &a[j + jb + j *
			     a_dim1], lda);
		}

/*              Compute inverse of current diagonal block */

		strti2_("Lower", diag, &jb, &a[j + j * a_dim1], lda, info);
/* L30: */
	    }
	}
    }

    return 0;

/*     End of STRTRI */

} /* strtri_ */

