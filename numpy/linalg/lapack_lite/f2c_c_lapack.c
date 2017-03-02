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

static integer c__1 = 1;
static complex c_b55 = {0.f,0.f};
static complex c_b56 = {1.f,0.f};
static integer c_n1 = -1;
static integer c__3 = 3;
static integer c__2 = 2;
static integer c__0 = 0;
static integer c__8 = 8;
static integer c__4 = 4;
static integer c__65 = 65;
static real c_b871 = 1.f;
static integer c__15 = 15;
static logical c_false = FALSE_;
static real c_b1101 = 0.f;
static integer c__9 = 9;
static real c_b1150 = -1.f;
static real c_b1794 = .5f;

/* Subroutine */ int cgebak_(char *job, char *side, integer *n, integer *ilo,
	integer *ihi, real *scale, integer *m, complex *v, integer *ldv,
	integer *info)
{
    /* System generated locals */
    integer v_dim1, v_offset, i__1;

    /* Local variables */
    static integer i__, k;
    static real s;
    static integer ii;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *,
	    complex *, integer *);
    static logical leftv;
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), xerbla_(char *, integer *);
    static logical rightv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGEBAK forms the right or left eigenvectors of a complex general
    matrix by backward transformation on the computed eigenvectors of the
    balanced matrix output by CGEBAL.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            Specifies the type of backward transformation required:
            = 'N', do nothing, return immediately;
            = 'P', do backward transformation for permutation only;
            = 'S', do backward transformation for scaling only;
            = 'B', do backward transformations for both permutation and
                   scaling.
            JOB must be the same as the argument JOB supplied to CGEBAL.

    SIDE    (input) CHARACTER*1
            = 'R':  V contains right eigenvectors;
            = 'L':  V contains left eigenvectors.

    N       (input) INTEGER
            The number of rows of the matrix V.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            The integers ILO and IHI determined by CGEBAL.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    SCALE   (input) REAL array, dimension (N)
            Details of the permutation and scaling factors, as returned
            by CGEBAL.

    M       (input) INTEGER
            The number of columns of the matrix V.  M >= 0.

    V       (input/output) COMPLEX array, dimension (LDV,M)
            On entry, the matrix of right or left eigenvectors to be
            transformed, as returned by CHSEIN or CTREVC.
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
	xerbla_("CGEBAK", &i__1);
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
		csscal_(m, &s, &v[i__ + v_dim1], ldv);
/* L10: */
	    }
	}

	if (leftv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = 1.f / scale[i__];
		csscal_(m, &s, &v[i__ + v_dim1], ldv);
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
		cswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
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
		cswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L50:
		;
	    }
	}
    }

    return 0;

/*     End of CGEBAK */

} /* cgebak_ */

/* Subroutine */ int cgebal_(char *job, integer *n, complex *a, integer *lda,
	integer *ilo, integer *ihi, real *scale, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1, r__2;

    /* Builtin functions */
    double r_imag(complex *), c_abs(complex *);

    /* Local variables */
    static real c__, f, g;
    static integer i__, j, k, l, m;
    static real r__, s, ca, ra;
    static integer ica, ira, iexc;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *,
	    complex *, integer *);
    static real sfmin1, sfmin2, sfmax1, sfmax2;
    extern integer icamax_(integer *, complex *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), xerbla_(char *, integer *);
    static logical noconv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CGEBAL balances a general complex matrix A.  This involves, first,
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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

    This subroutine is based on the EISPACK routine CBAL.

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
	xerbla_("CGEBAL", &i__1);
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

    cswap_(&l, &a[j * a_dim1 + 1], &c__1, &a[m * a_dim1 + 1], &c__1);
    i__1 = *n - k + 1;
    cswap_(&i__1, &a[j + k * a_dim1], lda, &a[m + k * a_dim1], lda);

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
	    i__2 = j + i__ * a_dim1;
	    if (a[i__2].r != 0.f || r_imag(&a[j + i__ * a_dim1]) != 0.f) {
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
	    i__3 = i__ + j * a_dim1;
	    if (a[i__3].r != 0.f || r_imag(&a[i__ + j * a_dim1]) != 0.f) {
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
	    i__3 = j + i__ * a_dim1;
	    c__ += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[j + i__
		    * a_dim1]), dabs(r__2));
	    i__3 = i__ + j * a_dim1;
	    r__ += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[i__ + j
		    * a_dim1]), dabs(r__2));
L150:
	    ;
	}
	ica = icamax_(&l, &a[i__ * a_dim1 + 1], &c__1);
	ca = c_abs(&a[ica + i__ * a_dim1]);
	i__2 = *n - k + 1;
	ira = icamax_(&i__2, &a[i__ + k * a_dim1], lda);
	ra = c_abs(&a[i__ + (ira + k - 1) * a_dim1]);

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
	csscal_(&i__2, &g, &a[i__ + k * a_dim1], lda);
	csscal_(&l, &f, &a[i__ * a_dim1 + 1], &c__1);

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

/*     End of CGEBAL */

} /* cgebal_ */

/* Subroutine */ int cgebd2_(integer *m, integer *n, complex *a, integer *lda,
	 real *d__, real *e, complex *tauq, complex *taup, complex *work,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__;
    static complex alpha;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *),
	    clarfg_(integer *, complex *, complex *, integer *, complex *),
	    clacgv_(integer *, complex *, integer *), xerbla_(char *, integer
	    *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGEBD2 reduces a complex general m by n matrix A to upper or lower
    real bidiagonal form B by a unitary transformation: Q' * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit,
            if m >= n, the diagonal and the first superdiagonal are
              overwritten with the upper bidiagonal matrix B; the
              elements below the diagonal, with the array TAUQ, represent
              the unitary matrix Q as a product of elementary
              reflectors, and the elements above the first superdiagonal,
              with the array TAUP, represent the unitary matrix P as
              a product of elementary reflectors;
            if m < n, the diagonal and the first subdiagonal are
              overwritten with the lower bidiagonal matrix B; the
              elements below the first subdiagonal, with the array TAUQ,
              represent the unitary matrix Q as a product of
              elementary reflectors, and the elements above the diagonal,
              with the array TAUP, represent the unitary matrix P as
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

    TAUQ    (output) COMPLEX array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    WORK    (workspace) COMPLEX array, dimension (max(M,N))

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

    If m >= n,

       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex
    vectors; v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in
    A(i+1:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in
    A(i,i+2:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n,

       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, v and u are complex vectors;
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
	xerbla_("CGEBD2", &i__1);
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    clarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1, &
		    tauq[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1.f, a[i__2].i = 0.f;

/*           Apply H(i)' to A(i:m,i+1:n) from the left */

	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    r_cnjg(&q__1, &tauq[i__]);
	    clarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &q__1,
		     &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = d__[i__3], a[i__2].i = 0.f;

	    if (i__ < *n) {

/*
                Generate elementary reflector G(i) to annihilate
                A(i,i+2:n)
*/

		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + (i__ + 1) * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		clarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
			taup[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + (i__ + 1) * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Apply G(i) to A(i+1:m,i+1:n) from the right */

		i__2 = *m - i__;
		i__3 = *n - i__;
		clarf_("Right", &i__2, &i__3, &a[i__ + (i__ + 1) * a_dim1],
			lda, &taup[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + (i__ + 1) * a_dim1;
		i__3 = i__;
		a[i__2].r = e[i__3], a[i__2].i = 0.f;
	    } else {
		i__2 = i__;
		taup[i__2].r = 0.f, taup[i__2].i = 0.f;
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector G(i) to annihilate A(i,i+1:n) */

	    i__2 = *n - i__ + 1;
	    clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    clarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
		    taup[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1.f, a[i__2].i = 0.f;

/*           Apply G(i) to A(i+1:m,i:n) from the right */

	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
/* Computing MIN */
	    i__4 = i__ + 1;
	    clarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &taup[
		    i__], &a[min(i__4,*m) + i__ * a_dim1], lda, &work[1]);
	    i__2 = *n - i__ + 1;
	    clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = d__[i__3], a[i__2].i = 0.f;

	    if (i__ < *m) {

/*
                Generate elementary reflector H(i) to annihilate
                A(i+2:m,i)
*/

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		clarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1,
			 &tauq[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Apply H(i)' to A(i+1:m,i+1:n) from the left */

		i__2 = *m - i__;
		i__3 = *n - i__;
		r_cnjg(&q__1, &tauq[i__]);
		clarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &
			c__1, &q__1, &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &
			work[1]);
		i__2 = i__ + 1 + i__ * a_dim1;
		i__3 = i__;
		a[i__2].r = e[i__3], a[i__2].i = 0.f;
	    } else {
		i__2 = i__;
		tauq[i__2].r = 0.f, tauq[i__2].i = 0.f;
	    }
/* L20: */
	}
    }
    return 0;

/*     End of CGEBD2 */

} /* cgebd2_ */

/* Subroutine */ int cgebrd_(integer *m, integer *n, complex *a, integer *lda,
	 real *d__, real *e, complex *tauq, complex *taup, complex *work,
	integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1;
    complex q__1;

    /* Local variables */
    static integer i__, j, nb, nx;
    static real ws;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *);
    static integer nbmin, iinfo, minmn;
    extern /* Subroutine */ int cgebd2_(integer *, integer *, complex *,
	    integer *, real *, real *, complex *, complex *, complex *,
	    integer *), clabrd_(integer *, integer *, integer *, complex *,
	    integer *, real *, real *, complex *, complex *, complex *,
	    integer *, complex *, integer *), xerbla_(char *, integer *);
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

    CGEBRD reduces a general complex M-by-N matrix A to upper or lower
    bidiagonal form B by a unitary transformation: Q**H * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N general matrix to be reduced.
            On exit,
            if m >= n, the diagonal and the first superdiagonal are
              overwritten with the upper bidiagonal matrix B; the
              elements below the diagonal, with the array TAUQ, represent
              the unitary matrix Q as a product of elementary
              reflectors, and the elements above the first superdiagonal,
              with the array TAUP, represent the unitary matrix P as
              a product of elementary reflectors;
            if m < n, the diagonal and the first subdiagonal are
              overwritten with the lower bidiagonal matrix B; the
              elements below the first subdiagonal, with the array TAUQ,
              represent the unitary matrix Q as a product of
              elementary reflectors, and the elements above the diagonal,
              with the array TAUP, represent the unitary matrix P as
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

    TAUQ    (output) COMPLEX array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

    If m >= n,

       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex
    vectors; v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in
    A(i+1:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in
    A(i,i+2:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n,

       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex
    vectors; v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

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
    i__1 = 1, i__2 = ilaenv_(&c__1, "CGEBRD", " ", m, n, &c_n1, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = max(i__1,i__2);
    lwkopt = (*m + *n) * nb;
    r__1 = (real) lwkopt;
    work[1].r = r__1, work[1].i = 0.f;
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
	xerbla_("CGEBRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    minmn = min(*m,*n);
    if (minmn == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = nb, i__2 = ilaenv_(&c__3, "CGEBRD", " ", m, n, &c_n1, &c_n1, (
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

		nbmin = ilaenv_(&c__2, "CGEBRD", " ", m, n, &c_n1, &c_n1, (
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
          Reduce rows and columns i:i+ib-1 to bidiagonal form and return
          the matrices X and Y which are needed to update the unreduced
          part of the matrix
*/

	i__3 = *m - i__ + 1;
	i__4 = *n - i__ + 1;
	clabrd_(&i__3, &i__4, &nb, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[
		i__], &tauq[i__], &taup[i__], &work[1], &ldwrkx, &work[ldwrkx
		* nb + 1], &ldwrky);

/*
          Update the trailing submatrix A(i+ib:m,i+ib:n), using
          an update of the form  A := A - V*Y' - X*U'
*/

	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	q__1.r = -1.f, q__1.i = -0.f;
	cgemm_("No transpose", "Conjugate transpose", &i__3, &i__4, &nb, &
		q__1, &a[i__ + nb + i__ * a_dim1], lda, &work[ldwrkx * nb +
		nb + 1], &ldwrky, &c_b56, &a[i__ + nb + (i__ + nb) * a_dim1],
		lda);
	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	q__1.r = -1.f, q__1.i = -0.f;
	cgemm_("No transpose", "No transpose", &i__3, &i__4, &nb, &q__1, &
		work[nb + 1], &ldwrkx, &a[i__ + (i__ + nb) * a_dim1], lda, &
		c_b56, &a[i__ + nb + (i__ + nb) * a_dim1], lda);

/*        Copy diagonal and off-diagonal elements of B back into A */

	if (*m >= *n) {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + j * a_dim1;
		i__5 = j;
		a[i__4].r = d__[i__5], a[i__4].i = 0.f;
		i__4 = j + (j + 1) * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.f;
/* L10: */
	    }
	} else {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + j * a_dim1;
		i__5 = j;
		a[i__4].r = d__[i__5], a[i__4].i = 0.f;
		i__4 = j + 1 + j * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.f;
/* L20: */
	    }
	}
/* L30: */
    }

/*     Use unblocked code to reduce the remainder of the matrix */

    i__2 = *m - i__ + 1;
    i__1 = *n - i__ + 1;
    cgebd2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__], &
	    tauq[i__], &taup[i__], &work[1], &iinfo);
    work[1].r = ws, work[1].i = 0.f;
    return 0;

/*     End of CGEBRD */

} /* cgebrd_ */

/* Subroutine */ int cgeev_(char *jobvl, char *jobvr, integer *n, complex *a,
	integer *lda, complex *w, complex *vl, integer *ldvl, complex *vr,
	integer *ldvr, complex *work, integer *lwork, real *rwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3, i__4;
    real r__1, r__2;
    complex q__1, q__2;

    /* Builtin functions */
    double sqrt(doublereal), r_imag(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, k, ihi;
    static real scl;
    static integer ilo;
    static real dum[1], eps;
    static complex tmp;
    static integer ibal;
    static char side[1];
    static integer maxb;
    static real anrm;
    static integer ierr, itau, iwrk, nout;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *);
    extern logical lsame_(char *, char *);
    extern doublereal scnrm2_(integer *, complex *, integer *);
    extern /* Subroutine */ int cgebak_(char *, char *, integer *, integer *,
	    integer *, real *, integer *, complex *, integer *, integer *), cgebal_(char *, integer *, complex *, integer *,
	    integer *, integer *, real *, integer *), slabad_(real *,
	    real *);
    static logical scalea;
    extern doublereal clange_(char *, integer *, integer *, complex *,
	    integer *, real *);
    static real cscale;
    extern /* Subroutine */ int cgehrd_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *, integer *),
	     clascl_(char *, integer *, integer *, real *, real *, integer *,
	    integer *, complex *, integer *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), clacpy_(char *, integer *, integer *, complex *, integer *,
	    complex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical select[1];
    static real bignum;
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int chseqr_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *), ctrevc_(char *,
	    char *, logical *, integer *, complex *, integer *, complex *,
	    integer *, complex *, integer *, integer *, integer *, complex *,
	    real *, integer *), cunghr_(integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    integer *);
    static integer minwrk, maxwrk;
    static logical wantvl;
    static real smlnum;
    static integer hswork, irwork;
    static logical lquery, wantvr;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CGEEV computes for an N-by-N complex nonsymmetric matrix A, the
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
            = 'V': left eigenvectors of are computed.

    JOBVR   (input) CHARACTER*1
            = 'N': right eigenvectors of A are not computed;
            = 'V': right eigenvectors of A are computed.

    N       (input) INTEGER
            The order of the matrix A. N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the N-by-N matrix A.
            On exit, A has been overwritten.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    W       (output) COMPLEX array, dimension (N)
            W contains the computed eigenvalues.

    VL      (output) COMPLEX array, dimension (LDVL,N)
            If JOBVL = 'V', the left eigenvectors u(j) are stored one
            after another in the columns of VL, in the same order
            as their eigenvalues.
            If JOBVL = 'N', VL is not referenced.
            u(j) = VL(:,j), the j-th column of VL.

    LDVL    (input) INTEGER
            The leading dimension of the array VL.  LDVL >= 1; if
            JOBVL = 'V', LDVL >= N.

    VR      (output) COMPLEX array, dimension (LDVR,N)
            If JOBVR = 'V', the right eigenvectors v(j) are stored one
            after another in the columns of VR, in the same order
            as their eigenvalues.
            If JOBVR = 'N', VR is not referenced.
            v(j) = VR(:,j), the j-th column of VR.

    LDVR    (input) INTEGER
            The leading dimension of the array VR.  LDVR >= 1; if
            JOBVR = 'V', LDVR >= N.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,2*N).
            For good performance, LWORK must generally be larger.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace) REAL array, dimension (2*N)

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = i, the QR algorithm failed to compute all the
                  eigenvalues, and no eigenvectors have been computed;
                  elements and i+1:N of W contain eigenvalues which have
                  converged.

    =====================================================================


       Test the input arguments
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;
    --rwork;

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
	*info = -8;
    } else if (*ldvr < 1 || wantvr && *ldvr < *n) {
	*info = -10;
    }

/*
       Compute workspace
        (Note: Comments in the code beginning "Workspace:" describe the
         minimal amount of workspace needed at that point in the code,
         as well as the preferred amount for good performance.
         CWorkspace refers to complex workspace, and RWorkspace to real
         workspace. NB refers to the optimal block size for the
         immediately following subroutine, as returned by ILAENV.
         HSWORK refers to the workspace preferred by CHSEQR, as
         calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
         the worst case.)
*/

    minwrk = 1;
    if (*info == 0 && (*lwork >= 1 || lquery)) {
	maxwrk = *n + *n * ilaenv_(&c__1, "CGEHRD", " ", n, &c__1, n, &c__0, (
		ftnlen)6, (ftnlen)1);
	if (! wantvl && ! wantvr) {
/* Computing MAX */
	    i__1 = 1, i__2 = *n << 1;
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "CHSEQR", "EN", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "CHSEQR", "EN", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = *n << 1;
	    hswork = max(i__1,i__2);
	    maxwrk = max(maxwrk,hswork);
	} else {
/* Computing MAX */
	    i__1 = 1, i__2 = *n << 1;
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + (*n - 1) * ilaenv_(&c__1, "CUNGHR",
		    " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "CHSEQR", "SV", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "CHSEQR", "SV", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = *n << 1;
	    hswork = max(i__1,i__2);
/* Computing MAX */
	    i__1 = max(maxwrk,hswork), i__2 = *n << 1;
	    maxwrk = max(i__1,i__2);
	}
	work[1].r = (real) maxwrk, work[1].i = 0.f;
    }
    if (*lwork < minwrk && ! lquery) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGEEV ", &i__1);
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

    anrm = clange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0.f && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	clascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }

/*
       Balance the matrix
       (CWorkspace: none)
       (RWorkspace: need N)
*/

    ibal = 1;
    cgebal_("B", n, &a[a_offset], lda, &ilo, &ihi, &rwork[ibal], &ierr);

/*
       Reduce to upper Hessenberg form
       (CWorkspace: need 2*N, prefer N+N*NB)
       (RWorkspace: none)
*/

    itau = 1;
    iwrk = itau + *n;
    i__1 = *lwork - iwrk + 1;
    cgehrd_(n, &ilo, &ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1,
	     &ierr);

    if (wantvl) {

/*
          Want left eigenvectors
          Copy Householder vectors to VL
*/

	*(unsigned char *)side = 'L';
	clacpy_("L", n, n, &a[a_offset], lda, &vl[vl_offset], ldvl)
		;

/*
          Generate unitary matrix in VL
          (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
          (RWorkspace: none)
*/

	i__1 = *lwork - iwrk + 1;
	cunghr_(n, &ilo, &ihi, &vl[vl_offset], ldvl, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VL
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	chseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vl[
		vl_offset], ldvl, &work[iwrk], &i__1, info);

	if (wantvr) {

/*
             Want left and right eigenvectors
             Copy Schur vectors to VR
*/

	    *(unsigned char *)side = 'B';
	    clacpy_("F", n, n, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr);
	}

    } else if (wantvr) {

/*
          Want right eigenvectors
          Copy Householder vectors to VR
*/

	*(unsigned char *)side = 'R';
	clacpy_("L", n, n, &a[a_offset], lda, &vr[vr_offset], ldvr)
		;

/*
          Generate unitary matrix in VR
          (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
          (RWorkspace: none)
*/

	i__1 = *lwork - iwrk + 1;
	cunghr_(n, &ilo, &ihi, &vr[vr_offset], ldvr, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VR
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	chseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);

    } else {

/*
          Compute eigenvalues only
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	chseqr_("E", "N", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);
    }

/*     If INFO > 0 from CHSEQR, then quit */

    if (*info > 0) {
	goto L50;
    }

    if (wantvl || wantvr) {

/*
          Compute left and/or right eigenvectors
          (CWorkspace: need 2*N)
          (RWorkspace: need 2*N)
*/

	irwork = ibal + *n;
	ctrevc_(side, "B", select, n, &a[a_offset], lda, &vl[vl_offset], ldvl,
		 &vr[vr_offset], ldvr, n, &nout, &work[iwrk], &rwork[irwork],
		&ierr);
    }

    if (wantvl) {

/*
          Undo balancing of left eigenvectors
          (CWorkspace: none)
          (RWorkspace: need N)
*/

	cgebak_("B", "L", n, &ilo, &ihi, &rwork[ibal], n, &vl[vl_offset],
		ldvl, &ierr);

/*        Normalize left eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scl = 1.f / scnrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
	    csscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
	    i__2 = *n;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = k + i__ * vl_dim1;
/* Computing 2nd power */
		r__1 = vl[i__3].r;
/* Computing 2nd power */
		r__2 = r_imag(&vl[k + i__ * vl_dim1]);
		rwork[irwork + k - 1] = r__1 * r__1 + r__2 * r__2;
/* L10: */
	    }
	    k = isamax_(n, &rwork[irwork], &c__1);
	    r_cnjg(&q__2, &vl[k + i__ * vl_dim1]);
	    r__1 = sqrt(rwork[irwork + k - 1]);
	    q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
	    tmp.r = q__1.r, tmp.i = q__1.i;
	    cscal_(n, &tmp, &vl[i__ * vl_dim1 + 1], &c__1);
	    i__2 = k + i__ * vl_dim1;
	    i__3 = k + i__ * vl_dim1;
	    r__1 = vl[i__3].r;
	    q__1.r = r__1, q__1.i = 0.f;
	    vl[i__2].r = q__1.r, vl[i__2].i = q__1.i;
/* L20: */
	}
    }

    if (wantvr) {

/*
          Undo balancing of right eigenvectors
          (CWorkspace: none)
          (RWorkspace: need N)
*/

	cgebak_("B", "R", n, &ilo, &ihi, &rwork[ibal], n, &vr[vr_offset],
		ldvr, &ierr);

/*        Normalize right eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scl = 1.f / scnrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
	    csscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
	    i__2 = *n;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = k + i__ * vr_dim1;
/* Computing 2nd power */
		r__1 = vr[i__3].r;
/* Computing 2nd power */
		r__2 = r_imag(&vr[k + i__ * vr_dim1]);
		rwork[irwork + k - 1] = r__1 * r__1 + r__2 * r__2;
/* L30: */
	    }
	    k = isamax_(n, &rwork[irwork], &c__1);
	    r_cnjg(&q__2, &vr[k + i__ * vr_dim1]);
	    r__1 = sqrt(rwork[irwork + k - 1]);
	    q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
	    tmp.r = q__1.r, tmp.i = q__1.i;
	    cscal_(n, &tmp, &vr[i__ * vr_dim1 + 1], &c__1);
	    i__2 = k + i__ * vr_dim1;
	    i__3 = k + i__ * vr_dim1;
	    r__1 = vr[i__3].r;
	    q__1.r = r__1, q__1.i = 0.f;
	    vr[i__2].r = q__1.r, vr[i__2].i = q__1.i;
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
	clascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &w[*info + 1]
		, &i__2, &ierr);
	if (*info > 0) {
	    i__1 = ilo - 1;
	    clascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &w[1], n,
		     &ierr);
	}
    }

    work[1].r = (real) maxwrk, work[1].i = 0.f;
    return 0;

/*     End of CGEEV */

} /* cgeev_ */

/* Subroutine */ int cgehd2_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__;
    static complex alpha;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *),
	    clarfg_(integer *, complex *, complex *, integer *, complex *),
	    xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGEHD2 reduces a complex general matrix A to upper Hessenberg form H
    by a unitary similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to CGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= max(1,N).

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the n by n general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the unitary matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX array, dimension (N)

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

    where tau is a complex scalar, and v is a complex vector with
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
	xerbla_("CGEHD2", &i__1);
	return 0;
    }

    i__1 = *ihi - 1;
    for (i__ = *ilo; i__ <= i__1; ++i__) {

/*        Compute elementary reflector H(i) to annihilate A(i+2:ihi,i) */

	i__2 = i__ + 1 + i__ * a_dim1;
	alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	i__2 = *ihi - i__;
/* Computing MIN */
	i__3 = i__ + 2;
	clarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[
		i__]);
	i__2 = i__ + 1 + i__ * a_dim1;
	a[i__2].r = 1.f, a[i__2].i = 0.f;

/*        Apply H(i) to A(1:ihi,i+1:ihi) from the right */

	i__2 = *ihi - i__;
	clarf_("Right", ihi, &i__2, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[(i__ + 1) * a_dim1 + 1], lda, &work[1]);

/*        Apply H(i)' to A(i+1:ihi,i+1:n) from the left */

	i__2 = *ihi - i__;
	i__3 = *n - i__;
	r_cnjg(&q__1, &tau[i__]);
	clarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &q__1,
		 &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &work[1]);

	i__2 = i__ + 1 + i__ * a_dim1;
	a[i__2].r = alpha.r, a[i__2].i = alpha.i;
/* L10: */
    }

    return 0;

/*     End of CGEHD2 */

} /* cgehd2_ */

/* Subroutine */ int cgehrd_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *lwork, integer
	*info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    complex q__1;

    /* Local variables */
    static integer i__;
    static complex t[4160]	/* was [65][64] */;
    static integer ib;
    static complex ei;
    static integer nb, nh, nx, iws;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int cgehd2_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *), clarfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *,
	    complex *, integer *, complex *, integer *, complex *, integer *,
	    complex *, integer *), clahrd_(
	    integer *, integer *, integer *, complex *, integer *, complex *,
	    complex *, integer *, complex *, integer *), xerbla_(char *,
	    integer *);
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

    CGEHRD reduces a complex general matrix A to upper Hessenberg form H
    by a unitary similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to CGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the N-by-N general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the unitary matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
            zero.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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

    where tau is a complex scalar, and v is a complex vector with
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
    i__1 = 64, i__2 = ilaenv_(&c__1, "CGEHRD", " ", n, ilo, ihi, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = min(i__1,i__2);
    lwkopt = *n * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
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
	xerbla_("CGEHRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Set elements 1:ILO-1 and IHI:N-1 of TAU to zero */

    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	tau[i__2].r = 0.f, tau[i__2].i = 0.f;
/* L10: */
    }
    i__1 = *n - 1;
    for (i__ = max(1,*ihi); i__ <= i__1; ++i__) {
	i__2 = i__;
	tau[i__2].r = 0.f, tau[i__2].i = 0.f;
/* L20: */
    }

/*     Quick return if possible */

    nh = *ihi - *ilo + 1;
    if (nh <= 1) {
	work[1].r = 1.f, work[1].i = 0.f;
	return 0;
    }

    nbmin = 2;
    iws = 1;
    if (nb > 1 && nb < nh) {

/*
          Determine when to cross over from blocked to unblocked code
          (last block is always handled by unblocked code).

   Computing MAX
*/
	i__1 = nb, i__2 = ilaenv_(&c__3, "CGEHRD", " ", n, ilo, ihi, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "CGEHRD", " ", n, ilo, ihi, &
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

	    clahrd_(ihi, &i__, &ib, &a[i__ * a_dim1 + 1], lda, &tau[i__], t, &
		    c__65, &work[1], &ldwork);

/*
             Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
             right, computing  A := A - Y * V'. V(i+ib,ib-1) must be set
             to 1.
*/

	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    ei.r = a[i__3].r, ei.i = a[i__3].i;
	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    a[i__3].r = 1.f, a[i__3].i = 0.f;
	    i__3 = *ihi - i__ - ib + 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemm_("No transpose", "Conjugate transpose", ihi, &i__3, &ib, &
		    q__1, &work[1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda,
		     &c_b56, &a[(i__ + ib) * a_dim1 + 1], lda);
	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    a[i__3].r = ei.r, a[i__3].i = ei.i;

/*
             Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
             left
*/

	    i__3 = *ihi - i__;
	    i__4 = *n - i__ - ib + 1;
	    clarfb_("Left", "Conjugate transpose", "Forward", "Columnwise", &
		    i__3, &i__4, &ib, &a[i__ + 1 + i__ * a_dim1], lda, t, &
		    c__65, &a[i__ + 1 + (i__ + ib) * a_dim1], lda, &work[1], &
		    ldwork);
/* L30: */
	}
    }

/*     Use unblocked code to reduce the rest of the matrix */

    cgehd2_(n, &i__, ihi, &a[a_offset], lda, &tau[1], &work[1], &iinfo);
    work[1].r = (real) iws, work[1].i = 0.f;

    return 0;

/*     End of CGEHRD */

} /* cgehrd_ */

/* Subroutine */ int cgelq2_(integer *m, integer *n, complex *a, integer *lda,
	 complex *tau, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, k;
    static complex alpha;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *),
	    clarfg_(integer *, complex *, complex *, integer *, complex *),
	    clacgv_(integer *, complex *, integer *), xerbla_(char *, integer
	    *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGELQ2 computes an LQ factorization of a complex m by n matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m by min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX array, dimension (M)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(k)' . . . H(2)' H(1)', where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; conjg(v(i+1:n)) is stored on exit in
    A(i,i+1:n), and tau in TAU(i).

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
	xerbla_("CGELQ2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i,i+1:n) */

	i__2 = *n - i__ + 1;
	clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	i__2 = i__ + i__ * a_dim1;
	alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	i__2 = *n - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	clarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &tau[i__]
		);
	if (i__ < *m) {

/*           Apply H(i) to A(i+1:m,i:n) from the right */

	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1.f, a[i__2].i = 0.f;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    clarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	}
	i__2 = i__ + i__ * a_dim1;
	a[i__2].r = alpha.r, a[i__2].i = alpha.i;
	i__2 = *n - i__ + 1;
	clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
/* L10: */
    }
    return 0;

/*     End of CGELQ2 */

} /* cgelq2_ */

/* Subroutine */ int cgelqf_(integer *m, integer *n, complex *a, integer *lda,
	 complex *tau, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int cgelq2_(integer *, integer *, complex *,
	    integer *, complex *, complex *, integer *), clarfb_(char *, char
	    *, char *, char *, integer *, integer *, integer *, complex *,
	    integer *, complex *, integer *, complex *, integer *, complex *,
	    integer *), clarft_(char *, char *
	    , integer *, integer *, complex *, integer *, complex *, complex *
	    , integer *), xerbla_(char *, integer *);
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

    CGELQF computes an LQ factorization of a complex M-by-N matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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

       Q = H(k)' . . . H(2)' H(1)', where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; conjg(v(i+1:n)) is stored on exit in
    A(i,i+1:n), and tau in TAU(i).

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
    nb = ilaenv_(&c__1, "CGELQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *m * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
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
	xerbla_("CGELQF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "CGELQF", " ", m, n, &c_n1, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "CGELQF", " ", m, n, &c_n1, &
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
	    cgelq2_(&ib, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *m) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *n - i__ + 1;
		clarft_("Forward", "Rowwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i+ib:m,i:n) from the right */

		i__3 = *m - i__ - ib + 1;
		i__4 = *n - i__ + 1;
		clarfb_("Right", "No transpose", "Forward", "Rowwise", &i__3,
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
	cgelq2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1].r = (real) iws, work[1].i = 0.f;
    return 0;

/*     End of CGELQF */

} /* cgelqf_ */

/* Subroutine */ int cgeqr2_(integer *m, integer *n, complex *a, integer *lda,
	 complex *tau, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, k;
    static complex alpha;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *),
	    clarfg_(integer *, complex *, complex *, integer *, complex *),
	    xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX array, dimension (N)

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
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
	xerbla_("CGEQR2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	i__2 = *m - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	clarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ * a_dim1]
		, &c__1, &tau[i__]);
	if (i__ < *n) {

/*           Apply H(i)' to A(i:m,i+1:n) from the left */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1.f, a[i__2].i = 0.f;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    r_cnjg(&q__1, &tau[i__]);
	    clarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &q__1,
		     &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = alpha.r, a[i__2].i = alpha.i;
	}
/* L10: */
    }
    return 0;

/*     End of CGEQR2 */

} /* cgeqr2_ */

/* Subroutine */ int cgeqrf_(integer *m, integer *n, complex *a, integer *lda,
	 complex *tau, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int cgeqr2_(integer *, integer *, complex *,
	    integer *, complex *, complex *, integer *), clarfb_(char *, char
	    *, char *, char *, integer *, integer *, integer *, complex *,
	    integer *, complex *, integer *, complex *, integer *, complex *,
	    integer *), clarft_(char *, char *
	    , integer *, integer *, complex *, integer *, complex *, complex *
	    , integer *), xerbla_(char *, integer *);
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

    CGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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

    where tau is a complex scalar, and v is a complex vector with
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
    nb = ilaenv_(&c__1, "CGEQRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *n * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
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
	xerbla_("CGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "CGEQRF", " ", m, n, &c_n1, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "CGEQRF", " ", m, n, &c_n1, &
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
	    cgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *n) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *m - i__ + 1;
		clarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i:m,i+ib:n) from the left */

		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		clarfb_("Left", "Conjugate transpose", "Forward", "Columnwise"
			, &i__3, &i__4, &ib, &a[i__ + i__ * a_dim1], lda, &
			work[1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda,
			&work[ib + 1], &ldwork);
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
	cgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1].r = (real) iws, work[1].i = 0.f;
    return 0;

/*     End of CGEQRF */

} /* cgeqrf_ */

/* Subroutine */ int cgesdd_(char *jobz, integer *m, integer *n, complex *a,
	integer *lda, real *s, complex *u, integer *ldu, complex *vt, integer
	*ldvt, complex *work, integer *lwork, real *rwork, integer *iwork,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1,
	    i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, ie, il, ir, iu, blk;
    static real dum[1], eps;
    static integer iru, ivt, iscl;
    static real anrm;
    static integer idum[1], ierr, itau, irvt;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *);
    extern logical lsame_(char *, char *);
    static integer chunk, minmn, wrkbl, itaup, itauq;
    static logical wntqa;
    static integer nwork;
    extern /* Subroutine */ int clacp2_(char *, integer *, integer *, real *,
	    integer *, complex *, integer *);
    static logical wntqn, wntqo, wntqs;
    static integer mnthr1, mnthr2;
    extern /* Subroutine */ int cgebrd_(integer *, integer *, complex *,
	    integer *, real *, real *, complex *, complex *, complex *,
	    integer *, integer *);
    extern doublereal clange_(char *, integer *, integer *, complex *,
	    integer *, real *);
    extern /* Subroutine */ int cgelqf_(integer *, integer *, complex *,
	    integer *, complex *, complex *, integer *, integer *), clacrm_(
	    integer *, integer *, complex *, integer *, real *, integer *,
	    complex *, integer *, real *), clarcm_(integer *, integer *, real
	    *, integer *, complex *, integer *, complex *, integer *, real *),
	     clascl_(char *, integer *, integer *, real *, real *, integer *,
	    integer *, complex *, integer *, integer *), sbdsdc_(char
	    *, char *, integer *, real *, real *, real *, integer *, real *,
	    integer *, real *, integer *, real *, integer *, integer *), cgeqrf_(integer *, integer *, complex *, integer
	    *, complex *, complex *, integer *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex
	    *, integer *, complex *, integer *), claset_(char *,
	    integer *, integer *, complex *, complex *, complex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int cungbr_(char *, integer *, integer *, integer
	    *, complex *, integer *, complex *, complex *, integer *, integer
	    *);
    static real bignum;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *), cunmbr_(char *, char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *), cunglq_(
	    integer *, integer *, integer *, complex *, integer *, complex *,
	    complex *, integer *, integer *);
    static integer ldwrkl;
    extern /* Subroutine */ int cungqr_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *, integer *);
    static integer ldwrkr, minwrk, ldwrku, maxwrk, ldwkvt;
    static real smlnum;
    static logical wntqas, lquery;
    static integer nrwork;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    CGESDD computes the singular value decomposition (SVD) of a complex
    M-by-N matrix A, optionally computing the left and/or right singular
    vectors, by using divide-and-conquer method. The SVD is written

         A = U * SIGMA * conjugate-transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
    V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**H, not V.

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
            = 'A':  all M columns of U and all N rows of V**H are
                    returned in the arrays U and VT;
            = 'S':  the first min(M,N) columns of U and the first
                    min(M,N) rows of V**H are returned in the arrays U
                    and VT;
            = 'O':  If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**H are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**H are overwritten
                    in the array VT;
            = 'N':  no columns of U or rows of V**H are computed.

    M       (input) INTEGER
            The number of rows of the input matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the input matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
            if JOBZ = 'O',  A is overwritten with the first N columns
                            of U (the left singular vectors, stored
                            columnwise) if M >= N;
                            A is overwritten with the first M rows
                            of V**H (the right singular vectors, stored
                            rowwise) otherwise.
            if JOBZ .ne. 'O', the contents of A are destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    S       (output) REAL array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    U       (output) COMPLEX array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
            UCOL = min(M,N) if JOBZ = 'S'.
            If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
            unitary matrix U;
            if JOBZ = 'S', U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
            if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.

    LDU     (input) INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.

    VT      (output) COMPLEX array, dimension (LDVT,N)
            If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
            N-by-N unitary matrix V**H;
            if JOBZ = 'S', VT contains the first min(M,N) rows of
            V**H (the right singular vectors, stored rowwise);
            if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.

    LDVT    (input) INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
            if JOBZ = 'S', LDVT >= min(M,N).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK >= 1.
            if JOBZ = 'N', LWORK >= 2*min(M,N)+max(M,N).
            if JOBZ = 'O',
                  LWORK >= 2*min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
            if JOBZ = 'S' or 'A',
                  LWORK >= min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
            For good performance, LWORK should generally be larger.
            If LWORK < 0 but other input arguments are legal, WORK(1)
            returns the optimal LWORK.

    RWORK   (workspace) REAL array, dimension (LRWORK)
            If JOBZ = 'N', LRWORK >= 7*min(M,N).
            Otherwise, LRWORK >= 5*min(M,N)*min(M,N) + 5*min(M,N)

    IWORK   (workspace) INTEGER array, dimension (8*min(M,N))

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The updating process of SBDSDC did not converge.

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
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    mnthr1 = (integer) (minmn * 17.f / 9.f);
    mnthr2 = (integer) (minmn * 5.f / 3.f);
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
         CWorkspace refers to complex workspace, and RWorkspace to
         real workspace. NB refers to the optimal block size for the
         immediately following subroutine, as returned by ILAENV.)
*/

    if (*info == 0 && *m > 0 && *n > 0) {
	if (*m >= *n) {

/*
             There is no complex work space needed for bidiagonal SVD
             The real work space needed for bidiagonal SVD is BDSPAC,
             BDSPAC = 3*N*N + 4*N
*/

	    if (*m >= mnthr1) {
		if (wntqn) {

/*                 Path 1 (M much larger than N, JOBZ='N') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "CGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl;
		    minwrk = *n * 3;
		} else if (wntqo) {

/*                 Path 2 (M much larger than N, JOBZ='O') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "CGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "CUNGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *n * *n + wrkbl;
		    minwrk = (*n << 1) * *n + *n * 3;
		} else if (wntqs) {

/*                 Path 3 (M much larger than N, JOBZ='S') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "CGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "CUNGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + *n * 3;
		} else if (wntqa) {

/*                 Path 4 (M much larger than N, JOBZ='A') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "CGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *m * ilaenv_(&c__1, "CUNGQR",
			    " ", m, m, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", n, n, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + (*n << 1) + *m;
		}
	    } else if (*m >= mnthr2) {

/*              Path 5 (M much larger than N, but not as much as MNTHR1) */

		maxwrk = (*n << 1) + (*m + *n) * ilaenv_(&c__1, "CGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = (*n << 1) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6 (M at least N, but not much larger) */

		maxwrk = (*n << 1) + (*m + *n) * ilaenv_(&c__1, "CGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = (*n << 1) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		}
	    }
	} else {

/*
             There is no complex work space needed for bidiagonal SVD
             The real work space needed for bidiagonal SVD is BDSPAC,
             BDSPAC = 3*M*M + 4*M
*/

	    if (*n >= mnthr1) {
		if (wntqn) {

/*                 Path 1t (N much larger than M, JOBZ='N') */

		    maxwrk = *m + *m * ilaenv_(&c__1, "CGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3;
		} else if (wntqo) {

/*                 Path 2t (N much larger than M, JOBZ='O') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "CGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "CUNGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *m * *m + wrkbl;
		    minwrk = (*m << 1) * *m + *m * 3;
		} else if (wntqs) {

/*                 Path 3t (N much larger than M, JOBZ='S') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "CGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "CUNGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + *m * 3;
		} else if (wntqa) {

/*                 Path 4t (N much larger than M, JOBZ='A') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "CGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *n * ilaenv_(&c__1, "CUNGLQ",
			    " ", n, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "CGEBRD", " ", m, m, &c_n1, &c_n1, (ftnlen)
			    6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + (*m << 1) + *n;
		}
	    } else if (*n >= mnthr2) {

/*              Path 5t (N much larger than M, but not as much as MNTHR1) */

		maxwrk = (*m << 1) + (*m + *n) * ilaenv_(&c__1, "CGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = (*m << 1) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "P", m, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "P", m, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "P", n, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6t (N greater than M, but not much larger) */

		maxwrk = (*m << 1) + (*m + *n) * ilaenv_(&c__1, "CGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = (*m << 1) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "PRC", m, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNMBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "PRC", m, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *n * ilaenv_(&c__1,
			    "CUNGBR", "PRC", n, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1,
			    "CUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		}
	    }
	}
	maxwrk = max(maxwrk,minwrk);
	work[1].r = (real) maxwrk, work[1].i = 0.f;
    }

    if (*lwork < minwrk && ! lquery) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGESDD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	if (*lwork >= 1) {
	    work[1].r = 1.f, work[1].i = 0.f;
	}
	return 0;
    }

/*     Get machine constants */

    eps = slamch_("P");
    smlnum = sqrt(slamch_("S")) / eps;
    bignum = 1.f / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = clange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0.f && anrm < smlnum) {
	iscl = 1;
	clascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	clascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
		ierr);
    }

    if (*m >= *n) {

/*
          A has at least as many rows as columns. If A has sufficiently
          more rows than columns, first reduce using the QR
          decomposition (if sufficient workspace available)
*/

	if (*m >= mnthr1) {

	    if (wntqn) {

/*
                Path 1 (M much larger than N, JOBZ='N')
                No singular vectors to be computed
*/

		itau = 1;
		nwork = itau + *n;

/*
                Compute A=Q*R
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: need 0)
*/

		i__1 = *lwork - nwork + 1;
		cgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out below R */

		i__1 = *n - 1;
		i__2 = *n - 1;
		claset_("L", &i__1, &i__2, &c_b55, &c_b55, &a[a_dim1 + 2],
			lda);
		ie = 1;
		itauq = 1;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in A
                (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
                (RWorkspace: need N)
*/

		i__1 = *lwork - nwork + 1;
		cgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nrwork = ie + *n;

/*
                Perform bidiagonal SVD, compute singular values only
                (CWorkspace: 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);

	    } else if (wntqo) {

/*
                Path 2 (M much larger than N, JOBZ='O')
                N left singular vectors to be overwritten on A and
                N right singular vectors to be computed in VT
*/

		iu = 1;

/*              WORK(IU) is N by N */

		ldwrku = *n;
		ir = iu + ldwrku * *n;
		if (*lwork >= *m * *n + *n * *n + *n * 3) {

/*                 WORK(IR) is M by N */

		    ldwrkr = *m;
		} else {
		    ldwrkr = (*lwork - *n * *n - *n * 3) / *n;
		}
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*
                Compute A=Q*R
                (CWorkspace: need N*N+2*N, prefer M*N+N+N*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy R to WORK( IR ), zeroing out below it */

		clacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		claset_("L", &i__1, &i__2, &c_b55, &c_b55, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in WORK(IR)
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+2*N*NB)
                (RWorkspace: need N)
*/

		i__1 = *lwork - nwork + 1;
		cgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of R in WORK(IRU) and computing right singular vectors
                of R in WORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by the left singular vectors of R
                (CWorkspace: need 2*N*N+3*N, prefer M*N+N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__1 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by the right singular vectors of R
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*
                Multiply Q in A by left singular vectors of R in
                WORK(IU), storing result in WORK(IR) and copying to A
                (CWorkspace: need 2*N*N, prefer N*N+M*N)
                (RWorkspace: 0)
*/

		i__1 = *m;
		i__2 = ldwrkr;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrkr);
		    cgemm_("N", "N", &chunk, n, n, &c_b56, &a[i__ + a_dim1],
			    lda, &work[iu], &ldwrku, &c_b55, &work[ir], &
			    ldwrkr);
		    clacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
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
                (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		clacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		claset_("L", &i__2, &i__1, &c_b55, &c_b55, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in WORK(IR)
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB)
                (RWorkspace: need N)
*/

		i__2 = *lwork - nwork + 1;
		cgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in A by left singular vectors of R in
                WORK(IR), storing result in U
                (CWorkspace: need N*N)
                (RWorkspace: 0)
*/

		clacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		cgemm_("N", "N", m, n, n, &c_b56, &a[a_offset], lda, &work[ir]
			, &ldwrkr, &c_b55, &u[u_offset], ldu);

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
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		clacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Generate Q in U
                (CWorkspace: need N+M, prefer N+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cungqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork],
			 &i__2, &ierr);

/*              Produce R in A, zeroing out below it */

		i__2 = *n - 1;
		i__1 = *n - 1;
		claset_("L", &i__2, &i__1, &c_b55, &c_b55, &a[a_dim1 + 2],
			lda);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*
                Bidiagonalize R in A
                (CWorkspace: need 3*N, prefer 2*N+2*N*NB)
                (RWorkspace: need N)
*/

		i__2 = *lwork - nwork + 1;
		cgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by left singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of R
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in U by left singular vectors of R in
                WORK(IU), storing result in A
                (CWorkspace: need N*N)
                (RWorkspace: 0)
*/

		cgemm_("N", "N", m, n, n, &c_b56, &u[u_offset], ldu, &work[iu]
			, &ldwrku, &c_b55, &a[a_offset], lda);

/*              Copy left singular vectors of A from A to U */

		clacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);

	    }

	} else if (*m >= mnthr2) {

/*
             MNTHR2 <= M < MNTHR1

             Path 5 (M much larger than N, but not as much as MNTHR1)
             Reduce to bidiagonal form without QR decomposition, use
             CUNGBR and matrix multiplication to compute singular vectors
*/

	    ie = 1;
	    nrwork = ie + *n;
	    itauq = 1;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*
             Bidiagonalize A
             (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
             (RWorkspace: need N)
*/

	    i__2 = *lwork - nwork + 1;
	    cgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		sbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		clacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &work[
			nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *n * 3) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		} else {

/*                 WORK(IU) is LDWRKU by N */

		    ldwrku = (*lwork - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in WORK(IU), copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		clarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &work[iu]
			, &ldwrku, &rwork[nrwork]);
		clacpy_("F", n, n, &work[iu], &ldwrku, &vt[vt_offset], ldvt);

/*
                Multiply Q in A by real matrix RWORK(IRU), storing the
                result in WORK(IU), copying to A
                (CWorkspace: need N*N, prefer M*N)
                (Rworkspace: need 3*N*N, prefer N*N+2*M*N)
*/

		nrwork = irvt;
		i__2 = *m;
		i__1 = ldwrku;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			i__1) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrku);
		    clacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru], n,
			    &work[iu], &ldwrku, &rwork[nrwork]);
		    clacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ +
			    a_dim1], lda);
/* L20: */
		}

	    } else if (wntqs) {

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		clacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		clacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cungbr_("Q", m, n, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		clarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		clacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need N*N+2*M*N)
*/

		nrwork = irvt;
		clacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset],
			 lda, &rwork[nrwork]);
		clacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    } else {

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		clacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		clacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		clarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		clacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: 0)
                (Rworkspace: need 3*N*N)
*/

		nrwork = irvt;
		clacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset],
			 lda, &rwork[nrwork]);
		clacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    }

	} else {

/*
             M .LT. MNTHR2

             Path 6 (M at least N, but not much larger)
             Reduce to bidiagonal form without QR decomposition
             Use CUNMBR to compute singular vectors
*/

	    ie = 1;
	    nrwork = ie + *n;
	    itauq = 1;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*
             Bidiagonalize A
             (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB)
             (RWorkspace: need N)
*/

	    i__1 = *lwork - nwork + 1;
	    cgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__1, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		sbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		if (*lwork >= *m * *n + *n * 3) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		} else {

/*                 WORK( IU ) is LDWRKU by N */

		    ldwrku = (*lwork - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: need 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

		if (*lwork >= *m * *n + *n * 3) {

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by left singular vectors of A, copying
                to A
                (Cworkspace: need M*N+2*N, prefer M*N+N+N*NB)
                (Rworkspace: need 0)
*/

		    claset_("F", m, n, &c_b55, &c_b55, &work[iu], &ldwrku);
		    clacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		    i__1 = *lwork - nwork + 1;
		    cunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			    ierr);
		    clacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {

/*
                   Generate Q in A
                   (Cworkspace: need 2*N, prefer N+N*NB)
                   (Rworkspace: need 0)
*/

		    i__1 = *lwork - nwork + 1;
		    cungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
			    work[nwork], &i__1, &ierr);

/*
                   Multiply Q in A by real matrix RWORK(IRU), storing the
                   result in WORK(IU), copying to A
                   (CWorkspace: need N*N, prefer M*N)
                   (Rworkspace: need 3*N*N, prefer N*N+2*M*N)
*/

		    nrwork = irvt;
		    i__1 = *m;
		    i__2 = ldwrku;
		    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			     i__2) {
/* Computing MIN */
			i__3 = *m - i__ + 1;
			chunk = min(i__3,ldwrku);
			clacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru],
				 n, &work[iu], &ldwrku, &rwork[nrwork]);
			clacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ +
				a_dim1], lda);
/* L30: */
		    }
		}

	    } else if (wntqs) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		claset_("F", m, n, &c_b55, &c_b55, &u[u_offset], ldu);
		clacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
	    } else {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		sbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*              Set the right corner of U to identity matrix */

		claset_("F", m, m, &c_b55, &c_b55, &u[u_offset], ldu);
		i__2 = *m - *n;
		i__1 = *m - *n;
		claset_("F", &i__2, &i__1, &c_b55, &c_b56, &u[*n + 1 + (*n +
			1) * u_dim1], ldu);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
	    }

	}

    } else {

/*
          A has more columns than rows. If A has sufficiently more
          columns than rows, first reduce using the LQ decomposition
          (if sufficient workspace available)
*/

	if (*n >= mnthr1) {

	    if (wntqn) {

/*
                Path 1t (N much larger than M, JOBZ='N')
                No singular vectors to be computed
*/

		itau = 1;
		nwork = itau + *m;

/*
                Compute A=L*Q
                (CWorkspace: need 2*M, prefer M+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Zero out above L */

		i__2 = *m - 1;
		i__1 = *m - 1;
		claset_("U", &i__2, &i__1, &c_b55, &c_b55, &a[(a_dim1 << 1) +
			1], lda);
		ie = 1;
		itauq = 1;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in A
                (CWorkspace: need 3*M, prefer 2*M+2*M*NB)
                (RWorkspace: need M)
*/

		i__2 = *lwork - nwork + 1;
		cgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		nrwork = ie + *m;

/*
                Perform bidiagonal SVD, compute singular values only
                (CWorkspace: 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("U", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);

	    } else if (wntqo) {

/*
                Path 2t (N much larger than M, JOBZ='O')
                M right singular vectors to be overwritten on A and
                M left singular vectors to be computed in U
*/

		ivt = 1;
		ldwkvt = *m;

/*              WORK(IVT) is M by M */

		il = ivt + ldwkvt * *m;
		if (*lwork >= *m * *n + *m * *m + *m * 3) {

/*                 WORK(IL) M by N */

		    ldwrkl = *m;
		    chunk = *n;
		} else {

/*                 WORK(IL) is M by CHUNK */

		    ldwrkl = *m;
		    chunk = (*lwork - *m * *m - *m * 3) / *m;
		}
		itau = il + ldwrkl * chunk;
		nwork = itau + *m;

/*
                Compute A=L*Q
                (CWorkspace: need 2*M, prefer M+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy L to WORK(IL), zeroing about above it */

		clacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		claset_("U", &i__2, &i__1, &c_b55, &c_b55, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		cunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in WORK(IL)
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
                (RWorkspace: need M)
*/

		i__2 = *lwork - nwork + 1;
		cgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		sbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by the left singular vectors of L
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by the right singular vectors of L
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__2 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply right singular vectors of L in WORK(IL) by Q
                in A, storing result in WORK(IL) and copying to A
                (CWorkspace: need 2*M*M, prefer M*M+M*N))
                (RWorkspace: 0)
*/

		i__2 = *n;
		i__1 = chunk;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			i__1) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    cgemm_("N", "N", m, &blk, m, &c_b56, &work[ivt], m, &a[
			    i__ * a_dim1 + 1], lda, &c_b55, &work[il], &
			    ldwrkl);
		    clacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1
			    + 1], lda);
/* L40: */
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
                (CWorkspace: need 2*M, prefer M+M*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy L to WORK(IL), zeroing out above it */

		clacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		claset_("U", &i__1, &i__2, &c_b55, &c_b55, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in WORK(IL)
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
                (RWorkspace: need M)
*/

		i__1 = *lwork - nwork + 1;
		cgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		sbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by left singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*
                Copy VT to WORK(IL), multiply right singular vectors of L
                in WORK(IL) by Q in A, storing result in VT
                (CWorkspace: need M*M)
                (RWorkspace: 0)
*/

		clacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		cgemm_("N", "N", m, n, m, &c_b56, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b55, &vt[vt_offset], ldvt);

	    } else if (wntqa) {

/*
                Path 9t (N much larger than M, JOBZ='A')
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
                (CWorkspace: need 2*M, prefer M+M*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		clacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Generate Q in VT
                (CWorkspace: need M+N, prefer M+N*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cunglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__1, &ierr);

/*              Produce L in A, zeroing out above it */

		i__1 = *m - 1;
		i__2 = *m - 1;
		claset_("U", &i__1, &i__2, &c_b55, &c_b55, &a[(a_dim1 << 1) +
			1], lda);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*
                Bidiagonalize L in A
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB)
                (RWorkspace: need M)
*/

		i__1 = *lwork - nwork + 1;
		cgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		sbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of L
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by right singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		clacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__1, &
			ierr);

/*
                Multiply right singular vectors of L in WORK(IVT) by
                Q in VT, storing result in A
                (CWorkspace: need M*M)
                (RWorkspace: 0)
*/

		cgemm_("N", "N", m, n, m, &c_b56, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b55, &a[a_offset], lda);

/*              Copy right singular vectors of A from A to VT */

		clacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

	    }

	} else if (*n >= mnthr2) {

/*
             MNTHR2 <= N < MNTHR1

             Path 5t (N much larger than M, but not as much as MNTHR1)
             Reduce to bidiagonal form without QR decomposition, use
             CUNGBR and matrix multiplication to compute singular vectors
*/


	    ie = 1;
	    nrwork = ie + *m;
	    itauq = 1;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*
             Bidiagonalize A
             (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
             (RWorkspace: M)
*/

	    i__1 = *lwork - nwork + 1;
	    cgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__1, &ierr);

	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		sbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		ivt = nwork;

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		clacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*
                Generate P**H in A
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		cungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &work[
			nwork], &i__1, &ierr);

		ldwkvt = *m;
		if (*lwork >= *m * *n + *m * 3) {

/*                 WORK( IVT ) is M by N */

		    nwork = ivt + ldwkvt * *n;
		    chunk = *n;
		} else {

/*                 WORK( IVT ) is M by CHUNK */

		    chunk = (*lwork - *m * 3) / *m;
		    nwork = ivt + ldwkvt * chunk;
		}

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRVT)
                storing the result in WORK(IVT), copying to U
                (Cworkspace: need 0)
                (Rworkspace: need 2*M*M)
*/

		clacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &work[ivt], &
			ldwkvt, &rwork[nrwork]);
		clacpy_("F", m, m, &work[ivt], &ldwkvt, &u[u_offset], ldu);

/*
                Multiply RWORK(IRVT) by P**H in A, storing the
                result in WORK(IVT), copying to A
                (CWorkspace: need M*M, prefer M*N)
                (Rworkspace: need 2*M*M, prefer 2*M*N)
*/

		nrwork = iru;
		i__1 = *n;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    clarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1],
			    lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
		    clacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ *
			    a_dim1 + 1], lda);
/* L50: */
		}
	    } else if (wntqs) {

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		clacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		clacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cungbr_("P", m, n, m, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need 3*M*M)
*/

		clacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset],
			 lda, &rwork[nrwork]);
		clacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need M*M+2*M*N)
*/

		nrwork = iru;
		clarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		clacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    } else {

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		clacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		clacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		cungbr_("P", n, n, m, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need 3*M*M)
*/

		clacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset],
			 lda, &rwork[nrwork]);
		clacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need M*M+2*M*N)
*/

		clarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		clacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    }

	} else {

/*
             N .LT. MNTHR2

             Path 6t (N greater than M, but not much larger)
             Reduce to bidiagonal form without LQ decomposition
             Use CUNMBR to compute singular vectors
*/

	    ie = 1;
	    nrwork = ie + *m;
	    itauq = 1;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*
             Bidiagonalize A
             (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
             (RWorkspace: M)
*/

	    i__2 = *lwork - nwork + 1;
	    cgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		sbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3) {

/*                 WORK( IVT ) is M by N */

		    claset_("F", m, n, &c_b55, &c_b55, &work[ivt], &ldwkvt);
		    nwork = ivt + ldwkvt * *n;
		} else {

/*                 WORK( IVT ) is M by CHUNK */

		    chunk = (*lwork - *m * 3) / *m;
		    nwork = ivt + ldwkvt * chunk;
		}

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: need 0)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *m * 3) {

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by right singular vectors of A,
                copying to A
                (Cworkspace: need M*N+2*M, prefer M*N+M+M*NB)
                (Rworkspace: need 0)
*/

		    clacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		    i__2 = *lwork - nwork + 1;
		    cunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2,
			    &ierr);
		    clacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda);
		} else {

/*
                   Generate P**H in A
                   (Cworkspace: need 2*M, prefer M+M*NB)
                   (Rworkspace: need 0)
*/

		    i__2 = *lwork - nwork + 1;
		    cungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
			    work[nwork], &i__2, &ierr);

/*
                   Multiply Q in A by real matrix RWORK(IRU), storing the
                   result in WORK(IU), copying to A
                   (CWorkspace: need M*M, prefer M*N)
                   (Rworkspace: need 3*M*M, prefer M*M+2*M*N)
*/

		    nrwork = iru;
		    i__2 = *n;
		    i__1 = chunk;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *n - i__ + 1;
			blk = min(i__3,chunk);
			clarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1]
				, lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
			clacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ *
				a_dim1 + 1], lda);
/* L60: */
		    }
		}
	    } else if (wntqs) {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		claset_("F", m, n, &c_b55, &c_b55, &vt[vt_offset], ldvt);
		clacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else {

/*
                Perform bidiagonal SVD, computing left singular vectors
                of bidiagonal matrix in RWORK(IRU) and computing right
                singular vectors of bidiagonal matrix in RWORK(IRVT)
                (CWorkspace: need 0)
                (RWorkspace: need BDSPAC)
*/

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;

		sbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		clacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		cunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Set the right corner of VT to identity matrix */

		i__1 = *n - *m;
		i__2 = *n - *m;
		claset_("F", &i__1, &i__2, &c_b55, &c_b56, &vt[*m + 1 + (*m +
			1) * vt_dim1], ldvt);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
                (RWorkspace: M*M)
*/

		claset_("F", n, n, &c_b55, &c_b55, &vt[vt_offset], ldvt);
		clacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		cunmbr_("P", "R", "C", n, n, m, &a[a_offset], lda, &work[
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

    work[1].r = (real) maxwrk, work[1].i = 0.f;

    return 0;

/*     End of CGESDD */

} /* cgesdd_ */

/* Subroutine */ int cgesv_(integer *n, integer *nrhs, complex *a, integer *
	lda, integer *ipiv, complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern /* Subroutine */ int cgetrf_(integer *, integer *, complex *,
	    integer *, integer *, integer *), xerbla_(char *, integer *), cgetrs_(char *, integer *, integer *, complex *, integer
	    *, integer *, complex *, integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    CGESV computes the solution to a complex system of linear equations
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (output) INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX array, dimension (LDB,NRHS)
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
	xerbla_("CGESV ", &i__1);
	return 0;
    }

/*     Compute the LU factorization of A. */

    cgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);
    if (*info == 0) {

/*        Solve the system A*X = B, overwriting B with X. */

	cgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &b[
		b_offset], ldb, info);
    }
    return 0;

/*     End of CGESV */

} /* cgesv_ */

/* Subroutine */ int cgetf2_(integer *m, integer *n, complex *a, integer *lda,
	 integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void c_div(complex *, complex *, complex *);

    /* Local variables */
    static integer j, jp;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), cgeru_(integer *, integer *, complex *, complex *,
	    integer *, complex *, integer *, complex *, integer *), cswap_(
	    integer *, complex *, integer *, complex *, integer *);
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGETF2 computes an LU factorization of a general m-by-n matrix A
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CGETF2", &i__1);
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
	jp = j - 1 + icamax_(&i__2, &a[j + j * a_dim1], &c__1);
	ipiv[j] = jp;
	i__2 = jp + j * a_dim1;
	if (a[i__2].r != 0.f || a[i__2].i != 0.f) {

/*           Apply the interchange to columns 1:N. */

	    if (jp != j) {
		cswap_(n, &a[j + a_dim1], lda, &a[jp + a_dim1], lda);
	    }

/*           Compute elements J+1:M of J-th column. */

	    if (j < *m) {
		i__2 = *m - j;
		c_div(&q__1, &c_b56, &a[j + j * a_dim1]);
		cscal_(&i__2, &q__1, &a[j + 1 + j * a_dim1], &c__1);
	    }

	} else if (*info == 0) {

	    *info = j;
	}

	if (j < min(*m,*n)) {

/*           Update trailing submatrix. */

	    i__2 = *m - j;
	    i__3 = *n - j;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgeru_(&i__2, &i__3, &q__1, &a[j + 1 + j * a_dim1], &c__1, &a[j +
		    (j + 1) * a_dim1], lda, &a[j + 1 + (j + 1) * a_dim1], lda)
		    ;
	}
/* L10: */
    }
    return 0;

/*     End of CGETF2 */

} /* cgetf2_ */

/* Subroutine */ int cgetrf_(integer *m, integer *n, complex *a, integer *lda,
	 integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1;

    /* Local variables */
    static integer i__, j, jb, nb;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *);
    static integer iinfo;
    extern /* Subroutine */ int ctrsm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *), cgetf2_(integer *,
	    integer *, complex *, integer *, integer *, integer *), xerbla_(
	    char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int claswp_(integer *, complex *, integer *,
	    integer *, integer *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGETRF computes an LU factorization of a general M-by-N matrix A
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CGETRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "CGETRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    if (nb <= 1 || nb >= min(*m,*n)) {

/*        Use unblocked code. */

	cgetf2_(m, n, &a[a_offset], lda, &ipiv[1], info);
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
	    cgetf2_(&i__3, &jb, &a[j + j * a_dim1], lda, &ipiv[j], &iinfo);

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
	    claswp_(&i__3, &a[a_offset], lda, &j, &i__4, &ipiv[1], &c__1);

	    if (j + jb <= *n) {

/*              Apply interchanges to columns J+JB:N. */

		i__3 = *n - j - jb + 1;
		i__4 = j + jb - 1;
		claswp_(&i__3, &a[(j + jb) * a_dim1 + 1], lda, &j, &i__4, &
			ipiv[1], &c__1);

/*              Compute block row of U. */

		i__3 = *n - j - jb + 1;
		ctrsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
			c_b56, &a[j + j * a_dim1], lda, &a[j + (j + jb) *
			a_dim1], lda);
		if (j + jb <= *m) {

/*                 Update trailing submatrix. */

		    i__3 = *m - j - jb + 1;
		    i__4 = *n - j - jb + 1;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "No transpose", &i__3, &i__4, &jb,
			    &q__1, &a[j + jb + j * a_dim1], lda, &a[j + (j +
			    jb) * a_dim1], lda, &c_b56, &a[j + jb + (j + jb) *
			     a_dim1], lda);
		}
	    }
/* L20: */
	}
    }
    return 0;

/*     End of CGETRF */

} /* cgetrf_ */

/* Subroutine */ int cgetrs_(char *trans, integer *n, integer *nrhs, complex *
	a, integer *lda, integer *ipiv, complex *b, integer *ldb, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrsm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *), xerbla_(char *,
	    integer *), claswp_(integer *, complex *, integer *,
	    integer *, integer *, integer *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CGETRS solves a system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B
    with a general N-by-N matrix A using the LU factorization computed
    by CGETRF.

    Arguments
    =========

    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B     (No transpose)
            = 'T':  A**T * X = B  (Transpose)
            = 'C':  A**H * X = B  (Conjugate transpose)

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) COMPLEX array, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U
            as computed by CGETRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from CGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX array, dimension (LDB,NRHS)
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
	xerbla_("CGETRS", &i__1);
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

	claswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);

/*        Solve L*X = B, overwriting B with X. */

	ctrsm_("Left", "Lower", "No transpose", "Unit", n, nrhs, &c_b56, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	ctrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b56, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A**T * X = B  or A**H * X = B.

          Solve U'*X = B, overwriting B with X.
*/

	ctrsm_("Left", "Upper", trans, "Non-unit", n, nrhs, &c_b56, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	ctrsm_("Left", "Lower", trans, "Unit", n, nrhs, &c_b56, &a[a_offset],
		lda, &b[b_offset], ldb);

/*        Apply row interchanges to the solution vectors. */

	claswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return 0;

/*     End of CGETRS */

} /* cgetrs_ */

/* Subroutine */ int cheevd_(char *jobz, char *uplo, integer *n, complex *a,
	integer *lda, real *w, complex *work, integer *lwork, real *rwork,
	integer *lrwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real eps;
    static integer inde;
    static real anrm;
    static integer imax;
    static real rmin, rmax;
    static integer lopt;
    static real sigma;
    extern logical lsame_(char *, char *);
    static integer iinfo;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static integer lwmin, liopt;
    static logical lower;
    static integer llrwk, lropt;
    static logical wantz;
    static integer indwk2, llwrk2;
    extern doublereal clanhe_(char *, char *, integer *, complex *, integer *,
	     real *);
    static integer iscale;
    extern /* Subroutine */ int clascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, complex *, integer *, integer *), cstedc_(char *, integer *, real *, real *, complex *,
	    integer *, complex *, integer *, real *, integer *, integer *,
	    integer *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int chetrd_(char *, integer *, complex *, integer
	    *, real *, real *, complex *, complex *, integer *, integer *), clacpy_(char *, integer *, integer *, complex *, integer
	    *, complex *, integer *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real bignum;
    static integer indtau, indrwk, indwrk, liwmin;
    extern /* Subroutine */ int ssterf_(integer *, real *, real *, integer *);
    static integer lrwmin;
    extern /* Subroutine */ int cunmtr_(char *, char *, char *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *);
    static integer llwork;
    static real smlnum;
    static logical lquery;


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CHEEVD computes all eigenvalues and, optionally, eigenvectors of a
    complex Hermitian matrix A.  If eigenvectors are desired, it uses a
    divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

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

    A       (input/output) COMPLEX array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the
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

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.
            If N <= 1,                LWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LWORK must be at least N + 1.
            If JOBZ  = 'V' and N > 1, LWORK must be at least 2*N + N**2.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace/output) REAL array,
                                           dimension (LRWORK)
            On exit, if INFO = 0, RWORK(1) returns the optimal LRWORK.

    LRWORK  (input) INTEGER
            The dimension of the array RWORK.
            If N <= 1,                LRWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LRWORK must be at least N.
            If JOBZ  = 'V' and N > 1, LRWORK must be at least
                           1 + 5*N + 2*N**2.

            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the RWORK array,
            returns this value as the first entry of the RWORK array, and
            no error message related to LRWORK is issued by XERBLA.

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

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    lower = lsame_(uplo, "L");
    lquery = *lwork == -1 || *lrwork == -1 || *liwork == -1;

    *info = 0;
    if (*n <= 1) {
	lwmin = 1;
	lrwmin = 1;
	liwmin = 1;
	lopt = lwmin;
	lropt = lrwmin;
	liopt = liwmin;
    } else {
	if (wantz) {
	    lwmin = (*n << 1) + *n * *n;
/* Computing 2nd power */
	    i__1 = *n;
	    lrwmin = *n * 5 + 1 + (i__1 * i__1 << 1);
	    liwmin = *n * 5 + 3;
	} else {
	    lwmin = *n + 1;
	    lrwmin = *n;
	    liwmin = 1;
	}
	lopt = lwmin;
	lropt = lrwmin;
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
    } else if (*lrwork < lrwmin && ! lquery) {
	*info = -10;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {
	work[1].r = (real) lopt, work[1].i = 0.f;
	rwork[1] = (real) lropt;
	iwork[1] = liopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHEEVD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	i__1 = a_dim1 + 1;
	w[1] = a[i__1].r;
	if (wantz) {
	    i__1 = a_dim1 + 1;
	    a[i__1].r = 1.f, a[i__1].i = 0.f;
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

    anrm = clanhe_("M", uplo, n, &a[a_offset], lda, &rwork[1]);
    iscale = 0;
    if (anrm > 0.f && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	clascl_(uplo, &c__0, &c__0, &c_b871, &sigma, n, n, &a[a_offset], lda,
		info);
    }

/*     Call CHETRD to reduce Hermitian matrix to tridiagonal form. */

    inde = 1;
    indtau = 1;
    indwrk = indtau + *n;
    indrwk = inde + *n;
    indwk2 = indwrk + *n * *n;
    llwork = *lwork - indwrk + 1;
    llwrk2 = *lwork - indwk2 + 1;
    llrwk = *lrwork - indrwk + 1;
    chetrd_(uplo, n, &a[a_offset], lda, &w[1], &rwork[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);
/* Computing MAX */
    i__1 = indwrk;
    r__1 = (real) lopt, r__2 = (real) (*n) + work[i__1].r;
    lopt = dmax(r__1,r__2);

/*
       For eigenvalues only, call SSTERF.  For eigenvectors, first call
       CSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call CUNMTR to multiply it to the
       Householder transformations represented as Householder vectors in
       A.
*/

    if (! wantz) {
	ssterf_(n, &w[1], &rwork[inde], info);
    } else {
	cstedc_("I", n, &w[1], &rwork[inde], &work[indwrk], n, &work[indwk2],
		&llwrk2, &rwork[indrwk], &llrwk, &iwork[1], liwork, info);
	cunmtr_("L", uplo, "N", n, n, &a[a_offset], lda, &work[indtau], &work[
		indwrk], n, &work[indwk2], &llwrk2, &iinfo);
	clacpy_("A", n, n, &work[indwrk], n, &a[a_offset], lda);
/*
   Computing MAX
   Computing 2nd power
*/
	i__3 = *n;
	i__4 = indwk2;
	i__1 = lopt, i__2 = *n + i__3 * i__3 + (integer) work[i__4].r;
	lopt = max(i__1,i__2);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
	if (*info == 0) {
	    imax = *n;
	} else {
	    imax = *info - 1;
	}
	r__1 = 1.f / sigma;
	sscal_(&imax, &r__1, &w[1], &c__1);
    }

    work[1].r = (real) lopt, work[1].i = 0.f;
    rwork[1] = (real) lropt;
    iwork[1] = liopt;

    return 0;

/*     End of CHEEVD */

} /* cheevd_ */

/* Subroutine */ int chetd2_(char *uplo, integer *n, complex *a, integer *lda,
	 real *d__, real *e, complex *tau, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2, q__3, q__4;

    /* Local variables */
    static integer i__;
    static complex taui;
    extern /* Subroutine */ int cher2_(char *, integer *, complex *, complex *
	    , integer *, complex *, integer *, complex *, integer *);
    static complex alpha;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int chemv_(char *, integer *, complex *, complex *
	    , integer *, complex *, integer *, complex *, complex *, integer *
	    ), caxpy_(integer *, complex *, complex *, integer *,
	    complex *, integer *);
    static logical upper;
    extern /* Subroutine */ int clarfg_(integer *, complex *, complex *,
	    integer *, complex *), xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    CHETD2 reduces a complex Hermitian matrix A to real symmetric
    tridiagonal form T by a unitary similarity transformation:
    Q' * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is stored:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = 'U', the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the unitary
            matrix Q as a product of elementary reflectors; if UPLO
            = 'L', the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the unitary matrix Q as a product
            of elementary reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    D       (output) REAL array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) COMPLEX array, dimension (N-1)
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

    where tau is a complex scalar, and v is a complex vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
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
	xerbla_("CHETD2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

    if (upper) {

/*        Reduce the upper triangle of A */

	i__1 = *n + *n * a_dim1;
	i__2 = *n + *n * a_dim1;
	r__1 = a[i__2].r;
	a[i__1].r = r__1, a[i__1].i = 0.f;
	for (i__ = *n - 1; i__ >= 1; --i__) {

/*
             Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(1:i-1,i+1)
*/

	    i__1 = i__ + (i__ + 1) * a_dim1;
	    alpha.r = a[i__1].r, alpha.i = a[i__1].i;
	    clarfg_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &taui);
	    i__1 = i__;
	    e[i__1] = alpha.r;

	    if (taui.r != 0.f || taui.i != 0.f) {

/*              Apply H(i) from both sides to A(1:i,1:i) */

		i__1 = i__ + (i__ + 1) * a_dim1;
		a[i__1].r = 1.f, a[i__1].i = 0.f;

/*              Compute  x := tau * A * v  storing x in TAU(1:i) */

		chemv_(uplo, &i__, &taui, &a[a_offset], lda, &a[(i__ + 1) *
			a_dim1 + 1], &c__1, &c_b55, &tau[1], &c__1)
			;

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		q__3.r = -.5f, q__3.i = -0.f;
		q__2.r = q__3.r * taui.r - q__3.i * taui.i, q__2.i = q__3.r *
			taui.i + q__3.i * taui.r;
		cdotc_(&q__4, &i__, &tau[1], &c__1, &a[(i__ + 1) * a_dim1 + 1]
			, &c__1);
		q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r *
			q__4.i + q__2.i * q__4.r;
		alpha.r = q__1.r, alpha.i = q__1.i;
		caxpy_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &tau[
			1], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		q__1.r = -1.f, q__1.i = -0.f;
		cher2_(uplo, &i__, &q__1, &a[(i__ + 1) * a_dim1 + 1], &c__1, &
			tau[1], &c__1, &a[a_offset], lda);

	    } else {
		i__1 = i__ + i__ * a_dim1;
		i__2 = i__ + i__ * a_dim1;
		r__1 = a[i__2].r;
		a[i__1].r = r__1, a[i__1].i = 0.f;
	    }
	    i__1 = i__ + (i__ + 1) * a_dim1;
	    i__2 = i__;
	    a[i__1].r = e[i__2], a[i__1].i = 0.f;
	    i__1 = i__ + 1;
	    i__2 = i__ + 1 + (i__ + 1) * a_dim1;
	    d__[i__1] = a[i__2].r;
	    i__1 = i__;
	    tau[i__1].r = taui.r, tau[i__1].i = taui.i;
/* L10: */
	}
	i__1 = a_dim1 + 1;
	d__[1] = a[i__1].r;
    } else {

/*        Reduce the lower triangle of A */

	i__1 = a_dim1 + 1;
	i__2 = a_dim1 + 1;
	r__1 = a[i__2].r;
	a[i__1].r = r__1, a[i__1].i = 0.f;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*
             Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(i+2:n,i)
*/

	    i__2 = i__ + 1 + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__;
/* Computing MIN */
	    i__3 = i__ + 2;
	    clarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &
		    taui);
	    i__2 = i__;
	    e[i__2] = alpha.r;

	    if (taui.r != 0.f || taui.i != 0.f) {

/*              Apply H(i) from both sides to A(i+1:n,i+1:n) */

		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute  x := tau * A * v  storing y in TAU(i:n-1) */

		i__2 = *n - i__;
		chemv_(uplo, &i__2, &taui, &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b55, &tau[
			i__], &c__1);

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		q__3.r = -.5f, q__3.i = -0.f;
		q__2.r = q__3.r * taui.r - q__3.i * taui.i, q__2.i = q__3.r *
			taui.i + q__3.i * taui.r;
		i__2 = *n - i__;
		cdotc_(&q__4, &i__2, &tau[i__], &c__1, &a[i__ + 1 + i__ *
			a_dim1], &c__1);
		q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r *
			q__4.i + q__2.i * q__4.r;
		alpha.r = q__1.r, alpha.i = q__1.i;
		i__2 = *n - i__;
		caxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
			i__], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		i__2 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cher2_(uplo, &i__2, &q__1, &a[i__ + 1 + i__ * a_dim1], &c__1,
			&tau[i__], &c__1, &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda);

	    } else {
		i__2 = i__ + 1 + (i__ + 1) * a_dim1;
		i__3 = i__ + 1 + (i__ + 1) * a_dim1;
		r__1 = a[i__3].r;
		a[i__2].r = r__1, a[i__2].i = 0.f;
	    }
	    i__2 = i__ + 1 + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = e[i__3], a[i__2].i = 0.f;
	    i__2 = i__;
	    i__3 = i__ + i__ * a_dim1;
	    d__[i__2] = a[i__3].r;
	    i__2 = i__;
	    tau[i__2].r = taui.r, tau[i__2].i = taui.i;
/* L20: */
	}
	i__1 = *n;
	i__2 = *n + *n * a_dim1;
	d__[i__1] = a[i__2].r;
    }

    return 0;

/*     End of CHETD2 */

} /* chetd2_ */

/* Subroutine */ int chetrd_(char *uplo, integer *n, complex *a, integer *lda,
	 real *d__, real *e, complex *tau, complex *work, integer *lwork,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1;

    /* Local variables */
    static integer i__, j, nb, kk, nx, iws;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    static logical upper;
    extern /* Subroutine */ int chetd2_(char *, integer *, complex *, integer
	    *, real *, real *, complex *, integer *), cher2k_(char *,
	    char *, integer *, integer *, complex *, complex *, integer *,
	    complex *, integer *, real *, complex *, integer *), clatrd_(char *, integer *, integer *, complex *, integer
	    *, real *, complex *, complex *, integer *), xerbla_(char
	    *, integer *);
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

    CHETRD reduces a complex Hermitian matrix A to real symmetric
    tridiagonal form T by a unitary similarity transformation:
    Q**H * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if UPLO = 'U', the diagonal and first superdiagonal
            of A are overwritten by the corresponding elements of the
            tridiagonal matrix T, and the elements above the first
            superdiagonal, with the array TAU, represent the unitary
            matrix Q as a product of elementary reflectors; if UPLO
            = 'L', the diagonal and first subdiagonal of A are over-
            written by the corresponding elements of the tridiagonal
            matrix T, and the elements below the first subdiagonal, with
            the array TAU, represent the unitary matrix Q as a product
            of elementary reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    D       (output) REAL array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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

    where tau is a complex scalar, and v is a complex vector with
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
    A(1:i-1,i+1), and tau in TAU(i).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(n-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
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

	nb = ilaenv_(&c__1, "CHETRD", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6,
		 (ftnlen)1);
	lwkopt = *n * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHETRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = nb, i__2 = ilaenv_(&c__3, "CHETRD", uplo, n, &c_n1, &c_n1, &
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
		nbmin = ilaenv_(&c__2, "CHETRD", uplo, n, &c_n1, &c_n1, &c_n1,
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
	    clatrd_(uplo, &i__3, &nb, &a[a_offset], lda, &e[1], &tau[1], &
		    work[1], &ldwork);

/*
             Update the unreduced submatrix A(1:i-1,1:i-1), using an
             update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cher2k_(uplo, "No transpose", &i__3, &nb, &q__1, &a[i__ * a_dim1
		    + 1], lda, &work[1], &ldwork, &c_b871, &a[a_offset], lda);

/*
             Copy superdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j - 1 + j * a_dim1;
		i__5 = j - 1;
		a[i__4].r = e[i__5], a[i__4].i = 0.f;
		i__4 = j;
		i__5 = j + j * a_dim1;
		d__[i__4] = a[i__5].r;
/* L10: */
	    }
/* L20: */
	}

/*        Use unblocked code to reduce the last or only block */

	chetd2_(uplo, &kk, &a[a_offset], lda, &d__[1], &e[1], &tau[1], &iinfo);
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
	    clatrd_(uplo, &i__3, &nb, &a[i__ + i__ * a_dim1], lda, &e[i__], &
		    tau[i__], &work[1], &ldwork);

/*
             Update the unreduced submatrix A(i+nb:n,i+nb:n), using
             an update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = *n - i__ - nb + 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cher2k_(uplo, "No transpose", &i__3, &nb, &q__1, &a[i__ + nb +
		    i__ * a_dim1], lda, &work[nb + 1], &ldwork, &c_b871, &a[
		    i__ + nb + (i__ + nb) * a_dim1], lda);

/*
             Copy subdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + 1 + j * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.f;
		i__4 = j;
		i__5 = j + j * a_dim1;
		d__[i__4] = a[i__5].r;
/* L30: */
	    }
/* L40: */
	}

/*        Use unblocked code to reduce the last or only block */

	i__1 = *n - i__ + 1;
	chetd2_(uplo, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__],
		&tau[i__], &iinfo);
    }

    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CHETRD */

} /* chetrd_ */

/* Subroutine */ int chseqr_(char *job, char *compz, integer *n, integer *ilo,
	 integer *ihi, complex *h__, integer *ldh, complex *w, complex *z__,
	integer *ldz, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4[2],
	    i__5, i__6;
    real r__1, r__2, r__3, r__4;
    complex q__1;
    char ch__1[2];

    /* Builtin functions */
    double r_imag(complex *);
    void r_cnjg(complex *, complex *);
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__, j, k, l;
    static complex s[225]	/* was [15][15] */, v[16];
    static integer i1, i2, ii, nh, nr, ns, nv;
    static complex vv[16];
    static integer itn;
    static complex tau;
    static integer its;
    static real ulp, tst1;
    static integer maxb, ierr;
    static real unfl;
    static complex temp;
    static real ovfl;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *), ccopy_(integer *, complex *, integer *,
	    complex *, integer *);
    static integer itemp;
    static real rtemp;
    static logical initz, wantt, wantz;
    static real rwork[1];
    extern doublereal slapy2_(real *, real *);
    extern /* Subroutine */ int slabad_(real *, real *), clarfg_(integer *,
	    complex *, complex *, integer *, complex *);
    extern integer icamax_(integer *, complex *, integer *);
    extern doublereal slamch_(char *), clanhs_(char *, integer *,
	    complex *, integer *, real *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), clahqr_(logical *, logical *, integer *, integer *, integer *,
	     complex *, integer *, complex *, integer *, integer *, complex *,
	     integer *, integer *), clacpy_(char *, integer *, integer *,
	    complex *, integer *, complex *, integer *), claset_(char
	    *, integer *, integer *, complex *, complex *, complex *, integer
	    *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int clarfx_(char *, integer *, integer *, complex
	    *, complex *, complex *, integer *, complex *);
    static real smlnum;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CHSEQR computes the eigenvalues of a complex upper Hessenberg
    matrix H, and, optionally, the matrices T and Z from the Schur
    decomposition H = Z T Z**H, where T is an upper triangular matrix
    (the Schur form), and Z is the unitary matrix of Schur vectors.

    Optionally Z may be postmultiplied into an input unitary matrix Q,
    so that this routine can give the Schur factorization of a matrix A
    which has been reduced to the Hessenberg form H by the unitary
    matrix Q:  A = Q*H*Q**H = (QZ)*T*(QZ)**H.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            = 'E': compute eigenvalues only;
            = 'S': compute eigenvalues and the Schur form T.

    COMPZ   (input) CHARACTER*1
            = 'N': no Schur vectors are computed;
            = 'I': Z is initialized to the unit matrix and the matrix Z
                   of Schur vectors of H is returned;
            = 'V': Z must contain an unitary matrix Q on entry, and
                   the product Q*Z is returned.

    N       (input) INTEGER
            The order of the matrix H.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that H is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to CGEBAL, and then passed to CGEHRD
            when the matrix output by CGEBAL is reduced to Hessenberg
            form. Otherwise ILO and IHI should be set to 1 and N
            respectively.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    H       (input/output) COMPLEX array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if JOB = 'S', H contains the upper triangular matrix
            T from the Schur decomposition (the Schur form). If
            JOB = 'E', the contents of H are unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    W       (output) COMPLEX array, dimension (N)
            The computed eigenvalues. If JOB = 'S', the eigenvalues are
            stored in the same order as on the diagonal of the Schur form
            returned in H, with W(i) = H(i,i).

    Z       (input/output) COMPLEX array, dimension (LDZ,N)
            If COMPZ = 'N': Z is not referenced.
            If COMPZ = 'I': on entry, Z need not be set, and on exit, Z
            contains the unitary matrix Z of the Schur vectors of H.
            If COMPZ = 'V': on entry Z must contain an N-by-N matrix Q,
            which is assumed to be equal to the unit matrix except for
            the submatrix Z(ILO:IHI,ILO:IHI); on exit Z contains Q*Z.
            Normally Q is the unitary matrix generated by CUNGHR after
            the call to CGEHRD which formed the Hessenberg matrix H.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.
            LDZ >= max(1,N) if COMPZ = 'I' or 'V'; LDZ >= 1 otherwise.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
            > 0:  if INFO = i, CHSEQR failed to compute all the
                  eigenvalues in a total of 30*(IHI-ILO+1) iterations;
                  elements 1:ilo-1 and i+1:n of W contain those
                  eigenvalues which have been successfully computed.

    =====================================================================


       Decode and test the input parameters
*/

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    /* Function Body */
    wantt = lsame_(job, "S");
    initz = lsame_(compz, "I");
    wantz = initz || lsame_(compz, "V");

    *info = 0;
    i__1 = max(1,*n);
    work[1].r = (real) i__1, work[1].i = 0.f;
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
	*info = -10;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHSEQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Initialize Z, if necessary */

    if (initz) {
	claset_("Full", n, n, &c_b55, &c_b56, &z__[z_offset], ldz);
    }

/*     Store the eigenvalues isolated by CGEBAL. */

    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	i__3 = i__ + i__ * h_dim1;
	w[i__2].r = h__[i__3].r, w[i__2].i = h__[i__3].i;
/* L10: */
    }
    i__1 = *n;
    for (i__ = *ihi + 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	i__3 = i__ + i__ * h_dim1;
	w[i__2].r = h__[i__3].r, w[i__2].i = h__[i__3].i;
/* L20: */
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }
    if (*ilo == *ihi) {
	i__1 = *ilo;
	i__2 = *ilo + *ilo * h_dim1;
	w[i__1].r = h__[i__2].r, w[i__1].i = h__[i__2].i;
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
	    i__3 = i__ + j * h_dim1;
	    h__[i__3].r = 0.f, h__[i__3].i = 0.f;
/* L30: */
	}
/* L40: */
    }
    nh = *ihi - *ilo + 1;

/*
       I1 and I2 are the indices of the first row and last column of H
       to which transformations must be applied. If eigenvalues only are
       being computed, I1 and I2 are re-set inside the main loop.
*/

    if (wantt) {
	i1 = 1;
	i2 = *n;
    } else {
	i1 = *ilo;
	i2 = *ihi;
    }

/*     Ensure that the subdiagonal elements are real. */

    i__1 = *ihi;
    for (i__ = *ilo + 1; i__ <= i__1; ++i__) {
	i__2 = i__ + (i__ - 1) * h_dim1;
	temp.r = h__[i__2].r, temp.i = h__[i__2].i;
	if (r_imag(&temp) != 0.f) {
	    r__1 = temp.r;
	    r__2 = r_imag(&temp);
	    rtemp = slapy2_(&r__1, &r__2);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.f;
	    q__1.r = temp.r / rtemp, q__1.i = temp.i / rtemp;
	    temp.r = q__1.r, temp.i = q__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		r_cnjg(&q__1, &temp);
		cscal_(&i__2, &q__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    cscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (i__ < *ihi) {
		i__2 = i__ + 1 + i__ * h_dim1;
		i__3 = i__ + 1 + i__ * h_dim1;
		q__1.r = temp.r * h__[i__3].r - temp.i * h__[i__3].i, q__1.i =
			 temp.r * h__[i__3].i + temp.i * h__[i__3].r;
		h__[i__2].r = q__1.r, h__[i__2].i = q__1.i;
	    }
	    if (wantz) {
		cscal_(&nh, &temp, &z__[*ilo + i__ * z_dim1], &c__1);
	    }
	}
/* L50: */
    }

/*
       Determine the order of the multi-shift QR algorithm to be used.

   Writing concatenation
*/
    i__4[0] = 1, a__1[0] = job;
    i__4[1] = 1, a__1[1] = compz;
    s_cat(ch__1, a__1, i__4, &c__2, (ftnlen)2);
    ns = ilaenv_(&c__4, "CHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
/* Writing concatenation */
    i__4[0] = 1, a__1[0] = job;
    i__4[1] = 1, a__1[1] = compz;
    s_cat(ch__1, a__1, i__4, &c__2, (ftnlen)2);
    maxb = ilaenv_(&c__8, "CHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if (ns <= 1 || ns > nh || maxb >= nh) {

/*        Use the standard double-shift algorithm */

	clahqr_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &w[1], ilo,
		ihi, &z__[z_offset], ldz, info);
	return 0;
    }
    maxb = max(2,maxb);
/* Computing MIN */
    i__1 = min(ns,maxb);
    ns = min(i__1,15);

/*
       Now 1 < NS <= MAXB < NH.

       Set machine-dependent constants for the stopping criterion.
       If norm(H) <= sqrt(OVFL), overflow should not occur.
*/

    unfl = slamch_("Safe minimum");
    ovfl = 1.f / unfl;
    slabad_(&unfl, &ovfl);
    ulp = slamch_("Precision");
    smlnum = unfl * (nh / ulp);

/*     ITN is the total number of multiple-shift QR iterations allowed. */

    itn = nh * 30;

/*
       The main loop begins here. I is the loop index and decreases from
       IHI to ILO in steps of at most MAXB. Each iteration of the loop
       works with the active submatrix in rows and columns L to I.
       Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
       H(L,L-1) is negligible so that the matrix splits.
*/

    i__ = *ihi;
L60:
    if (i__ < *ilo) {
	goto L180;
    }

/*
       Perform multiple-shift QR iterations on rows and columns ILO to I
       until a submatrix of order at most MAXB splits off at the bottom
       because a subdiagonal element has become negligible.
*/

    l = *ilo;
    i__1 = itn;
    for (its = 0; its <= i__1; ++its) {

/*        Look for a single small subdiagonal element. */

	i__2 = l + 1;
	for (k = i__; k >= i__2; --k) {
	    i__3 = k - 1 + (k - 1) * h_dim1;
	    i__5 = k + k * h_dim1;
	    tst1 = (r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = r_imag(&h__[k -
		    1 + (k - 1) * h_dim1]), dabs(r__2)) + ((r__3 = h__[i__5]
		    .r, dabs(r__3)) + (r__4 = r_imag(&h__[k + k * h_dim1]),
		    dabs(r__4)));
	    if (tst1 == 0.f) {
		i__3 = i__ - l + 1;
		tst1 = clanhs_("1", &i__3, &h__[l + l * h_dim1], ldh, rwork);
	    }
	    i__3 = k + (k - 1) * h_dim1;
/* Computing MAX */
	    r__2 = ulp * tst1;
	    if ((r__1 = h__[i__3].r, dabs(r__1)) <= dmax(r__2,smlnum)) {
		goto L80;
	    }
/* L70: */
	}
L80:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible. */

	    i__2 = l + (l - 1) * h_dim1;
	    h__[i__2].r = 0.f, h__[i__2].i = 0.f;
	}

/*        Exit from loop if a submatrix of order <= MAXB has split off. */

	if (l >= i__ - maxb + 1) {
	    goto L170;
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
		i__3 = ii;
		i__5 = ii + (ii - 1) * h_dim1;
		i__6 = ii + ii * h_dim1;
		r__3 = ((r__1 = h__[i__5].r, dabs(r__1)) + (r__2 = h__[i__6]
			.r, dabs(r__2))) * 1.5f;
		w[i__3].r = r__3, w[i__3].i = 0.f;
/* L90: */
	    }
	} else {

/*           Use eigenvalues of trailing submatrix of order NS as shifts. */

	    clacpy_("Full", &ns, &ns, &h__[i__ - ns + 1 + (i__ - ns + 1) *
		    h_dim1], ldh, s, &c__15);
	    clahqr_(&c_false, &c_false, &ns, &c__1, &ns, s, &c__15, &w[i__ -
		    ns + 1], &c__1, &ns, &z__[z_offset], ldz, &ierr);
	    if (ierr > 0) {

/*
                If CLAHQR failed to compute all NS eigenvalues, use the
                unconverged diagonal elements as the remaining shifts.
*/

		i__2 = ierr;
		for (ii = 1; ii <= i__2; ++ii) {
		    i__3 = i__ - ns + ii;
		    i__5 = ii + ii * 15 - 16;
		    w[i__3].r = s[i__5].r, w[i__3].i = s[i__5].i;
/* L100: */
		}
	    }
	}

/*
          Form the first column of (G-w(1)) (G-w(2)) . . . (G-w(ns))
          where G is the Hessenberg submatrix H(L:I,L:I) and w is
          the vector of shifts (stored in W). The result is
          stored in the local array V.
*/

	v[0].r = 1.f, v[0].i = 0.f;
	i__2 = ns + 1;
	for (ii = 2; ii <= i__2; ++ii) {
	    i__3 = ii - 1;
	    v[i__3].r = 0.f, v[i__3].i = 0.f;
/* L110: */
	}
	nv = 1;
	i__2 = i__;
	for (j = i__ - ns + 1; j <= i__2; ++j) {
	    i__3 = nv + 1;
	    ccopy_(&i__3, v, &c__1, vv, &c__1);
	    i__3 = nv + 1;
	    i__5 = j;
	    q__1.r = -w[i__5].r, q__1.i = -w[i__5].i;
	    cgemv_("No transpose", &i__3, &nv, &c_b56, &h__[l + l * h_dim1],
		    ldh, vv, &c__1, &q__1, v, &c__1);
	    ++nv;

/*
             Scale V(1:NV) so that max(abs(V(i))) = 1. If V is zero,
             reset it to the unit vector.
*/

	    itemp = icamax_(&nv, v, &c__1);
	    i__3 = itemp - 1;
	    rtemp = (r__1 = v[i__3].r, dabs(r__1)) + (r__2 = r_imag(&v[itemp
		    - 1]), dabs(r__2));
	    if (rtemp == 0.f) {
		v[0].r = 1.f, v[0].i = 0.f;
		i__3 = nv;
		for (ii = 2; ii <= i__3; ++ii) {
		    i__5 = ii - 1;
		    v[i__5].r = 0.f, v[i__5].i = 0.f;
/* L120: */
		}
	    } else {
		rtemp = dmax(rtemp,smlnum);
		r__1 = 1.f / rtemp;
		csscal_(&nv, &r__1, v, &c__1);
	    }
/* L130: */
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
	    i__3 = ns + 1, i__5 = i__ - k + 1;
	    nr = min(i__3,i__5);
	    if (k > l) {
		ccopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    clarfg_(&nr, v, &v[1], &c__1, &tau);
	    if (k > l) {
		i__3 = k + (k - 1) * h_dim1;
		h__[i__3].r = v[0].r, h__[i__3].i = v[0].i;
		i__3 = i__;
		for (ii = k + 1; ii <= i__3; ++ii) {
		    i__5 = ii + (k - 1) * h_dim1;
		    h__[i__5].r = 0.f, h__[i__5].i = 0.f;
/* L140: */
		}
	    }
	    v[0].r = 1.f, v[0].i = 0.f;

/*
             Apply G' from the left to transform the rows of the matrix
             in columns K to I2.
*/

	    i__3 = i2 - k + 1;
	    r_cnjg(&q__1, &tau);
	    clarfx_("Left", &nr, &i__3, v, &q__1, &h__[k + k * h_dim1], ldh, &
		    work[1]);

/*
             Apply G from the right to transform the columns of the
             matrix in rows I1 to min(K+NR,I).

   Computing MIN
*/
	    i__5 = k + nr;
	    i__3 = min(i__5,i__) - i1 + 1;
	    clarfx_("Right", &i__3, &nr, v, &tau, &h__[i1 + k * h_dim1], ldh,
		    &work[1]);

	    if (wantz) {

/*              Accumulate transformations in the matrix Z */

		clarfx_("Right", &nh, &nr, v, &tau, &z__[*ilo + k * z_dim1],
			ldz, &work[1]);
	    }
/* L150: */
	}

/*        Ensure that H(I,I-1) is real. */

	i__2 = i__ + (i__ - 1) * h_dim1;
	temp.r = h__[i__2].r, temp.i = h__[i__2].i;
	if (r_imag(&temp) != 0.f) {
	    r__1 = temp.r;
	    r__2 = r_imag(&temp);
	    rtemp = slapy2_(&r__1, &r__2);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.f;
	    q__1.r = temp.r / rtemp, q__1.i = temp.i / rtemp;
	    temp.r = q__1.r, temp.i = q__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		r_cnjg(&q__1, &temp);
		cscal_(&i__2, &q__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    cscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (wantz) {
		cscal_(&nh, &temp, &z__[*ilo + i__ * z_dim1], &c__1);
	    }
	}

/* L160: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L170:

/*
       A submatrix of order <= MAXB in rows and columns L to I has split
       off. Use the double-shift QR algorithm to handle it.
*/

    clahqr_(&wantt, &wantz, n, &l, &i__, &h__[h_offset], ldh, &w[1], ilo, ihi,
	     &z__[z_offset], ldz, info);
    if (*info > 0) {
	return 0;
    }

/*
       Decrement number of remaining iterations, and return to start of
       the main loop with a new value of I.
*/

    itn -= its;
    i__ = l - 1;
    goto L60;

L180:
    i__1 = max(1,*n);
    work[1].r = (real) i__1, work[1].i = 0.f;
    return 0;

/*     End of CHSEQR */

} /* chseqr_ */

/* Subroutine */ int clabrd_(integer *m, integer *n, integer *nb, complex *a,
	integer *lda, real *d__, real *e, complex *tauq, complex *taup,
	complex *x, integer *ldx, complex *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    complex q__1;

    /* Local variables */
    static integer i__;
    static complex alpha;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), cgemv_(char *, integer *, integer *, complex *,
	    complex *, integer *, complex *, integer *, complex *, complex *,
	    integer *), clarfg_(integer *, complex *, complex *,
	    integer *, complex *), clacgv_(integer *, complex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLABRD reduces the first NB rows and columns of a complex general
    m by n matrix A to upper or lower real bidiagonal form by a unitary
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by CGEBRD

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.

    N       (input) INTEGER
            The number of columns in the matrix A.

    NB      (input) INTEGER
            The number of leading rows and columns of A to be reduced.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit, the first NB rows and columns of the matrix are
            overwritten; the rest of the array is unchanged.
            If m >= n, elements on and below the diagonal in the first NB
              columns, with the array TAUQ, represent the unitary
              matrix Q as a product of elementary reflectors; and
              elements above the diagonal in the first NB rows, with the
              array TAUP, represent the unitary matrix P as a product
              of elementary reflectors.
            If m < n, elements below the diagonal in the first NB
              columns, with the array TAUQ, represent the unitary
              matrix Q as a product of elementary reflectors, and
              elements on and above the diagonal in the first NB rows,
              with the array TAUP, represent the unitary matrix P as
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

    TAUQ    (output) COMPLEX array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    X       (output) COMPLEX array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    LDX     (input) INTEGER
            The leading dimension of the array X. LDX >= max(1,M).

    Y       (output) COMPLEX array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    LDY     (output) INTEGER
            The leading dimension of the array Y. LDY >= max(1,N).

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

       Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex
    vectors.

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

	    i__2 = i__ - 1;
	    clacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + a_dim1], lda,
		     &y[i__ + y_dim1], ldy, &c_b56, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &x[i__ + x_dim1], ldx,
		     &a[i__ * a_dim1 + 1], &c__1, &c_b56, &a[i__ + i__ *
		    a_dim1], &c__1);

/*           Generate reflection Q(i) to annihilate A(i+1:m,i) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    clarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1, &
		    tauq[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *n) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ + (
			i__ + 1) * a_dim1], lda, &a[i__ + i__ * a_dim1], &
			c__1, &c_b55, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ +
			a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b55, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b56, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &x[i__ +
			x_dim1], ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b55, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("Conjugate transpose", &i__2, &i__3, &q__1, &a[(i__ +
			1) * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b56, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		cscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);

/*              Update A(i,i+1:n) */

		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		clacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__, &q__1, &y[i__ + 1 +
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b56, &a[i__ +
			(i__ + 1) * a_dim1], lda);
		clacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = i__ - 1;
		clacgv_(&i__2, &x[i__ + x_dim1], ldx);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("Conjugate transpose", &i__2, &i__3, &q__1, &a[(i__ +
			1) * a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b56,
			&a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		clacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*              Generate reflection P(i) to annihilate A(i,i+2:n) */

		i__2 = i__ + (i__ + 1) * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		clarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
			taup[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + (i__ + 1) * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		cgemv_("No transpose", &i__2, &i__3, &c_b56, &a[i__ + 1 + (
			i__ + 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1],
			 lda, &c_b55, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		cgemv_("Conjugate transpose", &i__2, &i__, &c_b56, &y[i__ + 1
			+ y_dim1], ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b55, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__, &q__1, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b56, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		cgemv_("No transpose", &i__2, &i__3, &c_b56, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b55, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b56, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		cscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i,i:n) */

	    i__2 = *n - i__ + 1;
	    clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &y[i__ + y_dim1], ldy,
		     &a[i__ + a_dim1], lda, &c_b56, &a[i__ + i__ * a_dim1],
		    lda);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &x[i__ + x_dim1], ldx);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("Conjugate transpose", &i__2, &i__3, &q__1, &a[i__ *
		    a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b56, &a[i__ +
		    i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*           Generate reflection P(i) to annihilate A(i,i+1:n) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    clarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
		    taup[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *m) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		cgemv_("No transpose", &i__2, &i__3, &c_b56, &a[i__ + 1 + i__
			* a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b55, &
			x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &y[i__ +
			y_dim1], ldy, &a[i__ + i__ * a_dim1], lda, &c_b55, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b56, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		cgemv_("No transpose", &i__2, &i__3, &c_b56, &a[i__ * a_dim1
			+ 1], lda, &a[i__ + i__ * a_dim1], lda, &c_b55, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b56, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		cscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);

/*              Update A(i+1:m,i) */

		i__2 = i__ - 1;
		clacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + 1 +
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b56, &a[i__ +
			1 + i__ * a_dim1], &c__1);
		i__2 = i__ - 1;
		clacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__, &q__1, &x[i__ + 1 +
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b56, &a[
			i__ + 1 + i__ * a_dim1], &c__1);

/*              Generate reflection Q(i) to annihilate A(i+2:m,i) */

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		clarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1,
			 &tauq[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ +
			1 + (i__ + 1) * a_dim1], lda, &a[i__ + 1 + i__ *
			a_dim1], &c__1, &c_b55, &y[i__ + 1 + i__ * y_dim1], &
			c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b55, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b56, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		cgemv_("Conjugate transpose", &i__2, &i__, &c_b56, &x[i__ + 1
			+ x_dim1], ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b55, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("Conjugate transpose", &i__, &i__2, &q__1, &a[(i__ + 1)
			 * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b56, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		cscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    } else {
		i__2 = *n - i__ + 1;
		clacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    }
/* L20: */
	}
    }
    return 0;

/*     End of CLABRD */

} /* clabrd_ */

/* Subroutine */ int clacgv_(integer *n, complex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, ioff;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    CLACGV conjugates a complex vector of length N.

    Arguments
    =========

    N       (input) INTEGER
            The length of the vector X.  N >= 0.

    X       (input/output) COMPLEX array, dimension
                           (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    INCX    (input) INTEGER
            The spacing between successive elements of X.

   =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*incx == 1) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    r_cnjg(&q__1, &x[i__]);
	    x[i__2].r = q__1.r, x[i__2].i = q__1.i;
/* L10: */
	}
    } else {
	ioff = 1;
	if (*incx < 0) {
	    ioff = 1 - (*n - 1) * *incx;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = ioff;
	    r_cnjg(&q__1, &x[ioff]);
	    x[i__2].r = q__1.r, x[i__2].i = q__1.i;
	    ioff += *incx;
/* L20: */
	}
    }
    return 0;

/*     End of CLACGV */

} /* clacgv_ */

/* Subroutine */ int clacp2_(char *uplo, integer *m, integer *n, real *a,
	integer *lda, complex *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLACP2 copies all or part of a real two-dimensional matrix A to a
    complex matrix B.

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
            The m by n matrix A.  If UPLO = 'U', only the upper trapezium
            is accessed; if UPLO = 'L', only the lower trapezium is
            accessed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (output) COMPLEX array, dimension (LDB,N)
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
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4], b[i__3].i = 0.f;
/* L10: */
	    }
/* L20: */
	}

    } else if (lsame_(uplo, "L")) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4], b[i__3].i = 0.f;
/* L30: */
	    }
/* L40: */
	}

    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4], b[i__3].i = 0.f;
/* L50: */
	    }
/* L60: */
	}
    }

    return 0;

/*     End of CLACP2 */

} /* clacp2_ */

/* Subroutine */ int clacpy_(char *uplo, integer *m, integer *n, complex *a,
	integer *lda, complex *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;

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

    CLACPY copies all or part of a two-dimensional matrix A to another
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

    A       (input) COMPLEX array, dimension (LDA,N)
            The m by n matrix A.  If UPLO = 'U', only the upper trapezium
            is accessed; if UPLO = 'L', only the lower trapezium is
            accessed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (output) COMPLEX array, dimension (LDB,N)
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
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4].r, b[i__3].i = a[i__4].i;
/* L10: */
	    }
/* L20: */
	}

    } else if (lsame_(uplo, "L")) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4].r, b[i__3].i = a[i__4].i;
/* L30: */
	    }
/* L40: */
	}

    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * a_dim1;
		b[i__3].r = a[i__4].r, b[i__3].i = a[i__4].i;
/* L50: */
	    }
/* L60: */
	}
    }

    return 0;

/*     End of CLACPY */

} /* clacpy_ */

/* Subroutine */ int clacrm_(integer *m, integer *n, complex *a, integer *lda,
	 real *b, integer *ldb, complex *c__, integer *ldc, real *rwork)
{
    /* System generated locals */
    integer b_dim1, b_offset, a_dim1, a_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    real r__1;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLACRM performs a very simple matrix-matrix multiplication:
             C := A * B,
    where A is M by N and complex; B is N by N and real;
    C is M by N and complex.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A and of the matrix C.
            M >= 0.

    N       (input) INTEGER
            The number of columns and rows of the matrix B and
            the number of columns of the matrix C.
            N >= 0.

    A       (input) COMPLEX array, dimension (LDA, N)
            A contains the M by N matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >=max(1,M).

    B       (input) REAL array, dimension (LDB, N)
            B contains the N by N matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B. LDB >=max(1,N).

    C       (input) COMPLEX array, dimension (LDC, N)
            C contains the M by N matrix C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >=max(1,N).

    RWORK   (workspace) REAL array, dimension (2*M*N)

    =====================================================================


       Quick return if possible.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --rwork;

    /* Function Body */
    if (*m == 0 || *n == 0) {
	return 0;
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    rwork[(j - 1) * *m + i__] = a[i__3].r;
/* L10: */
	}
/* L20: */
    }

    l = *m * *n + 1;
    sgemm_("N", "N", m, n, n, &c_b871, &rwork[1], m, &b[b_offset], ldb, &
	    c_b1101, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = l + (j - 1) * *m + i__ - 1;
	    c__[i__3].r = rwork[i__4], c__[i__3].i = 0.f;
/* L30: */
	}
/* L40: */
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    rwork[(j - 1) * *m + i__] = r_imag(&a[i__ + j * a_dim1]);
/* L50: */
	}
/* L60: */
    }
    sgemm_("N", "N", m, n, n, &c_b871, &rwork[1], m, &b[b_offset], ldb, &
	    c_b1101, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = i__ + j * c_dim1;
	    r__1 = c__[i__4].r;
	    i__5 = l + (j - 1) * *m + i__ - 1;
	    q__1.r = r__1, q__1.i = rwork[i__5];
	    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L70: */
	}
/* L80: */
    }

    return 0;

/*     End of CLACRM */

} /* clacrm_ */

/* Complex */ VOID cladiv_(complex * ret_val, complex *x, complex *y)
{
    /* System generated locals */
    real r__1, r__2, r__3, r__4;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    static real zi, zr;
    extern /* Subroutine */ int sladiv_(real *, real *, real *, real *, real *
	    , real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    CLADIV := X / Y, where X and Y are complex.  The computation of X / Y
    will not overflow on an intermediary step unless the results
    overflows.

    Arguments
    =========

    X       (input) COMPLEX
    Y       (input) COMPLEX
            The complex scalars X and Y.

    =====================================================================
*/


    r__1 = x->r;
    r__2 = r_imag(x);
    r__3 = y->r;
    r__4 = r_imag(y);
    sladiv_(&r__1, &r__2, &r__3, &r__4, &zr, &zi);
    q__1.r = zr, q__1.i = zi;
     ret_val->r = q__1.r,  ret_val->i = q__1.i;

    return ;

/*     End of CLADIV */

} /* cladiv_ */

/* Subroutine */ int claed0_(integer *qsiz, integer *n, real *d__, real *e,
	complex *q, integer *ldq, complex *qstore, integer *ldqs, real *rwork,
	 integer *iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, qstore_dim1, qstore_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, k, ll, iq, lgn, msd2, smm1, spm1, spm2;
    static real temp;
    static integer curr, iperm;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *,
	    complex *, integer *);
    static integer indxq, iwrem;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *,
	    integer *);
    static integer iqptr;
    extern /* Subroutine */ int claed7_(integer *, integer *, integer *,
	    integer *, integer *, integer *, real *, complex *, integer *,
	    real *, integer *, real *, integer *, integer *, integer *,
	    integer *, integer *, real *, complex *, real *, integer *,
	    integer *);
    static integer tlvls;
    extern /* Subroutine */ int clacrm_(integer *, integer *, complex *,
	    integer *, real *, integer *, complex *, integer *, real *);
    static integer igivcl;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static integer igivnm, submat, curprb, subpbs, igivpt, curlvl, matsiz,
	    iprmpt, smlsiz;
    extern /* Subroutine */ int ssteqr_(char *, integer *, real *, real *,
	    real *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    Using the divide and conquer method, CLAED0 computes all eigenvalues
    of a symmetric tridiagonal matrix which is one diagonal block of
    those from reducing a dense or band Hermitian matrix and
    corresponding eigenvectors of the dense or band matrix.

    Arguments
    =========

    QSIZ   (input) INTEGER
           The dimension of the unitary matrix used to reduce
           the full matrix to tridiagonal form.  QSIZ >= N if ICOMPQ = 1.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D      (input/output) REAL array, dimension (N)
           On entry, the diagonal elements of the tridiagonal matrix.
           On exit, the eigenvalues in ascending order.

    E      (input/output) REAL array, dimension (N-1)
           On entry, the off-diagonal elements of the tridiagonal matrix.
           On exit, E has been destroyed.

    Q      (input/output) COMPLEX array, dimension (LDQ,N)
           On entry, Q must contain an QSIZ x N matrix whose columns
           unitarily orthonormal. It is a part of the unitary matrix
           that reduces the full dense Hermitian matrix to a
           (reducible) symmetric tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    IWORK  (workspace) INTEGER array,
           the dimension of IWORK must be at least
                        6 + 6*N + 5*N*lg N
                        ( lg( N ) = smallest integer k
                                    such that 2^k >= N )

    RWORK  (workspace) REAL array,
                                 dimension (1 + 3*N + 2*N*lg N + 3*N**2)
                          ( lg( N ) = smallest integer k
                                      such that 2^k >= N )

    QSTORE (workspace) COMPLEX array, dimension (LDQS, N)
           Used to store parts of
           the eigenvector matrix when the updating matrix multiplies
           take place.

    LDQS   (input) INTEGER
           The leading dimension of the array QSTORE.
           LDQS >= max(1,N).

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    =====================================================================

    Warning:      N could be as big as QSIZ!


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
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

/*
       IF( ICOMPQ .LT. 0 .OR. ICOMPQ .GT. 2 ) THEN
          INFO = -1
       ELSE IF( ( ICOMPQ .EQ. 1 ) .AND. ( QSIZ .LT. MAX( 0, N ) ) )
      $        THEN
*/
    if (*qsiz < max(0,*n)) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldq < max(1,*n)) {
	*info = -6;
    } else if (*ldqs < max(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLAED0", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "CLAED0", " ", &c__0, &c__0, &c__0, &c__0, (
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
/*     Initialize pointers */
    i__1 = subpbs;
    for (i__ = 0; i__ <= i__1; ++i__) {
	iwork[iprmpt + i__] = 1;
	iwork[igivpt + i__] = 1;
/* L50: */
    }
    iwork[iqptr] = 1;

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
	ll = iq - 1 + iwork[iqptr + curr];
	ssteqr_("I", &matsiz, &d__[submat], &e[submat], &rwork[ll], &matsiz, &
		rwork[1], info);
	clacrm_(qsiz, &matsiz, &q[submat * q_dim1 + 1], ldq, &rwork[ll], &
		matsiz, &qstore[submat * qstore_dim1 + 1], ldqs, &rwork[iwrem]
		);
/* Computing 2nd power */
	i__2 = matsiz;
	iwork[iqptr + curr + 1] = iwork[iqptr + curr] + i__2 * i__2;
	++curr;
	if (*info > 0) {
	    *info = submat * (*n + 1) + submat + matsiz - 1;
	    return 0;
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
       into an eigensystem of size MATSIZ.  CLAED7 handles the case
       when the eigenvectors of a full or band Hermitian matrix (which
       was reduced to tridiagonal form) are desired.

       I am free to use Q as a valuable working space until Loop 150.
*/

	    claed7_(&matsiz, &msd2, qsiz, &tlvls, &curlvl, &curprb, &d__[
		    submat], &qstore[submat * qstore_dim1 + 1], ldqs, &e[
		    submat + msd2 - 1], &iwork[indxq + submat], &rwork[iq], &
		    iwork[iqptr], &iwork[iprmpt], &iwork[iperm], &iwork[
		    igivpt], &iwork[igivcl], &rwork[igivnm], &q[submat *
		    q_dim1 + 1], &rwork[iwrem], &iwork[subpbs + 1], info);
	    if (*info > 0) {
		*info = submat * (*n + 1) + submat + matsiz - 1;
		return 0;
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

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = iwork[indxq + i__];
	rwork[i__] = d__[j];
	ccopy_(qsiz, &qstore[j * qstore_dim1 + 1], &c__1, &q[i__ * q_dim1 + 1]
		, &c__1);
/* L100: */
    }
    scopy_(n, &rwork[1], &c__1, &d__[1], &c__1);

    return 0;

/*     End of CLAED0 */

} /* claed0_ */

/* Subroutine */ int claed7_(integer *n, integer *cutpnt, integer *qsiz,
	integer *tlvls, integer *curlvl, integer *curpbm, real *d__, complex *
	q, integer *ldq, real *rho, integer *indxq, real *qstore, integer *
	qptr, integer *prmptr, integer *perm, integer *givptr, integer *
	givcol, real *givnum, complex *work, real *rwork, integer *iwork,
	integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, k, n1, n2, iq, iw, iz, ptr, ind1, ind2, indx, curr,
	    indxc, indxp;
    extern /* Subroutine */ int claed8_(integer *, integer *, integer *,
	    complex *, integer *, real *, real *, integer *, real *, real *,
	    complex *, integer *, real *, integer *, integer *, integer *,
	    integer *, integer *, integer *, real *, integer *), slaed9_(
	    integer *, integer *, integer *, integer *, real *, real *,
	    integer *, real *, real *, real *, real *, integer *, integer *),
	    slaeda_(integer *, integer *, integer *, integer *, integer *,
	    integer *, integer *, integer *, real *, real *, integer *, real *
	    , real *, integer *);
    static integer idlmda;
    extern /* Subroutine */ int clacrm_(integer *, integer *, complex *,
	    integer *, real *, integer *, complex *, integer *, real *),
	    xerbla_(char *, integer *), slamrg_(integer *, integer *,
	    real *, integer *, integer *, integer *);
    static integer coltyp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLAED7 computes the updated eigensystem of a diagonal
    matrix after modification by a rank-one symmetric matrix. This
    routine is used only for the eigenproblem which requires all
    eigenvalues and optionally eigenvectors of a dense or banded
    Hermitian matrix that has been reduced to tridiagonal form.

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

    CUTPNT (input) INTEGER
           Contains the location of the last eigenvalue in the leading
           sub-matrix.  min(1,N) <= CUTPNT <= N.

    QSIZ   (input) INTEGER
           The dimension of the unitary matrix used to reduce
           the full matrix to tridiagonal form.  QSIZ >= N.

    TLVLS  (input) INTEGER
           The total number of merging levels in the overall divide and
           conquer tree.

    CURLVL (input) INTEGER
           The current level in the overall merge routine,
           0 <= curlvl <= tlvls.

    CURPBM (input) INTEGER
           The current problem in the current level in the overall
           merge routine (counting from upper left to lower right).

    D      (input/output) REAL array, dimension (N)
           On entry, the eigenvalues of the rank-1-perturbed matrix.
           On exit, the eigenvalues of the repaired matrix.

    Q      (input/output) COMPLEX array, dimension (LDQ,N)
           On entry, the eigenvectors of the rank-1-perturbed matrix.
           On exit, the eigenvectors of the repaired tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    RHO    (input) REAL
           Contains the subdiagonal element used to create the rank-1
           modification.

    INDXQ  (output) INTEGER array, dimension (N)
           This contains the permutation which will reintegrate the
           subproblem just solved back into sorted order,
           ie. D( INDXQ( I = 1, N ) ) will be in ascending order.

    IWORK  (workspace) INTEGER array, dimension (4*N)

    RWORK  (workspace) REAL array,
                                   dimension (3*N+2*QSIZ*N)

    WORK   (workspace) COMPLEX array, dimension (QSIZ*N)

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

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

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
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

/*
       IF( ICOMPQ.LT.0 .OR. ICOMPQ.GT.1 ) THEN
          INFO = -1
       ELSE IF( N.LT.0 ) THEN
*/
    if (*n < 0) {
	*info = -1;
    } else if (min(1,*n) > *cutpnt || *n < *cutpnt) {
	*info = -2;
    } else if (*qsiz < *n) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLAED7", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*
       The following values are for bookkeeping purposes only.  They are
       integer pointers which indicate the portion of the workspace
       used by a particular array in SLAED2 and SLAED3.
*/

    iz = 1;
    idlmda = iz + *n;
    iw = idlmda + *n;
    iq = iw + *n;

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
	    givcol[3], &givnum[3], &qstore[1], &qptr[1], &rwork[iz], &rwork[
	    iz + *n], info);

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

    claed8_(&k, n, qsiz, &q[q_offset], ldq, &d__[1], rho, cutpnt, &rwork[iz],
	    &rwork[idlmda], &work[1], qsiz, &rwork[iw], &iwork[indxp], &iwork[
	    indx], &indxq[1], &perm[prmptr[curr]], &givptr[curr + 1], &givcol[
	    (givptr[curr] << 1) + 1], &givnum[(givptr[curr] << 1) + 1], info);
    prmptr[curr + 1] = prmptr[curr] + *n;
    givptr[curr + 1] += givptr[curr];

/*     Solve Secular Equation. */

    if (k != 0) {
	slaed9_(&k, &c__1, &k, n, &d__[1], &rwork[iq], &k, rho, &rwork[idlmda]
		, &rwork[iw], &qstore[qptr[curr]], &k, info);
	clacrm_(qsiz, &k, &work[1], qsiz, &qstore[qptr[curr]], &k, &q[
		q_offset], ldq, &rwork[iq]);
/* Computing 2nd power */
	i__1 = k;
	qptr[curr + 1] = qptr[curr] + i__1 * i__1;
	if (*info != 0) {
	    return 0;
	}

/*     Prepare the INDXQ sorting premutation. */

	n1 = k;
	n2 = *n - k;
	ind1 = 1;
	ind2 = *n;
	slamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &indxq[1]);
    } else {
	qptr[curr + 1] = qptr[curr];
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    indxq[i__] = i__;
/* L20: */
	}
    }

    return 0;

/*     End of CLAED7 */

} /* claed7_ */

/* Subroutine */ int claed8_(integer *k, integer *n, integer *qsiz, complex *
	q, integer *ldq, real *d__, real *rho, integer *cutpnt, real *z__,
	real *dlamda, complex *q2, integer *ldq2, real *w, integer *indxp,
	integer *indx, integer *indxq, integer *perm, integer *givptr,
	integer *givcol, real *givnum, integer *info)
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
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *),
	    ccopy_(integer *, complex *, integer *, complex *, integer *),
	    csrot_(integer *, complex *, integer *, complex *, integer *,
	    real *, real *), scopy_(integer *, real *, integer *, real *,
	    integer *);
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex
	    *, integer *, complex *, integer *), xerbla_(char *,
	    integer *);
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slamrg_(integer *, integer *, real *, integer
	    *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       September 30, 1994


    Purpose
    =======

    CLAED8 merges the two sets of eigenvalues together into a single
    sorted set.  Then it tries to deflate the size of the problem.
    There are two ways in which deflation can occur:  when two or more
    eigenvalues are close together or if there is a tiny element in the
    Z vector.  For each such occurrence the order of the related secular
    equation problem is reduced by one.

    Arguments
    =========

    K      (output) INTEGER
           Contains the number of non-deflated eigenvalues.
           This is the order of the related secular equation.

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    QSIZ   (input) INTEGER
           The dimension of the unitary matrix used to reduce
           the dense or band matrix to tridiagonal form.
           QSIZ >= N if ICOMPQ = 1.

    Q      (input/output) COMPLEX array, dimension (LDQ,N)
           On entry, Q contains the eigenvectors of the partially solved
           system which has been previously updated in matrix
           multiplies with other partially solved eigensystems.
           On exit, Q contains the trailing (N-K) updated eigenvectors
           (those which were deflated) in its last N-K columns.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max( 1, N ).

    D      (input/output) REAL array, dimension (N)
           On entry, D contains the eigenvalues of the two submatrices to
           be combined.  On exit, D contains the trailing (N-K) updated
           eigenvalues (those which were deflated) sorted into increasing
           order.

    RHO    (input/output) REAL
           Contains the off diagonal element associated with the rank-1
           cut which originally split the two submatrices which are now
           being recombined. RHO is modified during the computation to
           the value required by SLAED3.

    CUTPNT (input) INTEGER
           Contains the location of the last eigenvalue in the leading
           sub-matrix.  MIN(1,N) <= CUTPNT <= N.

    Z      (input) REAL array, dimension (N)
           On input this vector contains the updating vector (the last
           row of the first sub-eigenvector matrix and the first row of
           the second sub-eigenvector matrix).  The contents of Z are
           destroyed during the updating process.

    DLAMDA (output) REAL array, dimension (N)
           Contains a copy of the first K eigenvalues which will be used
           by SLAED3 to form the secular equation.

    Q2     (output) COMPLEX array, dimension (LDQ2,N)
           If ICOMPQ = 0, Q2 is not referenced.  Otherwise,
           Contains a copy of the first K eigenvectors which will be used
           by SLAED7 in a matrix multiply (SGEMM) to update the new
           eigenvectors.

    LDQ2   (input) INTEGER
           The leading dimension of the array Q2.  LDQ2 >= max( 1, N ).

    W      (output) REAL array, dimension (N)
           This will hold the first k values of the final
           deflation-altered z-vector and will be passed to SLAED3.

    INDXP  (workspace) INTEGER array, dimension (N)
           This will contain the permutation used to place deflated
           values of D at the end of the array. On output INDXP(1:K)
           points to the nondeflated D-values and INDXP(K+1:N)
           points to the deflated eigenvalues.

    INDX   (workspace) INTEGER array, dimension (N)
           This will contain the permutation used to sort the contents of
           D into ascending order.

    INDXQ  (input) INTEGER array, dimension (N)
           This contains the permutation which separately sorts the two
           sub-problems in D into ascending order.  Note that elements in
           the second half of this permutation must first have CUTPNT
           added to their values in order to be accurate.

    PERM   (output) INTEGER array, dimension (N)
           Contains the permutations (from deflation and sorting) to be
           applied to each eigenblock.

    GIVPTR (output) INTEGER
           Contains the number of Givens rotations which took place in
           this subproblem.

    GIVCOL (output) INTEGER array, dimension (2, N)
           Each pair of numbers indicates a pair of columns to take place
           in a Givens rotation.

    GIVNUM (output) REAL array, dimension (2, N)
           Each number indicates the S value to be used in the
           corresponding Givens rotation.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --d__;
    --z__;
    --dlamda;
    q2_dim1 = *ldq2;
    q2_offset = 1 + q2_dim1;
    q2 -= q2_offset;
    --w;
    --indxp;
    --indx;
    --indxq;
    --perm;
    givcol -= 3;
    givnum -= 3;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -2;
    } else if (*qsiz < *n) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -5;
    } else if (*cutpnt < min(1,*n) || *cutpnt > *n) {
	*info = -8;
    } else if (*ldq2 < max(1,*n)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLAED8", &i__1);
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
	sscal_(&n2, &c_b1150, &z__[n1p1], &c__1);
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

/*     Calculate the allowable deflation tolerance */

    imax = isamax_(n, &z__[1], &c__1);
    jmax = isamax_(n, &d__[1], &c__1);
    eps = slamch_("Epsilon");
    tol = eps * 8.f * (r__1 = d__[jmax], dabs(r__1));

/*
       If the rank-1 modifier is small enough, no more needs to be done
       -- except to reorganize Q so that its columns correspond with the
       elements in D.
*/

    if (*rho * (r__1 = z__[imax], dabs(r__1)) <= tol) {
	*k = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    perm[j] = indxq[indx[j]];
	    ccopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1]
		    , &c__1);
/* L50: */
	}
	clacpy_("A", qsiz, n, &q2[q2_dim1 + 1], ldq2, &q[q_dim1 + 1], ldq);
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
		goto L100;
	    }
	} else {
	    jlam = j;
	    goto L70;
	}
/* L60: */
    }
L70:
    ++j;
    if (j > *n) {
	goto L90;
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
	    csrot_(qsiz, &q[indxq[indx[jlam]] * q_dim1 + 1], &c__1, &q[indxq[
		    indx[j]] * q_dim1 + 1], &c__1, &c__, &s);
	    t = d__[jlam] * c__ * c__ + d__[j] * s * s;
	    d__[j] = d__[jlam] * s * s + d__[j] * c__ * c__;
	    d__[jlam] = t;
	    --k2;
	    i__ = 1;
L80:
	    if (k2 + i__ <= *n) {
		if (d__[jlam] < d__[indxp[k2 + i__]]) {
		    indxp[k2 + i__ - 1] = indxp[k2 + i__];
		    indxp[k2 + i__] = jlam;
		    ++i__;
		    goto L80;
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
    goto L70;
L90:

/*     Record the last eigenvalue. */

    ++(*k);
    w[*k] = z__[jlam];
    dlamda[*k] = d__[jlam];
    indxp[*k] = jlam;

L100:

/*
       Sort the eigenvalues and corresponding eigenvectors into DLAMDA
       and Q2 respectively.  The eigenvalues/vectors which were not
       deflated go into the first K slots of DLAMDA and Q2 respectively,
       while those which were deflated go into the last N - K slots.
*/

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jp = indxp[j];
	dlamda[j] = d__[jp];
	perm[j] = indxq[indx[jp]];
	ccopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1], &
		c__1);
/* L110: */
    }

/*
       The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively.
*/

    if (*k < *n) {
	i__1 = *n - *k;
	scopy_(&i__1, &dlamda[*k + 1], &c__1, &d__[*k + 1], &c__1);
	i__1 = *n - *k;
	clacpy_("A", qsiz, &i__1, &q2[(*k + 1) * q2_dim1 + 1], ldq2, &q[(*k +
		1) * q_dim1 + 1], ldq);
    }

    return 0;

/*     End of CLAED8 */

} /* claed8_ */

/* Subroutine */ int clahqr_(logical *wantt, logical *wantz, integer *n,
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *w,
	integer *iloz, integer *ihiz, complex *z__, integer *ldz, integer *
	info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    double r_imag(complex *);
    void c_sqrt(complex *, complex *), r_cnjg(complex *, complex *);
    double c_abs(complex *);

    /* Local variables */
    static integer i__, j, k, l, m;
    static real s;
    static complex t, u, v[2], x, y;
    static integer i1, i2;
    static complex t1;
    static real t2;
    static complex v2;
    static real h10;
    static complex h11;
    static real h21;
    static complex h22;
    static integer nh, nz;
    static complex h11s;
    static integer itn, its;
    static real ulp;
    static complex sum;
    static real tst1;
    static complex temp;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), ccopy_(integer *, complex *, integer *, complex *,
	    integer *);
    static real rtemp, rwork[1];
    extern /* Subroutine */ int clarfg_(integer *, complex *, complex *,
	    integer *, complex *);
    extern /* Complex */ VOID cladiv_(complex *, complex *, complex *);
    extern doublereal slamch_(char *), clanhs_(char *, integer *,
	    complex *, integer *, real *);
    static real smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLAHQR is an auxiliary routine called by CHSEQR to update the
    eigenvalues and Schur decomposition already computed by CHSEQR, by
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
            It is assumed that H is already upper triangular in rows and
            columns IHI+1:N, and that H(ILO,ILO-1) = 0 (unless ILO = 1).
            CLAHQR works primarily with the Hessenberg submatrix in rows
            and columns ILO to IHI, but applies transformations to all of
            H if WANTT is .TRUE..
            1 <= ILO <= max(1,IHI); IHI <= N.

    H       (input/output) COMPLEX array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if WANTT is .TRUE., H is upper triangular in rows
            and columns ILO:IHI, with any 2-by-2 diagonal blocks in
            standard form. If WANTT is .FALSE., the contents of H are
            unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    W       (output) COMPLEX array, dimension (N)
            The computed eigenvalues ILO to IHI are stored in the
            corresponding elements of W. If WANTT is .TRUE., the
            eigenvalues are stored in the same order as on the diagonal
            of the Schur form returned in H, with W(i) = H(i,i).

    ILOZ    (input) INTEGER
    IHIZ    (input) INTEGER
            Specify the rows of Z to which transformations must be
            applied if WANTZ is .TRUE..
            1 <= ILOZ <= ILO; IHI <= IHIZ <= N.

    Z       (input/output) COMPLEX array, dimension (LDZ,N)
            If WANTZ is .TRUE., on entry Z must contain the current
            matrix Z of transformations accumulated by CHSEQR, and on
            exit Z has been updated; transformations are applied only to
            the submatrix Z(ILOZ:IHIZ,ILO:IHI).
            If WANTZ is .FALSE., Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            > 0: if INFO = i, CLAHQR failed to compute all the
                 eigenvalues ILO to IHI in a total of 30*(IHI-ILO+1)
                 iterations; elements i+1:ihi of W contain those
                 eigenvalues which have been successfully computed.

    =====================================================================
*/


    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --w;
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
	i__1 = *ilo;
	i__2 = *ilo + *ilo * h_dim1;
	w[i__1].r = h__[i__2].r, w[i__1].i = h__[i__2].i;
	return 0;
    }

    nh = *ihi - *ilo + 1;
    nz = *ihiz - *iloz + 1;

/*
       Set machine-dependent constants for the stopping criterion.
       If norm(H) <= sqrt(OVFL), overflow should not occur.
*/

    ulp = slamch_("Precision");
    smlnum = slamch_("Safe minimum") / ulp;

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
       IHI to ILO in steps of 1. Each iteration of the loop works
       with the active submatrix in rows and columns L to I.
       Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
       H(L,L-1) is negligible so that the matrix splits.
*/

    i__ = *ihi;
L10:
    if (i__ < *ilo) {
	goto L130;
    }

/*
       Perform QR iterations on rows and columns ILO to I until a
       submatrix of order 1 splits off at the bottom because a
       subdiagonal element has become negligible.
*/

    l = *ilo;
    i__1 = itn;
    for (its = 0; its <= i__1; ++its) {

/*        Look for a single small subdiagonal element. */

	i__2 = l + 1;
	for (k = i__; k >= i__2; --k) {
	    i__3 = k - 1 + (k - 1) * h_dim1;
	    i__4 = k + k * h_dim1;
	    tst1 = (r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = r_imag(&h__[k -
		    1 + (k - 1) * h_dim1]), dabs(r__2)) + ((r__3 = h__[i__4]
		    .r, dabs(r__3)) + (r__4 = r_imag(&h__[k + k * h_dim1]),
		    dabs(r__4)));
	    if (tst1 == 0.f) {
		i__3 = i__ - l + 1;
		tst1 = clanhs_("1", &i__3, &h__[l + l * h_dim1], ldh, rwork);
	    }
	    i__3 = k + (k - 1) * h_dim1;
/* Computing MAX */
	    r__2 = ulp * tst1;
	    if ((r__1 = h__[i__3].r, dabs(r__1)) <= dmax(r__2,smlnum)) {
		goto L30;
	    }
/* L20: */
	}
L30:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible */

	    i__2 = l + (l - 1) * h_dim1;
	    h__[i__2].r = 0.f, h__[i__2].i = 0.f;
	}

/*        Exit from loop if a submatrix of order 1 has split off. */

	if (l >= i__) {
	    goto L120;
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

	    i__2 = i__ + (i__ - 1) * h_dim1;
	    s = (r__1 = h__[i__2].r, dabs(r__1)) * .75f;
	    i__2 = i__ + i__ * h_dim1;
	    q__1.r = s + h__[i__2].r, q__1.i = h__[i__2].i;
	    t.r = q__1.r, t.i = q__1.i;
	} else {

/*           Wilkinson's shift. */

	    i__2 = i__ + i__ * h_dim1;
	    t.r = h__[i__2].r, t.i = h__[i__2].i;
	    i__2 = i__ - 1 + i__ * h_dim1;
	    i__3 = i__ + (i__ - 1) * h_dim1;
	    r__1 = h__[i__3].r;
	    q__1.r = r__1 * h__[i__2].r, q__1.i = r__1 * h__[i__2].i;
	    u.r = q__1.r, u.i = q__1.i;
	    if (u.r != 0.f || u.i != 0.f) {
		i__2 = i__ - 1 + (i__ - 1) * h_dim1;
		q__2.r = h__[i__2].r - t.r, q__2.i = h__[i__2].i - t.i;
		q__1.r = q__2.r * .5f, q__1.i = q__2.i * .5f;
		x.r = q__1.r, x.i = q__1.i;
		q__3.r = x.r * x.r - x.i * x.i, q__3.i = x.r * x.i + x.i *
			x.r;
		q__2.r = q__3.r + u.r, q__2.i = q__3.i + u.i;
		c_sqrt(&q__1, &q__2);
		y.r = q__1.r, y.i = q__1.i;
		if (x.r * y.r + r_imag(&x) * r_imag(&y) < 0.f) {
		    q__1.r = -y.r, q__1.i = -y.i;
		    y.r = q__1.r, y.i = q__1.i;
		}
		q__3.r = x.r + y.r, q__3.i = x.i + y.i;
		cladiv_(&q__2, &u, &q__3);
		q__1.r = t.r - q__2.r, q__1.i = t.i - q__2.i;
		t.r = q__1.r, t.i = q__1.i;
	    }
	}

/*        Look for two consecutive small subdiagonal elements. */

	i__2 = l + 1;
	for (m = i__ - 1; m >= i__2; --m) {

/*
             Determine the effect of starting the single-shift QR
             iteration at row M, and see if this would make H(M,M-1)
             negligible.
*/

	    i__3 = m + m * h_dim1;
	    h11.r = h__[i__3].r, h11.i = h__[i__3].i;
	    i__3 = m + 1 + (m + 1) * h_dim1;
	    h22.r = h__[i__3].r, h22.i = h__[i__3].i;
	    q__1.r = h11.r - t.r, q__1.i = h11.i - t.i;
	    h11s.r = q__1.r, h11s.i = q__1.i;
	    i__3 = m + 1 + m * h_dim1;
	    h21 = h__[i__3].r;
	    s = (r__1 = h11s.r, dabs(r__1)) + (r__2 = r_imag(&h11s), dabs(
		    r__2)) + dabs(h21);
	    q__1.r = h11s.r / s, q__1.i = h11s.i / s;
	    h11s.r = q__1.r, h11s.i = q__1.i;
	    h21 /= s;
	    v[0].r = h11s.r, v[0].i = h11s.i;
	    v[1].r = h21, v[1].i = 0.f;
	    i__3 = m + (m - 1) * h_dim1;
	    h10 = h__[i__3].r;
	    tst1 = ((r__1 = h11s.r, dabs(r__1)) + (r__2 = r_imag(&h11s), dabs(
		    r__2))) * ((r__3 = h11.r, dabs(r__3)) + (r__4 = r_imag(&
		    h11), dabs(r__4)) + ((r__5 = h22.r, dabs(r__5)) + (r__6 =
		    r_imag(&h22), dabs(r__6))));
	    if ((r__1 = h10 * h21, dabs(r__1)) <= ulp * tst1) {
		goto L50;
	    }
/* L40: */
	}
	i__2 = l + l * h_dim1;
	h11.r = h__[i__2].r, h11.i = h__[i__2].i;
	i__2 = l + 1 + (l + 1) * h_dim1;
	h22.r = h__[i__2].r, h22.i = h__[i__2].i;
	q__1.r = h11.r - t.r, q__1.i = h11.i - t.i;
	h11s.r = q__1.r, h11s.i = q__1.i;
	i__2 = l + 1 + l * h_dim1;
	h21 = h__[i__2].r;
	s = (r__1 = h11s.r, dabs(r__1)) + (r__2 = r_imag(&h11s), dabs(r__2))
		+ dabs(h21);
	q__1.r = h11s.r / s, q__1.i = h11s.i / s;
	h11s.r = q__1.r, h11s.i = q__1.i;
	h21 /= s;
	v[0].r = h11s.r, v[0].i = h11s.i;
	v[1].r = h21, v[1].i = 0.f;
L50:

/*        Single-shift QR step */

	i__2 = i__ - 1;
	for (k = m; k <= i__2; ++k) {

/*
             The first iteration of this loop determines a reflection G
             from the vector V and applies it from left and right to H,
             thus creating a nonzero bulge below the subdiagonal.

             Each subsequent iteration determines a reflection G to
             restore the Hessenberg form in the (K-1)th column, and thus
             chases the bulge one step toward the bottom of the active
             submatrix.

             V(2) is always real before the call to CLARFG, and hence
             after the call T2 ( = T1*V(2) ) is also real.
*/

	    if (k > m) {
		ccopy_(&c__2, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    clarfg_(&c__2, v, &v[1], &c__1, &t1);
	    if (k > m) {
		i__3 = k + (k - 1) * h_dim1;
		h__[i__3].r = v[0].r, h__[i__3].i = v[0].i;
		i__3 = k + 1 + (k - 1) * h_dim1;
		h__[i__3].r = 0.f, h__[i__3].i = 0.f;
	    }
	    v2.r = v[1].r, v2.i = v[1].i;
	    q__1.r = t1.r * v2.r - t1.i * v2.i, q__1.i = t1.r * v2.i + t1.i *
		    v2.r;
	    t2 = q__1.r;

/*
             Apply G from the left to transform the rows of the matrix
             in columns K to I2.
*/

	    i__3 = i2;
	    for (j = k; j <= i__3; ++j) {
		r_cnjg(&q__3, &t1);
		i__4 = k + j * h_dim1;
		q__2.r = q__3.r * h__[i__4].r - q__3.i * h__[i__4].i, q__2.i =
			 q__3.r * h__[i__4].i + q__3.i * h__[i__4].r;
		i__5 = k + 1 + j * h_dim1;
		q__4.r = t2 * h__[i__5].r, q__4.i = t2 * h__[i__5].i;
		q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
		sum.r = q__1.r, sum.i = q__1.i;
		i__4 = k + j * h_dim1;
		i__5 = k + j * h_dim1;
		q__1.r = h__[i__5].r - sum.r, q__1.i = h__[i__5].i - sum.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
		i__4 = k + 1 + j * h_dim1;
		i__5 = k + 1 + j * h_dim1;
		q__2.r = sum.r * v2.r - sum.i * v2.i, q__2.i = sum.r * v2.i +
			sum.i * v2.r;
		q__1.r = h__[i__5].r - q__2.r, q__1.i = h__[i__5].i - q__2.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
/* L60: */
	    }

/*
             Apply G from the right to transform the columns of the
             matrix in rows I1 to min(K+2,I).

   Computing MIN
*/
	    i__4 = k + 2;
	    i__3 = min(i__4,i__);
	    for (j = i1; j <= i__3; ++j) {
		i__4 = j + k * h_dim1;
		q__2.r = t1.r * h__[i__4].r - t1.i * h__[i__4].i, q__2.i =
			t1.r * h__[i__4].i + t1.i * h__[i__4].r;
		i__5 = j + (k + 1) * h_dim1;
		q__3.r = t2 * h__[i__5].r, q__3.i = t2 * h__[i__5].i;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		sum.r = q__1.r, sum.i = q__1.i;
		i__4 = j + k * h_dim1;
		i__5 = j + k * h_dim1;
		q__1.r = h__[i__5].r - sum.r, q__1.i = h__[i__5].i - sum.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
		i__4 = j + (k + 1) * h_dim1;
		i__5 = j + (k + 1) * h_dim1;
		r_cnjg(&q__3, &v2);
		q__2.r = sum.r * q__3.r - sum.i * q__3.i, q__2.i = sum.r *
			q__3.i + sum.i * q__3.r;
		q__1.r = h__[i__5].r - q__2.r, q__1.i = h__[i__5].i - q__2.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
/* L70: */
	    }

	    if (*wantz) {

/*              Accumulate transformations in the matrix Z */

		i__3 = *ihiz;
		for (j = *iloz; j <= i__3; ++j) {
		    i__4 = j + k * z_dim1;
		    q__2.r = t1.r * z__[i__4].r - t1.i * z__[i__4].i, q__2.i =
			     t1.r * z__[i__4].i + t1.i * z__[i__4].r;
		    i__5 = j + (k + 1) * z_dim1;
		    q__3.r = t2 * z__[i__5].r, q__3.i = t2 * z__[i__5].i;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    sum.r = q__1.r, sum.i = q__1.i;
		    i__4 = j + k * z_dim1;
		    i__5 = j + k * z_dim1;
		    q__1.r = z__[i__5].r - sum.r, q__1.i = z__[i__5].i -
			    sum.i;
		    z__[i__4].r = q__1.r, z__[i__4].i = q__1.i;
		    i__4 = j + (k + 1) * z_dim1;
		    i__5 = j + (k + 1) * z_dim1;
		    r_cnjg(&q__3, &v2);
		    q__2.r = sum.r * q__3.r - sum.i * q__3.i, q__2.i = sum.r *
			     q__3.i + sum.i * q__3.r;
		    q__1.r = z__[i__5].r - q__2.r, q__1.i = z__[i__5].i -
			    q__2.i;
		    z__[i__4].r = q__1.r, z__[i__4].i = q__1.i;
/* L80: */
		}
	    }

	    if (k == m && m > l) {

/*
                If the QR step was started at row M > L because two
                consecutive small subdiagonals were found, then extra
                scaling must be performed to ensure that H(M,M-1) remains
                real.
*/

		q__1.r = 1.f - t1.r, q__1.i = 0.f - t1.i;
		temp.r = q__1.r, temp.i = q__1.i;
		r__1 = c_abs(&temp);
		q__1.r = temp.r / r__1, q__1.i = temp.i / r__1;
		temp.r = q__1.r, temp.i = q__1.i;
		i__3 = m + 1 + m * h_dim1;
		i__4 = m + 1 + m * h_dim1;
		r_cnjg(&q__2, &temp);
		q__1.r = h__[i__4].r * q__2.r - h__[i__4].i * q__2.i, q__1.i =
			 h__[i__4].r * q__2.i + h__[i__4].i * q__2.r;
		h__[i__3].r = q__1.r, h__[i__3].i = q__1.i;
		if (m + 2 <= i__) {
		    i__3 = m + 2 + (m + 1) * h_dim1;
		    i__4 = m + 2 + (m + 1) * h_dim1;
		    q__1.r = h__[i__4].r * temp.r - h__[i__4].i * temp.i,
			    q__1.i = h__[i__4].r * temp.i + h__[i__4].i *
			    temp.r;
		    h__[i__3].r = q__1.r, h__[i__3].i = q__1.i;
		}
		i__3 = i__;
		for (j = m; j <= i__3; ++j) {
		    if (j != m + 1) {
			if (i2 > j) {
			    i__4 = i2 - j;
			    cscal_(&i__4, &temp, &h__[j + (j + 1) * h_dim1],
				    ldh);
			}
			i__4 = j - i1;
			r_cnjg(&q__1, &temp);
			cscal_(&i__4, &q__1, &h__[i1 + j * h_dim1], &c__1);
			if (*wantz) {
			    r_cnjg(&q__1, &temp);
			    cscal_(&nz, &q__1, &z__[*iloz + j * z_dim1], &
				    c__1);
			}
		    }
/* L90: */
		}
	    }
/* L100: */
	}

/*        Ensure that H(I,I-1) is real. */

	i__2 = i__ + (i__ - 1) * h_dim1;
	temp.r = h__[i__2].r, temp.i = h__[i__2].i;
	if (r_imag(&temp) != 0.f) {
	    rtemp = c_abs(&temp);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.f;
	    q__1.r = temp.r / rtemp, q__1.i = temp.i / rtemp;
	    temp.r = q__1.r, temp.i = q__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		r_cnjg(&q__1, &temp);
		cscal_(&i__2, &q__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    cscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (*wantz) {
		cscal_(&nz, &temp, &z__[*iloz + i__ * z_dim1], &c__1);
	    }
	}

/* L110: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L120:

/*     H(I,I-1) is negligible: one eigenvalue has converged. */

    i__1 = i__;
    i__2 = i__ + i__ * h_dim1;
    w[i__1].r = h__[i__2].r, w[i__1].i = h__[i__2].i;

/*
       Decrement number of remaining iterations, and return to start of
       the main loop with new value of I.
*/

    itn -= its;
    i__ = l - 1;
    goto L10;

L130:
    return 0;

/*     End of CLAHQR */

} /* clahqr_ */

/* Subroutine */ int clahrd_(integer *n, integer *k, integer *nb, complex *a,
	integer *lda, complex *tau, complex *t, integer *ldt, complex *y,
	integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    complex q__1;

    /* Local variables */
    static integer i__;
    static complex ei;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), cgemv_(char *, integer *, integer *, complex *,
	    complex *, integer *, complex *, integer *, complex *, complex *,
	    integer *), ccopy_(integer *, complex *, integer *,
	    complex *, integer *), caxpy_(integer *, complex *, complex *,
	    integer *, complex *, integer *), ctrmv_(char *, char *, char *,
	    integer *, complex *, integer *, complex *, integer *), clarfg_(integer *, complex *, complex *, integer
	    *, complex *), clacgv_(integer *, complex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLAHRD reduces the first NB columns of a complex general n-by-(n-k+1)
    matrix A so that elements below the k-th subdiagonal are zero. The
    reduction is performed by a unitary similarity transformation
    Q' * A * Q. The routine returns the matrices V and T which determine
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V * T.

    This is an auxiliary routine called by CGEHRD.

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.

    K       (input) INTEGER
            The offset for the reduction. Elements below the k-th
            subdiagonal in the first NB columns are reduced to zero.

    NB      (input) INTEGER
            The number of columns to be reduced.

    A       (input/output) COMPLEX array, dimension (LDA,N-K+1)
            On entry, the n-by-(n-k+1) general matrix A.
            On exit, the elements on and above the k-th subdiagonal in
            the first NB columns are overwritten with the corresponding
            elements of the reduced matrix; the elements below the k-th
            subdiagonal, with the array TAU, represent the matrix Q as a
            product of elementary reflectors. The other columns of A are
            unchanged. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX array, dimension (NB)
            The scalar factors of the elementary reflectors. See Further
            Details.

    T       (output) COMPLEX array, dimension (LDT,NB)
            The upper triangular matrix T.

    LDT     (input) INTEGER
            The leading dimension of the array T.  LDT >= NB.

    Y       (output) COMPLEX array, dimension (LDY,NB)
            The n-by-nb matrix Y.

    LDY     (input) INTEGER
            The leading dimension of the array Y. LDY >= max(1,N).

    Further Details
    ===============

    The matrix Q is represented as a product of nb elementary reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
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
	    clacgv_(&i__2, &a[*k + i__ - 1 + a_dim1], lda);
	    i__2 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", n, &i__2, &q__1, &y[y_offset], ldy, &a[*k
		    + i__ - 1 + a_dim1], lda, &c_b56, &a[i__ * a_dim1 + 1], &
		    c__1);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &a[*k + i__ - 1 + a_dim1], lda);

/*
             Apply I - V * T' * V' to this column (call it b) from the
             left, using the last column of T as workspace

             Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
                      ( V2 )             ( b2 )

             where V1 is unit lower triangular

             w := V1' * b1
*/

	    i__2 = i__ - 1;
	    ccopy_(&i__2, &a[*k + 1 + i__ * a_dim1], &c__1, &t[*nb * t_dim1 +
		    1], &c__1);
	    i__2 = i__ - 1;
	    ctrmv_("Lower", "Conjugate transpose", "Unit", &i__2, &a[*k + 1 +
		    a_dim1], lda, &t[*nb * t_dim1 + 1], &c__1);

/*           w := w + V2'*b2 */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[*k + i__ +
		    a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b56,
		    &t[*nb * t_dim1 + 1], &c__1);

/*           w := T'*w */

	    i__2 = i__ - 1;
	    ctrmv_("Upper", "Conjugate transpose", "Non-unit", &i__2, &t[
		    t_offset], ldt, &t[*nb * t_dim1 + 1], &c__1);

/*           b2 := b2 - V2*w */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &a[*k + i__ + a_dim1],
		     lda, &t[*nb * t_dim1 + 1], &c__1, &c_b56, &a[*k + i__ +
		    i__ * a_dim1], &c__1);

/*           b1 := b1 - V1*w */

	    i__2 = i__ - 1;
	    ctrmv_("Lower", "No transpose", "Unit", &i__2, &a[*k + 1 + a_dim1]
		    , lda, &t[*nb * t_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    caxpy_(&i__2, &q__1, &t[*nb * t_dim1 + 1], &c__1, &a[*k + 1 + i__
		    * a_dim1], &c__1);

	    i__2 = *k + i__ - 1 + (i__ - 1) * a_dim1;
	    a[i__2].r = ei.r, a[i__2].i = ei.i;
	}

/*
          Generate the elementary reflector H(i) to annihilate
          A(k+i+1:n,i)
*/

	i__2 = *k + i__ + i__ * a_dim1;
	ei.r = a[i__2].r, ei.i = a[i__2].i;
	i__2 = *n - *k - i__ + 1;
/* Computing MIN */
	i__3 = *k + i__ + 1;
	clarfg_(&i__2, &ei, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[i__])
		;
	i__2 = *k + i__ + i__ * a_dim1;
	a[i__2].r = 1.f, a[i__2].i = 0.f;

/*        Compute  Y(1:n,i) */

	i__2 = *n - *k - i__ + 1;
	cgemv_("No transpose", n, &i__2, &c_b56, &a[(i__ + 1) * a_dim1 + 1],
		lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b55, &y[i__ *
		y_dim1 + 1], &c__1);
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[*k + i__ +
		a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b55, &t[
		i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	q__1.r = -1.f, q__1.i = -0.f;
	cgemv_("No transpose", n, &i__2, &q__1, &y[y_offset], ldy, &t[i__ *
		t_dim1 + 1], &c__1, &c_b56, &y[i__ * y_dim1 + 1], &c__1);
	cscal_(n, &tau[i__], &y[i__ * y_dim1 + 1], &c__1);

/*        Compute T(1:i,i) */

	i__2 = i__ - 1;
	i__3 = i__;
	q__1.r = -tau[i__3].r, q__1.i = -tau[i__3].i;
	cscal_(&i__2, &q__1, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	ctrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[t_offset], ldt,
		&t[i__ * t_dim1 + 1], &c__1)
		;
	i__2 = i__ + i__ * t_dim1;
	i__3 = i__;
	t[i__2].r = tau[i__3].r, t[i__2].i = tau[i__3].i;

/* L10: */
    }
    i__1 = *k + *nb + *nb * a_dim1;
    a[i__1].r = ei.r, a[i__1].i = ei.i;

    return 0;

/*     End of CLAHRD */

} /* clahrd_ */

doublereal clange_(char *norm, integer *m, integer *n, complex *a, integer *
	lda, real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real ret_val, r__1, r__2;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int classq_(integer *, complex *, integer *, real
	    *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    CLANGE  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    complex matrix A.

    Description
    ===========

    CLANGE returns the value

       CLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in CLANGE as described
            above.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            CLANGE is set to zero.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            CLANGE is set to zero.

    A       (input) COMPLEX array, dimension (LDA,N)
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
		r__1 = value, r__2 = c_abs(&a[i__ + j * a_dim1]);
		value = dmax(r__1,r__2);
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
		sum += c_abs(&a[i__ + j * a_dim1]);
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
		work[i__] += c_abs(&a[i__ + j * a_dim1]);
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
	    classq_(m, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANGE */

} /* clange_ */

doublereal clanhe_(char *norm, char *uplo, integer *n, complex *a, integer *
	lda, real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, absa, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int classq_(integer *, complex *, integer *, real
	    *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    CLANHE  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    complex hermitian matrix A.

    Description
    ===========

    CLANHE returns the value

       CLANHE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in CLANHE as described
            above.

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            hermitian matrix A is to be referenced.
            = 'U':  Upper triangular part of A is referenced
            = 'L':  Lower triangular part of A is referenced

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, CLANHE is
            set to zero.

    A       (input) COMPLEX array, dimension (LDA,N)
            The hermitian matrix A.  If UPLO = 'U', the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced.  If UPLO = 'L', the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced. Note that the imaginary parts of the diagonal
            elements need not be set and are assumed to be zero.

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
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__1 = value, r__2 = c_abs(&a[i__ + j * a_dim1]);
		    value = dmax(r__1,r__2);
/* L10: */
		}
/* Computing MAX */
		i__2 = j + j * a_dim1;
		r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
		value = dmax(r__2,r__3);
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
		i__2 = j + j * a_dim1;
		r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
		value = dmax(r__2,r__3);
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__1 = value, r__2 = c_abs(&a[i__ + j * a_dim1]);
		    value = dmax(r__1,r__2);
/* L30: */
		}
/* L40: */
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*        Find normI(A) ( = norm1(A), since A is hermitian). */

	value = 0.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.f;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = c_abs(&a[i__ + j * a_dim1]);
		    sum += absa;
		    work[i__] += absa;
/* L50: */
		}
		i__2 = j + j * a_dim1;
		work[j] = sum + (r__1 = a[i__2].r, dabs(r__1));
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
		i__2 = j + j * a_dim1;
		sum = work[j] + (r__1 = a[i__2].r, dabs(r__1));
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = c_abs(&a[i__ + j * a_dim1]);
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
		classq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L110: */
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		classq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, &scale, &sum);
/* L120: */
	    }
	}
	sum *= 2;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    if (a[i__2].r != 0.f) {
		i__2 = i__ + i__ * a_dim1;
		absa = (r__1 = a[i__2].r, dabs(r__1));
		if (scale < absa) {
/* Computing 2nd power */
		    r__1 = scale / absa;
		    sum = sum * (r__1 * r__1) + 1.f;
		    scale = absa;
		} else {
/* Computing 2nd power */
		    r__1 = absa / scale;
		    sum += r__1 * r__1;
		}
	    }
/* L130: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANHE */

} /* clanhe_ */

doublereal clanhs_(char *norm, integer *n, complex *a, integer *lda, real *
	work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    real ret_val, r__1, r__2;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static real sum, scale;
    extern logical lsame_(char *, char *);
    static real value;
    extern /* Subroutine */ int classq_(integer *, complex *, integer *, real
	    *, real *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    CLANHS  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    Hessenberg matrix A.

    Description
    ===========

    CLANHS returns the value

       CLANHS = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in CLANHS as described
            above.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, CLANHS is
            set to zero.

    A       (input) COMPLEX array, dimension (LDA,N)
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
		r__1 = value, r__2 = c_abs(&a[i__ + j * a_dim1]);
		value = dmax(r__1,r__2);
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
		sum += c_abs(&a[i__ + j * a_dim1]);
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
		work[i__] += c_abs(&a[i__ + j * a_dim1]);
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
	    classq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANHS */

} /* clanhs_ */

/* Subroutine */ int clarcm_(integer *m, integer *n, real *a, integer *lda,
	complex *b, integer *ldb, complex *c__, integer *ldc, real *rwork)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    real r__1;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *,
	    integer *, real *, real *, integer *, real *, integer *, real *,
	    real *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLARCM performs a very simple matrix-matrix multiplication:
             C := A * B,
    where A is M by M and real; B is M by N and complex;
    C is M by N and complex.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A and of the matrix C.
            M >= 0.

    N       (input) INTEGER
            The number of columns and rows of the matrix B and
            the number of columns of the matrix C.
            N >= 0.

    A       (input) REAL array, dimension (LDA, M)
            A contains the M by M matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >=max(1,M).

    B       (input) REAL array, dimension (LDB, N)
            B contains the M by N matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B. LDB >=max(1,M).

    C       (input) COMPLEX array, dimension (LDC, N)
            C contains the M by N matrix C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >=max(1,M).

    RWORK   (workspace) REAL array, dimension (2*M*N)

    =====================================================================


       Quick return if possible.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --rwork;

    /* Function Body */
    if (*m == 0 || *n == 0) {
	return 0;
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * b_dim1;
	    rwork[(j - 1) * *m + i__] = b[i__3].r;
/* L10: */
	}
/* L20: */
    }

    l = *m * *n + 1;
    sgemm_("N", "N", m, n, m, &c_b871, &a[a_offset], lda, &rwork[1], m, &
	    c_b1101, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = l + (j - 1) * *m + i__ - 1;
	    c__[i__3].r = rwork[i__4], c__[i__3].i = 0.f;
/* L30: */
	}
/* L40: */
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    rwork[(j - 1) * *m + i__] = r_imag(&b[i__ + j * b_dim1]);
/* L50: */
	}
/* L60: */
    }
    sgemm_("N", "N", m, n, m, &c_b871, &a[a_offset], lda, &rwork[1], m, &
	    c_b1101, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = i__ + j * c_dim1;
	    r__1 = c__[i__4].r;
	    i__5 = l + (j - 1) * *m + i__ - 1;
	    q__1.r = r__1, q__1.i = rwork[i__5];
	    c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L70: */
	}
/* L80: */
    }

    return 0;

/*     End of CLARCM */

} /* clarcm_ */

/* Subroutine */ int clarf_(char *side, integer *m, integer *n, complex *v,
	integer *incv, complex *tau, complex *c__, integer *ldc, complex *
	work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    complex q__1;

    /* Local variables */
    extern /* Subroutine */ int cgerc_(integer *, integer *, complex *,
	    complex *, integer *, complex *, integer *, complex *, integer *),
	     cgemv_(char *, integer *, integer *, complex *, complex *,
	    integer *, complex *, integer *, complex *, complex *, integer *);
    extern logical lsame_(char *, char *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLARF applies a complex elementary reflector H to a complex M-by-N
    matrix C, from either the left or the right. H is represented in the
    form

          H = I - tau * v * v'

    where tau is a complex scalar and v is a complex vector.

    If tau = 0, then H is taken to be the unit matrix.

    To apply H' (the conjugate transpose of H), supply conjg(tau) instead
    tau.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': form  H * C
            = 'R': form  C * H

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    V       (input) COMPLEX array, dimension
                       (1 + (M-1)*abs(INCV)) if SIDE = 'L'
                    or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
            The vector v in the representation of H. V is not used if
            TAU = 0.

    INCV    (input) INTEGER
            The increment between elements of v. INCV <> 0.

    TAU     (input) COMPLEX
            The value tau in the representation of H.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension
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

	if (tau->r != 0.f || tau->i != 0.f) {

/*           w := C' * v */

	    cgemv_("Conjugate transpose", m, n, &c_b56, &c__[c_offset], ldc, &
		    v[1], incv, &c_b55, &work[1], &c__1);

/*           C := C - v * w' */

	    q__1.r = -tau->r, q__1.i = -tau->i;
	    cgerc_(m, n, &q__1, &v[1], incv, &work[1], &c__1, &c__[c_offset],
		    ldc);
	}
    } else {

/*        Form  C * H */

	if (tau->r != 0.f || tau->i != 0.f) {

/*           w := C * v */

	    cgemv_("No transpose", m, n, &c_b56, &c__[c_offset], ldc, &v[1],
		    incv, &c_b55, &work[1], &c__1);

/*           C := C - w * v' */

	    q__1.r = -tau->r, q__1.i = -tau->i;
	    cgerc_(m, n, &q__1, &work[1], &c__1, &v[1], incv, &c__[c_offset],
		    ldc);
	}
    }
    return 0;

/*     End of CLARF */

} /* clarf_ */

/* Subroutine */ int clarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, complex *v, integer *ldv,
	complex *t, integer *ldt, complex *c__, integer *ldc, complex *work,
	integer *ldwork)
{
    /* System generated locals */
    integer c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1,
	    work_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, j;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *,
	    complex *, integer *), ctrmm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *), clacgv_(integer *,
	    complex *, integer *);
    static char transt[1];


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLARFB applies a complex block reflector H or its transpose H' to a
    complex M-by-N matrix C, from either the left or the right.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply H or H' from the Left
            = 'R': apply H or H' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply H (No transpose)
            = 'C': apply H' (Conjugate transpose)

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

    V       (input) COMPLEX array, dimension
                                  (LDV,K) if STOREV = 'C'
                                  (LDV,M) if STOREV = 'R' and SIDE = 'L'
                                  (LDV,N) if STOREV = 'R' and SIDE = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
            if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
            if STOREV = 'R', LDV >= K.

    T       (input) COMPLEX array, dimension (LDT,K)
            The triangular K-by-K matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by H*C or H'*C or C*H or C*H'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension (LDWORK,K)

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
	*(unsigned char *)transt = 'C';
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
		    ccopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
		    clacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L10: */
		}

/*              W := W * V1 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b56,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2 */

		    i__1 = *m - *k;
		    cgemm_("Conjugate transpose", "No transpose", n, k, &i__1,
			     &c_b56, &c__[*k + 1 + c_dim1], ldc, &v[*k + 1 +
			    v_dim1], ldv, &c_b56, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2 * W' */

		    i__1 = *m - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &i__1, n, k,
			     &q__1, &v[*k + 1 + v_dim1], ldv, &work[
			    work_offset], ldwork, &c_b56, &c__[*k + 1 +
			    c_dim1], ldc);
		}

/*              W := W * V1' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k,
			&c_b56, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i -
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L40: */
		}

/*              W := W * V1 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b56,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2 */

		    i__1 = *n - *k;
		    cgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b56, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k +
			    1 + v_dim1], ldv, &c_b56, &work[work_offset],
			    ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C2 := C2 - W * V2' */

		    i__1 = *n - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", m, &i__1, k,
			     &q__1, &work[work_offset], ldwork, &v[*k + 1 +
			    v_dim1], ldv, &c_b56, &c__[(*k + 1) * c_dim1 + 1],
			     ldc);
		}

/*              W := W * V1' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k,
			&c_b56, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
		    clacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L70: */
		}

/*              W := W * V2 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b56,
			 &v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1 */

		    i__1 = *m - *k;
		    cgemm_("Conjugate transpose", "No transpose", n, k, &i__1,
			     &c_b56, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b56, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1 * W' */

		    i__1 = *m - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &i__1, n, k,
			     &q__1, &v[v_offset], ldv, &work[work_offset],
			    ldwork, &c_b56, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k,
			&c_b56, &v[*m - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = *m - *k + j + i__ * c_dim1;
			i__4 = *m - *k + j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i -
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L100: */
		}

/*              W := W * V2 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b56,
			 &v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1 */

		    i__1 = *n - *k;
		    cgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b56, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b56, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C1 := C1 - W * V1' */

		    i__1 = *n - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", m, &i__1, k,
			     &q__1, &work[work_offset], ldwork, &v[v_offset],
			    ldv, &c_b56, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k,
			&c_b56, &v[*n - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + (*n - *k + j) * c_dim1;
			i__4 = i__ + (*n - *k + j) * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
		    clacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L130: */
		}

/*              W := W * V1' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k,
			&c_b56, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2' */

		    i__1 = *m - *k;
		    cgemm_("Conjugate transpose", "Conjugate transpose", n, k,
			     &i__1, &c_b56, &c__[*k + 1 + c_dim1], ldc, &v[(*
			    k + 1) * v_dim1 + 1], ldv, &c_b56, &work[
			    work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2' * W' */

		    i__1 = *m - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, n, k, &q__1, &v[(*k + 1) * v_dim1 + 1], ldv,
			     &work[work_offset], ldwork, &c_b56, &c__[*k + 1
			    + c_dim1], ldc);
		}

/*              W := W * V1 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b56,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i -
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L160: */
		}

/*              W := W * V1' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k,
			&c_b56, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2' */

		    i__1 = *n - *k;
		    cgemm_("No transpose", "Conjugate transpose", m, k, &i__1,
			     &c_b56, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k
			    + 1) * v_dim1 + 1], ldv, &c_b56, &work[
			    work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C2 := C2 - W * V2 */

		    i__1 = *n - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "No transpose", m, &i__1, k, &q__1,
			     &work[work_offset], ldwork, &v[(*k + 1) * v_dim1
			    + 1], ldv, &c_b56, &c__[(*k + 1) * c_dim1 + 1],
			    ldc);
		}

/*              W := W * V1 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b56,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
		    clacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L190: */
		}

/*              W := W * V2' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k,
			&c_b56, &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1' */

		    i__1 = *m - *k;
		    cgemm_("Conjugate transpose", "Conjugate transpose", n, k,
			     &i__1, &c_b56, &c__[c_offset], ldc, &v[v_offset],
			     ldv, &c_b56, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1' * W' */

		    i__1 = *m - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, n, k, &q__1, &v[v_offset], ldv, &work[
			    work_offset], ldwork, &c_b56, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b56,
			 &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = *m - *k + j + i__ * c_dim1;
			i__4 = *m - *k + j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i -
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
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
		    ccopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L220: */
		}

/*              W := W * V2' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k,
			&c_b56, &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1' */

		    i__1 = *n - *k;
		    cgemm_("No transpose", "Conjugate transpose", m, k, &i__1,
			     &c_b56, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b56, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b56, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C1 := C1 - W * V1 */

		    i__1 = *n - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "No transpose", m, &i__1, k, &q__1,
			     &work[work_offset], ldwork, &v[v_offset], ldv, &
			    c_b56, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b56,
			 &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + (*n - *k + j) * c_dim1;
			i__4 = i__ + (*n - *k + j) * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L230: */
		    }
/* L240: */
		}

	    }

	}
    }

    return 0;

/*     End of CLARFB */

} /* clarfb_ */

/* Subroutine */ int clarfg_(integer *n, complex *alpha, complex *x, integer *
	incx, complex *tau)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;
    complex q__1, q__2;

    /* Builtin functions */
    double r_imag(complex *), r_sign(real *, real *);

    /* Local variables */
    static integer j, knt;
    static real beta;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *);
    static real alphi, alphr, xnorm;
    extern doublereal scnrm2_(integer *, complex *, integer *), slapy3_(real *
	    , real *, real *);
    extern /* Complex */ VOID cladiv_(complex *, complex *, complex *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *);
    static real safmin, rsafmn;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLARFG generates a complex elementary reflector H of order n, such
    that

          H' * ( alpha ) = ( beta ),   H' * H = I.
               (   x   )   (   0  )

    where alpha and beta are scalars, with beta real, and x is an
    (n-1)-element complex vector. H is represented in the form

          H = I - tau * ( 1 ) * ( 1 v' ) ,
                        ( v )

    where tau is a complex scalar and v is a complex (n-1)-element
    vector. Note that H is not hermitian.

    If the elements of x are all zero and alpha is real, then tau = 0
    and H is taken to be the unit matrix.

    Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .

    Arguments
    =========

    N       (input) INTEGER
            The order of the elementary reflector.

    ALPHA   (input/output) COMPLEX
            On entry, the value alpha.
            On exit, it is overwritten with the value beta.

    X       (input/output) COMPLEX array, dimension
                           (1+(N-2)*abs(INCX))
            On entry, the vector x.
            On exit, it is overwritten with the vector v.

    INCX    (input) INTEGER
            The increment between elements of X. INCX > 0.

    TAU     (output) COMPLEX
            The value tau.

    =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 0) {
	tau->r = 0.f, tau->i = 0.f;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = scnrm2_(&i__1, &x[1], incx);
    alphr = alpha->r;
    alphi = r_imag(alpha);

    if (xnorm == 0.f && alphi == 0.f) {

/*        H  =  I */

	tau->r = 0.f, tau->i = 0.f;
    } else {

/*        general case */

	r__1 = slapy3_(&alphr, &alphi, &xnorm);
	beta = -r_sign(&r__1, &alphr);
	safmin = slamch_("S") / slamch_("E");
	rsafmn = 1.f / safmin;

	if (dabs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

	    knt = 0;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    csscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    alphi *= rsafmn;
	    alphr *= rsafmn;
	    if (dabs(beta) < safmin) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = scnrm2_(&i__1, &x[1], incx);
	    q__1.r = alphr, q__1.i = alphi;
	    alpha->r = q__1.r, alpha->i = q__1.i;
	    r__1 = slapy3_(&alphr, &alphi, &xnorm);
	    beta = -r_sign(&r__1, &alphr);
	    r__1 = (beta - alphr) / beta;
	    r__2 = -alphi / beta;
	    q__1.r = r__1, q__1.i = r__2;
	    tau->r = q__1.r, tau->i = q__1.i;
	    q__2.r = alpha->r - beta, q__2.i = alpha->i;
	    cladiv_(&q__1, &c_b56, &q__2);
	    alpha->r = q__1.r, alpha->i = q__1.i;
	    i__1 = *n - 1;
	    cscal_(&i__1, alpha, &x[1], incx);

/*           If ALPHA is subnormal, it may lose relative accuracy */

	    alpha->r = beta, alpha->i = 0.f;
	    i__1 = knt;
	    for (j = 1; j <= i__1; ++j) {
		q__1.r = safmin * alpha->r, q__1.i = safmin * alpha->i;
		alpha->r = q__1.r, alpha->i = q__1.i;
/* L20: */
	    }
	} else {
	    r__1 = (beta - alphr) / beta;
	    r__2 = -alphi / beta;
	    q__1.r = r__1, q__1.i = r__2;
	    tau->r = q__1.r, tau->i = q__1.i;
	    q__2.r = alpha->r - beta, q__2.i = alpha->i;
	    cladiv_(&q__1, &c_b56, &q__2);
	    alpha->r = q__1.r, alpha->i = q__1.i;
	    i__1 = *n - 1;
	    cscal_(&i__1, alpha, &x[1], incx);
	    alpha->r = beta, alpha->i = 0.f;
	}
    }

    return 0;

/*     End of CLARFG */

} /* clarfg_ */

/* Subroutine */ int clarft_(char *direct, char *storev, integer *n, integer *
	k, complex *v, integer *ldv, complex *tau, complex *t, integer *ldt)
{
    /* System generated locals */
    integer t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3, i__4;
    complex q__1;

    /* Local variables */
    static integer i__, j;
    static complex vii;
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrmv_(char *, char *, char *, integer *,
	    complex *, integer *, complex *, integer *), clacgv_(integer *, complex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLARFT forms the triangular factor T of a complex block reflector H
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

    V       (input/output) COMPLEX array, dimension
                                 (LDV,K) if STOREV = 'C'
                                 (LDV,N) if STOREV = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i).

    T       (output) COMPLEX array, dimension (LDT,K)
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
	    i__2 = i__;
	    if (tau[i__2].r == 0.f && tau[i__2].i == 0.f) {

/*              H(i)  =  I */

		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j + i__ * t_dim1;
		    t[i__3].r = 0.f, t[i__3].i = 0.f;
/* L10: */
		}
	    } else {

/*              general case */

		i__2 = i__ + i__ * v_dim1;
		vii.r = v[i__2].r, vii.i = v[i__2].i;
		i__2 = i__ + i__ * v_dim1;
		v[i__2].r = 1.f, v[i__2].i = 0.f;
		if (lsame_(storev, "C")) {

/*                 T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i) */

		    i__2 = *n - i__ + 1;
		    i__3 = i__ - 1;
		    i__4 = i__;
		    q__1.r = -tau[i__4].r, q__1.i = -tau[i__4].i;
		    cgemv_("Conjugate transpose", &i__2, &i__3, &q__1, &v[i__
			    + v_dim1], ldv, &v[i__ + i__ * v_dim1], &c__1, &
			    c_b55, &t[i__ * t_dim1 + 1], &c__1);
		} else {

/*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:n) * V(i,i:n)' */

		    if (i__ < *n) {
			i__2 = *n - i__;
			clacgv_(&i__2, &v[i__ + (i__ + 1) * v_dim1], ldv);
		    }
		    i__2 = i__ - 1;
		    i__3 = *n - i__ + 1;
		    i__4 = i__;
		    q__1.r = -tau[i__4].r, q__1.i = -tau[i__4].i;
		    cgemv_("No transpose", &i__2, &i__3, &q__1, &v[i__ *
			    v_dim1 + 1], ldv, &v[i__ + i__ * v_dim1], ldv, &
			    c_b55, &t[i__ * t_dim1 + 1], &c__1);
		    if (i__ < *n) {
			i__2 = *n - i__;
			clacgv_(&i__2, &v[i__ + (i__ + 1) * v_dim1], ldv);
		    }
		}
		i__2 = i__ + i__ * v_dim1;
		v[i__2].r = vii.r, v[i__2].i = vii.i;

/*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i) */

		i__2 = i__ - 1;
		ctrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t[i__ * t_dim1 + 1], &c__1);
		i__2 = i__ + i__ * t_dim1;
		i__3 = i__;
		t[i__2].r = tau[i__3].r, t[i__2].i = tau[i__3].i;
	    }
/* L20: */
	}
    } else {
	for (i__ = *k; i__ >= 1; --i__) {
	    i__1 = i__;
	    if (tau[i__1].r == 0.f && tau[i__1].i == 0.f) {

/*              H(i)  =  I */

		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    i__2 = j + i__ * t_dim1;
		    t[i__2].r = 0.f, t[i__2].i = 0.f;
/* L30: */
		}
	    } else {

/*              general case */

		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			i__1 = *n - *k + i__ + i__ * v_dim1;
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = *n - *k + i__ + i__ * v_dim1;
			v[i__1].r = 1.f, v[i__1].i = 0.f;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(1:n-k+i,i+1:k)' * V(1:n-k+i,i)
*/

			i__1 = *n - *k + i__;
			i__2 = *k - i__;
			i__3 = i__;
			q__1.r = -tau[i__3].r, q__1.i = -tau[i__3].i;
			cgemv_("Conjugate transpose", &i__1, &i__2, &q__1, &v[
				(i__ + 1) * v_dim1 + 1], ldv, &v[i__ * v_dim1
				+ 1], &c__1, &c_b55, &t[i__ + 1 + i__ *
				t_dim1], &c__1);
			i__1 = *n - *k + i__ + i__ * v_dim1;
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    } else {
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			v[i__1].r = 1.f, v[i__1].i = 0.f;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(i+1:k,1:n-k+i) * V(i,1:n-k+i)'
*/

			i__1 = *n - *k + i__ - 1;
			clacgv_(&i__1, &v[i__ + v_dim1], ldv);
			i__1 = *k - i__;
			i__2 = *n - *k + i__;
			i__3 = i__;
			q__1.r = -tau[i__3].r, q__1.i = -tau[i__3].i;
			cgemv_("No transpose", &i__1, &i__2, &q__1, &v[i__ +
				1 + v_dim1], ldv, &v[i__ + v_dim1], ldv, &
				c_b55, &t[i__ + 1 + i__ * t_dim1], &c__1);
			i__1 = *n - *k + i__ - 1;
			clacgv_(&i__1, &v[i__ + v_dim1], ldv);
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    }

/*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i) */

		    i__1 = *k - i__;
		    ctrmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__
			    + 1 + (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ *
			     t_dim1], &c__1)
			    ;
		}
		i__1 = i__ + i__ * t_dim1;
		i__2 = i__;
		t[i__1].r = tau[i__2].r, t[i__1].i = tau[i__2].i;
	    }
/* L40: */
	}
    }
    return 0;

/*     End of CLARFT */

} /* clarft_ */

/* Subroutine */ int clarfx_(char *side, integer *m, integer *n, complex *v,
	complex *tau, complex *c__, integer *ldc, complex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8,
	    i__9, i__10, i__11;
    complex q__1, q__2, q__3, q__4, q__5, q__6, q__7, q__8, q__9, q__10,
	    q__11, q__12, q__13, q__14, q__15, q__16, q__17, q__18, q__19;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer j;
    static complex t1, t2, t3, t4, t5, t6, t7, t8, t9, v1, v2, v3, v4, v5, v6,
	     v7, v8, v9, t10, v10, sum;
    extern /* Subroutine */ int cgerc_(integer *, integer *, complex *,
	    complex *, integer *, complex *, integer *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLARFX applies a complex elementary reflector H to a complex m by n
    matrix C, from either the left or the right. H is represented in the
    form

          H = I - tau * v * v'

    where tau is a complex scalar and v is a complex vector.

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

    V       (input) COMPLEX array, dimension (M) if SIDE = 'L'
                                          or (N) if SIDE = 'R'
            The vector v in the representation of H.

    TAU     (input) COMPLEX
            The value tau in the representation of H.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension (N) if SIDE = 'L'
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
    if (tau->r == 0.f && tau->i == 0.f) {
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

	cgemv_("Conjugate transpose", m, n, &c_b56, &c__[c_offset], ldc, &v[1]
		, &c__1, &c_b55, &work[1], &c__1);

/*        C := C - tau * v * w' */

	q__1.r = -tau->r, q__1.i = -tau->i;
	cgerc_(m, n, &q__1, &v[1], &c__1, &work[1], &c__1, &c__[c_offset],
		ldc);
	goto L410;
L10:

/*        Special code for 1 x 1 Householder */

	q__3.r = tau->r * v[1].r - tau->i * v[1].i, q__3.i = tau->r * v[1].i
		+ tau->i * v[1].r;
	r_cnjg(&q__4, &v[1]);
	q__2.r = q__3.r * q__4.r - q__3.i * q__4.i, q__2.i = q__3.r * q__4.i
		+ q__3.i * q__4.r;
	q__1.r = 1.f - q__2.r, q__1.i = 0.f - q__2.i;
	t1.r = q__1.r, t1.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__1.r = t1.r * c__[i__3].r - t1.i * c__[i__3].i, q__1.i = t1.r *
		    c__[i__3].i + t1.i * c__[i__3].r;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L20: */
	}
	goto L410;
L30:

/*        Special code for 2 x 2 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__2.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__2.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__3.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__3.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L40: */
	}
	goto L410;
L50:

/*        Special code for 3 x 3 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__3.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__3.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__4.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__4.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__2.r = q__3.r + q__4.r, q__2.i = q__3.i + q__4.i;
	    i__4 = j * c_dim1 + 3;
	    q__5.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__5.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L60: */
	}
	goto L410;
L70:

/*        Special code for 4 x 4 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__4.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__4.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__5.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__5.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__3.r = q__4.r + q__5.r, q__3.i = q__4.i + q__5.i;
	    i__4 = j * c_dim1 + 3;
	    q__6.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__6.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__2.r = q__3.r + q__6.r, q__2.i = q__3.i + q__6.i;
	    i__5 = j * c_dim1 + 4;
	    q__7.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__7.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__1.r = q__2.r + q__7.r, q__1.i = q__2.i + q__7.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L80: */
	}
	goto L410;
L90:

/*        Special code for 5 x 5 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__5.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__5.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__6.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__6.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__4.r = q__5.r + q__6.r, q__4.i = q__5.i + q__6.i;
	    i__4 = j * c_dim1 + 3;
	    q__7.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__7.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__3.r = q__4.r + q__7.r, q__3.i = q__4.i + q__7.i;
	    i__5 = j * c_dim1 + 4;
	    q__8.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__8.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__2.r = q__3.r + q__8.r, q__2.i = q__3.i + q__8.i;
	    i__6 = j * c_dim1 + 5;
	    q__9.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__9.i = v5.r *
		    c__[i__6].i + v5.i * c__[i__6].r;
	    q__1.r = q__2.r + q__9.r, q__1.i = q__2.i + q__9.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L100: */
	}
	goto L410;
L110:

/*        Special code for 6 x 6 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	r_cnjg(&q__1, &v[6]);
	v6.r = q__1.r, v6.i = q__1.i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__6.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__6.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__7.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__7.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__5.r = q__6.r + q__7.r, q__5.i = q__6.i + q__7.i;
	    i__4 = j * c_dim1 + 3;
	    q__8.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__8.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__4.r = q__5.r + q__8.r, q__4.i = q__5.i + q__8.i;
	    i__5 = j * c_dim1 + 4;
	    q__9.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__9.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__3.r = q__4.r + q__9.r, q__3.i = q__4.i + q__9.i;
	    i__6 = j * c_dim1 + 5;
	    q__10.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__10.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__2.r = q__3.r + q__10.r, q__2.i = q__3.i + q__10.i;
	    i__7 = j * c_dim1 + 6;
	    q__11.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__11.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__1.r = q__2.r + q__11.r, q__1.i = q__2.i + q__11.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L120: */
	}
	goto L410;
L130:

/*        Special code for 7 x 7 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	r_cnjg(&q__1, &v[6]);
	v6.r = q__1.r, v6.i = q__1.i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	r_cnjg(&q__1, &v[7]);
	v7.r = q__1.r, v7.i = q__1.i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__7.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__7.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__8.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__8.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__6.r = q__7.r + q__8.r, q__6.i = q__7.i + q__8.i;
	    i__4 = j * c_dim1 + 3;
	    q__9.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__9.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__5.r = q__6.r + q__9.r, q__5.i = q__6.i + q__9.i;
	    i__5 = j * c_dim1 + 4;
	    q__10.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__10.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__4.r = q__5.r + q__10.r, q__4.i = q__5.i + q__10.i;
	    i__6 = j * c_dim1 + 5;
	    q__11.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__11.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__3.r = q__4.r + q__11.r, q__3.i = q__4.i + q__11.i;
	    i__7 = j * c_dim1 + 6;
	    q__12.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__12.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__2.r = q__3.r + q__12.r, q__2.i = q__3.i + q__12.i;
	    i__8 = j * c_dim1 + 7;
	    q__13.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__13.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__1.r = q__2.r + q__13.r, q__1.i = q__2.i + q__13.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L140: */
	}
	goto L410;
L150:

/*        Special code for 8 x 8 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	r_cnjg(&q__1, &v[6]);
	v6.r = q__1.r, v6.i = q__1.i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	r_cnjg(&q__1, &v[7]);
	v7.r = q__1.r, v7.i = q__1.i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	r_cnjg(&q__1, &v[8]);
	v8.r = q__1.r, v8.i = q__1.i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__8.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__8.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__9.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__9.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__7.r = q__8.r + q__9.r, q__7.i = q__8.i + q__9.i;
	    i__4 = j * c_dim1 + 3;
	    q__10.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__10.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__6.r = q__7.r + q__10.r, q__6.i = q__7.i + q__10.i;
	    i__5 = j * c_dim1 + 4;
	    q__11.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__11.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__5.r = q__6.r + q__11.r, q__5.i = q__6.i + q__11.i;
	    i__6 = j * c_dim1 + 5;
	    q__12.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__12.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__4.r = q__5.r + q__12.r, q__4.i = q__5.i + q__12.i;
	    i__7 = j * c_dim1 + 6;
	    q__13.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__13.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__3.r = q__4.r + q__13.r, q__3.i = q__4.i + q__13.i;
	    i__8 = j * c_dim1 + 7;
	    q__14.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__14.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__2.r = q__3.r + q__14.r, q__2.i = q__3.i + q__14.i;
	    i__9 = j * c_dim1 + 8;
	    q__15.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__15.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__1.r = q__2.r + q__15.r, q__1.i = q__2.i + q__15.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L160: */
	}
	goto L410;
L170:

/*        Special code for 9 x 9 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	r_cnjg(&q__1, &v[6]);
	v6.r = q__1.r, v6.i = q__1.i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	r_cnjg(&q__1, &v[7]);
	v7.r = q__1.r, v7.i = q__1.i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	r_cnjg(&q__1, &v[8]);
	v8.r = q__1.r, v8.i = q__1.i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	r_cnjg(&q__1, &v[9]);
	v9.r = q__1.r, v9.i = q__1.i;
	r_cnjg(&q__2, &v9);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t9.r = q__1.r, t9.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__9.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__9.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__10.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__10.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    q__8.r = q__9.r + q__10.r, q__8.i = q__9.i + q__10.i;
	    i__4 = j * c_dim1 + 3;
	    q__11.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__11.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__7.r = q__8.r + q__11.r, q__7.i = q__8.i + q__11.i;
	    i__5 = j * c_dim1 + 4;
	    q__12.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__12.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__6.r = q__7.r + q__12.r, q__6.i = q__7.i + q__12.i;
	    i__6 = j * c_dim1 + 5;
	    q__13.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__13.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__5.r = q__6.r + q__13.r, q__5.i = q__6.i + q__13.i;
	    i__7 = j * c_dim1 + 6;
	    q__14.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__14.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__4.r = q__5.r + q__14.r, q__4.i = q__5.i + q__14.i;
	    i__8 = j * c_dim1 + 7;
	    q__15.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__15.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__3.r = q__4.r + q__15.r, q__3.i = q__4.i + q__15.i;
	    i__9 = j * c_dim1 + 8;
	    q__16.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__16.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__2.r = q__3.r + q__16.r, q__2.i = q__3.i + q__16.i;
	    i__10 = j * c_dim1 + 9;
	    q__17.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, q__17.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    q__1.r = q__2.r + q__17.r, q__1.i = q__2.i + q__17.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 9;
	    i__3 = j * c_dim1 + 9;
	    q__2.r = sum.r * t9.r - sum.i * t9.i, q__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L180: */
	}
	goto L410;
L190:

/*        Special code for 10 x 10 Householder */

	r_cnjg(&q__1, &v[1]);
	v1.r = q__1.r, v1.i = q__1.i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	r_cnjg(&q__1, &v[2]);
	v2.r = q__1.r, v2.i = q__1.i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	r_cnjg(&q__1, &v[3]);
	v3.r = q__1.r, v3.i = q__1.i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	r_cnjg(&q__1, &v[4]);
	v4.r = q__1.r, v4.i = q__1.i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	r_cnjg(&q__1, &v[5]);
	v5.r = q__1.r, v5.i = q__1.i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	r_cnjg(&q__1, &v[6]);
	v6.r = q__1.r, v6.i = q__1.i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	r_cnjg(&q__1, &v[7]);
	v7.r = q__1.r, v7.i = q__1.i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	r_cnjg(&q__1, &v[8]);
	v8.r = q__1.r, v8.i = q__1.i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	r_cnjg(&q__1, &v[9]);
	v9.r = q__1.r, v9.i = q__1.i;
	r_cnjg(&q__2, &v9);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t9.r = q__1.r, t9.i = q__1.i;
	r_cnjg(&q__1, &v[10]);
	v10.r = q__1.r, v10.i = q__1.i;
	r_cnjg(&q__2, &v10);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t10.r = q__1.r, t10.i = q__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    q__10.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__10.i = v1.r
		    * c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    q__11.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__11.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    q__9.r = q__10.r + q__11.r, q__9.i = q__10.i + q__11.i;
	    i__4 = j * c_dim1 + 3;
	    q__12.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__12.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__8.r = q__9.r + q__12.r, q__8.i = q__9.i + q__12.i;
	    i__5 = j * c_dim1 + 4;
	    q__13.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__13.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__7.r = q__8.r + q__13.r, q__7.i = q__8.i + q__13.i;
	    i__6 = j * c_dim1 + 5;
	    q__14.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__14.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__6.r = q__7.r + q__14.r, q__6.i = q__7.i + q__14.i;
	    i__7 = j * c_dim1 + 6;
	    q__15.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__15.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__5.r = q__6.r + q__15.r, q__5.i = q__6.i + q__15.i;
	    i__8 = j * c_dim1 + 7;
	    q__16.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__16.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__4.r = q__5.r + q__16.r, q__4.i = q__5.i + q__16.i;
	    i__9 = j * c_dim1 + 8;
	    q__17.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__17.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__3.r = q__4.r + q__17.r, q__3.i = q__4.i + q__17.i;
	    i__10 = j * c_dim1 + 9;
	    q__18.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, q__18.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    q__2.r = q__3.r + q__18.r, q__2.i = q__3.i + q__18.i;
	    i__11 = j * c_dim1 + 10;
	    q__19.r = v10.r * c__[i__11].r - v10.i * c__[i__11].i, q__19.i =
		    v10.r * c__[i__11].i + v10.i * c__[i__11].r;
	    q__1.r = q__2.r + q__19.r, q__1.i = q__2.i + q__19.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 9;
	    i__3 = j * c_dim1 + 9;
	    q__2.r = sum.r * t9.r - sum.i * t9.i, q__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j * c_dim1 + 10;
	    i__3 = j * c_dim1 + 10;
	    q__2.r = sum.r * t10.r - sum.i * t10.i, q__2.i = sum.r * t10.i +
		    sum.i * t10.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
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

	cgemv_("No transpose", m, n, &c_b56, &c__[c_offset], ldc, &v[1], &
		c__1, &c_b55, &work[1], &c__1);

/*        C := C - tau * w * v' */

	q__1.r = -tau->r, q__1.i = -tau->i;
	cgerc_(m, n, &q__1, &work[1], &c__1, &v[1], &c__1, &c__[c_offset],
		ldc);
	goto L410;
L210:

/*        Special code for 1 x 1 Householder */

	q__3.r = tau->r * v[1].r - tau->i * v[1].i, q__3.i = tau->r * v[1].i
		+ tau->i * v[1].r;
	r_cnjg(&q__4, &v[1]);
	q__2.r = q__3.r * q__4.r - q__3.i * q__4.i, q__2.i = q__3.r * q__4.i
		+ q__3.i * q__4.r;
	q__1.r = 1.f - q__2.r, q__1.i = 0.f - q__2.i;
	t1.r = q__1.r, t1.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__1.r = t1.r * c__[i__3].r - t1.i * c__[i__3].i, q__1.i = t1.r *
		    c__[i__3].i + t1.i * c__[i__3].r;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L220: */
	}
	goto L410;
L230:

/*        Special code for 2 x 2 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__2.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__2.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__3.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__3.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L240: */
	}
	goto L410;
L250:

/*        Special code for 3 x 3 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__3.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__3.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__4.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__4.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__2.r = q__3.r + q__4.r, q__2.i = q__3.i + q__4.i;
	    i__4 = j + c_dim1 * 3;
	    q__5.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__5.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L260: */
	}
	goto L410;
L270:

/*        Special code for 4 x 4 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__4.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__4.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__5.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__5.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__3.r = q__4.r + q__5.r, q__3.i = q__4.i + q__5.i;
	    i__4 = j + c_dim1 * 3;
	    q__6.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__6.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__2.r = q__3.r + q__6.r, q__2.i = q__3.i + q__6.i;
	    i__5 = j + (c_dim1 << 2);
	    q__7.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__7.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__1.r = q__2.r + q__7.r, q__1.i = q__2.i + q__7.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L280: */
	}
	goto L410;
L290:

/*        Special code for 5 x 5 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__5.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__5.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__6.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__6.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__4.r = q__5.r + q__6.r, q__4.i = q__5.i + q__6.i;
	    i__4 = j + c_dim1 * 3;
	    q__7.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__7.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__3.r = q__4.r + q__7.r, q__3.i = q__4.i + q__7.i;
	    i__5 = j + (c_dim1 << 2);
	    q__8.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__8.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__2.r = q__3.r + q__8.r, q__2.i = q__3.i + q__8.i;
	    i__6 = j + c_dim1 * 5;
	    q__9.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__9.i = v5.r *
		    c__[i__6].i + v5.i * c__[i__6].r;
	    q__1.r = q__2.r + q__9.r, q__1.i = q__2.i + q__9.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L300: */
	}
	goto L410;
L310:

/*        Special code for 6 x 6 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__6.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__6.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__7.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__7.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__5.r = q__6.r + q__7.r, q__5.i = q__6.i + q__7.i;
	    i__4 = j + c_dim1 * 3;
	    q__8.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__8.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__4.r = q__5.r + q__8.r, q__4.i = q__5.i + q__8.i;
	    i__5 = j + (c_dim1 << 2);
	    q__9.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__9.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    q__3.r = q__4.r + q__9.r, q__3.i = q__4.i + q__9.i;
	    i__6 = j + c_dim1 * 5;
	    q__10.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__10.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__2.r = q__3.r + q__10.r, q__2.i = q__3.i + q__10.i;
	    i__7 = j + c_dim1 * 6;
	    q__11.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__11.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__1.r = q__2.r + q__11.r, q__1.i = q__2.i + q__11.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L320: */
	}
	goto L410;
L330:

/*        Special code for 7 x 7 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__7.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__7.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__8.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__8.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__6.r = q__7.r + q__8.r, q__6.i = q__7.i + q__8.i;
	    i__4 = j + c_dim1 * 3;
	    q__9.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__9.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    q__5.r = q__6.r + q__9.r, q__5.i = q__6.i + q__9.i;
	    i__5 = j + (c_dim1 << 2);
	    q__10.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__10.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__4.r = q__5.r + q__10.r, q__4.i = q__5.i + q__10.i;
	    i__6 = j + c_dim1 * 5;
	    q__11.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__11.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__3.r = q__4.r + q__11.r, q__3.i = q__4.i + q__11.i;
	    i__7 = j + c_dim1 * 6;
	    q__12.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__12.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__2.r = q__3.r + q__12.r, q__2.i = q__3.i + q__12.i;
	    i__8 = j + c_dim1 * 7;
	    q__13.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__13.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__1.r = q__2.r + q__13.r, q__1.i = q__2.i + q__13.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L340: */
	}
	goto L410;
L350:

/*        Special code for 8 x 8 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__8.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__8.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__9.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__9.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    q__7.r = q__8.r + q__9.r, q__7.i = q__8.i + q__9.i;
	    i__4 = j + c_dim1 * 3;
	    q__10.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__10.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__6.r = q__7.r + q__10.r, q__6.i = q__7.i + q__10.i;
	    i__5 = j + (c_dim1 << 2);
	    q__11.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__11.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__5.r = q__6.r + q__11.r, q__5.i = q__6.i + q__11.i;
	    i__6 = j + c_dim1 * 5;
	    q__12.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__12.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__4.r = q__5.r + q__12.r, q__4.i = q__5.i + q__12.i;
	    i__7 = j + c_dim1 * 6;
	    q__13.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__13.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__3.r = q__4.r + q__13.r, q__3.i = q__4.i + q__13.i;
	    i__8 = j + c_dim1 * 7;
	    q__14.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__14.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__2.r = q__3.r + q__14.r, q__2.i = q__3.i + q__14.i;
	    i__9 = j + (c_dim1 << 3);
	    q__15.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__15.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__1.r = q__2.r + q__15.r, q__1.i = q__2.i + q__15.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 3);
	    i__3 = j + (c_dim1 << 3);
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L360: */
	}
	goto L410;
L370:

/*        Special code for 9 x 9 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	v9.r = v[9].r, v9.i = v[9].i;
	r_cnjg(&q__2, &v9);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t9.r = q__1.r, t9.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__9.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__9.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__10.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__10.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    q__8.r = q__9.r + q__10.r, q__8.i = q__9.i + q__10.i;
	    i__4 = j + c_dim1 * 3;
	    q__11.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__11.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__7.r = q__8.r + q__11.r, q__7.i = q__8.i + q__11.i;
	    i__5 = j + (c_dim1 << 2);
	    q__12.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__12.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__6.r = q__7.r + q__12.r, q__6.i = q__7.i + q__12.i;
	    i__6 = j + c_dim1 * 5;
	    q__13.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__13.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__5.r = q__6.r + q__13.r, q__5.i = q__6.i + q__13.i;
	    i__7 = j + c_dim1 * 6;
	    q__14.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__14.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__4.r = q__5.r + q__14.r, q__4.i = q__5.i + q__14.i;
	    i__8 = j + c_dim1 * 7;
	    q__15.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__15.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__3.r = q__4.r + q__15.r, q__3.i = q__4.i + q__15.i;
	    i__9 = j + (c_dim1 << 3);
	    q__16.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__16.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__2.r = q__3.r + q__16.r, q__2.i = q__3.i + q__16.i;
	    i__10 = j + c_dim1 * 9;
	    q__17.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, q__17.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    q__1.r = q__2.r + q__17.r, q__1.i = q__2.i + q__17.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 3);
	    i__3 = j + (c_dim1 << 3);
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 9;
	    i__3 = j + c_dim1 * 9;
	    q__2.r = sum.r * t9.r - sum.i * t9.i, q__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L380: */
	}
	goto L410;
L390:

/*        Special code for 10 x 10 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	r_cnjg(&q__2, &v1);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t1.r = q__1.r, t1.i = q__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	r_cnjg(&q__2, &v2);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t2.r = q__1.r, t2.i = q__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	r_cnjg(&q__2, &v3);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t3.r = q__1.r, t3.i = q__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	r_cnjg(&q__2, &v4);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t4.r = q__1.r, t4.i = q__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	r_cnjg(&q__2, &v5);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t5.r = q__1.r, t5.i = q__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	r_cnjg(&q__2, &v6);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t6.r = q__1.r, t6.i = q__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	r_cnjg(&q__2, &v7);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t7.r = q__1.r, t7.i = q__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	r_cnjg(&q__2, &v8);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t8.r = q__1.r, t8.i = q__1.i;
	v9.r = v[9].r, v9.i = v[9].i;
	r_cnjg(&q__2, &v9);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t9.r = q__1.r, t9.i = q__1.i;
	v10.r = v[10].r, v10.i = v[10].i;
	r_cnjg(&q__2, &v10);
	q__1.r = tau->r * q__2.r - tau->i * q__2.i, q__1.i = tau->r * q__2.i
		+ tau->i * q__2.r;
	t10.r = q__1.r, t10.i = q__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    q__10.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, q__10.i = v1.r
		    * c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + (c_dim1 << 1);
	    q__11.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, q__11.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    q__9.r = q__10.r + q__11.r, q__9.i = q__10.i + q__11.i;
	    i__4 = j + c_dim1 * 3;
	    q__12.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, q__12.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    q__8.r = q__9.r + q__12.r, q__8.i = q__9.i + q__12.i;
	    i__5 = j + (c_dim1 << 2);
	    q__13.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, q__13.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    q__7.r = q__8.r + q__13.r, q__7.i = q__8.i + q__13.i;
	    i__6 = j + c_dim1 * 5;
	    q__14.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, q__14.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    q__6.r = q__7.r + q__14.r, q__6.i = q__7.i + q__14.i;
	    i__7 = j + c_dim1 * 6;
	    q__15.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, q__15.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    q__5.r = q__6.r + q__15.r, q__5.i = q__6.i + q__15.i;
	    i__8 = j + c_dim1 * 7;
	    q__16.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, q__16.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    q__4.r = q__5.r + q__16.r, q__4.i = q__5.i + q__16.i;
	    i__9 = j + (c_dim1 << 3);
	    q__17.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, q__17.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    q__3.r = q__4.r + q__17.r, q__3.i = q__4.i + q__17.i;
	    i__10 = j + c_dim1 * 9;
	    q__18.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, q__18.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    q__2.r = q__3.r + q__18.r, q__2.i = q__3.i + q__18.i;
	    i__11 = j + c_dim1 * 10;
	    q__19.r = v10.r * c__[i__11].r - v10.i * c__[i__11].i, q__19.i =
		    v10.r * c__[i__11].i + v10.i * c__[i__11].r;
	    q__1.r = q__2.r + q__19.r, q__1.i = q__2.i + q__19.i;
	    sum.r = q__1.r, sum.i = q__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    q__2.r = sum.r * t1.r - sum.i * t1.i, q__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 1);
	    i__3 = j + (c_dim1 << 1);
	    q__2.r = sum.r * t2.r - sum.i * t2.i, q__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    q__2.r = sum.r * t3.r - sum.i * t3.i, q__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 2);
	    i__3 = j + (c_dim1 << 2);
	    q__2.r = sum.r * t4.r - sum.i * t4.i, q__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    q__2.r = sum.r * t5.r - sum.i * t5.i, q__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    q__2.r = sum.r * t6.r - sum.i * t6.i, q__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    q__2.r = sum.r * t7.r - sum.i * t7.i, q__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + (c_dim1 << 3);
	    i__3 = j + (c_dim1 << 3);
	    q__2.r = sum.r * t8.r - sum.i * t8.i, q__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 9;
	    i__3 = j + c_dim1 * 9;
	    q__2.r = sum.r * t9.r - sum.i * t9.i, q__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
	    i__2 = j + c_dim1 * 10;
	    i__3 = j + c_dim1 * 10;
	    q__2.r = sum.r * t10.r - sum.i * t10.i, q__2.i = sum.r * t10.i +
		    sum.i * t10.r;
	    q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
	    c__[i__2].r = q__1.r, c__[i__2].i = q__1.i;
/* L400: */
	}
	goto L410;
    }
L410:
    return 0;

/*     End of CLARFX */

} /* clarfx_ */

/* Subroutine */ int clascl_(char *type__, integer *kl, integer *ku, real *
	cfrom, real *cto, integer *m, integer *n, complex *a, integer *lda,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1;

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

    CLASCL multiplies the M by N complex matrix A by the real scalar
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

    A       (input/output) COMPLEX array, dimension (LDA,M)
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
	xerbla_("CLASCL", &i__1);
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
		i__2 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + j * a_dim1;
		q__1.r = mul * a[i__4].r, q__1.i = mul * a[i__4].i;
		a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L140: */
	    }
/* L150: */
	}

    }

    if (! done) {
	goto L10;
    }

    return 0;

/*     End of CLASCL */

} /* clascl_ */

/* Subroutine */ int claset_(char *uplo, integer *m, integer *n, complex *
	alpha, complex *beta, complex *a, integer *lda)
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

    CLASET initializes a 2-D array A to BETA on the diagonal and
    ALPHA on the offdiagonals.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies the part of the matrix A to be set.
            = 'U':      Upper triangular part is set. The lower triangle
                        is unchanged.
            = 'L':      Lower triangular part is set. The upper triangle
                        is unchanged.
            Otherwise:  All of the matrix A is set.

    M       (input) INTEGER
            On entry, M specifies the number of rows of A.

    N       (input) INTEGER
            On entry, N specifies the number of columns of A.

    ALPHA   (input) COMPLEX
            All the offdiagonal array elements are set to ALPHA.

    BETA    (input) COMPLEX
            All the diagonal array elements are set to BETA.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n, i.ne.j;
                     A(i,i) = BETA , 1 <= i <= min(m,n)

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
          Set the diagonal to BETA and the strictly upper triangular
          part of the array to ALPHA.
*/

	i__1 = *n;
	for (j = 2; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = j - 1;
	    i__2 = min(i__3,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		a[i__3].r = alpha->r, a[i__3].i = alpha->i;
/* L10: */
	    }
/* L20: */
	}
	i__1 = min(*n,*m);
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = beta->r, a[i__2].i = beta->i;
/* L30: */
	}

    } else if (lsame_(uplo, "L")) {

/*
          Set the diagonal to BETA and the strictly lower triangular
          part of the array to ALPHA.
*/

	i__1 = min(*m,*n);
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j + 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		a[i__3].r = alpha->r, a[i__3].i = alpha->i;
/* L40: */
	    }
/* L50: */
	}
	i__1 = min(*n,*m);
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = beta->r, a[i__2].i = beta->i;
/* L60: */
	}

    } else {

/*
          Set the array to BETA on the diagonal and ALPHA on the
          offdiagonal.
*/

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		a[i__3].r = alpha->r, a[i__3].i = alpha->i;
/* L70: */
	    }
/* L80: */
	}
	i__1 = min(*m,*n);
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = beta->r, a[i__2].i = beta->i;
/* L90: */
	}
    }

    return 0;

/*     End of CLASET */

} /* claset_ */

/* Subroutine */ int clasr_(char *side, char *pivot, char *direct, integer *m,
	 integer *n, real *c__, real *s, complex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    complex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, j, info;
    static complex temp;
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

    CLASR   performs the transformation

       A := P*A,   when SIDE = 'L' or 'l'  (  Left-hand side )

       A := A*P',  when SIDE = 'R' or 'r'  ( Right-hand side )

    where A is an m by n complex matrix and P is an orthogonal matrix,
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CLASR ", &info);
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
			    i__3 = j + 1 + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + 1 + i__ * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__4 = j + i__ * a_dim1;
			    q__3.r = stemp * a[i__4].r, q__3.i = stemp * a[
				    i__4].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = j + i__ * a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__4 = j + i__ * a_dim1;
			    q__3.r = ctemp * a[i__4].r, q__3.i = ctemp * a[
				    i__4].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = j + 1 + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + 1 + i__ * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__3 = j + i__ * a_dim1;
			    q__3.r = stemp * a[i__3].r, q__3.i = stemp * a[
				    i__3].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = j + i__ * a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__3 = j + i__ * a_dim1;
			    q__3.r = ctemp * a[i__3].r, q__3.i = ctemp * a[
				    i__3].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
			    i__3 = j + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + i__ * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__4 = i__ * a_dim1 + 1;
			    q__3.r = stemp * a[i__4].r, q__3.i = stemp * a[
				    i__4].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = i__ * a_dim1 + 1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__4 = i__ * a_dim1 + 1;
			    q__3.r = ctemp * a[i__4].r, q__3.i = ctemp * a[
				    i__4].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = j + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + i__ * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__3 = i__ * a_dim1 + 1;
			    q__3.r = stemp * a[i__3].r, q__3.i = stemp * a[
				    i__3].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = i__ * a_dim1 + 1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__3 = i__ * a_dim1 + 1;
			    q__3.r = ctemp * a[i__3].r, q__3.i = ctemp * a[
				    i__3].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
			    i__3 = j + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + i__ * a_dim1;
			    i__4 = *m + i__ * a_dim1;
			    q__2.r = stemp * a[i__4].r, q__2.i = stemp * a[
				    i__4].i;
			    q__3.r = ctemp * temp.r, q__3.i = ctemp * temp.i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = *m + i__ * a_dim1;
			    i__4 = *m + i__ * a_dim1;
			    q__2.r = ctemp * a[i__4].r, q__2.i = ctemp * a[
				    i__4].i;
			    q__3.r = stemp * temp.r, q__3.i = stemp * temp.i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = j + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + i__ * a_dim1;
			    i__3 = *m + i__ * a_dim1;
			    q__2.r = stemp * a[i__3].r, q__2.i = stemp * a[
				    i__3].i;
			    q__3.r = ctemp * temp.r, q__3.i = ctemp * temp.i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = *m + i__ * a_dim1;
			    i__3 = *m + i__ * a_dim1;
			    q__2.r = ctemp * a[i__3].r, q__2.i = ctemp * a[
				    i__3].i;
			    q__3.r = stemp * temp.r, q__3.i = stemp * temp.i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
			    i__3 = i__ + (j + 1) * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + (j + 1) * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__4 = i__ + j * a_dim1;
			    q__3.r = stemp * a[i__4].r, q__3.i = stemp * a[
				    i__4].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = i__ + j * a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__4 = i__ + j * a_dim1;
			    q__3.r = ctemp * a[i__4].r, q__3.i = ctemp * a[
				    i__4].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = i__ + (j + 1) * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + (j + 1) * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__3 = i__ + j * a_dim1;
			    q__3.r = stemp * a[i__3].r, q__3.i = stemp * a[
				    i__3].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = i__ + j * a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__3 = i__ + j * a_dim1;
			    q__3.r = ctemp * a[i__3].r, q__3.i = ctemp * a[
				    i__3].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
			    i__3 = i__ + j * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + j * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__4 = i__ + a_dim1;
			    q__3.r = stemp * a[i__4].r, q__3.i = stemp * a[
				    i__4].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = i__ + a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__4 = i__ + a_dim1;
			    q__3.r = ctemp * a[i__4].r, q__3.i = ctemp * a[
				    i__4].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = i__ + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + j * a_dim1;
			    q__2.r = ctemp * temp.r, q__2.i = ctemp * temp.i;
			    i__3 = i__ + a_dim1;
			    q__3.r = stemp * a[i__3].r, q__3.i = stemp * a[
				    i__3].i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = i__ + a_dim1;
			    q__2.r = stemp * temp.r, q__2.i = stemp * temp.i;
			    i__3 = i__ + a_dim1;
			    q__3.r = ctemp * a[i__3].r, q__3.i = ctemp * a[
				    i__3].i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
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
			    i__3 = i__ + j * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + j * a_dim1;
			    i__4 = i__ + *n * a_dim1;
			    q__2.r = stemp * a[i__4].r, q__2.i = stemp * a[
				    i__4].i;
			    q__3.r = ctemp * temp.r, q__3.i = ctemp * temp.i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			    i__3 = i__ + *n * a_dim1;
			    i__4 = i__ + *n * a_dim1;
			    q__2.r = ctemp * a[i__4].r, q__2.i = ctemp * a[
				    i__4].i;
			    q__3.r = stemp * temp.r, q__3.i = stemp * temp.i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__3].r = q__1.r, a[i__3].i = q__1.i;
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
			    i__2 = i__ + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + j * a_dim1;
			    i__3 = i__ + *n * a_dim1;
			    q__2.r = stemp * a[i__3].r, q__2.i = stemp * a[
				    i__3].i;
			    q__3.r = ctemp * temp.r, q__3.i = ctemp * temp.i;
			    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i +
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			    i__2 = i__ + *n * a_dim1;
			    i__3 = i__ + *n * a_dim1;
			    q__2.r = ctemp * a[i__3].r, q__2.i = ctemp * a[
				    i__3].i;
			    q__3.r = stemp * temp.r, q__3.i = stemp * temp.i;
			    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i -
				    q__3.i;
			    a[i__2].r = q__1.r, a[i__2].i = q__1.i;
/* L230: */
			}
		    }
/* L240: */
		}
	    }
	}
    }

    return 0;

/*     End of CLASR */

} /* clasr_ */

/* Subroutine */ int classq_(integer *n, complex *x, integer *incx, real *
	scale, real *sumsq)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    static integer ix;
    static real temp1;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLASSQ returns the values scl and ssq such that

       ( scl**2 )*ssq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,

    where x( i ) = abs( X( 1 + ( i - 1 )*INCX ) ). The value of sumsq is
    assumed to be at least unity and the value of ssq will then satisfy

       1.0 .le. ssq .le. ( sumsq + 2*n ).

    scale is assumed to be non-negative and scl returns the value

       scl = max( scale, abs( real( x( i ) ) ), abs( aimag( x( i ) ) ) ),
              i

    scale and sumsq must be supplied in SCALE and SUMSQ respectively.
    SCALE and SUMSQ are overwritten by scl and ssq respectively.

    The routine makes only one pass through the vector X.

    Arguments
    =========

    N       (input) INTEGER
            The number of elements to be used from the vector X.

    X       (input) COMPLEX array, dimension (N)
            The vector x as described above.
               x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n.

    INCX    (input) INTEGER
            The increment between successive values of the vector X.
            INCX > 0.

    SCALE   (input/output) REAL
            On entry, the value  scale  in the equation above.
            On exit, SCALE is overwritten with the value  scl .

    SUMSQ   (input/output) REAL
            On entry, the value  sumsq  in the equation above.
            On exit, SUMSQ is overwritten with the value  ssq .

   =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n > 0) {
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    i__3 = ix;
	    if (x[i__3].r != 0.f) {
		i__3 = ix;
		temp1 = (r__1 = x[i__3].r, dabs(r__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    r__1 = *scale / temp1;
		    *sumsq = *sumsq * (r__1 * r__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    r__1 = temp1 / *scale;
		    *sumsq += r__1 * r__1;
		}
	    }
	    if (r_imag(&x[ix]) != 0.f) {
		temp1 = (r__1 = r_imag(&x[ix]), dabs(r__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    r__1 = *scale / temp1;
		    *sumsq = *sumsq * (r__1 * r__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    r__1 = temp1 / *scale;
		    *sumsq += r__1 * r__1;
		}
	    }
/* L10: */
	}
    }

    return 0;

/*     End of CLASSQ */

} /* classq_ */

/* Subroutine */ int claswp_(integer *n, complex *a, integer *lda, integer *
	k1, integer *k2, integer *ipiv, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    static integer i__, j, k, i1, i2, n32, ip, ix, ix0, inc;
    static complex temp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.

    Arguments
    =========

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
			i__5 = i__ + k * a_dim1;
			temp.r = a[i__5].r, temp.i = a[i__5].i;
			i__5 = i__ + k * a_dim1;
			i__6 = ip + k * a_dim1;
			a[i__5].r = a[i__6].r, a[i__5].i = a[i__6].i;
			i__5 = ip + k * a_dim1;
			a[i__5].r = temp.r, a[i__5].i = temp.i;
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
		    i__4 = i__ + k * a_dim1;
		    temp.r = a[i__4].r, temp.i = a[i__4].i;
		    i__4 = i__ + k * a_dim1;
		    i__5 = ip + k * a_dim1;
		    a[i__4].r = a[i__5].r, a[i__4].i = a[i__5].i;
		    i__4 = ip + k * a_dim1;
		    a[i__4].r = temp.r, a[i__4].i = temp.i;
/* L40: */
		}
	    }
	    ix += *incx;
/* L50: */
	}
    }

    return 0;

/*     End of CLASWP */

} /* claswp_ */

/* Subroutine */ int clatrd_(char *uplo, integer *n, integer *nb, complex *a,
	integer *lda, real *e, complex *tau, complex *w, integer *ldw)
{
    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2, q__3, q__4;

    /* Local variables */
    static integer i__, iw;
    static complex alpha;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *);
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *), chemv_(char *, integer *, complex *,
	    complex *, integer *, complex *, integer *, complex *, complex *,
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *,
	    integer *, complex *, integer *), clarfg_(integer *, complex *,
	    complex *, integer *, complex *), clacgv_(integer *, complex *,
	    integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLATRD reduces NB rows and columns of a complex Hermitian matrix A to
    Hermitian tridiagonal form by a unitary similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = 'U', CLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = 'L', CLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by CHETRD.

    Arguments
    =========

    UPLO    (input) CHARACTER
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is stored:
            = 'U': Upper triangular
            = 'L': Lower triangular

    N       (input) INTEGER
            The order of the matrix A.

    NB      (input) INTEGER
            The number of rows and columns to be reduced.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
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
              with the array TAU, represent the unitary matrix Q as a
              product of elementary reflectors;
            if UPLO = 'L', the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  unitary matrix Q as a
              product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    E       (output) REAL array, dimension (N-1)
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    TAU     (output) COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
            See Further Details.

    W       (output) COMPLEX array, dimension (LDW,NB)
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

    where tau is a complex scalar, and v is a complex vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
    and tau in TAU(i).

    The elements of the vectors v together form the n-by-nb matrix V
    which is needed, with W, to apply the transformation to the unreduced
    part of the matrix, using a Hermitian rank-2k update of the form:
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

		i__2 = i__ + i__ * a_dim1;
		i__3 = i__ + i__ * a_dim1;
		r__1 = a[i__3].r;
		a[i__2].r = r__1, a[i__2].i = 0.f;
		i__2 = *n - i__;
		clacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__, &i__2, &q__1, &a[(i__ + 1) *
			a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, &
			c_b56, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		clacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = *n - i__;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__, &i__2, &q__1, &w[(iw + 1) *
			w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b56, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + i__ * a_dim1;
		i__3 = i__ + i__ * a_dim1;
		r__1 = a[i__3].r;
		a[i__2].r = r__1, a[i__2].i = 0.f;
	    }
	    if (i__ > 1) {

/*
                Generate elementary reflector H(i) to annihilate
                A(1:i-2,i)
*/

		i__2 = i__ - 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = i__ - 1;
		clarfg_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &tau[i__
			- 1]);
		i__2 = i__ - 1;
		e[i__2] = alpha.r;
		i__2 = i__ - 1 + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute W(1:i-1,i) */

		i__2 = i__ - 1;
		chemv_("Upper", &i__2, &c_b56, &a[a_offset], lda, &a[i__ *
			a_dim1 + 1], &c__1, &c_b55, &w[iw * w_dim1 + 1], &
			c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &w[(
			    iw + 1) * w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1],
			    &c__1, &c_b55, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemv_("No transpose", &i__2, &i__3, &q__1, &a[(i__ + 1) *
			     a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b56, &w[iw * w_dim1 + 1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[(
			    i__ + 1) * a_dim1 + 1], lda, &a[i__ * a_dim1 + 1],
			     &c__1, &c_b55, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemv_("No transpose", &i__2, &i__3, &q__1, &w[(iw + 1) *
			    w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b56, &w[iw * w_dim1 + 1], &c__1);
		}
		i__2 = i__ - 1;
		cscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
		q__3.r = -.5f, q__3.i = -0.f;
		i__2 = i__ - 1;
		q__2.r = q__3.r * tau[i__2].r - q__3.i * tau[i__2].i, q__2.i =
			 q__3.r * tau[i__2].i + q__3.i * tau[i__2].r;
		i__3 = i__ - 1;
		cdotc_(&q__4, &i__3, &w[iw * w_dim1 + 1], &c__1, &a[i__ *
			a_dim1 + 1], &c__1);
		q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r *
			q__4.i + q__2.i * q__4.r;
		alpha.r = q__1.r, alpha.i = q__1.i;
		i__2 = i__ - 1;
		caxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw *
			w_dim1 + 1], &c__1);
	    }

/* L10: */
	}
    } else {

/*        Reduce first NB columns of lower triangle */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:n,i) */

	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__ + i__ * a_dim1;
	    r__1 = a[i__3].r;
	    a[i__2].r = r__1, a[i__2].i = 0.f;
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + a_dim1], lda,
		     &w[i__ + w_dim1], ldw, &c_b56, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__2, &i__3, &q__1, &w[i__ + w_dim1], ldw,
		     &a[i__ + a_dim1], lda, &c_b56, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    clacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__ + i__ * a_dim1;
	    r__1 = a[i__3].r;
	    a[i__2].r = r__1, a[i__2].i = 0.f;
	    if (i__ < *n) {

/*
                Generate elementary reflector H(i) to annihilate
                A(i+2:n,i)
*/

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		clarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1,
			 &tau[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;

/*              Compute W(i+1:n,i) */

		i__2 = *n - i__;
		chemv_("Lower", &i__2, &c_b56, &a[i__ + 1 + (i__ + 1) *
			a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b55, &w[i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &w[i__ +
			1 + w_dim1], ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b55, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &a[i__ + 1 +
			a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b56, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b55, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &w[i__ + 1 +
			w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b56, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		cscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
		q__3.r = -.5f, q__3.i = -0.f;
		i__2 = i__;
		q__2.r = q__3.r * tau[i__2].r - q__3.i * tau[i__2].i, q__2.i =
			 q__3.r * tau[i__2].i + q__3.i * tau[i__2].r;
		i__3 = *n - i__;
		cdotc_(&q__4, &i__3, &w[i__ + 1 + i__ * w_dim1], &c__1, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r *
			q__4.i + q__2.i * q__4.r;
		alpha.r = q__1.r, alpha.i = q__1.i;
		i__2 = *n - i__;
		caxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
	    }

/* L20: */
	}
    }

    return 0;

/*     End of CLATRD */

} /* clatrd_ */

/* Subroutine */ int clatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, complex *a, integer *lda, complex *x, real *scale,
	 real *cnorm, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    double r_imag(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, j;
    static real xj, rec, tjj;
    static integer jinc;
    static real xbnd;
    static integer imax;
    static real tmax;
    static complex tjjs;
    static real xmax, grow;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    static real tscal;
    static complex uscal;
    static integer jlast;
    extern /* Complex */ VOID cdotu_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    static complex csumj;
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *,
	    integer *, complex *, integer *);
    static logical upper;
    extern /* Subroutine */ int ctrsv_(char *, char *, char *, integer *,
	    complex *, integer *, complex *, integer *), slabad_(real *, real *);
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Complex */ VOID cladiv_(complex *, complex *, complex *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), xerbla_(char *, integer *);
    static real bignum;
    extern integer isamax_(integer *, real *, integer *);
    extern doublereal scasum_(integer *, complex *, integer *);
    static logical notran;
    static integer jfirst;
    static real smlnum;
    static logical nounit;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1992


    Purpose
    =======

    CLATRS solves one of the triangular systems

       A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b,

    with scaling to prevent overflow.  Here A is an upper or lower
    triangular matrix, A**T denotes the transpose of A, A**H denotes the
    conjugate transpose of A, x and b are n-element vectors, and s is a
    scaling factor, usually less than or equal to 1, chosen so that the
    components of x will be less than the overflow threshold.  If the
    unscaled problem will not cause overflow, the Level 2 BLAS routine
    CTRSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
    then s is set to 0 and a non-trivial solution to A*x = 0 is returned.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the matrix A is upper or lower triangular.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    TRANS   (input) CHARACTER*1
            Specifies the operation applied to A.
            = 'N':  Solve A * x = s*b     (No transpose)
            = 'T':  Solve A**T * x = s*b  (Transpose)
            = 'C':  Solve A**H * x = s*b  (Conjugate transpose)

    DIAG    (input) CHARACTER*1
            Specifies whether or not the matrix A is unit triangular.
            = 'N':  Non-unit triangular
            = 'U':  Unit triangular

    NORMIN  (input) CHARACTER*1
            Specifies whether CNORM has been set or not.
            = 'Y':  CNORM contains the column norms on entry
            = 'N':  CNORM is not set on entry.  On exit, the norms will
                    be computed and stored in CNORM.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input) COMPLEX array, dimension (LDA,N)
            The triangular matrix A.  If UPLO = 'U', the leading n by n
            upper triangular part of the array A contains the upper
            triangular matrix, and the strictly lower triangular part of
            A is not referenced.  If UPLO = 'L', the leading n by n lower
            triangular part of the array A contains the lower triangular
            matrix, and the strictly upper triangular part of A is not
            referenced.  If DIAG = 'U', the diagonal elements of A are
            also not referenced and are assumed to be 1.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max (1,N).

    X       (input/output) COMPLEX array, dimension (N)
            On entry, the right hand side b of the triangular system.
            On exit, X is overwritten by the solution vector x.

    SCALE   (output) REAL
            The scaling factor s for the triangular system
               A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b.
            If SCALE = 0, the matrix A is singular or badly scaled, and
            the vector x is an exact or approximate solution to A*x = 0.

    CNORM   (input or output) REAL array, dimension (N)

            If NORMIN = 'Y', CNORM is an input argument and CNORM(j)
            contains the norm of the off-diagonal part of the j-th column
            of A.  If TRANS = 'N', CNORM(j) must be greater than or equal
            to the infinity-norm, and if TRANS = 'T' or 'C', CNORM(j)
            must be greater than or equal to the 1-norm.

            If NORMIN = 'N', CNORM is an output argument and CNORM(j)
            returns the 1-norm of the offdiagonal part of the j-th column
            of A.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -k, the k-th argument had an illegal value

    Further Details
    ======= =======

    A rough bound on x is computed; if that is less than overflow, CTRSV
    is called, otherwise, specific code is used which checks for possible
    overflow or divide-by-zero at every operation.

    A columnwise scheme is used for solving A*x = b.  The basic algorithm
    if A is lower triangular is

         x[1:n] := b[1:n]
         for j = 1, ..., n
              x(j) := x(j) / A(j,j)
              x[j+1:n] := x[j+1:n] - x(j) * A[j+1:n,j]
         end

    Define bounds on the components of x after j iterations of the loop:
       M(j) = bound on x[1:j]
       G(j) = bound on x[j+1:n]
    Initially, let M(0) = 0 and G(0) = max{x(i), i=1,...,n}.

    Then for iteration j+1 we have
       M(j+1) <= G(j) / | A(j+1,j+1) |
       G(j+1) <= G(j) + M(j+1) * | A[j+2:n,j+1] |
              <= G(j) ( 1 + CNORM(j+1) / | A(j+1,j+1) | )

    where CNORM(j+1) is greater than or equal to the infinity-norm of
    column j+1 of A, not counting the diagonal.  Hence

       G(j) <= G(0) product ( 1 + CNORM(i) / | A(i,i) | )
                    1<=i<=j
    and

       |x(j)| <= ( G(0) / |A(j,j)| ) product ( 1 + CNORM(i) / |A(i,i)| )
                                     1<=i< j

    Since |x(j)| <= M(j), we use the Level 2 BLAS routine CTRSV if the
    reciprocal of the largest M(j), j=1,..,n, is larger than
    max(underflow, 1/overflow).

    The bound on x(j) is also used to determine when a step in the
    columnwise method can be performed without fear of overflow.  If
    the computed bound is greater than a large constant, x is scaled to
    prevent overflow, but if the bound overflows, x is set to 0, x(j) to
    1, and scale to 0, and a non-trivial solution to A*x = 0 is found.

    Similarly, a row-wise scheme is used to solve A**T *x = b  or
    A**H *x = b.  The basic algorithm for A upper triangular is

         for j = 1, ..., n
              x(j) := ( b(j) - A[1:j-1,j]' * x[1:j-1] ) / A(j,j)
         end

    We simultaneously compute two bounds
         G(j) = bound on ( b(i) - A[1:i-1,i]' * x[1:i-1] ), 1<=i<=j
         M(j) = bound on x(i), 1<=i<=j

    The initial values are G(0) = 0, M(0) = max{b(i), i=1,..,n}, and we
    add the constraint G(j) >= G(j-1) and M(j) >= M(j-1) for j >= 1.
    Then the bound on x(j) is

         M(j) <= M(j-1) * ( 1 + CNORM(j) ) / | A(j,j) |

              <= M(0) * product ( ( 1 + CNORM(i) ) / |A(i,i)| )
                        1<=i<=j

    and we can safely call CTRSV if 1/M(n) and 1/G(n) are both greater
    than max(underflow, 1/overflow).

    =====================================================================
*/


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --cnorm;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    notran = lsame_(trans, "N");
    nounit = lsame_(diag, "N");

/*     Test the input parameters. */

    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T") && !
	    lsame_(trans, "C")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (! lsame_(normin, "Y") && ! lsame_(normin,
	     "N")) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLATRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine machine dependent parameters to control overflow. */

    smlnum = slamch_("Safe minimum");
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);
    smlnum /= slamch_("Precision");
    bignum = 1.f / smlnum;
    *scale = 1.f;

    if (lsame_(normin, "N")) {

/*        Compute the 1-norm of each column, not including the diagonal. */

	if (upper) {

/*           A is upper triangular. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j - 1;
		cnorm[j] = scasum_(&i__2, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	    }
	} else {

/*           A is lower triangular. */

	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		cnorm[j] = scasum_(&i__2, &a[j + 1 + j * a_dim1], &c__1);
/* L20: */
	    }
	    cnorm[*n] = 0.f;
	}
    }

/*
       Scale the column norms by TSCAL if the maximum element in CNORM is
       greater than BIGNUM/2.
*/

    imax = isamax_(n, &cnorm[1], &c__1);
    tmax = cnorm[imax];
    if (tmax <= bignum * .5f) {
	tscal = 1.f;
    } else {
	tscal = .5f / (smlnum * tmax);
	sscal_(n, &tscal, &cnorm[1], &c__1);
    }

/*
       Compute a bound on the computed solution vector to see if the
       Level 2 BLAS routine CTRSV can be used.
*/

    xmax = 0.f;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__2 = j;
	r__3 = xmax, r__4 = (r__1 = x[i__2].r / 2.f, dabs(r__1)) + (r__2 =
		r_imag(&x[j]) / 2.f, dabs(r__2));
	xmax = dmax(r__3,r__4);
/* L30: */
    }
    xbnd = xmax;

    if (notran) {

/*        Compute the growth in A * x = b. */

	if (upper) {
	    jfirst = *n;
	    jlast = 1;
	    jinc = -1;
	} else {
	    jfirst = 1;
	    jlast = *n;
	    jinc = 1;
	}

	if (tscal != 1.f) {
	    grow = 0.f;
	    goto L60;
	}

	if (nounit) {

/*
             A is non-unit triangular.

             Compute GROW = 1/G(j) and XBND = 1/M(j).
             Initially, G(0) = max{x(i), i=1,...,n}.
*/

	    grow = .5f / dmax(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

		i__3 = j + j * a_dim1;
		tjjs.r = a[i__3].r, tjjs.i = a[i__3].i;
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			dabs(r__2));

		if (tjj >= smlnum) {

/*
                   M(j) = G(j-1) / abs(A(j,j))

   Computing MIN
*/
		    r__1 = xbnd, r__2 = dmin(1.f,tjj) * grow;
		    xbnd = dmin(r__1,r__2);
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.f;
		}

		if (tjj + cnorm[j] >= smlnum) {

/*                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) ) */

		    grow *= tjj / (tjj + cnorm[j]);
		} else {

/*                 G(j) could overflow, set GROW to 0. */

		    grow = 0.f;
		}
/* L40: */
	    }
	    grow = xbnd;
	} else {

/*
             A is unit triangular.

             Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.

   Computing MIN
*/
	    r__1 = 1.f, r__2 = .5f / dmax(xbnd,smlnum);
	    grow = dmin(r__1,r__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

/*              G(j) = G(j-1)*( 1 + CNORM(j) ) */

		grow *= 1.f / (cnorm[j] + 1.f);
/* L50: */
	    }
	}
L60:

	;
    } else {

/*        Compute the growth in A**T * x = b  or  A**H * x = b. */

	if (upper) {
	    jfirst = 1;
	    jlast = *n;
	    jinc = 1;
	} else {
	    jfirst = *n;
	    jlast = 1;
	    jinc = -1;
	}

	if (tscal != 1.f) {
	    grow = 0.f;
	    goto L90;
	}

	if (nounit) {

/*
             A is non-unit triangular.

             Compute GROW = 1/G(j) and XBND = 1/M(j).
             Initially, M(0) = max{x(i), i=1,...,n}.
*/

	    grow = .5f / dmax(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) ) */

		xj = cnorm[j] + 1.f;
/* Computing MIN */
		r__1 = grow, r__2 = xbnd / xj;
		grow = dmin(r__1,r__2);

		i__3 = j + j * a_dim1;
		tjjs.r = a[i__3].r, tjjs.i = a[i__3].i;
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			dabs(r__2));

		if (tjj >= smlnum) {

/*                 M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j)) */

		    if (xj > tjj) {
			xbnd *= tjj / xj;
		    }
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.f;
		}
/* L70: */
	    }
	    grow = dmin(grow,xbnd);
	} else {

/*
             A is unit triangular.

             Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.

   Computing MIN
*/
	    r__1 = 1.f, r__2 = .5f / dmax(xbnd,smlnum);
	    grow = dmin(r__1,r__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = ( 1 + CNORM(j) )*G(j-1) */

		xj = cnorm[j] + 1.f;
		grow /= xj;
/* L80: */
	    }
	}
L90:
	;
    }

    if (grow * tscal > smlnum) {

/*
          Use the Level 2 BLAS solve if the reciprocal of the bound on
          elements of X is not too small.
*/

	ctrsv_(uplo, trans, diag, n, &a[a_offset], lda, &x[1], &c__1);
    } else {

/*        Use a Level 1 BLAS solve, scaling intermediate results. */

	if (xmax > bignum * .5f) {

/*
             Scale X so that its components are less than or equal to
             BIGNUM in absolute value.
*/

	    *scale = bignum * .5f / xmax;
	    csscal_(n, scale, &x[1], &c__1);
	    xmax = bignum;
	} else {
	    xmax *= 2.f;
	}

	if (notran) {

/*           Solve A * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) / A(j,j), scaling x if necessary. */

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]),
			dabs(r__2));
		if (nounit) {
		    i__3 = j + j * a_dim1;
		    q__1.r = tscal * a[i__3].r, q__1.i = tscal * a[i__3].i;
		    tjjs.r = q__1.r, tjjs.i = q__1.i;
		} else {
		    tjjs.r = tscal, tjjs.i = 0.f;
		    if (tscal == 1.f) {
			goto L105;
		    }
		}
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			dabs(r__2));
		if (tjj > smlnum) {

/*                    abs(A(j,j)) > SMLNUM: */

		    if (tjj < 1.f) {
			if (xj > tjj * bignum) {

/*                          Scale x by 1/b(j). */

			    rec = 1.f / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    i__3 = j;
		    cladiv_(&q__1, &x[j], &tjjs);
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		} else if (tjj > 0.f) {

/*                    0 < abs(A(j,j)) <= SMLNUM: */

		    if (xj > tjj * bignum) {

/*
                         Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM
                         to avoid overflow when dividing by A(j,j).
*/

			rec = tjj * bignum / xj;
			if (cnorm[j] > 1.f) {

/*
                            Scale by 1/CNORM(j) to avoid overflow when
                            multiplying x(j) times column j.
*/

			    rec /= cnorm[j];
			}
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		    i__3 = j;
		    cladiv_(&q__1, &x[j], &tjjs);
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		} else {

/*
                      A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                      scale = 0, and compute a solution to A*x = 0.
*/

		    i__3 = *n;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L100: */
		    }
		    i__3 = j;
		    x[i__3].r = 1.f, x[i__3].i = 0.f;
		    xj = 1.f;
		    *scale = 0.f;
		    xmax = 0.f;
		}
L105:

/*
                Scale x if necessary to avoid overflow when adding a
                multiple of column j of A.
*/

		if (xj > 1.f) {
		    rec = 1.f / xj;
		    if (cnorm[j] > (bignum - xmax) * rec) {

/*                    Scale x by 1/(2*abs(x(j))). */

			rec *= .5f;
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
		    }
		} else if (xj * cnorm[j] > bignum - xmax) {

/*                 Scale x by 1/2. */

		    csscal_(n, &c_b1794, &x[1], &c__1);
		    *scale *= .5f;
		}

		if (upper) {
		    if (j > 1) {

/*
                      Compute the update
                         x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
*/

			i__3 = j - 1;
			i__4 = j;
			q__2.r = -x[i__4].r, q__2.i = -x[i__4].i;
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			caxpy_(&i__3, &q__1, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			i__3 = j - 1;
			i__ = icamax_(&i__3, &x[1], &c__1);
			i__3 = i__;
			xmax = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 =
				r_imag(&x[i__]), dabs(r__2));
		    }
		} else {
		    if (j < *n) {

/*
                      Compute the update
                         x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
*/

			i__3 = *n - j;
			i__4 = j;
			q__2.r = -x[i__4].r, q__2.i = -x[i__4].i;
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			caxpy_(&i__3, &q__1, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			i__3 = *n - j;
			i__ = j + icamax_(&i__3, &x[j + 1], &c__1);
			i__3 = i__;
			xmax = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 =
				r_imag(&x[i__]), dabs(r__2));
		    }
		}
/* L110: */
	    }

	} else if (lsame_(trans, "T")) {

/*           Solve A**T * x = b */

	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*
                Compute x(j) = b(j) - sum A(k,j)*x(k).
                                      k<>j
*/

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]),
			dabs(r__2));
		uscal.r = tscal, uscal.i = 0.f;
		rec = 1.f / dmax(xmax,1.f);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5f;
		    if (nounit) {
			i__3 = j + j * a_dim1;
			q__1.r = tscal * a[i__3].r, q__1.i = tscal * a[i__3]
				.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > 1.f) {

/*
                         Divide by A(j,j) when scaling x if A(j,j) > 1.

   Computing MIN
*/
			r__1 = 1.f, r__2 = rec * tjj;
			rec = dmin(r__1,r__2);
			cladiv_(&q__1, &uscal, &tjjs);
			uscal.r = q__1.r, uscal.i = q__1.i;
		    }
		    if (rec < 1.f) {
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0.f, csumj.i = 0.f;
		if (uscal.r == 1.f && uscal.i == 0.f) {

/*
                   If the scaling needed for A in the dot product is 1,
                   call CDOTU to perform the dot product.
*/

		    if (upper) {
			i__3 = j - 1;
			cdotu_(&q__1, &i__3, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    } else if (j < *n) {
			i__3 = *n - j;
			cdotu_(&q__1, &i__3, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
			i__3 = j - 1;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + j * a_dim1;
			    q__3.r = a[i__4].r * uscal.r - a[i__4].i *
				    uscal.i, q__3.i = a[i__4].r * uscal.i + a[
				    i__4].i * uscal.r;
			    i__5 = i__;
			    q__2.r = q__3.r * x[i__5].r - q__3.i * x[i__5].i,
				    q__2.i = q__3.r * x[i__5].i + q__3.i * x[
				    i__5].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i +
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L120: */
			}
		    } else if (j < *n) {
			i__3 = *n;
			for (i__ = j + 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + j * a_dim1;
			    q__3.r = a[i__4].r * uscal.r - a[i__4].i *
				    uscal.i, q__3.i = a[i__4].r * uscal.i + a[
				    i__4].i * uscal.r;
			    i__5 = i__;
			    q__2.r = q__3.r * x[i__5].r - q__3.i * x[i__5].i,
				    q__2.i = q__3.r * x[i__5].i + q__3.i * x[
				    i__5].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i +
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L130: */
			}
		    }
		}

		q__1.r = tscal, q__1.i = 0.f;
		if (uscal.r == q__1.r && uscal.i == q__1.i) {

/*
                   Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
                   was not used to scale the dotproduct.
*/

		    i__3 = j;
		    i__4 = j;
		    q__1.r = x[i__4].r - csumj.r, q__1.i = x[i__4].i -
			    csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		    if (nounit) {
			i__3 = j + j * a_dim1;
			q__1.r = tscal * a[i__3].r, q__1.i = tscal * a[i__3]
				.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
			if (tscal == 1.f) {
			    goto L145;
			}
		    }

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.f) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1.f / xj;
				csscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else if (tjj > 0.f) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else {

/*
                         A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                         scale = 0 and compute a solution to A**T *x = 0.
*/

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L140: */
			}
			i__3 = j;
			x[i__3].r = 1.f, x[i__3].i = 0.f;
			*scale = 0.f;
			xmax = 0.f;
		    }
L145:
		    ;
		} else {

/*
                   Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
                   product has already been divided by 1/A(j,j).
*/

		    i__3 = j;
		    cladiv_(&q__2, &x[j], &tjjs);
		    q__1.r = q__2.r - csumj.r, q__1.i = q__2.i - csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		}
/* Computing MAX */
		i__3 = j;
		r__3 = xmax, r__4 = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 =
			r_imag(&x[j]), dabs(r__2));
		xmax = dmax(r__3,r__4);
/* L150: */
	    }

	} else {

/*           Solve A**H * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*
                Compute x(j) = b(j) - sum A(k,j)*x(k).
                                      k<>j
*/

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]),
			dabs(r__2));
		uscal.r = tscal, uscal.i = 0.f;
		rec = 1.f / dmax(xmax,1.f);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5f;
		    if (nounit) {
			r_cnjg(&q__2, &a[j + j * a_dim1]);
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > 1.f) {

/*
                         Divide by A(j,j) when scaling x if A(j,j) > 1.

   Computing MIN
*/
			r__1 = 1.f, r__2 = rec * tjj;
			rec = dmin(r__1,r__2);
			cladiv_(&q__1, &uscal, &tjjs);
			uscal.r = q__1.r, uscal.i = q__1.i;
		    }
		    if (rec < 1.f) {
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0.f, csumj.i = 0.f;
		if (uscal.r == 1.f && uscal.i == 0.f) {

/*
                   If the scaling needed for A in the dot product is 1,
                   call CDOTC to perform the dot product.
*/

		    if (upper) {
			i__3 = j - 1;
			cdotc_(&q__1, &i__3, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    } else if (j < *n) {
			i__3 = *n - j;
			cdotc_(&q__1, &i__3, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
			i__3 = j - 1;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    r_cnjg(&q__4, &a[i__ + j * a_dim1]);
			    q__3.r = q__4.r * uscal.r - q__4.i * uscal.i,
				    q__3.i = q__4.r * uscal.i + q__4.i *
				    uscal.r;
			    i__4 = i__;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i,
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i +
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L160: */
			}
		    } else if (j < *n) {
			i__3 = *n;
			for (i__ = j + 1; i__ <= i__3; ++i__) {
			    r_cnjg(&q__4, &a[i__ + j * a_dim1]);
			    q__3.r = q__4.r * uscal.r - q__4.i * uscal.i,
				    q__3.i = q__4.r * uscal.i + q__4.i *
				    uscal.r;
			    i__4 = i__;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i,
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i +
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L170: */
			}
		    }
		}

		q__1.r = tscal, q__1.i = 0.f;
		if (uscal.r == q__1.r && uscal.i == q__1.i) {

/*
                   Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
                   was not used to scale the dotproduct.
*/

		    i__3 = j;
		    i__4 = j;
		    q__1.r = x[i__4].r - csumj.r, q__1.i = x[i__4].i -
			    csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		    if (nounit) {
			r_cnjg(&q__2, &a[j + j * a_dim1]);
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
			if (tscal == 1.f) {
			    goto L185;
			}
		    }

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.f) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1.f / xj;
				csscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else if (tjj > 0.f) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else {

/*
                         A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                         scale = 0 and compute a solution to A**H *x = 0.
*/

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L180: */
			}
			i__3 = j;
			x[i__3].r = 1.f, x[i__3].i = 0.f;
			*scale = 0.f;
			xmax = 0.f;
		    }
L185:
		    ;
		} else {

/*
                   Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
                   product has already been divided by 1/A(j,j).
*/

		    i__3 = j;
		    cladiv_(&q__2, &x[j], &tjjs);
		    q__1.r = q__2.r - csumj.r, q__1.i = q__2.i - csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		}
/* Computing MAX */
		i__3 = j;
		r__3 = xmax, r__4 = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 =
			r_imag(&x[j]), dabs(r__2));
		xmax = dmax(r__3,r__4);
/* L190: */
	    }
	}
	*scale /= tscal;
    }

/*     Scale the column norms by 1/TSCAL for return. */

    if (tscal != 1.f) {
	r__1 = 1.f / tscal;
	sscal_(n, &r__1, &cnorm[1], &c__1);
    }

    return 0;

/*     End of CLATRS */

} /* clatrs_ */

/* Subroutine */ int clauu2_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1;

    /* Local variables */
    static integer i__;
    static real aii;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *);
    static logical upper;
    extern /* Subroutine */ int clacgv_(integer *, complex *, integer *),
	    csscal_(integer *, real *, complex *, integer *), xerbla_(char *,
	    integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLAUU2 computes the product U * U' or L' * L, where the triangular
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CLAUU2", &i__1);
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
	    i__2 = i__ + i__ * a_dim1;
	    aii = a[i__2].r;
	    if (i__ < *n) {
		i__2 = i__ + i__ * a_dim1;
		i__3 = *n - i__;
		cdotc_(&q__1, &i__3, &a[i__ + (i__ + 1) * a_dim1], lda, &a[
			i__ + (i__ + 1) * a_dim1], lda);
		r__1 = aii * aii + q__1.r;
		a[i__2].r = r__1, a[i__2].i = 0.f;
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		q__1.r = aii, q__1.i = 0.f;
		cgemv_("No transpose", &i__2, &i__3, &c_b56, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			q__1, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		clacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
	    } else {
		csscal_(&i__, &aii, &a[i__ * a_dim1 + 1], &c__1);
	    }
/* L10: */
	}

    } else {

/*        Compute the product L' * L. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    aii = a[i__2].r;
	    if (i__ < *n) {
		i__2 = i__ + i__ * a_dim1;
		i__3 = *n - i__;
		cdotc_(&q__1, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		r__1 = aii * aii + q__1.r;
		a[i__2].r = r__1, a[i__2].i = 0.f;
		i__2 = i__ - 1;
		clacgv_(&i__2, &a[i__ + a_dim1], lda);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		q__1.r = aii, q__1.i = 0.f;
		cgemv_("Conjugate transpose", &i__2, &i__3, &c_b56, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			q__1, &a[i__ + a_dim1], lda);
		i__2 = i__ - 1;
		clacgv_(&i__2, &a[i__ + a_dim1], lda);
	    } else {
		csscal_(&i__, &aii, &a[i__ + a_dim1], lda);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of CLAUU2 */

} /* clauu2_ */

/* Subroutine */ int clauum_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, ib, nb;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *), cherk_(char *,
	    char *, integer *, integer *, real *, complex *, integer *, real *
	    , complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrmm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *);
    static logical upper;
    extern /* Subroutine */ int clauu2_(char *, integer *, complex *, integer
	    *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CLAUUM computes the product U * U' or L' * L, where the triangular
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CLAUUM", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "CLAUUM", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	clauu2_(uplo, n, &a[a_offset], lda, info);
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
		ctrmm_("Right", "Upper", "Conjugate transpose", "Non-unit", &
			i__3, &ib, &c_b56, &a[i__ + i__ * a_dim1], lda, &a[
			i__ * a_dim1 + 1], lda);
		clauu2_("Upper", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    cgemm_("No transpose", "Conjugate transpose", &i__3, &ib,
			    &i__4, &c_b56, &a[(i__ + ib) * a_dim1 + 1], lda, &
			    a[i__ + (i__ + ib) * a_dim1], lda, &c_b56, &a[i__
			    * a_dim1 + 1], lda);
		    i__3 = *n - i__ - ib + 1;
		    cherk_("Upper", "No transpose", &ib, &i__3, &c_b871, &a[
			    i__ + (i__ + ib) * a_dim1], lda, &c_b871, &a[i__
			    + i__ * a_dim1], lda);
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
		ctrmm_("Left", "Lower", "Conjugate transpose", "Non-unit", &
			ib, &i__3, &c_b56, &a[i__ + i__ * a_dim1], lda, &a[
			i__ + a_dim1], lda);
		clauu2_("Lower", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    cgemm_("Conjugate transpose", "No transpose", &ib, &i__3,
			    &i__4, &c_b56, &a[i__ + ib + i__ * a_dim1], lda, &
			    a[i__ + ib + a_dim1], lda, &c_b56, &a[i__ +
			    a_dim1], lda);
		    i__3 = *n - i__ - ib + 1;
		    cherk_("Lower", "Conjugate transpose", &ib, &i__3, &
			    c_b871, &a[i__ + ib + i__ * a_dim1], lda, &c_b871,
			     &a[i__ + i__ * a_dim1], lda);
		}
/* L20: */
	    }
	}
    }

    return 0;

/*     End of CLAUUM */

} /* clauum_ */

/* Subroutine */ int cpotf2_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer j;
    static real ajj;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *);
    static logical upper;
    extern /* Subroutine */ int clacgv_(integer *, complex *, integer *),
	    csscal_(integer *, real *, complex *, integer *), xerbla_(char *,
	    integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CPOTF2 computes the Cholesky factorization of a complex Hermitian
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
            Hermitian matrix A is stored.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
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
	xerbla_("CPOTF2", &i__1);
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

	    i__2 = j + j * a_dim1;
	    r__1 = a[i__2].r;
	    i__3 = j - 1;
	    cdotc_(&q__2, &i__3, &a[j * a_dim1 + 1], &c__1, &a[j * a_dim1 + 1]
		    , &c__1);
	    q__1.r = r__1 - q__2.r, q__1.i = -q__2.i;
	    ajj = q__1.r;
	    if (ajj <= 0.f) {
		i__2 = j + j * a_dim1;
		a[i__2].r = ajj, a[i__2].i = 0.f;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    i__2 = j + j * a_dim1;
	    a[i__2].r = ajj, a[i__2].i = 0.f;

/*           Compute elements J+1:N of row J. */

	    if (j < *n) {
		i__2 = j - 1;
		clacgv_(&i__2, &a[j * a_dim1 + 1], &c__1);
		i__2 = j - 1;
		i__3 = *n - j;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("Transpose", &i__2, &i__3, &q__1, &a[(j + 1) * a_dim1
			+ 1], lda, &a[j * a_dim1 + 1], &c__1, &c_b56, &a[j + (
			j + 1) * a_dim1], lda);
		i__2 = j - 1;
		clacgv_(&i__2, &a[j * a_dim1 + 1], &c__1);
		i__2 = *n - j;
		r__1 = 1.f / ajj;
		csscal_(&i__2, &r__1, &a[j + (j + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness. */

	    i__2 = j + j * a_dim1;
	    r__1 = a[i__2].r;
	    i__3 = j - 1;
	    cdotc_(&q__2, &i__3, &a[j + a_dim1], lda, &a[j + a_dim1], lda);
	    q__1.r = r__1 - q__2.r, q__1.i = -q__2.i;
	    ajj = q__1.r;
	    if (ajj <= 0.f) {
		i__2 = j + j * a_dim1;
		a[i__2].r = ajj, a[i__2].i = 0.f;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    i__2 = j + j * a_dim1;
	    a[i__2].r = ajj, a[i__2].i = 0.f;

/*           Compute elements J+1:N of column J. */

	    if (j < *n) {
		i__2 = j - 1;
		clacgv_(&i__2, &a[j + a_dim1], lda);
		i__2 = *n - j;
		i__3 = j - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__2, &i__3, &q__1, &a[j + 1 + a_dim1]
			, lda, &a[j + a_dim1], lda, &c_b56, &a[j + 1 + j *
			a_dim1], &c__1);
		i__2 = j - 1;
		clacgv_(&i__2, &a[j + a_dim1], lda);
		i__2 = *n - j;
		r__1 = 1.f / ajj;
		csscal_(&i__2, &r__1, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }
    goto L40;

L30:
    *info = j;

L40:
    return 0;

/*     End of CPOTF2 */

} /* cpotf2_ */

/* Subroutine */ int cpotrf_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    complex q__1;

    /* Local variables */
    static integer j, jb, nb;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *,
	    integer *, complex *, complex *, integer *, complex *, integer *,
	    complex *, complex *, integer *), cherk_(char *,
	    char *, integer *, integer *, real *, complex *, integer *, real *
	    , complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrsm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *);
    static logical upper;
    extern /* Subroutine */ int cpotf2_(char *, integer *, complex *, integer
	    *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A.

    The factorization has the form
       A = U**H * U,  if UPLO = 'U', or
       A = L  * L**H,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H*U or A = L*L**H.

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
	xerbla_("CPOTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "CPOTRF", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);
    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code. */

	cpotf2_(uplo, n, &a[a_offset], lda, info);
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
		cherk_("Upper", "Conjugate transpose", &jb, &i__3, &c_b1150, &
			a[j * a_dim1 + 1], lda, &c_b871, &a[j + j * a_dim1],
			lda);
		cpotf2_("Upper", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block row. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("Conjugate transpose", "No transpose", &jb, &i__3,
			    &i__4, &q__1, &a[j * a_dim1 + 1], lda, &a[(j + jb)
			     * a_dim1 + 1], lda, &c_b56, &a[j + (j + jb) *
			    a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    ctrsm_("Left", "Upper", "Conjugate transpose", "Non-unit",
			     &jb, &i__3, &c_b56, &a[j + j * a_dim1], lda, &a[
			    j + (j + jb) * a_dim1], lda);
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
		cherk_("Lower", "No transpose", &jb, &i__3, &c_b1150, &a[j +
			a_dim1], lda, &c_b871, &a[j + j * a_dim1], lda);
		cpotf2_("Lower", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block column. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &i__3, &jb,
			    &i__4, &q__1, &a[j + jb + a_dim1], lda, &a[j +
			    a_dim1], lda, &c_b56, &a[j + jb + j * a_dim1],
			    lda);
		    i__3 = *n - j - jb + 1;
		    ctrsm_("Right", "Lower", "Conjugate transpose", "Non-unit"
			    , &i__3, &jb, &c_b56, &a[j + j * a_dim1], lda, &a[
			    j + jb + j * a_dim1], lda);
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

/*     End of CPOTRF */

} /* cpotrf_ */

/* Subroutine */ int cpotri_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), clauum_(
	    char *, integer *, complex *, integer *, integer *),
	    ctrtri_(char *, char *, integer *, complex *, integer *, integer *
	    );


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    CPOTRI computes the inverse of a complex Hermitian positive definite
    matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
    computed by CPOTRF.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the triangular factor U or L from the Cholesky
            factorization A = U**H*U or A = L*L**H, as computed by
            CPOTRF.
            On exit, the upper or lower triangle of the (Hermitian)
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
	xerbla_("CPOTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Invert the triangular Cholesky factor U or L. */

    ctrtri_(uplo, "Non-unit", n, &a[a_offset], lda, info);
    if (*info > 0) {
	return 0;
    }

/*     Form inv(U)*inv(U)' or inv(L)'*inv(L). */

    clauum_(uplo, n, &a[a_offset], lda, info);

    return 0;

/*     End of CPOTRI */

} /* cpotri_ */

/* Subroutine */ int cpotrs_(char *uplo, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrsm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by CPOTRF.

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

    A       (input) COMPLEX array, dimension (LDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by CPOTRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input/output) COMPLEX array, dimension (LDB,NRHS)
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
	xerbla_("CPOTRS", &i__1);
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

	ctrsm_("Left", "Upper", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b56, &a[a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	ctrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b56, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A*X = B where A = L*L'.

          Solve L*X = B, overwriting B with X.
*/

	ctrsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b56, &
		a[a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	ctrsm_("Left", "Lower", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b56, &a[a_offset], lda, &b[b_offset], ldb);
    }

    return 0;

/*     End of CPOTRS */

} /* cpotrs_ */

/* Subroutine */ int csrot_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    complex q__1, q__2, q__3;

    /* Local variables */
    static integer i__, ix, iy;
    static complex ctemp;


/*
       applies a plane rotation, where the cos and sin (c and s) are real
       and the vectors cx and cy are complex.
       jack dongarra, linpack, 3/11/78.

    =====================================================================
*/


    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*
          code for unequal increments or equal increments not equal
            to 1
*/

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
	i__3 = iy;
	q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
	q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	ctemp.r = q__1.r, ctemp.i = q__1.i;
	i__2 = iy;
	i__3 = iy;
	q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
	i__4 = ix;
	q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
	i__2 = ix;
	cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
	i__3 = i__;
	q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
	q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	ctemp.r = q__1.r, ctemp.i = q__1.i;
	i__2 = i__;
	i__3 = i__;
	q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
	i__4 = i__;
	q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
	i__2 = i__;
	cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
/* L30: */
    }
    return 0;
} /* csrot_ */

/* Subroutine */ int cstedc_(char *compz, integer *n, real *d__, real *e,
	complex *z__, integer *ldz, complex *work, integer *lwork, real *
	rwork, integer *lrwork, integer *iwork, integer *liwork, integer *
	info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, k, m;
    static real p;
    static integer ii, ll, end, lgn;
    static real eps, tiny;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *,
	    complex *, integer *);
    static integer lwmin;
    extern /* Subroutine */ int claed0_(integer *, integer *, real *, real *,
	    complex *, integer *, complex *, integer *, real *, integer *,
	    integer *);
    static integer start;
    extern /* Subroutine */ int clacrm_(integer *, integer *, complex *,
	    integer *, real *, integer *, complex *, integer *, real *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex
	    *, integer *, complex *, integer *), xerbla_(char *,
	    integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *), sstedc_(char *, integer *, real *, real *, real *,
	    integer *, real *, integer *, integer *, integer *, integer *), slaset_(char *, integer *, integer *, real *, real *,
	    real *, integer *);
    static integer liwmin, icompz;
    extern /* Subroutine */ int csteqr_(char *, integer *, real *, real *,
	    complex *, integer *, real *, integer *);
    static real orgnrm;
    extern doublereal slanst_(char *, integer *, real *, real *);
    extern /* Subroutine */ int ssterf_(integer *, real *, real *, integer *);
    static integer lrwmin;
    static logical lquery;
    static integer smlsiz;
    extern /* Subroutine */ int ssteqr_(char *, integer *, real *, real *,
	    real *, integer *, real *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CSTEDC computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.
    The eigenvectors of a full or band complex Hermitian matrix can also
    be found if CHETRD or CHPTRD or CHBTRD has been used to reduce this
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
            = 'V':  Compute eigenvectors of original Hermitian matrix
                    also.  On entry, Z contains the unitary matrix used
                    to reduce the original matrix to tridiagonal form.

    N       (input) INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D       (input/output) REAL array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) REAL array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    Z       (input/output) COMPLEX array, dimension (LDZ,N)
            On entry, if COMPZ = 'V', then Z contains the unitary
            matrix used in the reduction to tridiagonal form.
            On exit, if INFO = 0, then if COMPZ = 'V', Z contains the
            orthonormal eigenvectors of the original Hermitian matrix,
            and if COMPZ = 'I', Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.
            If  COMPZ = 'N', then Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.  LDZ >= 1.
            If eigenvectors are desired, then LDZ >= max(1,N).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If COMPZ = 'N' or 'I', or N <= 1, LWORK must be at least 1.
            If COMPZ = 'V' and N > 1, LWORK must be at least N*N.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace/output) REAL array,
                                           dimension (LRWORK)
            On exit, if INFO = 0, RWORK(1) returns the optimal LRWORK.

    LRWORK  (input) INTEGER
            The dimension of the array RWORK.
            If COMPZ = 'N' or N <= 1, LRWORK must be at least 1.
            If COMPZ = 'V' and N > 1, LRWORK must be at least
                           1 + 3*N + 2*N*lg N + 3*N**2 ,
                           where lg( N ) = smallest integer k such
                           that 2**k >= N.
            If COMPZ = 'I' and N > 1, LRWORK must be at least
                           1 + 4*N + 2*N**2 .

            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the RWORK array,
            returns this value as the first entry of the RWORK array, and
            no error message related to LRWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If COMPZ = 'N' or N <= 1, LIWORK must be at least 1.
            If COMPZ = 'V' or N > 1,  LIWORK must be at least
                                      6 + 6*N + 5*N*lg N.
            If COMPZ = 'I' or N > 1,  LIWORK must be at least
                                      3 + 5*N .

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
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1 || *lrwork == -1 || *liwork == -1;

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
	lwmin = 1;
	liwmin = 1;
	lrwmin = 1;
    } else {
	lgn = (integer) (log((real) (*n)) / log(2.f));
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	if (pow_ii(&c__2, &lgn) < *n) {
	    ++lgn;
	}
	if (icompz == 1) {
	    lwmin = *n * *n;
/* Computing 2nd power */
	    i__1 = *n;
	    lrwmin = *n * 3 + 1 + (*n << 1) * lgn + i__1 * i__1 * 3;
	    liwmin = *n * 6 + 6 + *n * 5 * lgn;
	} else if (icompz == 2) {
	    lwmin = 1;
/* Computing 2nd power */
	    i__1 = *n;
	    lrwmin = (*n << 2) + 1 + (i__1 * i__1 << 1);
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
    } else if (*lrwork < lrwmin && ! lquery) {
	*info = -10;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {
	work[1].r = (real) lwmin, work[1].i = 0.f;
	rwork[1] = (real) lrwmin;
	iwork[1] = liwmin;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CSTEDC", &i__1);
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
	    i__1 = z_dim1 + 1;
	    z__[i__1].r = 1.f, z__[i__1].i = 0.f;
	}
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "CSTEDC", " ", &c__0, &c__0, &c__0, &c__0, (
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
	    csteqr_("I", n, &d__[1], &e[1], &z__[z_offset], ldz, &rwork[1],
		    info);
	    return 0;
	} else {
	    csteqr_("V", n, &d__[1], &e[1], &z__[z_offset], ldz, &rwork[1],
		    info);
	    return 0;
	}
    }

/*     If COMPZ = 'I', we simply call SSTEDC instead. */

    if (icompz == 2) {
	slaset_("Full", n, n, &c_b1101, &c_b871, &rwork[1], n);
	ll = *n * *n + 1;
	i__1 = *lrwork - ll + 1;
	sstedc_("I", n, &d__[1], &e[1], &rwork[1], n, &rwork[ll], &i__1, &
		iwork[1], liwork, info);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * z_dim1;
		i__4 = (j - 1) * *n + i__;
		z__[i__3].r = rwork[i__4], z__[i__3].i = 0.f;
/* L10: */
	    }
/* L20: */
	}
	return 0;
    }

/*
       From now on, only option left to be handled is COMPZ = 'V',
       i.e. ICOMPZ = 1.

       Scale.
*/

    orgnrm = slanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.f) {
	return 0;
    }

    eps = slamch_("Epsilon");

    start = 1;

/*     while ( START <= N ) */

L30:
    if (start <= *n) {

/*
       Let END be the position of the next subdiagonal entry such that
       E( END ) <= TINY or END = N if no such subdiagonal exists.  The
       matrix identified by the elements between START and END
       constitutes an independent sub-problem.
*/

	end = start;
L40:
	if (end < *n) {
	    tiny = eps * sqrt((r__1 = d__[end], dabs(r__1))) * sqrt((r__2 =
		    d__[end + 1], dabs(r__2)));
	    if ((r__1 = e[end], dabs(r__1)) > tiny) {
		++end;
		goto L40;
	    }
	}

/*        (Sub) Problem determined.  Compute its size and solve it. */

	m = end - start + 1;
	if (m > smlsiz) {
	    *info = smlsiz;

/*           Scale. */

	    orgnrm = slanst_("M", &m, &d__[start], &e[start]);
	    slascl_("G", &c__0, &c__0, &orgnrm, &c_b871, &m, &c__1, &d__[
		    start], &m, info);
	    i__1 = m - 1;
	    i__2 = m - 1;
	    slascl_("G", &c__0, &c__0, &orgnrm, &c_b871, &i__1, &c__1, &e[
		    start], &i__2, info);

	    claed0_(n, &m, &d__[start], &e[start], &z__[start * z_dim1 + 1],
		    ldz, &work[1], n, &rwork[1], &iwork[1], info);
	    if (*info > 0) {
		*info = (*info / (m + 1) + start - 1) * (*n + 1) + *info % (m
			+ 1) + start - 1;
		return 0;
	    }

/*           Scale back. */

	    slascl_("G", &c__0, &c__0, &c_b871, &orgnrm, &m, &c__1, &d__[
		    start], &m, info);

	} else {
	    ssteqr_("I", &m, &d__[start], &e[start], &rwork[1], &m, &rwork[m *
		     m + 1], info);
	    clacrm_(n, &m, &z__[start * z_dim1 + 1], ldz, &rwork[1], &m, &
		    work[1], n, &rwork[m * m + 1]);
	    clacpy_("A", n, &m, &work[1], n, &z__[start * z_dim1 + 1], ldz);
	    if (*info > 0) {
		*info = start * (*n + 1) + end;
		return 0;
	    }
	}

	start = end + 1;
	goto L30;
    }

/*
       endwhile

       If the problem split any number of times, then the eigenvalues
       will not be properly ordered.  Here we permute the eigenvalues
       (and the associated eigenvectors) into ascending order.
*/

    if (m != *n) {

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
/* L50: */
	    }
	    if (k != i__) {
		d__[k] = d__[i__];
		d__[i__] = p;
		cswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1 + 1],
			 &c__1);
	    }
/* L60: */
	}
    }

    work[1].r = (real) lwmin, work[1].i = 0.f;
    rwork[1] = (real) lrwmin;
    iwork[1] = liwmin;

    return 0;

/*     End of CSTEDC */

} /* cstedc_ */

/* Subroutine */ int csteqr_(char *compz, integer *n, real *d__, real *e,
	complex *z__, integer *ldz, real *work, integer *info)
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
    extern /* Subroutine */ int clasr_(char *, char *, char *, integer *,
	    integer *, real *, real *, complex *, integer *);
    static real anorm;
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *,
	    complex *, integer *);
    static integer lendm1, lendp1;
    extern /* Subroutine */ int slaev2_(real *, real *, real *, real *, real *
	    , real *, real *);
    extern doublereal slapy2_(real *, real *);
    static integer iscale;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int claset_(char *, integer *, integer *, complex
	    *, complex *, complex *, integer *);
    static real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static real safmax;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *,
	    real *, integer *, integer *, real *, integer *, integer *);
    static integer lendsv;
    extern /* Subroutine */ int slartg_(real *, real *, real *, real *, real *
	    );
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

    CSTEQR computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the implicit QL or QR method.
    The eigenvectors of a full or band complex Hermitian matrix can also
    be found if CHETRD or CHPTRD or CHBTRD has been used to reduce this
    matrix to tridiagonal form.

    Arguments
    =========

    COMPZ   (input) CHARACTER*1
            = 'N':  Compute eigenvalues only.
            = 'V':  Compute eigenvalues and eigenvectors of the original
                    Hermitian matrix.  On entry, Z must contain the
                    unitary matrix used to reduce the original matrix
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

    Z       (input/output) COMPLEX array, dimension (LDZ, N)
            On entry, if  COMPZ = 'V', then Z contains the unitary
            matrix used in the reduction to tridiagonal form.
            On exit, if INFO = 0, then if COMPZ = 'V', Z contains the
            orthonormal eigenvectors of the original Hermitian matrix,
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
                  matrix which is unitarily similar to the original
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
	xerbla_("CSTEQR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (icompz == 2) {
	    i__1 = z_dim1 + 1;
	    z__[i__1].r = 1.f, z__[i__1].i = 0.f;
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
	claset_("Full", n, n, &c_b55, &c_b56, &z__[z_offset], ldz);
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
		clasr_("R", "V", "B", n, &c__2, &work[l], &work[*n - 1 + l], &
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
	r__ = slapy2_(&g, &c_b871);
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
	    clasr_("R", "V", "B", n, &mm, &work[l], &work[*n - 1 + l], &z__[l
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
		clasr_("R", "V", "F", n, &c__2, &work[m], &work[*n - 1 + m], &
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
	r__ = slapy2_(&g, &c_b871);
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
	    clasr_("R", "V", "F", n, &mm, &work[m], &work[*n - 1 + m], &z__[m
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

    if (jtot == nmaxit) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (e[i__] != 0.f) {
		++(*info);
	    }
/* L150: */
	}
	return 0;
    }
    goto L10;

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
		cswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1 + 1],
			 &c__1);
	    }
/* L180: */
	}
    }
    return 0;

/*     End of CSTEQR */

} /* csteqr_ */

/* Subroutine */ int ctrevc_(char *side, char *howmny, logical *select,
	integer *n, complex *t, integer *ldt, complex *vl, integer *ldvl,
	complex *vr, integer *ldvr, integer *mm, integer *m, complex *work,
	real *rwork, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3;
    complex q__1, q__2;

    /* Builtin functions */
    double r_imag(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, j, k, ii, ki, is;
    static real ulp;
    static logical allv;
    static real unfl, ovfl, smin;
    static logical over;
    static real scale;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *);
    static real remax;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *,
	    complex *, integer *);
    static logical leftv, bothv, somev;
    extern /* Subroutine */ int slabad_(real *, real *);
    extern integer icamax_(integer *, complex *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer
	    *), xerbla_(char *, integer *), clatrs_(char *, char *,
	    char *, char *, integer *, complex *, integer *, complex *, real *
	    , real *, integer *);
    extern doublereal scasum_(integer *, complex *, integer *);
    static logical rightv;
    static real smlnum;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CTREVC computes some or all of the right and/or left eigenvectors of
    a complex upper triangular matrix T.

    The right eigenvector x and the left eigenvector y of T corresponding
    to an eigenvalue w are defined by:

                 T*x = w*x,     y'*T = w*y'

    where y' denotes the conjugate transpose of the vector y.

    If all eigenvectors are requested, the routine may either return the
    matrices X and/or Y of right or left eigenvectors of T, or the
    products Q*X and/or Q*Y, where Q is an input unitary
    matrix. If T was obtained from the Schur factorization of an
    original matrix A = Q*T*Q', then Q*X and Q*Y are the matrices of
    right or left eigenvectors of A.

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

    SELECT  (input) LOGICAL array, dimension (N)
            If HOWMNY = 'S', SELECT specifies the eigenvectors to be
            computed.
            If HOWMNY = 'A' or 'B', SELECT is not referenced.
            To select the eigenvector corresponding to the j-th
            eigenvalue, SELECT(j) must be set to .TRUE..

    N       (input) INTEGER
            The order of the matrix T. N >= 0.

    T       (input/output) COMPLEX array, dimension (LDT,N)
            The upper triangular matrix T.  T is modified, but restored
            on exit.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= max(1,N).

    VL      (input/output) COMPLEX array, dimension (LDVL,MM)
            On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
            contain an N-by-N matrix Q (usually the unitary matrix Q of
            Schur vectors returned by CHSEQR).
            On exit, if SIDE = 'L' or 'B', VL contains:
            if HOWMNY = 'A', the matrix Y of left eigenvectors of T;
                             VL is lower triangular. The i-th column
                             VL(i) of VL is the eigenvector corresponding
                             to T(i,i).
            if HOWMNY = 'B', the matrix Q*Y;
            if HOWMNY = 'S', the left eigenvectors of T specified by
                             SELECT, stored consecutively in the columns
                             of VL, in the same order as their
                             eigenvalues.
            If SIDE = 'R', VL is not referenced.

    LDVL    (input) INTEGER
            The leading dimension of the array VL.  LDVL >= max(1,N) if
            SIDE = 'L' or 'B'; LDVL >= 1 otherwise.

    VR      (input/output) COMPLEX array, dimension (LDVR,MM)
            On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
            contain an N-by-N matrix Q (usually the unitary matrix Q of
            Schur vectors returned by CHSEQR).
            On exit, if SIDE = 'R' or 'B', VR contains:
            if HOWMNY = 'A', the matrix X of right eigenvectors of T;
                             VR is upper triangular. The i-th column
                             VR(i) of VR is the eigenvector corresponding
                             to T(i,i).
            if HOWMNY = 'B', the matrix Q*X;
            if HOWMNY = 'S', the right eigenvectors of T specified by
                             SELECT, stored consecutively in the columns
                             of VR, in the same order as their
                             eigenvalues.
            If SIDE = 'L', VR is not referenced.

    LDVR    (input) INTEGER
            The leading dimension of the array VR.  LDVR >= max(1,N) if
             SIDE = 'R' or 'B'; LDVR >= 1 otherwise.

    MM      (input) INTEGER
            The number of columns in the arrays VL and/or VR. MM >= M.

    M       (output) INTEGER
            The number of columns in the arrays VL and/or VR actually
            used to store the eigenvectors.  If HOWMNY = 'A' or 'B', M
            is set to N.  Each selected eigenvector occupies one
            column.

    WORK    (workspace) COMPLEX array, dimension (2*N)

    RWORK   (workspace) REAL array, dimension (N)

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
    --rwork;

    /* Function Body */
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;

    allv = lsame_(howmny, "A");
    over = lsame_(howmny, "B");
    somev = lsame_(howmny, "S");

/*
       Set M to the number of columns required to store the selected
       eigenvectors.
*/

    if (somev) {
	*m = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (select[j]) {
		++(*m);
	    }
/* L10: */
	}
    } else {
	*m = *n;
    }

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
    } else if (*mm < *m) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTREVC", &i__1);
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

/*     Store the diagonal elements of T in working array WORK. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__ + *n;
	i__3 = i__ + i__ * t_dim1;
	work[i__2].r = t[i__3].r, work[i__2].i = t[i__3].i;
/* L20: */
    }

/*
       Compute 1-norm of each column of strictly upper triangular
       part of T to control overflow in triangular solver.
*/

    rwork[1] = 0.f;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	i__2 = j - 1;
	rwork[j] = scasum_(&i__2, &t[j * t_dim1 + 1], &c__1);
/* L30: */
    }

    if (rightv) {

/*        Compute right eigenvectors. */

	is = *m;
	for (ki = *n; ki >= 1; --ki) {

	    if (somev) {
		if (! select[ki]) {
		    goto L80;
		}
	    }
/* Computing MAX */
	    i__1 = ki + ki * t_dim1;
	    r__3 = ulp * ((r__1 = t[i__1].r, dabs(r__1)) + (r__2 = r_imag(&t[
		    ki + ki * t_dim1]), dabs(r__2)));
	    smin = dmax(r__3,smlnum);

	    work[1].r = 1.f, work[1].i = 0.f;

/*           Form right-hand side. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__2 = k;
		i__3 = k + ki * t_dim1;
		q__1.r = -t[i__3].r, q__1.i = -t[i__3].i;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
/* L40: */
	    }

/*
             Solve the triangular system:
                (T(1:KI-1,1:KI-1) - T(KI,KI))*X = SCALE*WORK.
*/

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__2 = k + k * t_dim1;
		i__3 = k + k * t_dim1;
		i__4 = ki + ki * t_dim1;
		q__1.r = t[i__3].r - t[i__4].r, q__1.i = t[i__3].i - t[i__4]
			.i;
		t[i__2].r = q__1.r, t[i__2].i = q__1.i;
		i__2 = k + k * t_dim1;
		if ((r__1 = t[i__2].r, dabs(r__1)) + (r__2 = r_imag(&t[k + k *
			 t_dim1]), dabs(r__2)) < smin) {
		    i__3 = k + k * t_dim1;
		    t[i__3].r = smin, t[i__3].i = 0.f;
		}
/* L50: */
	    }

	    if (ki > 1) {
		i__1 = ki - 1;
		clatrs_("Upper", "No transpose", "Non-unit", "Y", &i__1, &t[
			t_offset], ldt, &work[1], &scale, &rwork[1], info);
		i__1 = ki;
		work[i__1].r = scale, work[i__1].i = 0.f;
	    }

/*           Copy the vector x or Q*x to VR and normalize. */

	    if (! over) {
		ccopy_(&ki, &work[1], &c__1, &vr[is * vr_dim1 + 1], &c__1);

		ii = icamax_(&ki, &vr[is * vr_dim1 + 1], &c__1);
		i__1 = ii + is * vr_dim1;
		remax = 1.f / ((r__1 = vr[i__1].r, dabs(r__1)) + (r__2 =
			r_imag(&vr[ii + is * vr_dim1]), dabs(r__2)));
		csscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);

		i__1 = *n;
		for (k = ki + 1; k <= i__1; ++k) {
		    i__2 = k + is * vr_dim1;
		    vr[i__2].r = 0.f, vr[i__2].i = 0.f;
/* L60: */
		}
	    } else {
		if (ki > 1) {
		    i__1 = ki - 1;
		    q__1.r = scale, q__1.i = 0.f;
		    cgemv_("N", n, &i__1, &c_b56, &vr[vr_offset], ldvr, &work[
			    1], &c__1, &q__1, &vr[ki * vr_dim1 + 1], &c__1);
		}

		ii = icamax_(n, &vr[ki * vr_dim1 + 1], &c__1);
		i__1 = ii + ki * vr_dim1;
		remax = 1.f / ((r__1 = vr[i__1].r, dabs(r__1)) + (r__2 =
			r_imag(&vr[ii + ki * vr_dim1]), dabs(r__2)));
		csscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
	    }

/*           Set back the original diagonal elements of T. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__2 = k + k * t_dim1;
		i__3 = k + *n;
		t[i__2].r = work[i__3].r, t[i__2].i = work[i__3].i;
/* L70: */
	    }

	    --is;
L80:
	    ;
	}
    }

    if (leftv) {

/*        Compute left eigenvectors. */

	is = 1;
	i__1 = *n;
	for (ki = 1; ki <= i__1; ++ki) {

	    if (somev) {
		if (! select[ki]) {
		    goto L130;
		}
	    }
/* Computing MAX */
	    i__2 = ki + ki * t_dim1;
	    r__3 = ulp * ((r__1 = t[i__2].r, dabs(r__1)) + (r__2 = r_imag(&t[
		    ki + ki * t_dim1]), dabs(r__2)));
	    smin = dmax(r__3,smlnum);

	    i__2 = *n;
	    work[i__2].r = 1.f, work[i__2].i = 0.f;

/*           Form right-hand side. */

	    i__2 = *n;
	    for (k = ki + 1; k <= i__2; ++k) {
		i__3 = k;
		r_cnjg(&q__2, &t[ki + k * t_dim1]);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		work[i__3].r = q__1.r, work[i__3].i = q__1.i;
/* L90: */
	    }

/*
             Solve the triangular system:
                (T(KI+1:N,KI+1:N) - T(KI,KI))'*X = SCALE*WORK.
*/

	    i__2 = *n;
	    for (k = ki + 1; k <= i__2; ++k) {
		i__3 = k + k * t_dim1;
		i__4 = k + k * t_dim1;
		i__5 = ki + ki * t_dim1;
		q__1.r = t[i__4].r - t[i__5].r, q__1.i = t[i__4].i - t[i__5]
			.i;
		t[i__3].r = q__1.r, t[i__3].i = q__1.i;
		i__3 = k + k * t_dim1;
		if ((r__1 = t[i__3].r, dabs(r__1)) + (r__2 = r_imag(&t[k + k *
			 t_dim1]), dabs(r__2)) < smin) {
		    i__4 = k + k * t_dim1;
		    t[i__4].r = smin, t[i__4].i = 0.f;
		}
/* L100: */
	    }

	    if (ki < *n) {
		i__2 = *n - ki;
		clatrs_("Upper", "Conjugate transpose", "Non-unit", "Y", &
			i__2, &t[ki + 1 + (ki + 1) * t_dim1], ldt, &work[ki +
			1], &scale, &rwork[1], info);
		i__2 = ki;
		work[i__2].r = scale, work[i__2].i = 0.f;
	    }

/*           Copy the vector x or Q*x to VL and normalize. */

	    if (! over) {
		i__2 = *n - ki + 1;
		ccopy_(&i__2, &work[ki], &c__1, &vl[ki + is * vl_dim1], &c__1)
			;

		i__2 = *n - ki + 1;
		ii = icamax_(&i__2, &vl[ki + is * vl_dim1], &c__1) + ki - 1;
		i__2 = ii + is * vl_dim1;
		remax = 1.f / ((r__1 = vl[i__2].r, dabs(r__1)) + (r__2 =
			r_imag(&vl[ii + is * vl_dim1]), dabs(r__2)));
		i__2 = *n - ki + 1;
		csscal_(&i__2, &remax, &vl[ki + is * vl_dim1], &c__1);

		i__2 = ki - 1;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = k + is * vl_dim1;
		    vl[i__3].r = 0.f, vl[i__3].i = 0.f;
/* L110: */
		}
	    } else {
		if (ki < *n) {
		    i__2 = *n - ki;
		    q__1.r = scale, q__1.i = 0.f;
		    cgemv_("N", n, &i__2, &c_b56, &vl[(ki + 1) * vl_dim1 + 1],
			     ldvl, &work[ki + 1], &c__1, &q__1, &vl[ki *
			    vl_dim1 + 1], &c__1);
		}

		ii = icamax_(n, &vl[ki * vl_dim1 + 1], &c__1);
		i__2 = ii + ki * vl_dim1;
		remax = 1.f / ((r__1 = vl[i__2].r, dabs(r__1)) + (r__2 =
			r_imag(&vl[ii + ki * vl_dim1]), dabs(r__2)));
		csscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);
	    }

/*           Set back the original diagonal elements of T. */

	    i__2 = *n;
	    for (k = ki + 1; k <= i__2; ++k) {
		i__3 = k + k * t_dim1;
		i__4 = k + *n;
		t[i__3].r = work[i__4].r, t[i__3].i = work[i__4].i;
/* L120: */
	    }

	    ++is;
L130:
	    ;
	}
    }

    return 0;

/*     End of CTREVC */

} /* ctrevc_ */

/* Subroutine */ int ctrti2_(char *uplo, char *diag, integer *n, complex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    complex q__1;

    /* Builtin functions */
    void c_div(complex *, complex *, complex *);

    /* Local variables */
    static integer j;
    static complex ajj;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *);
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int ctrmv_(char *, char *, char *, integer *,
	    complex *, integer *, complex *, integer *), xerbla_(char *, integer *);
    static logical nounit;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CTRTI2 computes the inverse of a complex upper or lower triangular
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CTRTI2", &i__1);
	return 0;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		i__2 = j + j * a_dim1;
		c_div(&q__1, &c_b56, &a[j + j * a_dim1]);
		a[i__2].r = q__1.r, a[i__2].i = q__1.i;
		i__2 = j + j * a_dim1;
		q__1.r = -a[i__2].r, q__1.i = -a[i__2].i;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    } else {
		q__1.r = -1.f, q__1.i = -0.f;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    ctrmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a[j * a_dim1 + 1], &c__1);
	    i__2 = j - 1;
	    cscal_(&i__2, &ajj, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		i__1 = j + j * a_dim1;
		c_div(&q__1, &c_b56, &a[j + j * a_dim1]);
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = j + j * a_dim1;
		q__1.r = -a[i__1].r, q__1.i = -a[i__1].i;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    } else {
		q__1.r = -1.f, q__1.i = -0.f;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		ctrmv_("Lower", "No transpose", diag, &i__1, &a[j + 1 + (j +
			1) * a_dim1], lda, &a[j + 1 + j * a_dim1], &c__1);
		i__1 = *n - j;
		cscal_(&i__1, &ajj, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of CTRTI2 */

} /* ctrti2_ */

/* Subroutine */ int ctrtri_(char *uplo, char *diag, integer *n, complex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, i__1, i__2, i__3[2], i__4, i__5;
    complex q__1;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer j, jb, nb, nn;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrmm_(char *, char *, char *, char *,
	    integer *, integer *, complex *, complex *, integer *, complex *,
	    integer *), ctrsm_(char *, char *,
	     char *, char *, integer *, integer *, complex *, complex *,
	    integer *, complex *, integer *);
    static logical upper;
    extern /* Subroutine */ int ctrti2_(char *, char *, integer *, complex *,
	    integer *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical nounit;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CTRTRI computes the inverse of a complex upper or lower triangular
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
	xerbla_("CTRTRI", &i__1);
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
	    i__2 = *info + *info * a_dim1;
	    if (a[i__2].r == 0.f && a[i__2].i == 0.f) {
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
    i__3[0] = 1, a__1[0] = uplo;
    i__3[1] = 1, a__1[1] = diag;
    s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
    nb = ilaenv_(&c__1, "CTRTRI", ch__1, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	ctrti2_(uplo, diag, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code */

	if (upper) {

/*           Compute inverse of upper triangular matrix */

	    i__1 = *n;
	    i__2 = nb;
	    for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
/* Computing MIN */
		i__4 = nb, i__5 = *n - j + 1;
		jb = min(i__4,i__5);

/*              Compute rows 1:j-1 of current block column */

		i__4 = j - 1;
		ctrmm_("Left", "Upper", "No transpose", diag, &i__4, &jb, &
			c_b56, &a[a_offset], lda, &a[j * a_dim1 + 1], lda);
		i__4 = j - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		ctrsm_("Right", "Upper", "No transpose", diag, &i__4, &jb, &
			q__1, &a[j + j * a_dim1], lda, &a[j * a_dim1 + 1],
			lda);

/*              Compute inverse of current diagonal block */

		ctrti2_("Upper", diag, &jb, &a[j + j * a_dim1], lda, info);
/* L20: */
	    }
	} else {

/*           Compute inverse of lower triangular matrix */

	    nn = (*n - 1) / nb * nb + 1;
	    i__2 = -nb;
	    for (j = nn; i__2 < 0 ? j >= 1 : j <= 1; j += i__2) {
/* Computing MIN */
		i__1 = nb, i__4 = *n - j + 1;
		jb = min(i__1,i__4);
		if (j + jb <= *n) {

/*                 Compute rows j+jb:n of current block column */

		    i__1 = *n - j - jb + 1;
		    ctrmm_("Left", "Lower", "No transpose", diag, &i__1, &jb,
			    &c_b56, &a[j + jb + (j + jb) * a_dim1], lda, &a[j
			    + jb + j * a_dim1], lda);
		    i__1 = *n - j - jb + 1;
		    q__1.r = -1.f, q__1.i = -0.f;
		    ctrsm_("Right", "Lower", "No transpose", diag, &i__1, &jb,
			     &q__1, &a[j + j * a_dim1], lda, &a[j + jb + j *
			    a_dim1], lda);
		}

/*              Compute inverse of current diagonal block */

		ctrti2_("Lower", diag, &jb, &a[j + j * a_dim1], lda, info);
/* L30: */
	    }
	}
    }

    return 0;

/*     End of CTRTRI */

} /* ctrtri_ */

/* Subroutine */ int cung2r_(integer *m, integer *n, integer *k, complex *a,
	integer *lda, complex *tau, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    complex q__1;

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), clarf_(char *, integer *, integer *, complex *,
	    integer *, complex *, complex *, integer *, complex *),
	    xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CUNG2R generates an m by n complex matrix Q with orthonormal columns,
    which is defined as the first n columns of a product of k elementary
    reflectors of order m

          Q  =  H(1) H(2) . . . H(k)

    as returned by CGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by CGEQRF in the first k columns of its array
            argument A.
            On exit, the m by n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQRF.

    WORK    (workspace) COMPLEX array, dimension (N)

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
	xerbla_("CUNG2R", &i__1);
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
	    i__3 = l + j * a_dim1;
	    a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L10: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1.f, a[i__2].i = 0.f;
/* L20: */
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i) to A(i:m,i:n) from the left */

	if (i__ < *n) {
	    i__1 = i__ + i__ * a_dim1;
	    a[i__1].r = 1.f, a[i__1].i = 0.f;
	    i__1 = *m - i__ + 1;
	    i__2 = *n - i__;
	    clarf_("Left", &i__1, &i__2, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	}
	if (i__ < *m) {
	    i__1 = *m - i__;
	    i__2 = i__;
	    q__1.r = -tau[i__2].r, q__1.i = -tau[i__2].i;
	    cscal_(&i__1, &q__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
	}
	i__1 = i__ + i__ * a_dim1;
	i__2 = i__;
	q__1.r = 1.f - tau[i__2].r, q__1.i = 0.f - tau[i__2].i;
	a[i__1].r = q__1.r, a[i__1].i = q__1.i;

/*        Set A(1:i-1,i) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = l + i__ * a_dim1;
	    a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of CUNG2R */

} /* cung2r_ */

/* Subroutine */ int cungbr_(char *vect, integer *m, integer *n, integer *k,
	complex *a, integer *lda, complex *tau, complex *work, integer *lwork,
	 integer *info)
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
    extern /* Subroutine */ int cunglq_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *, integer *),
	     cungqr_(integer *, integer *, integer *, complex *, integer *,
	    complex *, complex *, integer *, integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CUNGBR generates one of the complex unitary matrices Q or P**H
    determined by CGEBRD when reducing a complex matrix A to bidiagonal
    form: A = Q * B * P**H.  Q and P**H are defined as products of
    elementary reflectors H(i) or G(i) respectively.

    If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
    is of order M:
    if m >= k, Q = H(1) H(2) . . . H(k) and CUNGBR returns the first n
    columns of Q, where m >= n >= k;
    if m < k, Q = H(1) H(2) . . . H(m-1) and CUNGBR returns Q as an
    M-by-M matrix.

    If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**H
    is of order N:
    if k < n, P**H = G(k) . . . G(2) G(1) and CUNGBR returns the first m
    rows of P**H, where n >= m >= k;
    if k >= n, P**H = G(n-1) . . . G(2) G(1) and CUNGBR returns P**H as
    an N-by-N matrix.

    Arguments
    =========

    VECT    (input) CHARACTER*1
            Specifies whether the matrix Q or the matrix P**H is
            required, as defined in the transformation applied by CGEBRD:
            = 'Q':  generate Q;
            = 'P':  generate P**H.

    M       (input) INTEGER
            The number of rows of the matrix Q or P**H to be returned.
            M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q or P**H to be returned.
            N >= 0.
            If VECT = 'Q', M >= N >= min(M,K);
            if VECT = 'P', N >= M >= min(N,K).

    K       (input) INTEGER
            If VECT = 'Q', the number of columns in the original M-by-K
            matrix reduced by CGEBRD.
            If VECT = 'P', the number of rows in the original K-by-N
            matrix reduced by CGEBRD.
            K >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by CGEBRD.
            On exit, the M-by-N matrix Q or P**H.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= M.

    TAU     (input) COMPLEX array, dimension
                                  (min(M,K)) if VECT = 'Q'
                                  (min(N,K)) if VECT = 'P'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i), which determines Q or P**H, as
            returned by CGEBRD in its array argument TAUQ or TAUP.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
	    nb = ilaenv_(&c__1, "CUNGQR", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	} else {
	    nb = ilaenv_(&c__1, "CUNGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	}
	lwkopt = max(1,mn) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNGBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
	return 0;
    }

    if (wantq) {

/*
          Form Q, determined by a call to CGEBRD to reduce an m-by-k
          matrix
*/

	if (*m >= *k) {

/*           If m >= k, assume m >= n >= k */

	    cungqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*
             If m < k, assume m = n

             Shift the vectors which define the elementary reflectors one
             column to the right, and set the first row and column of Q
             to those of the unit matrix
*/

	    for (j = *m; j >= 2; --j) {
		i__1 = j * a_dim1 + 1;
		a[i__1].r = 0.f, a[i__1].i = 0.f;
		i__1 = *m;
		for (i__ = j + 1; i__ <= i__1; ++i__) {
		    i__2 = i__ + j * a_dim1;
		    i__3 = i__ + (j - 1) * a_dim1;
		    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
/* L10: */
		}
/* L20: */
	    }
	    i__1 = a_dim1 + 1;
	    a[i__1].r = 1.f, a[i__1].i = 0.f;
	    i__1 = *m;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + a_dim1;
		a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L30: */
	    }
	    if (*m > 1) {

/*              Form Q(2:m,2:m) */

		i__1 = *m - 1;
		i__2 = *m - 1;
		i__3 = *m - 1;
		cungqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    } else {

/*
          Form P', determined by a call to CGEBRD to reduce a k-by-n
          matrix
*/

	if (*k < *n) {

/*           If k < n, assume k <= m <= n */

	    cunglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*
             If k >= n, assume m = n

             Shift the vectors which define the elementary reflectors one
             row downward, and set the first row and column of P' to
             those of the unit matrix
*/

	    i__1 = a_dim1 + 1;
	    a[i__1].r = 1.f, a[i__1].i = 0.f;
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + a_dim1;
		a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L40: */
	    }
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		for (i__ = j - 1; i__ >= 2; --i__) {
		    i__2 = i__ + j * a_dim1;
		    i__3 = i__ - 1 + j * a_dim1;
		    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
/* L50: */
		}
		i__2 = j * a_dim1 + 1;
		a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L60: */
	    }
	    if (*n > 1) {

/*              Form P'(2:n,2:n) */

		i__1 = *n - 1;
		i__2 = *n - 1;
		i__3 = *n - 1;
		cunglq_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNGBR */

} /* cungbr_ */

/* Subroutine */ int cunghr_(integer *n, integer *ilo, integer *ihi, complex *
	a, integer *lda, complex *tau, complex *work, integer *lwork, integer
	*info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, nb, nh, iinfo;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int cungqr_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *, integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CUNGHR generates a complex unitary matrix Q which is defined as the
    product of IHI-ILO elementary reflectors of order N, as returned by
    CGEHRD:

    Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix Q. N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            ILO and IHI must have the same values as in the previous call
            of CGEHRD. Q is equal to the unit matrix except in the
            submatrix Q(ilo+1:ihi,ilo+1:ihi).
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by CGEHRD.
            On exit, the N-by-N unitary matrix Q.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,N).

    TAU     (input) COMPLEX array, dimension (N-1)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEHRD.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
	nb = ilaenv_(&c__1, "CUNGQR", " ", &nh, &nh, &nh, &c_n1, (ftnlen)6, (
		ftnlen)1);
	lwkopt = max(1,nh) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNGHR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L10: */
	}
	i__2 = *ihi;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    i__4 = i__ + (j - 1) * a_dim1;
	    a[i__3].r = a[i__4].r, a[i__3].i = a[i__4].i;
/* L20: */
	}
	i__2 = *n;
	for (i__ = *ihi + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L30: */
	}
/* L40: */
    }
    i__1 = *ilo;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L50: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1.f, a[i__2].i = 0.f;
/* L60: */
    }
    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L70: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1.f, a[i__2].i = 0.f;
/* L80: */
    }

    if (nh > 0) {

/*        Generate Q(ilo+1:ihi,ilo+1:ihi) */

	cungqr_(&nh, &nh, &nh, &a[*ilo + 1 + (*ilo + 1) * a_dim1], lda, &tau[*
		ilo], &work[1], lwork, &iinfo);
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNGHR */

} /* cunghr_ */

/* Subroutine */ int cungl2_(integer *m, integer *n, integer *k, complex *a,
	integer *lda, complex *tau, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    complex q__1, q__2;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *,
	    integer *), clarf_(char *, integer *, integer *, complex *,
	    integer *, complex *, complex *, integer *, complex *),
	    clacgv_(integer *, complex *, integer *), xerbla_(char *, integer
	    *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CUNGL2 generates an m-by-n complex matrix Q with orthonormal rows,
    which is defined as the first m rows of a product of k elementary
    reflectors of order n

          Q  =  H(k)' . . . H(2)' H(1)'

    as returned by CGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by CGELQF in the first k rows of its array argument A.
            On exit, the m by n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGELQF.

    WORK    (workspace) COMPLEX array, dimension (M)

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
	xerbla_("CUNGL2", &i__1);
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
		i__3 = l + j * a_dim1;
		a[i__3].r = 0.f, a[i__3].i = 0.f;
/* L10: */
	    }
	    if (j > *k && j <= *m) {
		i__2 = j + j * a_dim1;
		a[i__2].r = 1.f, a[i__2].i = 0.f;
	    }
/* L20: */
	}
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i)' to A(i:m,i:n) from the right */

	if (i__ < *n) {
	    i__1 = *n - i__;
	    clacgv_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	    if (i__ < *m) {
		i__1 = i__ + i__ * a_dim1;
		a[i__1].r = 1.f, a[i__1].i = 0.f;
		i__1 = *m - i__;
		i__2 = *n - i__ + 1;
		r_cnjg(&q__1, &tau[i__]);
		clarf_("Right", &i__1, &i__2, &a[i__ + i__ * a_dim1], lda, &
			q__1, &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    i__1 = *n - i__;
	    i__2 = i__;
	    q__1.r = -tau[i__2].r, q__1.i = -tau[i__2].i;
	    cscal_(&i__1, &q__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	    i__1 = *n - i__;
	    clacgv_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	i__1 = i__ + i__ * a_dim1;
	r_cnjg(&q__2, &tau[i__]);
	q__1.r = 1.f - q__2.r, q__1.i = 0.f - q__2.i;
	a[i__1].r = q__1.r, a[i__1].i = q__1.i;

/*        Set A(i,1:i-1,i) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = i__ + l * a_dim1;
	    a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of CUNGL2 */

} /* cungl2_ */

/* Subroutine */ int cunglq_(integer *m, integer *n, integer *k, complex *a,
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int cungl2_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *), clarfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *,
	    complex *, integer *, complex *, integer *, complex *, integer *,
	    complex *, integer *), clarft_(
	    char *, char *, integer *, integer *, complex *, integer *,
	    complex *, complex *, integer *), xerbla_(char *,
	    integer *);
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

    CUNGLQ generates an M-by-N complex matrix Q with orthonormal rows,
    which is defined as the first M rows of a product of K elementary
    reflectors of order N

          Q  =  H(k)' . . . H(2)' H(1)'

    as returned by CGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by CGELQF in the first k rows of its array argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGELQF.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
            = 0:  successful exit;
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
    nb = ilaenv_(&c__1, "CUNGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*m) * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
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
	xerbla_("CUNGLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m <= 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "CUNGLQ", " ", m, n, k, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "CUNGLQ", " ", m, n, k, &c_n1,
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
		i__3 = i__ + j * a_dim1;
		a[i__3].r = 0.f, a[i__3].i = 0.f;
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
	cungl2_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
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
		clarft_("Forward", "Rowwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i+ib:m,i:n) from the right */

		i__2 = *m - i__ - ib + 1;
		i__3 = *n - i__ + 1;
		clarfb_("Right", "Conjugate transpose", "Forward", "Rowwise",
			&i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[
			ib + 1], &ldwork);
	    }

/*           Apply H' to columns i:n of current block */

	    i__2 = *n - i__ + 1;
	    cungl2_(&ib, &i__2, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set columns 1:i-1 of current block to zero */

	    i__2 = i__ - 1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + ib - 1;
		for (l = i__; l <= i__3; ++l) {
		    i__4 = l + j * a_dim1;
		    a[i__4].r = 0.f, a[i__4].i = 0.f;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1].r = (real) iws, work[1].i = 0.f;
    return 0;

/*     End of CUNGLQ */

} /* cunglq_ */

/* Subroutine */ int cungqr_(integer *m, integer *n, integer *k, complex *a,
	integer *lda, complex *tau, complex *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int cung2r_(integer *, integer *, integer *,
	    complex *, integer *, complex *, complex *, integer *), clarfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *,
	    complex *, integer *, complex *, integer *, complex *, integer *,
	    complex *, integer *), clarft_(
	    char *, char *, integer *, integer *, complex *, integer *,
	    complex *, complex *, integer *), xerbla_(char *,
	    integer *);
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

    CUNGQR generates an M-by-N complex matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by CGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by CGEQRF in the first k columns of its array
            argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQRF.

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
    nb = ilaenv_(&c__1, "CUNGQR", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*n) * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
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
	xerbla_("CUNGQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "CUNGQR", " ", m, n, k, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "CUNGQR", " ", m, n, k, &c_n1,
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
		i__3 = i__ + j * a_dim1;
		a[i__3].r = 0.f, a[i__3].i = 0.f;
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
	cung2r_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
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
		clarft_("Forward", "Columnwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i:m,i+ib:n) from the left */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__ - ib + 1;
		clarfb_("Left", "No transpose", "Forward", "Columnwise", &
			i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &
			work[ib + 1], &ldwork);
	    }

/*           Apply H to rows i:m of current block */

	    i__2 = *m - i__ + 1;
	    cung2r_(&i__2, &ib, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set rows 1:i-1 of current block to zero */

	    i__2 = i__ + ib - 1;
	    for (j = i__; j <= i__2; ++j) {
		i__3 = i__ - 1;
		for (l = 1; l <= i__3; ++l) {
		    i__4 = l + j * a_dim1;
		    a[i__4].r = 0.f, a[i__4].i = 0.f;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1].r = (real) iws, work[1].i = 0.f;
    return 0;

/*     End of CUNGQR */

} /* cungqr_ */

/* Subroutine */ int cunm2l_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, i1, i2, i3, mi, ni, nq;
    static complex aii;
    static logical left;
    static complex taui;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CUNM2L overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by CGEQLF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'C': apply Q' (Conjugate transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQLF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	xerbla_("CUNM2L", &i__1);
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

/*           H(i) or H(i)' is applied to C(1:m-k+i,1:n) */

	    mi = *m - *k + i__;
	} else {

/*           H(i) or H(i)' is applied to C(1:m,1:n-k+i) */

	    ni = *n - *k + i__;
	}

/*        Apply H(i) or H(i)' */

	if (notran) {
	    i__3 = i__;
	    taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	} else {
	    r_cnjg(&q__1, &tau[i__]);
	    taui.r = q__1.r, taui.i = q__1.i;
	}
	i__3 = nq - *k + i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = nq - *k + i__ + i__ * a_dim1;
	a[i__3].r = 1.f, a[i__3].i = 0.f;
	clarf_(side, &mi, &ni, &a[i__ * a_dim1 + 1], &c__1, &taui, &c__[
		c_offset], ldc, &work[1]);
	i__3 = nq - *k + i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
/* L10: */
    }
    return 0;

/*     End of CUNM2L */

} /* cunm2l_ */

/* Subroutine */ int cunm2r_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static complex aii;
    static logical left;
    static complex taui;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CUNM2R overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by CGEQRF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'C': apply Q' (Conjugate transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQRF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	xerbla_("CUNM2R", &i__1);
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

/*           H(i) or H(i)' is applied to C(i:m,1:n) */

	    mi = *m - i__ + 1;
	    ic = i__;
	} else {

/*           H(i) or H(i)' is applied to C(1:m,i:n) */

	    ni = *n - i__ + 1;
	    jc = i__;
	}

/*        Apply H(i) or H(i)' */

	if (notran) {
	    i__3 = i__;
	    taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	} else {
	    r_cnjg(&q__1, &tau[i__]);
	    taui.r = q__1.r, taui.i = q__1.i;
	}
	i__3 = i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = 1.f, a[i__3].i = 0.f;
	clarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], &c__1, &taui, &c__[ic
		+ jc * c_dim1], ldc, &work[1]);
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
/* L10: */
    }
    return 0;

/*     End of CUNM2R */

} /* cunm2r_ */

/* Subroutine */ int cunmbr_(char *vect, char *side, char *trans, integer *m,
	integer *n, integer *k, complex *a, integer *lda, complex *tau,
	complex *c__, integer *ldc, complex *work, integer *lwork, integer *
	info)
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
    extern /* Subroutine */ int cunmlq_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *);
    static logical notran;
    extern /* Subroutine */ int cunmqr_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *);
    static logical applyq;
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

    If VECT = 'Q', CUNMBR overwrites the general complex M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    If VECT = 'P', CUNMBR overwrites the general complex M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      P * C          C * P
    TRANS = 'C':      P**H * C       C * P**H

    Here Q and P**H are the unitary matrices determined by CGEBRD when
    reducing a complex matrix A to bidiagonal form: A = Q * B * P**H. Q
    and P**H are defined as products of elementary reflectors H(i) and
    G(i) respectively.

    Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
    order of the unitary matrix Q or P**H that is applied.

    If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
    if nq >= k, Q = H(1) H(2) . . . H(k);
    if nq < k, Q = H(1) H(2) . . . H(nq-1).

    If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
    if k < nq, P = G(1) G(2) . . . G(k);
    if k >= nq, P = G(1) G(2) . . . G(nq-1).

    Arguments
    =========

    VECT    (input) CHARACTER*1
            = 'Q': apply Q or Q**H;
            = 'P': apply P or P**H.

    SIDE    (input) CHARACTER*1
            = 'L': apply Q, Q**H, P or P**H from the Left;
            = 'R': apply Q, Q**H, P or P**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q or P;
            = 'C':  Conjugate transpose, apply Q**H or P**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            If VECT = 'Q', the number of columns in the original
            matrix reduced by CGEBRD.
            If VECT = 'P', the number of rows in the original
            matrix reduced by CGEBRD.
            K >= 0.

    A       (input) COMPLEX array, dimension
                                  (LDA,min(nq,K)) if VECT = 'Q'
                                  (LDA,nq)        if VECT = 'P'
            The vectors which define the elementary reflectors H(i) and
            G(i), whose products determine the matrices Q and P, as
            returned by CGEBRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If VECT = 'Q', LDA >= max(1,nq);
            if VECT = 'P', LDA >= max(1,min(nq,K)).

    TAU     (input) COMPLEX array, dimension (min(nq,K))
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i) which determines Q or P, as returned
            by CGEBRD in the array argument TAUQ or TAUP.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q
            or P*C or P**H*C or C*P or C*P**H.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
		nb = ilaenv_(&c__1, "CUNMQR", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "CUNMQR", ch__1, m, &i__1, &i__2, &c_n1, (
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
		nb = ilaenv_(&c__1, "CUNMLQ", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "CUNMLQ", ch__1, m, &i__1, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNMBR", &i__1);
	return 0;
    } else if (lquery) {
    }

/*     Quick return if possible */

    work[1].r = 1.f, work[1].i = 0.f;
    if (*m == 0 || *n == 0) {
	return 0;
    }

    if (applyq) {

/*        Apply Q */

	if (nq >= *k) {

/*           Q was determined by a call to CGEBRD with nq >= k */

	    cunmqr_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           Q was determined by a call to CGEBRD with nq < k */

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
	    cunmqr_(side, trans, &mi, &ni, &i__1, &a[a_dim1 + 2], lda, &tau[1]
		    , &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
	}
    } else {

/*        Apply P */

	if (notran) {
	    *(unsigned char *)transt = 'C';
	} else {
	    *(unsigned char *)transt = 'N';
	}
	if (nq > *k) {

/*           P was determined by a call to CGEBRD with nq > k */

	    cunmlq_(side, transt, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           P was determined by a call to CGEBRD with nq <= k */

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
	    cunmlq_(side, transt, &mi, &ni, &i__1, &a[(a_dim1 << 1) + 1], lda,
		     &tau[1], &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &
		    iinfo);
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMBR */

} /* cunmbr_ */

/* Subroutine */ int cunml2_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static complex aii;
    static logical left;
    static complex taui;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
	    , integer *, complex *, complex *, integer *, complex *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int clacgv_(integer *, complex *, integer *),
	    xerbla_(char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    CUNML2 overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k)' . . . H(2)' H(1)'

    as returned by CGELQF. Q is of order m if SIDE = 'L' and of order n
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q' from the Left
            = 'R': apply Q or Q' from the Right

    TRANS   (input) CHARACTER*1
            = 'N': apply Q  (No transpose)
            = 'C': apply Q' (Conjugate transpose)

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGELQF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX array, dimension
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	xerbla_("CUNML2", &i__1);
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

/*           H(i) or H(i)' is applied to C(i:m,1:n) */

	    mi = *m - i__ + 1;
	    ic = i__;
	} else {

/*           H(i) or H(i)' is applied to C(1:m,i:n) */

	    ni = *n - i__ + 1;
	    jc = i__;
	}

/*        Apply H(i) or H(i)' */

	if (notran) {
	    r_cnjg(&q__1, &tau[i__]);
	    taui.r = q__1.r, taui.i = q__1.i;
	} else {
	    i__3 = i__;
	    taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	}
	if (i__ < nq) {
	    i__3 = nq - i__;
	    clacgv_(&i__3, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	i__3 = i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = 1.f, a[i__3].i = 0.f;
	clarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], lda, &taui, &c__[ic +
		jc * c_dim1], ldc, &work[1]);
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
	if (i__ < nq) {
	    i__3 = nq - i__;
	    clacgv_(&i__3, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
/* L10: */
    }
    return 0;

/*     End of CUNML2 */

} /* cunml2_ */

/* Subroutine */ int cunmlq_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *lwork, integer *info)
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
    static complex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int cunml2_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *), clarfb_(char *, char *,
	    char *, char *, integer *, integer *, integer *, complex *,
	    integer *, complex *, integer *, complex *, integer *, complex *,
	    integer *), clarft_(char *, char *
	    , integer *, integer *, complex *, integer *, complex *, complex *
	    , integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
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

    CUNMLQ overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k)' . . . H(2)' H(1)'

    as returned by CGELQF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Conjugate transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGELQF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE 'L', and
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "CUNMLQ", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNMLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "CUNMLQ", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	cunml2_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
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
	    *(unsigned char *)transt = 'C';
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
	    clarft_("Forward", "Rowwise", &i__4, &ib, &a[i__ + i__ * a_dim1],
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

	    clarfb_(side, transt, "Forward", "Rowwise", &mi, &ni, &ib, &a[i__
		    + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc * c_dim1],
		    ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMLQ */

} /* cunmlq_ */

/* Subroutine */ int cunmql_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *lwork, integer *info)
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
    static complex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int cunm2l_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *), clarfb_(char *, char *,
	    char *, char *, integer *, integer *, integer *, complex *,
	    integer *, complex *, integer *, complex *, integer *, complex *,
	    integer *), clarft_(char *, char *
	    , integer *, integer *, complex *, integer *, complex *, complex *
	    , integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
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

    CUNMQL overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by CGEQLF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQLF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "CUNMQL", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNMQL", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "CUNMQL", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	cunm2l_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
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
	    clarft_("Backward", "Columnwise", &i__4, &ib, &a[i__ * a_dim1 + 1]
		    , lda, &tau[i__], t, &c__65);
	    if (left) {

/*              H or H' is applied to C(1:m-k+i+ib-1,1:n) */

		mi = *m - *k + i__ + ib - 1;
	    } else {

/*              H or H' is applied to C(1:m,1:n-k+i+ib-1) */

		ni = *n - *k + i__ + ib - 1;
	    }

/*           Apply H or H' */

	    clarfb_(side, trans, "Backward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ * a_dim1 + 1], lda, t, &c__65, &c__[c_offset], ldc, &
		    work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMQL */

} /* cunmql_ */

/* Subroutine */ int cunmqr_(char *side, char *trans, integer *m, integer *n,
	integer *k, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *lwork, integer *info)
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
    static complex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int cunm2r_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *), clarfb_(char *, char *,
	    char *, char *, integer *, integer *, integer *, complex *,
	    integer *, complex *, integer *, complex *, integer *, complex *,
	    integer *), clarft_(char *, char *
	    , integer *, integer *, complex *, integer *, complex *, complex *
	    , integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
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

    CUNMQR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by CGEQRF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Conjugate transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            CGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQRF.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
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
    } else if (! notran && ! lsame_(trans, "C")) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "CUNMQR", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNMQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "CUNMQR", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) {

/*        Use unblocked code */

	cunm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
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
	    clarft_("Forward", "Columnwise", &i__4, &ib, &a[i__ + i__ *
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

	    clarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc *
		    c_dim1], ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMQR */

} /* cunmqr_ */

/* Subroutine */ int cunmtr_(char *side, char *uplo, char *trans, integer *m,
	integer *n, complex *a, integer *lda, complex *tau, complex *c__,
	integer *ldc, complex *work, integer *lwork, integer *info)
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
    extern /* Subroutine */ int cunmql_(char *, char *, integer *, integer *,
	    integer *, complex *, integer *, complex *, complex *, integer *,
	    complex *, integer *, integer *), cunmqr_(char *,
	    char *, integer *, integer *, integer *, complex *, integer *,
	    complex *, complex *, integer *, complex *, integer *, integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    CUNMTR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix of order nq, with nq = m if
    SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
    nq-1 elementary reflectors, as returned by CHETRD:

    if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);

    if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    UPLO    (input) CHARACTER*1
            = 'U': Upper triangle of A contains elementary reflectors
                   from CHETRD;
            = 'L': Lower triangle of A contains elementary reflectors
                   from CHETRD.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Conjugate transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    A       (input) COMPLEX array, dimension
                                 (LDA,M) if SIDE = 'L'
                                 (LDA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by CHETRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.

    TAU     (input) COMPLEX array, dimension
                                 (M-1) if SIDE = 'L'
                                 (N-1) if SIDE = 'R'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CHETRD.

    C       (input/output) COMPLEX array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >=M*NB if SIDE = 'R', where NB is the optimal
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
	    "C")) {
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
		nb = ilaenv_(&c__1, "CUNMQL", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "CUNMQL", ch__1, m, &i__2, &i__3, &c_n1, (
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
		nb = ilaenv_(&c__1, "CUNMQR", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "CUNMQR", ch__1, m, &i__2, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("CUNMTR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || nq == 1) {
	work[1].r = 1.f, work[1].i = 0.f;
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

/*        Q was determined by a call to CHETRD with UPLO = 'U' */

	i__2 = nq - 1;
	cunmql_(side, trans, &mi, &ni, &i__2, &a[(a_dim1 << 1) + 1], lda, &
		tau[1], &c__[c_offset], ldc, &work[1], lwork, &iinfo);
    } else {

/*        Q was determined by a call to CHETRD with UPLO = 'L' */

	if (left) {
	    i1 = 2;
	    i2 = 1;
	} else {
	    i1 = 1;
	    i2 = 2;
	}
	i__2 = nq - 1;
	cunmqr_(side, trans, &mi, &ni, &i__2, &a[a_dim1 + 2], lda, &tau[1], &
		c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMTR */

} /* cunmtr_ */

