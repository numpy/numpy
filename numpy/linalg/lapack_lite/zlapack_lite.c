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



/* Table of constant values */

static integer c__1 = 1;
static doublecomplex c_b59 = {0.,0.};
static doublecomplex c_b60 = {1.,0.};
static integer c_n1 = -1;
static integer c__3 = 3;
static integer c__2 = 2;
static integer c__0 = 0;
static integer c__8 = 8;
static integer c__4 = 4;
static integer c__65 = 65;
static integer c__6 = 6;
static integer c__9 = 9;
static doublereal c_b324 = 0.;
static doublereal c_b1015 = 1.;
static integer c__15 = 15;
static logical c_false = FALSE_;
static doublereal c_b1294 = -1.;
static doublereal c_b2210 = .5;

/* Subroutine */ int zdrot_(integer *n, doublecomplex *cx, integer *incx,
	doublecomplex *cy, integer *incy, doublereal *c__, doublereal *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, ix, iy;
    static doublecomplex ctemp;


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
	z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
	i__3 = iy;
	z__3.r = *s * cy[i__3].r, z__3.i = *s * cy[i__3].i;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	ctemp.r = z__1.r, ctemp.i = z__1.i;
	i__2 = iy;
	i__3 = iy;
	z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
	i__4 = ix;
	z__3.r = *s * cx[i__4].r, z__3.i = *s * cx[i__4].i;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
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
	z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
	i__3 = i__;
	z__3.r = *s * cy[i__3].r, z__3.i = *s * cy[i__3].i;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	ctemp.r = z__1.r, ctemp.i = z__1.i;
	i__2 = i__;
	i__3 = i__;
	z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
	i__4 = i__;
	z__3.r = *s * cx[i__4].r, z__3.i = *s * cx[i__4].i;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
	i__2 = i__;
	cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
/* L30: */
    }
    return 0;
} /* zdrot_ */

/* Subroutine */ int zgebak_(char *job, char *side, integer *n, integer *ilo,
	integer *ihi, doublereal *scale, integer *m, doublecomplex *v,
	integer *ldv, integer *info)
{
    /* System generated locals */
    integer v_dim1, v_offset, i__1;

    /* Local variables */
    static integer i__, k;
    static doublereal s;
    static integer ii;
    extern logical lsame_(char *, char *);
    static logical leftv;
    extern /* Subroutine */ int zswap_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), xerbla_(char *, integer *),
	    zdscal_(integer *, doublereal *, doublecomplex *, integer *);
    static logical rightv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGEBAK forms the right or left eigenvectors of a complex general
    matrix by backward transformation on the computed eigenvectors of the
    balanced matrix output by ZGEBAL.

    Arguments
    =========

    JOB     (input) CHARACTER*1
            Specifies the type of backward transformation required:
            = 'N', do nothing, return immediately;
            = 'P', do backward transformation for permutation only;
            = 'S', do backward transformation for scaling only;
            = 'B', do backward transformations for both permutation and
                   scaling.
            JOB must be the same as the argument JOB supplied to ZGEBAL.

    SIDE    (input) CHARACTER*1
            = 'R':  V contains right eigenvectors;
            = 'L':  V contains left eigenvectors.

    N       (input) INTEGER
            The number of rows of the matrix V.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            The integers ILO and IHI determined by ZGEBAL.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    SCALE   (input) DOUBLE PRECISION array, dimension (N)
            Details of the permutation and scaling factors, as returned
            by ZGEBAL.

    M       (input) INTEGER
            The number of columns of the matrix V.  M >= 0.

    V       (input/output) COMPLEX*16 array, dimension (LDV,M)
            On entry, the matrix of right or left eigenvectors to be
            transformed, as returned by ZHSEIN or ZTREVC.
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
    } else if ((*ilo < 1) || (*ilo > max(1,*n))) {
	*info = -4;
    } else if ((*ihi < min(*ilo,*n)) || (*ihi > *n)) {
	*info = -5;
    } else if (*m < 0) {
	*info = -7;
    } else if (*ldv < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEBAK", &i__1);
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

    if ((lsame_(job, "S")) || (lsame_(job, "B"))) {

	if (rightv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = scale[i__];
		zdscal_(m, &s, &v[i__ + v_dim1], ldv);
/* L10: */
	    }
	}

	if (leftv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = 1. / scale[i__];
		zdscal_(m, &s, &v[i__ + v_dim1], ldv);
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
    if ((lsame_(job, "P")) || (lsame_(job, "B"))) {
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
		k = (integer) scale[i__];
		if (k == i__) {
		    goto L40;
		}
		zswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
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
		k = (integer) scale[i__];
		if (k == i__) {
		    goto L50;
		}
		zswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L50:
		;
	    }
	}
    }

    return 0;

/*     End of ZGEBAK */

} /* zgebak_ */

/* Subroutine */ int zgebal_(char *job, integer *n, doublecomplex *a, integer
	*lda, integer *ilo, integer *ihi, doublereal *scale, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1, d__2;

    /* Builtin functions */
    double d_imag(doublecomplex *), z_abs(doublecomplex *);

    /* Local variables */
    static doublereal c__, f, g;
    static integer i__, j, k, l, m;
    static doublereal r__, s, ca, ra;
    static integer ica, ira, iexc;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zswap_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static doublereal sfmin1, sfmin2, sfmax1, sfmax2;

    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    static logical noconv;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZGEBAL balances a general complex matrix A.  This involves, first,
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    SCALE   (output) DOUBLE PRECISION array, dimension (N)
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
	xerbla_("ZGEBAL", &i__1);
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
	    scale[i__] = 1.;
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
    scale[m] = (doublereal) j;
    if (j == m) {
	goto L30;
    }

    zswap_(&l, &a[j * a_dim1 + 1], &c__1, &a[m * a_dim1 + 1], &c__1);
    i__1 = *n - k + 1;
    zswap_(&i__1, &a[j + k * a_dim1], lda, &a[m + k * a_dim1], lda);

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
	    if ((a[i__2].r != 0.) || (d_imag(&a[j + i__ * a_dim1]) != 0.)) {
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
	    if ((a[i__3].r != 0.) || (d_imag(&a[i__ + j * a_dim1]) != 0.)) {
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
	scale[i__] = 1.;
/* L130: */
    }

    if (lsame_(job, "P")) {
	goto L210;
    }

/*
       Balance the submatrix in rows K to L.

       Iterative loop for norm reduction
*/

    sfmin1 = SAFEMINIMUM / PRECISION;
    sfmax1 = 1. / sfmin1;
    sfmin2 = sfmin1 * 8.;
    sfmax2 = 1. / sfmin2;
L140:
    noconv = FALSE_;

    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	c__ = 0.;
	r__ = 0.;

	i__2 = l;
	for (j = k; j <= i__2; ++j) {
	    if (j == i__) {
		goto L150;
	    }
	    i__3 = j + i__ * a_dim1;
	    c__ += (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[j + i__ *
		     a_dim1]), abs(d__2));
	    i__3 = i__ + j * a_dim1;
	    r__ += (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[i__ + j *
		     a_dim1]), abs(d__2));
L150:
	    ;
	}
	ica = izamax_(&l, &a[i__ * a_dim1 + 1], &c__1);
	ca = z_abs(&a[ica + i__ * a_dim1]);
	i__2 = *n - k + 1;
	ira = izamax_(&i__2, &a[i__ + k * a_dim1], lda);
	ra = z_abs(&a[i__ + (ira + k - 1) * a_dim1]);

/*        Guard against zero C or R due to underflow. */

	if ((c__ == 0.) || (r__ == 0.)) {
	    goto L200;
	}
	g = r__ / 8.;
	f = 1.;
	s = c__ + r__;
L160:
/* Computing MAX */
	d__1 = max(f,c__);
/* Computing MIN */
	d__2 = min(r__,g);
	if (((c__ >= g) || (max(d__1,ca) >= sfmax2)) || (min(d__2,ra) <=
		sfmin2)) {
	    goto L170;
	}
	f *= 8.;
	c__ *= 8.;
	ca *= 8.;
	r__ /= 8.;
	g /= 8.;
	ra /= 8.;
	goto L160;

L170:
	g = c__ / 8.;
L180:
/* Computing MIN */
	d__1 = min(f,c__), d__1 = min(d__1,g);
	if (((g < r__) || (max(r__,ra) >= sfmax2)) || (min(d__1,ca) <= sfmin2)
		) {
	    goto L190;
	}
	f /= 8.;
	c__ /= 8.;
	g /= 8.;
	ca /= 8.;
	r__ *= 8.;
	ra *= 8.;
	goto L180;

/*        Now balance. */

L190:
	if (c__ + r__ >= s * .95) {
	    goto L200;
	}
	if (f < 1. && scale[i__] < 1.) {
	    if (f * scale[i__] <= sfmin1) {
		goto L200;
	    }
	}
	if (f > 1. && scale[i__] > 1.) {
	    if (scale[i__] >= sfmax1 / f) {
		goto L200;
	    }
	}
	g = 1. / f;
	scale[i__] *= f;
	noconv = TRUE_;

	i__2 = *n - k + 1;
	zdscal_(&i__2, &g, &a[i__ + k * a_dim1], lda);
	zdscal_(&l, &f, &a[i__ * a_dim1 + 1], &c__1);

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

/*     End of ZGEBAL */

} /* zgebal_ */

/* Subroutine */ int zgebd2_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tauq,
	doublecomplex *taup, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__;
    static doublecomplex alpha;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), zlacgv_(integer *, doublecomplex *,
	    integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGEBD2 reduces a complex general m by n matrix A to upper or lower
    real bidiagonal form B by a unitary transformation: Q' * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    D       (output) DOUBLE PRECISION array, dimension (min(M,N))
            The diagonal elements of the bidiagonal matrix B:
            D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (min(M,N)-1)
            The off-diagonal elements of the bidiagonal matrix B:
            if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
            if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.

    TAUQ    (output) COMPLEX*16 array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    WORK    (workspace) COMPLEX*16 array, dimension (max(M,N))

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
	xerbla_("ZGEBD2", &i__1);
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
	    zlarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1, &
		    tauq[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;

/*           Apply H(i)' to A(i:m,i+1:n) from the left */

	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    d_cnjg(&z__1, &tauq[i__]);
	    zlarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &z__1,
		     &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = d__[i__3], a[i__2].i = 0.;

	    if (i__ < *n) {

/*
                Generate elementary reflector G(i) to annihilate
                A(i,i+2:n)
*/

		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + (i__ + 1) * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
			taup[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + (i__ + 1) * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Apply G(i) to A(i+1:m,i+1:n) from the right */

		i__2 = *m - i__;
		i__3 = *n - i__;
		zlarf_("Right", &i__2, &i__3, &a[i__ + (i__ + 1) * a_dim1],
			lda, &taup[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + (i__ + 1) * a_dim1;
		i__3 = i__;
		a[i__2].r = e[i__3], a[i__2].i = 0.;
	    } else {
		i__2 = i__;
		taup[i__2].r = 0., taup[i__2].i = 0.;
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Generate elementary reflector G(i) to annihilate A(i,i+1:n) */

	    i__2 = *n - i__ + 1;
	    zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    zlarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
		    taup[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;

/*           Apply G(i) to A(i+1:m,i:n) from the right */

	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
/* Computing MIN */
	    i__4 = i__ + 1;
	    zlarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &taup[
		    i__], &a[min(i__4,*m) + i__ * a_dim1], lda, &work[1]);
	    i__2 = *n - i__ + 1;
	    zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = d__[i__3], a[i__2].i = 0.;

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
		zlarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1,
			 &tauq[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Apply H(i)' to A(i+1:m,i+1:n) from the left */

		i__2 = *m - i__;
		i__3 = *n - i__;
		d_cnjg(&z__1, &tauq[i__]);
		zlarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &
			c__1, &z__1, &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &
			work[1]);
		i__2 = i__ + 1 + i__ * a_dim1;
		i__3 = i__;
		a[i__2].r = e[i__3], a[i__2].i = 0.;
	    } else {
		i__2 = i__;
		tauq[i__2].r = 0., tauq[i__2].i = 0.;
	    }
/* L20: */
	}
    }
    return 0;

/*     End of ZGEBD2 */

} /* zgebd2_ */

/* Subroutine */ int zgebrd_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tauq,
	doublecomplex *taup, doublecomplex *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, nb, nx;
    static doublereal ws;
    static integer nbmin, iinfo, minmn;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), zgebd2_(integer *, integer *,
	    doublecomplex *, integer *, doublereal *, doublereal *,
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *),
	    xerbla_(char *, integer *), zlabrd_(integer *, integer *,
	    integer *, doublecomplex *, integer *, doublereal *, doublereal *,
	     doublecomplex *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
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

    ZGEBRD reduces a general complex M-by-N matrix A to upper or lower
    bidiagonal form B by a unitary transformation: Q**H * A * P = B.

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns in the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    D       (output) DOUBLE PRECISION array, dimension (min(M,N))
            The diagonal elements of the bidiagonal matrix B:
            D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (min(M,N)-1)
            The off-diagonal elements of the bidiagonal matrix B:
            if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
            if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.

    TAUQ    (output) COMPLEX*16 array dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    i__1 = 1, i__2 = ilaenv_(&c__1, "ZGEBRD", " ", m, n, &c_n1, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = max(i__1,i__2);
    lwkopt = (*m + *n) * nb;
    d__1 = (doublereal) lwkopt;
    work[1].r = d__1, work[1].i = 0.;
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
	xerbla_("ZGEBRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    minmn = min(*m,*n);
    if (minmn == 0) {
	work[1].r = 1., work[1].i = 0.;
	return 0;
    }

    ws = (doublereal) max(*m,*n);
    ldwrkx = *m;
    ldwrky = *n;

    if (nb > 1 && nb < minmn) {

/*
          Set the crossover point NX.

   Computing MAX
*/
	i__1 = nb, i__2 = ilaenv_(&c__3, "ZGEBRD", " ", m, n, &c_n1, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = max(i__1,i__2);

/*        Determine when to switch from blocked to unblocked code. */

	if (nx < minmn) {
	    ws = (doublereal) ((*m + *n) * nb);
	    if ((doublereal) (*lwork) < ws) {

/*
                Not enough work space for the optimal NB, consider using
                a smaller block size.
*/

		nbmin = ilaenv_(&c__2, "ZGEBRD", " ", m, n, &c_n1, &c_n1, (
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
	zlabrd_(&i__3, &i__4, &nb, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[
		i__], &tauq[i__], &taup[i__], &work[1], &ldwrkx, &work[ldwrkx
		* nb + 1], &ldwrky);

/*
          Update the trailing submatrix A(i+ib:m,i+ib:n), using
          an update of the form  A := A - V*Y' - X*U'
*/

	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	z__1.r = -1., z__1.i = -0.;
	zgemm_("No transpose", "Conjugate transpose", &i__3, &i__4, &nb, &
		z__1, &a[i__ + nb + i__ * a_dim1], lda, &work[ldwrkx * nb +
		nb + 1], &ldwrky, &c_b60, &a[i__ + nb + (i__ + nb) * a_dim1],
		lda);
	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	z__1.r = -1., z__1.i = -0.;
	zgemm_("No transpose", "No transpose", &i__3, &i__4, &nb, &z__1, &
		work[nb + 1], &ldwrkx, &a[i__ + (i__ + nb) * a_dim1], lda, &
		c_b60, &a[i__ + nb + (i__ + nb) * a_dim1], lda);

/*        Copy diagonal and off-diagonal elements of B back into A */

	if (*m >= *n) {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + j * a_dim1;
		i__5 = j;
		a[i__4].r = d__[i__5], a[i__4].i = 0.;
		i__4 = j + (j + 1) * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.;
/* L10: */
	    }
	} else {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + j * a_dim1;
		i__5 = j;
		a[i__4].r = d__[i__5], a[i__4].i = 0.;
		i__4 = j + 1 + j * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.;
/* L20: */
	    }
	}
/* L30: */
    }

/*     Use unblocked code to reduce the remainder of the matrix */

    i__2 = *m - i__ + 1;
    i__1 = *n - i__ + 1;
    zgebd2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__], &
	    tauq[i__], &taup[i__], &work[1], &iinfo);
    work[1].r = ws, work[1].i = 0.;
    return 0;

/*     End of ZGEBRD */

} /* zgebrd_ */

/* Subroutine */ int zgeev_(char *jobvl, char *jobvr, integer *n,
	doublecomplex *a, integer *lda, doublecomplex *w, doublecomplex *vl,
	integer *ldvl, doublecomplex *vr, integer *ldvr, doublecomplex *work,
	integer *lwork, doublereal *rwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3, i__4;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double sqrt(doublereal), d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, k, ihi;
    static doublereal scl;
    static integer ilo;
    static doublereal dum[1], eps;
    static doublecomplex tmp;
    static integer ibal;
    static char side[1];
    static integer maxb;
    static doublereal anrm;
    static integer ierr, itau, iwrk, nout;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), dlabad_(doublereal *, doublereal *);
    extern doublereal dznrm2_(integer *, doublecomplex *, integer *);
    static logical scalea;

    static doublereal cscale;
    extern /* Subroutine */ int zgebak_(char *, char *, integer *, integer *,
	    integer *, doublereal *, integer *, doublecomplex *, integer *,
	    integer *), zgebal_(char *, integer *,
	    doublecomplex *, integer *, integer *, integer *, doublereal *,
	    integer *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static logical select[1];
    extern /* Subroutine */ int zdscal_(integer *, doublereal *,
	    doublecomplex *, integer *);
    static doublereal bignum;
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *,
	    integer *, doublereal *);
    extern /* Subroutine */ int zgehrd_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *), zlascl_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, integer *, doublecomplex *,
	     integer *, integer *), zlacpy_(char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *);
    static integer minwrk, maxwrk;
    static logical wantvl;
    static doublereal smlnum;
    static integer hswork, irwork;
    extern /* Subroutine */ int zhseqr_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *), ztrevc_(char *, char *, logical *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, integer *, integer *, doublecomplex *,
	     doublereal *, integer *);
    static logical lquery, wantvr;
    extern /* Subroutine */ int zunghr_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZGEEV computes for an N-by-N complex nonsymmetric matrix A, the
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the N-by-N matrix A.
            On exit, A has been overwritten.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    W       (output) COMPLEX*16 array, dimension (N)
            W contains the computed eigenvalues.

    VL      (output) COMPLEX*16 array, dimension (LDVL,N)
            If JOBVL = 'V', the left eigenvectors u(j) are stored one
            after another in the columns of VL, in the same order
            as their eigenvalues.
            If JOBVL = 'N', VL is not referenced.
            u(j) = VL(:,j), the j-th column of VL.

    LDVL    (input) INTEGER
            The leading dimension of the array VL.  LDVL >= 1; if
            JOBVL = 'V', LDVL >= N.

    VR      (output) COMPLEX*16 array, dimension (LDVR,N)
            If JOBVR = 'V', the right eigenvectors v(j) are stored one
            after another in the columns of VR, in the same order
            as their eigenvalues.
            If JOBVR = 'N', VR is not referenced.
            v(j) = VR(:,j), the j-th column of VR.

    LDVR    (input) INTEGER
            The leading dimension of the array VR.  LDVR >= 1; if
            JOBVR = 'V', LDVR >= N.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,2*N).
            For good performance, LWORK must generally be larger.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace) DOUBLE PRECISION array, dimension (2*N)

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
    } else if ((*ldvl < 1) || (wantvl && *ldvl < *n)) {
	*info = -8;
    } else if ((*ldvr < 1) || (wantvr && *ldvr < *n)) {
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
         HSWORK refers to the workspace preferred by ZHSEQR, as
         calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
         the worst case.)
*/

    minwrk = 1;
    if (*info == 0 && ((*lwork >= 1) || (lquery))) {
	maxwrk = *n + *n * ilaenv_(&c__1, "ZGEHRD", " ", n, &c__1, n, &c__0, (
		ftnlen)6, (ftnlen)1);
	if (! wantvl && ! wantvr) {
/* Computing MAX */
	    i__1 = 1, i__2 = (*n) << (1);
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "ZHSEQR", "EN", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "ZHSEQR", "EN", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = (*n) << (1);
	    hswork = max(i__1,i__2);
	    maxwrk = max(maxwrk,hswork);
	} else {
/* Computing MAX */
	    i__1 = 1, i__2 = (*n) << (1);
	    minwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + (*n - 1) * ilaenv_(&c__1, "ZUNGHR",
		    " ", n, &c__1, n, &c_n1, (ftnlen)6, (ftnlen)1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ilaenv_(&c__8, "ZHSEQR", "SV", n, &c__1, n, &c_n1, (ftnlen)
		    6, (ftnlen)2);
	    maxb = max(i__1,2);
/*
   Computing MIN
   Computing MAX
*/
	    i__3 = 2, i__4 = ilaenv_(&c__4, "ZHSEQR", "SV", n, &c__1, n, &
		    c_n1, (ftnlen)6, (ftnlen)2);
	    i__1 = min(maxb,*n), i__2 = max(i__3,i__4);
	    k = min(i__1,i__2);
/* Computing MAX */
	    i__1 = k * (k + 2), i__2 = (*n) << (1);
	    hswork = max(i__1,i__2);
/* Computing MAX */
	    i__1 = max(maxwrk,hswork), i__2 = (*n) << (1);
	    maxwrk = max(i__1,i__2);
	}
	work[1].r = (doublereal) maxwrk, work[1].i = 0.;
    }
    if (*lwork < minwrk && ! lquery) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEEV ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Get machine constants */

    eps = PRECISION;
    smlnum = SAFEMINIMUM;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = 1. / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = zlange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	zlascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }

/*
       Balance the matrix
       (CWorkspace: none)
       (RWorkspace: need N)
*/

    ibal = 1;
    zgebal_("B", n, &a[a_offset], lda, &ilo, &ihi, &rwork[ibal], &ierr);

/*
       Reduce to upper Hessenberg form
       (CWorkspace: need 2*N, prefer N+N*NB)
       (RWorkspace: none)
*/

    itau = 1;
    iwrk = itau + *n;
    i__1 = *lwork - iwrk + 1;
    zgehrd_(n, &ilo, &ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1,
	     &ierr);

    if (wantvl) {

/*
          Want left eigenvectors
          Copy Householder vectors to VL
*/

	*(unsigned char *)side = 'L';
	zlacpy_("L", n, n, &a[a_offset], lda, &vl[vl_offset], ldvl)
		;

/*
          Generate unitary matrix in VL
          (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
          (RWorkspace: none)
*/

	i__1 = *lwork - iwrk + 1;
	zunghr_(n, &ilo, &ihi, &vl[vl_offset], ldvl, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VL
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	zhseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vl[
		vl_offset], ldvl, &work[iwrk], &i__1, info);

	if (wantvr) {

/*
             Want left and right eigenvectors
             Copy Schur vectors to VR
*/

	    *(unsigned char *)side = 'B';
	    zlacpy_("F", n, n, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr);
	}

    } else if (wantvr) {

/*
          Want right eigenvectors
          Copy Householder vectors to VR
*/

	*(unsigned char *)side = 'R';
	zlacpy_("L", n, n, &a[a_offset], lda, &vr[vr_offset], ldvr)
		;

/*
          Generate unitary matrix in VR
          (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
          (RWorkspace: none)
*/

	i__1 = *lwork - iwrk + 1;
	zunghr_(n, &ilo, &ihi, &vr[vr_offset], ldvr, &work[itau], &work[iwrk],
		 &i__1, &ierr);

/*
          Perform QR iteration, accumulating Schur vectors in VR
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	zhseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);

    } else {

/*
          Compute eigenvalues only
          (CWorkspace: need 1, prefer HSWORK (see comments) )
          (RWorkspace: none)
*/

	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	zhseqr_("E", "N", n, &ilo, &ihi, &a[a_offset], lda, &w[1], &vr[
		vr_offset], ldvr, &work[iwrk], &i__1, info);
    }

/*     If INFO > 0 from ZHSEQR, then quit */

    if (*info > 0) {
	goto L50;
    }

    if ((wantvl) || (wantvr)) {

/*
          Compute left and/or right eigenvectors
          (CWorkspace: need 2*N)
          (RWorkspace: need 2*N)
*/

	irwork = ibal + *n;
	ztrevc_(side, "B", select, n, &a[a_offset], lda, &vl[vl_offset], ldvl,
		 &vr[vr_offset], ldvr, n, &nout, &work[iwrk], &rwork[irwork],
		&ierr);
    }

    if (wantvl) {

/*
          Undo balancing of left eigenvectors
          (CWorkspace: none)
          (RWorkspace: need N)
*/

	zgebak_("B", "L", n, &ilo, &ihi, &rwork[ibal], n, &vl[vl_offset],
		ldvl, &ierr);

/*        Normalize left eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scl = 1. / dznrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
	    zdscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
	    i__2 = *n;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = k + i__ * vl_dim1;
/* Computing 2nd power */
		d__1 = vl[i__3].r;
/* Computing 2nd power */
		d__2 = d_imag(&vl[k + i__ * vl_dim1]);
		rwork[irwork + k - 1] = d__1 * d__1 + d__2 * d__2;
/* L10: */
	    }
	    k = idamax_(n, &rwork[irwork], &c__1);
	    d_cnjg(&z__2, &vl[k + i__ * vl_dim1]);
	    d__1 = sqrt(rwork[irwork + k - 1]);
	    z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	    tmp.r = z__1.r, tmp.i = z__1.i;
	    zscal_(n, &tmp, &vl[i__ * vl_dim1 + 1], &c__1);
	    i__2 = k + i__ * vl_dim1;
	    i__3 = k + i__ * vl_dim1;
	    d__1 = vl[i__3].r;
	    z__1.r = d__1, z__1.i = 0.;
	    vl[i__2].r = z__1.r, vl[i__2].i = z__1.i;
/* L20: */
	}
    }

    if (wantvr) {

/*
          Undo balancing of right eigenvectors
          (CWorkspace: none)
          (RWorkspace: need N)
*/

	zgebak_("B", "R", n, &ilo, &ihi, &rwork[ibal], n, &vr[vr_offset],
		ldvr, &ierr);

/*        Normalize right eigenvectors and make largest component real */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scl = 1. / dznrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
	    zdscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
	    i__2 = *n;
	    for (k = 1; k <= i__2; ++k) {
		i__3 = k + i__ * vr_dim1;
/* Computing 2nd power */
		d__1 = vr[i__3].r;
/* Computing 2nd power */
		d__2 = d_imag(&vr[k + i__ * vr_dim1]);
		rwork[irwork + k - 1] = d__1 * d__1 + d__2 * d__2;
/* L30: */
	    }
	    k = idamax_(n, &rwork[irwork], &c__1);
	    d_cnjg(&z__2, &vr[k + i__ * vr_dim1]);
	    d__1 = sqrt(rwork[irwork + k - 1]);
	    z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	    tmp.r = z__1.r, tmp.i = z__1.i;
	    zscal_(n, &tmp, &vr[i__ * vr_dim1 + 1], &c__1);
	    i__2 = k + i__ * vr_dim1;
	    i__3 = k + i__ * vr_dim1;
	    d__1 = vr[i__3].r;
	    z__1.r = d__1, z__1.i = 0.;
	    vr[i__2].r = z__1.r, vr[i__2].i = z__1.i;
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
	zlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &w[*info + 1]
		, &i__2, &ierr);
	if (*info > 0) {
	    i__1 = ilo - 1;
	    zlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &w[1], n,
		     &ierr);
	}
    }

    work[1].r = (doublereal) maxwrk, work[1].i = 0.;
    return 0;

/*     End of ZGEEV */

} /* zgeev_ */

/* Subroutine */ int zgehd2_(integer *n, integer *ilo, integer *ihi,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__;
    static doublecomplex alpha;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGEHD2 reduces a complex general matrix A to upper Hessenberg form H
    by a unitary similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to ZGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= max(1,N).

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the n by n general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the unitary matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX*16 array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX*16 array, dimension (N)

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
    } else if ((*ilo < 1) || (*ilo > max(1,*n))) {
	*info = -2;
    } else if ((*ihi < min(*ilo,*n)) || (*ihi > *n)) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEHD2", &i__1);
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
	zlarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[
		i__]);
	i__2 = i__ + 1 + i__ * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;

/*        Apply H(i) to A(1:ihi,i+1:ihi) from the right */

	i__2 = *ihi - i__;
	zlarf_("Right", ihi, &i__2, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[(i__ + 1) * a_dim1 + 1], lda, &work[1]);

/*        Apply H(i)' to A(i+1:ihi,i+1:n) from the left */

	i__2 = *ihi - i__;
	i__3 = *n - i__;
	d_cnjg(&z__1, &tau[i__]);
	zlarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &z__1,
		 &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &work[1]);

	i__2 = i__ + 1 + i__ * a_dim1;
	a[i__2].r = alpha.r, a[i__2].i = alpha.i;
/* L10: */
    }

    return 0;

/*     End of ZGEHD2 */

} /* zgehd2_ */

/* Subroutine */ int zgehrd_(integer *n, integer *ilo, integer *ihi,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;

    /* Local variables */
    static integer i__;
    static doublecomplex t[4160]	/* was [65][64] */;
    static integer ib;
    static doublecomplex ei;
    static integer nb, nh, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), zgehd2_(integer *, integer *, integer
	    *, doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *),
	    zlahrd_(integer *, integer *, integer *, doublecomplex *, integer
	    *, doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZGEHRD reduces a complex general matrix A to upper Hessenberg form H
    by a unitary similarity transformation:  Q' * A * Q = H .

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to ZGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the N-by-N general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the unitary matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX*16 array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
            zero.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    i__1 = 64, i__2 = ilaenv_(&c__1, "ZGEHRD", " ", n, ilo, ihi, &c_n1, (
	    ftnlen)6, (ftnlen)1);
    nb = min(i__1,i__2);
    lwkopt = *n * nb;
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if ((*ilo < 1) || (*ilo > max(1,*n))) {
	*info = -2;
    } else if ((*ihi < min(*ilo,*n)) || (*ihi > *n)) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEHRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Set elements 1:ILO-1 and IHI:N-1 of TAU to zero */

    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	tau[i__2].r = 0., tau[i__2].i = 0.;
/* L10: */
    }
    i__1 = *n - 1;
    for (i__ = max(1,*ihi); i__ <= i__1; ++i__) {
	i__2 = i__;
	tau[i__2].r = 0., tau[i__2].i = 0.;
/* L20: */
    }

/*     Quick return if possible */

    nh = *ihi - *ilo + 1;
    if (nh <= 1) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = nb, i__2 = ilaenv_(&c__3, "ZGEHRD", " ", n, ilo, ihi, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZGEHRD", " ", n, ilo, ihi, &
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

    if ((nb < nbmin) || (nb >= nh)) {

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

	    zlahrd_(ihi, &i__, &ib, &a[i__ * a_dim1 + 1], lda, &tau[i__], t, &
		    c__65, &work[1], &ldwork);

/*
             Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
             right, computing  A := A - Y * V'. V(i+ib,ib-1) must be set
             to 1.
*/

	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    ei.r = a[i__3].r, ei.i = a[i__3].i;
	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    a[i__3].r = 1., a[i__3].i = 0.;
	    i__3 = *ihi - i__ - ib + 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemm_("No transpose", "Conjugate transpose", ihi, &i__3, &ib, &
		    z__1, &work[1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda,
		     &c_b60, &a[(i__ + ib) * a_dim1 + 1], lda);
	    i__3 = i__ + ib + (i__ + ib - 1) * a_dim1;
	    a[i__3].r = ei.r, a[i__3].i = ei.i;

/*
             Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
             left
*/

	    i__3 = *ihi - i__;
	    i__4 = *n - i__ - ib + 1;
	    zlarfb_("Left", "Conjugate transpose", "Forward", "Columnwise", &
		    i__3, &i__4, &ib, &a[i__ + 1 + i__ * a_dim1], lda, t, &
		    c__65, &a[i__ + 1 + (i__ + ib) * a_dim1], lda, &work[1], &
		    ldwork);
/* L30: */
	}
    }

/*     Use unblocked code to reduce the rest of the matrix */

    zgehd2_(n, &i__, ihi, &a[a_offset], lda, &tau[1], &work[1], &iinfo);
    work[1].r = (doublereal) iws, work[1].i = 0.;

    return 0;

/*     End of ZGEHRD */

} /* zgehrd_ */

/* Subroutine */ int zgelq2_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    static integer i__, k;
    static doublecomplex alpha;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), zlacgv_(integer *, doublecomplex *,
	    integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGELQ2 computes an LQ factorization of a complex m by n matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m by min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX*16 array, dimension (M)

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
	xerbla_("ZGELQ2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i,i+1:n) */

	i__2 = *n - i__ + 1;
	zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	i__2 = i__ + i__ * a_dim1;
	alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	i__2 = *n - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	zlarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &tau[i__]
		);
	if (i__ < *m) {

/*           Apply H(i) to A(i+1:m,i:n) from the right */

	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    zlarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	}
	i__2 = i__ + i__ * a_dim1;
	a[i__2].r = alpha.r, a[i__2].i = alpha.i;
	i__2 = *n - i__ + 1;
	zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
/* L10: */
    }
    return 0;

/*     End of ZGELQ2 */

} /* zgelq2_ */

/* Subroutine */ int zgelqf_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int zgelq2_(integer *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, integer *), xerbla_(
	    char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZGELQF computes an LQ factorization of a complex M-by-N matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    nb = ilaenv_(&c__1, "ZGELQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *m * nb;
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
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
	xerbla_("ZGELQF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "ZGELQF", " ", m, n, &c_n1, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZGELQF", " ", m, n, &c_n1, &
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
	    zgelq2_(&ib, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *m) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *n - i__ + 1;
		zlarft_("Forward", "Rowwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i+ib:m,i:n) from the right */

		i__3 = *m - i__ - ib + 1;
		i__4 = *n - i__ + 1;
		zlarfb_("Right", "No transpose", "Forward", "Rowwise", &i__3,
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
	zgelq2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1].r = (doublereal) iws, work[1].i = 0.;
    return 0;

/*     End of ZGELQF */

} /* zgelqf_ */

/* Subroutine */ int zgelsd_(integer *m, integer *n, integer *nrhs,
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb,
	doublereal *s, doublereal *rcond, integer *rank, doublecomplex *work,
	integer *lwork, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;
    doublereal d__1;
    doublecomplex z__1;

    /* Local variables */
    static integer ie, il, mm;
    static doublereal eps, anrm, bnrm;
    static integer itau, iascl, ibscl;
    static doublereal sfmin;
    static integer minmn, maxmn, itaup, itauq, mnthr, nwork;
    extern /* Subroutine */ int dlabad_(doublereal *, doublereal *);

    extern /* Subroutine */ int dlascl_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, integer *, doublereal *,
	    integer *, integer *), dlaset_(char *, integer *, integer
	    *, doublereal *, doublereal *, doublereal *, integer *),
	    xerbla_(char *, integer *), zgebrd_(integer *, integer *,
	    doublecomplex *, integer *, doublereal *, doublereal *,
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *,
	    integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *,
	    integer *, doublereal *);
    static doublereal bignum;
    extern /* Subroutine */ int zgelqf_(integer *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, integer *, integer *
	    ), zlalsd_(char *, integer *, integer *, integer *, doublereal *,
	    doublereal *, doublecomplex *, integer *, doublereal *, integer *,
	     doublecomplex *, doublereal *, integer *, integer *),
	    zlascl_(char *, integer *, integer *, doublereal *, doublereal *,
	    integer *, integer *, doublecomplex *, integer *, integer *), zgeqrf_(integer *, integer *, doublecomplex *, integer *,
	     doublecomplex *, doublecomplex *, integer *, integer *);
    static integer ldwork;
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *),
	    zlaset_(char *, integer *, integer *, doublecomplex *,
	    doublecomplex *, doublecomplex *, integer *);
    static integer minwrk, maxwrk;
    static doublereal smlnum;
    extern /* Subroutine */ int zunmbr_(char *, char *, char *, integer *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     doublecomplex *, integer *, doublecomplex *, integer *, integer *
	    );
    static logical lquery;
    static integer nrwork, smlsiz;
    extern /* Subroutine */ int zunmlq_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *), zunmqr_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    ZGELSD computes the minimum-norm solution to a real linear least
    squares problem:
        minimize 2-norm(| b - A*x |)
    using the singular value decomposition (SVD) of A. A is an M-by-N
    matrix which may be rank-deficient.

    Several right hand side vectors b and solution vectors x can be
    handled in a single call; they are stored as the columns of the
    M-by-NRHS right hand side matrix B and the N-by-NRHS solution
    matrix X.

    The problem is solved in three steps:
    (1) Reduce the coefficient matrix A to bidiagonal form with
        Householder tranformations, reducing the original problem
        into a "bidiagonal least squares problem" (BLS)
    (2) Solve the BLS using a divide and conquer approach.
    (3) Apply back all the Householder tranformations to solve
        the original least squares problem.

    The effective rank of A is determined by treating as zero those
    singular values which are less than RCOND times the largest singular
    value.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrices B and X. NRHS >= 0.

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, A has been destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
            On entry, the M-by-NRHS right hand side matrix B.
            On exit, B is overwritten by the N-by-NRHS solution matrix X.
            If m >= n and RANK = n, the residual sum-of-squares for
            the solution in the i-th column is given by the sum of
            squares of elements n+1:m in that column.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,M,N).

    S       (output) DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A in decreasing order.
            The condition number of A in the 2-norm = S(1)/S(min(m,n)).

    RCOND   (input) DOUBLE PRECISION
            RCOND is used to determine the effective rank of A.
            Singular values S(i) <= RCOND*S(1) are treated as zero.
            If RCOND < 0, machine precision is used instead.

    RANK    (output) INTEGER
            The effective rank of A, i.e., the number of singular values
            which are greater than RCOND*S(1).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK. LWORK must be at least 1.
            The exact minimum amount of workspace needed depends on M,
            N and NRHS. As long as LWORK is at least
                2 * N + N * NRHS
            if M is greater than or equal to N or
                2 * M + M * NRHS
            if M is less than N, the code will execute correctly.
            For good performance, LWORK should generally be larger.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace) DOUBLE PRECISION array, dimension at least
               10*N + 2*N*SMLSIZ + 8*N*NLVL + 3*SMLSIZ*NRHS +
               (SMLSIZ+1)**2
            if M is greater than or equal to N or
               10*M + 2*M*SMLSIZ + 8*M*NLVL + 3*SMLSIZ*NRHS +
               (SMLSIZ+1)**2
            if M is less than N, the code will execute correctly.
            SMLSIZ is returned by ILAENV and is equal to the maximum
            size of the subproblems at the bottom of the computation
            tree (usually about 25), and
               NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 )

    IWORK   (workspace) INTEGER array, dimension (LIWORK)
            LIWORK >= 3 * MINMN * NLVL + 11 * MINMN,
            where MINMN = MIN( M,N ).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -i, the i-th argument had an illegal value.
            > 0:  the algorithm for computing the SVD failed to converge;
                  if INFO = i, i off-diagonal elements of an intermediate
                  bidiagonal form did not converge to zero.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Ren-Cang Li, Computer Science Division, University of
         California at Berkeley, USA
       Osni Marques, LBNL/NERSC, USA

    =====================================================================


       Test the input arguments.
*/

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --s;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    maxmn = max(*m,*n);
    mnthr = ilaenv_(&c__6, "ZGELSD", " ", m, n, nrhs, &c_n1, (ftnlen)6, (
	    ftnlen)1);
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldb < max(1,maxmn)) {
	*info = -7;
    }

    smlsiz = ilaenv_(&c__9, "ZGELSD", " ", &c__0, &c__0, &c__0, &c__0, (
	    ftnlen)6, (ftnlen)1);

/*
       Compute workspace.
       (Note: Comments in the code beginning "Workspace:" describe the
       minimal amount of workspace needed at that point in the code,
       as well as the preferred amount for good performance.
       NB refers to the optimal block size for the immediately
       following subroutine, as returned by ILAENV.)
*/

    minwrk = 1;
    if (*info == 0) {
	maxwrk = 0;
	mm = *m;
	if (*m >= *n && *m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns. */

	    mm = *n;
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
		    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *nrhs * ilaenv_(&c__1, "ZUNMQR", "LC", m,
		    nrhs, n, &c_n1, (ftnlen)6, (ftnlen)2);
	    maxwrk = max(i__1,i__2);
	}
	if (*m >= *n) {

/*
             Path 1 - overdetermined or exactly determined.

   Computing MAX
*/
	    i__1 = maxwrk, i__2 = ((*n) << (1)) + (mm + *n) * ilaenv_(&c__1,
		    "ZGEBRD", " ", &mm, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1)
		    ;
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = ((*n) << (1)) + *nrhs * ilaenv_(&c__1,
		    "ZUNMBR", "QLC", &mm, nrhs, n, &c_n1, (ftnlen)6, (ftnlen)
		    3);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = ((*n) << (1)) + (*n - 1) * ilaenv_(&c__1,
		    "ZUNMBR", "PLN", n, nrhs, n, &c_n1, (ftnlen)6, (ftnlen)3);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * *nrhs;
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = ((*n) << (1)) + mm, i__2 = ((*n) << (1)) + *n * *nrhs;
	    minwrk = max(i__1,i__2);
	}
	if (*n > *m) {
	    if (*n >= mnthr) {

/*
                Path 2a - underdetermined, with many more columns
                than rows.
*/

		maxwrk = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &c_n1,
			&c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + ((*m) << (2)) + ((*m) << (1))
			* ilaenv_(&c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1, (
			ftnlen)6, (ftnlen)1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + ((*m) << (2)) + *nrhs *
			ilaenv_(&c__1, "ZUNMBR", "QLC", m, nrhs, m, &c_n1, (
			ftnlen)6, (ftnlen)3);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + ((*m) << (2)) + (*m - 1) *
			ilaenv_(&c__1, "ZUNMLQ", "LC", n, nrhs, m, &c_n1, (
			ftnlen)6, (ftnlen)2);
		maxwrk = max(i__1,i__2);
		if (*nrhs > 1) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + *m + *m * *nrhs;
		    maxwrk = max(i__1,i__2);
		} else {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + ((*m) << (1));
		    maxwrk = max(i__1,i__2);
		}
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + ((*m) << (2)) + *m * *nrhs;
		maxwrk = max(i__1,i__2);
	    } else {

/*              Path 2 - underdetermined. */

		maxwrk = ((*m) << (1)) + (*n + *m) * ilaenv_(&c__1, "ZGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		i__1 = maxwrk, i__2 = ((*m) << (1)) + *nrhs * ilaenv_(&c__1,
			"ZUNMBR", "QLC", m, nrhs, m, &c_n1, (ftnlen)6, (
			ftnlen)3);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			"ZUNMBR", "PLN", n, nrhs, m, &c_n1, (ftnlen)6, (
			ftnlen)3);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * *nrhs;
		maxwrk = max(i__1,i__2);
	    }
/* Computing MAX */
	    i__1 = ((*m) << (1)) + *n, i__2 = ((*m) << (1)) + *m * *nrhs;
	    minwrk = max(i__1,i__2);
	}
	minwrk = min(minwrk,maxwrk);
	d__1 = (doublereal) maxwrk;
	z__1.r = d__1, z__1.i = 0.;
	work[1].r = z__1.r, work[1].i = z__1.i;
	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGELSD", &i__1);
	return 0;
    } else if (lquery) {
	goto L10;
    }

/*     Quick return if possible. */

    if ((*m == 0) || (*n == 0)) {
	*rank = 0;
	return 0;
    }

/*     Get machine parameters. */

    eps = PRECISION;
    sfmin = SAFEMINIMUM;
    smlnum = sfmin / eps;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);

/*     Scale A if max entry outside range [SMLNUM,BIGNUM]. */

    anrm = zlange_("M", m, n, &a[a_offset], lda, &rwork[1]);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	zlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda,
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM. */

	zlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda,
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	i__1 = max(*m,*n);
	zlaset_("F", &i__1, nrhs, &c_b59, &c_b59, &b[b_offset], ldb);
	dlaset_("F", &minmn, &c__1, &c_b324, &c_b324, &s[1], &c__1)
		;
	*rank = 0;
	goto L10;
    }

/*     Scale B if max entry outside range [SMLNUM,BIGNUM]. */

    bnrm = zlange_("M", m, nrhs, &b[b_offset], ldb, &rwork[1]);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM. */

	zlascl_("G", &c__0, &c__0, &bnrm, &smlnum, m, nrhs, &b[b_offset], ldb,
		 info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM. */

	zlascl_("G", &c__0, &c__0, &bnrm, &bignum, m, nrhs, &b[b_offset], ldb,
		 info);
	ibscl = 2;
    }

/*     If M < N make sure B(M+1:N,:) = 0 */

    if (*m < *n) {
	i__1 = *n - *m;
	zlaset_("F", &i__1, nrhs, &c_b59, &c_b59, &b[*m + 1 + b_dim1], ldb);
    }

/*     Overdetermined case. */

    if (*m >= *n) {

/*        Path 1 - overdetermined or exactly determined. */

	mm = *m;
	if (*m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns */

	    mm = *n;
	    itau = 1;
	    nwork = itau + *n;

/*
             Compute A=Q*R.
             (RWorkspace: need N)
             (CWorkspace: need N, prefer N*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &i__1,
		     info);

/*
             Multiply B by transpose(Q).
             (RWorkspace: need N)
             (CWorkspace: need NRHS, prefer NRHS*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zunmqr_("L", "C", m, nrhs, n, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[nwork], &i__1, info);

/*           Zero out below R. */

	    if (*n > 1) {
		i__1 = *n - 1;
		i__2 = *n - 1;
		zlaset_("L", &i__1, &i__2, &c_b59, &c_b59, &a[a_dim1 + 2],
			lda);
	    }
	}

	itauq = 1;
	itaup = itauq + *n;
	nwork = itaup + *n;
	ie = 1;
	nrwork = ie + *n;

/*
          Bidiagonalize R in A.
          (RWorkspace: need N)
          (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB)
*/

	i__1 = *lwork - nwork + 1;
	zgebrd_(&mm, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], &
		work[itaup], &work[nwork], &i__1, info);

/*
          Multiply B by transpose of left bidiagonalizing vectors of R.
          (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB)
*/

	i__1 = *lwork - nwork + 1;
	zunmbr_("Q", "L", "C", &mm, nrhs, n, &a[a_offset], lda, &work[itauq],
		&b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	zlalsd_("U", &smlsiz, n, nrhs, &s[1], &rwork[ie], &b[b_offset], ldb,
		rcond, rank, &work[nwork], &rwork[nrwork], &iwork[1], info);
	if (*info != 0) {
	    goto L10;
	}

/*        Multiply B by right bidiagonalizing vectors of R. */

	i__1 = *lwork - nwork + 1;
	zunmbr_("P", "L", "N", n, nrhs, n, &a[a_offset], lda, &work[itaup], &
		b[b_offset], ldb, &work[nwork], &i__1, info);

    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = *m, i__2 = ((*m) << (1)) - 4, i__1 = max(i__1,i__2), i__1 =
		max(i__1,*nrhs), i__2 = *n - *m * 3;
	if (*n >= mnthr && *lwork >= ((*m) << (2)) + *m * *m + max(i__1,i__2))
		 {

/*
          Path 2a - underdetermined, with many more columns than rows
          and sufficient workspace for an efficient algorithm.
*/

	    ldwork = *m;
/*
   Computing MAX
   Computing MAX
*/
	    i__3 = *m, i__4 = ((*m) << (1)) - 4, i__3 = max(i__3,i__4), i__3 =
		     max(i__3,*nrhs), i__4 = *n - *m * 3;
	    i__1 = ((*m) << (2)) + *m * *lda + max(i__3,i__4), i__2 = *m * *
		    lda + *m + *m * *nrhs;
	    if (*lwork >= max(i__1,i__2)) {
		ldwork = *lda;
	    }
	    itau = 1;
	    nwork = *m + 1;

/*
          Compute A=L*Q.
          (CWorkspace: need 2*M, prefer M+M*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &i__1,
		     info);
	    il = nwork;

/*        Copy L to WORK(IL), zeroing out above its diagonal. */

	    zlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwork);
	    i__1 = *m - 1;
	    i__2 = *m - 1;
	    zlaset_("U", &i__1, &i__2, &c_b59, &c_b59, &work[il + ldwork], &
		    ldwork);
	    itauq = il + ldwork * *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;
	    ie = 1;
	    nrwork = ie + *m;

/*
          Bidiagonalize L in WORK(IL).
          (RWorkspace: need M)
          (CWorkspace: need M*M+4*M, prefer M*M+4*M+2*M*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zgebrd_(m, m, &work[il], &ldwork, &s[1], &rwork[ie], &work[itauq],
		     &work[itaup], &work[nwork], &i__1, info);

/*
          Multiply B by transpose of left bidiagonalizing vectors of L.
          (CWorkspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zunmbr_("Q", "L", "C", m, nrhs, m, &work[il], &ldwork, &work[
		    itauq], &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	    zlalsd_("U", &smlsiz, m, nrhs, &s[1], &rwork[ie], &b[b_offset],
		    ldb, rcond, rank, &work[nwork], &rwork[nrwork], &iwork[1],
		     info);
	    if (*info != 0) {
		goto L10;
	    }

/*        Multiply B by right bidiagonalizing vectors of L. */

	    i__1 = *lwork - nwork + 1;
	    zunmbr_("P", "L", "N", m, nrhs, m, &work[il], &ldwork, &work[
		    itaup], &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Zero out below first M rows of B. */

	    i__1 = *n - *m;
	    zlaset_("F", &i__1, nrhs, &c_b59, &c_b59, &b[*m + 1 + b_dim1],
		    ldb);
	    nwork = itau + *m;

/*
          Multiply transpose(Q) by B.
          (CWorkspace: need NRHS, prefer NRHS*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zunmlq_("L", "C", n, nrhs, m, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[nwork], &i__1, info);

	} else {

/*        Path 2 - remaining underdetermined cases. */

	    itauq = 1;
	    itaup = itauq + *m;
	    nwork = itaup + *m;
	    ie = 1;
	    nrwork = ie + *m;

/*
          Bidiagonalize A.
          (RWorkspace: need M)
          (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__1, info);

/*
          Multiply B by transpose of left bidiagonalizing vectors.
          (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB)
*/

	    i__1 = *lwork - nwork + 1;
	    zunmbr_("Q", "L", "C", m, nrhs, n, &a[a_offset], lda, &work[itauq]
		    , &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	    zlalsd_("L", &smlsiz, m, nrhs, &s[1], &rwork[ie], &b[b_offset],
		    ldb, rcond, rank, &work[nwork], &rwork[nrwork], &iwork[1],
		     info);
	    if (*info != 0) {
		goto L10;
	    }

/*        Multiply B by right bidiagonalizing vectors of A. */

	    i__1 = *lwork - nwork + 1;
	    zunmbr_("P", "L", "N", n, nrhs, m, &a[a_offset], lda, &work[itaup]
		    , &b[b_offset], ldb, &work[nwork], &i__1, info);

	}
    }

/*     Undo scaling. */

    if (iascl == 1) {
	zlascl_("G", &c__0, &c__0, &anrm, &smlnum, n, nrhs, &b[b_offset], ldb,
		 info);
	dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    } else if (iascl == 2) {
	zlascl_("G", &c__0, &c__0, &anrm, &bignum, n, nrhs, &b[b_offset], ldb,
		 info);
	dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    }
    if (ibscl == 1) {
	zlascl_("G", &c__0, &c__0, &smlnum, &bnrm, n, nrhs, &b[b_offset], ldb,
		 info);
    } else if (ibscl == 2) {
	zlascl_("G", &c__0, &c__0, &bignum, &bnrm, n, nrhs, &b[b_offset], ldb,
		 info);
    }

L10:
    d__1 = (doublereal) maxwrk;
    z__1.r = d__1, z__1.i = 0.;
    work[1].r = z__1.r, work[1].i = z__1.i;
    return 0;

/*     End of ZGELSD */

} /* zgelsd_ */

/* Subroutine */ int zgeqr2_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, k;
    static doublecomplex alpha;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace) COMPLEX*16 array, dimension (N)

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
	xerbla_("ZGEQR2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i+1:m,i) */

	i__2 = *m - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	zlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ * a_dim1]
		, &c__1, &tau[i__]);
	if (i__ < *n) {

/*           Apply H(i)' to A(i:m,i+1:n) from the left */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    d_cnjg(&z__1, &tau[i__]);
	    zlarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &z__1,
		     &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = alpha.r, a[i__2].i = alpha.i;
	}
/* L10: */
    }
    return 0;

/*     End of ZGEQR2 */

} /* zgeqr2_ */

/* Subroutine */ int zgeqrf_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int zgeqr2_(integer *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, integer *), xerbla_(
	    char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    nb = ilaenv_(&c__1, "ZGEQRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    lwkopt = *n * nb;
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
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
	xerbla_("ZGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    k = min(*m,*n);
    if (k == 0) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "ZGEQRF", " ", m, n, &c_n1, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZGEQRF", " ", m, n, &c_n1, &
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
	    zgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *n) {

/*
                Form the triangular factor of the block reflector
                H = H(i) H(i+1) . . . H(i+ib-1)
*/

		i__3 = *m - i__ + 1;
		zlarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i:m,i+ib:n) from the left */

		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		zlarfb_("Left", "Conjugate transpose", "Forward", "Columnwise"
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
	zgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }

    work[1].r = (doublereal) iws, work[1].i = 0.;
    return 0;

/*     End of ZGEQRF */

} /* zgeqrf_ */

/* Subroutine */ int zgesdd_(char *jobz, integer *m, integer *n,
	doublecomplex *a, integer *lda, doublereal *s, doublecomplex *u,
	integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work,
	integer *lwork, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1,
	    i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, ie, il, ir, iu, blk;
    static doublereal dum[1], eps;
    static integer iru, ivt, iscl;
    static doublereal anrm;
    static integer idum[1], ierr, itau, irvt;
    extern logical lsame_(char *, char *);
    static integer chunk, minmn;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer wrkbl, itaup, itauq;
    static logical wntqa;
    static integer nwork;
    static logical wntqn, wntqo, wntqs;
    extern /* Subroutine */ int zlacp2_(char *, integer *, integer *,
	    doublereal *, integer *, doublecomplex *, integer *);
    static integer mnthr1, mnthr2;
    extern /* Subroutine */ int dbdsdc_(char *, char *, integer *, doublereal
	    *, doublereal *, doublereal *, integer *, doublereal *, integer *,
	     doublereal *, integer *, doublereal *, integer *, integer *);

    extern /* Subroutine */ int dlascl_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, integer *, doublereal *,
	    integer *, integer *), xerbla_(char *, integer *),
	     zgebrd_(integer *, integer *, doublecomplex *, integer *,
	    doublereal *, doublereal *, doublecomplex *, doublecomplex *,
	    doublecomplex *, integer *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static doublereal bignum;
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *,
	    integer *, doublereal *);
    extern /* Subroutine */ int zgelqf_(integer *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, integer *, integer *
	    ), zlacrm_(integer *, integer *, doublecomplex *, integer *,
	    doublereal *, integer *, doublecomplex *, integer *, doublereal *)
	    , zlarcm_(integer *, integer *, doublereal *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublereal *), zlascl_(char *, integer *, integer *, doublereal *,
	     doublereal *, integer *, integer *, doublecomplex *, integer *,
	    integer *), zgeqrf_(integer *, integer *, doublecomplex *,
	     integer *, doublecomplex *, doublecomplex *, integer *, integer *
	    );
    static integer ldwrkl;
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *),
	    zlaset_(char *, integer *, integer *, doublecomplex *,
	    doublecomplex *, doublecomplex *, integer *);
    static integer ldwrkr, minwrk, ldwrku, maxwrk;
    extern /* Subroutine */ int zungbr_(char *, integer *, integer *, integer
	    *, doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);
    static integer ldwkvt;
    static doublereal smlnum;
    static logical wntqas;
    extern /* Subroutine */ int zunmbr_(char *, char *, char *, integer *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     doublecomplex *, integer *, doublecomplex *, integer *, integer *
	    ), zunglq_(integer *, integer *, integer *
	    , doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);
    static logical lquery;
    static integer nrwork;
    extern /* Subroutine */ int zungqr_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    ZGESDD computes the singular value decomposition (SVD) of a complex
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    S       (output) DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    U       (output) COMPLEX*16 array, dimension (LDU,UCOL)
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

    VT      (output) COMPLEX*16 array, dimension (LDVT,N)
            If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
            N-by-N unitary matrix V**H;
            if JOBZ = 'S', VT contains the first min(M,N) rows of
            V**H (the right singular vectors, stored rowwise);
            if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.

    LDVT    (input) INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
            if JOBZ = 'S', LDVT >= min(M,N).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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

    RWORK   (workspace) DOUBLE PRECISION array, dimension (LRWORK)
            If JOBZ = 'N', LRWORK >= 7*min(M,N).
            Otherwise, LRWORK >= 5*min(M,N)*min(M,N) + 5*min(M,N)

    IWORK   (workspace) INTEGER array, dimension (8*min(M,N))

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The updating process of DBDSDC did not converge.

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
    mnthr1 = (integer) (minmn * 17. / 9.);
    mnthr2 = (integer) (minmn * 5. / 3.);
    wntqa = lsame_(jobz, "A");
    wntqs = lsame_(jobz, "S");
    wntqas = (wntqa) || (wntqs);
    wntqo = lsame_(jobz, "O");
    wntqn = lsame_(jobz, "N");
    minwrk = 1;
    maxwrk = 1;
    lquery = *lwork == -1;

    if (! ((((wntqa) || (wntqs)) || (wntqo)) || (wntqn))) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (((*ldu < 1) || (wntqas && *ldu < *m)) || (wntqo && *m < *n && *
	    ldu < *m)) {
	*info = -8;
    } else if ((((*ldvt < 1) || (wntqa && *ldvt < *n)) || (wntqs && *ldvt <
	    minmn)) || (wntqo && *m >= *n && *ldvt < *n)) {
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

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + ((*n) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl;
		    minwrk = *n * 3;
		} else if (wntqo) {

/*                 Path 2 (M much larger than N, JOBZ='O') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "ZUNGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + ((*n) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *n * *n + wrkbl;
		    minwrk = ((*n) << (1)) * *n + *n * 3;
		} else if (wntqs) {

/*                 Path 3 (M much larger than N, JOBZ='S') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "ZUNGQR",
			    " ", m, n, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + ((*n) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + *n * 3;
		} else if (wntqa) {

/*                 Path 4 (M much larger than N, JOBZ='A') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *m * ilaenv_(&c__1, "ZUNGQR",
			    " ", m, m, n, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + ((*n) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + ((*n) << (1)) + *m;
		}
	    } else if (*m >= mnthr2) {

/*              Path 5 (M much larger than N, but not as much as MNTHR1) */

		maxwrk = ((*n) << (1)) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = ((*n) << (1)) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "P", n, n, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6 (M at least N, but not much larger) */

		maxwrk = ((*n) << (1)) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = ((*n) << (1)) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "PRC", n, n, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*n) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
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

		    maxwrk = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + ((*m) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3;
		} else if (wntqo) {

/*                 Path 2t (N much larger than M, JOBZ='O') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "ZUNGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + ((*m) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *m * *m + wrkbl;
		    minwrk = ((*m) << (1)) * *m + *m * 3;
		} else if (wntqs) {

/*                 Path 3t (N much larger than M, JOBZ='S') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "ZUNGLQ",
			    " ", m, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + ((*m) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + *m * 3;
		} else if (wntqa) {

/*                 Path 4t (N much larger than M, JOBZ='A') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *n * ilaenv_(&c__1, "ZUNGLQ",
			    " ", n, n, m, &c_n1, (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + ((*m) << (1)) *
			    ilaenv_(&c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1,
			    (ftnlen)6, (ftnlen)1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, m, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + ((*m) << (1)) + *n;
		}
	    } else if (*n >= mnthr2) {

/*              Path 5t (N much larger than M, but not as much as MNTHR1) */

		maxwrk = ((*m) << (1)) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = ((*m) << (1)) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "P", m, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "P", m, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "P", n, n, m, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "Q", m, m, n, &c_n1, (ftnlen)6, (ftnlen)
			    1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6t (N greater than M, but not much larger) */

		maxwrk = ((*m) << (1)) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD",
			" ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)1);
		minwrk = ((*m) << (1)) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "PRC", m, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNMBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "PRC", m, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *n * ilaenv_(&c__1,
			    "ZUNGBR", "PRC", n, n, m, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = ((*m) << (1)) + *m * ilaenv_(&c__1,
			    "ZUNGBR", "QLN", m, m, n, &c_n1, (ftnlen)6, (
			    ftnlen)3);
		    maxwrk = max(i__1,i__2);
		}
	    }
	}
	maxwrk = max(maxwrk,minwrk);
	work[1].r = (doublereal) maxwrk, work[1].i = 0.;
    }

    if (*lwork < minwrk && ! lquery) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGESDD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if ((*m == 0) || (*n == 0)) {
	if (*lwork >= 1) {
	    work[1].r = 1., work[1].i = 0.;
	}
	return 0;
    }

/*     Get machine constants */

    eps = PRECISION;
    smlnum = sqrt(SAFEMINIMUM) / eps;
    bignum = 1. / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = zlange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
	iscl = 1;
	zlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	zlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
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
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out below R */

		i__1 = *n - 1;
		i__2 = *n - 1;
		zlaset_("L", &i__1, &i__2, &c_b59, &c_b59, &a[a_dim1 + 2],
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
		zgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nrwork = ie + *n;

/*
                Perform bidiagonal SVD, compute singular values only
                (CWorkspace: 0)
                (RWorkspace: need BDSPAC)
*/

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
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
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy R to WORK( IR ), zeroing out below it */

		zlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		zlaset_("L", &i__1, &i__2, &c_b59, &c_b59, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		zungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
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
		zgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by the left singular vectors of R
                (CWorkspace: need 2*N*N+3*N, prefer M*N+N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by the right singular vectors of R
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
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
		    zgemm_("N", "N", &chunk, n, n, &c_b60, &a[i__ + a_dim1],
			    lda, &work[iu], &ldwrku, &c_b59, &work[ir], &
			    ldwrkr);
		    zlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
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
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		zlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		zlaset_("L", &i__2, &i__1, &c_b59, &c_b59, &work[ir + 1], &
			ldwrkr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		zungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
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
		zgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in A by left singular vectors of R in
                WORK(IR), storing result in U
                (CWorkspace: need N*N)
                (RWorkspace: 0)
*/

		zlacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		zgemm_("N", "N", m, n, n, &c_b60, &a[a_offset], lda, &work[ir]
			, &ldwrkr, &c_b59, &u[u_offset], ldu);

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
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Generate Q in U
                (CWorkspace: need N+M, prefer N+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		zungqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork],
			 &i__2, &ierr);

/*              Produce R in A, zeroing out below it */

		i__2 = *n - 1;
		i__1 = *n - 1;
		zlaset_("L", &i__2, &i__1, &c_b59, &c_b59, &a[a_dim1 + 2],
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
		zgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
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

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by left singular vectors of R
                (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of R
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*
                Multiply Q in U by left singular vectors of R in
                WORK(IU), storing result in A
                (CWorkspace: need N*N)
                (RWorkspace: 0)
*/

		zgemm_("N", "N", m, n, n, &c_b60, &u[u_offset], ldu, &work[iu]
			, &ldwrku, &c_b59, &a[a_offset], lda);

/*              Copy left singular vectors of A from A to U */

		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);

	    }

	} else if (*m >= mnthr2) {

/*
             MNTHR2 <= M < MNTHR1

             Path 5 (M much larger than N, but not as much as MNTHR1)
             Reduce to bidiagonal form without QR decomposition, use
             ZUNGBR and matrix multiplication to compute singular vectors
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
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
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

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*
                Generate Q in A
                (CWorkspace: need 2*N, prefer N+N*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &work[
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

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in WORK(IU), copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &work[iu]
			, &ldwrku, &rwork[nrwork]);
		zlacpy_("F", n, n, &work[iu], &ldwrku, &vt[vt_offset], ldvt);

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
		    zlacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru], n,
			    &work[iu], &ldwrku, &rwork[nrwork]);
		    zlacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ +
			    a_dim1], lda);
/* L20: */
		}

	    } else if (wntqs) {

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, n, n, &u[u_offset], ldu, &work[itauq], &work[
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need N*N+2*M*N)
*/

		nrwork = irvt;
		zlacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset],
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    } else {

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: 0)
*/

		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need 3*N*N)
*/

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: 0)
                (Rworkspace: need 3*N*N)
*/

		nrwork = irvt;
		zlacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset],
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    }

	} else {

/*
             M .LT. MNTHR2

             Path 6 (M at least N, but not much larger)
             Reduce to bidiagonal form without QR decomposition
             Use ZUNMBR to compute singular vectors
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
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__1, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
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

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (Cworkspace: need 2*N, prefer N+N*NB)
                (Rworkspace: need 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
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

		    zlaset_("F", m, n, &c_b59, &c_b59, &work[iu], &ldwrku);
		    zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		    i__1 = *lwork - nwork + 1;
		    zunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			    ierr);
		    zlacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {

/*
                   Generate Q in A
                   (Cworkspace: need 2*N, prefer N+N*NB)
                   (Rworkspace: need 0)
*/

		    i__1 = *lwork - nwork + 1;
		    zungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
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
			zlacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru],
				 n, &work[iu], &ldwrku, &rwork[nrwork]);
			zlacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ +
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		zlaset_("F", m, n, &c_b59, &c_b59, &u[u_offset], ldu);
		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
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
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*              Set the right corner of U to identity matrix */

		zlaset_("F", m, m, &c_b59, &c_b59, &u[u_offset], ldu);
		i__2 = *m - *n;
		i__1 = *m - *n;
		zlaset_("F", &i__2, &i__1, &c_b59, &c_b60, &u[*n + 1 + (*n +
			1) * u_dim1], ldu);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 2*N+M, prefer 2*N+M*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*N, prefer 2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
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
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Zero out above L */

		i__2 = *m - 1;
		i__1 = *m - 1;
		zlaset_("U", &i__2, &i__1, &c_b59, &c_b59, &a[((a_dim1) << (1)
			) + 1], lda);
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
		zgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		nrwork = ie + *m;

/*
                Perform bidiagonal SVD, compute singular values only
                (CWorkspace: 0)
                (RWorkspace: need BDSPAC)
*/

		dbdsdc_("U", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
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
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy L to WORK(IL), zeroing about above it */

		zlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		zlaset_("U", &i__2, &i__1, &c_b59, &c_b59, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
                (RWorkspace: 0)
*/

		i__2 = *lwork - nwork + 1;
		zunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
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
		zgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
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
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix WORK(IU)
                Overwrite WORK(IU) by the left singular vectors of L
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by the right singular vectors of L
                (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
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
		    zgemm_("N", "N", m, &blk, m, &c_b60, &work[ivt], m, &a[
			    i__ * a_dim1 + 1], lda, &c_b59, &work[il], &
			    ldwrkl);
		    zlacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1
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
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy L to WORK(IL), zeroing out above it */

		zlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		zlaset_("U", &i__1, &i__2, &c_b59, &c_b59, &work[il + ldwrkl],
			 &ldwrkl);

/*
                Generate Q in A
                (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		zunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
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
		zgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
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
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by left singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*
                Copy VT to WORK(IL), multiply right singular vectors of L
                in WORK(IL) by Q in A, storing result in VT
                (CWorkspace: need M*M)
                (RWorkspace: 0)
*/

		zlacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		zgemm_("N", "N", m, n, m, &c_b60, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b59, &vt[vt_offset], ldvt);

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
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*
                Generate Q in VT
                (CWorkspace: need M+N, prefer M+N*NB)
                (RWorkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		zunglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__1, &ierr);

/*              Produce L in A, zeroing out above it */

		i__1 = *m - 1;
		i__2 = *m - 1;
		zlaset_("U", &i__1, &i__2, &c_b59, &c_b59, &a[((a_dim1) << (1)
			) + 1], lda);
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
		zgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
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
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of L
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by right singular vectors of L
                (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB)
                (RWorkspace: 0)
*/

		zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__1, &
			ierr);

/*
                Multiply right singular vectors of L in WORK(IVT) by
                Q in VT, storing result in A
                (CWorkspace: need M*M)
                (RWorkspace: 0)
*/

		zgemm_("N", "N", m, n, m, &c_b60, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b59, &a[a_offset], lda);

/*              Copy right singular vectors of A from A to VT */

		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

	    }

	} else if (*n >= mnthr2) {

/*
             MNTHR2 <= N < MNTHR1

             Path 5t (N much larger than M, but not as much as MNTHR1)
             Reduce to bidiagonal form without QR decomposition, use
             ZUNGBR and matrix multiplication to compute singular vectors
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
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__1, &ierr);

	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		dbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
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

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*
                Generate P**H in A
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		i__1 = *lwork - nwork + 1;
		zungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &work[
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

		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRVT)
                storing the result in WORK(IVT), copying to U
                (Cworkspace: need 0)
                (Rworkspace: need 2*M*M)
*/

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &work[ivt], &
			ldwkvt, &rwork[nrwork]);
		zlacpy_("F", m, m, &work[ivt], &ldwkvt, &u[u_offset], ldu);

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
		    zlarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1],
			    lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
		    zlacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ *
			    a_dim1 + 1], lda);
/* L50: */
		}
	    } else if (wntqs) {

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", m, n, m, &vt[vt_offset], ldvt, &work[itaup], &
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
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need 3*M*M)
*/

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset],
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need M*M+2*M*N)
*/

		nrwork = iru;
		zlarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    } else {

/*
                Copy A to U, generate Q
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*
                Copy A to VT, generate P**H
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: 0)
*/

		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", n, n, m, &vt[vt_offset], ldvt, &work[itaup], &
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
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Multiply Q in U by real matrix RWORK(IRU), storing the
                result in A, copying to U
                (CWorkspace: need 0)
                (Rworkspace: need 3*M*M)
*/

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset],
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*
                Multiply real matrix RWORK(IRVT) by P**H in VT,
                storing the result in A, copying to VT
                (Cworkspace: need 0)
                (Rworkspace: need M*M+2*M*N)
*/

		zlarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    }

	} else {

/*
             N .LT. MNTHR2

             Path 6t (N greater than M, but not much larger)
             Reduce to bidiagonal form without LQ decomposition
             Use ZUNMBR to compute singular vectors
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
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq],
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*
                Compute singular values only
                (Cworkspace: 0)
                (Rworkspace: need BDSPAC)
*/

		dbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3) {

/*                 WORK( IVT ) is M by N */

		    zlaset_("F", m, n, &c_b59, &c_b59, &work[ivt], &ldwkvt);
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
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (Cworkspace: need 2*M, prefer M+M*NB)
                (Rworkspace: need 0)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *m * 3) {

/*
                Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT)
                Overwrite WORK(IVT) by right singular vectors of A,
                copying to A
                (Cworkspace: need M*N+2*M, prefer M*N+M+M*NB)
                (Rworkspace: need 0)
*/

		    zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		    i__2 = *lwork - nwork + 1;
		    zunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2,
			    &ierr);
		    zlacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda);
		} else {

/*
                   Generate P**H in A
                   (Cworkspace: need 2*M, prefer M+M*NB)
                   (Rworkspace: need 0)
*/

		    i__2 = *lwork - nwork + 1;
		    zungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
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
			zlarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1]
				, lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
			zlacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ *
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
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		zlaset_("F", m, n, &c_b59, &c_b59, &vt[vt_offset], ldvt);
		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
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

		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1],
			info);

/*
                Copy real matrix RWORK(IRU) to complex matrix U
                Overwrite U by left singular vectors of A
                (CWorkspace: need 3*M, prefer 2*M+M*NB)
                (RWorkspace: M*M)
*/

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Set the right corner of VT to identity matrix */

		i__1 = *n - *m;
		i__2 = *n - *m;
		zlaset_("F", &i__1, &i__2, &c_b59, &c_b60, &vt[*m + 1 + (*m +
			1) * vt_dim1], ldvt);

/*
                Copy real matrix RWORK(IRVT) to complex matrix VT
                Overwrite VT by right singular vectors of A
                (CWorkspace: need 2*M+N, prefer 2*M+N*NB)
                (RWorkspace: M*M)
*/

		zlaset_("F", n, n, &c_b59, &c_b59, &vt[vt_offset], ldvt);
		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    }

/*     Undo scaling if necessary */

    if (iscl == 1) {
	if (anrm > bignum) {
	    dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (anrm < smlnum) {
	    dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
    }

/*     Return optimal workspace in WORK(1) */

    work[1].r = (doublereal) maxwrk, work[1].i = 0.;

    return 0;

/*     End of ZGESDD */

} /* zgesdd_ */

/* Subroutine */ int zgesv_(integer *n, integer *nrhs, doublecomplex *a,
	integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern /* Subroutine */ int xerbla_(char *, integer *), zgetrf_(
	    integer *, integer *, doublecomplex *, integer *, integer *,
	    integer *), zgetrs_(char *, integer *, integer *, doublecomplex *,
	     integer *, integer *, doublecomplex *, integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    ZGESV computes the solution to a complex system of linear equations
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (output) INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
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
	xerbla_("ZGESV ", &i__1);
	return 0;
    }

/*     Compute the LU factorization of A. */

    zgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);
    if (*info == 0) {

/*        Solve the system A*X = B, overwriting B with X. */

	zgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &b[
		b_offset], ldb, info);
    }
    return 0;

/*     End of ZGESV */

} /* zgesv_ */

/* Subroutine */ int zgetf2_(integer *m, integer *n, doublecomplex *a,
	integer *lda, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void z_div(doublecomplex *, doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer j, jp;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), zgeru_(integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *), zswap_(integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(
	    char *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGETF2 computes an LU factorization of a general m-by-n matrix A
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZGETF2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if ((*m == 0) || (*n == 0)) {
	return 0;
    }

    i__1 = min(*m,*n);
    for (j = 1; j <= i__1; ++j) {

/*        Find pivot and test for singularity. */

	i__2 = *m - j + 1;
	jp = j - 1 + izamax_(&i__2, &a[j + j * a_dim1], &c__1);
	ipiv[j] = jp;
	i__2 = jp + j * a_dim1;
	if ((a[i__2].r != 0.) || (a[i__2].i != 0.)) {

/*           Apply the interchange to columns 1:N. */

	    if (jp != j) {
		zswap_(n, &a[j + a_dim1], lda, &a[jp + a_dim1], lda);
	    }

/*           Compute elements J+1:M of J-th column. */

	    if (j < *m) {
		i__2 = *m - j;
		z_div(&z__1, &c_b60, &a[j + j * a_dim1]);
		zscal_(&i__2, &z__1, &a[j + 1 + j * a_dim1], &c__1);
	    }

	} else if (*info == 0) {

	    *info = j;
	}

	if (j < min(*m,*n)) {

/*           Update trailing submatrix. */

	    i__2 = *m - j;
	    i__3 = *n - j;
	    z__1.r = -1., z__1.i = -0.;
	    zgeru_(&i__2, &i__3, &z__1, &a[j + 1 + j * a_dim1], &c__1, &a[j +
		    (j + 1) * a_dim1], lda, &a[j + 1 + (j + 1) * a_dim1], lda)
		    ;
	}
/* L10: */
    }
    return 0;

/*     End of ZGETF2 */

} /* zgetf2_ */

/* Subroutine */ int zgetrf_(integer *m, integer *n, doublecomplex *a,
	integer *lda, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, jb, nb, iinfo;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), ztrsm_(char *, char *, char *, char *,
	     integer *, integer *, doublecomplex *, doublecomplex *, integer *
	    , doublecomplex *, integer *),
	    zgetf2_(integer *, integer *, doublecomplex *, integer *, integer
	    *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlaswp_(integer *, doublecomplex *, integer *,
	     integer *, integer *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGETRF computes an LU factorization of a general M-by-N matrix A
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZGETRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if ((*m == 0) || (*n == 0)) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "ZGETRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    if ((nb <= 1) || (nb >= min(*m,*n))) {

/*        Use unblocked code. */

	zgetf2_(m, n, &a[a_offset], lda, &ipiv[1], info);
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
	    zgetf2_(&i__3, &jb, &a[j + j * a_dim1], lda, &ipiv[j], &iinfo);

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
	    zlaswp_(&i__3, &a[a_offset], lda, &j, &i__4, &ipiv[1], &c__1);

	    if (j + jb <= *n) {

/*              Apply interchanges to columns J+JB:N. */

		i__3 = *n - j - jb + 1;
		i__4 = j + jb - 1;
		zlaswp_(&i__3, &a[(j + jb) * a_dim1 + 1], lda, &j, &i__4, &
			ipiv[1], &c__1);

/*              Compute block row of U. */

		i__3 = *n - j - jb + 1;
		ztrsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
			c_b60, &a[j + j * a_dim1], lda, &a[j + (j + jb) *
			a_dim1], lda);
		if (j + jb <= *m) {

/*                 Update trailing submatrix. */

		    i__3 = *m - j - jb + 1;
		    i__4 = *n - j - jb + 1;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "No transpose", &i__3, &i__4, &jb,
			    &z__1, &a[j + jb + j * a_dim1], lda, &a[j + (j +
			    jb) * a_dim1], lda, &c_b60, &a[j + jb + (j + jb) *
			     a_dim1], lda);
		}
	    }
/* L20: */
	}
    }
    return 0;

/*     End of ZGETRF */

} /* zgetrf_ */

/* Subroutine */ int zgetrs_(char *trans, integer *n, integer *nrhs,
	doublecomplex *a, integer *lda, integer *ipiv, doublecomplex *b,
	integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ztrsm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *),
	    xerbla_(char *, integer *);
    static logical notran;
    extern /* Subroutine */ int zlaswp_(integer *, doublecomplex *, integer *,
	     integer *, integer *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZGETRS solves a system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B
    with a general N-by-N matrix A using the LU factorization computed
    by ZGETRF.

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

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U
            as computed by ZGETRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
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
	xerbla_("ZGETRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if ((*n == 0) || (*nrhs == 0)) {
	return 0;
    }

    if (notran) {

/*
          Solve A * X = B.

          Apply row interchanges to the right hand sides.
*/

	zlaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);

/*        Solve L*X = B, overwriting B with X. */

	ztrsm_("Left", "Lower", "No transpose", "Unit", n, nrhs, &c_b60, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	ztrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b60, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A**T * X = B  or A**H * X = B.

          Solve U'*X = B, overwriting B with X.
*/

	ztrsm_("Left", "Upper", trans, "Non-unit", n, nrhs, &c_b60, &a[
		a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	ztrsm_("Left", "Lower", trans, "Unit", n, nrhs, &c_b60, &a[a_offset],
		lda, &b[b_offset], ldb);

/*        Apply row interchanges to the solution vectors. */

	zlaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return 0;

/*     End of ZGETRS */

} /* zgetrs_ */

/* Subroutine */ int zheevd_(char *jobz, char *uplo, integer *n,
	doublecomplex *a, integer *lda, doublereal *w, doublecomplex *work,
	integer *lwork, doublereal *rwork, integer *lrwork, integer *iwork,
	integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal eps;
    static integer inde;
    static doublereal anrm;
    static integer imax;
    static doublereal rmin, rmax;
    static integer lopt;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *,
	    integer *);
    static doublereal sigma;
    extern logical lsame_(char *, char *);
    static integer iinfo, lwmin, liopt;
    static logical lower;
    static integer llrwk, lropt;
    static logical wantz;
    static integer indwk2, llwrk2;

    static integer iscale;
    static doublereal safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static doublereal bignum;
    extern doublereal zlanhe_(char *, char *, integer *, doublecomplex *,
	    integer *, doublereal *);
    static integer indtau;
    extern /* Subroutine */ int dsterf_(integer *, doublereal *, doublereal *,
	     integer *), zlascl_(char *, integer *, integer *, doublereal *,
	    doublereal *, integer *, integer *, doublecomplex *, integer *,
	    integer *), zstedc_(char *, integer *, doublereal *,
	    doublereal *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublereal *, integer *, integer *, integer *, integer
	    *);
    static integer indrwk, indwrk, liwmin;
    extern /* Subroutine */ int zhetrd_(char *, integer *, doublecomplex *,
	    integer *, doublereal *, doublereal *, doublecomplex *,
	    doublecomplex *, integer *, integer *), zlacpy_(char *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     integer *);
    static integer lrwmin, llwork;
    static doublereal smlnum;
    static logical lquery;
    extern /* Subroutine */ int zunmtr_(char *, char *, char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *);


/*
    -- LAPACK driver routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZHEEVD computes all eigenvalues and, optionally, eigenvectors of a
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

    A       (input/output) COMPLEX*16 array, dimension (LDA, N)
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

    W       (output) DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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

    RWORK   (workspace/output) DOUBLE PRECISION array,
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
    lquery = ((*lwork == -1) || (*lrwork == -1)) || (*liwork == -1);

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
	    lwmin = ((*n) << (1)) + *n * *n;
/* Computing 2nd power */
	    i__1 = *n;
	    lrwmin = *n * 5 + 1 + ((i__1 * i__1) << (1));
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
    if (! ((wantz) || (lsame_(jobz, "N")))) {
	*info = -1;
    } else if (! ((lower) || (lsame_(uplo, "U")))) {
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
	work[1].r = (doublereal) lopt, work[1].i = 0.;
	rwork[1] = (doublereal) lropt;
	iwork[1] = liopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHEEVD", &i__1);
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
	    a[i__1].r = 1., a[i__1].i = 0.;
	}
	return 0;
    }

/*     Get machine constants. */

    safmin = SAFEMINIMUM;
    eps = PRECISION;
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

/*     Scale matrix to allowable range, if necessary. */

    anrm = zlanhe_("M", uplo, n, &a[a_offset], lda, &rwork[1]);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	zlascl_(uplo, &c__0, &c__0, &c_b1015, &sigma, n, n, &a[a_offset], lda,
		 info);
    }

/*     Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */

    inde = 1;
    indtau = 1;
    indwrk = indtau + *n;
    indrwk = inde + *n;
    indwk2 = indwrk + *n * *n;
    llwork = *lwork - indwrk + 1;
    llwrk2 = *lwork - indwk2 + 1;
    llrwk = *lrwork - indrwk + 1;
    zhetrd_(uplo, n, &a[a_offset], lda, &w[1], &rwork[inde], &work[indtau], &
	    work[indwrk], &llwork, &iinfo);
/* Computing MAX */
    i__1 = indwrk;
    d__1 = (doublereal) lopt, d__2 = (doublereal) (*n) + work[i__1].r;
    lopt = (integer) max(d__1,d__2);

/*
       For eigenvalues only, call DSTERF.  For eigenvectors, first call
       ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call ZUNMTR to multiply it to the
       Householder transformations represented as Householder vectors in
       A.
*/

    if (! wantz) {
	dsterf_(n, &w[1], &rwork[inde], info);
    } else {
	zstedc_("I", n, &w[1], &rwork[inde], &work[indwrk], n, &work[indwk2],
		&llwrk2, &rwork[indrwk], &llrwk, &iwork[1], liwork, info);
	zunmtr_("L", uplo, "N", n, n, &a[a_offset], lda, &work[indtau], &work[
		indwrk], n, &work[indwk2], &llwrk2, &iinfo);
	zlacpy_("A", n, n, &work[indwrk], n, &a[a_offset], lda);
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
	d__1 = 1. / sigma;
	dscal_(&imax, &d__1, &w[1], &c__1);
    }

    work[1].r = (doublereal) lopt, work[1].i = 0.;
    rwork[1] = (doublereal) lropt;
    iwork[1] = liopt;

    return 0;

/*     End of ZHEEVD */

} /* zheevd_ */

/* Subroutine */ int zhetd2_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tau,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    static integer i__;
    static doublecomplex taui;
    extern /* Subroutine */ int zher2_(char *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static doublecomplex alpha;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zhemv_(char *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int zaxpy_(integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(
	    char *, integer *), zlarfg_(integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    ZHETD2 reduces a complex Hermitian matrix A to real symmetric
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    D       (output) DOUBLE PRECISION array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) COMPLEX*16 array, dimension (N-1)
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
	xerbla_("ZHETD2", &i__1);
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
	d__1 = a[i__2].r;
	a[i__1].r = d__1, a[i__1].i = 0.;
	for (i__ = *n - 1; i__ >= 1; --i__) {

/*
             Generate elementary reflector H(i) = I - tau * v * v'
             to annihilate A(1:i-1,i+1)
*/

	    i__1 = i__ + (i__ + 1) * a_dim1;
	    alpha.r = a[i__1].r, alpha.i = a[i__1].i;
	    zlarfg_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &taui);
	    i__1 = i__;
	    e[i__1] = alpha.r;

	    if ((taui.r != 0.) || (taui.i != 0.)) {

/*              Apply H(i) from both sides to A(1:i,1:i) */

		i__1 = i__ + (i__ + 1) * a_dim1;
		a[i__1].r = 1., a[i__1].i = 0.;

/*              Compute  x := tau * A * v  storing x in TAU(1:i) */

		zhemv_(uplo, &i__, &taui, &a[a_offset], lda, &a[(i__ + 1) *
			a_dim1 + 1], &c__1, &c_b59, &tau[1], &c__1)
			;

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		z__3.r = -.5, z__3.i = -0.;
		z__2.r = z__3.r * taui.r - z__3.i * taui.i, z__2.i = z__3.r *
			taui.i + z__3.i * taui.r;
		zdotc_(&z__4, &i__, &tau[1], &c__1, &a[(i__ + 1) * a_dim1 + 1]
			, &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r *
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		zaxpy_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &tau[
			1], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		z__1.r = -1., z__1.i = -0.;
		zher2_(uplo, &i__, &z__1, &a[(i__ + 1) * a_dim1 + 1], &c__1, &
			tau[1], &c__1, &a[a_offset], lda);

	    } else {
		i__1 = i__ + i__ * a_dim1;
		i__2 = i__ + i__ * a_dim1;
		d__1 = a[i__2].r;
		a[i__1].r = d__1, a[i__1].i = 0.;
	    }
	    i__1 = i__ + (i__ + 1) * a_dim1;
	    i__2 = i__;
	    a[i__1].r = e[i__2], a[i__1].i = 0.;
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
	d__1 = a[i__2].r;
	a[i__1].r = d__1, a[i__1].i = 0.;
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
	    zlarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &
		    taui);
	    i__2 = i__;
	    e[i__2] = alpha.r;

	    if ((taui.r != 0.) || (taui.i != 0.)) {

/*              Apply H(i) from both sides to A(i+1:n,i+1:n) */

		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute  x := tau * A * v  storing y in TAU(i:n-1) */

		i__2 = *n - i__;
		zhemv_(uplo, &i__2, &taui, &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b59, &tau[
			i__], &c__1);

/*              Compute  w := x - 1/2 * tau * (x'*v) * v */

		z__3.r = -.5, z__3.i = -0.;
		z__2.r = z__3.r * taui.r - z__3.i * taui.i, z__2.i = z__3.r *
			taui.i + z__3.i * taui.r;
		i__2 = *n - i__;
		zdotc_(&z__4, &i__2, &tau[i__], &c__1, &a[i__ + 1 + i__ *
			a_dim1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r *
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = *n - i__;
		zaxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
			i__], &c__1);

/*
                Apply the transformation as a rank-2 update:
                   A := A - v * w' - w * v'
*/

		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zher2_(uplo, &i__2, &z__1, &a[i__ + 1 + i__ * a_dim1], &c__1,
			&tau[i__], &c__1, &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda);

	    } else {
		i__2 = i__ + 1 + (i__ + 1) * a_dim1;
		i__3 = i__ + 1 + (i__ + 1) * a_dim1;
		d__1 = a[i__3].r;
		a[i__2].r = d__1, a[i__2].i = 0.;
	    }
	    i__2 = i__ + 1 + i__ * a_dim1;
	    i__3 = i__;
	    a[i__2].r = e[i__3], a[i__2].i = 0.;
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

/*     End of ZHETD2 */

} /* zhetd2_ */

/* Subroutine */ int zhetrd_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, doublereal *d__, doublereal *e, doublecomplex *tau,
	doublecomplex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, nb, kk, nx, iws;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    static logical upper;
    extern /* Subroutine */ int zhetd2_(char *, integer *, doublecomplex *,
	    integer *, doublereal *, doublereal *, doublecomplex *, integer *), zher2k_(char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublereal *, doublecomplex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlatrd_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublereal *, doublecomplex *,
	    doublecomplex *, integer *);
    static integer ldwork, lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZHETRD reduces a complex Hermitian matrix A to real symmetric
    tridiagonal form T by a unitary similarity transformation:
    Q**H * A * Q = T.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    D       (output) DOUBLE PRECISION array, dimension (N)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.

    TAU     (output) COMPLEX*16 array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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

	nb = ilaenv_(&c__1, "ZHETRD", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6,
		 (ftnlen)1);
	lwkopt = *n * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHETRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = nb, i__2 = ilaenv_(&c__3, "ZHETRD", uplo, n, &c_n1, &c_n1, &
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
		nbmin = ilaenv_(&c__2, "ZHETRD", uplo, n, &c_n1, &c_n1, &c_n1,
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
	    zlatrd_(uplo, &i__3, &nb, &a[a_offset], lda, &e[1], &tau[1], &
		    work[1], &ldwork);

/*
             Update the unreduced submatrix A(1:i-1,1:i-1), using an
             update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zher2k_(uplo, "No transpose", &i__3, &nb, &z__1, &a[i__ * a_dim1
		    + 1], lda, &work[1], &ldwork, &c_b1015, &a[a_offset], lda);

/*
             Copy superdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j - 1 + j * a_dim1;
		i__5 = j - 1;
		a[i__4].r = e[i__5], a[i__4].i = 0.;
		i__4 = j;
		i__5 = j + j * a_dim1;
		d__[i__4] = a[i__5].r;
/* L10: */
	    }
/* L20: */
	}

/*        Use unblocked code to reduce the last or only block */

	zhetd2_(uplo, &kk, &a[a_offset], lda, &d__[1], &e[1], &tau[1], &iinfo);
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
	    zlatrd_(uplo, &i__3, &nb, &a[i__ + i__ * a_dim1], lda, &e[i__], &
		    tau[i__], &work[1], &ldwork);

/*
             Update the unreduced submatrix A(i+nb:n,i+nb:n), using
             an update of the form:  A := A - V*W' - W*V'
*/

	    i__3 = *n - i__ - nb + 1;
	    z__1.r = -1., z__1.i = -0.;
	    zher2k_(uplo, "No transpose", &i__3, &nb, &z__1, &a[i__ + nb +
		    i__ * a_dim1], lda, &work[nb + 1], &ldwork, &c_b1015, &a[
		    i__ + nb + (i__ + nb) * a_dim1], lda);

/*
             Copy subdiagonal elements back into A, and diagonal
             elements into D
*/

	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		i__4 = j + 1 + j * a_dim1;
		i__5 = j;
		a[i__4].r = e[i__5], a[i__4].i = 0.;
		i__4 = j;
		i__5 = j + j * a_dim1;
		d__[i__4] = a[i__5].r;
/* L30: */
	    }
/* L40: */
	}

/*        Use unblocked code to reduce the last or only block */

	i__1 = *n - i__ + 1;
	zhetd2_(uplo, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__],
		&tau[i__], &iinfo);
    }

    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZHETRD */

} /* zhetrd_ */

/* Subroutine */ int zhseqr_(char *job, char *compz, integer *n, integer *ilo,
	 integer *ihi, doublecomplex *h__, integer *ldh, doublecomplex *w,
	doublecomplex *z__, integer *ldz, doublecomplex *work, integer *lwork,
	 integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4[2],
	    i__5, i__6;
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1;
    char ch__1[2];

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer i__, j, k, l;
    static doublecomplex s[225]	/* was [15][15] */, v[16];
    static integer i1, i2, ii, nh, nr, ns, nv;
    static doublecomplex vv[16];
    static integer itn;
    static doublecomplex tau;
    static integer its;
    static doublereal ulp, tst1;
    static integer maxb, ierr;
    static doublereal unfl;
    static doublecomplex temp;
    static doublereal ovfl;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *);
    static integer itemp;
    static doublereal rtemp;
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);
    static logical initz, wantt, wantz;
    static doublereal rwork[1];
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    extern doublereal dlapy2_(doublereal *, doublereal *);
    extern /* Subroutine */ int dlabad_(doublereal *, doublereal *);

    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zdscal_(integer *, doublereal *,
	    doublecomplex *, integer *), zlarfg_(integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern doublereal zlanhs_(char *, integer *, doublecomplex *, integer *,
	    doublereal *);
    extern /* Subroutine */ int zlahqr_(logical *, logical *, integer *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     integer *, integer *, doublecomplex *, integer *, integer *),
	    zlacpy_(char *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zlaset_(char *, integer *,
	    integer *, doublecomplex *, doublecomplex *, doublecomplex *,
	    integer *), zlarfx_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *);
    static doublereal smlnum;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZHSEQR computes the eigenvalues of a complex upper Hessenberg
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
            set by a previous call to ZGEBAL, and then passed to CGEHRD
            when the matrix output by ZGEBAL is reduced to Hessenberg
            form. Otherwise ILO and IHI should be set to 1 and N
            respectively.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    H       (input/output) COMPLEX*16 array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if JOB = 'S', H contains the upper triangular matrix
            T from the Schur decomposition (the Schur form). If
            JOB = 'E', the contents of H are unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    W       (output) COMPLEX*16 array, dimension (N)
            The computed eigenvalues. If JOB = 'S', the eigenvalues are
            stored in the same order as on the diagonal of the Schur form
            returned in H, with W(i) = H(i,i).

    Z       (input/output) COMPLEX*16 array, dimension (LDZ,N)
            If COMPZ = 'N': Z is not referenced.
            If COMPZ = 'I': on entry, Z need not be set, and on exit, Z
            contains the unitary matrix Z of the Schur vectors of H.
            If COMPZ = 'V': on entry Z must contain an N-by-N matrix Q,
            which is assumed to be equal to the unit matrix except for
            the submatrix Z(ILO:IHI,ILO:IHI); on exit Z contains Q*Z.
            Normally Q is the unitary matrix generated by ZUNGHR after
            the call to ZGEHRD which formed the Hessenberg matrix H.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.
            LDZ >= max(1,N) if COMPZ = 'I' or 'V'; LDZ >= 1 otherwise.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
            > 0:  if INFO = i, ZHSEQR failed to compute all the
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
    wantz = (initz) || (lsame_(compz, "V"));

    *info = 0;
    i__1 = max(1,*n);
    work[1].r = (doublereal) i__1, work[1].i = 0.;
    lquery = *lwork == -1;
    if (! lsame_(job, "E") && ! wantt) {
	*info = -1;
    } else if (! lsame_(compz, "N") && ! wantz) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if ((*ilo < 1) || (*ilo > max(1,*n))) {
	*info = -4;
    } else if ((*ihi < min(*ilo,*n)) || (*ihi > *n)) {
	*info = -5;
    } else if (*ldh < max(1,*n)) {
	*info = -7;
    } else if ((*ldz < 1) || (wantz && *ldz < max(1,*n))) {
	*info = -10;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHSEQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Initialize Z, if necessary */

    if (initz) {
	zlaset_("Full", n, n, &c_b59, &c_b60, &z__[z_offset], ldz);
    }

/*     Store the eigenvalues isolated by ZGEBAL. */

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
	    h__[i__3].r = 0., h__[i__3].i = 0.;
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
	if (d_imag(&temp) != 0.) {
	    d__1 = temp.r;
	    d__2 = d_imag(&temp);
	    rtemp = dlapy2_(&d__1, &d__2);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.;
	    z__1.r = temp.r / rtemp, z__1.i = temp.i / rtemp;
	    temp.r = z__1.r, temp.i = z__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		d_cnjg(&z__1, &temp);
		zscal_(&i__2, &z__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    zscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (i__ < *ihi) {
		i__2 = i__ + 1 + i__ * h_dim1;
		i__3 = i__ + 1 + i__ * h_dim1;
		z__1.r = temp.r * h__[i__3].r - temp.i * h__[i__3].i, z__1.i =
			 temp.r * h__[i__3].i + temp.i * h__[i__3].r;
		h__[i__2].r = z__1.r, h__[i__2].i = z__1.i;
	    }
	    if (wantz) {
		zscal_(&nh, &temp, &z__[*ilo + i__ * z_dim1], &c__1);
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
    ns = ilaenv_(&c__4, "ZHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
/* Writing concatenation */
    i__4[0] = 1, a__1[0] = job;
    i__4[1] = 1, a__1[1] = compz;
    s_cat(ch__1, a__1, i__4, &c__2, (ftnlen)2);
    maxb = ilaenv_(&c__8, "ZHSEQR", ch__1, n, ilo, ihi, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if (((ns <= 1) || (ns > nh)) || (maxb >= nh)) {

/*        Use the standard double-shift algorithm */

	zlahqr_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &w[1], ilo,
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

    unfl = SAFEMINIMUM;
    ovfl = 1. / unfl;
    dlabad_(&unfl, &ovfl);
    ulp = PRECISION;
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
	    tst1 = (d__1 = h__[i__3].r, abs(d__1)) + (d__2 = d_imag(&h__[k -
		    1 + (k - 1) * h_dim1]), abs(d__2)) + ((d__3 = h__[i__5].r,
		     abs(d__3)) + (d__4 = d_imag(&h__[k + k * h_dim1]), abs(
		    d__4)));
	    if (tst1 == 0.) {
		i__3 = i__ - l + 1;
		tst1 = zlanhs_("1", &i__3, &h__[l + l * h_dim1], ldh, rwork);
	    }
	    i__3 = k + (k - 1) * h_dim1;
/* Computing MAX */
	    d__2 = ulp * tst1;
	    if ((d__1 = h__[i__3].r, abs(d__1)) <= max(d__2,smlnum)) {
		goto L80;
	    }
/* L70: */
	}
L80:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible. */

	    i__2 = l + (l - 1) * h_dim1;
	    h__[i__2].r = 0., h__[i__2].i = 0.;
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

	if ((its == 20) || (its == 30)) {

/*           Exceptional shifts. */

	    i__2 = i__;
	    for (ii = i__ - ns + 1; ii <= i__2; ++ii) {
		i__3 = ii;
		i__5 = ii + (ii - 1) * h_dim1;
		i__6 = ii + ii * h_dim1;
		d__3 = ((d__1 = h__[i__5].r, abs(d__1)) + (d__2 = h__[i__6].r,
			 abs(d__2))) * 1.5;
		w[i__3].r = d__3, w[i__3].i = 0.;
/* L90: */
	    }
	} else {

/*           Use eigenvalues of trailing submatrix of order NS as shifts. */

	    zlacpy_("Full", &ns, &ns, &h__[i__ - ns + 1 + (i__ - ns + 1) *
		    h_dim1], ldh, s, &c__15);
	    zlahqr_(&c_false, &c_false, &ns, &c__1, &ns, s, &c__15, &w[i__ -
		    ns + 1], &c__1, &ns, &z__[z_offset], ldz, &ierr);
	    if (ierr > 0) {

/*
                If ZLAHQR failed to compute all NS eigenvalues, use the
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

	v[0].r = 1., v[0].i = 0.;
	i__2 = ns + 1;
	for (ii = 2; ii <= i__2; ++ii) {
	    i__3 = ii - 1;
	    v[i__3].r = 0., v[i__3].i = 0.;
/* L110: */
	}
	nv = 1;
	i__2 = i__;
	for (j = i__ - ns + 1; j <= i__2; ++j) {
	    i__3 = nv + 1;
	    zcopy_(&i__3, v, &c__1, vv, &c__1);
	    i__3 = nv + 1;
	    i__5 = j;
	    z__1.r = -w[i__5].r, z__1.i = -w[i__5].i;
	    zgemv_("No transpose", &i__3, &nv, &c_b60, &h__[l + l * h_dim1],
		    ldh, vv, &c__1, &z__1, v, &c__1);
	    ++nv;

/*
             Scale V(1:NV) so that max(abs(V(i))) = 1. If V is zero,
             reset it to the unit vector.
*/

	    itemp = izamax_(&nv, v, &c__1);
	    i__3 = itemp - 1;
	    rtemp = (d__1 = v[i__3].r, abs(d__1)) + (d__2 = d_imag(&v[itemp -
		    1]), abs(d__2));
	    if (rtemp == 0.) {
		v[0].r = 1., v[0].i = 0.;
		i__3 = nv;
		for (ii = 2; ii <= i__3; ++ii) {
		    i__5 = ii - 1;
		    v[i__5].r = 0., v[i__5].i = 0.;
/* L120: */
		}
	    } else {
		rtemp = max(rtemp,smlnum);
		d__1 = 1. / rtemp;
		zdscal_(&nv, &d__1, v, &c__1);
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
		zcopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    zlarfg_(&nr, v, &v[1], &c__1, &tau);
	    if (k > l) {
		i__3 = k + (k - 1) * h_dim1;
		h__[i__3].r = v[0].r, h__[i__3].i = v[0].i;
		i__3 = i__;
		for (ii = k + 1; ii <= i__3; ++ii) {
		    i__5 = ii + (k - 1) * h_dim1;
		    h__[i__5].r = 0., h__[i__5].i = 0.;
/* L140: */
		}
	    }
	    v[0].r = 1., v[0].i = 0.;

/*
             Apply G' from the left to transform the rows of the matrix
             in columns K to I2.
*/

	    i__3 = i2 - k + 1;
	    d_cnjg(&z__1, &tau);
	    zlarfx_("Left", &nr, &i__3, v, &z__1, &h__[k + k * h_dim1], ldh, &
		    work[1]);

/*
             Apply G from the right to transform the columns of the
             matrix in rows I1 to min(K+NR,I).

   Computing MIN
*/
	    i__5 = k + nr;
	    i__3 = min(i__5,i__) - i1 + 1;
	    zlarfx_("Right", &i__3, &nr, v, &tau, &h__[i1 + k * h_dim1], ldh,
		    &work[1]);

	    if (wantz) {

/*              Accumulate transformations in the matrix Z */

		zlarfx_("Right", &nh, &nr, v, &tau, &z__[*ilo + k * z_dim1],
			ldz, &work[1]);
	    }
/* L150: */
	}

/*        Ensure that H(I,I-1) is real. */

	i__2 = i__ + (i__ - 1) * h_dim1;
	temp.r = h__[i__2].r, temp.i = h__[i__2].i;
	if (d_imag(&temp) != 0.) {
	    d__1 = temp.r;
	    d__2 = d_imag(&temp);
	    rtemp = dlapy2_(&d__1, &d__2);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.;
	    z__1.r = temp.r / rtemp, z__1.i = temp.i / rtemp;
	    temp.r = z__1.r, temp.i = z__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		d_cnjg(&z__1, &temp);
		zscal_(&i__2, &z__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    zscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (wantz) {
		zscal_(&nh, &temp, &z__[*ilo + i__ * z_dim1], &c__1);
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

    zlahqr_(&wantt, &wantz, n, &l, &i__, &h__[h_offset], ldh, &w[1], ilo, ihi,
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
    work[1].r = (doublereal) i__1, work[1].i = 0.;
    return 0;

/*     End of ZHSEQR */

} /* zhseqr_ */

/* Subroutine */ int zlabrd_(integer *m, integer *n, integer *nb,
	doublecomplex *a, integer *lda, doublereal *d__, doublereal *e,
	doublecomplex *tauq, doublecomplex *taup, doublecomplex *x, integer *
	ldx, doublecomplex *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    doublecomplex z__1;

    /* Local variables */
    static integer i__;
    static doublecomplex alpha;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *),
	    zlarfg_(integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *), zlacgv_(integer *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLABRD reduces the first NB rows and columns of a complex general
    m by n matrix A to upper or lower real bidiagonal form by a unitary
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by ZGEBRD

    Arguments
    =========

    M       (input) INTEGER
            The number of rows in the matrix A.

    N       (input) INTEGER
            The number of columns in the matrix A.

    NB      (input) INTEGER
            The number of leading rows and columns of A to be reduced.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    D       (output) DOUBLE PRECISION array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    E       (output) DOUBLE PRECISION array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    TAUQ    (output) COMPLEX*16 array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the unitary matrix Q. See Further Details.

    TAUP    (output) COMPLEX*16 array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the unitary matrix P. See Further Details.

    X       (output) COMPLEX*16 array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    LDX     (input) INTEGER
            The leading dimension of the array X. LDX >= max(1,M).

    Y       (output) COMPLEX*16 array, dimension (LDY,NB)
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
    if ((*m <= 0) || (*n <= 0)) {
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:m,i) */

	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + a_dim1], lda,
		     &y[i__ + y_dim1], ldy, &c_b60, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + x_dim1], ldx,
		     &a[i__ * a_dim1 + 1], &c__1, &c_b60, &a[i__ + i__ *
		    a_dim1], &c__1);

/*           Generate reflection Q(i) to annihilate A(i+1:m,i) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    zlarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1, &
		    tauq[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *n) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ + (
			i__ + 1) * a_dim1], lda, &a[i__ + i__ * a_dim1], &
			c__1, &c_b59, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ +
			a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b59, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b60, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &x[i__ +
			x_dim1], ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b59, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[(i__ +
			1) * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b60, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);

/*              Update A(i,i+1:n) */

		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		zlacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &y[i__ + 1 +
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b60, &a[i__ +
			(i__ + 1) * a_dim1], lda);
		zlacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &x[i__ + x_dim1], ldx);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[(i__ +
			1) * a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b60,
			&a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*              Generate reflection P(i) to annihilate A(i,i+2:n) */

		i__2 = i__ + (i__ + 1) * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
			taup[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + (i__ + 1) * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		zgemv_("No transpose", &i__2, &i__3, &c_b60, &a[i__ + 1 + (
			i__ + 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1],
			 lda, &c_b59, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__, &c_b60, &y[i__ + 1
			+ y_dim1], ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b59, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b60, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		zgemv_("No transpose", &i__2, &i__3, &c_b60, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b59, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b60, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		zscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i,i:n) */

	    i__2 = *n - i__ + 1;
	    zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + y_dim1], ldy,
		     &a[i__ + a_dim1], lda, &c_b60, &a[i__ + i__ * a_dim1],
		    lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &x[i__ + x_dim1], ldx);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[i__ *
		    a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b60, &a[i__ +
		    i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*           Generate reflection P(i) to annihilate A(i,i+1:n) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    zlarfg_(&i__2, &alpha, &a[i__ + min(i__3,*n) * a_dim1], lda, &
		    taup[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *m) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		zgemv_("No transpose", &i__2, &i__3, &c_b60, &a[i__ + 1 + i__
			* a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b59, &
			x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &y[i__ +
			y_dim1], ldy, &a[i__ + i__ * a_dim1], lda, &c_b59, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b60, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		zgemv_("No transpose", &i__2, &i__3, &c_b60, &a[i__ * a_dim1
			+ 1], lda, &a[i__ + i__ * a_dim1], lda, &c_b59, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b60, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		zscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);

/*              Update A(i+1:m,i) */

		i__2 = i__ - 1;
		zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 +
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b60, &a[i__ +
			1 + i__ * a_dim1], &c__1);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &x[i__ + 1 +
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b60, &a[
			i__ + 1 + i__ * a_dim1], &c__1);

/*              Generate reflection Q(i) to annihilate A(i+2:m,i) */

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[min(i__3,*m) + i__ * a_dim1], &c__1,
			 &tauq[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ +
			1 + (i__ + 1) * a_dim1], lda, &a[i__ + 1 + i__ *
			a_dim1], &c__1, &c_b59, &y[i__ + 1 + i__ * y_dim1], &
			c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b59, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b60, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		zgemv_("Conjugate transpose", &i__2, &i__, &c_b60, &x[i__ + 1
			+ x_dim1], ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b59, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__, &i__2, &z__1, &a[(i__ + 1)
			 * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b60, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    } else {
		i__2 = *n - i__ + 1;
		zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    }
/* L20: */
	}
    }
    return 0;

/*     End of ZLABRD */

} /* zlabrd_ */

/* Subroutine */ int zlacgv_(integer *n, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, ioff;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLACGV conjugates a complex vector of length N.

    Arguments
    =========

    N       (input) INTEGER
            The length of the vector X.  N >= 0.

    X       (input/output) COMPLEX*16 array, dimension
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
	    d_cnjg(&z__1, &x[i__]);
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
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
	    d_cnjg(&z__1, &x[ioff]);
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	    ioff += *incx;
/* L20: */
	}
    }
    return 0;

/*     End of ZLACGV */

} /* zlacgv_ */

/* Subroutine */ int zlacp2_(char *uplo, integer *m, integer *n, doublereal *
	a, integer *lda, doublecomplex *b, integer *ldb)
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

    ZLACP2 copies all or part of a real two-dimensional matrix A to a
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

    A       (input) DOUBLE PRECISION array, dimension (LDA,N)
            The m by n matrix A.  If UPLO = 'U', only the upper trapezium
            is accessed; if UPLO = 'L', only the lower trapezium is
            accessed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (output) COMPLEX*16 array, dimension (LDB,N)
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
		b[i__3].r = a[i__4], b[i__3].i = 0.;
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
		b[i__3].r = a[i__4], b[i__3].i = 0.;
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
		b[i__3].r = a[i__4], b[i__3].i = 0.;
/* L50: */
	    }
/* L60: */
	}
    }

    return 0;

/*     End of ZLACP2 */

} /* zlacp2_ */

/* Subroutine */ int zlacpy_(char *uplo, integer *m, integer *n,
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb)
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

    ZLACPY copies all or part of a two-dimensional matrix A to another
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

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            The m by n matrix A.  If UPLO = 'U', only the upper trapezium
            is accessed; if UPLO = 'L', only the lower trapezium is
            accessed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (output) COMPLEX*16 array, dimension (LDB,N)
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

/*     End of ZLACPY */

} /* zlacpy_ */

/* Subroutine */ int zlacrm_(integer *m, integer *n, doublecomplex *a,
	integer *lda, doublereal *b, integer *ldb, doublecomplex *c__,
	integer *ldc, doublereal *rwork)
{
    /* System generated locals */
    integer b_dim1, b_offset, a_dim1, a_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLACRM performs a very simple matrix-matrix multiplication:
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

    A       (input) COMPLEX*16 array, dimension (LDA, N)
            A contains the M by N matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >=max(1,M).

    B       (input) DOUBLE PRECISION array, dimension (LDB, N)
            B contains the N by N matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B. LDB >=max(1,N).

    C       (input) COMPLEX*16 array, dimension (LDC, N)
            C contains the M by N matrix C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >=max(1,N).

    RWORK   (workspace) DOUBLE PRECISION array, dimension (2*M*N)

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
    if ((*m == 0) || (*n == 0)) {
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
    dgemm_("N", "N", m, n, n, &c_b1015, &rwork[1], m, &b[b_offset], ldb, &
	    c_b324, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = l + (j - 1) * *m + i__ - 1;
	    c__[i__3].r = rwork[i__4], c__[i__3].i = 0.;
/* L30: */
	}
/* L40: */
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    rwork[(j - 1) * *m + i__] = d_imag(&a[i__ + j * a_dim1]);
/* L50: */
	}
/* L60: */
    }
    dgemm_("N", "N", m, n, n, &c_b1015, &rwork[1], m, &b[b_offset], ldb, &
	    c_b324, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = i__ + j * c_dim1;
	    d__1 = c__[i__4].r;
	    i__5 = l + (j - 1) * *m + i__ - 1;
	    z__1.r = d__1, z__1.i = rwork[i__5];
	    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L70: */
	}
/* L80: */
    }

    return 0;

/*     End of ZLACRM */

} /* zlacrm_ */

/* Double Complex */ VOID zladiv_(doublecomplex * ret_val, doublecomplex *x,
	doublecomplex *y)
{
    /* System generated locals */
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static doublereal zi, zr;
    extern /* Subroutine */ int dladiv_(doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLADIV := X / Y, where X and Y are complex.  The computation of X / Y
    will not overflow on an intermediary step unless the results
    overflows.

    Arguments
    =========

    X       (input) COMPLEX*16
    Y       (input) COMPLEX*16
            The complex scalars X and Y.

    =====================================================================
*/


    d__1 = x->r;
    d__2 = d_imag(x);
    d__3 = y->r;
    d__4 = d_imag(y);
    dladiv_(&d__1, &d__2, &d__3, &d__4, &zr, &zi);
    z__1.r = zr, z__1.i = zi;
     ret_val->r = z__1.r,  ret_val->i = z__1.i;

    return ;

/*     End of ZLADIV */

} /* zladiv_ */

/* Subroutine */ int zlaed0_(integer *qsiz, integer *n, doublereal *d__,
	doublereal *e, doublecomplex *q, integer *ldq, doublecomplex *qstore,
	integer *ldqs, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, qstore_dim1, qstore_offset, i__1, i__2;
    doublereal d__1;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, k, ll, iq, lgn, msd2, smm1, spm1, spm2;
    static doublereal temp;
    static integer curr, iperm;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *,
	    doublereal *, integer *);
    static integer indxq, iwrem, iqptr, tlvls;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zlaed7_(integer *, integer *,
	    integer *, integer *, integer *, integer *, doublereal *,
	    doublecomplex *, integer *, doublereal *, integer *, doublereal *,
	     integer *, integer *, integer *, integer *, integer *,
	    doublereal *, doublecomplex *, doublereal *, integer *, integer *)
	    ;
    static integer igivcl;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlacrm_(integer *, integer *, doublecomplex *,
	     integer *, doublereal *, integer *, doublecomplex *, integer *,
	    doublereal *);
    static integer igivnm, submat, curprb, subpbs, igivpt;
    extern /* Subroutine */ int dsteqr_(char *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *);
    static integer curlvl, matsiz, iprmpt, smlsiz;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    Using the divide and conquer method, ZLAED0 computes all eigenvalues
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

    D      (input/output) DOUBLE PRECISION array, dimension (N)
           On entry, the diagonal elements of the tridiagonal matrix.
           On exit, the eigenvalues in ascending order.

    E      (input/output) DOUBLE PRECISION array, dimension (N-1)
           On entry, the off-diagonal elements of the tridiagonal matrix.
           On exit, E has been destroyed.

    Q      (input/output) COMPLEX*16 array, dimension (LDQ,N)
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

    RWORK  (workspace) DOUBLE PRECISION array,
                                 dimension (1 + 3*N + 2*N*lg N + 3*N**2)
                          ( lg( N ) = smallest integer k
                                      such that 2^k >= N )

    QSTORE (workspace) COMPLEX*16 array, dimension (LDQS, N)
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
	xerbla_("ZLAED0", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "ZLAED0", " ", &c__0, &c__0, &c__0, &c__0, (
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
	    iwork[((j) << (1)) - 1] = iwork[j] / 2;
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
	d__[smm1] -= (d__1 = e[smm1], abs(d__1));
	d__[submat] -= (d__1 = e[smm1], abs(d__1));
/* L40: */
    }

    indxq = ((*n) << (2)) + 3;

/*
       Set up workspaces for eigenvalues only/accumulate new vectors
       routine
*/

    temp = log((doublereal) (*n)) / log(2.);
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
    iq = igivnm + ((*n) << (1)) * lgn;
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
	dsteqr_("I", &matsiz, &d__[submat], &e[submat], &rwork[ll], &matsiz, &
		rwork[1], info);
	zlacrm_(qsiz, &matsiz, &q[submat * q_dim1 + 1], ldq, &rwork[ll], &
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
       into an eigensystem of size MATSIZ.  ZLAED7 handles the case
       when the eigenvectors of a full or band Hermitian matrix (which
       was reduced to tridiagonal form) are desired.

       I am free to use Q as a valuable working space until Loop 150.
*/

	    zlaed7_(&matsiz, &msd2, qsiz, &tlvls, &curlvl, &curprb, &d__[
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
	zcopy_(qsiz, &qstore[j * qstore_dim1 + 1], &c__1, &q[i__ * q_dim1 + 1]
		, &c__1);
/* L100: */
    }
    dcopy_(n, &rwork[1], &c__1, &d__[1], &c__1);

    return 0;

/*     End of ZLAED0 */

} /* zlaed0_ */

/* Subroutine */ int zlaed7_(integer *n, integer *cutpnt, integer *qsiz,
	integer *tlvls, integer *curlvl, integer *curpbm, doublereal *d__,
	doublecomplex *q, integer *ldq, doublereal *rho, integer *indxq,
	doublereal *qstore, integer *qptr, integer *prmptr, integer *perm,
	integer *givptr, integer *givcol, doublereal *givnum, doublecomplex *
	work, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, k, n1, n2, iq, iw, iz, ptr, ind1, ind2, indx, curr,
	    indxc, indxp;
    extern /* Subroutine */ int dlaed9_(integer *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, doublereal *, integer *, integer *),
	    zlaed8_(integer *, integer *, integer *, doublecomplex *, integer
	    *, doublereal *, doublereal *, integer *, doublereal *,
	    doublereal *, doublecomplex *, integer *, doublereal *, integer *,
	     integer *, integer *, integer *, integer *, integer *,
	    doublereal *, integer *), dlaeda_(integer *, integer *, integer *,
	     integer *, integer *, integer *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     integer *);
    static integer idlmda;
    extern /* Subroutine */ int dlamrg_(integer *, integer *, doublereal *,
	    integer *, integer *, integer *), xerbla_(char *, integer *), zlacrm_(integer *, integer *, doublecomplex *, integer *,
	     doublereal *, integer *, doublecomplex *, integer *, doublereal *
	    );
    static integer coltyp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLAED7 computes the updated eigensystem of a diagonal
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
          performed by the routine DLAED2.

          The second stage consists of calculating the updated
          eigenvalues. This is done by finding the roots of the secular
          equation via the routine DLAED4 (as called by SLAED3).
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

    D      (input/output) DOUBLE PRECISION array, dimension (N)
           On entry, the eigenvalues of the rank-1-perturbed matrix.
           On exit, the eigenvalues of the repaired matrix.

    Q      (input/output) COMPLEX*16 array, dimension (LDQ,N)
           On entry, the eigenvectors of the rank-1-perturbed matrix.
           On exit, the eigenvectors of the repaired tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    RHO    (input) DOUBLE PRECISION
           Contains the subdiagonal element used to create the rank-1
           modification.

    INDXQ  (output) INTEGER array, dimension (N)
           This contains the permutation which will reintegrate the
           subproblem just solved back into sorted order,
           ie. D( INDXQ( I = 1, N ) ) will be in ascending order.

    IWORK  (workspace) INTEGER array, dimension (4*N)

    RWORK  (workspace) DOUBLE PRECISION array,
                                   dimension (3*N+2*QSIZ*N)

    WORK   (workspace) COMPLEX*16 array, dimension (QSIZ*N)

    QSTORE (input/output) DOUBLE PRECISION array, dimension (N**2+1)
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

    GIVNUM (input) DOUBLE PRECISION array, dimension (2, N lg N)
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
    } else if ((min(1,*n) > *cutpnt) || (*n < *cutpnt)) {
	*info = -2;
    } else if (*qsiz < *n) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLAED7", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*
       The following values are for bookkeeping purposes only.  They are
       integer pointers which indicate the portion of the workspace
       used by a particular array in DLAED2 and SLAED3.
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
    dlaeda_(n, tlvls, curlvl, curpbm, &prmptr[1], &perm[1], &givptr[1], &
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

    zlaed8_(&k, n, qsiz, &q[q_offset], ldq, &d__[1], rho, cutpnt, &rwork[iz],
	    &rwork[idlmda], &work[1], qsiz, &rwork[iw], &iwork[indxp], &iwork[
	    indx], &indxq[1], &perm[prmptr[curr]], &givptr[curr + 1], &givcol[
	    ((givptr[curr]) << (1)) + 1], &givnum[((givptr[curr]) << (1)) + 1]
	    , info);
    prmptr[curr + 1] = prmptr[curr] + *n;
    givptr[curr + 1] += givptr[curr];

/*     Solve Secular Equation. */

    if (k != 0) {
	dlaed9_(&k, &c__1, &k, n, &d__[1], &rwork[iq], &k, rho, &rwork[idlmda]
		, &rwork[iw], &qstore[qptr[curr]], &k, info);
	zlacrm_(qsiz, &k, &work[1], qsiz, &qstore[qptr[curr]], &k, &q[
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
	dlamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &indxq[1]);
    } else {
	qptr[curr + 1] = qptr[curr];
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    indxq[i__] = i__;
/* L20: */
	}
    }

    return 0;

/*     End of ZLAED7 */

} /* zlaed7_ */

/* Subroutine */ int zlaed8_(integer *k, integer *n, integer *qsiz,
	doublecomplex *q, integer *ldq, doublereal *d__, doublereal *rho,
	integer *cutpnt, doublereal *z__, doublereal *dlamda, doublecomplex *
	q2, integer *ldq2, doublereal *w, integer *indxp, integer *indx,
	integer *indxq, integer *perm, integer *givptr, integer *givcol,
	doublereal *givnum, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, q2_dim1, q2_offset, i__1;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal c__;
    static integer i__, j;
    static doublereal s, t;
    static integer k2, n1, n2, jp, n1p1;
    static doublereal eps, tau, tol;
    static integer jlam, imax, jmax;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *,
	    integer *), dcopy_(integer *, doublereal *, integer *, doublereal
	    *, integer *), zdrot_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublereal *, doublereal *), zcopy_(
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *)
	    ;

    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dlamrg_(integer *, integer *, doublereal *,
	    integer *, integer *, integer *), xerbla_(char *, integer *), zlacpy_(char *, integer *, integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
       Courant Institute, NAG Ltd., and Rice University
       September 30, 1994


    Purpose
    =======

    ZLAED8 merges the two sets of eigenvalues together into a single
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

    Q      (input/output) COMPLEX*16 array, dimension (LDQ,N)
           On entry, Q contains the eigenvectors of the partially solved
           system which has been previously updated in matrix
           multiplies with other partially solved eigensystems.
           On exit, Q contains the trailing (N-K) updated eigenvectors
           (those which were deflated) in its last N-K columns.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max( 1, N ).

    D      (input/output) DOUBLE PRECISION array, dimension (N)
           On entry, D contains the eigenvalues of the two submatrices to
           be combined.  On exit, D contains the trailing (N-K) updated
           eigenvalues (those which were deflated) sorted into increasing
           order.

    RHO    (input/output) DOUBLE PRECISION
           Contains the off diagonal element associated with the rank-1
           cut which originally split the two submatrices which are now
           being recombined. RHO is modified during the computation to
           the value required by DLAED3.

    CUTPNT (input) INTEGER
           Contains the location of the last eigenvalue in the leading
           sub-matrix.  MIN(1,N) <= CUTPNT <= N.

    Z      (input) DOUBLE PRECISION array, dimension (N)
           On input this vector contains the updating vector (the last
           row of the first sub-eigenvector matrix and the first row of
           the second sub-eigenvector matrix).  The contents of Z are
           destroyed during the updating process.

    DLAMDA (output) DOUBLE PRECISION array, dimension (N)
           Contains a copy of the first K eigenvalues which will be used
           by DLAED3 to form the secular equation.

    Q2     (output) COMPLEX*16 array, dimension (LDQ2,N)
           If ICOMPQ = 0, Q2 is not referenced.  Otherwise,
           Contains a copy of the first K eigenvectors which will be used
           by DLAED7 in a matrix multiply (DGEMM) to update the new
           eigenvectors.

    LDQ2   (input) INTEGER
           The leading dimension of the array Q2.  LDQ2 >= max( 1, N ).

    W      (output) DOUBLE PRECISION array, dimension (N)
           This will hold the first k values of the final
           deflation-altered z-vector and will be passed to DLAED3.

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

    GIVNUM (output) DOUBLE PRECISION array, dimension (2, N)
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
    } else if ((*cutpnt < min(1,*n)) || (*cutpnt > *n)) {
	*info = -8;
    } else if (*ldq2 < max(1,*n)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLAED8", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    n1 = *cutpnt;
    n2 = *n - n1;
    n1p1 = n1 + 1;

    if (*rho < 0.) {
	dscal_(&n2, &c_b1294, &z__[n1p1], &c__1);
    }

/*     Normalize z so that norm(z) = 1 */

    t = 1. / sqrt(2.);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	indx[j] = j;
/* L10: */
    }
    dscal_(n, &t, &z__[1], &c__1);
    *rho = (d__1 = *rho * 2., abs(d__1));

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
    dlamrg_(&n1, &n2, &dlamda[1], &c__1, &c__1, &indx[1]);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = dlamda[indx[i__]];
	z__[i__] = w[indx[i__]];
/* L40: */
    }

/*     Calculate the allowable deflation tolerance */

    imax = idamax_(n, &z__[1], &c__1);
    jmax = idamax_(n, &d__[1], &c__1);
    eps = EPSILON;
    tol = eps * 8. * (d__1 = d__[jmax], abs(d__1));

/*
       If the rank-1 modifier is small enough, no more needs to be done
       -- except to reorganize Q so that its columns correspond with the
       elements in D.
*/

    if (*rho * (d__1 = z__[imax], abs(d__1)) <= tol) {
	*k = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    perm[j] = indxq[indx[j]];
	    zcopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1]
		    , &c__1);
/* L50: */
	}
	zlacpy_("A", qsiz, n, &q2[q2_dim1 + 1], ldq2, &q[q_dim1 + 1], ldq);
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
	if (*rho * (d__1 = z__[j], abs(d__1)) <= tol) {

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
    if (*rho * (d__1 = z__[j], abs(d__1)) <= tol) {

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

	tau = dlapy2_(&c__, &s);
	t = d__[j] - d__[jlam];
	c__ /= tau;
	s = -s / tau;
	if ((d__1 = t * c__ * s, abs(d__1)) <= tol) {

/*           Deflation is possible. */

	    z__[j] = tau;
	    z__[jlam] = 0.;

/*           Record the appropriate Givens rotation */

	    ++(*givptr);
	    givcol[((*givptr) << (1)) + 1] = indxq[indx[jlam]];
	    givcol[((*givptr) << (1)) + 2] = indxq[indx[j]];
	    givnum[((*givptr) << (1)) + 1] = c__;
	    givnum[((*givptr) << (1)) + 2] = s;
	    zdrot_(qsiz, &q[indxq[indx[jlam]] * q_dim1 + 1], &c__1, &q[indxq[
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
	zcopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1], &
		c__1);
/* L110: */
    }

/*
       The deflated eigenvalues and their corresponding vectors go back
       into the last N - K slots of D and Q respectively.
*/

    if (*k < *n) {
	i__1 = *n - *k;
	dcopy_(&i__1, &dlamda[*k + 1], &c__1, &d__[*k + 1], &c__1);
	i__1 = *n - *k;
	zlacpy_("A", qsiz, &i__1, &q2[(*k + 1) * q2_dim1 + 1], ldq2, &q[(*k +
		1) * q_dim1 + 1], ldq);
    }

    return 0;

/*     End of ZLAED8 */

} /* zlaed8_ */

/* Subroutine */ int zlahqr_(logical *wantt, logical *wantz, integer *n,
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh,
	doublecomplex *w, integer *iloz, integer *ihiz, doublecomplex *z__,
	integer *ldz, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void z_sqrt(doublecomplex *, doublecomplex *), d_cnjg(doublecomplex *,
	    doublecomplex *);
    double z_abs(doublecomplex *);

    /* Local variables */
    static integer i__, j, k, l, m;
    static doublereal s;
    static doublecomplex t, u, v[2], x, y;
    static integer i1, i2;
    static doublecomplex t1;
    static doublereal t2;
    static doublecomplex v2;
    static doublereal h10;
    static doublecomplex h11;
    static doublereal h21;
    static doublecomplex h22;
    static integer nh, nz;
    static doublecomplex h11s;
    static integer itn, its;
    static doublereal ulp;
    static doublecomplex sum;
    static doublereal tst1;
    static doublecomplex temp;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *);
    static doublereal rtemp, rwork[1];
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);

    extern /* Subroutine */ int zlarfg_(integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *);
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *,
	     doublecomplex *);
    extern doublereal zlanhs_(char *, integer *, doublecomplex *, integer *,
	    doublereal *);
    static doublereal smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLAHQR is an auxiliary routine called by ZHSEQR to update the
    eigenvalues and Schur decomposition already computed by ZHSEQR, by
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
            ZLAHQR works primarily with the Hessenberg submatrix in rows
            and columns ILO to IHI, but applies transformations to all of
            H if WANTT is .TRUE..
            1 <= ILO <= max(1,IHI); IHI <= N.

    H       (input/output) COMPLEX*16 array, dimension (LDH,N)
            On entry, the upper Hessenberg matrix H.
            On exit, if WANTT is .TRUE., H is upper triangular in rows
            and columns ILO:IHI, with any 2-by-2 diagonal blocks in
            standard form. If WANTT is .FALSE., the contents of H are
            unspecified on exit.

    LDH     (input) INTEGER
            The leading dimension of the array H. LDH >= max(1,N).

    W       (output) COMPLEX*16 array, dimension (N)
            The computed eigenvalues ILO to IHI are stored in the
            corresponding elements of W. If WANTT is .TRUE., the
            eigenvalues are stored in the same order as on the diagonal
            of the Schur form returned in H, with W(i) = H(i,i).

    ILOZ    (input) INTEGER
    IHIZ    (input) INTEGER
            Specify the rows of Z to which transformations must be
            applied if WANTZ is .TRUE..
            1 <= ILOZ <= ILO; IHI <= IHIZ <= N.

    Z       (input/output) COMPLEX*16 array, dimension (LDZ,N)
            If WANTZ is .TRUE., on entry Z must contain the current
            matrix Z of transformations accumulated by ZHSEQR, and on
            exit Z has been updated; transformations are applied only to
            the submatrix Z(ILOZ:IHIZ,ILO:IHI).
            If WANTZ is .FALSE., Z is not referenced.

    LDZ     (input) INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            > 0: if INFO = i, ZLAHQR failed to compute all the
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

    ulp = PRECISION;
    smlnum = SAFEMINIMUM / ulp;

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
	    tst1 = (d__1 = h__[i__3].r, abs(d__1)) + (d__2 = d_imag(&h__[k -
		    1 + (k - 1) * h_dim1]), abs(d__2)) + ((d__3 = h__[i__4].r,
		     abs(d__3)) + (d__4 = d_imag(&h__[k + k * h_dim1]), abs(
		    d__4)));
	    if (tst1 == 0.) {
		i__3 = i__ - l + 1;
		tst1 = zlanhs_("1", &i__3, &h__[l + l * h_dim1], ldh, rwork);
	    }
	    i__3 = k + (k - 1) * h_dim1;
/* Computing MAX */
	    d__2 = ulp * tst1;
	    if ((d__1 = h__[i__3].r, abs(d__1)) <= max(d__2,smlnum)) {
		goto L30;
	    }
/* L20: */
	}
L30:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible */

	    i__2 = l + (l - 1) * h_dim1;
	    h__[i__2].r = 0., h__[i__2].i = 0.;
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

	if ((its == 10) || (its == 20)) {

/*           Exceptional shift. */

	    i__2 = i__ + (i__ - 1) * h_dim1;
	    s = (d__1 = h__[i__2].r, abs(d__1)) * .75;
	    i__2 = i__ + i__ * h_dim1;
	    z__1.r = s + h__[i__2].r, z__1.i = h__[i__2].i;
	    t.r = z__1.r, t.i = z__1.i;
	} else {

/*           Wilkinson's shift. */

	    i__2 = i__ + i__ * h_dim1;
	    t.r = h__[i__2].r, t.i = h__[i__2].i;
	    i__2 = i__ - 1 + i__ * h_dim1;
	    i__3 = i__ + (i__ - 1) * h_dim1;
	    d__1 = h__[i__3].r;
	    z__1.r = d__1 * h__[i__2].r, z__1.i = d__1 * h__[i__2].i;
	    u.r = z__1.r, u.i = z__1.i;
	    if ((u.r != 0.) || (u.i != 0.)) {
		i__2 = i__ - 1 + (i__ - 1) * h_dim1;
		z__2.r = h__[i__2].r - t.r, z__2.i = h__[i__2].i - t.i;
		z__1.r = z__2.r * .5, z__1.i = z__2.i * .5;
		x.r = z__1.r, x.i = z__1.i;
		z__3.r = x.r * x.r - x.i * x.i, z__3.i = x.r * x.i + x.i *
			x.r;
		z__2.r = z__3.r + u.r, z__2.i = z__3.i + u.i;
		z_sqrt(&z__1, &z__2);
		y.r = z__1.r, y.i = z__1.i;
		if (x.r * y.r + d_imag(&x) * d_imag(&y) < 0.) {
		    z__1.r = -y.r, z__1.i = -y.i;
		    y.r = z__1.r, y.i = z__1.i;
		}
		z__3.r = x.r + y.r, z__3.i = x.i + y.i;
		zladiv_(&z__2, &u, &z__3);
		z__1.r = t.r - z__2.r, z__1.i = t.i - z__2.i;
		t.r = z__1.r, t.i = z__1.i;
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
	    z__1.r = h11.r - t.r, z__1.i = h11.i - t.i;
	    h11s.r = z__1.r, h11s.i = z__1.i;
	    i__3 = m + 1 + m * h_dim1;
	    h21 = h__[i__3].r;
	    s = (d__1 = h11s.r, abs(d__1)) + (d__2 = d_imag(&h11s), abs(d__2))
		     + abs(h21);
	    z__1.r = h11s.r / s, z__1.i = h11s.i / s;
	    h11s.r = z__1.r, h11s.i = z__1.i;
	    h21 /= s;
	    v[0].r = h11s.r, v[0].i = h11s.i;
	    v[1].r = h21, v[1].i = 0.;
	    i__3 = m + (m - 1) * h_dim1;
	    h10 = h__[i__3].r;
	    tst1 = ((d__1 = h11s.r, abs(d__1)) + (d__2 = d_imag(&h11s), abs(
		    d__2))) * ((d__3 = h11.r, abs(d__3)) + (d__4 = d_imag(&
		    h11), abs(d__4)) + ((d__5 = h22.r, abs(d__5)) + (d__6 =
		    d_imag(&h22), abs(d__6))));
	    if ((d__1 = h10 * h21, abs(d__1)) <= ulp * tst1) {
		goto L50;
	    }
/* L40: */
	}
	i__2 = l + l * h_dim1;
	h11.r = h__[i__2].r, h11.i = h__[i__2].i;
	i__2 = l + 1 + (l + 1) * h_dim1;
	h22.r = h__[i__2].r, h22.i = h__[i__2].i;
	z__1.r = h11.r - t.r, z__1.i = h11.i - t.i;
	h11s.r = z__1.r, h11s.i = z__1.i;
	i__2 = l + 1 + l * h_dim1;
	h21 = h__[i__2].r;
	s = (d__1 = h11s.r, abs(d__1)) + (d__2 = d_imag(&h11s), abs(d__2)) +
		abs(h21);
	z__1.r = h11s.r / s, z__1.i = h11s.i / s;
	h11s.r = z__1.r, h11s.i = z__1.i;
	h21 /= s;
	v[0].r = h11s.r, v[0].i = h11s.i;
	v[1].r = h21, v[1].i = 0.;
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

             V(2) is always real before the call to ZLARFG, and hence
             after the call T2 ( = T1*V(2) ) is also real.
*/

	    if (k > m) {
		zcopy_(&c__2, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    zlarfg_(&c__2, v, &v[1], &c__1, &t1);
	    if (k > m) {
		i__3 = k + (k - 1) * h_dim1;
		h__[i__3].r = v[0].r, h__[i__3].i = v[0].i;
		i__3 = k + 1 + (k - 1) * h_dim1;
		h__[i__3].r = 0., h__[i__3].i = 0.;
	    }
	    v2.r = v[1].r, v2.i = v[1].i;
	    z__1.r = t1.r * v2.r - t1.i * v2.i, z__1.i = t1.r * v2.i + t1.i *
		    v2.r;
	    t2 = z__1.r;

/*
             Apply G from the left to transform the rows of the matrix
             in columns K to I2.
*/

	    i__3 = i2;
	    for (j = k; j <= i__3; ++j) {
		d_cnjg(&z__3, &t1);
		i__4 = k + j * h_dim1;
		z__2.r = z__3.r * h__[i__4].r - z__3.i * h__[i__4].i, z__2.i =
			 z__3.r * h__[i__4].i + z__3.i * h__[i__4].r;
		i__5 = k + 1 + j * h_dim1;
		z__4.r = t2 * h__[i__5].r, z__4.i = t2 * h__[i__5].i;
		z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		sum.r = z__1.r, sum.i = z__1.i;
		i__4 = k + j * h_dim1;
		i__5 = k + j * h_dim1;
		z__1.r = h__[i__5].r - sum.r, z__1.i = h__[i__5].i - sum.i;
		h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
		i__4 = k + 1 + j * h_dim1;
		i__5 = k + 1 + j * h_dim1;
		z__2.r = sum.r * v2.r - sum.i * v2.i, z__2.i = sum.r * v2.i +
			sum.i * v2.r;
		z__1.r = h__[i__5].r - z__2.r, z__1.i = h__[i__5].i - z__2.i;
		h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
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
		z__2.r = t1.r * h__[i__4].r - t1.i * h__[i__4].i, z__2.i =
			t1.r * h__[i__4].i + t1.i * h__[i__4].r;
		i__5 = j + (k + 1) * h_dim1;
		z__3.r = t2 * h__[i__5].r, z__3.i = t2 * h__[i__5].i;
		z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		sum.r = z__1.r, sum.i = z__1.i;
		i__4 = j + k * h_dim1;
		i__5 = j + k * h_dim1;
		z__1.r = h__[i__5].r - sum.r, z__1.i = h__[i__5].i - sum.i;
		h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
		i__4 = j + (k + 1) * h_dim1;
		i__5 = j + (k + 1) * h_dim1;
		d_cnjg(&z__3, &v2);
		z__2.r = sum.r * z__3.r - sum.i * z__3.i, z__2.i = sum.r *
			z__3.i + sum.i * z__3.r;
		z__1.r = h__[i__5].r - z__2.r, z__1.i = h__[i__5].i - z__2.i;
		h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
/* L70: */
	    }

	    if (*wantz) {

/*              Accumulate transformations in the matrix Z */

		i__3 = *ihiz;
		for (j = *iloz; j <= i__3; ++j) {
		    i__4 = j + k * z_dim1;
		    z__2.r = t1.r * z__[i__4].r - t1.i * z__[i__4].i, z__2.i =
			     t1.r * z__[i__4].i + t1.i * z__[i__4].r;
		    i__5 = j + (k + 1) * z_dim1;
		    z__3.r = t2 * z__[i__5].r, z__3.i = t2 * z__[i__5].i;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    sum.r = z__1.r, sum.i = z__1.i;
		    i__4 = j + k * z_dim1;
		    i__5 = j + k * z_dim1;
		    z__1.r = z__[i__5].r - sum.r, z__1.i = z__[i__5].i -
			    sum.i;
		    z__[i__4].r = z__1.r, z__[i__4].i = z__1.i;
		    i__4 = j + (k + 1) * z_dim1;
		    i__5 = j + (k + 1) * z_dim1;
		    d_cnjg(&z__3, &v2);
		    z__2.r = sum.r * z__3.r - sum.i * z__3.i, z__2.i = sum.r *
			     z__3.i + sum.i * z__3.r;
		    z__1.r = z__[i__5].r - z__2.r, z__1.i = z__[i__5].i -
			    z__2.i;
		    z__[i__4].r = z__1.r, z__[i__4].i = z__1.i;
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

		z__1.r = 1. - t1.r, z__1.i = 0. - t1.i;
		temp.r = z__1.r, temp.i = z__1.i;
		d__1 = z_abs(&temp);
		z__1.r = temp.r / d__1, z__1.i = temp.i / d__1;
		temp.r = z__1.r, temp.i = z__1.i;
		i__3 = m + 1 + m * h_dim1;
		i__4 = m + 1 + m * h_dim1;
		d_cnjg(&z__2, &temp);
		z__1.r = h__[i__4].r * z__2.r - h__[i__4].i * z__2.i, z__1.i =
			 h__[i__4].r * z__2.i + h__[i__4].i * z__2.r;
		h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
		if (m + 2 <= i__) {
		    i__3 = m + 2 + (m + 1) * h_dim1;
		    i__4 = m + 2 + (m + 1) * h_dim1;
		    z__1.r = h__[i__4].r * temp.r - h__[i__4].i * temp.i,
			    z__1.i = h__[i__4].r * temp.i + h__[i__4].i *
			    temp.r;
		    h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
		}
		i__3 = i__;
		for (j = m; j <= i__3; ++j) {
		    if (j != m + 1) {
			if (i2 > j) {
			    i__4 = i2 - j;
			    zscal_(&i__4, &temp, &h__[j + (j + 1) * h_dim1],
				    ldh);
			}
			i__4 = j - i1;
			d_cnjg(&z__1, &temp);
			zscal_(&i__4, &z__1, &h__[i1 + j * h_dim1], &c__1);
			if (*wantz) {
			    d_cnjg(&z__1, &temp);
			    zscal_(&nz, &z__1, &z__[*iloz + j * z_dim1], &
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
	if (d_imag(&temp) != 0.) {
	    rtemp = z_abs(&temp);
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    h__[i__2].r = rtemp, h__[i__2].i = 0.;
	    z__1.r = temp.r / rtemp, z__1.i = temp.i / rtemp;
	    temp.r = z__1.r, temp.i = z__1.i;
	    if (i2 > i__) {
		i__2 = i2 - i__;
		d_cnjg(&z__1, &temp);
		zscal_(&i__2, &z__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__2 = i__ - i1;
	    zscal_(&i__2, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (*wantz) {
		zscal_(&nz, &temp, &z__[*iloz + i__ * z_dim1], &c__1);
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

/*     End of ZLAHQR */

} /* zlahqr_ */

/* Subroutine */ int zlahrd_(integer *n, integer *k, integer *nb,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *t,
	integer *ldt, doublecomplex *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    doublecomplex z__1;

    /* Local variables */
    static integer i__;
    static doublecomplex ei;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *),
	    zcopy_(integer *, doublecomplex *, integer *, doublecomplex *,
	    integer *), zaxpy_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, integer *), ztrmv_(char *, char *,
	    char *, integer *, doublecomplex *, integer *, doublecomplex *,
	    integer *), zlarfg_(integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *),
	    zlacgv_(integer *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLAHRD reduces the first NB columns of a complex general n-by-(n-k+1)
    matrix A so that elements below the k-th subdiagonal are zero. The
    reduction is performed by a unitary similarity transformation
    Q' * A * Q. The routine returns the matrices V and T which determine
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V * T.

    This is an auxiliary routine called by ZGEHRD.

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix A.

    K       (input) INTEGER
            The offset for the reduction. Elements below the k-th
            subdiagonal in the first NB columns are reduced to zero.

    NB      (input) INTEGER
            The number of columns to be reduced.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N-K+1)
            On entry, the n-by-(n-k+1) general matrix A.
            On exit, the elements on and above the k-th subdiagonal in
            the first NB columns are overwritten with the corresponding
            elements of the reduced matrix; the elements below the k-th
            subdiagonal, with the array TAU, represent the matrix Q as a
            product of elementary reflectors. The other columns of A are
            unchanged. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) COMPLEX*16 array, dimension (NB)
            The scalar factors of the elementary reflectors. See Further
            Details.

    T       (output) COMPLEX*16 array, dimension (LDT,NB)
            The upper triangular matrix T.

    LDT     (input) INTEGER
            The leading dimension of the array T.  LDT >= NB.

    Y       (output) COMPLEX*16 array, dimension (LDY,NB)
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
	    zlacgv_(&i__2, &a[*k + i__ - 1 + a_dim1], lda);
	    i__2 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", n, &i__2, &z__1, &y[y_offset], ldy, &a[*k
		    + i__ - 1 + a_dim1], lda, &c_b60, &a[i__ * a_dim1 + 1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[*k + i__ - 1 + a_dim1], lda);

/*
             Apply I - V * T' * V' to this column (call it b) from the
             left, using the last column of T as workspace

             Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
                      ( V2 )             ( b2 )

             where V1 is unit lower triangular

             w := V1' * b1
*/

	    i__2 = i__ - 1;
	    zcopy_(&i__2, &a[*k + 1 + i__ * a_dim1], &c__1, &t[*nb * t_dim1 +
		    1], &c__1);
	    i__2 = i__ - 1;
	    ztrmv_("Lower", "Conjugate transpose", "Unit", &i__2, &a[*k + 1 +
		    a_dim1], lda, &t[*nb * t_dim1 + 1], &c__1);

/*           w := w + V2'*b2 */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[*k + i__ +
		    a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b60,
		    &t[*nb * t_dim1 + 1], &c__1);

/*           w := T'*w */

	    i__2 = i__ - 1;
	    ztrmv_("Upper", "Conjugate transpose", "Non-unit", &i__2, &t[
		    t_offset], ldt, &t[*nb * t_dim1 + 1], &c__1);

/*           b2 := b2 - V2*w */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[*k + i__ + a_dim1],
		     lda, &t[*nb * t_dim1 + 1], &c__1, &c_b60, &a[*k + i__ +
		    i__ * a_dim1], &c__1);

/*           b1 := b1 - V1*w */

	    i__2 = i__ - 1;
	    ztrmv_("Lower", "No transpose", "Unit", &i__2, &a[*k + 1 + a_dim1]
		    , lda, &t[*nb * t_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zaxpy_(&i__2, &z__1, &t[*nb * t_dim1 + 1], &c__1, &a[*k + 1 + i__
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
	zlarfg_(&i__2, &ei, &a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[i__])
		;
	i__2 = *k + i__ + i__ * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;

/*        Compute  Y(1:n,i) */

	i__2 = *n - *k - i__ + 1;
	zgemv_("No transpose", n, &i__2, &c_b60, &a[(i__ + 1) * a_dim1 + 1],
		lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b59, &y[i__ *
		y_dim1 + 1], &c__1);
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[*k + i__ +
		a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b59, &t[
		i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	z__1.r = -1., z__1.i = -0.;
	zgemv_("No transpose", n, &i__2, &z__1, &y[y_offset], ldy, &t[i__ *
		t_dim1 + 1], &c__1, &c_b60, &y[i__ * y_dim1 + 1], &c__1);
	zscal_(n, &tau[i__], &y[i__ * y_dim1 + 1], &c__1);

/*        Compute T(1:i,i) */

	i__2 = i__ - 1;
	i__3 = i__;
	z__1.r = -tau[i__3].r, z__1.i = -tau[i__3].i;
	zscal_(&i__2, &z__1, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	ztrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[t_offset], ldt,
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

/*     End of ZLAHRD */

} /* zlahrd_ */

/* Subroutine */ int zlals0_(integer *icompq, integer *nl, integer *nr,
	integer *sqre, integer *nrhs, doublecomplex *b, integer *ldb,
	doublecomplex *bx, integer *ldbx, integer *perm, integer *givptr,
	integer *givcol, integer *ldgcol, doublereal *givnum, integer *ldgnum,
	 doublereal *poles, doublereal *difl, doublereal *difr, doublereal *
	z__, integer *k, doublereal *c__, doublereal *s, doublereal *rwork,
	integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, difr_dim1, difr_offset, givnum_dim1,
	    givnum_offset, poles_dim1, poles_offset, b_dim1, b_offset,
	    bx_dim1, bx_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static integer i__, j, m, n;
    static doublereal dj;
    static integer nlp1, jcol;
    static doublereal temp;
    static integer jrow;
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    static doublereal diflj, difrj, dsigj;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, doublereal *, integer *), zdrot_(integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublereal *, doublereal *);
    extern doublereal dlamc3_(doublereal *, doublereal *);
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), xerbla_(char *, integer *);
    static doublereal dsigjp;
    extern /* Subroutine */ int zdscal_(integer *, doublereal *,
	    doublecomplex *, integer *), zlascl_(char *, integer *, integer *,
	     doublereal *, doublereal *, integer *, integer *, doublecomplex *
	    , integer *, integer *), zlacpy_(char *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       December 1, 1999


    Purpose
    =======

    ZLALS0 applies back the multiplying factors of either the left or the
    right singular vector matrix of a diagonal matrix appended by a row
    to the right hand side matrix B in solving the least squares problem
    using the divide-and-conquer SVD approach.

    For the left singular vector matrix, three types of orthogonal
    matrices are involved:

    (1L) Givens rotations: the number of such rotations is GIVPTR; the
         pairs of columns/rows they were applied to are stored in GIVCOL;
         and the C- and S-values of these rotations are stored in GIVNUM.

    (2L) Permutation. The (NL+1)-st row of B is to be moved to the first
         row, and for J=2:N, PERM(J)-th row of B is to be moved to the
         J-th row.

    (3L) The left singular vector matrix of the remaining matrix.

    For the right singular vector matrix, four types of orthogonal
    matrices are involved:

    (1R) The right singular vector matrix of the remaining matrix.

    (2R) If SQRE = 1, one extra Givens rotation to generate the right
         null space.

    (3R) The inverse transformation of (2L).

    (4R) The inverse transformation of (1L).

    Arguments
    =========

    ICOMPQ (input) INTEGER
           Specifies whether singular vectors are to be computed in
           factored form:
           = 0: Left singular vector matrix.
           = 1: Right singular vector matrix.

    NL     (input) INTEGER
           The row dimension of the upper block. NL >= 1.

    NR     (input) INTEGER
           The row dimension of the lower block. NR >= 1.

    SQRE   (input) INTEGER
           = 0: the lower block is an NR-by-NR square matrix.
           = 1: the lower block is an NR-by-(NR+1) rectangular matrix.

           The bidiagonal matrix has row dimension N = NL + NR + 1,
           and column dimension M = N + SQRE.

    NRHS   (input) INTEGER
           The number of columns of B and BX. NRHS must be at least 1.

    B      (input/output) COMPLEX*16 array, dimension ( LDB, NRHS )
           On input, B contains the right hand sides of the least
           squares problem in rows 1 through M. On output, B contains
           the solution X in rows 1 through N.

    LDB    (input) INTEGER
           The leading dimension of B. LDB must be at least
           max(1,MAX( M, N ) ).

    BX     (workspace) COMPLEX*16 array, dimension ( LDBX, NRHS )

    LDBX   (input) INTEGER
           The leading dimension of BX.

    PERM   (input) INTEGER array, dimension ( N )
           The permutations (from deflation and sorting) applied
           to the two blocks.

    GIVPTR (input) INTEGER
           The number of Givens rotations which took place in this
           subproblem.

    GIVCOL (input) INTEGER array, dimension ( LDGCOL, 2 )
           Each pair of numbers indicates a pair of rows/columns
           involved in a Givens rotation.

    LDGCOL (input) INTEGER
           The leading dimension of GIVCOL, must be at least N.

    GIVNUM (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 )
           Each number indicates the C or S value used in the
           corresponding Givens rotation.

    LDGNUM (input) INTEGER
           The leading dimension of arrays DIFR, POLES and
           GIVNUM, must be at least K.

    POLES  (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 )
           On entry, POLES(1:K, 1) contains the new singular
           values obtained from solving the secular equation, and
           POLES(1:K, 2) is an array containing the poles in the secular
           equation.

    DIFL   (input) DOUBLE PRECISION array, dimension ( K ).
           On entry, DIFL(I) is the distance between I-th updated
           (undeflated) singular value and the I-th (undeflated) old
           singular value.

    DIFR   (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 ).
           On entry, DIFR(I, 1) contains the distances between I-th
           updated (undeflated) singular value and the I+1-th
           (undeflated) old singular value. And DIFR(I, 2) is the
           normalizing factor for the I-th right singular vector.

    Z      (input) DOUBLE PRECISION array, dimension ( K )
           Contain the components of the deflation-adjusted updating row
           vector.

    K      (input) INTEGER
           Contains the dimension of the non-deflated matrix,
           This is the order of the related secular equation. 1 <= K <=N.

    C      (input) DOUBLE PRECISION
           C contains garbage if SQRE =0 and the C-value of a Givens
           rotation related to the right null space if SQRE = 1.

    S      (input) DOUBLE PRECISION
           S contains garbage if SQRE =0 and the S-value of a Givens
           rotation related to the right null space if SQRE = 1.

    RWORK  (workspace) DOUBLE PRECISION array, dimension
           ( K*(1+NRHS) + 2*NRHS )

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Ren-Cang Li, Computer Science Division, University of
         California at Berkeley, USA
       Osni Marques, LBNL/NERSC, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    bx_dim1 = *ldbx;
    bx_offset = 1 + bx_dim1;
    bx -= bx_offset;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    difr_dim1 = *ldgnum;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    poles_dim1 = *ldgnum;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    --difl;
    --z__;
    --rwork;

    /* Function Body */
    *info = 0;

    if ((*icompq < 0) || (*icompq > 1)) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if ((*sqre < 0) || (*sqre > 1)) {
	*info = -4;
    }

    n = *nl + *nr + 1;

    if (*nrhs < 1) {
	*info = -5;
    } else if (*ldb < n) {
	*info = -7;
    } else if (*ldbx < n) {
	*info = -9;
    } else if (*givptr < 0) {
	*info = -11;
    } else if (*ldgcol < n) {
	*info = -13;
    } else if (*ldgnum < n) {
	*info = -15;
    } else if (*k < 1) {
	*info = -20;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLALS0", &i__1);
	return 0;
    }

    m = n + *sqre;
    nlp1 = *nl + 1;

    if (*icompq == 0) {

/*
          Apply back orthogonal transformations from the left.

          Step (1L): apply back the Givens rotations performed.
*/

	i__1 = *givptr;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    zdrot_(nrhs, &b[givcol[i__ + ((givcol_dim1) << (1))] + b_dim1],
		    ldb, &b[givcol[i__ + givcol_dim1] + b_dim1], ldb, &givnum[
		    i__ + ((givnum_dim1) << (1))], &givnum[i__ + givnum_dim1])
		    ;
/* L10: */
	}

/*        Step (2L): permute rows of B. */

	zcopy_(nrhs, &b[nlp1 + b_dim1], ldb, &bx[bx_dim1 + 1], ldbx);
	i__1 = n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    zcopy_(nrhs, &b[perm[i__] + b_dim1], ldb, &bx[i__ + bx_dim1],
		    ldbx);
/* L20: */
	}

/*
          Step (3L): apply the inverse of the left singular vector
          matrix to BX.
*/

	if (*k == 1) {
	    zcopy_(nrhs, &bx[bx_offset], ldbx, &b[b_offset], ldb);
	    if (z__[1] < 0.) {
		zdscal_(nrhs, &c_b1294, &b[b_offset], ldb);
	    }
	} else {
	    i__1 = *k;
	    for (j = 1; j <= i__1; ++j) {
		diflj = difl[j];
		dj = poles[j + poles_dim1];
		dsigj = -poles[j + ((poles_dim1) << (1))];
		if (j < *k) {
		    difrj = -difr[j + difr_dim1];
		    dsigjp = -poles[j + 1 + ((poles_dim1) << (1))];
		}
		if ((z__[j] == 0.) || (poles[j + ((poles_dim1) << (1))] == 0.)
			) {
		    rwork[j] = 0.;
		} else {
		    rwork[j] = -poles[j + ((poles_dim1) << (1))] * z__[j] /
			    diflj / (poles[j + ((poles_dim1) << (1))] + dj);
		}
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if ((z__[i__] == 0.) || (poles[i__ + ((poles_dim1) << (1))
			    ] == 0.)) {
			rwork[i__] = 0.;
		    } else {
			rwork[i__] = poles[i__ + ((poles_dim1) << (1))] * z__[
				i__] / (dlamc3_(&poles[i__ + ((poles_dim1) <<
				(1))], &dsigj) - diflj) / (poles[i__ + ((
				poles_dim1) << (1))] + dj);
		    }
/* L30: */
		}
		i__2 = *k;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    if ((z__[i__] == 0.) || (poles[i__ + ((poles_dim1) << (1))
			    ] == 0.)) {
			rwork[i__] = 0.;
		    } else {
			rwork[i__] = poles[i__ + ((poles_dim1) << (1))] * z__[
				i__] / (dlamc3_(&poles[i__ + ((poles_dim1) <<
				(1))], &dsigjp) + difrj) / (poles[i__ + ((
				poles_dim1) << (1))] + dj);
		    }
/* L40: */
		}
		rwork[1] = -1.;
		temp = dnrm2_(k, &rwork[1], &c__1);

/*
                Since B and BX are complex, the following call to DGEMV
                is performed in two steps (real and imaginary parts).

                CALL DGEMV( 'T', K, NRHS, ONE, BX, LDBX, WORK, 1, ZERO,
      $                     B( J, 1 ), LDB )
*/

		i__ = *k + ((*nrhs) << (1));
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = *k;
		    for (jrow = 1; jrow <= i__3; ++jrow) {
			++i__;
			i__4 = jrow + jcol * bx_dim1;
			rwork[i__] = bx[i__4].r;
/* L50: */
		    }
/* L60: */
		}
		dgemv_("T", k, nrhs, &c_b1015, &rwork[*k + 1 + ((*nrhs) << (1)
			)], k, &rwork[1], &c__1, &c_b324, &rwork[*k + 1], &
			c__1);
		i__ = *k + ((*nrhs) << (1));
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = *k;
		    for (jrow = 1; jrow <= i__3; ++jrow) {
			++i__;
			rwork[i__] = d_imag(&bx[jrow + jcol * bx_dim1]);
/* L70: */
		    }
/* L80: */
		}
		dgemv_("T", k, nrhs, &c_b1015, &rwork[*k + 1 + ((*nrhs) << (1)
			)], k, &rwork[1], &c__1, &c_b324, &rwork[*k + 1 + *
			nrhs], &c__1);
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = j + jcol * b_dim1;
		    i__4 = jcol + *k;
		    i__5 = jcol + *k + *nrhs;
		    z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		    b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L90: */
		}
		zlascl_("G", &c__0, &c__0, &temp, &c_b1015, &c__1, nrhs, &b[j
			+ b_dim1], ldb, info);
/* L100: */
	    }
	}

/*        Move the deflated rows of BX to B also. */

	if (*k < max(m,n)) {
	    i__1 = n - *k;
	    zlacpy_("A", &i__1, nrhs, &bx[*k + 1 + bx_dim1], ldbx, &b[*k + 1
		    + b_dim1], ldb);
	}
    } else {

/*
          Apply back the right orthogonal transformations.

          Step (1R): apply back the new right singular vector matrix
          to B.
*/

	if (*k == 1) {
	    zcopy_(nrhs, &b[b_offset], ldb, &bx[bx_offset], ldbx);
	} else {
	    i__1 = *k;
	    for (j = 1; j <= i__1; ++j) {
		dsigj = poles[j + ((poles_dim1) << (1))];
		if (z__[j] == 0.) {
		    rwork[j] = 0.;
		} else {
		    rwork[j] = -z__[j] / difl[j] / (dsigj + poles[j +
			    poles_dim1]) / difr[j + ((difr_dim1) << (1))];
		}
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (z__[j] == 0.) {
			rwork[i__] = 0.;
		    } else {
			d__1 = -poles[i__ + 1 + ((poles_dim1) << (1))];
			rwork[i__] = z__[j] / (dlamc3_(&dsigj, &d__1) - difr[
				i__ + difr_dim1]) / (dsigj + poles[i__ +
				poles_dim1]) / difr[i__ + ((difr_dim1) << (1))
				];
		    }
/* L110: */
		}
		i__2 = *k;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    if (z__[j] == 0.) {
			rwork[i__] = 0.;
		    } else {
			d__1 = -poles[i__ + ((poles_dim1) << (1))];
			rwork[i__] = z__[j] / (dlamc3_(&dsigj, &d__1) - difl[
				i__]) / (dsigj + poles[i__ + poles_dim1]) /
				difr[i__ + ((difr_dim1) << (1))];
		    }
/* L120: */
		}

/*
                Since B and BX are complex, the following call to DGEMV
                is performed in two steps (real and imaginary parts).

                CALL DGEMV( 'T', K, NRHS, ONE, B, LDB, WORK, 1, ZERO,
      $                     BX( J, 1 ), LDBX )
*/

		i__ = *k + ((*nrhs) << (1));
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = *k;
		    for (jrow = 1; jrow <= i__3; ++jrow) {
			++i__;
			i__4 = jrow + jcol * b_dim1;
			rwork[i__] = b[i__4].r;
/* L130: */
		    }
/* L140: */
		}
		dgemv_("T", k, nrhs, &c_b1015, &rwork[*k + 1 + ((*nrhs) << (1)
			)], k, &rwork[1], &c__1, &c_b324, &rwork[*k + 1], &
			c__1);
		i__ = *k + ((*nrhs) << (1));
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = *k;
		    for (jrow = 1; jrow <= i__3; ++jrow) {
			++i__;
			rwork[i__] = d_imag(&b[jrow + jcol * b_dim1]);
/* L150: */
		    }
/* L160: */
		}
		dgemv_("T", k, nrhs, &c_b1015, &rwork[*k + 1 + ((*nrhs) << (1)
			)], k, &rwork[1], &c__1, &c_b324, &rwork[*k + 1 + *
			nrhs], &c__1);
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = j + jcol * bx_dim1;
		    i__4 = jcol + *k;
		    i__5 = jcol + *k + *nrhs;
		    z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		    bx[i__3].r = z__1.r, bx[i__3].i = z__1.i;
/* L170: */
		}
/* L180: */
	    }
	}

/*
          Step (2R): if SQRE = 1, apply back the rotation that is
          related to the right null space of the subproblem.
*/

	if (*sqre == 1) {
	    zcopy_(nrhs, &b[m + b_dim1], ldb, &bx[m + bx_dim1], ldbx);
	    zdrot_(nrhs, &bx[bx_dim1 + 1], ldbx, &bx[m + bx_dim1], ldbx, c__,
		    s);
	}
	if (*k < max(m,n)) {
	    i__1 = n - *k;
	    zlacpy_("A", &i__1, nrhs, &b[*k + 1 + b_dim1], ldb, &bx[*k + 1 +
		    bx_dim1], ldbx);
	}

/*        Step (3R): permute rows of B. */

	zcopy_(nrhs, &bx[bx_dim1 + 1], ldbx, &b[nlp1 + b_dim1], ldb);
	if (*sqre == 1) {
	    zcopy_(nrhs, &bx[m + bx_dim1], ldbx, &b[m + b_dim1], ldb);
	}
	i__1 = n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    zcopy_(nrhs, &bx[i__ + bx_dim1], ldbx, &b[perm[i__] + b_dim1],
		    ldb);
/* L190: */
	}

/*        Step (4R): apply back the Givens rotations performed. */

	for (i__ = *givptr; i__ >= 1; --i__) {
	    d__1 = -givnum[i__ + givnum_dim1];
	    zdrot_(nrhs, &b[givcol[i__ + ((givcol_dim1) << (1))] + b_dim1],
		    ldb, &b[givcol[i__ + givcol_dim1] + b_dim1], ldb, &givnum[
		    i__ + ((givnum_dim1) << (1))], &d__1);
/* L200: */
	}
    }

    return 0;

/*     End of ZLALS0 */

} /* zlals0_ */

/* Subroutine */ int zlalsa_(integer *icompq, integer *smlsiz, integer *n,
	integer *nrhs, doublecomplex *b, integer *ldb, doublecomplex *bx,
	integer *ldbx, doublereal *u, integer *ldu, doublereal *vt, integer *
	k, doublereal *difl, doublereal *difr, doublereal *z__, doublereal *
	poles, integer *givptr, integer *givcol, integer *ldgcol, integer *
	perm, doublereal *givnum, doublereal *c__, doublereal *s, doublereal *
	rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, perm_dim1, perm_offset, difl_dim1,
	    difl_offset, difr_dim1, difr_offset, givnum_dim1, givnum_offset,
	    poles_dim1, poles_offset, u_dim1, u_offset, vt_dim1, vt_offset,
	    z_dim1, z_offset, b_dim1, b_offset, bx_dim1, bx_offset, i__1,
	    i__2, i__3, i__4, i__5, i__6;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    integer pow_ii(integer *, integer *);

    /* Local variables */
    static integer i__, j, i1, ic, lf, nd, ll, nl, nr, im1, nlf, nrf, lvl,
	    ndb1, nlp1, lvl2, nrp1, jcol, nlvl, sqre, jrow, jimag;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *);
    static integer jreal, inode, ndiml, ndimr;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zlals0_(integer *, integer *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, integer *, integer *, integer *,
	    integer *, doublereal *, integer *, doublereal *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     doublereal *, integer *), dlasdt_(integer *, integer *, integer *
	    , integer *, integer *, integer *, integer *), xerbla_(char *,
	    integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLALSA is an itermediate step in solving the least squares problem
    by computing the SVD of the coefficient matrix in compact form (The
    singular vectors are computed as products of simple orthorgonal
    matrices.).

    If ICOMPQ = 0, ZLALSA applies the inverse of the left singular vector
    matrix of an upper bidiagonal matrix to the right hand side; and if
    ICOMPQ = 1, ZLALSA applies the right singular vector matrix to the
    right hand side. The singular vector matrices were generated in
    compact form by ZLALSA.

    Arguments
    =========

    ICOMPQ (input) INTEGER
           Specifies whether the left or the right singular vector
           matrix is involved.
           = 0: Left singular vector matrix
           = 1: Right singular vector matrix

    SMLSIZ (input) INTEGER
           The maximum size of the subproblems at the bottom of the
           computation tree.

    N      (input) INTEGER
           The row and column dimensions of the upper bidiagonal matrix.

    NRHS   (input) INTEGER
           The number of columns of B and BX. NRHS must be at least 1.

    B      (input) COMPLEX*16 array, dimension ( LDB, NRHS )
           On input, B contains the right hand sides of the least
           squares problem in rows 1 through M. On output, B contains
           the solution X in rows 1 through N.

    LDB    (input) INTEGER
           The leading dimension of B in the calling subprogram.
           LDB must be at least max(1,MAX( M, N ) ).

    BX     (output) COMPLEX*16 array, dimension ( LDBX, NRHS )
           On exit, the result of applying the left or right singular
           vector matrix to B.

    LDBX   (input) INTEGER
           The leading dimension of BX.

    U      (input) DOUBLE PRECISION array, dimension ( LDU, SMLSIZ ).
           On entry, U contains the left singular vector matrices of all
           subproblems at the bottom level.

    LDU    (input) INTEGER, LDU = > N.
           The leading dimension of arrays U, VT, DIFL, DIFR,
           POLES, GIVNUM, and Z.

    VT     (input) DOUBLE PRECISION array, dimension ( LDU, SMLSIZ+1 ).
           On entry, VT' contains the right singular vector matrices of
           all subproblems at the bottom level.

    K      (input) INTEGER array, dimension ( N ).

    DIFL   (input) DOUBLE PRECISION array, dimension ( LDU, NLVL ).
           where NLVL = INT(log_2 (N/(SMLSIZ+1))) + 1.

    DIFR   (input) DOUBLE PRECISION array, dimension ( LDU, 2 * NLVL ).
           On entry, DIFL(*, I) and DIFR(*, 2 * I -1) record
           distances between singular values on the I-th level and
           singular values on the (I -1)-th level, and DIFR(*, 2 * I)
           record the normalizing factors of the right singular vectors
           matrices of subproblems on I-th level.

    Z      (input) DOUBLE PRECISION array, dimension ( LDU, NLVL ).
           On entry, Z(1, I) contains the components of the deflation-
           adjusted updating row vector for subproblems on the I-th
           level.

    POLES  (input) DOUBLE PRECISION array, dimension ( LDU, 2 * NLVL ).
           On entry, POLES(*, 2 * I -1: 2 * I) contains the new and old
           singular values involved in the secular equations on the I-th
           level.

    GIVPTR (input) INTEGER array, dimension ( N ).
           On entry, GIVPTR( I ) records the number of Givens
           rotations performed on the I-th problem on the computation
           tree.

    GIVCOL (input) INTEGER array, dimension ( LDGCOL, 2 * NLVL ).
           On entry, for each I, GIVCOL(*, 2 * I - 1: 2 * I) records the
           locations of Givens rotations performed on the I-th level on
           the computation tree.

    LDGCOL (input) INTEGER, LDGCOL = > N.
           The leading dimension of arrays GIVCOL and PERM.

    PERM   (input) INTEGER array, dimension ( LDGCOL, NLVL ).
           On entry, PERM(*, I) records permutations done on the I-th
           level of the computation tree.

    GIVNUM (input) DOUBLE PRECISION array, dimension ( LDU, 2 * NLVL ).
           On entry, GIVNUM(*, 2 *I -1 : 2 * I) records the C- and S-
           values of Givens rotations performed on the I-th level on the
           computation tree.

    C      (input) DOUBLE PRECISION array, dimension ( N ).
           On entry, if the I-th subproblem is not square,
           C( I ) contains the C-value of a Givens rotation related to
           the right null space of the I-th subproblem.

    S      (input) DOUBLE PRECISION array, dimension ( N ).
           On entry, if the I-th subproblem is not square,
           S( I ) contains the S-value of a Givens rotation related to
           the right null space of the I-th subproblem.

    RWORK  (workspace) DOUBLE PRECISION array, dimension at least
           max ( N, (SMLSZ+1)*NRHS*3 ).

    IWORK  (workspace) INTEGER array.
           The dimension must be at least 3 * N

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Ren-Cang Li, Computer Science Division, University of
         California at Berkeley, USA
       Osni Marques, LBNL/NERSC, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    bx_dim1 = *ldbx;
    bx_offset = 1 + bx_dim1;
    bx -= bx_offset;
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
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

    if ((*icompq < 0) || (*icompq > 1)) {
	*info = -1;
    } else if (*smlsiz < 3) {
	*info = -2;
    } else if (*n < *smlsiz) {
	*info = -3;
    } else if (*nrhs < 1) {
	*info = -4;
    } else if (*ldb < *n) {
	*info = -6;
    } else if (*ldbx < *n) {
	*info = -8;
    } else if (*ldu < *n) {
	*info = -10;
    } else if (*ldgcol < *n) {
	*info = -19;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLALSA", &i__1);
	return 0;
    }

/*     Book-keeping and  setting up the computation tree. */

    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;

    dlasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr],
	    smlsiz);

/*
       The following code applies back the left singular vector factors.
       For applying back the right singular vector factors, go to 170.
*/

    if (*icompq == 1) {
	goto L170;
    }

/*
       The nodes on the bottom level of the tree were solved
       by DLASDQ. The corresponding left and right singular vector
       matrices are in explicit form. First apply back the left
       singular vector matrices.
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
	nr = iwork[ndimr + i1];
	nlf = ic - nl;
	nrf = ic + 1;

/*
          Since B and BX are complex, the following call to DGEMM
          is performed in two steps (real and imaginary parts).

          CALL DGEMM( 'T', 'N', NL, NRHS, NL, ONE, U( NLF, 1 ), LDU,
       $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
*/

	j = (nl * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nl - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++j;
		i__4 = jrow + jcol * b_dim1;
		rwork[j] = b[i__4].r;
/* L10: */
	    }
/* L20: */
	}
	dgemm_("T", "N", &nl, nrhs, &nl, &c_b1015, &u[nlf + u_dim1], ldu, &
		rwork[((nl * *nrhs) << (1)) + 1], &nl, &c_b324, &rwork[1], &
		nl);
	j = (nl * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nl - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L30: */
	    }
/* L40: */
	}
	dgemm_("T", "N", &nl, nrhs, &nl, &c_b1015, &u[nlf + u_dim1], ldu, &
		rwork[((nl * *nrhs) << (1)) + 1], &nl, &c_b324, &rwork[nl * *
		nrhs + 1], &nl);
	jreal = 0;
	jimag = nl * *nrhs;
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nl - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++jreal;
		++jimag;
		i__4 = jrow + jcol * bx_dim1;
		i__5 = jreal;
		i__6 = jimag;
		z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		bx[i__4].r = z__1.r, bx[i__4].i = z__1.i;
/* L50: */
	    }
/* L60: */
	}

/*
          Since B and BX are complex, the following call to DGEMM
          is performed in two steps (real and imaginary parts).

          CALL DGEMM( 'T', 'N', NR, NRHS, NR, ONE, U( NRF, 1 ), LDU,
      $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
*/

	j = (nr * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nr - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++j;
		i__4 = jrow + jcol * b_dim1;
		rwork[j] = b[i__4].r;
/* L70: */
	    }
/* L80: */
	}
	dgemm_("T", "N", &nr, nrhs, &nr, &c_b1015, &u[nrf + u_dim1], ldu, &
		rwork[((nr * *nrhs) << (1)) + 1], &nr, &c_b324, &rwork[1], &
		nr);
	j = (nr * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nr - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L90: */
	    }
/* L100: */
	}
	dgemm_("T", "N", &nr, nrhs, &nr, &c_b1015, &u[nrf + u_dim1], ldu, &
		rwork[((nr * *nrhs) << (1)) + 1], &nr, &c_b324, &rwork[nr * *
		nrhs + 1], &nr);
	jreal = 0;
	jimag = nr * *nrhs;
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nr - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++jreal;
		++jimag;
		i__4 = jrow + jcol * bx_dim1;
		i__5 = jreal;
		i__6 = jimag;
		z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		bx[i__4].r = z__1.r, bx[i__4].i = z__1.i;
/* L110: */
	    }
/* L120: */
	}

/* L130: */
    }

/*
       Next copy the rows of B that correspond to unchanged rows
       in the bidiagonal matrix to BX.
*/

    i__1 = nd;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ic = iwork[inode + i__ - 1];
	zcopy_(nrhs, &b[ic + b_dim1], ldb, &bx[ic + bx_dim1], ldbx);
/* L140: */
    }

/*
       Finally go through the left singular vector matrices of all
       the other subproblems bottom-up on the tree.
*/

    j = pow_ii(&c__2, &nlvl);
    sqre = 0;

    for (lvl = nlvl; lvl >= 1; --lvl) {
	lvl2 = ((lvl) << (1)) - 1;

/*
          find the first node LF and last node LL on
          the current level LVL
*/

	if (lvl == 1) {
	    lf = 1;
	    ll = 1;
	} else {
	    i__1 = lvl - 1;
	    lf = pow_ii(&c__2, &i__1);
	    ll = ((lf) << (1)) - 1;
	}
	i__1 = ll;
	for (i__ = lf; i__ <= i__1; ++i__) {
	    im1 = i__ - 1;
	    ic = iwork[inode + im1];
	    nl = iwork[ndiml + im1];
	    nr = iwork[ndimr + im1];
	    nlf = ic - nl;
	    nrf = ic + 1;
	    --j;
	    zlals0_(icompq, &nl, &nr, &sqre, nrhs, &bx[nlf + bx_dim1], ldbx, &
		    b[nlf + b_dim1], ldb, &perm[nlf + lvl * perm_dim1], &
		    givptr[j], &givcol[nlf + lvl2 * givcol_dim1], ldgcol, &
		    givnum[nlf + lvl2 * givnum_dim1], ldu, &poles[nlf + lvl2 *
		     poles_dim1], &difl[nlf + lvl * difl_dim1], &difr[nlf +
		    lvl2 * difr_dim1], &z__[nlf + lvl * z_dim1], &k[j], &c__[
		    j], &s[j], &rwork[1], info);
/* L150: */
	}
/* L160: */
    }
    goto L330;

/*     ICOMPQ = 1: applying back the right singular vector factors. */

L170:

/*
       First now go through the right singular vector matrices of all
       the tree nodes top-down.
*/

    j = 0;
    i__1 = nlvl;
    for (lvl = 1; lvl <= i__1; ++lvl) {
	lvl2 = ((lvl) << (1)) - 1;

/*
          Find the first node LF and last node LL on
          the current level LVL.
*/

	if (lvl == 1) {
	    lf = 1;
	    ll = 1;
	} else {
	    i__2 = lvl - 1;
	    lf = pow_ii(&c__2, &i__2);
	    ll = ((lf) << (1)) - 1;
	}
	i__2 = lf;
	for (i__ = ll; i__ >= i__2; --i__) {
	    im1 = i__ - 1;
	    ic = iwork[inode + im1];
	    nl = iwork[ndiml + im1];
	    nr = iwork[ndimr + im1];
	    nlf = ic - nl;
	    nrf = ic + 1;
	    if (i__ == ll) {
		sqre = 0;
	    } else {
		sqre = 1;
	    }
	    ++j;
	    zlals0_(icompq, &nl, &nr, &sqre, nrhs, &b[nlf + b_dim1], ldb, &bx[
		    nlf + bx_dim1], ldbx, &perm[nlf + lvl * perm_dim1], &
		    givptr[j], &givcol[nlf + lvl2 * givcol_dim1], ldgcol, &
		    givnum[nlf + lvl2 * givnum_dim1], ldu, &poles[nlf + lvl2 *
		     poles_dim1], &difl[nlf + lvl * difl_dim1], &difr[nlf +
		    lvl2 * difr_dim1], &z__[nlf + lvl * z_dim1], &k[j], &c__[
		    j], &s[j], &rwork[1], info);
/* L180: */
	}
/* L190: */
    }

/*
       The nodes on the bottom level of the tree were solved
       by DLASDQ. The corresponding right singular vector
       matrices are in explicit form. Apply them back.
*/

    ndb1 = (nd + 1) / 2;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {
	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nr = iwork[ndimr + i1];
	nlp1 = nl + 1;
	if (i__ == nd) {
	    nrp1 = nr;
	} else {
	    nrp1 = nr + 1;
	}
	nlf = ic - nl;
	nrf = ic + 1;

/*
          Since B and BX are complex, the following call to DGEMM is
          performed in two steps (real and imaginary parts).

          CALL DGEMM( 'T', 'N', NLP1, NRHS, NLP1, ONE, VT( NLF, 1 ), LDU,
      $               B( NLF, 1 ), LDB, ZERO, BX( NLF, 1 ), LDBX )
*/

	j = (nlp1 * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nlp1 - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++j;
		i__4 = jrow + jcol * b_dim1;
		rwork[j] = b[i__4].r;
/* L200: */
	    }
/* L210: */
	}
	dgemm_("T", "N", &nlp1, nrhs, &nlp1, &c_b1015, &vt[nlf + vt_dim1],
		ldu, &rwork[((nlp1 * *nrhs) << (1)) + 1], &nlp1, &c_b324, &
		rwork[1], &nlp1);
	j = (nlp1 * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nlp1 - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L220: */
	    }
/* L230: */
	}
	dgemm_("T", "N", &nlp1, nrhs, &nlp1, &c_b1015, &vt[nlf + vt_dim1],
		ldu, &rwork[((nlp1 * *nrhs) << (1)) + 1], &nlp1, &c_b324, &
		rwork[nlp1 * *nrhs + 1], &nlp1);
	jreal = 0;
	jimag = nlp1 * *nrhs;
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nlf + nlp1 - 1;
	    for (jrow = nlf; jrow <= i__3; ++jrow) {
		++jreal;
		++jimag;
		i__4 = jrow + jcol * bx_dim1;
		i__5 = jreal;
		i__6 = jimag;
		z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		bx[i__4].r = z__1.r, bx[i__4].i = z__1.i;
/* L240: */
	    }
/* L250: */
	}

/*
          Since B and BX are complex, the following call to DGEMM is
          performed in two steps (real and imaginary parts).

          CALL DGEMM( 'T', 'N', NRP1, NRHS, NRP1, ONE, VT( NRF, 1 ), LDU,
      $               B( NRF, 1 ), LDB, ZERO, BX( NRF, 1 ), LDBX )
*/

	j = (nrp1 * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nrp1 - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++j;
		i__4 = jrow + jcol * b_dim1;
		rwork[j] = b[i__4].r;
/* L260: */
	    }
/* L270: */
	}
	dgemm_("T", "N", &nrp1, nrhs, &nrp1, &c_b1015, &vt[nrf + vt_dim1],
		ldu, &rwork[((nrp1 * *nrhs) << (1)) + 1], &nrp1, &c_b324, &
		rwork[1], &nrp1);
	j = (nrp1 * *nrhs) << (1);
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nrp1 - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L280: */
	    }
/* L290: */
	}
	dgemm_("T", "N", &nrp1, nrhs, &nrp1, &c_b1015, &vt[nrf + vt_dim1],
		ldu, &rwork[((nrp1 * *nrhs) << (1)) + 1], &nrp1, &c_b324, &
		rwork[nrp1 * *nrhs + 1], &nrp1);
	jreal = 0;
	jimag = nrp1 * *nrhs;
	i__2 = *nrhs;
	for (jcol = 1; jcol <= i__2; ++jcol) {
	    i__3 = nrf + nrp1 - 1;
	    for (jrow = nrf; jrow <= i__3; ++jrow) {
		++jreal;
		++jimag;
		i__4 = jrow + jcol * bx_dim1;
		i__5 = jreal;
		i__6 = jimag;
		z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		bx[i__4].r = z__1.r, bx[i__4].i = z__1.i;
/* L300: */
	    }
/* L310: */
	}

/* L320: */
    }

L330:

    return 0;

/*     End of ZLALSA */

} /* zlalsa_ */

/* Subroutine */ int zlalsd_(char *uplo, integer *smlsiz, integer *n, integer
	*nrhs, doublereal *d__, doublereal *e, doublecomplex *b, integer *ldb,
	 doublereal *rcond, integer *rank, doublecomplex *work, doublereal *
	rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *), log(doublereal), d_sign(doublereal *,
	    doublereal *);

    /* Local variables */
    static integer c__, i__, j, k;
    static doublereal r__;
    static integer s, u, z__;
    static doublereal cs;
    static integer bx;
    static doublereal sn;
    static integer st, vt, nm1, st1;
    static doublereal eps;
    static integer iwk;
    static doublereal tol;
    static integer difl, difr, jcol, irwb, perm, nsub, nlvl, sqre, bxst, jrow,
	     irwu, jimag;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *);
    static integer jreal, irwib, poles, sizei, irwrb, nsize;
    extern /* Subroutine */ int zdrot_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublereal *, doublereal *), zcopy_(
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *)
	    ;
    static integer irwvt, icmpq1, icmpq2;

    extern /* Subroutine */ int dlasda_(integer *, integer *, integer *,
	    integer *, doublereal *, doublereal *, doublereal *, integer *,
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *,
	     doublereal *, integer *, integer *, integer *, integer *,
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *,
	     integer *), dlascl_(char *, integer *, integer *, doublereal *,
	    doublereal *, integer *, integer *, doublereal *, integer *,
	    integer *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dlasdq_(char *, integer *, integer *, integer
	    *, integer *, integer *, doublereal *, doublereal *, doublereal *,
	     integer *, doublereal *, integer *, doublereal *, integer *,
	    doublereal *, integer *), dlaset_(char *, integer *,
	    integer *, doublereal *, doublereal *, doublereal *, integer *), dlartg_(doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *), xerbla_(char *, integer *);
    static integer givcol;
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int zlalsa_(integer *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *,
	     doublereal *, integer *, doublereal *, integer *, doublereal *,
	    doublereal *, doublereal *, doublereal *, integer *, integer *,
	    integer *, integer *, doublereal *, doublereal *, doublereal *,
	    doublereal *, integer *, integer *), zlascl_(char *, integer *,
	    integer *, doublereal *, doublereal *, integer *, integer *,
	    doublecomplex *, integer *, integer *), dlasrt_(char *,
	    integer *, doublereal *, integer *), zlacpy_(char *,
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *,
	     integer *), zlaset_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *);
    static doublereal orgnrm;
    static integer givnum, givptr, nrwork, irwwrk, smlszp;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1999


    Purpose
    =======

    ZLALSD uses the singular value decomposition of A to solve the least
    squares problem of finding X to minimize the Euclidean norm of each
    column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
    are N-by-NRHS. The solution X overwrites B.

    The singular values of A smaller than RCOND times the largest
    singular value are treated as zero in solving the least squares
    problem; in this case a minimum norm solution is returned.
    The actual singular values are returned in D in ascending order.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========

    UPLO   (input) CHARACTER*1
           = 'U': D and E define an upper bidiagonal matrix.
           = 'L': D and E define a  lower bidiagonal matrix.

    SMLSIZ (input) INTEGER
           The maximum size of the subproblems at the bottom of the
           computation tree.

    N      (input) INTEGER
           The dimension of the  bidiagonal matrix.  N >= 0.

    NRHS   (input) INTEGER
           The number of columns of B. NRHS must be at least 1.

    D      (input/output) DOUBLE PRECISION array, dimension (N)
           On entry D contains the main diagonal of the bidiagonal
           matrix. On exit, if INFO = 0, D contains its singular values.

    E      (input) DOUBLE PRECISION array, dimension (N-1)
           Contains the super-diagonal entries of the bidiagonal matrix.
           On exit, E has been destroyed.

    B      (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
           On input, B contains the right hand sides of the least
           squares problem. On output, B contains the solution X.

    LDB    (input) INTEGER
           The leading dimension of B in the calling subprogram.
           LDB must be at least max(1,N).

    RCOND  (input) DOUBLE PRECISION
           The singular values of A less than or equal to RCOND times
           the largest singular value are treated as zero in solving
           the least squares problem. If RCOND is negative,
           machine precision is used instead.
           For example, if diag(S)*X=B were the least squares problem,
           where diag(S) is a diagonal matrix of singular values, the
           solution would be X(i) = B(i) / S(i) if S(i) is greater than
           RCOND*max(S), and X(i) = 0 if S(i) is less than or equal to
           RCOND*max(S).

    RANK   (output) INTEGER
           The number of singular values of A greater than RCOND times
           the largest singular value.

    WORK   (workspace) COMPLEX*16 array, dimension at least
           (N * NRHS).

    RWORK  (workspace) DOUBLE PRECISION array, dimension at least
           (9*N + 2*N*SMLSIZ + 8*N*NLVL + 3*SMLSIZ*NRHS + (SMLSIZ+1)**2),
           where
           NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 )

    IWORK  (workspace) INTEGER array, dimension at least
           (3*N*NLVL + 11*N).

    INFO   (output) INTEGER
           = 0:  successful exit.
           < 0:  if INFO = -i, the i-th argument had an illegal value.
           > 0:  The algorithm failed to compute an singular value while
                 working on the submatrix lying in rows and columns
                 INFO/(N+1) through MOD(INFO,N+1).

    Further Details
    ===============

    Based on contributions by
       Ming Gu and Ren-Cang Li, Computer Science Division, University of
         California at Berkeley, USA
       Osni Marques, LBNL/NERSC, USA

    =====================================================================


       Test the input parameters.
*/

    /* Parameter adjustments */
    --d__;
    --e;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 1) {
	*info = -4;
    } else if ((*ldb < 1) || (*ldb < *n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLALSD", &i__1);
	return 0;
    }

    eps = EPSILON;

/*     Set up the tolerance. */

    if ((*rcond <= 0.) || (*rcond >= 1.)) {
	*rcond = eps;
    }

    *rank = 0;

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	if (d__[1] == 0.) {
	    zlaset_("A", &c__1, nrhs, &c_b59, &c_b59, &b[b_offset], ldb);
	} else {
	    *rank = 1;
	    zlascl_("G", &c__0, &c__0, &d__[1], &c_b1015, &c__1, nrhs, &b[
		    b_offset], ldb, info);
	    d__[1] = abs(d__[1]);
	}
	return 0;
    }

/*     Rotate the matrix if it is lower bidiagonal. */

    if (*(unsigned char *)uplo == 'L') {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (*nrhs == 1) {
		zdrot_(&c__1, &b[i__ + b_dim1], &c__1, &b[i__ + 1 + b_dim1], &
			c__1, &cs, &sn);
	    } else {
		rwork[((i__) << (1)) - 1] = cs;
		rwork[i__ * 2] = sn;
	    }
/* L10: */
	}
	if (*nrhs > 1) {
	    i__1 = *nrhs;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *n - 1;
		for (j = 1; j <= i__2; ++j) {
		    cs = rwork[((j) << (1)) - 1];
		    sn = rwork[j * 2];
		    zdrot_(&c__1, &b[j + i__ * b_dim1], &c__1, &b[j + 1 + i__
			    * b_dim1], &c__1, &cs, &sn);
/* L20: */
		}
/* L30: */
	    }
	}
    }

/*     Scale. */

    nm1 = *n - 1;
    orgnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.) {
	zlaset_("A", n, nrhs, &c_b59, &c_b59, &b[b_offset], ldb);
	return 0;
    }

    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, n, &c__1, &d__[1], n, info);
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, &nm1, &c__1, &e[1], &nm1,
	    info);

/*
       If N is smaller than the minimum divide size SMLSIZ, then solve
       the problem with another solver.
*/

    if (*n <= *smlsiz) {
	irwu = 1;
	irwvt = irwu + *n * *n;
	irwwrk = irwvt + *n * *n;
	irwrb = irwwrk;
	irwib = irwrb + *n * *nrhs;
	irwb = irwib + *n * *nrhs;
	dlaset_("A", n, n, &c_b324, &c_b1015, &rwork[irwu], n);
	dlaset_("A", n, n, &c_b324, &c_b1015, &rwork[irwvt], n);
	dlasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &rwork[irwvt], n,
		&rwork[irwu], n, &rwork[irwwrk], &c__1, &rwork[irwwrk], info);
	if (*info != 0) {
	    return 0;
	}

/*
          In the real version, B is passed to DLASDQ and multiplied
          internally by Q'. Here B is complex and that product is
          computed below in two steps (real and imaginary parts).
*/

	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		i__3 = jrow + jcol * b_dim1;
		rwork[j] = b[i__3].r;
/* L40: */
	    }
/* L50: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b1015, &rwork[irwu], n, &rwork[irwb],
		n, &c_b324, &rwork[irwrb], n);
	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L60: */
	    }
/* L70: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b1015, &rwork[irwu], n, &rwork[irwb],
		n, &c_b324, &rwork[irwib], n);
	jreal = irwrb - 1;
	jimag = irwib - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++jreal;
		++jimag;
		i__3 = jrow + jcol * b_dim1;
		i__4 = jreal;
		i__5 = jimag;
		z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L80: */
	    }
/* L90: */
	}

	tol = *rcond * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (d__[i__] <= tol) {
		zlaset_("A", &c__1, nrhs, &c_b59, &c_b59, &b[i__ + b_dim1],
			ldb);
	    } else {
		zlascl_("G", &c__0, &c__0, &d__[i__], &c_b1015, &c__1, nrhs, &
			b[i__ + b_dim1], ldb, info);
		++(*rank);
	    }
/* L100: */
	}

/*
          Since B is complex, the following call to DGEMM is performed
          in two steps (real and imaginary parts). That is for V * B
          (in the real version of the code V' is stored in WORK).

          CALL DGEMM( 'T', 'N', N, NRHS, N, ONE, WORK, N, B, LDB, ZERO,
      $               WORK( NWORK ), N )
*/

	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		i__3 = jrow + jcol * b_dim1;
		rwork[j] = b[i__3].r;
/* L110: */
	    }
/* L120: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b1015, &rwork[irwvt], n, &rwork[irwb],
		 n, &c_b324, &rwork[irwrb], n);
	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L130: */
	    }
/* L140: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b1015, &rwork[irwvt], n, &rwork[irwb],
		 n, &c_b324, &rwork[irwib], n);
	jreal = irwrb - 1;
	jimag = irwib - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++jreal;
		++jimag;
		i__3 = jrow + jcol * b_dim1;
		i__4 = jreal;
		i__5 = jimag;
		z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L150: */
	    }
/* L160: */
	}

/*        Unscale. */

	dlascl_("G", &c__0, &c__0, &c_b1015, &orgnrm, n, &c__1, &d__[1], n,
		info);
	dlasrt_("D", n, &d__[1], info);
	zlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, n, nrhs, &b[b_offset],
		ldb, info);

	return 0;
    }

/*     Book-keeping and setting up some constants. */

    nlvl = (integer) (log((doublereal) (*n) / (doublereal) (*smlsiz + 1)) /
	    log(2.)) + 1;

    smlszp = *smlsiz + 1;

    u = 1;
    vt = *smlsiz * *n + 1;
    difl = vt + smlszp * *n;
    difr = difl + nlvl * *n;
    z__ = difr + ((nlvl * *n) << (1));
    c__ = z__ + nlvl * *n;
    s = c__ + *n;
    poles = s + *n;
    givnum = poles + ((nlvl) << (1)) * *n;
    nrwork = givnum + ((nlvl) << (1)) * *n;
    bx = 1;

    irwrb = nrwork;
    irwib = irwrb + *smlsiz * *nrhs;
    irwb = irwib + *smlsiz * *nrhs;

    sizei = *n + 1;
    k = sizei + *n;
    givptr = k + *n;
    perm = givptr + *n;
    givcol = perm + nlvl * *n;
    iwk = givcol + ((nlvl * *n) << (1));

    st = 1;
    sqre = 0;
    icmpq1 = 1;
    icmpq2 = 0;
    nsub = 0;

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = d__[i__], abs(d__1)) < eps) {
	    d__[i__] = d_sign(&eps, &d__[i__]);
	}
/* L170: */
    }

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (((d__1 = e[i__], abs(d__1)) < eps) || (i__ == nm1)) {
	    ++nsub;
	    iwork[nsub] = st;

/*
             Subproblem found. First determine its size and then
             apply divide and conquer on it.
*/

	    if (i__ < nm1) {

/*              A subproblem with E(I) small for I < NM1. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else if ((d__1 = e[i__], abs(d__1)) >= eps) {

/*              A subproblem with E(NM1) not too small but I = NM1. */

		nsize = *n - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else {

/*
                A subproblem with E(NM1) small. This implies an
                1-by-1 subproblem at D(N), which is not solved
                explicitly.
*/

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
		++nsub;
		iwork[nsub] = *n;
		iwork[sizei + nsub - 1] = 1;
		zcopy_(nrhs, &b[*n + b_dim1], ldb, &work[bx + nm1], n);
	    }
	    st1 = st - 1;
	    if (nsize == 1) {

/*
                This is a 1-by-1 subproblem and is not solved
                explicitly.
*/

		zcopy_(nrhs, &b[st + b_dim1], ldb, &work[bx + st1], n);
	    } else if (nsize <= *smlsiz) {

/*              This is a small subproblem and is solved by DLASDQ. */

		dlaset_("A", &nsize, &nsize, &c_b324, &c_b1015, &rwork[vt +
			st1], n);
		dlaset_("A", &nsize, &nsize, &c_b324, &c_b1015, &rwork[u +
			st1], n);
		dlasdq_("U", &c__0, &nsize, &nsize, &nsize, &c__0, &d__[st], &
			e[st], &rwork[vt + st1], n, &rwork[u + st1], n, &
			rwork[nrwork], &c__1, &rwork[nrwork], info)
			;
		if (*info != 0) {
		    return 0;
		}

/*
                In the real version, B is passed to DLASDQ and multiplied
                internally by Q'. Here B is complex and that product is
                computed below in two steps (real and imaginary parts).
*/

		j = irwb - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++j;
			i__4 = jrow + jcol * b_dim1;
			rwork[j] = b[i__4].r;
/* L180: */
		    }
/* L190: */
		}
		dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b1015, &rwork[u +
			st1], n, &rwork[irwb], &nsize, &c_b324, &rwork[irwrb],
			 &nsize);
		j = irwb - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++j;
			rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L200: */
		    }
/* L210: */
		}
		dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b1015, &rwork[u +
			st1], n, &rwork[irwb], &nsize, &c_b324, &rwork[irwib],
			 &nsize);
		jreal = irwrb - 1;
		jimag = irwib - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++jreal;
			++jimag;
			i__4 = jrow + jcol * b_dim1;
			i__5 = jreal;
			i__6 = jimag;
			z__1.r = rwork[i__5], z__1.i = rwork[i__6];
			b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L220: */
		    }
/* L230: */
		}

		zlacpy_("A", &nsize, nrhs, &b[st + b_dim1], ldb, &work[bx +
			st1], n);
	    } else {

/*              A large problem. Solve it using divide and conquer. */

		dlasda_(&icmpq1, smlsiz, &nsize, &sqre, &d__[st], &e[st], &
			rwork[u + st1], n, &rwork[vt + st1], &iwork[k + st1],
			&rwork[difl + st1], &rwork[difr + st1], &rwork[z__ +
			st1], &rwork[poles + st1], &iwork[givptr + st1], &
			iwork[givcol + st1], n, &iwork[perm + st1], &rwork[
			givnum + st1], &rwork[c__ + st1], &rwork[s + st1], &
			rwork[nrwork], &iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
		bxst = bx + st1;
		zlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &b[st + b_dim1], ldb, &
			work[bxst], n, &rwork[u + st1], n, &rwork[vt + st1], &
			iwork[k + st1], &rwork[difl + st1], &rwork[difr + st1]
			, &rwork[z__ + st1], &rwork[poles + st1], &iwork[
			givptr + st1], &iwork[givcol + st1], n, &iwork[perm +
			st1], &rwork[givnum + st1], &rwork[c__ + st1], &rwork[
			s + st1], &rwork[nrwork], &iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
	    }
	    st = i__ + 1;
	}
/* L240: */
    }

/*     Apply the singular values and treat the tiny ones as zero. */

    tol = *rcond * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*
          Some of the elements in D can be negative because 1-by-1
          subproblems were not solved explicitly.
*/

	if ((d__1 = d__[i__], abs(d__1)) <= tol) {
	    zlaset_("A", &c__1, nrhs, &c_b59, &c_b59, &work[bx + i__ - 1], n);
	} else {
	    ++(*rank);
	    zlascl_("G", &c__0, &c__0, &d__[i__], &c_b1015, &c__1, nrhs, &
		    work[bx + i__ - 1], n, info);
	}
	d__[i__] = (d__1 = d__[i__], abs(d__1));
/* L250: */
    }

/*     Now apply back the right singular vectors. */

    icmpq2 = 1;
    i__1 = nsub;
    for (i__ = 1; i__ <= i__1; ++i__) {
	st = iwork[i__];
	st1 = st - 1;
	nsize = iwork[sizei + i__ - 1];
	bxst = bx + st1;
	if (nsize == 1) {
	    zcopy_(nrhs, &work[bxst], n, &b[st + b_dim1], ldb);
	} else if (nsize <= *smlsiz) {

/*
             Since B and BX are complex, the following call to DGEMM
             is performed in two steps (real and imaginary parts).

             CALL DGEMM( 'T', 'N', NSIZE, NRHS, NSIZE, ONE,
      $                  RWORK( VT+ST1 ), N, RWORK( BXST ), N, ZERO,
      $                  B( ST, 1 ), LDB )
*/

	    j = bxst - *n - 1;
	    jreal = irwb - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		j += *n;
		i__3 = nsize;
		for (jrow = 1; jrow <= i__3; ++jrow) {
		    ++jreal;
		    i__4 = j + jrow;
		    rwork[jreal] = work[i__4].r;
/* L260: */
		}
/* L270: */
	    }
	    dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b1015, &rwork[vt + st1],
		     n, &rwork[irwb], &nsize, &c_b324, &rwork[irwrb], &nsize);
	    j = bxst - *n - 1;
	    jimag = irwb - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		j += *n;
		i__3 = nsize;
		for (jrow = 1; jrow <= i__3; ++jrow) {
		    ++jimag;
		    rwork[jimag] = d_imag(&work[j + jrow]);
/* L280: */
		}
/* L290: */
	    }
	    dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b1015, &rwork[vt + st1],
		     n, &rwork[irwb], &nsize, &c_b324, &rwork[irwib], &nsize);
	    jreal = irwrb - 1;
	    jimag = irwib - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		i__3 = st + nsize - 1;
		for (jrow = st; jrow <= i__3; ++jrow) {
		    ++jreal;
		    ++jimag;
		    i__4 = jrow + jcol * b_dim1;
		    i__5 = jreal;
		    i__6 = jimag;
		    z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		    b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L300: */
		}
/* L310: */
	    }
	} else {
	    zlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &work[bxst], n, &b[st +
		    b_dim1], ldb, &rwork[u + st1], n, &rwork[vt + st1], &
		    iwork[k + st1], &rwork[difl + st1], &rwork[difr + st1], &
		    rwork[z__ + st1], &rwork[poles + st1], &iwork[givptr +
		    st1], &iwork[givcol + st1], n, &iwork[perm + st1], &rwork[
		    givnum + st1], &rwork[c__ + st1], &rwork[s + st1], &rwork[
		    nrwork], &iwork[iwk], info);
	    if (*info != 0) {
		return 0;
	    }
	}
/* L320: */
    }

/*     Unscale and sort the singular values. */

    dlascl_("G", &c__0, &c__0, &c_b1015, &orgnrm, n, &c__1, &d__[1], n, info);
    dlasrt_("D", n, &d__[1], info);
    zlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, n, nrhs, &b[b_offset], ldb,
	    info);

    return 0;

/*     End of ZLALSD */

} /* zlalsd_ */

doublereal zlange_(char *norm, integer *m, integer *n, doublecomplex *a,
	integer *lda, doublereal *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double z_abs(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static doublereal sum, scale;
    extern logical lsame_(char *, char *);
    static doublereal value;
    extern /* Subroutine */ int zlassq_(integer *, doublecomplex *, integer *,
	     doublereal *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLANGE  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    complex matrix A.

    Description
    ===========

    ZLANGE returns the value

       ZLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in ZLANGE as described
            above.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            ZLANGE is set to zero.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            ZLANGE is set to zero.

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            The m by n matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(M,1).

    WORK    (workspace) DOUBLE PRECISION array, dimension (LWORK),
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
	value = 0.;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		d__1 = value, d__2 = z_abs(&a[i__ + j * a_dim1]);
		value = max(d__1,d__2);
/* L10: */
	    }
/* L20: */
	}
    } else if ((lsame_(norm, "O")) || (*(unsigned char *
	    )norm == '1')) {

/*        Find norm1(A). */

	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += z_abs(&a[i__ + j * a_dim1]);
/* L30: */
	    }
	    value = max(value,sum);
/* L40: */
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.;
/* L50: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += z_abs(&a[i__ + j * a_dim1]);
/* L60: */
	    }
/* L70: */
	}
	value = 0.;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__1 = value, d__2 = work[i__];
	    value = max(d__1,d__2);
/* L80: */
	}
    } else if ((lsame_(norm, "F")) || (lsame_(norm,
	    "E"))) {

/*        Find normF(A). */

	scale = 0.;
	sum = 1.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    zlassq_(m, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of ZLANGE */

} /* zlange_ */

doublereal zlanhe_(char *norm, char *uplo, integer *n, doublecomplex *a,
	integer *lda, doublereal *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublereal ret_val, d__1, d__2, d__3;

    /* Builtin functions */
    double z_abs(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static doublereal sum, absa, scale;
    extern logical lsame_(char *, char *);
    static doublereal value;
    extern /* Subroutine */ int zlassq_(integer *, doublecomplex *, integer *,
	     doublereal *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLANHE  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    complex hermitian matrix A.

    Description
    ===========

    ZLANHE returns the value

       ZLANHE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in ZLANHE as described
            above.

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            hermitian matrix A is to be referenced.
            = 'U':  Upper triangular part of A is referenced
            = 'L':  Lower triangular part of A is referenced

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, ZLANHE is
            set to zero.

    A       (input) COMPLEX*16 array, dimension (LDA,N)
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

    WORK    (workspace) DOUBLE PRECISION array, dimension (LWORK),
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
	value = 0.;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    d__1 = value, d__2 = z_abs(&a[i__ + j * a_dim1]);
		    value = max(d__1,d__2);
/* L10: */
		}
/* Computing MAX */
		i__2 = j + j * a_dim1;
		d__2 = value, d__3 = (d__1 = a[i__2].r, abs(d__1));
		value = max(d__2,d__3);
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
		i__2 = j + j * a_dim1;
		d__2 = value, d__3 = (d__1 = a[i__2].r, abs(d__1));
		value = max(d__2,d__3);
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    d__1 = value, d__2 = z_abs(&a[i__ + j * a_dim1]);
		    value = max(d__1,d__2);
/* L30: */
		}
/* L40: */
	    }
	}
    } else if (((lsame_(norm, "I")) || (lsame_(norm,
	    "O"))) || (*(unsigned char *)norm == '1')) {

/*        Find normI(A) ( = norm1(A), since A is hermitian). */

	value = 0.;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = z_abs(&a[i__ + j * a_dim1]);
		    sum += absa;
		    work[i__] += absa;
/* L50: */
		}
		i__2 = j + j * a_dim1;
		work[j] = sum + (d__1 = a[i__2].r, abs(d__1));
/* L60: */
	    }
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		d__1 = value, d__2 = work[i__];
		value = max(d__1,d__2);
/* L70: */
	    }
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] = 0.;
/* L80: */
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j + j * a_dim1;
		sum = work[j] + (d__1 = a[i__2].r, abs(d__1));
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = z_abs(&a[i__ + j * a_dim1]);
		    sum += absa;
		    work[i__] += absa;
/* L90: */
		}
		value = max(value,sum);
/* L100: */
	    }
	}
    } else if ((lsame_(norm, "F")) || (lsame_(norm,
	    "E"))) {

/*        Find normF(A). */

	scale = 0.;
	sum = 1.;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		i__2 = j - 1;
		zlassq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L110: */
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		zlassq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, &scale, &sum);
/* L120: */
	    }
	}
	sum *= 2;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + i__ * a_dim1;
	    if (a[i__2].r != 0.) {
		i__2 = i__ + i__ * a_dim1;
		absa = (d__1 = a[i__2].r, abs(d__1));
		if (scale < absa) {
/* Computing 2nd power */
		    d__1 = scale / absa;
		    sum = sum * (d__1 * d__1) + 1.;
		    scale = absa;
		} else {
/* Computing 2nd power */
		    d__1 = absa / scale;
		    sum += d__1 * d__1;
		}
	    }
/* L130: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of ZLANHE */

} /* zlanhe_ */

doublereal zlanhs_(char *norm, integer *n, doublecomplex *a, integer *lda,
	doublereal *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double z_abs(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    static integer i__, j;
    static doublereal sum, scale;
    extern logical lsame_(char *, char *);
    static doublereal value;
    extern /* Subroutine */ int zlassq_(integer *, doublecomplex *, integer *,
	     doublereal *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLANHS  returns the value of the one norm,  or the Frobenius norm, or
    the  infinity norm,  or the  element of  largest absolute value  of a
    Hessenberg matrix A.

    Description
    ===========

    ZLANHS returns the value

       ZLANHS = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
            Specifies the value to be returned in ZLANHS as described
            above.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.  When N = 0, ZLANHS is
            set to zero.

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            The n by n upper Hessenberg matrix A; the part of A below the
            first sub-diagonal is not referenced.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(N,1).

    WORK    (workspace) DOUBLE PRECISION array, dimension (LWORK),
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
	value = 0.;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		d__1 = value, d__2 = z_abs(&a[i__ + j * a_dim1]);
		value = max(d__1,d__2);
/* L10: */
	    }
/* L20: */
	}
    } else if ((lsame_(norm, "O")) || (*(unsigned char *
	    )norm == '1')) {

/*        Find norm1(A). */

	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.;
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += z_abs(&a[i__ + j * a_dim1]);
/* L30: */
	    }
	    value = max(value,sum);
/* L40: */
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.;
/* L50: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += z_abs(&a[i__ + j * a_dim1]);
/* L60: */
	    }
/* L70: */
	}
	value = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__1 = value, d__2 = work[i__];
	    value = max(d__1,d__2);
/* L80: */
	}
    } else if ((lsame_(norm, "F")) || (lsame_(norm,
	    "E"))) {

/*        Find normF(A). */

	scale = 0.;
	sum = 1.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	    i__3 = *n, i__4 = j + 1;
	    i__2 = min(i__3,i__4);
	    zlassq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of ZLANHS */

} /* zlanhs_ */

/* Subroutine */ int zlarcm_(integer *m, integer *n, doublereal *a, integer *
	lda, doublecomplex *b, integer *ldb, doublecomplex *c__, integer *ldc,
	 doublereal *rwork)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *,
	    integer *, doublereal *, doublereal *, integer *, doublereal *,
	    integer *, doublereal *, doublereal *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLARCM performs a very simple matrix-matrix multiplication:
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

    A       (input) DOUBLE PRECISION array, dimension (LDA, M)
            A contains the M by M matrix A.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >=max(1,M).

    B       (input) DOUBLE PRECISION array, dimension (LDB, N)
            B contains the M by N matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B. LDB >=max(1,M).

    C       (input) COMPLEX*16 array, dimension (LDC, N)
            C contains the M by N matrix C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >=max(1,M).

    RWORK   (workspace) DOUBLE PRECISION array, dimension (2*M*N)

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
    if ((*m == 0) || (*n == 0)) {
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
    dgemm_("N", "N", m, n, m, &c_b1015, &a[a_offset], lda, &rwork[1], m, &
	    c_b324, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = l + (j - 1) * *m + i__ - 1;
	    c__[i__3].r = rwork[i__4], c__[i__3].i = 0.;
/* L30: */
	}
/* L40: */
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    rwork[(j - 1) * *m + i__] = d_imag(&b[i__ + j * b_dim1]);
/* L50: */
	}
/* L60: */
    }
    dgemm_("N", "N", m, n, m, &c_b1015, &a[a_offset], lda, &rwork[1], m, &
	    c_b324, &rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = i__ + j * c_dim1;
	    d__1 = c__[i__4].r;
	    i__5 = l + (j - 1) * *m + i__ - 1;
	    z__1.r = d__1, z__1.i = rwork[i__5];
	    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L70: */
	}
/* L80: */
    }

    return 0;

/*     End of ZLARCM */

} /* zlarcm_ */

/* Subroutine */ int zlarf_(char *side, integer *m, integer *n, doublecomplex
	*v, integer *incv, doublecomplex *tau, doublecomplex *c__, integer *
	ldc, doublecomplex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    doublecomplex z__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgerc_(integer *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLARF applies a complex elementary reflector H to a complex M-by-N
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

    V       (input) COMPLEX*16 array, dimension
                       (1 + (M-1)*abs(INCV)) if SIDE = 'L'
                    or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
            The vector v in the representation of H. V is not used if
            TAU = 0.

    INCV    (input) INTEGER
            The increment between elements of v. INCV <> 0.

    TAU     (input) COMPLEX*16
            The value tau in the representation of H.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension
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

	if ((tau->r != 0.) || (tau->i != 0.)) {

/*           w := C' * v */

	    zgemv_("Conjugate transpose", m, n, &c_b60, &c__[c_offset], ldc, &
		    v[1], incv, &c_b59, &work[1], &c__1);

/*           C := C - v * w' */

	    z__1.r = -tau->r, z__1.i = -tau->i;
	    zgerc_(m, n, &z__1, &v[1], incv, &work[1], &c__1, &c__[c_offset],
		    ldc);
	}
    } else {

/*        Form  C * H */

	if ((tau->r != 0.) || (tau->i != 0.)) {

/*           w := C * v */

	    zgemv_("No transpose", m, n, &c_b60, &c__[c_offset], ldc, &v[1],
		    incv, &c_b59, &work[1], &c__1);

/*           C := C - w * v' */

	    z__1.r = -tau->r, z__1.i = -tau->i;
	    zgerc_(m, n, &z__1, &work[1], &c__1, &v[1], incv, &c__[c_offset],
		    ldc);
	}
    }
    return 0;

/*     End of ZLARF */

} /* zlarf_ */

/* Subroutine */ int zlarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, doublecomplex *v, integer
	*ldv, doublecomplex *t, integer *ldt, doublecomplex *c__, integer *
	ldc, doublecomplex *work, integer *ldwork)
{
    /* System generated locals */
    integer c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1,
	    work_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, j;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), zcopy_(integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *), ztrmm_(char *, char *,
	    char *, char *, integer *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), zlacgv_(integer *, doublecomplex *,
	    integer *);
    static char transt[1];


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLARFB applies a complex block reflector H or its transpose H' to a
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

    V       (input) COMPLEX*16 array, dimension
                                  (LDV,K) if STOREV = 'C'
                                  (LDV,M) if STOREV = 'R' and SIDE = 'L'
                                  (LDV,N) if STOREV = 'R' and SIDE = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
            if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
            if STOREV = 'R', LDV >= K.

    T       (input) COMPLEX*16 array, dimension (LDT,K)
            The triangular K-by-K matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by H*C or H'*C or C*H or C*H'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension (LDWORK,K)

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
    if ((*m <= 0) || (*n <= 0)) {
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
		    zcopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
		    zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L10: */
		}

/*              W := W * V1 */

		ztrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b60,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2 */

		    i__1 = *m - *k;
		    zgemm_("Conjugate transpose", "No transpose", n, k, &i__1,
			     &c_b60, &c__[*k + 1 + c_dim1], ldc, &v[*k + 1 +
			    v_dim1], ldv, &c_b60, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ztrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2 * W' */

		    i__1 = *m - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "Conjugate transpose", &i__1, n, k,
			     &z__1, &v[*k + 1 + v_dim1], ldv, &work[
			    work_offset], ldwork, &c_b60, &c__[*k + 1 +
			    c_dim1], ldc);
		}

/*              W := W * V1' */

		ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k,
			&c_b60, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			d_cnjg(&z__2, &work[i__ + j * work_dim1]);
			z__1.r = c__[i__4].r - z__2.r, z__1.i = c__[i__4].i -
				z__2.i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L40: */
		}

/*              W := W * V1 */

		ztrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b60,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2 */

		    i__1 = *n - *k;
		    zgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b60, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k +
			    1 + v_dim1], ldv, &c_b60, &work[work_offset],
			    ldwork);
		}

/*              W := W * T  or  W * T' */

		ztrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C2 := C2 - W * V2' */

		    i__1 = *n - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "Conjugate transpose", m, &i__1, k,
			     &z__1, &work[work_offset], ldwork, &v[*k + 1 +
			    v_dim1], ldv, &c_b60, &c__[(*k + 1) * c_dim1 + 1],
			     ldc);
		}

/*              W := W * V1' */

		ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k,
			&c_b60, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			z__1.r = c__[i__4].r - work[i__5].r, z__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
		    zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L70: */
		}

/*              W := W * V2 */

		ztrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b60,
			 &v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1 */

		    i__1 = *m - *k;
		    zgemm_("Conjugate transpose", "No transpose", n, k, &i__1,
			     &c_b60, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b60, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ztrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1 * W' */

		    i__1 = *m - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "Conjugate transpose", &i__1, n, k,
			     &z__1, &v[v_offset], ldv, &work[work_offset],
			    ldwork, &c_b60, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k,
			&c_b60, &v[*m - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = *m - *k + j + i__ * c_dim1;
			i__4 = *m - *k + j + i__ * c_dim1;
			d_cnjg(&z__2, &work[i__ + j * work_dim1]);
			z__1.r = c__[i__4].r - z__2.r, z__1.i = c__[i__4].i -
				z__2.i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L100: */
		}

/*              W := W * V2 */

		ztrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b60,
			 &v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1 */

		    i__1 = *n - *k;
		    zgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b60, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b60, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ztrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (*n > *k) {

/*                 C1 := C1 - W * V1' */

		    i__1 = *n - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "Conjugate transpose", m, &i__1, k,
			     &z__1, &work[work_offset], ldwork, &v[v_offset],
			    ldv, &c_b60, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k,
			&c_b60, &v[*n - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + (*n - *k + j) * c_dim1;
			i__4 = i__ + (*n - *k + j) * c_dim1;
			i__5 = i__ + j * work_dim1;
			z__1.r = c__[i__4].r - work[i__5].r, z__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
		    zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L130: */
		}

/*              W := W * V1' */

		ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", n, k,
			&c_b60, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C2'*V2' */

		    i__1 = *m - *k;
		    zgemm_("Conjugate transpose", "Conjugate transpose", n, k,
			     &i__1, &c_b60, &c__[*k + 1 + c_dim1], ldc, &v[(*
			    k + 1) * v_dim1 + 1], ldv, &c_b60, &work[
			    work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ztrmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2' * W' */

		    i__1 = *m - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, n, k, &z__1, &v[(*k + 1) * v_dim1 + 1], ldv,
			     &work[work_offset], ldwork, &c_b60, &c__[*k + 1
			    + c_dim1], ldc);
		}

/*              W := W * V1 */

		ztrmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b60,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			d_cnjg(&z__2, &work[i__ + j * work_dim1]);
			z__1.r = c__[i__4].r - z__2.r, z__1.i = c__[i__4].i -
				z__2.i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
/* L160: */
		}

/*              W := W * V1' */

		ztrmm_("Right", "Upper", "Conjugate transpose", "Unit", m, k,
			&c_b60, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C2 * V2' */

		    i__1 = *n - *k;
		    zgemm_("No transpose", "Conjugate transpose", m, k, &i__1,
			     &c_b60, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k
			    + 1) * v_dim1 + 1], ldv, &c_b60, &work[
			    work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ztrmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C2 := C2 - W * V2 */

		    i__1 = *n - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "No transpose", m, &i__1, k, &z__1,
			     &work[work_offset], ldwork, &v[(*k + 1) * v_dim1
			    + 1], ldv, &c_b60, &c__[(*k + 1) * c_dim1 + 1],
			    ldc);
		}

/*              W := W * V1 */

		ztrmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b60,
			 &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			z__1.r = c__[i__4].r - work[i__5].r, z__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
		    zlacgv_(n, &work[j * work_dim1 + 1], &c__1);
/* L190: */
		}

/*              W := W * V2' */

		ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", n, k,
			&c_b60, &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (*m > *k) {

/*                 W := W + C1'*V1' */

		    i__1 = *m - *k;
		    zgemm_("Conjugate transpose", "Conjugate transpose", n, k,
			     &i__1, &c_b60, &c__[c_offset], ldc, &v[v_offset],
			     ldv, &c_b60, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ztrmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (*m > *k) {

/*                 C1 := C1 - V1' * W' */

		    i__1 = *m - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, n, k, &z__1, &v[v_offset], ldv, &work[
			    work_offset], ldwork, &c_b60, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ztrmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b60,
			 &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = *m - *k + j + i__ * c_dim1;
			i__4 = *m - *k + j + i__ * c_dim1;
			d_cnjg(&z__2, &work[i__ + j * work_dim1]);
			z__1.r = c__[i__4].r - z__2.r, z__1.i = c__[i__4].i -
				z__2.i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
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
		    zcopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
/* L220: */
		}

/*              W := W * V2' */

		ztrmm_("Right", "Lower", "Conjugate transpose", "Unit", m, k,
			&c_b60, &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		if (*n > *k) {

/*                 W := W + C1 * V1' */

		    i__1 = *n - *k;
		    zgemm_("No transpose", "Conjugate transpose", m, k, &i__1,
			     &c_b60, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b60, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ztrmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b60, &t[
			t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (*n > *k) {

/*                 C1 := C1 - W * V1 */

		    i__1 = *n - *k;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "No transpose", m, &i__1, k, &z__1,
			     &work[work_offset], ldwork, &v[v_offset], ldv, &
			    c_b60, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ztrmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b60,
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
			z__1.r = c__[i__4].r - work[i__5].r, z__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L230: */
		    }
/* L240: */
		}

	    }

	}
    }

    return 0;

/*     End of ZLARFB */

} /* zlarfb_ */

/* Subroutine */ int zlarfg_(integer *n, doublecomplex *alpha, doublecomplex *
	x, integer *incx, doublecomplex *tau)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double d_imag(doublecomplex *), d_sign(doublereal *, doublereal *);

    /* Local variables */
    static integer j, knt;
    static doublereal beta, alphi, alphr;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *);
    static doublereal xnorm;
    extern doublereal dlapy3_(doublereal *, doublereal *, doublereal *),
	    dznrm2_(integer *, doublecomplex *, integer *), dlamch_(char *);
    static doublereal safmin;
    extern /* Subroutine */ int zdscal_(integer *, doublereal *,
	    doublecomplex *, integer *);
    static doublereal rsafmn;
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *,
	     doublecomplex *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLARFG generates a complex elementary reflector H of order n, such
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

    ALPHA   (input/output) COMPLEX*16
            On entry, the value alpha.
            On exit, it is overwritten with the value beta.

    X       (input/output) COMPLEX*16 array, dimension
                           (1+(N-2)*abs(INCX))
            On entry, the vector x.
            On exit, it is overwritten with the vector v.

    INCX    (input) INTEGER
            The increment between elements of X. INCX > 0.

    TAU     (output) COMPLEX*16
            The value tau.

    =====================================================================
*/


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 0) {
	tau->r = 0., tau->i = 0.;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = dznrm2_(&i__1, &x[1], incx);
    alphr = alpha->r;
    alphi = d_imag(alpha);

    if (xnorm == 0. && alphi == 0.) {

/*        H  =  I */

	tau->r = 0., tau->i = 0.;
    } else {

/*        general case */

	d__1 = dlapy3_(&alphr, &alphi, &xnorm);
	beta = -d_sign(&d__1, &alphr);
	safmin = SAFEMINIMUM / EPSILON;
	rsafmn = 1. / safmin;

	if (abs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

	    knt = 0;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    zdscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    alphi *= rsafmn;
	    alphr *= rsafmn;
	    if (abs(beta) < safmin) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = dznrm2_(&i__1, &x[1], incx);
	    z__1.r = alphr, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    d__1 = dlapy3_(&alphr, &alphi, &xnorm);
	    beta = -d_sign(&d__1, &alphr);
	    d__1 = (beta - alphr) / beta;
	    d__2 = -alphi / beta;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    z__2.r = alpha->r - beta, z__2.i = alpha->i;
	    zladiv_(&z__1, &c_b60, &z__2);
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    i__1 = *n - 1;
	    zscal_(&i__1, alpha, &x[1], incx);

/*           If ALPHA is subnormal, it may lose relative accuracy */

	    alpha->r = beta, alpha->i = 0.;
	    i__1 = knt;
	    for (j = 1; j <= i__1; ++j) {
		z__1.r = safmin * alpha->r, z__1.i = safmin * alpha->i;
		alpha->r = z__1.r, alpha->i = z__1.i;
/* L20: */
	    }
	} else {
	    d__1 = (beta - alphr) / beta;
	    d__2 = -alphi / beta;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    z__2.r = alpha->r - beta, z__2.i = alpha->i;
	    zladiv_(&z__1, &c_b60, &z__2);
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    i__1 = *n - 1;
	    zscal_(&i__1, alpha, &x[1], incx);
	    alpha->r = beta, alpha->i = 0.;
	}
    }

    return 0;

/*     End of ZLARFG */

} /* zlarfg_ */

/* Subroutine */ int zlarft_(char *direct, char *storev, integer *n, integer *
	k, doublecomplex *v, integer *ldv, doublecomplex *tau, doublecomplex *
	t, integer *ldt)
{
    /* System generated locals */
    integer t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j;
    static doublecomplex vii;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *),
	    ztrmv_(char *, char *, char *, integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *),
	    zlacgv_(integer *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLARFT forms the triangular factor T of a complex block reflector H
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

    V       (input/output) COMPLEX*16 array, dimension
                                 (LDV,K) if STOREV = 'C'
                                 (LDV,N) if STOREV = 'R'
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V.
            If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i).

    T       (output) COMPLEX*16 array, dimension (LDT,K)
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
	    if (tau[i__2].r == 0. && tau[i__2].i == 0.) {

/*              H(i)  =  I */

		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j + i__ * t_dim1;
		    t[i__3].r = 0., t[i__3].i = 0.;
/* L10: */
		}
	    } else {

/*              general case */

		i__2 = i__ + i__ * v_dim1;
		vii.r = v[i__2].r, vii.i = v[i__2].i;
		i__2 = i__ + i__ * v_dim1;
		v[i__2].r = 1., v[i__2].i = 0.;
		if (lsame_(storev, "C")) {

/*                 T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i) */

		    i__2 = *n - i__ + 1;
		    i__3 = i__ - 1;
		    i__4 = i__;
		    z__1.r = -tau[i__4].r, z__1.i = -tau[i__4].i;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &v[i__
			    + v_dim1], ldv, &v[i__ + i__ * v_dim1], &c__1, &
			    c_b59, &t[i__ * t_dim1 + 1], &c__1);
		} else {

/*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:n) * V(i,i:n)' */

		    if (i__ < *n) {
			i__2 = *n - i__;
			zlacgv_(&i__2, &v[i__ + (i__ + 1) * v_dim1], ldv);
		    }
		    i__2 = i__ - 1;
		    i__3 = *n - i__ + 1;
		    i__4 = i__;
		    z__1.r = -tau[i__4].r, z__1.i = -tau[i__4].i;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &v[i__ *
			    v_dim1 + 1], ldv, &v[i__ + i__ * v_dim1], ldv, &
			    c_b59, &t[i__ * t_dim1 + 1], &c__1);
		    if (i__ < *n) {
			i__2 = *n - i__;
			zlacgv_(&i__2, &v[i__ + (i__ + 1) * v_dim1], ldv);
		    }
		}
		i__2 = i__ + i__ * v_dim1;
		v[i__2].r = vii.r, v[i__2].i = vii.i;

/*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i) */

		i__2 = i__ - 1;
		ztrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
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
	    if (tau[i__1].r == 0. && tau[i__1].i == 0.) {

/*              H(i)  =  I */

		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    i__2 = j + i__ * t_dim1;
		    t[i__2].r = 0., t[i__2].i = 0.;
/* L30: */
		}
	    } else {

/*              general case */

		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			i__1 = *n - *k + i__ + i__ * v_dim1;
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = *n - *k + i__ + i__ * v_dim1;
			v[i__1].r = 1., v[i__1].i = 0.;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(1:n-k+i,i+1:k)' * V(1:n-k+i,i)
*/

			i__1 = *n - *k + i__;
			i__2 = *k - i__;
			i__3 = i__;
			z__1.r = -tau[i__3].r, z__1.i = -tau[i__3].i;
			zgemv_("Conjugate transpose", &i__1, &i__2, &z__1, &v[
				(i__ + 1) * v_dim1 + 1], ldv, &v[i__ * v_dim1
				+ 1], &c__1, &c_b59, &t[i__ + 1 + i__ *
				t_dim1], &c__1);
			i__1 = *n - *k + i__ + i__ * v_dim1;
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    } else {
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			vii.r = v[i__1].r, vii.i = v[i__1].i;
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			v[i__1].r = 1., v[i__1].i = 0.;

/*
                      T(i+1:k,i) :=
                              - tau(i) * V(i+1:k,1:n-k+i) * V(i,1:n-k+i)'
*/

			i__1 = *n - *k + i__ - 1;
			zlacgv_(&i__1, &v[i__ + v_dim1], ldv);
			i__1 = *k - i__;
			i__2 = *n - *k + i__;
			i__3 = i__;
			z__1.r = -tau[i__3].r, z__1.i = -tau[i__3].i;
			zgemv_("No transpose", &i__1, &i__2, &z__1, &v[i__ +
				1 + v_dim1], ldv, &v[i__ + v_dim1], ldv, &
				c_b59, &t[i__ + 1 + i__ * t_dim1], &c__1);
			i__1 = *n - *k + i__ - 1;
			zlacgv_(&i__1, &v[i__ + v_dim1], ldv);
			i__1 = i__ + (*n - *k + i__) * v_dim1;
			v[i__1].r = vii.r, v[i__1].i = vii.i;
		    }

/*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i) */

		    i__1 = *k - i__;
		    ztrmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__
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

/*     End of ZLARFT */

} /* zlarft_ */

/* Subroutine */ int zlarfx_(char *side, integer *m, integer *n,
	doublecomplex *v, doublecomplex *tau, doublecomplex *c__, integer *
	ldc, doublecomplex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8,
	    i__9, i__10, i__11;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7, z__8, z__9, z__10,
	     z__11, z__12, z__13, z__14, z__15, z__16, z__17, z__18, z__19;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer j;
    static doublecomplex t1, t2, t3, t4, t5, t6, t7, t8, t9, v1, v2, v3, v4,
	    v5, v6, v7, v8, v9, t10, v10, sum;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgerc_(integer *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLARFX applies a complex elementary reflector H to a complex m by n
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

    V       (input) COMPLEX*16 array, dimension (M) if SIDE = 'L'
                                          or (N) if SIDE = 'R'
            The vector v in the representation of H.

    TAU     (input) COMPLEX*16
            The value tau in the representation of H.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by the matrix H * C if SIDE = 'L',
            or C * H if SIDE = 'R'.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension (N) if SIDE = 'L'
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
    if (tau->r == 0. && tau->i == 0.) {
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

	zgemv_("Conjugate transpose", m, n, &c_b60, &c__[c_offset], ldc, &v[1]
		, &c__1, &c_b59, &work[1], &c__1);

/*        C := C - tau * v * w' */

	z__1.r = -tau->r, z__1.i = -tau->i;
	zgerc_(m, n, &z__1, &v[1], &c__1, &work[1], &c__1, &c__[c_offset],
		ldc);
	goto L410;
L10:

/*        Special code for 1 x 1 Householder */

	z__3.r = tau->r * v[1].r - tau->i * v[1].i, z__3.i = tau->r * v[1].i
		+ tau->i * v[1].r;
	d_cnjg(&z__4, &v[1]);
	z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i = z__3.r * z__4.i
		+ z__3.i * z__4.r;
	z__1.r = 1. - z__2.r, z__1.i = 0. - z__2.i;
	t1.r = z__1.r, t1.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__1.r = t1.r * c__[i__3].r - t1.i * c__[i__3].i, z__1.i = t1.r *
		    c__[i__3].i + t1.i * c__[i__3].r;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L20: */
	}
	goto L410;
L30:

/*        Special code for 2 x 2 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__2.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__2.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__3.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__3.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L40: */
	}
	goto L410;
L50:

/*        Special code for 3 x 3 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__3.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__3.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__4.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__4.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__2.r = z__3.r + z__4.r, z__2.i = z__3.i + z__4.i;
	    i__4 = j * c_dim1 + 3;
	    z__5.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__5.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i + z__5.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L60: */
	}
	goto L410;
L70:

/*        Special code for 4 x 4 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__4.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__4.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__5.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__5.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__3.r = z__4.r + z__5.r, z__3.i = z__4.i + z__5.i;
	    i__4 = j * c_dim1 + 3;
	    z__6.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__6.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__2.r = z__3.r + z__6.r, z__2.i = z__3.i + z__6.i;
	    i__5 = j * c_dim1 + 4;
	    z__7.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__7.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__1.r = z__2.r + z__7.r, z__1.i = z__2.i + z__7.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L80: */
	}
	goto L410;
L90:

/*        Special code for 5 x 5 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__5.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__5.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__6.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__6.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__4.r = z__5.r + z__6.r, z__4.i = z__5.i + z__6.i;
	    i__4 = j * c_dim1 + 3;
	    z__7.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__7.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__3.r = z__4.r + z__7.r, z__3.i = z__4.i + z__7.i;
	    i__5 = j * c_dim1 + 4;
	    z__8.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__8.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__2.r = z__3.r + z__8.r, z__2.i = z__3.i + z__8.i;
	    i__6 = j * c_dim1 + 5;
	    z__9.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__9.i = v5.r *
		    c__[i__6].i + v5.i * c__[i__6].r;
	    z__1.r = z__2.r + z__9.r, z__1.i = z__2.i + z__9.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L100: */
	}
	goto L410;
L110:

/*        Special code for 6 x 6 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	d_cnjg(&z__1, &v[6]);
	v6.r = z__1.r, v6.i = z__1.i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__6.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__6.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__7.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__7.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__5.r = z__6.r + z__7.r, z__5.i = z__6.i + z__7.i;
	    i__4 = j * c_dim1 + 3;
	    z__8.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__8.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__4.r = z__5.r + z__8.r, z__4.i = z__5.i + z__8.i;
	    i__5 = j * c_dim1 + 4;
	    z__9.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__9.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__3.r = z__4.r + z__9.r, z__3.i = z__4.i + z__9.i;
	    i__6 = j * c_dim1 + 5;
	    z__10.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__10.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__2.r = z__3.r + z__10.r, z__2.i = z__3.i + z__10.i;
	    i__7 = j * c_dim1 + 6;
	    z__11.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__11.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__1.r = z__2.r + z__11.r, z__1.i = z__2.i + z__11.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L120: */
	}
	goto L410;
L130:

/*        Special code for 7 x 7 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	d_cnjg(&z__1, &v[6]);
	v6.r = z__1.r, v6.i = z__1.i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	d_cnjg(&z__1, &v[7]);
	v7.r = z__1.r, v7.i = z__1.i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__7.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__7.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__8.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__8.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__6.r = z__7.r + z__8.r, z__6.i = z__7.i + z__8.i;
	    i__4 = j * c_dim1 + 3;
	    z__9.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__9.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__5.r = z__6.r + z__9.r, z__5.i = z__6.i + z__9.i;
	    i__5 = j * c_dim1 + 4;
	    z__10.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__10.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__4.r = z__5.r + z__10.r, z__4.i = z__5.i + z__10.i;
	    i__6 = j * c_dim1 + 5;
	    z__11.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__11.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__3.r = z__4.r + z__11.r, z__3.i = z__4.i + z__11.i;
	    i__7 = j * c_dim1 + 6;
	    z__12.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__12.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__2.r = z__3.r + z__12.r, z__2.i = z__3.i + z__12.i;
	    i__8 = j * c_dim1 + 7;
	    z__13.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__13.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__1.r = z__2.r + z__13.r, z__1.i = z__2.i + z__13.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L140: */
	}
	goto L410;
L150:

/*        Special code for 8 x 8 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	d_cnjg(&z__1, &v[6]);
	v6.r = z__1.r, v6.i = z__1.i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	d_cnjg(&z__1, &v[7]);
	v7.r = z__1.r, v7.i = z__1.i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	d_cnjg(&z__1, &v[8]);
	v8.r = z__1.r, v8.i = z__1.i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__8.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__8.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__9.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__9.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__7.r = z__8.r + z__9.r, z__7.i = z__8.i + z__9.i;
	    i__4 = j * c_dim1 + 3;
	    z__10.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__10.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__6.r = z__7.r + z__10.r, z__6.i = z__7.i + z__10.i;
	    i__5 = j * c_dim1 + 4;
	    z__11.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__11.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__5.r = z__6.r + z__11.r, z__5.i = z__6.i + z__11.i;
	    i__6 = j * c_dim1 + 5;
	    z__12.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__12.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__4.r = z__5.r + z__12.r, z__4.i = z__5.i + z__12.i;
	    i__7 = j * c_dim1 + 6;
	    z__13.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__13.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__3.r = z__4.r + z__13.r, z__3.i = z__4.i + z__13.i;
	    i__8 = j * c_dim1 + 7;
	    z__14.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__14.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__2.r = z__3.r + z__14.r, z__2.i = z__3.i + z__14.i;
	    i__9 = j * c_dim1 + 8;
	    z__15.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__15.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__1.r = z__2.r + z__15.r, z__1.i = z__2.i + z__15.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L160: */
	}
	goto L410;
L170:

/*        Special code for 9 x 9 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	d_cnjg(&z__1, &v[6]);
	v6.r = z__1.r, v6.i = z__1.i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	d_cnjg(&z__1, &v[7]);
	v7.r = z__1.r, v7.i = z__1.i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	d_cnjg(&z__1, &v[8]);
	v8.r = z__1.r, v8.i = z__1.i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	d_cnjg(&z__1, &v[9]);
	v9.r = z__1.r, v9.i = z__1.i;
	d_cnjg(&z__2, &v9);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t9.r = z__1.r, t9.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__9.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__9.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__10.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__10.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    z__8.r = z__9.r + z__10.r, z__8.i = z__9.i + z__10.i;
	    i__4 = j * c_dim1 + 3;
	    z__11.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__11.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__7.r = z__8.r + z__11.r, z__7.i = z__8.i + z__11.i;
	    i__5 = j * c_dim1 + 4;
	    z__12.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__12.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__6.r = z__7.r + z__12.r, z__6.i = z__7.i + z__12.i;
	    i__6 = j * c_dim1 + 5;
	    z__13.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__13.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__5.r = z__6.r + z__13.r, z__5.i = z__6.i + z__13.i;
	    i__7 = j * c_dim1 + 6;
	    z__14.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__14.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__4.r = z__5.r + z__14.r, z__4.i = z__5.i + z__14.i;
	    i__8 = j * c_dim1 + 7;
	    z__15.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__15.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__3.r = z__4.r + z__15.r, z__3.i = z__4.i + z__15.i;
	    i__9 = j * c_dim1 + 8;
	    z__16.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__16.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__2.r = z__3.r + z__16.r, z__2.i = z__3.i + z__16.i;
	    i__10 = j * c_dim1 + 9;
	    z__17.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, z__17.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    z__1.r = z__2.r + z__17.r, z__1.i = z__2.i + z__17.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 9;
	    i__3 = j * c_dim1 + 9;
	    z__2.r = sum.r * t9.r - sum.i * t9.i, z__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L180: */
	}
	goto L410;
L190:

/*        Special code for 10 x 10 Householder */

	d_cnjg(&z__1, &v[1]);
	v1.r = z__1.r, v1.i = z__1.i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	d_cnjg(&z__1, &v[2]);
	v2.r = z__1.r, v2.i = z__1.i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__1, &v[3]);
	v3.r = z__1.r, v3.i = z__1.i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__1, &v[4]);
	v4.r = z__1.r, v4.i = z__1.i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	d_cnjg(&z__1, &v[5]);
	v5.r = z__1.r, v5.i = z__1.i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	d_cnjg(&z__1, &v[6]);
	v6.r = z__1.r, v6.i = z__1.i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	d_cnjg(&z__1, &v[7]);
	v7.r = z__1.r, v7.i = z__1.i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	d_cnjg(&z__1, &v[8]);
	v8.r = z__1.r, v8.i = z__1.i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	d_cnjg(&z__1, &v[9]);
	v9.r = z__1.r, v9.i = z__1.i;
	d_cnjg(&z__2, &v9);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t9.r = z__1.r, t9.i = z__1.i;
	d_cnjg(&z__1, &v[10]);
	v10.r = z__1.r, v10.i = z__1.i;
	d_cnjg(&z__2, &v10);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t10.r = z__1.r, t10.i = z__1.i;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j * c_dim1 + 1;
	    z__10.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__10.i = v1.r
		    * c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j * c_dim1 + 2;
	    z__11.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__11.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    z__9.r = z__10.r + z__11.r, z__9.i = z__10.i + z__11.i;
	    i__4 = j * c_dim1 + 3;
	    z__12.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__12.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__8.r = z__9.r + z__12.r, z__8.i = z__9.i + z__12.i;
	    i__5 = j * c_dim1 + 4;
	    z__13.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__13.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__7.r = z__8.r + z__13.r, z__7.i = z__8.i + z__13.i;
	    i__6 = j * c_dim1 + 5;
	    z__14.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__14.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__6.r = z__7.r + z__14.r, z__6.i = z__7.i + z__14.i;
	    i__7 = j * c_dim1 + 6;
	    z__15.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__15.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__5.r = z__6.r + z__15.r, z__5.i = z__6.i + z__15.i;
	    i__8 = j * c_dim1 + 7;
	    z__16.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__16.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__4.r = z__5.r + z__16.r, z__4.i = z__5.i + z__16.i;
	    i__9 = j * c_dim1 + 8;
	    z__17.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__17.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__3.r = z__4.r + z__17.r, z__3.i = z__4.i + z__17.i;
	    i__10 = j * c_dim1 + 9;
	    z__18.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, z__18.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    z__2.r = z__3.r + z__18.r, z__2.i = z__3.i + z__18.i;
	    i__11 = j * c_dim1 + 10;
	    z__19.r = v10.r * c__[i__11].r - v10.i * c__[i__11].i, z__19.i =
		    v10.r * c__[i__11].i + v10.i * c__[i__11].r;
	    z__1.r = z__2.r + z__19.r, z__1.i = z__2.i + z__19.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j * c_dim1 + 1;
	    i__3 = j * c_dim1 + 1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 2;
	    i__3 = j * c_dim1 + 2;
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 3;
	    i__3 = j * c_dim1 + 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 4;
	    i__3 = j * c_dim1 + 4;
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 5;
	    i__3 = j * c_dim1 + 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 6;
	    i__3 = j * c_dim1 + 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 7;
	    i__3 = j * c_dim1 + 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 8;
	    i__3 = j * c_dim1 + 8;
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 9;
	    i__3 = j * c_dim1 + 9;
	    z__2.r = sum.r * t9.r - sum.i * t9.i, z__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j * c_dim1 + 10;
	    i__3 = j * c_dim1 + 10;
	    z__2.r = sum.r * t10.r - sum.i * t10.i, z__2.i = sum.r * t10.i +
		    sum.i * t10.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
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

	zgemv_("No transpose", m, n, &c_b60, &c__[c_offset], ldc, &v[1], &
		c__1, &c_b59, &work[1], &c__1);

/*        C := C - tau * w * v' */

	z__1.r = -tau->r, z__1.i = -tau->i;
	zgerc_(m, n, &z__1, &work[1], &c__1, &v[1], &c__1, &c__[c_offset],
		ldc);
	goto L410;
L210:

/*        Special code for 1 x 1 Householder */

	z__3.r = tau->r * v[1].r - tau->i * v[1].i, z__3.i = tau->r * v[1].i
		+ tau->i * v[1].r;
	d_cnjg(&z__4, &v[1]);
	z__2.r = z__3.r * z__4.r - z__3.i * z__4.i, z__2.i = z__3.r * z__4.i
		+ z__3.i * z__4.r;
	z__1.r = 1. - z__2.r, z__1.i = 0. - z__2.i;
	t1.r = z__1.r, t1.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__1.r = t1.r * c__[i__3].r - t1.i * c__[i__3].i, z__1.i = t1.r *
		    c__[i__3].i + t1.i * c__[i__3].r;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L220: */
	}
	goto L410;
L230:

/*        Special code for 2 x 2 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__2.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__2.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__3.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__3.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L240: */
	}
	goto L410;
L250:

/*        Special code for 3 x 3 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__3.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__3.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__4.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__4.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__2.r = z__3.r + z__4.r, z__2.i = z__3.i + z__4.i;
	    i__4 = j + c_dim1 * 3;
	    z__5.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__5.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i + z__5.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L260: */
	}
	goto L410;
L270:

/*        Special code for 4 x 4 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__4.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__4.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__5.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__5.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__3.r = z__4.r + z__5.r, z__3.i = z__4.i + z__5.i;
	    i__4 = j + c_dim1 * 3;
	    z__6.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__6.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__2.r = z__3.r + z__6.r, z__2.i = z__3.i + z__6.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__7.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__7.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__1.r = z__2.r + z__7.r, z__1.i = z__2.i + z__7.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L280: */
	}
	goto L410;
L290:

/*        Special code for 5 x 5 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__5.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__5.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__6.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__6.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__4.r = z__5.r + z__6.r, z__4.i = z__5.i + z__6.i;
	    i__4 = j + c_dim1 * 3;
	    z__7.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__7.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__3.r = z__4.r + z__7.r, z__3.i = z__4.i + z__7.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__8.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__8.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__2.r = z__3.r + z__8.r, z__2.i = z__3.i + z__8.i;
	    i__6 = j + c_dim1 * 5;
	    z__9.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__9.i = v5.r *
		    c__[i__6].i + v5.i * c__[i__6].r;
	    z__1.r = z__2.r + z__9.r, z__1.i = z__2.i + z__9.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L300: */
	}
	goto L410;
L310:

/*        Special code for 6 x 6 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__6.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__6.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__7.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__7.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__5.r = z__6.r + z__7.r, z__5.i = z__6.i + z__7.i;
	    i__4 = j + c_dim1 * 3;
	    z__8.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__8.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__4.r = z__5.r + z__8.r, z__4.i = z__5.i + z__8.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__9.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__9.i = v4.r *
		    c__[i__5].i + v4.i * c__[i__5].r;
	    z__3.r = z__4.r + z__9.r, z__3.i = z__4.i + z__9.i;
	    i__6 = j + c_dim1 * 5;
	    z__10.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__10.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__2.r = z__3.r + z__10.r, z__2.i = z__3.i + z__10.i;
	    i__7 = j + c_dim1 * 6;
	    z__11.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__11.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__1.r = z__2.r + z__11.r, z__1.i = z__2.i + z__11.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L320: */
	}
	goto L410;
L330:

/*        Special code for 7 x 7 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__7.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__7.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__8.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__8.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__6.r = z__7.r + z__8.r, z__6.i = z__7.i + z__8.i;
	    i__4 = j + c_dim1 * 3;
	    z__9.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__9.i = v3.r *
		    c__[i__4].i + v3.i * c__[i__4].r;
	    z__5.r = z__6.r + z__9.r, z__5.i = z__6.i + z__9.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__10.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__10.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__4.r = z__5.r + z__10.r, z__4.i = z__5.i + z__10.i;
	    i__6 = j + c_dim1 * 5;
	    z__11.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__11.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__3.r = z__4.r + z__11.r, z__3.i = z__4.i + z__11.i;
	    i__7 = j + c_dim1 * 6;
	    z__12.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__12.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__2.r = z__3.r + z__12.r, z__2.i = z__3.i + z__12.i;
	    i__8 = j + c_dim1 * 7;
	    z__13.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__13.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__1.r = z__2.r + z__13.r, z__1.i = z__2.i + z__13.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L340: */
	}
	goto L410;
L350:

/*        Special code for 8 x 8 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__8.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__8.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__9.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__9.i = v2.r *
		    c__[i__3].i + v2.i * c__[i__3].r;
	    z__7.r = z__8.r + z__9.r, z__7.i = z__8.i + z__9.i;
	    i__4 = j + c_dim1 * 3;
	    z__10.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__10.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__6.r = z__7.r + z__10.r, z__6.i = z__7.i + z__10.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__11.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__11.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__5.r = z__6.r + z__11.r, z__5.i = z__6.i + z__11.i;
	    i__6 = j + c_dim1 * 5;
	    z__12.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__12.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__4.r = z__5.r + z__12.r, z__4.i = z__5.i + z__12.i;
	    i__7 = j + c_dim1 * 6;
	    z__13.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__13.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__3.r = z__4.r + z__13.r, z__3.i = z__4.i + z__13.i;
	    i__8 = j + c_dim1 * 7;
	    z__14.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__14.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__2.r = z__3.r + z__14.r, z__2.i = z__3.i + z__14.i;
	    i__9 = j + ((c_dim1) << (3));
	    z__15.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__15.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__1.r = z__2.r + z__15.r, z__1.i = z__2.i + z__15.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (3));
	    i__3 = j + ((c_dim1) << (3));
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L360: */
	}
	goto L410;
L370:

/*        Special code for 9 x 9 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	v9.r = v[9].r, v9.i = v[9].i;
	d_cnjg(&z__2, &v9);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t9.r = z__1.r, t9.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__9.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__9.i = v1.r *
		    c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__10.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__10.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    z__8.r = z__9.r + z__10.r, z__8.i = z__9.i + z__10.i;
	    i__4 = j + c_dim1 * 3;
	    z__11.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__11.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__7.r = z__8.r + z__11.r, z__7.i = z__8.i + z__11.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__12.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__12.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__6.r = z__7.r + z__12.r, z__6.i = z__7.i + z__12.i;
	    i__6 = j + c_dim1 * 5;
	    z__13.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__13.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__5.r = z__6.r + z__13.r, z__5.i = z__6.i + z__13.i;
	    i__7 = j + c_dim1 * 6;
	    z__14.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__14.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__4.r = z__5.r + z__14.r, z__4.i = z__5.i + z__14.i;
	    i__8 = j + c_dim1 * 7;
	    z__15.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__15.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__3.r = z__4.r + z__15.r, z__3.i = z__4.i + z__15.i;
	    i__9 = j + ((c_dim1) << (3));
	    z__16.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__16.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__2.r = z__3.r + z__16.r, z__2.i = z__3.i + z__16.i;
	    i__10 = j + c_dim1 * 9;
	    z__17.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, z__17.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    z__1.r = z__2.r + z__17.r, z__1.i = z__2.i + z__17.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (3));
	    i__3 = j + ((c_dim1) << (3));
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 9;
	    i__3 = j + c_dim1 * 9;
	    z__2.r = sum.r * t9.r - sum.i * t9.i, z__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L380: */
	}
	goto L410;
L390:

/*        Special code for 10 x 10 Householder */

	v1.r = v[1].r, v1.i = v[1].i;
	d_cnjg(&z__2, &v1);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t1.r = z__1.r, t1.i = z__1.i;
	v2.r = v[2].r, v2.i = v[2].i;
	d_cnjg(&z__2, &v2);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t2.r = z__1.r, t2.i = z__1.i;
	v3.r = v[3].r, v3.i = v[3].i;
	d_cnjg(&z__2, &v3);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t3.r = z__1.r, t3.i = z__1.i;
	v4.r = v[4].r, v4.i = v[4].i;
	d_cnjg(&z__2, &v4);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t4.r = z__1.r, t4.i = z__1.i;
	v5.r = v[5].r, v5.i = v[5].i;
	d_cnjg(&z__2, &v5);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t5.r = z__1.r, t5.i = z__1.i;
	v6.r = v[6].r, v6.i = v[6].i;
	d_cnjg(&z__2, &v6);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t6.r = z__1.r, t6.i = z__1.i;
	v7.r = v[7].r, v7.i = v[7].i;
	d_cnjg(&z__2, &v7);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t7.r = z__1.r, t7.i = z__1.i;
	v8.r = v[8].r, v8.i = v[8].i;
	d_cnjg(&z__2, &v8);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t8.r = z__1.r, t8.i = z__1.i;
	v9.r = v[9].r, v9.i = v[9].i;
	d_cnjg(&z__2, &v9);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t9.r = z__1.r, t9.i = z__1.i;
	v10.r = v[10].r, v10.i = v[10].i;
	d_cnjg(&z__2, &v10);
	z__1.r = tau->r * z__2.r - tau->i * z__2.i, z__1.i = tau->r * z__2.i
		+ tau->i * z__2.r;
	t10.r = z__1.r, t10.i = z__1.i;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + c_dim1;
	    z__10.r = v1.r * c__[i__2].r - v1.i * c__[i__2].i, z__10.i = v1.r
		    * c__[i__2].i + v1.i * c__[i__2].r;
	    i__3 = j + ((c_dim1) << (1));
	    z__11.r = v2.r * c__[i__3].r - v2.i * c__[i__3].i, z__11.i = v2.r
		    * c__[i__3].i + v2.i * c__[i__3].r;
	    z__9.r = z__10.r + z__11.r, z__9.i = z__10.i + z__11.i;
	    i__4 = j + c_dim1 * 3;
	    z__12.r = v3.r * c__[i__4].r - v3.i * c__[i__4].i, z__12.i = v3.r
		    * c__[i__4].i + v3.i * c__[i__4].r;
	    z__8.r = z__9.r + z__12.r, z__8.i = z__9.i + z__12.i;
	    i__5 = j + ((c_dim1) << (2));
	    z__13.r = v4.r * c__[i__5].r - v4.i * c__[i__5].i, z__13.i = v4.r
		    * c__[i__5].i + v4.i * c__[i__5].r;
	    z__7.r = z__8.r + z__13.r, z__7.i = z__8.i + z__13.i;
	    i__6 = j + c_dim1 * 5;
	    z__14.r = v5.r * c__[i__6].r - v5.i * c__[i__6].i, z__14.i = v5.r
		    * c__[i__6].i + v5.i * c__[i__6].r;
	    z__6.r = z__7.r + z__14.r, z__6.i = z__7.i + z__14.i;
	    i__7 = j + c_dim1 * 6;
	    z__15.r = v6.r * c__[i__7].r - v6.i * c__[i__7].i, z__15.i = v6.r
		    * c__[i__7].i + v6.i * c__[i__7].r;
	    z__5.r = z__6.r + z__15.r, z__5.i = z__6.i + z__15.i;
	    i__8 = j + c_dim1 * 7;
	    z__16.r = v7.r * c__[i__8].r - v7.i * c__[i__8].i, z__16.i = v7.r
		    * c__[i__8].i + v7.i * c__[i__8].r;
	    z__4.r = z__5.r + z__16.r, z__4.i = z__5.i + z__16.i;
	    i__9 = j + ((c_dim1) << (3));
	    z__17.r = v8.r * c__[i__9].r - v8.i * c__[i__9].i, z__17.i = v8.r
		    * c__[i__9].i + v8.i * c__[i__9].r;
	    z__3.r = z__4.r + z__17.r, z__3.i = z__4.i + z__17.i;
	    i__10 = j + c_dim1 * 9;
	    z__18.r = v9.r * c__[i__10].r - v9.i * c__[i__10].i, z__18.i =
		    v9.r * c__[i__10].i + v9.i * c__[i__10].r;
	    z__2.r = z__3.r + z__18.r, z__2.i = z__3.i + z__18.i;
	    i__11 = j + c_dim1 * 10;
	    z__19.r = v10.r * c__[i__11].r - v10.i * c__[i__11].i, z__19.i =
		    v10.r * c__[i__11].i + v10.i * c__[i__11].r;
	    z__1.r = z__2.r + z__19.r, z__1.i = z__2.i + z__19.i;
	    sum.r = z__1.r, sum.i = z__1.i;
	    i__2 = j + c_dim1;
	    i__3 = j + c_dim1;
	    z__2.r = sum.r * t1.r - sum.i * t1.i, z__2.i = sum.r * t1.i +
		    sum.i * t1.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (1));
	    i__3 = j + ((c_dim1) << (1));
	    z__2.r = sum.r * t2.r - sum.i * t2.i, z__2.i = sum.r * t2.i +
		    sum.i * t2.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 3;
	    i__3 = j + c_dim1 * 3;
	    z__2.r = sum.r * t3.r - sum.i * t3.i, z__2.i = sum.r * t3.i +
		    sum.i * t3.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (2));
	    i__3 = j + ((c_dim1) << (2));
	    z__2.r = sum.r * t4.r - sum.i * t4.i, z__2.i = sum.r * t4.i +
		    sum.i * t4.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 5;
	    i__3 = j + c_dim1 * 5;
	    z__2.r = sum.r * t5.r - sum.i * t5.i, z__2.i = sum.r * t5.i +
		    sum.i * t5.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 6;
	    i__3 = j + c_dim1 * 6;
	    z__2.r = sum.r * t6.r - sum.i * t6.i, z__2.i = sum.r * t6.i +
		    sum.i * t6.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 7;
	    i__3 = j + c_dim1 * 7;
	    z__2.r = sum.r * t7.r - sum.i * t7.i, z__2.i = sum.r * t7.i +
		    sum.i * t7.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + ((c_dim1) << (3));
	    i__3 = j + ((c_dim1) << (3));
	    z__2.r = sum.r * t8.r - sum.i * t8.i, z__2.i = sum.r * t8.i +
		    sum.i * t8.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 9;
	    i__3 = j + c_dim1 * 9;
	    z__2.r = sum.r * t9.r - sum.i * t9.i, z__2.i = sum.r * t9.i +
		    sum.i * t9.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
	    i__2 = j + c_dim1 * 10;
	    i__3 = j + c_dim1 * 10;
	    z__2.r = sum.r * t10.r - sum.i * t10.i, z__2.i = sum.r * t10.i +
		    sum.i * t10.r;
	    z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
	    c__[i__2].r = z__1.r, c__[i__2].i = z__1.i;
/* L400: */
	}
	goto L410;
    }
L410:
    return 0;

/*     End of ZLARFX */

} /* zlarfx_ */

/* Subroutine */ int zlascl_(char *type__, integer *kl, integer *ku,
	doublereal *cfrom, doublereal *cto, integer *m, integer *n,
	doublecomplex *a, integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, k1, k2, k3, k4;
    static doublereal mul, cto1;
    static logical done;
    static doublereal ctoc;
    extern logical lsame_(char *, char *);
    static integer itype;
    static doublereal cfrom1;

    static doublereal cfromc;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static doublereal bignum, smlnum;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       February 29, 1992


    Purpose
    =======

    ZLASCL multiplies the M by N complex matrix A by the real scalar
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

    CFROM   (input) DOUBLE PRECISION
    CTO     (input) DOUBLE PRECISION
            The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
            without over/underflow if the final result CTO*A(I,J)/CFROM
            can be represented without over/underflow.  CFROM must be
            nonzero.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,M)
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
    } else if (*cfrom == 0.) {
	*info = -4;
    } else if (*m < 0) {
	*info = -6;
    } else if (((*n < 0) || (itype == 4 && *n != *m)) || (itype == 5 && *n !=
	    *m)) {
	*info = -7;
    } else if (itype <= 3 && *lda < max(1,*m)) {
	*info = -9;
    } else if (itype >= 4) {
/* Computing MAX */
	i__1 = *m - 1;
	if ((*kl < 0) || (*kl > max(i__1,0))) {
	    *info = -2;
	} else /* if(complicated condition) */ {
/* Computing MAX */
	    i__1 = *n - 1;
	    if (((*ku < 0) || (*ku > max(i__1,0))) || (((itype == 4) || (
		    itype == 5)) && *kl != *ku)) {
		*info = -3;
	    } else if (((itype == 4 && *lda < *kl + 1) || (itype == 5 && *lda
		    < *ku + 1)) || (itype == 6 && *lda < ((*kl) << (1)) + *ku
		    + 1)) {
		*info = -9;
	    }
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLASCL", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if ((*n == 0) || (*m == 0)) {
	return 0;
    }

/*     Get machine parameters */

    smlnum = SAFEMINIMUM;
    bignum = 1. / smlnum;

    cfromc = *cfrom;
    ctoc = *cto;

L10:
    cfrom1 = cfromc * smlnum;
    cto1 = ctoc / bignum;
    if (abs(cfrom1) > abs(ctoc) && ctoc != 0.) {
	mul = smlnum;
	done = FALSE_;
	cfromc = cfrom1;
    } else if (abs(cto1) > abs(cfromc)) {
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__2].r = z__1.r, a[i__2].i = z__1.i;
/* L120: */
	    }
/* L130: */
	}

    } else if (itype == 6) {

/*        Band matrix */

	k1 = *kl + *ku + 2;
	k2 = *kl + 1;
	k3 = ((*kl) << (1)) + *ku + 1;
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
		z__1.r = mul * a[i__4].r, z__1.i = mul * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L140: */
	    }
/* L150: */
	}

    }

    if (! done) {
	goto L10;
    }

    return 0;

/*     End of ZLASCL */

} /* zlascl_ */

/* Subroutine */ int zlaset_(char *uplo, integer *m, integer *n,
	doublecomplex *alpha, doublecomplex *beta, doublecomplex *a, integer *
	lda)
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

    ZLASET initializes a 2-D array A to BETA on the diagonal and
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

    ALPHA   (input) COMPLEX*16
            All the offdiagonal array elements are set to ALPHA.

    BETA    (input) COMPLEX*16
            All the diagonal array elements are set to BETA.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

/*     End of ZLASET */

} /* zlaset_ */

/* Subroutine */ int zlasr_(char *side, char *pivot, char *direct, integer *m,
	 integer *n, doublereal *c__, doublereal *s, doublecomplex *a,
	integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    static integer i__, j, info;
    static doublecomplex temp;
    extern logical lsame_(char *, char *);
    static doublereal ctemp, stemp;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       October 31, 1992


    Purpose
    =======

    ZLASR   performs the transformation

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

    C, S    (input) DOUBLE PRECISION arrays, dimension
                    (M-1) if SIDE = 'L'
                    (N-1) if SIDE = 'R'
            c(k) and s(k) contain the cosine and sine that define the
            matrix P(k).  The two by two plane rotation part of the
            matrix P(k), R(k), is assumed to be of the form
            R( k ) = (  c( k )  s( k ) ).
                     ( -s( k )  c( k ) )

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
    if (! ((lsame_(side, "L")) || (lsame_(side, "R")))) {
	info = 1;
    } else if (! (((lsame_(pivot, "V")) || (lsame_(
	    pivot, "T"))) || (lsame_(pivot, "B")))) {
	info = 2;
    } else if (! ((lsame_(direct, "F")) || (lsame_(
	    direct, "B")))) {
	info = 3;
    } else if (*m < 0) {
	info = 4;
    } else if (*n < 0) {
	info = 5;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("ZLASR ", &info);
	return 0;
    }

/*     Quick return if possible */

    if ((*m == 0) || (*n == 0)) {
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = j + 1 + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + 1 + i__ * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__4 = j + i__ * a_dim1;
			    z__3.r = stemp * a[i__4].r, z__3.i = stemp * a[
				    i__4].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = j + i__ * a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__4 = j + i__ * a_dim1;
			    z__3.r = ctemp * a[i__4].r, z__3.i = ctemp * a[
				    i__4].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L10: */
			}
		    }
/* L20: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = j + 1 + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + 1 + i__ * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__3 = j + i__ * a_dim1;
			    z__3.r = stemp * a[i__3].r, z__3.i = stemp * a[
				    i__3].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = j + i__ * a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__3 = j + i__ * a_dim1;
			    z__3.r = ctemp * a[i__3].r, z__3.i = ctemp * a[
				    i__3].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = j + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + i__ * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__4 = i__ * a_dim1 + 1;
			    z__3.r = stemp * a[i__4].r, z__3.i = stemp * a[
				    i__4].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = i__ * a_dim1 + 1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__4 = i__ * a_dim1 + 1;
			    z__3.r = ctemp * a[i__4].r, z__3.i = ctemp * a[
				    i__4].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L50: */
			}
		    }
/* L60: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = j + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + i__ * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__3 = i__ * a_dim1 + 1;
			    z__3.r = stemp * a[i__3].r, z__3.i = stemp * a[
				    i__3].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = i__ * a_dim1 + 1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__3 = i__ * a_dim1 + 1;
			    z__3.r = ctemp * a[i__3].r, z__3.i = ctemp * a[
				    i__3].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = j + i__ * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = j + i__ * a_dim1;
			    i__4 = *m + i__ * a_dim1;
			    z__2.r = stemp * a[i__4].r, z__2.i = stemp * a[
				    i__4].i;
			    z__3.r = ctemp * temp.r, z__3.i = ctemp * temp.i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = *m + i__ * a_dim1;
			    i__4 = *m + i__ * a_dim1;
			    z__2.r = ctemp * a[i__4].r, z__2.i = ctemp * a[
				    i__4].i;
			    z__3.r = stemp * temp.r, z__3.i = stemp * temp.i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L90: */
			}
		    }
/* L100: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = j + i__ * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = j + i__ * a_dim1;
			    i__3 = *m + i__ * a_dim1;
			    z__2.r = stemp * a[i__3].r, z__2.i = stemp * a[
				    i__3].i;
			    z__3.r = ctemp * temp.r, z__3.i = ctemp * temp.i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = *m + i__ * a_dim1;
			    i__3 = *m + i__ * a_dim1;
			    z__2.r = ctemp * a[i__3].r, z__2.i = ctemp * a[
				    i__3].i;
			    z__3.r = stemp * temp.r, z__3.i = stemp * temp.i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__ + (j + 1) * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + (j + 1) * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__4 = i__ + j * a_dim1;
			    z__3.r = stemp * a[i__4].r, z__3.i = stemp * a[
				    i__4].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = i__ + j * a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__4 = i__ + j * a_dim1;
			    z__3.r = ctemp * a[i__4].r, z__3.i = ctemp * a[
				    i__4].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L130: */
			}
		    }
/* L140: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = i__ + (j + 1) * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + (j + 1) * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__3 = i__ + j * a_dim1;
			    z__3.r = stemp * a[i__3].r, z__3.i = stemp * a[
				    i__3].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = i__ + j * a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__3 = i__ + j * a_dim1;
			    z__3.r = ctemp * a[i__3].r, z__3.i = ctemp * a[
				    i__3].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__ + j * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + j * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__4 = i__ + a_dim1;
			    z__3.r = stemp * a[i__4].r, z__3.i = stemp * a[
				    i__4].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = i__ + a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__4 = i__ + a_dim1;
			    z__3.r = ctemp * a[i__4].r, z__3.i = ctemp * a[
				    i__4].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L170: */
			}
		    }
/* L180: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = i__ + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + j * a_dim1;
			    z__2.r = ctemp * temp.r, z__2.i = ctemp * temp.i;
			    i__3 = i__ + a_dim1;
			    z__3.r = stemp * a[i__3].r, z__3.i = stemp * a[
				    i__3].i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = i__ + a_dim1;
			    z__2.r = stemp * temp.r, z__2.i = stemp * temp.i;
			    i__3 = i__ + a_dim1;
			    z__3.r = ctemp * a[i__3].r, z__3.i = ctemp * a[
				    i__3].i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
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
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__ + j * a_dim1;
			    temp.r = a[i__3].r, temp.i = a[i__3].i;
			    i__3 = i__ + j * a_dim1;
			    i__4 = i__ + *n * a_dim1;
			    z__2.r = stemp * a[i__4].r, z__2.i = stemp * a[
				    i__4].i;
			    z__3.r = ctemp * temp.r, z__3.i = ctemp * temp.i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
			    i__3 = i__ + *n * a_dim1;
			    i__4 = i__ + *n * a_dim1;
			    z__2.r = ctemp * a[i__4].r, z__2.i = ctemp * a[
				    i__4].i;
			    z__3.r = stemp * temp.r, z__3.i = stemp * temp.i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L210: */
			}
		    }
/* L220: */
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if ((ctemp != 1.) || (stemp != 0.)) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    i__2 = i__ + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + j * a_dim1;
			    i__3 = i__ + *n * a_dim1;
			    z__2.r = stemp * a[i__3].r, z__2.i = stemp * a[
				    i__3].i;
			    z__3.r = ctemp * temp.r, z__3.i = ctemp * temp.i;
			    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i +
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
			    i__2 = i__ + *n * a_dim1;
			    i__3 = i__ + *n * a_dim1;
			    z__2.r = ctemp * a[i__3].r, z__2.i = ctemp * a[
				    i__3].i;
			    z__3.r = stemp * temp.r, z__3.i = stemp * temp.i;
			    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i -
				    z__3.i;
			    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
/* L230: */
			}
		    }
/* L240: */
		}
	    }
	}
    }

    return 0;

/*     End of ZLASR */

} /* zlasr_ */

/* Subroutine */ int zlassq_(integer *n, doublecomplex *x, integer *incx,
	doublereal *scale, doublereal *sumsq)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static integer ix;
    static doublereal temp1;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLASSQ returns the values scl and ssq such that

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

    X       (input) COMPLEX*16 array, dimension (N)
            The vector x as described above.
               x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n.

    INCX    (input) INTEGER
            The increment between successive values of the vector X.
            INCX > 0.

    SCALE   (input/output) DOUBLE PRECISION
            On entry, the value  scale  in the equation above.
            On exit, SCALE is overwritten with the value  scl .

    SUMSQ   (input/output) DOUBLE PRECISION
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
	    if (x[i__3].r != 0.) {
		i__3 = ix;
		temp1 = (d__1 = x[i__3].r, abs(d__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    d__1 = *scale / temp1;
		    *sumsq = *sumsq * (d__1 * d__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    d__1 = temp1 / *scale;
		    *sumsq += d__1 * d__1;
		}
	    }
	    if (d_imag(&x[ix]) != 0.) {
		temp1 = (d__1 = d_imag(&x[ix]), abs(d__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    d__1 = *scale / temp1;
		    *sumsq = *sumsq * (d__1 * d__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    d__1 = temp1 / *scale;
		    *sumsq += d__1 * d__1;
		}
	    }
/* L10: */
	}
    }

    return 0;

/*     End of ZLASSQ */

} /* zlassq_ */

/* Subroutine */ int zlaswp_(integer *n, doublecomplex *a, integer *lda,
	integer *k1, integer *k2, integer *ipiv, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    static integer i__, j, k, i1, i2, n32, ip, ix, ix0, inc;
    static doublecomplex temp;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.

    Arguments
    =========

    N       (input) INTEGER
            The number of columns of the matrix A.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    n32 = (*n / 32) << (5);
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

/*     End of ZLASWP */

} /* zlaswp_ */

/* Subroutine */ int zlatrd_(char *uplo, integer *n, integer *nb,
	doublecomplex *a, integer *lda, doublereal *e, doublecomplex *tau,
	doublecomplex *w, integer *ldw)
{
    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    static integer i__, iw;
    static doublecomplex alpha;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *),
	    zhemv_(char *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *), zaxpy_(integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), zlacgv_(integer *, doublecomplex *,
	    integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLATRD reduces NB rows and columns of a complex Hermitian matrix A to
    Hermitian tridiagonal form by a unitary similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by ZHETRD.

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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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

    E       (output) DOUBLE PRECISION array, dimension (N-1)
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    TAU     (output) COMPLEX*16 array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
            See Further Details.

    W       (output) COMPLEX*16 array, dimension (LDW,NB)
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
		d__1 = a[i__3].r;
		a[i__2].r = d__1, a[i__2].i = 0.;
		i__2 = *n - i__;
		zlacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__, &i__2, &z__1, &a[(i__ + 1) *
			a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, &
			c_b60, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__, &i__2, &z__1, &w[(iw + 1) *
			w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b60, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + i__ * a_dim1;
		i__3 = i__ + i__ * a_dim1;
		d__1 = a[i__3].r;
		a[i__2].r = d__1, a[i__2].i = 0.;
	    }
	    if (i__ > 1) {

/*
                Generate elementary reflector H(i) to annihilate
                A(1:i-2,i)
*/

		i__2 = i__ - 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = i__ - 1;
		zlarfg_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &tau[i__
			- 1]);
		i__2 = i__ - 1;
		e[i__2] = alpha.r;
		i__2 = i__ - 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute W(1:i-1,i) */

		i__2 = i__ - 1;
		zhemv_("Upper", &i__2, &c_b60, &a[a_offset], lda, &a[i__ *
			a_dim1 + 1], &c__1, &c_b59, &w[iw * w_dim1 + 1], &
			c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &w[(
			    iw + 1) * w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1],
			    &c__1, &c_b59, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    z__1.r = -1., z__1.i = -0.;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[(i__ + 1) *
			     a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b60, &w[iw * w_dim1 + 1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[(
			    i__ + 1) * a_dim1 + 1], lda, &a[i__ * a_dim1 + 1],
			     &c__1, &c_b59, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    z__1.r = -1., z__1.i = -0.;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &w[(iw + 1) *
			    w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b60, &w[iw * w_dim1 + 1], &c__1);
		}
		i__2 = i__ - 1;
		zscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
		z__3.r = -.5, z__3.i = -0.;
		i__2 = i__ - 1;
		z__2.r = z__3.r * tau[i__2].r - z__3.i * tau[i__2].i, z__2.i =
			 z__3.r * tau[i__2].i + z__3.i * tau[i__2].r;
		i__3 = i__ - 1;
		zdotc_(&z__4, &i__3, &w[iw * w_dim1 + 1], &c__1, &a[i__ *
			a_dim1 + 1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r *
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = i__ - 1;
		zaxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw *
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
	    d__1 = a[i__3].r;
	    a[i__2].r = d__1, a[i__2].i = 0.;
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + a_dim1], lda,
		     &w[i__ + w_dim1], ldw, &c_b60, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &w[i__ + w_dim1], ldw,
		     &a[i__ + a_dim1], lda, &c_b60, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__ + i__ * a_dim1;
	    d__1 = a[i__3].r;
	    a[i__2].r = d__1, a[i__2].i = 0.;
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
		zlarfg_(&i__2, &alpha, &a[min(i__3,*n) + i__ * a_dim1], &c__1,
			 &tau[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute W(i+1:n,i) */

		i__2 = *n - i__;
		zhemv_("Lower", &i__2, &c_b60, &a[i__ + 1 + (i__ + 1) *
			a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b59, &w[i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &w[i__ +
			1 + w_dim1], ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b59, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 +
			a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b60, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b59, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &w[i__ + 1 +
			w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b60, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
		z__3.r = -.5, z__3.i = -0.;
		i__2 = i__;
		z__2.r = z__3.r * tau[i__2].r - z__3.i * tau[i__2].i, z__2.i =
			 z__3.r * tau[i__2].i + z__3.i * tau[i__2].r;
		i__3 = *n - i__;
		zdotc_(&z__4, &i__3, &w[i__ + 1 + i__ * w_dim1], &c__1, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r *
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = *n - i__;
		zaxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
	    }

/* L20: */
	}
    }

    return 0;

/*     End of ZLATRD */

} /* zlatrd_ */

/* Subroutine */ int zlatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublecomplex *a, integer *lda, doublecomplex *x,
	doublereal *scale, doublereal *cnorm, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, j;
    static doublereal xj, rec, tjj;
    static integer jinc;
    static doublereal xbnd;
    static integer imax;
    static doublereal tmax;
    static doublecomplex tjjs;
    static doublereal xmax, grow;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *,
	    integer *);
    extern logical lsame_(char *, char *);
    static doublereal tscal;
    static doublecomplex uscal;
    static integer jlast;
    static doublecomplex csumj;
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    static logical upper;
    extern /* Double Complex */ VOID zdotu_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zaxpy_(integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), ztrsv_(
	    char *, char *, char *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), dlabad_(
	    doublereal *, doublereal *);

    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *);
    static doublereal bignum;
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *,
	     doublecomplex *);
    static logical notran;
    static integer jfirst;
    extern doublereal dzasum_(integer *, doublecomplex *, integer *);
    static doublereal smlnum;
    static logical nounit;


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1992


    Purpose
    =======

    ZLATRS solves one of the triangular systems

       A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b,

    with scaling to prevent overflow.  Here A is an upper or lower
    triangular matrix, A**T denotes the transpose of A, A**H denotes the
    conjugate transpose of A, x and b are n-element vectors, and s is a
    scaling factor, usually less than or equal to 1, chosen so that the
    components of x will be less than the overflow threshold.  If the
    unscaled problem will not cause overflow, the Level 2 BLAS routine
    ZTRSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
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

    A       (input) COMPLEX*16 array, dimension (LDA,N)
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

    X       (input/output) COMPLEX*16 array, dimension (N)
            On entry, the right hand side b of the triangular system.
            On exit, X is overwritten by the solution vector x.

    SCALE   (output) DOUBLE PRECISION
            The scaling factor s for the triangular system
               A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b.
            If SCALE = 0, the matrix A is singular or badly scaled, and
            the vector x is an exact or approximate solution to A*x = 0.

    CNORM   (input or output) DOUBLE PRECISION array, dimension (N)

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

    A rough bound on x is computed; if that is less than overflow, ZTRSV
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

    Since |x(j)| <= M(j), we use the Level 2 BLAS routine ZTRSV if the
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

    and we can safely call ZTRSV if 1/M(n) and 1/G(n) are both greater
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
	xerbla_("ZLATRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine machine dependent parameters to control overflow. */

    smlnum = SAFEMINIMUM;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    smlnum /= PRECISION;
    bignum = 1. / smlnum;
    *scale = 1.;

    if (lsame_(normin, "N")) {

/*        Compute the 1-norm of each column, not including the diagonal. */

	if (upper) {

/*           A is upper triangular. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j - 1;
		cnorm[j] = dzasum_(&i__2, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	    }
	} else {

/*           A is lower triangular. */

	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		cnorm[j] = dzasum_(&i__2, &a[j + 1 + j * a_dim1], &c__1);
/* L20: */
	    }
	    cnorm[*n] = 0.;
	}
    }

/*
       Scale the column norms by TSCAL if the maximum element in CNORM is
       greater than BIGNUM/2.
*/

    imax = idamax_(n, &cnorm[1], &c__1);
    tmax = cnorm[imax];
    if (tmax <= bignum * .5) {
	tscal = 1.;
    } else {
	tscal = .5 / (smlnum * tmax);
	dscal_(n, &tscal, &cnorm[1], &c__1);
    }

/*
       Compute a bound on the computed solution vector to see if the
       Level 2 BLAS routine ZTRSV can be used.
*/

    xmax = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__2 = j;
	d__3 = xmax, d__4 = (d__1 = x[i__2].r / 2., abs(d__1)) + (d__2 =
		d_imag(&x[j]) / 2., abs(d__2));
	xmax = max(d__3,d__4);
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

	if (tscal != 1.) {
	    grow = 0.;
	    goto L60;
	}

	if (nounit) {

/*
             A is non-unit triangular.

             Compute GROW = 1/G(j) and XBND = 1/M(j).
             Initially, G(0) = max{x(i), i=1,...,n}.
*/

	    grow = .5 / max(xbnd,smlnum);
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
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));

		if (tjj >= smlnum) {

/*
                   M(j) = G(j-1) / abs(A(j,j))

   Computing MIN
*/
		    d__1 = xbnd, d__2 = min(1.,tjj) * grow;
		    xbnd = min(d__1,d__2);
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.;
		}

		if (tjj + cnorm[j] >= smlnum) {

/*                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) ) */

		    grow *= tjj / (tjj + cnorm[j]);
		} else {

/*                 G(j) could overflow, set GROW to 0. */

		    grow = 0.;
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
	    d__1 = 1., d__2 = .5 / max(xbnd,smlnum);
	    grow = min(d__1,d__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

/*              G(j) = G(j-1)*( 1 + CNORM(j) ) */

		grow *= 1. / (cnorm[j] + 1.);
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

	if (tscal != 1.) {
	    grow = 0.;
	    goto L90;
	}

	if (nounit) {

/*
             A is non-unit triangular.

             Compute GROW = 1/G(j) and XBND = 1/M(j).
             Initially, M(0) = max{x(i), i=1,...,n}.
*/

	    grow = .5 / max(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) ) */

		xj = cnorm[j] + 1.;
/* Computing MIN */
		d__1 = grow, d__2 = xbnd / xj;
		grow = min(d__1,d__2);

		i__3 = j + j * a_dim1;
		tjjs.r = a[i__3].r, tjjs.i = a[i__3].i;
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));

		if (tjj >= smlnum) {

/*                 M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j)) */

		    if (xj > tjj) {
			xbnd *= tjj / xj;
		    }
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.;
		}
/* L70: */
	    }
	    grow = min(grow,xbnd);
	} else {

/*
             A is unit triangular.

             Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.

   Computing MIN
*/
	    d__1 = 1., d__2 = .5 / max(xbnd,smlnum);
	    grow = min(d__1,d__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = ( 1 + CNORM(j) )*G(j-1) */

		xj = cnorm[j] + 1.;
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

	ztrsv_(uplo, trans, diag, n, &a[a_offset], lda, &x[1], &c__1);
    } else {

/*        Use a Level 1 BLAS solve, scaling intermediate results. */

	if (xmax > bignum * .5) {

/*
             Scale X so that its components are less than or equal to
             BIGNUM in absolute value.
*/

	    *scale = bignum * .5 / xmax;
	    zdscal_(n, scale, &x[1], &c__1);
	    xmax = bignum;
	} else {
	    xmax *= 2.;
	}

	if (notran) {

/*           Solve A * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) / A(j,j), scaling x if necessary. */

		i__3 = j;
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]),
			abs(d__2));
		if (nounit) {
		    i__3 = j + j * a_dim1;
		    z__1.r = tscal * a[i__3].r, z__1.i = tscal * a[i__3].i;
		    tjjs.r = z__1.r, tjjs.i = z__1.i;
		} else {
		    tjjs.r = tscal, tjjs.i = 0.;
		    if (tscal == 1.) {
			goto L110;
		    }
		}
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));
		if (tjj > smlnum) {

/*                    abs(A(j,j)) > SMLNUM: */

		    if (tjj < 1.) {
			if (xj > tjj * bignum) {

/*                          Scale x by 1/b(j). */

			    rec = 1. / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    i__3 = j;
		    zladiv_(&z__1, &x[j], &tjjs);
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		} else if (tjj > 0.) {

/*                    0 < abs(A(j,j)) <= SMLNUM: */

		    if (xj > tjj * bignum) {

/*
                         Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM
                         to avoid overflow when dividing by A(j,j).
*/

			rec = tjj * bignum / xj;
			if (cnorm[j] > 1.) {

/*
                            Scale by 1/CNORM(j) to avoid overflow when
                            multiplying x(j) times column j.
*/

			    rec /= cnorm[j];
			}
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		    i__3 = j;
		    zladiv_(&z__1, &x[j], &tjjs);
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		} else {

/*
                      A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                      scale = 0, and compute a solution to A*x = 0.
*/

		    i__3 = *n;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			x[i__4].r = 0., x[i__4].i = 0.;
/* L100: */
		    }
		    i__3 = j;
		    x[i__3].r = 1., x[i__3].i = 0.;
		    xj = 1.;
		    *scale = 0.;
		    xmax = 0.;
		}
L110:

/*
                Scale x if necessary to avoid overflow when adding a
                multiple of column j of A.
*/

		if (xj > 1.) {
		    rec = 1. / xj;
		    if (cnorm[j] > (bignum - xmax) * rec) {

/*                    Scale x by 1/(2*abs(x(j))). */

			rec *= .5;
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
		    }
		} else if (xj * cnorm[j] > bignum - xmax) {

/*                 Scale x by 1/2. */

		    zdscal_(n, &c_b2210, &x[1], &c__1);
		    *scale *= .5;
		}

		if (upper) {
		    if (j > 1) {

/*
                      Compute the update
                         x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
*/

			i__3 = j - 1;
			i__4 = j;
			z__2.r = -x[i__4].r, z__2.i = -x[i__4].i;
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			zaxpy_(&i__3, &z__1, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			i__3 = j - 1;
			i__ = izamax_(&i__3, &x[1], &c__1);
			i__3 = i__;
			xmax = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(
				&x[i__]), abs(d__2));
		    }
		} else {
		    if (j < *n) {

/*
                      Compute the update
                         x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
*/

			i__3 = *n - j;
			i__4 = j;
			z__2.r = -x[i__4].r, z__2.i = -x[i__4].i;
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			zaxpy_(&i__3, &z__1, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			i__3 = *n - j;
			i__ = j + izamax_(&i__3, &x[j + 1], &c__1);
			i__3 = i__;
			xmax = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(
				&x[i__]), abs(d__2));
		    }
		}
/* L120: */
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
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]),
			abs(d__2));
		uscal.r = tscal, uscal.i = 0.;
		rec = 1. / max(xmax,1.);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5;
		    if (nounit) {
			i__3 = j + j * a_dim1;
			z__1.r = tscal * a[i__3].r, z__1.i = tscal * a[i__3]
				.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs),
			    abs(d__2));
		    if (tjj > 1.) {

/*
                         Divide by A(j,j) when scaling x if A(j,j) > 1.

   Computing MIN
*/
			d__1 = 1., d__2 = rec * tjj;
			rec = min(d__1,d__2);
			zladiv_(&z__1, &uscal, &tjjs);
			uscal.r = z__1.r, uscal.i = z__1.i;
		    }
		    if (rec < 1.) {
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0., csumj.i = 0.;
		if (uscal.r == 1. && uscal.i == 0.) {

/*
                   If the scaling needed for A in the dot product is 1,
                   call ZDOTU to perform the dot product.
*/

		    if (upper) {
			i__3 = j - 1;
			zdotu_(&z__1, &i__3, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    } else if (j < *n) {
			i__3 = *n - j;
			zdotu_(&z__1, &i__3, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
			i__3 = j - 1;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + j * a_dim1;
			    z__3.r = a[i__4].r * uscal.r - a[i__4].i *
				    uscal.i, z__3.i = a[i__4].r * uscal.i + a[
				    i__4].i * uscal.r;
			    i__5 = i__;
			    z__2.r = z__3.r * x[i__5].r - z__3.i * x[i__5].i,
				    z__2.i = z__3.r * x[i__5].i + z__3.i * x[
				    i__5].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i +
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L130: */
			}
		    } else if (j < *n) {
			i__3 = *n;
			for (i__ = j + 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + j * a_dim1;
			    z__3.r = a[i__4].r * uscal.r - a[i__4].i *
				    uscal.i, z__3.i = a[i__4].r * uscal.i + a[
				    i__4].i * uscal.r;
			    i__5 = i__;
			    z__2.r = z__3.r * x[i__5].r - z__3.i * x[i__5].i,
				    z__2.i = z__3.r * x[i__5].i + z__3.i * x[
				    i__5].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i +
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L140: */
			}
		    }
		}

		z__1.r = tscal, z__1.i = 0.;
		if (uscal.r == z__1.r && uscal.i == z__1.i) {

/*
                   Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
                   was not used to scale the dotproduct.
*/

		    i__3 = j;
		    i__4 = j;
		    z__1.r = x[i__4].r - csumj.r, z__1.i = x[i__4].i -
			    csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		    if (nounit) {
			i__3 = j + j * a_dim1;
			z__1.r = tscal * a[i__3].r, z__1.i = tscal * a[i__3]
				.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
			if (tscal == 1.) {
			    goto L160;
			}
		    }

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs),
			    abs(d__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1. / xj;
				zdscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else if (tjj > 0.) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else {

/*
                         A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                         scale = 0 and compute a solution to A**T *x = 0.
*/

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0., x[i__4].i = 0.;
/* L150: */
			}
			i__3 = j;
			x[i__3].r = 1., x[i__3].i = 0.;
			*scale = 0.;
			xmax = 0.;
		    }
L160:
		    ;
		} else {

/*
                   Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
                   product has already been divided by 1/A(j,j).
*/

		    i__3 = j;
		    zladiv_(&z__2, &x[j], &tjjs);
		    z__1.r = z__2.r - csumj.r, z__1.i = z__2.i - csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		}
/* Computing MAX */
		i__3 = j;
		d__3 = xmax, d__4 = (d__1 = x[i__3].r, abs(d__1)) + (d__2 =
			d_imag(&x[j]), abs(d__2));
		xmax = max(d__3,d__4);
/* L170: */
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
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]),
			abs(d__2));
		uscal.r = tscal, uscal.i = 0.;
		rec = 1. / max(xmax,1.);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5;
		    if (nounit) {
			d_cnjg(&z__2, &a[j + j * a_dim1]);
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs),
			    abs(d__2));
		    if (tjj > 1.) {

/*
                         Divide by A(j,j) when scaling x if A(j,j) > 1.

   Computing MIN
*/
			d__1 = 1., d__2 = rec * tjj;
			rec = min(d__1,d__2);
			zladiv_(&z__1, &uscal, &tjjs);
			uscal.r = z__1.r, uscal.i = z__1.i;
		    }
		    if (rec < 1.) {
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0., csumj.i = 0.;
		if (uscal.r == 1. && uscal.i == 0.) {

/*
                   If the scaling needed for A in the dot product is 1,
                   call ZDOTC to perform the dot product.
*/

		    if (upper) {
			i__3 = j - 1;
			zdotc_(&z__1, &i__3, &a[j * a_dim1 + 1], &c__1, &x[1],
				 &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    } else if (j < *n) {
			i__3 = *n - j;
			zdotc_(&z__1, &i__3, &a[j + 1 + j * a_dim1], &c__1, &
				x[j + 1], &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
			i__3 = j - 1;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    d_cnjg(&z__4, &a[i__ + j * a_dim1]);
			    z__3.r = z__4.r * uscal.r - z__4.i * uscal.i,
				    z__3.i = z__4.r * uscal.i + z__4.i *
				    uscal.r;
			    i__4 = i__;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i,
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i +
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L180: */
			}
		    } else if (j < *n) {
			i__3 = *n;
			for (i__ = j + 1; i__ <= i__3; ++i__) {
			    d_cnjg(&z__4, &a[i__ + j * a_dim1]);
			    z__3.r = z__4.r * uscal.r - z__4.i * uscal.i,
				    z__3.i = z__4.r * uscal.i + z__4.i *
				    uscal.r;
			    i__4 = i__;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i,
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i +
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L190: */
			}
		    }
		}

		z__1.r = tscal, z__1.i = 0.;
		if (uscal.r == z__1.r && uscal.i == z__1.i) {

/*
                   Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
                   was not used to scale the dotproduct.
*/

		    i__3 = j;
		    i__4 = j;
		    z__1.r = x[i__4].r - csumj.r, z__1.i = x[i__4].i -
			    csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		    if (nounit) {
			d_cnjg(&z__2, &a[j + j * a_dim1]);
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
			if (tscal == 1.) {
			    goto L210;
			}
		    }

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs),
			    abs(d__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1. / xj;
				zdscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else if (tjj > 0.) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else {

/*
                         A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
                         scale = 0 and compute a solution to A**H *x = 0.
*/

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0., x[i__4].i = 0.;
/* L200: */
			}
			i__3 = j;
			x[i__3].r = 1., x[i__3].i = 0.;
			*scale = 0.;
			xmax = 0.;
		    }
L210:
		    ;
		} else {

/*
                   Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
                   product has already been divided by 1/A(j,j).
*/

		    i__3 = j;
		    zladiv_(&z__2, &x[j], &tjjs);
		    z__1.r = z__2.r - csumj.r, z__1.i = z__2.i - csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		}
/* Computing MAX */
		i__3 = j;
		d__3 = xmax, d__4 = (d__1 = x[i__3].r, abs(d__1)) + (d__2 =
			d_imag(&x[j]), abs(d__2));
		xmax = max(d__3,d__4);
/* L220: */
	    }
	}
	*scale /= tscal;
    }

/*     Scale the column norms by 1/TSCAL for return. */

    if (tscal != 1.) {
	d__1 = 1. / tscal;
	dscal_(n, &d__1, &cnorm[1], &c__1);
    }

    return 0;

/*     End of ZLATRS */

} /* zlatrs_ */

/* Subroutine */ int zlauu2_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1;

    /* Local variables */
    static integer i__;
    static doublereal aii;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *), zlacgv_(
	    integer *, doublecomplex *, integer *);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLAUU2 computes the product U * U' or L' * L, where the triangular
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZLAUU2", &i__1);
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
		zdotc_(&z__1, &i__3, &a[i__ + (i__ + 1) * a_dim1], lda, &a[
			i__ + (i__ + 1) * a_dim1], lda);
		d__1 = aii * aii + z__1.r;
		a[i__2].r = d__1, a[i__2].i = 0.;
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		z__1.r = aii, z__1.i = 0.;
		zgemv_("No transpose", &i__2, &i__3, &c_b60, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			z__1, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
	    } else {
		zdscal_(&i__, &aii, &a[i__ * a_dim1 + 1], &c__1);
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
		zdotc_(&z__1, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		d__1 = aii * aii + z__1.r;
		a[i__2].r = d__1, a[i__2].i = 0.;
		i__2 = i__ - 1;
		zlacgv_(&i__2, &a[i__ + a_dim1], lda);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = aii, z__1.i = 0.;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b60, &a[i__ +
			1 + a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			z__1, &a[i__ + a_dim1], lda);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    } else {
		zdscal_(&i__, &aii, &a[i__ + a_dim1], lda);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of ZLAUU2 */

} /* zlauu2_ */

/* Subroutine */ int zlauum_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, ib, nb;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), zherk_(char *, char *, integer *,
	    integer *, doublereal *, doublecomplex *, integer *, doublereal *,
	     doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int ztrmm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *),
	    zlauu2_(char *, integer *, doublecomplex *, integer *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZLAUUM computes the product U * U' or L' * L, where the triangular
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZLAUUM", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "ZLAUUM", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if ((nb <= 1) || (nb >= *n)) {

/*        Use unblocked code */

	zlauu2_(uplo, n, &a[a_offset], lda, info);
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
		ztrmm_("Right", "Upper", "Conjugate transpose", "Non-unit", &
			i__3, &ib, &c_b60, &a[i__ + i__ * a_dim1], lda, &a[
			i__ * a_dim1 + 1], lda);
		zlauu2_("Upper", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    zgemm_("No transpose", "Conjugate transpose", &i__3, &ib,
			    &i__4, &c_b60, &a[(i__ + ib) * a_dim1 + 1], lda, &
			    a[i__ + (i__ + ib) * a_dim1], lda, &c_b60, &a[i__
			    * a_dim1 + 1], lda);
		    i__3 = *n - i__ - ib + 1;
		    zherk_("Upper", "No transpose", &ib, &i__3, &c_b1015, &a[
			    i__ + (i__ + ib) * a_dim1], lda, &c_b1015, &a[i__
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
		ztrmm_("Left", "Lower", "Conjugate transpose", "Non-unit", &
			ib, &i__3, &c_b60, &a[i__ + i__ * a_dim1], lda, &a[
			i__ + a_dim1], lda);
		zlauu2_("Lower", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    zgemm_("Conjugate transpose", "No transpose", &ib, &i__3,
			    &i__4, &c_b60, &a[i__ + ib + i__ * a_dim1], lda, &
			    a[i__ + ib + a_dim1], lda, &c_b60, &a[i__ +
			    a_dim1], lda);
		    i__3 = *n - i__ - ib + 1;
		    zherk_("Lower", "Conjugate transpose", &ib, &i__3, &
			    c_b1015, &a[i__ + ib + i__ * a_dim1], lda, &
			    c_b1015, &a[i__ + i__ * a_dim1], lda);
		}
/* L20: */
	    }
	}
    }

    return 0;

/*     End of ZLAUUM */

} /* zlauum_ */

/* Subroutine */ int zpotf2_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer j;
    static doublereal ajj;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *), zlacgv_(
	    integer *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZPOTF2 computes the Cholesky factorization of a complex Hermitian
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZPOTF2", &i__1);
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
	    d__1 = a[i__2].r;
	    i__3 = j - 1;
	    zdotc_(&z__2, &i__3, &a[j * a_dim1 + 1], &c__1, &a[j * a_dim1 + 1]
		    , &c__1);
	    z__1.r = d__1 - z__2.r, z__1.i = -z__2.i;
	    ajj = z__1.r;
	    if (ajj <= 0.) {
		i__2 = j + j * a_dim1;
		a[i__2].r = ajj, a[i__2].i = 0.;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    i__2 = j + j * a_dim1;
	    a[i__2].r = ajj, a[i__2].i = 0.;

/*           Compute elements J+1:N of row J. */

	    if (j < *n) {
		i__2 = j - 1;
		zlacgv_(&i__2, &a[j * a_dim1 + 1], &c__1);
		i__2 = j - 1;
		i__3 = *n - j;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Transpose", &i__2, &i__3, &z__1, &a[(j + 1) * a_dim1
			+ 1], lda, &a[j * a_dim1 + 1], &c__1, &c_b60, &a[j + (
			j + 1) * a_dim1], lda);
		i__2 = j - 1;
		zlacgv_(&i__2, &a[j * a_dim1 + 1], &c__1);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		zdscal_(&i__2, &d__1, &a[j + (j + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness. */

	    i__2 = j + j * a_dim1;
	    d__1 = a[i__2].r;
	    i__3 = j - 1;
	    zdotc_(&z__2, &i__3, &a[j + a_dim1], lda, &a[j + a_dim1], lda);
	    z__1.r = d__1 - z__2.r, z__1.i = -z__2.i;
	    ajj = z__1.r;
	    if (ajj <= 0.) {
		i__2 = j + j * a_dim1;
		a[i__2].r = ajj, a[i__2].i = 0.;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    i__2 = j + j * a_dim1;
	    a[i__2].r = ajj, a[i__2].i = 0.;

/*           Compute elements J+1:N of column J. */

	    if (j < *n) {
		i__2 = j - 1;
		zlacgv_(&i__2, &a[j + a_dim1], lda);
		i__2 = *n - j;
		i__3 = j - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[j + 1 + a_dim1]
			, lda, &a[j + a_dim1], lda, &c_b60, &a[j + 1 + j *
			a_dim1], &c__1);
		i__2 = j - 1;
		zlacgv_(&i__2, &a[j + a_dim1], lda);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		zdscal_(&i__2, &d__1, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }
    goto L40;

L30:
    *info = j;

L40:
    return 0;

/*     End of ZPOTF2 */

} /* zpotf2_ */

/* Subroutine */ int zpotrf_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    doublecomplex z__1;

    /* Local variables */
    static integer j, jb, nb;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), zherk_(char *, char *, integer *,
	    integer *, doublereal *, doublecomplex *, integer *, doublereal *,
	     doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int ztrsm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *),
	    zpotf2_(char *, integer *, doublecomplex *, integer *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZPOTRF computes the Cholesky factorization of a complex Hermitian
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZPOTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "ZPOTRF", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);
    if ((nb <= 1) || (nb >= *n)) {

/*        Use unblocked code. */

	zpotf2_(uplo, n, &a[a_offset], lda, info);
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
		zherk_("Upper", "Conjugate transpose", &jb, &i__3, &c_b1294, &
			a[j * a_dim1 + 1], lda, &c_b1015, &a[j + j * a_dim1],
			lda);
		zpotf2_("Upper", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block row. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("Conjugate transpose", "No transpose", &jb, &i__3,
			    &i__4, &z__1, &a[j * a_dim1 + 1], lda, &a[(j + jb)
			     * a_dim1 + 1], lda, &c_b60, &a[j + (j + jb) *
			    a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    ztrsm_("Left", "Upper", "Conjugate transpose", "Non-unit",
			     &jb, &i__3, &c_b60, &a[j + j * a_dim1], lda, &a[
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
		zherk_("Lower", "No transpose", &jb, &i__3, &c_b1294, &a[j +
			a_dim1], lda, &c_b1015, &a[j + j * a_dim1], lda);
		zpotf2_("Lower", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block column. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    z__1.r = -1., z__1.i = -0.;
		    zgemm_("No transpose", "Conjugate transpose", &i__3, &jb,
			    &i__4, &z__1, &a[j + jb + a_dim1], lda, &a[j +
			    a_dim1], lda, &c_b60, &a[j + jb + j * a_dim1],
			    lda);
		    i__3 = *n - j - jb + 1;
		    ztrsm_("Right", "Lower", "Conjugate transpose", "Non-unit"
			    , &i__3, &jb, &c_b60, &a[j + j * a_dim1], lda, &a[
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

/*     End of ZPOTRF */

} /* zpotrf_ */

/* Subroutine */ int zpotri_(char *uplo, integer *n, doublecomplex *a,
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), zlauum_(
	    char *, integer *, doublecomplex *, integer *, integer *),
	     ztrtri_(char *, char *, integer *, doublecomplex *, integer *,
	    integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       March 31, 1993


    Purpose
    =======

    ZPOTRI computes the inverse of a complex Hermitian positive definite
    matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
    computed by ZPOTRF.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the triangular factor U or L from the Cholesky
            factorization A = U**H*U or A = L*L**H, as computed by
            ZPOTRF.
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
	xerbla_("ZPOTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Invert the triangular Cholesky factor U or L. */

    ztrtri_(uplo, "Non-unit", n, &a[a_offset], lda, info);
    if (*info > 0) {
	return 0;
    }

/*     Form inv(U)*inv(U)' or inv(L)'*inv(L). */

    zlauum_(uplo, n, &a[a_offset], lda, info);

    return 0;

/*     End of ZPOTRI */

} /* zpotri_ */

/* Subroutine */ int zpotrs_(char *uplo, integer *n, integer *nrhs,
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb,
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int ztrsm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *),
	    xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZPOTRS solves a system of linear equations A*X = B with a Hermitian
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by ZPOTRF.

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

    A       (input) COMPLEX*16 array, dimension (LDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by ZPOTRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS)
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
	xerbla_("ZPOTRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if ((*n == 0) || (*nrhs == 0)) {
	return 0;
    }

    if (upper) {

/*
          Solve A*X = B where A = U'*U.

          Solve U'*X = B, overwriting B with X.
*/

	ztrsm_("Left", "Upper", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b60, &a[a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	ztrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b60, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*
          Solve A*X = B where A = L*L'.

          Solve L*X = B, overwriting B with X.
*/

	ztrsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b60, &
		a[a_offset], lda, &b[b_offset], ldb);

/*        Solve L'*X = B, overwriting B with X. */

	ztrsm_("Left", "Lower", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b60, &a[a_offset], lda, &b[b_offset], ldb);
    }

    return 0;

/*     End of ZPOTRS */

} /* zpotrs_ */

/* Subroutine */ int zstedc_(char *compz, integer *n, doublereal *d__,
	doublereal *e, doublecomplex *z__, integer *ldz, doublecomplex *work,
	integer *lwork, doublereal *rwork, integer *lrwork, integer *iwork,
	integer *liwork, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2;

    /* Builtin functions */
    double log(doublereal);
    integer pow_ii(integer *, integer *);
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, k, m;
    static doublereal p;
    static integer ii, ll, end, lgn;
    static doublereal eps, tiny;
    extern logical lsame_(char *, char *);
    static integer lwmin, start;
    extern /* Subroutine */ int zswap_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), zlaed0_(integer *, integer *,
	    doublereal *, doublereal *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublereal *, integer *, integer *);

    extern /* Subroutine */ int dlascl_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, integer *, doublereal *,
	    integer *, integer *), dstedc_(char *, integer *,
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *,
	     integer *, integer *, integer *, integer *), dlaset_(
	    char *, integer *, integer *, doublereal *, doublereal *,
	    doublereal *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int dsterf_(integer *, doublereal *, doublereal *,
	     integer *), zlacrm_(integer *, integer *, doublecomplex *,
	    integer *, doublereal *, integer *, doublecomplex *, integer *,
	    doublereal *);
    static integer liwmin, icompz;
    extern /* Subroutine */ int dsteqr_(char *, integer *, doublereal *,
	    doublereal *, doublereal *, integer *, doublereal *, integer *), zlacpy_(char *, integer *, integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *);
    static doublereal orgnrm;
    static integer lrwmin;
    static logical lquery;
    static integer smlsiz;
    extern /* Subroutine */ int zsteqr_(char *, integer *, doublereal *,
	    doublereal *, doublecomplex *, integer *, doublereal *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZSTEDC computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.
    The eigenvectors of a full or band complex Hermitian matrix can also
    be found if ZHETRD or ZHPTRD or ZHBTRD has been used to reduce this
    matrix to tridiagonal form.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.  See DLAED3 for details.

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

    D       (input/output) DOUBLE PRECISION array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) DOUBLE PRECISION array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    Z       (input/output) COMPLEX*16 array, dimension (LDZ,N)
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

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If COMPZ = 'N' or 'I', or N <= 1, LWORK must be at least 1.
            If COMPZ = 'V' and N > 1, LWORK must be at least N*N.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace/output) DOUBLE PRECISION array,
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
    lquery = ((*lwork == -1) || (*lrwork == -1)) || (*liwork == -1);

    if (lsame_(compz, "N")) {
	icompz = 0;
    } else if (lsame_(compz, "V")) {
	icompz = 1;
    } else if (lsame_(compz, "I")) {
	icompz = 2;
    } else {
	icompz = -1;
    }
    if ((*n <= 1) || (icompz <= 0)) {
	lwmin = 1;
	liwmin = 1;
	lrwmin = 1;
    } else {
	lgn = (integer) (log((doublereal) (*n)) / log(2.));
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
	    lrwmin = *n * 3 + 1 + ((*n) << (1)) * lgn + i__1 * i__1 * 3;
	    liwmin = *n * 6 + 6 + *n * 5 * lgn;
	} else if (icompz == 2) {
	    lwmin = 1;
/* Computing 2nd power */
	    i__1 = *n;
	    lrwmin = ((*n) << (2)) + 1 + ((i__1 * i__1) << (1));
	    liwmin = *n * 5 + 3;
	}
    }
    if (icompz < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if ((*ldz < 1) || (icompz > 0 && *ldz < max(1,*n))) {
	*info = -6;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -8;
    } else if (*lrwork < lrwmin && ! lquery) {
	*info = -10;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -12;
    }

    if (*info == 0) {
	work[1].r = (doublereal) lwmin, work[1].i = 0.;
	rwork[1] = (doublereal) lrwmin;
	iwork[1] = liwmin;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZSTEDC", &i__1);
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
	    z__[i__1].r = 1., z__[i__1].i = 0.;
	}
	return 0;
    }

    smlsiz = ilaenv_(&c__9, "ZSTEDC", " ", &c__0, &c__0, &c__0, &c__0, (
	    ftnlen)6, (ftnlen)1);

/*
       If the following conditional clause is removed, then the routine
       will use the Divide and Conquer routine to compute only the
       eigenvalues, which requires (3N + 3N**2) real workspace and
       (2 + 5N + 2N lg(N)) integer workspace.
       Since on many architectures DSTERF is much faster than any other
       algorithm for finding eigenvalues only, it is used here
       as the default.

       If COMPZ = 'N', use DSTERF to compute the eigenvalues.
*/

    if (icompz == 0) {
	dsterf_(n, &d__[1], &e[1], info);
	return 0;
    }

/*
       If N is smaller than the minimum divide size (SMLSIZ+1), then
       solve the problem with another solver.
*/

    if (*n <= smlsiz) {
	if (icompz == 0) {
	    dsterf_(n, &d__[1], &e[1], info);
	    return 0;
	} else if (icompz == 2) {
	    zsteqr_("I", n, &d__[1], &e[1], &z__[z_offset], ldz, &rwork[1],
		    info);
	    return 0;
	} else {
	    zsteqr_("V", n, &d__[1], &e[1], &z__[z_offset], ldz, &rwork[1],
		    info);
	    return 0;
	}
    }

/*     If COMPZ = 'I', we simply call DSTEDC instead. */

    if (icompz == 2) {
	dlaset_("Full", n, n, &c_b324, &c_b1015, &rwork[1], n);
	ll = *n * *n + 1;
	i__1 = *lrwork - ll + 1;
	dstedc_("I", n, &d__[1], &e[1], &rwork[1], n, &rwork[ll], &i__1, &
		iwork[1], liwork, info);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * z_dim1;
		i__4 = (j - 1) * *n + i__;
		z__[i__3].r = rwork[i__4], z__[i__3].i = 0.;
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

    orgnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.) {
	return 0;
    }

    eps = EPSILON;

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
	    tiny = eps * sqrt((d__1 = d__[end], abs(d__1))) * sqrt((d__2 =
		    d__[end + 1], abs(d__2)));
	    if ((d__1 = e[end], abs(d__1)) > tiny) {
		++end;
		goto L40;
	    }
	}

/*        (Sub) Problem determined.  Compute its size and solve it. */

	m = end - start + 1;
	if (m > smlsiz) {
	    *info = smlsiz;

/*           Scale. */

	    orgnrm = dlanst_("M", &m, &d__[start], &e[start]);
	    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, &m, &c__1, &d__[
		    start], &m, info);
	    i__1 = m - 1;
	    i__2 = m - 1;
	    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b1015, &i__1, &c__1, &e[
		    start], &i__2, info);

	    zlaed0_(n, &m, &d__[start], &e[start], &z__[start * z_dim1 + 1],
		    ldz, &work[1], n, &rwork[1], &iwork[1], info);
	    if (*info > 0) {
		*info = (*info / (m + 1) + start - 1) * (*n + 1) + *info % (m
			+ 1) + start - 1;
		return 0;
	    }

/*           Scale back. */

	    dlascl_("G", &c__0, &c__0, &c_b1015, &orgnrm, &m, &c__1, &d__[
		    start], &m, info);

	} else {
	    dsteqr_("I", &m, &d__[start], &e[start], &rwork[1], &m, &rwork[m *
		     m + 1], info);
	    zlacrm_(n, &m, &z__[start * z_dim1 + 1], ldz, &rwork[1], &m, &
		    work[1], n, &rwork[m * m + 1]);
	    zlacpy_("A", n, &m, &work[1], n, &z__[start * z_dim1 + 1], ldz);
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
		zswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1 + 1],
			 &c__1);
	    }
/* L60: */
	}
    }

    work[1].r = (doublereal) lwmin, work[1].i = 0.;
    rwork[1] = (doublereal) lrwmin;
    iwork[1] = liwmin;

    return 0;

/*     End of ZSTEDC */

} /* zstedc_ */

/* Subroutine */ int zsteqr_(char *compz, integer *n, doublereal *d__,
	doublereal *e, doublecomplex *z__, integer *ldz, doublereal *work,
	integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal), d_sign(doublereal *, doublereal *);

    /* Local variables */
    static doublereal b, c__, f, g;
    static integer i__, j, k, l, m;
    static doublereal p, r__, s;
    static integer l1, ii, mm, lm1, mm1, nm1;
    static doublereal rt1, rt2, eps;
    static integer lsv;
    static doublereal tst, eps2;
    static integer lend, jtot;
    extern /* Subroutine */ int dlae2_(doublereal *, doublereal *, doublereal
	    *, doublereal *, doublereal *);
    extern logical lsame_(char *, char *);
    static doublereal anorm;
    extern /* Subroutine */ int zlasr_(char *, char *, char *, integer *,
	    integer *, doublereal *, doublereal *, doublecomplex *, integer *), zswap_(integer *, doublecomplex *,
	    integer *, doublecomplex *, integer *), dlaev2_(doublereal *,
	    doublereal *, doublereal *, doublereal *, doublereal *,
	    doublereal *, doublereal *);
    static integer lendm1, lendp1;

    static integer iscale;
    extern /* Subroutine */ int dlascl_(char *, integer *, integer *,
	    doublereal *, doublereal *, integer *, integer *, doublereal *,
	    integer *, integer *);
    static doublereal safmin;
    extern /* Subroutine */ int dlartg_(doublereal *, doublereal *,
	    doublereal *, doublereal *, doublereal *);
    static doublereal safmax;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int dlasrt_(char *, integer *, doublereal *,
	    integer *);
    static integer lendsv;
    static doublereal ssfmin;
    static integer nmaxit, icompz;
    static doublereal ssfmax;
    extern /* Subroutine */ int zlaset_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZSTEQR computes all eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the implicit QL or QR method.
    The eigenvectors of a full or band complex Hermitian matrix can also
    be found if ZHETRD or ZHPTRD or ZHBTRD has been used to reduce this
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

    D       (input/output) DOUBLE PRECISION array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) DOUBLE PRECISION array, dimension (N-1)
            On entry, the (n-1) subdiagonal elements of the tridiagonal
            matrix.
            On exit, E has been destroyed.

    Z       (input/output) COMPLEX*16 array, dimension (LDZ, N)
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

    WORK    (workspace) DOUBLE PRECISION array, dimension (max(1,2*N-2))
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
    } else if ((*ldz < 1) || (icompz > 0 && *ldz < max(1,*n))) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZSTEQR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (icompz == 2) {
	    i__1 = z_dim1 + 1;
	    z__[i__1].r = 1., z__[i__1].i = 0.;
	}
	return 0;
    }

/*     Determine the unit roundoff and over/underflow thresholds. */

    eps = EPSILON;
/* Computing 2nd power */
    d__1 = eps;
    eps2 = d__1 * d__1;
    safmin = SAFEMINIMUM;
    safmax = 1. / safmin;
    ssfmax = sqrt(safmax) / 3.;
    ssfmin = sqrt(safmin) / eps2;

/*
       Compute the eigenvalues and eigenvectors of the tridiagonal
       matrix.
*/

    if (icompz == 2) {
	zlaset_("Full", n, n, &c_b59, &c_b60, &z__[z_offset], ldz);
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
	e[l1 - 1] = 0.;
    }
    if (l1 <= nm1) {
	i__1 = nm1;
	for (m = l1; m <= i__1; ++m) {
	    tst = (d__1 = e[m], abs(d__1));
	    if (tst == 0.) {
		goto L30;
	    }
	    if (tst <= sqrt((d__1 = d__[m], abs(d__1))) * sqrt((d__2 = d__[m
		    + 1], abs(d__2))) * eps) {
		e[m] = 0.;
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
    anorm = dlanst_("I", &i__1, &d__[l], &e[l]);
    iscale = 0;
    if (anorm == 0.) {
	goto L10;
    }
    if (anorm > ssfmax) {
	iscale = 1;
	i__1 = lend - l + 1;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &e[l], n,
		info);
    } else if (anorm < ssfmin) {
	iscale = 2;
	i__1 = lend - l + 1;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &d__[l], n,
		info);
	i__1 = lend - l;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &e[l], n,
		info);
    }

/*     Choose between QL and QR iteration */

    if ((d__1 = d__[lend], abs(d__1)) < (d__2 = d__[l], abs(d__2))) {
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
		d__2 = (d__1 = e[m], abs(d__1));
		tst = d__2 * d__2;
		if (tst <= eps2 * (d__1 = d__[m], abs(d__1)) * (d__2 = d__[m
			+ 1], abs(d__2)) + safmin) {
		    goto L60;
		}
/* L50: */
	    }
	}

	m = lend;

L60:
	if (m < lend) {
	    e[m] = 0.;
	}
	p = d__[l];
	if (m == l) {
	    goto L80;
	}

/*
          If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
          to compute its eigensystem.
*/

	if (m == l + 1) {
	    if (icompz > 0) {
		dlaev2_(&d__[l], &e[l], &d__[l + 1], &rt1, &rt2, &c__, &s);
		work[l] = c__;
		work[*n - 1 + l] = s;
		zlasr_("R", "V", "B", n, &c__2, &work[l], &work[*n - 1 + l], &
			z__[l * z_dim1 + 1], ldz);
	    } else {
		dlae2_(&d__[l], &e[l], &d__[l + 1], &rt1, &rt2);
	    }
	    d__[l] = rt1;
	    d__[l + 1] = rt2;
	    e[l] = 0.;
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

	g = (d__[l + 1] - p) / (e[l] * 2.);
	r__ = dlapy2_(&g, &c_b1015);
	g = d__[m] - p + e[l] / (g + d_sign(&r__, &g));

	s = 1.;
	c__ = 1.;
	p = 0.;

/*        Inner loop */

	mm1 = m - 1;
	i__1 = l;
	for (i__ = mm1; i__ >= i__1; --i__) {
	    f = s * e[i__];
	    b = c__ * e[i__];
	    dlartg_(&g, &f, &c__, &s, &r__);
	    if (i__ != m - 1) {
		e[i__ + 1] = r__;
	    }
	    g = d__[i__ + 1] - p;
	    r__ = (d__[i__] - g) * s + c__ * 2. * b;
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
	    zlasr_("R", "V", "B", n, &mm, &work[l], &work[*n - 1 + l], &z__[l
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
		d__2 = (d__1 = e[m - 1], abs(d__1));
		tst = d__2 * d__2;
		if (tst <= eps2 * (d__1 = d__[m], abs(d__1)) * (d__2 = d__[m
			- 1], abs(d__2)) + safmin) {
		    goto L110;
		}
/* L100: */
	    }
	}

	m = lend;

L110:
	if (m > lend) {
	    e[m - 1] = 0.;
	}
	p = d__[l];
	if (m == l) {
	    goto L130;
	}

/*
          If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
          to compute its eigensystem.
*/

	if (m == l - 1) {
	    if (icompz > 0) {
		dlaev2_(&d__[l - 1], &e[l - 1], &d__[l], &rt1, &rt2, &c__, &s)
			;
		work[m] = c__;
		work[*n - 1 + m] = s;
		zlasr_("R", "V", "F", n, &c__2, &work[m], &work[*n - 1 + m], &
			z__[(l - 1) * z_dim1 + 1], ldz);
	    } else {
		dlae2_(&d__[l - 1], &e[l - 1], &d__[l], &rt1, &rt2);
	    }
	    d__[l - 1] = rt1;
	    d__[l] = rt2;
	    e[l - 1] = 0.;
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

	g = (d__[l - 1] - p) / (e[l - 1] * 2.);
	r__ = dlapy2_(&g, &c_b1015);
	g = d__[m] - p + e[l - 1] / (g + d_sign(&r__, &g));

	s = 1.;
	c__ = 1.;
	p = 0.;

/*        Inner loop */

	lm1 = l - 1;
	i__1 = lm1;
	for (i__ = m; i__ <= i__1; ++i__) {
	    f = s * e[i__];
	    b = c__ * e[i__];
	    dlartg_(&g, &f, &c__, &s, &r__);
	    if (i__ != m) {
		e[i__ - 1] = r__;
	    }
	    g = d__[i__] - p;
	    r__ = (d__[i__ + 1] - g) * s + c__ * 2. * b;
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
	    zlasr_("R", "V", "F", n, &mm, &work[m], &work[*n - 1 + m], &z__[m
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
	dlascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
	i__1 = lendsv - lsv;
	dlascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &e[lsv], n,
		info);
    } else if (iscale == 2) {
	i__1 = lendsv - lsv + 1;
	dlascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &d__[lsv],
		n, info);
	i__1 = lendsv - lsv;
	dlascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &e[lsv], n,
		info);
    }

/*
       Check for no convergence to an eigenvalue after a total
       of N*MAXIT iterations.
*/

    if (jtot == nmaxit) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (e[i__] != 0.) {
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

	dlasrt_("I", n, &d__[1], info);

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
		zswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[k * z_dim1 + 1],
			 &c__1);
	    }
/* L180: */
	}
    }
    return 0;

/*     End of ZSTEQR */

} /* zsteqr_ */

/* Subroutine */ int ztrevc_(char *side, char *howmny, logical *select,
	integer *n, doublecomplex *t, integer *ldt, doublecomplex *vl,
	integer *ldvl, doublecomplex *vr, integer *ldvr, integer *mm, integer
	*m, doublecomplex *work, doublereal *rwork, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1,
	    i__2, i__3, i__4, i__5;
    doublereal d__1, d__2, d__3;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, j, k, ii, ki, is;
    static doublereal ulp;
    static logical allv;
    static doublereal unfl, ovfl, smin;
    static logical over;
    static doublereal scale;
    extern logical lsame_(char *, char *);
    static doublereal remax;
    static logical leftv, bothv;
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *, doublecomplex *, doublecomplex *, integer *);
    static logical somev;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *), dlabad_(doublereal *, doublereal *);

    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    static logical rightv;
    extern doublereal dzasum_(integer *, doublecomplex *, integer *);
    static doublereal smlnum;
    extern /* Subroutine */ int zlatrs_(char *, char *, char *, char *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublereal *, doublereal *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZTREVC computes some or all of the right and/or left eigenvectors of
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

    T       (input/output) COMPLEX*16 array, dimension (LDT,N)
            The upper triangular matrix T.  T is modified, but restored
            on exit.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= max(1,N).

    VL      (input/output) COMPLEX*16 array, dimension (LDVL,MM)
            On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
            contain an N-by-N matrix Q (usually the unitary matrix Q of
            Schur vectors returned by ZHSEQR).
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

    VR      (input/output) COMPLEX*16 array, dimension (LDVR,MM)
            On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
            contain an N-by-N matrix Q (usually the unitary matrix Q of
            Schur vectors returned by ZHSEQR).
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

    WORK    (workspace) COMPLEX*16 array, dimension (2*N)

    RWORK   (workspace) DOUBLE PRECISION array, dimension (N)

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
    rightv = (lsame_(side, "R")) || (bothv);
    leftv = (lsame_(side, "L")) || (bothv);

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
    } else if ((*ldvl < 1) || (leftv && *ldvl < *n)) {
	*info = -8;
    } else if ((*ldvr < 1) || (rightv && *ldvr < *n)) {
	*info = -10;
    } else if (*mm < *m) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTREVC", &i__1);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     Set the constants to control overflow. */

    unfl = SAFEMINIMUM;
    ovfl = 1. / unfl;
    dlabad_(&unfl, &ovfl);
    ulp = PRECISION;
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

    rwork[1] = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	i__2 = j - 1;
	rwork[j] = dzasum_(&i__2, &t[j * t_dim1 + 1], &c__1);
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
	    d__3 = ulp * ((d__1 = t[i__1].r, abs(d__1)) + (d__2 = d_imag(&t[
		    ki + ki * t_dim1]), abs(d__2)));
	    smin = max(d__3,smlnum);

	    work[1].r = 1., work[1].i = 0.;

/*           Form right-hand side. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__2 = k;
		i__3 = k + ki * t_dim1;
		z__1.r = -t[i__3].r, z__1.i = -t[i__3].i;
		work[i__2].r = z__1.r, work[i__2].i = z__1.i;
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
		z__1.r = t[i__3].r - t[i__4].r, z__1.i = t[i__3].i - t[i__4]
			.i;
		t[i__2].r = z__1.r, t[i__2].i = z__1.i;
		i__2 = k + k * t_dim1;
		if ((d__1 = t[i__2].r, abs(d__1)) + (d__2 = d_imag(&t[k + k *
			t_dim1]), abs(d__2)) < smin) {
		    i__3 = k + k * t_dim1;
		    t[i__3].r = smin, t[i__3].i = 0.;
		}
/* L50: */
	    }

	    if (ki > 1) {
		i__1 = ki - 1;
		zlatrs_("Upper", "No transpose", "Non-unit", "Y", &i__1, &t[
			t_offset], ldt, &work[1], &scale, &rwork[1], info);
		i__1 = ki;
		work[i__1].r = scale, work[i__1].i = 0.;
	    }

/*           Copy the vector x or Q*x to VR and normalize. */

	    if (! over) {
		zcopy_(&ki, &work[1], &c__1, &vr[is * vr_dim1 + 1], &c__1);

		ii = izamax_(&ki, &vr[is * vr_dim1 + 1], &c__1);
		i__1 = ii + is * vr_dim1;
		remax = 1. / ((d__1 = vr[i__1].r, abs(d__1)) + (d__2 = d_imag(
			&vr[ii + is * vr_dim1]), abs(d__2)));
		zdscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);

		i__1 = *n;
		for (k = ki + 1; k <= i__1; ++k) {
		    i__2 = k + is * vr_dim1;
		    vr[i__2].r = 0., vr[i__2].i = 0.;
/* L60: */
		}
	    } else {
		if (ki > 1) {
		    i__1 = ki - 1;
		    z__1.r = scale, z__1.i = 0.;
		    zgemv_("N", n, &i__1, &c_b60, &vr[vr_offset], ldvr, &work[
			    1], &c__1, &z__1, &vr[ki * vr_dim1 + 1], &c__1);
		}

		ii = izamax_(n, &vr[ki * vr_dim1 + 1], &c__1);
		i__1 = ii + ki * vr_dim1;
		remax = 1. / ((d__1 = vr[i__1].r, abs(d__1)) + (d__2 = d_imag(
			&vr[ii + ki * vr_dim1]), abs(d__2)));
		zdscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
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
	    d__3 = ulp * ((d__1 = t[i__2].r, abs(d__1)) + (d__2 = d_imag(&t[
		    ki + ki * t_dim1]), abs(d__2)));
	    smin = max(d__3,smlnum);

	    i__2 = *n;
	    work[i__2].r = 1., work[i__2].i = 0.;

/*           Form right-hand side. */

	    i__2 = *n;
	    for (k = ki + 1; k <= i__2; ++k) {
		i__3 = k;
		d_cnjg(&z__2, &t[ki + k * t_dim1]);
		z__1.r = -z__2.r, z__1.i = -z__2.i;
		work[i__3].r = z__1.r, work[i__3].i = z__1.i;
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
		z__1.r = t[i__4].r - t[i__5].r, z__1.i = t[i__4].i - t[i__5]
			.i;
		t[i__3].r = z__1.r, t[i__3].i = z__1.i;
		i__3 = k + k * t_dim1;
		if ((d__1 = t[i__3].r, abs(d__1)) + (d__2 = d_imag(&t[k + k *
			t_dim1]), abs(d__2)) < smin) {
		    i__4 = k + k * t_dim1;
		    t[i__4].r = smin, t[i__4].i = 0.;
		}
/* L100: */
	    }

	    if (ki < *n) {
		i__2 = *n - ki;
		zlatrs_("Upper", "Conjugate transpose", "Non-unit", "Y", &
			i__2, &t[ki + 1 + (ki + 1) * t_dim1], ldt, &work[ki +
			1], &scale, &rwork[1], info);
		i__2 = ki;
		work[i__2].r = scale, work[i__2].i = 0.;
	    }

/*           Copy the vector x or Q*x to VL and normalize. */

	    if (! over) {
		i__2 = *n - ki + 1;
		zcopy_(&i__2, &work[ki], &c__1, &vl[ki + is * vl_dim1], &c__1)
			;

		i__2 = *n - ki + 1;
		ii = izamax_(&i__2, &vl[ki + is * vl_dim1], &c__1) + ki - 1;
		i__2 = ii + is * vl_dim1;
		remax = 1. / ((d__1 = vl[i__2].r, abs(d__1)) + (d__2 = d_imag(
			&vl[ii + is * vl_dim1]), abs(d__2)));
		i__2 = *n - ki + 1;
		zdscal_(&i__2, &remax, &vl[ki + is * vl_dim1], &c__1);

		i__2 = ki - 1;
		for (k = 1; k <= i__2; ++k) {
		    i__3 = k + is * vl_dim1;
		    vl[i__3].r = 0., vl[i__3].i = 0.;
/* L110: */
		}
	    } else {
		if (ki < *n) {
		    i__2 = *n - ki;
		    z__1.r = scale, z__1.i = 0.;
		    zgemv_("N", n, &i__2, &c_b60, &vl[(ki + 1) * vl_dim1 + 1],
			     ldvl, &work[ki + 1], &c__1, &z__1, &vl[ki *
			    vl_dim1 + 1], &c__1);
		}

		ii = izamax_(n, &vl[ki * vl_dim1 + 1], &c__1);
		i__2 = ii + ki * vl_dim1;
		remax = 1. / ((d__1 = vl[i__2].r, abs(d__1)) + (d__2 = d_imag(
			&vl[ii + ki * vl_dim1]), abs(d__2)));
		zdscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);
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

/*     End of ZTREVC */

} /* ztrevc_ */

/* Subroutine */ int ztrti2_(char *uplo, char *diag, integer *n,
	doublecomplex *a, integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublecomplex z__1;

    /* Builtin functions */
    void z_div(doublecomplex *, doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer j;
    static doublecomplex ajj;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *);
    static logical upper;
    extern /* Subroutine */ int ztrmv_(char *, char *, char *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(char *, integer *);
    static logical nounit;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZTRTI2 computes the inverse of a complex upper or lower triangular
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZTRTI2", &i__1);
	return 0;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		i__2 = j + j * a_dim1;
		z_div(&z__1, &c_b60, &a[j + j * a_dim1]);
		a[i__2].r = z__1.r, a[i__2].i = z__1.i;
		i__2 = j + j * a_dim1;
		z__1.r = -a[i__2].r, z__1.i = -a[i__2].i;
		ajj.r = z__1.r, ajj.i = z__1.i;
	    } else {
		z__1.r = -1., z__1.i = -0.;
		ajj.r = z__1.r, ajj.i = z__1.i;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    ztrmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a[j * a_dim1 + 1], &c__1);
	    i__2 = j - 1;
	    zscal_(&i__2, &ajj, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		i__1 = j + j * a_dim1;
		z_div(&z__1, &c_b60, &a[j + j * a_dim1]);
		a[i__1].r = z__1.r, a[i__1].i = z__1.i;
		i__1 = j + j * a_dim1;
		z__1.r = -a[i__1].r, z__1.i = -a[i__1].i;
		ajj.r = z__1.r, ajj.i = z__1.i;
	    } else {
		z__1.r = -1., z__1.i = -0.;
		ajj.r = z__1.r, ajj.i = z__1.i;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		ztrmv_("Lower", "No transpose", diag, &i__1, &a[j + 1 + (j +
			1) * a_dim1], lda, &a[j + 1 + j * a_dim1], &c__1);
		i__1 = *n - j;
		zscal_(&i__1, &ajj, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of ZTRTI2 */

} /* ztrti2_ */

/* Subroutine */ int ztrtri_(char *uplo, char *diag, integer *n,
	doublecomplex *a, integer *lda, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, i__1, i__2, i__3[2], i__4, i__5;
    doublecomplex z__1;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    static integer j, jb, nb, nn;
    extern logical lsame_(char *, char *);
    static logical upper;
    extern /* Subroutine */ int ztrmm_(char *, char *, char *, char *,
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *),
	    ztrsm_(char *, char *, char *, char *, integer *, integer *,
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *,
	    integer *), ztrti2_(char *, char *
	    , integer *, doublecomplex *, integer *, integer *), xerbla_(char *, integer *);
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

    ZTRTRI computes the inverse of a complex upper or lower triangular
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

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
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
	xerbla_("ZTRTRI", &i__1);
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
	    if (a[i__2].r == 0. && a[i__2].i == 0.) {
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
    nb = ilaenv_(&c__1, "ZTRTRI", ch__1, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    if ((nb <= 1) || (nb >= *n)) {

/*        Use unblocked code */

	ztrti2_(uplo, diag, n, &a[a_offset], lda, info);
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
		ztrmm_("Left", "Upper", "No transpose", diag, &i__4, &jb, &
			c_b60, &a[a_offset], lda, &a[j * a_dim1 + 1], lda);
		i__4 = j - 1;
		z__1.r = -1., z__1.i = -0.;
		ztrsm_("Right", "Upper", "No transpose", diag, &i__4, &jb, &
			z__1, &a[j + j * a_dim1], lda, &a[j * a_dim1 + 1],
			lda);

/*              Compute inverse of current diagonal block */

		ztrti2_("Upper", diag, &jb, &a[j + j * a_dim1], lda, info);
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
		    ztrmm_("Left", "Lower", "No transpose", diag, &i__1, &jb,
			    &c_b60, &a[j + jb + (j + jb) * a_dim1], lda, &a[j
			    + jb + j * a_dim1], lda);
		    i__1 = *n - j - jb + 1;
		    z__1.r = -1., z__1.i = -0.;
		    ztrsm_("Right", "Lower", "No transpose", diag, &i__1, &jb,
			     &z__1, &a[j + j * a_dim1], lda, &a[j + jb + j *
			    a_dim1], lda);
		}

/*              Compute inverse of current diagonal block */

		ztrti2_("Lower", diag, &jb, &a[j + j * a_dim1], lda, info);
/* L30: */
	    }
	}
    }

    return 0;

/*     End of ZTRTRI */

} /* ztrtri_ */

/* Subroutine */ int zung2r_(integer *m, integer *n, integer *k,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZUNG2R generates an m by n complex matrix Q with orthonormal columns,
    which is defined as the first n columns of a product of k elementary
    reflectors of order m

          Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by ZGEQRF in the first k columns of its array
            argument A.
            On exit, the m by n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    WORK    (workspace) COMPLEX*16 array, dimension (N)

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
    } else if ((*n < 0) || (*n > *m)) {
	*info = -2;
    } else if ((*k < 0) || (*k > *n)) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNG2R", &i__1);
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
	    a[i__3].r = 0., a[i__3].i = 0.;
/* L10: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;
/* L20: */
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i) to A(i:m,i:n) from the left */

	if (i__ < *n) {
	    i__1 = i__ + i__ * a_dim1;
	    a[i__1].r = 1., a[i__1].i = 0.;
	    i__1 = *m - i__ + 1;
	    i__2 = *n - i__;
	    zlarf_("Left", &i__1, &i__2, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	}
	if (i__ < *m) {
	    i__1 = *m - i__;
	    i__2 = i__;
	    z__1.r = -tau[i__2].r, z__1.i = -tau[i__2].i;
	    zscal_(&i__1, &z__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
	}
	i__1 = i__ + i__ * a_dim1;
	i__2 = i__;
	z__1.r = 1. - tau[i__2].r, z__1.i = 0. - tau[i__2].i;
	a[i__1].r = z__1.r, a[i__1].i = z__1.i;

/*        Set A(1:i-1,i) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = l + i__ * a_dim1;
	    a[i__2].r = 0., a[i__2].i = 0.;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of ZUNG2R */

} /* zung2r_ */

/* Subroutine */ int zungbr_(char *vect, integer *m, integer *n, integer *k,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info)
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
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int zunglq_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *), zungqr_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNGBR generates one of the complex unitary matrices Q or P**H
    determined by ZGEBRD when reducing a complex matrix A to bidiagonal
    form: A = Q * B * P**H.  Q and P**H are defined as products of
    elementary reflectors H(i) or G(i) respectively.

    If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
    is of order M:
    if m >= k, Q = H(1) H(2) . . . H(k) and ZUNGBR returns the first n
    columns of Q, where m >= n >= k;
    if m < k, Q = H(1) H(2) . . . H(m-1) and ZUNGBR returns Q as an
    M-by-M matrix.

    If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**H
    is of order N:
    if k < n, P**H = G(k) . . . G(2) G(1) and ZUNGBR returns the first m
    rows of P**H, where n >= m >= k;
    if k >= n, P**H = G(n-1) . . . G(2) G(1) and ZUNGBR returns P**H as
    an N-by-N matrix.

    Arguments
    =========

    VECT    (input) CHARACTER*1
            Specifies whether the matrix Q or the matrix P**H is
            required, as defined in the transformation applied by ZGEBRD:
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
            matrix reduced by ZGEBRD.
            If VECT = 'P', the number of rows in the original K-by-N
            matrix reduced by ZGEBRD.
            K >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by ZGEBRD.
            On exit, the M-by-N matrix Q or P**H.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= M.

    TAU     (input) COMPLEX*16 array, dimension
                                  (min(M,K)) if VECT = 'Q'
                                  (min(N,K)) if VECT = 'P'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i), which determines Q or P**H, as
            returned by ZGEBRD in its array argument TAUQ or TAUP.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    } else if (((*n < 0) || (wantq && ((*n > *m) || (*n < min(*m,*k))))) || (!
	     wantq && ((*m > *n) || (*m < min(*n,*k))))) {
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
	    nb = ilaenv_(&c__1, "ZUNGQR", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	} else {
	    nb = ilaenv_(&c__1, "ZUNGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (
		    ftnlen)1);
	}
	lwkopt = max(1,mn) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNGBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if ((*m == 0) || (*n == 0)) {
	work[1].r = 1., work[1].i = 0.;
	return 0;
    }

    if (wantq) {

/*
          Form Q, determined by a call to ZGEBRD to reduce an m-by-k
          matrix
*/

	if (*m >= *k) {

/*           If m >= k, assume m >= n >= k */

	    zungqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
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
		a[i__1].r = 0., a[i__1].i = 0.;
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
	    a[i__1].r = 1., a[i__1].i = 0.;
	    i__1 = *m;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + a_dim1;
		a[i__2].r = 0., a[i__2].i = 0.;
/* L30: */
	    }
	    if (*m > 1) {

/*              Form Q(2:m,2:m) */

		i__1 = *m - 1;
		i__2 = *m - 1;
		i__3 = *m - 1;
		zungqr_(&i__1, &i__2, &i__3, &a[((a_dim1) << (1)) + 2], lda, &
			tau[1], &work[1], lwork, &iinfo);
	    }
	}
    } else {

/*
          Form P', determined by a call to ZGEBRD to reduce a k-by-n
          matrix
*/

	if (*k < *n) {

/*           If k < n, assume k <= m <= n */

	    zunglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*
             If k >= n, assume m = n

             Shift the vectors which define the elementary reflectors one
             row downward, and set the first row and column of P' to
             those of the unit matrix
*/

	    i__1 = a_dim1 + 1;
	    a[i__1].r = 1., a[i__1].i = 0.;
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + a_dim1;
		a[i__2].r = 0., a[i__2].i = 0.;
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
		a[i__2].r = 0., a[i__2].i = 0.;
/* L60: */
	    }
	    if (*n > 1) {

/*              Form P'(2:n,2:n) */

		i__1 = *n - 1;
		i__2 = *n - 1;
		i__3 = *n - 1;
		zunglq_(&i__1, &i__2, &i__3, &a[((a_dim1) << (1)) + 2], lda, &
			tau[1], &work[1], lwork, &iinfo);
	    }
	}
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNGBR */

} /* zungbr_ */

/* Subroutine */ int zunghr_(integer *n, integer *ilo, integer *ihi,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, nb, nh, iinfo;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int zungqr_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNGHR generates a complex unitary matrix Q which is defined as the
    product of IHI-ILO elementary reflectors of order N, as returned by
    ZGEHRD:

    Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Arguments
    =========

    N       (input) INTEGER
            The order of the matrix Q. N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            ILO and IHI must have the same values as in the previous call
            of ZGEHRD. Q is equal to the unit matrix except in the
            submatrix Q(ilo+1:ihi,ilo+1:ihi).
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the vectors which define the elementary reflectors,
            as returned by ZGEHRD.
            On exit, the N-by-N unitary matrix Q.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,N).

    TAU     (input) COMPLEX*16 array, dimension (N-1)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEHRD.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    } else if ((*ilo < 1) || (*ilo > max(1,*n))) {
	*info = -2;
    } else if ((*ihi < min(*ilo,*n)) || (*ihi > *n)) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,nh) && ! lquery) {
	*info = -8;
    }

    if (*info == 0) {
	nb = ilaenv_(&c__1, "ZUNGQR", " ", &nh, &nh, &nh, &c_n1, (ftnlen)6, (
		ftnlen)1);
	lwkopt = max(1,nh) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNGHR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1].r = 1., work[1].i = 0.;
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
	    a[i__3].r = 0., a[i__3].i = 0.;
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
	    a[i__3].r = 0., a[i__3].i = 0.;
/* L30: */
	}
/* L40: */
    }
    i__1 = *ilo;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0., a[i__3].i = 0.;
/* L50: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;
/* L60: */
    }
    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3].r = 0., a[i__3].i = 0.;
/* L70: */
	}
	i__2 = j + j * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;
/* L80: */
    }

    if (nh > 0) {

/*        Generate Q(ilo+1:ihi,ilo+1:ihi) */

	zungqr_(&nh, &nh, &nh, &a[*ilo + 1 + (*ilo + 1) * a_dim1], lda, &tau[*
		ilo], &work[1], lwork, &iinfo);
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNGHR */

} /* zunghr_ */

/* Subroutine */ int zungl2_(integer *m, integer *n, integer *k,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, j, l;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *,
	    doublecomplex *, integer *), zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlacgv_(integer *, doublecomplex *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNGL2 generates an m-by-n complex matrix Q with orthonormal rows,
    which is defined as the first m rows of a product of k elementary
    reflectors of order n

          Q  =  H(k)' . . . H(2)' H(1)'

    as returned by ZGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by ZGELQF in the first k rows of its array argument A.
            On exit, the m by n matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGELQF.

    WORK    (workspace) COMPLEX*16 array, dimension (M)

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
    } else if ((*k < 0) || (*k > *m)) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNGL2", &i__1);
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
		a[i__3].r = 0., a[i__3].i = 0.;
/* L10: */
	    }
	    if (j > *k && j <= *m) {
		i__2 = j + j * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;
	    }
/* L20: */
	}
    }

    for (i__ = *k; i__ >= 1; --i__) {

/*        Apply H(i)' to A(i:m,i:n) from the right */

	if (i__ < *n) {
	    i__1 = *n - i__;
	    zlacgv_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	    if (i__ < *m) {
		i__1 = i__ + i__ * a_dim1;
		a[i__1].r = 1., a[i__1].i = 0.;
		i__1 = *m - i__;
		i__2 = *n - i__ + 1;
		d_cnjg(&z__1, &tau[i__]);
		zlarf_("Right", &i__1, &i__2, &a[i__ + i__ * a_dim1], lda, &
			z__1, &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    i__1 = *n - i__;
	    i__2 = i__;
	    z__1.r = -tau[i__2].r, z__1.i = -tau[i__2].i;
	    zscal_(&i__1, &z__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	    i__1 = *n - i__;
	    zlacgv_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	i__1 = i__ + i__ * a_dim1;
	d_cnjg(&z__2, &tau[i__]);
	z__1.r = 1. - z__2.r, z__1.i = 0. - z__2.i;
	a[i__1].r = z__1.r, a[i__1].i = z__1.i;

/*        Set A(i,1:i-1) to zero */

	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = i__ + l * a_dim1;
	    a[i__2].r = 0., a[i__2].i = 0.;
/* L30: */
	}
/* L40: */
    }
    return 0;

/*     End of ZUNGL2 */

} /* zungl2_ */

/* Subroutine */ int zunglq_(integer *m, integer *n, integer *k,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int zungl2_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static logical lquery;
    static integer lwkopt;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNGLQ generates an M-by-N complex matrix Q with orthonormal rows,
    which is defined as the first M rows of a product of K elementary
    reflectors of order N

          Q  =  H(k)' . . . H(2)' H(1)'

    as returned by ZGELQF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. N >= M.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. M >= K >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the i-th row must contain the vector which defines
            the elementary reflector H(i), for i = 1,2,...,k, as returned
            by ZGELQF in the first k rows of its array argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGELQF.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    nb = ilaenv_(&c__1, "ZUNGLQ", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*m) * nb;
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if ((*k < 0) || (*k > *m)) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNGLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m <= 0) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "ZUNGLQ", " ", m, n, k, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZUNGLQ", " ", m, n, k, &c_n1,
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
		a[i__3].r = 0., a[i__3].i = 0.;
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
	zungl2_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
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
		zlarft_("Forward", "Rowwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H' to A(i+ib:m,i:n) from the right */

		i__2 = *m - i__ - ib + 1;
		i__3 = *n - i__ + 1;
		zlarfb_("Right", "Conjugate transpose", "Forward", "Rowwise",
			&i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[
			ib + 1], &ldwork);
	    }

/*           Apply H' to columns i:n of current block */

	    i__2 = *n - i__ + 1;
	    zungl2_(&ib, &i__2, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set columns 1:i-1 of current block to zero */

	    i__2 = i__ - 1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + ib - 1;
		for (l = i__; l <= i__3; ++l) {
		    i__4 = l + j * a_dim1;
		    a[i__4].r = 0., a[i__4].i = 0.;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1].r = (doublereal) iws, work[1].i = 0.;
    return 0;

/*     End of ZUNGLQ */

} /* zunglq_ */

/* Subroutine */ int zungqr_(integer *m, integer *n, integer *k,
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    static integer i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int zung2r_(integer *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNGQR generates an M-by-N complex matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)
            On entry, the i-th column must contain the vector which
            defines the elementary reflector H(i), for i = 1,2,...,k, as
            returned by ZGEQRF in the first k columns of its array
            argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    nb = ilaenv_(&c__1, "ZUNGQR", " ", m, n, k, &c_n1, (ftnlen)6, (ftnlen)1);
    lwkopt = max(1,*n) * nb;
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if ((*n < 0) || (*n > *m)) {
	*info = -2;
    } else if ((*k < 0) || (*k > *n)) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNGQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	work[1].r = 1., work[1].i = 0.;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "ZUNGQR", " ", m, n, k, &c_n1, (
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZUNGQR", " ", m, n, k, &c_n1,
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
		a[i__3].r = 0., a[i__3].i = 0.;
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
	zung2r_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
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
		zlarft_("Forward", "Columnwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(i:m,i+ib:n) from the left */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__ - ib + 1;
		zlarfb_("Left", "No transpose", "Forward", "Columnwise", &
			i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &
			work[ib + 1], &ldwork);
	    }

/*           Apply H to rows i:m of current block */

	    i__2 = *m - i__ + 1;
	    zung2r_(&i__2, &ib, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

/*           Set rows 1:i-1 of current block to zero */

	    i__2 = i__ + ib - 1;
	    for (j = i__; j <= i__2; ++j) {
		i__3 = i__ - 1;
		for (l = 1; l <= i__3; ++l) {
		    i__4 = l + j * a_dim1;
		    a[i__4].r = 0., a[i__4].i = 0.;
/* L30: */
		}
/* L40: */
	    }
/* L50: */
	}
    }

    work[1].r = (doublereal) iws, work[1].i = 0.;
    return 0;

/*     End of ZUNGQR */

} /* zungqr_ */

/* Subroutine */ int zunm2l_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, i1, i2, i3, mi, ni, nq;
    static doublecomplex aii;
    static logical left;
    static doublecomplex taui;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZUNM2L overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by ZGEQLF. Q is of order m if SIDE = 'L' and of order n
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

    A       (input) COMPLEX*16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQLF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension
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
    } else if ((*k < 0) || (*k > nq)) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNM2L", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	return 0;
    }

    if ((left && notran) || (! left && ! notran)) {
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
	    d_cnjg(&z__1, &tau[i__]);
	    taui.r = z__1.r, taui.i = z__1.i;
	}
	i__3 = nq - *k + i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = nq - *k + i__ + i__ * a_dim1;
	a[i__3].r = 1., a[i__3].i = 0.;
	zlarf_(side, &mi, &ni, &a[i__ * a_dim1 + 1], &c__1, &taui, &c__[
		c_offset], ldc, &work[1]);
	i__3 = nq - *k + i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
/* L10: */
    }
    return 0;

/*     End of ZUNM2L */

} /* zunm2l_ */

/* Subroutine */ int zunm2r_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static doublecomplex aii;
    static logical left;
    static doublecomplex taui;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZUNM2R overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order m if SIDE = 'L' and of order n
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

    A       (input) COMPLEX*16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension
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
    } else if ((*k < 0) || (*k > nq)) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNM2R", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	return 0;
    }

    if ((left && ! notran) || (! left && notran)) {
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
	    d_cnjg(&z__1, &tau[i__]);
	    taui.r = z__1.r, taui.i = z__1.i;
	}
	i__3 = i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = 1., a[i__3].i = 0.;
	zlarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], &c__1, &taui, &c__[ic
		+ jc * c_dim1], ldc, &work[1]);
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
/* L10: */
    }
    return 0;

/*     End of ZUNM2R */

} /* zunm2r_ */

/* Subroutine */ int zunmbr_(char *vect, char *side, char *trans, integer *m,
	integer *n, integer *k, doublecomplex *a, integer *lda, doublecomplex
	*tau, doublecomplex *c__, integer *ldc, doublecomplex *work, integer *
	lwork, integer *info)
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
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int zunmlq_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *), zunmqr_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    If VECT = 'Q', ZUNMBR overwrites the general complex M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    If VECT = 'P', ZUNMBR overwrites the general complex M-by-N matrix C
    with
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      P * C          C * P
    TRANS = 'C':      P**H * C       C * P**H

    Here Q and P**H are the unitary matrices determined by ZGEBRD when
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
            matrix reduced by ZGEBRD.
            If VECT = 'P', the number of rows in the original
            matrix reduced by ZGEBRD.
            K >= 0.

    A       (input) COMPLEX*16 array, dimension
                                  (LDA,min(nq,K)) if VECT = 'Q'
                                  (LDA,nq)        if VECT = 'P'
            The vectors which define the elementary reflectors H(i) and
            G(i), whose products determine the matrices Q and P, as
            returned by ZGEBRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If VECT = 'Q', LDA >= max(1,nq);
            if VECT = 'P', LDA >= max(1,min(nq,K)).

    TAU     (input) COMPLEX*16 array, dimension (min(nq,K))
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i) which determines Q or P, as returned
            by ZGEBRD in the array argument TAUQ or TAUP.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q
            or P*C or P**H*C or C*P or C*P**H.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
	if ((applyq && *lda < max(1,nq)) || (! applyq && *lda < max(i__1,i__2)
		)) {
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
		nb = ilaenv_(&c__1, "ZUNMQR", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "ZUNMQR", ch__1, m, &i__1, &i__2, &c_n1, (
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
		nb = ilaenv_(&c__1, "ZUNMLQ", ch__1, &i__1, n, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2, (ftnlen)2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "ZUNMLQ", ch__1, m, &i__1, &i__2, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNMBR", &i__1);
	return 0;
    } else if (lquery) {
    }

/*     Quick return if possible */

    work[1].r = 1., work[1].i = 0.;
    if ((*m == 0) || (*n == 0)) {
	return 0;
    }

    if (applyq) {

/*        Apply Q */

	if (nq >= *k) {

/*           Q was determined by a call to ZGEBRD with nq >= k */

	    zunmqr_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           Q was determined by a call to ZGEBRD with nq < k */

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
	    zunmqr_(side, trans, &mi, &ni, &i__1, &a[a_dim1 + 2], lda, &tau[1]
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

/*           P was determined by a call to ZGEBRD with nq > k */

	    zunmlq_(side, transt, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {

/*           P was determined by a call to ZGEBRD with nq <= k */

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
	    zunmlq_(side, transt, &mi, &ni, &i__1, &a[((a_dim1) << (1)) + 1],
		    lda, &tau[1], &c__[i1 + i2 * c_dim1], ldc, &work[1],
		    lwork, &iinfo);
	}
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNMBR */

} /* zunmbr_ */

/* Subroutine */ int zunml2_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    static integer i__, i1, i2, i3, ic, jc, mi, ni, nq;
    static doublecomplex aii;
    static logical left;
    static doublecomplex taui;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlacgv_(integer *, doublecomplex *, integer *);
    static logical notran;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    ZUNML2 overwrites the general complex m-by-n matrix C with

          Q * C  if SIDE = 'L' and TRANS = 'N', or

          Q'* C  if SIDE = 'L' and TRANS = 'C', or

          C * Q  if SIDE = 'R' and TRANS = 'N', or

          C * Q' if SIDE = 'R' and TRANS = 'C',

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k)' . . . H(2)' H(1)'

    as returned by ZGELQF. Q is of order m if SIDE = 'L' and of order n
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

    A       (input) COMPLEX*16 array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGELQF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the m-by-n matrix C.
            On exit, C is overwritten by Q*C or Q'*C or C*Q' or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace) COMPLEX*16 array, dimension
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
    } else if ((*k < 0) || (*k > nq)) {
	*info = -5;
    } else if (*lda < max(1,*k)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNML2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	return 0;
    }

    if ((left && notran) || (! left && ! notran)) {
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
	    d_cnjg(&z__1, &tau[i__]);
	    taui.r = z__1.r, taui.i = z__1.i;
	} else {
	    i__3 = i__;
	    taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	}
	if (i__ < nq) {
	    i__3 = nq - i__;
	    zlacgv_(&i__3, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	i__3 = i__ + i__ * a_dim1;
	aii.r = a[i__3].r, aii.i = a[i__3].i;
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = 1., a[i__3].i = 0.;
	zlarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], lda, &taui, &c__[ic +
		jc * c_dim1], ldc, &work[1]);
	i__3 = i__ + i__ * a_dim1;
	a[i__3].r = aii.r, a[i__3].i = aii.i;
	if (i__ < nq) {
	    i__3 = nq - i__;
	    zlacgv_(&i__3, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
/* L10: */
    }
    return 0;

/*     End of ZUNML2 */

} /* zunml2_ */

/* Subroutine */ int zunmlq_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork,
	 integer *info)
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
    static doublecomplex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int zunml2_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static logical notran;
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
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

    ZUNMLQ overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k)' . . . H(2)' H(1)'

    as returned by ZGELQF. Q is of order M if SIDE = 'L' and of order N
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

    A       (input) COMPLEX*16 array, dimension
                                 (LDA,M) if SIDE = 'L',
                                 (LDA,N) if SIDE = 'R'
            The i-th row must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGELQF in the first k rows of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,K).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGELQF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    } else if ((*k < 0) || (*k > nq)) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "ZUNMLQ", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNMLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	work[1].r = 1., work[1].i = 0.;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "ZUNMLQ", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if ((nb < nbmin) || (nb >= *k)) {

/*        Use unblocked code */

	zunml2_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if ((left && notran) || (! left && ! notran)) {
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
	    zlarft_("Forward", "Rowwise", &i__4, &ib, &a[i__ + i__ * a_dim1],
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

	    zlarfb_(side, transt, "Forward", "Rowwise", &mi, &ni, &ib, &a[i__
		    + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc * c_dim1],
		    ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNMLQ */

} /* zunmlq_ */

/* Subroutine */ int zunmql_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork,
	 integer *info)
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
    static doublecomplex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int zunm2l_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static logical notran;
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNMQL overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by ZGEQLF. Q is of order M if SIDE = 'L' and of order N
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

    A       (input) COMPLEX*16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQLF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    } else if ((*k < 0) || (*k > nq)) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "ZUNMQL", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNMQL", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	work[1].r = 1., work[1].i = 0.;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "ZUNMQL", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if ((nb < nbmin) || (nb >= *k)) {

/*        Use unblocked code */

	zunm2l_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if ((left && notran) || (! left && ! notran)) {
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
	    zlarft_("Backward", "Columnwise", &i__4, &ib, &a[i__ * a_dim1 + 1]
		    , lda, &tau[i__], t, &c__65);
	    if (left) {

/*              H or H' is applied to C(1:m-k+i+ib-1,1:n) */

		mi = *m - *k + i__ + ib - 1;
	    } else {

/*              H or H' is applied to C(1:m,1:n-k+i+ib-1) */

		ni = *n - *k + i__ + ib - 1;
	    }

/*           Apply H or H' */

	    zlarfb_(side, trans, "Backward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ * a_dim1 + 1], lda, t, &c__65, &c__[c_offset], ldc, &
		    work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNMQL */

} /* zunmql_ */

/* Subroutine */ int zunmqr_(char *side, char *trans, integer *m, integer *n,
	integer *k, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork,
	 integer *info)
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
    static doublecomplex t[4160]	/* was [65][64] */;
    static integer i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    static logical left;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    extern /* Subroutine */ int zunm2r_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *,
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int zlarfb_(char *, char *, char *, char *,
	    integer *, integer *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *, doublecomplex *, integer *,
	    doublecomplex *, integer *);
    static logical notran;
    static integer ldwork;
    extern /* Subroutine */ int zlarft_(char *, char *, integer *, integer *,
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *,
	    integer *);
    static integer lwkopt;
    static logical lquery;


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNMQR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
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

    A       (input) COMPLEX*16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX*16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
    } else if ((*k < 0) || (*k > nq)) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "ZUNMQR", ch__1, m, n, k, &c_n1, (
		ftnlen)6, (ftnlen)2);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZUNMQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (*k == 0)) {
	work[1].r = 1., work[1].i = 0.;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "ZUNMQR", ch__1, m, n, k, &c_n1, (
		    ftnlen)6, (ftnlen)2);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = nw;
    }

    if ((nb < nbmin) || (nb >= *k)) {

/*        Use unblocked code */

	zunm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {

/*        Use blocked code */

	if ((left && ! notran) || (! left && notran)) {
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
	    zlarft_("Forward", "Columnwise", &i__4, &ib, &a[i__ + i__ *
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

	    zlarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ + i__ * a_dim1], lda, t, &c__65, &c__[ic + jc *
		    c_dim1], ldc, &work[1], &ldwork);
/* L10: */
	}
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNMQR */

} /* zunmqr_ */

/* Subroutine */ int zunmtr_(char *side, char *uplo, char *trans, integer *m,
	integer *n, doublecomplex *a, integer *lda, doublecomplex *tau,
	doublecomplex *c__, integer *ldc, doublecomplex *work, integer *lwork,
	 integer *info)
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
    static integer lwkopt;
    static logical lquery;
    extern /* Subroutine */ int zunmql_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *), zunmqr_(char *, char *, integer *, integer *,
	    integer *, doublecomplex *, integer *, doublecomplex *,
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *);


/*
    -- LAPACK routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       June 30, 1999


    Purpose
    =======

    ZUNMTR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H

    where Q is a complex unitary matrix of order nq, with nq = m if
    SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
    nq-1 elementary reflectors, as returned by ZHETRD:

    if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);

    if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).

    Arguments
    =========

    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    UPLO    (input) CHARACTER*1
            = 'U': Upper triangle of A contains elementary reflectors
                   from ZHETRD;
            = 'L': Lower triangle of A contains elementary reflectors
                   from ZHETRD.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Conjugate transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    A       (input) COMPLEX*16 array, dimension
                                 (LDA,M) if SIDE = 'L'
                                 (LDA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by ZHETRD.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.

    TAU     (input) COMPLEX*16 array, dimension
                                 (M-1) if SIDE = 'L'
                                 (N-1) if SIDE = 'R'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZHETRD.

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)
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
		nb = ilaenv_(&c__1, "ZUNMQL", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "ZUNMQL", ch__1, m, &i__2, &i__3, &c_n1, (
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
		nb = ilaenv_(&c__1, "ZUNMQR", ch__1, &i__2, n, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    } else {
/* Writing concatenation */
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "ZUNMQR", ch__1, m, &i__2, &i__3, &c_n1, (
			ftnlen)6, (ftnlen)2);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    }

    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("ZUNMTR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (((*m == 0) || (*n == 0)) || (nq == 1)) {
	work[1].r = 1., work[1].i = 0.;
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

/*        Q was determined by a call to ZHETRD with UPLO = 'U' */

	i__2 = nq - 1;
	zunmql_(side, trans, &mi, &ni, &i__2, &a[((a_dim1) << (1)) + 1], lda,
		&tau[1], &c__[c_offset], ldc, &work[1], lwork, &iinfo);
    } else {

/*        Q was determined by a call to ZHETRD with UPLO = 'L' */

	if (left) {
	    i1 = 2;
	    i2 = 1;
	} else {
	    i1 = 1;
	    i2 = 2;
	}
	i__2 = nq - 1;
	zunmqr_(side, trans, &mi, &ni, &i__2, &a[a_dim1 + 2], lda, &tau[1], &
		c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
    }
    work[1].r = (doublereal) lwkopt, work[1].i = 0.;
    return 0;

/*     End of ZUNMTR */

} /* zunmtr_ */
