/*
 * NOTE: This is generated code. Look in numpy/linalg/lapack_lite for
 *       information on remaking this file.
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
static doublereal c_b32 = 0.;
static real c_b66 = 0.f;

doublereal dlamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static doublereal t;
    static integer it;
    static doublereal rnd, eps, base;
    static integer beta;
    static doublereal emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static doublereal rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static doublereal small, sfmin;
    extern /* Subroutine */ int dlamc2_(integer *, integer *, logical *,
	    doublereal *, integer *, doublereal *, integer *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMCH determines double precision machine parameters.

    Arguments
    =========

    CMACH   (input) CHARACTER*1
            Specifies the value to be returned by DLAMCH:
            = 'E' or 'e',   DLAMCH := eps
            = 'S' or 's ,   DLAMCH := sfmin
            = 'B' or 'b',   DLAMCH := base
            = 'P' or 'p',   DLAMCH := eps*base
            = 'N' or 'n',   DLAMCH := t
            = 'R' or 'r',   DLAMCH := rnd
            = 'M' or 'm',   DLAMCH := emin
            = 'U' or 'u',   DLAMCH := rmin
            = 'L' or 'l',   DLAMCH := emax
            = 'O' or 'o',   DLAMCH := rmax

            where

            eps   = relative machine precision
            sfmin = safe minimum, such that 1/sfmin does not overflow
            base  = base of the machine
            prec  = eps*base
            t     = number of (base) digits in the mantissa
            rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
            emin  = minimum exponent before (gradual) underflow
            rmin  = underflow threshold - base**(emin-1)
            emax  = largest exponent before overflow
            rmax  = overflow threshold  - (base**emax)*(1-eps)

   =====================================================================
*/


    if (first) {
	dlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (doublereal) beta;
	t = (doublereal) it;
	if (lrnd) {
	    rnd = 1.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (doublereal) imin;
	emax = (doublereal) imax;
	sfmin = rmin;
	small = 1. / rmax;
	if (small >= sfmin) {

/*
             Use SMALL plus a bit, to avoid the possibility of rounding
             causing overflow when computing  1/sfmin.
*/

	    sfmin = small * (eps + 1.);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* dlamch_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc1_(integer *beta, integer *t, logical *rnd, logical
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal a, b, c__, f, t1, t2;
    static integer lt;
    static doublereal one, qtr;
    static logical lrnd;
    static integer lbeta;
    static doublereal savec;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC1 determines the machine parameters given by BETA, T, RND, and
    IEEE1.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    IEEE1   (output) LOGICAL
            Specifies whether rounding appears to be done in the IEEE
            'round to nearest' style.

    Further Details
    ===============

    The routine is based on the routine  ENVRON  by Malcolm and
    incorporates suggestions by Gentleman and Marovich. See

       Malcolm M. A. (1972) Algorithms to reveal properties of
          floating-point arithmetic. Comms. of the ACM, 15, 949-951.

       Gentleman W. M. and Marovich S. B. (1974) More on algorithms
          that reveal properties of floating point arithmetic units.
          Comms. of the ACM, 17, 276-277.

   =====================================================================
*/


    if (first) {
	one = 1.;

/*
          LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA,
          IEEE1, T and RND.

          Throughout this routine  we use the function  DLAMC3  to ensure
          that relevant values are  stored and not held in registers,  or
          are not affected by optimizers.

          Compute  a = 2.0**m  with the  smallest positive integer m such
          that

             fl( a + 1.0 ) = a.
*/

	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = dlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = dlamc3_(&c__, &d__1);
	    goto L10;
	}
/*
   +       END WHILE

          Now compute  b = 2.0**m  with the smallest positive integer m
          such that

             fl( a + b ) .gt. a.
*/

	b = 1.;
	c__ = dlamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = dlamc3_(&a, &b);
	    goto L20;
	}
/*
   +       END WHILE

          Now compute the base.  a and c  are neighbouring floating point
          numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so
          their difference is beta. Adding 0.25 to c is to ensure that it
          is truncated to beta and not ( beta - 1 ).
*/

	qtr = one / 4;
	savec = c__;
	d__1 = -a;
	c__ = dlamc3_(&c__, &d__1);
	lbeta = (integer) (c__ + qtr);

/*
          Now determine whether rounding or chopping occurs,  by adding a
          bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a.
*/

	b = (doublereal) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = dlamc3_(&d__1, &d__2);
	c__ = dlamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = dlamc3_(&d__1, &d__2);
	c__ = dlamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*
          Try and decide whether rounding is done in the  IEEE  'round to
          nearest' style. B/2 is half a unit in the last place of the two
          numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit
          zero, and SAVEC is odd. Thus adding B/2 to A should not  change
          A, but adding B/2 to SAVEC should change SAVEC.
*/

	d__1 = b / 2;
	t1 = dlamc3_(&d__1, &a);
	d__1 = b / 2;
	t2 = dlamc3_(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*
          Now find  the  mantissa, t.  It should  be the  integer part of
          log to the base beta of a,  however it is safer to determine  t
          by powering.  So we find t as the smallest positive integer for
          which

             fl( beta**t + 1.0 ) = 1.0.
*/

	lt = 0;
	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = dlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = dlamc3_(&c__, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of DLAMC1 */

} /* dlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc2_(integer *beta, integer *t, logical *rnd,
	doublereal *eps, integer *emin, doublereal *rmin, integer *emax,
	doublereal *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 DLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    static doublereal a, b, c__;
    static integer i__, lt;
    static doublereal one, two;
    static logical ieee;
    static doublereal half;
    static logical lrnd;
    static doublereal leps, zero;
    static integer lbeta;
    static doublereal rbase;
    static integer lemin, lemax, gnmin;
    static doublereal small;
    static integer gpmin;
    static doublereal third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int dlamc1_(integer *, integer *, logical *,
	    logical *);
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;
    extern /* Subroutine */ int dlamc4_(integer *, doublereal *, integer *),
	    dlamc5_(integer *, integer *, integer *, logical *, integer *,
	    doublereal *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC2 determines the machine parameters specified in its argument
    list.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    EPS     (output) DOUBLE PRECISION
            The smallest positive number such that

               fl( 1.0 - EPS ) .LT. 1.0,

            where fl denotes the computed value.

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow occurs.

    RMIN    (output) DOUBLE PRECISION
            The smallest normalized number for the machine, given by
            BASE**( EMIN - 1 ), where  BASE  is the floating point value
            of BETA.

    EMAX    (output) INTEGER
            The maximum exponent before overflow occurs.

    RMAX    (output) DOUBLE PRECISION
            The largest positive number for the machine, given by
            BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point
            value of BETA.

    Further Details
    ===============

    The computation of  EPS  is based on a routine PARANOIA by
    W. Kahan of the University of California at Berkeley.

   =====================================================================
*/


    if (first) {
	zero = 0.;
	one = 1.;
	two = 2.;

/*
          LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of
          BETA, T, RND, EPS, EMIN and RMIN.

          Throughout this routine  we use the function  DLAMC3  to ensure
          that relevant values are stored  and not held in registers,  or
          are not affected by optimizers.

          DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1.
*/

	dlamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (doublereal) lbeta;
	i__1 = -lt;
	a = pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = dlamc3_(&b, &d__1);
	third = dlamc3_(&sixth, &sixth);
	d__1 = -half;
	b = dlamc3_(&third, &d__1);
	b = dlamc3_(&b, &sixth);
	b = abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c__ = dlamc3_(&d__1, &d__2);
	    d__1 = -c__;
	    c__ = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c__);
	    d__1 = -b;
	    c__ = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*
          Computation of EPS complete.

          Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)).
          Keep dividing  A by BETA until (gradual) underflow occurs. This
          is detected when we cannot recover the previous A.
*/

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    d__1 = small * rbase;
	    small = dlamc3_(&d__1, &zero);
/* L20: */
	}
	a = dlamc3_(&one, &small);
	dlamc4_(&ngpmin, &one, &lbeta);
	d__1 = -one;
	dlamc4_(&ngnmin, &d__1, &lbeta);
	dlamc4_(&gpmin, &a, &lbeta);
	d__1 = -a;
	dlamc4_(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*
              ( Non twos-complement machines, no gradual underflow;
                e.g.,  VAX )
*/
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*
              ( Non twos-complement machines, with gradual underflow;
                e.g., IEEE standard followers )
*/
	    } else {
		lemin = min(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = max(ngpmin,ngnmin);
/*
              ( Twos-complement machines, no gradual underflow;
                e.g., CYBER 205 )
*/
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - min(ngpmin,ngnmin) == 3) {
		lemin = max(ngpmin,ngnmin) - 1 + lt;
/*
              ( Twos-complement machines with gradual underflow;
                no known machine )
*/
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = min(ngpmin,ngnmin), i__1 = min(i__1,gpmin);
	    lemin = min(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/*
   **
   Comment out this if block if EMIN is ok
*/
	if (iwarn) {
	    first = TRUE_;
	    s_wsfe(&io___58);
	    do_fio(&c__1, (char *)&lemin, (ftnlen)sizeof(integer));
	    e_wsfe();
	}
/*
   **

          Assume IEEE arithmetic if we found denormalised  numbers above,
          or if arithmetic seems to round in the  IEEE style,  determined
          in routine DLAMC1. A true IEEE machine should have both  things
          true; however, faulty machines may have one or the other.
*/

	ieee = ieee || lieee1;

/*
          Compute  RMIN by successive division by  BETA. We could compute
          RMIN as BASE**( EMIN - 1 ),  but some machines underflow during
          this computation.
*/

	lrmin = 1.;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__1 = lrmin * rbase;
	    lrmin = dlamc3_(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	dlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* dlamc2_ */


/* *********************************************************************** */

doublereal dlamc3_(doublereal *a, doublereal *b)
{
    /* System generated locals */
    doublereal ret_val;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC3  is intended to force  A  and  B  to be stored prior to doing
    the addition of  A  and  B ,  for use in situations where optimizers
    might hold one of these in a register.

    Arguments
    =========

    A       (input) DOUBLE PRECISION
    B       (input) DOUBLE PRECISION
            The values A and B.

   =====================================================================
*/


    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* dlamc3_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc4_(integer *emin, doublereal *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal a;
    static integer i__;
    static doublereal b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern doublereal dlamc3_(doublereal *, doublereal *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC4 is a service routine for DLAMC2.

    Arguments
    =========

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow, computed by
            setting A = START and dividing by BASE until the previous A
            can not be recovered.

    START   (input) DOUBLE PRECISION
            The starting point for determining EMIN.

    BASE    (input) INTEGER
            The base of the machine.

   =====================================================================
*/


    a = *start;
    one = 1.;
    rbase = one / *base;
    zero = 0.;
    *emin = 1;
    d__1 = a * rbase;
    b1 = dlamc3_(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/*
   +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND.
      $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP
*/
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = dlamc3_(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = dlamc3_(&d__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = dlamc3_(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = dlamc3_(&d__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* dlamc4_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc5_(integer *beta, integer *p, integer *emin,
	logical *ieee, integer *emax, doublereal *rmax)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static integer i__;
    static doublereal y, z__;
    static integer try__, lexp;
    static doublereal oldy;
    static integer uexp, nbits;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static doublereal recbas;
    static integer exbits, expsum;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    DLAMC5 attempts to compute RMAX, the largest machine floating-point
    number, without overflow.  It assumes that EMAX + abs(EMIN) sum
    approximately to a power of 2.  It will fail on machines where this
    assumption does not hold, for example, the Cyber 205 (EMIN = -28625,
    EMAX = 28718).  It will also fail if the value supplied for EMIN is
    too large (i.e. too close to zero), probably with overflow.

    Arguments
    =========

    BETA    (input) INTEGER
            The base of floating-point arithmetic.

    P       (input) INTEGER
            The number of base BETA digits in the mantissa of a
            floating-point value.

    EMIN    (input) INTEGER
            The minimum exponent before (gradual) underflow.

    IEEE    (input) LOGICAL
            A logical flag specifying whether or not the arithmetic
            system is thought to comply with the IEEE standard.

    EMAX    (output) INTEGER
            The largest exponent before overflow

    RMAX    (output) DOUBLE PRECISION
            The largest machine floating-point number.

   =====================================================================


       First compute LEXP and UEXP, two powers of 2 that bound
       abs(EMIN). We then assume that EMAX + abs(EMIN) will sum
       approximately to the bound that is closest to abs(EMIN).
       (EMAX is the exponent of the required number RMAX).
*/

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*
       Now -LEXP is less than or equal to EMIN, and -UEXP is greater
       than or equal to EMIN. EXBITS is the number of bits needed to
       store the exponent.
*/

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*
       EXPSUM is the exponent range, approximately equal to
       EMAX - EMIN + 1 .
*/

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*
       NBITS is the total number of bits needed to store a
       floating-point number.
*/

    if (nbits % 2 == 1 && *beta == 2) {

/*
          Either there are an odd number of bits used to store a
          floating-point number, which is unlikely, or some bits are
          not used in the representation of numbers, which is possible,
          (e.g. Cray machines) or the mantissa has an implicit bit,
          (e.g. IEEE machines, Dec Vax machines), which is perhaps the
          most likely. We have to assume the last alternative.
          If this is true, then we need to reduce EMAX by one because
          there must be some way of representing zero in an implicit-bit
          system. On machines like Cray, we are reducing EMAX by one
          unnecessarily.
*/

	--(*emax);
    }

    if (*ieee) {

/*
          Assume we are on an IEEE machine which reserves one exponent
          for infinity and NaN.
*/

	--(*emax);
    }

/*
       Now create RMAX, the largest machine number, which should
       be equal to (1.0 - BETA**(-P)) * BETA**EMAX .

       First compute 1.0 - BETA**(-P), being careful that the
       result is less than 1.0 .
*/

    recbas = 1. / *beta;
    z__ = *beta - 1.;
    y = 0.;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.) {
	    oldy = y;
	}
	y = dlamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = y * *beta;
	y = dlamc3_(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* dlamc5_ */

logical lsame_(char *ca, char *cb)
{
    /* System generated locals */
    logical ret_val;

    /* Local variables */
    static integer inta, intb, zcode;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    LSAME returns .TRUE. if CA is the same letter as CB regardless of
    case.

    Arguments
    =========

    CA      (input) CHARACTER*1
    CB      (input) CHARACTER*1
            CA and CB specify the single characters to be compared.

   =====================================================================


       Test if the characters are equal
*/

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
	return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*
       Use 'Z' rather than 'A' so that ASCII can be detected on Prime
       machines, on which ICHAR returns a value with bit 8 set.
       ICHAR('A') on Prime machines returns 193 which is the same as
       ICHAR('A') on an EBCDIC machine.
*/

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*
          ASCII is assumed - ZCODE is the ASCII code of either lower or
          upper case 'Z'.
*/

	if (inta >= 97 && inta <= 122) {
	    inta += -32;
	}
	if (intb >= 97 && intb <= 122) {
	    intb += -32;
	}

    } else if (zcode == 233 || zcode == 169) {

/*
          EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
          upper case 'Z'.
*/

	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta
		>= 162 && inta <= 169) {
	    inta += 64;
	}
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb
		>= 162 && intb <= 169) {
	    intb += 64;
	}

    } else if (zcode == 218 || zcode == 250) {

/*
          ASCII is assumed, on Prime machines - ZCODE is the ASCII code
          plus 128 of either lower or upper case 'Z'.
*/

	if (inta >= 225 && inta <= 250) {
	    inta += -32;
	}
	if (intb >= 225 && intb <= 250) {
	    intb += -32;
	}
    }
    ret_val = inta == intb;

/*
       RETURN

       End of LSAME
*/

    return ret_val;
} /* lsame_ */

doublereal slamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    real ret_val;

    /* Local variables */
    static real t;
    static integer it;
    static real rnd, eps, base;
    static integer beta;
    static real emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static real rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static real small, sfmin;
    extern /* Subroutine */ int slamc2_(integer *, integer *, logical *, real
	    *, integer *, real *, integer *, real *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMCH determines single precision machine parameters.

    Arguments
    =========

    CMACH   (input) CHARACTER*1
            Specifies the value to be returned by SLAMCH:
            = 'E' or 'e',   SLAMCH := eps
            = 'S' or 's ,   SLAMCH := sfmin
            = 'B' or 'b',   SLAMCH := base
            = 'P' or 'p',   SLAMCH := eps*base
            = 'N' or 'n',   SLAMCH := t
            = 'R' or 'r',   SLAMCH := rnd
            = 'M' or 'm',   SLAMCH := emin
            = 'U' or 'u',   SLAMCH := rmin
            = 'L' or 'l',   SLAMCH := emax
            = 'O' or 'o',   SLAMCH := rmax

            where

            eps   = relative machine precision
            sfmin = safe minimum, such that 1/sfmin does not overflow
            base  = base of the machine
            prec  = eps*base
            t     = number of (base) digits in the mantissa
            rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
            emin  = minimum exponent before (gradual) underflow
            rmin  = underflow threshold - base**(emin-1)
            emax  = largest exponent before overflow
            rmax  = overflow threshold  - (base**emax)*(1-eps)

   =====================================================================
*/


    if (first) {
	slamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (real) beta;
	t = (real) it;
	if (lrnd) {
	    rnd = 1.f;
	    i__1 = 1 - it;
	    eps = pow_ri(&base, &i__1) / 2;
	} else {
	    rnd = 0.f;
	    i__1 = 1 - it;
	    eps = pow_ri(&base, &i__1);
	}
	prec = eps * base;
	emin = (real) imin;
	emax = (real) imax;
	sfmin = rmin;
	small = 1.f / rmax;
	if (small >= sfmin) {

/*
             Use SMALL plus a bit, to avoid the possibility of rounding
             causing overflow when computing  1/sfmin.
*/

	    sfmin = small * (eps + 1.f);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of SLAMCH */

} /* slamch_ */


/* *********************************************************************** */

/* Subroutine */ int slamc1_(integer *beta, integer *t, logical *rnd, logical
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    real r__1, r__2;

    /* Local variables */
    static real a, b, c__, f, t1, t2;
    static integer lt;
    static real one, qtr;
    static logical lrnd;
    static integer lbeta;
    static real savec;
    static logical lieee1;
    extern doublereal slamc3_(real *, real *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC1 determines the machine parameters given by BETA, T, RND, and
    IEEE1.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    IEEE1   (output) LOGICAL
            Specifies whether rounding appears to be done in the IEEE
            'round to nearest' style.

    Further Details
    ===============

    The routine is based on the routine  ENVRON  by Malcolm and
    incorporates suggestions by Gentleman and Marovich. See

       Malcolm M. A. (1972) Algorithms to reveal properties of
          floating-point arithmetic. Comms. of the ACM, 15, 949-951.

       Gentleman W. M. and Marovich S. B. (1974) More on algorithms
          that reveal properties of floating point arithmetic units.
          Comms. of the ACM, 17, 276-277.

   =====================================================================
*/


    if (first) {
	one = 1.f;

/*
          LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA,
          IEEE1, T and RND.

          Throughout this routine  we use the function  SLAMC3  to ensure
          that relevant values are  stored and not held in registers,  or
          are not affected by optimizers.

          Compute  a = 2.0**m  with the  smallest positive integer m such
          that

             fl( a + 1.0 ) = a.
*/

	a = 1.f;
	c__ = 1.f;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = slamc3_(&a, &one);
	    r__1 = -a;
	    c__ = slamc3_(&c__, &r__1);
	    goto L10;
	}
/*
   +       END WHILE

          Now compute  b = 2.0**m  with the smallest positive integer m
          such that

             fl( a + b ) .gt. a.
*/

	b = 1.f;
	c__ = slamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = slamc3_(&a, &b);
	    goto L20;
	}
/*
   +       END WHILE

          Now compute the base.  a and c  are neighbouring floating point
          numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so
          their difference is beta. Adding 0.25 to c is to ensure that it
          is truncated to beta and not ( beta - 1 ).
*/

	qtr = one / 4;
	savec = c__;
	r__1 = -a;
	c__ = slamc3_(&c__, &r__1);
	lbeta = c__ + qtr;

/*
          Now determine whether rounding or chopping occurs,  by adding a
          bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a.
*/

	b = (real) lbeta;
	r__1 = b / 2;
	r__2 = -b / 100;
	f = slamc3_(&r__1, &r__2);
	c__ = slamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	r__1 = b / 2;
	r__2 = b / 100;
	f = slamc3_(&r__1, &r__2);
	c__ = slamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*
          Try and decide whether rounding is done in the  IEEE  'round to
          nearest' style. B/2 is half a unit in the last place of the two
          numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit
          zero, and SAVEC is odd. Thus adding B/2 to A should not  change
          A, but adding B/2 to SAVEC should change SAVEC.
*/

	r__1 = b / 2;
	t1 = slamc3_(&r__1, &a);
	r__1 = b / 2;
	t2 = slamc3_(&r__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*
          Now find  the  mantissa, t.  It should  be the  integer part of
          log to the base beta of a,  however it is safer to determine  t
          by powering.  So we find t as the smallest positive integer for
          which

             fl( beta**t + 1.0 ) = 1.0.
*/

	lt = 0;
	a = 1.f;
	c__ = 1.f;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = slamc3_(&a, &one);
	    r__1 = -a;
	    c__ = slamc3_(&c__, &r__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of SLAMC1 */

} /* slamc1_ */


/* *********************************************************************** */

/* Subroutine */ int slamc2_(integer *beta, integer *t, logical *rnd, real *
	eps, integer *emin, real *rmin, integer *emax, real *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 SLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4, r__5;

    /* Local variables */
    static real a, b, c__;
    static integer i__, lt;
    static real one, two;
    static logical ieee;
    static real half;
    static logical lrnd;
    static real leps, zero;
    static integer lbeta;
    static real rbase;
    static integer lemin, lemax, gnmin;
    static real small;
    static integer gpmin;
    static real third, lrmin, lrmax, sixth;
    static logical lieee1;
    extern /* Subroutine */ int slamc1_(integer *, integer *, logical *,
	    logical *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int slamc4_(integer *, real *, integer *),
	    slamc5_(integer *, integer *, integer *, logical *, integer *,
	    real *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___144 = { 0, 6, 0, fmt_9999, 0 };


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC2 determines the machine parameters specified in its argument
    list.

    Arguments
    =========

    BETA    (output) INTEGER
            The base of the machine.

    T       (output) INTEGER
            The number of ( BETA ) digits in the mantissa.

    RND     (output) LOGICAL
            Specifies whether proper rounding  ( RND = .TRUE. )  or
            chopping  ( RND = .FALSE. )  occurs in addition. This may not
            be a reliable guide to the way in which the machine performs
            its arithmetic.

    EPS     (output) REAL
            The smallest positive number such that

               fl( 1.0 - EPS ) .LT. 1.0,

            where fl denotes the computed value.

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow occurs.

    RMIN    (output) REAL
            The smallest normalized number for the machine, given by
            BASE**( EMIN - 1 ), where  BASE  is the floating point value
            of BETA.

    EMAX    (output) INTEGER
            The maximum exponent before overflow occurs.

    RMAX    (output) REAL
            The largest positive number for the machine, given by
            BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point
            value of BETA.

    Further Details
    ===============

    The computation of  EPS  is based on a routine PARANOIA by
    W. Kahan of the University of California at Berkeley.

   =====================================================================
*/


    if (first) {
	zero = 0.f;
	one = 1.f;
	two = 2.f;

/*
          LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of
          BETA, T, RND, EPS, EMIN and RMIN.

          Throughout this routine  we use the function  SLAMC3  to ensure
          that relevant values are stored  and not held in registers,  or
          are not affected by optimizers.

          SLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1.
*/

	slamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (real) lbeta;
	i__1 = -lt;
	a = pow_ri(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	r__1 = -half;
	sixth = slamc3_(&b, &r__1);
	third = slamc3_(&sixth, &sixth);
	r__1 = -half;
	b = slamc3_(&third, &r__1);
	b = slamc3_(&b, &sixth);
	b = dabs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.f;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    r__1 = half * leps;
/* Computing 5th power */
	    r__3 = two, r__4 = r__3, r__3 *= r__3;
/* Computing 2nd power */
	    r__5 = leps;
	    r__2 = r__4 * (r__3 * r__3) * (r__5 * r__5);
	    c__ = slamc3_(&r__1, &r__2);
	    r__1 = -c__;
	    c__ = slamc3_(&half, &r__1);
	    b = slamc3_(&half, &c__);
	    r__1 = -b;
	    c__ = slamc3_(&half, &r__1);
	    b = slamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*
          Computation of EPS complete.

          Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)).
          Keep dividing  A by BETA until (gradual) underflow occurs. This
          is detected when we cannot recover the previous A.
*/

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    r__1 = small * rbase;
	    small = slamc3_(&r__1, &zero);
/* L20: */
	}
	a = slamc3_(&one, &small);
	slamc4_(&ngpmin, &one, &lbeta);
	r__1 = -one;
	slamc4_(&ngnmin, &r__1, &lbeta);
	slamc4_(&gpmin, &a, &lbeta);
	r__1 = -a;
	slamc4_(&gnmin, &r__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*
              ( Non twos-complement machines, no gradual underflow;
                e.g.,  VAX )
*/
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*
              ( Non twos-complement machines, with gradual underflow;
                e.g., IEEE standard followers )
*/
	    } else {
		lemin = min(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = max(ngpmin,ngnmin);
/*
              ( Twos-complement machines, no gradual underflow;
                e.g., CYBER 205 )
*/
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - min(ngpmin,ngnmin) == 3) {
		lemin = max(ngpmin,ngnmin) - 1 + lt;
/*
              ( Twos-complement machines with gradual underflow;
                no known machine )
*/
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = min(ngpmin,ngnmin), i__1 = min(i__1,gpmin);
	    lemin = min(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/*
   **
   Comment out this if block if EMIN is ok
*/
	if (iwarn) {
	    first = TRUE_;
	    s_wsfe(&io___144);
	    do_fio(&c__1, (char *)&lemin, (ftnlen)sizeof(integer));
	    e_wsfe();
	}
/*
   **

          Assume IEEE arithmetic if we found denormalised  numbers above,
          or if arithmetic seems to round in the  IEEE style,  determined
          in routine SLAMC1. A true IEEE machine should have both  things
          true; however, faulty machines may have one or the other.
*/

	ieee = ieee || lieee1;

/*
          Compute  RMIN by successive division by  BETA. We could compute
          RMIN as BASE**( EMIN - 1 ),  but some machines underflow during
          this computation.
*/

	lrmin = 1.f;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    r__1 = lrmin * rbase;
	    lrmin = slamc3_(&r__1, &zero);
/* L30: */
	}

/*        Finally, call SLAMC5 to compute EMAX and RMAX. */

	slamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of SLAMC2 */

} /* slamc2_ */


/* *********************************************************************** */

doublereal slamc3_(real *a, real *b)
{
    /* System generated locals */
    real ret_val;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC3  is intended to force  A  and  B  to be stored prior to doing
    the addition of  A  and  B ,  for use in situations where optimizers
    might hold one of these in a register.

    Arguments
    =========

    A       (input) REAL
    B       (input) REAL
            The values A and B.

   =====================================================================
*/


    ret_val = *a + *b;

    return ret_val;

/*     End of SLAMC3 */

} /* slamc3_ */


/* *********************************************************************** */

/* Subroutine */ int slamc4_(integer *emin, real *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static real a;
    static integer i__;
    static real b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern doublereal slamc3_(real *, real *);


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC4 is a service routine for SLAMC2.

    Arguments
    =========

    EMIN    (output) INTEGER
            The minimum exponent before (gradual) underflow, computed by
            setting A = START and dividing by BASE until the previous A
            can not be recovered.

    START   (input) REAL
            The starting point for determining EMIN.

    BASE    (input) INTEGER
            The base of the machine.

   =====================================================================
*/


    a = *start;
    one = 1.f;
    rbase = one / *base;
    zero = 0.f;
    *emin = 1;
    r__1 = a * rbase;
    b1 = slamc3_(&r__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/*
   +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND.
      $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP
*/
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	r__1 = a / *base;
	b1 = slamc3_(&r__1, &zero);
	r__1 = b1 * *base;
	c1 = slamc3_(&r__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	r__1 = a * rbase;
	b2 = slamc3_(&r__1, &zero);
	r__1 = b2 / rbase;
	c2 = slamc3_(&r__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of SLAMC4 */

} /* slamc4_ */


/* *********************************************************************** */

/* Subroutine */ int slamc5_(integer *beta, integer *p, integer *emin,
	logical *ieee, integer *emax, real *rmax)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static integer i__;
    static real y, z__;
    static integer try__, lexp;
    static real oldy;
    static integer uexp, nbits;
    extern doublereal slamc3_(real *, real *);
    static real recbas;
    static integer exbits, expsum;


/*
    -- LAPACK auxiliary routine (version 3.2) --
       Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
       November 2006


    Purpose
    =======

    SLAMC5 attempts to compute RMAX, the largest machine floating-point
    number, without overflow.  It assumes that EMAX + abs(EMIN) sum
    approximately to a power of 2.  It will fail on machines where this
    assumption does not hold, for example, the Cyber 205 (EMIN = -28625,
    EMAX = 28718).  It will also fail if the value supplied for EMIN is
    too large (i.e. too close to zero), probably with overflow.

    Arguments
    =========

    BETA    (input) INTEGER
            The base of floating-point arithmetic.

    P       (input) INTEGER
            The number of base BETA digits in the mantissa of a
            floating-point value.

    EMIN    (input) INTEGER
            The minimum exponent before (gradual) underflow.

    IEEE    (input) LOGICAL
            A logical flag specifying whether or not the arithmetic
            system is thought to comply with the IEEE standard.

    EMAX    (output) INTEGER
            The largest exponent before overflow

    RMAX    (output) REAL
            The largest machine floating-point number.

   =====================================================================


       First compute LEXP and UEXP, two powers of 2 that bound
       abs(EMIN). We then assume that EMAX + abs(EMIN) will sum
       approximately to the bound that is closest to abs(EMIN).
       (EMAX is the exponent of the required number RMAX).
*/

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*
       Now -LEXP is less than or equal to EMIN, and -UEXP is greater
       than or equal to EMIN. EXBITS is the number of bits needed to
       store the exponent.
*/

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*
       EXPSUM is the exponent range, approximately equal to
       EMAX - EMIN + 1 .
*/

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*
       NBITS is the total number of bits needed to store a
       floating-point number.
*/

    if (nbits % 2 == 1 && *beta == 2) {

/*
          Either there are an odd number of bits used to store a
          floating-point number, which is unlikely, or some bits are
          not used in the representation of numbers, which is possible,
          (e.g. Cray machines) or the mantissa has an implicit bit,
          (e.g. IEEE machines, Dec Vax machines), which is perhaps the
          most likely. We have to assume the last alternative.
          If this is true, then we need to reduce EMAX by one because
          there must be some way of representing zero in an implicit-bit
          system. On machines like Cray, we are reducing EMAX by one
          unnecessarily.
*/

	--(*emax);
    }

    if (*ieee) {

/*
          Assume we are on an IEEE machine which reserves one exponent
          for infinity and NaN.
*/

	--(*emax);
    }

/*
       Now create RMAX, the largest machine number, which should
       be equal to (1.0 - BETA**(-P)) * BETA**EMAX .

       First compute 1.0 - BETA**(-P), being careful that the
       result is less than 1.0 .
*/

    recbas = 1.f / *beta;
    z__ = *beta - 1.f;
    y = 0.f;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.f) {
	    oldy = y;
	}
	y = slamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.f) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__1 = y * *beta;
	y = slamc3_(&r__1, &c_b66);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of SLAMC5 */

} /* slamc5_ */

