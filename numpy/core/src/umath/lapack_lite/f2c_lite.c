#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "f2c.h"


extern void s_wsfe(cilist *f) {;}
extern void e_wsfe(void) {;}
extern void do_fio(integer *c, char *s, ftnlen l) {;}

/* You'll want this if you redo the *_lite.c files with the -C option
 * to f2c for checking array subscripts. (It's not suggested you do that
 * for production use, of course.) */
extern int
s_rnge(char *var, int index, char *routine, int lineno)
{
    fprintf(stderr, "array index out-of-bounds for %s[%d] in routine %s:%d\n",
            var, index, routine, lineno);
    fflush(stderr);
    abort();
}

#ifdef KR_headers
extern float sqrtf();
double f__cabsf(real, imag) float real, imag;
#else
#undef abs

double f__cabsf(float real, float imag)
#endif
{
float temp;

if(real < 0.0f)
	real = -real;
if(imag < 0.0f)
	imag = -imag;
if(imag > real){
	temp = real;
	real = imag;
	imag = temp;
}
if((imag+real) == real)
	return((float)real);

temp = imag/real;
temp = real*sqrtf(1.0 + temp*temp);  /*overflow!!*/
return(temp);
}


#ifdef KR_headers
extern double sqrt();
double f__cabs(real, imag) double real, imag;
#else
#undef abs

double f__cabs(double real, double imag)
#endif
{
double temp;

if(real < 0)
	real = -real;
if(imag < 0)
	imag = -imag;
if(imag > real){
	temp = real;
	real = imag;
	imag = temp;
}
if((imag+real) == real)
	return((double)real);

temp = imag/real;
temp = real*sqrt(1.0 + temp*temp);  /*overflow!!*/
return(temp);
}

 VOID
#ifdef KR_headers
r_cnjg(r, z) complex *r, *z;
#else
r_cnjg(complex *r, complex *z)
#endif
{
r->r = z->r;
r->i = - z->i;
}

 VOID
#ifdef KR_headers
d_cnjg(r, z) doublecomplex *r, *z;
#else
d_cnjg(doublecomplex *r, doublecomplex *z)
#endif
{
r->r = z->r;
r->i = - z->i;
}


#ifdef KR_headers
float r_imag(z) complex *z;
#else
float r_imag(complex *z)
#endif
{
return(z->i);
}

#ifdef KR_headers
double d_imag(z) doublecomplex *z;
#else
double d_imag(doublecomplex *z)
#endif
{
return(z->i);
}


#define log10e 0.43429448190325182765

#ifdef KR_headers
float logf();
float r_lg10(x) real *x;
#else
#undef abs

float r_lg10(real *x)
#endif
{
return( log10e * logf(*x) );
}

#ifdef KR_headers
double log();
double d_lg10(x) doublereal *x;
#else
#undef abs

double d_lg10(doublereal *x)
#endif
{
return( log10e * log(*x) );
}

#ifdef KR_headers
double r_sign(a,b) real *a, *b;
#else
double r_sign(real *a, real *b)
#endif
{
float x;
x = (*a >= 0.0f ? *a : - *a);
return( *b >= 0.0f ? x : -x);
}

#ifdef KR_headers
double d_sign(a,b) doublereal *a, *b;
#else
double d_sign(doublereal *a, doublereal *b)
#endif
{
double x;
x = (*a >= 0 ? *a : - *a);
return( *b >= 0 ? x : -x);
}


#ifdef KR_headers
double floor();
integer i_dnnt(x) doublereal *x;
#else
#undef abs

integer i_dnnt(doublereal *x)
#endif
{
return( (*x)>=0 ?
	floor(*x + .5) : -floor(.5 - *x) );
}


#ifdef KR_headers
double pow();
double pow_dd(ap, bp) doublereal *ap, *bp;
#else
#undef abs

double pow_dd(doublereal *ap, doublereal *bp)
#endif
{
return(pow(*ap, *bp) );
}


#ifdef KR_headers
double pow_ri(ap, bp) real *ap; integer *bp;
#else
double pow_ri(real *ap, integer *bp)
#endif
{
float pow, x;
integer n;
unsigned long u;

pow = 1;
x = *ap;
n = *bp;

if(n != 0)
	{
	if(n < 0)
		{
		n = -n;
		x = 1.0f/x;
		}
	for(u = n; ; )
		{
		if(u & 01)
			pow *= x;
		if(u >>= 1)
			x *= x;
		else
			break;
		}
	}
return(pow);
}

#ifdef KR_headers
double pow_di(ap, bp) doublereal *ap; integer *bp;
#else
double pow_di(doublereal *ap, integer *bp)
#endif
{
double pow, x;
integer n;
unsigned long u;

pow = 1;
x = *ap;
n = *bp;

if(n != 0)
	{
	if(n < 0)
		{
		n = -n;
		x = 1/x;
		}
	for(u = n; ; )
		{
		if(u & 01)
			pow *= x;
		if(u >>= 1)
			x *= x;
		else
			break;
		}
	}
return(pow);
}
/* Unless compiled with -DNO_OVERWRITE, this variant of s_cat allows the
 * target of a concatenation to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90).
 */
#define NO_OVERWRITE


#ifndef NO_OVERWRITE

#undef abs
#ifdef KR_headers
 extern char *F77_aloc();
 extern void free();
 extern void exit_();
#else

 extern char *F77_aloc(ftnlen, char*);
#endif

#endif /* NO_OVERWRITE */

 VOID
#ifdef KR_headers
s_cat(lp, rpp, rnp, np, ll) char *lp, *rpp[]; ftnlen rnp[], *np, ll;
#else
s_cat(char *lp, char *rpp[], ftnlen rnp[], ftnlen *np, ftnlen ll)
#endif
{
	ftnlen i, nc;
	char *rp;
	ftnlen n = *np;
#ifndef NO_OVERWRITE
	ftnlen L, m;
	char *lp0, *lp1;

	lp0 = 0;
	lp1 = lp;
	L = ll;
	i = 0;
	while(i < n) {
		rp = rpp[i];
		m = rnp[i++];
		if (rp >= lp1 || rp + m <= lp) {
			if ((L -= m) <= 0) {
				n = i;
				break;
				}
			lp1 += m;
			continue;
			}
		lp0 = lp;
		lp = lp1 = F77_aloc(L = ll, "s_cat");
		break;
		}
	lp1 = lp;
#endif /* NO_OVERWRITE */
	for(i = 0 ; i < n ; ++i) {
		nc = ll;
		if(rnp[i] < nc)
			nc = rnp[i];
		ll -= nc;
		rp = rpp[i];
		while(--nc >= 0)
			*lp++ = *rp++;
		}
	while(--ll >= 0)
		*lp++ = ' ';
#ifndef NO_OVERWRITE
	if (lp0) {
		memmove(lp0, lp1, L);
		free(lp1);
		}
#endif
	}


/* compare two strings */

#ifdef KR_headers
integer s_cmp(a0, b0, la, lb) char *a0, *b0; ftnlen la, lb;
#else
integer s_cmp(char *a0, char *b0, ftnlen la, ftnlen lb)
#endif
{
register unsigned char *a, *aend, *b, *bend;
a = (unsigned char *)a0;
b = (unsigned char *)b0;
aend = a + la;
bend = b + lb;

if(la <= lb)
	{
	while(a < aend)
		if(*a != *b)
			return( *a - *b );
		else
			{ ++a; ++b; }

	while(b < bend)
		if(*b != ' ')
			return( ' ' - *b );
		else	++b;
	}

else
	{
	while(b < bend)
		if(*a == *b)
			{ ++a; ++b; }
		else
			return( *a - *b );
	while(a < aend)
		if(*a != ' ')
			return(*a - ' ');
		else	++a;
	}
return(0);
}
/* Unless compiled with -DNO_OVERWRITE, this variant of s_copy allows the
 * target of an assignment to appear on its right-hand side (contrary
 * to the Fortran 77 Standard, but in accordance with Fortran 90),
 * as in  a(2:5) = a(4:7) .
 */



/* assign strings:  a = b */

#ifdef KR_headers
VOID s_copy(a, b, la, lb) register char *a, *b; ftnlen la, lb;
#else
void s_copy(register char *a, register char *b, ftnlen la, ftnlen lb)
#endif
{
	register char *aend, *bend;

	aend = a + la;

	if(la <= lb)
#ifndef NO_OVERWRITE
		if (a <= b || a >= b + la)
#endif
			while(a < aend)
				*a++ = *b++;
#ifndef NO_OVERWRITE
		else
			for(b += la; a < aend; )
				*--aend = *--b;
#endif

	else {
		bend = b + lb;
#ifndef NO_OVERWRITE
		if (a <= b || a >= bend)
#endif
			while(b < bend)
				*a++ = *b++;
#ifndef NO_OVERWRITE
		else {
			a += lb;
			while(b < bend)
				*--a = *--bend;
			a += lb;
			}
#endif
		while(a < aend)
			*a++ = ' ';
		}
	}


#ifdef KR_headers
double f__cabsf();
double c_abs(z) complex *z;
#else
double f__cabsf(float, float);
double c_abs(complex *z)
#endif
{
return( f__cabsf( z->r, z->i ) );
}

#ifdef KR_headers
double f__cabs();
double z_abs(z) doublecomplex *z;
#else
double f__cabs(double, double);
double z_abs(doublecomplex *z)
#endif
{
return( f__cabs( z->r, z->i ) );
}


#ifdef KR_headers
extern void sig_die();
VOID c_div(c, a, b) complex *a, *b, *c;
#else
extern void sig_die(char*, int);
void c_div(complex *c, complex *a, complex *b)
#endif
{
float ratio, den;
float abr, abi;

if( (abr = b->r) < 0.f)
	abr = - abr;
if( (abi = b->i) < 0.f)
	abi = - abi;
if( abr <= abi )
	{
	  /*Let IEEE Infinties handle this ;( */
	  /*if(abi == 0)
		sig_die("complex division by zero", 1);*/
	ratio = b->r / b->i ;
	den = b->i * (1 + ratio*ratio);
	c->r = (a->r*ratio + a->i) / den;
	c->i = (a->i*ratio - a->r) / den;
	}

else
	{
	ratio = b->i / b->r ;
	den = b->r * (1.f + ratio*ratio);
	c->r = (a->r + a->i*ratio) / den;
	c->i = (a->i - a->r*ratio) / den;
	}

}

#ifdef KR_headers
extern void sig_die();
VOID z_div(c, a, b) doublecomplex *a, *b, *c;
#else
extern void sig_die(char*, int);
void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
#endif
{
double ratio, den;
double abr, abi;

if( (abr = b->r) < 0.)
	abr = - abr;
if( (abi = b->i) < 0.)
	abi = - abi;
if( abr <= abi )
	{
	  /*Let IEEE Infinties handle this ;( */
	  /*if(abi == 0)
		sig_die("complex division by zero", 1);*/
	ratio = b->r / b->i ;
	den = b->i * (1 + ratio*ratio);
	c->r = (a->r*ratio + a->i) / den;
	c->i = (a->i*ratio - a->r) / den;
	}

else
	{
	ratio = b->i / b->r ;
	den = b->r * (1 + ratio*ratio);
	c->r = (a->r + a->i*ratio) / den;
	c->i = (a->i - a->r*ratio) / den;
	}

}


#ifdef KR_headers
float sqrtf(), f__cabsf();
VOID c_sqrt(r, z) complex *r, *z;
#else
#undef abs

extern double f__cabsf(float, float);
void c_sqrt(complex *r, complex *z)
#endif
{
float mag;

if( (mag = f__cabsf(z->r, z->i)) == 0.f)
	r->r = r->i = 0.f;
else if(z->r > 0.0f)
	{
	r->r = sqrtf(0.5f * (mag + z->r) );
	r->i = z->i / r->r / 2.0f;
	}
else
	{
	r->i = sqrtf(0.5f * (mag - z->r) );
	if(z->i < 0.0f)
		r->i = - r->i;
	r->r = z->i / r->i / 2.0f;
	}
}


#ifdef KR_headers
double sqrt(), f__cabs();
VOID z_sqrt(r, z) doublecomplex *r, *z;
#else
#undef abs

extern double f__cabs(double, double);
void z_sqrt(doublecomplex *r, doublecomplex *z)
#endif
{
double mag;

if( (mag = f__cabs(z->r, z->i)) == 0.)
	r->r = r->i = 0.;
else if(z->r > 0)
	{
	r->r = sqrt(0.5 * (mag + z->r) );
	r->i = z->i / r->r / 2;
	}
else
	{
	r->i = sqrt(0.5 * (mag - z->r) );
	if(z->i < 0)
		r->i = - r->i;
	r->r = z->i / r->i / 2;
	}
}
#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
integer pow_ii(ap, bp) integer *ap, *bp;
#else
integer pow_ii(integer *ap, integer *bp)
#endif
{
	integer pow, x, n;
	unsigned long u;

	x = *ap;
	n = *bp;

	if (n <= 0) {
		if (n == 0 || x == 1)
			return 1;
		if (x != -1)
			return x == 0 ? 1/x : 0;
		n = -n;
		}
	u = n;
	for(pow = 1; ; )
		{
		if(u & 01)
			pow *= x;
		if(u >>= 1)
			x *= x;
		else
			break;
		}
	return(pow);
	}
#ifdef __cplusplus
}
#endif

#ifdef KR_headers
extern void f_exit();
VOID s_stop(s, n) char *s; ftnlen n;
#else
#undef abs
#undef min
#undef max
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
extern "C" {
#endif
void f_exit(void);

int s_stop(char *s, ftnlen n)
#endif
{
int i;

if(n > 0)
	{
	fprintf(stderr, "STOP ");
	for(i = 0; i<n ; ++i)
		putc(*s++, stderr);
	fprintf(stderr, " statement executed\n");
	}
#ifdef NO_ONEXIT
f_exit();
#endif
exit(0);

/* We cannot avoid (useless) compiler diagnostics here:		*/
/* some compilers complain if there is no return statement,	*/
/* and others complain that this one cannot be reached.		*/

return 0; /* NOT REACHED */
}
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
}
#endif
