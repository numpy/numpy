#include "Python.h"
#include "Numeric/arrayobject.h"
#include "Numeric/ufuncobject.h"
#include "abstract.h"
#include <math.h>
#include "mconf_lite.h"

/* Fast umath module whose functions do not check for range and domain
   errors.

   Replacement for umath + additions for isnan, isfinite, and isinf
   Also allows comparison operations on complex numbers (just compares
   the real part) and logical operations.

   All logical operations return UBYTE arrays except for 
     logical_and, logical_or, and logical_xor
     which return their type so that reduce works correctly on them....
*/

#if defined _ISOC99_SOURCE || defined _XOPEN_SOURCE_EXTENDED \
    || defined _BSD_SOURCE || defined _SVID_SOURCE
#define HAVE_INVERSE_HYPERBOLIC 1
#endif

static PyObject *Array0d_FromDouble(double); 
/* Wrapper to include the correct version */

/* Complex functions */


#if !defined(HAVE_INVERSE_HYPERBOLIC)
static double acosh(double x)
{
    return log(x + sqrt((x-1.0)*(x+1.0)));
}

static double asinh(double xx)
{
    double x;
    int sign;
    if (xx < 0.0) {
	sign = -1;
	x = -xx;
    }
    else {
	sign = 1;
	x = xx;
    }
    return sign*log(x + sqrt(x*x+1.0));
}

static double atanh(double x)
{
    return 0.5*log((1.0+x)/(1.0-x));
}
#endif

#if defined(HAVE_HYPOT) 
#if !defined(NeXT) && !defined(_MSC_VER)
extern double hypot(double, double);
#endif
#else
double hypot(double x, double y)
{
    double yx;

    x = fabs(x);
    y = fabs(y);
    if (x < y) {
	double temp = x;
	x = y;
	y = temp;
    }
    if (x == 0.)
	return 0.;
    else {
	yx = y/x;
	return x*sqrt(1.+yx*yx);
    }
}
#endif

#ifdef i860
/* Cray APP has bogus definition of HUGE_VAL in <math.h> */
#undef HUGE_VAL
#endif

#ifdef HUGE_VAL
#define CHECK(x) if (errno != 0) ; 	else if (-HUGE_VAL <= (x) && (x) <= HUGE_VAL) ; 	else errno = ERANGE
#else
#define CHECK(x) /* Don't know how to check */
#endif

/* constants */
static Py_complex c_1 = {1., 0.};
static Py_complex c_half = {0.5, 0.};
static Py_complex c_i = {0., 1.};
static Py_complex c_i2 = {0., 0.5};
/*
static Py_complex c_mi = {0., -1.};
static Py_complex c_pi2 = {M_PI/2., 0.};
*/

static Py_complex
c_sum_(Py_complex a, Py_complex b)
{
    Py_complex r;
    r.real = a.real + b.real;
    r.imag = a.imag + b.imag;
    return r;
}

static Py_complex
c_diff_(Py_complex a, Py_complex b)
{
    Py_complex r;
    r.real = a.real - b.real;
    r.imag = a.imag - b.imag;
    return r;
}

static Py_complex
c_neg_(Py_complex a)
{
    Py_complex r;
    r.real = -a.real;
    r.imag = -a.imag;
    return r;
}

static Py_complex
c_prod_(Py_complex a, Py_complex b)
{
    Py_complex r;
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
    return r;
}

static Py_complex
c_pow_(Py_complex a, Py_complex b)
{
    Py_complex r;
    double vabs,len,at,phase;
    if (b.real == 0. && b.imag == 0.) {
	r.real = 1.;
	r.imag = 0.;
    }
    else if (a.real == 0. && a.imag == 0.) {
	if (b.imag != 0. || b.real < 0.)
	    errno = EDOM;
	r.real = 0.;
	r.imag = 0.;
    }
    else {
	vabs = hypot(a.real,a.imag);
	len = pow(vabs,b.real);
	at = atan2(a.imag, a.real);
	phase = at*b.real;
	if (b.imag != 0.0) {
	    len /= exp(at*b.imag);
	    phase += b.imag*log(vabs);
	}
	r.real = len*cos(phase);
	r.imag = len*sin(phase);
    }
    return r;
}

/* First, the C functions that do the real work */

static Py_complex 
c_quot_fast(Py_complex a, Py_complex b)
{
    /******************************************************************/
    
    /* This algorithm is better, and is pretty obvious:  first divide the
     * numerators and denominator by whichever of {b.real, b.imag} has
     * larger magnitude.  The earliest reference I found was to CACM
     * Algorithm 116 (Complex Division, Robert L. Smith, Stanford
     * University).  As usual, though, we're still ignoring all IEEE
     * endcases.
     */
    Py_complex r;  /* the result */

    const double abs_breal = b.real < 0 ? -b.real : b.real;
    const double abs_bimag = b.imag < 0 ? -b.imag : b.imag;

    if ((b.real == 0.0) && (b.imag == 0.0)) {
        r.real = a.real / b.real;
        r.imag = a.imag / b.imag;
	/* Using matlab's convention (x+0j is x):
	   (0+0j)/0 -> nan+0j
	   (0+xj)/0 -> nan+sign(x)*infj
	   (x+0j)/0 -> sign(x)*inf+0j
	*/
	if (a.imag == 0.0) {r.imag = 0.0;}
        return r;
    }
    if (abs_breal >= abs_bimag) {
	/* divide tops and bottom by b.real */
	const double ratio = b.imag / b.real;
	const double denom = b.real + b.imag * ratio;
	r.real = (a.real + a.imag * ratio) / denom;
	r.imag = (a.imag - a.real * ratio) / denom;
    }
    else {
	/* divide tops and bottom by b.imag */
	const double ratio = b.real / b.imag;
	const double denom = b.real * ratio + b.imag;
	r.real = (a.real * ratio + a.imag) / denom;
	r.imag = (a.imag * ratio - a.real) / denom;
    }
    return r;
}

#if PY_VERSION_HEX >= 0x02020000
static Py_complex 
c_quot_floor_fast(Py_complex a, Py_complex b)
{
  /* Not really sure what to do here, but it looks like Python takes the 
     floor of the real part and returns that as the answer.  So, we will do the same.
  */
  Py_complex r;

  r = c_quot_fast(a, b);
  r.imag = 0.0;
  r.real = floor(r.real);
  return r;
}
#endif

static Py_complex 
c_sqrt(Py_complex x)
{
    Py_complex r;
    double s,d;
    if (x.real == 0. && x.imag == 0.)
	r = x;
    else {
	s = sqrt(0.5*(fabs(x.real) + hypot(x.real,x.imag)));
	d = 0.5*x.imag/s;
	if (x.real > 0.) {
	    r.real = s;
	    r.imag = d;
	}
	else if (x.imag >= 0.) {
	    r.real = d;
	    r.imag = s;
	}
	else {
	    r.real = -d;
	    r.imag = -s;
	}
    }
    return r;
}

static Py_complex 
c_log(Py_complex x)
{
    Py_complex r;
    double l = hypot(x.real,x.imag);
    r.imag = atan2(x.imag, x.real);
    r.real = log(l);
    return r;
}

static Py_complex 
c_prodi(Py_complex x)
{
    Py_complex r;
    r.real = -x.imag;
    r.imag = x.real;
    return r;
}

static Py_complex 
c_acos(Py_complex x)
{
    return c_neg_(c_prodi(c_log(c_sum_(x,c_prod_(c_i,
					      c_sqrt(c_diff_(c_1,c_prod_(x,x))))))));
}

static Py_complex 
c_acosh(Py_complex x)
{
    return c_log(c_sum_(x,c_prod_(c_i,
				c_sqrt(c_diff_(c_1,c_prod_(x,x))))));
}

static Py_complex 
c_asin(Py_complex x)
{
    return c_neg_(c_prodi(c_log(c_sum_(c_prod_(c_i,x),
				       c_sqrt(c_diff_(c_1,c_prod_(x,x)))))));
}

static Py_complex 
c_asinh(Py_complex x)
{
    return c_neg_(c_log(c_diff_(c_sqrt(c_sum_(c_1,c_prod_(x,x))),x)));
}

static Py_complex 
c_atan(Py_complex x)
{
    return c_prod_(c_i2,c_log(c_quot_fast(c_sum_(c_i,x),c_diff_(c_i,x))));
}

static Py_complex 
c_atanh(Py_complex x)
{
    return c_prod_(c_half,c_log(c_quot_fast(c_sum_(c_1,x),c_diff_(c_1,x))));
}

static Py_complex 
c_cos(Py_complex x)
{
    Py_complex r;
    r.real = cos(x.real)*cosh(x.imag);
    r.imag = -sin(x.real)*sinh(x.imag);
    return r;
}

static Py_complex 
c_cosh(Py_complex x)
{
    Py_complex r;
    r.real = cos(x.imag)*cosh(x.real);
    r.imag = sin(x.imag)*sinh(x.real);
    return r;
}

static Py_complex 
c_exp(Py_complex x)
{
    Py_complex r;
    double l = exp(x.real);
    r.real = l*cos(x.imag);
    r.imag = l*sin(x.imag);
    return r;
}

static Py_complex 
c_log10(Py_complex x)
{
    Py_complex r;
    double l = hypot(x.real,x.imag);
    r.imag = atan2(x.imag, x.real)/log(10.);
    r.real = log10(l);
    return r;
}

static Py_complex 
c_sin(Py_complex x)
{
    Py_complex r;
    r.real = sin(x.real)*cosh(x.imag);
    r.imag = cos(x.real)*sinh(x.imag);
    return r;
}

static Py_complex 
c_sinh(Py_complex x)
{
    Py_complex r;
    r.real = cos(x.imag)*sinh(x.real);
    r.imag = sin(x.imag)*cosh(x.real);
    return r;
}

static Py_complex 
c_tan(Py_complex x)
{
    Py_complex r;
    double sr,cr,shi,chi;
    double rs,is,rc,ic;
    double d;
    sr = sin(x.real);
    cr = cos(x.real);
    shi = sinh(x.imag);
    chi = cosh(x.imag);
    rs = sr*chi;
    is = cr*shi;
    rc = cr*chi;
    ic = -sr*shi;
    d = rc*rc + ic*ic;
    r.real = (rs*rc+is*ic)/d;
    r.imag = (is*rc-rs*ic)/d;
    return r;
}

static Py_complex 
c_tanh(Py_complex x)
{
    Py_complex r;
    double si,ci,shr,chr;
    double rs,is,rc,ic;
    double d;
    si = sin(x.imag);
    ci = cos(x.imag);
    shr = sinh(x.real);
    chr = cosh(x.real);
    rs = ci*shr;
    is = si*chr;
    rc = ci*chr;
    ic = si*shr;
    d = rc*rc + ic*ic;
    r.real = (rs*rc+is*ic)/d;
    r.imag = (is*rc-rs*ic)/d;
    return r;
}


#ifdef PyArray_UNSIGNED_TYPES
#include "fastumath_unsigned.inc"
#else
#include "fastumath_nounsigned.inc"
#endif

static PyObject *Array0d_FromDouble(double val){
    PyArrayObject *a;
    a = (PyArrayObject *)PyArray_FromDims(0,NULL,PyArray_DOUBLE);
    memcpy(a->data,(char *)(&val),a->descr->elsize);
    return (PyObject *)a;
}

static double pinf_init(void) {
    double mul = 1e10;
    double tmp = 0.0;
    double pinf;

    pinf = mul;
    for (;;) {
	pinf *= mul;
	if (pinf == tmp) break;
	tmp = pinf;
    }
    return pinf;
}

static double pzero_init(void) {
    double div = 1e10;
    double tmp = 0.0;
    double pinf;

    pinf = div;
    for (;;) {
	pinf /= div;
	if (pinf == tmp) break;
	tmp = pinf;
    }
    return pinf;
}

/* Initialization function for the module (*must* be called initArray) */

static struct PyMethodDef methods[] = {
    {NULL,		NULL, 0}		/* sentinel */
};

DL_EXPORT(void) initfastumath(void) {
    PyObject *m, *d, *s, *f1;
    double pinf, pzero, nan;
    
    /* Create the module and add the functions */
    m = Py_InitModule("fastumath", methods); 

    /* Import the array and ufunc objects */
    import_array();
    import_ufunc();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    s = PyString_FromString("2.3");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Load the ufunc operators into the array module's namespace */
    InitOperators(d); 
    
    PyDict_SetItemString(d, "pi", s = PyFloat_FromDouble(atan(1.0) * 4.0));
    Py_DECREF(s);
    PyDict_SetItemString(d, "e", s = PyFloat_FromDouble(exp(1.0)));
    Py_DECREF(s);
    pinf = pinf_init();
    PyDict_SetItemString(d, "PINF", s = PyFloat_FromDouble(pinf));
    Py_DECREF(s);
    PyDict_SetItemString(d, "NINF", s = PyFloat_FromDouble(-pinf));
    Py_DECREF(s);
    pzero = pzero_init();
    PyDict_SetItemString(d, "PZERO", s = PyFloat_FromDouble(pzero));
    Py_DECREF(s);
    PyDict_SetItemString(d, "NZERO", s = PyFloat_FromDouble(-pzero));
    Py_DECREF(s);
    nan = pinf / pinf;
    PyDict_SetItemString(d, "NAN", s = PyFloat_FromDouble(nan));
    Py_DECREF(s);

    f1 = PyDict_GetItemString(d, "conjugate");  /* Borrowed reference */

    /* Setup the array object's numerical structures */
    PyArray_SetNumericOps(d);

    PyDict_SetItemString(d, "conj", f1); /* shorthand for conjugate */
  
    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("can't initialize module fastumath");
}

