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

   All logical operations return UBYTE arrays.
*/

#if defined _ISOC99_SOURCE || defined _XOPEN_SOURCE_EXTENDED \
    || defined _BSD_SOURCE || defined _SVID_SOURCE
#define HAVE_INVERSE_HYPERBOLIC 1
#endif

static PyObject *Array0d_FromDouble(double); 
/* Wrapper to include the correct version */

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

