#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "numpyos.h"

#ifndef LDBL_MAX_EXP
    #include <float.h>
#endif

/*
 * Heavily derived from PyLong_FromDouble
 * Notably, we can't set the digits directly, so have to shift and or instead.
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval)
{
    PyObject *v;
    PyObject *l_chunk_size;
    /*
     * number of bits to extract at a time. CPython uses 30, but that's because
     * it's tied to the internal long representation
     */
    const int chunk_size = NPY_BITSOF_LONGLONG;
    npy_longdouble frac;
    int i, ndig, expo, neg;
    neg = 0;

    if (npy_isinf(ldval)) {
        PyErr_SetString(PyExc_OverflowError,
                        "cannot convert longdouble infinity to integer");
        return NULL;
    }
    if (npy_isnan(ldval)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot convert longdouble NaN to integer");
        return NULL;
    }
    if (ldval < 0.0) {
        neg = 1;
        ldval = -ldval;
    }
    frac = npy_frexpl(ldval, &expo); /* ldval = frac*2**expo; 0.0 <= frac < 1.0 */
    v = PyLong_FromLong(0L);
    if (v == NULL)
        return NULL;
    if (expo <= 0)
        return v;

    ndig = (expo-1) / chunk_size + 1;

    l_chunk_size = PyLong_FromLong(chunk_size);
    if (l_chunk_size == NULL) {
        Py_DECREF(v);
        return NULL;
    }

    /* Get the MSBs of the integral part of the float */
    frac = npy_ldexpl(frac, (expo-1) % chunk_size + 1);
    for (i = ndig; --i >= 0; ) {
        npy_ulonglong chunk = (npy_ulonglong)frac;
        PyObject *l_chunk;
        /* v = v << chunk_size */
        Py_SETREF(v, PyNumber_Lshift(v, l_chunk_size));
        if (v == NULL) {
            goto done;
        }
        l_chunk = PyLong_FromUnsignedLongLong(chunk);
        if (l_chunk == NULL) {
            Py_DECREF(v);
            v = NULL;
            goto done;
        }
        /* v = v | chunk */
        Py_SETREF(v, PyNumber_Or(v, l_chunk));
        Py_DECREF(l_chunk);
        if (v == NULL) {
            goto done;
        }

        /* Remove the msbs, and repeat */
        frac = frac - (npy_longdouble) chunk;
        frac = npy_ldexpl(frac, chunk_size);
    }

    /* v = -v */
    if (neg) {
        Py_SETREF(v, PyNumber_Negative(v));
        if (v == NULL) {
            goto done;
        }
    }

done:
    Py_DECREF(l_chunk_size);
    return v;
}

// Overflow error function
npy_longdouble _ldbl_ovfl_err(void) {
    PyErr_SetString(PyExc_OverflowError, "Number too big to be represented as a np.longdouble --This is platform dependent--");
    return -1;
}

#define NPY_LGDB_SIGN_POS 1
#define NPY_LGDB_SIGN_NEG -1

#ifndef NPY_LDBL_MAX_EXP
    #if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
        #define NPY_LDBL_MAX_EXP DBL_MAX_EXP
    #else
        #define NPY_LDBL_MAX_EXP LDBL_MAX_EXP
    #endif
#endif

// Uses mathematical opperations to calculate the number, by using the numbers in the
// mantissa, scaling them approprietly using exp, and then adding them together
npy_longdouble _int_to_ld(int64_t *val, int exp, int sign) {
    if (PyErr_Occurred()) { return -1; }
    uint64_t mantissa[3];
    for (int i = 0; i < 3; i++) { mantissa[i] = (uint64_t)val[i]; }
    npy_longdouble ld;
    if (exp == 0) {
        ld = (npy_longdouble)sign * (npy_longdouble)mantissa[0];
        // Edge case for numbers that may or maynot be too big to be represented
    } else if (exp == NPY_LDBL_MAX_EXP && mantissa[0] == 0) {
        ld = (npy_longdouble)sign * (ldexpl((npy_longdouble)mantissa[1], exp - 64) +
            ldexpl((npy_longdouble)mantissa[2], exp - 128));
            // Sometimes it overflows in weird ways
        if (ld == (npy_longdouble)INFINITY || ld == (npy_longdouble)(-INFINITY) || ld == (npy_longdouble)NAN || ld == (npy_longdouble)(-NAN)) {
            return _ldbl_ovfl_err();
        }
    } else if (exp >= NPY_LDBL_MAX_EXP) {
        return _ldbl_ovfl_err();
    } else {
    ld = (npy_longdouble)sign * (ldexpl((npy_longdouble)mantissa[0], exp) +
        ldexpl((npy_longdouble)mantissa[1], exp - 64) +
        ldexpl((npy_longdouble)mantissa[2], exp - 128));
    }
    return ld;
}

// Helper function that get the exponent and mantissa, this works on all platforms
// This function works by getting the bits of the number in increments of 64 bits
// Since the size of the number is unknown, a while loop is needed.
// The more significant digits are stored in the smaller indexes
void _get_num(PyObject* py_int, int64_t* val, int *exp, int *ovf) {
    PyObject* shift = PyLong_FromLong(64);
    PyObject* temp = py_int;
    while (*ovf != 0) {
        *exp += 64;
        val[2] = val[1];
        val[1] = (uint64_t)PyLong_AsUnsignedLongLongMask(py_int);
        temp = PyNumber_Rshift(py_int, shift);
        if (temp != py_int) {  // Check if a new object was created
            Py_DECREF(py_int); // Decrement refcount of the old py_int
            py_int = temp;     // Update py_int to the new object
        }
        val[0] = (uint64_t)PyLong_AsLongLongAndOverflow(py_int, ovf);
    }
    Py_DECREF(shift);
    Py_DECREF(temp);
}

// helper function that gets the sign, and if the number is too big to be represented
// as a int64, calls _get_num to find the exponent and mantissa
void _fix_py_num(PyObject* py_int, int64_t* val, int *exp, int *sign) {
    int overflow;
    val[0] = PyLong_AsLongLongAndOverflow(py_int, &overflow);
    if (overflow == 1) {
        *sign = NPY_LGDB_SIGN_POS;
        PyObject* copy_int = PyNumber_Absolute(py_int);
        _get_num(copy_int, val, exp, &overflow);
    } else if (overflow == -1) {
        *sign = NPY_LGDB_SIGN_NEG;
        PyObject* copy_int = PyNumber_Negative(py_int); //create new PyLongObject
        _get_num(copy_int, val, exp, &overflow);
    } else {
        if (val[0] == 0) {*sign = 0;}
        else if (val[0] < 0) { val[0] = -val[0]; *sign = NPY_LGDB_SIGN_NEG;}
        else {*sign = NPY_LGDB_SIGN_POS;}
    } 
    
}
/*
The precision and max number is platform dependent, for 80 and 128 bit platforms
The largest number that can be converted is 2^16384 - 1.
In 64 bit platforms the largest number is 2^1024 - 1.
if the number to be converted is too big for a platform, it will give an error
In (my personal 80bit platform), I have tested that it converts up to the max
Now gives an overflow error if the number is too big, follows same rounding rules 
as python's float() function, including overflow.
*/
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj) {

    npy_longdouble value;
    if (PyFloat_Check(long_obj)) {
        value = (npy_longdouble)PyFloat_AsDouble(long_obj);
    } else if (PyLong_Check(long_obj)) {
        int sign;
        int E = 0;
        int64_t val[3]; //needs to be size 3 in 80bit prescision
        _fix_py_num(long_obj, val, &E, &sign);
        if (PyErr_Occurred()) { return -1; }
        value = _int_to_ld(val, E, sign);
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected a number (int or float)");
        return -1;
    }

    if (PyErr_Occurred()) { return -1; }
    return value;
}