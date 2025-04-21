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

npy_longdouble _ldbl_ovfl_err(void) {
    PyErr_SetString(PyExc_OverflowError, "Number too big to be represented as a np.longdouble --This is platform dependent--");
    return -1;
}

// When Double is the same as the long double, and it is little endian
// It then (thanfully) follows IEEE 754, so it is possible to do bitwise operations
#if LDBL_MANT_DIG == 53 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

#define NPY_LGDB_SIGN_POS 0
#define NPY_LGDB_SIGN_NEG 1

typedef union {
    uint64_t u64;
    npy_longdouble d;
} uint64_double_union;

npy_longdouble _int_to_ld(int64_t *val, int e, int s) {
    uint64_t mantissa[3];
    uint64_t exp = (uint64_t)e;
    uint64_t sign = (uint64_t)s;
    for (int i = 0; i < 3; i++) { mantissa[i] = (uint64_t)val[i]; }
    int foo = 0;
    uint64_double_union value;

    if (exp == 0) {
        s = (sign == 1) ? -1 : 1;
        return (npy_longdouble)s*(npy_longdouble)mantissa[0];
    } else if (mantissa[0] == 0) {
        mantissa[1] <<= 1;
        exp--;
    } else {
        for (int i = 63; i >= 0; i--) {
            if ((mantissa[0] >> i) & 1) {
                foo = i;
                break;
            }
        }
    }
    exp += foo;
    mantissa[0] = (mantissa[1] >> foo) | (mantissa[0] << (64 - foo));
    uint64_t guard = (mantissa[0] >> 11) & 1;
    mantissa[0] >>= 12;
    // Rounding (round to nearest even)
    if (guard) {
        mantissa[0]++;
        if (mantissa[0] == 0x10000000000000) {
            mantissa[0] = 0;
            exp++;
        }
    }
    if (exp >= DBL_MAX_EXP) {
        return _ldbl_ovfl_err();
    }
    sign <<= 63;
    exp += 1023;
    exp <<= 52;
    value.u64 = mantissa[0] | exp | sign;
    return value.d; 
}

// For 80bit platforms (x86 architectures), if it is little endian the layout of the longdouble is known
#elif LDBL_MANT_DIG == 64 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

#define NPY_LGDB_SIGN_POS 0
#define NPY_LGDB_SIGN_NEG 1

typedef union {
    npy_longdouble ld;
    uint64_t u64[2]; // Two 64-bit halves
} ld_128_union;

npy_longdouble _int_to_ld(int64_t *val, int exp, int sign) {
    uint64_t mantissa[3];
    for (int i = 0; i < 3; i++) { mantissa[i] = (uint64_t)val[i]; }

    ld_128_union value;
    uint64_t guard;
    int foo = 0;
    if (exp == 0) {
        sign = (sign == 1) ? -1 : 1;
        return (npy_longdouble)sign*(npy_longdouble)mantissa[0];
    } else if (mantissa[0] == 0) {
        guard = (mantissa[2] >> 63) & 1;
    } else {
        for (int i = 63; i >= 0; i--) {
            if ((mantissa[0] >> i) & 1) {
                foo = i + 1;
                break;
            }
        }
        guard = (mantissa[1] >> (foo + 1)) & 1;
    }
    exp += foo - 1;
    value.u64[0] = (mantissa[1] >> foo) | (mantissa[0] << (64 - foo));
    
    // Rounding (round to nearest even)
    if (guard) {
        value.u64[0]++;
        if (value.u64[0] == 0) {
            value.u64[0] = 0x8000000000000000;
            exp++;
        }
    }
    if (exp >= LDBL_MAX_EXP) {
        return _ldbl_ovfl_err();
    }
    value.u64[1] = (uint64_t)(exp + 16383) | (uint64_t)(sign << 15);
    return value.ld;
    
}
#else

#define NPY_LGDB_SIGN_POS 1
#define NPY_LGDB_SIGN_NEG -1

#ifndef NPY_LDBL_MAX_EXP
    #if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
        #define NPY_LDBL_MAX_EXP DBL_MAX_EXP
    #else
        #define NPY_LDBL_MAX_EXP LDBL_MAX_EXP
    #endif
#endif

// Since there are many types of long double, 64, 80, 128... 
// and no consensus, as everything changes platform to platform.
// For more performance needs to be platform specific
npy_longdouble _int_to_ld(int64_t *val, int exp, int sign) {
    uint64_t mantissa[3];
    for (int i = 0; i < 3; i++) { mantissa[i] = (uint64_t)val[i]; }
    npy_longdouble ld;
    if (exp == 0) {
        ld = (npy_longdouble)sign * (npy_longdouble)mantissa[0];
    } else if (exp == NPY_LDBL_MAX_EXP && mantissa[0] == 0) {
        ld = (npy_longdouble)sign * ((npy_longdouble)mantissa[1] * powl(2.0L, (npy_longdouble)(exp - 64)) + 
            (npy_longdouble)mantissa[2] * powl(2.0L, (npy_longdouble)(exp - 128)));
            //Sometimes it overflows in weird ways
        if (ld == (npy_longdouble)INFINITY || ld == (npy_longdouble)(-INFINITY) || ld == (npy_longdouble)NAN || ld == (npy_longdouble)(-NAN)) {
            return _ldbl_ovfl_err();
        }
    } else if (exp >= NPY_LDBL_MAX_EXP) {
        return _ldbl_ovfl_err();
    } else {
    ld = (npy_longdouble)sign * ((npy_longdouble)mantissa[0] * powl(2.0L, (npy_longdouble)(exp)) + 
    (npy_longdouble)mantissa[1] * powl(2.0L, (npy_longdouble)(exp - 64)) + 
    (npy_longdouble)mantissa[2] * powl(2.0L, (npy_longdouble)(exp - 128)));
    }
    return ld;
}
#endif

// Helper functions that get the exponent and mantissa, this works on all platforms
void _get_num(PyObject* py_int, int64_t* val, int *exp, int *ovf) {
    PyObject* shift = PyLong_FromLong(64);
    while (*ovf != 0) {
        *exp += 64;
        val[2] = val[1]; //only needed for 128 bit platform
        val[1] = (uint64_t)PyLong_AsUnsignedLongLongMask(py_int);
        py_int = PyNumber_Rshift(py_int, shift);
        val[0] = (uint64_t)PyLong_AsLongLongAndOverflow(py_int, ovf);
    }
}

void _fix_py_num(PyObject* py_int, int64_t* val, int *exp, int *sign) {

    int overflow;
    val[0] = PyLong_AsLongLongAndOverflow(py_int, &overflow);
    if (overflow == 1) {
        *sign = NPY_LGDB_SIGN_POS;
        _get_num(py_int, val, exp, &overflow);
    } else if (overflow == -1) {
        *sign = NPY_LGDB_SIGN_NEG;
        py_int = PyNumber_Negative(py_int);
        _get_num(py_int, val, exp, &overflow);
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
Now gives an overflow error if the number is too big, follows same rules as python's float()
*/
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj) {

    npy_longdouble value;
    if (PyFloat_Check(long_obj)) {
        value = (npy_longdouble)PyFloat_AsDouble(long_obj);
    } else if (PyLong_Check(long_obj)) {
        int sign;
        int E = 0;
        int64_t val[3]; //needs to be size 3 in 128bit prescision
        _fix_py_num(long_obj, val, &E, &sign);
        value = _int_to_ld(val, E, sign);
    } else {
        PyErr_SetString(PyExc_TypeError, "Expected a number (int or float)");
        return -1;
    }

    if (PyErr_Occurred()) {
        return -1;
    }
    return value;
}