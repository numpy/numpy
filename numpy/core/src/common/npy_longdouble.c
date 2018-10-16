#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "npy_pycompat.h"

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
