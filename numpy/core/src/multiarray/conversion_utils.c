#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "npy_3kcompat.h"

#include "common.h"
#include "arraytypes.h"

#include "conversion_utils.h"

/****************************************************************
* Useful function for conversion when used with PyArg_ParseTuple
****************************************************************/

/*NUMPY_API
 *
 * Useful to pass as converter function for O& processing in PyArgs_ParseTuple.
 *
 * This conversion function can be used with the "O&" argument for
 * PyArg_ParseTuple.  It will immediately return an object of array type
 * or will convert to a CARRAY any other object.
 *
 * If you use PyArray_Converter, you must DECREF the array when finished
 * as you get a new reference to it.
 */
NPY_NO_EXPORT int
PyArray_Converter(PyObject *object, PyObject **address)
{
    if (PyArray_Check(object)) {
        *address = object;
        Py_INCREF(object);
        return PY_SUCCEED;
    }
    else {
        *address = PyArray_FromAny(object, NULL, 0, 0, CARRAY, NULL);
        if (*address == NULL) {
            return PY_FAIL;
        }
        return PY_SUCCEED;
    }
}

/*NUMPY_API
 * Useful to pass as converter function for O& processing in
 * PyArgs_ParseTuple for output arrays
 */
NPY_NO_EXPORT int
PyArray_OutputConverter(PyObject *object, PyArrayObject **address)
{
    if (object == NULL || object == Py_None) {
        *address = NULL;
        return PY_SUCCEED;
    }
    if (PyArray_Check(object)) {
        *address = (PyArrayObject *)object;
        return PY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "output must be an array");
        *address = NULL;
        return PY_FAIL;
    }
}

/*NUMPY_API
 * Get intp chunk from sequence
 *
 * This function takes a Python sequence object and allocates and
 * fills in an intp array with the converted values.
 *
 * Remember to free the pointer seq.ptr when done using
 * PyDimMem_FREE(seq.ptr)**
 */
NPY_NO_EXPORT int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq)
{
    int len;
    int nd;

    seq->ptr = NULL;
    seq->len = 0;
    if (obj == Py_None) {
        return PY_SUCCEED;
    }
    len = PySequence_Size(obj);
    if (len == -1) {
        /* Check to see if it is a number */
        if (PyNumber_Check(obj)) {
            len = 1;
        }
    }
    if (len < 0) {
        PyErr_SetString(PyExc_TypeError,
                        "expected sequence object with len >= 0");
        return PY_FAIL;
    }
    if (len > MAX_DIMS) {
        PyErr_Format(PyExc_ValueError, "sequence too large; "   \
                     "must be smaller than %d", MAX_DIMS);
        return PY_FAIL;
    }
    if (len > 0) {
        seq->ptr = PyDimMem_NEW(len);
        if (seq->ptr == NULL) {
            PyErr_NoMemory();
            return PY_FAIL;
        }
    }
    seq->len = len;
    nd = PyArray_IntpFromSequence(obj, (intp *)seq->ptr, len);
    if (nd == -1 || nd != len) {
        PyDimMem_FREE(seq->ptr);
        seq->ptr = NULL;
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get buffer chunk from object
 *
 * this function takes a Python object which exposes the (single-segment)
 * buffer interface and returns a pointer to the data segment
 *
 * You should increment the reference count by one of buf->base
 * if you will hang on to a reference
 *
 * You only get a borrowed reference to the object. Do not free the
 * memory...
 */
NPY_NO_EXPORT int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf)
{
    Py_ssize_t buflen;

    buf->ptr = NULL;
    buf->flags = BEHAVED;
    buf->base = NULL;
    if (obj == Py_None) {
        return PY_SUCCEED;
    }
    if (PyObject_AsWriteBuffer(obj, &(buf->ptr), &buflen) < 0) {
        PyErr_Clear();
        buf->flags &= ~WRITEABLE;
        if (PyObject_AsReadBuffer(obj, (const void **)&(buf->ptr),
                                  &buflen) < 0) {
            return PY_FAIL;
        }
    }
    buf->len = (intp) buflen;

    /* Point to the base of the buffer object if present */
    if (PyBuffer_Check(obj)) {
        buf->base = ((PyArray_Chunk *)obj)->base;
    }
    if (buf->base == NULL) {
        buf->base = obj;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get axis from an object (possibly None) -- a converter function,
 */
NPY_NO_EXPORT int
PyArray_AxisConverter(PyObject *obj, int *axis)
{
    if (obj == Py_None) {
        *axis = MAX_DIMS;
    }
    else {
        *axis = (int) PyInt_AsLong(obj);
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert an object to true / false
 */
NPY_NO_EXPORT int
PyArray_BoolConverter(PyObject *object, Bool *val)
{
    if (PyObject_IsTrue(object)) {
        *val = TRUE;
    }
    else {
        *val = FALSE;
    }
    if (PyErr_Occurred()) {
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to endian
 */
NPY_NO_EXPORT int
PyArray_ByteorderConverter(PyObject *obj, char *endian)
{
    char *str;

    *endian = PyArray_SWAP;
    str = PyString_AsString(obj);
    if (!str) {
        return PY_FAIL;
    }
    if (strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Byteorder string must be at least length 1");
        return PY_FAIL;
    }
    *endian = str[0];
    if (str[0] != PyArray_BIG && str[0] != PyArray_LITTLE
        && str[0] != PyArray_NATIVE && str[0] != PyArray_IGNORE) {
        if (str[0] == 'b' || str[0] == 'B') {
            *endian = PyArray_BIG;
        }
        else if (str[0] == 'l' || str[0] == 'L') {
            *endian = PyArray_LITTLE;
        }
        else if (str[0] == 'n' || str[0] == 'N') {
            *endian = PyArray_NATIVE;
        }
        else if (str[0] == 'i' || str[0] == 'I') {
            *endian = PyArray_IGNORE;
        }
        else if (str[0] == 's' || str[0] == 'S') {
            *endian = PyArray_SWAP;
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "%s is an unrecognized byteorder",
                         str);
            return PY_FAIL;
        }
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to sort kind
 */
NPY_NO_EXPORT int
PyArray_SortkindConverter(PyObject *obj, NPY_SORTKIND *sortkind)
{
    char *str;

    *sortkind = PyArray_QUICKSORT;
    str = PyString_AsString(obj);
    if (!str) {
        return PY_FAIL;
    }
    if (strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "Sort kind string must be at least length 1");
        return PY_FAIL;
    }
    if (str[0] == 'q' || str[0] == 'Q') {
        *sortkind = PyArray_QUICKSORT;
    }
    else if (str[0] == 'h' || str[0] == 'H') {
        *sortkind = PyArray_HEAPSORT;
    }
    else if (str[0] == 'm' || str[0] == 'M') {
        *sortkind = PyArray_MERGESORT;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "%s is an unrecognized kind of sort",
                     str);
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert object to searchsorted side
 */
NPY_NO_EXPORT int
PyArray_SearchsideConverter(PyObject *obj, void *addr)
{
    NPY_SEARCHSIDE *side = (NPY_SEARCHSIDE *)addr;
    char *str = PyString_AsString(obj);

    if (!str || strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "expected nonempty string for keyword 'side'");
        return PY_FAIL;
    }

    if (str[0] == 'l' || str[0] == 'L') {
        *side = NPY_SEARCHLEFT;
    }
    else if (str[0] == 'r' || str[0] == 'R') {
        *side = NPY_SEARCHRIGHT;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is an invalid value for keyword 'side'", str);
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*****************************
* Other conversion functions
*****************************/

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_PyIntAsInt(PyObject *o)
{
    long long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    PyObject *arr;
    PyArray_Descr *descr;
    int ret;


    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (long) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (long) PyLong_AsLong(o);
        goto finish;
    }

    descr = &INT_Descr;
    arr = NULL;
    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Py_INCREF(descr);
        arr = PyArray_CastToType((PyArrayObject *)o, descr, 0);
    }
    if (PyArray_IsScalar(o, Integer)) {
        Py_INCREF(descr);
        arr = PyArray_FromScalar(o, descr);
    }
    if (arr != NULL) {
        ret = *((int *)PyArray_DATA(arr));
        Py_DECREF(arr);
        return ret;
    }
#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
    if (o->ob_type->tp_as_number != NULL &&         \
        o->ob_type->tp_as_number->nb_int != NULL) {
        obj = o->ob_type->tp_as_number->nb_int(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
    else if (o->ob_type->tp_as_number != NULL &&                    \
             o->ob_type->tp_as_number->nb_long != NULL) {
        obj = o->ob_type->tp_as_number->nb_long(o);
        if (obj == NULL) {
            return -1;
        }
        long_value = (long) PyLong_AsLong(obj);
        Py_DECREF(obj);
    }
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (SIZEOF_LONG > SIZEOF_INT)
    if ((long_value < INT_MIN) || (long_value > INT_MAX)) {
        PyErr_SetString(PyExc_ValueError, "integer won't fit into a C int");
        return -1;
    }
#endif
    return (int) long_value;
}

/*NUMPY_API*/
NPY_NO_EXPORT intp
PyArray_PyIntAsIntp(PyObject *o)
{
    longlong long_value = -1;
    PyObject *obj;
    static char *msg = "an integer is required";
    PyObject *arr;
    PyArray_Descr *descr;
    intp ret;

    if (!o) {
        PyErr_SetString(PyExc_TypeError, msg);
        return -1;
    }
    if (PyInt_Check(o)) {
        long_value = (longlong) PyInt_AS_LONG(o);
        goto finish;
    } else if (PyLong_Check(o)) {
        long_value = (longlong) PyLong_AsLongLong(o);
        goto finish;
    }

#if SIZEOF_INTP == SIZEOF_LONG
    descr = &LONG_Descr;
#elif SIZEOF_INTP == SIZEOF_INT
    descr = &INT_Descr;
#else
    descr = &LONGLONG_Descr;
#endif
    arr = NULL;

    if (PyArray_Check(o)) {
        if (PyArray_SIZE(o)!=1 || !PyArray_ISINTEGER(o)) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }
        Py_INCREF(descr);
        arr = PyArray_CastToType((PyArrayObject *)o, descr, 0);
    }
    else if (PyArray_IsScalar(o, Integer)) {
        Py_INCREF(descr);
        arr = PyArray_FromScalar(o, descr);
    }
    if (arr != NULL) {
        ret = *((intp *)PyArray_DATA(arr));
        Py_DECREF(arr);
        return ret;
    }

#if (PY_VERSION_HEX >= 0x02050000)
    if (PyIndex_Check(o)) {
        PyObject* value = PyNumber_Index(o);
        if (value == NULL) {
            return -1;
        }
        long_value = (longlong) PyInt_AsSsize_t(value);
        goto finish;
    }
#endif
    if (o->ob_type->tp_as_number != NULL &&                 \
        o->ob_type->tp_as_number->nb_long != NULL) {
        obj = o->ob_type->tp_as_number->nb_long(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else if (o->ob_type->tp_as_number != NULL &&            \
             o->ob_type->tp_as_number->nb_int != NULL) {
        obj = o->ob_type->tp_as_number->nb_int(o);
        if (obj != NULL) {
            long_value = (longlong) PyLong_AsLongLong(obj);
            Py_DECREF(obj);
        }
    }
    else {
        PyErr_SetString(PyExc_NotImplementedError,"");
    }

 finish:
    if error_converting(long_value) {
            PyErr_SetString(PyExc_TypeError, msg);
            return -1;
        }

#if (SIZEOF_LONGLONG > SIZEOF_INTP)
    if ((long_value < MIN_INTP) || (long_value > MAX_INTP)) {
        PyErr_SetString(PyExc_ValueError,
                        "integer won't fit into a C intp");
        return -1;
    }
#endif
    return (intp) long_value;
}

/*NUMPY_API
 * PyArray_IntpFromSequence
 * Returns the number of dimensions or -1 if an error occurred.
 * vals must be large enough to hold maxvals
 */
NPY_NO_EXPORT int
PyArray_IntpFromSequence(PyObject *seq, intp *vals, int maxvals)
{
    int nd, i;
    PyObject *op, *err;

    /*
     * Check to see if sequence is a single integer first.
     * or, can be made into one
     */
    if ((nd=PySequence_Length(seq)) == -1) {
        if (PyErr_Occurred()) PyErr_Clear();
#if SIZEOF_LONG >= SIZEOF_INTP
        if (!(op = PyNumber_Int(seq))) {
            return -1;
        }
#else
        if (!(op = PyNumber_Long(seq))) {
            return -1;
        }
#endif
        nd = 1;
#if SIZEOF_LONG >= SIZEOF_INTP
        vals[0] = (intp ) PyInt_AsLong(op);
#else
        vals[0] = (intp ) PyLong_AsLongLong(op);
#endif
        Py_DECREF(op);

        /*
         * Check wether there was an error - if the error was an overflow, raise
         * a ValueError instead to be more helpful
         */
        if(vals[0] == -1) {
            err = PyErr_Occurred();
            if (err  &&
                PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                PyErr_SetString(PyExc_ValueError,
                        "Maximum allowed dimension exceeded");
            }
            if(err != NULL) {
                return -1;
            }
        }
    }
    else {
        for (i = 0; i < MIN(nd,maxvals); i++) {
            op = PySequence_GetItem(seq, i);
            if (op == NULL) {
                return -1;
            }
#if SIZEOF_LONG >= SIZEOF_INTP
            vals[i]=(intp )PyInt_AsLong(op);
#else
            vals[i]=(intp )PyLong_AsLongLong(op);
#endif
            Py_DECREF(op);

            /*
             * Check wether there was an error - if the error was an overflow,
             * raise a ValueError instead to be more helpful
             */
            if(vals[0] == -1) {
                err = PyErr_Occurred();
                if (err  &&
                    PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
                    PyErr_SetString(PyExc_ValueError,
                            "Maximum allowed dimension exceeded");
                }
                if(err != NULL) {
                    return -1;
                }
            }
        }
    }
    return nd;
}

/*NUMPY_API
 * Typestr converter
 */
NPY_NO_EXPORT int
PyArray_TypestrConvert(int itemsize, int gentype)
{
    int newtype = gentype;

    if (gentype == PyArray_GENBOOLLTR) {
        if (itemsize == 1) {
            newtype = PyArray_BOOL;
        }
        else {
            newtype = PyArray_NOTYPE;
        }
    }
    else if (gentype == PyArray_SIGNEDLTR) {
        switch(itemsize) {
        case 1:
            newtype = PyArray_INT8;
            break;
        case 2:
            newtype = PyArray_INT16;
            break;
        case 4:
            newtype = PyArray_INT32;
            break;
        case 8:
            newtype = PyArray_INT64;
            break;
#ifdef PyArray_INT128
        case 16:
            newtype = PyArray_INT128;
            break;
#endif
        default:
            newtype = PyArray_NOTYPE;
        }
    }
    else if (gentype == PyArray_UNSIGNEDLTR) {
        switch(itemsize) {
        case 1:
            newtype = PyArray_UINT8;
            break;
        case 2:
            newtype = PyArray_UINT16;
            break;
        case 4:
            newtype = PyArray_UINT32;
            break;
        case 8:
            newtype = PyArray_UINT64;
            break;
#ifdef PyArray_INT128
        case 16:
            newtype = PyArray_UINT128;
            break;
#endif
        default:
            newtype = PyArray_NOTYPE;
            break;
        }
    }
    else if (gentype == PyArray_FLOATINGLTR) {
        switch(itemsize) {
        case 4:
            newtype = PyArray_FLOAT32;
            break;
        case 8:
            newtype = PyArray_FLOAT64;
            break;
#ifdef PyArray_FLOAT80
        case 10:
            newtype = PyArray_FLOAT80;
            break;
#endif
#ifdef PyArray_FLOAT96
        case 12:
            newtype = PyArray_FLOAT96;
            break;
#endif
#ifdef PyArray_FLOAT128
        case 16:
            newtype = PyArray_FLOAT128;
            break;
#endif
        default:
            newtype = PyArray_NOTYPE;
        }
    }
    else if (gentype == PyArray_COMPLEXLTR) {
        switch(itemsize) {
        case 8:
            newtype = PyArray_COMPLEX64;
            break;
        case 16:
            newtype = PyArray_COMPLEX128;
            break;
#ifdef PyArray_FLOAT80
        case 20:
            newtype = PyArray_COMPLEX160;
            break;
#endif
#ifdef PyArray_FLOAT96
        case 24:
            newtype = PyArray_COMPLEX192;
            break;
#endif
#ifdef PyArray_FLOAT128
        case 32:
            newtype = PyArray_COMPLEX256;
            break;
#endif
        default:
            newtype = PyArray_NOTYPE;
        }
    }
    return newtype;
}

/* Lifted from numarray */
/* TODO: not documented */
/*NUMPY_API
  PyArray_IntTupleFromIntp
*/
NPY_NO_EXPORT PyObject *
PyArray_IntTupleFromIntp(int len, intp *vals)
{
    int i;
    PyObject *intTuple = PyTuple_New(len);

    if (!intTuple) {
        goto fail;
    }
    for (i = 0; i < len; i++) {
#if SIZEOF_INTP <= SIZEOF_LONG
        PyObject *o = PyInt_FromLong((long) vals[i]);
#else
        PyObject *o = PyLong_FromLongLong((longlong) vals[i]);
#endif
        if (!o) {
            Py_DECREF(intTuple);
            intTuple = NULL;
            goto fail;
        }
        PyTuple_SET_ITEM(intTuple, i, o);
    }

 fail:
    return intTuple;
}

