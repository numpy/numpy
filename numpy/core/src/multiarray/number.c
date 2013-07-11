#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "number.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

NPY_NO_EXPORT NumericOps n_ops; /* NB: static objects initialized to zero */

/*
 * Dictionary can contain any of the numeric operations, by name.
 * Those not present will not be changed
 */

/* FIXME - macro contains a return */
#define SET(op)   temp = PyDict_GetItemString(dict, #op); \
    if (temp != NULL) { \
        if (!(PyCallable_Check(temp))) { \
            return -1; \
        } \
        Py_INCREF(temp); \
        Py_XDECREF(n_ops.op); \
        n_ops.op = temp; \
    }


/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(_ones_like);
    SET(sqrt);
    SET(negative);
    SET(absolute);
    SET(invert);
    SET(left_shift);
    SET(right_shift);
    SET(bitwise_and);
    SET(bitwise_or);
    SET(bitwise_xor);
    SET(less);
    SET(less_equal);
    SET(equal);
    SET(not_equal);
    SET(greater);
    SET(greater_equal);
    SET(floor_divide);
    SET(true_divide);
    SET(logical_or);
    SET(logical_and);
    SET(floor);
    SET(ceil);
    SET(maximum);
    SET(minimum);
    SET(rint);
    SET(conjugate);
    return 0;
}

/* FIXME - macro contains goto */
#define GET(op) if (n_ops.op &&                                         \
                    (PyDict_SetItemString(dict, #op, n_ops.op)==-1))    \
        goto fail;

/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(power);
    GET(square);
    GET(reciprocal);
    GET(_ones_like);
    GET(sqrt);
    GET(negative);
    GET(absolute);
    GET(invert);
    GET(left_shift);
    GET(right_shift);
    GET(bitwise_and);
    GET(bitwise_or);
    GET(bitwise_xor);
    GET(less);
    GET(less_equal);
    GET(equal);
    GET(not_equal);
    GET(greater);
    GET(greater_equal);
    GET(floor_divide);
    GET(true_divide);
    GET(logical_or);
    GET(logical_and);
    GET(floor);
    GET(ceil);
    GET(maximum);
    GET(minimum);
    GET(rint);
    GET(conjugate);
    return dict;

 fail:
    Py_DECREF(dict);
    return NULL;
}

static PyObject *
_get_keywords(int rtype, PyArrayObject *out)
{
    PyObject *kwds = NULL;
    if (rtype != NPY_NOTYPE || out != NULL) {
        kwds = PyDict_New();
        if (rtype != NPY_NOTYPE) {
            PyArray_Descr *descr;
            descr = PyArray_DescrFromType(rtype);
            if (descr) {
                PyDict_SetItemString(kwds, "dtype", (PyObject *)descr);
                Py_DECREF(descr);
            }
        }
        if (out != NULL) {
            PyDict_SetItemString(kwds, "out", (PyObject *)out);
        }
    }
    return kwds;
}

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "reduce");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out)
{
    PyObject *args, *ret = NULL, *meth;
    PyObject *kwds;
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    args = Py_BuildValue("(Oi)", m1, axis);
    kwds = _get_keywords(rtype, out);
    meth = PyObject_GetAttrString(op, "accumulate");
    if (meth && PyCallable_Check(meth)) {
        ret = PyObject_Call(meth, args, kwds);
    }
    Py_DECREF(args);
    Py_DECREF(meth);
    Py_XDECREF(kwds);
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    if (!PyArray_Check(m2)) {
          /*
           * Catch priority inversion and punt, but only if it's guaranteed
           * that we were called through m1 and the other guy is not an array
           * at all. Note that some arrays need to pass through here even
           * with priorities inverted, for example: float(17) * np.matrix(...)
           *
           * See also:
           * - https://github.com/numpy/numpy/issues/3502
           * - https://github.com/numpy/numpy/issues/3503
           */
          double m1_prio = PyArray_GetPriority(m1, NPY_SCALAR_PRIORITY);
          double m2_prio = PyArray_GetPriority(m2, NPY_SCALAR_PRIORITY);
          if (m1_prio < m2_prio) {
              Py_INCREF(Py_NotImplemented);
              return Py_NotImplemented;
          }
    }

    return PyObject_CallFunction(op, "OO", m1, m2);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "(O)", m1);
}

static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OOO", m1, m2, m1);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    if (op == NULL) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyObject_CallFunction(op, "OO", m1, m1);
}

static PyObject *
array_add(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
}

#if !defined(NPY_PY3K)
static PyObject *
array_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.divide);
}
#endif

static PyObject *
array_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.remainder);
}

/* Determine if object is a scalar and if so, convert the object
 *   to a double and place it in the out_exponent argument
 *   and return the "scalar kind" as a result.   If the object is
 *   not a scalar (or if there are other error conditions)
 *   return NPY_NOSCALAR, and out_exponent is undefined.
 */
static NPY_SCALARKIND
is_scalar_with_conversion(PyObject *o2, double* out_exponent)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyInt_Check(o2)) {
        *out_exponent = (double)PyInt_AsLong(o2);
        return NPY_INTPOS_SCALAR;
    }
    if (optimize_fpexps && PyFloat_Check(o2)) {
        *out_exponent = PyFloat_AsDouble(o2);
        return NPY_FLOAT_SCALAR;
    }
    if ((PyArray_IsZeroDim(o2) &&
         ((PyArray_ISINTEGER((PyArrayObject *)o2) ||
           (optimize_fpexps && PyArray_ISFLOAT((PyArrayObject *)o2))))) ||
        PyArray_IsScalar(o2, Integer) ||
        (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        if (temp != NULL) {
            *out_exponent = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            if (PyArray_IsZeroDim(o2)) {
                if (PyArray_ISINTEGER((PyArrayObject *)o2)) {
                    return NPY_INTPOS_SCALAR;
                }
                else { /* ISFLOAT */
                    return NPY_FLOAT_SCALAR;
                }
            }
            else if PyArray_IsScalar(o2, Integer) {
                    return NPY_INTPOS_SCALAR;
            }
            else { /* IsScalar(o2, Floating) */
                return NPY_FLOAT_SCALAR;
            }
        }
    }
    if (PyIndex_Check(o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value==NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return NPY_NOSCALAR;
        }
        val = PyInt_AsSsize_t(value);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double) val;
        return NPY_INTPOS_SCALAR;
    }
    return NPY_NOSCALAR;
}

/* optimize float array or complex array to a scalar power */
static PyObject *
fast_scalar_power(PyArrayObject *a1, PyObject *o2, int inplace)
{
    double exponent;
    NPY_SCALARKIND kind;   /* NPY_NOSCALAR is not scalar */

    if (PyArray_Check(a1) && ((kind=is_scalar_with_conversion(o2, &exponent))>0)) {
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exponent == 1.0) {
                /* we have to do this one special, as the
                   "copy" method of array objects isn't set
                   up early enough to be added
                   by PyArray_SetNumericOps.
                */
                if (inplace) {
                    Py_INCREF(a1);
                    return (PyObject *)a1;
                } else {
                    return PyArray_Copy(a1);
                }
            }
            else if (exponent == -1.0) {
                fastop = n_ops.reciprocal;
            }
            else if (exponent ==  0.0) {
                fastop = n_ops._ones_like;
            }
            else if (exponent ==  0.5) {
                fastop = n_ops.sqrt;
            }
            else if (exponent ==  2.0) {
                fastop = n_ops.square;
            }
            else {
                return NULL;
            }

            if (inplace) {
                return PyArray_GenericInplaceUnaryFunction(a1, fastop);
            } else {
                return PyArray_GenericUnaryFunction(a1, fastop);
            }
        }
        /* Because this is called with all arrays, we need to
         *  change the output if the kind of the scalar is different
         *  than that of the input and inplace is not on ---
         *  (thus, the input should be up-cast)
         */
        else if (exponent == 2.0) {
            fastop = n_ops.multiply;
            if (inplace) {
                return PyArray_GenericInplaceBinaryFunction
                    (a1, (PyObject *)a1, fastop);
            }
            else {
                PyArray_Descr *dtype = NULL;
                PyObject *res;

                /* We only special-case the FLOAT_SCALAR and integer types */
                if (kind == NPY_FLOAT_SCALAR && PyArray_ISINTEGER(a1)) {
                    dtype = PyArray_DescrFromType(NPY_DOUBLE);
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 == NULL) {
                        return NULL;
                    }
                }
                else {
                    Py_INCREF(a1);
                }
                res = PyArray_GenericBinaryFunction(a1, (PyObject *)a1, fastop);
                Py_DECREF(a1);
                return res;
            }
        }
    }
    return NULL;
}

static PyObject *
array_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 0);
    if (!value) {
        value = PyArray_GenericBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}


static PyObject *
array_negative(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.negative);
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.absolute);
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    return PyArray_GenericUnaryFunction(m1, n_ops.invert);
}

static PyObject *
array_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.multiply);
}

#if !defined(NPY_PY3K)
static PyObject *
array_inplace_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.divide);
}
#endif

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value;
    value = fast_scalar_power(a1, o2, 1);
    if (!value) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.floor_divide);
}

static PyObject *
array_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.true_divide);
}

static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.floor_divide);
}

static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.true_divide);
}

static int
_array_nonzero(PyArrayObject *mp)
{
    npy_intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        return PyArray_DESCR(mp)->f->nonzero(PyArray_DATA(mp), mp);
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array " \
                        "with more than one element is ambiguous. " \
                        "Use a.any() or a.all()");
        return -1;
    }
}



static PyObject *
array_divmod(PyArrayObject *op1, PyObject *op2)
{
    PyObject *divp, *modp, *result;

    divp = array_floor_divide(op1, op2);
    if (divp == NULL) {
        return NULL;
    }
    else if(divp == Py_NotImplemented) {
        return divp;
    }
    modp = array_remainder(op1, op2);
    if (modp == NULL) {
        Py_DECREF(divp);
        return NULL;
    }
    else if(modp == Py_NotImplemented) {
        Py_DECREF(divp);
        return modp;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}


NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be"\
                        " converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_int == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to int");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_float(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (pv == NULL) {
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a "\
                        "float; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_float == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to float");
        Py_DECREF(pv);
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;
}

#if !defined(NPY_PY3K)

static PyObject *
array_long(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_long == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to long");
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_oct(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_oct == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to oct");
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_hex(PyArrayObject *v)
{
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can "\
                        "be converted to Python scalars");
        return NULL;
    }
    pv = PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; "\
                        "scalar object is not a number");
        return NULL;
    }
    if (Py_TYPE(pv)->tp_as_number->nb_hex == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert "\
                        "scalar number to hex");
        return NULL;
    }
    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) &&
                    PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError,
                "object array may be self-referencing");
        return NULL;
    }
    pv2 = Py_TYPE(pv)->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;
}

#endif

static PyObject *
_array_copy_nice(PyArrayObject *self)
{
    return PyArray_Return((PyArrayObject *) PyArray_Copy(self));
}

static PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only integer arrays with "     \
                        "one element can be converted to an index");
        return NULL;
    }
    if (PyArray_NDIM(v) != 0) {
        if (DEPRECATE("converting an array with ndim > 0 to an index"
                      " will result in an error in the future") < 0) {
            return NULL;
        }
    }
    return PyArray_DESCR(v)->f->getitem(PyArray_DATA(v), v);
}


NPY_NO_EXPORT PyNumberMethods array_as_number = {
    (binaryfunc)array_add,                      /*nb_add*/
    (binaryfunc)array_subtract,                 /*nb_subtract*/
    (binaryfunc)array_multiply,                 /*nb_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_divide,                   /*nb_divide*/
#endif
    (binaryfunc)array_remainder,                /*nb_remainder*/
    (binaryfunc)array_divmod,                   /*nb_divmod*/
    (ternaryfunc)array_power,                   /*nb_power*/
    (unaryfunc)array_negative,                  /*nb_neg*/
    (unaryfunc)_array_copy_nice,                /*nb_pos*/
    (unaryfunc)array_absolute,                  /*(unaryfunc)array_abs,*/
    (inquiry)_array_nonzero,                    /*nb_nonzero*/
    (unaryfunc)array_invert,                    /*nb_invert*/
    (binaryfunc)array_left_shift,               /*nb_lshift*/
    (binaryfunc)array_right_shift,              /*nb_rshift*/
    (binaryfunc)array_bitwise_and,              /*nb_and*/
    (binaryfunc)array_bitwise_xor,              /*nb_xor*/
    (binaryfunc)array_bitwise_or,               /*nb_or*/
#if !defined(NPY_PY3K)
    0,                                          /*nb_coerce*/
#endif
    (unaryfunc)array_int,                       /*nb_int*/
#if defined(NPY_PY3K)
    0,                                          /*nb_reserved*/
#else
    (unaryfunc)array_long,                      /*nb_long*/
#endif
    (unaryfunc)array_float,                     /*nb_float*/
#if !defined(NPY_PY3K)
    (unaryfunc)array_oct,                       /*nb_oct*/
    (unaryfunc)array_hex,                       /*nb_hex*/
#endif

    /*
     * This code adds augmented assignment functionality
     * that was made available in Python 2.0
     */
    (binaryfunc)array_inplace_add,              /*inplace_add*/
    (binaryfunc)array_inplace_subtract,         /*inplace_subtract*/
    (binaryfunc)array_inplace_multiply,         /*inplace_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_inplace_divide,           /*inplace_divide*/
#endif
    (binaryfunc)array_inplace_remainder,        /*inplace_remainder*/
    (ternaryfunc)array_inplace_power,           /*inplace_power*/
    (binaryfunc)array_inplace_left_shift,       /*inplace_lshift*/
    (binaryfunc)array_inplace_right_shift,      /*inplace_rshift*/
    (binaryfunc)array_inplace_bitwise_and,      /*inplace_and*/
    (binaryfunc)array_inplace_bitwise_xor,      /*inplace_xor*/
    (binaryfunc)array_inplace_bitwise_or,       /*inplace_or*/

    (binaryfunc)array_floor_divide,             /*nb_floor_divide*/
    (binaryfunc)array_true_divide,              /*nb_true_divide*/
    (binaryfunc)array_inplace_floor_divide,     /*nb_inplace_floor_divide*/
    (binaryfunc)array_inplace_true_divide,      /*nb_inplace_true_divide*/
    (unaryfunc)array_index,                     /* nb_index */
};
