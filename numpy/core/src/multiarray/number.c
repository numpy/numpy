#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "common.h"
#include "number.h"
#include "temp_elide.h"

#include "binop_override.h"
#include "ufunc_override.h"
#include "abstractdtypes.h"
#include "common_dtype.h"
#include "convert_datatype.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

NPY_NO_EXPORT NumericOps n_ops; /* NB: static objects initialized to zero */

/*
 * Forward declarations. Might want to move functions around instead
 */
static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2);
static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo));
static PyObject *
array_inplace_matrix_multiply(PyArrayObject *m1, PyObject *m2);

/*
 * Dictionary can contain any of the numeric operations, by name.
 * Those not present will not be changed
 */

/* FIXME - macro contains a return */
#define SET(op)   temp = _PyDict_GetItemStringWithError(dict, #op); \
    if (temp == NULL && PyErr_Occurred()) { \
        return -1; \
    } \
    else if (temp != NULL) { \
        if (!(PyCallable_Check(temp))) { \
            return -1; \
        } \
        Py_INCREF(temp); \
        Py_XDECREF(n_ops.op); \
        n_ops.op = temp; \
    }

NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    SET(add);
    SET(subtract);
    SET(multiply);
    SET(divide);
    SET(remainder);
    SET(divmod);
    SET(power);
    SET(square);
    SET(reciprocal);
    SET(_ones_like);
    SET(sqrt);
    SET(cbrt);
    SET(negative);
    SET(positive);
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
    SET(matmul);
    SET(clip);
    return 0;
}

/*NUMPY_API
 *Set internal structure with number functions that all arrays will use
 */
NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict)
{
    /* 2018-09-09, 1.16 */
    if (DEPRECATE("PyArray_SetNumericOps is deprecated. Use "
        "PyUFunc_ReplaceLoopBySignature to replace ufunc inner loop functions "
        "instead.") < 0) {
        return -1;
    }
    return _PyArray_SetNumericOps(dict);
}

/* Note - macro contains goto */
#define GET(op) if (n_ops.op &&                                         \
                    (PyDict_SetItemString(dict, #op, n_ops.op)==-1))    \
        goto fail;

NPY_NO_EXPORT PyObject *
_PyArray_GetNumericOps(void)
{
    PyObject *dict;
    if ((dict = PyDict_New())==NULL)
        return NULL;
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(divmod);
    GET(power);
    GET(square);
    GET(reciprocal);
    GET(_ones_like);
    GET(sqrt);
    GET(negative);
    GET(positive);
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
    GET(matmul);
    GET(clip);
    return dict;

 fail:
    Py_DECREF(dict);
    return NULL;
}

/*NUMPY_API
  Get dictionary showing number functions that all arrays will use
*/
NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void)
{
    /* 2018-09-09, 1.16 */
    if (DEPRECATE("PyArray_GetNumericOps is deprecated.") < 0) {
        return NULL;
    }
    return _PyArray_GetNumericOps();
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
PyArray_GenericBinaryFunction(PyObject *m1, PyObject *m2, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m2, NULL);
}

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, NULL);
}

static PyObject *
PyArray_GenericInplaceBinaryFunction(PyArrayObject *m1,
                                     PyObject *m2, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m2, m1, NULL);
}

static PyObject *
PyArray_GenericInplaceUnaryFunction(PyArrayObject *m1, PyObject *op)
{
    return PyObject_CallFunctionObjArgs(op, m1, m1, NULL);
}

static PyObject *
array_add(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_add, array_add);
    if (try_binary_elide(m1, m2, &array_inplace_add, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_subtract(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_subtract, array_subtract);
    if (try_binary_elide(m1, m2, &array_inplace_subtract, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_multiply(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_multiply, array_multiply);
    if (try_binary_elide(m1, m2, &array_inplace_multiply, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.multiply);
}

static PyObject *
array_remainder(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_remainder, array_remainder);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_divmod(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_divmod, array_divmod);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.divmod);
}

static PyObject *
array_matrix_multiply(PyObject *m1, PyObject *m2)
{
    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_matrix_multiply, array_matrix_multiply);
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.matmul);
}

static PyObject *
array_inplace_matrix_multiply(PyArrayObject *self, PyObject *other)
{
    static PyObject *AxisError_cls = NULL;
    npy_cache_import("numpy.exceptions", "AxisError", &AxisError_cls);
    if (AxisError_cls == NULL) {
        return NULL;
    }

    INPLACE_GIVE_UP_IF_NEEDED(self, other,
            nb_inplace_matrix_multiply, array_inplace_matrix_multiply);

    /*
     * Unlike `matmul(a, b, out=a)` we ensure that the result is not broadcast
     * if the result without `out` would have less dimensions than `a`.
     * Since the signature of matmul is '(n?,k),(k,m?)->(n?,m?)' this is the
     * case exactly when the second operand has both core dimensions.
     *
     * The error here will be confusing, but for now, we enforce this by
     * passing the correct `axes=`.
     */
    static PyObject *axes_1d_obj_kwargs = NULL;
    static PyObject *axes_2d_obj_kwargs = NULL;
    if (NPY_UNLIKELY(axes_1d_obj_kwargs == NULL)) {
        axes_1d_obj_kwargs = Py_BuildValue(
                "{s, [(i), (i, i), (i)]}", "axes", -1, -2, -1, -1);
        if (axes_1d_obj_kwargs == NULL) {
            return NULL;
        }
    }
    if (NPY_UNLIKELY(axes_2d_obj_kwargs == NULL)) {
        axes_2d_obj_kwargs = Py_BuildValue(
                "{s, [(i, i), (i, i), (i, i)]}", "axes", -2, -1, -2, -1, -2, -1);
        if (axes_2d_obj_kwargs == NULL) {
            return NULL;
        }
    }

    PyObject *args = PyTuple_Pack(3, self, other, self);
    if (args == NULL) {
        return NULL;
    }
    PyObject *kwargs;
    if (PyArray_NDIM(self) == 1) {
        kwargs = axes_1d_obj_kwargs;
    }
    else {
        kwargs = axes_2d_obj_kwargs;
    }
    PyObject *res = PyObject_Call(n_ops.matmul, args, kwargs);
    Py_DECREF(args);

    if (res == NULL) {
        /*
         * AxisError should indicate that the axes argument didn't work out
         * which should mean the second operand not being 2 dimensional.
         */
        if (PyErr_ExceptionMatches(AxisError_cls)) {
            PyErr_SetString(PyExc_ValueError,
                "inplace matrix multiplication requires the first operand to "
                "have at least one and the second at least two dimensions.");
        }
    }

    return res;
}

/*
 * Determine if object is a scalar and if so, convert the object
 * to a double and place it in the out_exponent argument
 * and return the "scalar kind" as a result.   If the object is
 * not a scalar (or if there are other error conditions)
 * return NPY_NOSCALAR, and out_exponent is undefined.
 */
static NPY_SCALARKIND
is_scalar_with_conversion(PyObject *o2, double* out_exponent)
{
    PyObject *temp;
    const int optimize_fpexps = 1;

    if (PyLong_Check(o2)) {
        long tmp = PyLong_AsLong(o2);
        if (error_converting(tmp)) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double)tmp;
        return NPY_INTPOS_SCALAR;
    }

    if (optimize_fpexps && PyFloat_Check(o2)) {
        *out_exponent = PyFloat_AsDouble(o2);
        return NPY_FLOAT_SCALAR;
    }

    if (PyArray_Check(o2)) {
        if ((PyArray_NDIM((PyArrayObject *)o2) == 0) &&
                ((PyArray_ISINTEGER((PyArrayObject *)o2) ||
                 (optimize_fpexps && PyArray_ISFLOAT((PyArrayObject *)o2))))) {
            temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
            if (temp == NULL) {
                return NPY_NOSCALAR;
            }
            *out_exponent = PyFloat_AsDouble(o2);
            Py_DECREF(temp);
            if (PyArray_ISINTEGER((PyArrayObject *)o2)) {
                return NPY_INTPOS_SCALAR;
            }
            else { /* ISFLOAT */
                return NPY_FLOAT_SCALAR;
            }
        }
    }
    else if (PyArray_IsScalar(o2, Integer) ||
                (optimize_fpexps && PyArray_IsScalar(o2, Floating))) {
        temp = Py_TYPE(o2)->tp_as_number->nb_float(o2);
        if (temp == NULL) {
            return NPY_NOSCALAR;
        }
        *out_exponent = PyFloat_AsDouble(o2);
        Py_DECREF(temp);

        if (PyArray_IsScalar(o2, Integer)) {
                return NPY_INTPOS_SCALAR;
        }
        else { /* IsScalar(o2, Floating) */
            return NPY_FLOAT_SCALAR;
        }
    }
    else if (PyIndex_Check(o2)) {
        PyObject* value = PyNumber_Index(o2);
        Py_ssize_t val;
        if (value == NULL) {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return NPY_NOSCALAR;
        }
        val = PyLong_AsSsize_t(value);
        Py_DECREF(value);
        if (error_converting(val)) {
            PyErr_Clear();
            return NPY_NOSCALAR;
        }
        *out_exponent = (double) val;
        return NPY_INTPOS_SCALAR;
    }
    return NPY_NOSCALAR;
}

/*
 * optimize float array or complex array to a scalar power
 * returns 0 on success, -1 if no optimization is possible
 * the result is in value (can be NULL if an error occurred)
 */
static int
fast_scalar_power(PyObject *o1, PyObject *o2, int inplace,
                  PyObject **value)
{
    double exponent;
    NPY_SCALARKIND kind;   /* NPY_NOSCALAR is not scalar */

    if (PyArray_Check(o1) &&
            !PyArray_ISOBJECT((PyArrayObject *)o1) &&
            ((kind=is_scalar_with_conversion(o2, &exponent))>0)) {
        PyArrayObject *a1 = (PyArrayObject *)o1;
        PyObject *fastop = NULL;
        if (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) {
            if (exponent == 1.0) {
                fastop = n_ops.positive;
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
                return -1;
            }

            if (inplace || can_elide_temp_unary(a1)) {
                *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
            }
            else {
                *value = PyArray_GenericUnaryFunction(a1, fastop);
            }
            return 0;
        }
        /* Because this is called with all arrays, we need to
         *  change the output if the kind of the scalar is different
         *  than that of the input and inplace is not on ---
         *  (thus, the input should be up-cast)
         */
        else if (exponent == 2.0) {
            fastop = n_ops.square;
            if (inplace) {
                *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
            }
            else {
                /* We only special-case the FLOAT_SCALAR and integer types */
                if (kind == NPY_FLOAT_SCALAR && PyArray_ISINTEGER(a1)) {
                    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
                    a1 = (PyArrayObject *)PyArray_CastToType(a1, dtype,
                            PyArray_ISFORTRAN(a1));
                    if (a1 != NULL) {
                        /* cast always creates a new array */
                        *value = PyArray_GenericInplaceUnaryFunction(a1, fastop);
                        Py_DECREF(a1);
                    }
                }
                else {
                    *value = PyArray_GenericUnaryFunction(a1, fastop);
                }
            }
            return 0;
        }
    }
    /* no fast operation found */
    return -1;
}

static PyObject *
array_power(PyObject *a1, PyObject *o2, PyObject *modulo)
{
    PyObject *value = NULL;

    if (modulo != Py_None) {
        /* modular exponentiation is not implemented (gh-8804) */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    BINOP_GIVE_UP_IF_NEEDED(a1, o2, nb_power, array_power);
    if (fast_scalar_power(a1, o2, 0, &value) != 0) {
        value = PyArray_GenericBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_positive(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.positive);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.positive);
}

static PyObject *
array_negative(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.negative);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.negative);
}

static PyObject *
array_absolute(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1) && !PyArray_ISCOMPLEX(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.absolute);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.absolute);
}

static PyObject *
array_invert(PyArrayObject *m1)
{
    if (can_elide_temp_unary(m1)) {
        return PyArray_GenericInplaceUnaryFunction(m1, n_ops.invert);
    }
    return PyArray_GenericUnaryFunction(m1, n_ops.invert);
}

static PyObject *
array_left_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_lshift, array_left_shift);
    if (try_binary_elide(m1, m2, &array_inplace_left_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_right_shift(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_rshift, array_right_shift);
    if (try_binary_elide(m1, m2, &array_inplace_right_shift, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_bitwise_and(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_and, array_bitwise_and);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_and, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_bitwise_or(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_or, array_bitwise_or);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_or, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_bitwise_xor(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_xor, array_bitwise_xor);
    if (try_binary_elide(m1, m2, &array_inplace_bitwise_xor, &res, 1)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_inplace_add(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_add, array_inplace_add);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.add);
}

static PyObject *
array_inplace_subtract(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_subtract, array_inplace_subtract);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.subtract);
}

static PyObject *
array_inplace_multiply(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_multiply, array_inplace_multiply);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.multiply);
}

static PyObject *
array_inplace_remainder(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_remainder, array_inplace_remainder);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.remainder);
}

static PyObject *
array_inplace_power(PyArrayObject *a1, PyObject *o2, PyObject *NPY_UNUSED(modulo))
{
    /* modulo is ignored! */
    PyObject *value = NULL;

    INPLACE_GIVE_UP_IF_NEEDED(
            a1, o2, nb_inplace_power, array_inplace_power);
    if (fast_scalar_power((PyObject *)a1, o2, 1, &value) != 0) {
        value = PyArray_GenericInplaceBinaryFunction(a1, o2, n_ops.power);
    }
    return value;
}

static PyObject *
array_inplace_left_shift(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_lshift, array_inplace_left_shift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.left_shift);
}

static PyObject *
array_inplace_right_shift(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_rshift, array_inplace_right_shift);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.right_shift);
}

static PyObject *
array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_and, array_inplace_bitwise_and);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_and);
}

static PyObject *
array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_or, array_inplace_bitwise_or);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_or);
}

static PyObject *
array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_xor, array_inplace_bitwise_xor);
    return PyArray_GenericInplaceBinaryFunction(m1, m2, n_ops.bitwise_xor);
}

static PyObject *
array_floor_divide(PyObject *m1, PyObject *m2)
{
    PyObject *res;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_floor_divide, array_floor_divide);
    if (try_binary_elide(m1, m2, &array_inplace_floor_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.floor_divide);
}

static PyObject *
array_true_divide(PyObject *m1, PyObject *m2)
{
    PyObject *res;
    PyArrayObject *a1 = (PyArrayObject *)m1;

    BINOP_GIVE_UP_IF_NEEDED(m1, m2, nb_true_divide, array_true_divide);
    if (PyArray_CheckExact(m1) &&
            (PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1)) &&
            try_binary_elide(m1, m2, &array_inplace_true_divide, &res, 0)) {
        return res;
    }
    return PyArray_GenericBinaryFunction(m1, m2, n_ops.true_divide);
}

static PyObject *
array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_floor_divide, array_inplace_floor_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.floor_divide);
}

static PyObject *
array_inplace_true_divide(PyArrayObject *m1, PyObject *m2)
{
    INPLACE_GIVE_UP_IF_NEEDED(
            m1, m2, nb_inplace_true_divide, array_inplace_true_divide);
    return PyArray_GenericInplaceBinaryFunction(m1, m2,
                                                n_ops.true_divide);
}


static int
_array_nonzero(PyArrayObject *mp)
{
    npy_intp n;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        int res;
        if (Py_EnterRecursiveCall(" while converting array to bool")) {
            return -1;
        }
        res = PyArray_DESCR(mp)->f->nonzero(PyArray_DATA(mp), mp);
        /* nonzero has no way to indicate an error, but one can occur */
        if (PyErr_Occurred()) {
            res = -1;
        }
        Py_LeaveRecursiveCall();
        return res;
    }
    else if (n == 0) {
        /* 2017-09-25, 1.14 */
        if (DEPRECATE("The truth value of an empty array is ambiguous. "
                      "Returning False, but in future this will result in an error. "
                      "Use `array.size > 0` to check that an array is not empty.") < 0) {
            return -1;
        }
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array "
                        "with more than one element is ambiguous. "
                        "Use a.any() or a.all()");
        return -1;
    }
}

/*
 * Convert the array to a scalar if allowed, and apply the builtin function
 * to it. The where argument is passed onto Py_EnterRecursiveCall when the
 * array contains python objects.
 */
NPY_NO_EXPORT PyObject *
array_scalar_forward(PyArrayObject *v,
                     PyObject *(*builtin_func)(PyObject *),
                     const char *where)
{
    if (check_is_convertible_to_scalar(v) < 0) {
        return NULL;
    }

    PyObject *scalar;
    scalar = PyArray_GETITEM(v, PyArray_DATA(v));
    if (scalar == NULL) {
        return NULL;
    }

    /* Need to guard against recursion if our array holds references */
    if (PyDataType_REFCHK(PyArray_DESCR(v))) {
        PyObject *res;
        if (Py_EnterRecursiveCall(where) != 0) {
            Py_DECREF(scalar);
            return NULL;
        }
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        Py_LeaveRecursiveCall();
        return res;
    }
    else {
        PyObject *res;
        res = builtin_func(scalar);
        Py_DECREF(scalar);
        return res;
    }
}


NPY_NO_EXPORT PyObject *
array_float(PyArrayObject *v)
{
    return array_scalar_forward(v, &PyNumber_Float, " in ndarray.__float__");
}

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v)
{
    return array_scalar_forward(v, &PyNumber_Long, " in ndarray.__int__");
}

static PyObject *
array_index(PyArrayObject *v)
{
    if (!PyArray_ISINTEGER(v) || PyArray_NDIM(v) != 0) {
        PyErr_SetString(PyExc_TypeError,
            "only integer scalar arrays can be converted to a scalar index");
        return NULL;
    }
    return PyArray_GETITEM(v, PyArray_DATA(v));
}


NPY_NO_EXPORT PyNumberMethods array_as_number = {
    .nb_add = array_add,
    .nb_subtract = array_subtract,
    .nb_multiply = array_multiply,
    .nb_remainder = array_remainder,
    .nb_divmod = array_divmod,
    .nb_power = (ternaryfunc)array_power,
    .nb_negative = (unaryfunc)array_negative,
    .nb_positive = (unaryfunc)array_positive,
    .nb_absolute = (unaryfunc)array_absolute,
    .nb_bool = (inquiry)_array_nonzero,
    .nb_invert = (unaryfunc)array_invert,
    .nb_lshift = array_left_shift,
    .nb_rshift = array_right_shift,
    .nb_and = array_bitwise_and,
    .nb_xor = array_bitwise_xor,
    .nb_or = array_bitwise_or,

    .nb_int = (unaryfunc)array_int,
    .nb_float = (unaryfunc)array_float,

    .nb_inplace_add = (binaryfunc)array_inplace_add,
    .nb_inplace_subtract = (binaryfunc)array_inplace_subtract,
    .nb_inplace_multiply = (binaryfunc)array_inplace_multiply,
    .nb_inplace_remainder = (binaryfunc)array_inplace_remainder,
    .nb_inplace_power = (ternaryfunc)array_inplace_power,
    .nb_inplace_lshift = (binaryfunc)array_inplace_left_shift,
    .nb_inplace_rshift = (binaryfunc)array_inplace_right_shift,
    .nb_inplace_and = (binaryfunc)array_inplace_bitwise_and,
    .nb_inplace_xor = (binaryfunc)array_inplace_bitwise_xor,
    .nb_inplace_or = (binaryfunc)array_inplace_bitwise_or,

    .nb_floor_divide = array_floor_divide,
    .nb_true_divide = array_true_divide,
    .nb_inplace_floor_divide = (binaryfunc)array_inplace_floor_divide,
    .nb_inplace_true_divide = (binaryfunc)array_inplace_true_divide,

    .nb_index = (unaryfunc)array_index,

    .nb_matrix_multiply = array_matrix_multiply,
    .nb_inplace_matrix_multiply = (binaryfunc)array_inplace_matrix_multiply,
};
