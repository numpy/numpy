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
#include "convert_datatype.h"

/*************************************************************************
 ****************   Implement Number Protocol ****************************
 *************************************************************************/

// this is not in the global data struct to avoid needing to include the
// definition of the NumericOps struct in multiarraymodule.h
//
// it is filled in during module initialization in a thread-safe manner
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

/* FIXME - macro contains returns  */
#define SET(op) \
    res = PyDict_GetItemStringRef(dict, #op, &temp); \
    if (res == -1) { \
        return -1; \
    } \
    else if (res == 1) { \
        if (!(PyCallable_Check(temp))) { \
            Py_DECREF(temp); \
            return -1; \
        } \
        Py_XSETREF(n_ops.op, temp); \
    }

NPY_NO_EXPORT int
_PyArray_SetNumericOps(PyObject *dict)
{
    PyObject *temp = NULL;
    int res;
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

    // initialize static globals needed for matmul
    npy_static_pydata.axes_1d_obj_kwargs = Py_BuildValue(
            "{s, [(i), (i, i), (i)]}", "axes", -1, -2, -1, -1);
    if (npy_static_pydata.axes_1d_obj_kwargs == NULL) {
        return -1;
    }

    npy_static_pydata.axes_2d_obj_kwargs = Py_BuildValue(
            "{s, [(i, i), (i, i), (i, i)]}", "axes", -2, -1, -2, -1, -2, -1);
    if (npy_static_pydata.axes_2d_obj_kwargs == NULL) {
        return -1;
    }

    return 0;
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
    INPLACE_GIVE_UP_IF_NEEDED(self, other,
            nb_inplace_matrix_multiply, array_inplace_matrix_multiply);

    PyObject *args = PyTuple_Pack(3, self, other, self);
    if (args == NULL) {
        return NULL;
    }
    PyObject *kwargs;

    /*
     * Unlike `matmul(a, b, out=a)` we ensure that the result is not broadcast
     * if the result without `out` would have less dimensions than `a`.
     * Since the signature of matmul is '(n?,k),(k,m?)->(n?,m?)' this is the
     * case exactly when the second operand has both core dimensions.
     *
     * The error here will be confusing, but for now, we enforce this by
     * passing the correct `axes=`.
     */
    if (PyArray_NDIM(self) == 1) {
        kwargs = npy_static_pydata.axes_1d_obj_kwargs;
    }
    else {
        kwargs = npy_static_pydata.axes_2d_obj_kwargs;
    }
    PyObject *res = PyObject_Call(n_ops.matmul, args, kwargs);
    Py_DECREF(args);

    if (res == NULL) {
        /*
         * AxisError should indicate that the axes argument didn't work out
         * which should mean the second operand not being 2 dimensional.
         */
        if (PyErr_ExceptionMatches(npy_static_pydata.AxisError)) {
            PyErr_SetString(PyExc_ValueError,
                "inplace matrix multiplication requires the first operand to "
                "have at least one and the second at least two dimensions.");
        }
    }

    return res;
}

static int
fast_scalar_power(PyObject *o1, PyObject *o2, int inplace, PyObject **result)
{
    PyObject *fastop = NULL;
    if (PyLong_CheckExact(o2)) {
        int overflow = 0;
        long exp = PyLong_AsLongAndOverflow(o2, &overflow);
        if (overflow != 0) {
            return -1;
        }

        if (exp == -1) {
            fastop = n_ops.reciprocal;
        }
        else if (exp == 2) {
            fastop = n_ops.square;
        }
        else {
            return 1;
        }
    }
    else if (PyFloat_CheckExact(o2)) {
        double exp = PyFloat_AsDouble(o2);
        if (exp == 0.5) {
            fastop = n_ops.sqrt;
        }
        else {
            return 1;
        }
    }
    else {
        return 1;
    }

    PyArrayObject *a1 = (PyArrayObject *)o1;
    if (!(PyArray_ISFLOAT(a1) || PyArray_ISCOMPLEX(a1))) {
        return 1;
    }

    if (inplace || can_elide_temp_unary(a1)) {
        *result = PyArray_GenericInplaceUnaryFunction(a1, fastop);
    }
    else {
        *result = PyArray_GenericUnaryFunction(a1, fastop);
    }

    return 0;
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

    if (fast_scalar_power((PyObject *) a1, o2, 1, &value) != 0) {
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
        res = PyDataType_GetArrFuncs(PyArray_DESCR(mp))->nonzero(PyArray_DATA(mp), mp);
        /* nonzero has no way to indicate an error, but one can occur */
        if (PyErr_Occurred()) {
            res = -1;
        }
        Py_LeaveRecursiveCall();
        return res;
    }
    else if (n == 0) {
        PyErr_SetString(PyExc_ValueError,
                "The truth value of an empty array is ambiguous. "
                "Use `array.size > 0` to check that an array is not empty.");
        return -1;
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
