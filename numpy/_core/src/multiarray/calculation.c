#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "lowlevel_strided_loops.h"
#include "dtypemeta.h"

#include "npy_config.h"



#include "common.h"
#include "number.h"

#include "calculation.h"
#include "array_assign.h"

static double
power_of_ten(int n)
{
    static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    double ret;
    if (n < 9) {
        ret = p10[n];
    }
    else {
        ret = 1e9;
        while (n-- > 9) {
            ret *= 10.;
        }
    }
    return ret;
}

NPY_NO_EXPORT PyObject *
_PyArray_ArgMinMaxCommon(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims,
        npy_bool is_argmax)
{
    PyArrayObject *ap = NULL, *rp = NULL;
    PyArray_ArgFunc* arg_func = NULL;
    char *ip, *func_name;
    npy_intp *rptr;
    npy_intp i, n, m;
    int elsize;
    // Keep a copy because axis changes via call to PyArray_CheckAxis
    int axis_copy = axis;
    npy_intp _shape_buf[NPY_MAXDIMS];
    npy_intp *out_shape;
    // Keep the number of dimensions and shape of
    // original array. Helps when `keepdims` is True.
    npy_intp* original_op_shape = PyArray_DIMS(op);
    int out_ndim = PyArray_NDIM(op);
    NPY_BEGIN_THREADS_DEF;

    if ((ap = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /*
     * We need to permute the array so that axis is placed at the end.
     * And all other dimensions are shifted left.
     */
    if (axis != PyArray_NDIM(ap)-1) {
        PyArray_Dims newaxes;
        npy_intp dims[NPY_MAXDIMS];
        int j;

        newaxes.ptr = dims;
        newaxes.len = PyArray_NDIM(ap);
        for (j = 0; j < axis; j++) {
            dims[j] = j;
        }
        for (j = axis; j < PyArray_NDIM(ap) - 1; j++) {
            dims[j] = j + 1;
        }
        dims[PyArray_NDIM(ap) - 1] = axis;
        op = (PyArrayObject *)PyArray_Transpose(ap, &newaxes);
        Py_DECREF(ap);
        if (op == NULL) {
            return NULL;
        }
    }
    else {
        op = ap;
    }

    // Will get native-byte order contiguous copy.
    PyArray_Descr *descr = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(op));
    if (descr == NULL) {
        return NULL;
    }
    ap = (PyArrayObject *)PyArray_FromArray(op, descr, NPY_ARRAY_DEFAULT);

    Py_DECREF(op);
    if (ap == NULL) {
        return NULL;
    }

    // Decides the shape of the output array.
    if (!keepdims) {
        out_ndim = PyArray_NDIM(ap) - 1;
        out_shape = PyArray_DIMS(ap);
    }
    else {
        out_shape = _shape_buf;
        if (axis_copy == NPY_RAVEL_AXIS) {
            for (int i = 0; i < out_ndim; i++) {
                out_shape[i] = 1;
            }
        }
        else {
            /*
             * While `ap` may be transposed, we can ignore this for `out` because the
             * transpose only reorders the size 1 `axis` (not changing memory layout).
             */
            memcpy(out_shape, original_op_shape, out_ndim * sizeof(npy_intp));
            out_shape[axis] = 1;
        }
    }

    if (is_argmax) {
        func_name = "argmax";
        arg_func = PyDataType_GetArrFuncs(PyArray_DESCR(ap))->argmax;
    }
    else {
        func_name = "argmin";
        arg_func = PyDataType_GetArrFuncs(PyArray_DESCR(ap))->argmin;
    }
    if (arg_func == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "data type not ordered");
        goto fail;
    }
    elsize = PyArray_ITEMSIZE(ap);
    m = PyArray_DIMS(ap)[PyArray_NDIM(ap)-1];
    if (m == 0) {
        PyErr_Format(PyExc_ValueError,
                    "attempt to get %s of an empty sequence",
                    func_name);
        goto fail;
    }

    if (!out) {
        rp = (PyArrayObject *)PyArray_NewFromDescr(
                Py_TYPE(ap), PyArray_DescrFromType(NPY_INTP),
                out_ndim, out_shape, NULL, NULL,
                0, (PyObject *)ap);
        if (rp == NULL) {
            goto fail;
        }
    }
    else {
        if ((PyArray_NDIM(out) != out_ndim) ||
                !PyArray_CompareLists(PyArray_DIMS(out), out_shape,
                                        out_ndim)) {
            PyErr_Format(PyExc_ValueError,
                    "output array does not match result of np.%s.",
                    func_name);
            goto fail;
        }
        rp = (PyArrayObject *)PyArray_FromArray(out,
                              PyArray_DescrFromType(NPY_INTP),
                              NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (rp == NULL) {
            goto fail;
        }
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap));
    n = PyArray_SIZE(ap)/m;
    rptr = (npy_intp *)PyArray_DATA(rp);
    for (ip = PyArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
        arg_func(ip, m, rptr, ap);
        rptr += 1;
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap));

    Py_DECREF(ap);
    /* Trigger the WRITEBACKIFCOPY if necessary */
    if (out != NULL && out != rp) {
        PyArray_ResolveWritebackIfCopy(rp);
        Py_DECREF(rp);
        rp = out;
        Py_INCREF(rp);
    }
    return (PyObject *)rp;

 fail:
    Py_DECREF(ap);
    Py_XDECREF(rp);
    return NULL;
}

NPY_NO_EXPORT PyObject*
_PyArray_ArgMaxWithKeepdims(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims)
{
    return _PyArray_ArgMinMaxCommon(op, axis, out, keepdims, 1);
}

/*NUMPY_API
 * ArgMax
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMax(PyArrayObject *op, int axis, PyArrayObject *out)
{
    return _PyArray_ArgMinMaxCommon(op, axis, out, 0, 1);
}

NPY_NO_EXPORT PyObject *
_PyArray_ArgMinWithKeepdims(PyArrayObject *op,
        int axis, PyArrayObject *out, int keepdims)
{
    return _PyArray_ArgMinMaxCommon(op, axis, out, keepdims, 0);
}

/*NUMPY_API
 * ArgMin
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMin(PyArrayObject *op, int axis, PyArrayObject *out)
{
    return _PyArray_ArgMinMaxCommon(op, axis, out, 0, 0);
}

/*NUMPY_API
 * Max
 */
NPY_NO_EXPORT PyObject *
PyArray_Max(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    arr = (PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, n_ops.maximum, axis,
                                        PyArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Min
 */
NPY_NO_EXPORT PyObject *
PyArray_Min(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    arr=(PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, n_ops.minimum, axis,
                                        PyArray_DESCR(arr)->type_num, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Ptp
 */
NPY_NO_EXPORT PyObject *
PyArray_Ptp(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;
    PyObject *obj1 = NULL, *obj2 = NULL;

    arr=(PyArrayObject *)PyArray_CheckAxis(ap, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    obj1 = PyArray_Max(arr, axis, out);
    if (obj1 == NULL) {
        goto fail;
    }
    obj2 = PyArray_Min(arr, axis, NULL);
    if (obj2 == NULL) {
        goto fail;
    }
    Py_DECREF(arr);
    if (out) {
        ret = PyObject_CallFunction(n_ops.subtract, "OOO", out, obj2, out);
    }
    else {
        ret = PyNumber_Subtract(obj1, obj2);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;

 fail:
    Py_XDECREF(arr);
    Py_XDECREF(obj1);
    Py_XDECREF(obj2);
    return NULL;
}



/*NUMPY_API
 * Set variance to 1 to bypass square-root calculation and return variance
 * Std
 */
NPY_NO_EXPORT PyObject *
PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
            int variance)
{
    return __New_PyArray_Std(self, axis, rtype, out, variance, 0);
}

NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num)
{
    PyObject *obj1 = NULL, *obj2 = NULL, *obj3 = NULL;
    PyArrayObject *arr1 = NULL, *arr2 = NULL, *arrnew = NULL;
    PyObject *ret = NULL, *newshape = NULL;
    int i, n;
    npy_intp val;

    arrnew = (PyArrayObject *)PyArray_CheckAxis(self, &axis, 0);
    if (arrnew == NULL) {
        return NULL;
    }
    /* Compute and reshape mean */
    arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(
                    PyArray_Mean(arrnew, axis, rtype, NULL));
    if (arr1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    n = PyArray_NDIM(arrnew);
    newshape = PyTuple_New(n);
    if (newshape == NULL) {
        Py_DECREF(arr1);
        Py_DECREF(arrnew);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (i == axis) {
            val = 1;
        }
        else {
            val = PyArray_DIM(arrnew,i);
        }
        PyTuple_SET_ITEM(newshape, i, PyLong_FromSsize_t(val));
    }
    arr2 = (PyArrayObject *)PyArray_Reshape(arr1, newshape);
    Py_DECREF(arr1);
    Py_DECREF(newshape);
    if (arr2 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }

    /* Compute x = x - mx */
    arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(
                PyNumber_Subtract((PyObject *)arrnew, (PyObject *)arr2));
    Py_DECREF(arr2);
    if (arr1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    /* Compute x * x */
    if (PyArray_ISCOMPLEX(arr1)) {
        obj3 = PyArray_Conjugate(arr1, NULL);
    }
    else {
        obj3 = (PyObject *)arr1;
        Py_INCREF(arr1);
    }
    if (obj3 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    arr2 = (PyArrayObject *)PyArray_EnsureAnyArray(
                PyArray_GenericBinaryFunction((PyObject *)arr1, obj3,
                                               n_ops.multiply));
    Py_DECREF(arr1);
    Py_DECREF(obj3);
    if (arr2 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    if (PyArray_ISCOMPLEX(arr2)) {
        obj3 = PyObject_GetAttrString((PyObject *)arr2, "real");
        switch(rtype) {
        case NPY_CDOUBLE:
            rtype = NPY_DOUBLE;
            break;
        case NPY_CFLOAT:
            rtype = NPY_FLOAT;
            break;
        case NPY_CLONGDOUBLE:
            rtype = NPY_LONGDOUBLE;
            break;
        }
    }
    else {
        obj3 = (PyObject *)arr2;
        Py_INCREF(arr2);
    }
    if (obj3 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    /* Compute add.reduce(x*x,axis) */
    obj1 = PyArray_GenericReduceFunction((PyArrayObject *)obj3, n_ops.add,
                                         axis, rtype, NULL);
    Py_DECREF(obj3);
    Py_DECREF(arr2);
    if (obj1 == NULL) {
        Py_DECREF(arrnew);
        return NULL;
    }
    n = PyArray_DIM(arrnew,axis);
    Py_DECREF(arrnew);
    n = (n-num);
    if (n == 0) {
        n = 1;
    }
    obj2 = PyFloat_FromDouble(1.0/((double )n));
    if (obj2 == NULL) {
        Py_DECREF(obj1);
        return NULL;
    }
    ret = PyNumber_Multiply(obj1, obj2);
    Py_DECREF(obj1);
    Py_DECREF(obj2);

    if (!variance) {
        arr1 = (PyArrayObject *)PyArray_EnsureAnyArray(ret);
        /* sqrt() */
        ret = PyArray_GenericUnaryFunction(arr1, n_ops.sqrt);
        Py_DECREF(arr1);
    }
    if (ret == NULL) {
        return NULL;
    }
    if (PyArray_CheckExact(self)) {
        goto finish;
    }
    if (PyArray_Check(self) && Py_TYPE(self) == Py_TYPE(ret)) {
        goto finish;
    }
    arr1 = (PyArrayObject *)PyArray_EnsureArray(ret);
    if (arr1 == NULL) {
        return NULL;
    }
    ret = PyArray_View(arr1, NULL, Py_TYPE(self));
    Py_DECREF(arr1);

finish:
    if (out) {
        if (PyArray_AssignArray(out, (PyArrayObject *)ret,
                    NULL, NPY_DEFAULT_ASSIGN_CASTING) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(ret);
        Py_INCREF(out);
        return (PyObject *)out;
    }
    return ret;
}


/*NUMPY_API
 *Sum
 */
NPY_NO_EXPORT PyObject *
PyArray_Sum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr, n_ops.add, axis,
                                        rtype, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Prod
 */
NPY_NO_EXPORT PyObject *
PyArray_Prod(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr,
                                        n_ops.multiply, axis,
                                        rtype, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 *CumSum
 */
NPY_NO_EXPORT PyObject *
PyArray_CumSum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericAccumulateFunction((PyArrayObject *)arr,
                                            n_ops.add, axis,
                                            rtype, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * CumProd
 */
NPY_NO_EXPORT PyObject *
PyArray_CumProd(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }

    ret = PyArray_GenericAccumulateFunction((PyArrayObject *)arr,
                                            n_ops.multiply, axis,
                                            rtype, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Round
 */
NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out)
{
    PyObject *f, *ret = NULL, *tmp, *op1, *op2;
    int ret_int=0;
    PyArray_Descr *my_descr;
    if (out && (PyArray_SIZE(out) != PyArray_SIZE(a))) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid output shape");
        return NULL;
    }
    if (PyArray_ISCOMPLEX(a)) {
        PyObject *part;
        PyObject *round_part;
        PyObject *arr;
        int res;

        if (out) {
            arr = (PyObject *)out;
            Py_INCREF(arr);
        }
        else {
            arr = PyArray_NewCopy(a, NPY_KEEPORDER);
            if (arr == NULL) {
                return NULL;
            }
        }

        /* arr.real = a.real.round(decimals) */
        part = PyObject_GetAttrString((PyObject *)a, "real");
        if (part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        res = PyObject_SetAttrString(arr, "real", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(arr);
            return NULL;
        }

        /* arr.imag = a.imag.round(decimals) */
        part = PyObject_GetAttrString((PyObject *)a, "imag");
        if (part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        res = PyObject_SetAttrString(arr, "imag", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(arr);
            return NULL;
        }
        return arr;
    }
    /* do the most common case first */
    if (decimals >= 0) {
        if (PyArray_ISINTEGER(a)) {
            if (out) {
                if (PyArray_AssignArray(out, a,
                            NULL, NPY_DEFAULT_ASSIGN_CASTING) < 0) {
                    return NULL;
                }
                Py_INCREF(out);
                return (PyObject *)out;
            }
            else {
                return PyArray_NewCopy(a, NPY_KEEPORDER);
            }
        }
        if (decimals == 0) {
            if (out) {
                return PyObject_CallFunction(n_ops.rint, "OO", a, out);
            }
            return PyObject_CallFunction(n_ops.rint, "O", a);
        }
        op1 = n_ops.multiply;
        op2 = n_ops.true_divide;
    }
    else {
        op1 = n_ops.true_divide;
        op2 = n_ops.multiply;
        if (decimals == INT_MIN) {
            // not technically correct but it doesn't matter because no one in
            // this millenium is using floating point numbers with enough
            // accuracy for this to matter
            decimals = INT_MAX;
        }
        else {
            decimals = -decimals;
        }
    }
    if (!out) {
        if (PyArray_ISINTEGER(a)) {
            ret_int = 1;
            my_descr = PyArray_DescrFromType(NPY_DOUBLE);
        }
        else {
            Py_INCREF(PyArray_DESCR(a));
            my_descr = PyArray_DESCR(a);
        }
        out = (PyArrayObject *)PyArray_Empty(PyArray_NDIM(a), PyArray_DIMS(a),
                                             my_descr,
                                             PyArray_ISFORTRAN(a));
        if (out == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(out);
    }
    f = PyFloat_FromDouble(power_of_ten(decimals));
    if (f == NULL) {
        return NULL;
    }
    ret = PyObject_CallFunction(op1, "OOO", a, f, out);
    if (ret == NULL) {
        goto finish;
    }
    tmp = PyObject_CallFunction(n_ops.rint, "OO", ret, ret);
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    Py_DECREF(tmp);
    tmp = PyObject_CallFunction(op2, "OOO", ret, f, ret);
    if (tmp == NULL) {
        Py_DECREF(ret);
        ret = NULL;
        goto finish;
    }
    Py_DECREF(tmp);

 finish:
    Py_DECREF(f);
    Py_DECREF(out);
    if (ret_int && ret != NULL) {
        Py_INCREF(PyArray_DESCR(a));
        tmp = PyArray_CastToType((PyArrayObject *)ret,
                                 PyArray_DESCR(a), PyArray_ISFORTRAN(a));
        Py_DECREF(ret);
        return tmp;
    }
    return ret;
}


/*NUMPY_API
 * Mean
 */
NPY_NO_EXPORT PyObject *
PyArray_Mean(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *obj1 = NULL, *obj2 = NULL, *ret;
    PyArrayObject *arr;

    arr = (PyArrayObject *)PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    obj1 = PyArray_GenericReduceFunction(arr, n_ops.add, axis,
                                         rtype, out);
    obj2 = PyFloat_FromDouble((double)PyArray_DIM(arr,axis));
    Py_DECREF(arr);
    if (obj1 == NULL || obj2 == NULL) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    if (!out) {
        ret = PyNumber_TrueDivide(obj1, obj2);
    }
    else {
        ret = PyObject_CallFunction(n_ops.divide, "OOO", out, obj2, out);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;
}

/*NUMPY_API
 * Any
 */
NPY_NO_EXPORT PyObject *
PyArray_Any(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr,
                                        n_ops.logical_or, axis,
                                        NPY_BOOL, out);
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * All
 */
NPY_NO_EXPORT PyObject *
PyArray_All(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *arr, *ret;

    arr = PyArray_CheckAxis(self, &axis, 0);
    if (arr == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyArrayObject *)arr,
                                        n_ops.logical_and, axis,
                                        NPY_BOOL, out);
    Py_DECREF(arr);
    return ret;
}


/*NUMPY_API
 * Clip
 */
NPY_NO_EXPORT PyObject *
PyArray_Clip(PyArrayObject *self, PyObject *min, PyObject *max, PyArrayObject *out)
{
    /* Treat None the same as NULL */
    if (min == Py_None) {
        min = NULL;
    }
    if (max == Py_None) {
        max = NULL;
    }

    if ((max == NULL) && (min == NULL)) {
        PyErr_SetString(PyExc_ValueError,
                        "array_clip: must set either max or min");
        return NULL;
    }

    if (min == NULL) {
        return PyObject_CallFunctionObjArgs(n_ops.minimum, self, max, out, NULL);
    }
    else if (max == NULL) {
        return PyObject_CallFunctionObjArgs(n_ops.maximum, self, min, out, NULL);
    }
    else {
        return PyObject_CallFunctionObjArgs(n_ops.clip, self, min, max, out, NULL);
    }
}


/*NUMPY_API
 * Conjugate
 */
NPY_NO_EXPORT PyObject *
PyArray_Conjugate(PyArrayObject *self, PyArrayObject *out)
{
    if (PyArray_ISCOMPLEX(self) || PyArray_ISOBJECT(self) ||
            PyArray_ISUSERDEF(self)) {
        if (out == NULL) {
            return PyArray_GenericUnaryFunction(self,
                                                n_ops.conjugate);
        }
        else {
            return PyArray_GenericBinaryFunction((PyObject *)self,
                                                 (PyObject *)out,
                                                 n_ops.conjugate);
        }
    }
    else {
        PyArrayObject *ret;
        if (!PyArray_ISNUMBER(self)) {
            PyErr_SetString(PyExc_TypeError,
                "cannot conjugate non-numeric dtype");
            return NULL;
        }
        if (out) {
            if (PyArray_AssignArray(out, self,
                        NULL, NPY_DEFAULT_ASSIGN_CASTING) < 0) {
                return NULL;
            }
            ret = out;
        }
        else {
            ret = self;
        }
        Py_INCREF(ret);
        return (PyObject *)ret;
    }
}

/*NUMPY_API
 * Trace
 */
NPY_NO_EXPORT PyObject *
PyArray_Trace(PyArrayObject *self, int offset, int axis1, int axis2,
              int rtype, PyArrayObject *out)
{
    PyObject *diag = NULL, *ret = NULL;

    diag = PyArray_Diagonal(self, offset, axis1, axis2);
    if (diag == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyArrayObject *)diag, n_ops.add, -1, rtype, out);
    Py_DECREF(diag);
    return ret;
}
