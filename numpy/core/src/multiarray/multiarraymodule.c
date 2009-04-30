/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/

/* $Id: multiarraymodule.c,v 1.36 2005/09/14 00:14:00 teoliphant Exp $ */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "global.c"

#define PyAO PyArrayObject

/* Internal APIs */
#include "arrayobject.h"
#include "hashdescr.h"
#include "descriptor.h"

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj)
{
    PyObject *dtypedescr;
    PyArray_Descr *new;
    int ret;

    dtypedescr = PyObject_GetAttrString(obj, "dtype");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    /* Understand basic ctypes */
    dtypedescr = PyObject_GetAttrString(obj, "_type_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            PyObject *length;
            length = PyObject_GetAttrString(obj, "_length_");
            PyErr_Clear();
            if (length) {
                /* derived type */
                PyObject *newtup;
                PyArray_Descr *derived;
                newtup = Py_BuildValue("NO", new, length);
                ret = PyArray_DescrConverter(newtup, &derived);
                Py_DECREF(newtup);
                if (ret == PY_SUCCEED) {
                    return derived;
                }
                PyErr_Clear();
                return NULL;
            }
            return new;
        }
        PyErr_Clear();
        return NULL;
    }
    /* Understand ctypes structures --
       bit-fields are not supported
       automatically aligns */
    dtypedescr = PyObject_GetAttrString(obj, "_fields_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrAlignConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    return NULL;
}

/*
 * Including this file is the only way I know how to declare functions
 * static in each file, and store the pointers from functions in both
 * arrayobject.c and multiarraymodule.c for the C-API
 *
 * Declarying an external pointer-containing variable in arrayobject.c
 * and trying to copy it to PyArray_API, did not work.
 *
 * Think about two modules with a common api that import each other...
 *
 * This file would just be the module calls.
 */

//#include "arrayobject.c"


/* An Error object -- rarely used? */
static PyObject *MultiArrayError;

/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int *l1, int n)
{
    int s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT intp
PyArray_MultiplyList(intp *l1, int n)
{
    intp s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT intp
PyArray_OverflowMultiplyList(intp *l1, int n)
{
    intp prod = 1;
    intp imax = NPY_MAX_INTP;
    int i;

    for (i = 0; i < n; i++) {
        intp dim = l1[i];

	if (dim == 0) {
            return 0;
        }
	if (dim > imax) {
	    return -1;
        }
        imax /= dim;
	prod *= dim;
    }
    return prod;
}

/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, intp* ind)
{
    int n = obj->nd;
    intp *strides = obj->strides;
    char *dptr = obj->data;

    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;
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
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(intp *l1, intp *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

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
        PyObject *new;
        int res;

        if (out) {
            new = (PyObject *)out;
            Py_INCREF(new);
        }
        else {
            new = PyArray_Copy(a);
            if (new == NULL) {
                return NULL;
            }
        }

        /* new.real = a.real.round(decimals) */
        part = PyObject_GetAttrString(new, "real");
        if (part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        res = PyObject_SetAttrString(new, "real", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(new);
            return NULL;
        }

        /* new.imag = a.imag.round(decimals) */
        part = PyObject_GetAttrString(new, "imag");
        if (part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        part = PyArray_EnsureAnyArray(part);
        round_part = PyArray_Round((PyArrayObject *)part,
                                   decimals, NULL);
        Py_DECREF(part);
        if (round_part == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        res = PyObject_SetAttrString(new, "imag", round_part);
        Py_DECREF(round_part);
        if (res < 0) {
            Py_DECREF(new);
            return NULL;
        }
        return new;
    }
    /* do the most common case first */
    if (decimals >= 0) {
        if (PyArray_ISINTEGER(a)) {
            if (out) {
                if (PyArray_CopyAnyInto(out, a) < 0) {
                    return NULL;
                }
                Py_INCREF(out);
                return (PyObject *)out;
            }
            else {
                Py_INCREF(a);
                return (PyObject *)a;
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
        decimals = -decimals;
    }
    if (!out) {
        if (PyArray_ISINTEGER(a)) {
            ret_int = 1;
            my_descr = PyArray_DescrFromType(NPY_DOUBLE);
        }
        else {
            Py_INCREF(a->descr);
            my_descr = a->descr;
        }
        out = (PyArrayObject *)PyArray_Empty(a->nd, a->dimensions,
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
    if (ret_int) {
        Py_INCREF(a->descr);
        tmp = PyArray_CastToType((PyArrayObject *)ret,
                                 a->descr, PyArray_ISFORTRAN(a));
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
    PyObject *obj1 = NULL, *obj2 = NULL;
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    obj1 = PyArray_GenericReduceFunction((PyAO *)new, n_ops.add, axis,
                                         rtype, out);
    obj2 = PyFloat_FromDouble((double) PyArray_DIM(new,axis));
    Py_DECREF(new);
    if (obj1 == NULL || obj2 == NULL) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    if (!out) {
        ret = PyNumber_Divide(obj1, obj2);
    }
    else {
        ret = PyObject_CallFunction(n_ops.divide, "OOO", out, obj2, out);
    }
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    return ret;
}

/*NUMPY_API
 * Set variance to 1 to by-pass square-root calculation and return variance
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
    PyObject *obj1 = NULL, *obj2 = NULL, *obj3 = NULL, *new = NULL;
    PyObject *ret = NULL, *newshape = NULL;
    int i, n;
    intp val;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Compute and reshape mean */
    obj1 = PyArray_EnsureAnyArray(PyArray_Mean((PyAO *)new, axis, rtype, NULL));
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    n = PyArray_NDIM(new);
    newshape = PyTuple_New(n);
    if (newshape == NULL) {
        Py_DECREF(obj1);
        Py_DECREF(new);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (i == axis) {
            val = 1;
        }
        else {
            val = PyArray_DIM(new,i);
        }
        PyTuple_SET_ITEM(newshape, i, PyInt_FromLong((long)val));
    }
    obj2 = PyArray_Reshape((PyAO *)obj1, newshape);
    Py_DECREF(obj1);
    Py_DECREF(newshape);
    if (obj2 == NULL) {
        Py_DECREF(new);
        return NULL;
    }

    /* Compute x = x - mx */
    obj1 = PyArray_EnsureAnyArray(PyNumber_Subtract((PyObject *)new, obj2));
    Py_DECREF(obj2);
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    /* Compute x * x */
    if (PyArray_ISCOMPLEX(obj1)) {
	obj3 = PyArray_Conjugate((PyAO *)obj1, NULL);
    }
    else {
	obj3 = obj1;
	Py_INCREF(obj1);
    }
    if (obj3 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    obj2 = PyArray_EnsureAnyArray                                      \
        (PyArray_GenericBinaryFunction((PyAO *)obj1, obj3, n_ops.multiply));
    Py_DECREF(obj1);
    Py_DECREF(obj3);
    if (obj2 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    if (PyArray_ISCOMPLEX(obj2)) {
	obj3 = PyObject_GetAttrString(obj2, "real");
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
	obj3 = obj2;
	Py_INCREF(obj2);
    }
    if (obj3 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    /* Compute add.reduce(x*x,axis) */
    obj1 = PyArray_GenericReduceFunction((PyAO *)obj3, n_ops.add,
                                         axis, rtype, NULL);
    Py_DECREF(obj3);
    Py_DECREF(obj2);
    if (obj1 == NULL) {
        Py_DECREF(new);
        return NULL;
    }
    n = PyArray_DIM(new,axis);
    Py_DECREF(new);
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
        obj1 = PyArray_EnsureAnyArray(ret);
        /* sqrt() */
        ret = PyArray_GenericUnaryFunction((PyAO *)obj1, n_ops.sqrt);
        Py_DECREF(obj1);
    }
    if (ret == NULL || PyArray_CheckExact(self)) {
        return ret;
    }
    if (PyArray_Check(self) && self->ob_type == ret->ob_type) {
        return ret;
    }
    obj1 = PyArray_EnsureArray(ret);
    if (obj1 == NULL) {
        return NULL;
    }
    ret = PyArray_View((PyAO *)obj1, NULL, self->ob_type);
    Py_DECREF(obj1);
    if (out) {
        if (PyArray_CopyAnyInto(out, (PyArrayObject *)ret) < 0) {
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
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new, n_ops.add, axis,
                                        rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * Prod
 */
NPY_NO_EXPORT PyObject *
PyArray_Prod(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new, n_ops.multiply, axis,
                                        rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 *CumSum
 */
NPY_NO_EXPORT PyObject *
PyArray_CumSum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericAccumulateFunction((PyAO *)new, n_ops.add, axis,
                                            rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * CumProd
 */
NPY_NO_EXPORT PyObject *
PyArray_CumProd(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }

    ret = PyArray_GenericAccumulateFunction((PyAO *)new,
                                            n_ops.multiply, axis,
                                            rtype, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * Any
 */
NPY_NO_EXPORT PyObject *
PyArray_Any(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new,
                                        n_ops.logical_or, axis,
                                        PyArray_BOOL, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * All
 */
NPY_NO_EXPORT PyObject *
PyArray_All(PyArrayObject *self, int axis, PyArrayObject *out)
{
    PyObject *new, *ret;

    if ((new = _check_axis(self, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction((PyAO *)new,
                                        n_ops.logical_and, axis,
                                        PyArray_BOOL, out);
    Py_DECREF(new);
    return ret;
}


static PyObject *
_GenericBinaryOutFunction(PyArrayObject *m1, PyObject *m2, PyArrayObject *out,
			  PyObject *op)
{
    if (out == NULL) {
	return PyObject_CallFunction(op, "OO", m1, m2);
    }
    else {
	return PyObject_CallFunction(op, "OOO", m1, m2, out);
    }
}

static PyObject *
_slow_array_clip(PyArrayObject *self, PyObject *min, PyObject *max, PyArrayObject *out)
{
    PyObject *res1=NULL, *res2=NULL;

    if (max != NULL) {
	res1 = _GenericBinaryOutFunction(self, max, out, n_ops.minimum);
	if (res1 == NULL) {
            return NULL;
        }
    }
    else {
	res1 = (PyObject *)self;
	Py_INCREF(res1);
    }

    if (min != NULL) {
	res2 = _GenericBinaryOutFunction((PyArrayObject *)res1,
					 min, out, n_ops.maximum);
	if (res2 == NULL) {
            Py_XDECREF(res1);
            return NULL;
        }
    }
    else {
	res2 = res1;
	Py_INCREF(res2);
    }
    Py_DECREF(res1);
    return res2;
}

/*NUMPY_API
 * Clip
 */
NPY_NO_EXPORT PyObject *
PyArray_Clip(PyArrayObject *self, PyObject *min, PyObject *max, PyArrayObject *out)
{
    PyArray_FastClipFunc *func;
    int outgood = 0, ingood = 0;
    PyArrayObject *maxa = NULL;
    PyArrayObject *mina = NULL;
    PyArrayObject *newout = NULL, *newin = NULL;
    PyArray_Descr *indescr, *newdescr;
    char *max_data, *min_data;
    PyObject *zero;

    if ((max == NULL) && (min == NULL)) {
	PyErr_SetString(PyExc_ValueError, "array_clip: must set either max "\
			"or min");
	return NULL;
    }

    func = self->descr->f->fastclip;
    if (func == NULL || (min != NULL && !PyArray_CheckAnyScalar(min)) ||
        (max != NULL && !PyArray_CheckAnyScalar(max))) {
        return _slow_array_clip(self, min, max, out);
    }
    /* Use the fast scalar clip function */

    /* First we need to figure out the correct type */
    indescr = NULL;
    if (min != NULL) {
	indescr = PyArray_DescrFromObject(min, NULL);
	if (indescr == NULL) {
            return NULL;
        }
    }
    if (max != NULL) {
	newdescr = PyArray_DescrFromObject(max, indescr);
	Py_XDECREF(indescr);
	if (newdescr == NULL) {
            return NULL;
        }
    }
    else {
        /* Steal the reference */
	newdescr = indescr;
    }


    /*
     * Use the scalar descriptor only if it is of a bigger
     * KIND than the input array (and then find the
     * type that matches both).
     */
    if (PyArray_ScalarKind(newdescr->type_num, NULL) >
        PyArray_ScalarKind(self->descr->type_num, NULL)) {
        indescr = _array_small_type(newdescr, self->descr);
        func = indescr->f->fastclip;
        if (func == NULL) {
            return _slow_array_clip(self, min, max, out);
        }
    }
    else {
        indescr = self->descr;
        Py_INCREF(indescr);
    }
    Py_DECREF(newdescr);

    if (!PyDataType_ISNOTSWAPPED(indescr)) {
        PyArray_Descr *descr2;
        descr2 = PyArray_DescrNewByteorder(indescr, '=');
        Py_DECREF(indescr);
        if (descr2 == NULL) {
            goto fail;
        }
        indescr = descr2;
    }

    /* Convert max to an array */
    if (max != NULL) {
	maxa = (NPY_AO *)PyArray_FromAny(max, indescr, 0, 0,
					 NPY_DEFAULT, NULL);
	if (maxa == NULL) {
            return NULL;
        }
    }
    else {
	/* Side-effect of PyArray_FromAny */
	Py_DECREF(indescr);
    }

    /*
     * If we are unsigned, then make sure min is not < 0
     * This is to match the behavior of _slow_array_clip
     *
     * We allow min and max to go beyond the limits
     * for other data-types in which case they
     * are interpreted as their modular counterparts.
    */
    if (min != NULL) {
	if (PyArray_ISUNSIGNED(self)) {
	    int cmp;
	    zero = PyInt_FromLong(0);
	    cmp = PyObject_RichCompareBool(min, zero, Py_LT);
	    if (cmp == -1) {
                Py_DECREF(zero);
                goto fail;
            }
	    if (cmp == 1) {
		min = zero;
	    }
	    else {
		Py_DECREF(zero);
		Py_INCREF(min);
	    }
	}
	else {
	    Py_INCREF(min);
	}

	/* Convert min to an array */
	Py_INCREF(indescr);
	mina = (NPY_AO *)PyArray_FromAny(min, indescr, 0, 0,
					 NPY_DEFAULT, NULL);
	Py_DECREF(min);
	if (mina == NULL) {
            goto fail;
        }
    }


    /*
     * Check to see if input is single-segment, aligned,
     * and in native byteorder
     */
    if (PyArray_ISONESEGMENT(self) && PyArray_CHKFLAGS(self, ALIGNED) &&
        PyArray_ISNOTSWAPPED(self) && (self->descr == indescr)) {
        ingood = 1;
    }
    if (!ingood) {
        int flags;

        if (PyArray_ISFORTRAN(self)) {
            flags = NPY_FARRAY;
        }
        else {
            flags = NPY_CARRAY;
        }
        Py_INCREF(indescr);
        newin = (NPY_AO *)PyArray_FromArray(self, indescr, flags);
        if (newin == NULL) {
            goto fail;
        }
    }
    else {
        newin = self;
        Py_INCREF(newin);
    }

    /*
     * At this point, newin is a single-segment, aligned, and correct
     * byte-order array of the correct type
     *
     * if ingood == 0, then it is a copy, otherwise,
     * it is the original input.
     */

    /*
     * If we have already made a copy of the data, then use
     * that as the output array
     */
    if (out == NULL && !ingood) {
        out = newin;
    }

    /*
     * Now, we know newin is a usable array for fastclip,
     * we need to make sure the output array is available
     * and usable
     */
    if (out == NULL) {
        Py_INCREF(indescr);
        out = (NPY_AO*)PyArray_NewFromDescr(self->ob_type,
                                            indescr, self->nd,
                                            self->dimensions,
                                            NULL, NULL,
                                            PyArray_ISFORTRAN(self),
                                            (PyObject *)self);
        if (out == NULL) {
            goto fail;
        }
        outgood = 1;
    }
    else Py_INCREF(out);
    /* Input is good at this point */
    if (out == newin) {
        outgood = 1;
    }
    if (!outgood && PyArray_ISONESEGMENT(out) &&
        PyArray_CHKFLAGS(out, ALIGNED) && PyArray_ISNOTSWAPPED(out) &&
        PyArray_EquivTypes(out->descr, indescr)) {
        outgood = 1;
    }

    /*
     * Do we still not have a suitable output array?
     * Create one, now
     */
    if (!outgood) {
        int oflags;
        if (PyArray_ISFORTRAN(out))
            oflags = NPY_FARRAY;
        else
            oflags = NPY_CARRAY;
        oflags |= NPY_UPDATEIFCOPY | NPY_FORCECAST;
        Py_INCREF(indescr);
        newout = (NPY_AO*)PyArray_FromArray(out, indescr, oflags);
        if (newout == NULL) {
            goto fail;
        }
    }
    else {
        newout = out;
        Py_INCREF(newout);
    }

    /* make sure the shape of the output array is the same */
    if (!PyArray_SAMESHAPE(newin, newout)) {
        PyErr_SetString(PyExc_ValueError, "clip: Output array must have the"
                        "same shape as the input.");
        goto fail;
    }
    if (newout->data != newin->data) {
        memcpy(newout->data, newin->data, PyArray_NBYTES(newin));
    }

    /* Now we can call the fast-clip function */
    min_data = max_data = NULL;
    if (mina != NULL) {
	min_data = mina->data;
    }
    if (maxa != NULL) {
	max_data = maxa->data;
    }
    func(newin->data, PyArray_SIZE(newin), min_data, max_data, newout->data);

    /* Clean up temporary variables */
    Py_XDECREF(mina);
    Py_XDECREF(maxa);
    Py_DECREF(newin);
    /* Copy back into out if out was not already a nice array. */
    Py_DECREF(newout);
    return (PyObject *)out;

 fail:
    Py_XDECREF(maxa);
    Py_XDECREF(mina);
    Py_XDECREF(newin);
    PyArray_XDECREF_ERR(newout);
    return NULL;
}


/*NUMPY_API
 * Conjugate
 */
NPY_NO_EXPORT PyObject *
PyArray_Conjugate(PyArrayObject *self, PyArrayObject *out)
{
    if (PyArray_ISCOMPLEX(self)) {
        if (out == NULL) {
            return PyArray_GenericUnaryFunction(self,
                                                n_ops.conjugate);
        }
        else {
            return PyArray_GenericBinaryFunction(self,
                                                 (PyObject *)out,
                                                 n_ops.conjugate);
        }
    }
    else {
        PyArrayObject *ret;
        if (out) {
            if (PyArray_CopyAnyInto(out, self) < 0) {
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
    ret = PyArray_GenericReduceFunction((PyAO *)diag, n_ops.add, -1, rtype, out);
    Py_DECREF(diag);
    return ret;
}

/*
 * simulates a C-style 1-3 dimensional array which can be accesed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap;
    intp n, m, i, j;
    char **ptr2;
    char ***ptr3;

    if ((nd < 1) || (nd > 3)) {
        PyErr_SetString(PyExc_ValueError,
                        "C arrays of only 1-3 dimensions available");
        Py_XDECREF(typedescr);
        return -1;
    }
    if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
                                              CARRAY, NULL)) == NULL) {
        return -1;
    }
    switch(nd) {
    case 1:
        *((char **)ptr) = ap->data;
        break;
    case 2:
        n = ap->dimensions[0];
        ptr2 = (char **)_pya_malloc(n * sizeof(char *));
        if (!ptr2) {
            goto fail;
        }
        for (i = 0; i < n; i++) {
            ptr2[i] = ap->data + i*ap->strides[0];
        }
        *((char ***)ptr) = ptr2;
        break;
    case 3:
        n = ap->dimensions[0];
        m = ap->dimensions[1];
        ptr3 = (char ***)_pya_malloc(n*(m+1) * sizeof(char *));
        if (!ptr3) {
            goto fail;
        }
        for (i = 0; i < n; i++) {
            ptr3[i] = ptr3[n + (m-1)*i];
            for (j = 0; j < m; j++) {
                ptr3[i][j] = ap->data + i*ap->strides[0] + j*ap->strides[1];
            }
        }
        *((char ****)ptr) = ptr3;
    }
    memcpy(dims, ap->dimensions, nd*sizeof(intp));
    *op = (PyObject *)ap;
    return 0;

 fail:
    PyErr_SetString(PyExc_MemoryError, "no memory");
    return -1;
}

/* Deprecated --- Use PyArray_AsCArray instead */

/*NUMPY_API
 * Convert to a 1D C-array
 */
NPY_NO_EXPORT int
PyArray_As1D(PyObject **op, char **ptr, int *d1, int typecode)
{
    intp newd1;
    PyArray_Descr *descr;
    char msg[] = "PyArray_As1D: use PyArray_AsCArray.";

    if (DEPRECATE(msg) < 0) {
        return -1;
    }
    descr = PyArray_DescrFromType(typecode);
    if (PyArray_AsCArray(op, (void *)ptr, &newd1, 1, descr) == -1) {
        return -1;
    }
    *d1 = (int) newd1;
    return 0;
}

/*NUMPY_API
 * Convert to a 2D C-array
 */
NPY_NO_EXPORT int
PyArray_As2D(PyObject **op, char ***ptr, int *d1, int *d2, int typecode)
{
    intp newdims[2];
    PyArray_Descr *descr;
    char msg[] = "PyArray_As1D: use PyArray_AsCArray.";

    if (DEPRECATE(msg) < 0) {
        return -1;
    }
    descr = PyArray_DescrFromType(typecode);
    if (PyArray_AsCArray(op, (void *)ptr, newdims, 2, descr) == -1) {
        return -1;
    }
    *d1 = (int ) newdims[0];
    *d2 = (int ) newdims[1];
    return 0;
}

/* End Deprecated */

/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    PyArrayObject *ap = (PyArrayObject *)op;

    if ((ap->nd < 1) || (ap->nd > 3)) {
        return -1;
    }
    if (ap->nd >= 2) {
        _pya_free(ptr);
    }
    Py_DECREF(ap);
    return 0;
}


static PyObject *
_swap_and_concat(PyObject *op, int axis, int n)
{
    PyObject *newtup = NULL;
    PyObject *otmp, *arr;
    int i;

    newtup = PyTuple_New(n);
    if (newtup == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        otmp = PySequence_GetItem(op, i);
        arr = PyArray_FROM_O(otmp);
        Py_DECREF(otmp);
        if (arr == NULL) {
            goto fail;
        }
        otmp = PyArray_SwapAxes((PyArrayObject *)arr, axis, 0);
        Py_DECREF(arr);
        if (otmp == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(newtup, i, otmp);
    }
    otmp = PyArray_Concatenate(newtup, 0);
    Py_DECREF(newtup);
    if (otmp == NULL) {
        return NULL;
    }
    arr = PyArray_SwapAxes((PyArrayObject *)otmp, axis, 0);
    Py_DECREF(otmp);
    return arr;

 fail:
    Py_DECREF(newtup);
    return NULL;
}

/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is MAX_DIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    PyArrayObject *ret, **mps;
    PyObject *otmp;
    int i, n, tmp, nd = 0, new_dim;
    char *data;
    PyTypeObject *subtype;
    double prior1, prior2;
    intp numbytes;

    n = PySequence_Length(op);
    if (n == -1) {
        return NULL;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "concatenation of zero-length sequences is "\
                        "impossible");
        return NULL;
    }

    if ((axis < 0) || ((0 < axis) && (axis < MAX_DIMS))) {
        return _swap_and_concat(op, axis, n);
    }
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }

    /*
     * Make sure these arrays are legal to concatenate.
     * Must have same dimensions except d0
     */
    prior1 = PyArray_PRIORITY;
    subtype = &PyArray_Type;
    ret = NULL;
    for (i = 0; i < n; i++) {
        if (axis >= MAX_DIMS) {
            otmp = PyArray_Ravel(mps[i],0);
            Py_DECREF(mps[i]);
            mps[i] = (PyArrayObject *)otmp;
        }
        if (mps[i]->ob_type != subtype) {
            prior2 = PyArray_GetPriority((PyObject *)(mps[i]), 0.0);
            if (prior2 > prior1) {
                prior1 = prior2;
                subtype = mps[i]->ob_type;
            }
        }
    }

    new_dim = 0;
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i == 0) {
            nd = mps[i]->nd;
        }
        else {
            if (nd != mps[i]->nd) {
                PyErr_SetString(PyExc_ValueError,
                                "arrays must have same "\
                                "number of dimensions");
                goto fail;
            }
            if (!PyArray_CompareLists(mps[0]->dimensions+1,
                                      mps[i]->dimensions+1,
                                      nd-1)) {
                PyErr_SetString(PyExc_ValueError,
                                "array dimensions must "\
                                "agree except for d_0");
                goto fail;
            }
        }
        if (nd == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "0-d arrays can't be concatenated");
            goto fail;
        }
        new_dim += mps[i]->dimensions[0];
    }
    tmp = mps[0]->dimensions[0];
    mps[0]->dimensions[0] = new_dim;
    Py_INCREF(mps[0]->descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(subtype,
                                                mps[0]->descr, nd,
                                                mps[0]->dimensions,
                                                NULL, NULL, 0,
                                                (PyObject *)ret);
    mps[0]->dimensions[0] = tmp;

    if (ret == NULL) {
        goto fail;
    }
    data = ret->data;
    for (i = 0; i < n; i++) {
        numbytes = PyArray_NBYTES(mps[i]);
        memcpy(data, mps[i]->data, numbytes);
        data += numbytes;
    }

    PyArray_INCREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the byte to test */
    char byteorder;
    int elsize;

    elsize = arr->descr->elsize;
    byteorder = arr->descr->byteorder;
    ptr = arr->data;
    if (elsize > 1 &&
        (byteorder == PyArray_LITTLE ||
         (byteorder == PyArray_NATIVE &&
          PyArray_ISNBO(PyArray_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}


/*NUMPY_API
 * ScalarKind
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    if (PyTypeNum_ISSIGNED(typenum)) {
        if (arr && _signbit_set(*arr)) {
            return PyArray_INTNEG_SCALAR;
        }
        else {
            return PyArray_INTPOS_SCALAR;
        }
    }
    if (PyTypeNum_ISFLOAT(typenum)) {
        return PyArray_FLOAT_SCALAR;
    }
    if (PyTypeNum_ISUNSIGNED(typenum)) {
        return PyArray_INTPOS_SCALAR;
    }
    if (PyTypeNum_ISCOMPLEX(typenum)) {
        return PyArray_COMPLEX_SCALAR;
    }
    if (PyTypeNum_ISBOOL(typenum)) {
        return PyArray_BOOL_SCALAR;
    }

    if (PyTypeNum_ISUSERDEF(typenum)) {
        NPY_SCALARKIND retval;
        PyArray_Descr* descr = PyArray_DescrFromType(typenum);

        if (descr->f->scalarkind) {
            retval = descr->f->scalarkind((arr ? *arr : NULL));
        }
        else {
            retval = PyArray_NOSCALAR;
        }
        Py_DECREF(descr);
        return retval;
    }
    return PyArray_OBJECT_SCALAR;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    PyArray_Descr* from;
    int *castlist;

    if (scalar == PyArray_NOSCALAR) {
        return PyArray_CanCastSafely(thistype, neededtype);
    }
    from = PyArray_DescrFromType(thistype);
    if (from->f->cancastscalarkindto
        && (castlist = from->f->cancastscalarkindto[scalar])) {
        while (*castlist != PyArray_NOTYPE) {
            if (*castlist++ == neededtype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }
    Py_DECREF(from);

    switch(scalar) {
    case PyArray_BOOL_SCALAR:
    case PyArray_OBJECT_SCALAR:
        return PyArray_CanCastSafely(thistype, neededtype);
    default:
        if (PyTypeNum_ISUSERDEF(neededtype)) {
            return FALSE;
        }
        switch(scalar) {
        case PyArray_INTPOS_SCALAR:
            return (neededtype >= PyArray_BYTE);
        case PyArray_INTNEG_SCALAR:
            return (neededtype >= PyArray_BYTE)
                && !(PyTypeNum_ISUNSIGNED(neededtype));
        case PyArray_FLOAT_SCALAR:
            return (neededtype >= PyArray_FLOAT);
        case PyArray_COMPLEX_SCALAR:
            return (neededtype >= PyArray_CFLOAT);
        default:
            /* should never get here... */
            return 1;
        }
    }
}

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 */
static PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2,
                  int nd, intp dimensions[], int typenum)
{
    PyArrayObject *ret;
    PyTypeObject *subtype;
    double prior1, prior2;
    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    if (ap2->ob_type != ap1->ob_type) {
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);
    }
    else {
        prior1 = prior2 = 0.0;
        subtype = ap1->ob_type;
    }

    ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                       typenum, NULL, NULL, 0, 0,
                                       (PyObject *)
                                       (prior2 > prior1 ? ap2 : ap1));
    return ret;
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    PyArrayIterObject *it1, *it2;
    intp i, j, l;
    int typenum, nd, axis;
    intp is1, is2, os;
    char *op;
    intp dimensions[MAX_DIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0, BEHAVED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0, BEHAVED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    if (ap1->nd == 0 || ap2->nd == 0) {
        ret = (ap1->nd == 0 ? ap1 : ap2);
        ret = (PyArrayObject *)ret->ob_type->tp_as_number->nb_multiply(
                                            (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)ret;
    }

    l = ap1->dimensions[ap1->nd - 1];
    if (ap2->dimensions[ap2->nd - 1] != l) {
        PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
        goto fail;
    }

    nd = ap1->nd + ap2->nd - 2;
    j = 0;
    for (i = 0; i < ap1->nd - 1; i++) {
        dimensions[j++] = ap1->dimensions[i];
    }
    for (i = 0; i < ap2->nd - 1; i++) {
        dimensions[j++] = ap2->dimensions[i];
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
    if (ret == NULL) {
        goto fail;
    }
    dot = (ret->descr->f->dotfunc);
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }
    is1 = ap1->strides[ap1->nd - 1];
    is2 = ap2->strides[ap2->nd - 1];
    op = ret->data; os = ret->descr->elsize;
    axis = ap1->nd - 1;
    it1 = (PyArrayIterObject *) PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    axis = ap2->nd - 1;
    it2 = (PyArrayIterObject *) PyArray_IterAllButAxis((PyObject *)ap2, &axis);
    NPY_BEGIN_THREADS_DESCR(ap2->descr);
    while (1) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, ret);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        if (it1->index >= it1->size) {
            break;
        }
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(ap2->descr);
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 *Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    PyArrayIterObject *it1, *it2;
    intp i, j, l;
    int typenum, nd, axis, matchDim;
    intp is1, is2, os;
    char *op;
    intp dimensions[MAX_DIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0, BEHAVED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0, BEHAVED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    if (ap1->nd == 0 || ap2->nd == 0) {
        ret = (ap1->nd == 0 ? ap1 : ap2);
        ret = (PyArrayObject *)ret->ob_type->tp_as_number->nb_multiply(
                                        (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)ret;
    }
    l = ap1->dimensions[ap1->nd - 1];
    if (ap2->nd > 1) {
        matchDim = ap2->nd - 2;
    }
    else {
        matchDim = 0;
    }
    if (ap2->dimensions[matchDim] != l) {
        PyErr_SetString(PyExc_ValueError, "objects are not aligned");
        goto fail;
    }
    nd = ap1->nd + ap2->nd - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < ap1->nd - 1; i++) {
        dimensions[j++] = ap1->dimensions[i];
    }
    for (i = 0; i < ap2->nd - 2; i++) {
        dimensions[j++] = ap2->dimensions[i];
    }
    if(ap2->nd > 1) {
        dimensions[j++] = ap2->dimensions[ap2->nd-1];
    }
    /*
      fprintf(stderr, "nd=%d dimensions=", nd);
      for(i=0; i<j; i++)
      fprintf(stderr, "%d ", dimensions[i]);
      fprintf(stderr, "\n");
    */

    is1 = ap1->strides[ap1->nd-1]; is2 = ap2->strides[matchDim];
    /* Choose which subtype to return */
    ret = new_array_for_sum(ap1, ap2, nd, dimensions, typenum);
    if (ret == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
        memset(PyArray_DATA(ret), 0, PyArray_NBYTES(ret));
    }
    else {
        /* Ensure that multiarray.dot([],[]) -> 0 */
        memset(PyArray_DATA(ret), 0, PyArray_ITEMSIZE(ret));
    }

    dot = ret->descr->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    op = ret->data; os = ret->descr->elsize;
    axis = ap1->nd-1;
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
    NPY_BEGIN_THREADS_DESCR(ap2->descr);
    while (1) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, ret);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        if (it1->index >= it1->size) {
            break;
        }
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(ap2->descr);
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 * Fast Copy and Transpose
 */
NPY_NO_EXPORT PyObject *
PyArray_CopyAndTranspose(PyObject *op)
{
    PyObject *ret, *arr;
    int nd;
    intp dims[2];
    intp i,j;
    int elsize, str2;
    char *iptr;
    char *optr;

    /* make sure it is well-behaved */
    arr = PyArray_FromAny(op, NULL, 0, 0, CARRAY, NULL);
    if (arr == NULL) {
        return NULL;
    }
    nd = PyArray_NDIM(arr);
    if (nd == 1) {
        /* we will give in to old behavior */
        ret = PyArray_Copy((PyArrayObject *)arr);
        Py_DECREF(arr);
        return ret;
    }
    else if (nd != 2) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError,
                        "only 2-d arrays are allowed");
        return NULL;
    }

    /* Now construct output array */
    dims[0] = PyArray_DIM(arr,1);
    dims[1] = PyArray_DIM(arr,0);
    elsize = PyArray_ITEMSIZE(arr);
    Py_INCREF(PyArray_DESCR(arr));
    ret = PyArray_NewFromDescr(arr->ob_type,
                               PyArray_DESCR(arr),
                               2, dims,
                               NULL, NULL, 0, arr);
    if (ret == NULL) {
        Py_DECREF(arr);
        return NULL;
    }

    /* do 2-d loop */
    NPY_BEGIN_ALLOW_THREADS;
    optr = PyArray_DATA(ret);
    str2 = elsize*dims[0];
    for (i = 0; i < dims[0]; i++) {
        iptr = PyArray_BYTES(arr) + i*elsize;
        for (j = 0; j < dims[1]; j++) {
            /* optr[i,j] = iptr[j,i] */
            memcpy(optr, iptr, elsize);
            optr += elsize;
            iptr += str2;
        }
    }
    NPY_END_ALLOW_THREADS;
    Py_DECREF(arr);
    return ret;
}

/*NUMPY_API
 * Numeric.correlate(a1,a2,mode)
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    intp length;
    intp i, n1, n2, n, n_left, n_right;
    int typenum;
    intp is1, is2, os;
    char *ip1, *ip2, *op;
    PyArray_DotFunc *dot;
    PyArray_Descr *typec;

    NPY_BEGIN_THREADS_DEF;

        typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1, DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1, DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    n1 = ap1->dimensions[0];
    n2 = ap2->dimensions[0];
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
    }
    length = n1;
    n = n2;
    switch(mode) {
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (intp)(n/2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "mode must be 0, 1, or 2");
        goto fail;
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, 1, &length, typenum);
    if (ret == NULL) {
        goto fail;
    }
    dot = ret->descr->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
        goto fail;
    }

    NPY_BEGIN_THREADS_DESCR(ret->descr);
    is1 = ap1->strides[0];
    is2 = ap2->strides[0];
    op = ret->data;
    os = ret->descr->elsize;
    ip1 = ap1->data;
    ip2 = ap2->data + n_left*is2;
    n = n - n_left;
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        n++;
        ip2 -= is2;
        op += os;
    }
    for (i = 0; i < (n1 - n2 + 1); i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }
    for (i = 0; i < n_right; i++) {
        n--;
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }
    NPY_END_THREADS_DESCR(ret->descr);
    if (PyErr_Occurred()) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


/*NUMPY_API
 * ArgMin
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMin(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyObject *obj, *new, *ret;

    if (PyArray_ISFLEXIBLE(ap)) {
        PyErr_SetString(PyExc_TypeError,
                        "argmax is unsupported for this type");
        return NULL;
    }
    else if (PyArray_ISUNSIGNED(ap)) {
        obj = PyInt_FromLong((long) -1);
    }
    else if (PyArray_TYPE(ap) == PyArray_BOOL) {
        obj = PyInt_FromLong((long) 1);
    }
    else {
        obj = PyInt_FromLong((long) 0);
    }
    new = PyArray_EnsureAnyArray(PyNumber_Subtract(obj, (PyObject *)ap));
    Py_DECREF(obj);
    if (new == NULL) {
        return NULL;
    }
    ret = PyArray_ArgMax((PyArrayObject *)new, axis, out);
    Py_DECREF(new);
    return ret;
}

/*NUMPY_API
 * Max
 */
NPY_NO_EXPORT PyObject *
PyArray_Max(PyArrayObject *ap, int axis, PyArrayObject *out)
{
    PyArrayObject *arr;
    PyObject *ret;

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, n_ops.maximum, axis,
                                        arr->descr->type_num, out);
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

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
        return NULL;
    }
    ret = PyArray_GenericReduceFunction(arr, n_ops.minimum, axis,
                                        arr->descr->type_num, out);
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

    if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0)) == NULL) {
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
 * ArgMax
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgMax(PyArrayObject *op, int axis, PyArrayObject *out)
{
    PyArrayObject *ap = NULL, *rp = NULL;
    PyArray_ArgFunc* arg_func;
    char *ip;
    intp *rptr;
    intp i, n, m;
    int elsize;
    int copyret = 0;
    NPY_BEGIN_THREADS_DEF;

    if ((ap=(PyAO *)_check_axis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /*
     * We need to permute the array so that axis is placed at the end.
     * And all other dimensions are shifted left.
     */
    if (axis != ap->nd-1) {
        PyArray_Dims newaxes;
        intp dims[MAX_DIMS];
        int i;

        newaxes.ptr = dims;
        newaxes.len = ap->nd;
        for (i = 0; i < axis; i++) dims[i] = i;
        for (i = axis; i < ap->nd - 1; i++) dims[i] = i + 1;
        dims[ap->nd - 1] = axis;
        op = (PyAO *)PyArray_Transpose(ap, &newaxes);
        Py_DECREF(ap);
        if (op == NULL) {
            return NULL;
        }
    }
    else {
        op = ap;
    }

    /* Will get native-byte order contiguous copy. */
    ap = (PyArrayObject *)
        PyArray_ContiguousFromAny((PyObject *)op,
                                  op->descr->type_num, 1, 0);
    Py_DECREF(op);
    if (ap == NULL) {
        return NULL;
    }
    arg_func = ap->descr->f->argmax;
    if (arg_func == NULL) {
        PyErr_SetString(PyExc_TypeError, "data type not ordered");
        goto fail;
    }
    elsize = ap->descr->elsize;
    m = ap->dimensions[ap->nd-1];
    if (m == 0) {
        PyErr_SetString(MultiArrayError,
                        "attempt to get argmax/argmin "\
                        "of an empty sequence");
        goto fail;
    }

    if (!out) {
        rp = (PyArrayObject *)PyArray_New(ap->ob_type, ap->nd-1,
                                          ap->dimensions, PyArray_INTP,
                                          NULL, NULL, 0, 0,
                                          (PyObject *)ap);
        if (rp == NULL) {
            goto fail;
        }
    }
    else {
        if (PyArray_SIZE(out) !=
                PyArray_MultiplyList(ap->dimensions, ap->nd - 1)) {
            PyErr_SetString(PyExc_TypeError,
                            "invalid shape for output array.");
        }
        rp = (PyArrayObject *)\
            PyArray_FromArray(out,
                              PyArray_DescrFromType(PyArray_INTP),
                              NPY_CARRAY | NPY_UPDATEIFCOPY);
        if (rp == NULL) {
            goto fail;
        }
        if (rp != out) {
            copyret = 1;
        }
    }

    NPY_BEGIN_THREADS_DESCR(ap->descr);
    n = PyArray_SIZE(ap)/m;
    rptr = (intp *)rp->data;
    for (ip = ap->data, i = 0; i < n; i++, ip += elsize*m) {
        arg_func(ip, m, rptr, ap);
        rptr += 1;
    }
    NPY_END_THREADS_DESCR(ap->descr);

    Py_DECREF(ap);
    if (copyret) {
        PyArrayObject *obj;
        obj = (PyArrayObject *)rp->base;
        Py_INCREF(obj);
        Py_DECREF(rp);
        rp = obj;
    }
    return (PyObject *)rp;

 fail:
    Py_DECREF(ap);
    Py_XDECREF(rp);
    return NULL;
}


static PyObject *
array_putmask(PyObject *NPY_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *mask, *values;
    PyObject *array;

    static char *kwlist[] = {"arr", "mask", "values", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO:putmask", kwlist,
                                     &PyArray_Type,
                                     &array, &mask, &values)) {
        return NULL;
    }
    return PyArray_PutMask((PyArrayObject *)array, values, mask);
}

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
 * Convert an object to FORTRAN / C / ANY
 */
NPY_NO_EXPORT int
PyArray_OrderConverter(PyObject *object, NPY_ORDER *val)
{
    char *str;
    if (object == NULL || object == Py_None) {
        *val = PyArray_ANYORDER;
    }
    else if (!PyString_Check(object) || PyString_GET_SIZE(object) < 1) {
        if (PyObject_IsTrue(object)) {
            *val = PyArray_FORTRANORDER;
        }
        else {
            *val = PyArray_CORDER;
        }
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
        return PY_SUCCEED;
    }
    else {
        str = PyString_AS_STRING(object);
        if (str[0] == 'C' || str[0] == 'c') {
            *val = PyArray_CORDER;
        }
        else if (str[0] == 'F' || str[0] == 'f') {
            *val = PyArray_FORTRANORDER;
        }
        else if (str[0] == 'A' || str[0] == 'a') {
            *val = PyArray_ANYORDER;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "order not understood");
            return PY_FAIL;
        }
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Convert an object to NPY_RAISE / NPY_CLIP / NPY_WRAP
 */
NPY_NO_EXPORT int
PyArray_ClipmodeConverter(PyObject *object, NPY_CLIPMODE *val)
{
    if (object == NULL || object == Py_None) {
        *val = NPY_RAISE;
    }
    else if (PyString_Check(object)) {
        char *str;
        str = PyString_AS_STRING(object);
        if (str[0] == 'C' || str[0] == 'c') {
            *val = NPY_CLIP;
        }
        else if (str[0] == 'W' || str[0] == 'w') {
            *val = NPY_WRAP;
        }
        else if (str[0] == 'R' || str[0] == 'r') {
            *val = NPY_RAISE;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "clipmode not understood");
            return PY_FAIL;
        }
    }
    else {
        int number = PyInt_AsLong(object);
        if (number == -1 && PyErr_Occurred()) {
            goto fail;
        }
        if (number <= (int) NPY_RAISE
            && number >= (int) NPY_CLIP) {
            *val = (NPY_CLIPMODE) number;
        }
        else {
            goto fail;
        }
    }
    return PY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError,
                    "clipmode not understood");
    return PY_FAIL;
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


/*
 * compare the field dictionary for two types
 * return 1 if the same or 0 if not
 */
static int
_equivalent_fields(PyObject *field1, PyObject *field2) {

    int same, val;

    if (field1 == field2) {
        return 1;
    }
    if (field1 == NULL || field2 == NULL) {
        return 0;
    }
    val = PyObject_Compare(field1, field2);
    if (val != 0 || PyErr_Occurred()) {
        same = 0;
    }
    else {
        same = 1;
    }
    PyErr_Clear();
    return same;
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *typ1, PyArray_Descr *typ2)
{
    int typenum1 = typ1->type_num;
    int typenum2 = typ2->type_num;
    int size1 = typ1->elsize;
    int size2 = typ2->elsize;

    if (size1 != size2) {
        return FALSE;
    }
    if (PyArray_ISNBO(typ1->byteorder) != PyArray_ISNBO(typ2->byteorder)) {
        return FALSE;
    }
    if (typenum1 == PyArray_VOID
        || typenum2 == PyArray_VOID) {
        return ((typenum1 == typenum2) &&
                _equivalent_fields(typ1->fields, typ2->fields));
    }
    return (typ1->kind == typ2->kind);
}

/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    Bool ret;

    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    ret = PyArray_EquivTypes(d1, d2);
    Py_DECREF(d1);
    Py_DECREF(d2);
    return ret;
}

/*** END C-API FUNCTIONS **/

static PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin)
{
    intp newdims[MAX_DIMS];
    intp newstrides[MAX_DIMS];
    int i, k, num;
    PyObject *ret;

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = arr->descr->elsize;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = arr->dimensions[k];
        newstrides[i] = arr->strides[k];
    }
    Py_INCREF(arr->descr);
    ret = PyArray_NewFromDescr(arr->ob_type, arr->descr, ndmin,
                               newdims, newstrides, arr->data, arr->flags,
                               (PyObject *)arr);
    /* steals a reference to arr --- so don't increment here */
    PyArray_BASE(ret) = (PyObject *)arr;
    return ret;
}


#define _ARET(x) PyArray_Return((PyArrayObject *)(x))

#define STRIDING_OK(op, order) ((order) == PyArray_ANYORDER ||          \
                                ((order) == PyArray_CORDER &&           \
                                 PyArray_ISCONTIGUOUS(op)) ||           \
                                ((order) == PyArray_FORTRANORDER &&     \
                                 PyArray_ISFORTRAN(op)))

static PyObject *
_array_fromobject(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws)
{
    PyObject *op, *ret = NULL;
    static char *kwd[]= {"object", "dtype", "copy", "order", "subok",
                         "ndmin", NULL};
    Bool subok = FALSE;
    Bool copy = TRUE;
    int ndmin = 0, nd;
    PyArray_Descr *type = NULL;
    PyArray_Descr *oldtype = NULL;
    NPY_ORDER order=PyArray_ANYORDER;
    int flags = 0;

    if (PyTuple_GET_SIZE(args) > 2) {
        PyErr_SetString(PyExc_ValueError,
                        "only 2 non-keyword arguments accepted");
        return NULL;
    }
    if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O&O&O&O&i", kwd, &op,
                                    PyArray_DescrConverter2,
                                    &type,
                                    PyArray_BoolConverter, &copy,
                                    PyArray_OrderConverter, &order,
                                    PyArray_BoolConverter, &subok,
                                    &ndmin)) {
        goto clean_type;
    }

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "\
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto clean_type;
    }
    /* fast exit if simple call */
    if ((subok && PyArray_Check(op))
        || (!subok && PyArray_CheckExact(op))) {
        if (type == NULL) {
            if (!copy && STRIDING_OK(op, order)) {
                Py_INCREF(op);
                ret = op;
                goto finish;
            }
            else {
                ret = PyArray_NewCopy((PyArrayObject*)op, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = PyArray_DESCR(op);
        if (PyArray_EquivTypes(oldtype, type)) {
            if (!copy && STRIDING_OK(op, order)) {
                Py_INCREF(op);
                ret = op;
                goto finish;
            }
            else {
                ret = PyArray_NewCopy((PyArrayObject*)op, order);
                if (oldtype == type) {
                    goto finish;
                }
                Py_INCREF(oldtype);
                Py_DECREF(PyArray_DESCR(ret));
                PyArray_DESCR(ret) = oldtype;
                goto finish;
            }
        }
    }

    if (copy) {
        flags = ENSURECOPY;
    }
    if (order == PyArray_CORDER) {
        flags |= CONTIGUOUS;
    }
    else if ((order == PyArray_FORTRANORDER)
             /* order == PyArray_ANYORDER && */
             || (PyArray_Check(op) && PyArray_ISFORTRAN(op))) {
        flags |= FORTRAN;
    }
    if (!subok) {
        flags |= ENSUREARRAY;
    }

    flags |= NPY_FORCECAST;
    Py_XINCREF(type);
    ret = PyArray_CheckFromAny(op, type, 0, 0, flags, NULL);

 finish:
    Py_XDECREF(type);
    if (!ret || (nd=PyArray_NDIM(ret)) >= ndmin) {
        return ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones((PyArrayObject *)ret, nd, ndmin);

clean_type:
    Py_XDECREF(type);
    return NULL;
}

static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"shape","dtype","order",NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = PyArray_CORDER;
    Bool fortran;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&",
                                     kwlist, PyArray_IntpConverter,
                                     &shape,
                                     PyArray_DescrConverter,
                                     &typecode,
                                     PyArray_OrderConverter, &order)) {
        goto fail;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = TRUE;
    }
    else {
        fortran = FALSE;
    }
    ret = PyArray_Empty(shape.len, shape.ptr, typecode, fortran);
    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return NULL;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
static PyObject *
array_scalar(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dtype","obj", NULL};
    PyArray_Descr *typecode;
    PyObject *obj = NULL;
    int alloc = 0;
    void *dptr;
    PyObject *ret;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O",
                                     kwlist, &PyArrayDescr_Type,
                                     &typecode,
                                     &obj)) {
        return NULL;
    }
    if (typecode->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "itemsize cannot be zero");
        return NULL;
    }

    if (PyDataType_FLAGCHK(typecode, NPY_ITEM_IS_POINTER)) {
        if (obj == NULL) {
            obj = Py_None;
        }
        dptr = &obj;
    }
    else {
        if (obj == NULL) {
            dptr = _pya_malloc(typecode->elsize);
            if (dptr == NULL) {
                return PyErr_NoMemory();
            }
            memset(dptr, '\0', typecode->elsize);
            alloc = 1;
        }
        else {
            if (!PyString_Check(obj)) {
                PyErr_SetString(PyExc_TypeError,
                                "initializing object must "\
                                "be a string");
                return NULL;
            }
            if (PyString_GET_SIZE(obj) < typecode->elsize) {
                PyErr_SetString(PyExc_ValueError,
                                "initialization string is too"\
                                " small");
                return NULL;
            }
            dptr = PyString_AS_STRING(obj);
        }
    }

    ret = PyArray_Scalar(dptr, typecode, NULL);

    /* free dptr which contains zeros */
    if (alloc) {
        _pya_free(dptr);
    }
    return ret;
}

static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape","dtype","order",NULL}; /* XXX ? */
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = PyArray_CORDER;
    Bool fortran = FALSE;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&",
                                     kwlist, PyArray_IntpConverter,
                                     &shape,
                                     PyArray_DescrConverter,
                                     &typecode,
                                     PyArray_OrderConverter,
                                     &order)) {
        goto fail;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = TRUE;
    }
    else {
        fortran = FALSE;
    }
    ret = PyArray_Zeros(shape.len, shape.ptr, typecode, (int) fortran);
    PyDimMem_FREE(shape.ptr);
    return ret;

 fail:
    Py_XDECREF(typecode);
    PyDimMem_FREE(shape.ptr);
    return ret;
}

static PyObject *
array_fromstring(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    char *data;
    Py_ssize_t nin = -1;
    char *sep = NULL;
    Py_ssize_t s;
    static char *kwlist[] = {"string", "dtype", "count", "sep", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s#|O&"
                                     NPY_SSIZE_T_PYFMT "s", kwlist,
                                     &data, &s,
                                     PyArray_DescrConverter, &descr,
                                     &nin, &sep)) {
        Py_XDECREF(descr);
        return NULL;
    }
    return PyArray_FromString(data, (intp)s, descr, (intp)nin, sep);
}



static PyObject *
array_fromfile(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *file = NULL, *ret;
    FILE *fp;
    char *sep = "";
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"file", "dtype", "count", "sep", NULL};
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                                     "O|O&" NPY_SSIZE_T_PYFMT "s",
                                     kwlist,
                                     &file,
                                     PyArray_DescrConverter, &type,
                                     &nin, &sep)) {
        Py_XDECREF(type);
        return NULL;
    }
    if (PyString_Check(file) || PyUnicode_Check(file)) {
        file = PyObject_CallFunction((PyObject *)&PyFile_Type,
                                     "Os", file, "rb");
        if (file == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(file);
    }
    fp = PyFile_AsFile(file);
    if (fp == NULL) {
        PyErr_SetString(PyExc_IOError,
                        "first argument must be an open file");
        Py_DECREF(file);
        return NULL;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    ret = PyArray_FromFile(fp, type, (intp) nin, sep);
    Py_DECREF(file);
    return ret;
}

static PyObject *
array_fromiter(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *iter;
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"iter", "dtype", "count", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                                     "OO&|" NPY_SSIZE_T_PYFMT,
                                     kwlist,
                                     &iter,
                                     PyArray_DescrConverter, &descr,
                                     &nin)) {
        Py_XDECREF(descr);
        return NULL;
    }
    return PyArray_FromIter(iter, descr, (intp)nin);
}

static PyObject *
array_frombuffer(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = {"buffer", "dtype", "count", "offset", NULL};
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|O&"
                                     NPY_SSIZE_T_PYFMT
                                     NPY_SSIZE_T_PYFMT, kwlist,
                                     &obj,
                                     PyArray_DescrConverter, &type,
                                     &nin, &offset)) {
        Py_XDECREF(type);
        return NULL;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    return PyArray_FromBuffer(obj, type, (intp)nin, (intp)offset);
}

static PyObject *
array_concatenate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *a0;
    int axis = 0;
    static char *kwlist[] = {"seq", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
                                     &a0,
                                     PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return PyArray_Concatenate(a0, axis);
}

static PyObject *array_innerproduct(PyObject *NPY_UNUSED(dummy), PyObject *args) {
    PyObject *b0, *a0;

    if (!PyArg_ParseTuple(args, "OO", &a0, &b0)) {
        return NULL;
    }
    return _ARET(PyArray_InnerProduct(a0, b0));
}

static PyObject *array_matrixproduct(PyObject *NPY_UNUSED(dummy), PyObject *args) {
    PyObject *v, *a;

    if (!PyArg_ParseTuple(args, "OO", &a, &v)) {
        return NULL;
    }
    return _ARET(PyArray_MatrixProduct(a, v));
}

static PyObject *array_fastCopyAndTranspose(PyObject *NPY_UNUSED(dummy), PyObject *args) {
    PyObject *a0;

    if (!PyArg_ParseTuple(args, "O", &a0)) {
        return NULL;
    }
    return _ARET(PyArray_CopyAndTranspose(a0));
}

static PyObject *array_correlate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds) {
    PyObject *shape, *a0;
    int mode = 0;
    static char *kwlist[] = {"a", "v", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist,
                                     &a0, &shape, &mode)) {
        return NULL;
    }
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject *
array_arange(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws) {
    PyObject *o_start = NULL, *o_stop = NULL, *o_step = NULL;
    static char *kwd[]= {"start", "stop", "step", "dtype", NULL};
    PyArray_Descr *typecode = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kws, "O|OOO&", kwd, &o_start,
                                    &o_stop, &o_step,
                                    PyArray_DescrConverter2,
                                    &typecode)) {
        Py_XDECREF(typecode);
        return NULL;
    }
    return PyArray_ArangeObj(o_start, o_stop, o_step, typecode);
}

/*
 * Included at the very first so not auto-grabbed and thus not
 * labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_VERSION;
}

static PyObject *
array__get_ndarray_c_version(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist )) {
        return NULL;
    }
    return PyInt_FromLong( (long) PyArray_GetNDArrayCVersion() );
}

/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE;
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN;
    }
}

static PyObject *
array__reconstruct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    PyObject *ret;
    PyTypeObject *subtype;
    PyArray_Dims shape = {NULL, 0};
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTuple(args, "O!O&O&", &PyType_Type, &subtype,
                          PyArray_IntpConverter, &shape,
                          PyArray_DescrConverter, &dtype)) {
        goto fail;
    }
    if (!PyType_IsSubtype(subtype, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError,
                        "_reconstruct: First argument must be " \
                        "a sub-type of ndarray");
        goto fail;
    }
    ret = PyArray_NewFromDescr(subtype, dtype,
                               (int)shape.len, shape.ptr,
                               NULL, NULL, 0, NULL);
    if (shape.ptr) {
        PyDimMem_FREE(shape.ptr);
    }
    return ret;

 fail:
    Py_XDECREF(dtype);
    if (shape.ptr) {
        PyDimMem_FREE(shape.ptr);
    }
    return NULL;
}

static PyObject *
array_set_string_function(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *op = NULL;
    int repr=1;
    static char *kwlist[] = {"f", "repr", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist,
                                    &op, &repr)) {
        return NULL;
    }
    /* reset the array_repr function to built-in */
    if (op == Py_None) {
        op = NULL;
    }
    if (op != NULL && !PyCallable_Check(op)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be callable.");
        return NULL;
    }
    PyArray_SetStringFunction(op, repr);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
array_set_ops_function(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args), PyObject *kwds)
{
    PyObject *oldops = NULL;

    if ((oldops = PyArray_GetNumericOps()) == NULL) {
        return NULL;
    }
    /*
     * Should probably ensure that objects are at least callable
     *  Leave this to the caller for now --- error will be raised
     *  later when use is attempted
     */
    if (kwds && PyArray_SetNumericOps(kwds) == -1) {
        Py_DECREF(oldops);
        PyErr_SetString(PyExc_ValueError, "one or more objects not callable");
        return NULL;
    }
    return oldops;
}


/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    PyArrayObject *arr;
    PyObject *tup = NULL, *obj = NULL;
    PyObject *ret = NULL, *zero = NULL;

    arr = (PyArrayObject *)PyArray_FromAny(condition, NULL, 0, 0, 0, NULL);
    if (arr == NULL) {
        return NULL;
    }
    if ((x == NULL) && (y == NULL)) {
        ret = PyArray_Nonzero(arr);
        Py_DECREF(arr);
        return ret;
    }
    if ((x == NULL) || (y == NULL)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "either both or neither "
                        "of x and y should be given");
        return NULL;
    }


    zero = PyInt_FromLong((long) 0);
    obj = PyArray_EnsureAnyArray(PyArray_GenericBinaryFunction(arr, zero,
                                                               n_ops.not_equal));
    Py_DECREF(zero);
    Py_DECREF(arr);
    if (obj == NULL) {
        return NULL;
    }
    tup = Py_BuildValue("(OO)", y, x);
    if (tup == NULL) {
        Py_DECREF(obj);
        return NULL;
    }
    ret = PyArray_Choose((PyAO *)obj, tup, NULL, NPY_RAISE);
    Py_DECREF(obj);
    Py_DECREF(tup);
    return ret;
}

static PyObject *
array_where(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *obj = NULL, *x = NULL, *y = NULL;

    if (!PyArg_ParseTuple(args, "O|OO", &obj, &x, &y)) {
        return NULL;
    }
    return PyArray_Where(obj, x, y);
}

static PyObject *
array_lexsort(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    int axis = -1;
    PyObject *obj;
    static char *kwlist[] = {"keys", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist,
                                     &obj, &axis)) {
        return NULL;
    }
    return _ARET(PyArray_LexSort(obj, axis));
}

#undef _ARET

static PyObject *
array_can_cast_safely(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    Bool ret;
    PyObject *retobj = NULL;
    static char *kwlist[] = {"from", "to", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist,
                                    PyArray_DescrConverter, &d1,
                                    PyArray_DescrConverter, &d2)) {
        goto finish;
    }
    if (d1 == NULL || d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "did not understand one of the types; " \
                        "'None' not accepted");
        goto finish;
    }

    ret = PyArray_CanCastTo(d1, d2);
    retobj = ret ? Py_True : Py_False;
    Py_INCREF(retobj);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return retobj;
}

static PyObject *
new_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int size;

    if(!PyArg_ParseTuple(args, "i", &size)) {
        return NULL;
    }
    return PyBuffer_New(size);
}

static PyObject *
buffer_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    Py_ssize_t offset = 0, size = Py_END_OF_BUFFER, n;
    void *unused;
    static char *kwlist[] = {"object", "offset", "size", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|" NPY_SSIZE_T_PYFMT
                                     NPY_SSIZE_T_PYFMT, kwlist,
                                     &obj, &offset, &size)) {
        return NULL;
    }
    if (PyObject_AsWriteBuffer(obj, &unused, &n) < 0) {
        PyErr_Clear();
        return PyBuffer_FromObject(obj, offset, size);
    }
    else {
        return PyBuffer_FromReadWriteObject(obj, offset, size);
    }
}

#ifndef _MSC_VER
#include <setjmp.h>
#include <signal.h>
jmp_buf _NPY_SIGSEGV_BUF;
static void
_SigSegv_Handler(int signum)
{
    longjmp(_NPY_SIGSEGV_BUF, signum);
}
#endif

#define _test_code() {                          \
        test = *((char*)memptr);                \
        if (!ro) {                              \
            *((char *)memptr) = '\0';           \
            *((char *)memptr) = test;           \
        }                                       \
        test = *((char*)memptr+size-1);         \
        if (!ro) {                              \
            *((char *)memptr+size-1) = '\0';    \
            *((char *)memptr+size-1) = test;    \
        }                                       \
    }

static PyObject *
as_buffer(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *mem;
    Py_ssize_t size;
    Bool ro = FALSE, check = TRUE;
    void *memptr;
    static char *kwlist[] = {"mem", "size", "readonly", "check", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O" \
                                     NPY_SSIZE_T_PYFMT "|O&O&", kwlist,
                                     &mem, &size, PyArray_BoolConverter,
                                     &ro, PyArray_BoolConverter,
                                     &check)) {
        return NULL;
    }
    memptr = PyLong_AsVoidPtr(mem);
    if (memptr == NULL) {
        return NULL;
    }
    if (check) {
        /*
         * Try to dereference the start and end of the memory region
         * Catch segfault and report error if it occurs
         */
        char test;
        int err = 0;

#ifdef _MSC_VER
        __try {
            _test_code();
        }
        __except(1) {
            err = 1;
        }
#else
        PyOS_sighandler_t _npy_sig_save;
        _npy_sig_save = PyOS_setsig(SIGSEGV, _SigSegv_Handler);
        if (setjmp(_NPY_SIGSEGV_BUF) == 0) {
            _test_code();
        }
        else {
            err = 1;
        }
        PyOS_setsig(SIGSEGV, _npy_sig_save);
#endif
        if (err) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot use memory location as " \
                            "a buffer.");
            return NULL;
        }
    }


    if (ro) {
        return PyBuffer_FromMemory(memptr, size);
    }
    return PyBuffer_FromReadWriteMemory(memptr, size);
}

#undef _test_code

static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    unsigned int precision;
    longdouble x;
    static char *kwlist[] = {"x", "precision", NULL};
    static char repr[100];

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI", kwlist,
                                     &obj, &precision)) {
        return NULL;
    }
    if (!PyArray_IsScalar(obj, LongDouble)) {
        PyErr_SetString(PyExc_TypeError, "not a longfloat");
        return NULL;
    }
    x = ((PyLongDoubleScalarObject *)obj)->obval;
    if (precision > 70) {
        precision = 70;
    }
    format_longdouble(repr, 100, x, precision);
    return PyString_FromString(repr);
}

static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *array;
    PyObject *other;
    PyArrayObject *newarr, *newoth;
    int cmp_op;
    Bool rstrip;
    char *cmp_str;
    Py_ssize_t strlen;
    PyObject *res = NULL;
    static char msg[] = "comparision must be '==', '!=', '<', '>', '<=', '>='";
    static char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs#O&", kwlist,
                                     &array, &other,
                                     &cmp_str, &strlen,
                                     PyArray_BoolConverter, &rstrip)) {
        return NULL;
    }
    if (strlen < 1 || strlen > 2) {
        goto err;
    }
    if (strlen > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    newarr = (PyArrayObject *)PyArray_FROM_O(array);
    if (newarr == NULL) {
        return NULL;
    }
    newoth = (PyArrayObject *)PyArray_FROM_O(other);
    if (newoth == NULL) {
        Py_DECREF(newarr);
        return NULL;
    }
    if (PyArray_ISSTRING(newarr) && PyArray_ISSTRING(newoth)) {
        res = _strings_richcompare(newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "comparison of non-string arrays");
    }
    Py_DECREF(newarr);
    Py_DECREF(newoth);
    return res;

 err:
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
}


#ifndef __NPY_PRIVATE_NO_SIGNAL

SIGJMP_BUF _NPY_SIGINT_BUF;

/*NUMPY_API
 */
NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    PyOS_setsig(signum, SIG_IGN);
    SIGLONGJMP(_NPY_SIGINT_BUF, signum);
}

/*NUMPY_API
 */
NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    return (void *)&_NPY_SIGINT_BUF;
}

#else

static void
_PyArray_SigintHandler(int signum)
{
    return;
}

static void*
_PyArray_GetSigintBuf(void)
{
    return NULL;
}

#endif


static PyObject *
test_interrupt(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int kind = 0;
    int a = 0;

    if (!PyArg_ParseTuple(args, "|i", &kind)) {
        return NULL;
    }
    if (kind) {
        Py_BEGIN_ALLOW_THREADS;
        while (a >= 0) {
            if ((a % 1000 == 0) && PyOS_InterruptOccurred()) {
                break;
            }
            a += 1;
        }
        Py_END_ALLOW_THREADS;
    }
    else {
        NPY_SIGINT_ON
        while(a >= 0) {
            a += 1;
        }
        NPY_SIGINT_OFF
    }
    return PyInt_FromLong(a);
}

static struct PyMethodDef array_module_methods[] = {
    {"_get_ndarray_c_version",
        (PyCFunction)array__get_ndarray_c_version,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"_reconstruct",
        (PyCFunction)array__reconstruct,
        METH_VARARGS, NULL},
    {"set_string_function",
        (PyCFunction)array_set_string_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_numeric_ops",
        (PyCFunction)array_set_ops_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},

    {"array",
        (PyCFunction)_array_fromobject,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",
        (PyCFunction)array_arange,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros",
        (PyCFunction)array_zeros,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty",
        (PyCFunction)array_empty,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"where",
        (PyCFunction)array_where,
        METH_VARARGS, NULL},
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"putmask",
        (PyCFunction)array_putmask,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromstring",
        (PyCFunction)array_fromstring,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"fromiter",
        (PyCFunction)array_fromiter,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_VARARGS, NULL},
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_VARARGS, NULL},
    {"_fastCopyAndTranspose",
        (PyCFunction)array_fastCopyAndTranspose,
        METH_VARARGS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"newbuffer",
        (PyCFunction)new_buffer,
        METH_VARARGS, NULL},
    {"getbuffer",
        (PyCFunction)buffer_buffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"int_asbuffer",
        (PyCFunction)as_buffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"format_longfloat",
        (PyCFunction)format_longfloat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"compare_chararrays",
        (PyCFunction)compare_chararrays,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"test_interrupt",
        (PyCFunction)test_interrupt,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

#include "__multiarray_api.c"

/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(PyObject *NPY_UNUSED(dict))
{
    initialize_numeric_types();

    if (PyType_Ready(&PyBool_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyInt_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyFloat_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyComplex_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyString_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyUnicode_Type) < 0) {
        return -1;
    }

#define SINGLE_INHERIT(child, parent)                                   \
    Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;        \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    if (PyType_Ready(&PyGenericArrType_Type) < 0) {
        return -1;
    }
    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;       \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,               \
                      &Py##parent1##_Type);                             \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }                                                                   \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;

#define DUAL_INHERIT2(child, parent1, parent2)                          \
    Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;              \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent1##_Type,                      \
                      &Py##parent2##ArrType_Type);                      \
    Py##child##ArrType_Type.tp_richcompare =                            \
        Py##parent1##_Type.tp_richcompare;                              \
    Py##child##ArrType_Type.tp_compare =                                \
        Py##parent1##_Type.tp_compare;                                  \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    SINGLE_INHERIT(Bool, Generic);
    SINGLE_INHERIT(Byte, SignedInteger);
    SINGLE_INHERIT(Short, SignedInteger);
#if SIZEOF_INT == SIZEOF_LONG
    DUAL_INHERIT(Int, Int, SignedInteger);
#else
    SINGLE_INHERIT(Int, SignedInteger);
#endif
    DUAL_INHERIT(Long, Int, SignedInteger);
#if SIZEOF_LONGLONG == SIZEOF_LONG
    DUAL_INHERIT(LongLong, Int, SignedInteger);
#else
    SINGLE_INHERIT(LongLong, SignedInteger);
#endif

    /*
       fprintf(stderr,
        "tp_free = %p, PyObject_Del = %p, int_tp_free = %p, base.tp_free = %p\n",
         PyIntArrType_Type.tp_free, PyObject_Del, PyInt_Type.tp_free,
         PySignedIntegerArrType_Type.tp_free);
     */
    SINGLE_INHERIT(UByte, UnsignedInteger);
    SINGLE_INHERIT(UShort, UnsignedInteger);
    SINGLE_INHERIT(UInt, UnsignedInteger);
    SINGLE_INHERIT(ULong, UnsignedInteger);
    SINGLE_INHERIT(ULongLong, UnsignedInteger);

    SINGLE_INHERIT(Float, Floating);
    DUAL_INHERIT(Double, Float, Floating);
    SINGLE_INHERIT(LongDouble, Floating);

    SINGLE_INHERIT(CFloat, ComplexFloating);
    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
    SINGLE_INHERIT(CLongDouble, ComplexFloating);

    DUAL_INHERIT2(String, String, Character);
    DUAL_INHERIT2(Unicode, Unicode, Character);

    SINGLE_INHERIT(Void, Flexible);

    SINGLE_INHERIT(Object, Generic);

    return 0;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT

    /*
     * Clean up string and unicode array types so they act more like
     * strings -- get their tables from the standard types.
     */
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
    PyObject *s;
    PyObject *newd;

    newd = PyDict_New();

#define _addnew(val, one)                                       \
    PyDict_SetItemString(newd, #val, s=PyInt_FromLong(val));    \
    Py_DECREF(s);                                               \
    PyDict_SetItemString(newd, #one, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

#define _addone(val)                                            \
    PyDict_SetItemString(newd, #val, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

    _addnew(OWNDATA, O);
    _addnew(FORTRAN, F);
    _addnew(CONTIGUOUS, C);
    _addnew(ALIGNED, A);
    _addnew(UPDATEIFCOPY, U);
    _addnew(WRITEABLE, W);
    _addone(C_CONTIGUOUS);
    _addone(F_CONTIGUOUS);

#undef _addone
#undef _addnew

    PyDict_SetItemString(d, "_flagdict", newd);
    Py_DECREF(newd);
    return;
}


/* Initialization function for the module */

PyMODINIT_FUNC initmultiarray(void) {
    PyObject *m, *d, *s;
    PyObject *c_api;

    /* Create the module and add the functions */
    m = Py_InitModule("multiarray", array_module_methods);
    if (!m) {
        goto err;
    }

#ifdef MS_WIN64
  PyErr_WarnEx(PyExc_Warning,
        "Windows 64 bits support is experimental, and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        goto err;
    }
    PyArray_Type.tp_free = _pya_free;
    if (PyType_Ready(&PyArray_Type) < 0) {
        return;
    }
    if (setup_scalartypes(d) < 0) {
        goto err;
    }
    PyArrayIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_free = _pya_free;
    if (PyType_Ready(&PyArrayIter_Type) < 0) {
        return;
    }
    if (PyType_Ready(&PyArrayMapIter_Type) < 0) {
        return;
    }
    if (PyType_Ready(&PyArrayMultiIter_Type) < 0) {
        return;
    }
    PyArrayDescr_Type.tp_hash = PyArray_DescrHash;
    if (PyType_Ready(&PyArrayDescr_Type) < 0) {
        return;
    }
    if (PyType_Ready(&PyArrayFlags_Type) < 0) {
        return;
    }
    c_api = PyCObject_FromVoidPtr((void *)PyArray_API, NULL);
    PyDict_SetItemString(d, "_ARRAY_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }
    MultiArrayError = PyString_FromString ("multiarray.error");
    PyDict_SetItemString (d, "error", MultiArrayError);
    s = PyString_FromString("3.0");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

#define ADDCONST(NAME)                          \
    s = PyInt_FromLong(NPY_##NAME);             \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);
    ADDCONST(USE_GETITEM);
    ADDCONST(USE_SETITEM);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);
#undef ADDCONST

    Py_INCREF(&PyArray_Type);
    PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
    Py_INCREF(&PyArrayIter_Type);
    PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
    Py_INCREF(&PyArrayMultiIter_Type);
    PyDict_SetItemString(d, "broadcast",
                         (PyObject *)&PyArrayMultiIter_Type);
    Py_INCREF(&PyArrayDescr_Type);
    PyDict_SetItemString(d, "dtype", (PyObject *)&PyArrayDescr_Type);

    Py_INCREF(&PyArrayFlags_Type);
    PyDict_SetItemString(d, "flagsobj", (PyObject *)&PyArrayFlags_Type);

    set_flaginfo(d);

    if (set_typeinfo(d) != 0) {
        goto err;
    }
    return;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load multiarray module.");
    }
    return;
}
