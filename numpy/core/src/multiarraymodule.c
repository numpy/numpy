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

#define PyAO PyArrayObject

#include "hashdescr.c"

static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

static PyArray_Descr *
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

#include "arrayobject.c"


/* An Error object -- rarely used? */
static PyObject *MultiArrayError;

/*NUMPY_API
 * Multiply a List of ints
 */
static int
PyArray_MultiplyIntList(register int *l1, register int n)
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
static intp
PyArray_MultiplyList(register intp *l1, register int n)
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
static intp
PyArray_OverflowMultiplyList(register intp *l1, register int n)
{
    intp s = 1;

    while (n--) {
	if (*l1 == 0) {
            return 0;
        }
	if ((s > MAX_INTP / *l1) || (*l1 > MAX_INTP / s)) {
	    return -1;
        }
	s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Produce a pointer into array
 */
static void *
PyArray_GetPtr(PyArrayObject *obj, register intp* ind)
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
static int
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
static int
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

/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
static PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    PyObject *new = NULL;
    PyTypeObject *subtype;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = self->ob_type;
    }
    Py_INCREF(self->descr);
    new = PyArray_NewFromDescr(subtype,
                               self->descr,
                               self->nd, self->dimensions,
                               self->strides,
                               self->data,
                               self->flags, (PyObject *)self);
    if (new == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    PyArray_BASE(new) = (PyObject *)self;

    if (type != NULL) {
        if (PyObject_SetAttrString(new, "dtype",
                                   (PyObject *)type) < 0) {
            Py_DECREF(new);
            Py_DECREF(type);
            return NULL;
        }
        Py_DECREF(type);
    }
    return new;
}


/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
static PyObject *
PyArray_Ravel(PyArrayObject *a, NPY_ORDER fortran)
{
    PyArray_Dims newdim = {NULL,1};
    intp val[1] = {-1};

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_ISFORTRAN(a);
    }
    newdim.ptr = val;
    if (!fortran && PyArray_ISCONTIGUOUS(a)) {
        return PyArray_Newshape(a, &newdim, PyArray_CORDER);
    }
    else if (fortran && PyArray_ISFORTRAN(a)) {
        return PyArray_Newshape(a, &newdim, PyArray_FORTRANORDER);
    }
    else {
        return PyArray_Flatten(a, fortran);
    }
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
static PyObject *
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
 * Flatten
 */
static PyObject *
PyArray_Flatten(PyArrayObject *a, NPY_ORDER order)
{
    PyObject *ret;
    intp size;

    if (order == PyArray_ANYORDER) {
        order = PyArray_ISFORTRAN(a);
    }
    size = PyArray_SIZE(a);
    Py_INCREF(a->descr);
    ret = PyArray_NewFromDescr(a->ob_type,
                               a->descr,
                               1, &size,
                               NULL,
                               NULL,
                               0, (PyObject *)a);

    if (ret == NULL) {
        return NULL;
    }
    if (_flat_copyinto(ret, (PyObject *)a, order) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return ret;
}


/* For back-ward compatability -- Not recommended */

/*NUMPY_API
 * Reshape
 */
static PyObject *
PyArray_Reshape(PyArrayObject *self, PyObject *shape)
{
    PyObject *ret;
    PyArray_Dims newdims;

    if (!PyArray_IntpConverter(shape, &newdims)) {
        return NULL;
    }
    ret = PyArray_Newshape(self, &newdims, PyArray_CORDER);
    PyDimMem_FREE(newdims.ptr);
    return ret;
}

/* inserts 0 for strides where dimension will be 1 */
static int
_check_ones(PyArrayObject *self, int newnd, intp* newdims, intp *strides)
{
    int nd;
    intp *dims;
    Bool done=FALSE;
    int j, k;

    nd = self->nd;
    dims = self->dimensions;

    for (k = 0, j = 0; !done && (j < nd || k < newnd);) {
        if ((j<nd) && (k<newnd) && (newdims[k] == dims[j])) {
            strides[k] = self->strides[j];
            j++;
            k++;
        }
        else if ((k < newnd) && (newdims[k] == 1)) {
            strides[k] = 0;
            k++;
        }
        else if ((j<nd) && (dims[j] == 1)) {
            j++;
        }
        else {
            done = TRUE;
        }
    }
    if (done) {
        return -1;
    }
    return 0;
}

/*
 * attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "fortran" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in self->strides).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, intp* newdims,
                        intp *newstrides, int fortran)
{
    int oldnd;
    intp olddims[MAX_DIMS];
    intp oldstrides[MAX_DIMS];
    int oi, oj, ok, ni, nj, nk;
    int np, op;

    oldnd = 0;
    for (oi = 0; oi < self->nd; oi++) {
        if (self->dimensions[oi]!= 1) {
            olddims[oldnd] = self->dimensions[oi];
            oldstrides[oldnd] = self->strides[oi];
            oldnd++;
        }
    }

    /*
      fprintf(stderr, "_attempt_nocopy_reshape( (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,*), ", newdims[ni]);
      fprintf(stderr, "), fortran=%d)\n", fortran);
    */


    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }
    /* the current code does not handle 0-sized arrays, so give up */
    if (np == 0) {
        return 0;
    }

    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while(ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        for (ok = oi; ok < oj - 1; ok++) {
            if (fortran) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        if (fortran) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
      fprintf(stderr, "success: _attempt_nocopy_reshape (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,%d), ", newdims[ni], newstrides[ni]);
      fprintf(stderr, ")\n");
    */

    return 1;
}

static int
_fix_unknown_dimension(PyArray_Dims *newshape, intp s_original)
{
    intp *dimensions;
    intp i_unknown, s_known;
    int i, n;
    static char msg[] = "total size of new array must be unchanged";

    dimensions = newshape->ptr;
    n = newshape->len;
    s_known = 1;
    i_unknown = -1;

    for (i = 0; i < n; i++) {
        if (dimensions[i] < 0) {
            if (i_unknown == -1) {
                i_unknown = i;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                                "can only specify one"  \
                                " unknown dimension");
                return -1;
            }
        }
        else {
            s_known *= dimensions[i];
        }
    }

    if (i_unknown >= 0) {
        if ((s_known == 0) || (s_original % s_known != 0)) {
            PyErr_SetString(PyExc_ValueError, msg);
            return -1;
        }
        dimensions[i_unknown] = s_original/s_known;
    }
    else {
        if (s_original != s_known) {
            PyErr_SetString(PyExc_ValueError, msg);
            return -1;
        }
    }
    return 0;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on fortran argument.
 * copy-only-if-necessary
 */

/*NUMPY_API
 * New shape for an array
 */
static PyObject *
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER fortran)
{
    intp i;
    intp *dimensions = newdims->ptr;
    PyArrayObject *ret;
    int n = newdims->len;
    Bool same, incref = TRUE;
    intp *strides = NULL;
    intp newstrides[MAX_DIMS];
    int flags;

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_ISFORTRAN(self);
    }
    /*  Quick check to make sure anything actually needs to be done */
    if (n == self->nd) {
        same = TRUE;
        i = 0;
        while (same && i < n) {
            if (PyArray_DIM(self,i) != dimensions[i]) {
                same=FALSE;
            }
            i++;
        }
        if (same) {
            return PyArray_View(self, NULL, NULL);
        }
    }

    /*
     * Returns a pointer to an appropriate strides array
     * if all we are doing is inserting ones into the shape,
     * or removing ones from the shape
     * or doing a combination of the two
     * In this case we don't need to do anything but update strides and
     * dimensions.  So, we can handle non single-segment cases.
     */
    i = _check_ones(self, n, dimensions, newstrides);
    if (i == 0) {
        strides = newstrides;
    }
    flags = self->flags;

    if (strides == NULL) {
        /*
         * we are really re-shaping not just adding ones to the shape somewhere
         * fix any -1 dimensions and check new-dimensions against old size
         */
        if (_fix_unknown_dimension(newdims, PyArray_SIZE(self)) < 0) {
            return NULL;
        }
        /*
         * sometimes we have to create a new copy of the array
         * in order to get the right orientation and
         * because we can't just re-use the buffer with the
         * data in the order it is in.
         */
        if (!(PyArray_ISONESEGMENT(self)) ||
            (((PyArray_CHKFLAGS(self, NPY_CONTIGUOUS) &&
               fortran == NPY_FORTRANORDER) ||
              (PyArray_CHKFLAGS(self, NPY_FORTRAN) &&
                  fortran == NPY_CORDER)) && (self->nd > 1))) {
            int success = 0;
            success = _attempt_nocopy_reshape(self,n,dimensions,
                                              newstrides,fortran);
            if (success) {
                /* no need to copy the array after all */
                strides = newstrides;
                flags = self->flags;
            }
            else {
                PyObject *new;
                new = PyArray_NewCopy(self, fortran);
                if (new == NULL) {
                    return NULL;
                }
                incref = FALSE;
                self = (PyArrayObject *)new;
                flags = self->flags;
            }
        }

        /* We always have to interpret the contiguous buffer correctly */

        /* Make sure the flags argument is set. */
        if (n > 1) {
            if (fortran == NPY_FORTRANORDER) {
                flags &= ~NPY_CONTIGUOUS;
                flags |= NPY_FORTRAN;
            }
            else {
                flags &= ~NPY_FORTRAN;
                flags |= NPY_CONTIGUOUS;
            }
        }
    }
    else if (n > 0) {
        /*
         * replace any 0-valued strides with
         * appropriate value to preserve contiguousness
         */
        if (fortran == PyArray_FORTRANORDER) {
            if (strides[0] == 0) {
                strides[0] = self->descr->elsize;
            }
            for (i = 1; i < n; i++) {
                if (strides[i] == 0) {
                    strides[i] = strides[i-1] * dimensions[i-1];
                }
            }
        }
        else {
            if (strides[n-1] == 0) {
                strides[n-1] = self->descr->elsize;
            }
            for (i = n - 2; i > -1; i--) {
                if (strides[i] == 0) {
                    strides[i] = strides[i+1] * dimensions[i+1];
                }
            }
        }
    }

    Py_INCREF(self->descr);
    ret = (PyAO *)PyArray_NewFromDescr(self->ob_type,
                                       self->descr,
                                       n, dimensions,
                                       strides,
                                       self->data,
                                       flags, (PyObject *)self);

    if (ret == NULL) {
        goto fail;
    }
    if (incref) {
        Py_INCREF(self);
    }
    ret->base = (PyObject *)self;
    PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
    return (PyObject *)ret;

 fail:
    if (!incref) {
        Py_DECREF(self);
    }
    return NULL;
}



/*NUMPY_API
 *
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
static PyObject *
PyArray_Squeeze(PyArrayObject *self)
{
    int nd = self->nd;
    int newnd = nd;
    intp dimensions[MAX_DIMS];
    intp strides[MAX_DIMS];
    int i, j;
    PyObject *ret;

    if (nd == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    for (j = 0, i = 0; i < nd; i++) {
        if (self->dimensions[i] == 1) {
            newnd -= 1;
        }
        else {
            dimensions[j] = self->dimensions[i];
            strides[j++] = self->strides[i];
        }
    }

    Py_INCREF(self->descr);
    ret = PyArray_NewFromDescr(self->ob_type,
                               self->descr,
                               newnd, dimensions,
                               strides, self->data,
                               self->flags,
                               (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    PyArray_FLAGS(ret) &= ~OWNDATA;
    PyArray_BASE(ret) = (PyObject *)self;
    Py_INCREF(self);
    return (PyObject *)ret;
}


/*NUMPY_API
 * Mean
 */
static PyObject *
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
static PyObject *
PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
	    int variance)
{
    return __New_PyArray_Std(self, axis, rtype, out, variance, 0);
}

static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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


/*NUMPY_API
 * Compress
 */
static PyObject *
PyArray_Compress(PyArrayObject *self, PyObject *condition, int axis,
                 PyArrayObject *out)
{
    PyArrayObject *cond;
    PyObject *res, *ret;

    cond = (PyAO *)PyArray_FROM_O(condition);
    if (cond == NULL) {
        return NULL;
    }
    if (cond->nd != 1) {
        Py_DECREF(cond);
        PyErr_SetString(PyExc_ValueError,
                        "condition must be 1-d array");
        return NULL;
    }

    res = PyArray_Nonzero(cond);
    Py_DECREF(cond);
    if (res == NULL) {
        return res;
    }
    ret = PyArray_TakeFrom(self, PyTuple_GET_ITEM(res, 0), axis,
                           out, NPY_RAISE);
    Py_DECREF(res);
    return ret;
}

/*NUMPY_API
 * Nonzero
 */
static PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
    int n = self->nd, j;
    intp count = 0, i, size;
    PyArrayIterObject *it = NULL;
    PyObject *ret = NULL, *item;
    intp *dptr[MAX_DIMS];

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (it == NULL) {
        return NULL;
    }
    size = it->size;
    for (i = 0; i < size; i++) {
        if (self->descr->f->nonzero(it->dataptr, self)) {
            count++;
        }
        PyArray_ITER_NEXT(it);
    }

    PyArray_ITER_RESET(it);
    ret = PyTuple_New(n);
    if (ret == NULL) {
        goto fail;
    }
    for (j = 0; j < n; j++) {
        item = PyArray_New(self->ob_type, 1, &count,
                           PyArray_INTP, NULL, NULL, 0, 0,
                           (PyObject *)self);
        if (item == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(ret, j, item);
        dptr[j] = (intp *)PyArray_DATA(item);
    }
    if (n == 1) {
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                *(dptr[0])++ = i;
            }
            PyArray_ITER_NEXT(it);
        }
    }
    else {
        /* reset contiguous so that coordinates gets updated */
        it->contiguous = 0;
        for (i = 0; i < size; i++) {
            if (self->descr->f->nonzero(it->dataptr, self)) {
                for (j = 0; j < n; j++) {
                    *(dptr[j])++ = it->coordinates[j];
                }
            }
            PyArray_ITER_NEXT(it);
        }
    }

    Py_DECREF(it);
    return ret;

 fail:
    Py_XDECREF(ret);
    Py_XDECREF(it);
    return NULL;

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
static PyObject *
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
static PyObject *
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
static PyObject *
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

/*NUMPY_API
 * Diagonal
 */
static PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int n = self->nd;
    PyObject *new;
    PyArray_Dims newaxes;
    intp dims[MAX_DIMS];
    int i, pos;

    newaxes.ptr = dims;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "array.ndim must be >= 2");
        return NULL;
    }
    if (axis1 < 0) {
        axis1 += n;
    }
    if (axis2 < 0) {
        axis2 += n;
    }
    if ((axis1 == axis2) || (axis1 < 0) || (axis1 >= n) ||
        (axis2 < 0) || (axis2 >= n)) {
        PyErr_Format(PyExc_ValueError, "axis1(=%d) and axis2(=%d) "\
                     "must be different and within range (nd=%d)",
                     axis1, axis2, n);
        return NULL;
    }

    newaxes.len = n;
    /* insert at the end */
    newaxes.ptr[n-2] = axis1;
    newaxes.ptr[n-1] = axis2;
    pos = 0;
    for (i = 0; i < n; i++) {
        if ((i==axis1) || (i==axis2)) {
            continue;
        }
        newaxes.ptr[pos++] = i;
    }
    new = PyArray_Transpose(self, &newaxes);
    if (new == NULL) {
        return NULL;
    }
    self = (PyAO *)new;

    if (n == 2) {
        PyObject *a = NULL, *indices= NULL, *ret = NULL;
        intp n1, n2, start, stop, step, count;
        intp *dptr;

        n1 = self->dimensions[0];
        n2 = self->dimensions[1];
        step = n2 + 1;
        if (offset < 0) {
            start = -n2 * offset;
            stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
        }
        else {
            start = offset;
            stop = MIN(n1, n2-offset)*(n2+1) + offset;
        }

        /* count = ceil((stop-start)/step) */
        count = ((stop-start) / step) + (((stop-start) % step) != 0);
        indices = PyArray_New(&PyArray_Type, 1, &count,
                              PyArray_INTP, NULL, NULL, 0, 0, NULL);
        if (indices == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        dptr = (intp *)PyArray_DATA(indices);
        for (n1 = start; n1 < stop; n1 += step) {
            *dptr++ = n1;
        }
        a = PyArray_IterNew((PyObject *)self);
        Py_DECREF(self);
        if (a == NULL) {
            Py_DECREF(indices);
            return NULL;
        }
        ret = PyObject_GetItem(a, indices);
        Py_DECREF(a);
        Py_DECREF(indices);
        return ret;
    }

    else {
        /*
         * my_diagonal = []
         * for i in range (s [0]) :
         * my_diagonal.append (diagonal (a [i], offset))
         * return array (my_diagonal)
         */
        PyObject *mydiagonal = NULL, *new = NULL, *ret = NULL, *sel = NULL;
        intp i, n1;
        int res;
        PyArray_Descr *typecode;

        typecode = self->descr;
        mydiagonal = PyList_New(0);
        if (mydiagonal == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        n1 = self->dimensions[0];
        for (i = 0; i < n1; i++) {
            new = PyInt_FromLong((long) i);
            sel = PyArray_EnsureAnyArray(PyObject_GetItem((PyObject *)self, new));
            Py_DECREF(new);
            if (sel == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            new = PyArray_Diagonal((PyAO *)sel, offset, n-3, n-2);
            Py_DECREF(sel);
            if (new == NULL) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
            res = PyList_Append(mydiagonal, new);
            Py_DECREF(new);
            if (res < 0) {
                Py_DECREF(self);
                Py_DECREF(mydiagonal);
                return NULL;
            }
        }
        Py_DECREF(self);
        Py_INCREF(typecode);
        ret =  PyArray_FromAny(mydiagonal, typecode, 0, 0, 0, NULL);
        Py_DECREF(mydiagonal);
        return ret;
    }
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
static int
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
static int
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
static int
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
static int
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
static PyObject *
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

/*NUMPY_API
 * SwapAxes
 */
static PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    PyArray_Dims new_axes;
    intp dims[MAX_DIMS];
    int n, i, val;
    PyObject *ret;

    if (a1 == a2) {
        Py_INCREF(ap);
        return (PyObject *)ap;
    }

    n = ap->nd;
    if (n <= 1) {
        Py_INCREF(ap);
        return (PyObject *)ap;
    }

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis1 argument to swapaxes");
        return NULL;
    }
    if ((a2 < 0) || (a2 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis2 argument to swapaxes");
        return NULL;
    }
    new_axes.ptr = dims;
    new_axes.len = n;

    for (i = 0; i < n; i++) {
        if (i == a1) {
            val = a2;
        }
        else if (i == a2) {
            val = a1;
        }
        else {
            val = i;
        }
        new_axes.ptr[i] = val;
    }
    ret = PyArray_Transpose(ap, &new_axes);
    return ret;
}

/*NUMPY_API
 * Return Transpose.
 */
static PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
    intp *axes, axis;
    intp i, n;
    intp permutation[MAX_DIMS], reverse_permutation[MAX_DIMS];
    PyArrayObject *ret = NULL;

    if (permute == NULL) {
        n = ap->nd;
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != ap->nd) {
            PyErr_SetString(PyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            axis = axes[i];
            if (axis < 0) {
                axis = ap->nd + axis;
            }
            if (axis < 0 || axis >= ap->nd) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid axis for this array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                PyErr_SetString(PyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
        for (i = 0; i < n; i++) {
        }
    }

    /*
     * this allocates memory for dimensions and strides (but fills them
     * incorrectly), sets up descr, and points data at ap->data.
     */
    Py_INCREF(ap->descr);
    ret = (PyArrayObject *)\
        PyArray_NewFromDescr(ap->ob_type,
                             ap->descr,
                             n, ap->dimensions,
                             NULL, ap->data, ap->flags,
                             (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    ret->base = (PyObject *)ap;
    Py_INCREF(ap);

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        ret->dimensions[i] = ap->dimensions[permutation[i]];
        ret->strides[i] = ap->strides[permutation[i]];
    }
    PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
    return (PyObject *)ret;
}

/*NUMPY_API
 * Repeat the array.
 */
static PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
    intp *counts;
    intp n, n_outer, i, j, k, chunk, total;
    intp tmp;
    int nd;
    PyArrayObject *repeats = NULL;
    PyObject *ap = NULL;
    PyArrayObject *ret = NULL;
    char *new_data, *old_data;

    repeats = (PyAO *)PyArray_ContiguousFromAny(op, PyArray_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }
    nd = repeats->nd;
    counts = (intp *)repeats->data;

    if ((ap=_check_axis(aop, &axis, CARRAY))==NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    aop = (PyAO *)ap;
    if (nd == 1) {
        n = repeats->dimensions[0];
    }
    else {
        /* nd == 0 */
        n = aop->dimensions[axis];
    }
    if (aop->dimensions[axis] != n) {
        PyErr_SetString(PyExc_ValueError,
                        "a.shape[axis] != len(repeats)");
        goto fail;
    }

    if (nd == 0) {
        total = counts[0]*n;
    }
    else {

        total = 0;
        for (j = 0; j < n; j++) {
            if (counts[j] < 0) {
                PyErr_SetString(PyExc_ValueError, "count < 0");
                goto fail;
            }
            total += counts[j];
        }
    }


    /* Construct new array */
    aop->dimensions[axis] = total;
    Py_INCREF(aop->descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(aop->ob_type,
                                                aop->descr,
                                                aop->nd,
                                                aop->dimensions,
                                                NULL, NULL, 0,
                                                (PyObject *)aop);
    aop->dimensions[axis] = n;
    if (ret == NULL) {
        goto fail;
    }
    new_data = ret->data;
    old_data = aop->data;

    chunk = aop->descr->elsize;
    for(i = axis + 1; i < aop->nd; i++) {
        chunk *= aop->dimensions[i];
    }

    n_outer = 1;
    for (i = 0; i < axis; i++) {
        n_outer *= aop->dimensions[i];
    }
    for (i = 0; i < n_outer; i++) {
        for (j = 0; j < n; j++) {
            tmp = nd ? counts[j] : counts[0];
            for (k = 0; k < tmp; k++) {
                memcpy(new_data, old_data, chunk);
                new_data += chunk;
            }
            old_data += chunk;
        }
    }

    Py_DECREF(repeats);
    PyArray_INCREF(ret);
    Py_XDECREF(aop);
    return (PyObject *)ret;

 fail:
    Py_DECREF(repeats);
    Py_XDECREF(aop);
    Py_XDECREF(ret);
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
static NPY_SCALARKIND
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
static int
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

/* Raises error when len(op) == 0 */

/*NUMPY_API*/
static PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n, allscalars = 0;
    PyArrayObject **mps = NULL;
    PyObject *otmp;
    PyArray_Descr *intype = NULL, *stype = NULL;
    PyArray_Descr *newtype = NULL;
    NPY_SCALARKIND scalarkind = NPY_NOSCALAR, intypekind = NPY_NOSCALAR;

    *retn = n = PySequence_Length(op);
    if (n == 0) {
	PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        return (void*)PyErr_NoMemory();
    }

    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            mps[i] = (PyArrayObject *) array_big_item((PyArrayObject *)op, i);
        }
        if (!PyArray_ISCARRAY(op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        return mps;
    }

    for (i = 0; i < n; i++) {
        otmp = PySequence_GetItem(op, i);
        if (!PyArray_CheckAnyScalar(otmp)) {
            newtype = PyArray_DescrFromObject(otmp, intype);
            Py_XDECREF(intype);
            intype = newtype;
            mps[i] = NULL;
            intypekind = PyArray_ScalarKind(intype->type_num, NULL);
        }
        else {
            newtype = PyArray_DescrFromObject(otmp, stype);
            Py_XDECREF(stype);
            stype = newtype;
            scalarkind = PyArray_ScalarKind(newtype->type_num, NULL);
            mps[i] = (PyArrayObject *)Py_None;
            Py_INCREF(Py_None);
        }
        Py_XDECREF(otmp);
    }
    if (intype==NULL) {
        /* all scalars */
        allscalars = 1;
        intype = stype;
        Py_INCREF(intype);
        for (i = 0; i < n; i++) {
            Py_XDECREF(mps[i]);
            mps[i] = NULL;
        }
    }
    else if ((stype != NULL) && (intypekind != scalarkind)) {
        /*
         * we need to upconvert to type that
         * handles both intype and stype
         * also don't forcecast the scalars.
         */
        if (!PyArray_CanCoerceScalar(stype->type_num,
                                     intype->type_num,
                                     scalarkind)) {
            newtype = _array_small_type(intype, stype);
            Py_XDECREF(intype);
            intype = newtype;
        }
        for (i = 0; i < n; i++) {
            Py_XDECREF(mps[i]);
            mps[i] = NULL;
        }
    }


    /* Make sure all arrays are actual array objects. */
    for (i = 0; i < n; i++) {
        int flags = CARRAY;

        if ((otmp = PySequence_GetItem(op, i)) == NULL) {
            goto fail;
        }
        if (!allscalars && ((PyObject *)(mps[i]) == Py_None)) {
            /* forcecast scalars */
            flags |= FORCECAST;
            Py_DECREF(Py_None);
        }
        Py_INCREF(intype);
        mps[i] = (PyArrayObject*)
            PyArray_FromAny(otmp, intype, 0, 0, flags, NULL);
        Py_DECREF(otmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(intype);
    Py_XDECREF(stype);
    return mps;

 fail:
    Py_XDECREF(intype);
    Py_XDECREF(stype);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
}

/*NUMPY_API
 */
static PyObject *
PyArray_Choose(PyArrayObject *ip, PyObject *op, PyArrayObject *ret,
               NPY_CLIPMODE clipmode)
{
    int n, elsize;
    intp i;
    char *ret_data;
    PyArrayObject **mps, *ap;
    PyArrayMultiIterObject *multi = NULL;
    intp mi;
    int copyret = 0;
    ap = NULL;

    /*
     * Convert all inputs to arrays of a common type
     * Also makes them C-contiguous
     */
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    ap = (PyArrayObject *)PyArray_FROM_OT((PyObject *)ip, NPY_INTP);
    if (ap == NULL) {
        goto fail;
    }
    /* Broadcast all arrays to each other, index array at the end. */ 
    multi = (PyArrayMultiIterObject *)
	PyArray_MultiIterFromObjects((PyObject **)mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    /* Set-up return array */
    if (!ret) {
        Py_INCREF(mps[0]->descr);
        ret = (PyArrayObject *)PyArray_NewFromDescr(ap->ob_type,
                                                    mps[0]->descr,
                                                    multi->nd,
                                                    multi->dimensions,
                                                    NULL, NULL, 0,
                                                    (PyObject *)ap);
    }
    else {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;

        if ((PyArray_NDIM(ret) != multi->nd)
                || !PyArray_CompareLists(
                    PyArray_DIMS(ret), multi->dimensions, multi->nd)) {
	    PyErr_SetString(PyExc_TypeError,
                            "invalid shape for output array.");
            ret = NULL;
            goto fail;
        }
        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        Py_INCREF(mps[0]->descr);
        obj = (PyArrayObject *)PyArray_FromArray(ret, mps[0]->descr, flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
    }

    if (ret == NULL) {
        goto fail;
    }
    elsize = ret->descr->elsize;
    ret_data = ret->data;

    while (PyArray_MultiIter_NOTDONE(multi)) {
	mi = *((intp *)PyArray_MultiIter_DATA(multi, n));
        if (mi < 0 || mi >= n) {
            switch(clipmode) {
            case NPY_RAISE:
                PyErr_SetString(PyExc_ValueError,
                        "invalid entry in choice "\
                        "array");
                goto fail;
            case NPY_WRAP:
                if (mi < 0) {
                    while (mi < 0) {
                        mi += n;
                    }
                }
                else {
                    while (mi >= n) {
                        mi -= n;
                    }
                }
                break;
            case NPY_CLIP:
                if (mi < 0) {
                    mi = 0;
                }
                else if (mi >= n) {
                    mi = n - 1;
                }
                break;
            }
        }
        memmove(ret_data, PyArray_MultiIter_DATA(multi, mi), elsize);
        ret_data += elsize;
	PyArray_MultiIter_NEXT(multi);
    }

    PyArray_INCREF(ret);
    Py_DECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_DECREF(ap);
    PyDataMem_FREE(mps);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Py_INCREF(obj);
        Py_DECREF(ret);
        ret = (PyArrayObject *)obj;
    }
    return (PyObject *)ret;

 fail:
    Py_XDECREF(multi);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    Py_XDECREF(ap);
    PyDataMem_FREE(mps);
    PyArray_XDECREF_ERR(ret);
    return NULL;
}

/*
 * These algorithms use special sorting.  They are not called unless the
 * underlying sort function for the type is available.  Note that axis is
 * already valid. The sort functions require 1-d contiguous and well-behaved
 * data.  Therefore, a copy will be made of the data if needed before handing
 * it to the sorting routine.  An iterator is constructed and adjusted to walk
 * over all but the desired sorting axis.
 */
static int
_new_sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayIterObject *it;
    int needcopy = 0, swap;
    intp N, size;
    int elsize;
    intp astride;
    PyArray_SortFunc *sort;
    BEGIN_THREADS_DEF;

    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    swap = !PyArray_ISNOTSWAPPED(op);
    if (it == NULL) {
        return -1;
    }

    NPY_BEGIN_THREADS_DESCR(op->descr);
    sort = op->descr->f->sort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];

    needcopy = !(op->flags & ALIGNED) || (astride != (intp) elsize) || swap;
    if (needcopy) {
        char *buffer = PyDataMem_NEW(N*elsize);

        while (size--) {
            _unaligned_strided_byte_copy(buffer, (intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(buffer, (intp) elsize, N, elsize);
            }
            if (sort(buffer, N, op) < 0) {
                PyDataMem_FREE(buffer);
                goto fail;
            }
            if (swap) {
                _strided_byte_swap(buffer, (intp) elsize, N, elsize);
            }
            _unaligned_strided_byte_copy(it->dataptr, astride, buffer,
                                         (intp) elsize, N, elsize);
            PyArray_ITER_NEXT(it);
        }
        PyDataMem_FREE(buffer);
    }
    else {
        while (size--) {
            if (sort(it->dataptr, N, op) < 0) {
                goto fail;
            }
            PyArray_ITER_NEXT(it);
        }
    }
    NPY_END_THREADS_DESCR(op->descr);
    Py_DECREF(it);
    return 0;

 fail:
    NPY_END_THREADS;
    Py_DECREF(it);
    return 0;
}

static PyObject*
_new_argsort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{

    PyArrayIterObject *it = NULL;
    PyArrayIterObject *rit = NULL;
    PyObject *ret;
    int needcopy = 0, i;
    intp N, size;
    int elsize, swap;
    intp astride, rstride, *iptr;
    PyArray_ArgSortFunc *argsort;
    BEGIN_THREADS_DEF;

    ret = PyArray_New(op->ob_type, op->nd,
                          op->dimensions, PyArray_INTP,
                          NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        return NULL;
    }
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    rit = (PyArrayIterObject *)PyArray_IterAllButAxis(ret, &axis);
    if (rit == NULL || it == NULL) {
        goto fail;
    }
    swap = !PyArray_ISNOTSWAPPED(op);

    NPY_BEGIN_THREADS_DESCR(op->descr);
    argsort = op->descr->f->argsort[which];
    size = it->size;
    N = op->dimensions[axis];
    elsize = op->descr->elsize;
    astride = op->strides[axis];
    rstride = PyArray_STRIDE(ret,axis);

    needcopy = swap || !(op->flags & ALIGNED) || (astride != (intp) elsize) ||
            (rstride != sizeof(intp));
    if (needcopy) {
        char *valbuffer, *indbuffer;

        valbuffer = PyDataMem_NEW(N*elsize);
        indbuffer = PyDataMem_NEW(N*sizeof(intp));
        while (size--) {
            _unaligned_strided_byte_copy(valbuffer, (intp) elsize, it->dataptr,
                                         astride, N, elsize);
            if (swap) {
                _strided_byte_swap(valbuffer, (intp) elsize, N, elsize);
            }
            iptr = (intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(valbuffer, (intp *)indbuffer, N, op) < 0) {
                PyDataMem_FREE(valbuffer);
                PyDataMem_FREE(indbuffer);
                goto fail;
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(intp), N, sizeof(intp));
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
    }
    else {
        while (size--) {
            iptr = (intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            if (argsort(it->dataptr, (intp *)rit->dataptr, N, op) < 0) {
                goto fail;
            }
            PyArray_ITER_NEXT(it);
            PyArray_ITER_NEXT(rit);
        }
    }

    NPY_END_THREADS_DESCR(op->descr);

    Py_DECREF(it);
    Py_DECREF(rit);
    return ret;

 fail:
    NPY_END_THREADS;
    Py_DECREF(ret);
    Py_XDECREF(it);
    Py_XDECREF(rit);
    return NULL;
}


/* Be sure to save this global_compare when necessary */
static PyArrayObject *global_obj;

static int
qsortCompare (const void *a, const void *b)
{
    return global_obj->descr->f->compare(a,b,global_obj);
}

/*
 * Consumes reference to ap (op gets it) op contains a version of
 * the array with axes swapped if local variable axis is not the
 * last dimension.  Origin must be defined locally.
 */
#define SWAPAXES(op, ap) {                                      \
        orign = (ap)->nd-1;                                     \
        if (axis != orign) {                                    \
            (op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap));                                    \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/*
 * Consumes reference to ap (op gets it) origin must be previously
 * defined locally.  SWAPAXES must have been called previously.
 * op contains the swapped version of the array.
 */
#define SWAPBACK(op, ap) {                                      \
        if (axis != orign) {                                    \
            (op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
            Py_DECREF((ap));                                    \
            if ((op) == NULL) return NULL;                      \
        }                                                       \
        else (op) = (ap);                                       \
    }

/* These swap axes in-place if necessary */
#define SWAPINTP(a,b) {intp c; c=(a); (a) = (b); (b) = c;}
#define SWAPAXES2(ap) {                                                 \
        orign = (ap)->nd-1;                                             \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            PyArray_UpdateFlags(ap, CONTIGUOUS | FORTRAN);              \
        }                                                               \
    }

#define SWAPBACK2(ap) {                                                 \
        if (axis != orign) {                                            \
            SWAPINTP(ap->dimensions[axis], ap->dimensions[orign]);      \
            SWAPINTP(ap->strides[axis], ap->strides[orign]);            \
            PyArray_UpdateFlags(ap, CONTIGUOUS | FORTRAN);              \
        }                                                               \
    }

/*NUMPY_API
 * Sort an array in-place
 */
static int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *ap = NULL, *store_arr = NULL;
    char *ip;
    int i, n, m, elsize, orign;

    n = op->nd;
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        return 0;
    }
    if (axis < 0) {
        axis += n;
    }
    if ((axis < 0) || (axis >= n)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
        return -1;
    }
    if (!PyArray_ISWRITEABLE(op)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "attempted sort on unwriteable array.");
        return -1;
    }

    /* Determine if we should use type-specific algorithm or not */
    if (op->descr->f->sort[which] != NULL) {
        return _new_sort(op, axis, which);
    }
    if ((which != PyArray_QUICKSORT)
        || op->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "desired sort not supported for this type");
        return -1;
    }

    SWAPAXES2(op);

    ap = (PyArrayObject *)PyArray_FromAny((PyObject *)op,
                                          NULL, 1, 0,
                                          DEFAULT | UPDATEIFCOPY, NULL);
    if (ap == NULL) {
        goto fail;
    }
    elsize = ap->descr->elsize;
    m = ap->dimensions[ap->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(ap)/m;

    /* Store global -- allows re-entry -- restore before leaving*/
    store_arr = global_obj;
    global_obj = ap;
    for (ip = ap->data, i = 0; i < n; i++, ip += elsize*m) {
        qsort(ip, m, elsize, qsortCompare);
    }
    global_obj = store_arr;

    if (PyErr_Occurred()) {
        goto fail;
    }

 finish:
    Py_DECREF(ap);  /* Should update op if needed */
    SWAPBACK2(op);
    return 0;

 fail:
    Py_XDECREF(ap);
    SWAPBACK2(op);
    return -1;
}


static char *global_data;

static int
argsort_static_compare(const void *ip1, const void *ip2)
{
    int isize = global_obj->descr->elsize;
    const intp *ipa = ip1;
    const intp *ipb = ip2;
    return global_obj->descr->f->compare(global_data + (isize * *ipa),
                                         global_data + (isize * *ipb),
                                         global_obj);
}

/*NUMPY_API
 * ArgSort an array
 */
static PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *ap = NULL, *ret = NULL, *store, *op2;
    intp *ip;
    intp i, j, n, m, orign;
    int argsort_elsize;
    char *store_ptr;

    n = op->nd;
    if ((n == 0) || (PyArray_SIZE(op) == 1)) {
        ret = (PyArrayObject *)PyArray_New(op->ob_type, op->nd,
                                           op->dimensions,
                                           PyArray_INTP,
                                           NULL, NULL, 0, 0,
                                           (PyObject *)op);
        if (ret == NULL) {
            return NULL;
        }
        *((intp *)ret->data) = 0;
        return (PyObject *)ret;
    }

    /* Creates new reference op2 */
    if ((op2=(PyAO *)_check_axis(op, &axis, 0)) == NULL) {
        return NULL;
    }
    /* Determine if we should use new algorithm or not */
    if (op2->descr->f->argsort[which] != NULL) {
        ret = (PyArrayObject *)_new_argsort(op2, axis, which);
        Py_DECREF(op2);
        return (PyObject *)ret;
    }

    if ((which != PyArray_QUICKSORT) || op2->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "requested sort not available for type");
        Py_DECREF(op2);
        op = NULL;
        goto fail;
    }

    /* ap will contain the reference to op2 */
    SWAPAXES(ap, op2);
    op = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)ap,
                                                    PyArray_NOTYPE,
                                                    1, 0);
    Py_DECREF(ap);
    if (op == NULL) {
        return NULL;
    }
    ret = (PyArrayObject *)PyArray_New(op->ob_type, op->nd,
                                       op->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)op);
    if (ret == NULL) {
        goto fail;
    }
    ip = (intp *)ret->data;
    argsort_elsize = op->descr->elsize;
    m = op->dimensions[op->nd-1];
    if (m == 0) {
        goto finish;
    }
    n = PyArray_SIZE(op)/m;
    store_ptr = global_data;
    global_data = op->data;
    store = global_obj;
    global_obj = op;
    for (i = 0; i < n; i++, ip += m, global_data += m*argsort_elsize) {
        for (j = 0; j < m; j++) {
            ip[j] = j;
        }
        qsort((char *)ip, m, sizeof(intp), argsort_static_compare);
    }
    global_data = store_ptr;
    global_obj = store;

 finish:
    Py_DECREF(op);
    SWAPBACK(op, ret);
    return (PyObject *)op;

 fail:
    Py_XDECREF(op);
    Py_XDECREF(ret);
    return NULL;

}


/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
static PyObject *
PyArray_LexSort(PyObject *sort_keys, int axis)
{
    PyArrayObject **mps;
    PyArrayIterObject **its;
    PyArrayObject *ret = NULL;
    PyArrayIterObject *rit = NULL;
    int n;
    int nd;
    int needcopy=0, i,j;
    intp N, size;
    int elsize;
    int maxelsize;
    intp astride, rstride, *iptr;
    int object = 0;
    PyArray_ArgSortFunc *argsort;
    NPY_BEGIN_THREADS_DEF;

    if (!PySequence_Check(sort_keys)
           || ((n=PySequence_Size(sort_keys)) <= 0)) {
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");
        return NULL;
    }
    mps = (PyArrayObject **) _pya_malloc(n*sizeof(PyArrayObject));
    if (mps == NULL) {
        return PyErr_NoMemory();
    }
    its = (PyArrayIterObject **) _pya_malloc(n*sizeof(PyArrayIterObject));
    if (its == NULL) {
        _pya_free(mps);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
        its[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject *obj;
        obj = PySequence_GetItem(sort_keys, i);
        mps[i] = (PyArrayObject *)PyArray_FROM_O(obj);
        Py_DECREF(obj);
        if (mps[i] == NULL) {
            goto fail;
        }
        if (i > 0) {
            if ((mps[i]->nd != mps[0]->nd)
                || (!PyArray_CompareLists(mps[i]->dimensions,
                                       mps[0]->dimensions,
                                       mps[0]->nd))) {
                PyErr_SetString(PyExc_ValueError,
                                "all keys need to be the same shape");
                goto fail;
            }
        }
        if (!mps[i]->descr->f->argsort[PyArray_MERGESORT]) {
            PyErr_Format(PyExc_TypeError,
                         "merge sort not available for item %d", i);
            goto fail;
        }
        if (!object
            && PyDataType_FLAGCHK(mps[i]->descr, NPY_NEEDS_PYAPI)) {
            object = 1;
        }
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis
            ((PyObject *)mps[i], &axis);
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now we can check the axis */
    nd = mps[0]->nd;
    if ((nd == 0) || (PyArray_SIZE(mps[0]) == 1)) {
        ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                           mps[0]->dimensions,
                                           PyArray_INTP,
                                           NULL, NULL, 0, 0, NULL);

        if (ret == NULL) {
            goto fail;
        }
        *((intp *)(ret->data)) = 0;
        goto finish;
    }
    if (axis < 0) {
        axis += nd;
    }
    if ((axis < 0) || (axis >= nd)) {
        PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
        goto fail;
    }

    /* Now do the sorting */
    ret = (PyArrayObject *)PyArray_New(&PyArray_Type, mps[0]->nd,
                                       mps[0]->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, NULL);
    if (ret == NULL) {
        goto fail;
    }
    rit = (PyArrayIterObject *)
            PyArray_IterAllButAxis((PyObject *)ret, &axis);
    if (rit == NULL) {
        goto fail;
    }
    if (!object) {
        NPY_BEGIN_THREADS;
    }
    size = rit->size;
    N = mps[0]->dimensions[axis];
    rstride = PyArray_STRIDE(ret,axis);
    maxelsize = mps[0]->descr->elsize;
    needcopy = (rstride != sizeof(intp));
    for (j = 0; j < n && !needcopy; j++) {
        needcopy = PyArray_ISBYTESWAPPED(mps[j])
            || !(mps[j]->flags & ALIGNED)
            || (mps[j]->strides[axis] != (intp)mps[j]->descr->elsize);
        if (mps[j]->descr->elsize > maxelsize) {
            maxelsize = mps[j]->descr->elsize;
        }
    }

    if (needcopy) {
        char *valbuffer, *indbuffer;
        int *swaps;

        valbuffer = PyDataMem_NEW(N*maxelsize);
        indbuffer = PyDataMem_NEW(N*sizeof(intp));
        swaps = malloc(n*sizeof(int));
        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }
        while (size--) {
            iptr = (intp *)indbuffer;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                elsize = mps[j]->descr->elsize;
                astride = mps[j]->strides[axis];
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                _unaligned_strided_byte_copy(valbuffer, (intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (intp) elsize, N, elsize);
                }
                if (argsort(valbuffer, (intp *)indbuffer, N, mps[j]) < 0) {
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(intp), N, sizeof(intp));
            PyArray_ITER_NEXT(rit);
        }
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            iptr = (intp *)rit->dataptr;
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            for (j = 0; j < n; j++) {
                argsort = mps[j]->descr->f->argsort[PyArray_MERGESORT];
                if (argsort(its[j]->dataptr, (intp *)rit->dataptr,
                            N, mps[j]) < 0) {
                    goto fail;
                }
                PyArray_ITER_NEXT(its[j]);
            }
            PyArray_ITER_NEXT(rit);
        }
    }

    if (!object) {
        NPY_END_THREADS;
    }

 finish:
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    Py_XDECREF(rit);
    _pya_free(mps);
    _pya_free(its);
    return (PyObject *)ret;

 fail:
    NPY_END_THREADS;
    Py_XDECREF(rit);
    Py_XDECREF(ret);
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    _pya_free(mps);
    _pya_free(its);
    return NULL;
}


/** @brief Use bisection of sorted array to find first entries >= keys.
 *
 * For each key use bisection to find the first index i s.t. key <= arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_left(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for (i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 2);
            if (compare(parr + elsize*imid, pkey, key) < 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/** @brief Use bisection of sorted array to find first entries > keys.
 *
 * For each key use bisection to find the first index i s.t. key < arr[i].
 * When there is no such index i, set i = len(arr). Return the results in ret.
 * All arrays are assumed contiguous on entry and both arr and key must be of
 * the same comparable type.
 *
 * @param arr contiguous sorted array to be searched.
 * @param key contiguous array of keys.
 * @param ret contiguous array of intp for returned indices.
 * @return void
 */
static void
local_search_right(PyArrayObject *arr, PyArrayObject *key, PyArrayObject *ret)
{
    PyArray_CompareFunc *compare = key->descr->f->compare;
    intp nelts = arr->dimensions[arr->nd - 1];
    intp nkeys = PyArray_SIZE(key);
    char *parr = arr->data;
    char *pkey = key->data;
    intp *pret = (intp *)ret->data;
    int elsize = arr->descr->elsize;
    intp i;

    for(i = 0; i < nkeys; ++i) {
        intp imin = 0;
        intp imax = nelts;
        while (imin < imax) {
            intp imid = imin + ((imax - imin) >> 2);
            if (compare(parr + elsize*imid, pkey, key) <= 0) {
                imin = imid + 1;
            }
            else {
                imax = imid;
            }
        }
        *pret = imin;
        pret += 1;
        pkey += elsize;
    }
}


/*NUMPY_API
 * Convert object to searchsorted side
 */
static int
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


/*NUMPY_API
 * Numeric.searchsorted(a,v)
 */
static PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2, NPY_SEARCHSIDE side)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    NPY_BEGIN_THREADS_DEF;

    dtype = PyArray_DescrFromObject((PyObject *)op2, op1->descr);
    /* need ap1 as contiguous array and of right type */
    Py_INCREF(dtype);
    ap1 = (PyArrayObject *)PyArray_FromAny((PyObject *)op1, dtype,
					   1, 1, NPY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* need ap2 as contiguous array and of right type */
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, dtype,
                                          0, 0, NPY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    /* ret is a contiguous array of intp type to hold returned indices */
    ret = (PyArrayObject *)PyArray_New(ap2->ob_type, ap2->nd,
                                       ap2->dimensions, PyArray_INTP,
                                       NULL, NULL, 0, 0, (PyObject *)ap2);
    if (ret == NULL) {
        goto fail;
    }
    /* check that comparison function exists */
    if (ap2->descr->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "compare not supported for type");
        goto fail;
    }

    if (side == NPY_SEARCHLEFT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_left(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
    }
    else if (side == NPY_SEARCHRIGHT) {
        NPY_BEGIN_THREADS_DESCR(ap2->descr);
        local_search_right(ap1, ap2, ret);
        NPY_END_THREADS_DESCR(ap2->descr);
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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


/*NUMPY_API
 * Take
 */
static PyObject *
PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                 PyArrayObject *ret, NPY_CLIPMODE clipmode)
{
    PyArray_FastTakeFunc *func;
    PyArrayObject *self, *indices;
    intp nd, i, j, n, m, max_item, tmp, chunk, nelem;
    intp shape[MAX_DIMS];
    char *src, *dest;
    int copyret = 0;
    int err;

    indices = NULL;
    self = (PyAO *)_check_axis(self0, &axis, CARRAY);
    if (self == NULL) {
        return NULL;
    }
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         PyArray_INTP,
                                                         1, 0);
    if (indices == NULL) {
        goto fail;
    }
    n = m = chunk = 1;
    nd = self->nd + indices->nd - 1;
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = self->dimensions[i];
            n *= shape[i];
        }
        else {
            if (i < axis+indices->nd) {
                shape[i] = indices->dimensions[i-axis];
                m *= shape[i];
            }
            else {
                shape[i] = self->dimensions[i-indices->nd+1];
                chunk *= shape[i];
            }
        }
    }
    Py_INCREF(self->descr);
    if (!ret) {
        ret = (PyArrayObject *)PyArray_NewFromDescr(self->ob_type,
                                                    self->descr,
                                                    nd, shape,
                                                    NULL, NULL, 0,
                                                    (PyObject *)self);

        if (ret == NULL) {
            goto fail;
        }
    }
    else {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if ((ret->nd != nd) ||
            !PyArray_CompareLists(ret->dimensions, shape, nd)) {
            PyErr_SetString(PyExc_ValueError,
                            "bad shape in output array");
            ret = NULL;
            Py_DECREF(self->descr);
            goto fail;
        }

        if (clipmode == NPY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= NPY_ENSURECOPY;
        }
        obj = (PyArrayObject *)PyArray_FromArray(ret, self->descr,
                                                 flags);
        if (obj != ret) {
            copyret = 1;
        }
        ret = obj;
	if (ret == NULL) {
            goto fail;
        }
    }

    max_item = self->dimensions[axis];
    nelem = chunk;
    chunk = chunk * ret->descr->elsize;
    src = self->data;
    dest = ret->data;

    func = self->descr->f->fasttake;
    if (func == NULL) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = tmp + max_item;
                    }
                    if ((tmp < 0) || (tmp >= max_item)) {
                        PyErr_SetString(PyExc_IndexError,
                                "index out of range "\
                                "for array");
                        goto fail;
                    }
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
                    if (tmp < 0) {
                        while (tmp < 0) {
                            tmp += max_item;
                        }
                    }
                    else if (tmp >= max_item) {
                        while (tmp >= max_item) {
                            tmp -= max_item;
                        }
                    }
                    memmove(dest, src + tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    tmp = ((intp *)(indices->data))[j];
                    if (tmp < 0) {
                        tmp = 0;
                    }
                    else if (tmp >= max_item) {
                        tmp = max_item - 1;
                    }
                    memmove(dest, src+tmp*chunk, chunk);
                    dest += chunk;
                }
                src += chunk*max_item;
            }
            break;
        }
    }
    else {
        err = func(dest, src, (intp *)(indices->data),
                    max_item, n, m, nelem, clipmode);
        if (err) {
            goto fail;
        }
    }

    PyArray_INCREF(ret);
    Py_XDECREF(indices);
    Py_XDECREF(self);
    if (copyret) {
        PyObject *obj;
        obj = ret->base;
        Py_INCREF(obj);
        Py_DECREF(ret);
        ret = (PyArrayObject *)obj;
    }
    return (PyObject *)ret;

 fail:
    PyArray_XDECREF_ERR(ret);
    Py_XDECREF(indices);
    Py_XDECREF(self);
    return NULL;
}

/*NUMPY_API
 * Put values into an array
 */
static PyObject *
PyArray_PutTo(PyArrayObject *self, PyObject* values0, PyObject *indices0,
              NPY_CLIPMODE clipmode)
{
    PyArrayObject  *indices, *values;
    int i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    indices = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "put: first argument must be an array");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        if (clipmode == NPY_RAISE) {
            flags |= NPY_ENSURECOPY;
        }
        Py_INCREF(self->descr);
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }
    max_item = PyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                         PyArray_INTP, 0, 0);
    if (indices == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(indices);
    Py_INCREF(self->descr);
    values = (PyArrayObject *)PyArray_FromAny(values0, self->descr, 0, 0,
                                              DEFAULT | FORCECAST, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values);
    if (nv <= 0) {
        goto finish;
    }
    if (PyDataType_REFCHK(self->descr)) {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk*(i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    PyErr_SetString(PyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp*chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+tmp*chunk, self->descr);
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }
    else {
        switch(clipmode) {
        case NPY_RAISE:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = tmp + max_item;
                }
                if ((tmp < 0) || (tmp >= max_item)) {
                    PyErr_SetString(PyExc_IndexError,
                            "index out of " \
                            "range for array");
                    goto fail;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_WRAP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        case NPY_CLIP:
            for (i = 0; i < ni; i++) {
                src = values->data + chunk * (i % nv);
                tmp = ((intp *)(indices->data))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                memmove(dest + tmp * chunk, src, chunk);
            }
            break;
        }
    }

 finish:
    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        Py_DECREF(self);
    }
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        PyArray_XDECREF_ERR(self);
    }
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
 * Put values into an array according to a mask.
 */
static PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0)
{
    PyArray_FastPutmaskFunc *func;
    PyArrayObject  *mask, *values;
    int i, chunk, ni, max_item, nv, tmp;
    char *src, *dest;
    int copied = 0;

    mask = NULL;
    values = NULL;
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "\
                        "be an array");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(self)) {
        PyArrayObject *obj;
        int flags = NPY_CARRAY | NPY_UPDATEIFCOPY;

        Py_INCREF(self->descr);
        obj = (PyArrayObject *)PyArray_FromArray(self,
                                                 self->descr, flags);
        if (obj != self) {
            copied = 1;
        }
        self = obj;
    }

    max_item = PyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;
    mask = (PyArrayObject *)\
        PyArray_FROM_OTF(mask0, PyArray_BOOL, CARRAY | FORCECAST);
    if (mask == NULL) {
        goto fail;
    }
    ni = PyArray_SIZE(mask);
    if (ni != max_item) {
        PyErr_SetString(PyExc_ValueError,
                        "putmask: mask and data must be "\
                        "the same size");
        goto fail;
    }
    Py_INCREF(self->descr);
    values = (PyArrayObject *)\
        PyArray_FromAny(values0, self->descr, 0, 0, NPY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;
    }
    nv = PyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        Py_XDECREF(values);
        Py_XDECREF(mask);
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyDataType_REFCHK(self->descr)) {
        for (i = 0; i < ni; i++) {
            tmp = ((Bool *)(mask->data))[i];
            if (tmp) {
                src = values->data + chunk * (i % nv);
                PyArray_Item_INCREF(src, self->descr);
                PyArray_Item_XDECREF(dest+i*chunk, self->descr);
                memmove(dest + i * chunk, src, chunk);
            }
        }
    }
    else {
        func = self->descr->f->fastputmask;
        if (func == NULL) {
            for (i = 0; i < ni; i++) {
                tmp = ((Bool *)(mask->data))[i];
                if (tmp) {
                    src = values->data + chunk*(i % nv);
                    memmove(dest + i*chunk, src, chunk);
                }
            }
        }
        else {
            func(dest, mask->data, ni, values->data, nv);
        }
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    if (copied) {
        Py_DECREF(self);
    }
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(mask);
    Py_XDECREF(values);
    if (copied) {
        PyArray_XDECREF_ERR(self);
    }
    return NULL;
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
static int
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
static int
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
static int
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
static int
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
static int
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
static int
PyArray_TypestrConvert(int itemsize, int gentype)
{
    register int newtype = gentype;

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
static int
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
static int
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


/*
 * A tuple type would be either (generic typeobject, typesize)
 * or (fixed-length data-type, shape)
 *
 * or (inheriting data-type, new-data-type)
 * The new data-type must have the same itemsize as the inheriting data-type
 * unless the latter is 0
 *
 * Thus (int32, {'real':(int16,0),'imag',(int16,2)})
 *
 * is one way to specify a descriptor that will give
 * a['real'] and a['imag'] to an int32 array.
 *
 * leave type reference alone
 */
static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag)
{
    PyArray_Descr *new;
    PyArray_Descr *conv;

    *errflag = 0;
    if (!PyArray_DescrConverter(newobj, &conv)) {
        return NULL;
    }
    *errflag = 1;
    new = PyArray_DescrNew(type);
    if (new == NULL) {
        goto fail;
    }
    if (new->elsize && new->elsize != conv->elsize) {
        PyErr_SetString(PyExc_ValueError,
                        "mismatch in size of old "\
                        "and new data-descriptor");
        goto fail;
    }
    new->elsize = conv->elsize;
    if (conv->names) {
        new->fields = conv->fields;
        Py_XINCREF(new->fields);
        new->names = conv->names;
        Py_XINCREF(new->names);
    }
    new->hasobject = conv->hasobject;
    Py_DECREF(conv);
    *errflag = 0;
    return new;

 fail:
    Py_DECREF(conv);
    return NULL;
}

static PyArray_Descr *
_convert_from_tuple(PyObject *obj)
{
    PyArray_Descr *type, *res;
    PyObject *val;
    int errflag;

    if (PyTuple_GET_SIZE(obj) != 2) {
        return NULL;
    }
    if (!PyArray_DescrConverter(PyTuple_GET_ITEM(obj,0), &type)) {
        return NULL;
    }
    val = PyTuple_GET_ITEM(obj,1);
    /* try to interpret next item as a type */
    res = _use_inherit(type, val, &errflag);
    if (res || errflag) {
        Py_DECREF(type);
        if (res) {
            return res;
        }
        else {
            return NULL;
        }
    }
    PyErr_Clear();
    /*
     * We get here if res was NULL but errflag wasn't set
     * --- i.e. the conversion to a data-descr failed in _use_inherit
     */
    if (type->elsize == 0) {
        /* interpret next item as a typesize */
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));

        if (error_converting(itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid itemsize in generic type "\
                            "tuple");
            goto fail;
        }
        PyArray_DESCR_REPLACE(type);
        if (type->type_num == PyArray_UNICODE) {
            type->elsize = itemsize << 2;
        }
        else {
            type->elsize = itemsize;
        }
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to PyArray_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        PyArray_Descr *newdescr;

        if (!(PyArray_IntpConverter(val, &shape))
            || (shape.len > MAX_DIMS)) {
            PyDimMem_FREE(shape.ptr);
            PyErr_SetString(PyExc_ValueError,
                            "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /* If (type, 1) was given, it is equivalent to type...
           or (type, ()) was given it is equivalent to type... */
        if ((shape.len == 1 && shape.ptr[0] == 1 && PyNumber_Check(val))
            || (shape.len == 0 && PyTuple_Check(val))) {
            PyDimMem_FREE(shape.ptr);
            return type;
        }
        newdescr = PyArray_DescrNewFromType(PyArray_VOID);
        if (newdescr == NULL) {
            PyDimMem_FREE(shape.ptr);
            goto fail;
        }
        newdescr->elsize = type->elsize;
        newdescr->elsize *= PyArray_MultiplyList(shape.ptr, shape.len);
        PyDimMem_FREE(shape.ptr);
        newdescr->subarray = _pya_malloc(sizeof(PyArray_ArrayDescr));
        newdescr->subarray->base = type;
        newdescr->hasobject = type->hasobject;
        Py_INCREF(val);
        newdescr->subarray->shape = val;
        Py_XDECREF(newdescr->fields);
        Py_XDECREF(newdescr->names);
        newdescr->fields = NULL;
        newdescr->names = NULL;
        type = newdescr;
    }
    return type;

 fail:
    Py_XDECREF(type);
    return NULL;
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 */
static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    int n, i, totalsize;
    int ret;
    PyObject *fields, *item, *newobj;
    PyObject *name, *tup, *title;
    PyObject *nameslist;
    PyArray_Descr *new;
    PyArray_Descr *conv;
    int dtypeflags = 0;
    int maxalign = 0;


    n = PyList_GET_SIZE(obj);
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    totalsize = 0;
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        item = PyList_GET_ITEM(obj, i);
        if (!PyTuple_Check(item) || (PyTuple_GET_SIZE(item) < 2)) {
            goto fail;
        }
        name = PyTuple_GET_ITEM(item, 0);
        if (PyString_Check(name)) {
            title = NULL;
        }
        else if (PyTuple_Check(name)) {
            if (PyTuple_GET_SIZE(name) != 2) {
                goto fail;
            }
            title = PyTuple_GET_ITEM(name, 0);
            name = PyTuple_GET_ITEM(name, 1);
            if (!PyString_Check(name)) {
                goto fail;
            }
        }
        else {
            goto fail;
        }
        if (PyString_GET_SIZE(name)==0) {
            if (title == NULL) {
                name = PyString_FromFormat("f%d", i);
            }
            else {
                name = title;
                Py_INCREF(name);
            }
        }
        else {
            Py_INCREF(name);
        }
        PyTuple_SET_ITEM(nameslist, i, name);
        if (PyTuple_GET_SIZE(item) == 2) {
            ret = PyArray_DescrConverter(PyTuple_GET_ITEM(item, 1), &conv);
            if (ret == PY_FAIL) {
                PyObject_Print(PyTuple_GET_ITEM(item,1), stderr, 0);
            }
        }
        else if (PyTuple_GET_SIZE(item) == 3) {
            newobj = PyTuple_GetSlice(item, 1, 3);
            ret = PyArray_DescrConverter(newobj, &conv);
            Py_DECREF(newobj);
        }
        else {
            goto fail;
        }
        if (ret == PY_FAIL) {
            goto fail;
        }
        if ((PyDict_GetItem(fields, name) != NULL) ||
            (title &&
             (PyString_Check(title) || PyUnicode_Check(title)) &&
             (PyDict_GetItem(fields, title) != NULL))) {
            PyErr_SetString(PyExc_ValueError,
                            "two fields with the same name");
            goto fail;
        }
        dtypeflags |= (conv->hasobject & NPY_FROM_FIELDS);
        tup = PyTuple_New((title == NULL ? 2 : 3));
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         */
        if (title != NULL) {
            Py_INCREF(title);
            PyTuple_SET_ITEM(tup, 2, title);
            if (PyString_Check(title) || PyUnicode_Check(title)) {
                PyDict_SetItem(fields, title, tup);
            }
        }
        PyDict_SetItem(fields, name, tup);
        totalsize += conv->elsize;
        Py_DECREF(tup);
    }
    new = PyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->hasobject=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    return new;

 fail:
    Py_DECREF(fields);
    Py_DECREF(nameslist);
    return NULL;

}

/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
    int n, i;
    int totalsize;
    PyObject *fields;
    PyArray_Descr *conv = NULL;
    PyArray_Descr *new;
    PyObject *key, *tup;
    PyObject *nameslist = NULL;
    int ret;
    int maxalign = 0;
    int dtypeflags = 0;

    n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    key = PyList_GET_ITEM(obj, n-1);
    if (PyString_Check(key) && PyString_GET_SIZE(key) == 0) {
        n = n - 1;
    }
    /* End ignore code.*/
    totalsize = 0;
    if (n == 0) {
        return NULL;
    }
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        tup = PyTuple_New(2);
        key = PyString_FromFormat("f%d", i);
        ret = PyArray_DescrConverter(PyList_GET_ITEM(obj, i), &conv);
        if (ret == PY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(key);
            goto fail;
        }
        dtypeflags |= (conv->hasobject & NPY_FROM_FIELDS);
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));
        PyDict_SetItem(fields, key, tup);
        Py_DECREF(tup);
        PyTuple_SET_ITEM(nameslist, i, key);
        totalsize += conv->elsize;
    }
    new = PyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->hasobject=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    return new;

 fail:
    Py_DECREF(nameslist);
    Py_DECREF(fields);
    return NULL;
}


/*
 * comma-separated string
 * this is the format developed by the numarray records module
 * and implemented by the format parser in that module
 * this is an alternative implementation found in the _internal.py
 * file patterned after that one -- the approach is to try to convert
 * to a list (with tuples if any repeat information is present)
 * and then call the _convert_from_list)
 */
static PyArray_Descr *
_convert_from_commastring(PyObject *obj, int align)
{
    PyObject *listobj;
    PyArray_Descr *res;
    PyObject *_numpy_internal;

    if (!PyString_Check(obj)) {
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    listobj = PyObject_CallMethod(_numpy_internal, "_commastring", "O", obj);
    Py_DECREF(_numpy_internal);
    if (!listobj) {
        return NULL;
    }
    if (!PyList_Check(listobj) || PyList_GET_SIZE(listobj)<1) {
        PyErr_SetString(PyExc_RuntimeError, "_commastring is "  \
                        "not returning a list with len >= 1");
        return NULL;
    }
    if (PyList_GET_SIZE(listobj) == 1) {
        if (PyArray_DescrConverter(PyList_GET_ITEM(listobj, 0),
                                   &res) == NPY_FAIL) {
            res = NULL;
        }
    }
    else {
        res = _convert_from_list(listobj, align);
    }
    Py_DECREF(listobj);
    if (!res && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "invalid data-type");
        return NULL;
    }
    return res;
}



/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
 *
 * "names" --- field names
 * "formats" --- the data-type descriptors for the field.
 *
 * Optional:
 *
 * "offsets" --- integers indicating the offset into the
 * record of the start of the field.
 * if not given, then "consecutive offsets"
 * will be assumed and placed in the dictionary.
 *
 * "titles" --- Allows the use of an additional key
 * for the fields dictionary.(if these are strings
 * or unicode objects) or
 * this can also be meta-data to
 * be passed around with the field description.
 *
 * Attribute-lookup-based field names merely has to query the fields
 * dictionary of the data-descriptor.  Any result present can be used
 * to return the correct field.
 *
 * So, the notion of what is a name and what is a title is really quite
 * arbitrary.
 *
 * What does distinguish a title, however, is that if it is not None,
 * it will be placed at the end of the tuple inserted into the
 * fields dictionary.and can therefore be used to carry meta-data around.
 *
 * If the dictionary does not have "names" and "formats" entries,
 * then it will be checked for conformity and used directly.
 */
static PyArray_Descr *
_use_fields_dict(PyObject *obj, int align)
{
    PyObject *_numpy_internal;
    PyArray_Descr *res;

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = (PyArray_Descr *)PyObject_CallMethod(_numpy_internal,
                                               "_usefields",
                                               "Oi", obj, align);
    Py_DECREF(_numpy_internal);
    return res;
}

static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
    PyArray_Descr *new;
    PyObject *fields = NULL;
    PyObject *names, *offsets, *descrs, *titles;
    int n, i;
    int totalsize;
    int maxalign = 0;
    int dtypeflags = 0;

    fields = PyDict_New();
    if (fields == NULL) {
        return (PyArray_Descr *)PyErr_NoMemory();
    }
    names = PyDict_GetItemString(obj, "names");
    descrs = PyDict_GetItemString(obj, "formats");
    if (!names || !descrs) {
        Py_DECREF(fields);
        return _use_fields_dict(obj, align);
    }
    n = PyObject_Length(names);
    offsets = PyDict_GetItemString(obj, "offsets");
    titles = PyDict_GetItemString(obj, "titles");
    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                        "all items in the dictionary must have" \
                        " the same length.");
        goto fail;
    }

    totalsize = 0;
    for (i = 0; i < n; i++) {
        PyObject *tup, *descr, *index, *item, *name, *off;
        int len, ret, _align = 1;
        PyArray_Descr *newdescr;

        /* Build item to insert (descr, offset, [title])*/
        len = 2;
        item = NULL;
        index = PyInt_FromLong(i);
        if (titles) {
            item=PyObject_GetItem(titles, index);
            if (item && item != Py_None) {
                len = 3;
            }
            else {
                Py_XDECREF(item);
            }
            PyErr_Clear();
        }
        tup = PyTuple_New(len);
        descr = PyObject_GetItem(descrs, index);
        ret = PyArray_DescrConverter(descr, &newdescr);
        Py_DECREF(descr);
        if (ret == PY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(index);
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)newdescr);
        if (align) {
            _align = newdescr->alignment;
            maxalign = MAX(maxalign,_align);
        }
        if (offsets) {
            long offset;
            off = PyObject_GetItem(offsets, index);
            offset = PyInt_AsLong(off);
            PyTuple_SET_ITEM(tup, 1, off);
            if (offset < totalsize) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid offset (must be "\
                                "ordered)");
                ret = PY_FAIL;
            }
            if (offset > totalsize) totalsize = offset;
        }
        else {
            if (align && _align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(totalsize));
        }
        if (len == 3) {
            PyTuple_SET_ITEM(tup, 2, item);
        }
        name = PyObject_GetItem(names, index);
        Py_DECREF(index);
        if (!(PyString_Check(name) || PyUnicode_Check(name))) {
            PyErr_SetString(PyExc_ValueError,
                            "field names must be strings");
            ret = PY_FAIL;
        }

        /* Insert into dictionary */
        if (PyDict_GetItem(fields, name) != NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "name already used as a name or "\
                            "title");
            ret = PY_FAIL;
        }
        PyDict_SetItem(fields, name, tup);
        Py_DECREF(name);
        if (len == 3) {
            if ((PyString_Check(item) || PyUnicode_Check(item))
                && PyDict_GetItem(fields, item) != NULL) {
                PyErr_SetString(PyExc_ValueError,
                                "title already used as a "\
                                "name or title.");
                ret=PY_FAIL;
            }
            else {
                PyDict_SetItem(fields, item, tup);
            }
        }
        Py_DECREF(tup);
        if ((ret == PY_FAIL) || (newdescr->elsize == 0)) {
            goto fail;
        }
        dtypeflags |= (newdescr->hasobject & NPY_FROM_FIELDS);
        totalsize += newdescr->elsize;
    }

    new = PyArray_DescrNewFromType(PyArray_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = ((totalsize + maxalign - 1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    if (!PyTuple_Check(names)) {
        names = PySequence_Tuple(names);
    }
    else {
        Py_INCREF(names);
    }
    new->names = names;
    new->fields = fields;
    new->hasobject = dtypeflags;
    return new;

 fail:
    Py_XDECREF(fields);
    return NULL;
}

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static int
_check_for_commastring(char *type, int len)
{
    int i;

    /* Check for ints at start of string */
    if ((type[0] >= '0' && type[0] <= '9') ||
        ((len > 1) && _chk_byteorder(type[0]) &&
         (type[1] >= '0' && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1) && (type[0] == '(' && type[1] == ')')) ||
        ((len > 3) && _chk_byteorder(type[0]) &&
         (type[1] == '(' && type[2] == ')'))) {
        return 1;
    }
    /* Check for presence of commas */
    for (i = 1; i < len; i++) {
        if (type[i] == ',') {
            return 1;
        }
    }
    return 0;
}

#undef _chk_byteorder

/*NUMPY_API
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from PyArray_VOID.
*/
static int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    if PyDict_Check(obj) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if PyString_Check(obj) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if PyList_Check(obj) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
static int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if PyDict_Check(obj) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if PyString_Check(obj) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if PyList_Check(obj) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter2(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}


/*NUMPY_API
 * Get typenum from an object -- None goes to NULL
 */
static int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return PY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}

/*NUMPY_API
 * Get typenum from an object -- None goes to PyArray_DEFAULT
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 * new reference in *at
 */
static int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    char *type;
    int check_num = PyArray_NOTYPE + 10;
    int len;
    PyObject *item;
    int elsize = 0;
    char endian = '=';

    *at = NULL;
    /* default */
    if (obj == Py_None) {
        *at = PyArray_DescrFromType(PyArray_DEFAULT);
        return PY_SUCCEED;
    }
    if (PyArray_DescrCheck(obj)) {
        *at = (PyArray_Descr *)obj;
        Py_INCREF(*at);
        return PY_SUCCEED;
    }

    if (PyType_Check(obj)) {
        if (PyType_IsSubtype((PyTypeObject *)obj,
                             &PyGenericArrType_Type)) {
            *at = PyArray_DescrFromTypeObject(obj);
            if (*at) {
                return PY_SUCCEED;
            }
            else {
                return PY_FAIL;
            }
        }
        check_num = PyArray_OBJECT;
        if (obj == (PyObject *)(&PyInt_Type)) {
            check_num = PyArray_LONG;
        }
        else if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = PyArray_LONGLONG;
        }
        else if (obj == (PyObject *)(&PyFloat_Type)) {
            check_num = PyArray_DOUBLE;
        }
        else if (obj == (PyObject *)(&PyComplex_Type)) {
            check_num = PyArray_CDOUBLE;
        }
        else if (obj == (PyObject *)(&PyBool_Type)) {
            check_num = PyArray_BOOL;
        }
        else if (obj == (PyObject *)(&PyString_Type)) {
            check_num = PyArray_STRING;
        }
        else if (obj == (PyObject *)(&PyUnicode_Type)) {
            check_num = PyArray_UNICODE;
        }
        else if (obj == (PyObject *)(&PyBuffer_Type)) {
            check_num = PyArray_VOID;
        }
        else {
            *at = _arraydescr_fromobj(obj);
            if (*at) {
                return PY_SUCCEED;
            }
        }
        goto finish;
    }

    /* or a typecode string */
    if (PyString_Check(obj)) {
        /* Check for a string typecode. */
        type = PyString_AS_STRING(obj);
        len = PyString_GET_SIZE(obj);
        if (len <= 0) {
            goto fail;
        }
        /* check for commas present or first (or second) element a digit */
        if (_check_for_commastring(type, len)) {
            *at = _convert_from_commastring(obj, 0);
            if (*at) {
                return PY_SUCCEED;
            }
            return PY_FAIL;
        }
        check_num = (int) type[0];
        if ((char) check_num == '>' || (char) check_num == '<'
            || (char) check_num == '|' || (char) check_num == '=') {
            if (len <= 1) {
                goto fail;
            }
            endian = (char) check_num;
            type++; len--;
            check_num = (int) type[0];
            if (endian == '|') {
                endian = '=';
            }
        }
        if (len > 1) {
            elsize = atoi(type + 1);
            if (elsize == 0) {
                check_num = PyArray_NOTYPE+10;
            }
            /*
             * When specifying length of UNICODE
             * the number of characters is given to match
             * the STRING interface.  Each character can be
             * more than one byte and itemsize must be
             * the number of bytes.
             */
            else if (check_num == PyArray_UNICODELTR) {
                elsize <<= 2;
            }
            /* Support for generic processing c4, i4, f8, etc...*/
            else if ((check_num != PyArray_STRINGLTR)
                     && (check_num != PyArray_VOIDLTR)
                     && (check_num != PyArray_STRINGLTR2)) {
                check_num = PyArray_TypestrConvert(elsize, check_num);
                if (check_num == PyArray_NOTYPE) {
                    check_num += 10;
                }
                elsize = 0;
            }
        }
    }
    else if (PyTuple_Check(obj)) {
        /* or a tuple */
        *at = _convert_from_tuple(obj);
        if (*at == NULL){
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyList_Check(obj)) {
        /* or a list */
        *at = _convert_from_array_descr(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyDict_Check(obj)) {
        /* or a dictionary */
        *at = _convert_from_dict(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyArray_Check(obj)) {
        goto fail;
    }
    else {
        *at = _arraydescr_fromobj(obj);
        if (*at) {
            return PY_SUCCEED;
        }
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
        goto fail;
    }
    if (PyErr_Occurred()) {
        goto fail;
    }
    /*
      if (check_num == PyArray_NOTYPE) return PY_FAIL;
    */

 finish:
    if ((check_num == PyArray_NOTYPE + 10)
        || (*at = PyArray_DescrFromType(check_num)) == NULL) {
        /* Now check to see if the object is registered in typeDict */
        if (typeDict != NULL) {
            item = PyDict_GetItem(typeDict, obj);
            if (item) {
                return PyArray_DescrConverter(item, at);
            }
        }
        goto fail;
    }

    if (((*at)->elsize == 0) && (elsize != 0)) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->elsize = elsize;
    }
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && (*at)->byteorder != '|'
        && (*at)->byteorder != endian) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->byteorder = endian;
    }
    return PY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError, "data type not understood");
    *at = NULL;
    return PY_FAIL;
}

/*NUMPY_API
 * Convert object to endian
 */
static int
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
static int
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
static unsigned char
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
static unsigned char
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

/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals referenct to type
 */
static PyObject *
PyArray_Empty(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    if (!type) type = PyArray_DescrFromType(PyArray_DEFAULT);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type, nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (PyDataType_REFCHK(type)) {
        PyArray_FillObjectArray(ret, Py_None);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return NULL;
        }
    }
    return (PyObject *)ret;
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

/*NUMPY_API
 * Zeros
 *
 * steal a reference
 * accepts NULL type
 */
static PyObject *
PyArray_Zeros(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
    PyArrayObject *ret;

    if (!type) {
        type = PyArray_DescrFromType(PyArray_DEFAULT);
    }
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type,
                                                nd, dims,
                                                NULL, NULL,
                                                fortran, NULL);
    if (ret == NULL) {
        return NULL;
    }
    if (_zerofill(ret) < 0) {
        return NULL;
    }
    return (PyObject *)ret;

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
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O", &dict)) {
        return NULL;
    }
    /* Decrement old reference (if any)*/
    Py_XDECREF(typeDict);
    typeDict = dict;
    /* Create an internal reference to it */
    Py_INCREF(dict);
    Py_INCREF(Py_None);
    return Py_None;
}


/*
 * Reading from a file or a string.
 *
 * As much as possible, we try to use the same code for both files and strings,
 * so the semantics for fromstring and fromfile are the same, especially with
 * regards to the handling of text representations.
 */

typedef int (*next_element)(void **, void *, PyArray_Descr *, void *);
typedef int (*skip_separator)(void **, const char *, void *);

static int
fromstr_next_element(char **s, void *dptr, PyArray_Descr *dtype,
                     const char *end)
{
    int r = dtype->f->fromstr(*s, dptr, s, dtype);
    if (end != NULL && *s > end) {
        return -1;
    }
    return r;
}

static int
fromfile_next_element(FILE **fp, void *dptr, PyArray_Descr *dtype,
                      void *NPY_UNUSED(stream_data))
{
    /* the NULL argument is for backwards-compatibility */
    return dtype->f->scanfunc(*fp, dptr, NULL, dtype);
}

/*
 * Remove multiple whitespace from the separator, and add a space to the
 * beginning and end. This simplifies the separator-skipping code below.
 */
static char *
swab_separator(char *sep)
{
    int skip_space = 0;
    char *s, *start;

    s = start = malloc(strlen(sep)+3);
    /* add space to front if there isn't one */
    if (*sep != '\0' && !isspace(*sep)) {
        *s = ' '; s++;
    }
    while (*sep != '\0') {
        if (isspace(*sep)) {
            if (skip_space) {
                sep++;
            }
            else {
                *s = ' ';
                s++;
                sep++;
                skip_space = 1;
            }
        }
        else {
            *s = *sep;
            s++;
            sep++;
            skip_space = 0;
        }
    }
    /* add space to end if there isn't one */
    if (s != start && s[-1] == ' ') {
        *s = ' ';
        s++;
    }
    *s = '\0';
    return start;
}

/*
 * Assuming that the separator is the next bit in the string (file), skip it.
 *
 * Single spaces in the separator are matched to arbitrary-long sequences
 * of whitespace in the input. If the separator consists only of spaces,
 * it matches one or more whitespace characters.
 *
 * If we can't match the separator, return -2.
 * If we hit the end of the string (file), return -1.
 * Otherwise, return 0.
 */
static int
fromstr_skip_separator(char **s, const char *sep, const char *end)
{
    char *string = *s;
    int result = 0;
    while (1) {
        char c = *string;
        if (c == '\0' || (end != NULL && string >= end)) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            if (string != *s) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                continue;
            }
        }
        else if (*sep != c) {
            result = -2;
            break;
        }
        else {
            sep++;
        }
        string++;
    }
    *s = string;
    return result;
}

static int
fromfile_skip_separator(FILE **fp, const char *sep, void *NPY_UNUSED(stream_data))
{
    int result = 0;
    const char *sep_start = sep;

    while (1) {
        int c = fgetc(*fp);

        if (c == EOF) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            ungetc(c, *fp);
            if (sep != sep_start) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                sep_start++;
                ungetc(c, *fp);
            }
            else if (sep == sep_start) {
                sep_start--;
            }
        }
        else if (*sep != c) {
            ungetc(c, *fp);
            result = -2;
            break;
        }
        else {
            sep++;
        }
    }
    return result;
}

/*
 * Create an array by reading from the given stream, using the passed
 * next_element and skip_separator functions.
 */
#define FROM_BUFFER_SIZE 4096
static PyArrayObject *
array_from_text(PyArray_Descr *dtype, intp num, char *sep, size_t *nread,
                void *stream, next_element next, skip_separator skip_sep,
                void *stream_data)
{
    PyArrayObject *r;
    intp i;
    char *dptr, *clean_sep, *tmp;
    int err = 0;
    intp thisbuf = 0;
    intp size;
    intp bytes, totalbytes;

    size = (num >= 0) ? num : FROM_BUFFER_SIZE;
    r = (PyArrayObject *)
        PyArray_NewFromDescr(&PyArray_Type,
                             dtype,
                             1, &size,
                             NULL, NULL,
                             0, NULL);
    if (r == NULL) {
        return NULL;
    }
    clean_sep = swab_separator(sep);
    NPY_BEGIN_ALLOW_THREADS;
    totalbytes = bytes = size * dtype->elsize;
    dptr = r->data;
    for (i= 0; num < 0 || i < num; i++) {
        if (next(&stream, dptr, dtype, stream_data) < 0) {
            break;
        }
        *nread += 1;
        thisbuf += 1;
        dptr += dtype->elsize;
        if (num < 0 && thisbuf == size) {
            totalbytes += bytes;
            tmp = PyDataMem_RENEW(r->data, totalbytes);
            if (tmp == NULL) {
                err = 1;
                break;
            }
	    r->data = tmp;
            dptr = tmp + (totalbytes - bytes);
            thisbuf = 0;
        }
        if (skip_sep(&stream, clean_sep, stream_data) < 0) {
            break;
        }
    }
    if (num < 0) {
        tmp = PyDataMem_RENEW(r->data, (*nread)*dtype->elsize);
	if (tmp == NULL) {
            err = 1;
        }
	else {
	    PyArray_DIM(r,0) = *nread;
	    r->data = tmp;
	}
    }
    NPY_END_ALLOW_THREADS;
    free(clean_sep);
    if (err == 1) {
        PyErr_NoMemory();
    }
    if (PyErr_Occurred()) {
        Py_DECREF(r);
        return NULL;
    }
    return r;
}
#undef FROM_BUFFER_SIZE

/*NUMPY_API
 *
 * Given a pointer to a string ``data``, a string length ``slen``, and
 * a ``PyArray_Descr``, return an array corresponding to the data
 * encoded in that string.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * If ``slen`` is < 0, then the end of string is used for text data.
 * It is an error for ``slen`` to be < 0 for binary data (since embedded NULLs
 * would be the norm).
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 */
static PyObject *
PyArray_FromString(char *data, intp slen, PyArray_Descr *dtype,
                   intp num, char *sep)
{
    int itemsize;
    PyArrayObject *ret;
    Bool binary;

    if (dtype == NULL) {
        dtype=PyArray_DescrFromType(PyArray_DEFAULT);
    }
    if (PyDataType_FLAGCHK(dtype, NPY_ITEM_IS_POINTER)) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot create an object array from"    \
                        " a string");
        Py_DECREF(dtype);
        return NULL;
    }
    itemsize = dtype->elsize;
    if (itemsize == 0) {
        PyErr_SetString(PyExc_ValueError, "zero-valued itemsize");
        Py_DECREF(dtype);
        return NULL;
    }

    binary = ((sep == NULL) || (strlen(sep) == 0));
    if (binary) {
        if (num < 0 ) {
            if (slen % itemsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                                "string size must be a "\
                                "multiple of element size");
                Py_DECREF(dtype);
                return NULL;
            }
            num = slen/itemsize;
        }
        else {
            if (slen < num*itemsize) {
                PyErr_SetString(PyExc_ValueError,
                                "string is smaller than " \
                                "requested size");
                Py_DECREF(dtype);
                return NULL;
            }
        }
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, dtype,
                                 1, &num, NULL, NULL,
                                 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
        memcpy(ret->data, data, num*dtype->elsize);
    }
    else {
        /* read from character-based string */
        size_t nread = 0;
        char *end;

        if (dtype->f->scanfunc == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "don't know how to read "       \
                            "character strings with that "  \
                            "array type");
            Py_DECREF(dtype);
            return NULL;
        }
        if (slen < 0) {
            end = NULL;
        }
        else {
            end = data + slen;
        }
        ret = array_from_text(dtype, num, sep, &nread,
                              data,
                              (next_element) fromstr_next_element,
                              (skip_separator) fromstr_skip_separator,
                              end);
    }
    return (PyObject *)ret;
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



static PyArrayObject *
array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, intp num, size_t *nread)
{
    PyArrayObject *r;
    intp start, numbytes;

    if (num < 0) {
        int fail = 0;

        start = (intp )ftell(fp);
        if (start < 0) {
            fail = 1;
        }
        if (fseek(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = (intp) ftell(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (fseek(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
        if (fail) {
            PyErr_SetString(PyExc_IOError,
                            "could not seek in file");
            Py_DECREF(dtype);
            return NULL;
        }
        num = numbytes / dtype->elsize;
    }
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                              dtype,
                                              1, &num,
                                              NULL, NULL,
                                              0, NULL);
    if (r == NULL) {
        return NULL;
    }
    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(r->data, dtype->elsize, num, fp);
    NPY_END_ALLOW_THREADS;
    return r;
}

/*NUMPY_API
 *
 * Given a ``FILE *`` pointer ``fp``, and a ``PyArray_Descr``, return an
 * array corresponding to the data encoded in that file.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 *
 * For memory-mapped files, use the buffer interface. No more data than
 * necessary is read by this routine.
 */
static PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *dtype, intp num, char *sep)
{
    PyArrayObject *ret;
    size_t nread = 0;
    char *tmp;

    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "0-sized elements.");
        Py_DECREF(dtype);
        return NULL;
    }
    if ((sep == NULL) || (strlen(sep) == 0)) {
        ret = array_fromfile_binary(fp, dtype, num, &nread);
    }
    else {
        if (dtype->f->scanfunc == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "don't know how to read "       \
                            "character files with that "    \
                            "array type");
            Py_DECREF(dtype);
            return NULL;
        }
        ret = array_from_text(dtype, num, sep, &nread,
                              fp,
                              (next_element) fromfile_next_element,
                              (skip_separator) fromfile_skip_separator,
                              NULL);
    }
    if (((intp) nread) < num) {
        fprintf(stderr, "%ld items requested but only %ld read\n",
                (long) num, (long) nread);
	/* Make sure realloc is > 0 */
	tmp = PyDataMem_RENEW(ret->data,
			      NPY_MAX(nread,1) * ret->descr->elsize);
	/* FIXME: This should not raise a memory error when nread == 0
	   We should return an empty array or at least raise an  EOF Error.
	 */
        if ((tmp == NULL) || (nread == 0)) {
	    Py_DECREF(ret);
	    return PyErr_NoMemory();
	}
	ret->data = tmp;
        PyArray_DIM(ret,0) = nread;
    }
    return (PyObject *)ret;
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


/*NUMPY_API
 *
 * steals a reference to dtype (which cannot be NULL)
 */
static PyObject *
PyArray_FromIter(PyObject *obj, PyArray_Descr *dtype, intp count)
{
    PyObject *value;
    PyObject *iter = PyObject_GetIter(obj);
    PyArrayObject *ret = NULL;
    intp i, elsize, elcount;
    char *item, *new_data;

    if (iter == NULL) {
        goto done;
    }
    elcount = (count < 0) ? 0 : count;
    if ((elsize=dtype->elsize) == 0) {
        PyErr_SetString(PyExc_ValueError, "Must specify length "\
                        "when using variable-size data-type.");
        goto done;
    }

    /*
     * We would need to alter the memory RENEW code to decrement any
     * reference counts before throwing away any memory.
     */
    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError, "cannot create "\
                        "object arrays from iterator");
        goto done;
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1,
                                                &elcount, NULL,NULL, 0, NULL);
    dtype = NULL;
    if (ret == NULL) {
        goto done;
    }
    for (i = 0; (i < count || count == -1) &&
             (value = PyIter_Next(iter)); i++) {
        if (i >= elcount) {
            /*
              Grow ret->data:
              this is similar for the strategy for PyListObject, but we use
              50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            if (elcount <= (intp)((~(size_t)0) / elsize)) {
                new_data = PyDataMem_RENEW(ret->data, elcount * elsize);
            }
            else {
                new_data = NULL;
            }
            if (new_data == NULL) {
                PyErr_SetString(PyExc_MemoryError,
                                "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            ret->data = new_data;
        }
        ret->dimensions[0] = i + 1;

        if (((item = index2ptr(ret, i)) == NULL)
            || (ret->descr->f->setitem(value, item, ret) == -1)) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }

    if (i < count) {
        PyErr_SetString(PyExc_ValueError, "iterator too short");
        goto done;
    }

    /*
     * Realloc the data so that don't keep extra memory tied up
     * (assuming realloc is reasonably good about reusing space...)
     */
    if (i == 0) {
        i = 1;
    }
    new_data = PyDataMem_RENEW(ret->data, i * elsize);
    if (new_data == NULL) {
        PyErr_SetString(PyExc_MemoryError, "cannot allocate array memory");
        goto done;
    }
    ret->data = new_data;

 done:
    Py_XDECREF(iter);
    Py_XDECREF(dtype);
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
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


/*NUMPY_API*/
static PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   intp count, intp offset)
{
    PyArrayObject *ret;
    char *data;
    Py_ssize_t ts;
    intp s, n;
    int itemsize;
    int write = 1;


    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory"\
                        " buffer");
        Py_DECREF(type);
        return NULL;
    }
    if (type->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;
    }
    if (buf->ob_type->tp_as_buffer == NULL
        || (buf->ob_type->tp_as_buffer->bf_getwritebuffer == NULL
            && buf->ob_type->tp_as_buffer->bf_getreadbuffer == NULL)) {
        PyObject *newbuf;
        newbuf = PyObject_GetAttrString(buf, "__buffer__");
        if (newbuf == NULL) {
            Py_DECREF(type);
            return NULL;
        }
        buf = newbuf;
    }
    else {
        Py_INCREF(buf);
    }

    if (PyObject_AsWriteBuffer(buf, (void *)&data, &ts) == -1) {
        write = 0;
        PyErr_Clear();
        if (PyObject_AsReadBuffer(buf, (void *)&data, &ts) == -1) {
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((offset < 0) || (offset >= ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be positive and smaller than %"
                     INTP_FMT, (intp)ts);
    }

    data += offset;
    s = (intp)ts - offset;
    n = (intp)count;
    itemsize = type->elsize;
    if (n < 0 ) {
        if (s % itemsize != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size must be a multiple"\
                            " of element size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
        n = s/itemsize;
    }
    else {
        if (s < n*itemsize) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer is smaller than requested"\
                            " size");
            Py_DECREF(buf);
            Py_DECREF(type);
            return NULL;
        }
    }

    if ((ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                     type,
                                                     1, &n,
                                                     NULL, data,
                                                     DEFAULT,
                                                     NULL)) == NULL) {
        Py_DECREF(buf);
        return NULL;
    }

    if (!write) {
        ret->flags &= ~WRITEABLE;
    }
    /* Store a reference for decref on deallocation */
    ret->base = buf;
    PyArray_UpdateFlags(ret, ALIGNED);
    return (PyObject *)ret;
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

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static int _safe_ceil_to_intp(double value, intp* ret)
{
    double ivalue;

    ivalue = npy_ceil(value);
    if (ivalue < NPY_MIN_INTP || ivalue > NPY_MAX_INTP) {
        return -1;
    }

    *ret = (intp)ivalue;
    return 0;
}


/*NUMPY_API
  Arange,
*/
static PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    intp length;
    PyObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *obj;
    int ret;

    if (_safe_ceil_to_intp((stop - start)/step, &length)) {
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);
    }
    range = PyArray_New(&PyArray_Type, 1, &length, type_num,
                        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyArray_DESCR(range)->f;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_DATA(range), (PyArrayObject *)range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                         (PyArrayObject *)range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    funcs->fill(PyArray_DATA(range), length, (PyArrayObject *)range);
    if (PyErr_Occurred()) {
        goto fail;
    }
    return range;

 fail:
    Py_DECREF(range);
    return NULL;
}

/*
 * the formula is len = (intp) ceil((start - stop) / step);
 */
static intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    intp len, tmp;
    PyObject *val;
    double value;

    *next = PyNumber_Subtract(stop, start);
    if (!(*next)) {
        if (PyTuple_Check(stop)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");
        }
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;
    if (!val) {
        return -1;
    }
    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            Py_DECREF(val);
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &tmp)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
        len = MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        if (_safe_ceil_to_intp(value, &len)) {
            PyErr_SetString(PyExc_OverflowError,
                    "arange: overflow while computing length");
            return -1;
        }
    }
    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!next) {
            return -1;
        }
    }
    return len;
}

/*NUMPY_API
 *
 * ArangeObj,
 *
 * this doesn't change the references
 */
static PyObject *
PyArray_ArangeObj(PyObject *start, PyObject *stop, PyObject *step, PyArray_Descr *dtype)
{
    PyObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *next, *err;
    intp length;
    PyArray_Descr *native = NULL;
    int swap;

    if (!dtype) {
        PyArray_Descr *deftype;
        PyArray_Descr *newtype;
        /* intentionally made to be PyArray_LONG default */
        deftype = PyArray_DescrFromType(PyArray_LONG);
        newtype = PyArray_DescrFromObject(start, deftype);
        Py_DECREF(deftype);
        deftype = newtype;
        if (stop && stop != Py_None) {
            newtype = PyArray_DescrFromObject(stop, deftype);
            Py_DECREF(deftype);
            deftype = newtype;
        }
        if (step && step != Py_None) {
            newtype = PyArray_DescrFromObject(step, deftype);
            Py_DECREF(deftype);
            deftype = newtype;
        }
        dtype = deftype;
    }
    else {
        Py_INCREF(dtype);
    }
    if (!step || step == Py_None) {
        step = PyInt_FromLong(1);
    }
    else {
        Py_XINCREF(step);
    }
    if (!stop || stop == Py_None) {
        stop = start;
        start = PyInt_FromLong(0);
    }
    else {
        Py_INCREF(start);
    }
    /* calculate the length and next = start + step*/
    length = _calc_length(start, stop, step, &next,
                          PyTypeNum_ISCOMPLEX(dtype->type_num));
    err = PyErr_Occurred(); 
    if (err) {
        Py_DECREF(dtype);
        if (err && PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    if (length <= 0) {
        length = 0;
        range = PyArray_SimpleNewFromDescr(1, &length, dtype);
        Py_DECREF(step);
        Py_DECREF(start);
        return range;
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    if (!PyArray_ISNBO(dtype->byteorder)) {
        native = PyArray_DescrNewByteorder(dtype, PyArray_NATBYTE);
        swap = 1;
    }
    else {
        native = dtype;
        swap = 0;
    }

    range = PyArray_SimpleNewFromDescr(1, &length, native);
    if (range == NULL) {
        goto fail;
    }

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    funcs = PyArray_DESCR(range)->f;
    if (funcs->setitem(
                start, PyArray_DATA(range), (PyArrayObject *)range) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       (PyArrayObject *)range) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        goto fail;
    }
    funcs->fill(PyArray_DATA(range), length, (PyArrayObject *)range);
    if (PyErr_Occurred()) {
        goto fail;
    }
 finish:
    if (swap) {
        PyObject *new;
        new = PyArray_Byteswap((PyArrayObject *)range, 1);
        Py_DECREF(new);
        Py_DECREF(PyArray_DESCR(range));
        PyArray_DESCR(range) = dtype;  /* steals the reference */
    }
    Py_DECREF(start);
    Py_DECREF(step);
    Py_DECREF(next);
    return range;

 fail:
    Py_DECREF(start);
    Py_DECREF(step);
    Py_XDECREF(next);
    return NULL;
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
static unsigned int
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
static int
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
static PyObject *
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
static void
_PyArray_SigintHandler(int signum)
{
    PyOS_setsig(signum, SIG_IGN);
    SIGLONGJMP(_NPY_SIGINT_BUF, signum);
}

/*NUMPY_API
 */
static void*
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
