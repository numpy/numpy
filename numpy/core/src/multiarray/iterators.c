#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "arrayobject.h"
#include "iterators.h"
#include "ctors.h"
#include "common.h"

#define NEWAXIS_INDEX -1
#define ELLIPSIS_INDEX -2
#define SINGLE_INDEX -3

/*
 * Tries to convert 'o' into an npy_intp interpreted as an
 * index. Returns 1 if it was successful, 0 otherwise. Does
 * not set an exception.
 */
static int
coerce_index(PyObject *o, npy_intp *v)
{
    *v = PyArray_PyIntAsIntp(o);

    if ((*v) == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

/*
 * This function converts one element of the indexing tuple
 * into a step size and a number of steps, returning the
 * starting index. Non-slices are signalled in 'n_steps',
 * as NEWAXIS_INDEX, ELLIPSIS_INDEX, or SINGLE_INDEX.
 */
NPY_NO_EXPORT npy_intp
parse_index_entry(PyObject *op, npy_intp *step_size,
                  npy_intp *n_steps, npy_intp max,
                  int axis, int check_index)
{
    npy_intp i;

    if (op == Py_None) {
        *n_steps = NEWAXIS_INDEX;
        i = 0;
    }
    else if (op == Py_Ellipsis) {
        *n_steps = ELLIPSIS_INDEX;
        i = 0;
    }
    else if (PySlice_Check(op)) {
        npy_intp stop;
        if (NpySlice_GetIndicesEx(op, max, &i, &stop, step_size, n_steps) < 0) {
            goto fail;
        }
        if (*n_steps <= 0) {
            *n_steps = 0;
            *step_size = 1;
            i = 0;
        }
    }
    else if (coerce_index(op, &i)) {
        *n_steps = SINGLE_INDEX;
        *step_size = 0;
        if (check_index) {
            if (check_and_adjust_index(&i, max, axis, NULL) < 0) {
                goto fail;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_IndexError,
                        "each index entry must be either a "
                        "slice, an integer, Ellipsis, or "
                        "newaxis");
        goto fail;
    }
    return i;

 fail:
    return -1;
}


/*
 * Parses an index that has no fancy indexing. Populates
 * out_dimensions, out_strides, and out_offset.
 */
NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            npy_intp *out_dimensions,
            npy_intp *out_strides,
            npy_intp *out_offset,
            int check_index)
{
    int i, j, n;
    int nd_old, nd_new, n_add, n_ellipsis;
    npy_intp n_steps, start, offset, step_size;
    PyObject *op1 = NULL;
    int is_slice;

    if (PySlice_Check(op) || op == Py_Ellipsis || op == Py_None) {
        n = 1;
        op1 = op;
        Py_INCREF(op);
        /* this relies on the fact that n==1 for loop below */
        is_slice = 1;
    }
    else {
        if (!PySequence_Check(op)) {
            PyErr_SetString(PyExc_IndexError,
                            "index must be either an int "
                            "or a sequence");
            return -1;
        }
        n = PySequence_Length(op);
        is_slice = 0;
    }

    nd_old = nd_new = 0;

    offset = 0;
    for (i = 0; i < n; i++) {
        if (!is_slice) {
            op1 = PySequence_GetItem(op, i);
            if (op1 == NULL) {
                return -1;
            }
        }
        start = parse_index_entry(op1, &step_size, &n_steps,
                                  nd_old < PyArray_NDIM(self) ?
                                  PyArray_DIMS(self)[nd_old] : 0,
                                  nd_old, check_index ?
                                  nd_old < PyArray_NDIM(self) : 0);
        Py_DECREF(op1);
        if (start == -1) {
            break;
        }
        if (n_steps == NEWAXIS_INDEX) {
            out_dimensions[nd_new] = 1;
            out_strides[nd_new] = 0;
            nd_new++;
        }
        else if (n_steps == ELLIPSIS_INDEX) {
            for (j = i + 1, n_ellipsis = 0; j < n; j++) {
                op1 = PySequence_GetItem(op, j);
                if (op1 == Py_None) {
                    n_ellipsis++;
                }
                Py_DECREF(op1);
            }
            n_add = PyArray_NDIM(self)-(n-i-n_ellipsis-1+nd_old);
            if (n_add < 0) {
                PyErr_SetString(PyExc_IndexError, "too many indices");
                return -1;
            }
            for (j = 0; j < n_add; j++) {
                out_dimensions[nd_new] = PyArray_DIMS(self)[nd_old];
                out_strides[nd_new] = PyArray_STRIDES(self)[nd_old];
                nd_new++; nd_old++;
            }
        }
        else {
            if (nd_old >= PyArray_NDIM(self)) {
                PyErr_SetString(PyExc_IndexError, "too many indices");
                return -1;
            }
            offset += PyArray_STRIDES(self)[nd_old]*start;
            nd_old++;
            if (n_steps != SINGLE_INDEX) {
                out_dimensions[nd_new] = n_steps;
                out_strides[nd_new] = step_size *
                                            PyArray_STRIDES(self)[nd_old-1];
                nd_new++;
            }
        }
    }
    if (i < n) {
        return -1;
    }
    n_add = PyArray_NDIM(self)-nd_old;
    for (j = 0; j < n_add; j++) {
        out_dimensions[nd_new] = PyArray_DIMS(self)[nd_old];
        out_strides[nd_new] = PyArray_STRIDES(self)[nd_old];
        nd_new++;
        nd_old++;
    }
    *out_offset = offset;
    return nd_new;
}


/*********************** Element-wise Array Iterator ***********************/
/*  Aided by Peter J. Verveer's  nd_image package and numpy's arraymap  ****/
/*         and Python's array iterator                                   ***/

/* get the dataptr from its current coordinates for simple iterator */
static char*
get_ptr_simple(PyArrayIterObject* iter, npy_intp *coordinates)
{
    npy_intp i;
    char *ret;

    ret = PyArray_DATA(iter->ao);

    for(i = 0; i < PyArray_NDIM(iter->ao); ++i) {
            ret += coordinates[i] * iter->strides[i];
    }

    return ret;
}

/*
 * This is common initialization code between PyArrayIterObject and
 * PyArrayNeighborhoodIterObject
 *
 * Increase ao refcount
 */
static PyObject *
array_iter_base_init(PyArrayIterObject *it, PyArrayObject *ao)
{
    int nd, i;

    nd = PyArray_NDIM(ao);
    PyArray_UpdateFlags(ao, NPY_ARRAY_C_CONTIGUOUS);
    if (PyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    Py_INCREF(ao);
    it->ao = ao;
    it->size = PyArray_SIZE(ao);
    it->nd_m1 = nd - 1;
    if (nd != 0) {
        it->factors[nd-1] = 1;
    }
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = PyArray_DIMS(ao)[i] - 1;
        it->strides[i] = PyArray_STRIDES(ao)[i];
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * PyArray_DIMS(ao)[nd-i];
        }
        it->bounds[i][0] = 0;
        it->bounds[i][1] = PyArray_DIMS(ao)[i] - 1;
        it->limits[i][0] = 0;
        it->limits[i][1] = PyArray_DIMS(ao)[i] - 1;
        it->limits_sizes[i] = it->limits[i][1] - it->limits[i][0] + 1;
    }

    it->translate = &get_ptr_simple;
    PyArray_ITER_RESET(it);

    return (PyObject *)it;
}

static void
array_iter_base_dealloc(PyArrayIterObject *it)
{
    Py_XDECREF(it->ao);
}

/*NUMPY_API
 * Get Iterator.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterNew(PyObject *obj)
{
    PyArrayIterObject *it;
    PyArrayObject *ao;

    if (!PyArray_Check(obj)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    ao = (PyArrayObject *)obj;

    it = (PyArrayIterObject *)PyArray_malloc(sizeof(PyArrayIterObject));
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);
    /* it = PyObject_New(PyArrayIterObject, &PyArrayIter_Type);*/
    if (it == NULL) {
        return NULL;
    }

    array_iter_base_init(it, ao);
    return (PyObject *)it;
}

/*NUMPY_API
 * Get Iterator broadcast to a particular shape
 */
NPY_NO_EXPORT PyObject *
PyArray_BroadcastToShape(PyObject *obj, npy_intp *dims, int nd)
{
    PyArrayIterObject *it;
    int i, diff, j, compat, k;
    PyArrayObject *ao = (PyArrayObject *)obj;

    if (PyArray_NDIM(ao) > nd) {
        goto err;
    }
    compat = 1;
    diff = j = nd - PyArray_NDIM(ao);
    for (i = 0; i < PyArray_NDIM(ao); i++, j++) {
        if (PyArray_DIMS(ao)[i] == 1) {
            continue;
        }
        if (PyArray_DIMS(ao)[i] != dims[j]) {
            compat = 0;
            break;
        }
    }
    if (!compat) {
        goto err;
    }
    it = (PyArrayIterObject *)PyArray_malloc(sizeof(PyArrayIterObject));
    if (it == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);

    PyArray_UpdateFlags(ao, NPY_ARRAY_C_CONTIGUOUS);
    if (PyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    Py_INCREF(ao);
    it->ao = ao;
    it->size = PyArray_MultiplyList(dims, nd);
    it->nd_m1 = nd - 1;
    if (nd != 0) {
        it->factors[nd-1] = 1;
    }
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = dims[i] - 1;
        k = i - diff;
        if ((k < 0) || PyArray_DIMS(ao)[k] != dims[i]) {
            it->contiguous = 0;
            it->strides[i] = 0;
        }
        else {
            it->strides[i] = PyArray_STRIDES(ao)[k];
        }
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * dims[nd-i];
        }
    }
    PyArray_ITER_RESET(it);
    return (PyObject *)it;

 err:
    PyErr_SetString(PyExc_ValueError, "array is not broadcastable to "\
                    "correct shape");
    return NULL;
}





/*NUMPY_API
 * Get Iterator that iterates over all but one axis (don't use this with
 * PyArray_ITER_GOTO1D).  The axis will be over-written if negative
 * with the axis having the smallest stride.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterAllButAxis(PyObject *obj, int *inaxis)
{
    PyArrayObject *arr;
    PyArrayIterObject *it;
    int axis;

    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_ValueError,
                "Numpy IterAllButAxis requires an ndarray");
        return NULL;
    }
    arr = (PyArrayObject *)obj;

    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (it == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(arr)==0) {
        return (PyObject *)it;
    }
    if (*inaxis < 0) {
        int i, minaxis = 0;
        npy_intp minstride = 0;
        i = 0;
        while (minstride == 0 && i < PyArray_NDIM(arr)) {
            minstride = PyArray_STRIDE(arr,i);
            i++;
        }
        for (i = 1; i < PyArray_NDIM(arr); i++) {
            if (PyArray_STRIDE(arr,i) > 0 &&
                PyArray_STRIDE(arr, i) < minstride) {
                minaxis = i;
                minstride = PyArray_STRIDE(arr,i);
            }
        }
        *inaxis = minaxis;
    }
    axis = *inaxis;
    /* adjust so that will not iterate over axis */
    it->contiguous = 0;
    if (it->size != 0) {
        it->size /= PyArray_DIM(arr,axis);
    }
    it->dims_m1[axis] = 0;
    it->backstrides[axis] = 0;

    /*
     * (won't fix factors so don't use
     * PyArray_ITER_GOTO1D with this iterator)
     */
    return (PyObject *)it;
}

/*NUMPY_API
 * Adjusts previously broadcasted iterators so that the axis with
 * the smallest sum of iterator strides is not iterated over.
 * Returns dimension which is smallest in the range [0,multi->nd).
 * A -1 is returned if multi->nd == 0.
 *
 * don't use with PyArray_ITER_GOTO1D because factors are not adjusted
 */
NPY_NO_EXPORT int
PyArray_RemoveSmallest(PyArrayMultiIterObject *multi)
{
    PyArrayIterObject *it;
    int i, j;
    int axis;
    npy_intp smallest;
    npy_intp sumstrides[NPY_MAXDIMS];

    if (multi->nd == 0) {
        return -1;
    }
    for (i = 0; i < multi->nd; i++) {
        sumstrides[i] = 0;
        for (j = 0; j < multi->numiter; j++) {
            sumstrides[i] += multi->iters[j]->strides[i];
        }
    }
    axis = 0;
    smallest = sumstrides[0];
    /* Find longest dimension */
    for (i = 1; i < multi->nd; i++) {
        if (sumstrides[i] < smallest) {
            axis = i;
            smallest = sumstrides[i];
        }
    }
    for(i = 0; i < multi->numiter; i++) {
        it = multi->iters[i];
        it->contiguous = 0;
        if (it->size != 0) {
            it->size /= (it->dims_m1[axis]+1);
        }
        it->dims_m1[axis] = 0;
        it->backstrides[axis] = 0;
    }
    multi->size = multi->iters[0]->size;
    return axis;
}

/* Returns an array scalar holding the element desired */

static PyObject *
arrayiter_next(PyArrayIterObject *it)
{
    PyObject *ret;

    if (it->index < it->size) {
        ret = PyArray_ToScalar(it->dataptr, it->ao);
        PyArray_ITER_NEXT(it);
        return ret;
    }
    return NULL;
}

static void
arrayiter_dealloc(PyArrayIterObject *it)
{
    array_iter_base_dealloc(it);
    PyArray_free(it);
}

static Py_ssize_t
iter_length(PyArrayIterObject *self)
{
    return self->size;
}


static PyArrayObject *
iter_subscript_Bool(PyArrayIterObject *self, PyArrayObject *ind)
{
    npy_intp counter, strides;
    int itemsize;
    npy_intp count = 0;
    char *dptr, *optr;
    PyArrayObject *ret;
    int swap;
    PyArray_CopySwapFunc *copyswap;


    if (PyArray_NDIM(ind) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return NULL;
    }
    counter = PyArray_DIMS(ind)[0];
    if (counter > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "too many boolean indices");
        return NULL;
    }

    strides = PyArray_STRIDES(ind)[0];
    dptr = PyArray_DATA(ind);
    /* Get size of return array */
    while (counter--) {
        if (*((npy_bool *)dptr) != 0) {
            count++;
        }
        dptr += strides;
    }
    itemsize = PyArray_DESCR(self->ao)->elsize;
    Py_INCREF(PyArray_DESCR(self->ao));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             PyArray_DESCR(self->ao), 1, &count,
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (ret == NULL) {
        return NULL;
    }
    /* Set up loop */
    optr = PyArray_DATA(ret);
    counter = PyArray_DIMS(ind)[0];
    dptr = PyArray_DATA(ind);
    copyswap = PyArray_DESCR(self->ao)->f->copyswap;
    /* Loop over Boolean array */
    swap = (PyArray_ISNOTSWAPPED(self->ao) != PyArray_ISNOTSWAPPED(ret));
    while (counter--) {
        if (*((npy_bool *)dptr) != 0) {
            copyswap(optr, self->dataptr, swap, self->ao);
            optr += itemsize;
        }
        dptr += strides;
        PyArray_ITER_NEXT(self);
    }
    PyArray_ITER_RESET(self);
    return ret;
}

static PyObject *
iter_subscript_int(PyArrayIterObject *self, PyArrayObject *ind)
{
    npy_intp num;
    PyArrayObject *ret;
    PyArrayIterObject *ind_it;
    int itemsize;
    int swap;
    char *optr;
    npy_intp counter;
    PyArray_CopySwapFunc *copyswap;

    itemsize = PyArray_DESCR(self->ao)->elsize;
    if (PyArray_NDIM(ind) == 0) {
        num = *((npy_intp *)PyArray_DATA(ind));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            PyArray_ITER_RESET(self);
            return NULL;
        }
        else {
            PyObject *tmp;
            PyArray_ITER_GOTO1D(self, num);
            tmp = PyArray_ToScalar(self->dataptr, self->ao);
            PyArray_ITER_RESET(self);
            return tmp;
        }
    }

    Py_INCREF(PyArray_DESCR(self->ao));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             PyArray_DESCR(self->ao),
                             PyArray_NDIM(ind),
                             PyArray_DIMS(ind),
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (ret == NULL) {
        return NULL;
    }
    optr = PyArray_DATA(ret);
    ind_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    if (ind_it == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    counter = ind_it->size;
    copyswap = PyArray_DESCR(ret)->f->copyswap;
    swap = (PyArray_ISNOTSWAPPED(ret) != PyArray_ISNOTSWAPPED(self->ao));
    while (counter--) {
        num = *((npy_intp *)(ind_it->dataptr));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            Py_DECREF(ind_it);
            Py_DECREF(ret);
            PyArray_ITER_RESET(self);
            return NULL;
        }
        PyArray_ITER_GOTO1D(self, num);
        copyswap(optr, self->dataptr, swap, ret);
        optr += itemsize;
        PyArray_ITER_NEXT(ind_it);
    }
    Py_DECREF(ind_it);
    PyArray_ITER_RESET(self);
    return (PyObject *)ret;
}

/* Always returns arrays */
NPY_NO_EXPORT PyObject *
iter_subscript(PyArrayIterObject *self, PyObject *ind)
{
    PyArray_Descr *indtype = NULL;
    PyArray_Descr *dtype;
    npy_intp start, step_size;
    npy_intp n_steps;
    PyArrayObject *ret;
    char *dptr;
    int size;
    PyObject *obj = NULL;
    PyArray_CopySwapFunc *copyswap;

    if (ind == Py_Ellipsis) {
        ind = PySlice_New(NULL, NULL, NULL);
        obj = iter_subscript(self, ind);
        Py_DECREF(ind);
        return obj;
    }
    if (PyTuple_Check(ind)) {
        int len;
        len = PyTuple_GET_SIZE(ind);
        if (len > 1) {
            goto fail;
        }
        if (len == 0) {
            Py_INCREF(self->ao);
            return (PyObject *)self->ao;
        }
        ind = PyTuple_GET_ITEM(ind, 0);
    }

    /*
     * Tuples >1d not accepted --- i.e. no newaxis
     * Could implement this with adjusted strides and dimensions in iterator
     * Check for Boolean -- this is first because Bool is a subclass of Int
     */
    PyArray_ITER_RESET(self);

    if (PyBool_Check(ind)) {
        if (PyObject_IsTrue(ind)) {
            return PyArray_ToScalar(self->dataptr, self->ao);
        }
        else { /* empty array */
            npy_intp ii = 0;
            dtype = PyArray_DESCR(self->ao);
            Py_INCREF(dtype);
            ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                                     dtype,
                                     1, &ii,
                                     NULL, NULL, 0,
                                     (PyObject *)self->ao);
            return (PyObject *)ret;
        }
    }

    /* Check for Integer or Slice */
    if (PyLong_Check(ind) || PyInt_Check(ind) || PySlice_Check(ind)) {
        start = parse_index_entry(ind, &step_size, &n_steps,
                                  self->size, 0, 1);
        if (start == -1) {
            goto fail;
        }
        if (n_steps == ELLIPSIS_INDEX || n_steps == NEWAXIS_INDEX) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto fail;
        }
        PyArray_ITER_GOTO1D(self, start);
        if (n_steps == SINGLE_INDEX) { /* Integer */
            PyObject *tmp;
            tmp = PyArray_ToScalar(self->dataptr, self->ao);
            PyArray_ITER_RESET(self);
            return tmp;
        }
        size = PyArray_DESCR(self->ao)->elsize;
        dtype = PyArray_DESCR(self->ao);
        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                                 dtype,
                                 1, &n_steps,
                                 NULL, NULL,
                                 0, (PyObject *)self->ao);
        if (ret == NULL) {
            goto fail;
        }
        dptr = PyArray_DATA(ret);
        copyswap = PyArray_DESCR(ret)->f->copyswap;
        while (n_steps--) {
            copyswap(dptr, self->dataptr, 0, ret);
            start += step_size;
            PyArray_ITER_GOTO1D(self, start);
            dptr += size;
        }
        PyArray_ITER_RESET(self);
        return (PyObject *)ret;
    }

    /* convert to INTP array if Integer array scalar or List */
    indtype = PyArray_DescrFromType(NPY_INTP);
    if (PyArray_IsScalar(ind, Integer) || PyList_Check(ind)) {
        Py_INCREF(indtype);
        obj = PyArray_FromAny(ind, indtype, 0, 0, NPY_ARRAY_FORCECAST, NULL);
        if (obj == NULL) {
            goto fail;
        }
    }
    else {
        Py_INCREF(ind);
        obj = ind;
    }

    if (PyArray_Check(obj)) {
        /* Check for Boolean object */
        if (PyArray_TYPE((PyArrayObject *)obj) == NPY_BOOL) {
            ret = iter_subscript_Bool(self, (PyArrayObject *)obj);
            Py_DECREF(indtype);
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER((PyArrayObject *)obj)) {
            PyObject *new;
            new = PyArray_FromAny(obj, indtype, 0, 0,
                              NPY_ARRAY_FORCECAST | NPY_ARRAY_ALIGNED, NULL);
            if (new == NULL) {
                goto fail;
            }
            Py_DECREF(obj);
            obj = new;
            new = iter_subscript_int(self, (PyArrayObject *)obj);
            Py_DECREF(obj);
            return new;
        }
        else {
            goto fail;
        }
        Py_DECREF(obj);
        return (PyObject *)ret;
    }
    else {
        Py_DECREF(indtype);
    }


 fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    Py_XDECREF(indtype);
    Py_XDECREF(obj);
    return NULL;

}


static int
iter_ass_sub_Bool(PyArrayIterObject *self, PyArrayObject *ind,
                  PyArrayIterObject *val, int swap)
{
    npy_intp counter, strides;
    char *dptr;
    PyArray_CopySwapFunc *copyswap;

    if (PyArray_NDIM(ind) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return -1;
    }

    counter = PyArray_DIMS(ind)[0];
    if (counter > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array has too many values");
        return -1;
    }

    strides = PyArray_STRIDES(ind)[0];
    dptr = PyArray_DATA(ind);
    PyArray_ITER_RESET(self);
    /* Loop over Boolean array */
    copyswap = PyArray_DESCR(self->ao)->f->copyswap;
    while (counter--) {
        if (*((npy_bool *)dptr) != 0) {
            copyswap(self->dataptr, val->dataptr, swap, self->ao);
            PyArray_ITER_NEXT(val);
            if (val->index == val->size) {
                PyArray_ITER_RESET(val);
            }
        }
        dptr += strides;
        PyArray_ITER_NEXT(self);
    }
    PyArray_ITER_RESET(self);
    return 0;
}

static int
iter_ass_sub_int(PyArrayIterObject *self, PyArrayObject *ind,
                 PyArrayIterObject *val, int swap)
{
    npy_intp num;
    PyArrayIterObject *ind_it;
    npy_intp counter;
    PyArray_CopySwapFunc *copyswap;

    copyswap = PyArray_DESCR(self->ao)->f->copyswap;
    if (PyArray_NDIM(ind) == 0) {
        num = *((npy_intp *)PyArray_DATA(ind));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            return -1;
        }
        PyArray_ITER_GOTO1D(self, num);
        copyswap(self->dataptr, val->dataptr, swap, self->ao);
        return 0;
    }
    ind_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    if (ind_it == NULL) {
        return -1;
    }
    counter = ind_it->size;
    while (counter--) {
        num = *((npy_intp *)(ind_it->dataptr));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            Py_DECREF(ind_it);
            return -1;
        }
        PyArray_ITER_GOTO1D(self, num);
        copyswap(self->dataptr, val->dataptr, swap, self->ao);
        PyArray_ITER_NEXT(ind_it);
        PyArray_ITER_NEXT(val);
        if (val->index == val->size) {
            PyArray_ITER_RESET(val);
        }
    }
    Py_DECREF(ind_it);
    return 0;
}

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *self, PyObject *ind, PyObject *val)
{
    PyArrayObject *arrval = NULL;
    PyArrayIterObject *val_it = NULL;
    PyArray_Descr *type;
    PyArray_Descr *indtype = NULL;
    int swap, retval = -1;
    npy_intp start, step_size;
    npy_intp n_steps;
    PyObject *obj = NULL;
    PyArray_CopySwapFunc *copyswap;


    if (val == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    if (PyArray_FailUnlessWriteable(self->ao, "underlying array") < 0)
        return -1;

    if (ind == Py_Ellipsis) {
        ind = PySlice_New(NULL, NULL, NULL);
        retval = iter_ass_subscript(self, ind, val);
        Py_DECREF(ind);
        return retval;
    }

    if (PyTuple_Check(ind)) {
        int len;
        len = PyTuple_GET_SIZE(ind);
        if (len > 1) {
            goto finish;
        }
        ind = PyTuple_GET_ITEM(ind, 0);
    }

    type = PyArray_DESCR(self->ao);

    /*
     * Check for Boolean -- this is first because
     * Bool is a subclass of Int
     */
    if (PyBool_Check(ind)) {
        retval = 0;
        if (PyObject_IsTrue(ind)) {
            retval = PyArray_SETITEM(self->ao, self->dataptr, val);
        }
        goto finish;
    }

    if (PySequence_Check(ind) || PySlice_Check(ind)) {
        goto skip;
    }
    start = PyArray_PyIntAsIntp(ind);
    if (error_converting(start)) {
        PyErr_Clear();
    }
    else {
        if (check_and_adjust_index(&start, self->size, -1, NULL) < 0) {
            goto finish;
        }
        retval = 0;
        PyArray_ITER_GOTO1D(self, start);
        retval = type->f->setitem(val, self->dataptr, self->ao);
        PyArray_ITER_RESET(self);
        if (retval < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Error setting single item of array.");
        }
        goto finish;
    }

 skip:
    Py_INCREF(type);
    arrval = (PyArrayObject *)PyArray_FromAny(val, type, 0, 0,
                                              NPY_ARRAY_FORCECAST, NULL);
    if (arrval == NULL) {
        return -1;
    }
    val_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arrval);
    if (val_it == NULL) {
        goto finish;
    }
    if (val_it->size == 0) {
        retval = 0;
        goto finish;
    }

    copyswap = PyArray_DESCR(arrval)->f->copyswap;
    swap = (PyArray_ISNOTSWAPPED(self->ao)!=PyArray_ISNOTSWAPPED(arrval));

    /* Check Slice */
    if (PySlice_Check(ind)) {
        start = parse_index_entry(ind, &step_size, &n_steps, self->size, 0, 0);
        if (start == -1) {
            goto finish;
        }
        if (n_steps == ELLIPSIS_INDEX || n_steps == NEWAXIS_INDEX) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto finish;
        }
        PyArray_ITER_GOTO1D(self, start);
        if (n_steps == SINGLE_INDEX) {
            /* Integer */
            copyswap(self->dataptr, PyArray_DATA(arrval), swap, arrval);
            PyArray_ITER_RESET(self);
            retval = 0;
            goto finish;
        }
        while (n_steps--) {
            copyswap(self->dataptr, val_it->dataptr, swap, arrval);
            start += step_size;
            PyArray_ITER_GOTO1D(self, start);
            PyArray_ITER_NEXT(val_it);
            if (val_it->index == val_it->size) {
                PyArray_ITER_RESET(val_it);
            }
        }
        PyArray_ITER_RESET(self);
        retval = 0;
        goto finish;
    }

    /* convert to INTP array if Integer array scalar or List */
    indtype = PyArray_DescrFromType(NPY_INTP);
    if (PyList_Check(ind)) {
        Py_INCREF(indtype);
        obj = PyArray_FromAny(ind, indtype, 0, 0, NPY_ARRAY_FORCECAST, NULL);
    }
    else {
        Py_INCREF(ind);
        obj = ind;
    }

    if (obj != NULL && PyArray_Check(obj)) {
        /* Check for Boolean object */
        if (PyArray_TYPE((PyArrayObject *)obj)==NPY_BOOL) {
            if (iter_ass_sub_Bool(self, (PyArrayObject *)obj,
                                  val_it, swap) < 0) {
                goto finish;
            }
            retval=0;
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER((PyArrayObject *)obj)) {
            PyObject *new;
            Py_INCREF(indtype);
            new = PyArray_CheckFromAny(obj, indtype, 0, 0,
                           NPY_ARRAY_FORCECAST | NPY_ARRAY_BEHAVED_NS, NULL);
            Py_DECREF(obj);
            obj = new;
            if (new == NULL) {
                goto finish;
            }
            if (iter_ass_sub_int(self, (PyArrayObject *)obj,
                                 val_it, swap) < 0) {
                goto finish;
            }
            retval = 0;
        }
    }

 finish:
    if (!PyErr_Occurred() && retval < 0) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    Py_XDECREF(indtype);
    Py_XDECREF(obj);
    Py_XDECREF(val_it);
    Py_XDECREF(arrval);
    return retval;

}


static PyMappingMethods iter_as_mapping = {
    (lenfunc)iter_length,                   /*mp_length*/
    (binaryfunc)iter_subscript,             /*mp_subscript*/
    (objobjargproc)iter_ass_subscript,      /*mp_ass_subscript*/
};


/* Two options:
 *  1) underlying array is contiguous
 *     -- return 1-d wrapper around it
 *  2) underlying array is not contiguous
 *     -- make new 1-d contiguous array with updateifcopy flag set
 *        to copy back to the old array
 *
 *  If underlying array is readonly, then we make the output array readonly
 *     and updateifcopy does not apply.
 *
 *  Changed 2017-07-21, 1.14.0.
 *
 *  In order to start the process of removing UPDATEIFCOPY, see gh-7054, the
 *  behavior is changed to always return an non-writeable copy when the base
 *  array is non-contiguous. Doing that will hopefully smoke out those few
 *  folks who assign to the result with the expectation that the base array
 *  will be changed. At a later date non-contiguous arrays will always return
 *  writeable copies.
 *
 *  Note that the type and argument expected for the __array__ method is
 *  ignored.
 */
static PyArrayObject *
iter_array(PyArrayIterObject *it, PyObject *NPY_UNUSED(op))
{

    PyArrayObject *ret;
    npy_intp size;

    size = PyArray_SIZE(it->ao);
    Py_INCREF(PyArray_DESCR(it->ao));

    if (PyArray_ISCONTIGUOUS(it->ao)) {
        ret = (PyArrayObject *)PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DESCR(it->ao), 1, &size,
                NULL, PyArray_DATA(it->ao), PyArray_FLAGS(it->ao),
                (PyObject *)it->ao);
        if (ret == NULL) {
            return NULL;
        }
        Py_INCREF(it->ao);
        if (PyArray_SetBaseObject(ret, (PyObject *)it->ao) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }
    else {
        ret = (PyArrayObject *)PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DESCR(it->ao), 1, &size,
                NULL, NULL, 0,
                (PyObject *)it->ao);
        if (ret == NULL) {
            return NULL;
        }
        if (PyArray_CopyAnyInto(ret, it->ao) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return ret;

}

static PyObject *
iter_copy(PyArrayIterObject *it, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    return PyArray_Flatten(it->ao, 0);
}

static PyMethodDef iter_methods[] = {
    /* to get array */
    {"__array__",
        (PyCFunction)iter_array,
        METH_VARARGS, NULL},
    {"copy",
        (PyCFunction)iter_copy,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyObject *
iter_richcompare(PyArrayIterObject *self, PyObject *other, int cmp_op)
{
    PyArrayObject *new;
    PyObject *ret;
    new = (PyArrayObject *)iter_array(self, NULL);
    if (new == NULL) {
        return NULL;
    }
    ret = array_richcompare(new, other, cmp_op);
    PyArray_ResolveWritebackIfCopy(new);
    Py_DECREF(new);
    return ret;
}


static PyMemberDef iter_members[] = {
    {"base",
        T_OBJECT,
        offsetof(PyArrayIterObject, ao),
        READONLY, NULL},
    {"index",
        T_INT,
        offsetof(PyArrayIterObject, index),
        READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
iter_coords_get(PyArrayIterObject *self)
{
    int nd;
    nd = PyArray_NDIM(self->ao);
    if (self->contiguous) {
        /*
         * coordinates not kept track of ---
         * need to generate from index
         */
        npy_intp val;
        int i;
        val = self->index;
        for (i = 0; i < nd; i++) {
            if (self->factors[i] != 0) {
                self->coordinates[i] = val / self->factors[i];
                val = val % self->factors[i];
            } else {
                self->coordinates[i] = 0;
            }
        }
    }
    return PyArray_IntTupleFromIntp(nd, self->coordinates);
}

static PyGetSetDef iter_getsets[] = {
    {"coords",
        (getter)iter_coords_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

NPY_NO_EXPORT PyTypeObject PyArrayIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.flatiter",                           /* tp_name */
    sizeof(PyArrayIterObject),                  /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arrayiter_dealloc,              /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    &iter_as_mapping,                           /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)iter_richcompare,              /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    (iternextfunc)arrayiter_next,               /* tp_iternext */
    iter_methods,                               /* tp_methods */
    iter_members,                               /* tp_members */
    iter_getsets,                               /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};

/** END of Array Iterator **/

/* Adjust dimensionality and strides for index object iterators
   --- i.e. broadcast
*/
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Broadcast(PyArrayMultiIterObject *mit)
{
    int i, nd, k, j;
    npy_intp tmp;
    PyArrayIterObject *it;

    /* Discover the broadcast number of dimensions */
    for (i = 0, nd = 0; i < mit->numiter; i++) {
        nd = PyArray_MAX(nd, PyArray_NDIM(mit->iters[i]->ao));
    }
    mit->nd = nd;

    /* Discover the broadcast shape in each dimension */
    for (i = 0; i < nd; i++) {
        mit->dimensions[i] = 1;
        for (j = 0; j < mit->numiter; j++) {
            it = mit->iters[j];
            /* This prepends 1 to shapes not already equal to nd */
            k = i + PyArray_NDIM(it->ao) - nd;
            if (k >= 0) {
                tmp = PyArray_DIMS(it->ao)[k];
                if (tmp == 1) {
                    continue;
                }
                if (mit->dimensions[i] == 1) {
                    mit->dimensions[i] = tmp;
                }
                else if (mit->dimensions[i] != tmp) {
                    PyErr_SetString(PyExc_ValueError,
                                    "shape mismatch: objects" \
                                    " cannot be broadcast" \
                                    " to a single shape");
                    return -1;
                }
            }
        }
    }

    /*
     * Reset the iterator dimensions and strides of each iterator
     * object -- using 0 valued strides for broadcasting
     * Need to check for overflow
     */
    tmp = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (tmp < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "broadcast dimensions too large.");
        return -1;
    }
    mit->size = tmp;
    for (i = 0; i < mit->numiter; i++) {
        it = mit->iters[i];
        it->nd_m1 = mit->nd - 1;
        it->size = tmp;
        nd = PyArray_NDIM(it->ao);
        if (nd != 0) {
            it->factors[mit->nd-1] = 1;
        }
        for (j = 0; j < mit->nd; j++) {
            it->dims_m1[j] = mit->dimensions[j] - 1;
            k = j + nd - mit->nd;
            /*
             * If this dimension was added or shape of
             * underlying array was 1
             */
            if ((k < 0) ||
                PyArray_DIMS(it->ao)[k] != mit->dimensions[j]) {
                it->contiguous = 0;
                it->strides[j] = 0;
            }
            else {
                it->strides[j] = PyArray_STRIDES(it->ao)[k];
            }
            it->backstrides[j] = it->strides[j] * it->dims_m1[j];
            if (j > 0)
                it->factors[mit->nd-j-1] =
                    it->factors[mit->nd-j] * mit->dimensions[mit->nd-j];
        }
        PyArray_ITER_RESET(it);
    }
    return 0;
}

/*NUMPY_API
 * Get MultiIterator from array of Python objects and any additional
 *
 * PyObject **mps -- array of PyObjects
 * int n - number of PyObjects in the array
 * int nadd - number of additional arrays to include in the iterator.
 *
 * Returns a multi-iterator object.
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIterFromObjects(PyObject **mps, int n, int nadd, ...)
{
    va_list va;
    PyArrayMultiIterObject *multi;
    PyObject *current;
    PyObject *arr;

    int i, ntot, err=0;

    ntot = n + nadd;
    if (ntot < 1 || ntot > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Need at least 1 and at most %d "
                     "array objects.", NPY_MAXARGS);
        return NULL;
    }
    multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);

    for (i = 0; i < ntot; i++) {
        multi->iters[i] = NULL;
    }
    multi->numiter = ntot;
    multi->index = 0;

    va_start(va, nadd);
    for (i = 0; i < ntot; i++) {
        if (i < n) {
            current = mps[i];
        }
        else {
            current = va_arg(va, PyObject *);
        }
        arr = PyArray_FROM_O(current);
        if (arr == NULL) {
            err = 1;
            break;
        }
        else {
            multi->iters[i] = (PyArrayIterObject *)PyArray_IterNew(arr);
            if (multi->iters[i] == NULL) {
                err = 1;
                break;
            }
            Py_DECREF(arr);
        }
    }
    va_end(va);

    if (!err && PyArray_Broadcast(multi) < 0) {
        err = 1;
    }
    if (err) {
        Py_DECREF(multi);
        return NULL;
    }
    PyArray_MultiIter_RESET(multi);
    return (PyObject *)multi;
}

/*NUMPY_API
 * Get MultiIterator,
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIterNew(int n, ...)
{
    va_list va;
    PyArrayMultiIterObject *multi;
    PyObject *current;
    PyObject *arr;

    int i, err = 0;

    if (n < 1 || n > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Need at least 1 and at most %d "
                     "array objects.", NPY_MAXARGS);
        return NULL;
    }

    /* fprintf(stderr, "multi new...");*/

    multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);

    for (i = 0; i < n; i++) {
        multi->iters[i] = NULL;
    }
    multi->numiter = n;
    multi->index = 0;

    va_start(va, n);
    for (i = 0; i < n; i++) {
        current = va_arg(va, PyObject *);
        arr = PyArray_FROM_O(current);
        if (arr == NULL) {
            err = 1;
            break;
        }
        else {
            multi->iters[i] = (PyArrayIterObject *)PyArray_IterNew(arr);
            if (multi->iters[i] == NULL) {
                err = 1;
                break;
            }
            Py_DECREF(arr);
        }
    }
    va_end(va);

    if (!err && PyArray_Broadcast(multi) < 0) {
        err = 1;
    }
    if (err) {
        Py_DECREF(multi);
        return NULL;
    }
    PyArray_MultiIter_RESET(multi);
    return (PyObject *)multi;
}

static PyObject *
arraymultiter_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args, PyObject *kwds)
{

    Py_ssize_t n = 0;
    Py_ssize_t i, j, k;
    PyArrayMultiIterObject *multi;
    PyObject *arr;

    if (kwds != NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "keyword arguments not accepted.");
        return NULL;
    }

    for (j = 0; j < PyTuple_Size(args); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(args, j);

        if (PyObject_IsInstance(obj, (PyObject *)&PyArrayMultiIter_Type)) {
            /*
             * If obj is a multi-iterator, all its arrays will be added
             * to the new multi-iterator.
             */
            n += ((PyArrayMultiIterObject *)obj)->numiter;
        }
        else {
            /* If not, will try to convert it to a single array */
            ++n;
        }
    }
    if (n < 1 || n > NPY_MAXARGS) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        PyErr_Format(PyExc_ValueError,
                     "Need at least 1 and at most %d "
                     "array objects.", NPY_MAXARGS);
        return NULL;
    }

    multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);

    multi->numiter = n;
    multi->index = 0;
    i = 0;
    for (j = 0; j < PyTuple_GET_SIZE(args); ++j) {
        PyObject *obj = PyTuple_GET_ITEM(args, j);
        PyArrayIterObject *it;

        if (PyObject_IsInstance(obj, (PyObject *)&PyArrayMultiIter_Type)) {
            PyArrayMultiIterObject *mit = (PyArrayMultiIterObject *)obj;

            for (k = 0; k < mit->numiter; ++k) {
                arr = (PyObject *)mit->iters[k]->ao;
                assert (arr != NULL);
                it = (PyArrayIterObject *)PyArray_IterNew(arr);
                if (it == NULL) {
                    goto fail;
                }
                multi->iters[i++] = it;
            }
        }
        else {
            arr = PyArray_FROM_O(obj);
            if (arr == NULL) {
                goto fail;
            }
            it = (PyArrayIterObject *)PyArray_IterNew(arr);
            if (it == NULL) {
                goto fail;
            }
            multi->iters[i++] = it;
            Py_DECREF(arr);
        }
    }
    assert (i == n);
    if (PyArray_Broadcast(multi) < 0) {
        goto fail;
    }
    PyArray_MultiIter_RESET(multi);
    return (PyObject *)multi;

 fail:
    Py_DECREF(multi);
    return NULL;
}

static PyObject *
arraymultiter_next(PyArrayMultiIterObject *multi)
{
    PyObject *ret;
    int i, n;

    n = multi->numiter;
    ret = PyTuple_New(n);
    if (ret == NULL) {
        return NULL;
    }
    if (multi->index < multi->size) {
        for (i = 0; i < n; i++) {
            PyArrayIterObject *it=multi->iters[i];
            PyTuple_SET_ITEM(ret, i,
                             PyArray_ToScalar(it->dataptr, it->ao));
            PyArray_ITER_NEXT(it);
        }
        multi->index++;
        return ret;
    }
    Py_DECREF(ret);
    return NULL;
}

static void
arraymultiter_dealloc(PyArrayMultiIterObject *multi)
{
    int i;

    for (i = 0; i < multi->numiter; i++) {
        Py_XDECREF(multi->iters[i]);
    }
    Py_TYPE(multi)->tp_free((PyObject *)multi);
}

static PyObject *
arraymultiter_size_get(PyArrayMultiIterObject *self)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) self->size);
#else
    if (self->size < NPY_MAX_LONG) {
        return PyInt_FromLong((long) self->size);
    }
    else {
        return PyLong_FromLongLong((npy_longlong) self->size);
    }
#endif
}

static PyObject *
arraymultiter_index_get(PyArrayMultiIterObject *self)
{
#if NPY_SIZEOF_INTP <= NPY_SIZEOF_LONG
    return PyInt_FromLong((long) self->index);
#else
    if (self->size < NPY_MAX_LONG) {
        return PyInt_FromLong((long) self->index);
    }
    else {
        return PyLong_FromLongLong((npy_longlong) self->index);
    }
#endif
}

static PyObject *
arraymultiter_shape_get(PyArrayMultiIterObject *self)
{
    return PyArray_IntTupleFromIntp(self->nd, self->dimensions);
}

static PyObject *
arraymultiter_iters_get(PyArrayMultiIterObject *self)
{
    PyObject *res;
    int i, n;

    n = self->numiter;
    res = PyTuple_New(n);
    if (res == NULL) {
        return res;
    }
    for (i = 0; i < n; i++) {
        Py_INCREF(self->iters[i]);
        PyTuple_SET_ITEM(res, i, (PyObject *)self->iters[i]);
    }
    return res;
}

static PyGetSetDef arraymultiter_getsetlist[] = {
    {"size",
        (getter)arraymultiter_size_get,
        NULL,
        NULL, NULL},
    {"index",
        (getter)arraymultiter_index_get,
        NULL,
        NULL, NULL},
    {"shape",
        (getter)arraymultiter_shape_get,
        NULL,
        NULL, NULL},
    {"iters",
        (getter)arraymultiter_iters_get,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMemberDef arraymultiter_members[] = {
    {"numiter",
        T_INT,
        offsetof(PyArrayMultiIterObject, numiter),
        READONLY, NULL},
    {"nd",
        T_INT,
        offsetof(PyArrayMultiIterObject, nd),
        READONLY, NULL},
    {"ndim",
        T_INT,
        offsetof(PyArrayMultiIterObject, nd),
        READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraymultiter_reset(PyArrayMultiIterObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    PyArray_MultiIter_RESET(self);
    Py_RETURN_NONE;
}

static PyMethodDef arraymultiter_methods[] = {
    {"reset",
        (PyCFunction) arraymultiter_reset,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},      /* sentinal */
};

NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.broadcast",                          /* tp_name */
    sizeof(PyArrayMultiIterObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraymultiter_dealloc,          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    (iternextfunc)arraymultiter_next,           /* tp_iternext */
    arraymultiter_methods,                      /* tp_methods */
    arraymultiter_members,                      /* tp_members */
    arraymultiter_getsetlist,                   /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
    0,                                          /* tp_alloc */
    arraymultiter_new,                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};

/*========================= Neighborhood iterator ======================*/

static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter);

static char* _set_constant(PyArrayNeighborhoodIterObject* iter,
        PyArrayObject *fill)
{
    char *ret;
    PyArrayIterObject *ar = iter->_internal_iter;
    int storeflags, st;

    ret = PyDataMem_NEW(PyArray_DESCR(ar->ao)->elsize);
    if (ret == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (PyArray_ISOBJECT(ar->ao)) {
        memcpy(ret, PyArray_DATA(fill), sizeof(PyObject*));
        Py_INCREF(*(PyObject**)ret);
    } else {
        /* Non-object types */

        storeflags = PyArray_FLAGS(ar->ao);
        PyArray_ENABLEFLAGS(ar->ao, NPY_ARRAY_BEHAVED);
        st = PyArray_SETITEM(ar->ao, ret, (PyObject*)fill);
        ((PyArrayObject_fields *)ar->ao)->flags = storeflags;

        if (st < 0) {
            PyDataMem_FREE(ret);
            return NULL;
        }
    }

    return ret;
}

#define _INF_SET_PTR(c) \
    bd = coordinates[c] + p->coordinates[c]; \
    if (bd < p->limits[c][0] || bd > p->limits[c][1]) { \
        return niter->constant; \
    } \
    _coordinates[c] = bd;

/* set the dataptr from its current coordinates */
static char*
get_ptr_constant(PyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS];
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR

#define _NPY_IS_EVEN(x) ((x) % 2 == 0)

/* For an array x of dimension n, and given index i, returns j, 0 <= j < n
 * such as x[i] = x[j], with x assumed to be mirrored. For example, for x =
 * {1, 2, 3} (n = 3)
 *
 * index -5 -4 -3 -2 -1 0 1 2 3 4 5 6
 * value  2  3  3  2  1 1 2 3 3 2 1 1
 *
 * _npy_pos_index_mirror(4, 3) will return 1, because x[4] = x[1]*/
NPY_INLINE static npy_intp
__npy_pos_remainder(npy_intp i, npy_intp n)
{
    npy_intp k, l, j;

    /* Mirror i such as it is guaranteed to be positive */
    if (i < 0) {
        i = - i - 1;
    }

    /* compute k and l such as i = k * n + l, 0 <= l < k */
    k = i / n;
    l = i - k * n;

    if (_NPY_IS_EVEN(k)) {
        j = l;
    } else {
        j = n - 1 - l;
    }
    return j;
}
#undef _NPY_IS_EVEN

#define _INF_SET_PTR_MIRROR(c) \
    lb = p->limits[c][0]; \
    bd = coordinates[c] + p->coordinates[c] - lb; \
    _coordinates[c] = lb + __npy_pos_remainder(bd, p->limits_sizes[c]);

/* set the dataptr from its current coordinates */
static char*
get_ptr_mirror(PyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_MIRROR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR_MIRROR

/* compute l such as i = k * n + l, 0 <= l < |k| */
NPY_INLINE static npy_intp
__npy_euclidean_division(npy_intp i, npy_intp n)
{
    npy_intp l;

    l = i % n;
    if (l < 0) {
        l += n;
    }
    return l;
}

#define _INF_SET_PTR_CIRCULAR(c) \
    lb = p->limits[c][0]; \
    bd = coordinates[c] + p->coordinates[c] - lb; \
    _coordinates[c] = lb + __npy_euclidean_division(bd, p->limits_sizes[c]);

static char*
get_ptr_circular(PyArrayIterObject* _iter, npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_CIRCULAR(i)
    }
    return p->translate(p, _coordinates);
}

#undef _INF_SET_PTR_CIRCULAR

/*
 * fill and x->ao should have equivalent types
 */
/*NUMPY_API
 * A Neighborhood Iterator object.
*/
NPY_NO_EXPORT PyObject*
PyArray_NeighborhoodIterNew(PyArrayIterObject *x, npy_intp *bounds,
                            int mode, PyArrayObject* fill)
{
    int i;
    PyArrayNeighborhoodIterObject *ret;

    ret = PyArray_malloc(sizeof(*ret));
    if (ret == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)ret, &PyArrayNeighborhoodIter_Type);

    array_iter_base_init((PyArrayIterObject*)ret, x->ao);
    Py_INCREF(x);
    ret->_internal_iter = x;

    ret->nd = PyArray_NDIM(x->ao);

    for (i = 0; i < ret->nd; ++i) {
        ret->dimensions[i] = PyArray_DIMS(x->ao)[i];
    }

    /* Compute the neighborhood size and copy the shape */
    ret->size = 1;
    for (i = 0; i < ret->nd; ++i) {
        ret->bounds[i][0] = bounds[2 * i];
        ret->bounds[i][1] = bounds[2 * i + 1];
        ret->size *= (ret->bounds[i][1] - ret->bounds[i][0]) + 1;

        /* limits keep track of valid ranges for the neighborhood: if a bound
         * of the neighborhood is outside the array, then limits is the same as
         * boundaries. On the contrary, if a bound is strictly inside the
         * array, then limits correspond to the array range. For example, for
         * an array [1, 2, 3], if bounds are [-1, 3], limits will be [-1, 3],
         * but if bounds are [1, 2], then limits will be [0, 2].
         *
         * This is used by neighborhood iterators stacked on top of this one */
        ret->limits[i][0] = ret->bounds[i][0] < 0 ? ret->bounds[i][0] : 0;
        ret->limits[i][1] = ret->bounds[i][1] >= ret->dimensions[i] - 1 ?
                            ret->bounds[i][1] :
                            ret->dimensions[i] - 1;
        ret->limits_sizes[i] = (ret->limits[i][1] - ret->limits[i][0]) + 1;
    }

    switch (mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
            ret->constant = PyArray_Zero(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
            ret->constant = PyArray_One(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            /* New reference in returned value of _set_constant if array
             * object */
            assert(PyArray_EquivArrTypes(x->ao, fill) == NPY_TRUE);
            ret->constant = _set_constant(ret, fill);
            if (ret->constant == NULL) {
                goto clean_x;
            }
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_mirror;
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_circular;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unsupported padding mode");
            goto clean_x;
    }

    /*
     * XXX: we force x iterator to be non contiguous because we need
     * coordinates... Modifying the iterator here is not great
     */
    x->contiguous = 0;

    PyArrayNeighborhoodIter_Reset(ret);

    return (PyObject*)ret;

clean_x:
    Py_DECREF(ret->_internal_iter);
    array_iter_base_dealloc((PyArrayIterObject*)ret);
    PyArray_free((PyArrayObject*)ret);
    return NULL;
}

static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter)
{
    if (iter->mode == NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING) {
        if (PyArray_ISOBJECT(iter->_internal_iter->ao)) {
            Py_DECREF(*(PyObject**)iter->constant);
        }
    }
    PyDataMem_FREE(iter->constant);
    Py_DECREF(iter->_internal_iter);

    array_iter_base_dealloc((PyArrayIterObject*)iter);
    PyArray_free((PyArrayObject*)iter);
}

NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.neigh_internal_iter",                /* tp_name*/
    sizeof(PyArrayNeighborhoodIterObject),      /* tp_basicsize*/
    0,                                          /* tp_itemsize*/
    (destructor)neighiter_dealloc,              /* tp_dealloc*/
    0,                                          /* tp_print*/
    0,                                          /* tp_getattr*/
    0,                                          /* tp_setattr*/
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr*/
    0,                                          /* tp_as_number*/
    0,                                          /* tp_as_sequence*/
    0,                                          /* tp_as_mapping*/
    0,                                          /* tp_hash */
    0,                                          /* tp_call*/
    0,                                          /* tp_str*/
    0,                                          /* tp_getattro*/
    0,                                          /* tp_setattro*/
    0,                                          /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                         /* tp_flags*/
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    (iternextfunc)0,                            /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
