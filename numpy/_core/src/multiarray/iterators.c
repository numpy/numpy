#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_math.h"

#include "npy_config.h"



#include "arrayobject.h"
#include "iterators.h"
#include "ctors.h"
#include "common.h"
#include "conversion_utils.h"
#include "dtypemeta.h"
#include "array_coercion.h"
#include "item_selection.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"
#include "npy_pycompat.h"

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
        if (PySlice_GetIndicesEx(op, max, &i, &stop, step_size, n_steps) < 0) {
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


/*********************** Element-wise Array Iterator ***********************/
/*  Aided by Peter J. Verveer's  nd_image package and numpy's arraymap  ****/
/*         and Python's array iterator                                   ***/

/* get the dataptr from its current coordinates for simple iterator */
static char*
get_ptr_simple(PyArrayIterObject* iter, const npy_intp *coordinates)
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
 * Steals a reference to the array object which gets removed at deallocation,
 * if the iterator is allocated statically and its dealloc not called, it
 * can be thought of as borrowing the reference.
 */
NPY_NO_EXPORT void
PyArray_RawIterBaseInit(PyArrayIterObject *it, PyArrayObject *ao)
{
    int nd, i;

    nd = PyArray_NDIM(ao);
    /* The legacy iterator only supports 32 dimensions */
    assert(nd <= NPY_MAXDIMS_LEGACY_ITERS);
    if (PyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
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

    return;
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
    /*
     * Note that internally PyArray_RawIterBaseInit may be called directly on a
     * statically allocated PyArrayIterObject.
     */
    PyArrayIterObject *it;
    PyArrayObject *ao;

    if (!PyArray_Check(obj)) {
        PyErr_BadInternalCall();
        return NULL;
    }
    ao = (PyArrayObject *)obj;
    if (PyArray_NDIM(ao) > NPY_MAXDIMS_LEGACY_ITERS) {
        PyErr_Format(PyExc_RuntimeError,
                "this function only supports up to 32 dimensions but "
                "the array has %d.", PyArray_NDIM(ao));
        return NULL;
    }

    it = (PyArrayIterObject *)PyArray_malloc(sizeof(PyArrayIterObject));
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);
    /* it = PyObject_New(PyArrayIterObject, &PyArrayIter_Type);*/
    if (it == NULL) {
        return NULL;
    }

    Py_INCREF(ao);  /* PyArray_RawIterBaseInit steals a reference */
    PyArray_RawIterBaseInit(it, ao);
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
    /*
     * Note that it is possible to statically allocate a PyArrayIterObject,
     * which does not call this function.
     */
    array_iter_base_dealloc(it);
    PyArray_free(it);
}

static Py_ssize_t
iter_length(PyArrayIterObject *self)
{
    return self->size;
}


static PyArrayObject *
iter_subscript_Bool(PyArrayIterObject *self, PyArrayObject *ind,
                    NPY_cast_info *cast_info)
{
    npy_intp counter, strides;
    int itemsize;
    npy_intp count = 0;
    char *dptr, *optr;
    PyArrayObject *ret;

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
    /* Get size of return array */
    count = count_boolean_trues(PyArray_NDIM(ind), PyArray_DATA(ind),
                                PyArray_DIMS(ind), PyArray_STRIDES(ind));
    itemsize = PyArray_ITEMSIZE(self->ao);
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             dtype, 1, &count,
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (ret == NULL) {
        return NULL;
    }
    if (count > 0) {
        /* Set up loop */
        optr = PyArray_DATA(ret);
        counter = PyArray_DIMS(ind)[0];
        dptr = PyArray_DATA(ind);
        npy_intp one = 1;
        while (counter--) {
            if (*((npy_bool *)dptr) != 0) {
                char *args[2] = {self->dataptr, optr};
                npy_intp transfer_strides[2] = {itemsize, itemsize};
                if (cast_info->func(&cast_info->context, args, &one,
                                    transfer_strides, cast_info->auxdata) < 0) {
                    return NULL;
                }
                optr += itemsize;
            }
            dptr += strides;
            PyArray_ITER_NEXT(self);
        }
        PyArray_ITER_RESET(self);
    }
    return ret;
}

static PyObject *
iter_subscript_int(PyArrayIterObject *self, PyArrayObject *ind,
                   NPY_cast_info *cast_info)
{
    npy_intp num;
    PyArrayObject *ret;
    PyArrayIterObject *ind_it;
    int itemsize;
    char *optr;
    npy_intp counter;

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

    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             dtype,
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

    npy_intp one = 1;
    itemsize = dtype->elsize;
    counter = ind_it->size;
    while (counter--) {
        num = *((npy_intp *)(ind_it->dataptr));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            Py_CLEAR(ret);
            goto finish;
        }
        PyArray_ITER_GOTO1D(self, num);
        char *args[2] = {self->dataptr, optr};
        npy_intp transfer_strides[2] = {itemsize, itemsize};
        if (cast_info->func(&cast_info->context, args, &one,
                            transfer_strides, cast_info->auxdata) < 0) {
            Py_CLEAR(ret);
            goto finish;
        }
        optr += itemsize;
        PyArray_ITER_NEXT(ind_it);
    }

 finish:
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
    PyObject *new;
    NPY_cast_info cast_info = {.func = NULL};

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
        int istrue = PyObject_IsTrue(ind);
        if (istrue == -1) {
            goto fail;
        }
        if (istrue) {
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

    dtype = PyArray_DESCR(self->ao);
    size = dtype->elsize;

    /* set up a cast to handle item copying */

    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;
    npy_intp one = 1;
    /* We can assume the newly allocated output array is aligned */
    int is_aligned = IsUintAligned(self->ao);
    if (PyArray_GetDTypeTransferFunction(
                is_aligned, size, size, dtype, dtype, 0, &cast_info,
                &transfer_flags) < 0) {
        goto fail;
    }

    /* Check for Integer or Slice */
    if (PyLong_Check(ind) || PySlice_Check(ind)) {
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
            NPY_cast_info_xfree(&cast_info);
            return tmp;
        }
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
        while (n_steps--) {
            char *args[2] = {self->dataptr, dptr};
            npy_intp transfer_strides[2] = {size, size};
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto fail;
            }
            start += step_size;
            PyArray_ITER_GOTO1D(self, start);
            dptr += size;
        }
        PyArray_ITER_RESET(self);
        NPY_cast_info_xfree(&cast_info);
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

    if (!PyArray_Check(obj)) {
        PyArrayObject *tmp_arr = (PyArrayObject *) PyArray_FROM_O(obj);
        if (tmp_arr == NULL) {
            goto fail;
        }

        if (PyArray_SIZE(tmp_arr) == 0) {
            PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);
            Py_SETREF(obj, PyArray_FromArray(tmp_arr, indtype, NPY_ARRAY_FORCECAST));
            Py_DECREF(tmp_arr);
            if (obj == NULL) {
                goto fail;
            }
        }
        else {
            Py_SETREF(obj, (PyObject *) tmp_arr);
        }
    }

    /* Check for Boolean array */
    if (PyArray_TYPE((PyArrayObject *)obj) == NPY_BOOL) {
        ret = iter_subscript_Bool(self, (PyArrayObject *)obj, &cast_info);
        goto finish;
    }

    /* Only integer arrays left */
    if (!PyArray_ISINTEGER((PyArrayObject *)obj)) {
        goto fail;
    }

    Py_INCREF(indtype);
    new = PyArray_FromAny(obj, indtype, 0, 0,
                      NPY_ARRAY_FORCECAST | NPY_ARRAY_ALIGNED, NULL);
    if (new == NULL) {
        goto fail;
    }
    ret = (PyArrayObject *)iter_subscript_int(self, (PyArrayObject *)new,
                                              &cast_info);
    Py_DECREF(new);

 finish:
    Py_DECREF(indtype);
    Py_DECREF(obj);
    NPY_cast_info_xfree(&cast_info);
    return (PyObject *)ret;

 fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    Py_XDECREF(indtype);
    Py_XDECREF(obj);
    NPY_cast_info_xfree(&cast_info);

    return NULL;

}


static int
iter_ass_sub_Bool(PyArrayIterObject *self, PyArrayObject *ind,
                  PyArrayIterObject *val, NPY_cast_info *cast_info)
{
    npy_intp counter, strides;
    char *dptr;

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
    npy_intp one = 1;
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    int itemsize = dtype->elsize;
    npy_intp transfer_strides[2] = {itemsize, itemsize};
    while (counter--) {
        if (*((npy_bool *)dptr) != 0) {
            char *args[2] = {val->dataptr, self->dataptr};
            if (cast_info->func(&cast_info->context, args, &one,
                                transfer_strides, cast_info->auxdata) < 0) {
                return -1;
            }
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
                 PyArrayIterObject *val, NPY_cast_info *cast_info)
{
    npy_intp num;
    PyArrayIterObject *ind_it;
    npy_intp counter;

    npy_intp one = 1;
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    int itemsize = dtype->elsize;
    npy_intp transfer_strides[2] = {itemsize, itemsize};

    if (PyArray_NDIM(ind) == 0) {
        num = *((npy_intp *)PyArray_DATA(ind));
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            return -1;
        }
        PyArray_ITER_GOTO1D(self, num);
        char *args[2] = {val->dataptr, self->dataptr};
        if (cast_info->func(&cast_info->context, args, &one,
                            transfer_strides, cast_info->auxdata) < 0) {
            return -1;
        }
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
        char *args[2] = {val->dataptr, self->dataptr};
        if (cast_info->func(&cast_info->context, args, &one,
                            transfer_strides, cast_info->auxdata) < 0) {
            Py_DECREF(ind_it);
            return -1;
        }
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
    int retval = -1;
    npy_intp start, step_size;
    npy_intp n_steps;
    PyObject *obj = NULL;
    NPY_cast_info cast_info = {.func = NULL};

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
        int istrue = PyObject_IsTrue(ind);
        if (istrue == -1) {
            return -1;
        }
        if (istrue) {
            retval = PyArray_Pack(
                    PyArray_DESCR(self->ao), self->dataptr, val);
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
        PyArray_ITER_GOTO1D(self, start);
        retval = PyArray_Pack(PyArray_DESCR(self->ao), self->dataptr, val);
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

    /* set up cast to handle single-element copies into arrval */
    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;
    npy_intp one = 1;
    int itemsize = type->elsize;
    /* We can assume the newly allocated array is aligned */
    int is_aligned = IsUintAligned(self->ao);
    if (PyArray_GetDTypeTransferFunction(
                is_aligned, itemsize, itemsize, type, type, 0,
                &cast_info, &transfer_flags) < 0) {
        goto finish;
    }

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
        npy_intp transfer_strides[2] = {itemsize, itemsize};
        if (n_steps == SINGLE_INDEX) {
            char *args[2] = {PyArray_DATA(arrval), self->dataptr};
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto finish;
            }
            PyArray_ITER_RESET(self);
            retval = 0;
            goto finish;
        }
        while (n_steps--) {
            char *args[2] = {val_it->dataptr, self->dataptr};
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto finish;
            }
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
                                  val_it, &cast_info) < 0) {
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
                                 val_it, &cast_info) < 0) {
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
    NPY_cast_info_xfree(&cast_info);
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
iter_array(PyArrayIterObject *it, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{

    PyArrayObject *ret;
    npy_intp size;

    size = PyArray_SIZE(it->ao);
    Py_INCREF(PyArray_DESCR(it->ao));

    if (PyArray_ISCONTIGUOUS(it->ao)) {
        ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type, PyArray_DESCR(it->ao),
                1, &size, NULL, PyArray_DATA(it->ao),
                PyArray_FLAGS(it->ao), (PyObject *)it->ao, (PyObject *)it->ao);
        if (ret == NULL) {
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
        METH_VARARGS | METH_KEYWORDS, NULL},
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
    new = (PyArrayObject *)iter_array(self, NULL, NULL);
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
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
iter_index_get(PyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(self->index);
}

static PyObject *
iter_coords_get(PyArrayIterObject *self, void *NPY_UNUSED(ignored))
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
    {"index",
        (getter)iter_index_get,
        NULL, NULL, NULL},
    {"coords",
        (getter)iter_coords_get,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

NPY_NO_EXPORT PyTypeObject PyArrayIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.flatiter",
    .tp_basicsize = sizeof(PyArrayIterObject),
    .tp_dealloc = (destructor)arrayiter_dealloc,
    .tp_as_mapping = &iter_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = (richcmpfunc)iter_richcompare,
    .tp_iternext = (iternextfunc)arrayiter_next,
    .tp_methods = iter_methods,
    .tp_members = iter_members,
    .tp_getset = iter_getsets,
};

/** END of Array Iterator **/


static int
set_shape_mismatch_exception(PyArrayMultiIterObject *mit, int i1, int i2)
{
    PyObject *shape1, *shape2, *msg;

    shape1 = PyObject_GetAttrString((PyObject *) mit->iters[i1]->ao, "shape");
    if (shape1 == NULL) {
        return -1;
    }
    shape2 = PyObject_GetAttrString((PyObject *) mit->iters[i2]->ao, "shape");
    if (shape2 == NULL) {
        Py_DECREF(shape1);
        return -1;
    }
    msg = PyUnicode_FromFormat("shape mismatch: objects cannot be broadcast "
                               "to a single shape.  Mismatch is between arg %d "
                               "with shape %S and arg %d with shape %S.",
                               i1, shape1, i2, shape2);
    Py_DECREF(shape1);
    Py_DECREF(shape2);
    if (msg == NULL) {
        return -1;
    }
    PyErr_SetObject(PyExc_ValueError, msg);
    Py_DECREF(msg);
    return 0;
}

/* Adjust dimensionality and strides for index object iterators
   --- i.e. broadcast
*/
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Broadcast(PyArrayMultiIterObject *mit)
{
    int i, nd, k, j;
    int src_iter = -1;  /* Initializing avoids a compiler warning. */
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
                    src_iter = j;
                }
                else if (mit->dimensions[i] != tmp) {
                    set_shape_mismatch_exception(mit, src_iter, j);
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

static inline PyObject*
multiiter_wrong_number_of_args(void)
{
    return PyErr_Format(PyExc_ValueError,
                        "Need at least 0 and at most %d "
                        "array objects.", NPY_MAXARGS);
}

/*
 * Common implementation for all PyArrayMultiIterObject constructors.
 */
static PyObject*
multiiter_new_impl(int n_args, PyObject **args)
{
    PyArrayMultiIterObject *multi;
    int i;

    multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);
    multi->numiter = 0;

    for (i = 0; i < n_args; ++i) {
        PyObject *obj = args[i];
        PyObject *arr;
        PyArrayIterObject *it;

        if (PyObject_IsInstance(obj, (PyObject *)&PyArrayMultiIter_Type)) {
            PyArrayMultiIterObject *mit = (PyArrayMultiIterObject *)obj;
            int j;

            if (multi->numiter + mit->numiter > NPY_MAXARGS) {
                multiiter_wrong_number_of_args();
                goto fail;
            }
            for (j = 0; j < mit->numiter; ++j) {
                arr = (PyObject *)mit->iters[j]->ao;
                it = (PyArrayIterObject *)PyArray_IterNew(arr);
                if (it == NULL) {
                    goto fail;
                }
                multi->iters[multi->numiter++] = it;
            }
        }
        else if (multi->numiter < NPY_MAXARGS) {
            arr = PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
            if (arr == NULL) {
                goto fail;
            }
            it = (PyArrayIterObject *)PyArray_IterNew(arr);
            Py_DECREF(arr);
            if (it == NULL) {
                goto fail;
            }
            multi->iters[multi->numiter++] = it;
        }
        else {
            multiiter_wrong_number_of_args();
            goto fail;
        }
    }

    if (multi->numiter < 0) {
        multiiter_wrong_number_of_args();
        goto fail;
    }
    if (PyArray_Broadcast(multi) < 0) {
        goto fail;
    }
    PyArray_MultiIter_RESET(multi);

    return (PyObject *)multi;

fail:
    Py_DECREF(multi);

    return NULL;
}

/*NUMPY_API
 * Get MultiIterator from array of Python objects and any additional
 *
 * PyObject **mps - array of PyObjects
 * int n - number of PyObjects in the array
 * int nadd - number of additional arrays to include in the iterator.
 *
 * Returns a multi-iterator object.
 */
NPY_NO_EXPORT PyObject*
PyArray_MultiIterFromObjects(PyObject **mps, int n, int nadd, ...)
{
    PyObject *args_impl[NPY_MAXARGS];
    int ntot = n + nadd;
    int i;
    va_list va;

    if ((ntot > NPY_MAXARGS) || (ntot < 0)) {
        return multiiter_wrong_number_of_args();
    }

    for (i = 0; i < n; ++i) {
        args_impl[i] = mps[i];
    }

    va_start(va, nadd);
    for (; i < ntot; ++i) {
        args_impl[i] = va_arg(va, PyObject *);
    }
    va_end(va);

    return multiiter_new_impl(ntot, args_impl);
}

/*NUMPY_API
 * Get MultiIterator,
 */
NPY_NO_EXPORT PyObject*
PyArray_MultiIterNew(int n, ...)
{
    PyObject *args_impl[NPY_MAXARGS];
    int i;
    va_list va;

    if ((n > NPY_MAXARGS) || (n < 0)) {
        return multiiter_wrong_number_of_args();
    }

    va_start(va, n);
    for (i = 0; i < n; ++i) {
        args_impl[i] = va_arg(va, PyObject *);
    }
    va_end(va);

    return multiiter_new_impl(n, args_impl);
}


static PyObject*
arraymultiter_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args,
                  PyObject *kwds)
{
    PyObject *ret, *fast_seq;
    Py_ssize_t n;

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_ValueError,
                        "keyword arguments not accepted.");
        return NULL;
    }
    fast_seq = PySequence_Fast(args, "");  // noqa: borrowed-ref OK
    if (fast_seq == NULL) {
        return NULL;
    }
    NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST(args)
    n = PySequence_Fast_GET_SIZE(fast_seq);
    if (n > NPY_MAXARGS) {
        ret = multiiter_wrong_number_of_args();
    } else {
        ret = multiiter_new_impl(n, PySequence_Fast_ITEMS(fast_seq));
    }
    Py_DECREF(fast_seq);
    NPY_END_CRITICAL_SECTION_SEQUENCE_FAST()
    return ret;
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
arraymultiter_size_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(self->size);
}

static PyObject *
arraymultiter_index_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_PyIntFromIntp(self->index);
}

static PyObject *
arraymultiter_shape_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    return PyArray_IntTupleFromIntp(self->nd, self->dimensions);
}

static PyObject *
arraymultiter_iters_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
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
    {NULL, NULL, 0, NULL},      /* sentinel */
};

NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.broadcast",
    .tp_basicsize = sizeof(PyArrayMultiIterObject),
    .tp_dealloc = (destructor)arraymultiter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iternext = (iternextfunc)arraymultiter_next,
    .tp_methods = arraymultiter_methods,
    .tp_members = arraymultiter_members,
    .tp_getset = arraymultiter_getsetlist,
    .tp_new = arraymultiter_new,
};

/*========================= Neighborhood iterator ======================*/

static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter);

static char* _set_constant(PyArrayNeighborhoodIterObject* iter,
        PyArrayObject *fill)
{
    char *ret;
    PyArrayIterObject *ar = iter->_internal_iter;
    int storeflags, st;

    ret = PyDataMem_NEW(PyArray_ITEMSIZE(ar->ao));
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
get_ptr_constant(PyArrayIterObject* _iter, const npy_intp *coordinates)
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
static inline npy_intp
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
get_ptr_mirror(PyArrayIterObject* _iter, const npy_intp *coordinates)
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
static inline npy_intp
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
get_ptr_circular(PyArrayIterObject* _iter, const npy_intp *coordinates)
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
PyArray_NeighborhoodIterNew(PyArrayIterObject *x, const npy_intp *bounds,
                            int mode, PyArrayObject* fill)
{
    int i;
    PyArrayNeighborhoodIterObject *ret;

    ret = PyArray_malloc(sizeof(*ret));
    if (ret == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)ret, &PyArrayNeighborhoodIter_Type);

    Py_INCREF(x->ao);  /* PyArray_RawIterBaseInit steals a reference */
    PyArray_RawIterBaseInit((PyArrayIterObject*)ret, x->ao);
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
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.neigh_internal_iter",
    .tp_basicsize = sizeof(PyArrayNeighborhoodIterObject),
    .tp_dealloc = (destructor)neighiter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
