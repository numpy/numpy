#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "arrayobject.h"
#include "arrayiterators.h"

#define PseudoIndex -1
#define RubberIndex -2
#define SingleIndex -3

NPY_NO_EXPORT intp
parse_subindex(PyObject *op, intp *step_size, intp *n_steps, intp max)
{
    intp index;

    if (op == Py_None) {
        *n_steps = PseudoIndex;
        index = 0;
    }
    else if (op == Py_Ellipsis) {
        *n_steps = RubberIndex;
        index = 0;
    }
    else if (PySlice_Check(op)) {
        intp stop;
        if (slice_GetIndices((PySliceObject *)op, max,
                             &index, &stop, step_size, n_steps) < 0) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_IndexError,
                                "invalid slice");
            }
            goto fail;
        }
        if (*n_steps <= 0) {
            *n_steps = 0;
            *step_size = 1;
            index = 0;
        }
    }
    else {
        index = PyArray_PyIntAsIntp(op);
        if (error_converting(index)) {
            PyErr_SetString(PyExc_IndexError,
                            "each subindex must be either a "\
                            "slice, an integer, Ellipsis, or "\
                            "newaxis");
            goto fail;
        }
        *n_steps = SingleIndex;
        *step_size = 0;
        if (index < 0) {
            index += max;
        }
        if (index >= max || index < 0) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            goto fail;
        }
    }
    return index;

 fail:
    return -1;
}


NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            intp *dimensions, intp *strides, intp *offset_ptr)
{
    int i, j, n;
    int nd_old, nd_new, n_add, n_pseudo;
    intp n_steps, start, offset, step_size;
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
                            "index must be either an int "\
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
            if (!(op1=PySequence_GetItem(op, i))) {
                PyErr_SetString(PyExc_IndexError,
                                "invalid index");
                return -1;
            }
        }
        start = parse_subindex(op1, &step_size, &n_steps,
                               nd_old < self->nd ?
                               self->dimensions[nd_old] : 0);
        Py_DECREF(op1);
        if (start == -1) {
            break;
        }
        if (n_steps == PseudoIndex) {
            dimensions[nd_new] = 1; strides[nd_new] = 0;
            nd_new++;
        }
        else {
            if (n_steps == RubberIndex) {
                for (j = i + 1, n_pseudo = 0; j < n; j++) {
                    op1 = PySequence_GetItem(op, j);
                    if (op1 == Py_None) {
                        n_pseudo++;
                    }
                    Py_DECREF(op1);
                }
                n_add = self->nd-(n-i-n_pseudo-1+nd_old);
                if (n_add < 0) {
                    PyErr_SetString(PyExc_IndexError,
                                    "too many indices");
                    return -1;
                }
                for (j = 0; j < n_add; j++) {
                    dimensions[nd_new] = \
                        self->dimensions[nd_old];
                    strides[nd_new] = \
                        self->strides[nd_old];
                    nd_new++; nd_old++;
                }
            }
            else {
                if (nd_old >= self->nd) {
                    PyErr_SetString(PyExc_IndexError,
                                    "too many indices");
                    return -1;
                }
                offset += self->strides[nd_old]*start;
                nd_old++;
                if (n_steps != SingleIndex) {
                    dimensions[nd_new] = n_steps;
                    strides[nd_new] = step_size * \
                        self->strides[nd_old-1];
                    nd_new++;
                }
            }
        }
    }
    if (i < n) {
        return -1;
    }
    n_add = self->nd-nd_old;
    for (j = 0; j < n_add; j++) {
        dimensions[nd_new] = self->dimensions[nd_old];
        strides[nd_new] = self->strides[nd_old];
        nd_new++;
        nd_old++;
    }
    *offset_ptr = offset;
    return nd_new;
}

static int
slice_coerce_index(PyObject *o, intp *v)
{
    *v = PyArray_PyIntAsIntp(o);
    if (error_converting(*v)) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

/* This is basically PySlice_GetIndicesEx, but with our coercion
 * of indices to integers (plus, that function is new in Python 2.3) */
NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, intp length,
                 intp *start, intp *stop, intp *step,
                 intp *slicelength)
{
    intp defstop;

    if (r->step == Py_None) {
        *step = 1;
    }
    else {
        if (!slice_coerce_index(r->step, step)) {
            return -1;
        }
        if (*step == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "slice step cannot be zero");
            return -1;
        }
    }
    /* defstart = *step < 0 ? length - 1 : 0; */
    defstop = *step < 0 ? -1 : length;
    if (r->start == Py_None) {
        *start = *step < 0 ? length-1 : 0;
    }
    else {
        if (!slice_coerce_index(r->start, start)) {
            return -1;
        }
        if (*start < 0) {
            *start += length;
        }
        if (*start < 0) {
            *start = (*step < 0) ? -1 : 0;
        }
        if (*start >= length) {
            *start = (*step < 0) ? length - 1 : length;
        }
    }

    if (r->stop == Py_None) {
        *stop = defstop;
    }
    else {
        if (!slice_coerce_index(r->stop, stop)) {
            return -1;
        }
        if (*stop < 0) {
            *stop += length;
        }
        if (*stop < 0) {
            *stop = -1;
        }
        if (*stop > length) {
            *stop = length;
        }
    }

    if ((*step < 0 && *stop >= *start) ||
        (*step > 0 && *start >= *stop)) {
        *slicelength = 0;
    }
    else if (*step < 0) {
        *slicelength = (*stop - *start + 1) / (*step) + 1;
    }
    else {
        *slicelength = (*stop - *start - 1) / (*step) + 1;
    }

    return 0;
}

/*********************** Element-wise Array Iterator ***********************/
/*  Aided by Peter J. Verveer's  nd_image package and numpy's arraymap  ****/
/*         and Python's array iterator                                   ***/

/*NUMPY_API
 * Get Iterator.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterNew(PyObject *obj)
{
    PyArrayIterObject *it;
    int i, nd;
    PyArrayObject *ao = (PyArrayObject *)obj;

    if (!PyArray_Check(ao)) {
        PyErr_BadInternalCall();
        return NULL;
    }

    it = (PyArrayIterObject *)_pya_malloc(sizeof(PyArrayIterObject));
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);
    /* it = PyObject_New(PyArrayIterObject, &PyArrayIter_Type);*/
    if (it == NULL) {
        return NULL;
    }
    nd = ao->nd;
    PyArray_UpdateFlags(ao, CONTIGUOUS);
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
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = ao->dimensions[i] - 1;
        it->strides[i] = ao->strides[i];
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * ao->dimensions[nd-i];
        }
    }
    PyArray_ITER_RESET(it);

    return (PyObject *)it;
}

/*NUMPY_API
 * Get Iterator broadcast to a particular shape
 */
NPY_NO_EXPORT PyObject *
PyArray_BroadcastToShape(PyObject *obj, intp *dims, int nd)
{
    PyArrayIterObject *it;
    int i, diff, j, compat, k;
    PyArrayObject *ao = (PyArrayObject *)obj;

    if (ao->nd > nd) {
        goto err;
    }
    compat = 1;
    diff = j = nd - ao->nd;
    for (i = 0; i < ao->nd; i++, j++) {
        if (ao->dimensions[i] == 1) {
            continue;
        }
        if (ao->dimensions[i] != dims[j]) {
            compat = 0;
            break;
        }
    }
    if (!compat) {
        goto err;
    }
    it = (PyArrayIterObject *)_pya_malloc(sizeof(PyArrayIterObject));
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);

    if (it == NULL) {
        return NULL;
    }
    PyArray_UpdateFlags(ao, CONTIGUOUS);
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
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = dims[i] - 1;
        k = i - diff;
        if ((k < 0) || ao->dimensions[k] != dims[i]) {
            it->contiguous = 0;
            it->strides[i] = 0;
        }
        else {
            it->strides[i] = ao->strides[k];
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
    PyArrayIterObject *it;
    int axis;
    it = (PyArrayIterObject *)PyArray_IterNew(obj);
    if (it == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(obj)==0) {
        return (PyObject *)it;
    }
    if (*inaxis < 0) {
        int i, minaxis = 0;
        intp minstride = 0;
        i = 0;
        while (minstride == 0 && i < PyArray_NDIM(obj)) {
            minstride = PyArray_STRIDE(obj,i);
            i++;
        }
        for (i = 1; i < PyArray_NDIM(obj); i++) {
            if (PyArray_STRIDE(obj,i) > 0 &&
                PyArray_STRIDE(obj, i) < minstride) {
                minaxis = i;
                minstride = PyArray_STRIDE(obj,i);
            }
        }
        *inaxis = minaxis;
    }
    axis = *inaxis;
    /* adjust so that will not iterate over axis */
    it->contiguous = 0;
    if (it->size != 0) {
        it->size /= PyArray_DIM(obj,axis);
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
    intp smallest;
    intp sumstrides[NPY_MAXDIMS];

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
    Py_XDECREF(it->ao);
    _pya_free(it);
}

static Py_ssize_t
iter_length(PyArrayIterObject *self)
{
    return self->size;
}


static PyObject *
iter_subscript_Bool(PyArrayIterObject *self, PyArrayObject *ind)
{
    intp index, strides;
    int itemsize;
    intp count = 0;
    char *dptr, *optr;
    PyObject *r;
    int swap;
    PyArray_CopySwapFunc *copyswap;


    if (ind->nd != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return NULL;
    }
    index = ind->dimensions[0];
    if (index > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "too many boolean indices");
        return NULL;
    }

    strides = ind->strides[0];
    dptr = ind->data;
    /* Get size of return array */
    while (index--) {
        if (*((Bool *)dptr) != 0) {
            count++;
        }
        dptr += strides;
    }
    itemsize = self->ao->descr->elsize;
    Py_INCREF(self->ao->descr);
    r = PyArray_NewFromDescr(self->ao->ob_type,
                             self->ao->descr, 1, &count,
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (r == NULL) {
        return NULL;
    }
    /* Set up loop */
    optr = PyArray_DATA(r);
    index = ind->dimensions[0];
    dptr = ind->data;
    copyswap = self->ao->descr->f->copyswap;
    /* Loop over Boolean array */
    swap = (PyArray_ISNOTSWAPPED(self->ao) != PyArray_ISNOTSWAPPED(r));
    while (index--) {
        if (*((Bool *)dptr) != 0) {
            copyswap(optr, self->dataptr, swap, self->ao);
            optr += itemsize;
        }
        dptr += strides;
        PyArray_ITER_NEXT(self);
    }
    PyArray_ITER_RESET(self);
    return r;
}

static PyObject *
iter_subscript_int(PyArrayIterObject *self, PyArrayObject *ind)
{
    intp num;
    PyObject *r;
    PyArrayIterObject *ind_it;
    int itemsize;
    int swap;
    char *optr;
    intp index;
    PyArray_CopySwapFunc *copyswap;

    itemsize = self->ao->descr->elsize;
    if (ind->nd == 0) {
        num = *((intp *)ind->data);
        if (num < 0) {
            num += self->size;
        }
        if (num < 0 || num >= self->size) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds"   \
                         " 0<=index<%"INTP_FMT,
                         num, self->size);
            r = NULL;
        }
        else {
            PyArray_ITER_GOTO1D(self, num);
            r = PyArray_ToScalar(self->dataptr, self->ao);
        }
        PyArray_ITER_RESET(self);
        return r;
    }

    Py_INCREF(self->ao->descr);
    r = PyArray_NewFromDescr(self->ao->ob_type, self->ao->descr,
                             ind->nd, ind->dimensions,
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (r == NULL) {
        return NULL;
    }
    optr = PyArray_DATA(r);
    ind_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    if (ind_it == NULL) {
        Py_DECREF(r);
        return NULL;
    }
    index = ind_it->size;
    copyswap = PyArray_DESCR(r)->f->copyswap;
    swap = (PyArray_ISNOTSWAPPED(r) != PyArray_ISNOTSWAPPED(self->ao));
    while (index--) {
        num = *((intp *)(ind_it->dataptr));
        if (num < 0) {
            num += self->size;
        }
        if (num < 0 || num >= self->size) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds" \
                         " 0<=index<%"INTP_FMT,
			 num, self->size);
            Py_DECREF(ind_it);
            Py_DECREF(r);
            PyArray_ITER_RESET(self);
            return NULL;
        }
        PyArray_ITER_GOTO1D(self, num);
        copyswap(optr, self->dataptr, swap, r);
        optr += itemsize;
        PyArray_ITER_NEXT(ind_it);
    }
    Py_DECREF(ind_it);
    PyArray_ITER_RESET(self);
    return r;
}

/* Always returns arrays */
NPY_NO_EXPORT PyObject *
iter_subscript(PyArrayIterObject *self, PyObject *ind)
{
    PyArray_Descr *indtype = NULL;
    intp start, step_size;
    intp n_steps;
    PyObject *r;
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
     * Check for Boolean -- this is first becasue Bool is a subclass of Int
     */
    PyArray_ITER_RESET(self);

    if (PyBool_Check(ind)) {
        if (PyObject_IsTrue(ind)) {
            return PyArray_ToScalar(self->dataptr, self->ao);
        }
        else { /* empty array */
            intp ii = 0;
            Py_INCREF(self->ao->descr);
            r = PyArray_NewFromDescr(self->ao->ob_type,
                                     self->ao->descr,
                                     1, &ii,
                                     NULL, NULL, 0,
                                     (PyObject *)self->ao);
            return r;
        }
    }

    /* Check for Integer or Slice */
    if (PyLong_Check(ind) || PyInt_Check(ind) || PySlice_Check(ind)) {
        start = parse_subindex(ind, &step_size, &n_steps,
                               self->size);
        if (start == -1) {
            goto fail;
        }
        if (n_steps == RubberIndex || n_steps == PseudoIndex) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto fail;
        }
        PyArray_ITER_GOTO1D(self, start)
            if (n_steps == SingleIndex) { /* Integer */
                r = PyArray_ToScalar(self->dataptr, self->ao);
                PyArray_ITER_RESET(self);
                return r;
            }
        size = self->ao->descr->elsize;
        Py_INCREF(self->ao->descr);
        r = PyArray_NewFromDescr(self->ao->ob_type,
                                 self->ao->descr,
                                 1, &n_steps,
                                 NULL, NULL,
                                 0, (PyObject *)self->ao);
        if (r == NULL) {
            goto fail;
        }
        dptr = PyArray_DATA(r);
        copyswap = PyArray_DESCR(r)->f->copyswap;
        while (n_steps--) {
            copyswap(dptr, self->dataptr, 0, r);
            start += step_size;
            PyArray_ITER_GOTO1D(self, start)
                dptr += size;
        }
        PyArray_ITER_RESET(self);
        return r;
    }

    /* convert to INTP array if Integer array scalar or List */
    indtype = PyArray_DescrFromType(PyArray_INTP);
    if (PyArray_IsScalar(ind, Integer) || PyList_Check(ind)) {
        Py_INCREF(indtype);
        obj = PyArray_FromAny(ind, indtype, 0, 0, FORCECAST, NULL);
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
        if (PyArray_TYPE(obj)==PyArray_BOOL) {
            r = iter_subscript_Bool(self, (PyArrayObject *)obj);
            Py_DECREF(indtype);
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER(obj)) {
            PyObject *new;
            new = PyArray_FromAny(obj, indtype, 0, 0,
                                  FORCECAST | ALIGNED, NULL);
            if (new == NULL) {
                goto fail;
            }
            Py_DECREF(obj);
            obj = new;
            r = iter_subscript_int(self, (PyArrayObject *)obj);
        }
        else {
            goto fail;
        }
        Py_DECREF(obj);
        return r;
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
    intp index, strides;
    char *dptr;
    PyArray_CopySwapFunc *copyswap;

    if (ind->nd != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return -1;
    }

    index = ind->dimensions[0];
    if (index > self->size) {
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array has too many values");
        return -1;
    }

    strides = ind->strides[0];
    dptr = ind->data;
    PyArray_ITER_RESET(self);
    /* Loop over Boolean array */
    copyswap = self->ao->descr->f->copyswap;
    while (index--) {
        if (*((Bool *)dptr) != 0) {
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
    PyArray_Descr *typecode;
    intp num;
    PyArrayIterObject *ind_it;
    intp index;
    PyArray_CopySwapFunc *copyswap;

    typecode = self->ao->descr;
    copyswap = self->ao->descr->f->copyswap;
    if (ind->nd == 0) {
        num = *((intp *)ind->data);
        PyArray_ITER_GOTO1D(self, num);
        copyswap(self->dataptr, val->dataptr, swap, self->ao);
        return 0;
    }
    ind_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    if (ind_it == NULL) {
        return -1;
    }
    index = ind_it->size;
    while (index--) {
        num = *((intp *)(ind_it->dataptr));
        if (num < 0) {
            num += self->size;
        }
        if ((num < 0) || (num >= self->size)) {
            PyErr_Format(PyExc_IndexError,
                         "index %"INTP_FMT" out of bounds"           \
                         " 0<=index<%"INTP_FMT, num,
                         self->size);
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
    PyObject *arrval = NULL;
    PyArrayIterObject *val_it = NULL;
    PyArray_Descr *type;
    PyArray_Descr *indtype = NULL;
    int swap, retval = -1;
    intp start, step_size;
    intp n_steps;
    PyObject *obj = NULL;
    PyArray_CopySwapFunc *copyswap;


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

    type = self->ao->descr;

    /*
     * Check for Boolean -- this is first becasue
     * Bool is a subclass of Int
     */
    if (PyBool_Check(ind)) {
        retval = 0;
        if (PyObject_IsTrue(ind)) {
            retval = type->f->setitem(val, self->dataptr, self->ao);
        }
        goto finish;
    }

    if (PySequence_Check(ind) || PySlice_Check(ind)) {
        goto skip;
    }
    start = PyArray_PyIntAsIntp(ind);
    if (start==-1 && PyErr_Occurred()) {
        PyErr_Clear();
    }
    else {
        if (start < -self->size || start >= self->size) {
            PyErr_Format(PyExc_ValueError,
                         "index (%" NPY_INTP_FMT \
                         ") out of range", start);
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
    arrval = PyArray_FromAny(val, type, 0, 0, 0, NULL);
    if (arrval == NULL) {
        return -1;
    }
    val_it = (PyArrayIterObject *)PyArray_IterNew(arrval);
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
        start = parse_subindex(ind, &step_size, &n_steps, self->size);
        if (start == -1) {
            goto finish;
        }
        if (n_steps == RubberIndex || n_steps == PseudoIndex) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto finish;
        }
        PyArray_ITER_GOTO1D(self, start);
        if (n_steps == SingleIndex) {
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
    indtype = PyArray_DescrFromType(PyArray_INTP);
    if (PyList_Check(ind)) {
        Py_INCREF(indtype);
        obj = PyArray_FromAny(ind, indtype, 0, 0, FORCECAST, NULL);
    }
    else {
        Py_INCREF(ind);
        obj = ind;
    }

    if (obj != NULL && PyArray_Check(obj)) {
        /* Check for Boolean object */
        if (PyArray_TYPE(obj)==PyArray_BOOL) {
            if (iter_ass_sub_Bool(self, (PyArrayObject *)obj,
                                  val_it, swap) < 0) {
                goto finish;
            }
            retval=0;
        }
        /* Check for integer array */
        else if (PyArray_ISINTEGER(obj)) {
            PyObject *new;
            Py_INCREF(indtype);
            new = PyArray_CheckFromAny(obj, indtype, 0, 0,
                                       FORCECAST | BEHAVED_NS, NULL);
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
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)iter_length,                   /*mp_length*/
#else
    (inquiry)iter_length,                   /*mp_length*/
#endif
    (binaryfunc)iter_subscript,             /*mp_subscript*/
    (objobjargproc)iter_ass_subscript,      /*mp_ass_subscript*/
};



static PyObject *
iter_array(PyArrayIterObject *it, PyObject *NPY_UNUSED(op))
{

    PyObject *r;
    intp size;

    /* Any argument ignored */

    /* Two options:
     *  1) underlying array is contiguous
     *  -- return 1-d wrapper around it
     * 2) underlying array is not contiguous
     * -- make new 1-d contiguous array with updateifcopy flag set
     * to copy back to the old array
     */
    size = PyArray_SIZE(it->ao);
    Py_INCREF(it->ao->descr);
    if (PyArray_ISCONTIGUOUS(it->ao)) {
        r = PyArray_NewFromDescr(&PyArray_Type,
                                 it->ao->descr,
                                 1, &size,
                                 NULL, it->ao->data,
                                 it->ao->flags,
                                 (PyObject *)it->ao);
        if (r == NULL) {
            return NULL;
        }
    }
    else {
        r = PyArray_NewFromDescr(&PyArray_Type,
                                 it->ao->descr,
                                 1, &size,
                                 NULL, NULL,
                                 0, (PyObject *)it->ao);
        if (r == NULL) {
            return NULL;
        }
        if (_flat_copyinto(r, (PyObject *)it->ao,
                           PyArray_CORDER) < 0) {
            Py_DECREF(r);
            return NULL;
        }
        PyArray_FLAGS(r) |= UPDATEIFCOPY;
        it->ao->flags &= ~WRITEABLE;
    }
    Py_INCREF(it->ao);
    PyArray_BASE(r) = (PyObject *)it->ao;
    return r;

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
    {"__array__", (PyCFunction)iter_array, 1, NULL},
    {"copy", (PyCFunction)iter_copy, 1, NULL},
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
    Py_DECREF(new);
    return ret;
}


static PyMemberDef iter_members[] = {
    {"base", T_OBJECT, offsetof(PyArrayIterObject, ao), RO, NULL},
    {"index", T_INT, offsetof(PyArrayIterObject, index), RO, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
iter_coords_get(PyArrayIterObject *self)
{
    int nd;
    nd = self->ao->nd;
    if (self->contiguous) {
        /*
         * coordinates not kept track of ---
         * need to generate from index
         */
        intp val;
        int i;
        val = self->index;
        for (i = 0; i < nd; i++) {
            self->coordinates[i] = val / self->factors[i];
            val = val % self->factors[i];
        }
    }
    return PyArray_IntTupleFromIntp(nd, self->coordinates);
}

static PyGetSetDef iter_getsets[] = {
    {"coords",
     (getter)iter_coords_get,
     NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

NPY_NO_EXPORT PyTypeObject PyArrayIter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
    "numpy.flatiter",                            /* tp_name */
    sizeof(PyArrayIterObject),                   /* tp_basicsize */
    0,                                           /* tp_itemsize */
    /* methods */
    (destructor)arrayiter_dealloc,               /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    &iter_as_mapping,                            /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call */
    0,                                           /* tp_str */
    0,                                           /* tp_getattro */
    0,                                           /* tp_setattro */
    0,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                          /* tp_flags */
    0,                                           /* tp_doc */
    0,                                           /* tp_traverse */
    0,                                           /* tp_clear */
    (richcmpfunc)iter_richcompare,               /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    (iternextfunc)arrayiter_next,                /* tp_iternext */
    iter_methods,                                /* tp_methods */
    iter_members,                                /* tp_members */
    iter_getsets,                                /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,   				         /* tp_dictoffset */
    0,   				         /* tp_init */
    0,   				         /* tp_alloc */
    0,   				         /* tp_new */
    0,   				         /* tp_free */
    0,   				         /* tp_is_gc */
    0,   				         /* tp_bases */
    0,   				         /* tp_mro */
    0,   				         /* tp_cache */
    0,   				         /* tp_subclasses */
    0,   				         /* tp_weaklist */
    0,   				         /* tp_del */
#ifdef COUNT_ALLOCS
    /* these must be last and never explicitly initialized */
    0,                                           /* tp_allocs */
    0,                                           /* tp_frees */
    0,                                           /* tp_maxalloc */
    0,                                           /* tp_prev */
    0,                                           /* *tp_next */
#endif

};

/** END of Array Iterator **/

/*********************** Subscript Array Iterator *************************
 *                                                                        *
 * This object handles subscript behavior for array objects.              *
 *  It is an iterator object with a next method                           *
 *  It abstracts the n-dimensional mapping behavior to make the looping   *
 *     code more understandable (maybe)                                   *
 *     and so that indexing can be set up ahead of time                   *
 */


static int _nonzero_indices(PyObject *myBool, PyArrayIterObject **iters);
/* convert an indexing object to an INTP indexing array iterator
   if possible -- otherwise, it is a Slice or Ellipsis object
   and has to be interpreted on bind to a particular
   array so leave it NULL for now.
*/
static int
_convert_obj(PyObject *obj, PyArrayIterObject **iter)
{
    PyArray_Descr *indtype;
    PyObject *arr;

    if (PySlice_Check(obj) || (obj == Py_Ellipsis)) {
        return 0;
    }
    else if (PyArray_Check(obj) && PyArray_ISBOOL(obj)) {
        return _nonzero_indices(obj, iter);
    }
    else {
        indtype = PyArray_DescrFromType(PyArray_INTP);
        arr = PyArray_FromAny(obj, indtype, 0, 0, FORCECAST, NULL);
        if (arr == NULL) {
            return -1;
        }
        *iter = (PyArrayIterObject *)PyArray_IterNew(arr);
        Py_DECREF(arr);
        if (*iter == NULL) {
            return -1;
        }
    }
    return 1;
}

/* Adjust dimensionality and strides for index object iterators
   --- i.e. broadcast
*/
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Broadcast(PyArrayMultiIterObject *mit)
{
    int i, nd, k, j;
    intp tmp;
    PyArrayIterObject *it;

    /* Discover the broadcast number of dimensions */
    for (i = 0, nd = 0; i < mit->numiter; i++) {
        nd = MAX(nd, mit->iters[i]->ao->nd);
    }
    mit->nd = nd;

    /* Discover the broadcast shape in each dimension */
    for (i = 0; i < nd; i++) {
        mit->dimensions[i] = 1;
        for (j = 0; j < mit->numiter; j++) {
            it = mit->iters[j];
            /* This prepends 1 to shapes not already equal to nd */
            k = i + it->ao->nd - nd;
            if (k >= 0) {
                tmp = it->ao->dimensions[k];
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
        nd = it->ao->nd;
        it->factors[mit->nd-1] = 1;
        for (j = 0; j < mit->nd; j++) {
            it->dims_m1[j] = mit->dimensions[j] - 1;
            k = j + nd - mit->nd;
            /*
             * If this dimension was added or shape of
             * underlying array was 1
             */
            if ((k < 0) ||
                it->ao->dimensions[k] != mit->dimensions[j]) {
                it->contiguous = 0;
                it->strides[j] = 0;
            }
            else {
                it->strides[j] = it->ao->strides[k];
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

/* Reset the map iterator to the beginning */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *mit)
{
    int i,j; intp coord[MAX_DIMS];
    PyArrayIterObject *it;
    PyArray_CopySwapFunc *copyswap;

    mit->index = 0;

    copyswap = mit->iters[0]->ao->descr->f->copyswap;

    if (mit->subspace != NULL) {
        memcpy(coord, mit->bscoord, sizeof(intp)*mit->ait->ao->nd);
        PyArray_ITER_RESET(mit->subspace);
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            PyArray_ITER_RESET(it);
            j = mit->iteraxes[i];
            copyswap(coord+j,it->dataptr, !PyArray_ISNOTSWAPPED(it->ao),
                     it->ao);
        }
        PyArray_ITER_GOTO(mit->ait, coord);
        mit->subspace->dataptr = mit->ait->dataptr;
        mit->dataptr = mit->subspace->dataptr;
    }
    else {
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            if (it->size != 0) {
                PyArray_ITER_RESET(it);
                copyswap(coord+i,it->dataptr, !PyArray_ISNOTSWAPPED(it->ao),
                         it->ao);
            }
            else {
                coord[i] = 0;
            }
        }
        PyArray_ITER_GOTO(mit->ait, coord);
        mit->dataptr = mit->ait->dataptr;
    }
    return;
}

/*
 * This function needs to update the state of the map iterator
 * and point mit->dataptr to the memory-location of the next object
 */
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit)
{
    int i, j;
    intp coord[MAX_DIMS];
    PyArrayIterObject *it;
    PyArray_CopySwapFunc *copyswap;

    mit->index += 1;
    if (mit->index >= mit->size) {
        return;
    }
    copyswap = mit->iters[0]->ao->descr->f->copyswap;
    /* Sub-space iteration */
    if (mit->subspace != NULL) {
        PyArray_ITER_NEXT(mit->subspace);
        if (mit->subspace->index >= mit->subspace->size) {
            /* reset coord to coordinates of beginning of the subspace */
            memcpy(coord, mit->bscoord, sizeof(intp)*mit->ait->ao->nd);
            PyArray_ITER_RESET(mit->subspace);
            for (i = 0; i < mit->numiter; i++) {
                it = mit->iters[i];
                PyArray_ITER_NEXT(it);
                j = mit->iteraxes[i];
                copyswap(coord+j,it->dataptr, !PyArray_ISNOTSWAPPED(it->ao),
                         it->ao);
            }
            PyArray_ITER_GOTO(mit->ait, coord);
            mit->subspace->dataptr = mit->ait->dataptr;
        }
        mit->dataptr = mit->subspace->dataptr;
    }
    else {
        for (i = 0; i < mit->numiter; i++) {
            it = mit->iters[i];
            PyArray_ITER_NEXT(it);
            copyswap(coord+i,it->dataptr,
                     !PyArray_ISNOTSWAPPED(it->ao),
                     it->ao);
        }
        PyArray_ITER_GOTO(mit->ait, coord);
        mit->dataptr = mit->ait->dataptr;
    }
    return;
}

/*
 * Bind a mapiteration to a particular array
 *
 *  Determine if subspace iteration is necessary.  If so,
 *  1) Fill in mit->iteraxes
 *  2) Create subspace iterator
 *  3) Update nd, dimensions, and size.
 *
 *  Subspace iteration is necessary if:  arr->nd > mit->numiter
 *
 * Need to check for index-errors somewhere.
 *
 * Let's do it at bind time and also convert all <0 values to >0 here
 * as well.
 */
NPY_NO_EXPORT void
PyArray_MapIterBind(PyArrayMapIterObject *mit, PyArrayObject *arr)
{
    int subnd;
    PyObject *sub, *obj = NULL;
    int i, j, n, curraxis, ellipexp, noellip;
    PyArrayIterObject *it;
    intp dimsize;
    intp *indptr;

    subnd = arr->nd - mit->numiter;
    if (subnd < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "too many indices for array");
        return;
    }

    mit->ait = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (mit->ait == NULL) {
        return;
    }
    /* no subspace iteration needed.  Finish up and Return */
    if (subnd == 0) {
        n = arr->nd;
        for (i = 0; i < n; i++) {
            mit->iteraxes[i] = i;
        }
        goto finish;
    }

    /*
     * all indexing arrays have been converted to 0
     * therefore we can extract the subspace with a simple
     * getitem call which will use view semantics
     *
     * But, be sure to do it with a true array.
     */
    if (PyArray_CheckExact(arr)) {
        sub = array_subscript_simple(arr, mit->indexobj);
    }
    else {
        Py_INCREF(arr);
        obj = PyArray_EnsureArray((PyObject *)arr);
        if (obj == NULL) {
            goto fail;
        }
        sub = array_subscript_simple((PyArrayObject *)obj, mit->indexobj);
        Py_DECREF(obj);
    }

    if (sub == NULL) {
        goto fail;
    }
    mit->subspace = (PyArrayIterObject *)PyArray_IterNew(sub);
    Py_DECREF(sub);
    if (mit->subspace == NULL) {
        goto fail;
    }
    /* Expand dimensions of result */
    n = mit->subspace->ao->nd;
    for (i = 0; i < n; i++) {
        mit->dimensions[mit->nd+i] = mit->subspace->ao->dimensions[i];
    }
    mit->nd += n;

    /*
     * Now, we still need to interpret the ellipsis and slice objects
     * to determine which axes the indexing arrays are referring to
     */
    n = PyTuple_GET_SIZE(mit->indexobj);
    /* The number of dimensions an ellipsis takes up */
    ellipexp = arr->nd - n + 1;
    /*
     * Now fill in iteraxes -- remember indexing arrays have been
     * converted to 0's in mit->indexobj
     */
    curraxis = 0;
    j = 0;
    /* Only expand the first ellipsis */
    noellip = 1;
    memset(mit->bscoord, 0, sizeof(intp)*arr->nd);
    for (i = 0; i < n; i++) {
        /*
         * We need to fill in the starting coordinates for
         * the subspace
         */
        obj = PyTuple_GET_ITEM(mit->indexobj, i);
        if (PyInt_Check(obj) || PyLong_Check(obj)) {
            mit->iteraxes[j++] = curraxis++;
        }
        else if (noellip && obj == Py_Ellipsis) {
            curraxis += ellipexp;
            noellip = 0;
        }
        else {
            intp start = 0;
            intp stop, step;
            /* Should be slice object or another Ellipsis */
            if (obj == Py_Ellipsis) {
                mit->bscoord[curraxis] = 0;
            }
            else if (!PySlice_Check(obj) ||
                     (slice_GetIndices((PySliceObject *)obj,
                                       arr->dimensions[curraxis],
                                       &start, &stop, &step,
                                       &dimsize) < 0)) {
                PyErr_Format(PyExc_ValueError,
                             "unexpected object "       \
                             "(%s) in selection position %d",
                             obj->ob_type->tp_name, i);
                goto fail;
            }
            else {
                mit->bscoord[curraxis] = start;
            }
            curraxis += 1;
        }
    }

 finish:
    /* Here check the indexes (now that we have iteraxes) */
    mit->size = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (mit->size < 0) {
	PyErr_SetString(PyExc_ValueError,
			"dimensions too large in fancy indexing");
	goto fail;
    }
    if (mit->ait->size == 0 && mit->size != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid index into a 0-size array");
        goto fail;
    }

    for (i = 0; i < mit->numiter; i++) {
        intp indval;
        it = mit->iters[i];
        PyArray_ITER_RESET(it);
        dimsize = arr->dimensions[mit->iteraxes[i]];
        while (it->index < it->size) {
            indptr = ((intp *)it->dataptr);
            indval = *indptr;
            if (indval < 0) {
                indval += dimsize;
            }
            if (indval < 0 || indval >= dimsize) {
                PyErr_Format(PyExc_IndexError,
                             "index (%"INTP_FMT") out of range "\
                             "(0<=index<%"INTP_FMT") in dimension %d",
                             indval, (dimsize-1), mit->iteraxes[i]);
                goto fail;
            }
            PyArray_ITER_NEXT(it);
        }
        PyArray_ITER_RESET(it);
    }
    return;

 fail:
    Py_XDECREF(mit->subspace);
    Py_XDECREF(mit->ait);
    mit->subspace = NULL;
    mit->ait = NULL;
    return;
}

/*
 * This function takes a Boolean array and constructs index objects and
 * iterators as if nonzero(Bool) had been called
 */
static int
_nonzero_indices(PyObject *myBool, PyArrayIterObject **iters)
{
    PyArray_Descr *typecode;
    PyArrayObject *ba = NULL, *new = NULL;
    int nd, j;
    intp size, i, count;
    Bool *ptr;
    intp coords[MAX_DIMS], dims_m1[MAX_DIMS];
    intp *dptr[MAX_DIMS];

    typecode=PyArray_DescrFromType(PyArray_BOOL);
    ba = (PyArrayObject *)PyArray_FromAny(myBool, typecode, 0, 0,
                                          CARRAY, NULL);
    if (ba == NULL) {
        return -1;
    }
    nd = ba->nd;
    for (j = 0; j < nd; j++) {
        iters[j] = NULL;
    }
    size = PyArray_SIZE(ba);
    ptr = (Bool *)ba->data;
    count = 0;

    /* pre-determine how many nonzero entries there are */
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            count++;
        }
    }

    /* create count-sized index arrays for each dimension */
    for (j = 0; j < nd; j++) {
        new = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &count,
                                           PyArray_INTP, NULL, NULL,
                                           0, 0, NULL);
        if (new == NULL) {
            goto fail;
        }
        iters[j] = (PyArrayIterObject *)
            PyArray_IterNew((PyObject *)new);
        Py_DECREF(new);
        if (iters[j] == NULL) {
            goto fail;
        }
        dptr[j] = (intp *)iters[j]->ao->data;
        coords[j] = 0;
        dims_m1[j] = ba->dimensions[j]-1;
    }
    ptr = (Bool *)ba->data;
    if (count == 0) {
        goto finish;
    }

    /*
     * Loop through the Boolean array  and copy coordinates
     * for non-zero entries
     */
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            for (j = 0; j < nd; j++) {
                *(dptr[j]++) = coords[j];
            }
        }
        /* Borrowed from ITER_NEXT macro */
        for (j = nd - 1; j >= 0; j--) {
            if (coords[j] < dims_m1[j]) {
                coords[j]++;
                break;
            }
            else {
                coords[j] = 0;
            }
        }
    }

 finish:
    Py_DECREF(ba);
    return nd;

 fail:
    for (j = 0; j < nd; j++) {
        Py_XDECREF(iters[j]);
    }
    Py_XDECREF(ba);
    return -1;
}

NPY_NO_EXPORT PyObject *
PyArray_MapIterNew(PyObject *indexobj, int oned, int fancy)
{
    PyArrayMapIterObject *mit;
    PyArray_Descr *indtype;
    PyObject *arr = NULL;
    int i, n, started, nonindex;

    if (fancy == SOBJ_BADARRAY) {
        PyErr_SetString(PyExc_IndexError,                       \
                        "arrays used as indices must be of "    \
                        "integer (or boolean) type");
        return NULL;
    }
    if (fancy == SOBJ_TOOMANY) {
        PyErr_SetString(PyExc_IndexError, "too many indices");
        return NULL;
    }

    mit = (PyArrayMapIterObject *)_pya_malloc(sizeof(PyArrayMapIterObject));
    PyObject_Init((PyObject *)mit, &PyArrayMapIter_Type);
    if (mit == NULL) {
        return NULL;
    }
    for (i = 0; i < MAX_DIMS; i++) {
        mit->iters[i] = NULL;
    }
    mit->index = 0;
    mit->ait = NULL;
    mit->subspace = NULL;
    mit->numiter = 0;
    mit->consec = 1;
    Py_INCREF(indexobj);
    mit->indexobj = indexobj;

    if (fancy == SOBJ_LISTTUP) {
        PyObject *newobj;
        newobj = PySequence_Tuple(indexobj);
        if (newobj == NULL) {
            goto fail;
        }
        Py_DECREF(indexobj);
        indexobj = newobj;
        mit->indexobj = indexobj;
    }

#undef SOBJ_NOTFANCY
#undef SOBJ_ISFANCY
#undef SOBJ_BADARRAY
#undef SOBJ_TOOMANY
#undef SOBJ_LISTTUP

    if (oned) {
        return (PyObject *)mit;
    }
    /*
     * Must have some kind of fancy indexing if we are here
     * indexobj is either a list, an arrayobject, or a tuple
     * (with at least 1 list or arrayobject or Bool object)
     */

    /* convert all inputs to iterators */
    if (PyArray_Check(indexobj) && (PyArray_TYPE(indexobj) == PyArray_BOOL)) {
        mit->numiter = _nonzero_indices(indexobj, mit->iters);
        if (mit->numiter < 0) {
            goto fail;
        }
        mit->nd = 1;
        mit->dimensions[0] = mit->iters[0]->dims_m1[0]+1;
        Py_DECREF(mit->indexobj);
        mit->indexobj = PyTuple_New(mit->numiter);
        if (mit->indexobj == NULL) {
            goto fail;
        }
        for (i = 0; i < mit->numiter; i++) {
            PyTuple_SET_ITEM(mit->indexobj, i, PyInt_FromLong(0));
        }
    }

    else if (PyArray_Check(indexobj) || !PyTuple_Check(indexobj)) {
        mit->numiter = 1;
        indtype = PyArray_DescrFromType(PyArray_INTP);
        arr = PyArray_FromAny(indexobj, indtype, 0, 0, FORCECAST, NULL);
        if (arr == NULL) {
            goto fail;
        }
        mit->iters[0] = (PyArrayIterObject *)PyArray_IterNew(arr);
        if (mit->iters[0] == NULL) {
            Py_DECREF(arr);
            goto fail;
        }
        mit->nd = PyArray_NDIM(arr);
        memcpy(mit->dimensions, PyArray_DIMS(arr), mit->nd*sizeof(intp));
        mit->size = PyArray_SIZE(arr);
        Py_DECREF(arr);
        Py_DECREF(mit->indexobj);
        mit->indexobj = Py_BuildValue("(N)", PyInt_FromLong(0));
    }
    else {
        /* must be a tuple */
        PyObject *obj;
        PyArrayIterObject **iterp;
        PyObject *new;
        int numiters, j, n2;
        /*
         * Make a copy of the tuple -- we will be replacing
         * index objects with 0's
         */
        n = PyTuple_GET_SIZE(indexobj);
        n2 = n;
        new = PyTuple_New(n2);
        if (new == NULL) {
            goto fail;
        }
        started = 0;
        nonindex = 0;
        j = 0;
        for (i = 0; i < n; i++) {
            obj = PyTuple_GET_ITEM(indexobj,i);
            iterp = mit->iters + mit->numiter;
            if ((numiters=_convert_obj(obj, iterp)) < 0) {
                Py_DECREF(new);
                goto fail;
            }
            if (numiters > 0) {
                started = 1;
                if (nonindex) {
                    mit->consec = 0;
                }
                mit->numiter += numiters;
                if (numiters == 1) {
                    PyTuple_SET_ITEM(new,j++, PyInt_FromLong(0));
                }
                else {
                    /*
                     * we need to grow the new indexing object and fill
                     * it with 0s for each of the iterators produced
                     */
                    int k;
                    n2 += numiters - 1;
                    if (_PyTuple_Resize(&new, n2) < 0) {
                        goto fail;
                    }
                    for (k = 0; k < numiters; k++) {
                        PyTuple_SET_ITEM(new, j++, PyInt_FromLong(0));
                    }
                }
            }
            else {
                if (started) {
                    nonindex = 1;
                }
                Py_INCREF(obj);
                PyTuple_SET_ITEM(new,j++,obj);
            }
        }
        Py_DECREF(mit->indexobj);
        mit->indexobj = new;
        /*
         * Store the number of iterators actually converted
         * These will be mapped to actual axes at bind time
         */
        if (PyArray_Broadcast((PyArrayMultiIterObject *)mit) < 0) {
            goto fail;
        }
    }

    return (PyObject *)mit;

 fail:
    Py_DECREF(mit);
    return NULL;
}


static void
arraymapiter_dealloc(PyArrayMapIterObject *mit)
{
    int i;
    Py_XDECREF(mit->indexobj);
    Py_XDECREF(mit->ait);
    Py_XDECREF(mit->subspace);
    for (i = 0; i < mit->numiter; i++) {
        Py_XDECREF(mit->iters[i]);
    }
    _pya_free(mit);
}

/*
 * The mapiter object must be created new each time.  It does not work
 * to bind to a new array, and continue.
 *
 * This was the orginal intention, but currently that does not work.
 * Do not expose the MapIter_Type to Python.
 *
 * It's not very useful anyway, since mapiter(indexobj); mapiter.bind(a);
 * mapiter is equivalent to a[indexobj].flat but the latter gets to use
 * slice syntax.
 */
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
    "numpy.mapiter",                             /* tp_name */
    sizeof(PyArrayIterObject),                   /* tp_basicsize */
    0,                                           /* tp_itemsize */
    /* methods */
    (destructor)arraymapiter_dealloc,            /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call */
    0,                                           /* tp_str */
    0,                                           /* tp_getattro */
    0,                                           /* tp_setattro */
    0,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                          /* tp_flags */
    0,                                           /* tp_doc */
    (traverseproc)0,                             /* tp_traverse */
    0,                                           /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    (iternextfunc)0,                             /* tp_iternext */
    0,                                           /* tp_methods */
    0,                                           /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)0,                                 /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,   				         /* tp_del */

#ifdef COUNT_ALLOCS
    /* these must be last and never explicitly initialized */
    0,                                           /* tp_allocs */
    0,                                           /* tp_frees */
    0,                                           /* tp_maxalloc */
    0,                                           /* tp_prev */
    0,                                           /* *tp_next */
#endif
};

/** END of Subscript Iterator **/


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
    if (ntot < 2 || ntot > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Need between 2 and (%d) "                 \
                     "array objects (inclusive).", NPY_MAXARGS);
        return NULL;
    }
    multi = _pya_malloc(sizeof(PyArrayMultiIterObject));
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

    if (n < 2 || n > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Need between 2 and (%d) "                 \
                     "array objects (inclusive).", NPY_MAXARGS);
        return NULL;
    }

    /* fprintf(stderr, "multi new...");*/

    multi = _pya_malloc(sizeof(PyArrayMultiIterObject));
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

    int n, i;
    PyArrayMultiIterObject *multi;
    PyObject *arr;

    if (kwds != NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "keyword arguments not accepted.");
        return NULL;
    }

    n = PyTuple_Size(args);
    if (n < 2 || n > NPY_MAXARGS) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        PyErr_Format(PyExc_ValueError,
                     "Need at least two and fewer than (%d) "   \
                     "array objects.", NPY_MAXARGS);
        return NULL;
    }

    multi = _pya_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);

    multi->numiter = n;
    multi->index = 0;
    for (i = 0; i < n; i++) {
        multi->iters[i] = NULL;
    }
    for (i = 0; i < n; i++) {
        arr = PyArray_FromAny(PyTuple_GET_ITEM(args, i), NULL, 0, 0, 0, NULL);
        if (arr == NULL) {
            goto fail;
        }
        if ((multi->iters[i] = (PyArrayIterObject *)PyArray_IterNew(arr))
                == NULL) {
            goto fail;
        }
        Py_DECREF(arr);
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
    return NULL;
}

static void
arraymultiter_dealloc(PyArrayMultiIterObject *multi)
{
    int i;

    for (i = 0; i < multi->numiter; i++) {
        Py_XDECREF(multi->iters[i]);
    }
    multi->ob_type->tp_free((PyObject *)multi);
}

static PyObject *
arraymultiter_size_get(PyArrayMultiIterObject *self)
{
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) self->size);
#else
    if (self->size < MAX_LONG) {
        return PyInt_FromLong((long) self->size);
    }
    else {
        return PyLong_FromLongLong((longlong) self->size);
    }
#endif
}

static PyObject *
arraymultiter_index_get(PyArrayMultiIterObject *self)
{
#if SIZEOF_INTP <= SIZEOF_LONG
    return PyInt_FromLong((long) self->index);
#else
    if (self->size < MAX_LONG) {
        return PyInt_FromLong((long) self->index);
    }
    else {
        return PyLong_FromLongLong((longlong) self->index);
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
     NULL, NULL, NULL},
    {"index",
     (getter)arraymultiter_index_get,
     NULL, NULL, NULL},
    {"shape",
     (getter)arraymultiter_shape_get,
     NULL, NULL, NULL},
    {"iters",
     (getter)arraymultiter_iters_get,
     NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMemberDef arraymultiter_members[] = {
    {"numiter", T_INT, offsetof(PyArrayMultiIterObject, numiter),
     RO, NULL},
    {"nd", T_INT, offsetof(PyArrayMultiIterObject, nd), RO, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraymultiter_reset(PyArrayMultiIterObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    PyArray_MultiIter_RESET(self);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef arraymultiter_methods[] = {
    {"reset", (PyCFunction) arraymultiter_reset, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
    "numpy.broadcast",                           /* tp_name */
    sizeof(PyArrayMultiIterObject),              /* tp_basicsize */
    0,                                           /* tp_itemsize */
    /* methods */
    (destructor)arraymultiter_dealloc,           /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call */
    0,                                           /* tp_str */
    0,                                           /* tp_getattro */
    0,                                           /* tp_setattro */
    0,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                          /* tp_flags */
    0,                                           /* tp_doc */
    0,                                           /* tp_traverse */
    0,                                           /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    (iternextfunc)arraymultiter_next,            /* tp_iternext */
    arraymultiter_methods,                       /* tp_methods */
    arraymultiter_members,                       /* tp_members */
    arraymultiter_getsetlist,                    /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)0,                                 /* tp_init */
    0,                                           /* tp_alloc */
    arraymultiter_new,                           /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */

#ifdef COUNT_ALLOCS
    /* these must be last and never explicitly initialized */
    0,                                           /* tp_allocs */
    0,                                           /* tp_frees */
    0,                                           /* tp_maxalloc */
    0,                                           /* tp_prev */
    0,                                           /* *tp_next */
#endif
};
