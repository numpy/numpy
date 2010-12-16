#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define _MULTIARRAYMODULE
#include <numpy/ndarrayobject.h>

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "new_iterator.h"

typedef struct NewNpyArrayIterObject_tag NewNpyArrayIterObject;

struct NewNpyArrayIterObject_tag {
    PyObject_HEAD
    /* The iterator */
    PyArray_NpyIter *iter;
    /* Flag indicating iteration stopped */
    char finished;
    /* Cached values from the iterator */
    NpyIter_IterNext_Fn iternext;
    NpyIter_GetCoords_Fn getcoords;
    char **dataptrs;
    PyArray_Descr **dtypes;
    PyObject **objects;
    npy_intp *innerstrides, *innerloopsizeptr;
    char readflags[NPY_MAXARGS];
    char writeflags[NPY_MAXARGS];
};

static PyObject *
npyiter_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    NewNpyArrayIterObject *self;

    self = (NewNpyArrayIterObject *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        self->iter = NULL;
    }

    return (PyObject *)self;
}

static int
npyiter_init(NewNpyArrayIterObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"op", "flags", "op_flags", "op_dtypes",
                             "min_ndim", "max_ndim", "op_axes"};

    PyObject *op_in, *flags_in = NULL, *op_flags_in = NULL,
                *op_dtypes_in = NULL, *op_axes_in = NULL;

    npy_intp iiter;

    npy_intp niter = 0;
    PyObject *op[NPY_MAXARGS];
    npy_uint32 flags = 0;
    npy_uint32 op_flags[NPY_MAXARGS];
    PyArray_Descr *op_request_dtypes[NPY_MAXARGS];
    int min_ndim = -1, max_ndim = -1;
    npy_intp oa_ndim = 0;
    npy_intp op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    npy_intp *op_axes[NPY_MAXARGS];

    if (self->iter != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was already initialized");
        return -1;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOOiiO", kwlist,
                    &op_in, &flags_in, &op_flags_in, &op_dtypes_in,
                    &min_ndim, &max_ndim, &op_axes_in)) {
        return -1;
    }

    /* niter and op */
    if (PyTuple_Check(op_in) || PyList_Check(op_in)) {
        niter = PySequence_Size(op_in);
        if (niter == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Must provide at least one op");
            return -1;
        }
        if (niter > NPY_MAXARGS) {
            PyErr_SetString(PyExc_ValueError, "Too many ops");
            return -1;
        }
        /* Initialize the ops and dtypes to NULL */
        memset(op, 0, sizeof(op[0])*niter);
        memset(op_request_dtypes, 0, sizeof(op_request_dtypes[0])*niter);
    
        for (iiter = 0; iiter < niter; ++iiter) {
            PyObject *item = PySequence_GetItem(op_in, iiter);
            if (item == NULL) {
                goto fail;
            }
            else if (item == Py_None) {
                Py_DECREF(item);
                item = NULL;
            }
            op[iiter] = item;
        }
    }
    else {
        niter = 1;
        Py_INCREF(op_in);
        op[0] = op_in;
        op_request_dtypes[0] = NULL;
    }
    /* flags */
    if (flags_in == NULL) {
        /* Default to behaving similarly to the old iterator */
        flags = NPY_ITER_C_ORDER_INDEX |
                NPY_ITER_COORDS |
                NPY_ITER_FORCE_C_ORDER;
    }
    else if (PyTuple_Check(flags_in) || PyList_Check(flags_in)) {
        int iflags, nflags = PySequence_Size(flags_in);

        flags = 0;
        for (iflags = 0; iflags < nflags; ++iflags) {
            PyObject *f;
            char *str = NULL;
            Py_ssize_t length = 0;
            npy_uint32 flag = 0;

            f = PySequence_GetItem(flags_in, iflags);
            if (f == NULL) {
                goto fail;
            }
            if (PyString_AsStringAndSize(f, &str, &length) == -1) {
                Py_DECREF(f);
                goto fail;
            }
            /* Use switch statements to quickly isolate the right flag */
            switch (str[0]) {
                case 'c':
                    switch (str[1]) {
                        case '_':
                            if (strcmp(str, "c_order_index") == 0) {
                                flag = NPY_ITER_C_ORDER_INDEX;
                            }
                            break;
                        case 'o':
                            if (strcmp(str, "coords") == 0) {
                                flag = NPY_ITER_COORDS;
                            }
                            break;
                    }
                    break;
                case 'f':
                    if (length >= 7) switch (str[6]) {
                        case 'r':
                            if (strcmp(str, "f_order_index") == 0) {
                                flag = NPY_ITER_F_ORDER_INDEX;
                            }
                            break;
                        case 'c':
                            if (strcmp(str, "force_c_order") == 0) {
                                flag = NPY_ITER_FORCE_C_ORDER;
                            }
                            break;
                        case 'f':
                            if (strcmp(str, "force_f_order") == 0) {
                                flag = NPY_ITER_FORCE_F_ORDER;
                            }
                            break;
                        case 'a':
                            if (strcmp(str, "force_any_contiguous") == 0) {
                                flag = NPY_ITER_FORCE_ANY_CONTIGUOUS;
                            }
                            break;
                    }
                    break;
                case 'n':
                    if (strcmp(str, "no_inner_iteration") == 0) {
                        flag = NPY_ITER_NO_INNER_ITERATION;
                    }
                    break;
                case 'o':
                    if (strcmp(str, "offsets") == 0) {
                        flag = NPY_ITER_OFFSETS;
                    }
                    break;
            }
            if (flag == 0) {
                PyErr_Format(PyExc_ValueError,
                        "Unexpected flag \"%s\"", str);
                Py_DECREF(f);
                goto fail;
            }
            else {
                flags |= flag;
            }
            Py_DECREF(f);
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Parameter 2 must be a tuple of flags");
        goto fail;
    }
    /* op_flags */
    if (op_flags_in == NULL) {
        for (iiter = 0; iiter < niter; ++iiter) {
            if (op[iiter] != NULL && PyArray_Check(op[iiter]) &&
                PyArray_CHKFLAGS(op[iiter], NPY_WRITEABLE)) {
                op_flags[iiter] = NPY_ITER_READWRITE;
            }
            else {
                op_flags[iiter] = NPY_ITER_READONLY;
            }
        }
    }
    else if (PyTuple_Check(op_flags_in) || PyList_Check(op_flags_in)) {
        if (PySequence_Size(op_flags_in) != niter) {
            PyErr_SetString(PyExc_ValueError,
                    "op_flags must be a tuple/list matching the number of ops");
            goto fail;
        }
        for (iiter = 0; iiter < niter; ++iiter) {
            int iflags, nflags;
            PyObject *f = PySequence_GetItem(op_flags_in, iiter);
            if (f == NULL || (!PyTuple_Check(f) && !PyList_Check(f))) {
                PyObject_Print(f, stderr, 0);
                Py_XDECREF(f);
                PyErr_SetString(PyExc_ValueError,
                        "Each entry of op_flags must be a tuple/list");
                goto fail;
            }
            
            nflags = PySequence_Size(f);
            op_flags[iiter] = 0;
            for (iflags = 0; iflags < nflags; ++iflags) {
                PyObject *f2 = PySequence_GetItem(f, iflags);
                char *str = NULL;
                Py_ssize_t length = 0;
                npy_uint32 flag = 0;

                if (PyString_AsStringAndSize(f2, &str, &length) == -1) {
                    Py_XDECREF(f2);
                    Py_DECREF(f);
                    goto fail;
                }
                /* Use switch statements to quickly isolate the right flag */
                switch (str[0]) {
                    case 'r':
                        if (length >= 8) switch (str[4]) {
                            case 'w':
                                if (strcmp(str, "readwrite") == 0) {
                                    flag = NPY_ITER_READWRITE;
                                }
                                break;
                            case 'o':
                                if (strcmp(str, "readonly") == 0) {
                                    flag = NPY_ITER_READONLY;
                                }
                                break;
                        }
                        break;
                    case 'w':
                        if (strcmp(str, "writeonly") == 0) {
                            flag = NPY_ITER_WRITEONLY;
                        }
                        break;
                    case 'n':
                        switch (str[1]) {
                            case 'b':
                                if (strcmp(str, "nbo_aligned") == 0) {
                                    flag = NPY_ITER_NBO_ALIGNED;
                                }
                                break;
                            case 'o':
                                if (strcmp(str, "no_subtype") == 0) {
                                    flag = NPY_ITER_NO_SUBTYPE;
                                }
                                break;
                        }
                        break;
                    case 'a':
                        if (length >= 8) switch (str[8]) {
                            case 'p':
                                if (strcmp(str, "allow_copy") == 0) {
                                    flag = NPY_ITER_ALLOW_COPY;
                                }
                                break;
                            case 'd':
                                if (strcmp(str, "allow_updateifcopy") == 0) {
                                    flag = NPY_ITER_ALLOW_UPDATEIFCOPY;
                                }
                                break;
                            case 'f':
                                if (strcmp(str, "allow_safe_casts") == 0) {
                                    flag = NPY_ITER_ALLOW_SAFE_CASTS;
                                }
                                break;
                            case 'm':
                                if (strcmp(str,
                                               "allow_same_kind_casts") == 0) {
                                    flag = NPY_ITER_ALLOW_SAME_KIND_CASTS;
                                }
                                break;
                            case 's':
                                if (strcmp(str, "allow_unsafe_casts") == 0) {
                                    flag = NPY_ITER_ALLOW_UNSAFE_CASTS;
                                }
                                break;
                            case 'i':
                                if (strcmp(str,
                                          "allow_writeable_references") == 0) {
                                    flag = NPY_ITER_ALLOW_WRITEABLE_REFERENCES;
                                }
                                break;
                            case 0: /* zero terminator, size==8 */
                                if (strcmp(str, "allocate") == 0) {
                                    flag = NPY_ITER_ALLOCATE;
                                }
                                break;
                        }
                        break;
                }
                if (flag == 0) {
                    PyErr_Format(PyExc_ValueError,
                            "Unexpected flag \"%s\"", str);
                    Py_DECREF(f2);
                    Py_DECREF(f);
                    goto fail;
                }
                else {
                    op_flags[iiter] |= flag;
                }
                Py_DECREF(f2);
            }
            Py_DECREF(f);
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Must provide a tuple of flag-tuples in op_flags");
        goto fail;
    }
    /* op_request_dtypes */
    if (op_dtypes_in != NULL && op_dtypes_in != Py_None) {
        if (!PyTuple_Check(op_dtypes_in) && !PyList_Check(op_dtypes_in)) {
            PyErr_SetString(PyExc_ValueError,
                    "Must provide a tuple in op_dtypes");
            goto fail;
        }
        if (PySequence_Size(op_dtypes_in) != niter) {
            PyErr_SetString(PyExc_ValueError,
                   "op_dtypes must be a tuple/list matching the number of ops");
            goto fail;
        }
        for (iiter = 0; iiter < niter; ++iiter) {
            PyObject *dtype = PySequence_GetItem(op_dtypes_in, iiter);
            /* Make sure the entry is an array descr */
            if (PyObject_TypeCheck(dtype, &PyArrayDescr_Type)) {
                op_request_dtypes[iiter] = (PyArray_Descr *)dtype;
            }
            else if (dtype == Py_None) {
                Py_DECREF(dtype);
                op_request_dtypes[iiter] = NULL;
            }
            else {
                Py_DECREF(dtype);
                PyErr_SetString(PyExc_ValueError,
                        "Iterator requested dtypes must be array descrs.");
                goto fail;
            }
        }
    }
    /* op_axes */
    if (op_axes_in != NULL && op_axes_in != Py_None) {
        PyObject *a;
        if (!PyTuple_Check(op_axes_in) && !PyList_Check(op_axes_in)) {
            PyErr_SetString(PyExc_ValueError,
                    "Must provide a tuple in op_axes");
            goto fail;
        }
        if (PySequence_Size(op_axes_in) != niter) {
            PyErr_SetString(PyExc_ValueError,
                    "op_axes must be a tuple/list matching the number of ops");
            goto fail;
        }
        /* Copy the tuples into op_axes */
        for (iiter = 0; iiter < niter; ++iiter) {
            npy_intp idim;
            a = PySequence_GetItem(op_axes_in, iiter);
            if (a == NULL) {
                goto fail;
            }
            if (a == Py_None) {
                op_axes[iiter] = NULL;
            } else {
                if (!PyTuple_Check(a) && !PyList_Check(a)) {
                    PyErr_SetString(PyExc_ValueError,
                            "Each entry of op_axes must be None "
                            "or a tuple/list");
                    Py_DECREF(a);
                    goto fail;
                }
                if (oa_ndim == 0) {
                    oa_ndim = PySequence_Size(a);
                    if (oa_ndim == 0) {
                        PyErr_SetString(PyExc_ValueError,
                                "Must have at least one  dimension "
                                "in op_axes");
                        goto fail;
                    }
                    if (oa_ndim > NPY_MAXDIMS) {
                        PyErr_SetString(PyExc_ValueError,
                                "Too many dimensions in op_axes");
                        goto fail;
                    }
                }
                if (PySequence_Size(a) != oa_ndim) {
                    PyErr_SetString(PyExc_ValueError,
                            "Each entry of op_axes must have the same size");
                    Py_DECREF(a);
                    goto fail;
                }
                for (idim = 0; idim < oa_ndim; ++idim) {
                    PyObject *v = PySequence_GetItem(a, idim);
                    if (v == NULL) {
                        Py_DECREF(a);
                        goto fail;
                    }
                    /* numpy.newaxis is None */
                    if (v == Py_None) {
                        op_axes_arrays[iiter][idim] = -1;
                    }
                    else {
                        op_axes_arrays[iiter][idim] = PyInt_AsLong(v);
                        if (op_axes_arrays[iiter][idim]==-1 &&
                                                    PyErr_Occurred()) {
                            Py_DECREF(a);
                            Py_DECREF(v);
                            goto fail;
                        }
                    }
                    Py_DECREF(v);
                }
                Py_DECREF(a);
                op_axes[iiter] = op_axes_arrays[iiter];
            }
        }

        if (oa_ndim == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "If op_axes is provided, at least one list of axes "
                    "must be within it");
            goto fail;
        }
    }

    self->iter = NpyIter_MultiNew(niter, op, flags, op_flags,
                                  (PyArray_Descr**)op_request_dtypes,
                                  min_ndim, max_ndim, oa_ndim,
                                  oa_ndim > 0 ? op_axes : NULL);

    if (self->iter == NULL) {
        goto fail;
    }

    /* Cache some values for the member functions to use */
    self->iternext = NpyIter_GetIterNext(self->iter);
    if (NpyIter_HasCoords(self->iter)) {
        self->getcoords = NpyIter_GetGetCoords(self->iter);
    }
    else {
        self->getcoords = NULL;
    }

    self->dataptrs = NpyIter_GetDataPtrArray(self->iter);
    self->dtypes = NpyIter_GetDescrArray(self->iter);
    self->objects = NpyIter_GetObjectArray(self->iter);

    if (NpyIter_HasInnerLoop(self->iter)) {
        self->innerstrides = NULL;
        self->innerloopsizeptr = NULL;
    }
    else {
        self->innerstrides = NpyIter_GetInnerStrideArray(self->iter);
        self->innerloopsizeptr = NpyIter_GetInnerLoopSizePtr(self->iter);
    }

    /* Get the read/write settings */
    NpyIter_GetReadFlags(self->iter, self->readflags);
    NpyIter_GetWriteFlags(self->iter, self->writeflags);

    return 0;

fail:
    for (iiter = 0; iiter < niter; ++iiter) {
        Py_XDECREF(op[iiter]);
        Py_XDECREF(op_request_dtypes[iiter]);
    }
    return -1;
}

static void
npyiter_dealloc(NewNpyArrayIterObject *self)
{
    if (self->iter) {
        NpyIter_Deallocate(self->iter);
        self->iter = NULL;
    }
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
npyiter_reset(NewNpyArrayIterObject *self)
{
    if (self->iter) {
        NpyIter_Reset(self->iter);
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

static PyObject *
npyiter_iternext(NewNpyArrayIterObject *self)
{
    if (self->iter != NULL && !self->finished && self->iternext(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        self->finished = 1;
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_debug_print(NewNpyArrayIterObject *self)
{
    if (self->iter != NULL) {
        NpyIter_DebugPrint(self->iter);
    }
    else {
        printf("Iterator: (nil)\n");
    }

    Py_RETURN_NONE;
}

static PyObject *npyiter_value_get(NewNpyArrayIterObject *self)
{
    PyObject *ret;

    npy_intp iiter, niter;
    PyArray_Descr **dtypes;
    char **dataptrs;

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    niter = NpyIter_GetNIter(self->iter);
    dtypes = self->dtypes;
    dataptrs = self->dataptrs;

    if (NpyIter_HasOffsets(self->iter)) {
        npy_intp *offsets = (npy_intp *)dataptrs;
        /* Return an array with all the offsets */
        ret = PyTuple_New(niter);
        if (ret == NULL) {
            return NULL;
        }
        for (iiter = 0; iiter < niter; ++iiter) {
            PyTuple_SET_ITEM(ret, iiter, PyInt_FromLong(offsets[iiter]));
        }
    } else {
        /* Return a scalar or tuple of scalars with the values */
        if (niter == 1) {
            if (self->readflags[0]) {
                PyArray_Descr *dtype = dtypes[0];

                if (NpyIter_HasInnerLoop(self->iter)) {
                    ret = PyArray_Scalar(dataptrs[0], dtype, NULL);
                }
                else {
                    Py_INCREF(dtype);
                    ret = (PyObject *)PyArray_NewFromDescr(&PyArray_Type,
                                dtype, 1,
                                self->innerloopsizeptr,
                                &self->innerstrides[0],
                                dataptrs[0],
                                self->writeflags[0] ? NPY_WRITEABLE : 0,
                                NULL);
                    ((PyArrayObject *)ret)->base = (PyObject *)self;
                    Py_INCREF(self);
                    PyArray_UpdateFlags((PyArrayObject *)ret, NPY_UPDATE_ALL);
                }
            }
            else {
                Py_INCREF(Py_None);
                ret = Py_None;
            }
        }
        else {
            ret = PyTuple_New(niter);
            if (ret == NULL) {
                return NULL;
            }
            for (iiter = 0; iiter < niter; ++iiter) {
                if (self->readflags[iiter]) {
                    PyArray_Descr *dtype = dtypes[iiter];
                    PyObject *item;

                    if (NpyIter_HasInnerLoop(self->iter)) {
                        item = PyArray_Scalar(dataptrs[iiter], dtype, NULL);
                    }
                    else {
                        Py_INCREF(dtype);
                        item = (PyObject *)PyArray_NewFromDescr(&PyArray_Type,
                                dtype, 1,
                                self->innerloopsizeptr,
                                &self->innerstrides[iiter],
                                dataptrs[iiter],
                                self->writeflags[iiter] ? NPY_WRITEABLE : 0,
                                NULL);
                        ((PyArrayObject *)item)->base = (PyObject *)self;
                        Py_INCREF(self);
                        PyArray_UpdateFlags((PyArrayObject *)item,
                                                            NPY_UPDATE_ALL);
                    }
                    PyTuple_SET_ITEM(ret, iiter, item);
                        
                }
                else {
                    Py_INCREF(Py_None);
                    PyTuple_SET_ITEM(ret, iiter, Py_None);
                }
            }
        }
    }

    return ret;
}

static PyObject *npyiter_operands_get(NewNpyArrayIterObject *self)
{
    PyObject *ret;

    npy_intp iiter, niter;
    PyObject **objects;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    niter = NpyIter_GetNIter(self->iter);
    objects = self->objects;

    ret = PyTuple_New(niter);
    if (ret == NULL) {
        return NULL;
    }
    for (iiter = 0; iiter < niter; ++iiter) {
        PyObject *object = objects[iiter];

        Py_INCREF(object);
        PyTuple_SET_ITEM(ret, iiter, object);
    }

    return ret;
}

static PyObject *
npyiter_next(NewNpyArrayIterObject *self)
{
    PyObject *ret;

    if (self->iter == NULL || self->finished) {
        return NULL;
    }

    ret = npyiter_value_get(self);
    if (ret == NULL) {
        return NULL;
    }

    if (!self->iternext(self->iter)) {
        self->finished = 1;
    }
    
    return ret;
};

static PyObject *npyiter_shape_get(NewNpyArrayIterObject *self)
{
    PyObject *ret;
    npy_intp idim, ndim, shape[NPY_MAXDIMS];

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_GetShape(self->iter, shape) == NPY_SUCCEED) {
        ndim = NpyIter_GetNDim(self->iter);
        ret = PyTuple_New(ndim);
        if (ret != NULL) {
            for (idim = 0; idim < ndim; ++idim) {
                PyTuple_SET_ITEM(ret, idim,
                        PyInt_FromLong(shape[idim]));
            }
            return ret;
        }
    }

    return NULL;
}

static PyObject *npyiter_coords_get(NewNpyArrayIterObject *self)
{
    PyObject *ret;
    npy_intp idim, ndim, coords[NPY_MAXDIMS];

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasCoords(self->iter)) {
        ndim = NpyIter_GetNDim(self->iter);
        self->getcoords(self->iter, coords);
        ret = PyTuple_New(ndim);
        for (idim = 0; idim < ndim; ++idim) {
            PyTuple_SET_ITEM(ret, idim,
                    PyInt_FromLong(coords[idim]));
        }
        return ret;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have coordinates");
        return NULL;
    }
}

static int npyiter_coords_set(NewNpyArrayIterObject *self, PyObject *value)
{
    npy_intp idim, ndim, coords[NPY_MAXDIMS];

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot delete coordinates");
        return -1;
    }

    if (NpyIter_HasCoords(self->iter)) {
        ndim = NpyIter_GetNDim(self->iter);
        if (!PySequence_Check(value)) {
            PyErr_SetString(PyExc_ValueError,
                    "Coordinates must be set with a sequence");
            return -1;
        }
        if (PySequence_Size(value) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                    "Wrong number of coordinates");
            return -1;
        }
        for (idim = 0; idim < ndim; ++idim) {
            PyObject *v = PySequence_GetItem(value, idim);
            coords[idim] = PyInt_AsLong(v);
            if (coords[idim]==-1 && PyErr_Occurred()) {
                return -1;
            }
        }
        if (NpyIter_GotoCoords(self->iter, coords) != NPY_SUCCEED) {
            return -1;
        }
        self->finished = 0;
        
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have coordinates");
        return -1;
    }
}

static PyObject *npyiter_index_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp index = *NpyIter_GetIndexPtr(self->iter);
        return PyInt_FromLong(index);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return NULL;
    }
}

static int npyiter_index_set(NewNpyArrayIterObject *self, PyObject *value)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot delete index");
        return -1;
    }

    if (NpyIter_HasIndex(self->iter)) {
        npy_intp index;
        index = PyInt_AsLong(value);
        if (index==-1 && PyErr_Occurred()) {
            return -1;
        }
        if (NpyIter_GotoIndex(self->iter, index) != NPY_SUCCEED) {
            return -1;
        }
        self->finished = 0;
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return -1;
    }
}

static PyObject *npyiter_hascoords_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    if (NpyIter_HasCoords(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *npyiter_hasindex_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    if (NpyIter_HasIndex(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *npyiter_dtypes_get(NewNpyArrayIterObject *self)
{
    PyObject *ret;

    npy_intp iiter, niter;
    PyArray_Descr **dtypes;

    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    niter = NpyIter_GetNIter(self->iter);

    ret = PyTuple_New(niter);
    if (ret == NULL) {
        return NULL;
    }
    dtypes = self->dtypes;
    for (iiter = 0; iiter < niter; ++iiter) {
        PyArray_Descr *dtype = dtypes[iiter];

        Py_INCREF(dtype);
        PyTuple_SET_ITEM(ret, iiter, (PyObject *)dtype);
    }

    return ret;
}

static PyObject *npyiter_ndim_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    return PyInt_FromLong(NpyIter_GetNDim(self->iter));
}

static PyObject *npyiter_niter_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    return PyInt_FromLong(NpyIter_GetNIter(self->iter));
}

static PyObject *npyiter_itersize_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator was not constructed correctly");
        return NULL;
    }

    return PyInt_FromLong(NpyIter_GetIterSize(self->iter));
}

static PyObject *npyiter_finished_get(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL || !self->finished) {
        Py_RETURN_FALSE;
    }
    else {
        Py_RETURN_TRUE;
    }
}

NPY_NO_EXPORT Py_ssize_t
npyiter_seq_length(NewNpyArrayIterObject *self)
{
    if (self->iter == NULL) {
        return 0;
    }
    else {
        return NpyIter_GetNIter(self->iter);
    }
}

NPY_NO_EXPORT PyObject *
npyiter_seq_item(NewNpyArrayIterObject *self, Py_ssize_t i)
{
    PyObject *ret;

    npy_intp niter;
    char *dataptr;
    PyArray_Descr *dtype;

    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }
    niter = NpyIter_GetNIter(self->iter);
    /* Python negative indexing */
    if (i < 0) {
        i += niter;
    }
    if (i < 0 || i >= niter) {
        PyErr_Format(PyExc_IndexError,
                "Iterator operand index %d is out of bounds", (int)i);
        return NULL;
    }

#if 0
    /*
     * This check is disabled because it prevents things like
     * np.add(it[0], it[1], it[2]), where it[2] is a write-only
     * parameter.  When write-only, the value of it[i] is
     * likely random junk, as if it were allocated with an
     * np.empty(...) call.
     */
    if (!self->readflags[i]) {
        PyErr_Format(PyExc_RuntimeError,
                "Iterator operand %d is write-only", (int)i);
        return NULL;
    }
#endif

    dataptr = self->dataptrs[i];
    dtype = self->dtypes[i];

    if (NpyIter_HasInnerLoop(self->iter)) {
        ret = PyArray_Scalar(dataptr, dtype, NULL);
    }
    else {
        Py_INCREF(dtype);
        ret = (PyObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                                1, self->innerloopsizeptr,
                                &self->innerstrides[i], dataptr,
                                self->writeflags[i] ? NPY_WRITEABLE : 0, NULL);
        ((PyArrayObject *)ret)->base = (PyObject *)self;
        Py_INCREF(self);
        PyArray_UpdateFlags((PyArrayObject *)ret, NPY_UPDATE_ALL);
    }
    return ret;
}

NPY_NO_EXPORT int
npyiter_seq_ass_item(NewNpyArrayIterObject *self, Py_ssize_t i, PyObject *v)
{

    npy_intp niter;
    char *dataptr;
    PyArray_Descr *dtype;
    PyObject *object;

    if (v == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "can't delete iterator operands");
        return -1;
    }
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return -1;
    }
    niter = NpyIter_GetNIter(self->iter);
    /* Python negative indexing */
    if (i < 0) {
        i += niter;
    }
    if (i < 0 || i >= niter) {
        PyErr_Format(PyExc_IndexError,
                "Iterator operand index  %d is out of bounds", (int)i);
        return -1;
    }
    if (!self->writeflags[i]) {
        PyErr_Format(PyExc_RuntimeError,
                "Iterator operand %d is not writeable", (int)i);
        return -1;
    }

    dataptr = self->dataptrs[i];
    dtype = self->dtypes[i];
    object = self->objects[i];

    /*
     * TODO: When buffering is enabled for an operand, the object won't
     *       correspond to the data, so that will have to be accounted for
     */
    if (NpyIter_HasInnerLoop(self->iter)) {
        if (PyArray_Check(object)) {
            return dtype->f->setitem(v, dataptr, (PyArrayObject*)object);
        }
        else {
            return dtype->f->setitem(v, dataptr, NULL);
        }
    } else {
        PyArrayObject *tmp;
        int ret;
        Py_INCREF(dtype);
        tmp = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                                    1, self->innerloopsizeptr,
                                    &self->innerstrides[i], dataptr,
                                    NPY_WRITEABLE, NULL);
        if (tmp == NULL) {
            return -1;
        }
        PyArray_UpdateFlags(tmp, NPY_UPDATE_ALL);
        ret = PyArray_CopyObject(tmp, v);
        Py_DECREF(tmp);
        return ret;
    }
}

static PyMethodDef npyiter_methods[] = {
    {"reset", (PyCFunction)npyiter_reset, METH_NOARGS, NULL},
    {"iternext", (PyCFunction)npyiter_iternext, METH_NOARGS, NULL},
    {"debug_print", (PyCFunction)npyiter_debug_print, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef npyiter_members[] = {
    {NULL, 0, 0, 0, NULL},
};

static PyGetSetDef npyiter_getsets[] = {
    {"value",
        (getter)npyiter_value_get,
        NULL, NULL, NULL},
    {"operands",
        (getter)npyiter_operands_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)npyiter_shape_get,
        NULL, NULL, NULL},
    {"coords",
        (getter)npyiter_coords_get,
        (setter)npyiter_coords_set,
        NULL, NULL},
    {"index",
        (getter)npyiter_index_get,
        (setter)npyiter_index_set,
        NULL, NULL},
    {"hascoords",
        (getter)npyiter_hascoords_get,
        NULL, NULL, NULL},
    {"hasindex",
        (getter)npyiter_hasindex_get,
        NULL, NULL, NULL},
    {"dtypes",
        (getter)npyiter_dtypes_get,
        NULL, NULL, NULL},
    {"ndim",
        (getter)npyiter_ndim_get,
        NULL, NULL, NULL},
    {"niter",
        (getter)npyiter_niter_get,
        NULL, NULL, NULL},
    {"itersize",
        (getter)npyiter_itersize_get,
        NULL, NULL, NULL},
    {"finished",
        (getter)npyiter_finished_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL},
};

NPY_NO_EXPORT PySequenceMethods npyiter_as_sequence = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)npyiter_seq_length,            /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat*/
    (ssizeargfunc)NULL,                     /*sq_repeat*/
    (ssizeargfunc)npyiter_seq_item,         /*sq_item*/
    (ssizessizeargfunc)NULL,                /*sq_slice*/
    (ssizeobjargproc)npyiter_seq_ass_item,  /*sq_ass_item*/
    (ssizessizeobjargproc)NULL,             /*sq_ass_slice*/
    (objobjproc)NULL,                       /*sq_contains */
    (binaryfunc)NULL,                       /*sq_inplace_concat */
    (ssizeargfunc)NULL,                     /*sq_inplace_repeat */
#else
    (inquiry)npyiter_seq_length,            /*sq_length*/
    (binaryfunc)NULL,                       /*sq_concat is handled by nb_add*/
    (intargfunc)NULL,                       /*sq_repeat is handled nb_multiply*/
    (intargfunc)npyiter_seq_item,           /*sq_item*/
    (intintargfunc)NULL,                    /*sq_slice*/
    (intobjargproc)npyiter_seq_ass_item,    /*sq_ass_item*/
    (intintobjargproc)NULL,                 /*sq_ass_slice*/
    (objobjproc)NULL,                       /*sq_contains */
    (binaryfunc)NULL,                       /*sg_inplace_concat */
    (intargfunc)NULL                        /*sg_inplace_repeat */
#endif
};

NPY_NO_EXPORT PyTypeObject NpyIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.newiter",                            /* tp_name */
    sizeof(NewNpyArrayIterObject),               /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)npyiter_dealloc,                /* tp_dealloc */
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
    &npyiter_as_sequence,                       /* tp_as_sequence */
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
    (iternextfunc)npyiter_next,                 /* tp_iternext */
    npyiter_methods,                            /* tp_methods */
    npyiter_members,                            /* tp_members */
    npyiter_getsets,                            /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)npyiter_init,                     /* tp_init */
    0,                                          /* tp_alloc */
    npyiter_new,                                /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

