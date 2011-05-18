#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "arrayobject.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"

#include "convert.h"

/*
 * Converts a subarray of 'self' into lists, with starting data pointer
 * 'dataptr' and from dimension 'startdim' to the last dimension of 'self'.
 *
 * Returns a new reference.
 */
static PyObject *
recursive_tolist(PyArrayObject *self, char *dataptr, int startdim)
{
    npy_intp i, n, stride;
    PyObject *ret, *item;

    /* Base case */
    if (startdim >= PyArray_NDIM(self)) {
        return PyArray_DESCR(self)->f->getitem(dataptr,self);
    }

    n = PyArray_DIM(self, startdim);
    stride = PyArray_STRIDE(self, startdim);

    ret = PyList_New(n);
    if (ret == NULL) {
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        item = recursive_tolist(self, dataptr, startdim+1);
        if (item == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyList_SET_ITEM(ret, i, item);

        dataptr += stride;
    }

    return ret;
}

/*NUMPY_API
 * To List
 */
NPY_NO_EXPORT PyObject *
PyArray_ToList(PyArrayObject *self)
{
    return recursive_tolist(self, PyArray_DATA(self), 0);
}

/* XXX: FIXME --- add ordering argument to
   Allow Fortran ordering on write
   This will need the addition of a Fortran-order iterator.
 */

/*NUMPY_API
  To File
*/
NPY_NO_EXPORT int
PyArray_ToFile(PyArrayObject *self, FILE *fp, char *sep, char *format)
{
    npy_intp size;
    npy_intp n, n2;
    size_t n3, n4;
    PyArrayIterObject *it;
    PyObject *obj, *strobj, *tupobj, *byteobj;

    n3 = (sep ? strlen((const char *)sep) : 0);
    if (n3 == 0) {
        /* binary data */
        if (PyDataType_FLAGCHK(self->descr, NPY_LIST_PICKLE)) {
            PyErr_SetString(PyExc_ValueError, "cannot write " \
                    "object arrays to a file in "   \
                    "binary mode");
            return -1;
        }

        if (PyArray_ISCONTIGUOUS(self)) {
            size = PyArray_SIZE(self);
            NPY_BEGIN_ALLOW_THREADS;

#if defined (_MSC_VER) && defined(_WIN64)
            /* Workaround Win64 fwrite() bug. Ticket #1660 */
            {
                npy_intp maxsize = 2147483648 / self->descr->elsize;
                npy_intp chunksize;

                n = 0;
                while (size > 0) {
                    chunksize = (size > maxsize) ? maxsize : size;
                    n2 = fwrite((const void *)
                             ((char *)self->data + (n * self->descr->elsize)),
                             (size_t) self->descr->elsize,
                             (size_t) chunksize, fp);
                    if (n2 < chunksize) {
                        break;
                    }
                    n += n2;
                    size -= chunksize;
                }
                size = PyArray_SIZE(self);
            }
#else
            n = fwrite((const void *)self->data,
                    (size_t) self->descr->elsize,
                    (size_t) size, fp);
#endif
            NPY_END_ALLOW_THREADS;
            if (n < size) {
                PyErr_Format(PyExc_ValueError,
                        "%ld requested and %ld written",
                        (long) size, (long) n);
                return -1;
            }
        }
        else {
            NPY_BEGIN_THREADS_DEF;

            it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)self);
            NPY_BEGIN_THREADS;
            while (it->index < it->size) {
                if (fwrite((const void *)it->dataptr,
                            (size_t) self->descr->elsize,
                            1, fp) < 1) {
                    NPY_END_THREADS;
                    PyErr_Format(PyExc_IOError,
                            "problem writing element"\
                            " %"INTP_FMT" to file",
                            it->index);
                    Py_DECREF(it);
                    return -1;
                }
                PyArray_ITER_NEXT(it);
            }
            NPY_END_THREADS;
            Py_DECREF(it);
        }
    }
    else {
        /*
         * text data
         */

        it = (PyArrayIterObject *)
            PyArray_IterNew((PyObject *)self);
        n4 = (format ? strlen((const char *)format) : 0);
        while (it->index < it->size) {
            obj = self->descr->f->getitem(it->dataptr, self);
            if (obj == NULL) {
                Py_DECREF(it);
                return -1;
            }
            if (n4 == 0) {
                /*
                 * standard writing
                 */
                strobj = PyObject_Str(obj);
                Py_DECREF(obj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
            else {
                /*
                 * use format string
                 */
                tupobj = PyTuple_New(1);
                if (tupobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
                PyTuple_SET_ITEM(tupobj,0,obj);
                obj = PyUString_FromString((const char *)format);
                if (obj == NULL) {
                    Py_DECREF(tupobj);
                    Py_DECREF(it);
                    return -1;
                }
                strobj = PyUString_Format(obj, tupobj);
                Py_DECREF(obj);
                Py_DECREF(tupobj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
#if defined(NPY_PY3K)
            byteobj = PyUnicode_AsASCIIString(strobj);
#else
            byteobj = strobj;
#endif
            NPY_BEGIN_ALLOW_THREADS;
            n2 = PyBytes_GET_SIZE(byteobj);
            n = fwrite(PyBytes_AS_STRING(byteobj), 1, n2, fp);
            NPY_END_ALLOW_THREADS;
#if defined(NPY_PY3K)
            Py_DECREF(byteobj);
#endif
            if (n < n2) {
                PyErr_Format(PyExc_IOError,
                        "problem writing element %"INTP_FMT\
                        " to file", it->index);
                Py_DECREF(strobj);
                Py_DECREF(it);
                return -1;
            }
            /* write separator for all but last one */
            if (it->index != it->size-1) {
                if (fwrite(sep, 1, n3, fp) < n3) {
                    PyErr_Format(PyExc_IOError,
                            "problem writing "\
                            "separator to file");
                    Py_DECREF(strobj);
                    Py_DECREF(it);
                    return -1;
                }
            }
            Py_DECREF(strobj);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_ToString(PyArrayObject *self, NPY_ORDER order)
{
    intp numbytes;
    intp index;
    char *dptr;
    int elsize;
    PyObject *ret;
    PyArrayIterObject *it;

    if (order == NPY_ANYORDER)
        order = PyArray_ISFORTRAN(self);

    /*        if (PyArray_TYPE(self) == PyArray_OBJECT) {
              PyErr_SetString(PyExc_ValueError, "a string for the data" \
              "in an object array is not appropriate");
              return NULL;
              }
    */

    numbytes = PyArray_NBYTES(self);
    if ((PyArray_ISCONTIGUOUS(self) && (order == NPY_CORDER))
        || (PyArray_ISFORTRAN(self) && (order == NPY_FORTRANORDER))) {
        ret = PyBytes_FromStringAndSize(self->data, (Py_ssize_t) numbytes);
    }
    else {
        PyObject *new;
        if (order == NPY_FORTRANORDER) {
            /* iterators are always in C-order */
            new = PyArray_Transpose(self, NULL);
            if (new == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(self);
            new = (PyObject *)self;
        }
        it = (PyArrayIterObject *)PyArray_IterNew(new);
        Py_DECREF(new);
        if (it == NULL) {
            return NULL;
        }
        ret = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            Py_DECREF(it);
            return NULL;
        }
        dptr = PyBytes_AS_STRING(ret);
        index = it->size;
        elsize = self->descr->elsize;
        while (index--) {
            memcpy(dptr, it->dataptr, elsize);
            dptr += elsize;
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_FillWithScalar(PyArrayObject *arr, PyObject *obj)
{
    PyObject *newarr;
    int itemsize, swap;
    void *fromptr;
    PyArray_Descr *descr;
    intp size;
    PyArray_CopySwapFunc *copyswap;

    itemsize = arr->descr->elsize;
    if (PyArray_ISOBJECT(arr)) {
        fromptr = &obj;
        swap = 0;
        newarr = NULL;
    }
    else {
        descr = PyArray_DESCR(arr);
        Py_INCREF(descr);
        newarr = PyArray_FromAny(obj, descr, 0,0, ALIGNED, NULL);
        if (newarr == NULL) {
            return -1;
        }
        fromptr = PyArray_DATA(newarr);
        swap = (PyArray_ISNOTSWAPPED(arr) != PyArray_ISNOTSWAPPED(newarr));
    }
    size=PyArray_SIZE(arr);
    copyswap = arr->descr->f->copyswap;
    if (PyArray_ISONESEGMENT(arr)) {
        char *toptr=PyArray_DATA(arr);
        PyArray_FillWithScalarFunc* fillwithscalar =
            arr->descr->f->fillwithscalar;
        if (fillwithscalar && PyArray_ISALIGNED(arr)) {
            copyswap(fromptr, NULL, swap, newarr);
            fillwithscalar(toptr, size, fromptr, arr);
        }
        else {
            while (size--) {
                copyswap(toptr, fromptr, swap, arr);
                toptr += itemsize;
            }
        }
    }
    else {
        PyArrayIterObject *iter;

        iter = (PyArrayIterObject *)\
            PyArray_IterNew((PyObject *)arr);
        if (iter == NULL) {
            Py_XDECREF(newarr);
            return -1;
        }
        while (size--) {
            copyswap(iter->dataptr, fromptr, swap, arr);
            PyArray_ITER_NEXT(iter);
        }
        Py_DECREF(iter);
    }
    Py_XDECREF(newarr);
    return 0;
}

/*
 * Fills an array with zeros.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_FillWithZero(PyArrayObject *a)
{
    PyArray_StridedTransferFn *stransfer = NULL;
    void *transferdata = NULL;
    PyArray_Descr *dtype = PyArray_DESCR(a);
    NpyIter *iter;

    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp stride, *countptr;
    int needs_api;

    NPY_BEGIN_THREADS_DEF;

    if (!PyArray_ISWRITEABLE(a)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot write to array");
        return -1;
    }

    /* A zero-sized array needs no zeroing */
    if (PyArray_SIZE(a) == 0) {
        return 0;
    }

    /* If it's possible to do a simple memset, do so */
    if (!PyDataType_REFCHK(dtype) && (PyArray_ISCONTIGUOUS(a) ||
                                      PyArray_ISFORTRAN(a))) {
        memset(PyArray_DATA(a), 0, PyArray_NBYTES(a));
        return 0;
    }

    /* Use an iterator to go through all the data */
    iter = NpyIter_New(a, NPY_ITER_WRITEONLY|NPY_ITER_EXTERNAL_LOOP,
                    NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    if (iter == NULL) {
        return -1;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    stride = NpyIter_GetInnerStrideArray(iter)[0];
    countptr = NpyIter_GetInnerLoopSizePtr(iter);

    needs_api = NpyIter_IterationNeedsAPI(iter);

    /*
     * Because buffering is disabled in the iterator, the inner loop
     * strides will be the same throughout the iteration loop.  Thus,
     * we can pass them to this function to take advantage of
     * contiguous strides, etc.
     *
     * By setting the src_dtype to NULL, we get a function which sets
     * the destination to zeros.
     */
    if (PyArray_GetDTypeTransferFunction(
                    PyArray_ISALIGNED(a),
                    0, stride,
                    NULL, PyArray_DESCR(a),
                    0,
                    &stransfer, &transferdata,
                    &needs_api) != NPY_SUCCEED) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    do {
        stransfer(*dataptr, stride, NULL, 0,
                    *countptr, 0, transferdata);
    } while(iternext(iter));

    if (!needs_api) {
        NPY_END_THREADS;
    }

    PyArray_FreeStridedTransferData(transferdata);
    NpyIter_Deallocate(iter);

    return 0;
}

/*NUMPY_API
 * Copy an array.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *m1, NPY_ORDER order)
{
    PyArrayObject *ret = (PyArrayObject *)PyArray_NewLikeArray(
                                                    m1, order, NULL, 1);
    if (ret == NULL) {
        return NULL;
    }

    if (PyArray_CopyInto(ret, m1) == -1) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject *)ret;
}

/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    PyObject *new = NULL;
    PyTypeObject *subtype;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
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
