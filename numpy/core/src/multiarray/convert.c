#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "arrayobject.h"
#include "mapping.h"

#include "convert.h"

/*NUMPY_API
 * To List
 */
NPY_NO_EXPORT PyObject *
PyArray_ToList(PyArrayObject *self)
{
    PyObject *lp;
    PyArrayObject *v;
    intp sz, i;

    if (!PyArray_Check(self)) {
        return (PyObject *)self;
    }
    if (self->nd == 0) {
        return self->descr->f->getitem(self->data,self);
    }

    sz = self->dimensions[0];
    lp = PyList_New(sz);
    for (i = 0; i < sz; i++) {
        v = (PyArrayObject *)array_big_item(self, i);
        if (PyArray_Check(v) && (v->nd >= self->nd)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "array_item not returning smaller-" \
                            "dimensional array");
            Py_DECREF(v);
            Py_DECREF(lp);
            return NULL;
        }
        PyList_SetItem(lp, i, PyArray_ToList(v));
        Py_DECREF(v);
    }
    return lp;
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
    intp size;
    intp n, n2;
    size_t n3, n4;
    PyArrayIterObject *it;
    PyObject *obj, *strobj, *tupobj;

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
            n = fwrite((const void *)self->data,
                    (size_t) self->descr->elsize,
                    (size_t) size, fp);
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
                obj = PyString_FromString((const char *)format);
                if (obj == NULL) {
                    Py_DECREF(tupobj);
                    Py_DECREF(it);
                    return -1;
                }
                strobj = PyString_Format(obj, tupobj);
                Py_DECREF(obj);
                Py_DECREF(tupobj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
            NPY_BEGIN_ALLOW_THREADS;
            n2 = PyString_GET_SIZE(strobj);
            n = fwrite(PyString_AS_STRING(strobj), 1, n2, fp);
            NPY_END_ALLOW_THREADS;
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
        ret = PyString_FromStringAndSize(self->data, (Py_ssize_t) numbytes);
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
        ret = PyString_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            Py_DECREF(it);
            return NULL;
        }
        dptr = PyString_AS_STRING(ret);
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

/*NUMPY_API
  Copy an array.
*/
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *m1, NPY_ORDER fortran)
{
    PyArrayObject *ret;
    if (fortran == PyArray_ANYORDER)
        fortran = PyArray_ISFORTRAN(m1);

    Py_INCREF(m1->descr);
    ret = (PyArrayObject *)PyArray_NewFromDescr(m1->ob_type,
                                                m1->descr,
                                                m1->nd,
                                                m1->dimensions,
                                                NULL, NULL,
                                                fortran,
                                                (PyObject *)m1);
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
