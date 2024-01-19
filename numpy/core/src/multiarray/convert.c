#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "npy_config.h"

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "npy_pycompat.h"

#include "common.h"
#include "arrayobject.h"
#include "ctors.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"
#include "scalartypes.h"
#include "array_assign.h"

#include "convert.h"
#include "array_coercion.h"
#include "refcount.h"

#if defined(HAVE_FALLOCATE) && defined(__linux__)
#include <fcntl.h>
#endif

/*
 * allocate nbytes of diskspace for file fp
 * this allows the filesystem to make smarter allocation decisions and gives a
 * fast exit on not enough free space
 * returns -1 and raises exception on no space, ignores all other errors
 */
static int
npy_fallocate(npy_intp nbytes, FILE * fp)
{
    /*
     * unknown behavior on non-linux so don't try it
     * we don't want explicit zeroing to happen
     */
#if defined(HAVE_FALLOCATE) && defined(__linux__)
    int r;
    /* small files not worth the system call */
    if (nbytes < 16 * 1024 * 1024) {
        return 0;
    }

    /* btrfs can take a while to allocate making release worthwhile */
    NPY_BEGIN_ALLOW_THREADS;
    /*
     * flush in case there might be some unexpected interactions between the
     * fallocate call and unwritten data in the descriptor
     */
    fflush(fp);
    /*
     * the flag "1" (=FALLOC_FL_KEEP_SIZE) is needed for the case of files
     * opened in append mode (issue #8329)
     */
    r = fallocate(fileno(fp), 1, npy_ftell(fp), nbytes);
    NPY_END_ALLOW_THREADS;

    /*
     * early exit on no space, other errors will also get found during fwrite
     */
    if (r == -1 && errno == ENOSPC) {
        PyErr_Format(PyExc_OSError, "Not enough free space to write "
                     "%"NPY_INTP_FMT" bytes", nbytes);
        return -1;
    }
#endif
    return 0;
}

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
        return PyArray_GETITEM(self, dataptr);
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
        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_LIST_PICKLE)) {
            PyErr_SetString(PyExc_OSError,
                    "cannot write object arrays to a file in binary mode");
            return -1;
        }
        if (PyArray_DESCR(self)->elsize == 0) {
            /* For zero-width data types there's nothing to write */
            return 0;
        }
        if (npy_fallocate(PyArray_NBYTES(self), fp) != 0) {
            return -1;
        }

        if (PyArray_ISCONTIGUOUS(self)) {
            size = PyArray_SIZE(self);
            NPY_BEGIN_ALLOW_THREADS;

#if defined(_WIN64)
            /*
             * Workaround Win64 fwrite() bug. Issue gh-2256
             * The native 64 windows runtime has this issue, the above will
             * also trigger UCRT (which doesn't), so it could be more precise.
             *
             * If you touch this code, please run this test which is so slow
             * it was removed from the test suite. Note that the original
             * failure mode involves an infinite loop during tofile()
             *
             * import tempfile, numpy as np
             * from numpy.testing import (assert_)
             * fourgbplus = 2**32 + 2**16
             * testbytes = np.arange(8, dtype=np.int8)
             * n = len(testbytes)
             * flike = tempfile.NamedTemporaryFile()
             * f = flike.file
             * np.tile(testbytes, fourgbplus // testbytes.nbytes).tofile(f)
             * flike.seek(0)
             * a = np.fromfile(f, dtype=np.int8)
             * flike.close()
             * assert_(len(a) == fourgbplus)
             * # check only start and end for speed:
             * assert_((a[:n] == testbytes).all())
             * assert_((a[-n:] == testbytes).all())
             */
            {
                size_t maxsize = 2147483648 / (size_t)PyArray_DESCR(self)->elsize;
                size_t chunksize;

                n = 0;
                while (size > 0) {
                    chunksize = (size > maxsize) ? maxsize : size;
                    n2 = fwrite((const void *)
                             ((char *)PyArray_DATA(self) + (n * PyArray_DESCR(self)->elsize)),
                             (size_t) PyArray_DESCR(self)->elsize,
                             chunksize, fp);
                    if (n2 < chunksize) {
                        break;
                    }
                    n += n2;
                    size -= chunksize;
                }
                size = PyArray_SIZE(self);
            }
#else
            n = fwrite((const void *)PyArray_DATA(self),
                    (size_t) PyArray_DESCR(self)->elsize,
                    (size_t) size, fp);
#endif
            NPY_END_ALLOW_THREADS;
            if (n < size) {
                PyErr_Format(PyExc_OSError,
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
                            (size_t) PyArray_DESCR(self)->elsize,
                            1, fp) < 1) {
                    NPY_END_THREADS;
                    PyErr_Format(PyExc_OSError,
                            "problem writing element %" NPY_INTP_FMT
                            " to file", it->index);
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
            obj = PyArray_GETITEM(self, it->dataptr);
            if (obj == NULL) {
                Py_DECREF(it);
                return -1;
            }
            if (n4 == 0) {
                /*
                 * standard writing
                 */
                strobj = PyObject_Repr(obj);
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
                obj = PyUnicode_FromString((const char *)format);
                if (obj == NULL) {
                    Py_DECREF(tupobj);
                    Py_DECREF(it);
                    return -1;
                }
                strobj = PyUnicode_Format(obj, tupobj);
                Py_DECREF(obj);
                Py_DECREF(tupobj);
                if (strobj == NULL) {
                    Py_DECREF(it);
                    return -1;
                }
            }
            byteobj = PyUnicode_AsASCIIString(strobj);
            NPY_BEGIN_ALLOW_THREADS;
            n2 = PyBytes_GET_SIZE(byteobj);
            n = fwrite(PyBytes_AS_STRING(byteobj), 1, n2, fp);
            NPY_END_ALLOW_THREADS;
            Py_DECREF(byteobj);
            if (n < n2) {
                PyErr_Format(PyExc_OSError,
                        "problem writing element %" NPY_INTP_FMT
                        " to file", it->index);
                Py_DECREF(strobj);
                Py_DECREF(it);
                return -1;
            }
            /* write separator for all but last one */
            if (it->index != it->size-1) {
                if (fwrite(sep, 1, n3, fp) < n3) {
                    PyErr_Format(PyExc_OSError,
                            "problem writing separator to file");
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
    npy_intp numbytes;
    npy_intp i;
    char *dptr;
    int elsize;
    PyObject *ret;
    PyArrayIterObject *it;

    if (order == NPY_ANYORDER)
        order = PyArray_ISFORTRAN(self) ? NPY_FORTRANORDER : NPY_CORDER;

    /*        if (PyArray_TYPE(self) == NPY_OBJECT) {
              PyErr_SetString(PyExc_ValueError, "a string for the data" \
              "in an object array is not appropriate");
              return NULL;
              }
    */

    numbytes = PyArray_NBYTES(self);
    if ((PyArray_IS_C_CONTIGUOUS(self) && (order == NPY_CORDER))
        || (PyArray_IS_F_CONTIGUOUS(self) && (order == NPY_FORTRANORDER))) {
        ret = PyBytes_FromStringAndSize(PyArray_DATA(self), (Py_ssize_t) numbytes);
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
        i = it->size;
        elsize = PyArray_DESCR(self)->elsize;
        while (i--) {
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

    if (PyArray_FailUnlessWriteable(arr, "assignment destination") < 0) {
        return -1;
    }

    /*
     * If we knew that the output array has at least one element, we would
     * not actually need a helping buffer, we always null it, just in case.
     *
     * (The longlong here should help with alignment.)
     */
    npy_longlong value_buffer_stack[4] = {0};
    char *value_buffer_heap = NULL;
    char *value = (char *)value_buffer_stack;
    PyArray_Descr *descr = PyArray_DESCR(arr);

    if ((size_t)descr->elsize > sizeof(value_buffer_stack)) {
        /* We need a large temporary buffer... */
        value_buffer_heap = PyObject_Calloc(1, descr->elsize);
        if (value_buffer_heap == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        value = value_buffer_heap;
    }
    if (PyArray_Pack(descr, value, obj) < 0) {
        PyMem_FREE(value_buffer_heap);
        return -1;
    }

    /*
     * There is no cast anymore, the above already coerced using scalar
     * coercion rules
     */
    int retcode = raw_array_assign_scalar(
            PyArray_NDIM(arr), PyArray_DIMS(arr), descr,
            PyArray_BYTES(arr), PyArray_STRIDES(arr),
            descr, value);

    if (PyDataType_REFCHK(descr)) {
        PyArray_ClearBuffer(descr, value, 0, 1, 1);
    }
    PyMem_FREE(value_buffer_heap);
    return retcode;
}

/*
 * Internal function to fill an array with zeros.
 * Used in einsum and dot, which ensures the dtype is, in some sense, numerical
 * and not a str or struct
 *
 * dst: The destination array.
 * wheremask: If non-NULL, a boolean mask specifying where to set the values.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignZero(PyArrayObject *dst,
                   PyArrayObject *wheremask)
{
    int retcode = 0;
    if (PyArray_ISOBJECT(dst)) {
        PyObject * pZero = PyLong_FromLong(0);
        retcode = PyArray_AssignRawScalar(dst, PyArray_DESCR(dst),
                                     (char *)&pZero, wheremask, NPY_SAFE_CASTING);
        Py_DECREF(pZero);
    }
    else {
        /* Create a raw bool scalar with the value False */
        PyArray_Descr *bool_dtype = PyArray_DescrFromType(NPY_BOOL);
        if (bool_dtype == NULL) {
            return -1;
        }
        npy_bool value = 0;

        retcode = PyArray_AssignRawScalar(dst, bool_dtype, (char *)&value,
                                          wheremask, NPY_SAFE_CASTING);

        Py_DECREF(bool_dtype);
    }
    return retcode;
}


/*NUMPY_API
 * Copy an array.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *obj, NPY_ORDER order)
{
    PyArrayObject *ret;

    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "obj is NULL in PyArray_NewCopy");
        return NULL;
    }

    ret = (PyArrayObject *)PyArray_NewLikeArray(obj, order, NULL, 1);
    if (ret == NULL) {
        return NULL;
    }

    if (PyArray_AssignArray(ret, obj, NULL, NPY_UNSAFE_CASTING) < 0) {
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
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    PyTypeObject *subtype;
    int flags;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
    }

    dtype = PyArray_DESCR(self);
    flags = PyArray_FLAGS(self);

    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr_int(
            subtype, dtype,
            PyArray_NDIM(self), PyArray_DIMS(self), PyArray_STRIDES(self),
            PyArray_DATA(self),
            flags, (PyObject *)self, (PyObject *)self,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (ret == NULL) {
        Py_XDECREF(type);
        return NULL;
    }

    if (type != NULL) {
        if (PyObject_SetAttrString((PyObject *)ret, "dtype",
                                   (PyObject *)type) < 0) {
            Py_DECREF(ret);
            Py_DECREF(type);
            return NULL;
        }
        Py_DECREF(type);
    }
    return (PyObject *)ret;
}
