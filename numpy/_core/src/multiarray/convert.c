#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "npy_config.h"
#include "npy_pycompat.h"  // PyObject_GetOptionalAttr

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "alloc.h"
#include "common.h"
#include "arrayobject.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"
#include "scalartypes.h"
#include "array_assign.h"

#include "convert.h"
#include "array_coercion.h"
#include "refcount.h"
#include "getset.h"
#include "npy_static_data.h"

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
    npy_intp offset;
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
    offset = npy_ftell(fp);
    r = fallocate(fileno(fp), 1, offset, nbytes);
    NPY_END_ALLOW_THREADS;

    /*
     * early exit on no space, other errors will also get found during fwrite
     */
    if (r == -1 && errno == ENOSPC) {
        PyErr_Format(PyExc_OSError, "Not enough free space to write "
                     "%"NPY_INTP_FMT" bytes after offset %"NPY_INTP_FMT,
                     nbytes, offset);
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
        if (PyArray_ITEMSIZE(self) == 0) {
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
                size_t maxsize = 2147483648 / (size_t)PyArray_ITEMSIZE(self);
                size_t chunksize;

                n = 0;
                while (size > 0) {
                    chunksize = (size > maxsize) ? maxsize : size;
                    n2 = fwrite((const void *)
                             ((char *)PyArray_DATA(self) + (n * PyArray_ITEMSIZE(self))),
                             (size_t) PyArray_ITEMSIZE(self),
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
                    (size_t) PyArray_ITEMSIZE(self),
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
                            (size_t) PyArray_ITEMSIZE(self),
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
            /*
             * This is as documented.  If we have a low precision float value
             * then it may convert to float64 and store unnecessary digits.
             * TODO: This could be fixed, by not using `arr.item()` or using
             *       the array printing/formatting functionality.
             */
            obj = PyArray_GETITEM(self, it->dataptr);
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
    PyObject *ret;

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
        return PyBytes_FromStringAndSize(PyArray_DATA(self), (Py_ssize_t) numbytes);
    }

    /* Avoid Ravel where possible for fewer copies. */
    if (!PyDataType_REFCHK(PyArray_DESCR(self)) &&
        ((PyArray_DESCR(self)->flags & NPY_NEEDS_INIT) == 0)) {

        /* Allocate final Bytes Object */
        ret = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            return NULL;
        }

        /* Writable Buffer */
        char* dest = PyBytes_AS_STRING(ret);

        int flags = NPY_ARRAY_WRITEABLE;
        if (order == NPY_FORTRANORDER) {
            flags |= NPY_ARRAY_F_CONTIGUOUS;
        }

        Py_INCREF(PyArray_DESCR(self));
        /* Array view */
        PyArrayObject *dest_array = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type,
            PyArray_DESCR(self),
            PyArray_NDIM(self),
            PyArray_DIMS(self),
            NULL, // strides
            dest,
            flags,
            NULL
        );

        if (dest_array == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        /* Copy directly from source to destination with proper ordering */
        if (PyArray_CopyInto(dest_array, self) < 0) {
            Py_DECREF(dest_array);
            Py_DECREF(ret);
            return NULL;
        }

        Py_DECREF(dest_array);
        return ret;

    }

    /* Non-contiguous, Has References and/or Init Path.  */
    PyArrayObject *contig = (PyArrayObject *)PyArray_Ravel(self, order);
    if (contig == NULL) {
        return NULL;
    }

    ret = PyBytes_FromStringAndSize(PyArray_DATA(contig), numbytes);
    Py_DECREF(contig);
    return ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_FillWithScalar(PyArrayObject *arr, PyObject *obj)
{

    if (PyArray_FailUnlessWriteable(arr, "assignment destination") < 0) {
        return -1;
    }

    PyArray_Descr *descr = PyArray_DESCR(arr);

    /*
     * If we knew that the output array has at least one element, we would
     * not actually need a helping buffer, we always null it, just in case.
     * Use `long double` to ensure that the heap allocation is aligned.
     */
    size_t n_max_align_t = (descr->elsize + sizeof(long double) - 1) / sizeof(long double);
    NPY_ALLOC_WORKSPACE(value, long double, 2, n_max_align_t);
    if (value == NULL) {
        return -1;
    }
    if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
        memset(value, 0, descr->elsize);
    }

    if (PyArray_Pack(descr, value, obj) < 0) {
        npy_free_workspace(value);
        return -1;
    }

    /*
     * There is no cast anymore, the above already coerced using scalar
     * coercion rules
     */
    int retcode = raw_array_assign_scalar(
            PyArray_NDIM(arr), PyArray_DIMS(arr), descr,
            PyArray_BYTES(arr), PyArray_STRIDES(arr),
            descr, (void *)value, NPY_UNSAFE_CASTING);

    if (PyDataType_REFCHK(descr)) {
        PyArray_ClearBuffer(descr, (void *)value, 0, 1, 1);
    }
    npy_free_workspace(value);
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


static int
get_optional_set_dtype_and_dtype(
    PyTypeObject *subtype,
    PyObject **_set_dtype, PyObject **sub_dtype)
{
    if (PyObject_GetOptionalAttr(
            (PyObject *)subtype, npy_interned_str._set_dtype,
            _set_dtype) < 0) {
        return -1;
    }
    if (PyObject_GetOptionalAttr(
            (PyObject *)subtype, npy_interned_str.dtype, sub_dtype) < 0) {
        Py_XDECREF(*_set_dtype);
        return -1;
    }
    return 0;
}


/* Pick how a view() dtype change is propagated for `subtype`.
 *
 * Sets (at most) one of the three output flags; all three zero means
 * the legacy in-place path (subclass can't observe).
 *
 *   use_dtype_in_finalize  -- `_set_dtype = None`: class wants
 *                             __array_finalize__ to see the final dtype.
 *   use_set_dtype          -- subclass `_set_dtype` method: call it
 *                             as an adjust hook after viewing.
 *   use_dtype_prop         -- subclass `dtype` descriptor wins over
 *                             `_set_dtype`: call the setter (deprecated).
 *
 * Walk the MRO and see whether `_set_dtype` or `dtype` was overridden
 * more specifically: whichever's resolved value diverges from what
 * `subtype` sees at a lower MRO level wins.  Ties and "only
 * `_set_dtype` was overridden" both go to `_set_dtype`, matching the
 * direction numpy's deprecation points users toward.  Blind to
 * `__setattr__` / odd MI. */
static int
decide_view_dtype_path(
        PyTypeObject *subtype,
        int *use_dtype_in_finalize,
        int *use_set_dtype,
        int *use_dtype_prop)
{
    int ret = -1;
    *use_dtype_in_finalize = 1;  /* Future defaults. */
    *use_set_dtype = 0;
    *use_dtype_prop = 0;

    if (subtype == &PyArray_Type) {
        return 0;
    }

    PyObject *sub_set_dtype = NULL, *sub_dtype = NULL, *mro = NULL;
    if (get_optional_set_dtype_and_dtype(
            subtype, &sub_set_dtype, &sub_dtype) < 0) {
        goto finish;
    }

    int set_overridden =
            (sub_set_dtype != npy_static_pydata.ndarray_set_dtype);
    int dtype_overridden =
            (sub_dtype != npy_static_pydata.ndarray_dtype_descr);

    /* Default: `_set_dtype` wins (either it was overridden, or nothing
     * was); flipped only if the walk below finds `dtype` diverging
     * first, or `dtype` was the sole override. */
    int set_wins = set_overridden || !dtype_overridden;

    if (set_overridden && dtype_overridden) {
        /* Both overridden -- walk the MRO to see which was overridden
         * more specifically.  Pin `tp_mro`; under free-threading
         * `cls.__bases__ = ...` can replace it concurrently.  The tuple
         * owns refs to its entries, so the base types stay alive too. */
        mro = Py_XNewRef(subtype->tp_mro);
        Py_ssize_t n = mro != NULL ? PyTuple_GET_SIZE(mro) : 0;
        for (Py_ssize_t i = 1; i < n; i++) {
            PyTypeObject *base = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);
            PyObject *v_set, *v_dtype;
            if (get_optional_set_dtype_and_dtype(base, &v_set, &v_dtype) < 0) {
                goto finish;
            }
            /* NULL means this base doesn't know the name at all (an MI
             * sibling branch above ndarray).  That's "no information",
             * not divergence -- ndarray sits deeper in the MRO and will
             * provide the real baseline. */
            int set_div = v_set != NULL && v_set != sub_set_dtype;
            int dtype_div = v_dtype != NULL && v_dtype != sub_dtype;
            Py_XDECREF(v_set);
            Py_XDECREF(v_dtype);

            if (set_div || dtype_div) {
                /* First to diverge wins; tie (both) -> `_set_dtype`. */
                set_wins = set_div;
                break;
            }
        }
    }

    if (set_wins) {
        if (sub_set_dtype != Py_None) {
            *use_dtype_in_finalize = 0;
            *use_set_dtype = set_overridden;
        }
    }
    else {
        *use_dtype_in_finalize = 0;
        *use_dtype_prop = (sub_dtype != NULL
                           && Py_TYPE(sub_dtype)->tp_descr_set != NULL);
    }

    ret = 0;
  finish:
    Py_XDECREF(sub_set_dtype);
    Py_XDECREF(sub_dtype);
    Py_XDECREF(mro);
    return ret;
}


/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    PyObject *ret = NULL;
    int nd = PyArray_NDIM(self);
    npy_intp *dims = PyArray_DIMS(self);
    npy_intp *strides = PyArray_STRIDES(self);
    PyArray_Descr *dtype = PyArray_DESCR(self);
    int flags = PyArray_FLAGS(self);
    PyTypeObject *subtype;

    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
    }

    if (type == NULL) {
        /* No dtype change. */
        Py_INCREF(dtype);
        return PyArray_NewFromDescr_int(
                subtype, dtype, nd, dims, strides, PyArray_DATA(self),
                flags, (PyObject *)self, (PyObject *)self,
                _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    }

    /*
     * Changing dtype on a subclass.  We support 4 paths, based on whether
     * a subclass overrides _set_dtype or the dtype setter (where whichever
     * is overridden most recently wins):
     *
     * 1. If _set_dtype is None: create a new view with the new dtype.
     *    This is the future: __array_finalize__ sees final dtype and shape.
     * 2. subclass overrides _set_dtype: create subclass view first,
     *    then call _set_dtype (subclass handles dtype change).
     * 3. subclass overrides the dtype descriptor (e.g. property with
     *    setter): create subclass view first, use the setter, but
     *    emit a deprecation asking to implement _set_dtype instead.
     * 4. If _set_dtype and dtype are not set, call `__array_finalize__`
     *    with the old dtype and forcibly update the dtype (a subclass will be
     *    unaware of the change). This is the unfortunate historic behavior.
     *
     * (Base class ndarray uses path 1, but has no __array_finalize__,
     *  so it is the same as paths 2 and 4.)
     */
    int use_dtype_in_finalize, use_set_dtype, use_dtype_prop;
    if (decide_view_dtype_path(
            subtype, &use_dtype_in_finalize,
            &use_set_dtype, &use_dtype_prop) < 0) {
        goto finish;
    }

    if (use_dtype_in_finalize) {
        /*
         * Path 1: subclass lives in the future and its __array_finalize__
         * can handle getting the correct dtype+shape.
         */
        npy_intp newlastdim, newlaststride;
        /* Check whether the type is compatible. */
        Py_SETREF(type, _check_compatibility_with_new_dtype(
                      self, type, &newlastdim, &newlaststride));
        if (type == NULL) {
            return NULL;
        }
        /* Take view with old or adjusted dims (steals reference to type) */
        if (newlastdim < 0) {
            return PyArray_NewFromDescr_int(subtype, type,
                    nd, dims, strides, PyArray_DATA(self),
                    flags, (PyObject *)self, (PyObject *)self, 0);
        }
        else {
            NPY_ALLOC_WORKSPACE(newdims, npy_intp, 2 * 4, 2 * nd);
            if (newdims == NULL) {
                goto finish;
            }
            npy_intp *newstrides = newdims + nd;
            memcpy(newdims, dims, (nd-1)*sizeof(npy_intp));
            memcpy(newstrides, strides, (nd-1)*sizeof(npy_intp));
            newdims[nd-1] = newlastdim;
            newstrides[nd-1] = newlaststride;
            ret = PyArray_NewFromDescr_int(subtype, type,
                    nd, newdims, newstrides, PyArray_DATA(self),
                    flags, (PyObject *)self, (PyObject *)self, 0);
            npy_free_workspace(newdims);
            return ret;
        }
    }
    /*
     * Other paths: first create a view with the old dtype.
     */
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescr_int(
            subtype, dtype, nd, dims, strides, PyArray_DATA(self),
            flags, (PyObject *)self, (PyObject *)self,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (ret == NULL) {
        goto finish;
    }

    if (use_set_dtype) {
        /*
         * Path 2: subclass lives in future but needs to set dtype itself.
         */
        PyObject *res = PyObject_CallMethodOneArg(
                ret, npy_interned_str._set_dtype, (PyObject *)type);
        if (res == NULL) {
            Py_CLEAR(ret);
            goto finish;
        }
        Py_DECREF(res);
    }
    else if (use_dtype_prop) {
        /*
         * Path 3: subclass overrides dtype property.
         */
        if (PyObject_GenericSetAttr(
                ret, npy_interned_str.dtype, (PyObject *)type) < 0) {
            Py_CLEAR(ret);
            goto finish;
        }
        /*
         * Path 3: subclass overrides dtype property.
         * DEPRECATED 2026-04-13, NumPy 2.5.
         * After the deprecation, the decide_view_dtype_path helper isn't
         * needed.  `_set_dtype` is used unless it is the base-class
         * definition or None, which are the only 3 options left without
         * need for MRO walking.
         */
        if (DEPRECATE(
                "numpy.ndarray.view() used a custom `dtype` setter "
                "to change the dtype of the view.  Subclasses should "
                "implement `_set_dtype` instead. (Deprecated NumPy 2.5)") < 0) {
            Py_CLEAR(ret);
            goto finish;
        }
    }
    else {
        /*
         * Path 4: set dtype internally.
         */
        if (array_descr_set_internal(
                (PyArrayObject*)ret, (PyObject *)type) < 0) {
            Py_CLEAR(ret);
            goto finish;
        }
    }

finish:
    Py_DECREF(type);
    return ret;
}
