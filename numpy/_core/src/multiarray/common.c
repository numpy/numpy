#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"

#include "common.h"

#include "abstractdtypes.h"
#include "usertypes.h"

#include "npy_buffer.h"

#include "get_attr_string.h"
#include "mem_overlap.h"
#include "array_coercion.h"

/*
 * The casting to use for implicit assignment operations resulting from
 * in-place operations (like +=) and out= arguments. (Notice that this
 * variable is misnamed, but it's part of the public API so I'm not sure we
 * can just change it. Maybe someone should try and see if anyone notices.
 */
NPY_NO_EXPORT NPY_CASTING NPY_DEFAULT_ASSIGN_CASTING = NPY_SAME_KIND_CASTING;


NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op)
{
    if (PyFloat_Check(op)) {
        return PyArray_DescrFromType(NPY_DOUBLE);
    }
    else if (PyComplex_Check(op)) {
        return PyArray_DescrFromType(NPY_CDOUBLE);
    }
    else if (PyLong_Check(op)) {
        return NPY_DT_CALL_discover_descr_from_pyobject(
                &PyArray_PyLongDType, op);
    }
    return NULL;
}


/*
 * Get a suitable string dtype by calling `__str__`.
 * For `np.bytes_`, this assumes an ASCII encoding.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DTypeFromObjectStringDiscovery(
        PyObject *obj, PyArray_Descr *last_dtype, int string_type)
{
    npy_intp itemsize;

    if (string_type == NPY_STRING) {
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        /* assume that when we do the encoding elsewhere we'll use ASCII */
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
        if (itemsize > NPY_MAX_INT) {
            /* We can allow this, but should audit code paths before we do. */
            PyErr_Format(PyExc_TypeError,
                    "string of length %zd is too large to store inside array.", itemsize);
            return NULL;
        }
    }
    else if (string_type == NPY_UNICODE) {
        PyObject *temp = PyObject_Str(obj);
        if (temp == NULL) {
            return NULL;
        }
        itemsize = PyUnicode_GetLength(temp);
        Py_DECREF(temp);
        if (itemsize < 0) {
            return NULL;
        }
        if (itemsize > NPY_MAX_INT / 4) {
            PyErr_Format(PyExc_TypeError,
                    "string of length %zd is too large to store inside array.", itemsize);
            return NULL;
        }
        itemsize *= 4;  /* convert UCS4 codepoints to bytes */
    }
    else {
        return NULL;
    }
    if (last_dtype != NULL &&
        last_dtype->type_num == string_type &&
        last_dtype->elsize >= itemsize) {
        Py_INCREF(last_dtype);
        return last_dtype;
    }
    PyArray_Descr *dtype = PyArray_DescrNewFromType(string_type);
    if (dtype == NULL) {
        return NULL;
    }
    dtype->elsize = itemsize;
    return dtype;
}


/*
 * This function is now identical to the new PyArray_DiscoverDTypeAndShape
 * but only returns the dtype. It should in most cases be slowly phased out.
 * (Which may need some refactoring to PyArray_FromAny to make it simpler)
 */
NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, PyArray_Descr **out_dtype)
{
    coercion_cache_obj *cache = NULL;
    npy_intp shape[NPY_MAXDIMS];
    int ndim;

    ndim = PyArray_DiscoverDTypeAndShape(
            obj, maxdims, shape, &cache, NULL, NULL, out_dtype, 1, NULL);
    if (ndim < 0) {
        return -1;
    }
    npy_free_coercion_cache(cache);
    return 0;
}


NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base = PyArray_BASE(ap);
    Py_buffer view;

    /*
     * C-data wrapping arrays may not own their data while not having a base;
     * WRITEBACKIFCOPY arrays have a base, but do own their data.
     */
    if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
        /*
         * This is somewhat unsafe for directly wrapped non-writable C-arrays,
         * which do not know whether the memory area is writable or not and
         * do not own their data (but have no base).
         * It would be better if this returned PyArray_ISWRITEABLE(ap).
         * Since it is hard to deprecate, this is deprecated only on the Python
         * side, but not on in PyArray_UpdateFlags.
         */
        return NPY_TRUE;
    }

    /*
     * Get to the final base object.
     * If it is a writeable array, then return True if we can
     * find an array object or a writeable buffer object as
     * the final base object.
     */
    while (PyArray_Check(base)) {
        ap = (PyArrayObject *)base;
        base = PyArray_BASE(ap);

        if (PyArray_ISWRITEABLE(ap)) {
            /*
             * If any base is writeable, it must be OK to switch, note that
             * bases are typically collapsed to always point to the most
             * general one.
             */
            return NPY_TRUE;
        }

        if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
            /* there is no further base to test the writeable flag for */
            return NPY_FALSE;
        }
        assert(!PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA));
    }

    if (PyObject_GetBuffer(base, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
    PyBuffer_Release(&view);
    return NPY_TRUE;
}


/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param n Dimensionality of the shape
 * @param vals npy_intp pointer to shape array
 * @param ending String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp const *vals, char *ending)
{
    npy_intp i;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUnicode_FromFormat("()%s", ending);
    }

    PyObject *ret = PyUnicode_FromFormat("%" NPY_INTP_FMT, vals[i++]);
    if (ret == NULL) {
        return NULL;
    }
    for (; i < n; ++i) {
        PyObject *tmp;

        if (vals[i] < 0) {
            tmp = PyUnicode_FromString(",newaxis");
        }
        else {
            tmp = PyUnicode_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        Py_SETREF(ret, PyUnicode_Concat(ret, tmp));
        Py_DECREF(tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S,)%s", ret, ending));
    }
    else {
        Py_SETREF(ret, PyUnicode_FromFormat("(%S)%s", ret, ending));
    }
    return ret;
}


NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUnicode_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");

    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    shape1_i = PyLong_FromSsize_t(PyArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyArray_DIM(b, j));

    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    errmsg = PyUnicode_Format(format, fmt_args);
    if (errmsg != NULL) {
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
}

/**
 * unpack tuple of PyDataType_FIELDS(dtype) (descr, offset, title[not-needed])
 *
 * @param value should be the tuple.
 * @param descr will be set to the field's dtype
 * @param offset will be set to the field's offset
 *
 * @return -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset)
{
    PyObject * off;
    if (PyTuple_GET_SIZE(value) < 2) {
        return -1;
    }
    *descr = (PyArray_Descr *)PyTuple_GET_ITEM(value, 0);
    off  = PyTuple_GET_ITEM(value, 1);

    if (PyLong_Check(off)) {
        *offset = PyLong_AsSsize_t(off);
    }
    else {
        PyErr_SetString(PyExc_IndexError, "can't convert offset");
        return -1;
    }

    return 0;
}


/**
 * Unpack a field from a structured dtype. The field index must be valid.
 *
 * @param descr The dtype to unpack.
 * @param index The index of the field to unpack.
 * @param odescr will be set to the field's dtype
 * @param offset will be set to the field's offset
 *
 * @return -1 on failure, 0 on success.
 */
 NPY_NO_EXPORT int
 _unpack_field_index(
    _PyArray_LegacyDescr *descr,
    npy_intp index,
    PyArray_Descr **odescr,
    npy_intp *offset)
 {
    PyObject *key = PyTuple_GET_ITEM(descr->names, index);
    PyObject *tup = PyDict_GetItem(descr->fields, key);  // noqa: borrowed-ref OK
    return _unpack_field(tup, odescr, offset);
 }


/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype)
{
    PyArray_Descr *base = dtype;
    if (PyDataType_HASSUBARRAY(dtype)) {
        base = ((_PyArray_LegacyDescr *)dtype)->subarray->base;
    }

    return (PyDataType_HASFIELDS(base) ||
            PyDataType_FLAGCHK(base, NPY_ITEM_HASOBJECT) );
}

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result)
{
    PyArrayObject *out_buf;

    if (out) {
        int d;

        /* verify that out is usable */
        if (PyArray_NDIM(out) != nd ||
            PyArray_TYPE(out) != typenum ||
            !PyArray_ISCARRAY(out)) {
            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return 0;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                return 0;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = (PyArrayObject *)PyArray_NewLikeArray(out, NPY_CORDER,
                                                            NULL, 0);
            if (out_buf == NULL) {
                return NULL;
            }

            /* set copy-back */
            Py_INCREF(out);
            if (PyArray_SetWritebackIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                Py_DECREF(out_buf);
                return NULL;
            }
        }
        else {
            Py_INCREF(out);
            out_buf = out;
        }

        if (result) {
            Py_INCREF(out);
            *result = out;
        }

        return out_buf;
    }
    else {
        PyTypeObject *subtype;
        double prior1, prior2;
        /*
         * Need to choose an output array that can hold a sum
         * -- use priority to determine which subtype.
         */
        if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
            prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
            prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
            subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
        }
        else {
            prior1 = prior2 = 0.0;
            subtype = Py_TYPE(ap1);
        }

        out_buf = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0,
                                               (PyObject *)
                                               (prior2 > prior1 ? ap2 : ap1));

        if (out_buf != NULL && result) {
            Py_INCREF(out_buf);
            *result = out_buf;
        }

        return out_buf;
    }
}

NPY_NO_EXPORT int
check_is_convertible_to_scalar(PyArrayObject *v)
{
    if (PyArray_NDIM(v) == 0) {
        return 0;
    }

    PyErr_SetString(PyExc_TypeError,
            "only 0-dimensional arrays can be converted to Python scalars");
    return -1;
}
