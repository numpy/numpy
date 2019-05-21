#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "common.h"

#include "usertypes.h"

#include "common.h"
#include "buffer.h"

#include "get_attr_string.h"
#include "mem_overlap.h"

/*
 * The casting to use for implicit assignment operations resulting from
 * in-place operations (like +=) and out= arguments. (Notice that this
 * variable is misnamed, but it's part of the public API so I'm not sure we
 * can just change it. Maybe someone should try and see if anyone notices.
 */
/*
 * In numpy 1.6 and earlier, this was NPY_UNSAFE_CASTING. In a future
 * release, it will become NPY_SAME_KIND_CASTING.  Right now, during the
 * transitional period, we continue to follow the NPY_UNSAFE_CASTING rules (to
 * avoid breaking people's code), but we also check for whether the cast would
 * be allowed under the NPY_SAME_KIND_CASTING rules, and if not we issue a
 * warning (that people's code will be broken in a future release.)
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
    else if (PyInt_Check(op)) {
        /* bools are a subclass of int */
        if (PyBool_Check(op)) {
            return PyArray_DescrFromType(NPY_BOOL);
        }
        else {
            return  PyArray_DescrFromType(NPY_LONG);
        }
    }
    else if (PyLong_Check(op)) {
        /* check to see if integer can fit into a longlong or ulonglong
           and return that --- otherwise return object */
        if ((PyLong_AsLongLong(op) == -1) && PyErr_Occurred()) {
            PyErr_Clear();
        }
        else {
            return PyArray_DescrFromType(NPY_LONGLONG);
        }

        if ((PyLong_AsUnsignedLongLong(op) == (unsigned long long) -1)
            && PyErr_Occurred()){
            PyErr_Clear();
        }
        else {
            return PyArray_DescrFromType(NPY_ULONGLONG);
        }

        return PyArray_DescrFromType(NPY_OBJECT);
    }
    return NULL;
}

/*
 * These constants are used to signal that the recursive dtype determination in
 * PyArray_DTypeFromObject encountered a string type, and that the recursive
 * search must be restarted so that string representation lengths can be
 * computed for all scalar types.
 */
#define RETRY_WITH_STRING 1
#define RETRY_WITH_UNICODE 2

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 * This is reset to NULL on failure.
 *
 * Returns 0 on success, -1 on failure.
 */
 NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, PyArray_Descr **out_dtype)
{
    int res;

    res = PyArray_DTypeFromObjectHelper(obj, maxdims, out_dtype, 0);
    if (res == RETRY_WITH_STRING) {
        res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                            out_dtype, NPY_STRING);
        if (res == RETRY_WITH_UNICODE) {
            res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                                out_dtype, NPY_UNICODE);
        }
    }
    else if (res == RETRY_WITH_UNICODE) {
        res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                            out_dtype, NPY_UNICODE);
    }
    return res;
}

NPY_NO_EXPORT int
PyArray_DTypeFromObjectHelper(PyObject *obj, int maxdims,
                              PyArray_Descr **out_dtype, int string_type)
{
    int i, size;
    PyArray_Descr *dtype = NULL;
    PyObject *ip;
    Py_buffer buffer_view;
    /* types for sequence handling */
    PyObject ** objects;
    PyObject * seq;
    PyTypeObject * common_type;

    /* Check if it's an ndarray */
    if (PyArray_Check(obj)) {
        dtype = PyArray_DESCR((PyArrayObject *)obj);
        Py_INCREF(dtype);
        goto promote_types;
    }

    /* See if it's a python None */
    if (obj == Py_None) {
        dtype = PyArray_DescrFromType(NPY_OBJECT);
        if (dtype == NULL) {
            goto fail;
        }
        Py_INCREF(dtype);
        goto promote_types;
    }
    /* Check if it's a NumPy scalar */
    else if (PyArray_IsScalar(obj, Generic)) {
        if (!string_type) {
            dtype = PyArray_DescrFromScalar(obj);
            if (dtype == NULL) {
                goto fail;
            }
        }
        else {
            int itemsize;
            PyObject *temp;

            if (string_type == NPY_STRING) {
                if ((temp = PyObject_Str(obj)) == NULL) {
                    goto fail;
                }
#if defined(NPY_PY3K)
    #if PY_VERSION_HEX >= 0x03030000
                itemsize = PyUnicode_GetLength(temp);
    #else
                itemsize = PyUnicode_GET_SIZE(temp);
    #endif
#else
                itemsize = PyString_GET_SIZE(temp);
#endif
            }
            else if (string_type == NPY_UNICODE) {
#if defined(NPY_PY3K)
                if ((temp = PyObject_Str(obj)) == NULL) {
#else
                if ((temp = PyObject_Unicode(obj)) == NULL) {
#endif
                    goto fail;
                }
                itemsize = PyUnicode_GET_DATA_SIZE(temp);
#ifndef Py_UNICODE_WIDE
                itemsize <<= 1;
#endif
            }
            else {
                goto fail;
            }
            Py_DECREF(temp);
            if (*out_dtype != NULL &&
                    (*out_dtype)->type_num == string_type &&
                    (*out_dtype)->elsize >= itemsize) {
                return 0;
            }
            dtype = PyArray_DescrNewFromType(string_type);
            if (dtype == NULL) {
                goto fail;
            }
            dtype->elsize = itemsize;
        }
        goto promote_types;
    }

    /* Check if it's a Python scalar */
    dtype = _array_find_python_scalar_type(obj);
    if (dtype != NULL) {
        if (string_type) {
            int itemsize;
            PyObject *temp;

            if (string_type == NPY_STRING) {
                if ((temp = PyObject_Str(obj)) == NULL) {
                    goto fail;
                }
#if defined(NPY_PY3K)
    #if PY_VERSION_HEX >= 0x03030000
                itemsize = PyUnicode_GetLength(temp);
    #else
                itemsize = PyUnicode_GET_SIZE(temp);
    #endif
#else
                itemsize = PyString_GET_SIZE(temp);
#endif
            }
            else if (string_type == NPY_UNICODE) {
#if defined(NPY_PY3K)
                if ((temp = PyObject_Str(obj)) == NULL) {
#else
                if ((temp = PyObject_Unicode(obj)) == NULL) {
#endif
                    goto fail;
                }
                itemsize = PyUnicode_GET_DATA_SIZE(temp);
#ifndef Py_UNICODE_WIDE
                itemsize <<= 1;
#endif
            }
            else {
                goto fail;
            }
            Py_DECREF(temp);
            if (*out_dtype != NULL &&
                    (*out_dtype)->type_num == string_type &&
                    (*out_dtype)->elsize >= itemsize) {
                return 0;
            }
            dtype = PyArray_DescrNewFromType(string_type);
            if (dtype == NULL) {
                goto fail;
            }
            dtype->elsize = itemsize;
        }
        goto promote_types;
    }

    /* Check if it's an ASCII string */
    if (PyBytes_Check(obj)) {
        int itemsize = PyString_GET_SIZE(obj);

        /* If it's already a big enough string, don't bother type promoting */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_STRING &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        dtype = PyArray_DescrNewFromType(NPY_STRING);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = itemsize;
        goto promote_types;
    }

    /* Check if it's a Unicode string */
    if (PyUnicode_Check(obj)) {
        int itemsize = PyUnicode_GET_DATA_SIZE(obj);
#ifndef Py_UNICODE_WIDE
        itemsize <<= 1;
#endif

        /*
         * If it's already a big enough unicode object,
         * don't bother type promoting
         */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_UNICODE &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        dtype = PyArray_DescrNewFromType(NPY_UNICODE);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = itemsize;
        goto promote_types;
    }

    /* PEP 3118 buffer interface */
    if (PyObject_CheckBuffer(obj) == 1) {
        memset(&buffer_view, 0, sizeof(Py_buffer));
        if (PyObject_GetBuffer(obj, &buffer_view,
                               PyBUF_FORMAT|PyBUF_STRIDES) == 0 ||
            PyObject_GetBuffer(obj, &buffer_view,
                               PyBUF_FORMAT|PyBUF_SIMPLE) == 0) {

            PyErr_Clear();
            dtype = _descriptor_from_pep3118_format(buffer_view.format);
            PyBuffer_Release(&buffer_view);
            _dealloc_cached_buffer_info(obj);
            if (dtype) {
                goto promote_types;
            }
        }
        else if (PyObject_GetBuffer(obj, &buffer_view, PyBUF_STRIDES) == 0 ||
                 PyObject_GetBuffer(obj, &buffer_view, PyBUF_SIMPLE) == 0) {

            PyErr_Clear();
            dtype = PyArray_DescrNewFromType(NPY_VOID);
            dtype->elsize = buffer_view.itemsize;
            PyBuffer_Release(&buffer_view);
            _dealloc_cached_buffer_info(obj);
            goto promote_types;
        }
        else {
            PyErr_Clear();
        }
    }

    /* The array interface */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array_interface__");
    if (ip != NULL) {
        if (PyDict_Check(ip)) {
            PyObject *typestr;
#if defined(NPY_PY3K)
            PyObject *tmp = NULL;
#endif
            typestr = PyDict_GetItemString(ip, "typestr");
#if defined(NPY_PY3K)
            /* Allow unicode type strings */
            if (typestr && PyUnicode_Check(typestr)) {
                tmp = PyUnicode_AsASCIIString(typestr);
                typestr = tmp;
            }
#endif
            if (typestr && PyBytes_Check(typestr)) {
                dtype =_array_typedescr_fromstr(PyBytes_AS_STRING(typestr));
#if defined(NPY_PY3K)
                if (tmp == typestr) {
                    Py_DECREF(tmp);
                }
#endif
                Py_DECREF(ip);
                if (dtype == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }

    /* The array struct interface */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array_struct__");
    if (ip != NULL) {
        PyArrayInterface *inter;
        char buf[40];

        if (NpyCapsule_Check(ip)) {
            inter = (PyArrayInterface *)NpyCapsule_AsVoidPtr(ip);
            if (inter->two == 2) {
                PyOS_snprintf(buf, sizeof(buf),
                        "|%c%d", inter->typekind, inter->itemsize);
                dtype = _array_typedescr_fromstr(buf);
                Py_DECREF(ip);
                if (dtype == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }

    /* The old buffer interface */
#if !defined(NPY_PY3K)
    if (PyBuffer_Check(obj)) {
        dtype = PyArray_DescrNewFromType(NPY_VOID);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = Py_TYPE(obj)->tp_as_sequence->sq_length(obj);
        PyErr_Clear();
        goto promote_types;
    }
#endif

    /* The __array__ attribute */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array__");
    if (ip != NULL) {
        Py_DECREF(ip);
        ip = PyObject_CallMethod(obj, "__array__", NULL);
        if(ip && PyArray_Check(ip)) {
            dtype = PyArray_DESCR((PyArrayObject *)ip);
            Py_INCREF(dtype);
            Py_DECREF(ip);
            goto promote_types;
        }
        Py_XDECREF(ip);
        if (PyErr_Occurred()) {
            goto fail;
        }
    }

    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, and obj isn't a sequence-like object, the output
     * dtype should be either OBJECT or a user-defined type.
     *
     * Note that some libraries define sequence-like classes but want them to
     * be treated as objects, and they expect numpy to treat it as an object if
     * __len__ is not defined.
     */
    if (maxdims == 0 || !PySequence_Check(obj) || PySequence_Size(obj) < 0) {
        /* clear any PySequence_Size error which corrupts further calls */
        PyErr_Clear();

        if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
            Py_XDECREF(*out_dtype);
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
        }
        return 0;
    }

    /*
     * The C-API recommends calling PySequence_Fast before any of the other
     * PySequence_Fast* functions. This is required for PyPy
     */
    seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        goto fail;
    }

    /* Recursive case, first check the sequence contains only one type */
    size = PySequence_Fast_GET_SIZE(seq);
    /* objects is borrowed, do not release seq */
    objects = PySequence_Fast_ITEMS(seq);
    common_type = size > 0 ? Py_TYPE(objects[0]) : NULL;
    for (i = 1; i < size; ++i) {
        if (Py_TYPE(objects[i]) != common_type) {
            common_type = NULL;
            break;
        }
    }

    /* all types are the same and scalar, one recursive call is enough */
    if (common_type != NULL && !string_type &&
            (common_type == &PyFloat_Type ||
/* TODO: we could add longs if we add a range check */
#if !defined(NPY_PY3K)
             common_type == &PyInt_Type ||
#endif
             common_type == &PyBool_Type ||
             common_type == &PyComplex_Type)) {
        size = 1;
    }

    /* Recursive call for each sequence item */
    for (i = 0; i < size; ++i) {
        int res = PyArray_DTypeFromObjectHelper(objects[i], maxdims - 1,
                                                out_dtype, string_type);
        if (res < 0) {
            Py_DECREF(seq);
            goto fail;
        }
        else if (res > 0) {
            Py_DECREF(seq);
            return res;
        }
    }

    Py_DECREF(seq);

    return 0;


promote_types:
    /* Set 'out_dtype' if it's NULL */
    if (*out_dtype == NULL) {
        if (!string_type && dtype->type_num == NPY_STRING) {
            Py_DECREF(dtype);
            return RETRY_WITH_STRING;
        }
        if (!string_type && dtype->type_num == NPY_UNICODE) {
            Py_DECREF(dtype);
            return RETRY_WITH_UNICODE;
        }
        *out_dtype = dtype;
        return 0;
    }
    /* Do type promotion with 'out_dtype' */
    else {
        PyArray_Descr *res_dtype = PyArray_PromoteTypes(dtype, *out_dtype);
        Py_DECREF(dtype);
        if (res_dtype == NULL) {
            goto fail;
        }
        if (!string_type &&
                res_dtype->type_num == NPY_UNICODE &&
                (*out_dtype)->type_num != NPY_UNICODE) {
            Py_DECREF(res_dtype);
            return RETRY_WITH_UNICODE;
        }
        if (!string_type &&
                res_dtype->type_num == NPY_STRING &&
                (*out_dtype)->type_num != NPY_STRING) {
            Py_DECREF(res_dtype);
            return RETRY_WITH_STRING;
        }
        Py_DECREF(*out_dtype);
        *out_dtype = res_dtype;
        return 0;
    }

fail:
    Py_XDECREF(*out_dtype);
    *out_dtype = NULL;
    return -1;
}

#undef RETRY_WITH_STRING
#undef RETRY_WITH_UNICODE

/* new reference */
NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *c_str)
{
    PyArray_Descr *descr = NULL;
    PyObject *stringobj = PyString_FromString(c_str);

    if (stringobj == NULL) {
        return NULL;
    }
    if (PyArray_DescrConverter(stringobj, &descr) != NPY_SUCCEED) {
        Py_DECREF(stringobj);
        return NULL;
    }
    Py_DECREF(stringobj);
    return descr;
}


NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, npy_intp i)
{
    npy_intp dim0;

    if (PyArray_NDIM(mp) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = PyArray_DIMS(mp)[0];
    if (check_and_adjust_index(&i, dim0, 0, NULL) < 0)
        return NULL;
    if (i == 0) {
        return PyArray_DATA(mp);
    }
    return PyArray_BYTES(mp)+i*PyArray_STRIDES(mp)[0];
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        PyObject *zero = PyInt_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        npy_intp n = PyArray_NBYTES(ret);
        memset(PyArray_DATA(ret), 0, n);
    }
    return 0;
}

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base = PyArray_BASE(ap);
#if defined(NPY_PY3K)
    Py_buffer view;
#else
    void *dummy;
    Py_ssize_t n;
#endif

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

        if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
            return (npy_bool) PyArray_ISWRITEABLE(ap);
        }
        assert(!PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA));
    }

#if defined(NPY_PY3K)
    if (PyObject_GetBuffer(base, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
    PyBuffer_Release(&view);
    /*
     * The first call to PyObject_GetBuffer stores a reference to a struct
     * _buffer_info_t (from buffer.c, with format, ndim, strides and shape) in
     * a static dictionary, with id(base) as the key. Usually we release it
     * after the call to PyBuffer_Release, via a call to
     * _dealloc_cached_buffer_info, but in this case leave it in the cache to
     * speed up future calls to _IsWriteable.
     */
#else
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
#endif
    return NPY_TRUE;
}


/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param Dimensionality of the shape
 * @param npy_intp pointer to shape array
 * @param String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUString_FromFormat("()%s", ending);
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        tmp = PyUString_FromFormat(",)%s", ending);
    }
    else {
        tmp = PyUString_FromFormat(")%s", ending);
    }
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}


NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUString_FromString("shapes %s and %s not aligned:"
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

    errmsg = PyUString_Format(format, fmt_args);
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
 * unpack tuple of dtype->fields (descr, offset, title[not-needed])
 *
 * @param "value" should be the tuple.
 *
 * @return "descr" will be set to the field's dtype
 * @return "offset" will be set to the field's offset
 *
 * returns -1 on failure, 0 on success.
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

    if (PyInt_Check(off)) {
        *offset = PyInt_AsSsize_t(off);
    }
    else if (PyLong_Check(off)) {
        *offset = PyLong_AsSsize_t(off);
    }
    else {
        PyErr_SetString(PyExc_IndexError, "can't convert offset");
        return -1;
    }

    return 0;
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
        base = dtype->subarray->base;
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



