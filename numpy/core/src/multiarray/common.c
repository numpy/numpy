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

/*
 * PyArray_GetAttrString_SuppressException:
 *
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns attribute value on success, 0 on failure.
 */
PyObject *
PyArray_GetAttrString_SuppressException(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* We do not need to check for special attributes on trivial types */
    if (obj == Py_None ||
            PyList_CheckExact(obj) ||
            PyTuple_CheckExact(obj)) {
        return NULL;
    }

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
#if defined(NPY_PY3K)
        PyObject *w = PyUnicode_InternFromString(name);
#else
        PyObject *w = PyString_InternFromString(name);
#endif
        if (w == NULL) {
            return (PyObject *)NULL;
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return res;
}



NPY_NO_EXPORT NPY_CASTING NPY_DEFAULT_ASSIGN_CASTING = NPY_INTERNAL_UNSAFE_CASTING_BUT_WARN_UNLESS_SAME_KIND;


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

#if !defined(NPY_PY3K)
static PyArray_Descr *
_use_default_type(PyObject *op)
{
    int typenum, l;
    PyObject *type;

    typenum = -1;
    l = 0;
    type = (PyObject *)Py_TYPE(op);
    while (l < NPY_NUMUSERTYPES) {
        if (type == (PyObject *)(userdescrs[l]->typeobj)) {
            typenum = l + NPY_USERDEF;
            break;
        }
        l++;
    }
    if (typenum == -1) {
        typenum = NPY_OBJECT;
    }
    return PyArray_DescrFromType(typenum);
}
#endif

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
                    return -1;
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
                    return -1;
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
                    return -1;
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
                    return -1;
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
            PyObject_GetBuffer(obj, &buffer_view, PyBUF_FORMAT) == 0) {
    
            PyErr_Clear();
            dtype = _descriptor_from_pep3118_format(buffer_view.format);
            PyBuffer_Release(&buffer_view);
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
            goto promote_types;
        }
        else {
            PyErr_Clear();
        }
    }

    /* The array interface */
    ip = PyArray_GetAttrString_SuppressException(obj, "__array_interface__");
    if (ip != NULL) {
        if (PyDict_Check(ip)) {
            PyObject *typestr;
#if defined(NPY_PY3K)
            PyObject *tmp = NULL;
#endif
            typestr = PyDict_GetItemString(ip, "typestr");
#if defined(NPY_PY3K)
            /* Allow unicode type strings */
            if (PyUnicode_Check(typestr)) {
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
    ip = PyArray_GetAttrString_SuppressException(obj, "__array_struct__");
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
    ip = PyArray_GetAttrString_SuppressException(obj, "__array__");
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

    /* Not exactly sure what this is about... */
#if !defined(NPY_PY3K)
    if (PyInstance_Check(obj)) {
        dtype = _use_default_type(obj);
        if (dtype == NULL) {
            goto fail;
        }
        else {
            goto promote_types;
        }
    }
#endif

    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, the output dtype should be OBJECT
     */
    if (maxdims == 0 || !PySequence_Check(obj)) {
        if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
            Py_XDECREF(*out_dtype);
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
        }
        return 0;
    }

    /* Recursive case */
    size = PySequence_Size(obj);
    if (size < 0) {
        goto fail;
    }
    /* Recursive call for each sequence item */
    for (i = 0; i < size; ++i) {
        int res;
        ip = PySequence_GetItem(obj, i);
        if (ip == NULL) {
            goto fail;
        }
        res = PyArray_DTypeFromObjectHelper(ip, maxdims - 1,
                                            out_dtype, string_type);
        if (res < 0) {
            Py_DECREF(ip);
            goto fail;
        }
        else if (res > 0) {
            Py_DECREF(ip);
            return res;
        }
        Py_DECREF(ip);
    }

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
            return -1;
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

NPY_NO_EXPORT int
check_and_adjust_index(npy_intp *index, npy_intp max_item, int axis)
{
    /* Check that index is valid, taking into account negative indices */
    if ((*index < -max_item) || (*index >= max_item)) {
        /* Try to be as clear as possible about what went wrong. */
        if (axis >= 0) {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for axis %d with size %"NPY_INTP_FMT,
                         *index, axis, max_item);
        } else {
            PyErr_Format(PyExc_IndexError,
                         "index %"NPY_INTP_FMT" is out of bounds "
                         "for size %"NPY_INTP_FMT,
                         *index, max_item);
        }
        return -1;
    }
    /* adjust negative indices */
    if (*index < 0) {
        *index += max_item;
    }
    return 0;
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
    if (check_and_adjust_index(&i, dim0, 0) < 0)
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

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap)
{
    unsigned int i, aligned = 1;
    const unsigned int alignment = PyArray_DESCR(ap)->alignment;

    /* The special casing for STRING and VOID types was removed
     * in accordance with http://projects.scipy.org/numpy/ticket/1227
     * It used to be that IsAligned always returned True for these
     * types, which is indeed the case when they are created using
     * PyArray_DescrConverter(), but not necessarily when using
     * PyArray_DescrAlignConverter(). */

    if (alignment == 1) {
        return 1;
    }
    aligned = npy_is_aligned(PyArray_DATA(ap), alignment);

    for (i = 0; i < PyArray_NDIM(ap); i++) {
#if NPY_RELAXED_STRIDES_CHECKING
        if (PyArray_DIM(ap, i) > 1) {
            /* if shape[i] == 1, the stride is never used */
            aligned &= npy_is_aligned((void*)PyArray_STRIDES(ap)[i],
                                      alignment);
        }
        else if (PyArray_DIM(ap, i) == 0) {
            /* an array with zero elements is always aligned */
            return 1;
        }
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        aligned &= npy_is_aligned((void*)PyArray_STRIDES(ap)[i], alignment);
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
    }
    return aligned != 0;
}

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base=PyArray_BASE(ap);
    void *dummy;
    Py_ssize_t n;

    /* If we own our own data, then no-problem */
    if ((base == NULL) || (PyArray_FLAGS(ap) & NPY_ARRAY_OWNDATA)) {
        return NPY_TRUE;
    }
    /*
     * Get to the final base object
     * If it is a writeable array, then return TRUE
     * If we can find an array object
     * or a writeable buffer object as the final base object
     * or a string object (for pickling support memory savings).
     * - this last could be removed if a proper pickleable
     * buffer was added to Python.
     *
     * MW: I think it would better to disallow switching from READONLY
     *     to WRITEABLE like this...
     */

    while(PyArray_Check(base)) {
        if (PyArray_CHKFLAGS((PyArrayObject *)base, NPY_ARRAY_OWNDATA)) {
            return (npy_bool) (PyArray_ISWRITEABLE((PyArrayObject *)base));
        }
        base = PyArray_BASE((PyArrayObject *)base);
    }

    /*
     * here so pickle support works seamlessly
     * and unpickled array can be set and reset writeable
     * -- could be abused --
     */
    if (PyString_Check(base)) {
        return NPY_TRUE;
    }
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        return NPY_FALSE;
    }
    return NPY_TRUE;
}


/* Gets a half-open range [start, end) of offsets from the data pointer */
NPY_NO_EXPORT void
offset_bounds_from_strides(const int itemsize, const int nd,
                           const npy_intp *dims, const npy_intp *strides,
                           npy_intp *lower_offset, npy_intp *upper_offset) {
    npy_intp max_axis_offset;
    npy_intp lower = 0;
    npy_intp upper = 0;
    int i;

    for (i = 0; i < nd; i++) {
        if (dims[i] == 0) {
            /* If the array size is zero, return an empty range */
            *lower_offset = 0;
            *upper_offset = 0;
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        max_axis_offset = strides[i] * (dims[i] - 1);
        if (max_axis_offset > 0) {
            upper += max_axis_offset;
        }
        else {
            lower += max_axis_offset;
        }
    }
    /* Return a half-open range */
    upper += itemsize;
    *lower_offset = lower;
    *upper_offset = upper;
}
