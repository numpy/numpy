/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <errno.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "numpy/npy_math.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_import.h"


#include "_datetime.h"
#include "common.h"
#include "conversion_utils.h"  /* for PyArray_TypestrConvert */
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "descriptor.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"  // for thread unsafe state access
#include "alloc.h"
#include "assert.h"
#include "npy_buffer.h"
#include "dtypemeta.h"
#include "stringdtype/dtype.h"
#include "array_coercion.h"

#ifndef PyDictProxy_Check
#define PyDictProxy_Check(obj) (Py_TYPE(obj) == &PyDictProxy_Type)
#endif

static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

static PyArray_Descr *
_try_convert_from_inherit_tuple(PyArray_Descr *type, PyObject *newobj);

static PyArray_Descr *
_convert_from_any(PyObject *obj, int align);

/*
 * This function creates a dtype object when the object is a ctypes subclass.
 *
 * Returns `Py_NotImplemented` if the type is not a ctypes subclass.
 */
static PyArray_Descr *
_try_convert_from_ctypes_type(PyTypeObject *type)
{
    PyObject *_numpy_dtype_ctypes;
    PyObject *res;

    if (!npy_ctypes_check(type)) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }

    /* Call the python function of the same name. */
    _numpy_dtype_ctypes = PyImport_ImportModule("numpy._core._dtype_ctypes");
    if (_numpy_dtype_ctypes == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype_ctypes, "dtype_from_ctypes_type", "O", (PyObject *)type);
    Py_DECREF(_numpy_dtype_ctypes);
    if (res == NULL) {
        return NULL;
    }

    /*
     * sanity check that dtype_from_ctypes_type returned the right type,
     * since getting it wrong would give segfaults.
     */
    if (!PyObject_TypeCheck(res, &PyArrayDescr_Type)) {
        Py_DECREF(res);
        PyErr_BadInternalCall();
        return NULL;
    }

    return (PyArray_Descr *)res;
}

/*
 * This function creates a dtype object when the object has a "dtype" attribute,
 * and it can be converted to a dtype object.
 *
 * Returns `Py_NotImplemented` if this is not possible.
 * Currently the only failure mode for a NULL return is a RecursionError.
 */
static PyArray_Descr *
_try_convert_from_dtype_attr(PyObject *obj)
{
    /* For arbitrary objects that have a "dtype" attribute */
    PyObject *dtypedescr = PyObject_GetAttrString(obj, "dtype");
    if (dtypedescr == NULL) {
        /*
         * This can be reached due to recursion limit being hit while fetching
         * the attribute (tested for py3.7). This removes the custom message.
         */
        goto fail;
    }

    if (PyArray_DescrCheck(dtypedescr)) {
        /* The dtype attribute is already a valid descriptor */
        return (PyArray_Descr *)dtypedescr;
    }

    if (Py_EnterRecursiveCall(
            " while trying to convert the given data type from its "
            "`.dtype` attribute.") != 0) {
        Py_DECREF(dtypedescr);
        return NULL;
    }

    PyArray_Descr *newdescr = _convert_from_any(dtypedescr, 0);
    Py_DECREF(dtypedescr);
    Py_LeaveRecursiveCall();
    if (newdescr == NULL) {
        goto fail;
    }

    Py_DECREF(newdescr);
    PyErr_SetString(PyExc_ValueError, "dtype attribute is not a valid dtype instance");
    return NULL;

  fail:
    /* Ignore all but recursion errors, to give ctypes a full try. */
    if (!PyErr_ExceptionMatches(PyExc_RecursionError)) {
        PyErr_Clear();
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }
    return NULL;
}

/* Expose to another file with a prefixed name */
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_try_convert_from_dtype_attr(PyObject *obj)
{
    return _try_convert_from_dtype_attr(obj);
}

/*
 * Sets the global typeDict object, which is a dictionary mapping
 * dtype names to numpy scalar types.
 */
NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O:set_typeDict", &dict)) {
        return NULL;
    }
    /* Decrement old reference (if any)*/
    Py_XDECREF(typeDict);
    typeDict = dict;
    /* Create an internal reference to it */
    Py_INCREF(dict);
    Py_RETURN_NONE;
}

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static int
_check_for_commastring(const char *type, Py_ssize_t len)
{
    Py_ssize_t i;
    int sqbracket;

    /* Check for ints at start of string */
    if ((type[0] >= '0'
                && type[0] <= '9')
            || ((len > 1)
                && _chk_byteorder(type[0])
                && (type[1] >= '0'
                && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1)
                && (type[0] == '('
                && type[1] == ')'))
            || ((len > 3)
                && _chk_byteorder(type[0])
                && (type[1] == '('
                && type[2] == ')'))) {
        return 1;
    }
    /*
     * Check for presence of commas outside square [] brackets. This
     * allows commas inside of [], for parameterized dtypes to use.
     */
    sqbracket = 0;
    for (i = 0; i < len; i++) {
        switch (type[i]) {
            case ',':
                if (sqbracket == 0) {
                    return 1;
                }
                break;
            case '[':
                ++sqbracket;
                break;
            case ']':
                --sqbracket;
                break;
        }
    }
    return 0;
}

#undef _chk_byteorder

static int
is_datetime_typestr(char const *type, Py_ssize_t len)
{
    if (len < 2) {
        return 0;
    }
    if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm')) {
        return 1;
    }
    if (len < 10) {
        return 0;
    }
    if (strncmp(type, "datetime64", 10) == 0) {
        return 1;
    }
    if (len < 11) {
        return 0;
    }
    if (strncmp(type, "timedelta64", 11) == 0) {
        return 1;
    }
    return 0;
}

static PyArray_Descr *
_convert_from_tuple(PyObject *obj, int align)
{
    if (PyTuple_GET_SIZE(obj) != 2) {
        PyErr_Format(PyExc_TypeError,
	        "Tuple must have size 2, but has size %zd",
	        PyTuple_GET_SIZE(obj));
        return NULL;
    }
    PyArray_Descr *type = _convert_from_any(PyTuple_GET_ITEM(obj, 0), align);
    if (type == NULL) {
        return NULL;
    }
    PyObject *val = PyTuple_GET_ITEM(obj,1);
    /* try to interpret next item as a type */
    PyArray_Descr *res = _try_convert_from_inherit_tuple(type, val);
    if ((PyObject *)res != Py_NotImplemented) {
        Py_DECREF(type);
        return res;
    }
    Py_DECREF(res);
    /*
     * We get here if _try_convert_from_inherit_tuple failed without crashing
     */
    if (PyDataType_ISUNSIZED(type)) {
        /* interpret next item as a typesize */
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));
        if (type->type_num == NPY_UNICODE) {
            if (itemsize > NPY_MAX_INT / 4) {
                itemsize = -1;
            }
            else {
                itemsize *= 4;
            }
        }
        if (itemsize < 0) {
            /* Error may or may not be set by PyIntAsInt. */
            PyErr_SetString(PyExc_ValueError,
                    "invalid itemsize in generic type tuple");
            Py_DECREF(type);
            return NULL;
        }
        PyArray_DESCR_REPLACE(type);
        if (type == NULL) {
            return NULL;
        }

        type->elsize = itemsize;
        return type;
    }
    else if (type->metadata && (PyDict_Check(val) || PyDictProxy_Check(val))) {
        /* Assume it's a metadata dictionary */
        if (PyDict_Merge(type->metadata, val, 0) == -1) {
            Py_DECREF(type);
            return NULL;
        }
        return type;
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to NPY_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        if (!(PyArray_IntpConverter(val, &shape)) || (shape.len > NPY_MAXDIMS)) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /* if (type, ()) was given it is equivalent to type... */
        if (shape.len == 0 && PyTuple_Check(val)) {
            npy_free_cache_dim_obj(shape);
            return type;
        }

        /* validate and set shape */
        for (int i=0; i < shape.len; i++) {
            if (shape.ptr[i] < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension smaller then zero.");
                goto fail;
            }
            if (shape.ptr[i] > NPY_MAX_INT) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension does not fit into a C int.");
                goto fail;
            }
        }
        npy_intp items = PyArray_OverflowMultiplyList(shape.ptr, shape.len);
        int overflowed;
        int nbytes;
        if (items < 0 || items > NPY_MAX_INT) {
            overflowed = 1;
        }
        else {
            overflowed = npy_mul_with_overflow_int(
                &nbytes, type->elsize, (int) items);
        }
        if (overflowed) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid shape in fixed-type tuple: dtype size in "
                            "bytes must fit into a C int.");
            goto fail;
        }
        _PyArray_LegacyDescr *newdescr = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
        if (newdescr == NULL) {
            goto fail;
        }
        newdescr->elsize = nbytes;
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            Py_DECREF(newdescr);
            PyErr_NoMemory();
            goto fail;
        }
        newdescr->flags = type->flags;
        newdescr->alignment = type->alignment;
        newdescr->subarray->base = type;
        type = NULL;
        Py_XDECREF(newdescr->fields);
        Py_XDECREF(newdescr->names);
        newdescr->fields = NULL;
        newdescr->names = NULL;

        /*
         * Create a new subarray->shape tuple (it can be an arbitrary
         * sequence of integer like objects, neither of which is safe.
         */
        newdescr->subarray->shape = PyTuple_New(shape.len);
        if (newdescr->subarray->shape == NULL) {
            Py_DECREF(newdescr);
            goto fail;
        }
        for (int i=0; i < shape.len; i++) {
            PyTuple_SET_ITEM(newdescr->subarray->shape, i,
                             PyLong_FromLong((long)shape.ptr[i]));

            if (PyTuple_GET_ITEM(newdescr->subarray->shape, i) == NULL) {
                Py_DECREF(newdescr);
                goto fail;
            }
        }

        npy_free_cache_dim_obj(shape);
        return (PyArray_Descr *)newdescr;

    fail:
        Py_XDECREF(type);
        npy_free_cache_dim_obj(shape);
        return NULL;
    }
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 *
 * field-name can be a string or a 2-tuple
 * data-type can now be a list, string, or 2-tuple
 *          (string, metadata dictionary)
 */
static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    int n = PyList_GET_SIZE(obj);
    PyObject *nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int maxalign = 1;
    int totalsize = 0;
    PyObject *fields = PyDict_New();
    if (!fields) {
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(obj, i);
        if (!PyTuple_Check(item) || (PyTuple_GET_SIZE(item) < 2)) {
            PyErr_Format(PyExc_TypeError,
			 "Field elements must be 2- or 3-tuples, got '%R'",
			 item);
            goto fail;
        }
        PyObject *name = PyTuple_GET_ITEM(item, 0);
        PyObject *title;
        if (PyUnicode_Check(name)) {
            title = NULL;
        }
        else if (PyTuple_Check(name)) {
            if (PyTuple_GET_SIZE(name) != 2) {
                PyErr_Format(PyExc_TypeError,
				"If a tuple, the first element of a field tuple must have "
				"two elements, not %zd",
			       	PyTuple_GET_SIZE(name));
                goto fail;
            }
            title = PyTuple_GET_ITEM(name, 0);
            name = PyTuple_GET_ITEM(name, 1);
            if (!PyUnicode_Check(name)) {
                PyErr_SetString(PyExc_TypeError, "Field name must be a str");
                goto fail;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
			            "First element of field tuple is "
			            "neither a tuple nor str");
            goto fail;
        }

        /* Insert name into nameslist */
        Py_INCREF(name);

        if (PyUnicode_GetLength(name) == 0) {
            Py_DECREF(name);
            if (title == NULL) {
                name = PyUnicode_FromFormat("f%d", i);
                if (name == NULL) {
                    goto fail;
                }
            }
            /* On Py3, allow only non-empty Unicode strings as field names */
            else if (PyUnicode_Check(title) && PyUnicode_GetLength(title) > 0) {
                name = title;
                Py_INCREF(name);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "Field titles must be non-empty strings");
                goto fail;
            }
        }
        PyTuple_SET_ITEM(nameslist, i, name);

        /* Process rest */
        PyArray_Descr *conv;
        if (PyTuple_GET_SIZE(item) == 2) {
            conv = _convert_from_any(PyTuple_GET_ITEM(item, 1), align);
            if (conv == NULL) {
                goto fail;
            }
        }
        else if (PyTuple_GET_SIZE(item) == 3) {
            PyObject *newobj = PyTuple_GetSlice(item, 1, 3);
            conv = _convert_from_any(newobj, align);
            Py_DECREF(newobj);
            if (conv == NULL) {
                goto fail;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError,
                    "Field elements must be tuples with at most 3 elements, got '%R'", item);
            goto fail;
        }
        if (PyObject_IsInstance((PyObject *)conv, (PyObject *)&PyArray_StringDType)) {
            PyErr_Format(PyExc_TypeError,
                         "StringDType is not currently supported for structured dtype fields.");
            goto fail;
        }
        if ((PyDict_GetItemWithError(fields, name) != NULL)
             || (title
                 && PyUnicode_Check(title)
                 && (PyDict_GetItemWithError(fields, title) != NULL))) {
            PyErr_Format(PyExc_ValueError,
                    "field %R occurs more than once", name);
            Py_DECREF(conv);
            goto fail;
        }
        else if (PyErr_Occurred()) {
            /* Dict lookup crashed */
            Py_DECREF(conv);
            goto fail;
        }
        dtypeflags |= (conv->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align = conv->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }
        PyObject *tup = PyTuple_New((title == NULL ? 2 : 3));
        if (tup == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        PyTuple_SET_ITEM(tup, 1, PyLong_FromLong((long) totalsize));

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         * and if it is not the same as the name.
         */
        if (title != NULL) {
            Py_INCREF(title);
            PyTuple_SET_ITEM(tup, 2, title);
            if (PyDict_SetItem(fields, name, tup) < 0) {
                goto fail;
            }
            if (PyUnicode_Check(title)) {
                PyObject *existing = PyDict_GetItemWithError(fields, title);
                if (existing == NULL && PyErr_Occurred()) {
                    goto fail;
                }
                if (existing != NULL) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    Py_DECREF(tup);
                    goto fail;
                }
                if (PyDict_SetItem(fields, title, tup) < 0) {
                    goto fail;
                }
            }
        }
        else {
            if (PyDict_SetItem(fields, name, tup) < 0) {
                goto fail;
            }
        }

        totalsize += conv->elsize;
        Py_DECREF(tup);
    }

    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }

    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->flags = dtypeflags;

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    return (PyArray_Descr *)new;

 fail:
    Py_DECREF(fields);
    Py_DECREF(nameslist);
    return NULL;

}

/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
    int n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    PyObject *last_item = PyList_GET_ITEM(obj, n-1);
    if (PyUnicode_Check(last_item)) {
        Py_ssize_t s = PySequence_Size(last_item);
        if (s < 0) {
            return NULL;
        }
        if (s == 0) {
            n = n - 1;
        }
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Expected at least one field name");
        return NULL;
    }
    PyObject *nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    PyObject *fields = PyDict_New();
    if (!fields) {
        Py_DECREF(nameslist);
        return NULL;
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int maxalign = 1;
    int totalsize = 0;
    for (int i = 0; i < n; i++) {
        PyArray_Descr *conv = _convert_from_any(
                PyList_GET_ITEM(obj, i), align);
        if (conv == NULL) {
            goto fail;
        }
        dtypeflags |= (conv->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align = conv->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }
        PyObject *size_obj = PyLong_FromLong((long) totalsize);
        if (!size_obj) {
            Py_DECREF(conv);
            goto fail;
        }
        PyObject *tup = PyTuple_New(2);
        if (!tup) {
            Py_DECREF(size_obj);
            Py_DECREF(conv);
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        PyTuple_SET_ITEM(tup, 1, size_obj);
        PyObject *key = PyUnicode_FromFormat("f%d", i);
        if (!key) {
            Py_DECREF(tup);
            goto fail;
        }
        /* steals a reference to key */
        PyTuple_SET_ITEM(nameslist, i, key);
        int ret = PyDict_SetItem(fields, key, tup);
        Py_DECREF(tup);
        if (ret < 0) {
            goto fail;
        }
        totalsize += conv->elsize;
    }
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    new->fields = fields;
    new->names = nameslist;
    new->flags = dtypeflags;
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    return (PyArray_Descr *)new;

 fail:
    Py_DECREF(nameslist);
    Py_DECREF(fields);
    return NULL;
}


/*
 * comma-separated string
 * this is the format developed by the numarray records module and implemented
 * by the format parser in that module this is an alternative implementation
 * found in the _internal.py file patterned after that one -- the approach is
 * to convert strings like "i,i" or "i1,2i,(3,4)f4" to a list of simple dtypes or
 * (dtype, repeat) tuples. A single tuple is expected for strings such as "(2,)i".
 *
 * TODO: Calling Python from C like this in critical-path code is not
 *       a good idea. This should all be converted to C code.
 */
static PyArray_Descr *
_convert_from_commastring(PyObject *obj, int align)
{
    PyObject *parsed;
    PyArray_Descr *res;
    assert(PyUnicode_Check(obj));
    if (npy_cache_import_runtime(
            "numpy._core._internal", "_commastring",
            &npy_runtime_imports._commastring) == -1) {
        return NULL;
    }
    parsed = PyObject_CallOneArg(npy_runtime_imports._commastring, obj);
    if (parsed == NULL) {
        return NULL;
    }
    if ((PyTuple_Check(parsed) && PyTuple_GET_SIZE(parsed) == 2)) {
        res = _convert_from_any(parsed, align);
    }
    else if (PyList_Check(parsed) && PyList_GET_SIZE(parsed) >= 1) {
        res = _convert_from_list(parsed, align);
    }
    else {
        PyErr_SetString(PyExc_RuntimeError,
                "_commastring should return a tuple with len == 2, or "
                "a list with len >= 1");
        res = NULL;
    }
    Py_DECREF(parsed);
    return res;
}

static int
_is_tuple_of_integers(PyObject *obj)
{
    int i;

    if (!PyTuple_Check(obj)) {
        return 0;
    }
    for (i = 0; i < PyTuple_GET_SIZE(obj); i++) {
        if (!PyArray_IsIntegerScalar(PyTuple_GET_ITEM(obj, i))) {
            return 0;
        }
    }
    return 1;
}

/*
 * helper function for _try_convert_from_inherit_tuple to disallow dtypes of the form
 * (old_dtype, new_dtype) where either of the dtypes contains python
 * objects - these dtypes are not useful and can be a source of segfaults,
 * when an attempt is made to interpret a python object as a different dtype
 * or vice versa
 * an exception is made for dtypes of the form ('O', [('name', 'O')]), which
 * people have been using to add a field to an object array without fields
 */
static int
_validate_union_object_dtype(_PyArray_LegacyDescr *new, _PyArray_LegacyDescr *conv)
{
    PyObject *name, *tup;
    PyArray_Descr *dtype;

    if (!PyDataType_REFCHK((PyArray_Descr *)new)
            && !PyDataType_REFCHK((PyArray_Descr *)conv)) {
        return 0;
    }
    if (PyDataType_HASFIELDS(new) || new->kind != 'O') {
        goto fail;
    }
    if (!PyDataType_HASFIELDS(conv) || PyTuple_GET_SIZE(conv->names) != 1) {
        goto fail;
    }
    name = PyTuple_GET_ITEM(conv->names, 0);
    if (name == NULL) {
        return -1;
    }
    tup = PyDict_GetItemWithError(conv->fields, name);
    if (tup == NULL) {
        if (!PyErr_Occurred()) {
            /* fields was missing the name it claimed to contain */
            PyErr_BadInternalCall();
        }
        return -1;
    }
    dtype = (PyArray_Descr *)PyTuple_GET_ITEM(tup, 0);
    if (dtype == NULL) {
        return -1;
    }
    if (dtype->kind != 'O') {
        goto fail;
    }
    return 0;

fail:
    PyErr_SetString(PyExc_ValueError,
            "dtypes of the form (old_dtype, new_dtype) containing the object "
            "dtype are not supported");
    return -1;
}

/*
 * A tuple type would be either (generic typeobject, typesize)
 * or (fixed-length data-type, shape)
 *
 * or (inheriting data-type, new-data-type)
 * The new data-type must have the same itemsize as the inheriting data-type
 * unless the latter is 0
 *
 * Thus (int32, {'real':(int16,0),'imag',(int16,2)})
 *
 * is one way to specify a descriptor that will give
 * a['real'] and a['imag'] to an int32 array.
 *
 * leave type reference alone
 *
 * Returns `Py_NotImplemented` if the second tuple item is not
 * appropriate.
 */
static PyArray_Descr *
_try_convert_from_inherit_tuple(PyArray_Descr *type, PyObject *newobj)
{
    if (PyArray_IsScalar(newobj, Integer) || _is_tuple_of_integers(newobj)) {
        /* It's a subarray or flexible type instead */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }
    _PyArray_LegacyDescr *conv = (_PyArray_LegacyDescr *)_convert_from_any(newobj, 0);
    if (conv == NULL) {
        /* Let someone else try to convert this */
        PyErr_Clear();
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }
    if (!PyDataType_ISLEGACY(type) || !PyDataType_ISLEGACY(conv)) {
        /* 
         * This specification should probably be never supported, but
         * certainly not for new-style DTypes.
         */
        Py_DECREF(conv);
        Py_INCREF(Py_NotImplemented);
        return (PyArray_Descr *)Py_NotImplemented;
    }
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNew(type);
    if (new == NULL) {
        goto fail;
    }
    if (PyDataType_ISUNSIZED(new)) {
        new->elsize = conv->elsize;
    }
    else if (new->elsize != conv->elsize) {
        PyErr_SetString(PyExc_ValueError,
                "mismatch in size of old and new data-descriptor");
        Py_DECREF(new);
        goto fail;
    }
    else if (_validate_union_object_dtype(new, conv) < 0) {
        Py_DECREF(new);
        goto fail;
    }

    if (PyDataType_HASFIELDS(conv)) {
        Py_XDECREF(new->fields);
        new->fields = conv->fields;
        Py_XINCREF(new->fields);

        Py_XDECREF(new->names);
        new->names = conv->names;
        Py_XINCREF(new->names);
    }
    if (conv->metadata != NULL) {
        Py_XDECREF(new->metadata);
        new->metadata = conv->metadata;
        Py_XINCREF(new->metadata);
    }
    /*
     * Certain flags must be inherited from the fields.  This is needed
     * only for void dtypes (or subclasses of it such as a record dtype).
     * For other dtypes, the field part will only be used for direct field
     * access and thus flag inheritance should not be necessary.
     * (We only allow object fields if the dtype is object as well.)
     * This ensures copying over of the NPY_FROM_FIELDS "inherited" flags.
     */
    if (new->type_num == NPY_VOID) {
        new->flags = conv->flags;
    }
    Py_DECREF(conv);
    return (PyArray_Descr *)new;

 fail:
    Py_DECREF(conv);
    return NULL;
}

/*
 * Validates that any field of the structured array 'dtype' which has
 * the NPY_ITEM_HASOBJECT flag set does not overlap with another field.
 *
 * This algorithm is worst case O(n^2). It could be done with a sort
 * and sweep algorithm, but the structured dtype representation is
 * rather ugly right now, so writing something better can wait until
 * that representation is made sane.
 *
 * Returns 0 on success, -1 if an exception is raised.
 */
static int
_validate_object_field_overlap(_PyArray_LegacyDescr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, j, names_size;
    PyArray_Descr *fld_dtype, *fld2_dtype;
    int fld_offset, fld2_offset;

    /* Get some properties from the dtype */
    names = dtype->names;
    names_size = PyTuple_GET_SIZE(names);
    fields = dtype->fields;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            return -1;
        }
        tup = PyDict_GetItemWithError(fields, key);
        if (tup == NULL) {
            if (!PyErr_Occurred()) {
                /* fields was missing the name it claimed to contain */
                PyErr_BadInternalCall();
            }
            return -1;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            return -1;
        }

        /* If this field has objects, check for overlaps */
        if (PyDataType_REFCHK(fld_dtype)) {
            for (j = 0; j < names_size; ++j) {
                if (i != j) {
                    key = PyTuple_GET_ITEM(names, j);
                    if (key == NULL) {
                        return -1;
                    }
                    tup = PyDict_GetItemWithError(fields, key);
                    if (tup == NULL) {
                        if (!PyErr_Occurred()) {
                            /* fields was missing the name it claimed to contain */
                            PyErr_BadInternalCall();
                        }
                        return -1;
                    }
                    if (!PyArg_ParseTuple(tup, "Oi|O", &fld2_dtype,
                                                &fld2_offset, &title)) {
                        return -1;
                    }
                    /* Raise an exception if it overlaps */
                    if (fld_offset < fld2_offset + fld2_dtype->elsize &&
                                fld2_offset < fld_offset + fld_dtype->elsize) {
                        PyErr_SetString(PyExc_TypeError,
                                "Cannot create a NumPy dtype with overlapping "
                                "object fields");
                        return -1;
                    }
                }
            }
        }
    }

    /* It passed all the overlap tests */
    return 0;
}

/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
 *
 * can also have an additional key called "metadata" which can be any dictionary
 *
 * "names" --- field names
 * "formats" --- the data-type descriptors for the field.
 *
 * Optional:
 *
 * "offsets" --- integers indicating the offset into the
 * record of the start of the field.
 * if not given, then "consecutive offsets"
 * will be assumed and placed in the dictionary.
 *
 * "titles" --- Allows the use of an additional key
 * for the fields dictionary.(if these are strings
 * or unicode objects) or
 * this can also be meta-data to
 * be passed around with the field description.
 *
 * Attribute-lookup-based field names merely has to query the fields
 * dictionary of the data-descriptor.  Any result present can be used
 * to return the correct field.
 *
 * So, the notion of what is a name and what is a title is really quite
 * arbitrary.
 *
 * What does distinguish a title, however, is that if it is not None,
 * it will be placed at the end of the tuple inserted into the
 * fields dictionary.and can therefore be used to carry meta-data around.
 *
 * If the dictionary does not have "names" and "formats" entries,
 * then it will be checked for conformity and used directly.
 */
static PyArray_Descr *
_convert_from_field_dict(PyObject *obj, int align)
{
    PyObject *_numpy_internal;
    PyArray_Descr *res;

    _numpy_internal = PyImport_ImportModule("numpy._core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = (PyArray_Descr *)PyObject_CallMethod(_numpy_internal,
            "_usefields", "Oi", obj, align);
    Py_DECREF(_numpy_internal);
    return res;
}

/*
 * Creates a struct dtype object from a Python dictionary.
 */
static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
    PyObject *fields = PyDict_New();
    if (fields == NULL) {
        return (PyArray_Descr *)PyErr_NoMemory();
    }
    /*
     * Use PyMapping_GetItemString to support dictproxy objects as well.
     */
    PyObject *names = PyMapping_GetItemString(obj, "names");
    if (names == NULL) {
        Py_DECREF(fields);
        /* XXX should check this is a KeyError */
        PyErr_Clear();
        return _convert_from_field_dict(obj, align);
    }
    PyObject *descrs = PyMapping_GetItemString(obj, "formats");
    if (descrs == NULL) {
        Py_DECREF(fields);
        /* XXX should check this is a KeyError */
        PyErr_Clear();
        Py_DECREF(names);
        return _convert_from_field_dict(obj, align);
    }
    int n = PyObject_Length(names);
    PyObject *offsets = PyMapping_GetItemString(obj, "offsets");
    if (!offsets) {
        PyErr_Clear();
    }
    PyObject *titles = PyMapping_GetItemString(obj, "titles");
    if (!titles) {
        PyErr_Clear();
    }

    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                "'names', 'formats', 'offsets', and 'titles' dict "
                "entries must have the same length");
        goto fail;
    }

    /*
     * If a property 'aligned' is in the dict, it overrides the align flag
     * to be True if it not already true.
     */
    PyObject *tmp = PyMapping_GetItemString(obj, "aligned");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        if (tmp == Py_True) {
            align = 1;
        }
        else if (tmp != Py_False) {
            Py_DECREF(tmp);
            PyErr_SetString(PyExc_ValueError,
                    "NumPy dtype descriptor includes 'aligned' entry, "
                    "but its value is neither True nor False");
            goto fail;
        }
        Py_DECREF(tmp);
    }

    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int totalsize = 0;
    int maxalign = 1;
    int has_out_of_order_fields = 0;
    for (int i = 0; i < n; i++) {
        /* Build item to insert (descr, offset, [title])*/
        int len = 2;
        PyObject *title = NULL;
        PyObject *ind = PyLong_FromLong(i);
        if (titles) {
            title=PyObject_GetItem(titles, ind);
            if (title && title != Py_None) {
                len = 3;
            }
            else {
                Py_XDECREF(title);
            }
            PyErr_Clear();
        }
        PyObject *tup = PyTuple_New(len);
        PyObject *descr = PyObject_GetItem(descrs, ind);
        if (!descr) {
            Py_DECREF(tup);
            Py_DECREF(ind);
            goto fail;
        }
        PyArray_Descr *newdescr = _convert_from_any(descr, align);
        Py_DECREF(descr);
        if (newdescr == NULL) {
            Py_DECREF(tup);
            Py_DECREF(ind);
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)newdescr);
        int _align = 1;
        if (align) {
            _align = newdescr->alignment;
            maxalign = PyArray_MAX(maxalign,_align);
        }
        if (offsets) {
            PyObject *off = PyObject_GetItem(offsets, ind);
            if (!off) {
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }
            long offset = PyArray_PyIntAsInt(off);
            if (error_converting(offset)) {
                Py_DECREF(off);
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }
            Py_DECREF(off);
            if (offset < 0) {
                PyErr_Format(PyExc_ValueError, "offset %ld cannot be negative",
                             offset);
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }

            PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(offset));
            /* Flag whether the fields are specified out of order */
            if (offset < totalsize) {
                has_out_of_order_fields = 1;
            }
            /* If align=True, enforce field alignment */
            if (align && offset % newdescr->alignment != 0) {
                PyErr_Format(PyExc_ValueError,
                        "offset %ld for NumPy dtype with fields is "
                        "not divisible by the field alignment %d "
                        "with align=True",
                        offset, newdescr->alignment);
                Py_DECREF(ind);
                Py_DECREF(tup);
                goto fail;
            }
            else if (offset + newdescr->elsize > totalsize) {
                totalsize = offset + newdescr->elsize;
            }
        }
        else {
            if (align && _align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(totalsize));
            totalsize += newdescr->elsize;
        }
        if (len == 3) {
            PyTuple_SET_ITEM(tup, 2, title);
        }
        PyObject *name = PyObject_GetItem(names, ind);
        Py_DECREF(ind);
        if (!name) {
            Py_DECREF(tup);
            goto fail;
        }
        if (!PyUnicode_Check(name)) {
            PyErr_SetString(PyExc_ValueError,
                    "field names must be strings");
            Py_DECREF(tup);
            goto fail;
        }

        /* Insert into dictionary */
        if (PyDict_GetItemWithError(fields, name) != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "name already used as a name or title");
            Py_DECREF(tup);
            goto fail;
        }
        else if (PyErr_Occurred()) {
            /* MemoryError during dict lookup */
            Py_DECREF(tup);
            goto fail;
        }
        int ret = PyDict_SetItem(fields, name, tup);
        Py_DECREF(name);
        if (ret < 0) {
            Py_DECREF(tup);
            goto fail;
        }
        if (len == 3) {
            if (PyUnicode_Check(title)) {
                if (PyDict_GetItemWithError(fields, title) != NULL) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    Py_DECREF(tup);
                    goto fail;
                }
                else if (PyErr_Occurred()) {
                    /* MemoryError during dict lookup */
                    goto fail;
                }
                if (PyDict_SetItem(fields, title, tup) < 0) {
                    Py_DECREF(tup);
                    goto fail;
                }
            }
        }
        Py_DECREF(tup);
        dtypeflags |= (newdescr->flags & NPY_FROM_FIELDS);
    }

    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    if (!PyTuple_Check(names)) {
        Py_SETREF(names, PySequence_Tuple(names));
        if (names == NULL) {
            Py_DECREF(new);
            goto fail;
        }
    }
    new->names = names;
    new->fields = fields;
    new->flags = dtypeflags;
    /* new takes responsibility for DECREFing names, fields */
    names = NULL;
    fields = NULL;

    /*
     * If the fields weren't in order, and there was an OBJECT type,
     * need to verify that no OBJECT types overlap with something else.
     */
    if (has_out_of_order_fields && PyDataType_REFCHK((PyArray_Descr *)new)) {
        if (_validate_object_field_overlap(new) < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
    }

    /* Override the itemsize if provided */
    tmp = PyMapping_GetItemString(obj, "itemsize");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        int itemsize = (int)PyArray_PyIntAsInt(tmp);
        Py_DECREF(tmp);
        if (error_converting(itemsize)) {
            Py_DECREF(new);
            goto fail;
        }
        /* Make sure the itemsize isn't made too small */
        if (itemsize < new->elsize) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires %d bytes, "
                    "cannot override to smaller itemsize of %d",
                    new->elsize, itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* If align is set, make sure the alignment divides into the size */
        if (align && new->alignment > 0 && itemsize % new->alignment != 0) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires alignment of %d bytes, "
                    "which is not divisible into the specified itemsize %d",
                    new->alignment, itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* Set the itemsize */
        new->elsize = itemsize;
    }

    /* Add the metadata if provided */
    PyObject *metadata = PyMapping_GetItemString(obj, "metadata");

    if (metadata == NULL) {
        PyErr_Clear();
    }
    else if (new->metadata == NULL) {
        new->metadata = metadata;
    }
    else {
        int ret = PyDict_Merge(new->metadata, metadata, 0);
        Py_DECREF(metadata);
        if (ret < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    return (PyArray_Descr *)new;

 fail:
    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    return NULL;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    PyArray_Descr *old;
    PyArray_Descr *new;

    old = PyArray_DescrFromType(type_num);
    if (old == NULL) {
        return NULL;
    }
    new = PyArray_DescrNew(old);
    Py_DECREF(old);
    return new;
}

/*NUMPY_API
 * Get typenum from an object -- None goes to NULL
 */
NPY_NO_EXPORT int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}


/**
 * Check the descriptor is a legacy "flexible" DType instance, this is
 * an instance which is (normally) not attached to an array, such as a string
 * of length 0 or a datetime with no unit.
 * These should be largely deprecated, and represent only the DType class
 * for most `dtype` parameters.
 *
 * TODO: This function should eventually receive a deprecation warning and
 *       be removed.
 *
 * @param descr descriptor to be checked
 * @param DType pointer to the DType of the descriptor
 * @return 1 if this is not a concrete dtype instance 0 otherwise
 */
static int
descr_is_legacy_parametric_instance(PyArray_Descr *descr,
                                    PyArray_DTypeMeta *DType)
{
    if (!NPY_DT_is_legacy(DType)) {
        return 0;
    }

    if (PyDataType_ISUNSIZED(descr)) {
        return 1;
    }
    /* Flexible descr with generic time unit (which can be adapted) */
    if (PyDataType_ISDATETIME(descr)) {
        PyArray_DatetimeMetaData *meta;
        meta = get_datetime_metadata_from_dtype(descr);
        if (meta->base == NPY_FR_GENERIC) {
            return 1;
        }
    }
    return 0;
}


/**
 * Given a descriptor (dtype instance), handles conversion of legacy flexible
 * "unsized" descriptors to their DType.  It returns the DType and descriptor
 * both results can be NULL (if the input is).  But it always sets the DType
 * when a descriptor is set.
 *
 * @param dtype Input descriptor to be converted
 * @param out_descr Output descriptor
 * @param out_DType DType of the output descriptor
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_ExtractDTypeAndDescriptor(PyArray_Descr *dtype,
        PyArray_Descr **out_descr, PyArray_DTypeMeta **out_DType)
{
    *out_DType = NULL;
    *out_descr = NULL;

    if (dtype != NULL) {
        *out_DType = NPY_DTYPE(dtype);
        Py_INCREF(*out_DType);
        if (!descr_is_legacy_parametric_instance((PyArray_Descr *)dtype,
                                                    *out_DType)) {
            *out_descr = (PyArray_Descr *)dtype;
            Py_INCREF(*out_descr);
        }
    }
    return 0;
}


/**
 * Converter function filling in an npy_dtype_info struct on success.
 *
 * @param obj representing a dtype instance (descriptor) or DType class.
 * @param[out] dt_info npy_dtype_info filled with the DType class and dtype/descriptor
 *         instance.  The class is always set while the instance may be NULL.
 *         On error, both will be NULL.
 * @return 0 on failure and 1 on success (as a converter)
 */
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterRequired(PyObject *obj, npy_dtype_info *dt_info)
{
    /*
     * Allow dtype classes pass, this could also be generalized to at least
     * some scalar types (right now most of these give instances or)
     */
    dt_info->dtype = NULL;
    dt_info->descr = NULL;

    if (PyObject_TypeCheck(obj, &PyArrayDTypeMeta_Type)) {
        if (obj == (PyObject *)&PyArrayDescr_Type) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot convert np.dtype into a dtype.");
            return NPY_FAIL;
        }
        Py_INCREF(obj);
        dt_info->dtype = (PyArray_DTypeMeta *)obj;
        dt_info->descr = NULL;
        return NPY_SUCCEED;
    }
    PyArray_Descr *descr;
    if (PyArray_DescrConverter(obj, &descr) != NPY_SUCCEED) {
        return NPY_FAIL;
    }
    /*
     * The above converts e.g. "S" or "S0" to the prototype instance, we make
     * it behave the same as the DType.  This is not fully correct, "S0" should
     * be considered an instance with actual 0 length.
     * TODO: It would be nice to fix that eventually.
     */
    int res = PyArray_ExtractDTypeAndDescriptor(
                descr, &dt_info->descr, &dt_info->dtype);
    Py_DECREF(descr);
    if (res < 0) {
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}


/**
 * Converter function filling in an npy_dtype_info struct on success.  It
 * accepts `None` and does nothing in that case (user must initialize to
 * NULL anyway).
 *
 * @param obj None or obj representing a dtype instance (descr) or DType class.
 * @param[out] dt_info filled with the DType class and dtype/descriptor
 *         instance.  If `obj` is None, is not modified.  Otherwise the class
 *         is always set while the instance may be NULL.
 *         On error, both will be NULL.
 * @return 0 on failure and 1 on success (as a converter)
 */
NPY_NO_EXPORT int
PyArray_DTypeOrDescrConverterOptional(PyObject *obj, npy_dtype_info *dt_info)
{
    if (obj == Py_None) {
        /* caller must have initialized for the optional version */
        return NPY_SUCCEED;
    }
    return PyArray_DTypeOrDescrConverterRequired(obj, dt_info);
}

/*NUMPY_API
 *
 * Given a DType class, returns the default instance (descriptor).  This
 * checks for a `singleton` first and only calls the `default_descr` function
 * if necessary.
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_GetDefaultDescr(PyArray_DTypeMeta *DType)
{
    if (DType->singleton != NULL) {
        Py_INCREF(DType->singleton);
        return DType->singleton;
    }
    return NPY_DT_CALL_default_descr(DType);
}


/**
 * Get a dtype instance from a python type
 */
static PyArray_Descr *
_convert_from_type(PyObject *obj) {
    PyTypeObject *typ = (PyTypeObject*)obj;

    if (PyType_IsSubtype(typ, &PyGenericArrType_Type)) {
        return PyArray_DescrFromTypeObject(obj);
    }
    else if (typ == &PyLong_Type) {
        return PyArray_DescrFromType(NPY_INTP);
    }
    else if (typ == &PyFloat_Type) {
        return PyArray_DescrFromType(NPY_DOUBLE);
    }
    else if (typ == &PyComplex_Type) {
        return PyArray_DescrFromType(NPY_CDOUBLE);
    }
    else if (typ == &PyBool_Type) {
        return PyArray_DescrFromType(NPY_BOOL);
    }
    else if (typ == &PyBytes_Type) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=bytes/"S" in coercion: It should not rely on "S0".
         */
        return PyArray_DescrFromType(NPY_STRING);
    }
    else if (typ == &PyUnicode_Type) {
        /*
         * TODO: This should be deprecated, and have special handling for
         *       dtype=str/"U" in coercion: It should not rely on "U0".
         */
        return PyArray_DescrFromType(NPY_UNICODE);
    }
    else if (typ == &PyMemoryView_Type) {
        return PyArray_DescrFromType(NPY_VOID);
    }
    else if (typ == &PyBaseObject_Type) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    else {
        PyObject *DType = PyArray_DiscoverDTypeFromScalarType(typ);
        if (DType != NULL) {
            return PyArray_GetDefaultDescr((PyArray_DTypeMeta *)DType);
        }
        PyArray_Descr *ret = _try_convert_from_dtype_attr(obj);
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);

        /*
         * Note: this comes after _try_convert_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        ret = _try_convert_from_ctypes_type(typ);
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);

        /*
         * All other classes are treated as object. This can be convenient
         * to convey an intention of using it for a specific python type
         * and possibly allow converting to a new type-specific dtype in the future. It may make sense to
         * only allow this only within `dtype=...` keyword argument context
         * in the future.
         */
        return PyArray_DescrFromType(NPY_OBJECT);
    }
}


static PyArray_Descr *
_convert_from_str(PyObject *obj, int align);

static PyArray_Descr *
_convert_from_any(PyObject *obj, int align)
{
    /* default */
    if (obj == Py_None) {
        return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    else if (PyArray_DescrCheck(obj)) {
        PyArray_Descr *ret = (PyArray_Descr *)obj;
        Py_INCREF(ret);
        return ret;
    }
    else if (PyType_Check(obj)) {
        return _convert_from_type(obj);
    }
    /* or a typecode string */
    else if (PyBytes_Check(obj)) {
        /* Allow bytes format strings: convert to unicode */
        PyObject *obj2 = PyUnicode_FromEncodedObject(obj, NULL, NULL);
        if (obj2 == NULL) {
            /* Convert the exception into a TypeError */
            if (PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
                PyErr_SetString(PyExc_TypeError,
                        "data type not understood");
            }
            return NULL;
        }
        PyArray_Descr *ret = _convert_from_str(obj2, align);
        Py_DECREF(obj2);
        return ret;
    }
    else if (PyUnicode_Check(obj)) {
        return _convert_from_str(obj, align);
    }
    else if (PyTuple_Check(obj)) {
        /* or a tuple */
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a tuple object" ) != 0) {
            return NULL;
        }
        PyArray_Descr *ret = _convert_from_tuple(obj, align);
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (PyList_Check(obj)) {
        /* or a list */
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a list object" ) != 0) {
            return NULL;
        }
        PyArray_Descr *ret = _convert_from_array_descr(obj, align);
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (PyDict_Check(obj) || PyDictProxy_Check(obj)) {
        /* or a dictionary */
        if (Py_EnterRecursiveCall(
                " while trying to convert the given data type from"
                " a dict object" ) != 0) {
            return NULL;
        }
        PyArray_Descr *ret = _convert_from_dict(obj, align);
        Py_LeaveRecursiveCall();
        return ret;
    }
    else if (PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Cannot construct a dtype from an array");
        return NULL;
    }
    else {
        PyArray_Descr *ret = _try_convert_from_dtype_attr(obj);
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);
        /*
         * Note: this comes after _try_convert_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        ret = _try_convert_from_ctypes_type(Py_TYPE(obj));
        if ((PyObject *)ret != Py_NotImplemented) {
            return ret;
        }
        Py_DECREF(ret);
        PyErr_Format(PyExc_TypeError, "Cannot interpret '%R' as a data type", obj);
        return NULL;
    }
}


/*NUMPY_API
 * Get typenum from an object -- None goes to NPY_DEFAULT_TYPE
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 *
 * Returns a new reference in *at, but the returned should not be
 * modified as it may be one of the canonical immutable objects or
 * a reference to the input obj.
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    *at = _convert_from_any(obj, 0);
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/** Convert a bytestring specification into a dtype */
static PyArray_Descr *
_convert_from_str(PyObject *obj, int align)
{
    /* Check for a string typecode. */
    Py_ssize_t len = 0;
    char const *type = PyUnicode_AsUTF8AndSize(obj, &len);
    if (type == NULL) {
        return NULL;
    }

    /* Empty string is invalid */
    if (len == 0) {
        goto fail;
    }

    /* check for commas present or first (or second) element a digit */
    if (_check_for_commastring(type, len)) {
        return _convert_from_commastring(obj, align);
    }

    /* Process the endian character. '|' is replaced by '='*/
    char endian = '=';
    switch (type[0]) {
        case '>':
        case '<':
        case '=':
            endian = type[0];
            ++type;
            --len;
            break;

        case '|':
            endian = '=';
            ++type;
            --len;
            break;
    }

    /* Just an endian character is invalid */
    if (len == 0) {
        goto fail;
    }

    /* Check for datetime format */
    if (is_datetime_typestr(type, len)) {
        PyArray_Descr *ret = parse_dtype_from_datetime_typestr(type, len);
        if (ret == NULL) {
            return NULL;
        }
        /* ret has byte order '=' at this point */
        if (!PyArray_ISNBO(endian)) {
            ret->byteorder = endian;
        }
        return ret;
    }

    int check_num = NPY_NOTYPE + 10;
    int elsize = 0;
    /* A typecode like 'd' */
    if (len == 1) {
        /* Python byte string characters are unsigned */
        check_num = (unsigned char) type[0];
    }
    /* Possibly a kind + size like 'f8' but also could be 'bool' */
    else {
        char *typeend = NULL;
        int kind;

        /* Attempt to parse the integer, make sure it's the rest of the string */
        errno = 0;
        long result = strtol(type + 1, &typeend, 10);
        npy_bool some_parsing_happened = !(type == typeend);
        npy_bool entire_string_consumed = *typeend == '\0';
        npy_bool parsing_succeeded =
                (errno == 0) && some_parsing_happened && entire_string_consumed;
        // make sure it doesn't overflow or go negative
        if (result > INT_MAX || result < 0) {
            goto fail;
        }

        elsize = (int)result;


        if (parsing_succeeded && typeend - type == len) {

            kind = type[0];
            switch (kind) {
                case NPY_STRINGLTR:
                    check_num = NPY_STRING;
                    break;

                case NPY_DEPRECATED_STRINGLTR2:
                    if (DEPRECATE("Data type alias 'a' was deprecated in NumPy 2.0. "
                                  "Use the 'S' alias instead.") < 0) {
                        return NULL;
                    }
                    check_num = NPY_STRING;
                    break;

                /*
                 * When specifying length of UNICODE
                 * the number of characters is given to match
                 * the STRING interface.  Each character can be
                 * more than one byte and itemsize must be
                 * the number of bytes.
                 */
                case NPY_UNICODELTR:
                    check_num = NPY_UNICODE;
                    if (elsize > (NPY_MAX_INT / 4)) {
                        goto fail;
                    }
                    elsize *= 4;
                    break;

                case NPY_VOIDLTR:
                    check_num = NPY_VOID;
                    break;

                default:
                    if (elsize == 0) {
                        check_num = NPY_NOTYPE+10;
                    }
                    /* Support for generic processing c8, i4, f8, etc...*/
                    else {
                        check_num = PyArray_TypestrConvert(elsize, kind);
                        if (check_num == NPY_NOTYPE) {
                            check_num += 10;
                        }
                        elsize = 0;
                    }
            }
        }
        else if (parsing_succeeded) {
            goto fail;
        }
    }

    if (PyErr_Occurred()) {
        goto fail;
    }

    PyArray_Descr *ret;
    if ((check_num == NPY_NOTYPE + 10) ||
            (ret = PyArray_DescrFromType(check_num)) == NULL) {
        PyErr_Clear();
        /* Now check to see if the object is registered in typeDict */
        if (typeDict == NULL) {
            goto fail;
        }
        PyObject *item = PyDict_GetItemWithError(typeDict, obj);
        if (item == NULL) {
            if (PyErr_Occurred()) {
                return NULL;
            }
            if (
                strcmp(type, "int0") == 0 || strcmp(type, "uint0") == 0 ||
                strcmp(type, "void0") == 0 || strcmp(type, "object0") == 0 ||
                strcmp(type, "str0") == 0 || strcmp(type, "bytes0") == 0 ||
                strcmp(type, "bool8") == 0
            ) {
                PyErr_Format(PyExc_TypeError,
                        "Alias %R was removed in NumPy 2.0. Use a name "
                        "without a digit at the end.", obj);
                return NULL;
            }

            goto fail;
        }

        if (strcmp(type, "a") == 0) {
            if (DEPRECATE("Data type alias 'a' was deprecated in NumPy 2.0. "
                          "Use the 'S' alias instead.") < 0) {
                return NULL;
            }
        }

        /*
         * Probably only ever dispatches to `_convert_from_type`, but who
         * knows what users are injecting into `np.typeDict`.
         */
        return _convert_from_any(item, align);
    }

    if (PyDataType_ISUNSIZED(ret) && ret->elsize != elsize) {
        PyArray_DESCR_REPLACE(ret);
        if (ret == NULL) {
            return NULL;
        }
        ret->elsize = elsize;
    }
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && ret->byteorder != '|' && ret->byteorder != endian) {
        PyArray_DESCR_REPLACE(ret);
        if (ret == NULL) {
            return NULL;
        }
        ret->byteorder = endian;
    }
    return ret;

fail:
    PyErr_Format(PyExc_TypeError, "data type %R not understood", obj);
    return NULL;
}

/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This lets us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base_descr)
{
    if (!PyDataType_ISLEGACY(base_descr)) {
        /* 
         * The main use of this function is mutating strings, so probably
         * disallowing this is fine in practice.
         */
        PyErr_SetString(PyExc_RuntimeError,
            "cannot use `PyArray_DescrNew` on new style DTypes.");
        return NULL;
    }
    _PyArray_LegacyDescr *base = (_PyArray_LegacyDescr *)base_descr;
    _PyArray_LegacyDescr *newdescr = PyObject_New(_PyArray_LegacyDescr, Py_TYPE(base));

    if (newdescr == NULL) {
        return NULL;
    }
    /* Don't copy PyObject_HEAD part */
    memcpy((char *)newdescr + sizeof(PyObject),
           (char *)base + sizeof(PyObject),
           sizeof(_PyArray_LegacyDescr) - sizeof(PyObject));

    /*
     * The c_metadata has a by-value ownership model, need to clone it
     * (basically a deep copy, but the auxdata clone function has some
     * flexibility still) so the new PyArray_Descr object owns
     * a copy of the data. Having both 'base' and 'newdescr' point to
     * the same auxdata pointer would cause a double-free of memory.
     */
    if (base->c_metadata != NULL) {
        newdescr->c_metadata = NPY_AUXDATA_CLONE(base->c_metadata);
        if (newdescr->c_metadata == NULL) {
            PyErr_NoMemory();
            /* TODO: This seems wrong, as the old fields get decref'd? */
            Py_DECREF(newdescr);
            return NULL;
        }
    }

    if (newdescr->fields == Py_None) {
        newdescr->fields = NULL;
    }
    Py_XINCREF(newdescr->fields);
    Py_XINCREF(newdescr->names);
    if (newdescr->subarray) {
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            Py_DECREF(newdescr);
            return (PyArray_Descr *)PyErr_NoMemory();
        }
        memcpy(newdescr->subarray, base->subarray, sizeof(PyArray_ArrayDescr));
        Py_INCREF(newdescr->subarray->shape);
        Py_INCREF(newdescr->subarray->base);
    }
    Py_XINCREF(newdescr->typeobj);
    Py_XINCREF(newdescr->metadata);
    newdescr->hash = -1;

    return (PyArray_Descr *)newdescr;
}

/*
 * should never be called for builtin-types unless
 * there is a reference-count problem
 */
static void
arraydescr_dealloc(PyArray_Descr *self)
{
    Py_XDECREF(self->typeobj);
    if (!PyDataType_ISLEGACY(self)) {
        /* non legacy dtypes must not have fields, etc. */
        Py_TYPE(self)->tp_free((PyObject *)self);
        return;
    }
    _PyArray_LegacyDescr *lself = (_PyArray_LegacyDescr *)self;

    if (lself->fields == Py_None) {
        fprintf(stderr, "*** Reference count error detected: "
                "an attempt was made to deallocate the dtype %d (%c) ***\n",
                self->type_num, self->type);
        assert(0);
        Py_INCREF(self);
        Py_INCREF(self);
        return;
    }
    Py_XDECREF(lself->names);
    Py_XDECREF(lself->fields);
    if (lself->subarray) {
        Py_XDECREF(lself->subarray->shape);
        Py_DECREF(lself->subarray->base);
        PyArray_free(lself->subarray);
    }
    Py_XDECREF(lself->metadata);
    NPY_AUXDATA_FREE(lself->c_metadata);
    lself->c_metadata = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*
 * we need to be careful about setting attributes because these
 * objects are pointed to by arrays that depend on them for interpreting
 * data.  Currently no attributes of data-type objects can be set
 * directly except names.
 */
static PyMemberDef arraydescr_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_Descr, typeobj), READONLY, NULL},
    {"kind",
        T_CHAR, offsetof(PyArray_Descr, kind), READONLY, NULL},
    {"char",
        T_CHAR, offsetof(PyArray_Descr, type), READONLY, NULL},
    {"num",
        T_INT, offsetof(PyArray_Descr, type_num), READONLY, NULL},
    {"byteorder",
        T_CHAR, offsetof(PyArray_Descr, byteorder), READONLY, NULL},
    {"itemsize",
        T_PYSSIZET, offsetof(PyArray_Descr, elsize), READONLY, NULL},
    {"alignment",
        T_PYSSIZET, offsetof(PyArray_Descr, alignment), READONLY, NULL},
    {"flags",
#if NPY_ULONGLONG == NPY_UINT64
        T_ULONGLONG, offsetof(PyArray_Descr, flags), READONLY, NULL},
#else
    #error Assuming long long is 64bit, if not replace with getter function.
#endif
  {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraydescr_subdescr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("OO",
            PyDataType_SUBARRAY(self)->base, PyDataType_SUBARRAY(self)->shape);
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_typestr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_ISLEGACY(NPY_DTYPE(self))) {
        return (PyObject *) Py_TYPE(self)->tp_str((PyObject *)self);
    }

    char basic_ = self->kind;
    char endian = self->byteorder;
    int size = self->elsize;
    PyObject *ret;

    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (self->type_num == NPY_UNICODE) {
        size >>= 2;
    }
    if (self->type_num == NPY_OBJECT) {
        ret = PyUnicode_FromFormat("%c%c", endian, basic_);
    }
    else {
        ret = PyUnicode_FromFormat("%c%c%d", endian, basic_, size);
    }
    if (ret == NULL) {
        return NULL;
    }

    if (PyDataType_ISDATETIME(self)) {
        PyArray_DatetimeMetaData *meta;
        meta = get_datetime_metadata_from_dtype(self);
        if (meta == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyObject *umeta = metastr_to_unicode(meta, 0);
        if (umeta == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        Py_SETREF(ret, PyUnicode_Concat(ret, umeta));
        Py_DECREF(umeta);
    }
    return ret;
}

static PyObject *
arraydescr_name_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    /* let python handle this */
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "_name_get", "O", self);
    Py_DECREF(_numpy_dtype);
    return res;
}

static PyObject *
arraydescr_base_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    Py_INCREF(PyDataType_SUBARRAY(self)->base);
    return (PyObject *)(PyDataType_SUBARRAY(self)->base);
}

static PyObject *
arraydescr_shape_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASSUBARRAY(self)) {
        return PyTuple_New(0);
    }
    assert(PyTuple_Check(PyDataType_SUBARRAY(self)->shape));
    Py_INCREF(PyDataType_SUBARRAY(self)->shape);
    return PyDataType_SUBARRAY(self)->shape;
}

static PyObject *
arraydescr_ndim_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    Py_ssize_t ndim;

    if (!PyDataType_HASSUBARRAY(self)) {
        return PyLong_FromLong(0);
    }

    /*
     * PyTuple_Size has built in check
     * for tuple argument
     */
    ndim = PyTuple_Size(PyDataType_SUBARRAY(self)->shape);
    return PyLong_FromLong(ndim);
}


NPY_NO_EXPORT PyObject *
arraydescr_protocol_descr_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *dobj, *res;
    PyObject *_numpy_internal;

    if (!PyDataType_HASFIELDS(self)) {
        /* get default */
        dobj = PyTuple_New(2);
        if (dobj == NULL) {
            return NULL;
        }
        PyTuple_SET_ITEM(dobj, 0, PyUnicode_FromString(""));
        PyTuple_SET_ITEM(dobj, 1, arraydescr_protocol_typestr_get(self, NULL));
        res = PyList_New(1);
        if (res == NULL) {
            Py_DECREF(dobj);
            return NULL;
        }
        PyList_SET_ITEM(res, 0, dobj);
        return res;
    }

    _numpy_internal = PyImport_ImportModule("numpy._core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_internal, "_array_descr", "O", self);
    Py_DECREF(_numpy_internal);
    return res;
}

/*
 * returns 1 for a builtin type
 * and 2 for a user-defined data-type descriptor
 * return 0 if neither (i.e. it's a copy of one)
 */
static PyObject *
arraydescr_isbuiltin_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    long val;
    val = 0;
    if (PyDataType_FIELDS(self) == Py_None) {
        val = 1;
    }
    if (PyTypeNum_ISUSERDEF(self->type_num)) {
        val = 2;
    }
    return PyLong_FromLong(val);
}

static int
_arraydescr_isnative(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        return PyArray_ISNBO(self->byteorder);
    }
    else {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(PyDataType_FIELDS(self), &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return -1;
            }
            if (!_arraydescr_isnative(new)) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * return Py_True if this data-type descriptor
 * has native byteorder if no fields are defined
 *
 * or if all sub-fields have native-byteorder if
 * fields are defined
 */
static PyObject *
arraydescr_isnative_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;
    int retval;
    retval = _arraydescr_isnative(self);
    if (retval == -1) {
        return NULL;
    }
    ret = retval ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

static PyObject *
arraydescr_isalignedstruct_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;
    ret = (self->flags&NPY_ALIGNED_STRUCT) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

static PyObject *
arraydescr_fields_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASFIELDS(self)) {
        Py_RETURN_NONE;
    }
    return PyDictProxy_New(PyDataType_FIELDS(self));
}

static PyObject *
arraydescr_metadata_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (self->metadata == NULL) {
        Py_RETURN_NONE;
    }
    return PyDictProxy_New(self->metadata);
}

static PyObject *
arraydescr_hasobject_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
arraydescr_names_get(PyArray_Descr *self, void *NPY_UNUSED(ignored))
{
    if (!PyDataType_HASFIELDS(self)) {
        Py_RETURN_NONE;
    }
    Py_INCREF(PyDataType_NAMES(self));
    return PyDataType_NAMES(self);
}

static int
arraydescr_names_set(
        _PyArray_LegacyDescr *self, PyObject *val, void *NPY_UNUSED(ignored))
{
    int N = 0;
    int i;
    PyObject *new_names;
    PyObject *new_fields;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete dtype names attribute");
        return -1;
    }
    if (!PyDataType_HASFIELDS(self)) {
        PyErr_SetString(PyExc_ValueError,
                "there are no fields defined");
        return -1;
    }

    N = PyTuple_GET_SIZE(self->names);
    if (!PySequence_Check(val) || PyObject_Size((PyObject *)val) != N) {
        /* Should be a TypeError, but this should be deprecated anyway. */
        PyErr_Format(PyExc_ValueError,
                "must replace all names at once with a sequence of length %d",
                N);
        return -1;
    }
    /* Make sure all entries are strings */
    for (i = 0; i < N; i++) {
        PyObject *item;
        int valid;
        item = PySequence_GetItem(val, i);
        valid = PyUnicode_Check(item);
        if (!valid) {
            PyErr_Format(PyExc_ValueError,
                    "item #%d of names is of type %s and not string",
                    i, Py_TYPE(item)->tp_name);
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }
    /* Invalidate cached hash value */
    self->hash = -1;
    /* Update dictionary keys in fields */
    new_names = PySequence_Tuple(val);
    if (new_names == NULL) {
        return -1;
    }
    new_fields = PyDict_New();
    if (new_fields == NULL) {
        Py_DECREF(new_names);
        return -1;
    }
    for (i = 0; i < N; i++) {
        PyObject *key;
        PyObject *item;
        PyObject *new_key;
        int ret;
        key = PyTuple_GET_ITEM(self->names, i);
        /* Borrowed references to item and new_key */
        item = PyDict_GetItemWithError(self->fields, key);
        if (item == NULL) {
            if (!PyErr_Occurred()) {
                /* fields was missing the name it claimed to contain */
                PyErr_BadInternalCall();
            }
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        new_key = PyTuple_GET_ITEM(new_names, i);
        /* Check for duplicates */
        ret = PyDict_Contains(new_fields, new_key);
        if (ret < 0) {
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        else if (ret != 0) {
            PyErr_SetString(PyExc_ValueError, "Duplicate field names given.");
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        if (PyDict_SetItem(new_fields, new_key, item) < 0) {
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
    }

    /* Replace names */
    Py_DECREF(self->names);
    self->names = new_names;

    /* Replace fields */
    Py_DECREF(self->fields);
    self->fields = new_fields;

    return 0;
}

static PyGetSetDef arraydescr_getsets[] = {
    {"subdtype",
        (getter)arraydescr_subdescr_get,
        NULL, NULL, NULL},
    {"descr",
        (getter)arraydescr_protocol_descr_get,
        NULL, NULL, NULL},
    {"str",
        (getter)arraydescr_protocol_typestr_get,
        NULL, NULL, NULL},
    {"name",
        (getter)arraydescr_name_get,
        NULL, NULL, NULL},
    {"base",
        (getter)arraydescr_base_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)arraydescr_shape_get,
        NULL, NULL, NULL},
    {"ndim",
        (getter)arraydescr_ndim_get,
        NULL, NULL, NULL},
    {"isbuiltin",
        (getter)arraydescr_isbuiltin_get,
        NULL, NULL, NULL},
    {"isnative",
        (getter)arraydescr_isnative_get,
        NULL, NULL, NULL},
    {"isalignedstruct",
        (getter)arraydescr_isalignedstruct_get,
        NULL, NULL, NULL},
    {"fields",
        (getter)arraydescr_fields_get,
        NULL, NULL, NULL},
    {"metadata",
        (getter)arraydescr_metadata_get,
        NULL, NULL, NULL},
    {"names",
        (getter)arraydescr_names_get,
        (setter)arraydescr_names_set,
        NULL, NULL},
    {"hasobject",
        (getter)arraydescr_hasobject_get,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *
arraydescr_new(PyTypeObject *subtype,
                PyObject *args, PyObject *kwds)
{
    if (subtype != &PyArrayDescr_Type) {
        if (Py_TYPE(subtype) == &PyArrayDTypeMeta_Type &&
                (NPY_DT_SLOTS((PyArray_DTypeMeta *)subtype)) != NULL &&
                !NPY_DT_is_legacy((PyArray_DTypeMeta *)subtype) &&
                subtype->tp_new != PyArrayDescr_Type.tp_new) {
            /*
             * Appears to be a properly initialized user DType. Allocate
             * it and initialize the main part as best we can.
             * TODO: This should probably be a user function, and enforce
             *       things like the `elsize` being correctly set.
             * TODO: This is EXPERIMENTAL API!
             */
            PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)subtype;
            PyArray_Descr *descr = (PyArray_Descr *)subtype->tp_alloc(subtype, 0);
            if (descr == 0) {
                PyErr_NoMemory();
                return NULL;
            }
            Py_XINCREF(DType->scalar_type);
            descr->typeobj = DType->scalar_type;
            descr->type_num = DType->type_num;
            descr->flags = NPY_USE_GETITEM|NPY_USE_SETITEM;
            descr->byteorder = '|';  /* If DType uses it, let it override */
            descr->elsize = -1;  /* Initialize to invalid value */
            descr->hash = -1;
            return (PyObject *)descr;
        }
        /* The DTypeMeta class should prevent this from happening. */
        PyErr_Format(PyExc_SystemError,
                "'%S' must not inherit np.dtype.__new__(). User DTypes should "
                "currently call `PyArrayDescr_Type.tp_new` from their new.",
                subtype);
        return NULL;
    }

    PyObject *odescr, *metadata=NULL;
    PyArray_Descr *conv;
    npy_bool align = NPY_FALSE;
    npy_bool copy = NPY_FALSE;
    npy_bool copied = NPY_FALSE;

    static char *kwlist[] = {"dtype", "align", "copy", "metadata", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O!:dtype", kwlist,
                &odescr,
                PyArray_BoolConverter, &align,
                PyArray_BoolConverter, &copy,
                &PyDict_Type, &metadata)) {
        return NULL;
    }

    conv = _convert_from_any(odescr, align);
    if (conv == NULL) {
        return NULL;
    }

    /* Get a new copy of it unless it's already a copy */
    if (copy && PyDataType_FIELDS(conv) == Py_None) {
        PyArray_DESCR_REPLACE(conv);
        if (conv == NULL) {
            return NULL;
        }
        copied = NPY_TRUE;
    }

    if ((metadata != NULL)) {
        if (!PyDataType_ISLEGACY(conv)) {
            PyErr_SetString(PyExc_TypeError,
                    "cannot attach metadata to new style DType");
            Py_DECREF(conv);
            return NULL;
        }
        /*
         * We need to be sure to make a new copy of the data-type and any
         * underlying dictionary
         */
        if (!copied) {
            PyArray_DESCR_REPLACE(conv);
            if (conv == NULL) {
                return NULL;
            }
            copied = NPY_TRUE;
        }
        _PyArray_LegacyDescr *lconv = (_PyArray_LegacyDescr *)conv;
        if ((lconv->metadata != NULL)) {
            /*
             * Make a copy of the metadata before merging with the
             * input metadata so that this data-type descriptor has
             * it's own copy
             */
            /* Save a reference */
            odescr = lconv->metadata;
            lconv->metadata = PyDict_Copy(odescr);
            /* Decrement the old reference */
            Py_DECREF(odescr);

            /*
             * Update conv->metadata with anything new in metadata
             * keyword, but do not over-write anything already there
             */
            if (PyDict_Merge(lconv->metadata, metadata, 0) != 0) {
                Py_DECREF(conv);
                return NULL;
            }
        }
        else {
            /* Make a copy of the input dictionary */
            lconv->metadata = PyDict_Copy(metadata);
        }
    }

    return (PyObject *)conv;
}


/*
 * Return a tuple of
 * (cleaned metadata dictionary, tuple with (str, num))
 */
static PyObject *
_get_pickleabletype_from_datetime_metadata(PyArray_Descr *dtype)
{
    PyObject *ret, *dt_tuple;
    PyArray_DatetimeMetaData *meta;

    /* Create the 2-item tuple to return */
    ret = PyTuple_New(2);
    if (ret == NULL) {
        return NULL;
    }

    /* Store the metadata dictionary */
    if (dtype->metadata != NULL) {
        Py_INCREF(dtype->metadata);
        PyTuple_SET_ITEM(ret, 0, dtype->metadata);
    } else {
        PyTuple_SET_ITEM(ret, 0, PyDict_New());
    }

    /* Convert the datetime metadata into a tuple */
    meta = get_datetime_metadata_from_dtype(dtype);
    if (meta == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    /* Use a 4-tuple that numpy 1.6 knows how to unpickle */
    dt_tuple = PyTuple_New(4);
    if (dt_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[meta->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyLong_FromLong(meta->num));
    PyTuple_SET_ITEM(dt_tuple, 2,
            PyLong_FromLong(1));
    PyTuple_SET_ITEM(dt_tuple, 3,
            PyLong_FromLong(1));

    PyTuple_SET_ITEM(ret, 1, dt_tuple);

    return ret;
}

/*
 * return a tuple of (callable object, args, state).
 *
 * TODO: This method needs to change so that unpickling doesn't
 *       use __setstate__. This is required for the dtype
 *       to be an immutable object.
 */
static PyObject *
arraydescr_reduce(PyArray_Descr *self, PyObject *NPY_UNUSED(args))
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 4;
    PyObject *ret, *mod, *obj;
    PyObject *state;
    char endian;
    int elsize, alignment;

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy._core._multiarray_umath");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    obj = PyObject_GetAttr(mod, npy_interned_str.dtype);
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, obj);
    if (PyTypeNum_ISUSERDEF(self->type_num)
            || ((self->type_num == NPY_VOID
                    && self->typeobj != &PyVoidArrType_Type))) {
        obj = (PyObject *)self->typeobj;
        Py_INCREF(obj);
    }
    else if (!NPY_DT_is_legacy(NPY_DTYPE(self))) {
        PyErr_SetString(PyExc_RuntimeError,
                "Custom dtypes cannot use the default pickle implementation "
                "for NumPy dtypes. Add a custom pickle implementation to the "
                "DType to avoid this error");
        return NULL;
    }
    else {
        elsize = self->elsize;
        if (self->type_num == NPY_UNICODE) {
            elsize >>= 2;
        }
        obj = PyUnicode_FromFormat("%c%d",self->kind, elsize);
    }
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(NOO)", obj, Py_False, Py_True));

    /*
     * Now return the state which is at least byteorder,
     * subarray, and fields
     */
    endian = self->byteorder;
    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (PyDataType_ISDATETIME(self)) {
        PyObject *newobj;
        state = PyTuple_New(9);
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(version));
        /*
         * newobj is a tuple of the Python metadata dictionary
         * and tuple of date_time info (str, num)
         */
        newobj = _get_pickleabletype_from_datetime_metadata(self);
        if (newobj == NULL) {
            Py_DECREF(state);
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(state, 8, newobj);
    }
    else if (self->metadata) {
        state = PyTuple_New(9);
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(version));
        Py_INCREF(self->metadata);
        PyTuple_SET_ITEM(state, 8, self->metadata);
    }
    else { /* Use version 3 pickle format */
        state = PyTuple_New(8);
        PyTuple_SET_ITEM(state, 0, PyLong_FromLong(3));
    }

    PyTuple_SET_ITEM(state, 1, PyUnicode_FromFormat("%c", endian));
    PyTuple_SET_ITEM(state, 2, arraydescr_subdescr_get(self, NULL));
    if (PyDataType_HASFIELDS(self)) {
        Py_INCREF(PyDataType_NAMES(self));
        Py_INCREF(PyDataType_FIELDS(self));
        PyTuple_SET_ITEM(state, 3, PyDataType_NAMES(self));
        PyTuple_SET_ITEM(state, 4, PyDataType_FIELDS(self));
    }
    else {
        PyTuple_SET_ITEM(state, 3, Py_None);
        PyTuple_SET_ITEM(state, 4, Py_None);
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
    }

    /* for extended types it also includes elsize and alignment */
    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        elsize = self->elsize;
        alignment = self->alignment;
    }
    else {
        elsize = -1;
        alignment = -1;
    }
    PyTuple_SET_ITEM(state, 5, PyLong_FromLong(elsize));
    PyTuple_SET_ITEM(state, 6, PyLong_FromLong(alignment));
    PyTuple_SET_ITEM(state, 7, PyLong_FromUnsignedLongLong(self->flags));

    PyTuple_SET_ITEM(ret, 2, state);
    return ret;
}

/*
 * returns NPY_OBJECT_DTYPE_FLAGS if this data-type has an object portion used
 * when setting the state because hasobject is not stored.
 */
static char
_descr_find_object(PyArray_Descr *self)
{
    if (self->flags
            || self->type_num == NPY_OBJECT
            || self->kind == 'O') {
        return NPY_OBJECT_DTYPE_FLAGS;
    }
    if (PyDataType_HASFIELDS(self)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(PyDataType_FIELDS(self), &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                PyErr_Clear();
                return 0;
            }
            if (_descr_find_object(new)) {
                new->flags = NPY_OBJECT_DTYPE_FLAGS;
                return NPY_OBJECT_DTYPE_FLAGS;
            }
        }
    }
    return 0;
}

/*
 * state is at least byteorder, subarray, and fields but could include elsize
 * and alignment for EXTENDED arrays
 */
static PyObject *
arraydescr_setstate(_PyArray_LegacyDescr *self, PyObject *args)
{
    int elsize = -1, alignment = -1;
    int version = 4;
    char endian;
    PyObject *endian_obj;
    PyObject *subarray, *fields, *names = NULL, *metadata=NULL;
    int incref_names = 1;
    int int_dtypeflags = 0;
    npy_uint64 dtypeflags;

    if (!PyDataType_ISLEGACY(self)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Cannot unpickle new style DType without custom methods.");
        return NULL;
    }

    if (self->fields == Py_None) {
        Py_RETURN_NONE;
    }
    if (PyTuple_GET_SIZE(args) != 1
            || !(PyTuple_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }
    switch (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0))) {
    case 9:
        if (!PyArg_ParseTuple(args, "(iOOOOiiiO):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags, &metadata)) {
            PyErr_Clear();
            return NULL;
        }
        break;
    case 8:
        if (!PyArg_ParseTuple(args, "(iOOOOiii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags)) {
            return NULL;
        }
        break;
    case 7:
        if (!PyArg_ParseTuple(args, "(iOOOOii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    case 6:
        if (!PyArg_ParseTuple(args, "(iOOOii):__setstate__",
                    &version,
                    &endian_obj, &subarray, &fields,
                    &elsize, &alignment)) {
            return NULL;
        }
        break;
    case 5:
        version = 0;
        if (!PyArg_ParseTuple(args, "(OOOii):__setstate__",
                    &endian_obj, &subarray, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    default:
        /* raise an error */
        if (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0)) > 5) {
            version = PyLong_AsLong(PyTuple_GET_ITEM(args, 0));
        }
        else {
            version = -1;
        }
    }

    /*
     * If we ever need another pickle format, increment the version
     * number. But we should still be able to handle the old versions.
     */
    if (version < 0 || version > 4) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return NULL;
    }
    /* Invalidate cached hash value */
    self->hash = -1;

    if (version == 1 || version == 0) {
        if (fields != Py_None) {
            PyObject *key, *list;
            key = PyLong_FromLong(-1);
            list = PyDict_GetItemWithError(fields, key);
            if (!list) {
                if (!PyErr_Occurred()) {
                    /* fields was missing the name it claimed to contain */
                    PyErr_BadInternalCall();
                }
                return NULL;
            }
            Py_INCREF(list);
            names = list;
            PyDict_DelItem(fields, key);
            incref_names = 0;
        }
        else {
            names = Py_None;
        }
    }

    /* Parse endian */
    if (PyUnicode_Check(endian_obj) || PyBytes_Check(endian_obj)) {
        PyObject *tmp = NULL;
        char *str;
        Py_ssize_t len;

        if (PyUnicode_Check(endian_obj)) {
            tmp = PyUnicode_AsASCIIString(endian_obj);
            if (tmp == NULL) {
                return NULL;
            }
            endian_obj = tmp;
        }

        if (PyBytes_AsStringAndSize(endian_obj, &str, &len) < 0) {
            Py_XDECREF(tmp);
            return NULL;
        }
        if (len != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "endian is not 1-char string in Numpy dtype unpickling");
            Py_XDECREF(tmp);
            return NULL;
        }
        endian = str[0];
        Py_XDECREF(tmp);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "endian is not a string in Numpy dtype unpickling");
        return NULL;
    }

    if ((fields == Py_None && names != Py_None) ||
        (names == Py_None && fields != Py_None)) {
        PyErr_Format(PyExc_ValueError,
                "inconsistent fields and names in Numpy dtype unpickling");
        return NULL;
    }

    if (names != Py_None && !PyTuple_Check(names)) {
        PyErr_Format(PyExc_ValueError,
                "non-tuple names in Numpy dtype unpickling");
        return NULL;
    }

    if (fields != Py_None && !PyDict_Check(fields)) {
        PyErr_Format(PyExc_ValueError,
                "non-dict fields in Numpy dtype unpickling");
        return NULL;
    }

    if (endian != '|' && PyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    self->byteorder = endian;
    if (self->subarray) {
        Py_XDECREF(self->subarray->base);
        Py_XDECREF(self->subarray->shape);
        PyArray_free(self->subarray);
    }
    self->subarray = NULL;

    if (subarray != Py_None) {
        PyObject *subarray_shape;

        /*
         * Ensure that subarray[0] is an ArrayDescr and
         * that subarray_shape obtained from subarray[1] is a tuple of integers.
         */
        if (!(PyTuple_Check(subarray) &&
              PyTuple_Size(subarray) == 2 &&
              PyArray_DescrCheck(PyTuple_GET_ITEM(subarray, 0)))) {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray in __setstate__");
            return NULL;
        }
        subarray_shape = PyTuple_GET_ITEM(subarray, 1);
        if (PyNumber_Check(subarray_shape)) {
            PyObject *tmp;
            tmp = PyNumber_Long(subarray_shape);
            if (tmp == NULL) {
                return NULL;
            }
            subarray_shape = Py_BuildValue("(O)", tmp);
            Py_DECREF(tmp);
            if (subarray_shape == NULL) {
                return NULL;
            }
        }
        else if (_is_tuple_of_integers(subarray_shape)) {
            Py_INCREF(subarray_shape);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray shape in __setstate__");
            return NULL;
        }

        self->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (self->subarray == NULL) {
            return PyErr_NoMemory();
        }
        self->subarray->base = (PyArray_Descr *)PyTuple_GET_ITEM(subarray, 0);
        Py_INCREF(self->subarray->base);
        self->subarray->shape = subarray_shape;
    }

    if (fields != Py_None) {
        /*
         * Ensure names are of appropriate string type
         */
        Py_ssize_t i;
        int names_ok = 1;
        PyObject *name;

        for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
            name = PyTuple_GET_ITEM(names, i);
            if (!PyUnicode_Check(name)) {
                names_ok = 0;
                break;
            }
        }

        if (names_ok) {
            Py_XDECREF(self->fields);
            self->fields = fields;
            Py_INCREF(fields);
            Py_XDECREF(self->names);
            self->names = names;
            if (incref_names) {
                Py_INCREF(names);
            }
        }
        else {
            /*
             * To support pickle.load(f, encoding='bytes') for loading Py2
             * generated pickles on Py3, we need to be more lenient and convert
             * field names from byte strings to unicode.
             */
            PyObject *tmp, *new_name, *field;

            tmp = PyDict_New();
            if (tmp == NULL) {
                return NULL;
            }
            Py_XDECREF(self->fields);
            self->fields = tmp;

            tmp = PyTuple_New(PyTuple_GET_SIZE(names));
            if (tmp == NULL) {
                return NULL;
            }
            Py_XDECREF(self->names);
            self->names = tmp;

            for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
                name = PyTuple_GET_ITEM(names, i);
                field = PyDict_GetItemWithError(fields, name);
                if (!field) {
                    if (!PyErr_Occurred()) {
                        /* fields was missing the name it claimed to contain */
                        PyErr_BadInternalCall();
                    }
                    return NULL;
                }

                if (PyUnicode_Check(name)) {
                    new_name = name;
                    Py_INCREF(new_name);
                }
                else {
                    new_name = PyUnicode_FromEncodedObject(name, "ASCII", "strict");
                    if (new_name == NULL) {
                        return NULL;
                    }
                }

                PyTuple_SET_ITEM(self->names, i, new_name);
                if (PyDict_SetItem(self->fields, new_name, field) != 0) {
                    return NULL;
                }
            }
        }
    }

    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        self->elsize = elsize;
        self->alignment = alignment;
    }

    /*
     * We use an integer converted to char for backward compatibility with
     * pickled arrays. Pickled arrays created with previous versions encoded
     * flags as an int even though it actually was a char in the PyArray_Descr
     * structure
     */
    if (int_dtypeflags < 0 && int_dtypeflags >= -128) {
        /* NumPy used to use a char. So normalize if signed. */
        int_dtypeflags += 128;
    }
    dtypeflags = int_dtypeflags;
    if (dtypeflags != int_dtypeflags) {
        PyErr_Format(PyExc_ValueError,
                     "incorrect value for flags variable (overflow)");
        return NULL;
    }
    else {
        self->flags = dtypeflags;
    }

    if (version < 3) {
        self->flags = _descr_find_object((PyArray_Descr *)self);
    }

    /*
     * We have a borrowed reference to metadata so no need
     * to alter reference count when throwing away Py_None.
     */
    if (metadata == Py_None) {
        metadata = NULL;
    }

    if (PyDataType_ISDATETIME(self) && (metadata != NULL)) {
        PyObject *old_metadata;
        PyArray_DatetimeMetaData temp_dt_data;

        if ((! PyTuple_Check(metadata)) || (PyTuple_Size(metadata) != 2)) {
            PyErr_Format(PyExc_ValueError,
                    "Invalid datetime dtype (metadata, c_metadata): %R",
                    metadata);
            return NULL;
        }

        if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                    PyTuple_GET_ITEM(metadata, 1),
                                    &temp_dt_data,
                                    NPY_TRUE) < 0) {
            return NULL;
        }

        old_metadata = self->metadata;
        self->metadata = PyTuple_GET_ITEM(metadata, 0);
        memcpy((char *) &((PyArray_DatetimeDTypeMetaData *)self->c_metadata)->meta,
               (char *) &temp_dt_data,
               sizeof(PyArray_DatetimeMetaData));
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }
    else {
        PyObject *old_metadata = self->metadata;
        self->metadata = metadata;
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }

    Py_RETURN_NONE;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from NPY_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    *at = _convert_from_any(obj, 1);
    return (*at) ? NPY_SUCCEED : NPY_FAIL;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        return PyArray_DescrAlignConverter(obj, at);
    }
}



/*NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewByteorder(PyArray_Descr *oself, char newendian)
{
    char endian;

    if (!PyDataType_ISLEGACY(oself)) {
        PyErr_SetString(PyExc_TypeError,
            "Cannot use DescrNewByteOrder for this new style DTypes.");
        return NULL;
    }

    _PyArray_LegacyDescr *self = (_PyArray_LegacyDescr *)oself;
    _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNew(oself);
    if (new == NULL) {
        return NULL;
    }
    endian = new->byteorder;
    if (endian != NPY_IGNORE) {
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            if (PyArray_ISNBO(endian)) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new->byteorder = endian;
        }
        else if (newendian != NPY_IGNORE) {
            new->byteorder = newendian;
        }
    }
    if (PyDataType_HASFIELDS(new)) {
        PyObject *newfields;
        PyObject *key, *value;
        PyObject *newvalue;
        PyObject *old;
        PyArray_Descr *newdescr;
        Py_ssize_t pos = 0;
        int len, i;

        newfields = PyDict_New();
        if (newfields == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        /* make new dictionary with replaced PyArray_Descr Objects */
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            if (!PyUnicode_Check(key) || !PyTuple_Check(value) ||
                ((len=PyTuple_GET_SIZE(value)) < 2)) {
                continue;
            }
            old = PyTuple_GET_ITEM(value, 0);
            if (!PyArray_DescrCheck(old)) {
                continue;
            }
            newdescr = PyArray_DescrNewByteorder(
                    (PyArray_Descr *)old, newendian);
            if (newdescr == NULL) {
                Py_DECREF(newfields); Py_DECREF(new);
                return NULL;
            }
            newvalue = PyTuple_New(len);
            PyTuple_SET_ITEM(newvalue, 0, (PyObject *)newdescr);
            for (i = 1; i < len; i++) {
                old = PyTuple_GET_ITEM(value, i);
                Py_INCREF(old);
                PyTuple_SET_ITEM(newvalue, i, old);
            }
            int ret = PyDict_SetItem(newfields, key, newvalue);
            Py_DECREF(newvalue);
            if (ret < 0) {
                Py_DECREF(newfields);
                Py_DECREF(new);
                return NULL;
            }
        }
        Py_DECREF(new->fields);
        new->fields = newfields;
    }
    if (new->subarray) {
        Py_DECREF(new->subarray->base);
        new->subarray->base = PyArray_DescrNewByteorder(
                self->subarray->base, newendian);
        if (new->subarray->base == NULL) {
            Py_DECREF(new);
            return NULL;
        }
    }
    return (PyArray_Descr *)new;
}


static PyObject *
arraydescr_newbyteorder(PyArray_Descr *self, PyObject *args)
{
    char endian=NPY_SWAP;

    if (!PyArg_ParseTuple(args, "|O&:newbyteorder", PyArray_ByteorderConverter,
                &endian)) {
        return NULL;
    }
    return (PyObject *)PyArray_DescrNewByteorder(self, endian);
}

static PyObject *
arraydescr_class_getitem(PyObject *cls, PyObject *args)
{
    const Py_ssize_t args_len = PyTuple_Check(args) ? PyTuple_Size(args) : 1;

    if (args_len != 1) {
        return PyErr_Format(PyExc_TypeError,
                            "Too %s arguments for %s",
                            args_len > 1 ? "many" : "few",
                            ((PyTypeObject *)cls)->tp_name);
    }
    return Py_GenericAlias(cls, args);
}

static PyMethodDef arraydescr_methods[] = {
    /* for pickling */
    {"__reduce__",
        (PyCFunction)arraydescr_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction)arraydescr_setstate,
        METH_VARARGS, NULL},
    {"newbyteorder",
        (PyCFunction)arraydescr_newbyteorder,
        METH_VARARGS, NULL},
    /* for typing; requires python >= 3.9 */
    {"__class_getitem__",
        (PyCFunction)arraydescr_class_getitem,
        METH_CLASS | METH_O, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

/*
 * Checks whether the structured data type in 'dtype'
 * has a simple layout, where all the fields are in order,
 * and follow each other with no alignment padding.
 *
 * When this returns true, the dtype can be reconstructed
 * from a list of the field names and dtypes with no additional
 * dtype parameters.
 *
 * Returns 1 if it has a simple layout, 0 otherwise.
 */
NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, names_size;
    PyArray_Descr *fld_dtype;
    int fld_offset;
    npy_intp total_offset;

    /* Get some properties from the dtype */
    names = PyDataType_NAMES(dtype);
    names_size = PyTuple_GET_SIZE(names);
    fields = PyDataType_FIELDS(dtype);

    /* Start at offset zero */
    total_offset = 0;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            return 0;
        }
        tup = PyDict_GetItem(fields, key);
        if (tup == NULL) {
            return 0;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            PyErr_Clear();
            return 0;
        }
        /* If this field doesn't follow the pattern, not a simple layout */
        if (total_offset != fld_offset) {
            return 0;
        }
        /* Get the next offset */
        total_offset += fld_dtype->elsize;
    }

    /*
     * If the itemsize doesn't match the final offset, it's
     * not a simple layout.
     */
    if (total_offset != dtype->elsize) {
        return 0;
    }

    /* It's a simple layout, since all the above tests passed */
    return 1;
}

/*
 * The general dtype repr function.
 */
static PyObject *
arraydescr_repr(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "__repr__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}
/*
 * The general dtype str function.
 */
static PyObject *
arraydescr_str(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy._core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "__str__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}

static PyObject *
arraydescr_richcompare(PyArray_Descr *self, PyObject *other, int cmp_op)
{
    PyArray_Descr *new = _convert_from_any(other, 0);
    if (new == NULL) {
        /* Cannot convert `other` to dtype */
        PyErr_Clear();
        Py_RETURN_NOTIMPLEMENTED;
    }

    npy_bool ret;
    switch (cmp_op) {
    case Py_LT:
        ret = !PyArray_EquivTypes(self, new) && PyArray_CanCastTo(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_LE:
        ret = PyArray_CanCastTo(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_EQ:
        ret = PyArray_EquivTypes(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_NE:
        ret = !PyArray_EquivTypes(self, new);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_GT:
        ret = !PyArray_EquivTypes(self, new) && PyArray_CanCastTo(new, self);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    case Py_GE:
        ret = PyArray_CanCastTo(new, self);
        Py_DECREF(new);
        return PyBool_FromLong(ret);
    default:
        Py_DECREF(new);
        Py_RETURN_NOTIMPLEMENTED;
    }
}

static int
descr_nonzero(PyObject *NPY_UNUSED(self))
{
    /* `bool(np.dtype(...)) == True` for all dtypes. Needed to override default
     * nonzero implementation, which checks if `len(object) > 0`. */
    return 1;
}

static PyNumberMethods descr_as_number = {
    .nb_bool = (inquiry)descr_nonzero,
};

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

static Py_ssize_t
descr_length(PyObject *self0)
{
    PyArray_Descr *self = (PyArray_Descr *)self0;

    if (PyDataType_HASFIELDS(self)) {
        return PyTuple_GET_SIZE(PyDataType_NAMES(self));
    }
    else {
        return 0;
    }
}

static PyObject *
descr_repeat(PyObject *self, Py_ssize_t length)
{
    PyObject *tup;
    PyArray_Descr *new;
    if (length < 0) {
        return PyErr_Format(PyExc_ValueError,
                "Array length must be >= 0, not %"NPY_INTP_FMT, (npy_intp)length);
    }
    tup = Py_BuildValue("O" NPY_SSIZE_T_PYFMT, self, length);
    if (tup == NULL) {
        return NULL;
    }
    new = _convert_from_any(tup, 0);
    Py_DECREF(tup);
    return (PyObject *)new;
}

static int
_check_has_fields(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        PyErr_Format(PyExc_KeyError, "There are no fields in dtype %S.", self);
        return -1;
    }
    else {
        return 0;
    }
}

static PyObject *
_subscript_by_name(_PyArray_LegacyDescr *self, PyObject *op)
{
    PyObject *obj = PyDict_GetItemWithError(self->fields, op);
    if (obj == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_KeyError,
                    "Field named %R not found.", op);
        }
        return NULL;
    }
    PyObject *descr = PyTuple_GET_ITEM(obj, 0);
    Py_INCREF(descr);
    return descr;
}

static PyObject *
_subscript_by_index(_PyArray_LegacyDescr *self, Py_ssize_t i)
{
    PyObject *name = PySequence_GetItem(self->names, i);
    PyObject *ret;
    if (name == NULL) {
        PyErr_Format(PyExc_IndexError,
                     "Field index %zd out of range.", i);
        return NULL;
    }
    ret = _subscript_by_name(self, name);
    Py_DECREF(name);
    return ret;
}

static npy_bool
_is_list_of_strings(PyObject *obj)
{
    int seqlen, i;
    if (!PyList_CheckExact(obj)) {
        return NPY_FALSE;
    }
    seqlen = PyList_GET_SIZE(obj);
    for (i = 0; i < seqlen; i++) {
        PyObject *item = PyList_GET_ITEM(obj, i);
        if (!PyUnicode_Check(item)) {
            return NPY_FALSE;
        }
    }

    return NPY_TRUE;
}

NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(_PyArray_LegacyDescr *self, PyObject *ind)
{
    int seqlen, i;
    PyObject *fields = NULL;
    PyObject *names = NULL;

    seqlen = PySequence_Size(ind);
    if (seqlen == -1) {
        return NULL;
    }

    fields = PyDict_New();
    if (fields == NULL) {
        goto fail;
    }
    names = PyTuple_New(seqlen);
    if (names == NULL) {
        goto fail;
    }

    for (i = 0; i < seqlen; i++) {
        PyObject *name;
        PyObject *tup;

        name = PySequence_GetItem(ind, i);
        if (name == NULL) {
            goto fail;
        }

        /* Let the names tuple steal a reference now, so we don't need to
         * decref name if an error occurs further on.
         */
        PyTuple_SET_ITEM(names, i, name);

        tup = PyDict_GetItemWithError(self->fields, name);
        if (tup == NULL) {
            if (!PyErr_Occurred()) {
                PyErr_SetObject(PyExc_KeyError, name);
            }
            goto fail;
        }

        /* disallow use of titles as index */
        if (PyTuple_Size(tup) == 3) {
            PyObject *title = PyTuple_GET_ITEM(tup, 2);
            int titlecmp = PyObject_RichCompareBool(title, name, Py_EQ);
            if (titlecmp < 0) {
                goto fail;
            }
            if (titlecmp == 1) {
                /* if title == name, we were given a title, not a field name */
                PyErr_SetString(PyExc_KeyError,
                            "cannot use field titles in multi-field index");
                goto fail;
            }
            if (PyDict_SetItem(fields, title, tup) < 0) {
                goto fail;
            }
        }
        /* disallow duplicate field indices */
        if (PyDict_Contains(fields, name)) {
            PyObject *msg = NULL;
            PyObject *fmt = PyUnicode_FromString(
                                   "duplicate field of name {!r}");
            if (fmt != NULL) {
                msg = PyObject_CallMethod(fmt, "format", "O", name);
                Py_DECREF(fmt);
            }
            PyErr_SetObject(PyExc_ValueError, msg);
            Py_XDECREF(msg);
            goto fail;
        }
        if (PyDict_SetItem(fields, name, tup) < 0) {
            goto fail;
        }
    }

    _PyArray_LegacyDescr *view_dtype = (_PyArray_LegacyDescr *)PyArray_DescrNewFromType(NPY_VOID);
    if (view_dtype == NULL) {
        goto fail;
    }
    view_dtype->elsize = self->elsize;
    view_dtype->names = names;
    view_dtype->fields = fields;
    view_dtype->flags = self->flags;
    return (PyArray_Descr *)view_dtype;

fail:
    Py_XDECREF(fields);
    Py_XDECREF(names);
    return NULL;
}

static PyObject *
descr_subscript(PyArray_Descr *self, PyObject *op)
{
    _PyArray_LegacyDescr *lself = (_PyArray_LegacyDescr *)self;
    if (_check_has_fields(self) < 0) {
        return NULL;
    }

    if (PyUnicode_Check(op)) {
        return _subscript_by_name(lself, op);
    }
    else if (_is_list_of_strings(op)) {
        return (PyObject *)arraydescr_field_subset_view(lself, op);
    }
    else {
        Py_ssize_t i = PyArray_PyIntAsIntp(op);
        if (error_converting(i)) {
            /* if converting to an int gives a type error, adjust the message */
            PyObject *err = PyErr_Occurred();
            if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {
                PyErr_SetString(PyExc_TypeError,
                        "Field key must be an integer field offset, "
                        "single field name, or list of field names.");
            }
            return NULL;
        }
        return _subscript_by_index(lself, i);
    }
}

static PySequenceMethods descr_as_sequence = {
    (lenfunc) descr_length,                  /* sq_length */
    (binaryfunc) NULL,                       /* sq_concat */
    (ssizeargfunc) descr_repeat,             /* sq_repeat */
    (ssizeargfunc) NULL,                     /* sq_item */
    (ssizessizeargfunc) NULL,                /* sq_slice */
    (ssizeobjargproc) NULL,                  /* sq_ass_item */
    (ssizessizeobjargproc) NULL,             /* sq_ass_slice */
    (objobjproc) NULL,                       /* sq_contains */
    (binaryfunc) NULL,                       /* sq_inplace_concat */
    (ssizeargfunc) NULL,                     /* sq_inplace_repeat */
};

static PyMappingMethods descr_as_mapping = {
    descr_length,                                /* mp_length*/
    (binaryfunc)descr_subscript,                 /* mp_subscript*/
    (objobjargproc)NULL,                         /* mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/


/*
 * NOTE: Since this is a MetaClass, the name has Full appended here, the
 *       correct name of the type is PyArrayDescr_Type.
 */
NPY_NO_EXPORT PyArray_DTypeMeta PyArrayDescr_TypeFull = {
    {{
        /* NULL represents `type`, this is set to DTypeMeta at import time */
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numpy.dtype",
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_dealloc = (destructor)arraydescr_dealloc,
        .tp_repr = (reprfunc)arraydescr_repr,
        .tp_as_number = &descr_as_number,
        .tp_as_sequence = &descr_as_sequence,
        .tp_as_mapping = &descr_as_mapping,
        .tp_str = (reprfunc)arraydescr_str,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_richcompare = (richcmpfunc)arraydescr_richcompare,
        .tp_methods = arraydescr_methods,
        .tp_members = arraydescr_members,
        .tp_getset = arraydescr_getsets,
        .tp_new = arraydescr_new,
    },},
    .singleton = NULL,
    .type_num = -1,
    .scalar_type = NULL,
    .flags = NPY_DT_ABSTRACT,
};
