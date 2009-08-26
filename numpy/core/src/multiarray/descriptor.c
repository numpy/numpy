/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "config.h"

#include "common.h"

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag);

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj)
{
    PyObject *dtypedescr;
    PyArray_Descr *new;
    int ret;

    dtypedescr = PyObject_GetAttrString(obj, "dtype");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    /* Understand basic ctypes */
    dtypedescr = PyObject_GetAttrString(obj, "_type_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            PyObject *length;
            length = PyObject_GetAttrString(obj, "_length_");
            PyErr_Clear();
            if (length) {
                /* derived type */
                PyObject *newtup;
                PyArray_Descr *derived;
                newtup = Py_BuildValue("NO", new, length);
                ret = PyArray_DescrConverter(newtup, &derived);
                Py_DECREF(newtup);
                if (ret == PY_SUCCEED) {
                    return derived;
                }
                PyErr_Clear();
                return NULL;
            }
            return new;
        }
        PyErr_Clear();
        return NULL;
    }
    /* Understand ctypes structures --
       bit-fields are not supported
       automatically aligns */
    dtypedescr = PyObject_GetAttrString(obj, "_fields_");
    PyErr_Clear();
    if (dtypedescr) {
        ret = PyArray_DescrAlignConverter(dtypedescr, &new);
        Py_DECREF(dtypedescr);
        if (ret == PY_SUCCEED) {
            return new;
        }
        PyErr_Clear();
    }
    return NULL;
}

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O", &dict)) {
        return NULL;
    }
    /* Decrement old reference (if any)*/
    Py_XDECREF(typeDict);
    typeDict = dict;
    /* Create an internal reference to it */
    Py_INCREF(dict);
    Py_INCREF(Py_None);
    return Py_None;
}

static int
_check_for_commastring(char *type, int len)
{
    int i;

    /* Check for ints at start of string */
    if ((type[0] >= '0' && type[0] <= '9') ||
        ((len > 1) && _chk_byteorder(type[0]) &&
         (type[1] >= '0' && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1) && (type[0] == '(' && type[1] == ')')) ||
        ((len > 3) && _chk_byteorder(type[0]) &&
         (type[1] == '(' && type[2] == ')'))) {
        return 1;
    }
    /* Check for presence of commas */
    for (i = 1; i < len; i++) {
        if (type[i] == ',') {
            return 1;
        }
    }
    return 0;
}

#undef _chk_byteorder

static PyArray_Descr *
_convert_from_tuple(PyObject *obj)
{
    PyArray_Descr *type, *res;
    PyObject *val;
    int errflag;

    if (PyTuple_GET_SIZE(obj) != 2) {
        return NULL;
    }
    if (!PyArray_DescrConverter(PyTuple_GET_ITEM(obj,0), &type)) {
        return NULL;
    }
    val = PyTuple_GET_ITEM(obj,1);
    /* try to interpret next item as a type */
    res = _use_inherit(type, val, &errflag);
    if (res || errflag) {
        Py_DECREF(type);
        if (res) {
            return res;
        }
        else {
            return NULL;
        }
    }
    PyErr_Clear();
    /*
     * We get here if res was NULL but errflag wasn't set
     * --- i.e. the conversion to a data-descr failed in _use_inherit
     */
    if (type->elsize == 0) {
        /* interpret next item as a typesize */
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));

        if (error_converting(itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid itemsize in generic type "\
                            "tuple");
            goto fail;
        }
        PyArray_DESCR_REPLACE(type);
        if (type->type_num == PyArray_UNICODE) {
            type->elsize = itemsize << 2;
        }
        else {
            type->elsize = itemsize;
        }
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to PyArray_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        PyArray_Descr *newdescr;

        if (!(PyArray_IntpConverter(val, &shape))
            || (shape.len > MAX_DIMS)) {
            PyDimMem_FREE(shape.ptr);
            PyErr_SetString(PyExc_ValueError,
                            "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /* If (type, 1) was given, it is equivalent to type...
           or (type, ()) was given it is equivalent to type... */
        if ((shape.len == 1 && shape.ptr[0] == 1 && PyNumber_Check(val))
            || (shape.len == 0 && PyTuple_Check(val))) {
            PyDimMem_FREE(shape.ptr);
            return type;
        }
        newdescr = PyArray_DescrNewFromType(PyArray_VOID);
        if (newdescr == NULL) {
            PyDimMem_FREE(shape.ptr);
            goto fail;
        }
        newdescr->elsize = type->elsize;
        newdescr->elsize *= PyArray_MultiplyList(shape.ptr, shape.len);
        PyDimMem_FREE(shape.ptr);
        newdescr->subarray = _pya_malloc(sizeof(PyArray_ArrayDescr));
        newdescr->subarray->base = type;
        newdescr->hasobject = type->hasobject;
        Py_INCREF(val);
        newdescr->subarray->shape = val;
        Py_XDECREF(newdescr->fields);
        Py_XDECREF(newdescr->names);
        newdescr->fields = NULL;
        newdescr->names = NULL;
        type = newdescr;
    }
    return type;

 fail:
    Py_XDECREF(type);
    return NULL;
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 */
static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    int n, i, totalsize;
    int ret;
    PyObject *fields, *item, *newobj;
    PyObject *name, *tup, *title;
    PyObject *nameslist;
    PyArray_Descr *new;
    PyArray_Descr *conv;
    int dtypeflags = 0;
    int maxalign = 0;


    n = PyList_GET_SIZE(obj);
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    totalsize = 0;
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        item = PyList_GET_ITEM(obj, i);
        if (!PyTuple_Check(item) || (PyTuple_GET_SIZE(item) < 2)) {
            goto fail;
        }
        name = PyTuple_GET_ITEM(item, 0);
        if (PyString_Check(name)) {
            title = NULL;
        }
        else if (PyTuple_Check(name)) {
            if (PyTuple_GET_SIZE(name) != 2) {
                goto fail;
            }
            title = PyTuple_GET_ITEM(name, 0);
            name = PyTuple_GET_ITEM(name, 1);
            if (!PyString_Check(name)) {
                goto fail;
            }
        }
        else {
            goto fail;
        }
        if (PyString_GET_SIZE(name)==0) {
            if (title == NULL) {
                name = PyString_FromFormat("f%d", i);
            }
            else {
                name = title;
                Py_INCREF(name);
            }
        }
        else {
            Py_INCREF(name);
        }
        PyTuple_SET_ITEM(nameslist, i, name);
        if (PyTuple_GET_SIZE(item) == 2) {
            ret = PyArray_DescrConverter(PyTuple_GET_ITEM(item, 1), &conv);
            if (ret == PY_FAIL) {
                PyObject_Print(PyTuple_GET_ITEM(item,1), stderr, 0);
            }
        }
        else if (PyTuple_GET_SIZE(item) == 3) {
            newobj = PyTuple_GetSlice(item, 1, 3);
            ret = PyArray_DescrConverter(newobj, &conv);
            Py_DECREF(newobj);
        }
        else {
            goto fail;
        }
        if (ret == PY_FAIL) {
            goto fail;
        }
        if ((PyDict_GetItem(fields, name) != NULL) ||
            (title &&
             (PyString_Check(title) || PyUnicode_Check(title)) &&
             (PyDict_GetItem(fields, title) != NULL))) {
            PyErr_SetString(PyExc_ValueError,
                            "two fields with the same name");
            goto fail;
        }
        dtypeflags |= (conv->hasobject & NPY_FROM_FIELDS);
        tup = PyTuple_New((title == NULL ? 2 : 3));
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         */
        if (title != NULL) {
            Py_INCREF(title);
            PyTuple_SET_ITEM(tup, 2, title);
            if (PyString_Check(title) || PyUnicode_Check(title)) {
                PyDict_SetItem(fields, title, tup);
            }
        }
        PyDict_SetItem(fields, name, tup);
        totalsize += conv->elsize;
        Py_DECREF(tup);
    }
    new = PyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->hasobject=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    return new;

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
    int n, i;
    int totalsize;
    PyObject *fields;
    PyArray_Descr *conv = NULL;
    PyArray_Descr *new;
    PyObject *key, *tup;
    PyObject *nameslist = NULL;
    int ret;
    int maxalign = 0;
    int dtypeflags = 0;

    n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    key = PyList_GET_ITEM(obj, n-1);
    if (PyString_Check(key) && PyString_GET_SIZE(key) == 0) {
        n = n - 1;
    }
    /* End ignore code.*/
    totalsize = 0;
    if (n == 0) {
        return NULL;
    }
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        tup = PyTuple_New(2);
        key = PyString_FromFormat("f%d", i);
        ret = PyArray_DescrConverter(PyList_GET_ITEM(obj, i), &conv);
        if (ret == PY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(key);
            goto fail;
        }
        dtypeflags |= (conv->hasobject & NPY_FROM_FIELDS);
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            maxalign = MAX(maxalign, _align);
        }
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));
        PyDict_SetItem(fields, key, tup);
        Py_DECREF(tup);
        PyTuple_SET_ITEM(nameslist, i, key);
        totalsize += conv->elsize;
    }
    new = PyArray_DescrNewFromType(PyArray_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->hasobject=dtypeflags;
    if (maxalign > 1) {
        totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    return new;

 fail:
    Py_DECREF(nameslist);
    Py_DECREF(fields);
    return NULL;
}


/*
 * comma-separated string
 * this is the format developed by the numarray records module
 * and implemented by the format parser in that module
 * this is an alternative implementation found in the _internal.py
 * file patterned after that one -- the approach is to try to convert
 * to a list (with tuples if any repeat information is present)
 * and then call the _convert_from_list)
 */
static PyArray_Descr *
_convert_from_commastring(PyObject *obj, int align)
{
    PyObject *listobj;
    PyArray_Descr *res;
    PyObject *_numpy_internal;

    if (!PyString_Check(obj)) {
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    listobj = PyObject_CallMethod(_numpy_internal, "_commastring", "O", obj);
    Py_DECREF(_numpy_internal);
    if (!listobj) {
        return NULL;
    }
    if (!PyList_Check(listobj) || PyList_GET_SIZE(listobj)<1) {
        PyErr_SetString(PyExc_RuntimeError, "_commastring is "  \
                        "not returning a list with len >= 1");
        return NULL;
    }
    if (PyList_GET_SIZE(listobj) == 1) {
        if (PyArray_DescrConverter(PyList_GET_ITEM(listobj, 0),
                                   &res) == NPY_FAIL) {
            res = NULL;
        }
    }
    else {
        res = _convert_from_list(listobj, align);
    }
    Py_DECREF(listobj);
    if (!res && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "invalid data-type");
        return NULL;
    }
    return res;
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
 */
static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag)
{
    PyArray_Descr *new;
    PyArray_Descr *conv;

    *errflag = 0;
    if (!PyArray_DescrConverter(newobj, &conv)) {
        return NULL;
    }
    *errflag = 1;
    new = PyArray_DescrNew(type);
    if (new == NULL) {
        goto fail;
    }
    if (new->elsize && new->elsize != conv->elsize) {
        PyErr_SetString(PyExc_ValueError,
                        "mismatch in size of old "\
                        "and new data-descriptor");
        goto fail;
    }
    new->elsize = conv->elsize;
    if (conv->names) {
        new->fields = conv->fields;
        Py_XINCREF(new->fields);
        new->names = conv->names;
        Py_XINCREF(new->names);
    }
    new->hasobject = conv->hasobject;
    Py_DECREF(conv);
    *errflag = 0;
    return new;

 fail:
    Py_DECREF(conv);
    return NULL;
}

/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
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
_use_fields_dict(PyObject *obj, int align)
{
    PyObject *_numpy_internal;
    PyArray_Descr *res;

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = (PyArray_Descr *)PyObject_CallMethod(_numpy_internal,
                                               "_usefields",
                                               "Oi", obj, align);
    Py_DECREF(_numpy_internal);
    return res;
}

static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
    PyArray_Descr *new;
    PyObject *fields = NULL;
    PyObject *names, *offsets, *descrs, *titles;
    int n, i;
    int totalsize;
    int maxalign = 0;
    int dtypeflags = 0;

    fields = PyDict_New();
    if (fields == NULL) {
        return (PyArray_Descr *)PyErr_NoMemory();
    }
    names = PyDict_GetItemString(obj, "names");
    descrs = PyDict_GetItemString(obj, "formats");
    if (!names || !descrs) {
        Py_DECREF(fields);
        return _use_fields_dict(obj, align);
    }
    n = PyObject_Length(names);
    offsets = PyDict_GetItemString(obj, "offsets");
    titles = PyDict_GetItemString(obj, "titles");
    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                        "all items in the dictionary must have" \
                        " the same length.");
        goto fail;
    }

    totalsize = 0;
    for (i = 0; i < n; i++) {
        PyObject *tup, *descr, *index, *item, *name, *off;
        int len, ret, _align = 1;
        PyArray_Descr *newdescr;

        /* Build item to insert (descr, offset, [title])*/
        len = 2;
        item = NULL;
        index = PyInt_FromLong(i);
        if (titles) {
            item=PyObject_GetItem(titles, index);
            if (item && item != Py_None) {
                len = 3;
            }
            else {
                Py_XDECREF(item);
            }
            PyErr_Clear();
        }
        tup = PyTuple_New(len);
        descr = PyObject_GetItem(descrs, index);
        ret = PyArray_DescrConverter(descr, &newdescr);
        Py_DECREF(descr);
        if (ret == PY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(index);
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)newdescr);
        if (align) {
            _align = newdescr->alignment;
            maxalign = MAX(maxalign,_align);
        }
        if (offsets) {
            long offset;
            off = PyObject_GetItem(offsets, index);
            offset = PyInt_AsLong(off);
            PyTuple_SET_ITEM(tup, 1, off);
            if (offset < totalsize) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid offset (must be "\
                                "ordered)");
                ret = PY_FAIL;
            }
            if (offset > totalsize) totalsize = offset;
        }
        else {
            if (align && _align > 1) {
                totalsize = ((totalsize + _align - 1)/_align)*_align;
            }
            PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(totalsize));
        }
        if (len == 3) {
            PyTuple_SET_ITEM(tup, 2, item);
        }
        name = PyObject_GetItem(names, index);
        Py_DECREF(index);
        if (!(PyString_Check(name) || PyUnicode_Check(name))) {
            PyErr_SetString(PyExc_ValueError,
                            "field names must be strings");
            ret = PY_FAIL;
        }

        /* Insert into dictionary */
        if (PyDict_GetItem(fields, name) != NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "name already used as a name or "\
                            "title");
            ret = PY_FAIL;
        }
        PyDict_SetItem(fields, name, tup);
        Py_DECREF(name);
        if (len == 3) {
            if ((PyString_Check(item) || PyUnicode_Check(item))
                && PyDict_GetItem(fields, item) != NULL) {
                PyErr_SetString(PyExc_ValueError,
                                "title already used as a "\
                                "name or title.");
                ret=PY_FAIL;
            }
            else {
                PyDict_SetItem(fields, item, tup);
            }
        }
        Py_DECREF(tup);
        if ((ret == PY_FAIL) || (newdescr->elsize == 0)) {
            goto fail;
        }
        dtypeflags |= (newdescr->hasobject & NPY_FROM_FIELDS);
        totalsize += newdescr->elsize;
    }

    new = PyArray_DescrNewFromType(PyArray_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = ((totalsize + maxalign - 1)/maxalign)*maxalign;
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    if (!PyTuple_Check(names)) {
        names = PySequence_Tuple(names);
    }
    else {
        Py_INCREF(names);
    }
    new->names = names;
    new->fields = fields;
    new->hasobject = dtypeflags;
    return new;

 fail:
    Py_XDECREF(fields);
    return NULL;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    PyArray_Descr *old;
    PyArray_Descr *new;

    old = PyArray_DescrFromType(type_num);
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
        return PY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}

/*NUMPY_API
 * Get typenum from an object -- None goes to PyArray_DEFAULT
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 * new reference in *at
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    char *type;
    int check_num = PyArray_NOTYPE + 10;
    int len;
    PyObject *item;
    int elsize = 0;
    char endian = '=';

    *at = NULL;
    /* default */
    if (obj == Py_None) {
        *at = PyArray_DescrFromType(PyArray_DEFAULT);
        return PY_SUCCEED;
    }
    if (PyArray_DescrCheck(obj)) {
        *at = (PyArray_Descr *)obj;
        Py_INCREF(*at);
        return PY_SUCCEED;
    }

    if (PyType_Check(obj)) {
        if (PyType_IsSubtype((PyTypeObject *)obj,
                             &PyGenericArrType_Type)) {
            *at = PyArray_DescrFromTypeObject(obj);
            if (*at) {
                return PY_SUCCEED;
            }
            else {
                return PY_FAIL;
            }
        }
        check_num = PyArray_OBJECT;
        if (obj == (PyObject *)(&PyInt_Type)) {
            check_num = PyArray_LONG;
        }
        else if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = PyArray_LONGLONG;
        }
        else if (obj == (PyObject *)(&PyFloat_Type)) {
            check_num = PyArray_DOUBLE;
        }
        else if (obj == (PyObject *)(&PyComplex_Type)) {
            check_num = PyArray_CDOUBLE;
        }
        else if (obj == (PyObject *)(&PyBool_Type)) {
            check_num = PyArray_BOOL;
        }
        else if (obj == (PyObject *)(&PyString_Type)) {
            check_num = PyArray_STRING;
        }
        else if (obj == (PyObject *)(&PyUnicode_Type)) {
            check_num = PyArray_UNICODE;
        }
        else if (obj == (PyObject *)(&PyBuffer_Type)) {
            check_num = PyArray_VOID;
        }
        else {
            *at = _arraydescr_fromobj(obj);
            if (*at) {
                return PY_SUCCEED;
            }
        }
        goto finish;
    }

    /* or a typecode string */
    if (PyString_Check(obj)) {
        /* Check for a string typecode. */
        type = PyString_AS_STRING(obj);
        len = PyString_GET_SIZE(obj);
        if (len <= 0) {
            goto fail;
        }
        /* check for commas present or first (or second) element a digit */
        if (_check_for_commastring(type, len)) {
            *at = _convert_from_commastring(obj, 0);
            if (*at) {
                return PY_SUCCEED;
            }
            return PY_FAIL;
        }
        check_num = (int) type[0];
        if ((char) check_num == '>' || (char) check_num == '<'
            || (char) check_num == '|' || (char) check_num == '=') {
            if (len <= 1) {
                goto fail;
            }
            endian = (char) check_num;
            type++; len--;
            check_num = (int) type[0];
            if (endian == '|') {
                endian = '=';
            }
        }
        if (len > 1) {
            elsize = atoi(type + 1);
            if (elsize == 0) {
                check_num = PyArray_NOTYPE+10;
            }
            /*
             * When specifying length of UNICODE
             * the number of characters is given to match
             * the STRING interface.  Each character can be
             * more than one byte and itemsize must be
             * the number of bytes.
             */
            else if (check_num == PyArray_UNICODELTR) {
                elsize <<= 2;
            }
            /* Support for generic processing c4, i4, f8, etc...*/
            else if ((check_num != PyArray_STRINGLTR)
                     && (check_num != PyArray_VOIDLTR)
                     && (check_num != PyArray_STRINGLTR2)) {
                check_num = PyArray_TypestrConvert(elsize, check_num);
                if (check_num == PyArray_NOTYPE) {
                    check_num += 10;
                }
                elsize = 0;
            }
        }
    }
    else if (PyTuple_Check(obj)) {
        /* or a tuple */
        *at = _convert_from_tuple(obj);
        if (*at == NULL){
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyList_Check(obj)) {
        /* or a list */
        *at = _convert_from_array_descr(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyDict_Check(obj)) {
        /* or a dictionary */
        *at = _convert_from_dict(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return PY_FAIL;
            }
            goto fail;
        }
        return PY_SUCCEED;
    }
    else if (PyArray_Check(obj)) {
        goto fail;
    }
    else {
        *at = _arraydescr_fromobj(obj);
        if (*at) {
            return PY_SUCCEED;
        }
        if (PyErr_Occurred()) {
            return PY_FAIL;
        }
        goto fail;
    }
    if (PyErr_Occurred()) {
        goto fail;
    }
    /*
      if (check_num == PyArray_NOTYPE) return PY_FAIL;
    */

 finish:
    if ((check_num == PyArray_NOTYPE + 10)
        || (*at = PyArray_DescrFromType(check_num)) == NULL) {
        /* Now check to see if the object is registered in typeDict */
        if (typeDict != NULL) {
            item = PyDict_GetItem(typeDict, obj);
            if (item) {
                return PyArray_DescrConverter(item, at);
            }
        }
        goto fail;
    }

    if (((*at)->elsize == 0) && (elsize != 0)) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->elsize = elsize;
    }
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && (*at)->byteorder != '|'
        && (*at)->byteorder != endian) {
        PyArray_DESCR_REPLACE(*at);
        (*at)->byteorder = endian;
    }
    return PY_SUCCEED;

 fail:
    PyErr_SetString(PyExc_TypeError, "data type not understood");
    *at = NULL;
    return PY_FAIL;
}

/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This let's us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base)
{
    PyArray_Descr *new = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);

    if (new == NULL) {
        return NULL;
    }
    /* Don't copy PyObject_HEAD part */
    memcpy((char *)new + sizeof(PyObject),
           (char *)base + sizeof(PyObject),
           sizeof(PyArray_Descr) - sizeof(PyObject));

    if (new->fields == Py_None) {
        new->fields = NULL;
    }
    Py_XINCREF(new->fields);
    Py_XINCREF(new->names);
    if (new->subarray) {
        new->subarray = _pya_malloc(sizeof(PyArray_ArrayDescr));
        memcpy(new->subarray, base->subarray, sizeof(PyArray_ArrayDescr));
        Py_INCREF(new->subarray->shape);
        Py_INCREF(new->subarray->base);
    }
    Py_XINCREF(new->typeobj);
    return new;
}

/*
 * should never be called for builtin-types unless
 * there is a reference-count problem
 */
static void
arraydescr_dealloc(PyArray_Descr *self)
{
    if (self->fields == Py_None) {
        fprintf(stderr, "*** Reference count error detected: \n" \
                "an attempt was made to deallocate %d (%c) ***\n",
                self->type_num, self->type);
        Py_INCREF(self);
        Py_INCREF(self);
        return;
    }
    Py_XDECREF(self->typeobj);
    Py_XDECREF(self->names);
    Py_XDECREF(self->fields);
    if (self->subarray) {
        Py_DECREF(self->subarray->shape);
        Py_DECREF(self->subarray->base);
        _pya_free(self->subarray);
    }
    self->ob_type->tp_free((PyObject *)self);
}

/*
 * we need to be careful about setting attributes because these
 * objects are pointed to by arrays that depend on them for interpreting
 * data.  Currently no attributes of data-type objects can be set
 * directly except names.
 */
static PyMemberDef arraydescr_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_Descr, typeobj), RO, NULL},
    {"kind",
        T_CHAR, offsetof(PyArray_Descr, kind), RO, NULL},
    {"char",
        T_CHAR, offsetof(PyArray_Descr, type), RO, NULL},
    {"num",
        T_INT, offsetof(PyArray_Descr, type_num), RO, NULL},
    {"byteorder",
        T_CHAR, offsetof(PyArray_Descr, byteorder), RO, NULL},
    {"itemsize",
        T_INT, offsetof(PyArray_Descr, elsize), RO, NULL},
    {"alignment",
        T_INT, offsetof(PyArray_Descr, alignment), RO, NULL},
    {"flags",
        T_UBYTE, offsetof(PyArray_Descr, hasobject), RO, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraydescr_subdescr_get(PyArray_Descr *self)
{
    if (self->subarray == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("OO", (PyObject *)self->subarray->base,
                         self->subarray->shape);
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_typestr_get(PyArray_Descr *self)
{
    char basic_ = self->kind;
    char endian = self->byteorder;
    int size = self->elsize;

    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (self->type_num == PyArray_UNICODE) {
        size >>= 2;
    }
    return PyString_FromFormat("%c%c%d", endian, basic_, size);
}

static PyObject *
arraydescr_typename_get(PyArray_Descr *self)
{
    int len;
    PyTypeObject *typeobj = self->typeobj;
    PyObject *res;
    char *s;
    /* fixme: not reentrant */
    static int prefix_len = 0;

    if (PyTypeNum_ISUSERDEF(self->type_num)) {
        s = strrchr(typeobj->tp_name, '.');
        if (s == NULL) {
            res = PyString_FromString(typeobj->tp_name);
        }
        else {
            res = PyString_FromStringAndSize(s + 1, strlen(s) - 1);
        }
        return res;
    }
    else {
        if (prefix_len == 0) {
            prefix_len = strlen("numpy.");
        }
        len = strlen(typeobj->tp_name);
        if (*(typeobj->tp_name + (len-1)) == '_') {
            len -= 1;
        }
        len -= prefix_len;
        res = PyString_FromStringAndSize(typeobj->tp_name+prefix_len, len);
    }
    if (PyTypeNum_ISFLEXIBLE(self->type_num) && self->elsize != 0) {
        PyObject *p;
        p = PyString_FromFormat("%d", self->elsize * 8);
        PyString_ConcatAndDel(&res, p);
    }
    return res;
}

static PyObject *
arraydescr_base_get(PyArray_Descr *self)
{
    if (self->subarray == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    Py_INCREF(self->subarray->base);
    return (PyObject *)(self->subarray->base);
}

static PyObject *
arraydescr_shape_get(PyArray_Descr *self)
{
    if (self->subarray == NULL) {
        return PyTuple_New(0);
    }
    if (PyTuple_Check(self->subarray->shape)) {
        Py_INCREF(self->subarray->shape);
        return (PyObject *)(self->subarray->shape);
    }
    return Py_BuildValue("(O)", self->subarray->shape);
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_descr_get(PyArray_Descr *self)
{
    PyObject *dobj, *res;
    PyObject *_numpy_internal;

    if (self->names == NULL) {
        /* get default */
        dobj = PyTuple_New(2);
        if (dobj == NULL) {
            return NULL;
        }
        PyTuple_SET_ITEM(dobj, 0, PyString_FromString(""));
        PyTuple_SET_ITEM(dobj, 1, arraydescr_protocol_typestr_get(self));
        res = PyList_New(1);
        if (res == NULL) {
            Py_DECREF(dobj);
            return NULL;
        }
        PyList_SET_ITEM(res, 0, dobj);
        return res;
    }

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
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
arraydescr_isbuiltin_get(PyArray_Descr *self)
{
    long val;
    val = 0;
    if (self->fields == Py_None) {
        val = 1;
    }
    if (PyTypeNum_ISUSERDEF(self->type_num)) {
        val = 2;
    }
    return PyInt_FromLong(val);
}

static int
_arraydescr_isnative(PyArray_Descr *self)
{
    if (self->names == NULL) {
        return PyArray_ISNBO(self->byteorder);
    }
    else {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
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
arraydescr_isnative_get(PyArray_Descr *self)
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
arraydescr_fields_get(PyArray_Descr *self)
{
    if (self->names == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return PyDictProxy_New(self->fields);
}

static PyObject *
arraydescr_hasobject_get(PyArray_Descr *self)
{
    PyObject *res;
    if (PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT)) {
        res = Py_True;
    }
    else {
        res = Py_False;
    }
    Py_INCREF(res);
    return res;
}

static PyObject *
arraydescr_names_get(PyArray_Descr *self)
{
    if (self->names == NULL) {
	Py_INCREF(Py_None);
	return Py_None;
    }
    Py_INCREF(self->names);
    return self->names;
}

static int
arraydescr_names_set(PyArray_Descr *self, PyObject *val)
{
    int N = 0;
    int i;
    PyObject *new_names;
    if (self->names == NULL) {
	PyErr_SetString(PyExc_ValueError, "there are no fields defined");
	return -1;
    }

    N = PyTuple_GET_SIZE(self->names);
    if (!PySequence_Check(val) || PyObject_Size((PyObject *)val) != N) {
	PyErr_Format(PyExc_ValueError, "must replace all names at once" \
		     " with a sequence of length %d", N);
	return -1;
    }
    /* Make sure all entries are strings */
    for (i = 0; i < N; i++) {
	PyObject *item;
	int valid = 1;
	item = PySequence_GetItem(val, i);
	valid = PyString_Check(item);
	Py_DECREF(item);
	if (!valid) {
	    PyErr_Format(PyExc_ValueError,
			 "item #%d of names is of type %s and not string",
			 i, item->ob_type->tp_name);
	    return -1;
	}
    }
    /* Update dictionary keys in fields */
    new_names = PySequence_Tuple(val);
    for (i = 0; i < N; i++) {
	PyObject *key;
	PyObject *item;
	PyObject *new_key;
	key = PyTuple_GET_ITEM(self->names, i);
	/* Borrowed reference to item */
	item = PyDict_GetItem(self->fields, key);
	Py_INCREF(item); /* Hold on to it even through DelItem */
	new_key = PyTuple_GET_ITEM(new_names, i);
	PyDict_DelItem(self->fields, key);
	PyDict_SetItem(self->fields, new_key, item);
	Py_DECREF(item); /* self->fields now holds reference */
    }

    /* Replace names */
    Py_DECREF(self->names);
    self->names = new_names;

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
        (getter)arraydescr_typename_get,
        NULL, NULL, NULL},
    {"base",
        (getter)arraydescr_base_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)arraydescr_shape_get,
        NULL, NULL, NULL},
    {"isbuiltin",
        (getter)arraydescr_isbuiltin_get,
        NULL, NULL, NULL},
    {"isnative",
        (getter)arraydescr_isnative_get,
        NULL, NULL, NULL},
    {"fields",
        (getter)arraydescr_fields_get,
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
arraydescr_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args, PyObject *kwds)
{
    PyObject *odescr;
    PyArray_Descr *descr, *conv;
    Bool align = FALSE;
    Bool copy = FALSE;
    static char *kwlist[] = {"dtype", "align", "copy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&",
                                     kwlist, &odescr,
                                     PyArray_BoolConverter, &align,
                                     PyArray_BoolConverter, &copy)) {
        return NULL;
    }
    if (align) {
        if (!PyArray_DescrAlignConverter(odescr, &conv)) {
            return NULL;
        }
    }
    else if (!PyArray_DescrConverter(odescr, &conv)) {
        return NULL;
    }
    /* Get a new copy of it unless it's already a copy */
    if (copy && conv->fields == Py_None) {
        descr = PyArray_DescrNew(conv);
        Py_DECREF(conv);
        conv = descr;
    }
    return (PyObject *)conv;
}


/* return a tuple of (callable object, args, state). */
static PyObject *
arraydescr_reduce(PyArray_Descr *self, PyObject *NPY_UNUSED(args))
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 3;
    PyObject *ret, *mod, *obj;
    PyObject *state;
    char endian;
    int elsize, alignment;

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy.core.multiarray");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    obj = PyObject_GetAttrString(mod, "dtype");
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, obj);
    if (PyTypeNum_ISUSERDEF(self->type_num) ||
        ((self->type_num == PyArray_VOID &&
          self->typeobj != &PyVoidArrType_Type))) {
        obj = (PyObject *)self->typeobj;
        Py_INCREF(obj);
    }
    else {
        elsize = self->elsize;
        if (self->type_num == PyArray_UNICODE) {
            elsize >>= 2;
        }
        obj = PyString_FromFormat("%c%d",self->kind, elsize);
    }
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(Nii)", obj, 0, 1));

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
    state = PyTuple_New(8);
    PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));
    PyTuple_SET_ITEM(state, 1, PyString_FromFormat("%c", endian));
    PyTuple_SET_ITEM(state, 2, arraydescr_subdescr_get(self));
    if (self->names) {
        Py_INCREF(self->names);
        Py_INCREF(self->fields);
        PyTuple_SET_ITEM(state, 3, self->names);
        PyTuple_SET_ITEM(state, 4, self->fields);
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
    PyTuple_SET_ITEM(state, 5, PyInt_FromLong(elsize));
    PyTuple_SET_ITEM(state, 6, PyInt_FromLong(alignment));
    PyTuple_SET_ITEM(state, 7, PyInt_FromLong(self->hasobject));
    PyTuple_SET_ITEM(ret, 2, state);
    return ret;
}

/* returns 1 if this data-type has an object portion
   used when setting the state because hasobject is not stored.
*/
static int
_descr_find_object(PyArray_Descr *self)
{
    if (self->hasobject || self->type_num == PyArray_OBJECT ||
        self->kind == 'O') {
        return NPY_OBJECT_DTYPE_FLAGS;
    }
    if (PyDescr_HASFIELDS(self)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(self->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                PyErr_Clear();
                return 0;
            }
            if (_descr_find_object(new)) {
                new->hasobject = NPY_OBJECT_DTYPE_FLAGS;
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
arraydescr_setstate(PyArray_Descr *self, PyObject *args)
{
    int elsize = -1, alignment = -1;
    int version = 3;
    char endian;
    PyObject *subarray, *fields, *names = NULL;
    int incref_names = 1;
    int dtypeflags = 0;

    if (self->fields == Py_None) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyTuple_GET_SIZE(args) != 1 ||
        !(PyTuple_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }
    switch (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0))) {
    case 8:
        if (!PyArg_ParseTuple(args, "(icOOOiii)", &version, &endian,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &dtypeflags)) {
            return NULL;
        }
        break;
    case 7:
        if (!PyArg_ParseTuple(args, "(icOOOii)", &version, &endian,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    case 6:
        if (!PyArg_ParseTuple(args, "(icOOii)", &version,
                    &endian, &subarray, &fields,
                    &elsize, &alignment)) {
            PyErr_Clear();
        }
        break;
    case 5:
        version = 0;
        if (!PyArg_ParseTuple(args, "(cOOii)",
                    &endian, &subarray, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    default:
        /* raise an error */
        version = -1;
    }

    /*
     * If we ever need another pickle format, increment the version
     * number. But we should still be able to handle the old versions.
     */
    if (version < 0 || version > 3) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return NULL;
    }

    if (version == 1 || version == 0) {
        if (fields != Py_None) {
            PyObject *key, *list;
            key = PyInt_FromLong(-1);
            list = PyDict_GetItem(fields, key);
            if (!list) {
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


    if ((fields == Py_None && names != Py_None) ||
        (names == Py_None && fields != Py_None)) {
        PyErr_Format(PyExc_ValueError,
                     "inconsistent fields and names");
        return NULL;
    }

    if (endian != '|' && PyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    self->byteorder = endian;
    if (self->subarray) {
        Py_XDECREF(self->subarray->base);
        Py_XDECREF(self->subarray->shape);
        _pya_free(self->subarray);
    }
    self->subarray = NULL;

    if (subarray != Py_None) {
        self->subarray = _pya_malloc(sizeof(PyArray_ArrayDescr));
        self->subarray->base = (PyArray_Descr *)PyTuple_GET_ITEM(subarray, 0);
        Py_INCREF(self->subarray->base);
        self->subarray->shape = PyTuple_GET_ITEM(subarray, 1);
        Py_INCREF(self->subarray->shape);
    }

    if (fields != Py_None) {
        Py_XDECREF(self->fields);
        self->fields = fields;
        Py_INCREF(fields);
        Py_XDECREF(self->names);
        self->names = names;
        if (incref_names) {
            Py_INCREF(names);
        }
    }

    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        self->elsize = elsize;
        self->alignment = alignment;
    }

    self->hasobject = dtypeflags;
    if (version < 3) {
        self->hasobject = _descr_find_object(self);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

/*NUMPY_API
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from PyArray_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    if PyDict_Check(obj) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if PyString_Check(obj) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if PyList_Check(obj) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}

/*NUMPY_API
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if PyDict_Check(obj) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if PyString_Check(obj) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if PyList_Check(obj) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter2(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return PY_FAIL;
    }
    return PY_SUCCEED;
}



 /*NUMPY_API
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
PyArray_DescrNewByteorder(PyArray_Descr *self, char newendian)
{
    PyArray_Descr *new;
    char endian;

    new = PyArray_DescrNew(self);
    endian = new->byteorder;
    if (endian != PyArray_IGNORE) {
        if (newendian == PyArray_SWAP) {
            /* swap byteorder */
            if PyArray_ISNBO(endian) {
                endian = PyArray_OPPBYTE;
            }
            else {
                endian = PyArray_NATBYTE;
            }
            new->byteorder = endian;
        }
        else if (newendian != PyArray_IGNORE) {
            new->byteorder = newendian;
        }
    }
    if (new->names) {
        PyObject *newfields;
        PyObject *key, *value;
        PyObject *newvalue;
        PyObject *old;
        PyArray_Descr *newdescr;
        Py_ssize_t pos = 0;
        int len, i;

        newfields = PyDict_New();
        /* make new dictionary with replaced PyArray_Descr Objects */
        while(PyDict_Next(self->fields, &pos, &key, &value)) {
	    if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyString_Check(key) ||
                !PyTuple_Check(value) ||
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
            PyDict_SetItem(newfields, key, newvalue);
            Py_DECREF(newvalue);
        }
        Py_DECREF(new->fields);
        new->fields = newfields;
    }
    if (new->subarray) {
        Py_DECREF(new->subarray->base);
        new->subarray->base = PyArray_DescrNewByteorder
            (self->subarray->base, newendian);
    }
    return new;
}


static PyObject *
arraydescr_newbyteorder(PyArray_Descr *self, PyObject *args)
{
    char endian=PyArray_SWAP;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_ByteorderConverter,
                          &endian)) {
        return NULL;
    }
    return (PyObject *)PyArray_DescrNewByteorder(self, endian);
}

static PyMethodDef arraydescr_methods[] = {
    /* for pickling */
    {"__reduce__",
        (PyCFunction)arraydescr_reduce, METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction)arraydescr_setstate, METH_VARARGS, NULL},
    {"newbyteorder",
        (PyCFunction)arraydescr_newbyteorder, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyObject *
arraydescr_str(PyArray_Descr *self)
{
    PyObject *sub;

    if (self->names) {
        PyObject *lst;
        lst = arraydescr_protocol_descr_get(self);
        if (!lst) {
            sub = PyString_FromString("<err>");
            PyErr_Clear();
        }
        else {
            sub = PyObject_Str(lst);
        }
        Py_XDECREF(lst);
        if (self->type_num != PyArray_VOID) {
            PyObject *p;
            PyObject *t=PyString_FromString("'");
            p = arraydescr_protocol_typestr_get(self);
            PyString_Concat(&p, t);
            PyString_ConcatAndDel(&t, p);
            p = PyString_FromString("(");
            PyString_ConcatAndDel(&p, t);
            PyString_ConcatAndDel(&p, PyString_FromString(", "));
            PyString_ConcatAndDel(&p, sub);
            PyString_ConcatAndDel(&p, PyString_FromString(")"));
            sub = p;
        }
    }
    else if (self->subarray) {
        PyObject *p;
        PyObject *t = PyString_FromString("(");
        PyObject *sh;
        p = arraydescr_str(self->subarray->base);
        if (!self->subarray->base->names && !self->subarray->base->subarray) {
            PyObject *t=PyString_FromString("'");
            PyString_Concat(&p, t);
            PyString_ConcatAndDel(&t, p);
            p = t;
        }
        PyString_ConcatAndDel(&t, p);
        PyString_ConcatAndDel(&t, PyString_FromString(","));
        if (!PyTuple_Check(self->subarray->shape)) {
            sh = Py_BuildValue("(O)", self->subarray->shape);
        }
        else {
            sh = self->subarray->shape;
            Py_INCREF(sh);
        }
        PyString_ConcatAndDel(&t, PyObject_Str(sh));
        Py_DECREF(sh);
        PyString_ConcatAndDel(&t, PyString_FromString(")"));
        sub = t;
    }
    else if (PyDataType_ISFLEXIBLE(self) || !PyArray_ISNBO(self->byteorder)) {
        sub = arraydescr_protocol_typestr_get(self);
    }
    else {
        sub = arraydescr_typename_get(self);
    }
    return sub;
}

static PyObject *
arraydescr_repr(PyArray_Descr *self)
{
    PyObject *sub, *s;
    s = PyString_FromString("dtype(");
    sub = arraydescr_str(self);
    if (!self->names && !self->subarray) {
        PyObject *t=PyString_FromString("'");
        PyString_Concat(&sub, t);
        PyString_ConcatAndDel(&t, sub);
        sub = t;
    }
    PyString_ConcatAndDel(&s, sub);
    sub = PyString_FromString(")");
    PyString_ConcatAndDel(&s, sub);
    return s;
}

static PyObject *
arraydescr_richcompare(PyArray_Descr *self, PyObject *other, int cmp_op)
{
    PyArray_Descr *new = NULL;
    PyObject *result = Py_NotImplemented;
    if (!PyArray_DescrCheck(other)) {
        if (PyArray_DescrConverter(other, &new) == PY_FAIL) {
            return NULL;
        }
    }
    else {
        new = (PyArray_Descr *)other;
        Py_INCREF(new);
    }
    switch (cmp_op) {
    case Py_LT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_LE:
        if (PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_EQ:
        if (PyArray_EquivTypes(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_NE:
        if (PyArray_EquivTypes(self, new))
            result = Py_False;
        else
            result = Py_True;
        break;
    case Py_GT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_GE:
        if (PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    default:
        result = Py_NotImplemented;
    }

    Py_XDECREF(new);
    Py_INCREF(result);
    return result;
}

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

static Py_ssize_t
descr_length(PyObject *self0)
{
    PyArray_Descr *self = (PyArray_Descr *)self0;

    if (self->names) {
        return PyTuple_GET_SIZE(self->names);
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
                            "Array length must be >= 0, not %"INTP_FMT,
                            length);
    }
    tup = Py_BuildValue("O" NPY_SSIZE_T_PYFMT, self, length);
    if (tup == NULL) {
        return NULL;
    }
    PyArray_DescrConverter(tup, &new);
    Py_DECREF(tup);
    return (PyObject *)new;
}

static PyObject *
descr_subscript(PyArray_Descr *self, PyObject *op)
{
    PyObject *retval;

    if (!self->names) {
        PyObject *astr;
        astr = arraydescr_str(self);
        PyErr_Format(PyExc_KeyError,
                     "There are no fields in dtype %s.",
                     PyString_AsString(astr));
        Py_DECREF(astr);
        return NULL;
    }
    if (PyString_Check(op) || PyUnicode_Check(op)) {
        PyObject *obj = PyDict_GetItem(self->fields, op);
        PyObject *descr;

        if (obj == NULL) {
            PyErr_Format(PyExc_KeyError,
                    "Field named \'%s\' not found.",
                    PyString_AsString(op));
            return NULL;
        }
        descr = PyTuple_GET_ITEM(obj, 0);
        Py_INCREF(descr);
        retval = descr;
    }
    else if (PyInt_Check(op)) {
        PyObject *name;
        int size = PyTuple_GET_SIZE(self->names);
        int value = PyArray_PyIntAsInt(op);

        if (PyErr_Occurred()) {
            return NULL;
        }
        if (value < 0) {
            value += size;
        }
        if (value < 0 || value >= size) {
            PyErr_Format(PyExc_IndexError,
                    "Field index out of range.");
            return NULL;
        }
        name = PyTuple_GET_ITEM(self->names, value);
        retval = descr_subscript(self, name);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Field key must be an integer, string, or unicode.");
        return NULL;
    }
    return retval;
}

static PySequenceMethods descr_as_sequence = {
    descr_length,
    (binaryfunc)NULL,
    descr_repeat,
    NULL, NULL,
    NULL,                                        /* sq_ass_item */
    NULL,                                        /* ssizessizeobjargproc sq_ass_slice */
    0,                                           /* sq_contains */
    0,                                           /* sq_inplace_concat */
    0,                                           /* sq_inplace_repeat */
};

static PyMappingMethods descr_as_mapping = {
    descr_length,                                /* mp_length*/
    (binaryfunc)descr_subscript,                 /* mp_subscript*/
    (objobjargproc)NULL,                         /* mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/

NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.dtype",                              /* tp_name */
    sizeof(PyArray_Descr),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraydescr_dealloc,             /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    (void *)0,                                  /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)arraydescr_repr,                  /* tp_repr */
    0,                                          /* tp_as_number */
    &descr_as_sequence,                         /* tp_as_sequence */
    &descr_as_mapping,                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)arraydescr_str,                   /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)arraydescr_richcompare,        /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    arraydescr_methods,                         /* tp_methods */
    arraydescr_members,                         /* tp_members */
    arraydescr_getsets,                         /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    arraydescr_new,                             /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
};
