/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "arrayobject.h"

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

    if (self->names) {
        if (PyString_Check(op) || PyUnicode_Check(op)) {
            PyObject *obj = PyDict_GetItem(self->fields, op);
            if (obj != NULL) {
                PyObject *descr = PyTuple_GET_ITEM(obj, 0);
                Py_INCREF(descr);
                return descr;
            }
            else {
                PyErr_Format(PyExc_KeyError,
                             "field named \'%s\' not found.",
                             PyString_AsString(op));
            }
        }
        else {
            PyObject *name;
            int value = PyArray_PyIntAsInt(op);
            if (!PyErr_Occurred()) {
                int size = PyTuple_GET_SIZE(self->names);
                if (value < 0) {
                    value += size;
                }
                if (value < 0 || value >= size) {
                    PyErr_Format(PyExc_IndexError,
                                 "0<=index<%d not %d",
                                 size, value);
                    return NULL;
                }
                name = PyTuple_GET_ITEM(self->names, value);
                return descr_subscript(self, name);
            }
        }
        PyErr_SetString(PyExc_ValueError,
                        "only integers, strings or unicode values "
                        "allowed for getting fields.");
    }
    else {
        PyObject *astr;
        astr = arraydescr_str(self);
        PyErr_Format(PyExc_KeyError,
                     "there are no fields in dtype %s.",
                     PyString_AsString(astr));
        Py_DECREF(astr);
    }
    return NULL;
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
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
    "numpy.dtype",                               /* tp_name */
    sizeof(PyArray_Descr),                       /* tp_basicsize */
    0,                                           /* tp_itemsize */
    /* methods */
    (destructor)arraydescr_dealloc,              /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    (reprfunc)arraydescr_repr,                   /* tp_repr */
    0,                                           /* tp_as_number */
    &descr_as_sequence,                          /* tp_as_sequence */
    &descr_as_mapping,                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call */
    (reprfunc)arraydescr_str,                    /* tp_str */
    0,                                           /* tp_getattro */
    0,                                           /* tp_setattro */
    0,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                          /* tp_flags */
    0,                                           /* tp_doc */
    0,                                           /* tp_traverse */
    0,                                           /* tp_clear */
    (richcmpfunc)arraydescr_richcompare,         /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    arraydescr_methods,                          /* tp_methods */
    arraydescr_members,                          /* tp_members */
    arraydescr_getsets,                          /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    0,                                           /* tp_init */
    0,                                           /* tp_alloc */
    arraydescr_new,                              /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */

#ifdef COUNT_ALLOCS
    /* these must be last and never explicitly initialized */
    0,                                           /* tp_allocs */
    0,                                           /* tp_frees */
    0,                                           /* tp_maxalloc */
    0,                                           /* tp_prev */
    0,                                           /* *tp_next */
#endif
};
