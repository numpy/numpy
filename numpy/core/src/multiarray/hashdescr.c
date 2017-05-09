#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"

#include "npy_pycompat.h"

#include "hashdescr.h"

/*
 * How does this work ? The hash is computed from a list which contains all the
 * information specific to a type. The hard work is to build the list
 * (_array_descr_walk). The list is built as follows:
 *      * If the dtype is builtin (no fields, no subarray), then the list
 *      contains 6 items which uniquely define one dtype (_array_descr_builtin)
 *      * If the dtype is a compound array, one walk on each field. For each
 *      field, we append title, names, offset to the final list used for
 *      hashing, and then append the list recursively built for each
 *      corresponding dtype (_array_descr_walk_fields)
 *      * If the dtype is a subarray, one adds the shape tuple to the list, and
 *      then append the list recursively built for each corresponding dtype
 *      (_array_descr_walk_subarray)
 *
 */

static int _is_array_descr_builtin(PyArray_Descr* descr);
static int _array_descr_walk(PyArray_Descr* descr, PyObject *l);
static int _array_descr_walk_fields(PyObject *names, PyObject* fields, PyObject* l);
static int _array_descr_builtin(PyArray_Descr* descr, PyObject *l);

/*
 * normalize endian character: always return 'I', '<' or '>'
 */
static char _normalize_byteorder(char byteorder)
{
   switch(byteorder) {
       case '=':
           if (PyArray_GetEndianness() == NPY_CPU_BIG) {
               return '>';
           }
           else {
               return '<';
           }
       default:
           return byteorder;
   }
}

/*
 * Return true if descr is a builtin type
 */
static int _is_array_descr_builtin(PyArray_Descr* descr)
{
    if (descr->fields != NULL && descr->fields != Py_None) {
        return 0;
    }
    if (PyDataType_HASSUBARRAY(descr)) {
        return 0;
    }
    return 1;
}

/*
 * Add to l all the items which uniquely define a builtin type
 */
static int _array_descr_builtin(PyArray_Descr* descr, PyObject *l)
{
    Py_ssize_t i;
    PyObject *t, *item;
    char nbyteorder = _normalize_byteorder(descr->byteorder);

    /*
     * For builtin type, hash relies on : kind + byteorder + flags +
     * type_num + elsize + alignment
     */
    t = Py_BuildValue("(cccii)", descr->kind, nbyteorder,
            descr->flags, descr->elsize, descr->alignment);

    for(i = 0; i < PyTuple_Size(t); ++i) {
        item = PyTuple_GetItem(t, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while computing builting hash");
            goto clean_t;
        }
        PyList_Append(l, item);
    }

    Py_DECREF(t);
    return 0;

clean_t:
    Py_DECREF(t);
    return -1;
}

/*
 * Walk inside the fields and add every item which will be used for hashing
 * into the list l
 *
 * Return 0 on success
 */
static int _array_descr_walk_fields(PyObject *names, PyObject* fields, PyObject* l)
{
    PyObject *key, *value, *foffset, *fdescr, *ftitle;
    Py_ssize_t pos = 0;
    int st;

    if (!PyTuple_Check(names)) {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) names is not a tuple ???");
        return -1;
    }
    if (!PyDict_Check(fields)) {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) fields is not a dict ???");
        return -1;
    }

    for (pos = 0; pos < PyTuple_GET_SIZE(names); pos++) {
        /*
         * For each field, add the key + descr + offset to l
         */
        key = PyTuple_GET_ITEM(names, pos);
        value = PyDict_GetItem(fields, key);
        /* XXX: are those checks necessary ? */
        if (value == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) names and fields inconsistent ???");
            return -1;
        }
        if (!PyUString_Check(key)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) key of dtype dict not a string ???");
            return -1;
        }
        if (!PyTuple_Check(value)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) value of dtype dict not a dtype ???");
            return -1;
        }
        if (PyTuple_GET_SIZE(value) < 2) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Less than 2 items in dtype dict ???");
            return -1;
        }
        PyList_Append(l, key);

        fdescr = PyTuple_GET_ITEM(value, 0);
        if (!PyArray_DescrCheck(fdescr)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) First item in compound dtype tuple not a descr ???");
            return -1;
        }
        else {
            Py_INCREF(fdescr);
            st = _array_descr_walk((PyArray_Descr*)fdescr, l);
            Py_DECREF(fdescr);
            if (st) {
                return -1;
            }
        }

        foffset = PyTuple_GET_ITEM(value, 1);
        if (!PyInt_Check(foffset)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Second item in compound dtype tuple not an int ???");
            return -1;
        }
        else {
            PyList_Append(l, foffset);
        }

        if (PyTuple_GET_SIZE(value) > 2) {
            ftitle = PyTuple_GET_ITEM(value, 2);
            PyList_Append(l, ftitle);
        }
    }

    return 0;
}

/*
 * Walk into subarray, and add items for hashing in l
 *
 * Return 0 on success
 */
static int _array_descr_walk_subarray(PyArray_ArrayDescr* adescr, PyObject *l)
{
    PyObject *item;
    Py_ssize_t i;
    int st;

    /*
     * Add shape and descr itself to the list of object to hash
     */
    if (PyTuple_Check(adescr->shape)) {
        for(i = 0; i < PyTuple_Size(adescr->shape); ++i) {
            item = PyTuple_GetItem(adescr->shape, i);
            if (item == NULL) {
                PyErr_SetString(PyExc_SystemError,
                        "(Hash) Error while getting shape item of subarray dtype ???");
                return -1;
            }
            PyList_Append(l, item);
        }
    }
    else if (PyInt_Check(adescr->shape)) {
        PyList_Append(l, adescr->shape);
    }
    else {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Shape of subarray dtype neither a tuple or int ???");
        return -1;
    }

    Py_INCREF(adescr->base);
    st = _array_descr_walk(adescr->base, l);
    Py_DECREF(adescr->base);

    return st;
}

/*
 * 'Root' function to walk into a dtype. May be called recursively
 */
static int _array_descr_walk(PyArray_Descr* descr, PyObject *l)
{
    int st;

    if (_is_array_descr_builtin(descr)) {
        return _array_descr_builtin(descr, l);
    }
    else {
        if(descr->fields != NULL && descr->fields != Py_None) {
            st = _array_descr_walk_fields(descr->names, descr->fields, l);
            if (st) {
                return -1;
            }
        }
        if(PyDataType_HASSUBARRAY(descr)) {
            st = _array_descr_walk_subarray(descr->subarray, l);
            if (st) {
                return -1;
            }
        }
    }

    return 0;
}

/*
 * Return 0 if successful
 */
static int _PyArray_DescrHashImp(PyArray_Descr *descr, npy_hash_t *hash)
{
    PyObject *l, *tl;
    int st;

    l = PyList_New(0);
    if (l == NULL) {
        return -1;
    }

    st = _array_descr_walk(descr, l);
    if (st) {
        Py_DECREF(l);
        return -1;
    }

    /*
     * Convert the list to tuple and compute the tuple hash using python
     * builtin function
     */
    tl = PyList_AsTuple(l);
    Py_DECREF(l);
    if (tl == NULL)
        return -1;

    *hash = PyObject_Hash(tl);
    Py_DECREF(tl);
    if (*hash == -1) {
        /* XXX: does PyObject_Hash set an exception on failure ? */
#if 0
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Error while hashing final tuple");
#endif
        return -1;
    }

    return 0;
}

NPY_NO_EXPORT npy_hash_t
PyArray_DescrHash(PyObject* odescr)
{
    PyArray_Descr *descr;
    int st;

    if (!PyArray_DescrCheck(odescr)) {
        PyErr_SetString(PyExc_ValueError,
                "PyArray_DescrHash argument must be a type descriptor");
        return -1;
    }
    descr = (PyArray_Descr*)odescr;

    if (descr->hash == -1) {
        st = _PyArray_DescrHashImp(descr, &descr->hash);
        if (st) {
            return -1;
        }
    }

    return descr->hash;
}
