#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

/*
 * How this works ? The hash is computed from a list which contains all the
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
static int _array_descr_walk_fields(PyObject* fields, PyObject* l);
static int _array_descr_builtin(PyArray_Descr* descr, PyObject *l);

/*
 * Return true if descr is a builtin type
 */
static int _is_array_descr_builtin(PyArray_Descr* descr)
{
        if (descr->fields != NULL && descr->fields != Py_None) {
                return 0;
        }
        if (descr->subarray != NULL) {
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

    /*
     * For builtin type, hash relies on : kind + byteorder + hasobject +
     * type_num + elsize + alignment
     */
    t = Py_BuildValue("(ccciii)", descr->kind, descr->byteorder,
            descr->hasobject, descr->type_num, descr->elsize,
            descr->alignment);

    for(i = 0; i < PyTuple_Size(t); ++i) {
        item = PyTuple_GetItem(t, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while computing builting hash");
            goto clean_t;
        }
        Py_INCREF(item);
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
static int _array_descr_walk_fields(PyObject* fields, PyObject* l)
{
    PyObject *key, *value, *foffset, *fdescr;
    int pos = 0;
    int st;

    while (PyDict_Next(fields, &pos, &key, &value)) {
        /*
         * For each field, add the key + descr + offset to l
         */

        /* XXX: are those checks necessary ? */
        if (!PyString_Check(key)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) key of dtype dict not a string ???");
            return -1;
        }
        if (!PyTuple_Check(value)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) value of dtype dict not a dtype ???");
            return -1;
        }
        if (PyTuple_Size(value) < 2) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Less than 2 items in dtype dict ???");
            return -1;
        }
        Py_INCREF(key);
        PyList_Append(l, key);

        fdescr = PyTuple_GetItem(value, 0);
        if (!PyArray_DescrCheck(fdescr)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) First item in compound dtype tuple not a descr ???");
            return -1;
        } else {
            Py_INCREF(fdescr);
            st = _array_descr_walk((PyArray_Descr*)fdescr, l);
            Py_DECREF(fdescr);
            if (st) {
                return -1;
            }
        }

        foffset = PyTuple_GetItem(value, 1);
        if (!PyInt_Check(foffset)) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Second item in compound dtype tuple not an int ???");
            return -1;
        } else {
            Py_INCREF(foffset);
            PyList_Append(l, foffset);
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
    if ( !PyTuple_Check(adescr->shape)) {
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Shape of subarray dtype an tuple ???");
        return -1;
    }

    for(i = 0; i < PyTuple_Size(adescr->shape); ++i) {
        item = PyTuple_GetItem(adescr->shape, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while getting shape item of subarray dtype ???");
            return -1;
        }
        Py_INCREF(item);
        PyList_Append(l, item);
    }

    Py_INCREF(adescr->base);
    st = _array_descr_walk(adescr->base, l);
    Py_DECREF(adescr->base);

    return st;
}

/*
 * 'Root' function to walk into a dtype. May be call recursively
 */
static int _array_descr_walk(PyArray_Descr* descr, PyObject *l)
{
    int st;

    if (_is_array_descr_builtin(descr)) {
        return _array_descr_builtin(descr, l);
    } else {
        if(descr->fields != NULL && descr->fields != Py_None) {
            if (!PyDict_Check(descr->fields)) {
                PyErr_SetString(PyExc_SystemError,
                        "(Hash) fields is not a dict ???");
                return -1;
            }
            st = _array_descr_walk_fields(descr->fields, l);
            if (st) {
                return -1;
            }
        }
        if(descr->subarray != NULL) {
            st = _array_descr_walk_subarray(descr->subarray, l);
            if (st) {
                return -1;
            }
        }
    }

    return 0;
}

/*
 * Return 0 if successfull
 *
 * -1: memory-like error
 * -2: hashing error (bug)
 * -3: unexpected error
 */
static int _PyArray_DescrHashImp(PyArray_Descr *descr, long *hash)
{
    PyObject *l, *tl, *item;
    Py_ssize_t i;
    int st;

    l = PyList_New(0);
    if (l == NULL) {
        return -1;
    }

    st = _array_descr_walk(descr, l);
    if (st) {
        goto clean_l;
    }

    /*
     * Convert the list to tuple and compute the tuple hash using python
     * builtin function
     */
    tl = PyTuple_New(PyList_Size(l));
    for(i = 0; i < PyList_Size(l); ++i) {
        item = PyList_GetItem(l, i);
        if (item == NULL) {
            PyErr_SetString(PyExc_SystemError,
                    "(Hash) Error while translating the list into a tuple " \
                    "(NULL item)");
            goto clean_tl;
        }
        PyTuple_SetItem(tl, i, item);
    }

    *hash = PyObject_Hash(tl);
    if (*hash == -1) {
        /* XXX: does PyObject_Hash set an exception on failure ? */
#if 0
        PyErr_SetString(PyExc_SystemError,
                "(Hash) Error while hashing final tuple");
#endif
        goto clean_tl;
    }
    Py_DECREF(tl);
    Py_DECREF(l);

    return 0;

clean_tl:
    Py_DECREF(tl);
clean_l:
    Py_DECREF(l);
    return -1;
}

long PyArray_DescrHash(PyObject* odescr)
{
    PyArray_Descr *descr;
    int st;
    long hash;

    if (!PyArray_DescrCheck(odescr)) {
        PyErr_SetString(PyExc_ValueError,
                "PyArray_DescrHash argument must be a type descriptor");
        return -1;
    }
    descr = (PyArray_Descr*)odescr;

    st = _PyArray_DescrHashImp(descr, &hash);
    if (st) {
        return -1;
    }

    return hash;
}
