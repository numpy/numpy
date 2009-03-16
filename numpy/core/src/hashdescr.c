#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

/* Main function for hasing, walk recursively for non-builtin types */
static int _hash_imp(PyArray_Descr* descr, PyObject *l);

static int builtin_hash(PyArray_Descr* descr, PyObject *l)
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
 * Walk inside the fields and add every item which will be used for hashing in
 * the list l
 *
 * May recursively call into _hash_imp
 *
 * Return 0 on success
 * Return -3 for unexpected error
 */
static int walk_fields(PyObject* fields, PyObject* l)
{
    PyObject *key, *value, *foffset, *fdescr;
    int pos = 0;

    while (PyDict_Next(fields, &pos, &key, &value)) {
        /*
         * For each field, add the key + descr + offset
         */

        /* XXX: are those checks necessary ? */
        if (!PyString_Check(key)) {
            return -3;
        }
        if (!PyTuple_Check(value)) {
            return -3;
        }
        if (PyTuple_Size(value) < 2) {
            return -3;
        }
        Py_INCREF(key);
        PyList_Append(l, key);

        fdescr = PyTuple_GetItem(value, 0);
        if (!PyArray_DescrCheck(fdescr)) {
            return -3;
        } else {
            Py_INCREF(fdescr);
            _hash_imp((PyArray_Descr*)fdescr, l);
            Py_DECREF(fdescr);
        }

        foffset = PyTuple_GetItem(value, 1);
        if (!PyInt_Check(foffset)) {
            return -3;
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
static int walk_subarray(PyArray_ArrayDescr* adescr, PyObject *l)
{
    PyObject *item;
    Py_ssize_t i;
    int st;

    /*
     * Add shape and descr itself to the list of object to hash
     */
    if ( !PyTuple_Check(adescr->shape)) {
        return -3;
    }

    for(i = 0; i < PyTuple_Size(adescr->shape); ++i) {
        item = PyTuple_GetItem(adescr->shape, i);
        if (item == NULL) {
            return -1;
        }
        Py_INCREF(item);
        PyList_Append(l, item);
    }

    Py_INCREF(adescr->base);
    st = _hash_imp(adescr->base, l);
    Py_DECREF(adescr->base);

    return st;
}

/*
 * Return true if descr is a builtin type
 */
static int isbuiltin(PyArray_Descr* descr)
{
        if (descr->fields != NULL && descr->fields != Py_None) {
                return 0;
        }
        if (descr->subarray != NULL) {
                return 0;
        }
        return 1;
}

static int _hash_imp(PyArray_Descr* descr, PyObject *l)
{
    int st;

    if (isbuiltin(descr)) {
        return builtin_hash(descr, l);
    } else {
        if(descr->fields != NULL && descr->fields != Py_None) {
            if (!PyDict_Check(descr->fields)) {
                return -3;
            }
            st = walk_fields(descr->fields, l);
            if (st) {
                printf("Error while walking fields\n");
                return -3;
            }
        }
        if(descr->subarray != NULL) {
            st = walk_subarray(descr->subarray, l);
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
    int st = -3;

    l = PyList_New(0);
    if (l == NULL) {
        return -1;
    }

    st = _hash_imp(descr, l);
    if (st) {
        printf("Error while computing hash\n");
        st = -3;
        goto clean_l;
    }

    /*
     * Convert the list to tuple
     */
    tl = PyTuple_New(PyList_Size(l));
    for(i = 0; i < PyList_Size(l); ++i) {
        item = PyList_GetItem(l, i);
        if (item == NULL) {
            st = -3;
            goto clean_tl;
        }
        PyTuple_SetItem(tl, i, item);
    }

    *hash = PyObject_Hash(tl);
    if (*hash == -1) {
        st = -2;
        goto clean_tl;
    }
    Py_DECREF(tl);
    Py_DECREF(l);

    return 0;

clean_tl:
    Py_DECREF(tl);
clean_l:
    Py_DECREF(l);
    return st;
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
