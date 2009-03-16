#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

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
    if (isbuiltin(descr)) {
        return builtin_hash(descr, l);
    } else {
        return -1;
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
