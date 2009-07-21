/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */
#include <Python.h>

#include "numpy/ndarrayobject.h"

/*
 * TODO:
 *  - Handle mode
 *  - Handle any dtype
 *  - Handle Object type
 */
static int copy_double(PyArrayIterObject *itx, PyArrayNeighborhoodIterObject *niterx,
        npy_intp *bounds,
        PyObject **out)
{
    npy_intp i, j;
    double *ptr;
    npy_intp odims[NPY_MAXDIMS];
    PyArrayObject *aout;

    /* For each point in itx, copy the current neighborhood into an array which
     * is appended at the output list */
    for(i = 0; i < itx->size; ++i) {
        PyArrayNeighborhoodIter_ResetMirror(niterx);

        for(j = 0; j < itx->ao->nd; ++j) {
            odims[0] = bounds[2 * j + 1] - bounds[2 * j] + 1;
        }
        aout = (PyArrayObject*)PyArray_SimpleNew(itx->ao->nd, odims, NPY_DOUBLE);
        if (aout == NULL) {
            return -1;
        }

        ptr = (double*)aout->data;

        for(j = 0; j < niterx->size; ++j) {
            *ptr = *((double*)niterx->dataptr);
            PyArrayNeighborhoodIter_NextMirror(niterx);
            ptr += 1;
        }

        Py_INCREF(aout);
        PyList_Append(*out, (PyObject*)aout);
        Py_DECREF(aout);
        PyArray_ITER_NEXT(itx);
    }

    return 0;
}

static PyObject*
test_neighborhood_iterator(PyObject* NPY_UNUSED(self), PyObject* args)
{
    PyObject *x, *c, *out, *b;
    PyArrayObject *ax;
    PyArrayIterObject *itx;
    int i, typenum;
    npy_intp bounds[NPY_MAXDIMS*2];
    PyArrayNeighborhoodIterObject *niterx;
    PyArrayNeighborhoodIterMode mode;

    if (!PyArg_ParseTuple(args, "OOO", &x, &b, &c)) {
        return NULL;
    }

    if (!PySequence_Check(b)) {
        return NULL;
    }

    typenum = PyArray_ObjectType(x, 0);
    typenum = PyArray_ObjectType(c, typenum);

    ax = (PyArrayObject*)PyArray_FromObject(x, typenum, 1, 10);
    if (ax == NULL) {
        printf("Bleh\n");
        return NULL;
    }
    if (PySequence_Size(b) != 2 * ax->nd) {
        PyErr_SetString(PyExc_ValueError,
                "bounds sequence size not compatible with x input");
        return NULL;
    }

    out = PyList_New(0);
    if (out == NULL) {
        printf("Bleh\n");
        return NULL;
    }

    itx = (PyArrayIterObject*)PyArray_IterNew(x);
    if (itx == NULL) {
        printf("bleh\n");
        return NULL;
    }

    /* Compute boundaries for the neighborhood iterator */
    for(i = 0; i < 2 * ax->nd; ++i) {
        PyObject* bound;
        bound = PySequence_GetItem(b, i);
        if (!PyInt_Check(bound)) {
            PyErr_SetString(PyExc_ValueError, "bound not long");
            return NULL;
        }
        bounds[i] = PyInt_AsLong(bound);
        Py_DECREF(bound);
    }

    /* Create the neighborhood iterator */
    mode.mode = NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING;
    mode.constant = NULL;
    niterx = (PyArrayNeighborhoodIterObject*)PyArray_NeighborhoodIterNew(
                    (PyArrayIterObject*)itx, bounds, &mode);
    if (niterx == NULL) {
        printf("bleh\n");
        return NULL;
    }

    switch (typenum) {
        case NPY_DOUBLE:
            copy_double(itx, niterx, bounds, &out);
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Type not supported");
            return NULL;
    }

    Py_DECREF((PyArrayIterObject*)niterx);
    Py_DECREF((PyArrayIterObject*)itx);

    Py_DECREF(ax);

    return out;
}

static PyMethodDef Multiarray_TestsMethods[] = {
    {"test_neighborhood_iterator",  test_neighborhood_iterator, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initmultiarray_tests(void)
{
    PyObject *m;

    m = Py_InitModule("multiarray_tests", Multiarray_TestsMethods);
    if (m == NULL) return;

    import_array();

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load umath_tests module.");
    }
}
