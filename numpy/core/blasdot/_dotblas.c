/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"

#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#ifndef CBLAS_HEADER
#define CBLAS_HEADER "cblas.h"
#endif
#include CBLAS_HEADER

#include <assert.h>
#include <limits.h>
#include <stdio.h>


static char module_doc[] =
    "This module provides a BLAS optimized\n"
    "matrix multiply, inner product and dot for numpy arrays";


static PyArray_DotFunc *oldFunctions[NPY_NTYPES];


/*
 * Helper: call appropriate BLAS dot function for typenum.
 * Strides are NumPy strides.
 */
static void
blas_dot(int typenum, npy_intp n,
         void *a, npy_intp stridea, void *b, npy_intp strideb, void *res)
{
    PyArray_DotFunc *dot;

    dot = oldFunctions[typenum];
    assert(dot != NULL);
    dot(a, stridea, b, strideb, res, n, NULL);
}


static const double oneD[2] = {1.0, 0.0}, zeroD[2] = {0.0, 0.0};
static const float oneF[2] = {1.0, 0.0}, zeroF[2] = {0.0, 0.0};


/*
 * Initialize oldFunctions table.
 */
static void
init_oldFunctions(void)
{
    PyArray_Descr *descr;
    int i;

    /* Initialise the array of dot functions */
    for (i = 0; i < NPY_NTYPES; i++)
        oldFunctions[i] = NULL;

    /* index dot functions we want to use here */
    descr = PyArray_DescrFromType(NPY_FLOAT);
    oldFunctions[NPY_FLOAT] = descr->f->dotfunc;

    descr = PyArray_DescrFromType(NPY_DOUBLE);
    oldFunctions[NPY_DOUBLE] = descr->f->dotfunc;

    descr = PyArray_DescrFromType(NPY_CFLOAT);
    oldFunctions[NPY_CFLOAT] = descr->f->dotfunc;

    descr = PyArray_DescrFromType(NPY_CDOUBLE);
    oldFunctions[NPY_CDOUBLE] = descr->f->dotfunc;
}


typedef enum {_scalar, _column, _row, _matrix} MatrixShape;


/*
 * vdot(a,b)
 *
 * Returns the dot product of a and b for scalars and vectors of
 * floating point and complex types.  The first argument, a, is conjugated.
 */
static PyObject *dotblas_vdot(PyObject *NPY_UNUSED(dummy), PyObject *args) {
    PyObject *op1, *op2;
    PyArrayObject *ap1 = NULL, *ap2  = NULL, *ret = NULL;
    int l;
    int typenum;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Multiplies op1 and op2, each of which must be vector.
     */
    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    type = PyArray_DescrFromType(typenum);
    Py_INCREF(type);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, type, 0, 0, 0, NULL);
    if (ap1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }
    op1 = PyArray_Flatten(ap1, 0);
    if (op1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;

    ap2 = (PyArrayObject *)PyArray_FromAny(op2, type, 0, 0, 0, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    op2 = PyArray_Flatten(ap2, 0);
    if (op2 == NULL) {
        goto fail;
    }
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;

    if (typenum != NPY_FLOAT && typenum != NPY_DOUBLE &&
        typenum != NPY_CFLOAT && typenum != NPY_CDOUBLE) {
        if (PyTypeNum_ISCOMPLEX(typenum)) {
            op1 = PyArray_Conjugate(ap1, NULL);
            if (op1 == NULL) {
                goto fail;
            }
            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)op1;
        }
        ret = (PyArrayObject *)PyArray_InnerProduct((PyObject *)ap1,
                                                    (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return PyArray_Return(ret);
    }

    if (PyArray_DIM(ap2, 0) != PyArray_DIM(ap1, PyArray_NDIM(ap1)-1)) {
        PyErr_SetString(PyExc_ValueError, "vectors have different lengths");
        goto fail;
    }
    l = PyArray_DIM(ap1, PyArray_NDIM(ap1)-1);

    ret = (PyArrayObject *)PyArray_SimpleNew(0, dimensions, typenum);
    if (ret == NULL) {
        goto fail;
    }

    NPY_BEGIN_ALLOW_THREADS;

    /* Dot product between two vectors -- Level 1 BLAS */
    if (typenum == NPY_DOUBLE || typenum == NPY_FLOAT) {
        blas_dot(typenum, l, PyArray_DATA(ap1), PyArray_ITEMSIZE(ap1),
                 PyArray_DATA(ap2), PyArray_ITEMSIZE(ap2), PyArray_DATA(ret));
    }
    else if (typenum == NPY_CDOUBLE) {
        cblas_zdotc_sub(l, (double *)PyArray_DATA(ap1), 1,
                        (double *)PyArray_DATA(ap2), 1, (double *)PyArray_DATA(ret));
    }
    else if (typenum == NPY_CFLOAT) {
        cblas_cdotc_sub(l, (float *)PyArray_DATA(ap1), 1,
                        (float *)PyArray_DATA(ap2), 1, (float *)PyArray_DATA(ret));
    }

    NPY_END_ALLOW_THREADS;

    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return PyArray_Return(ret);

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

static struct PyMethodDef dotblas_module_methods[] = {
    {"vdot",
        (PyCFunction)dotblas_vdot,
        1, NULL},
    {NULL, NULL, 0, NULL}               /* sentinel */
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_dotblas",
        NULL,
        -1,
        dotblas_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__dotblas(void)
#else
#define RETVAL
PyMODINIT_FUNC init_dotblas(void)
#endif
{
#if defined(NPY_PY3K)
    PyObject *m;
    m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("_dotblas", dotblas_module_methods, module_doc);
#endif

    /* Import the array object */
    import_array();

    /* initialize oldFunctions table */
    init_oldFunctions();

    return RETVAL;
}
