/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"

#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "ufunc_override.h"
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
 * Helper: dispatch to appropriate cblas_?gemm for typenum.
 */
static void
gemm(int typenum, enum CBLAS_ORDER order,
     enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     PyArrayObject *A, int lda, PyArrayObject *B, int ldb, PyArrayObject *R)
{
    const void *Adata = PyArray_DATA(A), *Bdata = PyArray_DATA(B);
    void *Rdata = PyArray_DATA(R);

    int ldc = PyArray_DIM(R, 1) > 1 ? PyArray_DIM(R, 1) : 1;

    switch (typenum) {
        case NPY_DOUBLE:
            cblas_dgemm(order, transA, transB, m, n, k, 1.,
                        Adata, lda, Bdata, ldb, 0., Rdata, ldc);
            break;
        case NPY_FLOAT:
            cblas_sgemm(order, transA, transB, m, n, k, 1.f,
                        Adata, lda, Bdata, ldb, 0.f, Rdata, ldc);
            break;
        case NPY_CDOUBLE:
            cblas_zgemm(order, transA, transB, m, n, k, oneD,
                        Adata, lda, Bdata, ldb, zeroD, Rdata, ldc);
            break;
        case NPY_CFLOAT:
            cblas_cgemm(order, transA, transB, m, n, k, oneF,
                        Adata, lda, Bdata, ldb, zeroF, Rdata, ldc);
            break;
    }
}


/*
 * Helper: dispatch to appropriate cblas_?gemv for typenum.
 */
static void
gemv(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     PyArrayObject *A, int lda, PyArrayObject *X, int incX,
     PyArrayObject *R)
{
    const void *Adata = PyArray_DATA(A), *Xdata = PyArray_DATA(X);
    void *Rdata = PyArray_DATA(R);

    int m = PyArray_DIM(A, 0), n = PyArray_DIM(A, 1);

    switch (typenum) {
        case NPY_DOUBLE:
            cblas_dgemv(order, trans, m, n, 1., Adata, lda, Xdata, incX,
                        0., Rdata, 1);
            break;
        case NPY_FLOAT:
            cblas_sgemv(order, trans, m, n, 1.f, Adata, lda, Xdata, incX,
                        0.f, Rdata, 1);
            break;
        case NPY_CDOUBLE:
            cblas_zgemv(order, trans, m, n, oneD, Adata, lda, Xdata, incX,
                        zeroD, Rdata, 1);
            break;
        case NPY_CFLOAT:
            cblas_cgemv(order, trans, m, n, oneF, Adata, lda, Xdata, incX,
                        zeroF, Rdata, 1);
            break;
    }
}


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


static MatrixShape
_select_matrix_shape(PyArrayObject *array)
{
    switch (PyArray_NDIM(array)) {
        case 0:
            return _scalar;
        case 1:
            if (PyArray_DIM(array, 0) > 1)
                return _column;
            return _scalar;
        case 2:
            if (PyArray_DIM(array, 0) > 1) {
                if (PyArray_DIM(array, 1) == 1)
                    return _column;
                else
                    return _matrix;
            }
            if (PyArray_DIM(array, 1) == 1)
                return _scalar;
            return _row;
    }
    return _matrix;
}


/*
 * This also makes sure that the data segment is aligned with
 * an itemsize address as well by returning one if not true.
 */
static int
_bad_strides(PyArrayObject *ap)
{
    int itemsize = PyArray_ITEMSIZE(ap);
    int i, N=PyArray_NDIM(ap);
    npy_intp *strides = PyArray_STRIDES(ap);

    if (((npy_intp)(PyArray_DATA(ap)) % itemsize) != 0) {
        return 1;
    }
    for (i = 0; i < N; i++) {
        if ((strides[i] < 0) || (strides[i] % itemsize) != 0) {
            return 1;
        }
    }

    return 0;
}

/*
 * innerproduct(a,b)
 *
 * Returns the inner product of a and b for arrays of
 * floating point types. Like the generic NumPy equivalent the product
 * sum is over the last dimension of a and b.
 * NB: The first argument is not conjugated.
 */

static PyObject *
dotblas_innerproduct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *ap1, *ap2, *ret;
    int j, l, lda, ldb;
    int typenum, nd;
    npy_intp dimensions[NPY_MAXDIMS];
    PyTypeObject *subtype;
    double prior1, prior2;

    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }

    /*
     * Inner product using the BLAS.  The product sum is taken along the last
     * dimensions of the two arrays.
     * Only speeds things up for float double and complex types.
     */


    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    /* This function doesn't handle other types */
    if ((typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE &&
            typenum != NPY_FLOAT && typenum != NPY_CFLOAT)) {
        return PyArray_Return((PyArrayObject *)PyArray_InnerProduct(op1, op2));
    }

    ret = NULL;
    ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
    if (ap1 == NULL) {
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
    if (ap2 == NULL) {
        goto fail;
    }

    if ((PyArray_NDIM(ap1) > 2) || (PyArray_NDIM(ap2) > 2)) {
        /* This function doesn't handle dimensions greater than 2 */
        ret = (PyArrayObject *)PyArray_InnerProduct((PyObject *)ap1,
                                                    (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return PyArray_Return(ret);
    }

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        /* One of ap1 or ap2 is a scalar */
        if (PyArray_NDIM(ap1) == 0) {
            /* Make ap2 the scalar */
            PyArrayObject *t = ap1;
            ap1 = ap2;
            ap2 = t;
        }
        for (l = 1, j = 0; j < PyArray_NDIM(ap1); j++) {
            dimensions[j] = PyArray_DIM(ap1, j);
            l *= dimensions[j];
        }
        nd = PyArray_NDIM(ap1);
    }
    else {
        /*
         * (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2)
         *  Both ap1 and ap2 are vectors or matrices
         */
        l = PyArray_DIM(ap1, PyArray_NDIM(ap1)-1);

        if (PyArray_DIM(ap2, PyArray_NDIM(ap2)-1) != l) {
            PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
            goto fail;
        }
        nd = PyArray_NDIM(ap1)+PyArray_NDIM(ap2)-2;

        if (nd == 1)
            dimensions[0] = (PyArray_NDIM(ap1) == 2) ? PyArray_DIM(ap1, 0) : PyArray_DIM(ap2, 0);
        else if (nd == 2) {
            dimensions[0] = PyArray_DIM(ap1, 0);
            dimensions[1] = PyArray_DIM(ap2, 0);
        }
    }

    /* Choose which subtype to return */
    prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
    prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
    subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));

    ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                       typenum, NULL, NULL, 0, 0,
                                       (PyObject *)\
                                       (prior2 > prior1 ? ap2 : ap1));

    if (ret == NULL) {
        goto fail;
    }
    NPY_BEGIN_ALLOW_THREADS
    memset(PyArray_DATA(ret), 0, PyArray_NBYTES(ret));

    if (PyArray_NDIM(ap2) == 0) {
        /* Multiplication by a scalar -- Level 1 BLAS */
        if (typenum == NPY_DOUBLE) {
            cblas_daxpy(l, *((double *)PyArray_DATA(ap2)), (double *)PyArray_DATA(ap1), 1,
                        (double *)PyArray_DATA(ret), 1);
        }
        else if (typenum == NPY_CDOUBLE) {
            cblas_zaxpy(l, (double *)PyArray_DATA(ap2), (double *)PyArray_DATA(ap1), 1,
                        (double *)PyArray_DATA(ret), 1);
        }
        else if (typenum == NPY_FLOAT) {
            cblas_saxpy(l, *((float *)PyArray_DATA(ap2)), (float *)PyArray_DATA(ap1), 1,
                        (float *)PyArray_DATA(ret), 1);
        }
        else if (typenum == NPY_CFLOAT) {
            cblas_caxpy(l, (float *)PyArray_DATA(ap2), (float *)PyArray_DATA(ap1), 1,
                        (float *)PyArray_DATA(ret), 1);
        }
    }
    else if (PyArray_NDIM(ap1) == 1 && PyArray_NDIM(ap2) == 1) {
        /* Dot product between two vectors -- Level 1 BLAS */
        blas_dot(typenum, l, PyArray_DATA(ap1), PyArray_ITEMSIZE(ap1),
                 PyArray_DATA(ap2), PyArray_ITEMSIZE(ap2), PyArray_DATA(ret));
    }
    else if (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 1) {
        /* Matrix-vector multiplication -- Level 2 BLAS */
        lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        gemv(typenum, CblasRowMajor, CblasNoTrans, ap1, lda, ap2, 1, ret);
    }
    else if (PyArray_NDIM(ap1) == 1 && PyArray_NDIM(ap2) == 2) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        lda = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);
        gemv(typenum, CblasRowMajor, CblasNoTrans, ap2, lda, ap1, 1, ret);
    }
    else {
        /*
         * (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2)
         * Matrix matrix multiplication -- Level 3 BLAS
         */
        lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        ldb = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);
        gemm(typenum, CblasRowMajor, CblasNoTrans, CblasTrans,
             PyArray_DIM(ap1, 0), PyArray_DIM(ap2, 0), PyArray_DIM(ap1, 1),
             ap1, lda, ap2, ldb, ret);
    }
    NPY_END_ALLOW_THREADS
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return PyArray_Return(ret);

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


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
    {"inner",
        (PyCFunction)dotblas_innerproduct,
        1, NULL},
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
