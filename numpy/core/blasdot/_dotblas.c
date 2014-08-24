/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"

#include "numpy/arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "common.h"
#include "ufunc_override.h"
#ifndef CBLAS_HEADER
#define CBLAS_HEADER "cblas.h"
#endif
#include CBLAS_HEADER

#include <assert.h>
#include <limits.h>
#include <stdio.h>

static char module_doc[] =
"This module provides a BLAS optimized\nmatrix multiply, inner product and dot for numpy arrays";

static PyArray_DotFunc *oldFunctions[NPY_NTYPES];

#define MIN(a, b)   ((a) < (b) ? (a) : (b))

/*
 * Convert NumPy stride to BLAS stride. Returns 0 if conversion cannot be done
 * (BLAS won't handle negative or zero strides the way we want).
 */
static NPY_INLINE int
blas_stride(npy_intp stride, unsigned itemsize)
{
    if (stride <= 0 || stride % itemsize != 0) {
        return 0;
    }
    stride /= itemsize;

    if (stride > INT_MAX) {
        return 0;
    }
    return stride;
}

/*
 * The following functions do a "chunked" dot product using BLAS when
 * sizeof(npy_intp) > sizeof(int), because BLAS libraries can typically not
 * handle more than INT_MAX elements per call.
 *
 * The chunksize is the greatest power of two less than INT_MAX.
 */
#if NPY_MAX_INTP > INT_MAX
# define CHUNKSIZE  (INT_MAX / 2 + 1)
#else
# define CHUNKSIZE  NPY_MAX_INTP
#endif

static void
FLOAT_dot(void *a, npy_intp stridea, void *b, npy_intp strideb, void *res,
          npy_intp n, void *tmp)
{
    int na = blas_stride(stridea, sizeof(float));
    int nb = blas_stride(strideb, sizeof(float));

    if (na && nb) {
        double r = 0.;          /* double for stability */
        float *fa = a, *fb = b;

        while (n > 0) {
            int chunk = MIN(n, CHUNKSIZE);

            r += cblas_sdot(chunk, fa, na, fb, nb);
            fa += chunk * na;
            fb += chunk * nb;
            n -= chunk;
        }
        *((float *)res) = r;
    }
    else {
        oldFunctions[NPY_FLOAT](a, stridea, b, strideb, res, n, tmp);
    }
}

static void
DOUBLE_dot(void *a, npy_intp stridea, void *b, npy_intp strideb, void *res,
           npy_intp n, void *tmp)
{
    int na = blas_stride(stridea, sizeof(double));
    int nb = blas_stride(strideb, sizeof(double));

    if (na && nb) {
        double r = 0.;
        double *da = a, *db = b;

        while (n > 0) {
            int chunk = MIN(n, CHUNKSIZE);

            r += cblas_ddot(chunk, da, na, db, nb);
            da += chunk * na;
            db += chunk * nb;
            n -= chunk;
        }
        *((double *)res) = r;
    }
    else {
        oldFunctions[NPY_DOUBLE](a, stridea, b, strideb, res, n, tmp);
    }
}

static void
CFLOAT_dot(void *a, npy_intp stridea, void *b, npy_intp strideb, void *res,
           npy_intp n, void *tmp)
{
    int na = blas_stride(stridea, sizeof(npy_cfloat));
    int nb = blas_stride(strideb, sizeof(npy_cfloat));

    if (na && nb) {
        cblas_cdotu_sub((int)n, (float *)a, na, (float *)b, nb, (float *)res);
    }
    else {
        oldFunctions[NPY_CFLOAT](a, stridea, b, strideb, res, n, tmp);
    }
}

static void
CDOUBLE_dot(void *a, npy_intp stridea, void *b, npy_intp strideb, void *res,
            npy_intp n, void *tmp)
{
    int na = blas_stride(stridea, sizeof(npy_cdouble));
    int nb = blas_stride(strideb, sizeof(npy_cdouble));

    if (na && nb) {
        cblas_zdotu_sub((int)n, (double *)a, na, (double *)b, nb,
                        (double *)res);
    }
    else {
        oldFunctions[NPY_CDOUBLE](a, stridea, b, strideb, res, n, tmp);
    }
}

/*
 * Helper: call appropriate BLAS dot function for typenum.
 * Strides are NumPy strides.
 */
static void
blas_dot(int typenum, npy_intp n,
         void *a, npy_intp stridea, void *b, npy_intp strideb, void *res)
{
    PyArray_DotFunc *dot = NULL;
    switch (typenum) {
        case NPY_DOUBLE:
            dot = DOUBLE_dot;
            break;
        case NPY_FLOAT:
            dot = FLOAT_dot;
            break;
        case NPY_CDOUBLE:
            dot = CDOUBLE_dot;
            break;
        case NPY_CFLOAT:
            dot = CFLOAT_dot;
            break;
    }
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


static npy_bool altered=NPY_FALSE;

/*
 * alterdot() changes all dot functions to use blas.
 */
static PyObject *
dotblas_alterdot(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *descr;

    if (!PyArg_ParseTuple(args, "")) return NULL;

    /* Replace the dot functions to the ones using blas */

    if (!altered) {
        descr = PyArray_DescrFromType(NPY_FLOAT);
        oldFunctions[NPY_FLOAT] = descr->f->dotfunc;
        descr->f->dotfunc = (PyArray_DotFunc *)FLOAT_dot;

        descr = PyArray_DescrFromType(NPY_DOUBLE);
        oldFunctions[NPY_DOUBLE] = descr->f->dotfunc;
        descr->f->dotfunc = (PyArray_DotFunc *)DOUBLE_dot;

        descr = PyArray_DescrFromType(NPY_CFLOAT);
        oldFunctions[NPY_CFLOAT] = descr->f->dotfunc;
        descr->f->dotfunc = (PyArray_DotFunc *)CFLOAT_dot;

        descr = PyArray_DescrFromType(NPY_CDOUBLE);
        oldFunctions[NPY_CDOUBLE] = descr->f->dotfunc;
        descr->f->dotfunc = (PyArray_DotFunc *)CDOUBLE_dot;

        altered = NPY_TRUE;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * restoredot() restores dots to defaults.
 */
static PyObject *
dotblas_restoredot(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *descr;

    if (!PyArg_ParseTuple(args, "")) return NULL;

    if (altered) {
        descr = PyArray_DescrFromType(NPY_FLOAT);
        descr->f->dotfunc = oldFunctions[NPY_FLOAT];
        oldFunctions[NPY_FLOAT] = NULL;
        Py_XDECREF(descr);

        descr = PyArray_DescrFromType(NPY_DOUBLE);
        descr->f->dotfunc = oldFunctions[NPY_DOUBLE];
        oldFunctions[NPY_DOUBLE] = NULL;
        Py_XDECREF(descr);

        descr = PyArray_DescrFromType(NPY_CFLOAT);
        descr->f->dotfunc = oldFunctions[NPY_CFLOAT];
        oldFunctions[NPY_CFLOAT] = NULL;
        Py_XDECREF(descr);

        descr = PyArray_DescrFromType(NPY_CDOUBLE);
        descr->f->dotfunc = oldFunctions[NPY_CDOUBLE];
        oldFunctions[NPY_CDOUBLE] = NULL;
        Py_XDECREF(descr);

        altered = NPY_FALSE;
    }

    Py_INCREF(Py_None);
    return Py_None;
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


/* This also makes sure that the data segment is aligned with
   an itemsize address as well by returning one if not true.
*/
static int
_bad_strides(PyArrayObject *ap)
{
    register int itemsize = PyArray_ITEMSIZE(ap);
    register int i, N=PyArray_NDIM(ap);
    register npy_intp *strides = PyArray_STRIDES(ap);

    if (((npy_intp)(PyArray_DATA(ap)) % itemsize) != 0)
        return 1;
    for (i=0; i<N; i++) {
        if ((strides[i] < 0) || (strides[i] % itemsize) != 0)
            return 1;
    }

    return 0;
}

/*
 * dot(a,b)
 * Returns the dot product of a and b for arrays of floating point types.
 * Like the generic numpy equivalent the product sum is over
 * the last dimension of a and the second-to-last dimension of b.
 * NB: The first argument is not conjugated.;
 */
static PyObject *
dotblas_matrixproduct(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject* kwargs)
{
    static PyObject *cached_npy_dot = NULL;
    PyObject *override = NULL;
    PyObject *module;
    PyObject *op1, *op2;
    PyArrayObject *ap1 = NULL, *ap2 = NULL, *out = NULL, *ret = NULL;
    int errval;
    int j, lda, ldb;
    npy_intp l;
    int typenum, nd;
    npy_intp ap1stride = 0;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp numbytes;
    double prior1, prior2;
    PyTypeObject *subtype;
    PyArray_Descr *dtype;
    MatrixShape ap1shape, ap2shape;
    char* kwords[] = {"a", "b", "out", NULL };

    if (cached_npy_dot == NULL) {
        module = PyImport_ImportModule("numpy.core._dotblas");
        cached_npy_dot = PyDict_GetItemString(PyModule_GetDict(module), "dot");

        Py_INCREF(cached_npy_dot);
        Py_DECREF(module);
    }

    errval = PyUFunc_CheckOverride(cached_npy_dot, "__call__", args, kwargs, 
                                   &override, 2);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwords,
                                    &op1, &op2, &out)) {
        return NULL;
    }
    if ((PyObject *)out == Py_None) {
        out = NULL;
    }

    /*
     * "Matrix product" using the BLAS.
     * Only works for float double and complex types.
     */

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    /* This function doesn't handle other types */
    if ((typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE &&
         typenum != NPY_FLOAT && typenum != NPY_CFLOAT)) {
        return PyArray_Return((PyArrayObject *)PyArray_MatrixProduct2(
                                                    (PyObject *)op1,
                                                    (PyObject *)op2,
                                                    out));
    }

    dtype = PyArray_DescrFromType(typenum);
    if (dtype == NULL) {
        return NULL;
    }
    Py_INCREF(dtype);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, dtype, 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, dtype, 0, 0, NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

    if ((PyArray_NDIM(ap1) > 2) || (PyArray_NDIM(ap2) > 2)) {
        /*
         * This function doesn't handle dimensions greater than 2
         * (or negative striding)  -- other
         * than to ensure the dot function is altered
         */
        if (!altered) {
            /* need to alter dot product */
            PyObject *tmp1, *tmp2;
            tmp1 = PyTuple_New(0);
            tmp2 = dotblas_alterdot(NULL, tmp1);
            Py_DECREF(tmp1);
            Py_DECREF(tmp2);
        }
        ret = (PyArrayObject *)PyArray_MatrixProduct2((PyObject *)ap1,
                                                      (PyObject *)ap2,
                                                      out);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return PyArray_Return(ret);
    }

    if (_bad_strides(ap1)) {
            op1 = PyArray_NewCopy(ap1, NPY_ANYORDER);
            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)op1;
            if (ap1 == NULL) {
                goto fail;
            }
    }
    if (_bad_strides(ap2)) {
            op2 = PyArray_NewCopy(ap2, NPY_ANYORDER);
            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)op2;
            if (ap2 == NULL) {
                goto fail;
            }
    }
    ap1shape = _select_matrix_shape(ap1);
    ap2shape = _select_matrix_shape(ap2);

    if (ap1shape == _scalar || ap2shape == _scalar) {
        PyArrayObject *oap1, *oap2;
        oap1 = ap1; oap2 = ap2;
        /* One of ap1 or ap2 is a scalar */
        if (ap1shape == _scalar) {              /* Make ap2 the scalar */
            PyArrayObject *t = ap1;
            ap1 = ap2;
            ap2 = t;
            ap1shape = ap2shape;
            ap2shape = _scalar;
        }

        if (ap1shape == _row) {
            ap1stride = PyArray_STRIDE(ap1, 1);
        }
        else if (PyArray_NDIM(ap1) > 0) {
            ap1stride = PyArray_STRIDE(ap1, 0);
        }

        if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
            npy_intp *thisdims;
            if (PyArray_NDIM(ap1) == 0) {
                nd = PyArray_NDIM(ap2);
                thisdims = PyArray_DIMS(ap2);
            }
            else {
                nd = PyArray_NDIM(ap1);
                thisdims = PyArray_DIMS(ap1);
            }
            l = 1;
            for (j = 0; j < nd; j++) {
                dimensions[j] = thisdims[j];
                l *= dimensions[j];
            }
        }
        else {
            l = PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1);

            if (PyArray_DIM(oap2, 0) != l) {
                not_aligned(oap1, PyArray_NDIM(oap1) - 1, oap2, 0);
                goto fail;
            }
            nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
            /*
             * nd = 0 or 1 or 2. If nd == 0 do nothing ...
             */
            if (nd == 1) {
                /*
                 * Either PyArray_NDIM(ap1) is 1 dim or PyArray_NDIM(ap2) is 1 dim
                 * and the other is 2-dim
                 */
                dimensions[0] = (PyArray_NDIM(oap1) == 2) ? PyArray_DIM(oap1, 0) : PyArray_DIM(oap2, 1);
                l = dimensions[0];
                /*
                 * Fix it so that dot(shape=(N,1), shape=(1,))
                 * and dot(shape=(1,), shape=(1,N)) both return
                 * an (N,) array (but use the fast scalar code)
                 */
            }
            else if (nd == 2) {
                dimensions[0] = PyArray_DIM(oap1, 0);
                dimensions[1] = PyArray_DIM(oap2, 1);
                /*
                 * We need to make sure that dot(shape=(1,1), shape=(1,N))
                 * and dot(shape=(N,1),shape=(1,1)) uses
                 * scalar multiplication appropriately
                 */
                if (ap1shape == _row) {
                    l = dimensions[1];
                }
                else {
                    l = dimensions[0];
                }
            }

            /* Check if the summation dimension is 0-sized */
            if (PyArray_DIM(oap1, PyArray_NDIM(oap1) - 1) == 0) {
                l = 0;
            }
        }
    }
    else {
        /*
         * (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2)
         * Both ap1 and ap2 are vectors or matrices
         */
        l = PyArray_DIM(ap1, PyArray_NDIM(ap1) - 1);

        if (PyArray_DIM(ap2, 0) != l) {
            not_aligned(ap1, PyArray_NDIM(ap1) - 1, ap2, 0);
            goto fail;
        }
        nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;

        if (nd == 1)
            dimensions[0] = (PyArray_NDIM(ap1) == 2) ? PyArray_DIM(ap1, 0) : PyArray_DIM(ap2, 1);
        else if (nd == 2) {
            dimensions[0] = PyArray_DIM(ap1, 0);
            dimensions[1] = PyArray_DIM(ap2, 1);
        }
    }

    /* Choose which subtype to return */
    if (Py_TYPE(ap1) != Py_TYPE(ap2)) {
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
    }
    else {
        prior1 = prior2 = 0.0;
        subtype = Py_TYPE(ap1);
    }

    if (out) {
        int d;
        /* verify that out is usable */
        if (Py_TYPE(out) != subtype ||
            PyArray_NDIM(out) != nd ||
            PyArray_TYPE(out) != typenum ||
            !PyArray_ISCARRAY(out)) {

            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable "
                "(must have the right type, nr dimensions, and be a C-Array)");
            goto fail;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                goto fail;
            }
        }
        Py_INCREF(out);
        ret = out;
    } else {
        ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                           typenum, NULL, NULL, 0, 0,
                                           (PyObject *)
                                           (prior2 > prior1 ? ap2 : ap1));
    }

    if (ret == NULL) {
        goto fail;
    }
    numbytes = PyArray_NBYTES(ret);
    memset(PyArray_DATA(ret), 0, numbytes);
    if (numbytes==0 || l == 0) {
            Py_DECREF(ap1);
            Py_DECREF(ap2);
            return PyArray_Return(ret);
    }

    if (ap2shape == _scalar) {
        /*
         * Multiplication by a scalar -- Level 1 BLAS
         * if ap1shape is a matrix and we are not contiguous, then we can't
         * just blast through the entire array using a single striding factor
         */
        NPY_BEGIN_ALLOW_THREADS;

        if (typenum == NPY_DOUBLE) {
            if (l == 1) {
                *((double *)PyArray_DATA(ret)) = *((double *)PyArray_DATA(ap2)) *
                    *((double *)PyArray_DATA(ap1));
            }
            else if (ap1shape != _matrix) {
                cblas_daxpy(l, *((double *)PyArray_DATA(ap2)), (double *)PyArray_DATA(ap1),
                            ap1stride/sizeof(double), (double *)PyArray_DATA(ret), 1);
            }
            else {
                int maxind, oind, i, a1s, rets;
                char *ptr, *rptr;
                double val;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1-maxind;
                ptr = PyArray_DATA(ap1);
                rptr = PyArray_DATA(ret);
                l = PyArray_DIM(ap1, maxind);
                val = *((double *)PyArray_DATA(ap2));
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(double);
                rets = PyArray_STRIDE(ret, maxind) / sizeof(double);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    cblas_daxpy(l, val, (double *)ptr, a1s,
                                (double *)rptr, rets);
                    ptr += PyArray_STRIDE(ap1, oind);
                    rptr += PyArray_STRIDE(ret, oind);
                }
            }
        }
        else if (typenum == NPY_CDOUBLE) {
            if (l == 1) {
                npy_cdouble *ptr1, *ptr2, *res;

                ptr1 = (npy_cdouble *)PyArray_DATA(ap2);
                ptr2 = (npy_cdouble *)PyArray_DATA(ap1);
                res = (npy_cdouble *)PyArray_DATA(ret);
                res->real = ptr1->real * ptr2->real - ptr1->imag * ptr2->imag;
                res->imag = ptr1->real * ptr2->imag + ptr1->imag * ptr2->real;
            }
            else if (ap1shape != _matrix) {
                cblas_zaxpy(l, (double *)PyArray_DATA(ap2), (double *)PyArray_DATA(ap1),
                            ap1stride/sizeof(npy_cdouble), (double *)PyArray_DATA(ret), 1);
            }
            else {
                int maxind, oind, i, a1s, rets;
                char *ptr, *rptr;
                double *pval;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1-maxind;
                ptr = PyArray_DATA(ap1);
                rptr = PyArray_DATA(ret);
                l = PyArray_DIM(ap1, maxind);
                pval = (double *)PyArray_DATA(ap2);
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(npy_cdouble);
                rets = PyArray_STRIDE(ret, maxind) / sizeof(npy_cdouble);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    cblas_zaxpy(l, pval, (double *)ptr, a1s,
                                (double *)rptr, rets);
                    ptr += PyArray_STRIDE(ap1, oind);
                    rptr += PyArray_STRIDE(ret, oind);
                }
            }
        }
        else if (typenum == NPY_FLOAT) {
            if (l == 1) {
                *((float *)PyArray_DATA(ret)) = *((float *)PyArray_DATA(ap2)) *
                    *((float *)PyArray_DATA(ap1));
            }
            else if (ap1shape != _matrix) {
                cblas_saxpy(l, *((float *)PyArray_DATA(ap2)), (float *)PyArray_DATA(ap1),
                            ap1stride/sizeof(float), (float *)PyArray_DATA(ret), 1);
            }
            else {
                int maxind, oind, i, a1s, rets;
                char *ptr, *rptr;
                float val;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1-maxind;
                ptr = PyArray_DATA(ap1);
                rptr = PyArray_DATA(ret);
                l = PyArray_DIM(ap1, maxind);
                val = *((float *)PyArray_DATA(ap2));
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(float);
                rets = PyArray_STRIDE(ret, maxind) / sizeof(float);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    cblas_saxpy(l, val, (float *)ptr, a1s,
                                (float *)rptr, rets);
                    ptr += PyArray_STRIDE(ap1, oind);
                    rptr += PyArray_STRIDE(ret, oind);
                }
            }
        }
        else if (typenum == NPY_CFLOAT) {
            if (l == 1) {
                npy_cfloat *ptr1, *ptr2, *res;

                ptr1 = (npy_cfloat *)PyArray_DATA(ap2);
                ptr2 = (npy_cfloat *)PyArray_DATA(ap1);
                res = (npy_cfloat *)PyArray_DATA(ret);
                res->real = ptr1->real * ptr2->real - ptr1->imag * ptr2->imag;
                res->imag = ptr1->real * ptr2->imag + ptr1->imag * ptr2->real;
            }
            else if (ap1shape != _matrix) {
                cblas_caxpy(l, (float *)PyArray_DATA(ap2), (float *)PyArray_DATA(ap1),
                            ap1stride/sizeof(npy_cfloat), (float *)PyArray_DATA(ret), 1);
            }
            else {
                int maxind, oind, i, a1s, rets;
                char *ptr, *rptr;
                float *pval;

                maxind = (PyArray_DIM(ap1, 0) >= PyArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1-maxind;
                ptr = PyArray_DATA(ap1);
                rptr = PyArray_DATA(ret);
                l = PyArray_DIM(ap1, maxind);
                pval = (float *)PyArray_DATA(ap2);
                a1s = PyArray_STRIDE(ap1, maxind) / sizeof(npy_cfloat);
                rets = PyArray_STRIDE(ret, maxind) / sizeof(npy_cfloat);
                for (i = 0; i < PyArray_DIM(ap1, oind); i++) {
                    cblas_caxpy(l, pval, (float *)ptr, a1s,
                                (float *)rptr, rets);
                    ptr += PyArray_STRIDE(ap1, oind);
                    rptr += PyArray_STRIDE(ret, oind);
                }
            }
        }
        NPY_END_ALLOW_THREADS;
    }
    else if ((ap2shape == _column) && (ap1shape != _matrix)) {
        NPY_BEGIN_ALLOW_THREADS;

        /* Dot product between two vectors -- Level 1 BLAS */
        blas_dot(typenum, l,
                 PyArray_DATA(ap1), PyArray_STRIDE(ap1, (ap1shape == _row)),
                 PyArray_DATA(ap2), PyArray_STRIDE(ap2, 0),
                 PyArray_DATA(ret));
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape == _matrix && ap2shape != _matrix) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        /* lda must be MAX(M,1) */
        enum CBLAS_ORDER Order;
        int ap2s;

        if (!PyArray_ISONESEGMENT(ap1)) {
            PyObject *new;
            new = PyArray_Copy(ap1);
            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyArray_ISCONTIGUOUS(ap1)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        ap2s = PyArray_STRIDE(ap2, 0) / PyArray_ITEMSIZE(ap2);
        gemv(typenum, Order, CblasNoTrans, ap1, lda, ap2, ap2s, ret);
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape != _matrix && ap2shape == _matrix) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        enum CBLAS_ORDER Order;
        int ap1s;

        if (!PyArray_ISONESEGMENT(ap2)) {
            PyObject *new;
            new = PyArray_Copy(ap2);
            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyArray_ISCONTIGUOUS(ap2)) {
            Order = CblasRowMajor;
            lda = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }
        if (ap1shape == _row) {
            ap1s = PyArray_STRIDE(ap1, 1) / PyArray_ITEMSIZE(ap1);
        }
        else {
            ap1s = PyArray_STRIDE(ap1, 0) / PyArray_ITEMSIZE(ap1);
        }
        gemv(typenum, Order, CblasTrans, ap2, lda, ap1, ap1s, ret);
        NPY_END_ALLOW_THREADS;
    }
    else {
        /*
         * (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2)
         * Matrix matrix multiplication -- Level 3 BLAS
         *  L x M  multiplied by M x N
         */
        enum CBLAS_ORDER Order;
        enum CBLAS_TRANSPOSE Trans1, Trans2;
        int M, N, L;

        /* Optimization possible: */
        /*
         * We may be able to handle single-segment arrays here
         * using appropriate values of Order, Trans1, and Trans2.
         */
        if (!PyArray_IS_C_CONTIGUOUS(ap2) && !PyArray_IS_F_CONTIGUOUS(ap2)) {
            PyObject *new = PyArray_Copy(ap2);

            Py_DECREF(ap2);
            ap2 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        if (!PyArray_IS_C_CONTIGUOUS(ap1) && !PyArray_IS_F_CONTIGUOUS(ap1)) {
            PyObject *new = PyArray_Copy(ap1);

            Py_DECREF(ap1);
            ap1 = (PyArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_ALLOW_THREADS;

        Order = CblasRowMajor;
        Trans1 = CblasNoTrans;
        Trans2 = CblasNoTrans;
        L = PyArray_DIM(ap1, 0);
        N = PyArray_DIM(ap2, 1);
        M = PyArray_DIM(ap2, 0);
        lda = (PyArray_DIM(ap1, 1) > 1 ? PyArray_DIM(ap1, 1) : 1);
        ldb = (PyArray_DIM(ap2, 1) > 1 ? PyArray_DIM(ap2, 1) : 1);

        /*
         * Avoid temporary copies for arrays in Fortran order
         */
        if (PyArray_IS_F_CONTIGUOUS(ap1)) {
            Trans1 = CblasTrans;
            lda = (PyArray_DIM(ap1, 0) > 1 ? PyArray_DIM(ap1, 0) : 1);
        }
        if (PyArray_IS_F_CONTIGUOUS(ap2)) {
            Trans2 = CblasTrans;
            ldb = (PyArray_DIM(ap2, 0) > 1 ? PyArray_DIM(ap2, 0) : 1);
        }
        gemm(typenum, Order, Trans1, Trans2, L, N, M, ap1, lda, ap2, ldb, ret);
        NPY_END_ALLOW_THREADS;
    }


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

    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;

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
    if (ap1 == NULL) return NULL;
    ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
    if (ap2 == NULL) goto fail;

    if ((PyArray_NDIM(ap1) > 2) || (PyArray_NDIM(ap2) > 2)) {
        /* This function doesn't handle dimensions greater than 2 -- other
           than to ensure the dot function is altered
        */
        if (!altered) {
            /* need to alter dot product */
            PyObject *tmp1, *tmp2;
            tmp1 = PyTuple_New(0);
            tmp2 = dotblas_alterdot(NULL, tmp1);
            Py_DECREF(tmp1);
            Py_DECREF(tmp2);
        }
        ret = (PyArrayObject *)PyArray_InnerProduct((PyObject *)ap1,
                                                    (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return PyArray_Return(ret);
    }

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        /* One of ap1 or ap2 is a scalar */
        if (PyArray_NDIM(ap1) == 0) {             /* Make ap2 the scalar */
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
    else { /* (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2) */
        /*  Both ap1 and ap2 are vectors or matrices */
        l = PyArray_DIM(ap1, PyArray_NDIM(ap1)-1);

        if (PyArray_DIM(ap2, PyArray_NDIM(ap2)-1) != l) {
            not_aligned(ap1, PyArray_NDIM(ap1) - 1,
                        ap2, PyArray_NDIM(ap2) - 1);
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

    if (ret == NULL) goto fail;
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
    else { /* (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2) */
        /* Matrix matrix multiplication -- Level 3 BLAS */
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
    PyArrayObject *ap1=NULL, *ap2=NULL, *ret=NULL;
    int l;
    int typenum;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Multiplies op1 and op2, each of which must be vector.
     */

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    type = PyArray_DescrFromType(typenum);
    Py_INCREF(type);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, type, 0, 0, 0, NULL);
    if (ap1==NULL) {Py_DECREF(type); goto fail;}
    op1 = PyArray_Flatten(ap1, 0);
    if (op1==NULL) {Py_DECREF(type); goto fail;}
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;

    ap2 = (PyArrayObject *)PyArray_FromAny(op2, type, 0, 0, 0, NULL);
    if (ap2==NULL) goto fail;
    op2 = PyArray_Flatten(ap2, 0);
    if (op2 == NULL) goto fail;
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;

    if (typenum != NPY_FLOAT && typenum != NPY_DOUBLE &&
        typenum != NPY_CFLOAT && typenum != NPY_CDOUBLE) {
        if (!altered) {
            /* need to alter dot product */
            PyObject *tmp1, *tmp2;
            tmp1 = PyTuple_New(0);
            tmp2 = dotblas_alterdot(NULL, tmp1);
            Py_DECREF(tmp1);
            Py_DECREF(tmp2);
        }
        if (PyTypeNum_ISCOMPLEX(typenum)) {
            op1 = PyArray_Conjugate(ap1, NULL);
            if (op1==NULL) goto fail;
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
    if (ret == NULL) goto fail;

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
    {"dot",  (PyCFunction)dotblas_matrixproduct, METH_VARARGS|METH_KEYWORDS, NULL},
    {"inner",   (PyCFunction)dotblas_innerproduct,  1, NULL},
    {"vdot", (PyCFunction)dotblas_vdot, 1, NULL},
    {"alterdot", (PyCFunction)dotblas_alterdot, 1, NULL},
    {"restoredot", (PyCFunction)dotblas_restoredot, 1, NULL},
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
    int i;

    PyObject *d, *s, *m;
    m = PyModule_Create(&moduledef);
#else
    int i;

    PyObject *d, *s;
    Py_InitModule3("_dotblas", dotblas_module_methods, module_doc);
#endif

    /* add the functions */

    /* Import the array object */
    import_array();

    /* Initialise the array of dot functions */
    for (i = 0; i < NPY_NTYPES; i++)
        oldFunctions[i] = NULL;

    /* alterdot at load */
    d = PyTuple_New(0);
    s = dotblas_alterdot(NULL, d);
    Py_DECREF(d);
    Py_DECREF(s);

    return RETVAL;
}
