static char module_doc[] =
"This module provides a BLAS optimized\nmatrix multiply, inner product and dot for scipy arrays";


#include "Python.h"
#include "scipy/arrayobject.h"
#ifndef CBLAS_HEADER
#define CBLAS_HEADER "cblas.h"
#endif
#include CBLAS_HEADER

#include <stdio.h>

static void 
FLOAT_dot(void *a, intp stridea, void *b, intp strideb, void *res, 
	  intp n, void *tmp)
{
    register int na = stridea / sizeof(float);
    register int nb = strideb / sizeof(float);

    *((float *)res) = cblas_sdot((int)n, (float *)a, na, (float *)b, nb);
}

static void 
DOUBLE_dot(void *a, intp stridea, void *b, intp strideb, void *res, 
	   intp n, void *tmp)
{
    register int na = stridea / sizeof(double);
    register int nb = strideb / sizeof(double);

    *((double *)res) = cblas_ddot((int)n, (double *)a, na, (double *)b, nb);
}

static void 
CFLOAT_dot(void *a, intp stridea, void *b, intp strideb, void *res, 
	   intp n, void *tmp)
{
    
    register int na = stridea / sizeof(cfloat);
    register int nb = strideb / sizeof(cfloat);

    cblas_cdotu_sub((int)n, (float *)a, na, (float *)b, nb, (float *)res);
}

static void 
CDOUBLE_dot(void *a, intp stridea, void *b, intp strideb, void *res, 
	    intp n, void *tmp)
{
    register int na = stridea / sizeof(cdouble);
    register int nb = strideb / sizeof(cdouble);

    cblas_zdotu_sub((int)n, (double *)a, na, (double *)b, nb, (double *)res);
}


static PyArray_DotFunc *oldFunctions[PyArray_NTYPES];
static Bool altered=FALSE;

static char doc_alterdot[] = "alterdot() changes all dot functions to use blas.";

static PyObject *
dotblas_alterdot(PyObject *dummy, PyObject *args) 
{
    PyArray_Descr *descr;
    
    if (!PyArg_ParseTuple(args, "")) return NULL;

    /* Replace the dot functions to the ones using blas */

    if (!altered) {
	descr = PyArray_DescrFromType(PyArray_FLOAT);
	oldFunctions[PyArray_FLOAT] = descr->dotfunc;
	descr->dotfunc = (PyArray_DotFunc *)FLOAT_dot;
	
	descr = PyArray_DescrFromType(PyArray_DOUBLE);
	oldFunctions[PyArray_DOUBLE] = descr->dotfunc;
	descr->dotfunc = (PyArray_DotFunc *)DOUBLE_dot;
	
	descr = PyArray_DescrFromType(PyArray_CFLOAT);
	oldFunctions[PyArray_CFLOAT] = descr->dotfunc;
	descr->dotfunc = (PyArray_DotFunc *)CFLOAT_dot;
	
	descr = PyArray_DescrFromType(PyArray_CDOUBLE);
	oldFunctions[PyArray_CDOUBLE] = descr->dotfunc;
	descr->dotfunc = (PyArray_DotFunc *)CDOUBLE_dot;

	altered = TRUE;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static char doc_restoredot[] = "restoredot() restores dots to defaults.";

static PyObject *
dotblas_restoredot(PyObject *dummy, PyObject *args) 
{
    PyArray_Descr *descr;

    if (!PyArg_ParseTuple(args, "")) return NULL;

    if (altered) {
	descr = PyArray_DescrFromType(PyArray_FLOAT);
	descr->dotfunc = oldFunctions[PyArray_FLOAT];
	oldFunctions[PyArray_FLOAT] = NULL;

	descr = PyArray_DescrFromType(PyArray_DOUBLE);
	descr->dotfunc = oldFunctions[PyArray_DOUBLE];
	oldFunctions[PyArray_DOUBLE] = NULL;

	descr = PyArray_DescrFromType(PyArray_CFLOAT);
	descr->dotfunc = oldFunctions[PyArray_CFLOAT];
	oldFunctions[PyArray_CFLOAT] = NULL;

	descr = PyArray_DescrFromType(PyArray_CDOUBLE);
	descr->dotfunc = oldFunctions[PyArray_CDOUBLE];
	oldFunctions[PyArray_CDOUBLE] = NULL;
	
	altered = FALSE;
    }

    Py_INCREF(Py_None);
    return Py_None;
}
      

static char doc_matrixproduct[] = "matrixproduct(a,b)\nReturns the dot product of a and b for arrays of floating point types.\nLike the generic scipy equivalent the product sum is over\nthe last dimension of a and the second-to-last dimension of b.\nNB: The first argument is not conjugated.";

static PyObject *
dotblas_matrixproduct(PyObject *dummy, PyObject *args) 
{
    PyObject *op1, *op2;
    PyArrayObject *ap1, *ap2, *ret;
    int j, l, lda, ldb;
    int typenum, nd;
    intp dimensions[MAX_DIMS];
    static const float oneF[2] = {1.0, 0.0};
    static const float zeroF[2] = {0.0, 0.0};
    static const double oneD[2] = {1.0, 0.0};
    static const double zeroD[2] = {0.0, 0.0};
    double prior1, prior2;
    PyTypeObject *subtype;
    PyArray_Typecode dtype = {PyArray_NOTYPE, 0, 0};


    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
	
    /* 
     * "Matrix product" using the BLAS.  
     * Only works for float double and complex types.
     */


    typenum = PyArray_ObjectType(op1, 0);  
    typenum = PyArray_ObjectType(op2, typenum);

    /* This function doesn't handle other types */
    if ((typenum != PyArray_DOUBLE && typenum != PyArray_CDOUBLE &&
	 typenum != PyArray_FLOAT && typenum != PyArray_CFLOAT)) {
            return PyArray_Return((PyArrayObject *)PyArray_MatrixProduct(op1, op2));
    }

    ret = NULL;
    dtype.type_num = typenum;
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, &dtype, 0, 0, CARRAY_FLAGS);
    if (ap1 == NULL) return NULL;
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, &dtype, 0, 0, CARRAY_FLAGS);
    if (ap2 == NULL) goto fail;

    if ((ap1->nd > 2) || (ap2->nd > 2)) {  
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
	ret = (PyArrayObject *)PyArray_MatrixProduct((PyObject *)ap1, 
						     (PyObject *)ap2);
	Py_DECREF(ap1); 
	Py_DECREF(ap2);
	return PyArray_Return(ret);
    }

    if (ap1->nd == 0 || ap2->nd == 0) {
	/* One of ap1 or ap2 is a scalar */
	if (ap1->nd == 0) {		/* Make ap2 the scalar */
	    PyArrayObject *t = ap1;
	    ap1 = ap2;
	    ap2 = t;
	}
	for (l = 1, j = 0; j < ap1->nd; j++) {
	    dimensions[j] = ap1->dimensions[j];
	    l *= dimensions[j];
	}
	nd = ap1->nd;
    }
    else { /* (ap1->nd <= 2 && ap2->nd <= 2) */
	/*  Both ap1 and ap2 are vectors or matrices */
	l = ap1->dimensions[ap1->nd-1];
	
	if (ap2->dimensions[0] != l) {
	    PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
	    goto fail;
	}
	nd = ap1->nd+ap2->nd-2;
 
	if (nd == 1) 
	    dimensions[0] = (ap1->nd == 2) ? ap1->dimensions[0] : ap2->dimensions[1];
	else if (nd == 2) {
	    dimensions[0] = ap1->dimensions[0];
	    dimensions[1] = ap2->dimensions[1];
	}
    }

    /* Choose which subtype to return */
    prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
    prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
    subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);
    
    ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions, 
				       typenum, NULL, NULL, 0, 0, 
				       (PyObject *)\
				       (prior2 > prior1 ? ap2 : ap1));  

    if (ret == NULL) goto fail;
    memset(ret->data, 0, PyArray_NBYTES(ret));

    if (ap2->nd == 0) {
	/* Multiplication by a scalar -- Level 1 BLAS */
	if (typenum == PyArray_DOUBLE) {
	    cblas_daxpy(l, *((double *)ap2->data), (double *)ap1->data, 1,
			(double *)ret->data, 1);
	} 
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zaxpy(l, (double *)ap2->data, (double *)ap1->data, 1,
			(double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_saxpy(l, *((float *)ap2->data), (float *)ap1->data, 1,
			(float *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_caxpy(l, (float *)ap2->data, (float *)ap1->data, 1,
			(float *)ret->data, 1);
	}
    }
    else if (ap1->nd == 1 && ap2->nd == 1) {
	/* Dot product between two vectors -- Level 1 BLAS */
	if (typenum == PyArray_DOUBLE) {
	    double result = cblas_ddot(l, (double *)ap1->data, 1, 
				       (double *)ap2->data, 1);
	    *((double *)ret->data) = result;
	}
	else if (typenum == PyArray_FLOAT) {
	    float result = cblas_sdot(l, (float *)ap1->data, 1, 
				      (float *)ap2->data, 1);
	    *((float *)ret->data) = result;
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zdotu_sub(l, (double *)ap1->data, 1, 
			    (double *)ap2->data, 1, (double *)ret->data);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cdotu_sub(l, (float *)ap1->data, 1, 
			    (float *)ap2->data, 1, (float *)ret->data);
	    fprintf(stderr, "Here...\n");
	}
    }
    else if (ap1->nd == 2 && ap2->nd == 1) {
	/* Matrix vector multiplication -- Level 2 BLAS */
	/* lda must be MAX(M,1) */
	lda = (ap1->dimensions[1] > 1 ? ap1->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			1.0, (double *)ap1->data, lda, 
			(double *)ap2->data, 1, 0.0, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			1.0, (float *)ap1->data, lda, 
			(float *)ap2->data, 1, 0.0, (float *)ret->data, 1);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			oneD, (double *)ap1->data, lda, 
			(double *)ap2->data, 1, zeroD, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			oneF, (float *)ap1->data, lda, 
			(float *)ap2->data, 1, zeroF, (float *)ret->data, 1);
	}
    }
    else if (ap1->nd == 1 && ap2->nd == 2) {
	/* Vector matrix multiplication -- Level 2 BLAS */
	lda = (ap2->dimensions[1] > 1 ? ap2->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemv(CblasRowMajor, 
			CblasTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			1.0, (double *)ap2->data, lda,
			(double *)ap1->data, 1, 0.0, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemv(CblasRowMajor, 
			CblasTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			1.0, (float *)ap2->data, lda,
			(float *)ap1->data, 1, 0.0, (float *)ret->data, 1);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemv(CblasRowMajor, 
			CblasTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			oneD, (double *)ap2->data, lda,
			(double *)ap1->data, 1, zeroD, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemv(CblasRowMajor, 
			CblasTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			oneF, (float *)ap2->data, lda,
			(float *)ap1->data, 1, zeroF, (float *)ret->data, 1);
	}
    }
    else { /* (ap1->nd == 2 && ap2->nd == 2) */
	/* Matrix matrix multiplication -- Level 3 BLAS */  
	lda = (ap1->dimensions[1] > 1 ? ap1->dimensions[1] : 1);
	ldb = (ap2->dimensions[1] > 1 ? ap2->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			ap1->dimensions[0], ap2->dimensions[1], ap2->dimensions[0],
			1.0, (double *)ap1->data, lda,
			(double *)ap2->data, ldb,
			0.0, (double *)ret->data, ldb);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			ap1->dimensions[0], ap2->dimensions[1], ap2->dimensions[0],
			1.0, (float *)ap1->data, lda,
			(float *)ap2->data, ldb,
			0.0, (float *)ret->data, ldb);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			ap1->dimensions[0], ap2->dimensions[1], ap2->dimensions[0],
			oneD, (double *)ap1->data, lda,
			(double *)ap2->data, ldb,
			zeroD, (double *)ret->data, ldb);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			ap1->dimensions[0], ap2->dimensions[1], ap2->dimensions[0],
			oneF, (float *)ap1->data, lda,
			(float *)ap2->data, ldb,
			zeroF, (float *)ret->data, ldb);
	}
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


static char doc_innerproduct[] = "innerproduct(a,b)\nReturns the inner product of a and b for arrays of floating point types.\nLike the generic Numeric equivalent the product sum is over\nthe last dimension of a and b.\nNB: The first argument is not conjugated.";

static PyObject *
dotblas_innerproduct(PyObject *dummy, PyObject *args) 
{
    PyObject *op1, *op2;
    PyArrayObject *ap1, *ap2, *ret;
    int j, l, lda, ldb;
    int typenum, nd;
    intp dimensions[MAX_DIMS];
    static const float oneF[2] = {1.0, 0.0};
    static const float zeroF[2] = {0.0, 0.0};
    static const double oneD[2] = {1.0, 0.0};
    static const double zeroD[2] = {0.0, 0.0};
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
    if ((typenum != PyArray_DOUBLE && typenum != PyArray_CDOUBLE &&
	 typenum != PyArray_FLOAT && typenum != PyArray_CFLOAT)) {
            return PyArray_Return((PyArrayObject *)PyArray_InnerProduct(op1, op2));
    }

    ret = NULL;
    ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
    if (ap1 == NULL) return NULL;
    ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
    if (ap2 == NULL) goto fail;


    if ((ap1->nd > 2) || (ap2->nd > 2)) {  
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

    if (ap1->nd == 0 || ap2->nd == 0) {
	/* One of ap1 or ap2 is a scalar */
	if (ap1->nd == 0) {		/* Make ap2 the scalar */
	    PyArrayObject *t = ap1;
	    ap1 = ap2;
	    ap2 = t;
	}
	for (l = 1, j = 0; j < ap1->nd; j++) {
	    dimensions[j] = ap1->dimensions[j];
	    l *= dimensions[j];
	}
	nd = ap1->nd;
    }
    else { /* (ap1->nd <= 2 && ap2->nd <= 2) */
	/*  Both ap1 and ap2 are vectors or matrices */
	l = ap1->dimensions[ap1->nd-1];
	
	if (ap2->dimensions[ap2->nd-1] != l) {
	    PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
	    goto fail;
	}
	nd = ap1->nd+ap2->nd-2;
 
	if (nd == 1) 
	    dimensions[0] = (ap1->nd == 2) ? ap1->dimensions[0] : ap2->dimensions[0];
	else if (nd == 2) {
	    dimensions[0] = ap1->dimensions[0];
	    dimensions[1] = ap2->dimensions[0];
	}
    }

    /* Choose which subtype to return */
    prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
    prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
    subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);
    
    ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions, 
				       typenum, NULL, NULL, 0, 0, 
				       (PyObject *)\
				       (prior2 > prior1 ? ap2 : ap1));
    
    if (ret == NULL) goto fail;
    memset(ret->data, 0, PyArray_NBYTES(ret));

    if (ap2->nd == 0) {
	/* Multiplication by a scalar -- Level 1 BLAS */
	if (typenum == PyArray_DOUBLE) {
	    cblas_daxpy(l, *((double *)ap2->data), (double *)ap1->data, 1,
			(double *)ret->data, 1);
	} 
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zaxpy(l, (double *)ap2->data, (double *)ap1->data, 1,
			(double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_saxpy(l, *((float *)ap2->data), (float *)ap1->data, 1,
			(float *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_caxpy(l, (float *)ap2->data, (float *)ap1->data, 1,
			(float *)ret->data, 1);
	}
    }
    else if (ap1->nd == 1 && ap2->nd == 1) {
	/* Dot product between two vectors -- Level 1 BLAS */
	if (typenum == PyArray_DOUBLE) {
	    double result = cblas_ddot(l, (double *)ap1->data, 1, 
				       (double *)ap2->data, 1);
	    *((double *)ret->data) = result;
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zdotu_sub(l, (double *)ap1->data, 1, 
			    (double *)ap2->data, 1, (double *)ret->data);
	}
	else if (typenum == PyArray_FLOAT) {
	    float result = cblas_sdot(l, (float *)ap1->data, 1, 
				      (float *)ap2->data, 1);
	    *((float *)ret->data) = result;
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cdotu_sub(l, (float *)ap1->data, 1, 
			    (float *)ap2->data, 1, (float *)ret->data);
	}
    }
    else if (ap1->nd == 2 && ap2->nd == 1) {
	/* Matrix-vector multiplication -- Level 2 BLAS */
	lda = (ap1->dimensions[1] > 1 ? ap1->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			1.0, (double *)ap1->data, lda,
			(double *)ap2->data, 1, 0.0, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			oneD, (double *)ap1->data, lda,
			(double *)ap2->data, 1, zeroD, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			1.0, (float *)ap1->data, lda,
			(float *)ap2->data, 1, 0.0, (float *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemv(CblasRowMajor, 
			CblasNoTrans,  ap1->dimensions[0], ap1->dimensions[1], 
			oneF, (float *)ap1->data, lda,
			(float *)ap2->data, 1, zeroF, (float *)ret->data, 1);
	}
    }
    else if (ap1->nd == 1 && ap2->nd == 2) {
	/* Vector matrix multiplication -- Level 2 BLAS */
	lda = (ap2->dimensions[1] > 1 ? ap2->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemv(CblasRowMajor, 
			CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			1.0, (double *)ap2->data, lda,
			(double *)ap1->data, 1, 0.0, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemv(CblasRowMajor, 
			CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			oneD, (double *)ap2->data, lda,
			(double *)ap1->data, 1, zeroD, (double *)ret->data, 1);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemv(CblasRowMajor, 
			CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			1.0, (float *)ap2->data, lda,
			(float *)ap1->data, 1, 0.0, (float *)ret->data, 1);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemv(CblasRowMajor, 
			CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
			oneF, (float *)ap2->data, lda,
			(float *)ap1->data, 1, zeroF, (float *)ret->data, 1);
	}
    }
    else { /* (ap1->nd == 2 && ap2->nd == 2) */  
	/* Matrix matrix multiplication -- Level 3 BLAS */  
	lda = (ap1->dimensions[1] > 1 ? ap1->dimensions[1] : 1);
	ldb = (ap2->dimensions[1] > 1 ? ap2->dimensions[1] : 1);
	if (typenum == PyArray_DOUBLE) {
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			ap1->dimensions[0], ap2->dimensions[0], ap1->dimensions[1],
			1.0, (double *)ap1->data, lda,
			(double *)ap2->data, ldb,
			0.0, (double *)ret->data, ldb);
	}
	else if (typenum == PyArray_FLOAT) {
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			ap1->dimensions[0], ap2->dimensions[0], ap1->dimensions[1],
			1.0, (float *)ap1->data, lda,
			(float *)ap2->data, ldb,
			0.0, (float *)ret->data, ldb);
	}
	else if (typenum == PyArray_CDOUBLE) {
	    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			ap1->dimensions[0], ap2->dimensions[0], ap1->dimensions[1],
			oneD, (double *)ap1->data, lda,
			(double *)ap2->data, ldb,
			zeroD, (double *)ret->data, ldb);
	}
	else if (typenum == PyArray_CFLOAT) {
	    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			ap1->dimensions[0], ap2->dimensions[0], ap1->dimensions[1],
			oneF, (float *)ap1->data, lda,
			(float *)ap2->data, ldb,
			zeroF, (float *)ret->data, ldb);
	}
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


static char doc_vdot[] = "vdot(a,b)\nReturns the dot product of a and b for scalars and vectors\nof floating point and complex types.  The first argument, a, is conjugated.";


static PyObject *dotblas_vdot(PyObject *dummy, PyObject *args) {
    PyObject *op1, *op2;
    PyArrayObject *ap1=NULL, *ap2=NULL, *ret=NULL;
    int l;
    int typenum;
    intp dimensions[MAX_DIMS];
    PyArray_Typecode type;

    if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
	
    /* 
     * Conjugating dot product using the BLAS for vectors.
     * Multiplies op1 and op2, each of which must be vector.
     */

    typenum = PyArray_ObjectType(op1, 0);  
    typenum = PyArray_ObjectType(op2, typenum);
    
    type.type_num = typenum;
    
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, &type, 0, 0, 0);
    if (ap1==NULL) goto fail;
    op1 = PyArray_Flatten(ap1, 0);
    if (op1==NULL) goto fail;
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;
    
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, &type, 0, 0, 0);
    if (ap2==NULL) goto fail;
    op2 = PyArray_Flatten(ap2, 0);
    if (op2 == NULL) goto fail;
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;
    
    if (typenum != PyArray_FLOAT && typenum != PyArray_DOUBLE &&
	typenum != PyArray_CFLOAT && typenum != PyArray_CDOUBLE) {
	if (!altered) {
	    /* need to alter dot product */
	    PyObject *tmp1, *tmp2;
	    tmp1 = PyTuple_New(0);
	    tmp2 = dotblas_alterdot(NULL, tmp1);
	    Py_DECREF(tmp1); 
	    Py_DECREF(tmp2);
	}
	if (PyTypeNum_ISCOMPLEX(typenum)) {
	    op1 = PyArray_Conjugate(ap1);
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

    if (ap2->dimensions[0] != ap1->dimensions[ap1->nd-1]) {
	PyErr_SetString(PyExc_ValueError, "vectors have different lengths");
	goto fail;
    }
    l = ap1->dimensions[ap1->nd-1];
  
    ret = (PyArrayObject *)PyArray_SimpleNew(0, dimensions, typenum);
    if (ret == NULL) goto fail;


    /* Dot product between two vectors -- Level 1 BLAS */
    if (typenum == PyArray_DOUBLE) {
	*((double *)ret->data) = cblas_ddot(l, (double *)ap1->data, 1, 
					    (double *)ap2->data, 1);
    }
    else if (typenum == PyArray_FLOAT) {
	*((float *)ret->data) = cblas_sdot(l, (float *)ap1->data, 1, 
					   (float *)ap2->data, 1);
    }
    else if (typenum == PyArray_CDOUBLE) {
	cblas_zdotc_sub(l, (double *)ap1->data, 1, 
			(double *)ap2->data, 1, (double *)ret->data);
    }
    else if (typenum == PyArray_CFLOAT) {
	cblas_cdotc_sub(l, (float *)ap1->data, 1, 
			(float *)ap2->data, 1, (float *)ret->data);
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

static struct PyMethodDef dotblas_module_methods[] = {
    {"dot",  (PyCFunction)dotblas_matrixproduct, 1, doc_matrixproduct},
    {"inner",   (PyCFunction)dotblas_innerproduct,  1, doc_innerproduct},
    {"vdot", (PyCFunction)dotblas_vdot, 1, doc_vdot},
    {"alterdot", (PyCFunction)dotblas_alterdot, 1, doc_alterdot},
    {"restoredot", (PyCFunction)dotblas_restoredot, 1, doc_restoredot},
    {NULL,		NULL, 0}		/* sentinel */
};

/* Initialization function for the module */
DL_EXPORT(void) init_dotblas(void) {
    int i;
    PyObject *m, *d, *s;
    
    /* Create the module and add the functions */
    m = Py_InitModule3("_dotblas", dotblas_module_methods, module_doc);

    /* Import the array object */
    import_array();

    /* Initialise the array of dot functions */
    for (i = 0; i < PyArray_NTYPES; i++)
	oldFunctions[i] = NULL;

    /* alterdot at load */
    d = PyTuple_New(0);
    s = dotblas_alterdot(NULL, d);
    Py_DECREF(d);
    Py_DECREF(s);
    
    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("can't initialize module _dotblas");
}
