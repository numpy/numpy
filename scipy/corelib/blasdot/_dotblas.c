static char module_doc[] =
"This module provides a BLAS optimized\nmatrix multiply, inner product and dot for Numeric arrays";


#include "Python.h"
#include "Numeric/arrayobject.h"
#ifndef CBLAS_HEADER
#define CBLAS_HEADER <cblas.h>
#endif
#include CBLAS_HEADER

#include <stdio.h>

/* Defined to be the same as MAX_DIMS in multiarray */
#define MAX_DIMS 40



static void FLOAT_dot(void *a, int stridea, void *b, int strideb, void *res, int n)
{
  *((float *)res) = cblas_sdot(n, (float *)a, stridea, (float *)b, strideb);
}

static void DOUBLE_dot(void *a, int stridea, void *b, int strideb, void *res, int n)
{
  *((double *)res) = cblas_ddot(n, (double *)a, stridea, (double *)b, strideb);
}

static void CFLOAT_dot(void *a, int stridea, void *b, int strideb, void *res, int n)
{
  cblas_cdotu_sub(n, (double *)a, stridea, (double *)b, strideb,
                  (double *)res);
}

static void CDOUBLE_dot(void *a, int stridea, void *b, int strideb, void *res, int n)
{
  cblas_zdotu_sub(n, (double *)a, stridea, (double *)b, strideb,
                  (double *)res);
}




typedef void (DotFunction)(void *, int, void *, int, void *, int);

static DotFunction *dotFunctions[PyArray_NTYPES];


static char doc_matrixproduct[] = "matrixproduct(a,b)\nReturns the dot product of a and b for arrays of floating point types.\nLike the generic Numeric equivalent the product sum is over\nthe last dimension of a and the second-to-last dimension of b.\nNB: The first argument is not conjugated.";


static PyObject *dotblas_matrixproduct(PyObject *dummy, PyObject *args) {
  PyObject *op1, *op2;
  PyArrayObject *ap1, *ap2, *ret;
  int i, j, l, lda, ldb, matchDim = -1, otherDim = -1;
  int typenum;
  int dimensions[MAX_DIMS], nd;
  static const float oneF[2] = {1.0, 0.0};
  static const float zeroF[2] = {0.0, 0.0};
  static const double oneD[2] = {1.0, 0.0};
  static const double zeroD[2] = {0.0, 0.0};


  if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
	
  /* 
   * "Matrix product" using the BLAS.  
   * Only works for float double and complex types.
   */


  typenum = PyArray_ObjectType(op1, 0);  
  typenum = PyArray_ObjectType(op2, typenum);

  ret = NULL;
  ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
  if (ap1 == NULL) return NULL;
  ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
  if (ap2 == NULL) goto fail;

  if (typenum != PyArray_FLOAT && typenum != PyArray_DOUBLE &&
      typenum != PyArray_CFLOAT && typenum != PyArray_CDOUBLE) {
   PyErr_SetString(PyExc_TypeError, "at least one argument must be (possibly complex) float or double");
    goto fail;
  }

  
  if (ap1->nd < 0 || ap2->nd < 0) {
    PyErr_SetString(PyExc_TypeError, "negative dimensioned arrays!");
    goto fail;
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
  else if (ap1->nd <= 2 && ap2->nd <= 2) {
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
  else {
    /* At least one of ap1 or ap2 has dimension > 2. */
    l = ap1->dimensions[ap1->nd-1];
    matchDim = ap2->nd - 2;
    otherDim = ap2->nd - 1;
    
    if (ap2->dimensions[matchDim] != l) {
	PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
	goto fail;
    }

    nd = ap1->nd+ap2->nd-2;
    j = 0;
    for(i=0; i<ap1->nd-1; i++) {
	dimensions[j++] = ap1->dimensions[i];
    }
    for(i=0; i<ap2->nd-2; i++) {
	dimensions[j++] = ap2->dimensions[i];
    }
    if(ap2->nd > 1) {
      dimensions[j++] = ap2->dimensions[ap2->nd-1];
    }
  }

  ret = (PyArrayObject *)PyArray_FromDims(nd, dimensions, typenum);
  if (ret == NULL) goto fail;

  if (ap2->nd == 0) {
    /* Multiplication by a scalar -- Level 1 BLAS */
    if (typenum == PyArray_DOUBLE) {
      cblas_daxpy(l, *((double *)ap2->data), (double *)ap1->data, 1,
		  (double *)ret->data, 1);
    } 
    else  if (typenum == PyArray_FLOAT) {
      cblas_saxpy(l, *((float *)ap2->data), (float *)ap1->data, 1,
		  (float *)ret->data, 1);
    }
    else if (typenum == PyArray_CDOUBLE) {
      cblas_zaxpy(l, (double *)ap2->data, (double *)ap1->data, 1,
		  (double *)ret->data, 1);
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
  else if (ap1->nd == 2 && ap2->nd == 2) {
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
  else {
    /* Deal with arrays with dimension greater than two.  All we do is copy
     * the original multiarraymodule.c/PyArray_InnerProduct but use the BLAS
     * one dimensional functions instead of the macro versions in
     * multiarraymodule
     */

    int is1, is2, is1r, is2r, n1, n2, os, i1, i2;
    char *ip1, *ip2, *op;
    DotFunction *dot = dotFunctions[(int)(ret->descr->type_num)];
    if (dot == NULL) {
      PyErr_SetString(PyExc_ValueError, 
                      "dotblas matrixMultiply not available for this type");
      goto fail;
    }

    n1 = PyArray_SIZE(ap1)/l;
    n2 = PyArray_SIZE(ap2)/l;
    is1 = ap1->strides[ap1->nd-1]/ap1->descr->elsize;
    is2 = ap2->strides[matchDim]/ap2->descr->elsize;
    if(ap1->nd > 1)
      is1r = ap1->strides[ap1->nd-2];
    else
      is1r = ap1->strides[ap1->nd-1];
    is2r = ap2->strides[otherDim];
    op = ret->data;
    os = ret->descr->elsize;

    ip1 = ap1->data;
    for(i1=0; i1<n1; i1++) {
      ip2 = ap2->data;
      for(i2=0; i2<n2; i2++) {
        dot(ip1, is1, ip2, is2, op, l);
        ip2 += is2r;
        op += os;
      }
      ip1 += is1r;
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

static PyObject *dotblas_innerproduct(PyObject *dummy, PyObject *args) {
  PyObject *op1, *op2;
  PyArrayObject *ap1, *ap2, *ret;
  int i, j, l, lda, ldb;
  int typenum;
  int dimensions[MAX_DIMS], nd;
  static const float oneF[2] = {1.0, 0.0};
  static const float zeroF[2] = {0.0, 0.0};
  static const double oneD[2] = {1.0, 0.0};
  static const double zeroD[2] = {0.0, 0.0};


  if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
	
  /* 
   * Inner product using the BLAS.  The product sum is taken along the last
   * dimensions of the two arrays.
   * Only works for float double and complex types.
   */


  typenum = PyArray_ObjectType(op1, 0);  
  typenum = PyArray_ObjectType(op2, typenum);

  ret = NULL;
  ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
  if (ap1 == NULL) return NULL;
  ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
  if (ap2 == NULL) goto fail;

  if (typenum != PyArray_FLOAT && typenum != PyArray_DOUBLE &&
      typenum != PyArray_CFLOAT && typenum != PyArray_CDOUBLE) {
    PyErr_SetString(PyExc_TypeError, "at least one argument must be (possibly complex) float or double");
    goto fail;
  }

  
  if (ap1->nd < 0 || ap2->nd < 0) {
    PyErr_SetString(PyExc_TypeError, "negative dimensioned arrays!");
    goto fail;
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
  else if (ap1->nd <= 2 && ap2->nd <= 2) {
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
  else {
    /* At least one of ap1 or ap2 has dimension > 2. */
    l = ap1->dimensions[ap1->nd-1];
    
    if (ap2->dimensions[ap2->nd-1] != l) {
      PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
      goto fail;
    }

    nd = ap1->nd+ap2->nd-2;
    j = 0;
    for(i=0; i<ap1->nd-1; i++) {
	dimensions[j++] = ap1->dimensions[i];
    }
    for(i=0; i<ap2->nd-1; i++) {
	dimensions[j++] = ap2->dimensions[i];
    }
  }

  ret = (PyArrayObject *)PyArray_FromDims(nd, dimensions, typenum);
  if (ret == NULL) goto fail;

  if (ap2->nd == 0) {
    /* Multiplication by a scalar -- Level 1 BLAS */
    if (typenum == PyArray_DOUBLE) {
      cblas_daxpy(l, *((double *)ap2->data), (double *)ap1->data, 1,
		  (double *)ret->data, 1);
    } 
    else  if (typenum == PyArray_FLOAT) {
      cblas_saxpy(l, *((float *)ap2->data), (float *)ap1->data, 1,
		  (float *)ret->data, 1);
    }
    else if (typenum == PyArray_CDOUBLE) {
      cblas_zaxpy(l, (double *)ap2->data, (double *)ap1->data, 1,
		  (double *)ret->data, 1);
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
		  CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
		  1.0, (double *)ap2->data, lda,
		  (double *)ap1->data, 1, 0.0, (double *)ret->data, 1);
    }
    else if (typenum == PyArray_FLOAT) {
      cblas_sgemv(CblasRowMajor, 
		  CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
		  1.0, (float *)ap2->data, lda,
		  (float *)ap1->data, 1, 0.0, (float *)ret->data, 1);
    }
    else if (typenum == PyArray_CDOUBLE) {
      cblas_zgemv(CblasRowMajor, 
		  CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
		  oneD, (double *)ap2->data, lda,
		  (double *)ap1->data, 1, zeroD, (double *)ret->data, 1);
    }
    else if (typenum == PyArray_CFLOAT) {
      cblas_cgemv(CblasRowMajor, 
		  CblasNoTrans,  ap2->dimensions[0], ap2->dimensions[1], 
		  oneF, (float *)ap2->data, lda,
		  (float *)ap1->data, 1, zeroF, (float *)ret->data, 1);
    }
  }
  else if (ap1->nd == 2 && ap2->nd == 2) {
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
  else {
    /* Deal with arrays with dimension greater than two.  All we do is copy
     * the original multiarraymodule.c/PyArray_InnerProduct but use the BLAS
     * one dimensional functions instead of the macro versions in
     * multiarraymodule
     */

    int is1, is2, n1, n2, os, i1, i2, s1, s2;
    char *ip1, *ip2, *op;
    DotFunction *dot = dotFunctions[(int)(ret->descr->type_num)];
    if (dot == NULL) {
      PyErr_SetString(PyExc_ValueError, 
                      "dotblas inner product not available for this type");
      goto fail;
    }

    n1 = PyArray_SIZE(ap1)/l;
    n2 = PyArray_SIZE(ap2)/l;
    is1 = ap1->strides[ap1->nd-1];
    s1 = is1/ap1->descr->elsize;
    is2 = ap2->strides[ap2->nd-1];
    s2 = is2/ap2->descr->elsize;
    op = ret->data;
    os = ret->descr->elsize;
	
    ip1 = ap1->data;
    for(i1=0; i1<n1; i1++) {
      ip2 = ap2->data;
      for(i2=0; i2<n2; i2++) {
        dot(ip1, s1, ip2, s2, op, l);
        ip2 += is2*l;
        op += os;
      }
      ip1 += is1*l;
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
  PyArrayObject *ap1, *ap2, *ret;
  int l;
  int typenum;
  int dimensions[MAX_DIMS];

  if (!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
	
  /* 
   * Conjugating dot product using the BLAS for vectors.
   * Multiplies op1 and op2, each of which must be vector.
   */

  typenum = PyArray_ObjectType(op1, 0);  
  typenum = PyArray_ObjectType(op2, typenum);


  ret = NULL;
  ap1 = (PyArrayObject *)PyArray_ContiguousFromObject(op1, typenum, 0, 0);
  if (ap1 == NULL) return NULL;
  ap2 = (PyArrayObject *)PyArray_ContiguousFromObject(op2, typenum, 0, 0);
  if (ap2 == NULL) goto fail;

  if (typenum != PyArray_FLOAT && typenum != PyArray_DOUBLE &&
      typenum != PyArray_CFLOAT && typenum != PyArray_CDOUBLE) {
    PyErr_SetString(PyExc_TypeError, "at least one argument must be (possibly complex) float or double");
    goto fail;
  }

  if (ap1->nd != 1 || ap2->nd != 1) {
    PyErr_SetString(PyExc_TypeError, "arguments must be vectors");
    goto fail;
  }

  if (ap2->dimensions[0] != ap1->dimensions[ap1->nd-1]) {
    PyErr_SetString(PyExc_ValueError, "vectors have different lengths");
    goto fail;
  }
  l = ap1->dimensions[ap1->nd-1];
  
  ret = (PyArrayObject *)PyArray_FromDims(0, dimensions, typenum);
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
  {"matrixproduct",  (PyCFunction)dotblas_matrixproduct, 1, doc_matrixproduct},
  {"innerproduct",   (PyCFunction)dotblas_innerproduct,  1, doc_innerproduct},
  {"vdot", (PyCFunction)dotblas_vdot, 1, doc_vdot},
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

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    s = PyString_FromString("$Id: _dotblas.c,v 1.3 2005/04/06 22:40:23 dmcooke Exp $");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Initialise the array of dot functions */
    for (i = 0; i < PyArray_NTYPES; i++)
      dotFunctions[i] = NULL;
    dotFunctions[PyArray_FLOAT] = FLOAT_dot;
    dotFunctions[PyArray_DOUBLE] = DOUBLE_dot;
    dotFunctions[PyArray_CFLOAT] = CFLOAT_dot;
    dotFunctions[PyArray_CDOUBLE] = CDOUBLE_dot;

    
    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("can't initialize module _dotblas");
}
