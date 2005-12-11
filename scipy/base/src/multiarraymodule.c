/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file 
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for scipy_core in 2005 

  Travis E. Oliphant
  Assistant Professor at
  Brigham Young University
 
*/

/* $Id: multiarraymodule.c,v 1.36 2005/09/14 00:14:00 teoliphant Exp $ */

#include "Python.h"
#include "structmember.h"
/*#include <string.h>
#include <math.h>
*/

#define _MULTIARRAYMODULE
#include "scipy/arrayobject.h"

#define PyAO PyArrayObject

static PyObject *typeDict=NULL;   /* Must be explicitly loaded */



static PyArray_Descr *
_arraydescr_fromobj(PyObject *obj)
{
	PyObject *dtypedescr;
	PyArray_Descr *new;
	int ret;
	
	dtypedescr = PyObject_GetAttrString(obj, "dtypedescr");
	PyErr_Clear();
	if (dtypedescr) {
		ret = PyArray_DescrConverter(dtypedescr, &new);
		Py_DECREF(dtypedescr);
		if (ret) return new;
		PyErr_Clear();
	}
	return NULL;
}


/* Including this file is the only way I know how to declare functions
   static in each file, and store the pointers from functions in both
   arrayobject.c and multiarraymodule.c for the C-API 

   Declarying an external pointer-containing variable in arrayobject.c
   and trying to copy it to PyArray_API, did not work.

   Think about two modules with a common api that import each other...

   This file would just be the module calls. 
*/

#include "arrayobject.c"


/* An Error object -- rarely used? */
static PyObject *MultiArrayError;

/*MULTIARRAY_API
 Multiply a List of ints
*/
static int
PyArray_MultiplyIntList(register int *l1, register int n) 
{
	register int s=1;
        while (n--) s *= (*l1++);
        return s;
}

/*MULTIARRAY_API
 Multiply a List
*/
static intp 
PyArray_MultiplyList(register intp *l1, register int n) 
{
	register intp s=1;
        while (n--) s *= (*l1++);
        return s;
}

/*MULTIARRAY_API
 Produce a pointer into array
*/
static char *
PyArray_GetPtr(PyArrayObject *obj, register intp* ind)
{
	register int n = obj->nd;
	register intp *strides = obj->strides;
	register char *dptr = obj->data;
	
	while (n--) dptr += (*strides++) * (*ind++);
	return dptr;
}

/*MULTIARRAY_API
 Get axis from an object (possibly None) -- a converter function,
*/
static int 
PyArray_AxisConverter(PyObject *obj, int *axis)
{
	if (obj == Py_None) {
		*axis = MAX_DIMS;
	}
	else {
		*axis = (int) PyInt_AsLong(obj);
		if (PyErr_Occurred()) {
			return PY_FAIL;
		}
	}
	return PY_SUCCEED;
}

/*MULTIARRAY_API
 Compare Lists
*/
static int 
PyArray_CompareLists(intp *l1, intp *l2, int n) 
{
        int i;
        for(i=0;i<n;i++) {
                if (l1[i] != l2[i]) return 0;
        } 
        return 1;
}

/* steals a reference to type -- accepts NULL */
/*MULTIARRAY_API
 View
*/
static PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type)
{
	PyObject *new=NULL;
 
	Py_INCREF(self->descr);
	new = PyArray_NewFromDescr(self->ob_type,
				   self->descr,
				   self->nd, self->dimensions,
				   self->strides,
				   self->data,
				   self->flags, (PyObject *)self);
	
	if (new==NULL) return NULL;
	Py_INCREF(self);
        PyArray_BASE(new) = (PyObject *)self;
	
	if (type != NULL) {
		if (PyObject_SetAttrString(new, "dtype", 
					   (PyObject *)type) < 0) {
			Py_DECREF(new);
			Py_DECREF(type);
			return NULL;
		}
		Py_DECREF(type);
	}
	return new;	
}

/*MULTIARRAY_API
 Ravel
*/
static PyObject *
PyArray_Ravel(PyArrayObject *a, int fortran)
{
	PyArray_Dims newdim = {NULL,1};
	intp val[1] = {-1};

	if (fortran < 0) fortran = PyArray_ISFORTRAN(a);

	newdim.ptr = val;
	if (!fortran && PyArray_ISCONTIGUOUS(a)) {
		if (a->nd == 1) {
			Py_INCREF(a);
			return (PyObject *)a;
		}
		return PyArray_Newshape(a, &newdim);
	}
	else
	        return PyArray_Flatten(a, fortran);
}

/*MULTIARRAY_API
 Flatten
*/
static PyObject *
PyArray_Flatten(PyArrayObject *a, int fortran)
{
	PyObject *ret, *new;
	intp size;

	if (fortran < 0) fortran = PyArray_ISFORTRAN(a);

	size = PyArray_SIZE(a);
	Py_INCREF(a->descr);
	ret = PyArray_NewFromDescr(a->ob_type,
				   a->descr,
				   1, &size,
				   NULL,
				   NULL,
				   0, (PyObject *)a);
	
	if (ret== NULL) return NULL;
	if (fortran) {
		new = PyArray_Transpose(a, NULL);
		if (new == NULL) {
			Py_DECREF(ret);
			return NULL;
		}
	}
	else {
		Py_INCREF(a);
		new = (PyObject *)a;
	}
	if (PyArray_CopyInto((PyArrayObject *)ret, (PyArrayObject *)new) < 0) {
		Py_DECREF(ret);
		Py_DECREF(new);
		return NULL;
	}
	Py_DECREF(new);
	return ret;
}


/* For back-ward compatability *

/ * Not recommended */

/*MULTIARRAY_API
 Reshape an array
*/
static PyObject *
PyArray_Reshape(PyArrayObject *self, PyObject *shape) 
{
        PyObject *ret;
        PyArray_Dims newdims;

        if (!PyArray_IntpConverter(shape, &newdims)) return NULL;
        ret = PyArray_Newshape(self, &newdims);
        PyDimMem_FREE(newdims.ptr);
        return ret;
}

static int
_check_ones(PyArrayObject *self, int newnd, intp* newdims, intp *strides)
{
	int nd;
	intp *dims;
	Bool done=FALSE;
	int j, k;

	nd = self->nd;
	dims = self->dimensions;

	for (k=0, j=0; !done && (j<nd || k<newnd);) {
		if ((j<nd) && (k<newnd) && (newdims[k]==dims[j])) {
			strides[k] = self->strides[j];
			j++; k++;
		}
		else if ((k<newnd) && (newdims[k]==1)) {
			strides[k] = 0;
			k++;
		}
		else if ((j<nd) && (dims[j]==1)) {
			j++;
		}
		else done=TRUE;
	}
	if (done) return -1;
	return 0;
}

/* Returns a new array 
   with the new shape from the data
   in the old array
*/

/*MULTIARRAY_API
 New shape for an array
*/
static PyObject * 
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims)
{
        intp i, s_original, i_unknown, s_known;
        intp *dimensions = newdims->ptr;
        PyArrayObject *ret;
	char msg[] = "total size of new array must be unchanged";
	int n = newdims->len;
        Bool same;
	intp *strides = NULL;
	intp newstrides[MAX_DIMS];

        /*  Quick check to make sure anything needs to be done */
        if (n == self->nd) {
                same = TRUE;
                i=0;
                while(same && i<n) {
                        if (PyArray_DIM(self,i) != dimensions[i]) 
                                same=FALSE;
                        i++;
                }
                if (same) return PyArray_View(self, NULL);
        }
	
	/* Returns a pointer to an appropriate strides array
	   if all we are doing is inserting ones into the shape,
	   or removing ones from the shape 
	   or doing a combination of the two*/
	i=_check_ones(self, n, dimensions, newstrides);
	if (i==0) strides=newstrides;
	
	if (strides==NULL) {
		if (!PyArray_ISCONTIGUOUS(self)) {
			PyErr_SetString(PyExc_ValueError, 
					"changing shape that way "\
					"only works on contiguous arrays");
			return NULL;
		}
		
		s_known = 1;
		i_unknown = -1;
		
		for(i=0; i<n; i++) {
			if (dimensions[i] < 0) {
				if (i_unknown == -1) {
					i_unknown = i;
				} else {
					PyErr_SetString(PyExc_ValueError, 
							"can only specify one" \
							" unknown dimension");
					return NULL;
				}
			} else {
				s_known *= dimensions[i];
			}
		}
		
		s_original = PyArray_SIZE(self);
		
		if (i_unknown >= 0) {
			if ((s_known == 0) || (s_original % s_known != 0)) {
				PyErr_SetString(PyExc_ValueError, msg);
				return NULL;
			}
			dimensions[i_unknown] = s_original/s_known;
		} else {
			if (s_original != s_known) {
				PyErr_SetString(PyExc_ValueError, msg);
				return NULL;
			}
		}
	}
        
	Py_INCREF(self->descr);
	ret = (PyAO *)PyArray_NewFromDescr(self->ob_type,
					   self->descr,
					   n, dimensions,
					   strides,
					   self->data,
					   self->flags, (PyObject *)self);
	
	if (ret== NULL) return NULL;
	
        Py_INCREF(self);
        ret->base = (PyObject *)self;
	PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
	
        return (PyObject *)ret;
}

/* return a new view of the array object with all of its unit-length 
   dimensions squeezed out if needed, otherwise
   return the same array.
 */

/*MULTIARRAY_API*/
static PyObject *
PyArray_Squeeze(PyArrayObject *self)
{
	int nd = self->nd;
	int newnd = nd;
	intp dimensions[MAX_DIMS];
	intp strides[MAX_DIMS];
	int i,j;
	PyObject *ret;

	if (nd == 0) {
		Py_INCREF(self);
		return (PyObject *)self;
	}
	for (j=0, i=0; i<nd; i++) {
		if (self->dimensions[i] == 1) {
			newnd -= 1;
		}
		else {
			dimensions[j] = self->dimensions[i];
			strides[j++] = self->strides[i];
		}
	}
	
	Py_INCREF(self->descr);
	ret = PyArray_NewFromDescr(self->ob_type, 
				   self->descr,
				   newnd, dimensions, 
				   strides, self->data, 
				   self->flags,
				   (PyObject *)self);
	if (ret == NULL) return NULL;
	PyArray_FLAGS(ret) &= ~OWN_DATA;
	PyArray_BASE(ret) = (PyObject *)self;
	Py_INCREF(self);
	return (PyObject *)ret;
}


/*MULTIARRAY_API
 Mean
*/
static PyObject *
PyArray_Mean(PyArrayObject *self, int axis, int rtype)
{
	PyObject *obj1=NULL, *obj2=NULL;
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	obj1 = PyArray_GenericReduceFunction((PyAO *)new, n_ops.add, axis,
					     rtype);
	obj2 = PyFloat_FromDouble((double) PyArray_DIM(new,axis));
        Py_DECREF(new);
	if (obj1 == NULL || obj2 == NULL) {
		Py_XDECREF(obj1);
		Py_XDECREF(obj2);
		return NULL;
	}

	ret = PyNumber_Divide(obj1, obj2);
	Py_DECREF(obj1);
	Py_DECREF(obj2);
	return ret;
}

/* Set variance to 1 to by-pass square-root calculation and return variance */
/*MULTIARRAY_API
 Std
*/
static PyObject *
PyArray_Std(PyArrayObject *self, int axis, int rtype, int variance)
{
	PyObject *obj1=NULL, *obj2=NULL, *new=NULL;
	PyObject *ret=NULL, *newshape=NULL;
	int i, n;
	intp val;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;
	
	/* Compute and reshape mean */
	obj1 = PyArray_EnsureArray(PyArray_Mean((PyAO *)new, axis, rtype));
	if (obj1 == NULL) {Py_DECREF(new); return NULL;} 
	n = PyArray_NDIM(new);
	newshape = PyTuple_New(n);
	if (newshape == NULL) {Py_DECREF(obj1); Py_DECREF(new); return NULL;}
	for (i=0; i<n; i++) {
		if (i==axis) val = 1;
		else val = PyArray_DIM(new,i);
		PyTuple_SET_ITEM(newshape, i, PyInt_FromLong((long)val));
	}
	obj2 = PyArray_Reshape((PyAO *)obj1, newshape);
	Py_DECREF(obj1);
	Py_DECREF(newshape);
	if (obj2 == NULL) {Py_DECREF(new); return NULL;}

	/* Compute x = x - mx */
	obj1 = PyNumber_Subtract((PyObject *)new, obj2);
	Py_DECREF(obj2);
	if (obj1 == NULL) {Py_DECREF(new); return NULL;}

	/* Compute x * x */
	obj2 = PyArray_EnsureArray(PyNumber_Multiply(obj1, obj1));
	Py_DECREF(obj1);
	if (obj2 == NULL) {Py_DECREF(new); return NULL;}

	/* Compute add.reduce(x*x,axis) */
	obj1 = PyArray_GenericReduceFunction((PyAO *)obj2, n_ops.add,
					     axis, rtype);
	Py_DECREF(obj2);
	if (obj1 == NULL) {Py_DECREF(new); return NULL;}

	n = PyArray_DIM(new,axis)-1;
	Py_DECREF(new);
	if (n<=0) n=1;
	obj2 = PyFloat_FromDouble(1.0/((double )n));
	if (obj2 == NULL) {Py_DECREF(obj1); return NULL;}
	ret = PyNumber_Multiply(obj1, obj2);
	Py_DECREF(obj1);
	Py_DECREF(obj2);

        if (variance) return ret;
	
	ret = PyArray_EnsureArray(ret);

	/* sqrt() */
	obj1 = PyArray_GenericUnaryFunction((PyAO *)ret, n_ops.sqrt);
	Py_DECREF(ret);

	return obj1;
}


/*MULTIARRAY_API
 Sum
*/
static PyObject *
PyArray_Sum(PyArrayObject *self, int axis, int rtype)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericReduceFunction((PyAO *)new, n_ops.add, axis, 
					    rtype);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 Prod
*/
static PyObject *
PyArray_Prod(PyArrayObject *self, int axis, int rtype)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericReduceFunction((PyAO *)new, n_ops.multiply, axis,
					    rtype);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 CumSum
*/
static PyObject *
PyArray_CumSum(PyArrayObject *self, int axis, int rtype)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericAccumulateFunction((PyAO *)new, n_ops.add, axis,
						rtype);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 CumProd
*/
static PyObject *
PyArray_CumProd(PyArrayObject *self, int axis, int rtype)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericAccumulateFunction((PyAO *)new, 
						n_ops.multiply, axis,
						rtype);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 Any
*/
static PyObject *
PyArray_Any(PyArrayObject *self, int axis)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericReduceFunction((PyAO *)new, 
					    n_ops.logical_or, axis, 
					    PyArray_BOOL);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 All
*/
static PyObject *
PyArray_All(PyArrayObject *self, int axis)
{
	PyObject *new, *ret;

	if ((new = _check_axis(self, &axis, 0))==NULL) return NULL;

	ret = PyArray_GenericReduceFunction((PyAO *)new, 
					    n_ops.logical_and, axis, 
					    PyArray_BOOL);
	Py_DECREF(new);
	return ret;
}


/*MULTIARRAY_API
 Compress
*/
static PyObject *
PyArray_Compress(PyArrayObject *self, PyObject *condition, int axis)
{
        PyArrayObject *cond;
	PyObject *res, *ret;

        cond = (PyAO *)PyArray_FromAny(condition, NULL, 0, 0, 0);
        if (cond == NULL) return NULL;
        
        if (cond->nd != 1) {
                Py_DECREF(cond);
                PyErr_SetString(PyExc_ValueError, 
				"condition must be 1-d array");
                return NULL;
        }

        res = PyArray_Nonzero(cond);
        Py_DECREF(cond);
	ret = PyArray_Take(self, res, axis);
	Py_DECREF(res);
	return ret;
}

/*MULTIARRAY_API
 Nonzero
*/
static PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
        int n=self->nd, j;
	intp count=0, i, size;
	PyArrayIterObject *it=NULL;
	PyObject *ret=NULL, *item;
	intp *dptr[MAX_DIMS];

	it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
	if (it==NULL) return NULL;

	size = it->size;
	for (i=0; i<size; i++) {
		if (self->descr->f->nonzero(it->dataptr, self)) count++;
		PyArray_ITER_NEXT(it);
	}

	PyArray_ITER_RESET(it);
	if (n==1) {
		ret = PyArray_New(self->ob_type, 1, &count, PyArray_INTP, 
				  NULL, NULL, 0, 0, (PyObject *)self);
		if (ret == NULL) goto fail;
		dptr[0] = (intp *)PyArray_DATA(ret);
		
		for (i=0; i<size; i++) {
			if (self->descr->f->nonzero(it->dataptr, self)) 
				*(dptr[0])++ = i;
			PyArray_ITER_NEXT(it);
		}		
	}
	else {
		ret = PyTuple_New(n);
		for (j=0; j<n; j++) {
			item = PyArray_New(self->ob_type, 1, &count, 
					   PyArray_INTP, NULL, NULL, 0, 0,
					   (PyObject *)self);
			if (item == NULL) goto fail;
			PyTuple_SET_ITEM(ret, j, item);
			dptr[j] = (intp *)PyArray_DATA(item);
		}
		
		/* reset contiguous so that coordinates gets updated */
		it->contiguous = 0;
		for (i=0; i<size; i++) {
			if (self->descr->f->nonzero(it->dataptr, self)) 
				for (j=0; j<n; j++) 
					*(dptr[j])++ = it->coordinates[j];
			PyArray_ITER_NEXT(it);
		}
	}

	Py_DECREF(it);
	return ret;

 fail:
	Py_XDECREF(ret);
	Py_XDECREF(it);
	return NULL;
        
}

/*MULTIARRAY_API
 Clip
*/
static PyObject *
PyArray_Clip(PyArrayObject *self, PyObject *min, PyObject *max)
{
	PyObject *selector=NULL, *newtup=NULL, *ret=NULL;
	PyObject *res1=NULL, *res2=NULL, *res3=NULL;
	PyObject *two;

	two = PyInt_FromLong((long)2);
	res1 = PyArray_GenericBinaryFunction(self, max, n_ops.greater);
	res2 = PyArray_GenericBinaryFunction(self, min, n_ops.less);
	if ((res1 == NULL) || (res2 == NULL)) {
		Py_DECREF(two);
		Py_XDECREF(res1);
		Py_XDECREF(res2);
	}
	res3 = PyNumber_Multiply(two, res1);
	Py_DECREF(two); 
	Py_DECREF(res1); 
	if (res3 == NULL) return NULL;

	selector = PyArray_EnsureArray(PyNumber_Add(res2, res3));
	Py_DECREF(res2);
	Py_DECREF(res3);
	if (selector == NULL) return NULL;

	newtup = Py_BuildValue("(OOO)", (PyObject *)self, min, max);
	if (newtup == NULL) {Py_DECREF(selector); return NULL;}
	ret = PyArray_Choose((PyAO *)selector, newtup);
	Py_DECREF(selector);
	Py_DECREF(newtup);
	return ret;
}

/*MULTIARRAY_API
 Conjugate
*/
static PyObject *
PyArray_Conjugate(PyArrayObject *self)
{
	if (PyArray_ISCOMPLEX(self)) {
		PyObject *new;
		intp size, i;
		/* Make a copy */
		new = PyArray_NewCopy(self, -1);
		if (new==NULL) return NULL;
		size = PyArray_SIZE(new);
		if (self->descr->type_num == PyArray_CFLOAT) {
			cfloat *dptr = (cfloat *) PyArray_DATA(new);
			for (i=0; i<size; i++) {
				dptr->imag = -dptr->imag;
				dptr++;
			}
		}
		else if (self->descr->type_num == PyArray_CDOUBLE) {
			cdouble *dptr = (cdouble *)PyArray_DATA(new);
			for (i=0; i<size; i++) {
				dptr->imag = -dptr->imag;
				dptr++;
			}
		}
		else if (self->descr->type_num == PyArray_CLONGDOUBLE) {
			clongdouble *dptr = (clongdouble *)PyArray_DATA(new);
			for (i=0; i<size; i++) {
				dptr->imag = -dptr->imag;
				dptr++;
			}			
		}		
		return new;
	}
	else {
		Py_INCREF(self);
		return (PyObject *) self;
	}
}

/*MULTIARRAY_API
 Trace
*/
static PyObject *
PyArray_Trace(PyArrayObject *self, int offset, int axis1, int axis2, 
int rtype)
{
	PyObject *diag=NULL, *ret=NULL;

	diag = PyArray_Diagonal(self, offset, axis1, axis2);
	if (diag == NULL) return NULL;
	ret = PyArray_GenericReduceFunction((PyAO *)diag, n_ops.add, -1, rtype);
	Py_DECREF(diag);
	return ret;
}

/*MULTIARRAY_API
 Diagonal
*/
static PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
	int n = self->nd;
	PyObject *new;
	PyArray_Dims newaxes;
	intp dims[MAX_DIMS];
	int i, pos;	

	newaxes.ptr = dims;
	if (n < 2) {
		PyErr_SetString(PyExc_ValueError, 
				"array.ndim must be >= 2");
		return NULL;
	}
	if (axis1 < 0) axis1 += n;
	if (axis2 < 0) axis2 += n;
	if ((axis1 == axis2) || (axis1 < 0) || (axis1 >= n) ||	\
	    (axis2 < 0) || (axis2 >= n)) {
		PyErr_Format(PyExc_ValueError, "axis1(=%d) and axis2(=%d) "\
			     "must be different and within range (nd=%d)",
			     axis1, axis2, n);
		return NULL;
	}
      
	newaxes.len = n;
	/* insert at the end */
	newaxes.ptr[n-2] = axis1;
	newaxes.ptr[n-1] = axis2;
	pos = 0;
	for (i=0; i<n; i++) {
		if ((i==axis1) || (i==axis2)) continue;
		newaxes.ptr[pos++] = i;
	}
	new = PyArray_Transpose(self, &newaxes);
	if (new == NULL) return NULL;
	self = (PyAO *)new;
	
	if (n == 2) {
		PyObject *a=NULL, *indices=NULL, *ret=NULL;
		intp n1, n2, start, stop, step, count;
		intp *dptr;
		n1 = self->dimensions[0];
		n2 = self->dimensions[1];
		step = n2+1;
		if (offset < 0) {
			start = -n2 * offset;
			stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
		}
		else {
			start = offset;
			stop = MIN(n1, n2-offset)*(n2+1) + offset;
		}
		
		/* count = ceil((stop-start)/step) */
		count = ((stop-start) / step) + (((stop-start) % step) != 0);
			
		indices = PyArray_New(&PyArray_Type, 1, &count, 
				      PyArray_INTP, NULL, NULL, 0, 0, NULL);
		if (indices == NULL) {
			Py_DECREF(self); return NULL;
		}
		dptr = (intp *)PyArray_DATA(indices);
		for (n1=start; n1<stop; n1+=step) *dptr++ = n1;
		a = PyArray_IterNew((PyObject *)self);
		Py_DECREF(self);
		if (a == NULL) {Py_DECREF(indices); return NULL;}
		ret = PyObject_GetItem(a, indices);
		Py_DECREF(a);
		Py_DECREF(indices);
		return ret;
	}	

	else {
		/* 
		   my_diagonal = []
		   for i in range (s [0]) :
		       my_diagonal.append (diagonal (a [i], offset))
		   return array (my_diagonal)	
		*/	
		PyObject *mydiagonal=NULL, *new=NULL, *ret=NULL, *sel=NULL;
		intp i, n1;
		int res;
		PyArray_Descr *typecode;

		typecode = self->descr;

		mydiagonal = PyList_New(0);
		if (mydiagonal == NULL) {Py_DECREF(self); return NULL;}
		n1 = self->dimensions[0];
		for (i=0; i<n1; i++) {
			new = PyInt_FromLong((long) i);
			sel = PyArray_EnsureArray(PyObject_GetItem((PyObject *)self, new));
			Py_DECREF(new);
			if (sel == NULL) {
				Py_DECREF(self);
				Py_DECREF(mydiagonal);
				return NULL;
			}
			new = PyArray_Diagonal((PyAO *)sel, offset, n-3, n-2);
			Py_DECREF(sel);
			if (new == NULL) {
				Py_DECREF(self);
				Py_DECREF(mydiagonal);
				return NULL;
			}
			res = PyList_Append(mydiagonal, new);
			Py_DECREF(new);
			if (res < 0) {
				Py_DECREF(self);
				Py_DECREF(mydiagonal);
				return NULL;
			}
		}
		Py_DECREF(self);
		Py_INCREF(typecode);
		ret =  PyArray_FromAny(mydiagonal, typecode, 0, 0, 0);
		Py_DECREF(mydiagonal);
		return ret;
	}
}

/* simulates a C-style 1-3 dimensional array which can be accesed using 
    ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation 
    for 2-d and 3-d.

    For 2-d and up, ptr is NOT equivalent to a statically defined
    2-d or 3-d array.  In particular, it cannot be passed into a 
    function that requires a true pointer to a fixed-size array. 
*/

/* steals a reference to typedescr -- can be NULL*/
/*MULTIARRAY_API
 Simulat a C-array
*/
static int
PyArray_AsCArray(PyObject **op, void *ptr, intp *dims, int nd, 
		 PyArray_Descr* typedescr)
{
	PyArrayObject *ap;
	intp n, m, i, j;
	char **ptr2;
	char ***ptr3;

	if ((nd < 1) || (nd > 3)) {
		PyErr_SetString(PyExc_ValueError,
				"C arrays of only 1-3 dimensions available");
		Py_XDECREF(typedescr);
		return -1;
	}
	if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
						  CARRAY_FLAGS)) == NULL)
		return -1;
	switch(nd) {
	case 1:
		*((char **)ptr) = ap->data;
		break;
	case 2:
		n = ap->dimensions[0];
		ptr2 = (char **)malloc(n * sizeof(char *));
		if (!ptr2) goto fail;
		for (i=0; i<n; i++) {
			ptr2[i] = ap->data + i*ap->strides[0];
		}
		*((char ***)ptr) = ptr2;
		break;		
	case 3:
		n = ap->dimensions[0];
		m = ap->dimensions[1];
		ptr3 = (char ***)malloc(n*(m+1) * sizeof(char *));
		if (!ptr3) goto fail;
		for (i=0; i<n; i++) {
			ptr3[i] = ptr3[n + (m-1)*i];
			for (j=0; j<m; j++) {
				ptr3[i][j] = ap->data + i*ap->strides[0] + \
					j*ap->strides[1];
			}
		}
		*((char ****)ptr) = ptr3;
	}
	memcpy(dims, ap->dimensions, nd*sizeof(intp));
	*op = (PyObject *)ap;
	return 0;

 fail:
	PyErr_SetString(PyExc_MemoryError, "no memory");
	return -1;
}

/* Deprecated --- Use PyArray_AsCArray instead */

/*MULTIARRAY_API
 Convert to a 1D C-array
*/
static int 
PyArray_As1D(PyObject **op, char **ptr, int *d1, int typecode) 
{
	intp newd1;
	PyArray_Descr *descr;

	descr = PyArray_DescrFromType(typecode);	
	if (PyArray_AsCArray(op, (void *)ptr, &newd1, 1, descr) == -1)
		return -1;	
	*d1 = (int) newd1;
	return 0;
}

/*MULTIARRAY_API
 Convert to a 2D C-array
*/
static int 
PyArray_As2D(PyObject **op, char ***ptr, int *d1, int *d2, int typecode) 
{
	intp newdims[2];
	PyArray_Descr *descr;

	descr = PyArray_DescrFromType(typecode);	
	if (PyArray_AsCArray(op, (void *)ptr, newdims, 2, descr) == -1)
		return -1;

	*d1 = (int ) newdims[0];
	*d2 = (int ) newdims[1];
        return 0;
}

/* End Deprecated */

/*MULTIARRAY_API
 Free pointers created if As2D is called
*/
static int 
PyArray_Free(PyObject *op, void *ptr) 
{
        PyArrayObject *ap = (PyArrayObject *)op;
	
        if ((ap->nd < 1) || (ap->nd > 3)) 
		return -1;
        if (ap->nd >= 2) {
		free(ptr);
        }
        Py_DECREF(ap);
        return 0;
}


static PyObject *
_swap_and_concat(PyObject *op, int axis, int n)
{
	PyObject *newtup=NULL;
	PyObject *otmp, *arr;
	int i;

	newtup = PyTuple_New(n);
	if (newtup==NULL) return NULL;
	for (i=0; i<n; i++) {
		otmp = PySequence_GetItem(op, i);
		arr = PyArray_FROM_O(otmp);
		Py_DECREF(otmp);
		if (arr==NULL) goto fail;
		otmp = PyArray_SwapAxes((PyArrayObject *)arr, axis, 0);
		Py_DECREF(arr);
		if (otmp == NULL) goto fail;
		PyTuple_SET_ITEM(newtup, i, otmp);
	}
	otmp = PyArray_Concatenate(newtup, 0);
	Py_DECREF(newtup);
	if (otmp == NULL) return NULL;
	arr = PyArray_SwapAxes((PyArrayObject *)otmp, axis, 0);
	Py_DECREF(otmp);
	return arr;
	
 fail:
	Py_DECREF(newtup);
	return NULL;
}

/*op is a python object supporting the sequence interface.
  Its elements will be concatenated together to form a single 
  multidimensional array.*/
/* If axis is MAX_DIMS or bigger, then each sequence object will 
   be flattened before concatenation 
*/
/*MULTIARRAY_API
 Concatenate an arbitrary Python sequence into an array.
*/
static PyObject *
PyArray_Concatenate(PyObject *op, int axis) 
{
	PyArrayObject *ret, **mps;
	PyObject *otmp;
	int i, n, tmp, nd=0, new_dim;
	char *data;	
	PyTypeObject *subtype;
	double prior1, prior2;
	intp numbytes;

	n = PySequence_Length(op);
	if (n == -1) {
		return NULL;
	}
	if (n == 0) {
		PyErr_SetString(PyExc_ValueError, 
				"concatenation of zero-length sequences is "\
				"impossible");
		return NULL;
	}

	if ((axis < 0) || ((0 < axis) && (axis < MAX_DIMS)))
		return _swap_and_concat(op, axis, n);
	
	mps = PyArray_ConvertToCommonType(op, &n);
	if (mps == NULL) return NULL;
	
	/* Make sure these arrays are legal to concatenate. */
	/* Must have same dimensions except d0 */
	
	prior1 = 0.0;
	subtype = &PyArray_Type;
	ret = NULL;
	for(i=0; i<n; i++) {
		if (axis >= MAX_DIMS) {
			otmp = PyArray_Ravel(mps[i],0);
			Py_DECREF(mps[i]);
			mps[i] = (PyArrayObject *)otmp;
		}
		prior2 = PyArray_GetPriority((PyObject *)(mps[i]), 0.0);
		if (prior2 > prior1) {
			prior1 = prior2;
			subtype = mps[i]->ob_type;
			ret = mps[i];
		}
	}
	
	new_dim = 0;
	for(i=0; i<n; i++) {
		if (mps[i] == NULL) goto fail;
		if (i == 0) nd = mps[i]->nd;
		else {
			if (nd != mps[i]->nd) {
				PyErr_SetString(PyExc_ValueError, 
						"arrays must have same "\
						"number of dimensions");
				goto fail;
			}
			if (!PyArray_CompareLists(mps[0]->dimensions+1, 
						  mps[i]->dimensions+1, 
						  nd-1)) {
				PyErr_SetString(PyExc_ValueError, 
						"array dimensions must "\
						"agree except for d_0");
				goto fail;
			}
		}
		if (nd == 0) {
			PyErr_SetString(PyExc_ValueError, 
					"0-d arrays can't be concatenated");
			goto fail;
		}
		new_dim += mps[i]->dimensions[0];
	}
	
	tmp = mps[0]->dimensions[0];
	mps[0]->dimensions[0] = new_dim;
	Py_INCREF(mps[0]->descr);
	ret = (PyArrayObject *)PyArray_NewFromDescr(subtype, 
						    mps[0]->descr, nd,
						    mps[0]->dimensions, 
						    NULL, NULL, 0,
						    (PyObject *)ret);
	mps[0]->dimensions[0] = tmp;
	
	if (ret == NULL) goto fail;
	
	data = ret->data;
	for(i=0; i<n; i++) {
		numbytes = PyArray_NBYTES(mps[i]);
		memcpy(data, mps[i]->data, numbytes);
		data += numbytes;
	}
	
	PyArray_INCREF(ret);
	for(i=0; i<n; i++) Py_XDECREF(mps[i]);
	PyDataMem_FREE(mps);
	return (PyObject *)ret;
	
 fail:
	Py_XDECREF(ret);
	for(i=0; i<n; i++) Py_XDECREF(mps[i]);
	PyDataMem_FREE(mps);
	return NULL;
}

/*MULTIARRAY_API
 SwapAxes
*/
static PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
	PyArray_Dims new_axes;
	intp dims[MAX_DIMS];
	int n, i, val;
	PyObject *ret;
		
	if (a1 == a2) {
		Py_INCREF(ap);
		return (PyObject *)ap;
	}
	
	n = ap->nd;
	if (n <= 1) {
		Py_INCREF(ap);
		return (PyObject *)ap;
	}

	if (a1 < 0) a1 += n;
	if (a2 < 0) a2 += n;
	if ((a1 < 0) || (a1 >= n)) {
		PyErr_SetString(PyExc_ValueError, 
				"bad axis1 argument to swapaxes");
		return NULL;
	}
	if ((a2 < 0) || (a2 >= n)) {
		PyErr_SetString(PyExc_ValueError, 
				"bad axis2 argument to swapaxes");
		return NULL;
	}
	new_axes.ptr = dims;
	new_axes.len = n;

	for (i=0; i<n; i++) {
		if (i == a1) val = a2;
		else if (i == a2) val = a1;
		else val = i;
		new_axes.ptr[i] = val;
	}
	ret = PyArray_Transpose(ap, &new_axes);
	return ret;
}

/*MULTIARRAY_API
 Return Transpose.
*/
static PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
	intp *axes, axis;
	intp i, n;
	intp permutation[MAX_DIMS];
	PyArrayObject *ret = NULL;

	if (permute == NULL) {
		n = ap->nd;
		for(i=0; i<n; i++)
			permutation[i] = n-1-i;
	} else {
		n = permute->len;
		axes = permute->ptr;
		if (n > ap->nd) {
			PyErr_SetString(PyExc_ValueError, 
					"too many axes for this array");
			return NULL;
		}
		for(i=0; i<n; i++) {
			axis = axes[i];
			if (axis < 0) axis = ap->nd+axis;
			if (axis < 0 || axis >= ap->nd) {
				PyErr_SetString(PyExc_ValueError, 
						"invalid axis for this array");
				return NULL;
			}
			permutation[i] = axis;
		}
	}
	
	/* this allocates memory for dimensions and strides (but fills them
	   incorrectly), sets up descr, and points data at ap->data. */
	Py_INCREF(ap->descr);
	ret = (PyArrayObject *)\
		PyArray_NewFromDescr(ap->ob_type, 
				     ap->descr, 
				     n, permutation, 
				     NULL, ap->data, ap->flags,
				     (PyObject *)ap);
	if (ret == NULL) return NULL;
	
	/* point at true owner of memory: */
	ret->base = (PyObject *)ap;
	Py_INCREF(ap);
	
	for(i=0; i<n; i++) {
		ret->dimensions[i] = ap->dimensions[permutation[i]];
		ret->strides[i] = ap->strides[permutation[i]];
	}
	PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);	

	return (PyObject *)ret;	
}

/*MULTIARRAY_API
 Repeat the array.
*/
static PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
	intp *counts;
	intp n, n_outer, i, j, k, chunk, total;
	intp tmp;
	int nd;
	PyArrayObject *repeats=NULL;
	PyObject *ap=NULL;
	PyArrayObject *ret=NULL;
	char *new_data, *old_data;

	repeats = (PyAO *)PyArray_ContiguousFromAny(op, PyArray_INTP, 0, 1);
	if (repeats == NULL) return NULL;
	nd = repeats->nd;
	counts = (intp *)repeats->data;

	if ((ap=_check_axis(aop, &axis, CARRAY_FLAGS))==NULL) {
		Py_DECREF(repeats);
		return NULL;
	}

	aop = (PyAO *)ap;

	if (nd == 1)
		n = repeats->dimensions[0];
	else /* nd == 0 */
		n = aop->dimensions[axis];

	if (aop->dimensions[axis] != n) {
		PyErr_SetString(PyExc_ValueError, 
				"a.shape[axis] != len(repeats)");
		goto fail;
	}

	
	if (nd == 0) 
		total = counts[0]*n;
	else {
		
		total = 0;
		for(j=0; j<n; j++) {
			if (counts[j] < 0) {
				PyErr_SetString(PyExc_ValueError, "count < 0");
				goto fail;
			}
			total += counts[j];
		}
	}


	/* Construct new array */
	aop->dimensions[axis] = total;
	Py_INCREF(aop->descr);
	ret = (PyArrayObject *)PyArray_NewFromDescr(aop->ob_type, 
						    aop->descr,
						    aop->nd,
						    aop->dimensions,
						    NULL, NULL, 0,
						    (PyObject *)aop);
	aop->dimensions[axis] = n;
	
	if (ret == NULL) goto fail;
	
	new_data = ret->data;
	old_data = aop->data;
	
	chunk = aop->descr->elsize;
	for(i=axis+1; i<aop->nd; i++) {
		chunk *= aop->dimensions[i];
	}
	
	n_outer = 1;
	for(i=0; i<axis; i++) n_outer *= aop->dimensions[i];

	for(i=0; i<n_outer; i++) {
		for(j=0; j<n; j++) {
			tmp = (nd ? counts[j] : counts[0]);
			for(k=0; k<tmp; k++) {
				memcpy(new_data, old_data, chunk);
				new_data += chunk;
			}
			old_data += chunk;
		}
	}

	Py_DECREF(repeats);
	PyArray_INCREF(ret);
	Py_XDECREF(aop);
	return (PyObject *)ret;
	
 fail:
	Py_DECREF(repeats);
	Py_XDECREF(aop);
	Py_XDECREF(ret);
	return NULL;
}

/*OBJECT_API*/
static PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
	int i, n, allscalars=0; 
	PyArrayObject **mps=NULL;
	PyObject *otmp;
	PyArray_Descr *intype=NULL, *stype=NULL;
	PyArray_Descr *newtype=NULL;

	
	*retn = n = PySequence_Length(op);
	if (PyErr_Occurred()) {*retn = 0; return NULL;}
	
	mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
	if (mps == NULL) {
		*retn = 0;
		return (void*)PyErr_NoMemory();
	}
	
	for(i=0; i<n; i++) {
		otmp = PySequence_GetItem(op, i);
		if (!PyArray_CheckAnyScalar(otmp)) {			
			newtype = PyArray_DescrFromObject(otmp, intype);
			Py_XDECREF(intype);
			intype = newtype;
			mps[i] = NULL;
		}
		else {
			newtype = PyArray_DescrFromObject(otmp, stype);
			Py_XDECREF(stype);
			stype = newtype;
			mps[i] = (PyArrayObject *)Py_None;
			Py_INCREF(Py_None);
		}
		Py_XDECREF(otmp);
	}
	if (intype==NULL) { /* all scalars */
		allscalars = 1;
		intype = stype;
		Py_INCREF(intype);
		for (i=0; i<n; i++) {
			Py_XDECREF(mps[i]);
			mps[i] = NULL;
		}
	}
	/* Make sure all arrays are actual array objects. */
	for(i=0; i<n; i++) {
		int flags = CARRAY_FLAGS;
		if ((otmp = PySequence_GetItem(op, i)) == NULL) 
			goto fail;
		if (!allscalars && ((PyObject *)(mps[i]) == Py_None)) {
			/* forcecast scalars */
			flags |= FORCECAST;
			Py_DECREF(Py_None);
		}
		mps[i] = (PyArrayObject*)
                        PyArray_FromAny(otmp, intype, 0, 0, flags);
		Py_DECREF(otmp);
		Py_XDECREF(stype);
	}	
	return mps;

 fail:
	Py_XDECREF(intype); 
	Py_XDECREF(stype);
	*retn = 0;
	for (i=0; i<n; i++) Py_XDECREF(mps[i]);
	PyDataMem_FREE(mps);
	return NULL;
}


/*MULTIARRAY_API
 Numeric.choose()
*/
static PyObject *
PyArray_Choose(PyArrayObject *ip, PyObject *op)
{
	intp *sizes, offset;
	int i,n,m,elsize;
	char *ret_data;
	PyArrayObject **mps, *ap, *ret;
	intp *self_data, mi;
	ap = NULL;
	ret = NULL;
	       	
	/* Convert all inputs to arrays of a common type */
	mps = PyArray_ConvertToCommonType(op, &n);
	if (mps == NULL) return NULL;

	sizes = (intp *)malloc(n*sizeof(intp));
	if (sizes == NULL) goto fail;
	
	ap = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)ip, 
							   PyArray_INTP, 
							   0, 0);
	if (ap == NULL) goto fail;
	
	/* Check the dimensions of the arrays */
	for(i=0; i<n; i++) {
		if (mps[i] == NULL) goto fail;
		if (ap->nd < mps[i]->nd) {
			PyErr_SetString(PyExc_ValueError, 
					"too many dimensions");
			goto fail;
		}
		if (!PyArray_CompareLists(ap->dimensions+(ap->nd-mps[i]->nd),
				   mps[i]->dimensions, mps[i]->nd)) {
			PyErr_SetString(PyExc_ValueError, 
					"array dimensions must agree");
			goto fail;
		}
		sizes[i] = PyArray_NBYTES(mps[i]);
	}
	
	Py_INCREF(mps[0]->descr);
	ret = (PyArrayObject *)PyArray_NewFromDescr(ap->ob_type, 
						    mps[0]->descr,
						    ap->nd,
						    ap->dimensions, 
						    NULL, NULL, 0,
						    (PyObject *)ap);
	if (ret == NULL) goto fail;
	
	elsize = ret->descr->elsize;
	m = PyArray_SIZE(ret);
	self_data = (intp *)ap->data;
	ret_data = ret->data;
	
	for (i=0; i<m; i++) {
		mi = *self_data;
		if (mi < 0 || mi >= n) {
			PyErr_SetString(PyExc_ValueError, 
					"invalid entry in choice array");
			goto fail;
		}
		offset = i*elsize;
		if (offset >= sizes[mi]) {offset = offset % sizes[mi]; }
		memmove(ret_data, mps[mi]->data+offset, elsize);
		ret_data += elsize; self_data++;
	}
	
	PyArray_INCREF(ret);
	for(i=0; i<n; i++) Py_XDECREF(mps[i]);
	Py_DECREF(ap);
	PyDataMem_FREE(mps);
	free(sizes);

	return (PyObject *)ret;
	
 fail:
	for(i=0; i<n; i++) Py_XDECREF(mps[i]);
	Py_XDECREF(ap);
	PyDataMem_FREE(mps);
	free(sizes);
	Py_XDECREF(ret);
	return NULL;
}


/* Be sure to save this global_compare when necessary */

static PyArrayObject *global_obj;

static int 
qsortCompare (const void *a, const void *b) 
{
	return global_obj->descr->f->compare(a,b,global_obj);
}

#define SWAPAXES(op, ap) {						\
		orign = (ap)->nd-1;					\
		if (axis != orign) {					\
			(op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
			Py_DECREF((ap));				\
			if ((op) == NULL) return NULL;			\
		}							\
		else (op) = (ap);					\
	}

#define SWAPBACK(op, ap) { \
		if (axis != orign) { \
			(op) = (PyAO *)PyArray_SwapAxes((ap), axis, orign); \
			Py_DECREF((ap));				\
			if ((op) == NULL) return NULL;			\
		}							\
		else (op) = (ap);					\
	}

/*MULTIARRAY_API
 Sort an array
*/
static PyObject *
PyArray_Sort(PyArrayObject *op, int axis) 
{
	PyArrayObject *ap=NULL, *store_arr=NULL;
	char *ip;
	int i, n, m, elsize, orign;

	if ((ap = (PyAO*) _check_axis(op, &axis, 0))==NULL) return NULL;

	SWAPAXES(op, ap);

	ap = (PyArrayObject *)PyArray_CopyFromObject((PyObject *)op, 
						     PyArray_NOTYPE,
						     1, 0);
	Py_DECREF(op);

	if (ap == NULL) return NULL;

	if (ap->descr->f->compare == NULL) {
		PyErr_SetString(PyExc_TypeError, 
				"compare not supported for type");
		Py_DECREF(ap);
		return NULL;
	}
	
	elsize = ap->descr->elsize;
	m = ap->dimensions[ap->nd-1];
	if (m == 0) goto finish;

	n = PyArray_SIZE(ap)/m;

	/* Store global -- allows re-entry -- restore before leaving*/
	store_arr = global_obj; 
	global_obj = ap;
	
	for (ip=ap->data, i=0; i<n; i++, ip+=elsize*m) {
		qsort(ip, m, elsize, qsortCompare);
	}
	
	global_obj = store_arr;
		
	if (PyErr_Occurred()) {
		Py_DECREF(ap);
		return NULL;	
	}

 finish:
	SWAPBACK(op, ap);

	return (PyObject *)op;
}


static char *global_data;

static int 
argsort_static_compare(const void *ip1, const void *ip2) 
{
	int isize = global_obj->descr->elsize;
	const intp *ipa = ip1;
	const intp *ipb = ip2;	
	return global_obj->descr->f->compare(global_data + (isize * *ipa),
                                          global_data + (isize * *ipb), 
					  global_obj);
}

/*MULTIARRAY_API
 ArgSort an array
*/
static PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis) 
{
	PyArrayObject *ap, *ret, *store;
	intp *ip;
	intp i, j, n, m, orign;
	int argsort_elsize;
	char *store_ptr;

	if ((ap = (PyAO *)_check_axis(op, &axis, 0))==NULL) return NULL;

	SWAPAXES(op, ap);

	ap = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)op, 
							   PyArray_NOTYPE,
							   1, 0);
	Py_DECREF(op);

	if (ap == NULL) return NULL;
	
	ret = (PyArrayObject *)PyArray_New(ap->ob_type, ap->nd,
					   ap->dimensions, PyArray_INTP,
					   NULL, NULL, 0, 0, (PyObject *)ap);
	if (ret == NULL) goto fail;
	
	if (ap->descr->f->compare == NULL) {
		PyErr_SetString(PyExc_TypeError, 
				"compare not supported for type");
		goto fail;
	}
	
	ip = (intp *)ret->data;
	argsort_elsize = ap->descr->elsize;
	m = ap->dimensions[ap->nd-1];
	if (m == 0) goto finish;

	n = PyArray_SIZE(ap)/m;
	store_ptr = global_data;
	global_data = ap->data;
	store = global_obj;
	global_obj = ap;
	for (i=0; i<n; i++, ip+=m, global_data += m*argsort_elsize) {
		for(j=0; j<m; j++) ip[j] = j;
		qsort((char *)ip, m, sizeof(intp),
		      argsort_static_compare);
	}
	global_data = store_ptr;
	global_obj = store;


 finish:
	Py_DECREF(ap);
	SWAPBACK(op, ret);
	return (PyObject *)op;

 fail:
	Py_XDECREF(ap);
	Py_XDECREF(ret);
	return NULL;

}  

static void 
local_where(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject *ret)
{	
	PyArray_CompareFunc *compare = ap2->descr->f->compare;
	intp  min_i, max_i, i, j;
	int location, elsize = ap1->descr->elsize;
	intp elements = ap1->dimensions[ap1->nd-1];
	intp n = PyArray_SIZE(ap2);
	intp *rp = (intp *)ret->data;
	char *ip = ap2->data;
	char *vp = ap1->data;

	for (j=0; j<n; j++, ip+=elsize, rp++) {
		min_i = 0;
		max_i = elements;
		while (min_i != max_i) {
			i = (max_i-min_i)/2 + min_i;
			location = compare(ip, vp+elsize*i, ap2);
			if (location == 0) {
				while (i > 0) {
					if (compare(ip, vp+elsize*(--i), ap2) \
					    != 0) {
						i = i+1; break;
					}
				}
				min_i = i;
				break;
			}
			else if (location < 0) {
				max_i = i;
			} else {
				min_i = i+1;
			}
		}
		*rp = min_i;
	}
}

/*MULTIARRAY_API
 Numeric.searchsorted(a,v)
*/
static PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2) 
{
	PyArrayObject *ap1=NULL, *ap2=NULL, *ret=NULL;
	int typenum = 0;

	/* 
        PyObject *args;
        args = Py_BuildValue("O",op2);
	Py_DELEGATE_ARGS(((PyObject *)op1), searchsorted, args);
        Py_XDECREF(args);
	*/

	typenum = PyArray_ObjectType((PyObject *)op1, 0);
	typenum = PyArray_ObjectType(op2, typenum);
	ret = NULL;
	ap1 = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject *)op1, 
							    typenum, 
							    1, 1);
	if (ap1 == NULL) return NULL;
	ap2 = (PyArrayObject *)PyArray_ContiguousFromAny(op2, typenum, 
							    0, 0);
	if (ap2 == NULL) goto fail;
	
	ret = (PyArrayObject *)PyArray_New(ap2->ob_type, ap2->nd, 
					   ap2->dimensions, PyArray_INTP,
					   NULL, NULL, 0, 0, (PyObject *)ap2);
	if (ret == NULL) goto fail;

	if (ap2->descr->f->compare == NULL) {
		PyErr_SetString(PyExc_TypeError, 
				"compare not supported for type");
		goto fail;
	}
	
	local_where(ap1, ap2, ret);   
	
	Py_DECREF(ap1);
	Py_DECREF(ap2);
	return (PyObject *)ret;
	
 fail:
	Py_XDECREF(ap1);
	Py_XDECREF(ap2);
	Py_XDECREF(ret);
	return NULL;
}



/* Could perhaps be redone to not make contiguous arrays 
 */

/*MULTIARRAY_API
 Numeric.innerproduct(a,v)
*/
static PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2) 
{
	PyArrayObject *ap1, *ap2, *ret=NULL;
	intp i, j, l, i1, i2, n1, n2;
	int typenum;
	intp is1, is2, os;
	char *ip1, *ip2, *op;
	intp dimensions[MAX_DIMS], nd;
	PyArray_DotFunc *dot;
	PyTypeObject *subtype;
        double prior1, prior2;
	
	typenum = PyArray_ObjectType(op1, 0);  
	typenum = PyArray_ObjectType(op2, typenum);
		
	ap1 = (PyArrayObject *)PyArray_ContiguousFromAny(op1, typenum, 
							    0, 0);
	if (ap1 == NULL) return NULL;
	ap2 = (PyArrayObject *)PyArray_ContiguousFromAny(op2, typenum, 
							    0, 0);
	if (ap2 == NULL) goto fail;
	
	if (ap1->nd == 0 || ap2->nd == 0) {
		ret = (ap1->nd == 0 ? ap1 : ap2);
		ret = (PyArrayObject *)ret->ob_type->tp_as_number->\
			nb_multiply((PyObject *)ap1, (PyObject *)ap2);
		Py_DECREF(ap1);
		Py_DECREF(ap2);
		return (PyObject *)ret;
	}
	
	l = ap1->dimensions[ap1->nd-1];
	
	if (ap2->dimensions[ap2->nd-1] != l) {
		PyErr_SetString(PyExc_ValueError, "matrices are not aligned");
		goto fail;
	}
	
	if (l == 0) n1 = n2 = 0;
	else {
		n1 = PyArray_SIZE(ap1)/l;
		n2 = PyArray_SIZE(ap2)/l;
	}

	nd = ap1->nd+ap2->nd-2;
	j = 0;
	for(i=0; i<ap1->nd-1; i++) {
		dimensions[j++] = ap1->dimensions[i];
	}
	for(i=0; i<ap2->nd-1; i++) {
		dimensions[j++] = ap2->dimensions[i];
	}


	/* Need to choose an output array that can hold a sum 
	    -- use priority to determine which subtype.
	 */
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);
       
	ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions, 
					   typenum, NULL, NULL, 0, 0, 
                                           (PyObject *)
					   (prior2 > prior1 ? ap2 : ap1));
	if (ret == NULL) goto fail;

	dot = (ret->descr->f->dotfunc);
	
	if (dot == NULL) {
		PyErr_SetString(PyExc_ValueError, 
				"dot not available for this type");
		goto fail;
	}

	
	is1 = ap1->strides[ap1->nd-1]; 
	is2 = ap2->strides[ap2->nd-1];
	op = ret->data; os = ret->descr->elsize;
	
	ip1 = ap1->data;
	for(i1=0; i1<n1; i1++) {
		ip2 = ap2->data;
		for(i2=0; i2<n2; i2++) {
			dot(ip1, is1, ip2, is2, op, l, ret);
			ip2 += is2*l;
			op += os;
		}
		ip1 += is1*l;
	}
	if (PyErr_Occurred()) goto fail;
		
	
	Py_DECREF(ap1);
	Py_DECREF(ap2);
	return (PyObject *)ret;
	
 fail:
	Py_XDECREF(ap1);
	Py_XDECREF(ap2);
	Py_XDECREF(ret);
	return NULL;
}


/* just like inner product but does the swapaxes stuff on the fly */
/*MULTIARRAY_API
 Numeric.matrixproduct(a,v)
*/
static PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2) 
{
	PyArrayObject *ap1, *ap2, *ret=NULL;
	intp i, j, l, i1, i2, n1, n2;
	int typenum;
	intp is1, is2, os;
	char *ip1, *ip2, *op;
	intp dimensions[MAX_DIMS], nd;
	PyArray_DotFunc *dot;
	intp matchDim, otherDim, is2r, is1r;
	PyTypeObject *subtype;
        double prior1, prior2;
	PyArray_Descr *typec;
        
	typenum = PyArray_ObjectType(op1, 0);  
	typenum = PyArray_ObjectType(op2, typenum);	
	
	typec = PyArray_DescrFromType(typenum);
	Py_INCREF(typec);
	ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0, 
					       DEFAULT_FLAGS);
	if (ap1 == NULL) {Py_DECREF(typec); return NULL;}
	ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
					       DEFAULT_FLAGS);
	if (ap2 == NULL) goto fail;
	
	if (ap1->nd == 0 || ap2->nd == 0) {
		ret = (ap1->nd == 0 ? ap1 : ap2);
		ret = (PyArrayObject *)ret->ob_type->tp_as_number->\
			nb_multiply((PyObject *)ap1, (PyObject *)ap2);
		Py_DECREF(ap1);
		Py_DECREF(ap2);
		return (PyObject *)ret;
	}
	
	l = ap1->dimensions[ap1->nd-1];
	if (ap2->nd > 1) {
		matchDim = ap2->nd - 2;
		otherDim = ap2->nd - 1;
	}
	else {
		matchDim = 0;
		otherDim = 0;
	}

	if (ap2->dimensions[matchDim] != l) {
		PyErr_SetString(PyExc_ValueError, "objects are not aligned");
		goto fail;
	}
	
	if (l == 0) n1 = n2 = 0;
	else {
		n1 = PyArray_SIZE(ap1)/l;
		n2 = PyArray_SIZE(ap2)/l;
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
	/*
	fprintf(stderr, "nd=%d dimensions=", nd);
	   for(i=0; i<j; i++) 
	   fprintf(stderr, "%d ", dimensions[i]);
	   fprintf(stderr, "\n"); 
	*/

        /* Choose which subtype to return */
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);

	ret = (PyArrayObject *)PyArray_New(subtype, nd, dimensions, 
					   typenum, NULL, NULL, 0, 0, 
                                           (PyObject *)
					   (prior2 > prior1 ? ap2 : ap1));
	if (ret == NULL) goto fail;

	dot = ret->descr->f->dotfunc;
	if (dot == NULL) {
		PyErr_SetString(PyExc_ValueError, 
				"dot not available for this type");
		goto fail;
	}
		
	is1 = ap1->strides[ap1->nd-1]; is2 = ap2->strides[matchDim];
	if(ap1->nd > 1)
		is1r = ap1->strides[ap1->nd-2];
	else
		is1r = ap1->strides[ap1->nd-1];
	is2r = ap2->strides[otherDim];

	op = ret->data; os = ret->descr->elsize;

	ip1 = ap1->data;
	for(i1=0; i1<n1; i1++) {
		ip2 = ap2->data;
		for(i2=0; i2<n2; i2++) {
			dot(ip1, is1, ip2, is2, op, l, ret);
			ip2 += is2r;
			op += os;
		}
		ip1 += is1r;
	}
	if (PyErr_Occurred()) goto fail;
	
	
	Py_DECREF(ap1);
	Py_DECREF(ap2);
	return (PyObject *)ret;
	
 fail:
	Py_XDECREF(ap1);
	Py_XDECREF(ap2);
	Py_XDECREF(ret);
	return NULL;
}

/*MULTIARRAY_API
 Fast Copy and Transpose
*/
static PyObject *
PyArray_CopyAndTranspose(PyObject *op) 
{
	PyObject *ret, *arr;
	int nd;
	intp dims[2];
	intp i,j;
	int elsize, str2;
	char *iptr;
	char *optr;

	/* make sure it is well-behaved */
	arr = PyArray_FromAny(op, NULL, 0, 0, CARRAY_FLAGS);
	nd = PyArray_NDIM(arr);
	if (nd == 1) {     /* we will give in to old behavior */
		ret = PyArray_Copy((PyArrayObject *)arr);
		Py_DECREF(arr);
		return ret;		
	}
	else if (nd != 2) {
		Py_DECREF(arr);
		PyErr_SetString(PyExc_ValueError, 
				"only 2-d arrays are allowed");
		return NULL;
	}

	/* Now construct output array */
	dims[0] = PyArray_DIM(arr,1);
	dims[1] = PyArray_DIM(arr,0);
	elsize = PyArray_ITEMSIZE(arr);
	
	Py_INCREF(PyArray_DESCR(arr));
	ret = PyArray_NewFromDescr(arr->ob_type, 
				   PyArray_DESCR(arr),
				   2, dims, 
				   NULL, NULL, 0, arr);

	if (ret == NULL) {
		Py_DECREF(arr);
		return NULL;
	}
	/* do 2-d loop */
	optr = PyArray_DATA(ret);
	str2 = elsize*dims[0];
	for (i=0; i<dims[0]; i++) {
		iptr = PyArray_DATA(arr) + i*elsize;
		for (j=0; j<dims[1]; j++) {
			/* optr[i,j] = iptr[j,i] */
			memcpy(optr, iptr, elsize);
			optr += elsize;
			iptr += str2;
		}
	}
	Py_DECREF(arr);
	return ret;
}
 
/*MULTIARRAY_API
 Numeric.correlate(a1,a2,mode)
*/
static PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode) 
{
	PyArrayObject *ap1, *ap2, *ret=NULL;
	intp length;
	intp i, n1, n2, n, n_left, n_right;
	int typenum;
	intp is1, is2, os;
	char *ip1, *ip2, *op;
	PyArray_DotFunc *dot;
	PyArray_Descr *typec;
        double prior1, prior2;
        PyTypeObject *subtype=NULL;
	
	typenum = PyArray_ObjectType(op1, 0);  
	typenum = PyArray_ObjectType(op2, typenum);
	
	typec = PyArray_DescrFromType(typenum);
	Py_INCREF(typec);
	ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
					       DEFAULT_FLAGS);
	if (ap1 == NULL) {Py_DECREF(typec); return NULL;}
	ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
					       DEFAULT_FLAGS);
	if (ap2 == NULL) goto fail;
	
	n1 = ap1->dimensions[0];
	n2 = ap2->dimensions[0];

	if (n1 < n2) { 
		ret = ap1; ap1 = ap2; ap2 = ret; 
		ret = NULL; i = n1;n1=n2;n2=i;
	}
	length = n1;
	n = n2;
	switch(mode) {
	case 0:	
		length = length-n+1;
		n_left = n_right = 0;
		break;
	case 1:
		n_left = (intp)(n/2);
		n_right = n-n_left-1;
		break;
	case 2:
		n_right = n-1;
		n_left = n-1;
		length = length+n-1;
		break;
	default:
		PyErr_SetString(PyExc_ValueError, 
				"mode must be 0, 1, or 2");
		goto fail;
	}

	/* Need to choose an output array that can hold a sum 
	    -- use priority to determine which subtype.
	 */
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? ap2->ob_type : ap1->ob_type);
	
	ret = (PyArrayObject *)PyArray_New(subtype, 1,
					   &length, typenum, 
					   NULL, NULL, 0, 0,
                                           (PyObject *)
					   (prior2 > prior1 ? ap2 : ap1));
	if (ret == NULL) goto fail;
	
	dot = ret->descr->f->dotfunc;
	if (dot == NULL) {
		PyErr_SetString(PyExc_ValueError, 
				"function not available for this data type");
		goto fail;
	}
	
	is1 = ap1->strides[0]; is2 = ap2->strides[0];
	op = ret->data; os = ret->descr->elsize;
	
	ip1 = ap1->data; ip2 = ap2->data+n_left*is2;
	n = n-n_left;
	for(i=0; i<n_left; i++) {
		dot(ip1, is1, ip2, is2, op, n, ret);
		n++;
		ip2 -= is2;
		op += os;
	}
	for(i=0; i<(n1-n2+1); i++) {
		dot(ip1, is1, ip2, is2, op, n, ret);
		ip1 += is1;
		op += os;
	}
	for(i=0; i<n_right; i++) {
		n--;
		dot(ip1, is1, ip2, is2, op, n, ret);
		ip1 += is1;
		op += os;
	}
	if (PyErr_Occurred()) goto fail;
	Py_DECREF(ap1);
	Py_DECREF(ap2);
	return (PyObject *)ret;
	
 fail:
	Py_XDECREF(ap1);
	Py_XDECREF(ap2);
	Py_XDECREF(ret);
	return NULL;
}


/*MULTIARRAY_API
 ArgMin
*/
static PyObject *
PyArray_ArgMin(PyArrayObject *ap, int axis) 
{
	PyObject *obj, *new, *ret;

	if (PyArray_ISFLEXIBLE(ap)) {
		PyErr_SetString(PyExc_TypeError, 
				"argmax is unsupported for this type");
		return NULL;
	}
	else if (PyArray_ISUNSIGNED(ap)) 
		obj = PyInt_FromLong((long) -1);
	
	else if (PyArray_TYPE(ap)==PyArray_BOOL) 
		obj = PyInt_FromLong((long) 1);
	
	else 
		obj = PyInt_FromLong((long) 0);
	
	new = PyArray_EnsureArray(PyNumber_Subtract(obj, (PyObject *)ap));
	Py_DECREF(obj);
	if (new == NULL) return NULL;
	ret = PyArray_ArgMax((PyArrayObject *)new, axis);
	Py_DECREF(new);
	return ret;
}

/*MULTIARRAY_API
 Max
*/
static PyObject *
PyArray_Max(PyArrayObject *ap, int axis)
{
	PyArrayObject *arr;
	PyObject *ret;

	if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0))==NULL)
		return NULL;
	ret = PyArray_GenericReduceFunction(arr, n_ops.maximum, axis, 
					    arr->descr->type_num);
	Py_DECREF(arr);
	return ret;	    
}

/*MULTIARRAY_API
 Min
*/
static PyObject *
PyArray_Min(PyArrayObject *ap, int axis)
{
	PyArrayObject *arr;
	PyObject *ret;

	if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0))==NULL)
		return NULL;
	ret = PyArray_GenericReduceFunction(arr, n_ops.minimum, axis,
					    arr->descr->type_num);
	Py_DECREF(arr);
	return ret;	    
}

/*MULTIARRAY_API
 Ptp
*/
static PyObject *
PyArray_Ptp(PyArrayObject *ap, int axis)
{
	PyArrayObject *arr;
	PyObject *ret;
	PyObject *obj1=NULL, *obj2=NULL;

	if ((arr=(PyArrayObject *)_check_axis(ap, &axis, 0))==NULL)
		return NULL;
	obj1 = PyArray_Max(arr, axis);
	if (obj1 == NULL) goto fail;
	obj2 = PyArray_Min(arr, axis);
	if (obj2 == NULL) goto fail;
	Py_DECREF(arr);
	ret = PyNumber_Subtract(obj1, obj2);
	Py_DECREF(obj1);
	Py_DECREF(obj2);
	return ret;

 fail:
	Py_XDECREF(arr);
	Py_XDECREF(obj1);
	Py_XDECREF(obj2);
	return NULL;
}


/*MULTIARRAY_API
 ArgMax
*/
static PyObject *
PyArray_ArgMax(PyArrayObject *op, int axis) 
{
	PyArrayObject *ap=NULL, *rp=NULL;
	PyArray_ArgFunc* arg_func;
	char *ip;
	intp *rptr;
	intp i, n, orign, m;
	int elsize;
	
	if ((ap=(PyAO *)_check_axis(op, &axis, 0))==NULL) return NULL;

	SWAPAXES(op, ap);

	ap = (PyArrayObject *)\
		PyArray_ContiguousFromAny((PyObject *)op, 
					  PyArray_NOTYPE, 1, 0);

	Py_DECREF(op);
	if (ap == NULL) return NULL;
	
	arg_func = ap->descr->f->argmax;
	if (arg_func == NULL) {
		PyErr_SetString(PyExc_TypeError, "data type not ordered");
		goto fail;
	}

	rp = (PyArrayObject *)PyArray_New(ap->ob_type, ap->nd-1,
					  ap->dimensions, PyArray_INTP,
					  NULL, NULL, 0, 0, 
                                          (PyObject *)ap);
	if (rp == NULL) goto fail;


	elsize = ap->descr->elsize;
	m = ap->dimensions[ap->nd-1];
	if (m == 0) {
		PyErr_SetString(MultiArrayError, 
				"attempt to get argmax/argmin "\
				"of an empty sequence??");
		goto fail;
	}
	n = PyArray_SIZE(ap)/m;
	rptr = (intp *)rp->data;
	for (ip = ap->data, i=0; i<n; i++, ip+=elsize*m) {
		arg_func(ip, m, rptr, ap);
		rptr += 1;
	}
	Py_DECREF(ap);

	SWAPBACK(op, rp);     /* op now contains the return */
  
	return (PyObject *)op;
	
 fail:
	Py_DECREF(ap);
	Py_XDECREF(rp);
	return NULL;
}  


/*MULTIARRAY_API
 Take
*/
static PyObject *
PyArray_Take(PyArrayObject *self0, PyObject *indices0, int axis)
{
        PyArrayObject *self, *indices, *ret;
        intp nd, i, j, n, m, max_item, tmp, chunk;
	intp shape[MAX_DIMS];
        char *src, *dest;
	
        indices = ret = NULL;
	self = (PyAO *)_check_axis(self0, &axis, CARRAY_FLAGS);
        if (self == NULL) return NULL;
	
        indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0, 
							     PyArray_INTP, 
								1, 0);
        if (indices == NULL) goto fail;
	
        n = m = chunk = 1;
        nd = self->nd + indices->nd - 1;
        for (i=0; i< nd; i++) {
                if (i < axis) {
                        shape[i] = self->dimensions[i];
                        n *= shape[i];
                } else {
                        if (i < axis+indices->nd) {
                                shape[i] = indices->dimensions[i-axis];
                                m *= shape[i];
                        } else {
                                shape[i] = self->dimensions[i-indices->nd+1];
                                chunk *= shape[i];
                        }
                }
        }
	Py_INCREF(self->descr);
        ret = (PyArrayObject *)PyArray_NewFromDescr(self->ob_type, 
						    self->descr,
						    nd, shape, 
						    NULL, NULL, 0, 
						    (PyObject *)self);
	
        if (ret == NULL) goto fail;
	
        max_item = self->dimensions[axis];
        chunk = chunk * ret->descr->elsize;
        src = self->data;
        dest = ret->data;
	
        for(i=0; i<n; i++) {
                for(j=0; j<m; j++) {
                        tmp = ((intp *)(indices->data))[j];
                        if (tmp < 0) tmp = tmp+max_item;
                        if ((tmp < 0) || (tmp >= max_item)) {
                                PyErr_SetString(PyExc_IndexError, 
						"index out of range for "\
						"array");
                                goto fail;
                        }
                        memmove(dest, src+tmp*chunk, chunk);
                        dest += chunk;
                }
                src += chunk*max_item;
        }
	
        PyArray_INCREF(ret);

        Py_XDECREF(indices);
        Py_XDECREF(self);

        return (PyObject *)ret;
	
	
 fail:
        Py_XDECREF(ret);
        Py_XDECREF(indices);
        Py_XDECREF(self);
        return NULL;
}

/*MULTIARRAY_API
 Put values into an array
*/
static PyObject *
PyArray_Put(PyArrayObject *self, PyObject* values0, PyObject *indices0) 
{
        PyArrayObject  *indices, *values;
        int i, chunk, ni, max_item, nv, tmp, thistype; 
        char *src, *dest;

        indices = NULL;
        values = NULL;

        if (!PyArray_Check(self)) {
                PyErr_SetString(PyExc_TypeError, "put: first argument must be an array");
                return NULL;
        }
        if (!PyArray_ISCONTIGUOUS(self)) {
                PyErr_SetString(PyExc_ValueError, "put: first argument must be contiguous");
                return NULL;
        }
        max_item = PyArray_SIZE(self);
        dest = self->data;
        chunk = self->descr->elsize;

        indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0, PyArray_INTP, 0, 0);
        if (indices == NULL) goto fail;
        ni = PyArray_SIZE(indices);

	thistype = self->descr->type_num;
        values = (PyArrayObject *)\
		PyArray_ContiguousFromAny(values0, thistype, 0, 0);
        if (values == NULL) goto fail;
        nv = PyArray_SIZE(values);
        if (nv > 0) { /* nv == 0 for a null array */
                if (thistype == PyArray_OBJECT) {   
                        for(i=0; i<ni; i++) {
                                src = values->data + chunk * (i % nv);
                                tmp = ((intp *)(indices->data))[i];
                                if (tmp < 0) tmp = tmp+max_item;
                                if ((tmp < 0) || (tmp >= max_item)) {
                                        PyErr_SetString(PyExc_IndexError, "index out of range for array");
                                        goto fail;
                                }
                                Py_INCREF(*((PyObject **)src));
                                Py_XDECREF(*((PyObject **)(dest+tmp*chunk)));
                                memmove(dest + tmp * chunk, src, chunk);
                        }
                }
                else {
                        for(i=0; i<ni; i++) {
                                src = values->data + chunk * (i % nv);
                                tmp = ((intp *)(indices->data))[i];
                                if (tmp < 0) tmp = tmp+max_item;
                                if ((tmp < 0) || (tmp >= max_item)) {
                                        PyErr_SetString(PyExc_IndexError, "index out of range for array");
                                        goto fail;
                                }
                                memmove(dest + tmp * chunk, src, chunk);
                        }
                }

        }

        Py_XDECREF(values);
        Py_XDECREF(indices);
        Py_INCREF(Py_None);
        return Py_None;
	
 fail:
        Py_XDECREF(indices);
        Py_XDECREF(values);
        return NULL;
}

/*MULTIARRAY_API
 Put values into an array according to a mask.
*/
static PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0) 
{
        PyArrayObject  *mask, *values;
        int i, chunk, ni, max_item, nv, tmp, thistype;
        char *src, *dest;

        mask = NULL;
        values = NULL;

        if (!PyArray_Check(self)) {
                PyErr_SetString(PyExc_TypeError, 
				"putmask: first argument must "\
				"be an array");
                return NULL;
        }
        if (!PyArray_ISCONTIGUOUS(self)) {
                PyErr_SetString(PyExc_ValueError, 
				"putmask: first argument must be contiguous");
                return NULL;
        }

        max_item = PyArray_SIZE(self);
        dest = self->data;
        chunk = self->descr->elsize;

        mask = (PyArrayObject *)\
		PyArray_FROM_OTF(mask0, PyArray_BOOL, CARRAY_FLAGS | FORCECAST);
	if (mask == NULL) goto fail;
        ni = PyArray_SIZE(mask);
        if (ni != max_item) {
                PyErr_SetString(PyExc_ValueError, 
				"putmask: mask and data must be "\
				"the same size");
                goto fail;
        }

	thistype = self->descr->type_num;
        values = (PyArrayObject *)\
		PyArray_ContiguousFromAny(values0, thistype, 0, 0);
	if (values == NULL) goto fail;
        nv = PyArray_SIZE(values);	 /* zero if null array */
        if (nv > 0) {
                if (thistype == PyArray_OBJECT) {
                        for(i=0; i<ni; i++) {
                                src = values->data + chunk * (i % nv);
                                tmp = ((Bool *)(mask->data))[i];
                                if (tmp) {
					Py_INCREF(*((PyObject **)src));
                                        Py_XDECREF(*((PyObject **)(dest+i*chunk)));
                                        memmove(dest + i * chunk, src, chunk);
                                }
			}
                }
                else {
                        for(i=0; i<ni; i++) {
                                src = values->data + chunk * (i % nv);
                                tmp = ((Bool *)(mask->data))[i];
                                if (tmp) memmove(dest + i * chunk, src, chunk);
			}
		}
        }

        Py_XDECREF(values);
        Py_XDECREF(mask);
        Py_INCREF(Py_None);
        return Py_None;
	
 fail:
        Py_XDECREF(mask);
        Py_XDECREF(values);
        return NULL;
}


/* This conversion function can be used with the "O&" argument for
   PyArg_ParseTuple.  It will immediately return an object of array type
   or will convert to a CARRAY any other object.  

   If you use PyArray_Converter, you must DECREF the array when finished
   as you get a new reference to it.
*/
    
/*MULTIARRAY_API
 Useful to pass as converter function for O& processing in
 PyArgs_ParseTuple.
*/
static int 
PyArray_Converter(PyObject *object, PyObject **address) 
{
        if (PyArray_Check(object)) {
                *address = object;
		Py_INCREF(object);
                return PY_SUCCEED;
        }
        else {
		*address = PyArray_FromAny(object, NULL, 0, 0, CARRAY_FLAGS);
		if (*address == NULL) return PY_FAIL;
		return PY_SUCCEED;
        }
}

/*MULTIARRAY_API
 Convert an object to true / false
*/
static int
PyArray_BoolConverter(PyObject *object, Bool *val)
{    
    if (PyObject_IsTrue(object))
        *val=TRUE;
    else *val=FALSE;
    if (PyErr_Occurred())
        return PY_FAIL;
    return PY_SUCCEED;
}


/*MULTIARRAY_API
 Typestr converter
*/
static int
PyArray_TypestrConvert(int itemsize, int gentype)
{
	register int newtype = gentype;
	
	if (gentype == PyArray_SIGNEDLTR) {
		switch(itemsize) {
		case 1:
			newtype = PyArray_INT8;
			break;
		case 2:
			newtype = PyArray_INT16;
			break;
		case 4:
			newtype = PyArray_INT32;
			break;
		case 8:
			newtype = PyArray_INT64;
			break;
#ifdef PyArray_INT128
		case 16:
			newtype = PyArray_INT128;
			break;
#endif
		default:
			newtype = PyArray_NOTYPE;
		}
	}

	else if (gentype == PyArray_UNSIGNEDLTR) {
		switch(itemsize) {
		case 1:
			newtype = PyArray_UINT8;
			break;
		case 2:
			newtype = PyArray_UINT16;
			break;
		case 4:
			newtype = PyArray_UINT32;
			break;
		case 8:
			newtype = PyArray_UINT64;
			break;
#ifdef PyArray_INT128
		case 16:
			newtype = PyArray_UINT128;
			break;
#endif
		default:
			newtype = PyArray_NOTYPE;
			break;
		}
	}
	else if (gentype == PyArray_FLOATINGLTR) {
		switch(itemsize) {
		case 4:
			newtype = PyArray_FLOAT32;
			break;
		case 8:
			newtype = PyArray_FLOAT64;
			break;
#ifdef PyArray_FLOAT80
                case 10:
			newtype = PyArray_FLOAT80;
			break;
#endif
#ifdef PyArray_FLOAT96
		case 12:
			newtype = PyArray_FLOAT96;
			break;
#endif		       
#ifdef PyArray_FLOAT128
		case 16:
			newtype = PyArray_FLOAT128;
			break;
#endif
		default:
			newtype = PyArray_NOTYPE;
		}		
	}
	
	else if (gentype == PyArray_COMPLEXLTR) {
		switch(itemsize) {
		case 8:
			newtype = PyArray_COMPLEX64;
			break;
		case 16:
			newtype = PyArray_COMPLEX128;
			break;
#ifdef PyArray_FLOAT80
                case 20:
			newtype = PyArray_COMPLEX160;
			break;
#endif
#ifdef PyArray_FLOAT96
		case 24:
			newtype = PyArray_COMPLEX192;			
			break;
#endif		       
#ifdef PyArray_FLOAT128
		case 32:
			newtype = PyArray_COMPLEX256;
			break;
#endif
		default:
			newtype = PyArray_NOTYPE;
		}		
	}

	return newtype;
}


/* this function takes a Python object which exposes the (single-segment)
   buffer interface and returns a pointer to the data segment
   
   You should increment the reference count by one of buf->base
   if you will hang on to a reference

   You only get a borrowed reference to the object. Do not free the
   memory...
*/


/*MULTIARRAY_API
 Get buffer chunk from object
*/
static int
PyArray_BufferConverter(PyObject *obj, PyArray_Chunk *buf)
{
        int buflen;

        buf->ptr = NULL;
        buf->flags = BEHAVED_FLAGS;
        buf->base = NULL;

	if (obj == Py_None)
		return PY_SUCCEED;

        if (PyObject_AsWriteBuffer(obj, &(buf->ptr), &buflen) < 0) {
                PyErr_Clear();
                buf->flags &= ~WRITEABLE;
                if (PyObject_AsReadBuffer(obj, (const void **)&(buf->ptr), 
                                          &buflen) < 0)
                        return PY_FAIL;
        }
        buf->len = (intp) buflen;
        
        /* Point to the base of the buffer object if present */
        if (PyBuffer_Check(obj)) buf->base = ((PyArray_Chunk *)obj)->base;
        if (buf->base == NULL) buf->base = obj;
        
        return PY_SUCCEED;                    
}



/* This function takes a Python sequence object and allocates and
   fills in an intp array with the converted values.

   **Remember to free the pointer seq.ptr when done using
   PyDimMem_FREE(seq.ptr)**
*/

/*MULTIARRAY_API
 Get intp chunk from sequence
*/
static int
PyArray_IntpConverter(PyObject *obj, PyArray_Dims *seq)
{
        int len;
        int nd;

        seq->ptr = NULL;
        if (obj == Py_None) return PY_SUCCEED;
        len = PySequence_Size(obj);
        if (len == -1) { /* Check to see if it is a number */
                if (PyNumber_Check(obj)) len = 1;
        }
        if (len < 0) {
                PyErr_SetString(PyExc_TypeError, 
                                "expected sequence object with len >= 0");
                return PY_FAIL;
        }
        if (len > MAX_DIMS) {
                PyErr_Format(PyExc_ValueError, "sequence too large; "   \
                             "must be smaller than %d", MAX_DIMS);
                return PY_FAIL;
        }
	if (len > 0) {
		seq->ptr = PyDimMem_NEW(len);
		if (seq->ptr == NULL) {
			PyErr_NoMemory();
			return PY_FAIL;
		}
	}
        seq->len = len;
        nd = PyArray_IntpFromSequence(obj, (intp *)seq->ptr, len);
        if (nd == -1 || nd != len) {
		PyDimMem_FREE(seq->ptr);
		seq->ptr=NULL;
		return PY_FAIL;
	}
        return PY_SUCCEED;
}


/* A tuple type would be either (generic typeobject, typesize) 
   or (fixed-length data-type, shape) 

   or (inheriting data-type, new-data-type)
   The new data-type must have the same itemsize as the inheriting data-type
   unless the latter is 0 
   
   Thus (int32, {'real':(int16,0),'imag',(int16,2)})

   is one way to specify a descriptor that will give 
   a['real'] and a['imag'] to an int32 array.
*/

/* leave type reference alone */
static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag) 
{
	PyArray_Descr *new;
	PyArray_Descr *conv;
	
	*errflag = 0;
	if (!PyArray_DescrConverter(newobj, &conv)) {
		return NULL;
	}
	*errflag = 1;
	if (type == &OBJECT_Descr) {
		PyErr_SetString(PyExc_ValueError,
				"cannot base a new descriptor on an"\
				" OBJECT descriptor.");
		return NULL;
	}
	new = PyArray_DescrNew(type);
	if (new == NULL) return NULL;

	if (new->elsize && new->elsize != conv->elsize) {
		PyErr_SetString(PyExc_ValueError, 
				"mismatch in size of old"\
				"and new data-descriptor");
		return NULL;
	}
	new->elsize = conv->elsize;
	new->fields = conv->fields;
	Py_XINCREF(new->fields);
	Py_DECREF(conv);
	*errflag = 0;
	return new;
}

static PyArray_Descr *
_convert_from_tuple(PyObject *obj) 
{
	PyArray_Descr *type, *res;
	PyObject *val;
	int errflag;
	
	if (PyTuple_GET_SIZE(obj) < 2) return NULL;

	if (!PyArray_DescrConverter(PyTuple_GET_ITEM(obj,0), &type)) 
		return NULL;
	val = PyTuple_GET_ITEM(obj,1);
	/* try to interpret next item as a type */
	res = _use_inherit(type, val, &errflag);
	if (res || errflag) {
		Py_DECREF(type);
		if (res) return res;
		else return NULL;
	}
	PyErr_Clear();
	/* We get here if res was NULL but errflag wasn't set
	   --- i.e. the conversion to a data-descr failed in _use_inherit
	*/

	if (type->elsize == 0) { /* interpret next item as a typesize */
		int itemsize;
		itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));
		if (error_converting(itemsize)) {
			PyErr_SetString(PyExc_ValueError, 
					"invalid itemsize in generic type "\
					"tuple");
			goto fail;
		}
		PyArray_DESCR_REPLACE(type);
		type->elsize = itemsize;
	}
	else {
		/* interpret next item as shape (if it's a tuple)
		   and reset the type to PyArray_VOID with 
		   anew fields attribute. 
	       */
		PyArray_Dims shape={NULL,-1};
		PyArray_Descr *newdescr;
		if (!(PyArray_IntpConverter(val, &shape)) || 
		    (shape.len > MAX_DIMS)) {
			PyDimMem_FREE(shape.ptr);
			PyErr_SetString(PyExc_ValueError, 
					"invalid shape in fixed-type tuple.");
			goto fail;
		}
		newdescr = PyArray_DescrNew(type);
		if (newdescr == NULL) {PyDimMem_FREE(shape.ptr); goto fail;}
		newdescr->elsize *= PyArray_MultiplyList(shape.ptr, 
							 shape.len);
		PyDimMem_FREE(shape.ptr);
		newdescr->type_num = PyArray_VOID;
		newdescr->subarray = malloc(sizeof(PyArray_ArrayDescr));
		newdescr->subarray->base = type;
		Py_INCREF(val);
		newdescr->subarray->shape = val;
		Py_XDECREF(newdescr->fields);
		newdescr->fields = NULL;
		type = newdescr;
	}
	return type;

 fail:
	Py_XDECREF(type);
	return NULL;
}

/* a list specifying a data-type can just be
   a list of formats.  The names for the fields
   will default to f1, f2, f3, and so forth.
*/

static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
	int n, i;
	int totalsize;
	PyObject *fields;
	PyArray_Descr *conv=NULL;
	PyArray_Descr *new;
	PyObject *key, *tup;
       	int ret;
	int maxalign=0;

	
	n = PyList_GET_SIZE(obj);
	totalsize = 0;
	if (n==0) return NULL;
	fields = PyDict_New();
	for (i=0; i<n; i++) {
		tup = PyTuple_New(2);
		key = PyString_FromFormat("f%d", i+1);
		ret = PyArray_DescrConverter(PyList_GET_ITEM(obj, i), &conv);
		PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
		PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));
		PyDict_SetItem(fields, key, tup);
		Py_DECREF(tup);
		Py_DECREF(key);
		if (ret == PY_FAIL) goto fail;
		totalsize += conv->elsize;
		if (align) {
			int _align;
			_align = conv->alignment;
			if (_align > 1) totalsize =			\
				((totalsize + _align - 1)/_align)*_align;
			maxalign = MAX(maxalign, _align);
		}
	}
	new = PyArray_DescrNewFromType(PyArray_VOID);
	new->fields = fields;
	if (maxalign > 1) {
		totalsize = ((totalsize+maxalign-1)/maxalign)*maxalign;
	}
	if (align) new->alignment = maxalign;
	new->elsize = totalsize;
	return new;

 fail:
	Py_DECREF(fields);
	return NULL;
}


/* a dictionary specifying a data-type
   must have at least two and up to four
   keys These must all be sequences of the same length.

   "names" --- field names 
   "formats" --- the data-type descriptors for the field.
   
   Optional:

   "offsets" --- integers indicating the offset into the 
                 record of the start of the field.
		 if not given, then "consecutive offsets" 
		 will be assumed and placed in the dictionary.
   
   "titles" --- Allows the use of an additional key
                 for the fields dictionary.
                            
Attribute-lookup-based field names merely has to query the fields 
dictionary of the data-descriptor.  Any result present can be used
to return the correct field.

So, the notion of what is a name and what is a title is really quite
arbitrary.  

What does distinguish a title, however, is that if it is not None, 
it will be placed at the end of the tuple inserted into the 
fields dictionary.

If the dictionary does not have "names" and "formats" entries,
then it will be checked for conformity and used directly. 
*/

static PyArray_Descr *
_use_fields_dict(PyObject *obj)
{
        static PyObject *module=NULL;

        if (module==NULL) {
                module = PyImport_ImportModule("scipy.base._internal");
                if (module == NULL) return NULL;
        }
        return (PyArray_Descr *)PyObject_CallMethod(module, "_usefields", 
						    "O", obj);
}

static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
	PyArray_Descr *new;
	PyObject *fields=NULL;
	PyObject *names, *offsets, *descrs, *titles;
	int n, i;
	int totalsize;
	int maxalign=0;
	
	fields = PyDict_New();
	if (fields == NULL) return (PyArray_Descr *)PyErr_NoMemory();
	
	names = PyDict_GetItemString(obj, "names");
	descrs = PyDict_GetItemString(obj, "formats");

	if (!names || !descrs) {
		Py_DECREF(fields);
		return _use_fields_dict(obj);
	}
	n = PyObject_Length(names);
	offsets = PyDict_GetItemString(obj, "offsets");
	titles = PyDict_GetItemString(obj, "titles");
	if ((n > PyObject_Length(descrs)) ||			\
	    (offsets && (n > PyObject_Length(offsets))) ||	\
	    (titles && (n > PyObject_Length(titles)))) {
		PyErr_SetString(PyExc_ValueError,
				"all items in the dictionary must have" \
				" the same length.");
		goto fail;
	}

	totalsize = 0;
	for(i=0; i<n; i++) {
		PyObject *tup, *descr, *index, *item, *name, *off;
		int len, ret;
		PyArray_Descr *newdescr;

		/* Build item to insert (descr, offset, [title])*/
		len = 2;
		item = NULL;
		index = PyInt_FromLong(i);
		if (titles) {
			item=PyObject_GetItem(titles, index);
			if (item && item != Py_None) len = 3;
			else Py_XDECREF(item);
			PyErr_Clear();
		}
		tup = PyTuple_New(len);
		descr = PyObject_GetItem(descrs, index);
		ret = PyArray_DescrConverter(descr, &newdescr);
		Py_DECREF(descr);
		PyTuple_SET_ITEM(tup, 0, (PyObject *)newdescr);
		if (offsets) {
			long offset;
			off = PyObject_GetItem(offsets, index);
			offset = PyInt_AsLong(off);
			PyTuple_SET_ITEM(tup, 1, off);
			if (offset < totalsize) {
				PyErr_SetString(PyExc_ValueError,
						"invalid offset (must be "\
						"ordered)");
				ret = PY_FAIL;
			}
			if (offset > totalsize) totalsize = offset;
		}
		else 
			PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(totalsize));
		if (len == 3) PyTuple_SET_ITEM(tup, 2, item);
		name = PyObject_GetItem(names, index);
		Py_DECREF(index);

		/* Insert into dictionary */
		PyDict_SetItem(fields, name, tup);
		Py_DECREF(name);
		if (len == 3) PyDict_SetItem(fields, item, tup);
		Py_DECREF(tup);
		if ((ret == PY_FAIL) || (newdescr->elsize == 0)) goto fail;
		totalsize += newdescr->elsize;
		if (align) {
			int _align = newdescr->alignment;
			if (_align > 1) totalsize =			\
				((totalsize + _align - 1)/_align)*_align;
			maxalign = MAX(maxalign,_align);
		}
	}

	new = PyArray_DescrNewFromType(PyArray_VOID);
	if (new == NULL) goto fail;
	if (maxalign > 1)
		totalsize = ((totalsize + maxalign - 1)/maxalign)*maxalign;
	if (align) new->alignment = maxalign;
	new->elsize = totalsize;
	new->fields = fields;
	return new;

 fail:
	Py_XDECREF(fields);
	return NULL;
}

/* 
   any object with 
   the .fields attribute and/or .itemsize attribute 
   (if the .fields attribute does not give
    the total size -- i.e. a partial record naming).
   If itemsize is given it must be >= size computed from fields
   
   The .fields attribute must return a convertible dictionary if 
   present.  Result inherits from PyArray_VOID.
*/


/*MULTIARRAY_API
 Get typenum from an object -- None goes to NULL
*/
static int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
	if (obj == Py_None) {
		*at = NULL;
		return PY_SUCCEED;
	}
	else return PyArray_DescrConverter(obj, at);
}

/* This function takes a Python object representing a type and converts it 
   to a the correct PyArray_Descr * structure to describe the type.
   
   Many objects can be used to represent a data-type which in SciPy is
   quite a flexible concept. 

   This is the central code that converts Python objects to 
   Type-descriptor objects that are used throughout scipy.
 */

/* new reference in *at */
/*MULTIARRAY_API
 Get typenum from an object -- None goes to &LONG_descr
*/
static int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
        char *type;
        int check_num=PyArray_NOTYPE+10;
	int len;
	PyObject *item;
	int elsize = 0;


	*at=NULL;
	
	/* default */
        if (obj == Py_None) {
		*at = PyArray_DescrFromType(PyArray_LONG);
		return PY_SUCCEED;
	}

	
	if (PyArray_DescrCheck(obj)) {
		*at = (PyArray_Descr *)obj;
		Py_INCREF(*at);
		return PY_SUCCEED;
	}
	
        if (PyType_Check(obj)) {
		if (PyType_IsSubtype((PyTypeObject *)obj, 
				     &PyGenericArrType_Type)) {
			*at = PyArray_DescrFromTypeObject(obj);
			if (*at) return PY_SUCCEED;
			else return PY_FAIL;
		}
		check_num = PyArray_OBJECT;
		if (obj == (PyObject *)(&PyInt_Type))
			check_num = PyArray_LONG;
		else if (obj == (PyObject *)(&PyFloat_Type)) 
			check_num = PyArray_DOUBLE;
		else if (obj == (PyObject *)(&PyComplex_Type)) 
			check_num = PyArray_CDOUBLE;
		else if (obj == (PyObject *)(&PyBool_Type))
			check_num = PyArray_BOOL;
                else if (obj == (PyObject *)(&PyString_Type))
                        check_num = PyArray_STRING;
                else if (obj == (PyObject *)(&PyUnicode_Type))
                        check_num = PyArray_UNICODE;
		else if (obj == (PyObject *)(&PyBuffer_Type))
			check_num = PyArray_VOID;
		else {
			*at = _arraydescr_fromobj(obj);
			if (*at) return PY_SUCCEED;
		}
		goto finish;
	}

	/* or a typecode string */

	if (PyString_Check(obj)) {
		/* Check for a string typecode. */
		type = PyString_AS_STRING(obj);
		len = PyString_GET_SIZE(obj);		
		if (len > 0) {
			check_num = (int) type[0];
		}
		if (len > 1) {
			elsize = atoi(type+1);
			/* When specifying length of UNICODE
			   the number of characters is given to match 
			   the STRING interface.  Each character can be
			   more than one byte and itemsize must be
			   the number of bytes.
			*/
			if (check_num == PyArray_UNICODELTR)
				elsize *= sizeof(Py_UNICODE);
			
			/* Support for generic processing 
			   c4, i4, f8, etc...
			*/
			else if ((check_num != PyArray_STRINGLTR) &&
				 (check_num != PyArray_VOIDLTR)) {
				if (elsize == 0) {
					/* reset because string conversion failed */
					check_num = PyArray_NOTYPE+10;
				}
				else {
					check_num =			\
						PyArray_TypestrConvert(elsize,
								       check_num);
					elsize = 0;
				}
			}
		}
	}
	/* or a tuple */
 	else if (PyTuple_Check(obj)) {
		*at = _convert_from_tuple(obj);
		if (*at == NULL) return PY_FAIL;
		return PY_SUCCEED;
	}
	/* or a list */
	else if (PyList_Check(obj)) {
		*at = _convert_from_list(obj,0);
		if (*at == NULL) return PY_FAIL;
		return PY_SUCCEED;
	}
	/* or a dictionary */
	else if (PyDict_Check(obj)) {
		*at = _convert_from_dict(obj,0);
		if (*at == NULL) return PY_FAIL;
		return PY_SUCCEED;
	}
        else {
		*at = _arraydescr_fromobj(obj);
		if (*at) return PY_SUCCEED;
		goto fail;
	}
	if (PyErr_Occurred()) goto fail;

	/*
	if (check_num == PyArray_NOTYPE) return PY_FAIL;
	*/

 finish:
        if ((check_num == PyArray_NOTYPE+10) ||			\
	    (*at = PyArray_DescrFromType(check_num))==NULL) {
		/* Now check to see if the object is registered
		   in typeDict */
		if (typeDict != NULL) {
			item = PyDict_GetItem(typeDict, obj);
			if (item) return PyArray_DescrConverter(item, at);
		}
		goto fail;
	}
	
	if (((*at)->elsize == 0) && (elsize != 0)) {
		PyArray_DESCR_REPLACE(*at);
		(*at)->elsize = elsize;
	}
	
        return PY_SUCCEED;

 fail:
	PyErr_SetString(PyExc_TypeError, 
			"data type not understood");
	*at=NULL;
	return PY_FAIL;
}	


/* This function returns true if the two typecodes are 
   equivalent (same basic kind and same itemsize).
*/

/*MULTIARRAY_API*/
static Bool
PyArray_EquivalentTypes(PyArray_Descr *typ1, PyArray_Descr *typ2)
{
	register int typenum1=typ1->type_num;
	register int typenum2=typ2->type_num;
	register int size1=typ1->elsize;
	register int size2=typ2->elsize;

	if (size1 != size2) return FALSE;
	if (typ1->fields != typ2->fields) return FALSE;

	if (typenum1 == PyArray_VOID || \
	    typenum2 == PyArray_VOID) {
		return ((typenum1 == typenum2) && 
			(typ1->typeobj == typ2->typeobj) &&
			(typ1->fields == typ2->fields));
	}
	return (typ1->kind == typ2->kind);
}

/*** END C-API FUNCTIONS **/


#define _ARET(x) PyArray_Return((PyArrayObject *)(x))

static char doc_fromobject[] = "array(object, dtype=None, copy=1, fortran=0, "\
        "subok=0)\n"\
        "will return a new array formed from the given object type given.\n"\
        "Object can anything with an __array__ method, or any object\n"\
        "exposing the array interface, or any (nested) sequence.\n"\
        "If no type is given, then the type will be determined as the\n"\
        "minimum type required to hold the objects in the sequence.\n"\
        "If copy is zero and sequence is already an array with the right \n"\
        "type, a reference will be returned.  If the sequence is an array,\n"\
        "type can be used only to upcast the array.  For downcasting \n"\
        "use .astype(t) method.  If subok is true, then subclasses of the\n"\
        "array may be returned. Otherwise, a base-class ndarray is returned";

static PyObject *
_array_fromobject(PyObject *ignored, PyObject *args, PyObject *kws)
{
	PyObject *op, *ret=NULL;
	static char *kwd[]= {"object", "dtype", "copy", "fortran", "subok", 
                             NULL};
        Bool subok=FALSE;
	Bool copy=TRUE;
	PyArray_Descr *type=NULL;
	PyArray_Descr *oldtype=NULL;
	Bool fortran=FALSE;
	int flags=0;

	if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O&O&O&O&", kwd, &op, 
					PyArray_DescrConverter2,
                                        &type, 
					PyArray_BoolConverter, &copy, 
					PyArray_BoolConverter, &fortran,
                                        PyArray_BoolConverter, &subok)) 
		return NULL;

	/* fast exit if simple call */
	if ((PyArray_CheckExact(op) || PyBigArray_CheckExact(op))) {
		if (type==NULL) {
			if (!copy && fortran==PyArray_ISFORTRAN(op)) {
				Py_INCREF(op);
				return op;
			}
			else {
				return PyArray_NewCopy((PyArrayObject*)op, 
						       fortran);
			}
		}
		/* One more chance */
		oldtype = PyArray_DESCR(op);
		if (PyArray_EquivalentTypes(oldtype, type)) {
			if (!copy && fortran==PyArray_ISFORTRAN(op)) {
				Py_INCREF(op);
				return op;
			}
			else {
				ret = PyArray_NewCopy((PyArrayObject*)op,
						      fortran);
				if (oldtype == type) return ret;
				Py_INCREF(oldtype);
				Py_DECREF(PyArray_DESCR(ret));
				PyArray_DESCR(ret) = oldtype;
				return ret;
			}
		}
	}

	if (copy) {
		flags = ENSURECOPY;
	}
	if (fortran) {
		flags |= FORTRAN;
	}
        if (!subok) {
                flags |= ENSUREARRAY;
        }

	if ((ret = PyArray_FromAny(op, type, 0, 0, flags)) == NULL) 
		return NULL;

	return ret;
}

/* accepts NULL type */
/* steals referenct to type */
/*MULTIARRAY_API
 Empty
*/
static PyObject *
PyArray_Empty(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
	PyArrayObject *ret;
        
	if (!type) type = PyArray_DescrFromType(PyArray_LONG);
	ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, 
						    type, nd, dims, 
						    NULL, NULL,
						    fortran, NULL);
	if (ret == NULL) return NULL;
        
	if ((PyArray_TYPE(ret) == PyArray_OBJECT)) {
                PyArray_FillObjectArray(ret, Py_None);
	}
	return (PyObject *)ret;
}


static char doc_empty[] = "empty((d1,...,dn),dtype=int,fortran=0) will return a new array\n of shape (d1,...,dn) and given type with all its entries uninitialized. This can be faster than zeros.";

static PyObject *
array_empty(PyObject *ignored, PyObject *args, PyObject *kwds) 
{
        
	static char *kwlist[] = {"shape","dtype","fortran",NULL};
	PyArray_Descr *typecode=NULL;
        PyArray_Dims shape = {NULL, 0};
	Bool fortran = FALSE;	
        PyObject *ret=NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&",
					 kwlist, PyArray_IntpConverter,
                                         &shape, 
                                         PyArray_DescrConverter,
					 &typecode, 
					 PyArray_BoolConverter, &fortran)) 
		goto fail;

	ret = PyArray_Empty(shape.len, shape.ptr, typecode, fortran);        
        PyDimMem_FREE(shape.ptr);
        return ret;

 fail:
	PyDimMem_FREE(shape.ptr);
	return ret;
}

static char doc_scalar[] = "scalar(dtypestr,obj) will return a new scalar array of the given type initialized with obj. Mainly for pickle support. typestr must be a valid data typestr (complete with < > or |).  If dtypestr is object, then obj can be any object, otherwise obj must be a string. If obj is not given it will be interpreted as None for object type and zeros for all other types.";

static PyObject *
array_scalar(PyObject *ignored, PyObject *args, PyObject *kwds) 
{
        
	static char *kwlist[] = {"dtypestr","obj", NULL};
	PyArray_Descr *typecode;
	PyObject *obj=NULL;
	char *typestr;
	int typestrlen;
	int swap, alloc=0;
	void *dptr;
	PyObject *ret;

	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "z#|O",
					 kwlist, &typestr, &typestrlen,
					 &obj)) 
		return NULL;
	
	if (!(typecode=_array_typedescr_fromstr(typestr, &swap)))
		return NULL;
	
	if (typecode->elsize == 0) {
		PyErr_SetString(PyExc_ValueError,		\
				"itemsize cannot be zero");
		Py_DECREF(typecode);
		return NULL;
	}

	if (typecode->type_num == PyArray_OBJECT) {
		if (obj == NULL) obj = Py_None;
		dptr = &obj;
		swap = 0;
	}
	else {
		if (obj == NULL) {
			dptr = malloc(typecode->elsize);
			if (dptr == NULL) {
				Py_DECREF(typecode);		
				return PyErr_NoMemory();
			}
			memset(dptr, '\0', typecode->elsize);
			alloc = 1;
		}
		else {
			if (!PyString_Check(obj)) {
				PyErr_SetString(PyExc_TypeError, 
						"initializing object must "\
						"be a string");
				Py_DECREF(typecode);				
				return NULL;
			}
			if (PyString_GET_SIZE(obj) < typecode->elsize) {
				PyErr_SetString(PyExc_ValueError,
						"initialization string is too"\
						" small");
				Py_DECREF(typecode);				
				return NULL;
			}
			dptr = PyString_AS_STRING(obj);
		}
	}

	ret = PyArray_Scalar(dptr, swap, typecode, NULL);
	Py_DECREF(typecode);
	
	/* free dptr which contains zeros */
	if (alloc) free(dptr);
	return ret;
}


/* steal a reference */
/* accepts NULL type */
/*MULTIARRAY_API
 Zeros
*/
static PyObject *
PyArray_Zeros(int nd, intp *dims, PyArray_Descr *type, int fortran)
{
	PyArrayObject *ret;
	intp n;

	if (!type) type = PyArray_DescrFromType(PyArray_LONG);
	ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, 
						    type,
						    nd, dims, 
						    NULL, NULL, 
						    fortran, NULL);
	if (ret == NULL) return NULL;
        
	if ((PyArray_TYPE(ret) == PyArray_OBJECT)) {
		PyObject *zero = PyInt_FromLong(0);
                PyArray_FillObjectArray(ret, zero);
                Py_DECREF(zero);
	}
	else {
		n = PyArray_NBYTES(ret);
		memset(ret->data, 0, n);
	}
	return (PyObject *)ret;

}

static char doc_zeros[] = "zeros((d1,...,dn),dtype=int,fortran=0) will return a new array of shape (d1,...,dn) and type typecode with all it's entries initialized to zero.";


static PyObject *
array_zeros(PyObject *ignored, PyObject *args, PyObject *kwds) 
{
	static char *kwlist[] = {"shape","dtype","fortran",NULL};
	PyArray_Descr *typecode=NULL;
        PyArray_Dims shape = {NULL, 0};
	Bool fortran = FALSE;	
        PyObject *ret=NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&",
					 kwlist, PyArray_IntpConverter,
                                         &shape, 
                                         PyArray_DescrConverter,
					 &typecode, 
					 PyArray_BoolConverter,
					 &fortran)) 
		goto fail;

	ret = PyArray_Zeros(shape.len, shape.ptr, typecode, (int) fortran);
        PyDimMem_FREE(shape.ptr);
        return ret;

 fail:
	PyDimMem_FREE(shape.ptr);
	return ret;
}

static char doc_set_typeDict[] = "set_typeDict(dict) set the internal "\
	"dictionary that can look up an array type using a registered "\
	"code";

static PyObject *
array_set_typeDict(PyObject *ignored, PyObject *args)
{
	PyObject *dict;
	if (!PyArg_ParseTuple(args, "O", &dict)) return NULL;
	Py_XDECREF(typeDict); /* Decrement old reference (if any)*/
	typeDict = dict;
	Py_INCREF(dict);  /* Create an internal reference to it */
	Py_INCREF(Py_None);
	return Py_None;
}

/* steals a reference to dtype -- accepts NULL */
/*OBJECT_API*/
static PyObject *
PyArray_FromString(char *data, intp slen, PyArray_Descr *dtype, 
		   intp n, int swap)
{
	int itemsize;
	PyArrayObject *ret;

	if (dtype == NULL)
		dtype=PyArray_DescrFromType(PyArray_LONG);
	
	if (dtype == &OBJECT_Descr) {
		PyErr_SetString(PyExc_ValueError, 
				"Cannot create an object array from a"\
				" string.");
		Py_DECREF(dtype);
		return NULL;
	}
	
	itemsize = dtype->elsize;
	if (itemsize == 0) {
		PyErr_SetString(PyExc_ValueError, "zero-valued itemsize");
		Py_DECREF(dtype);
		return NULL;
	}
	
	if (n < 0 ) {
		if (slen % itemsize != 0) {
			PyErr_SetString(PyExc_ValueError, 
					"string size must be a multiple"\
					" of element size");
			Py_DECREF(dtype);
			return NULL;
		}
		n = slen/itemsize;
	} else {
		if (slen < n*itemsize) {
			PyErr_SetString(PyExc_ValueError, 
					"string is smaller than requested"\
					" size");
			Py_DECREF(dtype);
			return NULL;
		}
	}

	if ((ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, 
							 dtype,
							 1, &n, 
							 NULL, NULL,
							 0, NULL)) == NULL)
		return NULL;
		
	memcpy(ret->data, data, n*dtype->elsize);
	if (swap) ret->flags &= ~NOTSWAPPED;
	return (PyObject *)ret;
}

static char doc_fromString[] = "fromstring(string, dtype=int, count=-1, swap=False) returns a new 1d array initialized from the raw binary data in string.  If count is positive, the new array will have count elements, otherwise it's size is determined by the size of string.";

static PyObject *
array_fromString(PyObject *ignored, PyObject *args, PyObject *keywds)
{
	char *data;
	longlong nin=-1;
	int s;
	static char *kwlist[] = {"string", "dtype", "count", "swap",NULL};
	PyArray_Descr *descr=NULL;
	int swapped=FALSE;

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "s#|O&LO&", kwlist, 
					 &data, &s, 
					 PyArray_DescrConverter, &descr,
					 &nin, 
					 PyArray_BoolConverter,
					 &swapped)) {
		return NULL;
	}

	return PyArray_FromString(data, (intp)s, descr, (intp)nin, swapped);
}

/* This needs an open file object and reads it in directly. 
   memory-mapped files handled differently through buffer interface.

file pointer number in resulting 1d array 
(can easily reshape later, -1 for to end of file)
type of array
sep is a separator string for character-based data (or NULL for binary)
   " " means whitespace
*/

/*OBJECT_API*/
static PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *typecode, intp num, char *sep)
{
	PyArrayObject *r;
	size_t nread = 0;
	PyArray_ScanFunc *scan;
	Bool binary;

	if (typecode->elsize == 0) {
		PyErr_SetString(PyExc_ValueError, "0-sized elements.");
		return NULL;
	}

	binary = ((sep == NULL) || (strlen(sep) == 0));
	if (num == -1 && binary) {  /* Get size for binary file*/
		intp start, numbytes;
		start = (intp )ftell(fp);
		fseek(fp, 0, SEEK_END);
		numbytes = (intp )ftell(fp) - start;
		rewind(fp);
		if (numbytes == -1) {
			PyErr_SetString(PyExc_IOError, 
					"could not seek in file");
			return NULL;
		}
		num = numbytes / typecode->elsize;
	}
	
	if (binary) { /* binary data */
		r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, 
							  typecode,
							  1, &num, 
							  NULL, NULL, 
							  0, NULL);
		if (r==NULL) return NULL;
		nread = fread(r->data, typecode->elsize, num, fp);
	}
	else {  /* character reading */
		intp i;
		char *dptr;
		int done=0;

		scan = typecode->f->scanfunc;
		if (scan == NULL) {
			PyErr_SetString(PyExc_ValueError, 
					"don't know how to read "	\
					"character files with that "	\
					"array type");
			return NULL;
		}

		if (num != -1) {  /* number to read is known */
			r = (PyArrayObject *)\
				PyArray_NewFromDescr(&PyArray_Type, 
						     typecode, 
						     1, &num, 
						     NULL, NULL, 
						     0, NULL);
			if (r==NULL) return NULL;
			dptr = r->data;
			for (i=0; i < num; i++) {
				if (done) break;
				done = scan(fp, dptr, sep, NULL);
				if (done < -2) break;
				nread += 1;
				dptr += r->descr->elsize;
			}
			if (PyErr_Occurred()) {
				Py_DECREF(r);
				return NULL;
			}
		}
		else { /* we have to watch for the end of the file and 
			  reallocate at the end */
#define _FILEBUFNUM 4096
			intp thisbuf=0;
			intp size = _FILEBUFNUM;
			intp bytes;
			intp totalbytes;

			r = (PyArrayObject *)\
				PyArray_NewFromDescr(&PyArray_Type, 
						     typecode,
						     1, &size, 
						     NULL, NULL, 
						     0, NULL);
			if (r==NULL) return NULL;
			totalbytes = bytes = size * typecode->elsize;
			dptr = r->data;
			while (!done) {
				done = scan(fp, dptr, sep, NULL);

				/* end of file reached trying to 
				   scan value.  done is 1 or 2
				   if end of file reached trying to
				   scan separator.  Still good value.
				*/
				if (done < -2) break;
				thisbuf += 1;
				nread += 1;
				dptr += r->descr->elsize;
				if (!done && thisbuf == size) {
					totalbytes += bytes;
					r->data = PyDataMem_RENEW(r->data, 
								  totalbytes);
					dptr = r->data + (totalbytes - bytes);
					thisbuf = 0;
				}
			}
			if (PyErr_Occurred()) {
				Py_DECREF(r);
				return NULL;
			}
			r->data = PyDataMem_RENEW(r->data, nread*r->descr->elsize);
			PyArray_DIM(r,0) = nread;
			num = nread;
#undef _FILEBUFNUM
		}
	}
	if (nread < num) {
		fprintf(stderr, "%ld items requested but only %ld read\n", 
			(long) num, (long) nread);
		r->data = PyDataMem_RENEW(r->data, nread * r->descr->elsize);
		PyArray_DIM(r,0) = nread;
	}
	return (PyObject *)r;
}

static char doc_fromfile[] = \
	"fromfile(file=, dtype=int, count=-1, sep='')\n"	\
	"\n"\
	"  Return an array of the given data type from a \n"\
	"  (text or binary) file.   The file argument can be an open file\n"\
	"  or a string with the name of a file to read from.  If\n"\
	"  count==-1, then the entire file is read, otherwise count is\n"\
	"  the number of items of the given type read in.  If sep is ''\n"\
	"  then read a binary file, otherwise it gives the separator\n"\
	"  between elements in a text file.\n"\
	"\n"\
	"  WARNING: This function should be used sparingly, as it is not\n"\
	"  a robust method of persistence.  But it can be useful to\n"\
	"  read in simply-formatted or binary data quickly.";

static PyObject *
array_fromfile(PyObject *ignored, PyObject *args, PyObject *keywds)
{
	PyObject *file=NULL, *ret;
	FILE *fp;
	char *sep="";
	char *mode=NULL;
	longlong nin=-1;
	static char *kwlist[] = {"file", "dtype", "count", "sep", NULL};
	PyArray_Descr *type=NULL;
	
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|O&Ls", kwlist, 
					 &file,
					 PyArray_DescrConverter, &type,
					 &nin, &sep)) {
		return NULL;
	}

	if (type == NULL) type = PyArray_DescrFromType(PyArray_LONG);

	if (PyString_Check(file)) {
		if (sep == "") mode="rb";
		else mode="r";
		file = PyFile_FromString(PyString_AS_STRING(file), mode);
		if (file==NULL) return NULL;
	}
	else {
		Py_INCREF(file);
	}
	fp = PyFile_AsFile(file);
	if (fp == NULL) {
		PyErr_SetString(PyExc_IOError, 
				"first argument must be an open file");
		Py_DECREF(file);
		return NULL;
	}
	ret = PyArray_FromFile(fp, type, (intp) nin, sep);
	Py_DECREF(file);
	return ret;
}

/*OBJECT_API*/
static PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type, 
		   intp count, intp offset, int swapped) 
{
	PyArrayObject *ret;
	char *data;
	int ts;
	intp s, n;
	int itemsize;
	int write=1;


	if (type->type_num == PyArray_OBJECT) {
		PyErr_SetString(PyExc_ValueError, 
				"cannot create an OBJECT array from memory"\
				" buffer");
		Py_DECREF(type);
		return NULL;
	}
	if (type->elsize == 0) {
		PyErr_SetString(PyExc_ValueError, 
				"itemsize cannot be zero in type");
		Py_DECREF(type);
		return NULL; 		
	}

	if (buf->ob_type->tp_as_buffer == NULL || \
	    (buf->ob_type->tp_as_buffer->bf_getwritebuffer == NULL &&	\
	     buf->ob_type->tp_as_buffer->bf_getreadbuffer == NULL)) {
		PyObject *newbuf;
		newbuf = PyObject_GetAttrString(buf, "__buffer__");
		if (newbuf == NULL) {Py_DECREF(type); return NULL;}
		buf = newbuf;
	}
	else {Py_INCREF(buf);}

	if (PyObject_AsWriteBuffer(buf, (void *)&data, &ts)==-1) {
		write = 0;
		PyErr_Clear();
		if (PyObject_AsReadBuffer(buf, (void *)&data, &ts)==-1) {
			Py_DECREF(buf);
			Py_DECREF(type);
			return NULL;
		}
	}
	
	if ((offset < 0) || (offset >= ts)) {
		PyErr_Format(PyExc_ValueError,
			     "offset must be positive and smaller than %"
			     INTP_FMT, (intp)ts);
	}

	data += offset;
	s = (intp)ts - offset;
	n = (intp)count;
	itemsize = type->elsize;
	
	if (n < 0 ) {
		if (s % itemsize != 0) {
			PyErr_SetString(PyExc_ValueError, 
					"buffer size must be a multiple"\
					" of element size");
			Py_DECREF(buf);
			Py_DECREF(type);
			return NULL;
		}
		n = s/itemsize;
	} else {
		if (s < n*itemsize) {
			PyErr_SetString(PyExc_ValueError, 
					"buffer is smaller than requested"\
					" size");
			Py_DECREF(buf);
			Py_DECREF(type);
			return NULL;
		}
	}
	
	if ((ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, 
							 type, 
							 1, &n, 
							 NULL, data, 
							 DEFAULT_FLAGS,
							 NULL)) == NULL) {
		Py_DECREF(buf);
		return NULL;
	}
	
	if (!write) ret->flags &= ~WRITEABLE;
	if (swapped) ret->flags &= ~NOTSWAPPED;

	/* Store a reference for decref on deallocation */
	ret->base = buf;
	PyArray_UpdateFlags(ret, ALIGNED);
	return (PyObject *)ret;
}

static char doc_frombuffer[] = \
	"frombuffer(buffer=, dtype=int, count=-1, offset=0, swap=0)\n"\
	"\n"								\
	"  Returns a 1-d array of data type dtype from buffer. The buffer\n"\
	"   argument must be an object that exposes the buffer interface.\n"\
	"   If count is -1 then the entire buffer is used, otherwise, count\n"\
	"   is the size of the output.  If offset is given then jump that\n"\
	"   far into the buffer. If the buffer has data that is out\n" \
	"   not in machine byte-order, than set swap=1.  The data will not\n"
	"   be byteswapped, but the array will manage it in future\n"\
	"   operations.\n";

static PyObject *
array_frombuffer(PyObject *ignored, PyObject *args, PyObject *keywds)
{
	PyObject *obj=NULL;
	longlong nin=-1, offset=0;
	static char *kwlist[] = {"buffer", "dtype", "count", 
				 "swap", NULL};
	PyArray_Descr *type=NULL;
	int swapped=0;

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|O&LLi", kwlist, 
					 &obj,
					 PyArray_DescrConverter, &type,
					 &nin, &offset, &swapped)) {
		return NULL;
	}
	if (type==NULL)
		type = PyArray_DescrFromType(PyArray_LONG);
	
	return PyArray_FromBuffer(obj, type, (intp)nin, 
				  (intp)offset, swapped);
}


static char doc_concatenate[] = "concatenate((a1,a2,...),axis=None).";

static PyObject *
array_concatenate(PyObject *dummy, PyObject *args, PyObject *kwds) 
{
	PyObject *a0;
	int axis=0;
	static char *kwlist[] = {"seq", "axis", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
					 &a0,
					 PyArray_AxisConverter, &axis))
		return NULL;
	return PyArray_Concatenate(a0, axis);
}

static char doc_innerproduct[] = \
	"inner(a,b) returns the dot product of two arrays, which has\n"\
	"shape a.shape[:-1] + b.shape[:-1] with elements computed by\n" \
	"the product of the elements from the last dimensions of a and b.";

static PyObject *array_innerproduct(PyObject *dummy, PyObject *args) {
	PyObject *b0, *a0;
	
	if (!PyArg_ParseTuple(args, "OO", &a0, &b0)) return NULL;
	
	return _ARET(PyArray_InnerProduct(a0, b0));
}

static char doc_matrixproduct[] = \
	"dot(a,v) returns matrix-multiplication between a and b.  \n"\
	"The product-sum is over the last dimension of a and the \n"\
	"second-to-last dimension of b.";

static PyObject *array_matrixproduct(PyObject *dummy, PyObject *args) {
	PyObject *v, *a;
	
	if (!PyArg_ParseTuple(args, "OO", &a, &v)) return NULL;
	
	return _ARET(PyArray_MatrixProduct(a, v));
}

static char doc_fastCopyAndTranspose[] = "_fastCopyAndTranspose(a)";

static PyObject *array_fastCopyAndTranspose(PyObject *dummy, PyObject *args) {
	PyObject *a0;
	
	if (!PyArg_ParseTuple(args, "O", &a0)) return NULL;
	
	return _ARET(PyArray_CopyAndTranspose(a0));
}

static char doc_correlate[] = "cross_correlate(a,v, mode=0)";

static PyObject *array_correlate(PyObject *dummy, PyObject *args, PyObject *kwds) {
	PyObject *shape, *a0;
	int mode=0;
	static char *kwlist[] = {"a", "v", "mode", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist, 
					 &a0, &shape, &mode)) return NULL;
	
	return PyArray_Correlate(a0, shape, mode);
}


/*MULTIARRAY_API
 Arange, 
*/
static PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
	intp length, i;
	PyObject *range;
	char *rptr;
	int elsize, type;
	double value;
	PyArray_Descr *dbl_descr;

	length = (intp ) ceil((stop - start)/step);
    
	if (length <= 0) {
		length = 0;
		return PyArray_New(&PyArray_Type, 1, &length, type_num,
				   NULL, NULL, 0, 0, NULL);
	}

	range = PyArray_New(&PyArray_Type, 1, &length, type_num, 
			    NULL, NULL, 0, 0, NULL);
	if (range == NULL) return NULL;
	dbl_descr = PyArray_DescrFromType(PyArray_DOUBLE);
    
	rptr = ((PyArrayObject *)range)->data;
	elsize = ((PyArrayObject *)range)->descr->elsize;
	type = ((PyArrayObject *)range)->descr->type_num;
	for (i=0; i < length; i++) {
		value = start + i*step;
		dbl_descr->f->cast[type]((char*)&value, rptr, 1, NULL, 
				      (PyArrayObject *)range);
		rptr += elsize;
	}
    
	return range;
}


static char doc_arange[] = "arange(start, stop=None, step=1, dtype=int)\n\n  Just like range() except it returns an array whose type can be\n specified by the keyword argument typecode.";

static PyObject *
array_arange(PyObject *ignored, PyObject *args, PyObject *kws) {
	PyObject *o_start=NULL, *o_stop=Py_None, *o_step=NULL;
	static char *kwd[]= {"start", "stop", "step", "dtype", NULL};
	double start, stop, step;
	PyArray_Descr *typecode=NULL;
	int type_num;
	int deftype = PyArray_LONG;
	
	if(!PyArg_ParseTupleAndKeywords(args, kws, "O|OOO&", kwd, &o_start,
					&o_stop, &o_step, 
					PyArray_DescrConverter, 
					&typecode)) 
		return NULL;

	deftype = PyArray_ObjectType(o_start, deftype);
	if (o_stop != Py_None) {
		deftype = PyArray_ObjectType(o_stop, deftype);
	}
	if (o_step != NULL) {
		deftype = PyArray_ObjectType(o_step, deftype);
	}

	if (typecode == NULL) {
		type_num = deftype;
	}
	else {
		type_num = typecode->type_num;
	}

	start = PyFloat_AsDouble(o_start);
	if error_converting(start) return NULL;

	if (o_step == NULL) {
		step = 1;
	}
	else {
		step = PyFloat_AsDouble(o_step);
		if error_converting(step) return NULL;
	}

	if (o_stop == Py_None) {
		stop = start;
		start = 0;
	}
	else {
		stop = PyFloat_AsDouble(o_stop);
		if error_converting(stop) return NULL;
	}

	return PyArray_Arange(start, stop, step, type_num);
}

#undef _ARET

/*****
      static char doc_arrayMap[] = "arrayMap(func, [a1,...,an])";

      static PyObject *array_arrayMap(PyObject *dummy, PyObject *args) {
      PyObject *shape, *a0;
  
      if (PyArg_ParseTuple(args, "OO", &a0, &shape) == NULL) return NULL;
	
      return PyArray_Map(a0, shape);
      }
*****/

static char 
doc_set_string_function[] = "set_string_function(f, repr=1) sets the python function f to be the function used to obtain a pretty printable string version of a array whenever a array is printed.  f(M) should expect a array argument M, and should return a string consisting of the desired representation of M for printing.";

static PyObject *
array_set_string_function(PyObject *dummy, PyObject *args, PyObject *kwds) 
{
	PyObject *op;
	int repr=1;
	static char *kwlist[] = {"f", "repr", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, 
					&op, &repr)) return NULL; 
	if (!PyCallable_Check(op)) {
		PyErr_SetString(PyExc_TypeError, 
				"Argument must be callable.");
		return NULL;
	}
	PyArray_SetStringFunction(op, repr);
	Py_INCREF(Py_None);
	return Py_None;
}

static char 
doc_set_ops_function[] = "set_numeric_ops(op=func, ...) sets some or all of the number methods for all array objects.  Don't forget **dict can be used as the argument list.  Returns the functions that were replaced -- can be stored and set later.";

static PyObject *
array_set_ops_function(PyObject *self, PyObject *args, PyObject *kwds) 
{
	PyObject *oldops=NULL;
	
	if ((oldops = PyArray_GetNumericOps())==NULL) return NULL;

	/* Should probably ensure that objects are at least callable */
	/*  Leave this to the caller for now --- error will be raised
	    later when use is attempted 
	*/
	if (kwds && PyArray_SetNumericOps(kwds) == -1) {
		Py_DECREF(oldops);
		PyErr_SetString(PyExc_ValueError, 
				"one or more objects not callable");
		return NULL;
	}
	return oldops;
}


/*MULTIARRAY_API
 Where
*/
static PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
	PyArrayObject *arr;
	PyObject *tup=NULL, *obj=NULL;
	PyObject *ret=NULL, *zero=NULL;


	arr = (PyArrayObject *)PyArray_FromAny(condition, NULL, 0, 0, 0);
	if (arr == NULL) return NULL;

	if ((x==NULL) && (y==NULL)) {
		ret = PyArray_Nonzero(arr);
		Py_DECREF(arr);
		return ret;
	}

	if ((x==NULL) || (y==NULL)) {
		Py_DECREF(arr);
		PyErr_SetString(PyExc_ValueError, "either both or neither "
				"of x and y should be given");
		return NULL;
	}


	zero = PyInt_FromLong((long) 0);

	obj = PyArray_EnsureArray(PyArray_GenericBinaryFunction(arr, zero, 
								n_ops.not_equal));
	Py_DECREF(zero);
	Py_DECREF(arr);
	if (obj == NULL) return NULL;

	tup = Py_BuildValue("(OO)", y, x);
	if (tup == NULL) {Py_DECREF(obj); return NULL;}

	ret = PyArray_Choose((PyAO *)obj, tup);

	Py_DECREF(obj);
	Py_DECREF(tup);
	return ret;
}

static char doc_where[] = "where(condition, | x, y) is shaped like condition"\
	" and has elements of x and y where condition is respectively true or"\
	" false.  If x or y are not given, then it is equivalent to"\
	" nonzero(condition).";

static PyObject *
array_where(PyObject *ignored, PyObject *args)
{
	PyObject *obj=NULL, *x=NULL, *y=NULL;
	
	if (!PyArg_ParseTuple(args, "O|OO", &obj, &x, &y)) return NULL;

	return PyArray_Where(obj, x, y);

}

static char doc_register_dtype[] = \
	"register_dtype(a) registers a new type object -- gives it a typenum";

static PyObject *
array_register_dtype(PyObject *dummy, PyObject *args)
{
	PyObject *dtype;
	int ret;
	
	if (!PyArg_ParseTuple(args, "O", &dtype)) return NULL;
	
	ret = PyArray_RegisterDataType((PyTypeObject *)dtype);
	if (ret < 0)
		return NULL;
	return PyInt_FromLong((long) ret);
}

static char doc_can_cast_safely[] = \
	"can_cast_safely(from=d1, to=d2) returns True if data type d1 "\
	"can be cast to data type d2 without losing precision.";

static PyObject *
array_can_cast_safely(PyObject *dummy, PyObject *args, PyObject *kwds)
{
	PyArray_Descr *d1=NULL;
	PyArray_Descr *d2=NULL;
	Bool ret;
	PyObject *retobj;
	static char *kwlist[] = {"from", "to", NULL};

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist, 
					PyArray_DescrConverter, &d1,
					PyArray_DescrConverter, &d2))
		return NULL;
	if (d1 == NULL || d2 == NULL) {
		PyErr_SetString(PyExc_TypeError, 
				"did not understand one of the types; "	\
				"'None' not accepted");
		return NULL;
	}
		
	ret = PyArray_CanCastTo(d1, d2);
	retobj = (ret ? Py_True : Py_False);
	Py_INCREF(retobj);
	return retobj;
}

static char doc_new_buffer[] = \
	"newbuffer(size) return a new uninitialized buffer object of size "
	"bytes";

static PyObject *
new_buffer(PyObject *dummy, PyObject *args)
{
	int size;

	if(!PyArg_ParseTuple(args, "i", &size))
		return NULL;
	
	return PyBuffer_New(size);
}

static char doc_buffer_buffer[] = \
	"getbuffer(obj [,offset[, size]]) create a buffer object from the "\
	"given object\n  referencing a slice of length size starting at "\
	"offset.  Default\n is the entire buffer. A read-write buffer is "\
	"attempted followed by a read-only buffer.";

static PyObject *
buffer_buffer(PyObject *dummy, PyObject *args, PyObject *kwds)
{
	PyObject *obj;
	int offset=0, size=Py_END_OF_BUFFER, n;
	void *unused;
	static char *kwlist[] = {"object", "offset", "size", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwlist, 
					 &obj, &offset, &size))
		return NULL;


	if (PyObject_AsWriteBuffer(obj, &unused, &n) < 0) {
		PyErr_Clear();
		return PyBuffer_FromObject(obj, offset, size);		
	}
	else
		return PyBuffer_FromReadWriteObject(obj, offset, size);
}


static struct PyMethodDef array_module_methods[] = {
	{"set_string_function", (PyCFunction)array_set_string_function, 
	 METH_VARARGS|METH_KEYWORDS, doc_set_string_function},
	{"set_numeric_ops", (PyCFunction)array_set_ops_function,
	 METH_VARARGS|METH_KEYWORDS, doc_set_ops_function},
	{"set_typeDict", (PyCFunction)array_set_typeDict,
	 METH_VARARGS, doc_set_typeDict},

	{"array",	(PyCFunction)_array_fromobject, 
	 METH_VARARGS|METH_KEYWORDS, doc_fromobject},
	{"arange",  (PyCFunction)array_arange, 
	 METH_VARARGS|METH_KEYWORDS, doc_arange},
	{"zeros",	(PyCFunction)array_zeros, 
	 METH_VARARGS|METH_KEYWORDS, doc_zeros},
	{"empty",	(PyCFunction)array_empty, 
	 METH_VARARGS|METH_KEYWORDS, doc_empty},
	{"scalar",      (PyCFunction)array_scalar,
	 METH_VARARGS|METH_KEYWORDS, doc_scalar},
	{"where",  (PyCFunction)array_where,
	 METH_VARARGS, doc_where},
	{"fromstring",(PyCFunction)array_fromString,
	 METH_VARARGS|METH_KEYWORDS, doc_fromString},
	{"concatenate", (PyCFunction)array_concatenate, 
	 METH_VARARGS|METH_KEYWORDS, doc_concatenate},
	{"inner", (PyCFunction)array_innerproduct, 
	 METH_VARARGS, doc_innerproduct}, 
	{"dot", (PyCFunction)array_matrixproduct, 
	 METH_VARARGS, doc_matrixproduct}, 
	{"_fastCopyAndTranspose", (PyCFunction)array_fastCopyAndTranspose, 
	 METH_VARARGS, doc_fastCopyAndTranspose},
	{"correlate", (PyCFunction)array_correlate, 
	 METH_VARARGS | METH_KEYWORDS, doc_correlate},
	{"frombuffer", (PyCFunction)array_frombuffer,
	 METH_VARARGS | METH_KEYWORDS, doc_frombuffer},
	{"fromfile", (PyCFunction)array_fromfile,
	 METH_VARARGS | METH_KEYWORDS, doc_fromfile},
	{"register_dtype", (PyCFunction)array_register_dtype,
	 METH_VARARGS, doc_register_dtype},
	{"can_cast", (PyCFunction)array_can_cast_safely,
	 METH_VARARGS | METH_KEYWORDS, doc_can_cast_safely},		
	{"newbuffer", (PyCFunction)new_buffer,
	 METH_VARARGS, doc_new_buffer},	
	{"getbuffer", (PyCFunction)buffer_buffer,
	 METH_VARARGS | METH_KEYWORDS, doc_buffer_buffer},	
	{NULL,		NULL, 0}		/* sentinel */
};

#include "__multiarray_api.c"

/* Establish scalar-type hierarchy */

/*  For dual inheritance we need to make sure that the objects being
    inherited from have the tp->mro object initialized.  This is
    not necessarily true for the basic type objects of Python (it is 
    checked for single inheritance but not dual in PyType_Ready).

    Thus, we call PyType_Ready on the standard Python Types, here.
*/ 
static int
setup_scalartypes(PyObject *dict)
{

	initialize_numeric_types();

        if (PyType_Ready(&PyBool_Type) < 0) return -1;
        if (PyType_Ready(&PyInt_Type) < 0) return -1;
        if (PyType_Ready(&PyFloat_Type) < 0) return -1;
        if (PyType_Ready(&PyComplex_Type) < 0) return -1;
        if (PyType_Ready(&PyString_Type) < 0) return -1;
        if (PyType_Ready(&PyUnicode_Type) < 0) return -1;

#define SINGLE_INHERIT(child, parent)                                   \
        Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;	\
        if (PyType_Ready(&Py##child##ArrType_Type) < 0) {		\
                PyErr_Print();                                          \
                PyErr_Format(PyExc_SystemError,                         \
			     "could not initialize Py%sArrType_Type",  \
                             #child);                                   \
                return -1;						\
        }
        
        if (PyType_Ready(&PyGenericArrType_Type) < 0)
                return -1;

        SINGLE_INHERIT(Numeric, Generic);
        SINGLE_INHERIT(Integer, Numeric);
        SINGLE_INHERIT(Inexact, Numeric);
        SINGLE_INHERIT(SignedInteger, Integer);
        SINGLE_INHERIT(UnsignedInteger, Integer);
        SINGLE_INHERIT(Floating, Inexact);
        SINGLE_INHERIT(ComplexFloating, Inexact);
        SINGLE_INHERIT(Flexible, Generic);
        SINGLE_INHERIT(Character, Flexible);
	
#define DUAL_INHERIT(child, parent1, parent2)                           \
        Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;	\
        Py##child##ArrType_Type.tp_bases =                              \
                Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,	\
			      &Py##parent1##_Type);			\
        if (PyType_Ready(&Py##child##ArrType_Type) < 0) {               \
                PyErr_Print();                                          \
		PyErr_Format(PyExc_SystemError,                         \
			     "could not initialize Py%sArrType_Type",   \
                             #child);                                   \
                return -1;                                              \
        }\
        Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;

#define DUAL_INHERIT2(child, parent1, parent2)				\
        Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;		\
        Py##child##ArrType_Type.tp_bases =                              \
                Py_BuildValue("(OO)", &Py##parent1##_Type,		\
			      &Py##parent2##ArrType_Type);		\
	Py##child##ArrType_Type.tp_richcompare =			\
		Py##parent1##_Type.tp_richcompare;			\
	Py##child##ArrType_Type.tp_compare =				\
		Py##parent1##_Type.tp_compare;				\
        Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;	\
        if (PyType_Ready(&Py##child##ArrType_Type) < 0) {               \
                PyErr_Print();                                          \
		PyErr_Format(PyExc_SystemError,                         \
			     "could not initialize Py%sArrType_Type",   \
                             #child);                                   \
                return -1;                                              \
        }

        SINGLE_INHERIT(Bool, Generic);
        SINGLE_INHERIT(Byte, SignedInteger);
        SINGLE_INHERIT(Short, SignedInteger);
#if SIZEOF_INT == SIZEOF_LONG
        DUAL_INHERIT(Int, Int, SignedInteger);
#else
        SINGLE_INHERIT(Int, SignedInteger);
#endif
        DUAL_INHERIT(Long, Int, SignedInteger);
#if SIZEOF_LONGLONG == SIZEOF_LONG
        DUAL_INHERIT(LongLong, Int, SignedInteger);
#else
        SINGLE_INHERIT(LongLong, SignedInteger);
#endif

        /* fprintf(stderr, "tp_free = %p, PyObject_Del = %p, int_tp_free = %p, base.tp_free = %p\n", PyIntArrType_Type.tp_free, PyObject_Del, PyInt_Type.tp_free, PySignedIntegerArrType_Type.tp_free);
	 */
	SINGLE_INHERIT(UByte, UnsignedInteger);
        SINGLE_INHERIT(UShort, UnsignedInteger);
        SINGLE_INHERIT(UInt, UnsignedInteger);
        SINGLE_INHERIT(ULong, UnsignedInteger);
        SINGLE_INHERIT(ULongLong, UnsignedInteger);

        SINGLE_INHERIT(Float, Floating);
        DUAL_INHERIT(Double, Float, Floating);
        SINGLE_INHERIT(LongDouble, Floating);

        SINGLE_INHERIT(CFloat, ComplexFloating);
        DUAL_INHERIT(CDouble, Complex, ComplexFloating);
        SINGLE_INHERIT(CLongDouble, ComplexFloating);

        DUAL_INHERIT2(String, String, Character);
        DUAL_INHERIT2(Unicode, Unicode, Character);
	
        SINGLE_INHERIT(Void, Flexible);
        
        SINGLE_INHERIT(Object, Generic);

        return 0;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT

	/* Clean up string and unicode array types so they act more like
	   strings -- get their tables from the standard types.
	*/
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
        PyObject *s;
        PyObject *newd;
        
        newd = PyDict_New();

        PyDict_SetItemString(newd, "OWNDATA", s=PyInt_FromLong(OWNDATA));
        Py_DECREF(s);
        PyDict_SetItemString(newd, "FORTRAN", s=PyInt_FromLong(FORTRAN));
        Py_DECREF(s);
        PyDict_SetItemString(newd, "CONTIGUOUS", s=PyInt_FromLong(CONTIGUOUS));
        Py_DECREF(s);
        PyDict_SetItemString(newd, "ALIGNED", s=PyInt_FromLong(ALIGNED));
        Py_DECREF(s);

        PyDict_SetItemString(newd, "NOTSWAPPED", s=PyInt_FromLong(NOTSWAPPED));
        Py_DECREF(s);
        PyDict_SetItemString(newd, "UPDATEIFCOPY", s=PyInt_FromLong(UPDATEIFCOPY));
        Py_DECREF(s);
        PyDict_SetItemString(newd, "WRITEABLE", s=PyInt_FromLong(WRITEABLE));
        Py_DECREF(s);
        
        PyDict_SetItemString(d, "_flagdict", newd);
        Py_DECREF(newd);
        return;
}


/* Initialization function for the module */

DL_EXPORT(void) initmultiarray(void) {
	PyObject *m, *d, *s;
	PyObject *c_api;
	
	/* Create the module and add the functions */
	m = Py_InitModule("multiarray", array_module_methods);
	if (!m) goto err;

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	if (!d) goto err; 

	/* Create the module and add the functions */
	if (PyType_Ready(&PyBigArray_Type) < 0) 
		return;

        PyArray_Type.tp_base = &PyBigArray_Type;

        PyArray_Type.tp_as_mapping = &array_as_mapping;
	/* Even though, this would be inherited, it needs to be set now
	   so that the __getitem__ will map to the as_mapping descriptor
	*/
        PyArray_Type.tp_as_number = &array_as_number;               
	/* For good measure */
	PyArray_Type.tp_as_sequence = &array_as_sequence;
	PyArray_Type.tp_as_buffer = &array_as_buffer;	
        PyArray_Type.tp_flags = (Py_TPFLAGS_DEFAULT 
				 | Py_TPFLAGS_BASETYPE
				 | Py_TPFLAGS_CHECKTYPES);
        PyArray_Type.tp_doc = Arraytype__doc__;

	if (PyType_Ready(&PyArray_Type) < 0)
                return;

        if (setup_scalartypes(d) < 0) goto err;

	PyArrayIter_Type.tp_iter = PyObject_SelfIter;
	PyArrayMultiIter_Type.tp_iter = PyObject_SelfIter;
	if (PyType_Ready(&PyArrayIter_Type) < 0)
		return; 
        
	if (PyType_Ready(&PyArrayMapIter_Type) < 0)
                return; 

	if (PyType_Ready(&PyArrayMultiIter_Type) < 0)
		return;

	if (PyType_Ready(&PyArrayDescr_Type) < 0)
		return;

	c_api = PyCObject_FromVoidPtr((void *)PyArray_API, NULL);
	if (PyErr_Occurred()) goto err;
	PyDict_SetItemString(d, "_ARRAY_API", c_api);
	Py_DECREF(c_api);
	if (PyErr_Occurred()) goto err;

	MultiArrayError = PyString_FromString ("multiarray.error");
	PyDict_SetItemString (d, "error", MultiArrayError);
	
	s = PyString_FromString("3.0");
	PyDict_SetItemString(d, "__version__", s);
	Py_DECREF(s);
        Py_INCREF(&PyBigArray_Type);
	PyDict_SetItemString(d, "bigndarray", (PyObject *)&PyBigArray_Type);
        Py_INCREF(&PyArray_Type);
	PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
        Py_INCREF(&PyArrayIter_Type);
	PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
        Py_INCREF(&PyArrayMultiIter_Type);
	PyDict_SetItemString(d, "broadcast", 
			     (PyObject *)&PyArrayMultiIter_Type);
	Py_INCREF(&PyArrayDescr_Type);
	PyDict_SetItemString(d, "dtypedescr", (PyObject *)&PyArrayDescr_Type);

	/* Doesn't need to be exposed to Python 
        Py_INCREF(&PyArrayMapIter_Type);
	PyDict_SetItemString(d, "mapiter", (PyObject *)&PyArrayMapIter_Type);
	*/
        set_flaginfo(d);

	if (set_typeinfo(d) == 0) 
                return;  /* otherwise there is an error */


 err:	
	/* Check for errors */
	if (PyErr_Occurred())
                PyErr_Print();
		Py_FatalError("can't initialize module multiarray");

	return;
}

