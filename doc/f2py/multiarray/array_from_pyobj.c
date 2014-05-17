/*
 * File: array_from_pyobj.c
 *
 * Description:
 * ------------
 * Provides array_from_pyobj function that returns a contigious array
 * object with the given dimensions and required storage order, either
 * in row-major (C) or column-major (Fortran) order. The function
 * array_from_pyobj is very flexible about its Python object argument
 * that can be any number, list, tuple, or array.
 *
 * array_from_pyobj is used in f2py generated Python extension
 * modules.
 *
 * Author: Pearu Peterson <pearu@cens.ioc.ee>
 * Created: 13-16 January 2002
 * $Id: array_from_pyobj.c,v 1.1 2002/01/16 18:57:33 pearu Exp $
 */


#define ARR_IS_NULL(arr,mess) \
if (arr==NULL) { \
    fprintf(stderr,"array_from_pyobj:" mess); \
    return NULL; \
}

#define CHECK_DIMS_DEFINED(rank,dims,mess) \
if (count_nonpos(rank,dims)) { \
  fprintf(stderr,"array_from_pyobj:" mess); \
  return NULL; \
}

#define HAS_PROPER_ELSIZE(arr,type_num) \
  ((PyArray_DescrFromType(type_num)->elsize) == (arr)->descr->elsize)

/* static */
/* void f2py_show_args(const int type_num, */
/* 		    const int *dims, */
/* 		    const int rank, */
/* 		    const int intent) { */
/*   int i; */
/*   fprintf(stderr,"array_from_pyobj:\n\ttype_num=%d\n\trank=%d\n\tintent=%d\n",\ */
/* 	  type_num,rank,intent); */
/*   for (i=0;i<rank;++i) */
/*     fprintf(stderr,"\tdims[%d]=%d\n",i,dims[i]); */
/* } */

static
int count_nonpos(const int rank,
		 const int *dims) {
  int i=0,r=0;
  while (i<rank) {
    if (dims[i] <= 0) ++r;
    ++i;
  }
  return r;
}

static void lazy_transpose(PyArrayObject* arr);
static int check_and_fix_dimensions(const PyArrayObject* arr,
				    const int rank,
				    int *dims);
static
int array_has_column_major_storage(const PyArrayObject *ap);

static
PyArrayObject* array_from_pyobj(const int type_num,
				int *dims,
				const int rank,
				const int intent,
				PyObject *obj) {
  /* Note about reference counting
     -----------------------------
     If the caller returns the array to Python, it must be done with
     Py_BuildValue("N",arr).
     Otherwise, if obj!=arr then the caller must call Py_DECREF(arr).
  */

/*   f2py_show_args(type_num,dims,rank,intent); */

  if (intent & F2PY_INTENT_CACHE) {
    /* Don't expect correct storage order or anything reasonable when
       returning cache array. */
    if ((intent & F2PY_INTENT_HIDE)
	|| (obj==Py_None)) {
      PyArrayObject *arr = NULL;
      CHECK_DIMS_DEFINED(rank,dims,"optional,intent(cache) must"
			 " have defined dimensions.\n");
      arr = (PyArrayObject *)PyArray_FromDims(rank,dims,type_num);
      ARR_IS_NULL(arr,"FromDims failed: optional,intent(cache)\n");
      if (intent & F2PY_INTENT_OUT)
	Py_INCREF(arr);
      return arr;
    }
    if (PyArray_Check(obj)
	&& ISCONTIGUOUS((PyArrayObject *)obj)
	&& HAS_PROPER_ELSIZE((PyArrayObject *)obj,type_num)
	) {
      if (check_and_fix_dimensions((PyArrayObject *)obj,rank,dims))
	return NULL; /*XXX: set exception */
      if (intent & F2PY_INTENT_OUT)
	Py_INCREF(obj);
      return (PyArrayObject *)obj;
    }
    ARR_IS_NULL(NULL,"intent(cache) must be contiguous array with a proper elsize.\n");
  }

  if (intent & F2PY_INTENT_HIDE) {
    PyArrayObject *arr = NULL;
    CHECK_DIMS_DEFINED(rank,dims,"intent(hide) must have defined dimensions.\n");
    arr = (PyArrayObject *)PyArray_FromDims(rank,dims,type_num);
    ARR_IS_NULL(arr,"FromDims failed: intent(hide)\n");
    if (intent & F2PY_INTENT_OUT) {
      if ((!(intent & F2PY_INTENT_C)) && (rank>1)) {
	lazy_transpose(arr);
	arr->flags &= ~NPY_CONTIGUOUS;
      }
      Py_INCREF(arr);
    }
    return arr;
  }

  if (PyArray_Check(obj)) { /* here we have always intent(in) or
			       intent(inout) */

    PyArrayObject *arr = (PyArrayObject *)obj;
    int is_cont = (intent & F2PY_INTENT_C) ?
      (ISCONTIGUOUS(arr)) : (array_has_column_major_storage(arr));

    if (check_and_fix_dimensions(arr,rank,dims))
      return NULL; /*XXX: set exception */

    if ((intent & F2PY_INTENT_COPY)
	|| (! (is_cont
	       && HAS_PROPER_ELSIZE(arr,type_num)
	       && PyArray_CanCastSafely(arr->descr->type_num,type_num)))) {
      PyArrayObject *tmp_arr = NULL;
      if (intent & F2PY_INTENT_INOUT) {
	ARR_IS_NULL(NULL,"intent(inout) array must be contiguous and"
		    " with a proper type and size.\n")
	  }
      if ((rank>1) && (! (intent & F2PY_INTENT_C)))
	lazy_transpose(arr);
      if (PyArray_CanCastSafely(arr->descr->type_num,type_num)) {
	tmp_arr = (PyArrayObject *)PyArray_CopyFromObject(obj,type_num,0,0);
	ARR_IS_NULL(arr,"CopyFromObject failed: array.\n");
      } else {
	tmp_arr = (PyArrayObject *)PyArray_FromDims(arr->nd,
						    arr->dimensions,
						    type_num);
	ARR_IS_NULL(tmp_arr,"FromDims failed: array with unsafe cast.\n");
	if (copy_ND_array(arr,tmp_arr))
	  ARR_IS_NULL(NULL,"copy_ND_array failed: array with unsafe cast.\n");
      }
      if ((rank>1) && (! (intent & F2PY_INTENT_C))) {
	lazy_transpose(arr);
	lazy_transpose(tmp_arr);
	tmp_arr->flags &= ~NPY_CONTIGUOUS;
      }
      arr = tmp_arr;
    }
    if (intent & F2PY_INTENT_OUT)
      Py_INCREF(arr);
      return arr;
  }

  if ((obj==Py_None) && (intent & F2PY_OPTIONAL)) {
    PyArrayObject *arr = NULL;
    CHECK_DIMS_DEFINED(rank,dims,"optional must have defined dimensions.\n");
    arr = (PyArrayObject *)PyArray_FromDims(rank,dims,type_num);
    ARR_IS_NULL(arr,"FromDims failed: optional.\n");
    if (intent & F2PY_INTENT_OUT) {
      if ((!(intent & F2PY_INTENT_C)) && (rank>1)) {
	lazy_transpose(arr);
	arr->flags &= ~NPY_CONTIGUOUS;
      }
      Py_INCREF(arr);
    }
    return arr;
  }

  if (intent & F2PY_INTENT_INOUT) {
    ARR_IS_NULL(NULL,"intent(inout) argument must be an array.\n");
  }

  {
    PyArrayObject *arr = (PyArrayObject *) \
      PyArray_ContiguousFromObject(obj,type_num,0,0);
    ARR_IS_NULL(arr,"ContiguousFromObject failed: not a sequence.\n");
    if (check_and_fix_dimensions(arr,rank,dims))
      return NULL; /*XXX: set exception */
    if ((rank>1) && (! (intent & F2PY_INTENT_C))) {
      PyArrayObject *tmp_arr = NULL;
      lazy_transpose(arr);
      arr->flags &= ~NPY_CONTIGUOUS;
      tmp_arr = (PyArrayObject *) PyArray_CopyFromObject((PyObject *)arr,type_num,0,0);
      Py_DECREF(arr);
      arr = tmp_arr;
      ARR_IS_NULL(arr,"CopyFromObject(Array) failed: intent(fortran)\n");
      lazy_transpose(arr);
      arr->flags &= ~NPY_CONTIGUOUS;
    }
    if (intent & F2PY_INTENT_OUT)
      Py_INCREF(arr);
    return arr;
  }

}

           /*****************************************/
           /* Helper functions for array_from_pyobj */
           /*****************************************/

static
int array_has_column_major_storage(const PyArrayObject *ap) {
  /* array_has_column_major_storage(a) is equivalent to
     transpose(a).iscontiguous() but more efficient.

     This function can be used in order to decide whether to use a
     Fortran or C version of a wrapped function. This is relevant, for
     example, in choosing a clapack or flapack function depending on
     the storage order of array arguments.
  */
  int sd;
  int i;
  sd = ap->descr->elsize;
  for (i=0;i<ap->nd;++i) {
    if (ap->dimensions[i] == 0) return 1;
    if (ap->strides[i] != sd) return 0;
    sd *= ap->dimensions[i];
  }
  return 1;
}

static
void lazy_transpose(PyArrayObject* arr) {
  /*
    Changes the order of array strides and dimensions.  This
    corresponds to the lazy transpose of a Numeric array in-situ.
    Note that this function is assumed to be used even times for a
    given array. Otherwise, the caller should set flags &= ~NPY_CONTIGUOUS.
   */
  int rank,i,s,j;
  rank = arr->nd;
  if (rank < 2) return;

  for(i=0,j=rank-1;i<rank/2;++i,--j) {
    s = arr->strides[i];
    arr->strides[i] = arr->strides[j];
    arr->strides[j] = s;
    s = arr->dimensions[i];
    arr->dimensions[i] = arr->dimensions[j];
    arr->dimensions[j] = s;
  }
}

static
int check_and_fix_dimensions(const PyArrayObject* arr,const int rank,int *dims) {
  /*
    This function fills in blanks (that are -1's) in dims list using
    the dimensions from arr. It also checks that non-blank dims will
    match with the corresponding values in arr dimensions.
   */
  const int arr_size = (arr->nd)?PyArray_Size((PyObject *)arr):1;

  if (rank > arr->nd) { /* [1,2] -> [[1],[2]]; 1 -> [[1]]  */
    int new_size = 1;
    int free_axe = -1;
    int i;
    /* Fill dims where -1 or 0; check dimensions; calc new_size; */
    for(i=0;i<arr->nd;++i) {
      if (dims[i] >= 0) {
	if (dims[i]!=arr->dimensions[i]) {
	  fprintf(stderr,"%d-th dimension must be fixed to %d but got %d\n",
		  i,dims[i],arr->dimensions[i]);
	  return 1;
	}
	if (!dims[i]) dims[i] = 1;
      } else {
	dims[i] = arr->dimensions[i] ? arr->dimensions[i] : 1;
      }
      new_size *= dims[i];
    }
    for(i=arr->nd;i<rank;++i)
      if (dims[i]>1) {
	fprintf(stderr,"%d-th dimension must be %d but got 0 (not defined).\n",
		i,dims[i]);
	return 1;
      } else if (free_axe<0)
	free_axe = i;
      else
	dims[i] = 1;
    if (free_axe>=0) {
      dims[free_axe] = arr_size/new_size;
      new_size *= dims[free_axe];
    }
    if (new_size != arr_size) {
      fprintf(stderr,"confused: new_size=%d, arr_size=%d (maybe too many free"
	      " indices)\n",new_size,arr_size);
      return 1;
    }
  } else {
    int i;
    for (i=rank;i<arr->nd;++i)
      if (arr->dimensions[i]>1) {
	fprintf(stderr,"too many axes: %d, expected rank=%d\n",arr->nd,rank);
	return 1;
      }
    for (i=0;i<rank;++i)
      if (dims[i]>=0) {
	if (arr->dimensions[i]!=dims[i]) {
	  fprintf(stderr,"%d-th dimension must be fixed to %d but got %d\n",
		  i,dims[i],arr->dimensions[i]);
	  return 1;
	}
	if (!dims[i]) dims[i] = 1;
      } else
	dims[i] = arr->dimensions[i];
  }
  return 0;
}

/* End of file: array_from_pyobj.c */
