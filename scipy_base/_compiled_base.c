
#include "Python.h"
#include "Numeric/arrayobject.h"


static char doc_base_unique[] = "Return the unique elements of a 1-D sequence.";

static PyObject *base_unique(PyObject *self, PyObject *args, PyObject *kwdict)
{
  /* Returns a 1-D array containing the unique elements of a 1-D sequence.
   */

  void *new_mem=NULL;
  PyArrayObject *ainput=NULL, *aoutput=NULL;
  int asize, abytes, new;
  int copied=0, nd;
  int instride=0, elsize, k, j, dims[1];
  char *ip, *op; /* Current memory buffer */
  char *op2;
  
  static char *kwlist[] = {"input", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O!", kwlist, &PyArray_Type, &ainput)) 
    return NULL;
  
  if (ainput->nd > 1) {
    PyErr_SetString(PyExc_ValueError, "Input array must be < 1 dimensional");
    return NULL;
  }
  asize = PyArray_SIZE(ainput);
  elsize = ainput->descr->elsize;
  abytes = asize * elsize;
  nd = ainput->nd;
  if (nd > 0) {
    instride = ainput->strides[0];
  }

  new_mem = (void *)PyMem_Malloc((size_t) abytes);
  if (new_mem == NULL) {
    return PyErr_NoMemory();
  }
  
  ip = ainput->data;
  op = new_mem;
  for (k=0; k < asize; k++,ip+=instride) {
    new = 1;  /* Assume it is new */
    op2 = new_mem;
    for (j=0; j < copied; j++,op2+=elsize) {
      if (memcmp(op2,ip,elsize) == 0) {  /* Is a match found? */
        new = 0;
        break;
      }
    }
    /* No match found, copy this one over */
    if (new) {
      memcpy(op,ip,elsize);
      copied += 1;
      op += elsize; /* Get ready to put next match */
    }
  }

  dims[0] = copied;
  /* Make output array */
  if ((aoutput = (PyArrayObject *)PyArray_FromDims(nd, 
               dims, ainput->descr->type_num))==NULL) goto fail;

  memcpy(aoutput->data,new_mem,elsize*copied);
  /* Reallocate memory to new-size */
  PyMem_Free(new_mem);
  return PyArray_Return(aoutput);  
  
 fail:
  if (new_mem != NULL) PyMem_Free(new_mem);
  Py_XDECREF(aoutput);
  return NULL;
}


static char doc_base_insert[] = "Insert vals sequenctially into equivalent 1-d positions indicated by mask.";

static PyObject *base_insert(PyObject *self, PyObject *args, PyObject *kwdict)
{
  /* Returns input array with values inserted sequentially into places 
     indicated by the mask
   */

  PyObject *mask=NULL, *vals=NULL;
  PyArrayObject *ainput=NULL, *amask=NULL, *avals=NULL, *avalscast=NULL, *tmp=NULL;
  int numvals, totmask, sameshape;
  char *input_data, *mptr, *vptr, *zero;
  int melsize, delsize, copied, nd;
  int *instrides, *inshape;
  int mindx, rem_indx, indx, i, k, objarray;
  
  static char *kwlist[] = {"input","mask","vals",NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O!OO", kwlist, &PyArray_Type, &ainput, &mask, &vals))
    return NULL;

  /* Fixed problem with OBJECT ARRAYS
  if (ainput->descr->type_num == PyArray_OBJECT) {
    PyErr_SetString(PyExc_ValueError, "Not currently supported for Object arrays.");
    return NULL;
  }
  */

  amask = (PyArrayObject *)PyArray_ContiguousFromObject(mask, PyArray_NOTYPE, 0, 0);
  if (amask == NULL) return NULL;
  /* Cast an object array */
  if (amask->descr->type_num == PyArray_OBJECT) {
    tmp = (PyArrayObject *)PyArray_Cast(amask, PyArray_LONG);
    if (tmp == NULL) goto fail;
    Py_DECREF(amask);
    amask = tmp;
  }

  sameshape = 1;
  if (amask->nd == ainput->nd) {
    for (k=0; k < amask->nd; k++) 
      if (amask->dimensions[k] != ainput->dimensions[k])
        sameshape = 0;
  }
  else { /* Test to see if amask is 1d */
    if (amask->nd != 1) sameshape = 0;
    else if ((PyArray_SIZE(ainput)) != PyArray_SIZE(amask)) sameshape = 0;
  }
  if (!sameshape) {
    PyErr_SetString(PyExc_ValueError, "Mask array must be 1D or same shape as input array.");
    goto fail;
  }

  avals = (PyArrayObject *)PyArray_FromObject(vals, PyArray_NOTYPE, 0, 1);
  if (avals == NULL) goto fail;
  avalscast = (PyArrayObject *)PyArray_Cast(avals, ainput->descr->type_num);
  if (avalscast == NULL) goto fail;
  Py_DECREF(avals);

  numvals = PyArray_SIZE(avalscast);
  nd = ainput->nd;
  input_data = ainput->data;
  mptr = amask->data;
  melsize = amask->descr->elsize;
  vptr = avalscast->data;
  delsize = avalscast->descr->elsize;
  zero = amask->descr->zero;
  objarray = (ainput->descr->type_num == PyArray_OBJECT);
  
  /* Handle zero-dimensional case separately */
  if (nd == 0) {
    if (memcmp(mptr,zero,melsize) != 0) {
      /* Copy value element over to input array */
      memcpy(input_data,vptr,delsize);
      if (objarray) Py_INCREF(*((PyObject **)vptr));
    }
    Py_DECREF(amask);
    Py_DECREF(avalscast);
    Py_INCREF(Py_None);
    return Py_None;
  }

  /* Walk through mask array, when non-zero is encountered
     copy next value in the vals array to the input array.
     If we get through the value array, repeat it as necessary. 
  */
  totmask = PyArray_SIZE(amask);
  copied = 0;
  instrides = ainput->strides;
  inshape = ainput->dimensions;
  for (mindx = 0; mindx < totmask; mindx++) { 
    if (memcmp(mptr,zero,melsize) != 0) {      
      /* compute indx into input array 
       */
      rem_indx = mindx;
      indx = 0;
      for(i=nd-1; i > 0; --i) {
        indx += (rem_indx % inshape[i]) * instrides[i];
        rem_indx /= inshape[i];
      }
      indx += rem_indx * instrides[0];
      /* fprintf(stderr, "mindx = %d, indx=%d\n", mindx, indx); */
      /* Copy value element over to input array */
      memcpy(input_data+indx,vptr,delsize);
      if (objarray) Py_INCREF(*((PyObject **)vptr));
      vptr += delsize;
      copied += 1;
      /* If we move past value data.  Reset */
      if (copied >= numvals) vptr = avalscast->data;
    }
    mptr += melsize;
  }

  Py_DECREF(amask);
  Py_DECREF(avalscast);
  Py_INCREF(Py_None);
  return Py_None;
  
 fail:
  Py_XDECREF(amask);
  Py_XDECREF(avals);
  Py_XDECREF(avalscast);
  return NULL;
}


/* Decrement the reference count of all objects in **arrays. */
static void cleanup_arrays(PyArrayObject **arrays, int number)
{
  int k;
  for (k=0; k < number; k++)
    Py_XDECREF((PyObject *)arrays[k]);
  return;
}

/* All rank-0 arrays are converted to rank-1 arrays */
/* The number of dimensions of each array with rank less than
    the rank of the array with the most dimensions is increased by 
    prepending with a dimenson length of one so that all arrays have
    the same rank. */
/* Dimensions are checked and unmatched dimensions triggers an error */
/* Strides for dimensions whose real length is one is set to zero but the dimension
   length is set to the maximum dimensions for the collection of inputs  */
static int setup_input_arrays(PyTupleObject *inputs, PyArrayObject **inputarrays, int nin)
{
  int i, k;
  int maxrank=1;
  int *maxdims;
  PyObject *inputobj;
  PyArrayObject *ain, *tmparray;

  /* Convert nested sequences to arrays or just increase reference count
     if already an array */
  for (i=0; i < nin; i++) {
    ain = NULL;
    inputobj = PyTuple_GET_ITEM(inputs,i);
    ain = (PyArrayObject *)PyArray_FromObject(inputobj,PyArray_ObjectType(inputobj,0),0,0);
    if (NULL == ain) {
      cleanup_arrays(inputarrays,i);
      return -1;
    }
    if (PyArray_SIZE(ain)==0) {
      cleanup_arrays(inputarrays,i);
      PyErr_SetString(PyExc_IndexError,"arraymap: Input arrays of zero-dimensions not supported.");
      return -1;
    }
    if (ain->nd > maxrank) maxrank = ain->nd;
    if (ain->nd == 0) {  /* turn into 1-d array */
      /* convert to rank-1 array */
      if ((ain->dimensions = (int *)malloc(sizeof(int))) == NULL) {
        PyErr_SetString(PyExc_MemoryError, "arraymap: can't allocate memory for input arrays");
        cleanup_arrays(inputarrays,i);
        return -1;
      }
      if ((ain->strides = (int *)malloc(sizeof(int))) == NULL) {
        PyErr_SetString(PyExc_MemoryError, "arraymap: can't allocate memory for input arrays");
        cleanup_arrays(inputarrays,i);
        free(ain->dimensions);
        return -1;
      }
      ain->nd = 1;
      ain->dimensions[0] = 1;
      ain->strides[0] = ain->descr->elsize;
    }
    inputarrays[i] = ain;
  }

  maxdims = (int*)malloc(2*sizeof(int)*maxrank);
  if (NULL == maxdims) {
    PyErr_SetString(PyExc_MemoryError, "arraymap: can't allocate memory for input arrays");
    cleanup_arrays(inputarrays,nin);
    return -1;
  }


  /* Reshape all arrays so they have the same rank (pre-pend with length 1 dimensions) */
  /* We want to replace the header information without copying the data. 
     Keeping the reference count correct can be tricky.
     We want to make a new array object with a different header and decrease the 
     reference count of the old one without deallocating the data section */
  for (i=0; i < nin; i++) {
    ain = inputarrays[i];

    /* Initialize all dimensions to 1 */
    /* Change array shape */
    for (k=0; k < maxrank; k++) 
      maxdims[k] = 1; 
    for (k=maxrank-ain->nd; k< maxrank; k++) 
      maxdims[k] = ain->dimensions[k-maxrank+ain->nd];

    tmparray = (PyArrayObject *)PyArray_FromDimsAndData(maxrank,maxdims,ain->descr->type,ain->data);
    if (NULL == tmparray) {
      free(maxdims);
      cleanup_arrays(inputarrays,nin);
      return -1;
    }
    memmove(tmparray->strides,ain->strides,sizeof(int)*tmparray->nd);
    tmparray->base = (PyObject *)ain;  /* When tmparray is deallocated ain will be too */
    inputarrays[i] = tmparray;  /* tmparray is new array */
  }

  /* Find dimension length for the output arrays (maximum length for each
     dimension) */
  for (k=0; k < maxrank; k++) { 
    maxdims[k] = 1;
    for (i=0; i < nin; i++) 
      if (inputarrays[i]->dimensions[k] > maxdims[k])
	maxdims[k] = inputarrays[i]->dimensions[k];
  }

  /* Now set all lengths for input array dimensions to maxdims 
       and make strides equal to zero for arrays whose
       real length is 1 for a particular dimension
  */

  for (i=0; i<nin; i++) {
    ain = inputarrays[i];
    for (k=0; k< maxrank; k++) {
      if (1 == ain->dimensions[k]) {
	ain->strides[k] = 0;
	ain->dimensions[k] = maxdims[k];
      }
      else if (ain->dimensions[k] != maxdims[k]) {
	PyErr_SetString(PyExc_ValueError,"arraymap: Frames are not aligned (mismatched dimensions).");
	cleanup_arrays(inputarrays,nin);
	free(maxdims);
	return -1;
      }
    }
  }

  free(maxdims);
  return 0;

}

static int type_from_object(PyObject *obj)
{
  if (PyArray_Check(obj))
    return ((PyArrayObject *)obj)->descr->type_num;
  if (PyComplex_Check(obj)) return PyArray_CDOUBLE;
  if (PyFloat_Check(obj)) return PyArray_DOUBLE;
  if (PyInt_Check(obj) || PyLong_Check(obj)) return PyArray_LONG;
  PyErr_SetString(PyExc_ValueError, "arraymap: Invalid type for output array.");
  return -1;
}

static int type_from_char(char typechar)
{
  switch(typechar) {
  case 'c': return PyArray_CHAR;
  case 'b': return PyArray_UBYTE;
  case '1': return PyArray_SBYTE;
  case 's': return PyArray_SHORT;
  case 'i': return PyArray_INT;
#ifdef PyArray_UNSIGNED_TYPES
  case 'w': return PyArray_USHORT;
  case 'u': return PyArray_UINT;
#endif
  case 'l': return PyArray_LONG;
  case 'f': return PyArray_FLOAT;
  case 'd': return PyArray_DOUBLE;
  case 'F': return PyArray_CFLOAT;
  case 'D': return PyArray_CDOUBLE;
  default:
    PyErr_SetString(PyExc_ValueError, "arraymap: Invalid type for array");
    return -1;
  }
}



/* This sets up the output arrays by calling the function with arguments 
     the first element of each input arrays.  If otypes is NULL, the
     returned value type is used to establish the type of the output
     arrays, otherwise the characters in otypes determine the
     output types */
static int setup_output_arrays(PyObject *func, PyArrayObject **inarr, int nin, PyArrayObject ***outarr, char *otypes, int numtypes)
{
  PyObject *arglist, *result;
  PyObject *tmpobject;
  PyArrayObject *tmparr;
  int i, nout;
  int nd, *dimensions, type_num;

  nd = inarr[0]->nd;
  dimensions = inarr[0]->dimensions;

  if ((numtypes == 0) || (otypes == NULL)) { 
    /* Call function to get number of outputs */

    /* Build argument list */
    if ((arglist = PyTuple_New(nin)) == NULL) {
      return -1;
    }
    /* Construct input argument by creating a tuple with an element
     from each input array (cast to an appropriate Python Object) */
    for (i=0; i < nin; i++) {
      tmparr = inarr[i];
      /* Get first data point */
      tmpobject = tmparr->descr->getitem((void *)tmparr->data);
      if (NULL == tmpobject) {
	Py_DECREF(arglist);
	return -1;
      }
      PyTuple_SET_ITEM(arglist, i, tmpobject);  /* arg1 owns reference to tmpobj now */
    }    
    /* Call Python Function */
    if ((result=PyEval_CallObject(func, arglist))==NULL) {
      Py_DECREF(arglist);
      return -1;
    }

    Py_DECREF(arglist);

    /* If result is a tuple, create output_arrays according 
       to output.  */
    if (PyTuple_Check(result)) {
      nout = PyTuple_GET_SIZE(result);
      *outarr = (PyArrayObject **)malloc(nout*sizeof(PyArrayObject *));
      if (NULL == *outarr) {
	PyErr_SetString(PyExc_MemoryError, "arraymap: Cannot allocate memory for output arrays.");
	Py_DECREF(result);
	return -1;
      }
      /* Create nout output arrays */
      for (i=0; i < nout; i++) {
	/* Determine type */
	if ((type_num=type_from_object(PyTuple_GET_ITEM(result, i)))==-1) {
	  cleanup_arrays(*outarr,i);
	  Py_DECREF(result);
	  free(*outarr);
	  return -1;
	}
	/* Create output array */
	(*outarr)[i] = (PyArrayObject *)PyArray_FromDims(nd,dimensions,type_num);
	if (NULL == (*outarr)[i]) {
	  cleanup_arrays(*outarr,i);
	  Py_DECREF(result);
	  free(*outarr);
	  return -1;
	}
      }
    }
    else {           /* Only a single output result */
      nout = 1;
      *outarr = (PyArrayObject **)malloc(nout*sizeof(PyArrayObject *));
      if (NULL==*outarr) {
	PyErr_SetString(PyExc_MemoryError,"arraymap: Cannot allocate memory for output arrays.");
	Py_DECREF(result);
	return -1;
      }
      if ((type_num = type_from_object(result))==-1) {
	Py_DECREF(result);
	free(*outarr);
	return -1;
      }
      (*outarr)[0] = (PyArrayObject *)PyArray_FromDims(nd,dimensions,type_num);
      if (NULL == (*outarr)[0]) {
	Py_DECREF(result);
	free(*outarr);
	return -1;
      }
    }
    Py_DECREF(result);
  }

  else { /* Character output types entered */
    nout = numtypes;
    *outarr = (PyArrayObject **)malloc(nout*sizeof(PyArrayObject *));
    if (NULL==*outarr) {
      PyErr_SetString(PyExc_MemoryError,"arraymap: Cannot allocate memory for output arrays.");
      return -1;
    }
    /* Create Output arrays */
    for (i=0; i < nout; i++) {
      /* Get type */
      if ((type_num = type_from_char(otypes[i]))==-1) {
	cleanup_arrays(*outarr,i);
	free(*outarr);
	return -1;
      }
      /* Create array */
      (*outarr)[i] = (PyArrayObject *)PyArray_FromDims(nd,dimensions,type_num);
      if (NULL == (*outarr)[i]) {
	cleanup_arrays(*outarr,i);
	free(*outarr);
	return -1;
      }
    }     
  } 
  return nout;
}


/* Corresponding dimensions are assumed to match, check before calling. */
/* No rank-0 arrays (make them rank-1 arrays) */

/* This replicates the standard Ufunc broadcasting rule that if the
   dimension length is one, incrementing does not occur for that dimension.  

   This is currently done by setting the stride in that dimension to
   zero during input array setup.

   The purpose of this function is to perform a for loop over arbitrary
   discontiguous N-D arrays, call the Python function for each set of 
   corresponding elements and place the results in the output_array.
*/   
#define INCREMENT(ret_ind, nd, max_ind) \
{ \
  int k; \
  k = (nd) - 1; \
  if (++(ret_ind)[k] >= (max_ind)[k]) { \
    while (k >= 0 && ((ret_ind)[k] >= (max_ind)[k]-1)) \
      (ret_ind)[k--] = 0; \
    if (k >= 0) (ret_ind)[k]++; \
    else (ret_ind)[0] = (max_ind)[0]; \
  }  \
}

#define CALCINDEX(indx, nd_index, strides, ndim) \
{ \
  int i; \
 \
  indx = 0; \
  for (i=0; i < (ndim); i++)  \
    indx += (nd_index)[i]*(strides)[i]; \
} 

static int loop_over_arrays(PyObject *func, PyArrayObject **inarr, int nin, PyArrayObject **outarr, int nout)
{
  int i, loop_index;
  int *nd_index, indx_in, indx_out;
  PyArrayObject *in, *out, *tmparr;
  PyObject *result, *tmpobj, *arglist;

  in = inarr[0];     /* For any shape information needed */
  out = outarr[0];
  /* Allocate the N-D index initalized to zero. */
  nd_index = (int *)calloc(in->nd,sizeof(int));
  if (NULL == nd_index) {
    PyErr_SetString(PyExc_MemoryError,"arraymap: Cannot allocate memory for arrays.");
    return -1;
  }
  /* Build argument list */
  if ((arglist = PyTuple_New(nin)) == NULL) {
    free(nd_index);
    return -1;
  }

  loop_index = PyArray_Size((PyObject *)in);  /* Total number of Python function calls */

  while(loop_index--) { 
    /* Create input argument list with current element from the input
       arrays 
    */
    for (i=0; i < nin; i++) {
      tmparr = inarr[i];
      /* Find linear index into this input array */
      CALCINDEX(indx_in,nd_index,tmparr->strides,in->nd);
      /* Get object at this index */
      tmpobj = tmparr->descr->getitem((void *)(tmparr->data+indx_in));
      if (NULL == tmpobj) {
	Py_DECREF(arglist);
	free(nd_index);
	return -1;
      }
      /* This steals reference of tmpobj */
      PyTuple_SET_ITEM(arglist, i, tmpobj);  
    }
    /* Call Python Function for this set of inputs */
    if ((result=PyEval_CallObject(func, arglist))==NULL) {
      Py_DECREF(arglist);
      free(nd_index);
      return -1;
    } 

    /* Find index into (all) output arrays */
    CALCINDEX(indx_out,nd_index,out->strides,out->nd);

    /* Copy the results to the output arrays */
    if (1==nout) {
      if ((outarr[0]->descr->setitem(result,(outarr[0]->data+indx_out)))==-1) {
	free(nd_index);
	Py_DECREF(arglist);
	Py_DECREF(result);
	return -1;
      }
    }
    else if (PyTuple_Check(result)) {
      for (i=0; i<nout; i++) {
	if ((outarr[i]->descr->setitem(PyTuple_GET_ITEM(result,i),(outarr[i]->data+indx_out)))==-1) {
	  free(nd_index);
	  Py_DECREF(arglist);
	  Py_DECREF(result);
          return -1;
	}
      }
    }
    else { 
      PyErr_SetString(PyExc_ValueError,"arraymap: Function output of incorrect type.");
      free(nd_index);
      Py_DECREF(arglist);
      Py_DECREF(result);
      return -1;
    }

    /* Increment the index counter */
    INCREMENT(nd_index,in->nd,in->dimensions);
    Py_DECREF(result);

  }
  Py_DECREF(arglist);
  free(nd_index);
  return 0;
} 

static PyObject *build_output(PyArrayObject **outarr,int nout)
{
  int i;
  PyObject *out;

  if (1==nout) return PyArray_Return(outarr[0]);
  if ((out=PyTuple_New(nout))==NULL) return NULL;
  for (i=0; i<nout; i++) PyTuple_SET_ITEM(out,i,(PyObject *)(outarr[i]));
  return out;
}

static char arraymap_doc[] = "c1,..,cn = arraymap(pyfunc,inputs{,outputtypes})\n\n  Loop over the elements of the inputs tuple, applying pyfunc to the set\n  formed from each element of inputs.  Place the output in arrays c1,...,cn.\n  This function can make any pyfunc with scalar inputs and scalar outputs\n  emulate a ufunc, except none of the inputs can be length 0 arrays.\n";

static PyObject *map_PyFunc(PyObject *self, PyObject *args)
{
  PyObject *Pyfunc, *out;
  PyTupleObject *inputs;
  PyArrayObject **inputarrays, **outputarrays;
  char *otypes=NULL;
  int nin, nout, numtypes = 0;
  if (!PyArg_ParseTuple ( args, "OO!|s#;cephes.arraymap", &Pyfunc, &PyTuple_Type, (PyObject **)&inputs, &otypes, &numtypes )) return NULL;

  if (!PyCallable_Check(Pyfunc)) {
    PyErr_SetString(PyExc_TypeError, "arraymap: First argument is not a callable object.");
    return NULL;  
  }
    
  nin = PyTuple_GET_SIZE(inputs);
  inputarrays = calloc(nin,sizeof(PyArrayObject *));
  if (NULL == inputarrays) {
     PyErr_SetString(PyExc_MemoryError,"arraymap: Cannot allocate memory for input arrays.");
     return NULL;
  }
  if (setup_input_arrays(inputs,inputarrays,nin) == -1) {
    free(inputarrays);
    return NULL;
  }

  /* Construct output arrays */
  if (-1 == (nout=setup_output_arrays(Pyfunc,inputarrays,nin,&outputarrays,otypes,numtypes))) {
    cleanup_arrays(inputarrays,nin);
    free(inputarrays);
    return NULL;
  }

  /* Loop over the input arrays and place in output-arrays */
  if (-1 == loop_over_arrays(Pyfunc,inputarrays,nin,outputarrays,nout)) {
    cleanup_arrays(inputarrays,nin);
    free(inputarrays);
    cleanup_arrays(outputarrays,nout);
    free(outputarrays);
    return NULL;
  }

  cleanup_arrays(inputarrays,nin);
  free(inputarrays);
  if ((out = build_output(outputarrays,nout))==NULL) {
    cleanup_arrays(outputarrays,nout);
    free(outputarrays);
    return NULL;
  }
  free(outputarrays);
  return out;
}


/* Initialization function for the module (*must* be called initArray) */

static struct PyMethodDef methods[] = {
    {"arraymap", map_PyFunc, METH_VARARGS, arraymap_doc},
    {"_unique",	 (PyCFunction)base_unique, METH_VARARGS | METH_KEYWORDS, doc_base_unique},
    {"_insert",	 (PyCFunction)base_insert, METH_VARARGS | METH_KEYWORDS, doc_base_insert},
    {NULL, NULL}    /* sentinel */
};

DL_EXPORT(void) init_compiled_base(void) {
    PyObject *m, *d, *s;
  
    /* Create the module and add the functions */
    m = Py_InitModule("_compiled_base", methods); 

    /* Import the array and ufunc objects */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    s = PyString_FromString("0.2");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("can't initialize module _compiled_base");
}

