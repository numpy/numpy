
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


/* Initialization function for the module (*must* be called initArray) */

static struct PyMethodDef methods[] = {
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

