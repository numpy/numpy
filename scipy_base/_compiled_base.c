
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
  void *ip, *op; /* Current memory buffer */
  void *op2;
  
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


/* Initialization function for the module (*must* be called initArray) */

static struct PyMethodDef methods[] = {
    {"_unique",	 (PyCFunction)base_unique, METH_VARARGS | METH_KEYWORDS, doc_base_unique},
    {NULL, NULL}    /* sentinel */
};

DL_EXPORT(void) init_compiled_base(void) {
    PyObject *m, *d, *s, *f1;
  
    /* Create the module and add the functions */
    m = Py_InitModule("_compiled_base", methods); 

    /* Import the array and ufunc objects */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    s = PyString_FromString("0.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("can't initialize module _compiled_base");
}

