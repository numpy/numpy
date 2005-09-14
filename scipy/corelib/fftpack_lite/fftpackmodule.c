#include "fftpack.h"
#include "Python.h"
#include "Numeric/arrayobject.h"

static PyObject *ErrorObject;

/* ----------------------------------------------------- */

static char fftpack_cfftf__doc__[] ="";

PyObject *
fftpack_cfftf(PyObject *self, PyObject *args)
{
  PyObject *op1, *op2;
  PyArrayObject *data;
  double *wsave, *dptr;
  int npts, nsave, nrepeats, i;

  if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
  data = (PyArrayObject *)PyArray_CopyFromObject(op1, PyArray_CDOUBLE, 1, 0);
	if (data == NULL) return NULL;	
  if (PyArray_As1D(&op2, (char **)&wsave, &nsave, PyArray_DOUBLE) == -1) 
    goto fail;
  if (data == NULL) goto fail;

  npts = data->dimensions[data->nd-1];
  if (nsave != npts*4+15) {
    PyErr_SetString(ErrorObject, "invalid work array for fft size");
    goto fail;
  }

  nrepeats = PyArray_SIZE(data)/npts;
  dptr = (double *)data->data;
  for (i=0; i<nrepeats; i++) {
    cfftf(npts, dptr, wsave);
    dptr += npts*2;
  }
  PyArray_Free(op2, (char *)wsave);
  return (PyObject *)data;
fail:
  PyArray_Free(op2, (char *)wsave);
  Py_DECREF(data);
  return NULL;
}

static char fftpack_cfftb__doc__[] ="";

PyObject *
fftpack_cfftb(PyObject *self, PyObject *args)
{
  PyObject *op1, *op2;
  PyArrayObject *data;
  double *wsave, *dptr;
  int npts, nsave, nrepeats, i;

  if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
  data = (PyArrayObject *)PyArray_CopyFromObject(op1, PyArray_CDOUBLE, 1, 0);
	if (data == NULL) return NULL;	
  if (PyArray_As1D(&op2, (char **)&wsave, &nsave, PyArray_DOUBLE) == -1) 
    goto fail;
  if (data == NULL) goto fail;

  npts = data->dimensions[data->nd-1];
  if (nsave != npts*4+15) {
    PyErr_SetString(ErrorObject, "invalid work array for fft size");
    goto fail;
  }

  nrepeats = PyArray_SIZE(data)/npts;
  dptr = (double *)data->data;
  for (i=0; i<nrepeats; i++) {
    cfftb(npts, dptr, wsave);
    dptr += npts*2;
  }
  PyArray_Free(op2, (char *)wsave);
  return (PyObject *)data;
fail:
  PyArray_Free(op2, (char *)wsave);
  Py_DECREF(data);
  return NULL;
}

static char fftpack_cffti__doc__[] ="";

static PyObject *
fftpack_cffti(PyObject *self, PyObject *args)
{
  PyArrayObject *op;
  int dim, n;

  if (!PyArg_ParseTuple(args, "i", &n)) return NULL;

  dim = 4*n+15; /*Magic size needed by cffti*/
  /*Create a 1 dimensional array of dimensions of type double*/
  op = (PyArrayObject *)PyArray_FromDims(1, &dim, PyArray_DOUBLE);
  if (op == NULL) return NULL;

  cffti(n, (double *)((PyArrayObject*)op)->data);

  return (PyObject *)op;
}

static char fftpack_rfftf__doc__[] ="";

PyObject *
fftpack_rfftf(PyObject *self, PyObject *args)
{
  PyObject *op1, *op2;
  PyArrayObject *data, *ret;
  double *wsave, *dptr, *rptr;
  int npts, nsave, nrepeats, i, rstep;

  if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
  data = (PyArrayObject *)PyArray_ContiguousFromObject(op1, PyArray_DOUBLE, 1, 0);
  if (data == NULL) return NULL;
  npts = data->dimensions[data->nd-1];
  data->dimensions[data->nd-1] = npts/2+1;
  ret = (PyArrayObject *)PyArray_FromDims(data->nd, data->dimensions, PyArray_CDOUBLE);
  data->dimensions[data->nd-1] = npts;
  rstep = (ret->dimensions[ret->nd-1])*2;

  if (PyArray_As1D(&op2, (char **)&wsave, &nsave, PyArray_DOUBLE) == -1) 
    goto fail;
  if (data == NULL || ret == NULL) goto fail;

  if (nsave != npts*2+15) {
    PyErr_SetString(ErrorObject, "invalid work array for fft size");
    goto fail;
  }

  nrepeats = PyArray_SIZE(data)/npts;
  rptr = (double *)ret->data;
  dptr = (double *)data->data;
  
  for (i=0; i<nrepeats; i++) {
	memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
    rfftf(npts, rptr+1, wsave);
	rptr[0] = rptr[1];
	rptr[1] = 0.0;
    rptr += rstep;
	dptr += npts;
  }
  PyArray_Free(op2, (char *)wsave);
  Py_DECREF(data);
  return (PyObject *)ret;
fail:
  PyArray_Free(op2, (char *)wsave);
  Py_XDECREF(data);
  Py_XDECREF(ret);
  return NULL;
}

static char fftpack_rfftb__doc__[] ="";


PyObject *
fftpack_rfftb(PyObject *self, PyObject *args)
{
  PyObject *op1, *op2;
  PyArrayObject *data, *ret;
  double *wsave, *dptr, *rptr;
  int npts, nsave, nrepeats, i;

  if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) return NULL;
  data = (PyArrayObject *)PyArray_ContiguousFromObject(op1, PyArray_CDOUBLE, 1, 0);
  if (data == NULL) return NULL;
  npts = data->dimensions[data->nd-1];
  ret = (PyArrayObject *)PyArray_FromDims(data->nd, data->dimensions, PyArray_DOUBLE);

  if (PyArray_As1D(&op2, (char **)&wsave, &nsave, PyArray_DOUBLE) == -1) 
    goto fail;
  if (data == NULL || ret == NULL) goto fail;

  if (nsave != npts*2+15) {
    PyErr_SetString(ErrorObject, "invalid work array for fft size");
    goto fail;
  }

  nrepeats = PyArray_SIZE(ret)/npts;
  rptr = (double *)ret->data;
  dptr = (double *)data->data;
  
  for (i=0; i<nrepeats; i++) {
	memcpy((char *)(rptr+1), (dptr+2), (npts-1)*sizeof(double));
	rptr[0] = dptr[0];
    rfftb(npts, rptr, wsave);
    rptr += npts;
	dptr += npts*2;
  }
  PyArray_Free(op2, (char *)wsave);
  Py_DECREF(data);
  return (PyObject *)ret;
fail:
  PyArray_Free(op2, (char *)wsave);
  Py_XDECREF(data);
  Py_XDECREF(ret);
  return NULL;
}


static char fftpack_rffti__doc__[] ="";

static PyObject *
fftpack_rffti(PyObject *self, PyObject *args)
{
  PyArrayObject *op;
  int dim, n;

  if (!PyArg_ParseTuple(args, "i", &n)) return NULL;

  dim = 2*n+15; /*Magic size needed by rffti*/
  /*Create a 1 dimensional array of dimensions of type double*/
  op = (PyArrayObject *)PyArray_FromDims(1, &dim, PyArray_DOUBLE);
  if (op == NULL) return NULL;

  rffti(n, (double *)((PyArrayObject*)op)->data);

  return (PyObject *)op;
}


/* List of methods defined in the module */

static struct PyMethodDef fftpack_methods[] = {
 {"cfftf",	fftpack_cfftf,	1,	fftpack_cfftf__doc__},
 {"cfftb",	fftpack_cfftb,	1,	fftpack_cfftb__doc__},
 {"cffti",	fftpack_cffti,	1,	fftpack_cffti__doc__},
 {"rfftf",	fftpack_rfftf,	1,	fftpack_rfftf__doc__},
 {"rfftb",	fftpack_rfftb,	1,	fftpack_rfftb__doc__},
 {"rffti",	fftpack_rffti,	1,	fftpack_rffti__doc__},
 {NULL,		NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initfftpack) */

static char fftpack_module_documentation[] = 
""
;

DL_EXPORT(void)
initfftpack(void)
{
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule4("fftpack", fftpack_methods,
		fftpack_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);

	/* Import the array object */
	import_array();

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyErr_NewException("fftpack.error", NULL, NULL);
	PyDict_SetItemString(d, "error", ErrorObject);

	/* XXXX Add constants here */
	
	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module fftpack");
}
