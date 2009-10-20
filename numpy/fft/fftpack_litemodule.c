#include "fftpack.h"
#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject *ErrorObject;

/* ----------------------------------------------------- */

static char fftpack_cfftf__doc__[] = "";

PyObject *
fftpack_cfftf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data;
    PyArray_Descr *descr;
    double *wsave, *dptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_CopyFromObject(op1,
            PyArray_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    descr = PyArray_DescrFromType(PyArray_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL) {
        goto fail;
    }

    npts = data->dimensions[data->nd - 1];
    if (nsave != npts*4 + 15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(data)/npts;
    dptr = (double *)data->data;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        cfftf(npts, dptr, wsave);
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
    PyArray_Free(op2, (char *)wsave);
    return (PyObject *)data;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return NULL;
}

static char fftpack_cfftb__doc__[] = "";

PyObject *
fftpack_cfftb(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data;
    PyArray_Descr *descr;
    double *wsave, *dptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_CopyFromObject(op1,
            PyArray_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    descr = PyArray_DescrFromType(PyArray_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL) {
        goto fail;
    }

    npts = data->dimensions[data->nd - 1];
    if (nsave != npts*4 + 15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(data)/npts;
    dptr = (double *)data->data;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        cfftb(npts, dptr, wsave);
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
    PyArray_Free(op2, (char *)wsave);
    return (PyObject *)data;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return NULL;
}

static char fftpack_cffti__doc__[] ="";

static PyObject *
fftpack_cffti(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyArrayObject *op;
    npy_intp dim;
    long n;

    if (!PyArg_ParseTuple(args, "l", &n)) {
        return NULL;
    }
    /*Magic size needed by cffti*/
    dim = 4*n + 15;
    /*Create a 1 dimensional array of dimensions of type double*/
    op = (PyArrayObject *)PyArray_SimpleNew(1, &dim, PyArray_DOUBLE);
    if (op == NULL) {
        return NULL;
    }

    NPY_SIGINT_ON;
    cffti(n, (double *)((PyArrayObject*)op)->data);
    NPY_SIGINT_OFF;

    return (PyObject *)op;
}

static char fftpack_rfftf__doc__[] ="";

PyObject *
fftpack_rfftf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data, *ret;
    PyArray_Descr *descr;
    double *wsave, *dptr, *rptr;
    npy_intp nsave;
    int npts, nrepeats, i, rstep;

    if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_ContiguousFromObject(op1,
            PyArray_DOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    npts = data->dimensions[data->nd-1];
    data->dimensions[data->nd - 1] = npts/2 + 1;
    ret = (PyArrayObject *)PyArray_Zeros(data->nd, data->dimensions,
            PyArray_DescrFromType(PyArray_CDOUBLE), 0);
    data->dimensions[data->nd - 1] = npts;
    rstep = (ret->dimensions[ret->nd - 1])*2;

    descr = PyArray_DescrFromType(PyArray_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL || ret == NULL) {
        goto fail;
    }
    if (nsave != npts*2+15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(data)/npts;
    rptr = (double *)ret->data;
    dptr = (double *)data->data;


    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
        rfftf(npts, rptr+1, wsave);
        rptr[0] = rptr[1];
        rptr[1] = 0.0;
        rptr += rstep;
        dptr += npts;
    }
    NPY_SIGINT_OFF;
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
fftpack_rfftb(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data, *ret;
    PyArray_Descr *descr;
    double *wsave, *dptr, *rptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_ContiguousFromObject(op1,
            PyArray_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    npts = data->dimensions[data->nd - 1];
    ret = (PyArrayObject *)PyArray_Zeros(data->nd, data->dimensions,
            PyArray_DescrFromType(PyArray_DOUBLE), 0);

    descr = PyArray_DescrFromType(PyArray_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL || ret == NULL) {
        goto fail;
    }
    if (nsave != npts*2 + 15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(ret)/npts;
    rptr = (double *)ret->data;
    dptr = (double *)data->data;

    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        memcpy((char *)(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double));
        rptr[0] = dptr[0];
        rfftb(npts, rptr, wsave);
        rptr += npts;
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
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
fftpack_rffti(PyObject *NPY_UNUSED(self), PyObject *args)
{
  PyArrayObject *op;
  npy_intp dim;
  long n;

  if (!PyArg_ParseTuple(args, "l", &n)) {
      return NULL;
  }
  /*Magic size needed by rffti*/
  dim = 2*n + 15;
  /*Create a 1 dimensional array of dimensions of type double*/
  op = (PyArrayObject *)PyArray_SimpleNew(1, &dim, PyArray_DOUBLE);
  if (op == NULL) {
      return NULL;
  }
  NPY_SIGINT_ON;
  rffti(n, (double *)((PyArrayObject*)op)->data);
  NPY_SIGINT_OFF;

  return (PyObject *)op;
}


/* List of methods defined in the module */

static struct PyMethodDef fftpack_methods[] = {
    {"cfftf",   fftpack_cfftf,  1,      fftpack_cfftf__doc__},
    {"cfftb",   fftpack_cfftb,  1,      fftpack_cfftb__doc__},
    {"cffti",   fftpack_cffti,  1,      fftpack_cffti__doc__},
    {"rfftf",   fftpack_rfftf,  1,      fftpack_rfftf__doc__},
    {"rfftb",   fftpack_rfftb,  1,      fftpack_rfftb__doc__},
    {"rffti",   fftpack_rffti,  1,      fftpack_rffti__doc__},
    {NULL, NULL, 0, NULL}          /* sentinel */
};


/* Initialization function for the module (*must* be called initfftpack) */

static char fftpack_module_documentation[] = "" ;

PyMODINIT_FUNC initfftpack_lite(void)
{
    PyObject *m, *d;

    /* Create the module and add the functions */
    m = Py_InitModule4("fftpack_lite", fftpack_methods,
            fftpack_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);

    /* Import the array object */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    ErrorObject = PyErr_NewException("fftpack.error", NULL, NULL);
    PyDict_SetItemString(d, "error", ErrorObject);

    /* XXXX Add constants here */

}
