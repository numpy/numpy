#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "fftpack.h"

static PyObject *ErrorObject;

static const char fftpack_cfftf__doc__[] = "";

static PyObject *
fftpack_cfftf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data;
    PyArray_Descr *descr;
    double *wsave, *dptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO:cfftf", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_CopyFromObject(op1,
            NPY_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL) {
        goto fail;
    }

    npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    if (nsave != npts*4 + 15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(data)/npts;
    dptr = (double *)PyArray_DATA(data);
    Py_BEGIN_ALLOW_THREADS;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        npy_cfftf(npts, dptr, wsave);
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
    Py_END_ALLOW_THREADS;
    PyArray_Free(op2, (char *)wsave);
    return (PyObject *)data;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return NULL;
}

static const char fftpack_cfftb__doc__[] = "";

static PyObject *
fftpack_cfftb(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data;
    PyArray_Descr *descr;
    double *wsave, *dptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO:cfftb", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_CopyFromObject(op1,
            NPY_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (PyArray_AsCArray(&op2, (void *)&wsave, &nsave, 1, descr) == -1) {
        goto fail;
    }
    if (data == NULL) {
        goto fail;
    }

    npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    if (nsave != npts*4 + 15) {
        PyErr_SetString(ErrorObject, "invalid work array for fft size");
        goto fail;
    }

    nrepeats = PyArray_SIZE(data)/npts;
    dptr = (double *)PyArray_DATA(data);
    Py_BEGIN_ALLOW_THREADS;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        npy_cfftb(npts, dptr, wsave);
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
    Py_END_ALLOW_THREADS;
    PyArray_Free(op2, (char *)wsave);
    return (PyObject *)data;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return NULL;
}

static const char fftpack_cffti__doc__[] = "";

static PyObject *
fftpack_cffti(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyArrayObject *op;
    npy_intp dim;
    long n;

    if (!PyArg_ParseTuple(args, "l:cffti", &n)) {
        return NULL;
    }
    /*Magic size needed by npy_cffti*/
    dim = 4*n + 15;
    /*Create a 1 dimensional array of dimensions of type double*/
    op = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
    if (op == NULL) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    NPY_SIGINT_ON;
    npy_cffti(n, (double *)PyArray_DATA((PyArrayObject*)op));
    NPY_SIGINT_OFF;
    Py_END_ALLOW_THREADS;

    return (PyObject *)op;
}

static const char fftpack_rfftf__doc__[] = "";

static PyObject *
fftpack_rfftf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data, *ret;
    PyArray_Descr *descr;
    double *wsave = NULL, *dptr, *rptr;
    npy_intp nsave;
    int npts, nrepeats, i, rstep;

    if(!PyArg_ParseTuple(args, "OO:rfftf", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_ContiguousFromObject(op1,
            NPY_DOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    /* FIXME, direct access changing contents of data->dimensions */
    npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    PyArray_DIMS(data)[PyArray_NDIM(data) - 1] = npts/2 + 1;
    ret = (PyArrayObject *)PyArray_Zeros(PyArray_NDIM(data),
            PyArray_DIMS(data), PyArray_DescrFromType(NPY_CDOUBLE), 0);
    if (ret == NULL) {
        goto fail;
    }
    PyArray_DIMS(data)[PyArray_NDIM(data) - 1] = npts;
    rstep = PyArray_DIM(ret, PyArray_NDIM(ret) - 1)*2;

    descr = PyArray_DescrFromType(NPY_DOUBLE);
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
    rptr = (double *)PyArray_DATA(ret);
    dptr = (double *)PyArray_DATA(data);

    Py_BEGIN_ALLOW_THREADS;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
        npy_rfftf(npts, rptr+1, wsave);
        rptr[0] = rptr[1];
        rptr[1] = 0.0;
        rptr += rstep;
        dptr += npts;
    }
    NPY_SIGINT_OFF;
    Py_END_ALLOW_THREADS;
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return (PyObject *)ret;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_XDECREF(data);
    Py_XDECREF(ret);
    return NULL;
}

static const char fftpack_rfftb__doc__[] = "";

static PyObject *
fftpack_rfftb(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *op1, *op2;
    PyArrayObject *data, *ret;
    PyArray_Descr *descr;
    double *wsave, *dptr, *rptr;
    npy_intp nsave;
    int npts, nrepeats, i;

    if(!PyArg_ParseTuple(args, "OO:rfftb", &op1, &op2)) {
        return NULL;
    }
    data = (PyArrayObject *)PyArray_ContiguousFromObject(op1,
            NPY_CDOUBLE, 1, 0);
    if (data == NULL) {
        return NULL;
    }
    npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    ret = (PyArrayObject *)PyArray_Zeros(PyArray_NDIM(data), PyArray_DIMS(data),
            PyArray_DescrFromType(NPY_DOUBLE), 0);

    descr = PyArray_DescrFromType(NPY_DOUBLE);
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
    rptr = (double *)PyArray_DATA(ret);
    dptr = (double *)PyArray_DATA(data);

    Py_BEGIN_ALLOW_THREADS;
    NPY_SIGINT_ON;
    for (i = 0; i < nrepeats; i++) {
        memcpy((char *)(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double));
        rptr[0] = dptr[0];
        npy_rfftb(npts, rptr, wsave);
        rptr += npts;
        dptr += npts*2;
    }
    NPY_SIGINT_OFF;
    Py_END_ALLOW_THREADS;
    PyArray_Free(op2, (char *)wsave);
    Py_DECREF(data);
    return (PyObject *)ret;

fail:
    PyArray_Free(op2, (char *)wsave);
    Py_XDECREF(data);
    Py_XDECREF(ret);
    return NULL;
}

static const char fftpack_rffti__doc__[] = "";

static PyObject *
fftpack_rffti(PyObject *NPY_UNUSED(self), PyObject *args)
{
  PyArrayObject *op;
  npy_intp dim;
  long n;

  if (!PyArg_ParseTuple(args, "l:rffti", &n)) {
      return NULL;
  }
  /*Magic size needed by npy_rffti*/
  dim = 2*n + 15;
  /*Create a 1 dimensional array of dimensions of type double*/
  op = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
  if (op == NULL) {
      return NULL;
  }
  Py_BEGIN_ALLOW_THREADS;
  NPY_SIGINT_ON;
  npy_rffti(n, (double *)PyArray_DATA((PyArrayObject*)op));
  NPY_SIGINT_OFF;
  Py_END_ALLOW_THREADS;

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

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "fftpack_lite",
        NULL,
        -1,
        fftpack_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyMODINIT_FUNC PyInit_fftpack_lite(void)
#else
#define RETVAL
PyMODINIT_FUNC
initfftpack_lite(void)
#endif
{
    PyObject *m,*d;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    static const char fftpack_module_documentation[] = "";

    m = Py_InitModule4("fftpack_lite", fftpack_methods,
            fftpack_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);
#endif

    /* Import the array object */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    ErrorObject = PyErr_NewException("fftpack.error", NULL, NULL);
    PyDict_SetItemString(d, "error", ErrorObject);

    /* XXXX Add constants here */

    return RETVAL;
}
