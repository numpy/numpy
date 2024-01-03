/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*
 *  Main implementation file.
 *
 *  Copyright (C) 2004-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"

#include "pocketfft/pocketfft.h"

static PyObject *
execute_complex(PyObject *a1, int is_forward, double fct)
{
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_CDOUBLE), 1, 0,
            NPY_ARRAY_ENSURECOPY | NPY_ARRAY_DEFAULT |
            NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;

    int npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    cfft_plan plan=NULL;

    int nrepeats = PyArray_SIZE(data)/npts;
    double *dptr = (double *)PyArray_DATA(data);
    int fail=0;
    Py_BEGIN_ALLOW_THREADS;
    plan = make_cfft_plan(npts);
    if (!plan) fail=1;
    if (!fail)
      for (int i = 0; i < nrepeats; i++) {
          int res = is_forward ?
            cfft_forward(plan, dptr, fct) : cfft_backward(plan, dptr, fct);
          if (res!=0) { fail=1; break; }
          dptr += npts*2;
      }
    if (plan) destroy_cfft_plan(plan);
    Py_END_ALLOW_THREADS;
    if (fail) {
      Py_XDECREF(data);
      return PyErr_NoMemory();
    }
    return (PyObject *)data;
}

static PyObject *
execute_real_forward(PyObject *a1, double fct)
{
    rfft_plan plan=NULL;
    int fail = 0;
    npy_intp tdim[NPY_MAXDIMS];

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
            NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;

    int ndim = PyArray_NDIM(data);
    const npy_intp *odim = PyArray_DIMS(data);
    int npts = odim[ndim - 1];
    for (int d=0; d<ndim-1; ++d)
      tdim[d] = odim[d];
    tdim[ndim-1] = npts/2 + 1;
    PyArrayObject *ret = (PyArrayObject *)PyArray_Empty(ndim,
            tdim, PyArray_DescrFromType(NPY_CDOUBLE), 0);
    if (!ret) fail=1;
    if (!fail) {
      int rstep = PyArray_DIM(ret, PyArray_NDIM(ret) - 1)*2;

      int nrepeats = PyArray_SIZE(data)/npts;
      double *rptr = (double *)PyArray_DATA(ret),
             *dptr = (double *)PyArray_DATA(data);

      Py_BEGIN_ALLOW_THREADS;
      plan = make_rfft_plan(npts);
      if (!plan) fail=1;
      if (!fail)
        for (int i = 0; i < nrepeats; i++) {
            rptr[rstep-1] = 0.0;
            memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
            if (rfft_forward(plan, rptr+1, fct)!=0) {fail=1; break;}
            rptr[0] = rptr[1];
            rptr[1] = 0.0;
            rptr += rstep;
            dptr += npts;
      }
      if (plan) destroy_rfft_plan(plan);
      Py_END_ALLOW_THREADS;
    }
    if (fail) {
      Py_XDECREF(data);
      Py_XDECREF(ret);
      return PyErr_NoMemory();
    }
    Py_DECREF(data);
    return (PyObject *)ret;
}
static PyObject *
execute_real_backward(PyObject *a1, double fct)
{
    rfft_plan plan=NULL;
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_CDOUBLE), 1, 0,
            NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;
    int npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    PyArrayObject *ret = (PyArrayObject *)PyArray_Empty(PyArray_NDIM(data),
            PyArray_DIMS(data), PyArray_DescrFromType(NPY_DOUBLE), 0);
    int fail = 0;
    if (!ret) fail=1;
    if (!fail) {
      int nrepeats = PyArray_SIZE(ret)/npts;
      double *rptr = (double *)PyArray_DATA(ret),
             *dptr = (double *)PyArray_DATA(data);

      Py_BEGIN_ALLOW_THREADS;
      plan = make_rfft_plan(npts);
      if (!plan) fail=1;
      if (!fail) {
        for (int i = 0; i < nrepeats; i++) {
          memcpy((char *)(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double));
          rptr[0] = dptr[0];
          if (rfft_backward(plan, rptr, fct)!=0) {fail=1; break;}
          rptr += npts;
          dptr += npts*2;
        }
      }
      if (plan) destroy_rfft_plan(plan);
      Py_END_ALLOW_THREADS;
    }
    if (fail) {
      Py_XDECREF(data);
      Py_XDECREF(ret);
      return PyErr_NoMemory();
    }
    Py_DECREF(data);
    return (PyObject *)ret;
}

static PyObject *
execute_real(PyObject *a1, int is_forward, double fct)
{
    return is_forward ? execute_real_forward(a1, fct)
                      : execute_real_backward(a1, fct);
}

static const char execute__doc__[] = "";

static PyObject *
execute(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a1;
    int is_real, is_forward;
    double fct;

    if(!PyArg_ParseTuple(args, "Oiid:execute", &a1, &is_real, &is_forward, &fct)) {
        return NULL;
    }

    return is_real ? execute_real(a1, is_forward, fct)
                   : execute_complex(a1, is_forward, fct);
}

/* List of methods defined in the module */

static struct PyMethodDef methods[] = {
    {"execute",   execute,   1, execute__doc__},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pocketfft_internal",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__pocketfft_internal(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    /* Import the array object */
    import_array();

    /* XXXX Add constants here */

    return m;
}
