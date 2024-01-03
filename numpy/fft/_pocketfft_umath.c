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
#include <assert.h>
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_config.h"

#include "pocketfft/pocketfft.h"

/*
 * Copy all nin elements of input to the first nin of the output,
 * and any set any remaining nout-nin output elements to 0
 * (if nout < nin, copy only nout).
 */

static inline void
copy_data(char* in, npy_intp step_in, npy_intp nin,
          char* out, npy_intp step_out, npy_intp nout, npy_intp elsize)
{
    npy_intp ncopy = nin <= nout? nin : nout;
    if (step_in == elsize && step_out == elsize) {
        memcpy(out, in, ncopy*elsize);
    }
    else {
        char *ip = in, *op = out;
        for (npy_intp i = 0; i < ncopy; i++, ip += step_in, op += step_out) {
            memcpy(op, ip, elsize);
        }
    }
    if (nin < nout) {
        char *op = out + nin*elsize;
        if (step_out == elsize) {
            memset(op, 0, (nout-nin)*elsize);
        }
        else {
            for (npy_intp i = nin; i < nout; i++, op += step_out) {
                memset(op, 0, elsize);
            }
        }
    }
}


/*
 * Loops calling the pocketfft code.
 *
 * Unfortunately, the gufunc machinery does not (yet?) allow forcing contiguous
 * inner loop data, so we create a contiguous output buffer if needed
 * (input gets copied to output before processing, so can be non-contiguous).
 */
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    int (*cfft_function)(cfft_plan, double *, double) = func;
    npy_intp npts = nout;
    cfft_plan plan;
    char *buff = NULL;
    int no_mem = 1;

    if (nout == 0) {
        return;  /* no output to set */
    }

    plan = make_cfft_plan(npts);
    if (plan == NULL) {
        goto fail;
    }
    if (step_out != sizeof(npy_cdouble)) {
        buff = malloc(npts * sizeof(npy_cdouble));
        if (buff == NULL) {
            goto fail;
        }
    }

    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        double fct = *(double *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        if (ip != op_or_buff) {  /* no copy needed if in-place already */
            copy_data(ip, step_in, nin,
                      op_or_buff, sizeof(npy_cdouble), npts, sizeof(npy_cdouble));
        }
        if ((no_mem = cfft_function(plan, (double *)op_or_buff, fct)) != 0) {
            break;
        }
        if (op_or_buff == buff) {
            copy_data(op_or_buff, sizeof(npy_cdouble), npts,
                      op, step_out, npts, sizeof(npy_cdouble));
        }
    }
  fail:
    free(buff);
    destroy_cfft_plan(plan);  /* uses free so can be passed NULL */
    if (no_mem) {
        /* TODO: Requires use of new ufunc API to indicate error return */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    return;
}



static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func,
    npy_intp npts)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    rfft_plan plan;
    char *buff = NULL;
    int no_mem = 1;

    if (nout == 0) {
        return;
    }

    plan = make_rfft_plan(npts);
    if (plan == NULL) {
        goto fail;
    }
    if (step_out != sizeof(npy_cdouble)){
        buff = malloc(nout * sizeof(npy_cdouble));
        if (buff == NULL) {
            goto fail;
        }
    }

    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        double fct = *(double *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        double *op_double = (double *)op_or_buff;
        /* Copy dat to buffer in starting at position 1, as expected for FFTpack order */
        copy_data(ip, step_in, nin,
                  (char *)&op_double[1], sizeof(npy_double), nout*2 - 1, sizeof(npy_double));
        if ((no_mem = rfft_forward(plan, &op_double[1], fct)) != 0) {
            break;
        }
        op_double[0] = op_double[1];
        op_double[1] = 0.;
        if (op_or_buff == buff) {
            copy_data(op_or_buff, sizeof(npy_cdouble), nout,
                      op, step_out, nout, sizeof(npy_cdouble));
        }
    }
  fail:
    free(buff);
    destroy_rfft_plan(plan);
    if (no_mem) {
        /* TODO: Requires use of new ufunc API to indicate error return */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    return;
}

/*
 * For the forward real, we cannot know what the requested number of points is
 * just based on the number of points in the complex output array (e.g., 10
 * and 11 real input points both lead to 6 complex output points), so we
 * define versions for both even and odd number of points.
 */
static void
rfft_n_even_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    npy_intp npts = 2 * dimensions[2] - 2;
    rfft_impl(args, dimensions, steps, func, npts);
}

static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    npy_intp npts = 2 * dimensions[2] - 1;
    rfft_impl(args, dimensions, steps, func, npts);
}


static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    npy_intp npts = nout;
    rfft_plan plan;
    char *buff = NULL;
    int no_mem = 1;

    if (nout == 0) {
        return;
    }

    plan = make_rfft_plan(npts);
    if (plan == NULL) {
        goto fail;
    }
    if (step_out != sizeof(npy_double)) {
        buff = malloc(npts * sizeof(npy_double));
        if (buff == NULL) {
            goto fail;
        }
    }

    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        double fct = *(double *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        double *op_double = (double *)op_or_buff;
        /* Copy complex input to buffer in FFTpack order */
        op_double[0] = ((double *)ip)[0];
        if (nin > 1) {
            copy_data(ip + step_in, step_in, nin - 2,
                      (char *)&op_double[1], sizeof(npy_cdouble), npts / 2 - 1,
                      sizeof(npy_cdouble));
            npy_intp ncopied = (npts / 2 - 1  < nin - 2 ? npts / 2 - 1 : nin - 2);
            double *last = (double *)(ip + (ncopied + 1) * step_in);
            op_double[ncopied * 2 + 1] = last[0];
            if (npts % 2 == 1) {  /* For odd n, we still imag real of the last point */
                op_double[ncopied * 2 + 2] = last[1];
            }
        }
        if ((no_mem = rfft_backward(plan, op_double, fct)) != 0) {
            break;
        }
        if (op_or_buff == buff) {
            copy_data(op_or_buff, sizeof(npy_double), npts,
                      op, step_out, npts, sizeof(npy_double));
        }
    }
  fail:
    free(buff);
    destroy_rfft_plan(plan);
    if (no_mem) {
        /* TODO: Requires use of new ufunc API to indicate error return */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    return;
}


static PyUFuncGenericFunction fft_functions[] = { fft_loop };
static char fft_types[] = { NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE};
static void *fft_data[] = { &cfft_forward };
static void *ifft_data[] = { &cfft_backward };

static PyUFuncGenericFunction rfft_n_even_functions[] = { rfft_n_even_loop };
static PyUFuncGenericFunction rfft_n_odd_functions[] = { rfft_n_odd_loop };
static char rfft_types[] = { NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE};
static void *rfft_data[] = { (void *)NULL };

static PyUFuncGenericFunction irfft_functions[] = { irfft_loop };
static char irfft_types[] = { NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *irfft_data[] = { (void *)NULL };

static int
add_gufuncs(PyObject *dictionary) {
    PyObject *f;

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, fft_data, fft_types, 1, 2, 1, PyUFunc_None,
        "fft", "complex forward FFT\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "fft", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, ifft_data, fft_types, 1, 2, 1, PyUFunc_None,
        "ifft", "complex backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "ifft", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_even_functions, rfft_data, rfft_types, 1, 2, 1, PyUFunc_None,
        "rfft_n_even", "real forward FFT for even n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_even", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_odd_functions, rfft_data, rfft_types, 1, 2, 1, PyUFunc_None,
        "rfft_n_odd", "real forward FFT for odd n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_odd", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        irfft_functions, irfft_data, irfft_types, 1, 2, 1, PyUFunc_None,
        "irfft", "real backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "irfft", f);
    Py_DECREF(f);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_umath_pocketfft",
    .m_size = -1,
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__pocketfft_umath(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    /* Import the array and ufunc objects */
    import_array();
    import_ufunc();

    PyObject *d = PyModule_GetDict(m);
    if (add_gufuncs(d) < 0) {
        Py_DECREF(d);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
