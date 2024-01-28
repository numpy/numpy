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

#include "pocketfft/pocketfft_hdronly.h"
using pocketfft::detail::FORWARD;
using pocketfft::detail::BACKWARD;
using pocketfft::detail::get_plan;
using pocketfft::detail::pocketfft_r;
using pocketfft::detail::pocketfft_c;
using pocketfft::detail::cmplx;

/*
 * Copy all nin elements of input to the first nin of the output,
 * and any set any remaining nout-nin output elements to 0
 * (if nout < nin, copy only nout).
 */
template <typename T>
static inline void
copy_data(char* in, npy_intp step_in, npy_intp nin,
          char* out, npy_intp step_out, npy_intp nout)
{
    npy_intp ncopy = nin <= nout? nin : nout;
    if (ncopy > 0) {
        if (step_in == sizeof(T) && step_out == sizeof(T)) {
            memcpy(out, in, ncopy*sizeof(T));
        }
        else {
            char *ip = in, *op = out;
            for (npy_intp i = 0; i < ncopy; i++, ip += step_in, op += step_out) {
                memcpy(op, ip, sizeof(T));
            }
        }
    }
    else {
        assert(ncopy == 0);
    }
    if (nout > ncopy) {
        char *op = out + ncopy*sizeof(T);
        if (step_out == sizeof(T)) {
            memset(op, 0, (nout-ncopy)*sizeof(T));
        }
        else {
            for (npy_intp i = ncopy; i < nout; i++, op += step_out) {
                memset(op, 0, sizeof(T));
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
template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    npy_intp npts = nout;
    char *buff = NULL;
    bool direction = *((bool *)func);

    if (nout == 0) {
        return;  /* no output to set */
    }

    auto plan = get_plan<pocketfft_c<T>>(npts);

    if (step_out != sizeof(cmplx<T>)) {
        buff = (char *)malloc(npts * sizeof(cmplx<T>));
        if (buff == NULL) {
            goto fail;
        }
    }

    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        T fct = *(T *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        /*
         * pocketfft works in-place, so we need to copy the data
         * (except if we want to be in-place)
         */
        if (ip != op_or_buff) {
            copy_data<cmplx<T>>(ip, step_in, nin,
                                op_or_buff, sizeof(cmplx<T>), npts);
        }
        plan->exec((cmplx<T> *)op_or_buff, fct, direction);
        if (op_or_buff == buff) {
            copy_data<cmplx<T>>(op_or_buff, sizeof(cmplx<T>), npts,
                                op, step_out, npts);
        }
    }
    free(buff);
    return;

  fail:
    /* TODO: Requires use of new ufunc API to indicate error return */
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;
    return;
}



template <typename T>
static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func,
    npy_intp npts)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    char *buff = NULL;

    if (nout == 0) {
        return;
    }
    auto plan = get_plan<pocketfft_r<T>>(npts);
    if (step_out != sizeof(cmplx<T>)){
        buff = (char *)malloc(nout * sizeof(cmplx<T>));
        if (buff == NULL) {
            goto fail;
        }
    }
    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        T fct = *(T *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        T *op_T = (T *)op_or_buff;
        /*
         * Pocketfft works in-place and for real transforms the frequency data
         * thus needs to be compressed, using that there will be no imaginary
         * component for the zero-frequency item (which is the sum of all
         * inputs and thus has to be real), nor one for the Nyquist frequency
         * for even number of points. Pocketfft uses FFTpack order,
         * R0,R1,I1,...Rn-1,In-1,Rn[,In] (last for npts odd only). To make
         * unpacking easy, we place the real data offset by one in the buffer,
         * so that we just have to move R0 and create I0=0. Note that
         * copy_data will zero the In component for even number of points.
         */
        copy_data<T>(ip, step_in, nin,
                     (char *)&op_T[1], sizeof(T), nout*2 - 1);
        plan->exec(&op_T[1], fct, FORWARD);
        op_T[0] = op_T[1];
        op_T[1] = (T)0;
        if (op_or_buff == buff) {
            copy_data<cmplx<T>>(op_or_buff, sizeof(cmplx<T>), nout,
                                op, step_out, nout);
        }
    }
    free(buff);
    return;

  fail:
    /* TODO: Requires use of new ufunc API to indicate error return */
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;
    return;
}

/*
 * For the forward real, we cannot know what the requested number of points is
 * just based on the number of points in the complex output array (e.g., 10
 * and 11 real input points both lead to 6 complex output points), so we
 * define versions for both even and odd number of points.
 */
template <typename T>
static void
rfft_n_even_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    npy_intp npts = 2 * dimensions[2] - 2;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    npy_intp npts = 2 * dimensions[2] - 1;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    npy_intp npts = nout;
    char *buff = NULL;

    if (nout == 0) {
        return;
    }
    auto plan = get_plan<pocketfft_r<T>>(npts);

    if (step_out != sizeof(T)) {
        buff = (char *)malloc(npts * sizeof(T));
        if (buff == NULL) {
            goto fail;
        }
    }

    for (npy_intp i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        T fct = *(T *)fp;
        char *op_or_buff = buff == NULL ? op : buff;
        T *op_T = (T *)op_or_buff;
        /*
         * Pocket_fft works in-place and for inverse real transforms the
         * frequency data thus needs to be compressed, removing the imaginary
         * component of the zero-frequency item (which is the sum of all
         * inputs and thus has to be real), as well as the imaginary component
         * of the Nyquist frequency for even number of points. We thus copy
         * the data to the buffer in the following order (also used by
         * FFTpack): R0,R1,I1,...Rn-1,In-1,Rn[,In] (last for npts odd only).
         */
        op_T[0] = ((T *)ip)[0];  /* copy R0 */
        if (npts > 1) {
            /*
             * Copy R1,I1... up to Rn-1,In-1 if possible, stopping earlier
             * if not all the input points are needed or if the input is short
             * (in the latter case, zeroing after).
             */
            copy_data<cmplx<T>>(ip + step_in, step_in, nin - 1,
                                (char *)&op_T[1], sizeof(cmplx<T>), (npts - 1) / 2);
            /* For even npts, we still need to set Rn. */
            if (npts % 2 == 0) {
                op_T[npts - 1] = (npts / 2 >= nin) ? (T)0 :
                    ((T *)(ip + (npts / 2) * step_in))[0];
            }
        }
        plan->exec(op_T, fct, BACKWARD);
        if (op_or_buff == buff) {
            copy_data<T>(op_or_buff, sizeof(T), npts,
                         op, step_out, npts);
        }
    }
    free(buff);
    return;

  fail:
    /* TODO: Requires use of new ufunc API to indicate error return */
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;
    return;
}

static PyUFuncGenericFunction fft_functions[] = {
    fft_loop<npy_double>,
    fft_loop<npy_float>,
    fft_loop<npy_longdouble>
};
static char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};
static void *fft_data[] = {
    (void*)&FORWARD,
    (void*)&FORWARD,
    (void*)&FORWARD
};
static void *ifft_data[] = {
    (void*)&BACKWARD,
    (void*)&BACKWARD,
    (void*)&BACKWARD
};

static PyUFuncGenericFunction rfft_n_even_functions[] = {
    rfft_n_even_loop<npy_double>,
    rfft_n_even_loop<npy_float>,
    rfft_n_even_loop<npy_longdouble>
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    rfft_n_odd_loop<npy_double>,
    rfft_n_odd_loop<npy_float>,
    rfft_n_odd_loop<npy_longdouble>
};
static char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

static PyUFuncGenericFunction irfft_functions[] = {
    irfft_loop<npy_double>,
    irfft_loop<npy_float>,
    irfft_loop<npy_longdouble>
};
static char irfft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE
};

static int
add_gufuncs(PyObject *dictionary) {
    PyObject *f;

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, fft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "fft", "complex forward FFT\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "fft", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, ifft_data, fft_types, 3, 2, 1, PyUFunc_None,
        "ifft", "complex backward FFT\n", 0, "(m),()->(n)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "ifft", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_even_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_even", "real forward FFT for even n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_even", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_odd_functions, NULL, rfft_types, 3, 2, 1, PyUFunc_None,
        "rfft_n_odd", "real forward FFT for odd n\n", 0, "(n),()->(m)");
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "rfft_n_odd", f);
    Py_DECREF(f);
    f = PyUFunc_FromFuncAndDataAndSignature(
        irfft_functions, NULL, irfft_types, 3, 2, 1, PyUFunc_None,
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
    "_multiarray_umath",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
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
