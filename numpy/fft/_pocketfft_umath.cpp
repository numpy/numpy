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
#include <assert.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_config.h"

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft/pocketfft_hdronly.h"

/*
 * In order to ensure that C++ exceptions are converted to Python
 * ones before crossing over to the C machinery, we must catch them.
 * This template can be used to wrap a C++ written ufunc to do this via:
 *      wrap_legacy_cpp_ufunc<cpp_ufunc>
 */
template<PyUFuncGenericFunction cpp_ufunc>
static void
wrap_legacy_cpp_ufunc(char **args, npy_intp const *dimensions,
                      npy_intp const *steps, void *func)
{
    NPY_ALLOW_C_API_DEF
    try {
        cpp_ufunc(args, dimensions, steps, func);
    }
    catch (std::bad_alloc& e) {
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    catch (const std::exception& e) {
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
    }
}

/*
 * Transfer to and from a contiguous buffer.
 * copy_input: copy min(nin, n) elements from input to buffer and zero rest.
 * copy_output: copy n elements from buffer to output.
 */
template <typename T>
static inline void
copy_input(char *in, npy_intp step_in, size_t nin,
           T buff[], size_t n)
{
    size_t ncopy = nin <= n ? nin : n;
    char *ip = in;
    size_t i;
    for (i = 0; i < ncopy; i++, ip += step_in) {
      buff[i] = *(T *)ip;
    }
    for (; i < n; i++) {
      buff[i] = 0;
    }
}

template <typename T>
static inline void
copy_output(T buff[], char *out, npy_intp step_out, size_t n)
{
    char *op = out;
    for (size_t i = 0; i < n; i++, op += step_out) {
        *(T *)op = buff[i];
    }
}

/*
 * Gufunc loops calling the pocketfft code.
 */
template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    bool direction = *((bool *)func); /* pocketfft::FORWARD or BACKWARD */

    assert (nout > 0);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * For the common case of nin >= nout, fixed factor, and suitably sized
     * outer loop, we call pocketfft directly to benefit from its vectorization.
     * (For nin>nout, this just removes the extra input points, as required;
     * the vlen constraint avoids compiling extra code for longdouble, which
     * cannot be vectorized so does not benefit.)
     */
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= nout && sf == 0) {
        std::vector<size_t> shape = { n_outer, nout };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        std::vector<size_t> axes = { 1 };
        pocketfft::c2c(shape, strides_in, strides_out, axes, direction,
                       (std::complex<T> *)ip, (std::complex<T> *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(nout);
    auto buffered = (step_out != sizeof(std::complex<T>));
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        if (ip != (char*)op_or_buff) {
            copy_input(ip, step_in, nin, op_or_buff, nout);
        }
        plan->exec((pocketfft::detail::cmplx<T> *)op_or_buff, *(T *)fp, direction);
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    return;
}

template <typename T>
static void
rfft_impl(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *func, size_t npts)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    npy_intp si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    npy_intp step_in = steps[3], step_out = steps[4];

    assert (nout > 0 && nout == npts / 2 + 1);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts && sf == 0) {
        std::vector<size_t> shape_in = { n_outer, npts };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        std::vector<size_t> axes = { 1 };
        pocketfft::r2c(shape_in, strides_in, strides_out, axes, pocketfft::FORWARD,
                       (T *)ip, (std::complex<T> *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(npts);
    auto buffered = (step_out != sizeof(std::complex<T>));
    pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
    auto nin_used = nin <= npts ? nin : npts;
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
        /*
         * The internal pocketfft routines work in-place and for real
         * transforms the frequency data thus needs to be compressed, using
         * that there will be no imaginary component for the zero-frequency
         * item (which is the sum of all inputs and thus has to be real), nor
         * one for the Nyquist frequency for even number of points.
         * Pocketfft uses FFTpack order, R0,R1,I1,...Rn-1,In-1,Rn[,In] (last
         * for npts odd only). To make unpacking easy, we place the real data
         * offset by one in the buffer, so that we just have to move R0 and
         * create I0=0. Note that copy_input will zero the In component for
         * even number of points.
         */
        copy_input(ip, step_in, nin_used, &((T *)op_or_buff)[1], nout*2 - 1);
        plan->exec(&((T *)op_or_buff)[1], *(T *)fp, pocketfft::FORWARD);
        op_or_buff[0] = op_or_buff[0].imag();  // I0->R0, I0=0
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
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
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0);
    size_t npts = 2 * nout - 2;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
rfft_n_odd_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    size_t nout = (size_t)dimensions[2];
    assert (nout > 0);
    size_t npts = 2 * nout - 1;
    rfft_impl<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func)
{
    char *ip = args[0], *fp = args[1], *op = args[2];
    size_t n_outer = (size_t)dimensions[0];
    ptrdiff_t si = steps[0], sf = steps[1], so = steps[2];
    size_t nin = (size_t)dimensions[1], nout = (size_t)dimensions[2];
    ptrdiff_t step_in = steps[3], step_out = steps[4];

    assert(nout > 0);

#ifndef POCKETFFT_NO_VECTORS
    /*
     * Call pocketfft directly if vectorization is possible.
     */
    size_t npts_in = nout / 2 + 1;
    constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
    if (vlen > 1 && n_outer >= vlen && nin >= npts_in && sf == 0) {
        std::vector<size_t> axes = { 1 };
        std::vector<size_t> shape_out = { n_outer, nout };
        std::vector<ptrdiff_t> strides_in = { si, step_in };
        std::vector<ptrdiff_t> strides_out = { so, step_out};
        pocketfft::c2r(shape_out, strides_in, strides_out, axes, pocketfft::BACKWARD,
                       (std::complex<T> *)ip, (T *)op, *(T *)fp);
        return;
    }
#endif
    /*
     * Otherwise, use a non-vectorized loop in which we try to minimize copies.
     * We do still need a buffer if the output is not contiguous.
     */
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(nout);
    auto buffered = (step_out != sizeof(T));
    pocketfft::detail::arr<T> buff(buffered ? nout : 0);
    for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
        T *op_or_buff = buffered ? buff.data() : (T *)op;
        /*
         * Pocket_fft works in-place and for inverse real transforms the
         * frequency data thus needs to be compressed, removing the imaginary
         * component of the zero-frequency item (which is the sum of all
         * inputs and thus has to be real), as well as the imaginary component
         * of the Nyquist frequency for even number of points. We thus copy
         * the data to the buffer in the following order (also used by
         * FFTpack): R0,R1,I1,...Rn-1,In-1,Rn[,In] (last for npts odd only).
         */
        op_or_buff[0] = ((T *)ip)[0];  /* copy R0 */
        if (nout > 1) {
            /*
             * Copy R1,I1... up to Rn-1,In-1 if possible, stopping earlier
             * if not all the input points are needed or if the input is short
             * (in the latter case, zeroing after).
             */
            copy_input(ip + step_in, step_in, nin - 1,
                       (std::complex<T> *)&op_or_buff[1], (nout - 1) / 2);
            /* For even nout, we still need to set Rn. */
            if (nout % 2 == 0) {
                op_or_buff[nout - 1] = (nout / 2 >= nin) ? (T)0 :
                    ((T *)(ip + (nout / 2) * step_in))[0];
            }
        }
        plan->exec(op_or_buff, *(T *)fp, pocketfft::BACKWARD);
        if (buffered) {
            copy_output(op_or_buff, op, step_out, nout);
        }
    }
    return;
}

static PyUFuncGenericFunction fft_functions[] = {
    wrap_legacy_cpp_ufunc<fft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<fft_loop<npy_longdouble>>
};
static const char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};
static void *const fft_data[] = {
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD,
    (void*)&pocketfft::FORWARD
};
static void *const ifft_data[] = {
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD,
    (void*)&pocketfft::BACKWARD
};

static PyUFuncGenericFunction rfft_n_even_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<npy_longdouble>>
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<npy_longdouble>>
};
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CLONGDOUBLE
};

static PyUFuncGenericFunction irfft_functions[] = {
    wrap_legacy_cpp_ufunc<irfft_loop<npy_double>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_float>>,
    wrap_legacy_cpp_ufunc<irfft_loop<npy_longdouble>>
};
static const char irfft_types[] = {
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

static int module_loaded = 0;

static int
_pocketfft_umath_exec(PyObject *m)
{
    // https://docs.python.org/3/howto/isolating-extensions.html#opt-out-limiting-to-one-module-object-per-process
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;

    /* Import the array and ufunc objects */
    if (PyArray_ImportNumPyAPI() < 0) {
        return -1;
    }
    if (PyUFunc_ImportUFuncAPI() < 0) {
        return -1;
    }

    PyObject *d = PyModule_GetDict(m);
    if (add_gufuncs(d) < 0) {
        Py_DECREF(d);
        return -1;
    }

    return 0;
}

static struct PyModuleDef_Slot _pocketfft_umath_slots[] = {
    {Py_mod_exec, (void*)_pocketfft_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    "_pocketfft_umath",     /* m_name */
    NULL,                   /* m_doc */
    0,                      /* m_size */
    NULL,                   /* m_methods */
    _pocketfft_umath_slots, /* m_slots */
};

PyMODINIT_FUNC PyInit__pocketfft_umath(void) {
    return PyModuleDef_Init(&moduledef);
}
